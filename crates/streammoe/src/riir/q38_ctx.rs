//! [`RsCtx`] glue for Qwen3.8-Flash-Next (`model-qwen3-8-flash-next`).
//!
//! The family's forward is [`Q38Engine`]; `RsCtx` keeps its expert
//! files, weight file, hot tier, IO pool, checkpoint store and memory
//! cap, and routes every eval and state op here (small `cfg` hunks in
//! `riir/mod.rs`). The qwen3_5_moe layer caches, linear buffers and
//! per-layer `LayerStates` stay empty for this family.
//!
//! What it does not do (rejected at open or at the call):
//! - a partial `layer_range` / `eval_stage` with hidden rows (the
//!   residual crossing a stage is `hc · H` wide);
//! - KV formats other than f32, `max_requests > 1`,
//!   `max_decode_tokens > 1` (`eval_tokens` passes, `rewind_tokens`);
//! - `memory_seq_rm` to a middle position (`false`: linear-attention
//!   and PLE conv state cannot be truncated — use checkpoints);
//! - M-RoPE positions (`eval_stage_mrope`).
//!
//! Prefill runs in chunks of `STREAMMOE_Q38_CHUNK` tokens (default
//! 512; [`engine::prefill_chunk`]) through [`Q38Engine::eval`]; under a
//! cap [`plan`] may pick a smaller chunk ([`Q38PrefillIo`]).

use std::path::PathBuf;

use super::budget::{Account, BudgetPlan};
use super::config::OpenConfig;
use super::model_layer::qwen3_8_flash_next::engine::{self, Q38Engine, Q38EngineError, Q38PlanBytes, Q38PrefillIo};
use super::snapshot::state_snapshot::StateSnapshotError;
use super::io::expert_io_mode::ExpertIoMode;
use super::{KvFormat, MetalBackend, PrefetchState, RsCtx, RsError, StageInput, StageOutput};

/// The family engine and where its model lives.
pub(crate) struct Q38Slot {
    /// `experts_dir` (holds `ngram.mfng` and `packed_experts/`).
    pub root: PathBuf,
    pub engine: Option<Q38Engine<MetalBackend>>,
    /// Prefill settings the budget plan fitted to the cap (`None`
    /// uncapped: [`Q38PrefillIo::for_root`]).
    pub io: Option<Q38PrefillIo>,
}

impl Q38Slot {
    pub fn new(root: &std::path::Path, io: Option<Q38PrefillIo>) -> Self {
        Self { root: root.to_path_buf(), engine: None, io }
    }
}

fn unsupported(detail: &str) -> RsError {
    RsError::InvalidConfig { detail: format!("Qwen3.8-Flash-Next: {detail}") }
}

/// Reject the [`OpenConfig`] settings this family's engine lacks.
pub(crate) fn check_config(config: &OpenConfig) -> Result<(), RsError> {
    if !super::range::is_full(&config.resolved_layer_range()) {
        return Err(unsupported("layer ranges are not supported yet (the residual is hc·H wide)"));
    }
    if config.kv_format != KvFormat::F32 {
        return Err(unsupported("only the f32 KV format is supported"));
    }
    if config.max_requests > 1 {
        return Err(unsupported("max_requests > 1 is not supported"));
    }
    if config.max_decode_tokens > 1 {
        return Err(unsupported("max_decode_tokens > 1 is not supported"));
    }
    Ok(())
}

/// The engine's buffers in the budget plan, on the existing accounts:
/// QSA K / V → `Kv`; Gated DeltaNet state plus the host family state
/// (indexer keys, PLE state, n-gram row cache) → `LinearState`; the
/// decode expert staging (K demand slots + two prefetch sets) →
/// `ExpertSlots`; the prefill expert staging → `PrefillStaging`;
/// everything sized by the prefill chunk → `PrefillScratch`. No KV
/// scratch, no multi-decode buffers.
///
/// Returns the prefill settings the engine must be built with: the
/// environment's ([`Q38PrefillIo::for_root`]), shrunk while the plan
/// does not fit `cap` (the chunk halved down to 64 tokens), so the
/// hot tier is what gives way last.
pub(crate) fn plan(
    plan: &mut BudgetPlan,
    n_ctx: usize,
    expert_size: u64,
    root: &std::path::Path,
    prefetch_experts: usize,
    cap: u64,
) -> Result<Q38PrefillIo, RsError> {
    let ngram = super::model_layer::qwen3_8_flash_next::ngram::NgramTable::open(
        &engine::ngram_path(root),
        engine::ngram_cache_rows(),
    )
    .map_err(|e| unsupported(&format!("n-gram table: {e}")))?;
    let mut io = Q38PrefillIo::for_root(root, n_ctx, expert_size as usize);
    let want = io;
    loop {
        let p: Q38PlanBytes =
            engine::plan_bytes_io(n_ctx, expert_size as usize, ngram.cache_bytes(), prefetch_experts, &io);
        plan.set(Account::Kv, p.kv);
        plan.set(Account::KvScratch, 0);
        plan.set(Account::LinearState, p.linear_state + p.family_state);
        plan.set(Account::ExpertSlots, p.expert_slots);
        plan.set(Account::PrefillStaging, p.prefill_staging);
        plan.set(Account::PrefillScratch, p.scratch);
        plan.set(Account::MultiDecode, 0);
        if plan.check(cap).is_ok() || !io.shrink() {
            break;
        }
    }
    if io != want {
        eprintln!("[q38] prefill settings {want:?} do not fit the cap; using {io:?}");
    }
    Ok(io)
}

fn eval_err(e: Q38EngineError) -> RsError {
    eprintln!("[q38] eval failed: {e}");
    RsError::EvalFailed
}

fn snap_err(e: Q38EngineError) -> StateSnapshotError {
    match e {
        Q38EngineError::Snapshot(s) => s,
        other => {
            eprintln!("[q38] state op failed: {other}");
            StateSnapshotError::BuffersNotReady
        }
    }
}

impl RsCtx<MetalBackend> {
    /// Build the backend and the engine (idempotent): every buffer is
    /// allocated and zeroed here, so under a cap this runs at open.
    pub(crate) fn ensure_q38(&mut self) -> Result<(), RsError> {
        if self.q38.engine.is_some() {
            return Ok(());
        }
        self.ensure_backend()?;
        let es = self.experts.expert_size();
        // Speculative expert prefetch with the context's settings
        // (`prefetch_experts`, `prefetch_lookahead`); this family's
        // decode always preads, so it runs in either IO mode.
        let mut pf = PrefetchState::new(crate::riir::variants::VARIANT.num_layers, ExpertIoMode::Pread);
        pf.set_max_experts(self.prefetch.max_experts());
        pf.set_lookahead(self.prefetch.lookahead());
        pf.set_guess(self.prefetch.guess());
        pf.set_threads_from_env();
        let Self { backend, wf, q38, n_ctx, .. } = self;
        let b = backend.as_mut().expect("just built");
        let io = q38.io.unwrap_or_else(|| Q38PrefillIo::for_root(&q38.root, *n_ctx, es));
        let e = Q38Engine::with_options(b, wf, &q38.root, *n_ctx, es, Some(pf), io).map_err(|e| {
            eprintln!("[q38] engine init failed: {e}");
            RsError::InitFailed
        })?;
        q38.engine = Some(e);
        Ok(())
    }

    /// `step_internal` / the per-token path for this family: tokens
    /// only, positions continuing the sequence.
    pub(crate) fn q38_step(
        &mut self,
        input: StageInput<'_>,
        start_pos: i32,
        output: StageOutput<'_>,
    ) -> Result<(), RsError> {
        let StageInput::Tokens(tokens) = input else {
            return Err(RsError::StageMismatch {
                detail: "Qwen3.8-Flash-Next takes token input only (no hidden-row stages)".into(),
            });
        };
        let logits = match output {
            StageOutput::Logits(l) => Some(l),
            StageOutput::Discard => None,
            StageOutput::Hidden(_) => {
                return Err(RsError::StageMismatch {
                    detail: "Qwen3.8-Flash-Next has no hidden-row stage output".into(),
                })
            }
        };
        self.ensure_q38()?;
        let Self { backend, wf, experts, io_pool, q38, .. } = self;
        let e = q38.engine.as_mut().expect("just built");
        let at = e.pos() as i32;
        if start_pos != at {
            return Err(RsError::StageMismatch {
                detail: format!(
                    "Qwen3.8-Flash-Next continues at position {at}, got {start_pos}; \
                     memory_clear or restore_to first"
                ),
            });
        }
        let b = backend.as_mut().expect("built with the engine");
        let r = e.eval(b, wf, experts, io_pool, tokens, logits);
        // A failed token can leave prefetch reads in flight.
        e.drain_prefetch();
        r.map_err(eval_err)
    }

    /// Prefetch `(hits, misses)` of the engine ([`RsCtx::prefetch_stats`]).
    pub(crate) fn q38_prefetch_stats(&self) -> (u64, u64) {
        self.q38.engine.as_ref().map_or((0, 0), |e| e.prefetch_stats())
    }

    pub(crate) fn q38_reset_prefetch_stats(&self) {
        if let Some(e) = self.q38.engine.as_ref() {
            e.reset_prefetch_stats();
        }
    }

    pub(crate) fn q38_clear(&mut self) {
        let Self { backend, q38, .. } = self;
        if let (Some(e), Some(b)) = (q38.engine.as_mut(), backend.as_mut()) {
            if let Err(err) = e.reset(b) {
                eprintln!("[q38] reset failed: {err}");
            }
        }
    }

    /// Only a truncation to the start (`p0 <= 0`) or one past the end
    /// is possible.
    pub(crate) fn q38_seq_rm(&mut self, p0: i32, p1: i32) -> bool {
        let pos = self.q38.engine.as_ref().map_or(0, |e| e.pos()) as i32;
        let p1 = if p1 < 0 { i32::MAX } else { p1 };
        if p0 >= pos || p1 <= p0 {
            return true;
        }
        if p0 <= 0 && p1 >= pos {
            self.q38_clear();
            return true;
        }
        false
    }

    pub(crate) fn q38_pos_max(&self) -> i32 {
        self.q38.engine.as_ref().map_or(-1, |e| e.pos() as i32 - 1)
    }

    pub(crate) fn q38_state_size(&self) -> usize {
        self.q38.engine.as_ref().map_or(engine::state_size_at(0), |e| e.state_size())
    }

    pub(crate) fn q38_state_save(&mut self, buf: &mut [u8]) -> Result<usize, StateSnapshotError> {
        self.ensure_q38().map_err(|_| StateSnapshotError::BuffersNotReady)?;
        let Self { backend, q38, .. } = self;
        let e = q38.engine.as_ref().expect("just built");
        e.state_save(backend.as_ref().expect("built"), buf).map_err(snap_err)
    }

    pub(crate) fn q38_state_load(&mut self, buf: &[u8]) -> Result<(), StateSnapshotError> {
        self.ensure_q38().map_err(|_| StateSnapshotError::BuffersNotReady)?;
        let Self { backend, q38, .. } = self;
        let e = q38.engine.as_mut().expect("just built");
        e.state_load(backend.as_mut().expect("built"), buf).map_err(snap_err)
    }
}
