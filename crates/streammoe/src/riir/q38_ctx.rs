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
//! 64; [`engine::prefill_chunk`]) through [`Q38Engine::eval`].

use std::path::PathBuf;

use super::budget::{Account, BudgetPlan};
use super::config::OpenConfig;
use super::model_layer::qwen3_8_flash_next::engine::{self, Q38Engine, Q38EngineError, Q38PlanBytes};
use super::snapshot::state_snapshot::StateSnapshotError;
use super::{KvFormat, MetalBackend, RsCtx, RsError, StageInput, StageOutput};

/// The family engine and where its model lives.
pub(crate) struct Q38Slot {
    /// `experts_dir` (holds `ngram.mfng` and `packed_experts/`).
    pub root: PathBuf,
    pub engine: Option<Q38Engine<MetalBackend>>,
}

impl Q38Slot {
    pub fn new(root: &std::path::Path) -> Self {
        Self { root: root.to_path_buf(), engine: None }
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
/// K-expert staging → `ExpertSlots`; per-token scratch at the prefill
/// chunk's rows → `PrefillScratch`. No prefill staging (a chunk's
/// experts go through the same `top_k` slots, `top_k` at a time), no KV
/// scratch, no multi-decode buffers.
pub(crate) fn plan(plan: &mut BudgetPlan, n_ctx: usize, expert_size: u64, root: &std::path::Path) -> Result<Q38PlanBytes, RsError> {
    let ngram = super::model_layer::qwen3_8_flash_next::ngram::NgramTable::open(
        &engine::ngram_path(root),
        engine::ngram_cache_rows(),
    )
    .map_err(|e| unsupported(&format!("n-gram table: {e}")))?;
    let p = engine::plan_bytes(n_ctx, expert_size as usize, ngram.cache_bytes());
    plan.set(Account::Kv, p.kv);
    plan.set(Account::KvScratch, 0);
    plan.set(Account::LinearState, p.linear_state + p.family_state);
    plan.set(Account::ExpertSlots, p.expert_slots);
    plan.set(Account::PrefillStaging, 0);
    plan.set(Account::PrefillScratch, p.scratch);
    plan.set(Account::MultiDecode, 0);
    Ok(p)
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
        let Self { backend, wf, q38, n_ctx, .. } = self;
        let b = backend.as_mut().expect("just built");
        let e = Q38Engine::new(b, wf, &q38.root, *n_ctx, es).map_err(|e| {
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
        e.eval(b, wf, experts, io_pool, tokens, logits).map_err(eval_err)
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
