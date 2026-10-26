//! Runtime open-time configuration — [`OpenConfig`].
//!
//! Model *shape* stays compile-time ([`crate::riir::variants`]);
//! everything a deployment needs to tune per machine lives here:
//! context length (which sizes every KV cache), the expert-IO path,
//! and the IO thread count. [`crate::riir::RsCtx::open`] is
//! [`crate::riir::RsCtx::open_with`] with [`OpenConfig::default`],
//! so existing callers see no behavior change.

pub use crate::riir::attn::kv_quant::KvFormat;
use crate::riir::budget::DEFAULT_RUNTIME_OVERHEAD;
use crate::riir::io::expert_io_mode::ExpertIoMode;
pub use crate::riir::io::prefetch::PrefetchGuess;
use crate::riir::variants::MAX_SEQ_LEN;

/// Default IO worker-thread count — the historical hard-coded value.
pub const DEFAULT_IO_THREADS: usize = 8;

/// Default [`OpenConfig::prefill_chunk`] under a `mem_cap`. The
/// batched-prefill scratch is linear in the chunk (≈ 0.32 MiB per
/// token for OLMoE-1B-7B, ≈ 0.45 MiB for Qwen3.6-35B-A3B), so the
/// uncapped [`crate::riir::BATCHED_CHUNK_SIZE`] (8192) pins 2.6 GiB
/// (OLMoE) / 3.6 GiB (a3b). 512 keeps it at 165 / 232 MiB; longer
/// prompts run as several chunks.
pub const CAPPED_PREFILL_CHUNK: usize = 512;

/// Default [`OpenConfig::attn_threads`]. On an M3 Pro (OLMoE shape,
/// 2,700 positions) decode SDPA per layer took 4.0 ms serial, 2.4 /
/// 1.1 / 1.0 ms on 2 / 4 / 6 threads: 4 gets most of it while leaving
/// cores to other work on the machine.
pub const DEFAULT_ATTN_THREADS: usize = 4;

/// Default [`OpenConfig::prefetch_experts`] under a `mem_cap`: the
/// strongest 4 guesses per layer (see that field for the numbers).
pub const DEFAULT_CAPPED_PREFETCH_EXPERTS: usize = 4;

/// The prefill chunk a context with `n_ctx` positions uses when
/// [`OpenConfig::prefill_chunk`] is `None`:
/// [`crate::riir::BATCHED_CHUNK_SIZE`] uncapped,
/// [`CAPPED_PREFILL_CHUNK`] under a cap, never more than `n_ctx`.
pub fn default_prefill_chunk(n_ctx: usize, capped: bool) -> usize {
    let chunk = if capped {
        CAPPED_PREFILL_CHUNK
    } else {
        crate::riir::BATCHED_CHUNK_SIZE
    };
    chunk.min(n_ctx)
}

/// Open-time knobs for [`crate::riir::RsCtx::open_with`].
///
/// Construct with [`OpenConfig::default`] and override fields:
///
/// ```ignore
/// let cfg = OpenConfig { n_ctx: 32_768, ..OpenConfig::default() };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenConfig {
    /// Maximum positions per sequence. Sizes every full-attention KV
    /// cache (`n_ctx × num_kv_heads × head_dim × 4 B` per side per
    /// layer). Must be in `1..=MAX_SEQ_LEN`. The default,
    /// `MAX_SEQ_LEN`, keeps the historical lazily-committed 1M-row
    /// reservation; set it to what you actually need.
    pub n_ctx: usize,
    /// Expert-IO path. `None` defers to `STREAMMOE_EXPERT_IO`, then to
    /// the RAM gate in [`crate::riir::io::expert_io_mode::select`].
    /// `Some(_)` wins over the env var.
    pub expert_io: Option<ExpertIoMode>,
    /// Worker threads for parallel expert `pread` and prefetch.
    /// Must be ≥ 1. Pool reads block, so this also bounds the reads
    /// the pool has in flight.
    pub io_threads: usize,
    /// Maximum expert reads in flight at once across every reader —
    /// decode misses, speculative prefetch, prefill staging and hot-
    /// tier seeding ([`crate::riir::io::io_limit::IoGate`]). `None`
    /// (default) adds no bound beyond `io_threads`. Must be ≥ 1.
    pub io_queue_depth: Option<usize>,
    /// Optional limit on expert-read bandwidth, bytes per second
    /// ([`crate::riir::io::io_limit::RateLimiter`]); speculative
    /// prefetch counts against it like any other read. `None`
    /// (default) is unlimited. Must be ≥ 1.
    pub io_bytes_per_sec: Option<u64>,
    /// Hard memory cap in bytes. `None` (default) = no cap, historical
    /// behavior. `Some(cap)`:
    /// - expert IO is forced to [`ExpertIoMode::Pread`]; asking for
    ///   `Mmap` is a config error (mmap'd pages escape accounting);
    /// - `open_with` fails if the [`crate::riir::budget::BudgetPlan`]
    ///   exceeds the cap;
    /// - every resource is built and the KV caches touched at open, then
    ///   the measured footprint is checked against the cap;
    /// - each `eval_*` call re-checks the footprint.
    pub mem_cap: Option<u64>,
    /// Bytes reserved inside `mem_cap` for what the budget plan does
    /// not itemize (Metal/driver state, small fixed buffers, heap).
    /// The prefill scratch is planned separately (`prefill_scratch`).
    /// Default [`DEFAULT_RUNTIME_OVERHEAD`] (512 MiB, measured + margin;
    /// a capped open logs the measured `unplanned` bytes). Only used
    /// when `mem_cap` is set.
    pub runtime_overhead: u64,
    /// Size of the RAM hot tier for experts
    /// ([`crate::riir::io::hot_tier`]). `None`: under a cap, everything
    /// left after the budget plan; without a cap, no hot tier.
    /// `Some(0)` disables it. Requires `Pread` expert IO.
    pub hot_tier_bytes: Option<u64>,
    /// Routing profile to seed the hot tier from at open (written by
    /// [`crate::riir::RsCtx::save_routing_profile`]). Most-picked
    /// experts are loaded first.
    pub hot_tier_profile: Option<std::path::PathBuf>,
    /// Optional fast tier for expert files (plan phase 6, "disk
    /// tier"): any `packed_experts/layer_NN.bin` found under this
    /// directory — e.g. a copy of the hottest layers on the internal
    /// SSD — is read from there instead of `experts_dir`. Same format,
    /// per-layer granularity; costs disk, not RAM. A copy whose size
    /// differs from the primary file fails `open`. `None` (default):
    /// everything from `experts_dir`.
    pub fast_experts_dir: Option<std::path::PathBuf>,
    /// Contiguous layer range this context runs (plan phase 3, mode B).
    /// `None` (default) = the whole model. `Some(a..b)` loads, plans and
    /// runs only layers `a..b`; the embedding exists only when `a == 0`
    /// and the final norm + LM head only when `b == num_layers`. Drive a
    /// partial range with [`crate::riir::RsCtx::eval_stage`]; the
    /// whole-model `eval_prompt` / `eval_token` need the full range.
    pub layer_range: Option<std::ops::Range<usize>>,
    /// Requests (sequences) the context can hold at once (plan phase
    /// 4). Each gets its own KV caches and linear-attn state for the
    /// layer range, all allocated at open and counted in the budget
    /// plan; admission beyond this fails cleanly. Must be ≥ 1; the
    /// default, 1, is the single-sequence context.
    pub max_requests: usize,
    /// How speculative decode prefetch guesses each layer's experts
    /// (plan phase 6). Default [`PrefetchGuess::PrevToken`] is the
    /// historical behavior; [`PrefetchGuess::CrossLayer`] predicts
    /// layer L+1 from layer L's residual. Bit-identical logits either
    /// way; only pread-mode GQA decode prefetches.
    pub prefetch_guess: PrefetchGuess,
    /// Issue layer L+1's speculative prefetch while layer L's
    /// attention runs on the GPU, instead of at the top of L+1 (plan
    /// phase 6 overlap). Uses the two existing prefetch sets — no new
    /// buffers. Needs `prefetch_guess = PrevToken` (a cross-layer
    /// guess is only known after L's attention). Default `false`.
    pub prefetch_lookahead: bool,
    /// Experts per layer each speculative decode prefetch reads, the
    /// strongest guesses first (previous token: highest router
    /// weight; cross-layer: highest guessed score). `Some(0)` turns
    /// speculative prefetch off (misses are read on demand only).
    /// `None` (default): [`DEFAULT_CAPPED_PREFETCH_EXPERTS`] under a
    /// cap, else all K. Never changes the logits.
    ///
    /// Under a cap every wrong guess costs SSD bandwidth the demand
    /// reads need: on OLMoE-1B-7B at a 6 GiB cap, reading all 8
    /// guesses wasted 42% of decode reads, and top 4 was the fastest
    /// setting (8.5 vs 7.4 tok/s with a 0.35 GiB hot tier, 11.8 vs
    /// 11.3 with 1.19 GiB); prefetch off read least but stalled on
    /// every miss.
    pub prefetch_experts: Option<usize>,
    /// Bytes of fixed buffers the embedding application holds next to
    /// the context — e.g. `streammoe-net`'s frame buffers (plan phase 5).
    /// Counted in the budget plan as `host_buffers`, so the cap covers
    /// them; allocate (and touch) them before `open_with` so the
    /// at-open footprint check sees them. Only used with `mem_cap`.
    pub host_buffers: u64,
    /// Tokens per batched-prefill chunk. Every prefill scratch buffer
    /// (attention / MoE graph stacks, hidden double buffer, token ids)
    /// is sized for one chunk; longer prompts run as several chunks.
    /// `None`: [`default_prefill_chunk`] — 8192 uncapped (the
    /// historical behavior), [`CAPPED_PREFILL_CHUNK`] under `mem_cap`.
    /// Always clamped to `n_ctx`. Must be ≥ 1. Counted in the budget
    /// plan as `prefill_scratch`.
    pub prefill_chunk: Option<usize>,
    /// Storage format of the GQA KV cache ([`KvFormat`]). `F32`
    /// (default) is the historical layout; `F16` halves the cache,
    /// `Turbo4` / `Turbo3` (TurboQuant) shrink it ~7.5× / ~9.8× at a
    /// small accuracy cost. Counted per format in the budget plan.
    /// Non-F32 formats run decode attention on the CPU path. MLA
    /// variants accept only `F32`.
    pub kv_format: KvFormat,
    /// Threads the CPU decode attention (SDPA) splits its heads
    /// across; `0` / `1` run it on the calling thread. Default
    /// [`DEFAULT_ATTN_THREADS`]. Output is bit-identical for any
    /// value. This path runs whenever the GPU SDPA fast path does not
    /// (pread / capped runs, any non-F32 `kv_format`, kv_len < 32);
    /// at long context it is a large share of decode time. Thread
    /// stacks are inside `runtime_overhead`.
    pub attn_threads: usize,
    /// Snapshots `checkpoint_pos` keeps (LRU, pos 0 protected).
    /// Under `mem_cap` this many buffers of the largest state at
    /// `n_ctx` are planned (`checkpoints` account), allocated and
    /// touched at open, and reused — checkpoints never allocate after
    /// open. `set_max_checkpoints` can lower it later but not raise it
    /// past this under a cap. Default
    /// [`crate::riir::DEFAULT_MAX_CHECKPOINTS`].
    pub max_checkpoints: usize,
}

impl Default for OpenConfig {
    fn default() -> Self {
        Self {
            n_ctx: MAX_SEQ_LEN,
            expert_io: None,
            io_threads: DEFAULT_IO_THREADS,
            io_queue_depth: None,
            io_bytes_per_sec: None,
            mem_cap: None,
            runtime_overhead: DEFAULT_RUNTIME_OVERHEAD,
            hot_tier_bytes: None,
            hot_tier_profile: None,
            fast_experts_dir: None,
            layer_range: None,
            max_requests: 1,
            prefetch_guess: PrefetchGuess::PrevToken,
            prefetch_lookahead: false,
            prefetch_experts: None,
            host_buffers: 0,
            prefill_chunk: None,
            kv_format: KvFormat::F32,
            attn_threads: DEFAULT_ATTN_THREADS,
            max_checkpoints: crate::riir::DEFAULT_MAX_CHECKPOINTS,
        }
    }
}

impl OpenConfig {
    /// Check field ranges. Returns a human-readable reason on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.n_ctx == 0 || self.n_ctx > MAX_SEQ_LEN {
            return Err(format!(
                "n_ctx {} out of range 1..={MAX_SEQ_LEN}",
                self.n_ctx
            ));
        }
        if self.io_threads == 0 {
            return Err("io_threads must be ≥ 1".into());
        }
        if self.max_requests == 0 {
            return Err("max_requests must be ≥ 1".into());
        }
        if self.prefill_chunk == Some(0) {
            return Err("prefill_chunk must be ≥ 1".into());
        }
        if self.io_queue_depth == Some(0) {
            return Err("io_queue_depth must be ≥ 1".into());
        }
        if self.io_bytes_per_sec == Some(0) {
            return Err("io_bytes_per_sec must be ≥ 1".into());
        }
        if self.prefetch_lookahead && self.prefetch_guess != PrefetchGuess::PrevToken {
            return Err(
                "prefetch_lookahead needs prefetch_guess = PrevToken (a cross-layer \
                 guess is only known after the previous layer's attention)"
                    .into(),
            );
        }
        if self.hot_tier_bytes.is_some_and(|b| b > 0) && self.expert_io == Some(ExpertIoMode::Mmap)
        {
            return Err("hot_tier_bytes needs expert_io = Pread (mmap mode binds the files)".into());
        }
        if self.mem_cap.is_some() && self.expert_io == Some(ExpertIoMode::Mmap) {
            return Err(
                "expert_io = Mmap is not allowed with mem_cap: mmap'd expert \
                 pages are file-backed and escape the cap"
                    .into(),
            );
        }
        if let Some(r) = &self.layer_range {
            crate::riir::range::validate(r)?;
        }
        self.kv_format.validate(&crate::riir::variants::VARIANT)?;
        Ok(())
    }

    /// The prefill chunk to use: [`Self::prefill_chunk`] or
    /// [`default_prefill_chunk`], clamped to `n_ctx`.
    pub fn resolved_prefill_chunk(&self) -> usize {
        match self.prefill_chunk {
            Some(c) => c.min(self.n_ctx),
            None => default_prefill_chunk(self.n_ctx, self.mem_cap.is_some()),
        }
    }

    /// Experts per layer a speculative prefetch reads:
    /// [`Self::prefetch_experts`], else
    /// [`DEFAULT_CAPPED_PREFETCH_EXPERTS`] under a cap, else all K.
    pub fn resolved_prefetch_experts(&self) -> usize {
        match self.prefetch_experts {
            Some(n) => n,
            None if self.mem_cap.is_some() => DEFAULT_CAPPED_PREFETCH_EXPERTS,
            None => crate::riir::moe::expert_forward::MAX_K,
        }
    }

    /// The layer range to run: [`Self::layer_range`] or the whole model.
    pub fn resolved_layer_range(&self) -> std::ops::Range<usize> {
        self.layer_range
            .clone()
            .unwrap_or_else(crate::riir::range::full_range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_historical_behavior() {
        let c = OpenConfig::default();
        assert_eq!(c.n_ctx, MAX_SEQ_LEN);
        assert_eq!(c.expert_io, None);
        assert_eq!(c.io_threads, DEFAULT_IO_THREADS);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_rejects_out_of_range() {
        let bad_ctx = OpenConfig { n_ctx: 0, ..OpenConfig::default() };
        assert!(bad_ctx.validate().is_err());
        let too_big = OpenConfig { n_ctx: MAX_SEQ_LEN + 1, ..OpenConfig::default() };
        assert!(too_big.validate().is_err());
        let no_threads = OpenConfig { io_threads: 0, ..OpenConfig::default() };
        assert!(no_threads.validate().is_err());
        let no_requests = OpenConfig { max_requests: 0, ..OpenConfig::default() };
        assert!(no_requests.validate().is_err());
        let no_depth = OpenConfig { io_queue_depth: Some(0), ..OpenConfig::default() };
        assert!(no_depth.validate().is_err());
        let no_rate = OpenConfig { io_bytes_per_sec: Some(0), ..OpenConfig::default() };
        assert!(no_rate.validate().is_err());
        let bounded = OpenConfig {
            io_queue_depth: Some(4),
            io_bytes_per_sec: Some(1 << 30),
            ..OpenConfig::default()
        };
        assert!(bounded.validate().is_ok());
        let ahead = OpenConfig { prefetch_lookahead: true, ..OpenConfig::default() };
        assert!(ahead.validate().is_ok());
        let ahead_cross = OpenConfig { prefetch_guess: PrefetchGuess::CrossLayer, ..ahead };
        assert!(ahead_cross.validate().is_err());
    }

    #[test]
    fn prefill_chunk_defaults_and_clamps() {
        let c = OpenConfig { n_ctx: 2048, ..OpenConfig::default() };
        assert_eq!(c.resolved_prefill_chunk(), 2048);
        assert_eq!(OpenConfig::default().resolved_prefill_chunk(), crate::riir::BATCHED_CHUNK_SIZE);
        let capped = OpenConfig { mem_cap: Some(3 << 30), ..c.clone() };
        assert_eq!(capped.resolved_prefill_chunk(), CAPPED_PREFILL_CHUNK);
        let tiny = OpenConfig { n_ctx: 100, ..capped.clone() };
        assert_eq!(tiny.resolved_prefill_chunk(), 100);
        let chosen = OpenConfig { prefill_chunk: Some(1024), ..capped };
        assert_eq!(chosen.resolved_prefill_chunk(), 1024);
        assert!(OpenConfig { prefill_chunk: Some(0), ..c }.validate().is_err());
    }

    #[test]
    fn prefetch_experts_defaults_to_top_4_under_a_cap() {
        let c = OpenConfig::default();
        assert_eq!(c.resolved_prefetch_experts(), crate::riir::moe::expert_forward::MAX_K);
        let capped = OpenConfig { mem_cap: Some(6 << 30), ..c };
        assert_eq!(capped.resolved_prefetch_experts(), DEFAULT_CAPPED_PREFETCH_EXPERTS);
        let off = OpenConfig { prefetch_experts: Some(0), ..capped };
        assert_eq!(off.resolved_prefetch_experts(), 0);
    }

    #[test]
    fn mmap_rejected_under_cap() {
        let c = OpenConfig {
            mem_cap: Some(10 << 30),
            expert_io: Some(ExpertIoMode::Mmap),
            ..OpenConfig::default()
        };
        assert!(c.validate().is_err());
        let ok = OpenConfig { expert_io: Some(ExpertIoMode::Pread), ..c };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn layer_range_validated() {
        let n = crate::riir::variants::VARIANT.num_layers;
        let c = OpenConfig { layer_range: Some(0..n), ..OpenConfig::default() };
        assert!(c.validate().is_ok());
        assert_eq!(OpenConfig::default().resolved_layer_range(), 0..n);
        let empty = OpenConfig { layer_range: Some(3..3), ..OpenConfig::default() };
        assert!(empty.validate().is_err());
        let past = OpenConfig { layer_range: Some(1..n + 1), ..OpenConfig::default() };
        assert!(past.validate().is_err());
    }
}
