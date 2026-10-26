//! Hard memory cap — open-time plan, footprint measurement, checks.
//!
//! A context opened with [`crate::riir::OpenConfig::mem_cap`] set must
//! never push the process past the cap. Enforcement has three layers:
//!
//! 1. **Plan** ([`BudgetPlan::for_config`]) — before anything is
//!    allocated, add up every fixed account from the variant shape and
//!    the config. A plan over the cap fails `open`.
//! 2. **Measure at open** — under a cap, `open_with` builds every
//!    resource eagerly and touches the KV caches, then samples
//!    [`footprint`]. Over the cap fails `open`.
//! 3. **Tripwire** — each `eval_*` call re-samples the footprint and
//!    returns [`crate::riir::RsError::OverBudget`] if it crossed the
//!    cap (an accounting bug, by construction). Both checks compare the
//!    lifetime peak too, so a transient breach since the start of
//!    `open_with` is reported ([`check_footprint`] has the semantics).
//!
//! `phys_footprint` counts anonymous and GPU (IOAccelerator) memory
//! but not clean file-backed pages, so mmap'd model files sit outside
//! it. Under a cap the engine reads experts with `pread` (never
//! mmap-direct); `F_NOCACHE` reads that keep them out of the page
//! cache are part of the same plan (`docs/fork-upgrade-plan.md`).

use std::ops::Range;

use crate::riir::attn::kv_quant::KvFormat;
use crate::riir::io::expert_io_mode::ExpertIoMode;
use crate::riir::moe::expert_forward::MAX_K;
use crate::riir::variants::{AttnKind, LayerKind, VARIANT, Variant};

const GIB: u64 = 1 << 30;
const TWO_MIB: u64 = 2 << 20;

/// Default reservation for what the plan does not itemize: Metal
/// device/driver state and pipelines, the small fixed pool buffers
/// (per-token layer buffers, router/combine scratch, logits), thread
/// stacks, the heap, the driver state of the first GPU submit and
/// per-eval host vectors. Measured on OLMoE-1B-7B (M3 Pro,
/// 2026-10-25): ≈ 60 MiB above the exactly-planned bytes at open, and
/// the peak through a 25-token prefill + 32 decode steps stays within
/// 5 MiB of that. 512 MiB is that plus a > 5× margin for longer
/// prefill chunks and larger variants (`docs/fork-upgrade-plan.md`
/// §10). Only used under a cap.
pub const DEFAULT_RUNTIME_OVERHEAD: u64 = GIB / 2;

/// One named slice of the cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Account {
    /// Non-expert weights (`model_weights.bin`).
    Weights,
    /// Full-attention KV caches at `n_ctx`, for every request slot,
    /// in the context's [`KvFormat`].
    Kv,
    /// The f32 attention scratch a non-F32 KV cache dequantizes into
    /// during batched prefill ([`kv_scratch_bytes`]); one, shared.
    KvScratch,
    /// Preallocated snapshot buffers for `checkpoint_pos` /
    /// `restore_to`: `max_checkpoints` × the largest state at `n_ctx`
    /// (`state_snapshot::state_size_full`).
    Checkpoints,
    /// Linear-attention conv + recurrence state (host + GPU copies),
    /// for every request slot.
    LinearState,
    /// Expert slot ring: synced + two prefetch sets, 2 MiB-aligned.
    ExpertSlots,
    /// Prefill expert staging (pread mode only).
    PrefillStaging,
    /// Batched-prefill scratch sized by the context's prefill chunk
    /// ([`prefill_scratch_bytes`]).
    PrefillScratch,
    /// Metal/driver/runtime overhead and scratch.
    RuntimeOverhead,
    /// RAM hot tier of experts (sized from what the cap leaves).
    HotTier,
    /// Fixed buffers the application holds next to the context
    /// ([`crate::riir::OpenConfig::host_buffers`], e.g. network frames).
    HostBuffers,
}

impl Account {
    /// Stable lowercase name for logs and errors.
    pub fn name(self) -> &'static str {
        match self {
            Account::Weights => "weights",
            Account::Kv => "kv",
            Account::KvScratch => "kv_scratch",
            Account::Checkpoints => "checkpoints",
            Account::LinearState => "linear_state",
            Account::ExpertSlots => "expert_slots",
            Account::PrefillStaging => "prefill_staging",
            Account::PrefillScratch => "prefill_scratch",
            Account::RuntimeOverhead => "runtime_overhead",
            Account::HotTier => "hot_tier",
            Account::HostBuffers => "host_buffers",
        }
    }
}

/// Byte totals per [`Account`] for one context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetPlan {
    /// `(account, bytes)` in a fixed order.
    pub entries: Vec<(Account, u64)>,
}

fn round_up(x: u64, to: u64) -> u64 {
    x.div_ceil(to) * to
}

/// KV bytes per position summed over every full-attn layer.
pub fn kv_bytes_per_token(v: Variant) -> u64 {
    kv_bytes_per_token_in(v, &(0..v.num_layers))
}

/// [`kv_bytes_per_token`] over the full-attn layers in `range` only.
pub fn kv_bytes_per_token_in(v: Variant, range: &Range<usize>) -> u64 {
    kv_bytes_per_token_fmt(v, range, KvFormat::F32)
}

/// [`kv_bytes_per_token_in`] with GQA rows stored as `fmt`.
pub fn kv_bytes_per_token_fmt(v: Variant, range: &Range<usize>, fmt: KvFormat) -> u64 {
    let f32b = std::mem::size_of::<f32>() as u64;
    let per_layer = match v.attn_kind {
        AttnKind::Gqa => 2 * fmt.row_bytes(&v) as u64,
        AttnKind::Mla => (v.kv_lora_rank + v.qk_rope_head_dim) as u64 * f32b,
    };
    full_attn_layers_in(v, range) * per_layer
}

fn full_attn_layers_in(v: Variant, range: &Range<usize>) -> u64 {
    range
        .clone()
        .filter(|&i| v.layer_kind(i) == LayerKind::FullAttn)
        .count() as u64
}

/// [`Account::KvScratch`]: the f32 K + V prefill scratch at `n_ctx`
/// a non-F32 cache needs (`KvCache::f32_scratch`); 0 for F32 or a
/// range without GQA layers.
pub fn kv_scratch_bytes(v: Variant, range: &Range<usize>, n_ctx: usize, fmt: KvFormat) -> u64 {
    if fmt == KvFormat::F32 || v.attn_kind != AttnKind::Gqa || full_attn_layers_in(v, range) == 0 {
        return 0;
    }
    2 * n_ctx as u64 * (v.num_kv_heads * v.head_dim) as u64 * 4
}

/// Linear-attn state bytes (host `LinearAttnState` + GPU mirror).
pub fn linear_state_bytes(v: Variant) -> u64 {
    linear_state_bytes_in(v, &(0..v.num_layers))
}

/// [`linear_state_bytes`] over the linear-attn layers in `range` only.
pub fn linear_state_bytes_in(v: Variant, range: &Range<usize>) -> u64 {
    let f32b = std::mem::size_of::<f32>() as u64;
    let linear_layers = range
        .clone()
        .filter(|&i| v.layer_kind(i) == LayerKind::LinearAttn)
        .count() as u64;
    let conv = ((Variant::CONV_KERNEL_SIZE - 1) * v.linear_conv_dim()) as u64;
    let ssm = (v.linear_num_v_heads * Variant::LINEAR_VALUE_DIM * Variant::LINEAR_KEY_DIM) as u64;
    2 * linear_layers * (conv + ssm) * f32b
}

/// Bytes the batched-prefill scratch structs request from the pool
/// for a `chunk`-token prefill chunk: `LinearAttnGraphScratch`,
/// `FullAttnGraphScratch`, `MoeGraphScratch` (minus its pread
/// `expert_base`, which is [`Account::PrefillStaging`]),
/// `HiddenDoubleBuffer` and `HeadTailScratch`. Mirrors their `new`
/// allocation formulas term for term; `prefill_scratch_matches_pool`
/// keeps the two in step. Zero for MLA variants, which prefill
/// token by token and never build these.
pub fn prefill_scratch_bytes(v: Variant, chunk: usize) -> u64 {
    if v.attn_kind == AttnKind::Mla {
        return 0;
    }
    let f32b = std::mem::size_of::<f32>();
    let hidden = v.hidden_dim;
    let q_dim = v.num_attn_heads * v.head_dim;
    let kv_dim = v.num_kv_heads * v.head_dim;
    let k = v.num_experts_per_tok;

    // LinearAttnGraphScratch — no slots without linear-attn layers.
    let lin_chunk = if v.has_linear_attn() { chunk } else { 0 };
    let num_v = v.linear_num_v_heads;
    let linear = lin_chunk
        * (2 * hidden
            + 2 * v.linear_conv_dim()
            + 2 * v.linear_total_value()
            + 4 * num_v
            + num_v * Variant::LINEAR_VALUE_DIM
            + v.num_experts);

    // FullAttnGraphScratch transients + the inv_freq table (+ the
    // `[chunk, 3]` i32 M-RoPE positions where the variant has them).
    let k_raw = if v.qk_norm == crate::riir::variants::QkNorm::Full { kv_dim } else { 0 };
    let q_gate = if v.attn_output_gate { q_dim } else { 0 };
    let mrope = if crate::riir::variants::MROPE_SECTION.is_some() { 3 } else { 0 };
    let full = chunk
        * (2 * hidden + v.q_proj_dim() + k_raw + 2 * kv_dim + 2 * q_dim + q_gate + v.num_experts + mrope)
        + v.rotary_dim() / 2;

    // MoeGraphScratch, `expert_base` excluded.
    let shared_down = if v.has_shared_expert() { hidden } else { 0 };
    let max_k = MAX_K;
    let moe = chunk
        * (3 * v.shared_intermediate
            + shared_down
            + 3 * k * v.moe_intermediate
            + k * hidden
            + 3 * hidden
            + 1
            + 2 * max_k
            + k * hidden
            + 3 * k
            + v.num_experts.max(1))
        + v.num_experts.max(1);

    // HiddenDoubleBuffer + HeadTailScratch.
    let hdb = 2 * chunk * hidden;
    let hts = chunk + v.vocab_size;

    ((linear + full + moe + hdb + hts) * f32b) as u64
}

impl BudgetPlan {
    /// Plan for the active variant.
    ///
    /// `weights_bytes` is the size of `model_weights.bin`;
    /// `runtime_overhead` is [`crate::riir::OpenConfig::runtime_overhead`].
    pub fn for_config(
        n_ctx: usize,
        mode: ExpertIoMode,
        weights_bytes: u64,
        runtime_overhead: u64,
        prefill_chunk: usize,
    ) -> Self {
        Self::for_range(
            n_ctx,
            mode,
            weights_bytes,
            runtime_overhead,
            &(0..VARIANT.num_layers),
            1,
            prefill_chunk,
        )
    }

    /// [`Self::for_config`] for a context running only the layers in
    /// `range` (plan phase 3) with a pool of `max_requests` sequence
    /// states (plan phase 4): KV and linear-attn state count only those
    /// layers, once per request slot; `weights_bytes` is what the
    /// context loads for them. `prefill_chunk` is
    /// [`crate::riir::OpenConfig::resolved_prefill_chunk`]; the
    /// prefill scratch is shared by every range and request.
    pub fn for_range(
        n_ctx: usize,
        mode: ExpertIoMode,
        weights_bytes: u64,
        runtime_overhead: u64,
        range: &Range<usize>,
        max_requests: usize,
        prefill_chunk: usize,
    ) -> Self {
        let reqs = max_requests.max(1) as u64;
        let v = VARIANT;
        let expert = v.expert_size_4bit() as u64;
        // `MoeBuffers::new`: MAX_K synced + 2 × MAX_K prefetch slots.
        let slots = 3 * MAX_K as u64 * round_up(expert, TWO_MIB);
        let staging = if mode.is_pread() {
            v.num_experts as u64 * expert
        } else {
            0
        };
        Self {
            entries: vec![
                (Account::Weights, weights_bytes),
                (Account::Kv, reqs * kv_bytes_per_token_in(v, range) * n_ctx as u64),
                (Account::LinearState, reqs * linear_state_bytes_in(v, range)),
                (Account::ExpertSlots, slots),
                (Account::PrefillStaging, staging),
                (Account::PrefillScratch, prefill_scratch_bytes(v, prefill_chunk)),
                (Account::RuntimeOverhead, runtime_overhead),
            ],
        }
    }

    /// Add (or replace) one account.
    pub fn set(&mut self, account: Account, bytes: u64) {
        match self.entries.iter_mut().find(|(a, _)| *a == account) {
            Some(e) => e.1 = bytes,
            None => self.entries.push((account, bytes)),
        }
    }

    /// Sum of every account.
    pub fn total(&self) -> u64 {
        self.entries.iter().map(|&(_, b)| b).sum()
    }

    /// Bytes for one account (0 if absent).
    pub fn get(&self, account: Account) -> u64 {
        self.entries
            .iter()
            .find(|&&(a, _)| a == account)
            .map_or(0, |&(_, b)| b)
    }

    /// Fail if the plan exceeds `cap`, naming every account.
    pub fn check(&self, cap: u64) -> Result<(), String> {
        let total = self.total();
        if total <= cap {
            return Ok(());
        }
        Err(format!(
            "planned {} exceeds cap {} ({})",
            fmt_gib(total),
            fmt_gib(cap),
            self.breakdown()
        ))
    }

    /// `name=1.23 GiB, …` for logs and errors.
    pub fn breakdown(&self) -> String {
        self.entries
            .iter()
            .map(|&(a, b)| format!("{}={}", a.name(), fmt_gib(b)))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// `1.23 GiB`.
pub fn fmt_gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / GIB as f64)
}

/// Process memory as the OS accounts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footprint {
    /// Current `phys_footprint` in bytes.
    pub current: u64,
    /// Lifetime peak `phys_footprint` in bytes.
    pub peak: u64,
}

/// Sample this process's `phys_footprint` (what Activity Monitor's
/// "Memory" column and jetsam use). `None` if the syscall fails.
pub fn footprint() -> Option<Footprint> {
    let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a valid, writable `rusage_info_v4`, matching
    // the `RUSAGE_INFO_V4` flavor; the kernel writes at most its size.
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            (&mut info as *mut libc::rusage_info_v4).cast(),
        )
    };
    if rc != 0 {
        return None;
    }
    Some(Footprint {
        current: info.ri_phys_footprint,
        peak: info.ri_lifetime_max_phys_footprint,
    })
}

/// Fail if the current footprint, or the peak reached since the
/// context started opening, is over `cap`. A failed sample is treated
/// as a failure — under a hard cap, unknown is not safe.
///
/// The OS only keeps a *lifetime* peak (`ri_lifetime_max_phys_footprint`,
/// since process start). `baseline_peak` is that peak sampled at the
/// start of `open_with`: a peak above it was reached during this
/// context's life and is checked against the cap, so a transient
/// breach between two checks is reported, not missed. A peak no
/// higher than the baseline predates the context (an earlier, larger
/// context in the same process) and is ignored — which also means a
/// transient that stays below that older peak cannot be seen.
pub fn check_footprint(cap: u64, baseline_peak: u64, when: &str) -> Result<Footprint, String> {
    let fp = footprint().ok_or_else(|| format!("{when}: could not read phys_footprint"))?;
    if fp.current > cap {
        return Err(format!(
            "{when}: footprint {} exceeds cap {} (peak {})",
            fmt_gib(fp.current),
            fmt_gib(cap),
            fmt_gib(fp.peak)
        ));
    }
    if fp.peak > baseline_peak && fp.peak > cap {
        return Err(format!(
            "{when}: peak footprint {} since open exceeds cap {} (now {})",
            fmt_gib(fp.peak),
            fmt_gib(cap),
            fmt_gib(fp.current)
        ));
    }
    Ok(fp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_scales_with_n_ctx() {
        let a = BudgetPlan::for_config(1024, ExpertIoMode::Pread, 0, 0, 512);
        let b = BudgetPlan::for_config(2048, ExpertIoMode::Pread, 0, 0, 512);
        assert_eq!(b.get(Account::Kv), 2 * a.get(Account::Kv));
        assert_eq!(a.get(Account::Kv), 1024 * kv_bytes_per_token(VARIANT));
    }

    #[test]
    fn staging_only_in_pread_mode() {
        let p = BudgetPlan::for_config(1, ExpertIoMode::Pread, 0, 0, 1);
        let m = BudgetPlan::for_config(1, ExpertIoMode::Mmap, 0, 0, 1);
        assert!(p.get(Account::PrefillStaging) > 0);
        assert_eq!(m.get(Account::PrefillStaging), 0);
    }

    #[test]
    fn total_and_check() {
        let p = BudgetPlan::for_config(8, ExpertIoMode::Pread, 1000, 2000, 8);
        assert_eq!(p.total(), p.entries.iter().map(|e| e.1).sum::<u64>());
        assert!(p.check(p.total()).is_ok());
        let err = p.check(p.total() - 1).unwrap_err();
        assert!(err.contains("weights=") && err.contains("kv="), "{err}");
    }

    /// a3b numbers from the plan doc: 40 KiB of KV per token.
    #[cfg(feature = "model-qwen3-6-35b-a3b")]
    #[test]
    fn a3b_kv_per_token_is_40_kib() {
        assert_eq!(kv_bytes_per_token(VARIANT), 40 * 1024);
    }

    /// A split's per-range KV and linear state add up to the whole
    /// model's.
    #[test]
    fn range_accounts_partition_the_model() {
        let n = VARIANT.num_layers;
        let full = BudgetPlan::for_config(512, ExpertIoMode::Pread, 0, 0, 512);
        let cuts = [0, n / 2, n * 3 / 4, n];
        let (mut kv, mut lin) = (0, 0);
        for w in cuts.windows(2) {
            let p = BudgetPlan::for_range(512, ExpertIoMode::Pread, 0, 0, &(w[0]..w[1]), 1, 512);
            kv += p.get(Account::Kv);
            lin += p.get(Account::LinearState);
        }
        assert_eq!(kv, full.get(Account::Kv));
        assert_eq!(lin, full.get(Account::LinearState));
    }

    /// Per-request state scales with the pool; shared accounts don't.
    #[test]
    fn request_pool_scales_state_accounts() {
        let r = 0..VARIANT.num_layers;
        let one = BudgetPlan::for_range(256, ExpertIoMode::Pread, 7, 9, &r, 1, 256);
        let four = BudgetPlan::for_range(256, ExpertIoMode::Pread, 7, 9, &r, 4, 256);
        assert_eq!(four.get(Account::Kv), 4 * one.get(Account::Kv));
        assert_eq!(four.get(Account::LinearState), 4 * one.get(Account::LinearState));
        for a in [Account::Weights, Account::ExpertSlots, Account::PrefillStaging, Account::PrefillScratch] {
            assert_eq!(four.get(a), one.get(a), "{}", a.name());
        }
    }

    /// The plan's `prefill_scratch` is exactly what the scratch
    /// structs request from a real Metal pool, at several chunks.
    #[test]
    fn prefill_scratch_matches_pool() {
        use crate::riir::attn::full_attn_forward::FullAttnGraphScratch;
        use crate::riir::attn::linear_attn_forward::{
            HeadTailScratch, HiddenDoubleBuffer, LinearAttnGraphScratch, MoeGraphScratch,
        };
        use crate::riir::backend::MetalBufferPool;

        let v = VARIANT;
        if v.attn_kind == AttnKind::Mla {
            assert_eq!(prefill_scratch_bytes(v, 512), 0);
            return;
        }
        let device = metal::Device::system_default().expect("Metal device");
        for chunk in [1, 37, 512] {
            let mut pool = MetalBufferPool::new(device.clone());
            let _lin = LinearAttnGraphScratch::new(&mut pool, chunk);
            let _full = FullAttnGraphScratch::new(&mut pool, chunk);
            // Mmap mode: no `expert_base` (that is `prefill_staging`).
            let _moe = MoeGraphScratch::new(&mut pool, v.num_experts_per_tok, ExpertIoMode::Mmap, chunk);
            let _hdb = HiddenDoubleBuffer::new(&mut pool, chunk);
            let _hts = HeadTailScratch::new(&mut pool, chunk);
            let want = prefill_scratch_bytes(v, chunk);
            assert_eq!(pool.requested_bytes(), want, "chunk {chunk}");
            // Physical buffers only add page rounding (and the 1-byte
            // floor for empty slots).
            let phys = pool.total_bytes();
            assert!(phys >= want && phys - want <= TWO_MIB, "chunk {chunk}: phys {phys} want {want}");
        }
    }

    #[test]
    fn footprint_is_readable_and_nonzero() {
        let fp = footprint().expect("proc_pid_rusage");
        assert!(fp.current > 0);
        // Not `peak >= current`: the kernel updates the lifetime max
        // lazily, so with other test threads allocating the current
        // footprint can read briefly above it (seen intermittently).
        assert!(fp.peak > 0);
        assert!(check_footprint(u64::MAX, 0, "test").is_ok());
        assert!(check_footprint(1, 0, "test").is_err());
        // A peak from before the baseline is not this context's.
        let below = fp.peak.max(fp.current) + (1 << 30);
        assert!(check_footprint(below, u64::MAX, "test").is_ok());
        // One reached after it is, even once the current is back down.
        // (Only the error side is race-free: the peak is monotone, but
        // parallel tests can raise it past any baseline sampled here.)
        let between = (fp.current + fp.peak) / 2;
        if fp.peak > fp.current + 2 {
            assert!(check_footprint(between, 0, "test").is_err());
        }
    }
}
