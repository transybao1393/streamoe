//! QSA over the budget on the GPU: block-selected attention as one
//! Metal kernel over per-query row lists.
//!
//! Past `budget + compress_ratio - 1` visible tokens each query attends
//! over its own subset of the cache, so the dense `SdpaCausalTiled`
//! does not apply. The split here:
//!
//! 1. Host: [`qsa_select_rows`] — the indexer's block scores and top-k
//!    per query ([`QsaIndexerCache::select`], queries in parallel), as
//!    a `[n, max_rows]` row-index table plus a count per query. A query
//!    still within the budget gets all its visible rows, so one table
//!    covers a chunk that crosses the limit.
//! 2. GPU: [`QsaGpuKernels::encode_sdpa_rows`] — `q38_qsa_sdpa_rows`
//!    gathers each query's K / V rows straight from the f32 caches and
//!    runs GQA softmax attention plus the sigmoid output gate, writing
//!    the `o_proj` input. It replaces [`super::push_qsa_sdpa`] for that
//!    chunk; `q` / `gate` / the caches are the buffers
//!    [`super::push_qsa_pre_sdpa`] filled.
//!
//! The engine flow for a chunk with [`super::qsa_needs_sparse`]:
//! `push_qsa_pre_sdpa` → commit, read back `idx_qk` →
//! [`super::qsa_indexer_step`] → [`qsa_select_rows`] →
//! [`QsaRowBuffers::upload`] → `encode_sdpa_rows` → `push_qsa_o_proj`.
//!
//! The scoring stays on the CPU: it is `n_blocks × idx_heads ×
//! idx_head_dim` MACs per query (~0.8 M at 8k context) against
//! `heads × 2051 × head_dim × 2` ≈ 25 M for the attention the kernel
//! takes over. Measured (`select_rows_cpu_cost`, ignored test, release,
//! M3 Pro, real shape, one layer):
//!
//! | context | selection, all queries (11 threads) | attention of 64 sparse queries: CPU 1 thread / GPU kernel |
//! | --- | --- | --- |
//! | 4,096 | 0.05 s (12 µs / query) | 440 ms / 7.6 ms |
//! | 8,192 | 0.22–0.26 s (27–32 µs / query) | 460–480 ms / 7.6 ms |
//!
//! So an 8k prompt spends ~3 s in selection and ~8.5 s in this kernel
//! over the 12 QSA layers (6,141 sparse queries each). The kernel is
//! at ~7% of f32 peak: each (token, 4-head group) threadgroup gathers
//! its own rows; tiling K / V across the queries of a chunk (they
//! share most blocks) is the next step if it matters.
//!
//! The kernel's online softmax sums in another order than
//! [`super::sdpa_rows`], so the two agree to f32 rounding (tests: max
//! |diff| ≤ 2e-5 · max |ref|), not bit for bit.

use metal::{BufferRef, CommandBufferRef, CompileOptions, ComputePipelineState, Device, MTLSize};
use rayon::prelude::*;

use super::{QsaIndexerCache, QsaMode, QsaShape};
use crate::riir::backend::gpu::metal::MtlBuffer;
use crate::riir::model_layer::qwen3_8_flash_next::gated_residual_gpu::Q38GpuError;

/// Metal source of the sparse attention kernel. Compiled per shape
/// with `QSA_PER` (`head_dim / 32`) and `QSA_HG` (query heads per
/// threadgroup) defined in front ([`QsaGpuKernels::new`]): with
/// run-time loop bounds the per-lane arrays left registers and the
/// kernel ran ~10× slower.
pub const QSA_GPU_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint QSA_SGS = 8;          // simdgroups per threadgroup
constant uint QSA_MAX_PER_LANE = 8; // head_dim <= 256
constant uint QSA_MAX_HG = 4;       // query heads per threadgroup

// One threadgroup per (token, group of hg query heads sharing a KV
// head), so each gathered K / V row is read once for hg heads. Each
// simdgroup walks rows sg, sg + 8, ... of the token's list with an
// online softmax per head; lanes hold head_dim / 32 channels each.
// The 8 partial states are merged head by head through threadgroup
// memory, then out = sigmoid(gate) * (o / l).
kernel void q38_qsa_sdpa_rows(
    device const float* q        [[buffer(0)]],   // [n, heads * head_dim]
    device const float* gate     [[buffer(1)]],   // [n, heads * head_dim]
    device const float* k_cache  [[buffer(2)]],   // [ctx, kv_dim]
    device const float* v_cache  [[buffer(3)]],   // [ctx, kv_dim]
    device const uint*  rows     [[buffer(4)]],   // [n, max_rows]
    device const uint*  counts   [[buffer(5)]],   // [n]
    device float*       out      [[buffer(6)]],   // [n, heads * head_dim]
    constant uint&  num_heads    [[buffer(7)]],
    constant uint&  heads_per_kv [[buffer(8)]],
    constant uint&  head_dim     [[buffer(9)]],
    constant uint&  kv_dim       [[buffer(10)]],
    constant uint&  max_rows     [[buffer(11)]],
    constant float& scale        [[buffer(12)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_m[QSA_SGS * QSA_MAX_HG];
    threadgroup float tg_l[QSA_SGS * QSA_MAX_HG];
    threadgroup float tg_o[QSA_SGS * 256];

    const uint hg = QSA_HG;     // query heads per threadgroup, divides heads_per_kv
    uint groups = num_heads / hg;
    uint t = tg / groups;
    uint h0 = (tg - t * groups) * hg;
    uint kvh = h0 / heads_per_kv;
    const uint per = QSA_PER;   // head_dim / 32, compile-time so the arrays stay in registers
    uint q_dim = num_heads * head_dim;

    float qv[QSA_MAX_HG][QSA_MAX_PER_LANE];
    float o[QSA_MAX_HG][QSA_MAX_PER_LANE];
    float m[QSA_MAX_HG];
    float l[QSA_MAX_HG];
    for (uint j = 0; j < hg; j++) {
        device const float* qh = q + (ulong)t * q_dim + (h0 + j) * head_dim;
        for (uint i = 0; i < per; i++) {
            qv[j][i] = qh[lane * per + i] * scale;
            o[j][i] = 0.0f;
        }
        m[j] = -INFINITY;
        l[j] = 0.0f;
    }
    uint cnt = counts[t];
    device const uint* rl = rows + (ulong)t * max_rows;
    float kr[QSA_MAX_PER_LANE];
    float vr[QSA_MAX_PER_LANE];
    for (uint r = sg; r < cnt; r += QSA_SGS) {
        ulong base = (ulong)rl[r] * kv_dim + kvh * head_dim + lane * per;
        for (uint i = 0; i < per; i++) { kr[i] = k_cache[base + i]; vr[i] = v_cache[base + i]; }
        for (uint j = 0; j < hg; j++) {
            float dot = 0.0f;
            for (uint i = 0; i < per; i++) dot = fma(qv[j][i], kr[i], dot);
            float s = simd_sum(dot);
            float m_new = max(m[j], s);
            float corr = exp(m[j] - m_new);
            float p = exp(s - m_new);
            l[j] = l[j] * corr + p;
            for (uint i = 0; i < per; i++) o[j][i] = fma(p, vr[i], o[j][i] * corr);
            m[j] = m_new;
        }
    }
    if (lane == 0) {
        for (uint j = 0; j < hg; j++) { tg_m[sg * QSA_MAX_HG + j] = m[j]; tg_l[sg * QSA_MAX_HG + j] = l[j]; }
    }
    for (uint j = 0; j < hg; j++) {
        for (uint i = 0; i < per; i++) tg_o[sg * 256 + lane * per + i] = o[j][i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float mm = -INFINITY;
            for (uint s = 0; s < QSA_SGS; s++) mm = max(mm, tg_m[s * QSA_MAX_HG + j]);
            float ll = 0.0f;
            float acc[QSA_MAX_PER_LANE];
            for (uint i = 0; i < per; i++) acc[i] = 0.0f;
            for (uint s = 0; s < QSA_SGS; s++) {
                // A simdgroup with no rows has m = -inf, l = 0: weight 0.
                float ls = tg_l[s * QSA_MAX_HG + j];
                float w = (ls > 0.0f) ? exp(tg_m[s * QSA_MAX_HG + j] - mm) : 0.0f;
                ll += ls * w;
                for (uint i = 0; i < per; i++) acc[i] = fma(tg_o[s * 256 + lane * per + i], w, acc[i]);
            }
            uint hoff = (h0 + j) * head_dim;
            device const float* gh = gate + (ulong)t * q_dim + hoff;
            device float* oh = out + (ulong)t * q_dim + hoff;
            float inv = 1.0f / ll;
            for (uint i = 0; i < per; i++) {
                uint c = lane * per + i;
                oh[c] = acc[i] * inv * (1.0f / (1.0f + exp(-gh[c])));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

/// Per-query row lists of a chunk, `[n, max_rows]` (rows past a
/// query's count are 0 and unread).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QsaRowLists {
    pub n_tokens: usize,
    pub max_rows: usize,
    /// Rows each query attends to.
    pub counts: Vec<u32>,
    /// `[n_tokens, max_rows]` cache row indices, increasing per query.
    pub rows: Vec<u32>,
    /// Queries whose list is a strict subset of their visible tokens.
    pub n_sparse: usize,
}

/// The rows each query of a chunk attends to: token `t` is cache row
/// `kv_start + t` and sees `0..=kv_start + t`; `q_idx` are the chunk's
/// indexer queries ([`super::qsa_indexer_step`]) and `indexer` already
/// holds the chunk. [`QsaMode::Sparse`] keeps the top blocks past the
/// budget ([`QsaIndexerCache::select`]); [`QsaMode::Dense`] keeps
/// everything. Queries are scored over `pool` (or the global rayon
/// pool); the result does not depend on it.
pub fn qsa_select_rows(
    shape: &QsaShape,
    mode: QsaMode,
    pool: Option<&rayon::ThreadPool>,
    q_idx: &[f32],
    kv_start: usize,
    n_tokens: usize,
    indexer: &QsaIndexerCache,
) -> QsaRowLists {
    let iqd = shape.idx_q_dim();
    assert_eq!(q_idx.len(), n_tokens * iqd, "qsa_select_rows: q_idx length");
    assert!(indexer.len() >= kv_start + n_tokens, "qsa_select_rows: indexer cache behind the chunk");
    let pick = |t: usize| -> Option<Vec<u32>> {
        match mode {
            QsaMode::Dense => None,
            QsaMode::Sparse => indexer.select(&q_idx[t * iqd..(t + 1) * iqd], kv_start + t + 1),
        }
    };
    let picked: Vec<Option<Vec<u32>>> = match pool {
        Some(p) => p.install(|| (0..n_tokens).into_par_iter().map(pick).collect()),
        None => (0..n_tokens).into_par_iter().map(pick).collect(),
    };
    let counts: Vec<u32> = picked
        .iter()
        .enumerate()
        .map(|(t, r)| r.as_ref().map_or(kv_start + t + 1, |r| r.len()) as u32)
        .collect();
    let max_rows = counts.iter().copied().max().unwrap_or(0) as usize;
    let mut rows = vec![0u32; n_tokens * max_rows];
    let mut n_sparse = 0;
    for (t, (r, dst)) in picked.iter().zip(rows.chunks_mut(max_rows.max(1))).enumerate() {
        match r {
            Some(r) => {
                dst[..r.len()].copy_from_slice(r);
                n_sparse += 1;
            }
            None => {
                for (i, d) in dst[..kv_start + t + 1].iter_mut().enumerate() {
                    *d = i as u32;
                }
            }
        }
    }
    QsaRowLists { n_tokens, max_rows, counts, rows, n_sparse }
}

/// The compiled sparse-attention kernel.
pub struct QsaGpuKernels {
    pub sdpa_rows: ComputePipelineState,
    /// The shape the kernel was compiled for.
    pub shape: QsaShape,
    /// Query heads per threadgroup (largest of 4, 3, 2, 1 dividing
    /// `heads / kv_heads`).
    pub heads_per_tg: usize,
}

fn set_const<T: Copy>(enc: &metal::ComputeCommandEncoderRef, index: u64, v: &T) {
    enc.set_bytes(index, std::mem::size_of::<T>() as u64, (v as *const T).cast());
}

impl QsaGpuKernels {
    /// Compile the kernel for `shape` (`head_dim % 32 == 0`,
    /// `head_dim <= 256`).
    pub fn new(device: &Device, shape: &QsaShape) -> Result<Self, Q38GpuError> {
        assert!(shape.head_dim % 32 == 0 && shape.head_dim <= 256, "q38_qsa_sdpa_rows: head_dim {}", shape.head_dim);
        let hpk = shape.num_heads / shape.num_kv_heads;
        let hg = (1..=4).rev().find(|g| hpk % g == 0).unwrap_or(1);
        let src = format!("#define QSA_PER {}\n#define QSA_HG {hg}\n{QSA_GPU_SHADERS}", shape.head_dim / 32);
        let lib = device.new_library_with_source(&src, &CompileOptions::new()).map_err(Q38GpuError::Compile)?;
        let name = "q38_qsa_sdpa_rows";
        let f = lib.get_function(name, None).map_err(|msg| Q38GpuError::Pipeline { name, msg })?;
        let sdpa_rows =
            device.new_compute_pipeline_state_with_function(&f).map_err(|msg| Q38GpuError::Pipeline { name, msg })?;
        Ok(Self { sdpa_rows, shape: *shape, heads_per_tg: hg })
    }

    /// Gated block-selected attention for `n_tokens` queries:
    /// `out[t] = sigmoid(gate[t]) ⊙ softmax(q[t] · K[rows_t] / √d) V[rows_t]`
    /// per head (GQA). `q` / `gate` / `out` are `[n, q_dim]` f32 at byte
    /// offsets; `k_cache` / `v_cache` `[ctx, kv_dim]` f32; `rows` /
    /// `counts` as in [`QsaRowLists`] (upload with
    /// [`QsaRowBuffers::upload`]). The attention shape is the one
    /// [`Self::new`] compiled for.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_sdpa_rows(
        &self,
        cmdbuf: &CommandBufferRef,
        q: &BufferRef,
        q_off: u64,
        gate: &BufferRef,
        gate_off: u64,
        k_cache: &BufferRef,
        v_cache: &BufferRef,
        rows: &BufferRef,
        counts: &BufferRef,
        max_rows: usize,
        out: &BufferRef,
        out_off: u64,
        n_tokens: usize,
    ) {
        let shape = &self.shape;
        if n_tokens == 0 {
            return;
        }
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.sdpa_rows);
        enc.set_buffer(0, Some(q), q_off);
        enc.set_buffer(1, Some(gate), gate_off);
        enc.set_buffer(2, Some(k_cache), 0);
        enc.set_buffer(3, Some(v_cache), 0);
        enc.set_buffer(4, Some(rows), 0);
        enc.set_buffer(5, Some(counts), 0);
        enc.set_buffer(6, Some(out), out_off);
        set_const(enc, 7, &(shape.num_heads as u32));
        set_const(enc, 8, &((shape.num_heads / shape.num_kv_heads) as u32));
        set_const(enc, 9, &(shape.head_dim as u32));
        set_const(enc, 10, &(shape.kv_dim() as u32));
        set_const(enc, 11, &(max_rows as u32));
        set_const(enc, 12, &(1.0f32 / (shape.head_dim as f32).sqrt()));
        enc.dispatch_thread_groups(
            MTLSize::new((n_tokens * shape.num_heads / self.heads_per_tg) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
}

/// GPU copies of a [`QsaRowLists`], grown on demand.
pub struct QsaRowBuffers {
    device: Device,
    pub rows: MtlBuffer<u32>,
    pub counts: MtlBuffer<u32>,
    pub max_rows: usize,
    pub n_tokens: usize,
}

impl QsaRowBuffers {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            rows: MtlBuffer::with_len(device, 1),
            counts: MtlBuffer::with_len(device, 1),
            max_rows: 0,
            n_tokens: 0,
        }
    }

    /// Copy `lists` in (before the command buffer that reads them is
    /// committed).
    pub fn upload(&mut self, lists: &QsaRowLists) {
        if self.rows.len() < lists.rows.len() {
            self.rows = MtlBuffer::with_len(&self.device, lists.rows.len());
        }
        if self.counts.len() < lists.counts.len() {
            self.counts = MtlBuffer::with_len(&self.device, lists.counts.len());
        }
        self.rows.as_mut_slice()[..lists.rows.len()].copy_from_slice(&lists.rows);
        self.counts.as_mut_slice()[..lists.counts.len()].copy_from_slice(&lists.counts);
        self.max_rows = lists.max_rows;
        self.n_tokens = lists.n_tokens;
    }
}

#[cfg(test)]
mod tests {
    use super::super::{qsa_attend_rows, QsaIndexerCache};
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f32(&mut self) -> f32 {
            (self.u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.f32() * scale).collect()
        }
    }

    /// A context of `ctx` tokens whose last `n` are the chunk: random
    /// (normed-looking) q / gate / K / V and indexer keys / queries.
    struct Case {
        shape: QsaShape,
        kv_start: usize,
        n: usize,
        q: Vec<f32>,
        gate: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        q_idx: Vec<f32>,
        indexer: QsaIndexerCache,
    }

    impl Case {
        fn new(shape: QsaShape, ctx: usize, n: usize, seed: u64) -> Case {
            let mut r = Rng(seed);
            let s = &shape;
            let k_norm: Vec<f32> = (0..s.idx_head_dim).map(|_| 1.0 + 0.1 * r.f32()).collect();
            let mut indexer = QsaIndexerCache::new(s, k_norm);
            let pos: Vec<[i32; 3]> = (0..ctx as i32).map(|p| [p; 3]).collect();
            indexer.append(&r.vec(ctx * s.idx_head_dim, 1.0), &pos);
            Case {
                shape,
                kv_start: ctx - n,
                n,
                q: r.vec(n * s.q_dim(), 1.0),
                gate: r.vec(n * s.q_dim(), 2.0),
                k: r.vec(ctx * s.kv_dim(), 1.0),
                v: r.vec(ctx * s.kv_dim(), 1.0),
                q_idx: r.vec(n * s.idx_q_dim(), 1.0),
                indexer,
            }
        }

        fn cpu(&self, mode: QsaMode) -> Vec<f32> {
            let mut out = vec![0f32; self.n * self.shape.q_dim()];
            qsa_attend_rows(&self.shape, mode, None, &self.q, &self.gate, &self.k, &self.v, self.kv_start, &self.q_idx, &self.indexer, &mut out);
            out
        }

        fn gpu(&self, mode: QsaMode) -> (Vec<f32>, QsaRowLists) {
            let device = Device::system_default().expect("Metal");
            let k = QsaGpuKernels::new(&device, &self.shape).expect("compile");
            let lists = qsa_select_rows(&self.shape, mode, None, &self.q_idx, self.kv_start, self.n, &self.indexer);
            let mut rb = QsaRowBuffers::new(&device);
            rb.upload(&lists);
            let (q, g) = (MtlBuffer::with_data(&device, &self.q), MtlBuffer::with_data(&device, &self.gate));
            let (kc, vc) = (MtlBuffer::with_data(&device, &self.k), MtlBuffer::with_data(&device, &self.v));
            let out = MtlBuffer::<f32>::with_len(&device, self.n * self.shape.q_dim());
            let queue = device.new_command_queue();
            let cb = queue.new_command_buffer();
            k.encode_sdpa_rows(cb, q.raw(), 0, g.raw(), 0, kc.raw(), vc.raw(), rb.rows.raw(), rb.counts.raw(), rb.max_rows, out.raw(), 0, self.n);
            cb.commit();
            cb.wait_until_completed();
            (out.to_vec(), lists)
        }
    }

    fn check(label: &str, got: &[f32], want: &[f32]) {
        assert!(got.iter().all(|v| v.is_finite()), "{label}: non-finite");
        let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let d = got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("[qsa gpu] {label}: max|diff| {d:.3e} (max|ref| {scale:.3e})");
        assert!(d <= 2e-5 * scale, "{label}: {d} vs {scale}");
    }

    /// Real shape (24 / 2 heads × 256, budget 2048): a chunk of 24
    /// queries that crosses 2051 visible tokens, GPU vs the CPU sparse
    /// path, both modes; the row lists match the CPU selection.
    #[test]
    fn sdpa_rows_real_shape_matches_cpu_sparse() {
        if Device::system_default().is_none() {
            return;
        }
        let c = Case::new(QsaShape::QWEN3_8_FLASH_NEXT, 2040 + 24, 24, 0x0051_5341);
        for mode in [QsaMode::Sparse, QsaMode::Dense] {
            let (g, lists) = c.gpu(mode);
            let want_sparse = if mode == QsaMode::Sparse { 2040 + 24 - 2051 } else { 0 };
            assert_eq!(lists.n_sparse, want_sparse, "{mode:?}");
            if mode == QsaMode::Sparse {
                assert_eq!(lists.max_rows, 2051);
            }
            for t in 0..c.n {
                let rows = &lists.rows[t * lists.max_rows..][..lists.counts[t] as usize];
                let want = c.indexer.select(&c.q_idx[t * c.shape.idx_q_dim()..][..c.shape.idx_q_dim()], c.kv_start + t + 1);
                match (mode, want) {
                    (QsaMode::Sparse, Some(w)) => assert_eq!(rows, &w[..]),
                    _ => assert!(rows.iter().enumerate().all(|(i, &r)| r as usize == i) && rows.len() == c.kv_start + t + 1),
                }
            }
            check(&format!("real {mode:?}"), &g, &c.cpu(mode));
        }
    }

    /// Tiny budget (16 tokens): 40-token context, chunk of 30 queries,
    /// most of them sparse.
    #[test]
    fn sdpa_rows_tiny_matches_cpu_sparse() {
        if Device::system_default().is_none() {
            return;
        }
        let shape = QsaShape { hidden: 128, budget: 16, ..QsaShape::QWEN3_8_FLASH_NEXT };
        let c = Case::new(shape, 40, 30, 0x0051_5342);
        let (g, lists) = c.gpu(QsaMode::Sparse);
        assert_eq!(lists.n_sparse, 40 - 19);
        check("tiny sparse", &g, &c.cpu(QsaMode::Sparse));
    }

    /// CPU cost of the parts that stay on the host — the indexer's
    /// scoring and top-k for every query of a prefill — and of the
    /// attention the GPU kernel replaces, at 4k and 8k context, real
    /// shape. `cargo test --release ... select_rows_cpu_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn select_rows_cpu_cost() {
        use std::time::Instant;
        let s = QsaShape::QWEN3_8_FLASH_NEXT;
        let device = Device::system_default().expect("Metal");
        let kern = QsaGpuKernels::new(&device, &s).expect("compile");
        for ctx in [4096usize, 8192] {
            let mut r = Rng(ctx as u64);
            let k_norm: Vec<f32> = (0..s.idx_head_dim).map(|_| 1.0 + 0.1 * r.f32()).collect();
            let mut indexer = QsaIndexerCache::new(&s, k_norm);
            let pos: Vec<[i32; 3]> = (0..ctx as i32).map(|p| [p; 3]).collect();
            indexer.append(&r.vec(ctx * s.idx_head_dim, 1.0), &pos);
            // Selection for every query of the prefill, in 512-token chunks.
            let t0 = Instant::now();
            let mut sparse = 0;
            let chunk = 512;
            let q_idx = r.vec(chunk * s.idx_q_dim(), 1.0);
            let mut start = 0;
            while start < ctx {
                let n = chunk.min(ctx - start);
                let l = qsa_select_rows(&s, QsaMode::Sparse, None, &q_idx[..n * s.idx_q_dim()], start, n, &indexer);
                sparse += l.n_sparse;
                start += n;
            }
            let sel = t0.elapsed();
            // Attention for the last chunk: CPU sdpa vs the GPU kernel.
            let n = 64;
            let (q, gate) = (r.vec(n * s.q_dim(), 1.0), r.vec(n * s.q_dim(), 1.0));
            let (k, v) = (r.vec(ctx * s.kv_dim(), 1.0), r.vec(ctx * s.kv_dim(), 1.0));
            let qi = r.vec(n * s.idx_q_dim(), 1.0);
            let t1 = Instant::now();
            let mut out = vec![0f32; n * s.q_dim()];
            qsa_attend_rows(&s, QsaMode::Sparse, None, &q, &gate, &k, &v, ctx - n, &qi, &indexer, &mut out);
            let cpu_attn = t1.elapsed();
            let lists = qsa_select_rows(&s, QsaMode::Sparse, None, &qi, ctx - n, n, &indexer);
            let mut rb = QsaRowBuffers::new(&device);
            rb.upload(&lists);
            let (qb, gb) = (MtlBuffer::with_data(&device, &q), MtlBuffer::with_data(&device, &gate));
            let (kb, vb) = (MtlBuffer::with_data(&device, &k), MtlBuffer::with_data(&device, &v));
            let ob = MtlBuffer::<f32>::with_len(&device, n * s.q_dim());
            let queue = device.new_command_queue();
            // GPU time per dispatch, 8 dispatches per command buffer
            // (one alone mostly measures the command buffer round trip).
            let mut best = f64::MAX;
            for _ in 0..5 {
                let cb = queue.new_command_buffer();
                for _ in 0..8 {
                    kern.encode_sdpa_rows(cb, qb.raw(), 0, gb.raw(), 0, kb.raw(), vb.raw(), rb.rows.raw(), rb.counts.raw(), rb.max_rows, ob.raw(), 0, n);
                }
                cb.commit();
                cb.wait_until_completed();
                let (a, b) = crate::riir::backend::gpu::gpu_timing::gpu_span(cb);
                let gpu = b - a;
                best = best.min(gpu / 8.0);
            }
            eprintln!(
                "[qsa cost] ctx {ctx}: select for {ctx} queries ({sparse} sparse) {:.3} s / layer, {:.1} us / query; \
                 attention of {n} queries: CPU (1 thread) {:.1} ms, GPU kernel {:.2} ms",
                sel.as_secs_f64(),
                sel.as_secs_f64() * 1e6 / ctx as f64,
                cpu_attn.as_secs_f64() * 1e3,
                best * 1e3
            );
        }
    }
}
