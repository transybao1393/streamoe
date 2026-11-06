//! The family's extra ops between graph [`Op`]s, on both backends.
//!
//! The engine path ([`super::engine`]) runs every step it can as an
//! existing graph [`Op`] — matvecs, the Gated DeltaNet conv /
//! recurrence, the QSA projections / RoPE / SDPA, the expert
//! permute-fuse and combine — so the CPU backend checks the Metal one
//! op for op. What qwen3_5_moe has no op for is a [`FamOp`]:
//!
//! - the Gated Residual's elementwise kernels (`hc_norm` with `1 + w`,
//!   `silu(v / hc)`, the copy mix, the injection weights, the inject
//!   and the stream expand) — on Metal the pipelines of
//!   [`Q38GpuKernels`];
//! - per-head RMSNorm with a `1 + w` gain (the QSA Q / K norms) —
//!   [`FamOp::HeadNormOnePlus`] replaces the graph's
//!   `Op::RmsNormPerHeadNTokens`, whose plain bf16 gain would need
//!   `1 + w` rounded to bf16 in the weight file (up to 2^-9 relative
//!   error per channel);
//! - the Gated DeltaNet output norm with a sigmoid gate
//!   ([`FamOp::GatedRmsNormSigmoid`]; the graph's
//!   `Op::GatedRmsNormNTokens` gates with SiLU);
//! - `dst += src` ([`FamOp::Add`], the PLE addition).
//! - the MoE row gather ([`FamOp::GatherRows`]: each expert assignment
//!   row from its token's row, on the GPU);
//! - QSA past the budget ([`FamOp::SparseRows`]: attention over host-
//!   selected cache rows, `qsa::gpu` on Metal).
//!
//! **Norm gains.** Every `1 + w` gain is applied in the kernels, from
//! the checkpoint's zero-centred bf16 values in the weight file. The
//! converter and the loader leave them as stored.
//!
//! A pass is a list of [`Step`]s encoded into one command buffer
//! ([`run_steps`]).
//!
//! **One place per kernel signature.** On Metal every [`FamOp`] arm is
//! one call to a shared encoder: the Gated Residual kernels through
//! [`Q38GpuKernels`]'s `encode_*` functions (the ones the T-token
//! Gated Residual path uses), the engine-only kernels through
//! [`Q38EngineKernels`]. No arm binds kernel arguments itself, so a
//! kernel that gains an argument changes in one function.
//!
//! Every op takes a row count (`n_tokens` / `rows` / `n_groups` /
//! `n`): a prefill chunk of `T` tokens is the same ops at `T` rows.

use std::cell::RefCell;

use metal::{BufferRef, CommandBufferRef, CompileOptions, ComputePipelineState, Device, MTLSize, NSUInteger};

use super::gated_residual_gpu::{Q38GpuError, Q38GpuKernels};
use super::qsa::gpu::{QsaGpuKernels, QsaRowBuffers, QsaRowLists};
use super::qsa::{sdpa_rows, QsaShape};
use crate::riir::backend::buftype::{Buf, HiddenBuf};
use crate::riir::backend::{Backend, BufId, BufferPool, CpuBackend, GraphError, MetalBackend, Op};
use crate::riir::io::embedding::bf16_to_f32;

/// Every family buffer carries the `HiddenBuf` role tag; [`retag`]
/// gives it the tag an [`Op`] field wants. The family's dataflow
/// reuses one buffer in several qwen3_5_moe roles (the Gated
/// Residual's output is both the attention and the MoE input), so the
/// tags are labels here, not a check.
pub type FBuf = BufId<HiddenBuf>;

/// Re-label a buffer id for an [`Op`] field.
#[inline]
pub fn retag<A: Buf, B: Buf>(id: BufId<A>) -> BufId<B> {
    BufId::from_raw(id.raw())
}

/// A family op (see the module docs). Offsets are byte offsets into
/// the weight file; gains are bf16.
#[derive(Debug, Clone, Copy)]
pub enum FamOp {
    /// `out[g·group + i] = x̂ · (1 + gain[g·group + i])`, RMS over each
    /// of `n_groups` groups of `group` (the Gated Residual's
    /// `hc_norm`).
    /// `rows` rows of `gain_len` values (`gain_len / group` groups
    /// each), every row scaled by the same `gain_len` gains.
    HcNorm { x: FBuf, gain_off: u64, out: FBuf, group: u32, gain_len: u32, rows: u32, eps: f32 },
    /// `v[i] = silu(v[i] · scale)`, `n` values.
    SiluScale { v: FBuf, scale: f32, n: u32 },
    /// `out[t, j] = mean_c sigmoid(mix[t, c·H + j]) · normed[t, c·H + j]`.
    HcMix { mix: FBuf, normed: FBuf, out: FBuf, hidden: u32, hc: u32, n_tokens: u32 },
    /// `inj[t, c] = 2 · sigmoid(inj[t, c] / hc)`.
    InjWeights { inj: FBuf, hc: u32, n_tokens: u32 },
    /// `hyper[t, c·H + j] += inj[t, c] · block[t, j]`.
    HcInject { hyper: FBuf, block: FBuf, inj: FBuf, hidden: u32, hc: u32, n_tokens: u32 },
    /// `hyper[t, c·H + j] = h[t, j]`.
    HcExpand { h: FBuf, hyper: FBuf, hidden: u32, hc: u32, n_tokens: u32 },
    /// In place: each `dim` slice of `x` (`n_groups` of them) RMS-
    /// normalized and scaled by `1 + gain[i]` (`gain` `[dim]`, shared).
    HeadNormOnePlus { x: FBuf, gain_off: u64, dim: u32, n_groups: u32, eps: f32 },
    /// `out = v̂ · w · sigmoid(z)` per `dim` slice (`n_groups` of them;
    /// plain gain `w` `[dim]`, shared).
    GatedRmsNormSigmoid { values: FBuf, z: FBuf, weight_off: u64, out: FBuf, dim: u32, n_groups: u32, eps: f32 },
    /// `dst[i] += src[i]`, `n` values.
    Add { dst: FBuf, src: FBuf, n: u32 },
    /// `dst[r, :] = src[idx[r], :]` for `n_rows` rows of `width` (`idx`
    /// i32: the MoE assignment rows from the token rows, on the GPU).
    GatherRows { src: FBuf, idx: FBuf, dst: FBuf, width: u32, n_rows: u32 },
    /// QSA over the budget: gated attention of `n_tokens` queries (`q` /
    /// `gate` `[n, q_dim]`) over the cache rows of the last
    /// [`Q38Exec::fam_set_rows`] lists, into `out` `[n, q_dim]`. Metal:
    /// [`QsaGpuKernels::encode_sdpa_rows`] (no K / V copies to the
    /// host); CPU: [`sdpa_rows`] per query.
    SparseRows { q: FBuf, gate: FBuf, k_cache: FBuf, v_cache: FBuf, out: FBuf, n_tokens: u32 },
    /// The PLE gate / conv ([`super::ple_gpu`]).
    Ple(super::ple_gpu::PleOp),
}

/// One entry of a pass.
#[derive(Debug)]
pub enum Step {
    G(Op),
    F(FamOp),
}

/// A backend that also runs [`FamOp`]s.
pub trait Q38Exec: Backend<Error = GraphError> {
    /// Compiled family kernels and the QSA row lists (Metal), or the
    /// QSA shape and row lists (CPU).
    type Fam;
    fn fam_new(&self, qsa: &QsaShape) -> Result<Self::Fam, Q38GpuError>;
    /// The row lists the next [`FamOp::SparseRows`] reads. Call it
    /// after the previous pass has completed (the lists live in one
    /// buffer).
    fn fam_set_rows(&self, fam: &mut Self::Fam, lists: &QsaRowLists);
    fn encode_fam(&self, fam: &Self::Fam, op: &FamOp, ctx: &mut Self::EncodeCtx);
}

/// Encode `steps` into one command buffer and wait for it.
pub fn run_steps<B: Q38Exec>(b: &B, fam: &B::Fam, steps: &[Step], label: &'static str) -> Result<(), GraphError> {
    let mut ctx = b.begin_encoding();
    for s in steps {
        match s {
            Step::G(op) => b.encode_op(op, &mut ctx),
            Step::F(f) => b.encode_fam(fam, f, &mut ctx),
        }
    }
    b.submit_and_wait(ctx, label)
}

/// [`run_steps`] without waiting: committed behind every earlier
/// command buffer (Metal; [`Backend::wait_pending`] waits), run inline
/// on the CPU backend.
pub fn run_steps_nowait<B: Q38Exec>(b: &B, fam: &B::Fam, steps: &[Step], label: &'static str) -> Result<(), GraphError> {
    let mut ctx = b.begin_encoding();
    for s in steps {
        match s {
            Step::G(op) => b.encode_op(op, &mut ctx),
            Step::F(f) => b.encode_fam(fam, f, &mut ctx),
        }
    }
    b.submit_nowait(ctx, label)
}

// --- Metal -------------------------------------------------------------

/// Kernels the Gated Residual library does not have.
pub const Q38_ENGINE_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float q38e_bf16(ushort b) { return as_type<float>(uint(b) << 16); }

// Threadgroup sum of `v` (any size up to 1024); every thread gets it.
inline float q38e_tg_sum(float v, threadgroup float* sh, uint lid, uint tg) {
    float s = simd_sum(v);
    if ((lid % 32) == 0) sh[lid / 32] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) {
        float t = (lid < (tg + 31) / 32) ? sh[lid] : 0.0f;
        t = simd_sum(t);
        if (lid == 0) sh[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sh[0];
}

// In place: group g of `dim` values, RMS-normalized, times (1 + gain[i]).
kernel void q38e_head_norm_one_plus(
    device float*        x    [[buffer(0)]],
    device const ushort* gain [[buffer(1)]],
    constant uint&       dim  [[buffer(2)]],
    constant float&      eps  [[buffer(3)]],
    uint g   [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    threadgroup float sh[32];
    uint base = g * dim;
    float acc = 0.0f;
    for (uint i = lid; i < dim; i += tg) { float v = x[base + i]; acc += v * v; }
    float s = q38e_tg_sum(acc, sh, lid, tg);
    float inv = 1.0f / sqrt(s / float(dim) + eps);
    for (uint i = lid; i < dim; i += tg) {
        x[base + i] = x[base + i] * inv * (1.0f + q38e_bf16(gain[i]));
    }
}

// out = rms_norm(values) * w * sigmoid(z), per group of `dim`.
kernel void q38e_gated_rms_norm_sigmoid(
    device const float*  values [[buffer(0)]],
    device const float*  z      [[buffer(1)]],
    device const ushort* weight [[buffer(2)]],
    device float*        out    [[buffer(3)]],
    constant uint&       dim    [[buffer(4)]],
    constant float&      eps    [[buffer(5)]],
    uint g   [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    threadgroup float sh[32];
    uint base = g * dim;
    float acc = 0.0f;
    for (uint i = lid; i < dim; i += tg) { float v = values[base + i]; acc += v * v; }
    float s = q38e_tg_sum(acc, sh, lid, tg);
    float inv = 1.0f / sqrt(s / float(dim) + eps);
    for (uint i = lid; i < dim; i += tg) {
        float zi = z[base + i];
        float gate = 1.0f / (1.0f + exp(-zi));
        out[base + i] = values[base + i] * inv * q38e_bf16(weight[i]) * gate;
    }
}

kernel void q38e_add(
    device float*       dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) dst[i] += src[i];
}

// dst[r, :] = src[idx[r], :] over total = n_rows * width.
kernel void q38e_gather_rows(
    device const float* src   [[buffer(0)]],
    device const int*   idx   [[buffer(1)]],
    device float*       dst   [[buffer(2)]],
    constant uint&      width [[buffer(3)]],
    constant uint&      total [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    uint r = i / width;
    dst[i] = src[(ulong)idx[r] * width + (i - r * width)];
}
"#;

/// Encoders of the kernels in [`Q38_ENGINE_SHADERS`]: the one place
/// that binds their arguments.
pub struct Q38EngineKernels {
    pub head_norm: ComputePipelineState,
    pub gated_norm_sigmoid: ComputePipelineState,
    pub add: ComputePipelineState,
    pub gather_rows: ComputePipelineState,
    pub ple: super::ple_gpu::PleGpuKernels,
}

fn grid1(n: u32, tg: u32) -> (MTLSize, MTLSize) {
    (MTLSize::new(n.div_ceil(tg) as NSUInteger, 1, 1), MTLSize::new(tg as NSUInteger, 1, 1))
}

fn set_u32(enc: &metal::ComputeCommandEncoderRef, i: u64, v: u32) {
    enc.set_bytes(i, 4, (&v as *const u32).cast());
}

fn set_f32(enc: &metal::ComputeCommandEncoderRef, i: u64, v: f32) {
    enc.set_bytes(i, 4, (&v as *const f32).cast());
}

impl Q38EngineKernels {
    pub fn new(device: &Device) -> Result<Self, Q38GpuError> {
        let lib = device
            .new_library_with_source(Q38_ENGINE_SHADERS, &CompileOptions::new())
            .map_err(Q38GpuError::Compile)?;
        let p = |name: &'static str| -> Result<ComputePipelineState, Q38GpuError> {
            let f = lib.get_function(name, None).map_err(|msg| Q38GpuError::Pipeline { name, msg })?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|msg| Q38GpuError::Pipeline { name, msg })
        };
        Ok(Self {
            head_norm: p("q38e_head_norm_one_plus")?,
            gated_norm_sigmoid: p("q38e_gated_rms_norm_sigmoid")?,
            add: p("q38e_add")?,
            gather_rows: p("q38e_gather_rows")?,
            ple: super::ple_gpu::PleGpuKernels::new(device)?,
        })
    }

    /// In place: `n_groups` slices of `dim`, RMS-normalized, times
    /// `1 + gain[i]` (bf16 `[dim]` at byte `gain_off` of `gain`).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_head_norm_one_plus(
        &self,
        cmdbuf: &CommandBufferRef,
        x: &BufferRef,
        gain: &BufferRef,
        gain_off: u64,
        dim: u32,
        n_groups: u32,
        eps: f32,
    ) {
        if n_groups == 0 {
            return;
        }
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.head_norm);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(gain), gain_off as NSUInteger);
        set_u32(enc, 2, dim);
        set_f32(enc, 3, eps);
        enc.dispatch_thread_groups(MTLSize::new(n_groups as NSUInteger, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
    }

    /// `out = rms_norm(values) · w · sigmoid(z)` per `dim` slice
    /// (`n_groups` of them; bf16 `w` `[dim]` at byte `weight_off`).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_gated_rms_norm_sigmoid(
        &self,
        cmdbuf: &CommandBufferRef,
        values: &BufferRef,
        z: &BufferRef,
        weight: &BufferRef,
        weight_off: u64,
        out: &BufferRef,
        dim: u32,
        n_groups: u32,
        eps: f32,
    ) {
        if n_groups == 0 {
            return;
        }
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.gated_norm_sigmoid);
        enc.set_buffer(0, Some(values), 0);
        enc.set_buffer(1, Some(z), 0);
        enc.set_buffer(2, Some(weight), weight_off as NSUInteger);
        enc.set_buffer(3, Some(out), 0);
        set_u32(enc, 4, dim);
        set_f32(enc, 5, eps);
        enc.dispatch_thread_groups(MTLSize::new(n_groups as NSUInteger, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
    }

    /// `dst[i] += src[i]`, `n` values.
    pub fn encode_add(&self, cmdbuf: &CommandBufferRef, dst: &BufferRef, src: &BufferRef, n: u32) {
        if n == 0 {
            return;
        }
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.add);
        enc.set_buffer(0, Some(dst), 0);
        enc.set_buffer(1, Some(src), 0);
        set_u32(enc, 2, n);
        let (g, t) = grid1(n, 256);
        enc.dispatch_thread_groups(g, t);
        enc.end_encoding();
    }

    /// A PLE gate / conv op on its own encoder ([`super::ple_gpu::PleGpuKernels::encode`]
    /// binds the arguments).
    pub fn encode_ple(
        &self,
        cmdbuf: &CommandBufferRef,
        op: &super::ple_gpu::PleOp,
        pool: &crate::riir::backend::gpu::MetalBufferPool,
        wbuf: &BufferRef,
    ) {
        let enc = cmdbuf.new_compute_command_encoder();
        self.ple.encode(enc, op, pool, wbuf);
        enc.end_encoding();
    }

    /// `dst[r, :] = src[idx[r], :]`, `n_rows` rows of `width` (`idx` i32).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_gather_rows(
        &self,
        cmdbuf: &CommandBufferRef,
        src: &BufferRef,
        idx: &BufferRef,
        dst: &BufferRef,
        width: u32,
        n_rows: u32,
    ) {
        let total = width * n_rows;
        if total == 0 {
            return;
        }
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.gather_rows);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(idx), 0);
        enc.set_buffer(2, Some(dst), 0);
        set_u32(enc, 3, width);
        set_u32(enc, 4, total);
        let (g, t) = grid1(total, 256);
        enc.dispatch_thread_groups(g, t);
        enc.end_encoding();
    }
}

/// The family's Metal pipelines.
pub struct Q38MetalFam {
    pub gr: Q38GpuKernels,
    pub eng: Q38EngineKernels,
    pub qsa: QsaGpuKernels,
    pub rows: QsaRowBuffers,
}

impl Q38Exec for MetalBackend {
    type Fam = Q38MetalFam;

    fn fam_new(&self, qsa: &QsaShape) -> Result<Q38MetalFam, Q38GpuError> {
        let device = self.metal().device();
        Ok(Q38MetalFam {
            gr: Q38GpuKernels::new(device)?,
            eng: Q38EngineKernels::new(device)?,
            qsa: QsaGpuKernels::new(device, qsa)?,
            rows: QsaRowBuffers::new(device),
        })
    }

    fn fam_set_rows(&self, fam: &mut Q38MetalFam, lists: &QsaRowLists) {
        fam.rows.upload(lists);
    }

    fn encode_fam(&self, fam: &Q38MetalFam, op: &FamOp, ctx: &mut Self::EncodeCtx) {
        let cmd = ctx.cmdbuf();
        let pool = self.pool();
        let h = |id: FBuf| -> &BufferRef { pool.handle(id) };
        let wbuf = self.weight_buf().buffer();
        let u = |v: u32| v as usize;
        match *op {
            FamOp::HcNorm { x, gain_off, out, group, gain_len, rows, eps } => {
                fam.gr.encode_hc_norm_rows(cmd, h(x), wbuf, gain_off, h(out), u(group), u(gain_len), u(rows), eps)
            }
            FamOp::SiluScale { v, scale, n } => fam.gr.encode_silu_scale(cmd, h(v), scale, u(n)),
            FamOp::HcMix { mix, normed, out, hidden, hc, n_tokens } => {
                fam.gr.encode_hc_mix_n(cmd, h(mix), h(normed), h(out), u(hidden), u(hc), u(n_tokens))
            }
            FamOp::InjWeights { inj, hc, n_tokens } => fam.gr.encode_inj_weights_n(cmd, h(inj), u(hc), u(n_tokens)),
            FamOp::HcInject { hyper, block, inj, hidden, hc, n_tokens } => {
                fam.gr.encode_inject_n(cmd, h(hyper), h(block), 0, h(inj), u(hidden), u(hc), u(n_tokens))
            }
            FamOp::HcExpand { h: src, hyper, hidden, hc, n_tokens } => {
                fam.gr.encode_expand_n(cmd, h(src), h(hyper), u(hidden), u(hc), u(n_tokens))
            }
            FamOp::HeadNormOnePlus { x, gain_off, dim, n_groups, eps } => {
                fam.eng.encode_head_norm_one_plus(cmd, h(x), wbuf, gain_off, dim, n_groups, eps)
            }
            FamOp::GatedRmsNormSigmoid { values, z, weight_off, out, dim, n_groups, eps } => {
                fam.eng.encode_gated_rms_norm_sigmoid(cmd, h(values), h(z), wbuf, weight_off, h(out), dim, n_groups, eps)
            }
            FamOp::Add { dst, src, n } => fam.eng.encode_add(cmd, h(dst), h(src), n),
            FamOp::GatherRows { src, idx, dst, width, n_rows } => {
                fam.eng.encode_gather_rows(cmd, h(src), h(idx), h(dst), width, n_rows)
            }
            FamOp::SparseRows { q, gate, k_cache, v_cache, out, n_tokens } => {
                let r = &fam.rows;
                debug_assert_eq!(r.n_tokens, n_tokens as usize, "SparseRows: row lists of another chunk");
                fam.qsa.encode_sdpa_rows(
                    cmd,
                    h(q),
                    0,
                    h(gate),
                    0,
                    h(k_cache),
                    h(v_cache),
                    r.rows.raw(),
                    r.counts.raw(),
                    r.max_rows,
                    h(out),
                    0,
                    u(n_tokens),
                )
            }
            FamOp::Ple(ref p) => fam.eng.encode_ple(cmd, p, pool, wbuf),
        }
    }
}

// --- CPU ---------------------------------------------------------------

fn f32s(c: &RefCell<Vec<u8>>) -> std::cell::Ref<'_, [f32]> {
    std::cell::Ref::map(c.borrow(), |v| bytemuck::cast_slice(v.as_slice()))
}

fn f32s_mut(c: &RefCell<Vec<u8>>) -> std::cell::RefMut<'_, [f32]> {
    std::cell::RefMut::map(c.borrow_mut(), |v| bytemuck::cast_slice_mut(v.as_mut_slice()))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `n` bf16 values at byte `off` of the CPU backend's weight file.
fn bf16_gain(b: &CpuBackend, off: u64, n: usize) -> Vec<f32> {
    let bytes = b.weight_file().bytes_at(off, 2 * n).expect("q38: gain offset out of the weight file");
    bytes.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
}

/// RMS-normalize each `group` slice of `x` into `out`, times `gain(i)`.
fn rms_groups(x: &[f32], out: &mut [f32], group: usize, eps: f32, gain: impl Fn(usize, usize) -> f32) {
    for (g, (xs, os)) in x.chunks(group).zip(out.chunks_mut(group)).enumerate() {
        let mut s = 0.0f32;
        for &v in xs {
            s += v * v;
        }
        let inv = 1.0 / (s / group as f32 + eps).sqrt();
        for (i, (o, &v)) in os.iter_mut().zip(xs).enumerate() {
            *o = v * inv * gain(g, i);
        }
    }
}

/// The CPU backend's family state: the QSA shape and the last row
/// lists ([`FamOp::SparseRows`]).
pub struct Q38CpuFam {
    pub qsa: QsaShape,
    pub rows: QsaRowLists,
}

impl Q38Exec for CpuBackend {
    type Fam = Q38CpuFam;

    fn fam_new(&self, qsa: &QsaShape) -> Result<Q38CpuFam, Q38GpuError> {
        Ok(Q38CpuFam { qsa: *qsa, rows: QsaRowLists::default() })
    }

    fn fam_set_rows(&self, fam: &mut Q38CpuFam, lists: &QsaRowLists) {
        fam.rows = lists.clone();
    }

    fn encode_fam(&self, fam: &Q38CpuFam, op: &FamOp, _ctx: &mut ()) {
        let pool = self.pool();
        match *op {
            FamOp::HcNorm { x, gain_off, out, group, gain_len, rows, eps } => {
                let (group, gl) = (group as usize, gain_len as usize);
                let n = gl * rows as usize;
                let gain = bf16_gain(self, gain_off, gl);
                let xv = f32s(pool.handle(x));
                let mut ov = f32s_mut(pool.handle(out));
                rms_groups(&xv[..n], &mut ov[..n], group, eps, |g, i| 1.0 + gain[(g * group + i) % gl]);
            }
            FamOp::SiluScale { v, scale, n } => {
                let mut vv = f32s_mut(pool.handle(v));
                for x in vv[..n as usize].iter_mut() {
                    let t = *x * scale;
                    *x = t / (1.0 + (-t).exp());
                }
            }
            FamOp::HcMix { mix, normed, out, hidden, hc, n_tokens } => {
                let (h, hc) = (hidden as usize, hc as usize);
                let (m, nv) = (f32s(pool.handle(mix)), f32s(pool.handle(normed)));
                let mut ov = f32s_mut(pool.handle(out));
                for t in 0..n_tokens as usize {
                    let row = t * hc * h;
                    for j in 0..h {
                        let mut acc = 0.0f32;
                        for c in 0..hc {
                            acc += sigmoid(m[row + c * h + j]) * nv[row + c * h + j];
                        }
                        ov[t * h + j] = acc / hc as f32;
                    }
                }
            }
            FamOp::InjWeights { inj, hc, n_tokens } => {
                let mut iv = f32s_mut(pool.handle(inj));
                for v in iv[..(hc * n_tokens) as usize].iter_mut() {
                    *v = 2.0 * sigmoid(*v / hc as f32);
                }
            }
            FamOp::HcInject { hyper, block, inj, hidden, hc, n_tokens } => {
                let (h, hc) = (hidden as usize, hc as usize);
                let (bv, iv) = (f32s(pool.handle(block)), f32s(pool.handle(inj)));
                let mut hv = f32s_mut(pool.handle(hyper));
                for t in 0..n_tokens as usize {
                    let (b, w, y) = (&bv[t * h..(t + 1) * h], &iv[t * hc..(t + 1) * hc], t * hc * h);
                    for c in 0..hc {
                        for j in 0..h {
                            hv[y + c * h + j] = b[j].mul_add(w[c], hv[y + c * h + j]);
                        }
                    }
                }
            }
            FamOp::HcExpand { h, hyper, hidden, hc, n_tokens } => {
                let (hd, hc, n) = (hidden as usize, hc as usize, n_tokens as usize);
                let src = f32s(pool.handle(h))[..n * hd].to_vec();
                let mut hv = f32s_mut(pool.handle(hyper));
                for t in 0..n {
                    for c in 0..hc {
                        hv[(t * hc + c) * hd..(t * hc + c + 1) * hd].copy_from_slice(&src[t * hd..(t + 1) * hd]);
                    }
                }
            }
            FamOp::HeadNormOnePlus { x, gain_off, dim, n_groups, eps } => {
                let (d, n) = (dim as usize, (dim * n_groups) as usize);
                let gain = bf16_gain(self, gain_off, d);
                let mut xv = f32s_mut(pool.handle(x));
                let src = xv[..n].to_vec();
                rms_groups(&src, &mut xv[..n], d, eps, |_, i| 1.0 + gain[i]);
            }
            FamOp::GatedRmsNormSigmoid { values, z, weight_off, out, dim, n_groups, eps } => {
                let (d, n) = (dim as usize, (dim * n_groups) as usize);
                let w = bf16_gain(self, weight_off, d);
                let (vv, zv) = (f32s(pool.handle(values)), f32s(pool.handle(z)));
                let mut ov = f32s_mut(pool.handle(out));
                rms_groups(&vv[..n], &mut ov[..n], d, eps, |g, i| w[i] * sigmoid(zv[g * d + i]));
            }
            FamOp::Add { dst, src, n } => {
                let sv = f32s(pool.handle(src))[..n as usize].to_vec();
                let mut dv = f32s_mut(pool.handle(dst));
                for (d, s) in dv.iter_mut().zip(&sv) {
                    *d += s;
                }
            }
            FamOp::GatherRows { src, idx, dst, width, n_rows } => {
                let w = width as usize;
                let sv = f32s(pool.handle(src));
                let iv = std::cell::Ref::map(pool.handle(idx).borrow(), |v| bytemuck::cast_slice::<u8, i32>(v.as_slice()));
                let mut dv = f32s_mut(pool.handle(dst));
                for (r, row) in dv[..w * n_rows as usize].chunks_mut(w).enumerate() {
                    let s = iv[r] as usize;
                    row.copy_from_slice(&sv[s * w..(s + 1) * w]);
                }
            }
            FamOp::SparseRows { q, gate, k_cache, v_cache, out, n_tokens } => {
                let (s, r) = (&fam.qsa, &fam.rows);
                debug_assert_eq!(r.n_tokens, n_tokens as usize, "SparseRows: row lists of another chunk");
                let qd = s.q_dim();
                let (qv, gv) = (f32s(pool.handle(q)), f32s(pool.handle(gate)));
                let (kc, vc) = (f32s(pool.handle(k_cache)), f32s(pool.handle(v_cache)));
                let mut ov = f32s_mut(pool.handle(out));
                for t in 0..n_tokens as usize {
                    let rows = &r.rows[t * r.max_rows..t * r.max_rows + r.counts[t] as usize];
                    let (qt, gt) = (&qv[t * qd..(t + 1) * qd], &gv[t * qd..(t + 1) * qd]);
                    sdpa_rows(s, qt, gt, &kc, &vc, 0, Some(rows), &mut ov[t * qd..(t + 1) * qd]);
                }
            }
            FamOp::Ple(ref p) => super::ple_gpu::apply_cpu(self, p),
        }
    }
}
