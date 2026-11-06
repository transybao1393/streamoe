//! Gated Residual on Metal.
//!
//! Same math as [`super::gated_residual`], as a chain of small
//! dispatches on one command buffer:
//!
//! 1. `q38_hc_norm` — one threadgroup per copy: RMS over its `H`
//!    values, gain `1 + w` (BF16 gains, as stored in the weight file).
//! 2. `mix_down` matvec → `q38_silu_scale` (`silu(v / hc)`).
//! 3. `mix_up` matvec → `q38_hc_mix` (`x[j] = mean_c sigmoid(m) · n̂`).
//! 4. `block_inject` matvec → `q38_inj_weights` (`2 · sigmoid(v / hc)`).
//!
//! and after the block, `q38_hc_inject` (`hyper[c, j] += inj[c] ·
//! y[j]`); `q38_hc_expand` builds the model's input stream.
//!
//! The matvecs go through [`GpuLinear`]: [`GpuQ4Linear`] for the real
//! model's MLX 4-bit group-32 weights (the engine's matvec kernels and
//! MLX `qmm` with the group passed at run time), [`GpuDenseLinear`]
//! (f32) for the tests.
//!
//! Every step takes a row count: `encode_*_n` run `T` tokens of a
//! prefill chunk at once (`hyper` `[T, hc · H]`, block input / output
//! `[T, H]`, injection weights `[T, hc]`); the one-token functions are
//! `T = 1`.
//!
//! The kernels live in their own small library ([`Q38_SHADERS`]),
//! compiled once by [`Q38GpuKernels::new`], so they don't touch
//! `shaders.metal`.

use metal::{
    BufferRef, CommandBufferRef, CompileOptions, ComputePipelineState, Device, MTLSize,
};

use streammoe_metal::{Kernels, QmmCall, QuantWeights};

use crate::riir::backend::gpu::gpu_matvec::MatvecPipelines;
use crate::riir::backend::gpu::metal::{MetalContext, MetalError, MtlBuffer};

/// Errors building the family's Metal kernels.
#[derive(Debug, thiserror::Error)]
pub enum Q38GpuError {
    #[error("compiling the qwen3_8_flash_next Metal library: {0}")]
    Compile(String),
    #[error("kernel {name}: {msg}")]
    Pipeline { name: &'static str, msg: String },
}

/// Metal source of the family's own kernels.
pub const Q38_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float q38_bf16(ushort b) { return as_type<float>(uint(b) << 16); }
inline float q38_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// Grouped RMSNorm with gain (1 + w): threadgroup g normalizes
// x[g*group .. (g+1)*group). Gains are bf16, gain_len of them, reused
// for every row (gain index (g*group + i) % gain_len), so T rows of
// hc groups each run as one dispatch of T*hc groups.
kernel void q38_hc_norm(
    device const float*  x        [[buffer(0)]],
    device const ushort* gain     [[buffer(1)]],
    device float*        out      [[buffer(2)]],
    constant uint&       group    [[buffer(3)]],
    constant float&      eps      [[buffer(4)]],
    constant uint&       gain_len [[buffer(5)]],
    uint g   [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    threadgroup float shared_sum[32];
    uint base = g * group;
    float acc = 0.0f;
    for (uint i = lid; i < group; i += tg) {
        float v = x[base + i];
        acc += v * v;
    }
    float s = simd_sum(acc);
    if ((lid % 32) == 0) shared_sum[lid / 32] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) {
        float v = (lid < (tg + 31) / 32) ? shared_sum[lid] : 0.0f;
        v = simd_sum(v);
        if (lid == 0) shared_sum[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / sqrt(shared_sum[0] / float(group) + eps);
    for (uint i = lid; i < group; i += tg) {
        out[base + i] = x[base + i] * inv * (1.0f + q38_bf16(gain[(base + i) % gain_len]));
    }
}

// out[row] = W[row, :] . x, W row-major f32. One 32-thread threadgroup per row.
kernel void q38_matvec_f32(
    device const float* w      [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      in_dim [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    device const float* wr = w + ulong(row) * in_dim;
    float acc = 0.0f;
    for (uint i = lid; i < in_dim; i += 32) acc = fma(wr[i], x[i], acc);
    acc = simd_sum(acc);
    if (lid == 0) out[row] = acc;
}

// v[i] = silu(v[i] * scale)
kernel void q38_silu_scale(
    device float*   v     [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint&  n     [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float t = v[i] * scale;
    v[i] = t / (1.0f + exp(-t));
}

// out[t*H + j] = (1/hc) sum_c sigmoid(mix[t, c*H + j]) * normed[t, c*H + j]
// over total = T*H outputs.
kernel void q38_hc_mix(
    device const float* mix    [[buffer(0)]],
    device const float* normed [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      hidden [[buffer(3)]],
    constant uint&      hc     [[buffer(4)]],
    constant uint&      total  [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    uint t = i / hidden;
    uint j = i - t * hidden;
    uint row = t * hc * hidden;
    float acc = 0.0f;
    for (uint c = 0; c < hc; c++) {
        uint k = row + c * hidden + j;
        acc += q38_sigmoid(mix[k]) * normed[k];
    }
    out[i] = acc / float(hc);
}

// inj[i] = 2 * sigmoid(inj[i] / hc) for i < n (T rows of hc)
kernel void q38_inj_weights(
    device float*  inj [[buffer(0)]],
    constant uint& hc  [[buffer(1)]],
    constant uint& n   [[buffer(2)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= n) return;
    inj[c] = 2.0f * q38_sigmoid(inj[c] / float(hc));
}

// hyper[t, c*H + j] += inj[t, c] * block[t, j] over total = T*hc*H
kernel void q38_hc_inject(
    device float*       hyper  [[buffer(0)]],
    device const float* block  [[buffer(1)]],
    device const float* inj    [[buffer(2)]],
    constant uint&      hidden [[buffer(3)]],
    constant uint&      total  [[buffer(4)]],
    constant uint&      hc     [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    uint r = i / hidden;          // t * hc + c
    uint j = i - r * hidden;
    uint t = r / hc;
    hyper[i] = fma(block[t * hidden + j], inj[r], hyper[i]);
}

// hyper[t, c*H + j] = h[t, j] over total = T*hc*H
kernel void q38_hc_expand(
    device const float* h      [[buffer(0)]],
    device float*       hyper  [[buffer(1)]],
    constant uint&      hidden [[buffer(2)]],
    constant uint&      total  [[buffer(3)]],
    constant uint&      hc     [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    uint r = i / hidden;
    uint t = r / hc;
    hyper[i] = h[t * hidden + (i - r * hidden)];
}
"#;

/// `out = W · x` encoded on a command buffer. `x` / `out` are f32
/// buffers with byte offsets.
pub trait GpuLinear {
    fn in_dim(&self) -> usize;
    fn out_dim(&self) -> usize;
    fn encode(&self, cmdbuf: &CommandBufferRef, x: &BufferRef, x_off: u64, out: &BufferRef, out_off: u64);
    /// `n_tokens` rows at once: `x` is `[n, in_dim]`, `out`
    /// `[n, out_dim]` (f32, byte offsets). The default encodes one
    /// [`Self::encode`] per row, so it equals the per-token calls bit
    /// for bit; [`GpuQ4Linear`] batches.
    fn encode_n(&self, cmdbuf: &CommandBufferRef, x: &BufferRef, x_off: u64, out: &BufferRef, out_off: u64, n_tokens: usize) {
        let (xi, oi) = (4 * self.in_dim() as u64, 4 * self.out_dim() as u64);
        for t in 0..n_tokens as u64 {
            self.encode(cmdbuf, x, x_off + t * xi, out, out_off + t * oi);
        }
    }
}

/// The family's compiled kernels.
pub struct Q38GpuKernels {
    pub hc_norm: ComputePipelineState,
    pub matvec_f32: ComputePipelineState,
    pub silu_scale: ComputePipelineState,
    pub hc_mix: ComputePipelineState,
    pub inj_weights: ComputePipelineState,
    pub hc_inject: ComputePipelineState,
    pub hc_expand: ComputePipelineState,
}

fn grid1(n: usize, tg: usize) -> (MTLSize, MTLSize) {
    (MTLSize::new(n.div_ceil(tg) as u64, 1, 1), MTLSize::new(tg as u64, 1, 1))
}

/// Set a small POD constant (`set_bytes` with the value's own size).
fn set_const<T: Copy>(enc: &metal::ComputeCommandEncoderRef, index: u64, v: &T) {
    enc.set_bytes(index, std::mem::size_of::<T>() as u64, (v as *const T).cast());
}

impl Q38GpuKernels {
    pub fn new(device: &Device) -> Result<Self, Q38GpuError> {
        let lib = device
            .new_library_with_source(Q38_SHADERS, &CompileOptions::new())
            .map_err(Q38GpuError::Compile)?;
        let p = |name: &'static str| -> Result<ComputePipelineState, Q38GpuError> {
            let f = lib.get_function(name, None).map_err(|msg| Q38GpuError::Pipeline { name, msg })?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|msg| Q38GpuError::Pipeline { name, msg })
        };
        Ok(Self {
            hc_norm: p("q38_hc_norm")?,
            matvec_f32: p("q38_matvec_f32")?,
            silu_scale: p("q38_silu_scale")?,
            hc_mix: p("q38_hc_mix")?,
            inj_weights: p("q38_inj_weights")?,
            hc_inject: p("q38_hc_inject")?,
            hc_expand: p("q38_hc_expand")?,
        })
    }

    /// Grouped RMSNorm, gain `1 + w`: `out = hc_norm(x)` over
    /// `n_groups` groups of `group`. `gain` is BF16 at byte offset
    /// `gain_off`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_hc_norm(
        &self,
        cmdbuf: &CommandBufferRef,
        x: &BufferRef,
        gain: &BufferRef,
        gain_off: u64,
        out: &BufferRef,
        group: usize,
        n_groups: usize,
        eps: f32,
    ) {
        self.encode_hc_norm_rows(cmdbuf, x, gain, gain_off, out, group, group * n_groups, 1, eps)
    }

    /// [`Self::encode_hc_norm`] over `rows` rows of `gain_len` values
    /// (`gain_len / group` groups per row); every row is scaled by the
    /// same `gain_len` gains.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_hc_norm_rows(
        &self,
        cmdbuf: &CommandBufferRef,
        x: &BufferRef,
        gain: &BufferRef,
        gain_off: u64,
        out: &BufferRef,
        group: usize,
        gain_len: usize,
        rows: usize,
        eps: f32,
    ) {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.hc_norm);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(gain), gain_off);
        enc.set_buffer(2, Some(out), 0);
        set_const(enc, 3, &(group as u32));
        set_const(enc, 4, &eps);
        set_const(enc, 5, &(gain_len as u32));
        enc.dispatch_thread_groups(MTLSize::new((rows * gain_len / group) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }

    /// One Gated Residual: reads `hyper` (`[hc · H]`), writes the
    /// block input to `bufs.mixed` and, with `block_inject`, the
    /// injection weights to `bufs.inj`. `hyper` is left as is.
    pub fn encode_gated_residual_mix(
        &self,
        cmdbuf: &CommandBufferRef,
        w: &GpuHcWeights<'_>,
        hyper: &BufferRef,
        bufs: &GpuHcBuffers,
        eps: f32,
    ) {
        self.encode_gated_residual_mix_n(cmdbuf, w, hyper, bufs, eps, 1)
    }

    /// [`Self::encode_gated_residual_mix`] for `n_tokens` rows:
    /// `hyper` is `[n, hc · H]`; writes `bufs.mixed` `[n, H]` and
    /// `bufs.inj` `[n, hc]`. `bufs` must hold `n` rows
    /// ([`GpuHcBuffers::with_tokens`]). Each elementwise step does the
    /// one-token arithmetic per row, so the result equals `n`
    /// one-token calls bit for bit when the linears' `encode_n` does
    /// (the default `GpuLinear::encode_n`, or [`GpuQ4Linear`] with qmv
    /// and the GEMM off); otherwise it is within the matvec kernels'
    /// rounding.
    pub fn encode_gated_residual_mix_n(
        &self,
        cmdbuf: &CommandBufferRef,
        w: &GpuHcWeights<'_>,
        hyper: &BufferRef,
        bufs: &GpuHcBuffers,
        eps: f32,
        n_tokens: usize,
    ) {
        let (h, hc) = (bufs.hidden, bufs.hc);
        let width = h * hc;
        let n = n_tokens;
        debug_assert_eq!(w.mix_down.in_dim(), width);
        debug_assert_eq!(w.mix_up.out_dim(), width);
        assert!(n <= bufs.max_tokens(), "encode_gated_residual_mix_n: {n} tokens, buffers hold {}", bufs.max_tokens());
        if n == 0 {
            return;
        }
        self.encode_hc_norm_rows(cmdbuf, hyper, w.hc_norm.0, w.hc_norm.1, bufs.normed.raw(), h, width, n, eps);

        w.mix_down.encode_n(cmdbuf, bufs.normed.raw(), 0, bufs.low.raw(), 0, n);
        let lowrank = w.mix_down.out_dim();
        self.encode_silu_scale(cmdbuf, bufs.low.raw(), 1.0f32 / hc as f32, n * lowrank);
        w.mix_up.encode_n(cmdbuf, bufs.low.raw(), 0, bufs.mix.raw(), 0, n);
        self.encode_hc_mix_n(cmdbuf, bufs.mix.raw(), bufs.normed.raw(), bufs.mixed.raw(), h, hc, n);
        if let Some(bi) = w.block_inject {
            bi.encode_n(cmdbuf, bufs.normed.raw(), 0, bufs.inj.raw(), 0, n);
            self.encode_inj_weights_n(cmdbuf, bufs.inj.raw(), hc, n);
        }
    }

    /// `v[i] = silu(v[i] · scale)` for `n` values.
    pub fn encode_silu_scale(&self, cmdbuf: &CommandBufferRef, v: &BufferRef, scale: f32, n: usize) {
        if n == 0 {
            return;
        }
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.silu_scale);
        enc.set_buffer(0, Some(v), 0);
        set_const(enc, 1, &scale);
        set_const(enc, 2, &(n as u32));
        let (g, t) = grid1(n, 64);
        enc.dispatch_thread_groups(g, t);
        enc.end_encoding();
    }

    /// The copy mix for `n_tokens` rows: `out [n, H]`,
    /// `out[t, j] = mean_c sigmoid(mix[t, c·H + j]) · normed[t, c·H + j]`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_hc_mix_n(
        &self,
        cmdbuf: &CommandBufferRef,
        mix: &BufferRef,
        normed: &BufferRef,
        out: &BufferRef,
        hidden: usize,
        hc: usize,
        n_tokens: usize,
    ) {
        let total = n_tokens * hidden;
        if total == 0 {
            return;
        }
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.hc_mix);
        enc.set_buffer(0, Some(mix), 0);
        enc.set_buffer(1, Some(normed), 0);
        enc.set_buffer(2, Some(out), 0);
        set_const(enc, 3, &(hidden as u32));
        set_const(enc, 4, &(hc as u32));
        set_const(enc, 5, &(total as u32));
        let (g, t) = grid1(total, 256);
        enc.dispatch_thread_groups(g, t);
        enc.end_encoding();
    }

    /// Injection weights in place for `n_tokens` rows of `hc`:
    /// `inj = 2 · sigmoid(inj / hc)`.
    pub fn encode_inj_weights_n(&self, cmdbuf: &CommandBufferRef, inj: &BufferRef, hc: usize, n_tokens: usize) {
        let n = n_tokens * hc;
        if n == 0 {
            return;
        }
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.inj_weights);
        enc.set_buffer(0, Some(inj), 0);
        set_const(enc, 1, &(hc as u32));
        set_const(enc, 2, &(n as u32));
        let (g, t) = grid1(n, 32);
        enc.dispatch_thread_groups(g, t);
        enc.end_encoding();
    }

    /// `hyper[c · H + j] += inj[c] · block_out[j]`.
    pub fn encode_inject(
        &self,
        cmdbuf: &CommandBufferRef,
        hyper: &BufferRef,
        block_out: &BufferRef,
        block_off: u64,
        inj: &BufferRef,
        hidden: usize,
        hc: usize,
    ) {
        self.encode_inject_n(cmdbuf, hyper, block_out, block_off, inj, hidden, hc, 1)
    }

    /// [`Self::encode_inject`] for `n_tokens` rows: `hyper [n, hc · H]
    /// += block_out [n, H] ⊗ inj [n, hc]`, row by row.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_inject_n(
        &self,
        cmdbuf: &CommandBufferRef,
        hyper: &BufferRef,
        block_out: &BufferRef,
        block_off: u64,
        inj: &BufferRef,
        hidden: usize,
        hc: usize,
        n_tokens: usize,
    ) {
        let total = hidden * hc * n_tokens;
        if total == 0 {
            return;
        }
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.hc_inject);
        enc.set_buffer(0, Some(hyper), 0);
        enc.set_buffer(1, Some(block_out), block_off);
        enc.set_buffer(2, Some(inj), 0);
        set_const(enc, 3, &(hidden as u32));
        set_const(enc, 4, &(total as u32));
        set_const(enc, 5, &(hc as u32));
        let (g, t) = grid1(total, 256);
        enc.dispatch_thread_groups(g, t);
        enc.end_encoding();
    }

    /// `hyper = hidden.repeat(hc)`.
    pub fn encode_expand(&self, cmdbuf: &CommandBufferRef, hidden_buf: &BufferRef, hyper: &BufferRef, hidden: usize, hc: usize) {
        self.encode_expand_n(cmdbuf, hidden_buf, hyper, hidden, hc, 1)
    }

    /// [`Self::encode_expand`] for `n_tokens` rows: `hyper [n, hc · H]`
    /// from `hidden_buf [n, H]`.
    pub fn encode_expand_n(
        &self,
        cmdbuf: &CommandBufferRef,
        hidden_buf: &BufferRef,
        hyper: &BufferRef,
        hidden: usize,
        hc: usize,
        n_tokens: usize,
    ) {
        let total = hidden * hc * n_tokens;
        if total == 0 {
            return;
        }
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.hc_expand);
        enc.set_buffer(0, Some(hidden_buf), 0);
        enc.set_buffer(1, Some(hyper), 0);
        set_const(enc, 2, &(hidden as u32));
        set_const(enc, 3, &(total as u32));
        set_const(enc, 4, &(hc as u32));
        let (g, t) = grid1(total, 256);
        enc.dispatch_thread_groups(g, t);
        enc.end_encoding();
    }
}

/// One Gated Residual's weights on the GPU.
pub struct GpuHcWeights<'a> {
    /// BF16 gains `[hc · H]` (applied as `1 + w`) and their byte offset
    /// in the buffer (the weight file's buffer in the engine).
    pub hc_norm: (&'a BufferRef, u64),
    pub mix_down: &'a dyn GpuLinear,
    pub mix_up: &'a dyn GpuLinear,
    /// `None` for the final mixer.
    pub block_inject: Option<&'a dyn GpuLinear>,
}

/// Scratch and outputs of [`Q38GpuKernels::encode_gated_residual_mix`].
pub struct GpuHcBuffers {
    pub hidden: usize,
    pub hc: usize,
    /// `[hc · H]`.
    pub normed: MtlBuffer<f32>,
    /// `[lowrank]`.
    pub low: MtlBuffer<f32>,
    /// `[hc · H]`.
    pub mix: MtlBuffer<f32>,
    /// Block input `[H]`.
    pub mixed: MtlBuffer<f32>,
    /// Injection weights `[hc]`.
    pub inj: MtlBuffer<f32>,
}

impl GpuHcBuffers {
    pub fn new(device: &Device, hidden: usize, hc: usize, lowrank: usize) -> Self {
        Self::with_tokens(device, hidden, hc, lowrank, 1)
    }

    /// Buffers for up to `max_tokens` rows
    /// ([`Q38GpuKernels::encode_gated_residual_mix_n`]): each is
    /// `[max_tokens, …]`.
    pub fn with_tokens(device: &Device, hidden: usize, hc: usize, lowrank: usize, max_tokens: usize) -> Self {
        let n = max_tokens.max(1);
        Self {
            hidden,
            hc,
            normed: MtlBuffer::with_len(device, n * hc * hidden),
            low: MtlBuffer::with_len(device, n * lowrank),
            mix: MtlBuffer::with_len(device, n * hc * hidden),
            mixed: MtlBuffer::with_len(device, n * hidden),
            inj: MtlBuffer::with_len(device, n * hc),
        }
    }

    /// Rows the buffers hold.
    pub fn max_tokens(&self) -> usize {
        self.mixed.len() / self.hidden.max(1)
    }
}

/// Dense f32 `[out_dim, in_dim]` weights on the GPU (tests).
pub struct GpuDenseLinear {
    pipe: ComputePipelineState,
    pub w: MtlBuffer<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl GpuDenseLinear {
    pub fn new(device: &Device, k: &Q38GpuKernels, out_dim: usize, in_dim: usize, w: &[f32]) -> Self {
        assert_eq!(w.len(), out_dim * in_dim);
        Self {
            pipe: k.matvec_f32.clone(),
            w: MtlBuffer::with_data(device, w),
            out_dim,
            in_dim,
        }
    }
}

impl GpuLinear for GpuDenseLinear {
    fn in_dim(&self) -> usize {
        self.in_dim
    }
    fn out_dim(&self) -> usize {
        self.out_dim
    }
    fn encode(&self, cmdbuf: &CommandBufferRef, x: &BufferRef, x_off: u64, out: &BufferRef, out_off: u64) {
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipe);
        enc.set_buffer(0, Some(self.w.raw()), 0);
        enc.set_buffer(1, Some(x), x_off);
        enc.set_buffer(2, Some(out), out_off);
        set_const(enc, 3, &(self.in_dim as u32));
        enc.dispatch_thread_groups(MTLSize::new(self.out_dim as u64, 1, 1), MTLSize::new(32, 1, 1));
        enc.end_encoding();
    }
}

/// An MLX affine 4-bit matrix `[out_dim, in_dim]` with an explicit
/// group size (32 for this family's projections, Gated Residual and
/// PLE alike), read from a Metal buffer at byte offsets — the weight
/// file's buffer in the engine. The group is passed to the kernels at
/// run time, so it works under any variant feature (the engine's
/// `encode_matvec*` take the variant's compile-time `GROUP_SIZE`).
///
/// Kernels, by row count:
///
/// - 1 row: `dequant_matvec_4bit_qmv` when [`Self::qmv`] and the shape
///   fits (`in_dim % 32 == 0`, 16-byte aligned weight and input), else
///   `_v3` (`in_dim <= 4096`) or `_fast`.
/// - `n >= gemm_min_tokens` rows and a [`Kernels`] handle: MLX
///   `affine_qmm_t` at the group (32 or 64).
/// - other `n`: `_v3_n_tokens` / `_fast_n_tokens`, whose per-row
///   arithmetic is the one-row `_v3` / `_fast` kernel's — so with
///   `qmv = false` and no GEMM, `n` rows equal `n` one-row calls bit
///   for bit.
///
/// All of them match the CPU dequant reference to f32 rounding
/// (`tests/group32_kernels.rs`).
pub struct GpuQ4Linear {
    pub buf: metal::Buffer,
    pub w_off: u64,
    pub s_off: u64,
    pub b_off: u64,
    pub out_dim: usize,
    pub in_dim: usize,
    pub group: usize,
    /// Use `qmv` for one row when it fits (default true).
    pub qmv: bool,
    /// Rows from which the GEMM runs (default 16; `usize::MAX` never).
    pub gemm_min_tokens: usize,
    pipes: MatvecPipelines,
    kernels: Option<Kernels>,
}

impl GpuQ4Linear {
    /// `buf` holds the packed weights at `w_off`, bf16 scales at
    /// `s_off` and bf16 biases at `b_off`. The GEMM is enabled
    /// (`metal.kernels()`) when the group is 32 or 64.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        metal: &mut MetalContext,
        buf: metal::Buffer,
        w_off: u64,
        s_off: u64,
        b_off: u64,
        out_dim: usize,
        in_dim: usize,
        group: usize,
    ) -> Result<Self, MetalError> {
        assert!(group % 8 == 0 && in_dim % group == 0, "GpuQ4Linear: in_dim {in_dim} not a multiple of group {group}");
        let pipes = MatvecPipelines::fetch(metal)?;
        let kernels = streammoe_metal::QMM_GROUP_SIZES.contains(&(group as u32)).then(|| metal.kernels().clone());
        Ok(Self { buf, w_off, s_off, b_off, out_dim, in_dim, group, qmv: true, gemm_min_tokens: 16, pipes, kernels })
    }

    /// Copy MLX bytes (`weight` u32-packed, `scales` / `biases` bf16)
    /// into a new shared buffer, 16-byte aligned sections.
    #[allow(clippy::too_many_arguments)]
    pub fn from_bytes(
        metal: &mut MetalContext,
        w: &[u8],
        s: &[u8],
        b: &[u8],
        out_dim: usize,
        in_dim: usize,
        group: usize,
    ) -> Result<Self, MetalError> {
        assert_eq!(w.len(), out_dim * in_dim / 2, "GpuQ4Linear: weight bytes");
        assert_eq!(s.len(), out_dim * (in_dim / group) * 2, "GpuQ4Linear: scale bytes");
        assert_eq!(b.len(), s.len(), "GpuQ4Linear: bias bytes");
        let s_off = w.len().next_multiple_of(16);
        let b_off = (s_off + s.len()).next_multiple_of(16);
        let mut bytes = vec![0u8; b_off + b.len()];
        bytes[..w.len()].copy_from_slice(w);
        bytes[s_off..s_off + s.len()].copy_from_slice(s);
        bytes[b_off..].copy_from_slice(b);
        let buf = metal.device().new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Self::new(metal, buf, 0, s_off as u64, b_off as u64, out_dim, in_dim, group)
    }

    fn set_common(&self, enc: &metal::ComputeCommandEncoderRef, x: &BufferRef, x_off: u64, out: &BufferRef, out_off: u64) {
        enc.set_buffer(0, Some(&self.buf), self.w_off);
        enc.set_buffer(1, Some(&self.buf), self.s_off);
        enc.set_buffer(2, Some(&self.buf), self.b_off);
        enc.set_buffer(3, Some(x), x_off);
        enc.set_buffer(4, Some(out), out_off);
        set_const(enc, 5, &(self.out_dim as u32));
        set_const(enc, 6, &(self.in_dim as u32));
        set_const(enc, 7, &(self.group as u32));
    }
}

impl GpuLinear for GpuQ4Linear {
    fn in_dim(&self) -> usize {
        self.in_dim
    }
    fn out_dim(&self) -> usize {
        self.out_dim
    }
    fn encode(&self, cmdbuf: &CommandBufferRef, x: &BufferRef, x_off: u64, out: &BufferRef, out_off: u64) {
        let tiles = self.out_dim.div_ceil(8) as u64;
        let use_qmv = self.qmv && self.in_dim % 32 == 0 && self.group % 32 == 0 && self.w_off % 16 == 0 && x_off % 16 == 0;
        let enc = cmdbuf.new_compute_command_encoder();
        let (pso, tgs, threads) = if use_qmv {
            (&self.pipes.qmv_4bit, tiles, 64)
        } else if self.in_dim <= 4096 {
            (&self.pipes.v3_4bit, tiles, 256)
        } else {
            (&self.pipes.fast_4bit, self.out_dim as u64, 64)
        };
        enc.set_compute_pipeline_state(pso);
        self.set_common(enc, x, x_off, out, out_off);
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(threads, 1, 1));
        enc.end_encoding();
    }
    fn encode_n(&self, cmdbuf: &CommandBufferRef, x: &BufferRef, x_off: u64, out: &BufferRef, out_off: u64, n_tokens: usize) {
        if n_tokens == 0 {
            return;
        }
        if n_tokens == 1 && self.qmv {
            return self.encode(cmdbuf, x, x_off, out, out_off);
        }
        if let Some(k) = self.kernels.as_ref().filter(|_| n_tokens >= self.gemm_min_tokens && self.in_dim % 32 == 0) {
            k.encode(
                cmdbuf,
                &QmmCall {
                    weights: QuantWeights {
                        buffer: &self.buf,
                        packed_offset: self.w_off,
                        scales_offset: self.s_off,
                        biases_offset: self.b_off,
                        group_size: self.group as u32,
                    },
                    input: x,
                    input_offset: x_off,
                    output: out,
                    output_offset: out_off,
                    in_dim: self.in_dim as u32,
                    out_dim: self.out_dim as u32,
                    n_tokens: n_tokens as u32,
                },
            );
            return;
        }
        let n = n_tokens as u32;
        let tiles = self.out_dim.div_ceil(8) as u32;
        let enc = cmdbuf.new_compute_command_encoder();
        if self.in_dim <= 4096 {
            enc.set_compute_pipeline_state(&self.pipes.v3_4bit_n);
            self.set_common(enc, x, x_off, out, out_off);
            set_const(enc, 8, &n);
            set_const(enc, 9, &tiles);
            enc.dispatch_thread_groups(MTLSize::new(tiles as u64 * n as u64, 1, 1), MTLSize::new(256, 1, 1));
        } else {
            enc.set_compute_pipeline_state(&self.pipes.fast_4bit_n);
            self.set_common(enc, x, x_off, out, out_off);
            set_const(enc, 8, &n);
            enc.dispatch_thread_groups(MTLSize::new(self.out_dim as u64 * n as u64, 1, 1), MTLSize::new(64, 1, 1));
        }
        enc.end_encoding();
    }
}

#[cfg(test)]
mod tests {
    use super::super::gated_residual::{
        gated_residual_mix, hc_expand, inject, DenseLinear, GatedResidualScratch, GatedResidualWeights,
    };
    use super::super::tiny::gen_tensor;
    use super::*;
    use crate::riir::io::embedding::bf16_to_f32;

    fn f32_to_bf16_rne(x: f32) -> u16 {
        let b = x.to_bits();
        let r = 0x7FFF + ((b >> 16) & 1);
        ((b + r) >> 16) as u16
    }

    fn max_rel(name: &str, gpu: &[f32], cpu: &[f32]) -> f32 {
        assert_eq!(gpu.len(), cpu.len());
        let mut worst = 0.0f32;
        for (i, (&g, &c)) in gpu.iter().zip(cpu).enumerate() {
            assert!(g.is_finite(), "{name}[{i}] not finite");
            worst = worst.max((g - c).abs() / (1.0 + c.abs()));
        }
        eprintln!("[q38 gpu {name}] {} values, worst rel diff {worst:.2e}", gpu.len());
        worst
    }

    /// CPU vs Metal for the whole Gated Residual round trip — expand,
    /// mix with inject weights, inject, final mixer — at the tiny size
    /// and at the real one (H 2560, hc 4, rank 320).
    fn cpu_vs_metal(hidden: usize, hc: usize, lowrank: usize, tag: &str) {
        let Some(device) = Device::system_default() else {
            eprintln!("[q38 gpu] skipping: no Metal device");
            return;
        };
        let k = Q38GpuKernels::new(&device).expect("compile q38 kernels");
        let width = hidden * hc;
        // Gains as BF16, the stored format; the CPU path gets the same
        // values widened.
        let gain_bf16: Vec<u16> = gen_tensor(&format!("{tag}.hc_norm"), width, 0.2, 0.0)
            .into_iter()
            .map(f32_to_bf16_rne)
            .collect();
        let gain: Vec<f32> = gain_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        let down_w = gen_tensor(&format!("{tag}.down"), lowrank * width, 1.0 / (width as f32).sqrt(), 0.0);
        let up_w = gen_tensor(&format!("{tag}.up"), width * lowrank, 2.0 / (lowrank as f32).sqrt(), 0.0);
        let inj_w = gen_tensor(&format!("{tag}.inject"), hc * width, 1.0 / (width as f32).sqrt(), 0.0);
        let emb = gen_tensor(&format!("{tag}.emb"), hidden, 1.0, 0.0);
        let block = gen_tensor(&format!("{tag}.block"), hidden, 1.0, 0.0);

        // CPU.
        let (down, up, bi) = (
            DenseLinear::new(lowrank, width, down_w.clone()),
            DenseLinear::new(width, lowrank, up_w.clone()),
            DenseLinear::new(hc, width, inj_w.clone()),
        );
        let cw = GatedResidualWeights { hc_norm: &gain, mix_down: &down, mix_up: &up, block_inject: Some(&bi) };
        let mut hyper = vec![0.0; width];
        hc_expand(&emb, hc, &mut hyper);
        // Make the copies differ so the per-copy norm and the mean
        // matter: one inject first.
        let mut s = GatedResidualScratch::new(width, lowrank);
        let mut mixed0 = vec![0.0; hidden];
        let mut inj0 = vec![0.0; hc];
        gated_residual_mix(&cw, hidden, hc, 1e-6, &hyper, &mut s, &mut mixed0, Some(&mut inj0));
        inject(&mut hyper, &block, &inj0);
        let mut mixed1 = vec![0.0; hidden];
        let mut inj1 = vec![0.0; hc];
        gated_residual_mix(&cw, hidden, hc, 1e-6, &hyper, &mut s, &mut mixed1, Some(&mut inj1));
        let cw_final = GatedResidualWeights { block_inject: None, ..cw };
        let mut fin = vec![0.0; hidden];
        gated_residual_mix(&cw_final, hidden, hc, 1e-6, &hyper, &mut s, &mut fin, None);

        // Metal, one command buffer per stage so each can be read back.
        let gd = GpuDenseLinear::new(&device, &k, lowrank, width, &down_w);
        let gu = GpuDenseLinear::new(&device, &k, width, lowrank, &up_w);
        let gi = GpuDenseLinear::new(&device, &k, hc, width, &inj_w);
        let gain_buf = MtlBuffer::<u16>::with_data(&device, &gain_bf16);
        let gw = GpuHcWeights { hc_norm: (gain_buf.raw(), 0), mix_down: &gd, mix_up: &gu, block_inject: Some(&gi) };
        let bufs = GpuHcBuffers::new(&device, hidden, hc, lowrank);
        let emb_buf = MtlBuffer::with_data(&device, &emb);
        let block_buf = MtlBuffer::with_data(&device, &block);
        let hyper_buf = MtlBuffer::<f32>::with_len(&device, width);
        let queue = device.new_command_queue();

        let cb = queue.new_command_buffer();
        k.encode_expand(cb, emb_buf.raw(), hyper_buf.raw(), hidden, hc);
        k.encode_gated_residual_mix(cb, &gw, hyper_buf.raw(), &bufs, 1e-6);
        cb.commit();
        cb.wait_until_completed();
        let tol = 2e-5;
        assert!(max_rel(&format!("{tag} mixed0"), &bufs.mixed.to_vec(), &mixed0) < tol);
        assert!(max_rel(&format!("{tag} inj0"), &bufs.inj.to_vec(), &inj0) < tol);

        let cb = queue.new_command_buffer();
        k.encode_inject(cb, hyper_buf.raw(), block_buf.raw(), 0, bufs.inj.raw(), hidden, hc);
        k.encode_gated_residual_mix(cb, &gw, hyper_buf.raw(), &bufs, 1e-6);
        cb.commit();
        cb.wait_until_completed();
        assert!(max_rel(&format!("{tag} hyper"), &hyper_buf.to_vec(), &hyper) < tol);
        assert!(max_rel(&format!("{tag} mixed1"), &bufs.mixed.to_vec(), &mixed1) < tol);
        assert!(max_rel(&format!("{tag} inj1"), &bufs.inj.to_vec(), &inj1) < tol);

        let gw_final = GpuHcWeights { block_inject: None, ..gw };
        let cb = queue.new_command_buffer();
        k.encode_gated_residual_mix(cb, &gw_final, hyper_buf.raw(), &bufs, 1e-6);
        cb.commit();
        cb.wait_until_completed();
        assert!(max_rel(&format!("{tag} final"), &bufs.mixed.to_vec(), &fin) < tol);
    }

    #[test]
    fn gated_residual_cpu_vs_metal_tiny() {
        cpu_vs_metal(32, 4, 8, "gpu_tiny");
    }

    #[test]
    fn gated_residual_cpu_vs_metal_real_size() {
        cpu_vs_metal(2560, 4, 320, "gpu_real");
    }

    /// `n` tokens through `encode_expand_n` → `encode_gated_residual_mix_n`
    /// → `encode_inject_n` → `encode_gated_residual_mix_n` against the
    /// CPU per token, and the batched mix against the one-token GPU
    /// mix row by row (bit for bit: the dense linear's `encode_n` is
    /// per row).
    fn cpu_vs_metal_n(hidden: usize, hc: usize, lowrank: usize, n: usize, tag: &str) {
        let Some(device) = Device::system_default() else {
            eprintln!("[q38 gpu] skipping: no Metal device");
            return;
        };
        let k = Q38GpuKernels::new(&device).expect("compile q38 kernels");
        let width = hidden * hc;
        let gain_bf16: Vec<u16> =
            gen_tensor(&format!("{tag}.hc_norm"), width, 0.2, 0.0).into_iter().map(f32_to_bf16_rne).collect();
        let gain: Vec<f32> = gain_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        let down_w = gen_tensor(&format!("{tag}.down"), lowrank * width, 1.0 / (width as f32).sqrt(), 0.0);
        let up_w = gen_tensor(&format!("{tag}.up"), width * lowrank, 2.0 / (lowrank as f32).sqrt(), 0.0);
        let inj_w = gen_tensor(&format!("{tag}.inject"), hc * width, 1.0 / (width as f32).sqrt(), 0.0);
        let emb = gen_tensor(&format!("{tag}.emb"), n * hidden, 1.0, 0.0);
        let block = gen_tensor(&format!("{tag}.block"), n * hidden, 1.0, 0.0);

        let (down, up, bi) = (
            DenseLinear::new(lowrank, width, down_w.clone()),
            DenseLinear::new(width, lowrank, up_w.clone()),
            DenseLinear::new(hc, width, inj_w.clone()),
        );
        let cw = GatedResidualWeights { hc_norm: &gain, mix_down: &down, mix_up: &up, block_inject: Some(&bi) };
        let mut s = GatedResidualScratch::new(width, lowrank);
        let mut hyper = vec![0.0; n * width];
        let (mut mixed0, mut inj0, mut mixed1, mut inj1) =
            (vec![0.0; n * hidden], vec![0.0; n * hc], vec![0.0; n * hidden], vec![0.0; n * hc]);
        for t in 0..n {
            let ht = &mut hyper[t * width..(t + 1) * width];
            hc_expand(&emb[t * hidden..(t + 1) * hidden], hc, ht);
            let (m, i) = (&mut mixed0[t * hidden..(t + 1) * hidden], &mut inj0[t * hc..(t + 1) * hc]);
            gated_residual_mix(&cw, hidden, hc, 1e-6, ht, &mut s, m, Some(&mut *i));
            inject(ht, &block[t * hidden..(t + 1) * hidden], i);
            let (m, i) = (&mut mixed1[t * hidden..(t + 1) * hidden], &mut inj1[t * hc..(t + 1) * hc]);
            gated_residual_mix(&cw, hidden, hc, 1e-6, ht, &mut s, m, Some(i));
        }

        let gd = GpuDenseLinear::new(&device, &k, lowrank, width, &down_w);
        let gu = GpuDenseLinear::new(&device, &k, width, lowrank, &up_w);
        let gi = GpuDenseLinear::new(&device, &k, hc, width, &inj_w);
        let gain_buf = MtlBuffer::<u16>::with_data(&device, &gain_bf16);
        let gw = GpuHcWeights { hc_norm: (gain_buf.raw(), 0), mix_down: &gd, mix_up: &gu, block_inject: Some(&gi) };
        let bufs = GpuHcBuffers::with_tokens(&device, hidden, hc, lowrank, n);
        assert_eq!(bufs.max_tokens(), n);
        let emb_buf = MtlBuffer::with_data(&device, &emb);
        let block_buf = MtlBuffer::with_data(&device, &block);
        let hyper_buf = MtlBuffer::<f32>::with_len(&device, n * width);
        let queue = device.new_command_queue();
        let cb = queue.new_command_buffer();
        k.encode_expand_n(cb, emb_buf.raw(), hyper_buf.raw(), hidden, hc, n);
        k.encode_gated_residual_mix_n(cb, &gw, hyper_buf.raw(), &bufs, 1e-6, n);
        cb.commit();
        cb.wait_until_completed();
        let tol = 2e-5;
        assert!(max_rel(&format!("{tag} n{n} mixed0"), &bufs.mixed.to_vec(), &mixed0) < tol);
        assert!(max_rel(&format!("{tag} n{n} inj0"), &bufs.inj.to_vec(), &inj0) < tol);
        let cb = queue.new_command_buffer();
        k.encode_inject_n(cb, hyper_buf.raw(), block_buf.raw(), 0, bufs.inj.raw(), hidden, hc, n);
        k.encode_gated_residual_mix_n(cb, &gw, hyper_buf.raw(), &bufs, 1e-6, n);
        cb.commit();
        cb.wait_until_completed();
        assert!(max_rel(&format!("{tag} n{n} hyper"), &hyper_buf.to_vec(), &hyper) < tol);
        let (mixed_n, inj_n) = (bufs.mixed.to_vec(), bufs.inj.to_vec());
        assert!(max_rel(&format!("{tag} n{n} mixed1"), &mixed_n, &mixed1) < tol);
        assert!(max_rel(&format!("{tag} n{n} inj1"), &inj_n, &inj1) < tol);

        // Batched vs one token at a time on the GPU: bit for bit.
        let hyper_n = hyper_buf.to_vec();
        let one = GpuHcBuffers::new(&device, hidden, hc, lowrank);
        for t in 0..n {
            let row = MtlBuffer::with_data(&device, &hyper_n[t * width..(t + 1) * width]);
            let cb = queue.new_command_buffer();
            k.encode_gated_residual_mix(cb, &gw, row.raw(), &one, 1e-6);
            cb.commit();
            cb.wait_until_completed();
            assert!(one.mixed.to_vec().iter().zip(&mixed_n[t * hidden..]).all(|(a, b)| a.to_bits() == b.to_bits()), "{tag}: row {t} mixed");
            assert!(one.inj.to_vec().iter().zip(&inj_n[t * hc..]).all(|(a, b)| a.to_bits() == b.to_bits()), "{tag}: row {t} inj");
        }
    }

    #[test]
    fn gated_residual_n_tokens_cpu_vs_metal_tiny() {
        cpu_vs_metal_n(32, 4, 8, 5, "gpu_n_tiny");
    }

    #[test]
    fn gated_residual_n_tokens_cpu_vs_metal_real_size() {
        cpu_vs_metal_n(2560, 4, 320, 6, "gpu_n_real");
    }

    // --- GpuQ4Linear ---------------------------------------------------

    /// Random MLX 4-bit bytes `(weight, scales, biases)`.
    pub(crate) fn rand_q4(out_dim: usize, in_dim: usize, group: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut x = seed | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let w: Vec<u8> = (0..out_dim * in_dim / 2).map(|_| next() as u8).collect();
        let groups = out_dim * in_dim / group;
        let mut sb = |scale: f32| -> Vec<u8> {
            (0..groups)
                .flat_map(|_| {
                    let u = (next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                    f32_to_bf16_rne(u * scale).to_le_bytes()
                })
                .collect()
        };
        let s = sb(0.05);
        let b = sb(0.02);
        (w, s, b)
    }

    /// GpuQ4Linear at group 32 on the PLE and Gated Residual shapes:
    /// every kernel choice against the CPU dequant
    /// (`ple::matvec_4bit_tokens`), and `n` rows on `_n_tokens` against
    /// `n` one-row calls bit for bit.
    #[test]
    fn q4_linear_group32_matches_cpu() {
        use super::super::ple::matvec_4bit_tokens;
        let mut metal = MetalContext::new().expect("Metal");
        let queue = metal.queue_clone();
        let shapes = [
            (10240usize, 2560usize, "ple key_proj"),
            (2560, 2560, "ple value_proj"),
            (320, 10240, "hc mix_down"),
            (10240, 320, "hc mix_up"),
            (4, 10240, "hc block_inject"),
            (128, 128, "ple tiny key_proj"),
        ];
        const N: usize = 20;
        for (si, &(out_dim, in_dim, name)) in shapes.iter().enumerate() {
            let (w, s, b) = rand_q4(out_dim, in_dim, 32, 0x9e37 + si as u64);
            let mut lin = GpuQ4Linear::from_bytes(&mut metal, &w, &s, &b, out_dim, in_dim, 32).unwrap();
            let x = gen_tensor(&format!("q4lin.{si}"), N * in_dim, 1.0, 0.0);
            let mut want = vec![0f32; N * out_dim];
            matvec_4bit_tokens(&w, &s, &b, 32, in_dim, out_dim, &x, &mut want).unwrap();
            let xb = MtlBuffer::with_data(metal.device(), &x);
            let ob = MtlBuffer::<f32>::with_len(metal.device(), N * out_dim);
            let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            let check = |label: &str, got: &[f32], want: &[f32]| {
                let d = got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                eprintln!("[q4 linear] {name} {label}: max|diff| / max|ref| {:.2e}", d / scale);
                assert!(d <= 2e-5 * scale, "{name} {label}");
            };
            for (label, qmv, gemm_min, n) in
                [("1 row qmv", true, usize::MAX, 1), ("1 row v3/fast", false, usize::MAX, 1), ("n rows matvec", false, usize::MAX, N), ("n rows gemm", false, 1, N)]
            {
                lin.qmv = qmv;
                lin.gemm_min_tokens = gemm_min;
                let cb = queue.new_command_buffer();
                lin.encode_n(cb, xb.raw(), 0, ob.raw(), 0, n);
                cb.commit();
                cb.wait_until_completed();
                check(label, &ob.to_vec()[..n * out_dim], &want[..n * out_dim]);
            }
            // n rows on the matvec_n kernel == n one-row v3 / fast calls.
            lin.qmv = false;
            lin.gemm_min_tokens = usize::MAX;
            let cb = queue.new_command_buffer();
            lin.encode_n(cb, xb.raw(), 0, ob.raw(), 0, N);
            cb.commit();
            cb.wait_until_completed();
            let batched = ob.to_vec();
            let cb = queue.new_command_buffer();
            for t in 0..N as u64 {
                lin.encode(cb, xb.raw(), t * 4 * in_dim as u64, ob.raw(), t * 4 * out_dim as u64);
            }
            cb.commit();
            cb.wait_until_completed();
            assert!(batched.iter().zip(ob.to_vec()).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: batched != per row");
        }
    }
}
