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
//! The matvecs go through [`GpuLinear`]. The real model's weights are
//! MLX 4-bit group 32 — the group-size-32 kernels are another module's
//! work; anything that can encode `out = W · x` on a command buffer
//! plugs in. [`GpuDenseLinear`] (f32) serves the tests.
//!
//! The kernels live in their own small library ([`Q38_SHADERS`]),
//! compiled once by [`Q38GpuKernels::new`], so they don't touch
//! `shaders.metal`.

use metal::{
    BufferRef, CommandBufferRef, CompileOptions, ComputePipelineState, Device, MTLSize,
};

use crate::riir::backend::gpu::metal::MtlBuffer;

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
// x[g*group .. (g+1)*group). Gains are bf16.
kernel void q38_hc_norm(
    device const float*  x     [[buffer(0)]],
    device const ushort* gain  [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       group [[buffer(3)]],
    constant float&      eps   [[buffer(4)]],
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
        out[base + i] = x[base + i] * inv * (1.0f + q38_bf16(gain[base + i]));
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

// out[j] = (1/hc) sum_c sigmoid(mix[c*H + j]) * normed[c*H + j]
kernel void q38_hc_mix(
    device const float* mix    [[buffer(0)]],
    device const float* normed [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      hidden [[buffer(3)]],
    constant uint&      hc     [[buffer(4)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= hidden) return;
    float acc = 0.0f;
    for (uint c = 0; c < hc; c++) {
        uint k = c * hidden + j;
        acc += q38_sigmoid(mix[k]) * normed[k];
    }
    out[j] = acc / float(hc);
}

// inj[c] = 2 * sigmoid(inj[c] / hc)
kernel void q38_inj_weights(
    device float*  inj [[buffer(0)]],
    constant uint& hc  [[buffer(1)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= hc) return;
    inj[c] = 2.0f * q38_sigmoid(inj[c] / float(hc));
}

// hyper[c*H + j] += inj[c] * block[j]
kernel void q38_hc_inject(
    device float*       hyper  [[buffer(0)]],
    device const float* block  [[buffer(1)]],
    device const float* inj    [[buffer(2)]],
    constant uint&      hidden [[buffer(3)]],
    constant uint&      total  [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    uint c = i / hidden;
    uint j = i - c * hidden;
    hyper[i] = fma(block[j], inj[c], hyper[i]);
}

// hyper[c*H + j] = h[j]
kernel void q38_hc_expand(
    device const float* h      [[buffer(0)]],
    device float*       hyper  [[buffer(1)]],
    constant uint&      hidden [[buffer(2)]],
    constant uint&      total  [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    hyper[i] = h[i % hidden];
}
"#;

/// `out = W · x` encoded on a command buffer. `x` / `out` are f32
/// buffers with byte offsets.
pub trait GpuLinear {
    fn in_dim(&self) -> usize;
    fn out_dim(&self) -> usize;
    fn encode(&self, cmdbuf: &CommandBufferRef, x: &BufferRef, x_off: u64, out: &BufferRef, out_off: u64);
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
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.hc_norm);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(gain), gain_off);
        enc.set_buffer(2, Some(out), 0);
        set_const(enc, 3, &(group as u32));
        set_const(enc, 4, &eps);
        enc.dispatch_thread_groups(MTLSize::new(n_groups as u64, 1, 1), MTLSize::new(256, 1, 1));
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
        let (h, hc) = (bufs.hidden, bufs.hc);
        let width = h * hc;
        debug_assert_eq!(w.mix_down.in_dim(), width);
        debug_assert_eq!(w.mix_up.out_dim(), width);
        self.encode_hc_norm(cmdbuf, hyper, w.hc_norm.0, w.hc_norm.1, bufs.normed.raw(), h, hc, eps);

        w.mix_down.encode(cmdbuf, bufs.normed.raw(), 0, bufs.low.raw(), 0);
        let lowrank = w.mix_down.out_dim();
        {
            let enc = cmdbuf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.silu_scale);
            enc.set_buffer(0, Some(bufs.low.raw()), 0);
            set_const(enc, 1, &(1.0f32 / hc as f32));
            set_const(enc, 2, &(lowrank as u32));
            let (g, t) = grid1(lowrank, 64);
            enc.dispatch_thread_groups(g, t);
            enc.end_encoding();
        }
        w.mix_up.encode(cmdbuf, bufs.low.raw(), 0, bufs.mix.raw(), 0);
        {
            let enc = cmdbuf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.hc_mix);
            enc.set_buffer(0, Some(bufs.mix.raw()), 0);
            enc.set_buffer(1, Some(bufs.normed.raw()), 0);
            enc.set_buffer(2, Some(bufs.mixed.raw()), 0);
            set_const(enc, 3, &(h as u32));
            set_const(enc, 4, &(hc as u32));
            let (g, t) = grid1(h, 256);
            enc.dispatch_thread_groups(g, t);
            enc.end_encoding();
        }
        if let Some(bi) = w.block_inject {
            bi.encode(cmdbuf, bufs.normed.raw(), 0, bufs.inj.raw(), 0);
            let enc = cmdbuf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.inj_weights);
            enc.set_buffer(0, Some(bufs.inj.raw()), 0);
            set_const(enc, 1, &(hc as u32));
            let (g, t) = grid1(hc, 32);
            enc.dispatch_thread_groups(g, t);
            enc.end_encoding();
        }
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
        let total = hidden * hc;
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.hc_inject);
        enc.set_buffer(0, Some(hyper), 0);
        enc.set_buffer(1, Some(block_out), block_off);
        enc.set_buffer(2, Some(inj), 0);
        set_const(enc, 3, &(hidden as u32));
        set_const(enc, 4, &(total as u32));
        let (g, t) = grid1(total, 256);
        enc.dispatch_thread_groups(g, t);
        enc.end_encoding();
    }

    /// `hyper = hidden.repeat(hc)`.
    pub fn encode_expand(&self, cmdbuf: &CommandBufferRef, hidden_buf: &BufferRef, hyper: &BufferRef, hidden: usize, hc: usize) {
        let total = hidden * hc;
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.hc_expand);
        enc.set_buffer(0, Some(hidden_buf), 0);
        enc.set_buffer(1, Some(hyper), 0);
        set_const(enc, 2, &(hidden as u32));
        set_const(enc, 3, &(total as u32));
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
        Self {
            hidden,
            hc,
            normed: MtlBuffer::with_len(device, hc * hidden),
            low: MtlBuffer::with_len(device, lowrank),
            mix: MtlBuffer::with_len(device, hc * hidden),
            mixed: MtlBuffer::with_len(device, hidden),
            inj: MtlBuffer::with_len(device, hc),
        }
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
}
