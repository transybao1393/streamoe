//! The PLE math after the projections, on the GPU (and the CPU
//! backend), for the engine.
//!
//! [`super::ple::forward_with_projections`] runs `norm_key`,
//! `norm_query`, the signed-sqrt gate, `norm_conv` and the dilated
//! depthwise conv on the host, so a GPU engine had to read the
//! residual back at the PLE layer. Here they are two family ops
//! ([`PleOp`], run through `FamOp::Ple`):
//!
//! ```text
//! Gate:  u_c = sigmoid(sgnsqrt((norm_key(k)_c · norm_query(x)_c) / √H)) · v      per token, stream c
//! Conv:  n = norm_conv(u);  y = silu(Σ_j w[ch, j] · hist[t - (K-1-j)·D][ch]);  x += u + y
//!        conv state (the last (K-1)·D rows of n) updated in place
//! ```
//!
//! With the projections as two graph matvecs (`key_proj`, `value_proj`)
//! and the conv state in a GPU buffer, the whole PLE inject stays in
//! the token's command buffer; only the n-gram rows (a host table
//! lookup) are uploaded. Both ops take `n_tokens` rows (the conv runs
//! the rows in order, each tap reading the state or an earlier row of
//! the same call), so a prefill chunk equals its tokens one at a time
//! up to the GPU's summation order.
//!
//! Gains are the stored zero-centred bf16 values (`1 + w` in the
//! kernels), at byte offsets into the weight file, like the other
//! family ops.

use std::cell::RefCell;

use metal::{CompileOptions, ComputePipelineState, Device, MTLSize, NSUInteger};

use super::engine_exec::FBuf;
use super::gated_residual_gpu::Q38GpuError;
use crate::riir::backend::gpu::MetalBufferPool;
use crate::riir::backend::{Backend, BufferPool, CpuBackend};
use crate::riir::io::embedding::bf16_to_f32;

/// A PLE op (see the module docs). Offsets are byte offsets into the
/// weight file.
#[derive(Debug, Clone, Copy)]
pub enum PleOp {
    /// `u [T, hc·H] = sigmoid(g) · v` with `g` the signed-sqrt gate of
    /// `norm_key(k)` and `norm_query(x)` per stream; `k`, `x`
    /// `[T, hc·H]`, `v [T, H]`.
    Gate {
        k: FBuf,
        x: FBuf,
        v: FBuf,
        u: FBuf,
        norm_key_off: u64,
        norm_query_off: u64,
        hidden: u32,
        hc: u32,
        n_tokens: u32,
        eps: f32,
    },
    /// `n = norm_conv(u)` (into `n [T, hc·H]`), the dilated depthwise
    /// conv over `state` (`(K-1)·D` rows of `hc·H`, oldest first) and
    /// `n`, then `x += u + silu(conv)`; `state` keeps the last rows.
    Conv {
        u: FBuf,
        n: FBuf,
        state: FBuf,
        x: FBuf,
        norm_conv_off: u64,
        conv_off: u64,
        hidden: u32,
        hc: u32,
        kernel: u32,
        dilation: u32,
        n_tokens: u32,
        eps: f32,
    },
}

/// Metal source of the PLE kernels.
pub const PLE_GPU_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float q38p_bf16(ushort b) { return as_type<float>(uint(b) << 16); }

inline float q38p_tg_sum(float v, threadgroup float* sh, uint lid, uint tg) {
    float s = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
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

// One threadgroup per (token t, stream s): g = t * hc + s.
kernel void q38p_ple_gate(
    device const float*  k      [[buffer(0)]],
    device const float*  x      [[buffer(1)]],
    device const float*  v      [[buffer(2)]],
    device float*        u      [[buffer(3)]],
    device const ushort* gk     [[buffer(4)]],
    device const ushort* gq     [[buffer(5)]],
    constant uint&       hidden [[buffer(6)]],
    constant uint&       hc     [[buffer(7)]],
    constant float&      eps    [[buffer(8)]],
    uint g   [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    threadgroup float sh[32];
    uint t = g / hc, s = g % hc;
    uint base = t * hc * hidden + s * hidden;
    uint goff = s * hidden;
    float a = 0.0f, b = 0.0f;
    for (uint i = lid; i < hidden; i += tg) {
        float kv = k[base + i], xv = x[base + i];
        a += kv * kv;
        b += xv * xv;
    }
    float inv_k = 1.0f / sqrt(q38p_tg_sum(a, sh, lid, tg) / float(hidden) + eps);
    float inv_q = 1.0f / sqrt(q38p_tg_sum(b, sh, lid, tg) / float(hidden) + eps);
    float d = 0.0f;
    for (uint i = lid; i < hidden; i += tg) {
        float kn = k[base + i] * inv_k * (1.0f + q38p_bf16(gk[goff + i]));
        float qn = x[base + i] * inv_q * (1.0f + q38p_bf16(gq[goff + i]));
        d += kn * qn;
    }
    float gg = q38p_tg_sum(d, sh, lid, tg) / sqrt(float(hidden));
    float sgn = gg > 0.0f ? 1.0f : (gg < 0.0f ? -1.0f : 0.0f);
    gg = sqrt(max(fabs(gg), 1e-6f)) * sgn;
    float sg = 1.0f / (1.0f + exp(-gg));
    for (uint i = lid; i < hidden; i += tg) {
        u[base + i] = sg * v[t * hidden + i];
    }
}

// n = rms_norm(u) * (1 + w) per group of `hidden` (T * hc groups).
kernel void q38p_ple_conv_norm(
    device const float*  u      [[buffer(0)]],
    device float*        n      [[buffer(1)]],
    device const ushort* gc     [[buffer(2)]],
    constant uint&       hidden [[buffer(3)]],
    constant uint&       hc     [[buffer(4)]],
    constant float&      eps    [[buffer(5)]],
    uint g   [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    threadgroup float sh[32];
    uint base = g * hidden;
    uint goff = (g % hc) * hidden;
    float a = 0.0f;
    for (uint i = lid; i < hidden; i += tg) { float vv = u[base + i]; a += vv * vv; }
    float inv = 1.0f / sqrt(q38p_tg_sum(a, sh, lid, tg) / float(hidden) + eps);
    for (uint i = lid; i < hidden; i += tg) {
        n[base + i] = u[base + i] * inv * (1.0f + q38p_bf16(gc[goff + i]));
    }
}

// One thread per channel: the T rows in order, then the state shift.
kernel void q38p_ple_conv(
    device const float*  u        [[buffer(0)]],
    device const float*  n        [[buffer(1)]],
    device float*        state    [[buffer(2)]],
    device float*        x        [[buffer(3)]],
    device const ushort* w        [[buffer(4)]],
    constant uint&       width    [[buffer(5)]],
    constant uint&       kernel_k [[buffer(6)]],
    constant uint&       dil      [[buffer(7)]],
    constant uint&       n_tokens [[buffer(8)]],
    uint ch [[thread_position_in_grid]])
{
    if (ch >= width) return;
    uint s_rows = (kernel_k - 1) * dil;
    for (uint t = 0; t < n_tokens; t++) {
        float acc = 0.0f;
        for (uint j = 0; j < kernel_k; j++) {
            uint src = s_rows + t - (kernel_k - 1 - j) * dil;
            float hv = src < s_rows ? state[src * width + ch] : n[(src - s_rows) * width + ch];
            acc += q38p_bf16(w[ch * kernel_k + j]) * hv;
        }
        float y = acc / (1.0f + exp(-acc));
        x[t * width + ch] += u[t * width + ch] + y;
    }
    for (uint r = 0; r < s_rows; r++) {
        uint src = n_tokens + r;
        state[r * width + ch] = src < s_rows ? state[src * width + ch] : n[(src - s_rows) * width + ch];
    }
}
"#;

/// A buffer and a byte offset.
pub type BufAt<'a> = (&'a metal::BufferRef, u64);

/// The PLE pipelines.
pub struct PleGpuKernels {
    pub gate: ComputePipelineState,
    pub conv_norm: ComputePipelineState,
    pub conv: ComputePipelineState,
}

fn set_u32(enc: &metal::ComputeCommandEncoderRef, i: u64, v: u32) {
    enc.set_bytes(i, 4, (&v as *const u32).cast());
}

fn set_f32(enc: &metal::ComputeCommandEncoderRef, i: u64, v: f32) {
    enc.set_bytes(i, 4, (&v as *const f32).cast());
}

impl PleGpuKernels {
    pub fn new(device: &Device) -> Result<Self, Q38GpuError> {
        let lib = device
            .new_library_with_source(PLE_GPU_SHADERS, &CompileOptions::new())
            .map_err(Q38GpuError::Compile)?;
        let p = |name: &'static str| -> Result<ComputePipelineState, Q38GpuError> {
            let f = lib.get_function(name, None).map_err(|msg| Q38GpuError::Pipeline { name, msg })?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|msg| Q38GpuError::Pipeline { name, msg })
        };
        Ok(Self { gate: p("q38p_ple_gate")?, conv_norm: p("q38p_ple_conv_norm")?, conv: p("q38p_ple_conv")? })
    }

    /// Encode [`PleOp::Gate`] on raw buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_gate(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        k: BufAt,
        x: BufAt,
        v: BufAt,
        u: BufAt,
        norm_key: BufAt,
        norm_query: BufAt,
        hidden: u32,
        hc: u32,
        n_tokens: u32,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.gate);
        for (i, (b, o)) in [k, x, v, u, norm_key, norm_query].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), o as NSUInteger);
        }
        set_u32(enc, 6, hidden);
        set_u32(enc, 7, hc);
        set_f32(enc, 8, eps);
        enc.dispatch_thread_groups(MTLSize::new((hc * n_tokens) as NSUInteger, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Encode [`PleOp::Conv`] on raw buffers (two dispatches: the norm,
    /// then the conv). Uses memory barriers between them.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_conv(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        u: BufAt,
        n: BufAt,
        state: BufAt,
        x: BufAt,
        norm_conv: BufAt,
        conv: BufAt,
        hidden: u32,
        hc: u32,
        kernel: u32,
        dilation: u32,
        n_tokens: u32,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.conv_norm);
        enc.set_buffer(0, Some(u.0), u.1 as NSUInteger);
        enc.set_buffer(1, Some(n.0), n.1 as NSUInteger);
        enc.set_buffer(2, Some(norm_conv.0), norm_conv.1 as NSUInteger);
        set_u32(enc, 3, hidden);
        set_u32(enc, 4, hc);
        set_f32(enc, 5, eps);
        enc.dispatch_thread_groups(MTLSize::new((hc * n_tokens) as NSUInteger, 1, 1), MTLSize::new(256, 1, 1));
        enc.memory_barrier_with_resources(&[n.0]);
        let width = hidden * hc;
        enc.set_compute_pipeline_state(&self.conv);
        for (i, (b, o)) in [u, n, state, x, conv].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), o as NSUInteger);
        }
        set_u32(enc, 5, width);
        set_u32(enc, 6, kernel);
        set_u32(enc, 7, dilation);
        set_u32(enc, 8, n_tokens);
        enc.dispatch_thread_groups(MTLSize::new(width.div_ceil(256) as NSUInteger, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Encode `op` (engine buffers from `pool`, gains and conv weights
    /// in `wbuf`).
    pub fn encode(&self, enc: &metal::ComputeCommandEncoderRef, op: &PleOp, pool: &MetalBufferPool, wbuf: &metal::BufferRef) {
        let h = |id: FBuf| -> BufAt { (pool.handle(id), 0) };
        match *op {
            PleOp::Gate { k, x, v, u, norm_key_off, norm_query_off, hidden, hc, n_tokens, eps } => self.encode_gate(
                enc,
                h(k),
                h(x),
                h(v),
                h(u),
                (wbuf, norm_key_off),
                (wbuf, norm_query_off),
                hidden,
                hc,
                n_tokens,
                eps,
            ),
            PleOp::Conv { u, n, state, x, norm_conv_off, conv_off, hidden, hc, kernel, dilation, n_tokens, eps } => {
                self.encode_conv(
                    enc,
                    h(u),
                    h(n),
                    h(state),
                    h(x),
                    (wbuf, norm_conv_off),
                    (wbuf, conv_off),
                    hidden,
                    hc,
                    kernel,
                    dilation,
                    n_tokens,
                    eps,
                )
            }
        }
    }
}

// --- CPU -----------------------------------------------------------------

fn f32s(c: &RefCell<Vec<u8>>) -> std::cell::Ref<'_, [f32]> {
    std::cell::Ref::map(c.borrow(), |v| bytemuck::cast_slice(v.as_slice()))
}

fn f32s_mut(c: &RefCell<Vec<u8>>) -> std::cell::RefMut<'_, [f32]> {
    std::cell::RefMut::map(c.borrow_mut(), |v| bytemuck::cast_slice_mut(v.as_mut_slice()))
}

fn bf16s(b: &CpuBackend, off: u64, n: usize) -> Vec<f32> {
    let bytes = b.weight_file().bytes_at(off, 2 * n).expect("q38 ple: weight offset out of the weight file");
    bytes.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
}

/// [`PleOp::Gate`] on host slices (`gk`, `gq`: the bf16 gains as f32).
#[allow(clippy::too_many_arguments)]
pub fn gate_host(k: &[f32], x: &[f32], v: &[f32], u: &mut [f32], gk: &[f32], gq: &[f32], hidden: usize, hc: usize, eps: f32) {
    let hh = hidden * hc;
    let t_n = v.len() / hidden;
    let inv = |s: &[f32]| 1.0 / (s.iter().map(|a| a * a).sum::<f32>() / hidden as f32 + eps).sqrt();
    for t in 0..t_n {
        for s in 0..hc {
            let base = t * hh + s * hidden;
            let (ks, xs) = (&k[base..base + hidden], &x[base..base + hidden]);
            let (ik, iq) = (inv(ks), inv(xs));
            let mut d = 0f32;
            for i in 0..hidden {
                d += ks[i] * ik * (1.0 + gk[s * hidden + i]) * (xs[i] * iq * (1.0 + gq[s * hidden + i]));
            }
            let g = d / (hidden as f32).sqrt();
            let sign = if g > 0.0 { 1.0 } else if g < 0.0 { -1.0 } else { 0.0 };
            let g = g.abs().max(1e-6).sqrt() * sign;
            let sg = 1.0 / (1.0 + (-g).exp());
            for i in 0..hidden {
                u[base + i] = sg * v[t * hidden + i];
            }
        }
    }
}

/// [`PleOp::Conv`] on host slices (`gc`, `w`: the bf16 values as f32;
/// `w` is `[hc·H, K]`, `k` fastest).
#[allow(clippy::too_many_arguments)]
pub fn conv_host(
    u: &[f32],
    n: &mut [f32],
    state: &mut [f32],
    x: &mut [f32],
    gc: &[f32],
    w: &[f32],
    hidden: usize,
    hc: usize,
    kernel: usize,
    dilation: usize,
    eps: f32,
) {
    let width = hidden * hc;
    let t_n = u.len() / width;
    for (g, (us, ns)) in u.chunks(hidden).zip(n.chunks_mut(hidden)).enumerate() {
        let inv = 1.0 / (us.iter().map(|a| a * a).sum::<f32>() / hidden as f32 + eps).sqrt();
        let goff = (g % hc) * hidden;
        for i in 0..hidden {
            ns[i] = us[i] * inv * (1.0 + gc[goff + i]);
        }
    }
    let s_rows = (kernel - 1) * dilation;
    for ch in 0..width {
        let hist = |src: usize, state: &[f32]| if src < s_rows { state[src * width + ch] } else { n[(src - s_rows) * width + ch] };
        for t in 0..t_n {
            let mut acc = 0f32;
            for j in 0..kernel {
                acc += w[ch * kernel + j] * hist(s_rows + t - (kernel - 1 - j) * dilation, state);
            }
            x[t * width + ch] += u[t * width + ch] + acc / (1.0 + (-acc).exp());
        }
        for r in 0..s_rows {
            let v = hist(t_n + r, state);
            state[r * width + ch] = v;
        }
    }
}

/// Run `op` on the CPU backend.
pub fn apply_cpu(b: &CpuBackend, op: &PleOp) {
    let pool = b.pool();
    match *op {
        PleOp::Gate { k, x, v, u, norm_key_off, norm_query_off, hidden, hc, n_tokens, eps } => {
            let (h, hc, t) = (hidden as usize, hc as usize, n_tokens as usize);
            let (gk, gq) = (bf16s(b, norm_key_off, h * hc), bf16s(b, norm_query_off, h * hc));
            let (kv, xv, vv) = (f32s(pool.handle(k)), f32s(pool.handle(x)), f32s(pool.handle(v)));
            let mut uv = f32s_mut(pool.handle(u));
            let hh = h * hc;
            gate_host(&kv[..t * hh], &xv[..t * hh], &vv[..t * h], &mut uv[..t * hh], &gk, &gq, h, hc, eps);
        }
        PleOp::Conv { u, n, state, x, norm_conv_off, conv_off, hidden, hc, kernel, dilation, n_tokens, eps } => {
            let (h, hc, t, kk, d) = (hidden as usize, hc as usize, n_tokens as usize, kernel as usize, dilation as usize);
            let hh = h * hc;
            let (gc, w) = (bf16s(b, norm_conv_off, hh), bf16s(b, conv_off, hh * kk));
            let uv = f32s(pool.handle(u));
            let mut nv = f32s_mut(pool.handle(n));
            let mut sv = f32s_mut(pool.handle(state));
            let mut xv = f32s_mut(pool.handle(x));
            let s_len = (kk - 1) * d * hh;
            conv_host(&uv[..t * hh], &mut nv[..t * hh], &mut sv[..s_len], &mut xv[..t * hh], &gc, &w, h, hc, kk, d, eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ple::{matvec_4bit_tokens, PleConfig};
    use super::*;
    use crate::riir::backend::gpu::metal::{MetalContext, MtlBuffer};

    fn unhex(v: &serde_json::Value) -> Vec<u8> {
        let s = v.as_str().unwrap();
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    fn f32s_json(v: &serde_json::Value) -> Vec<f32> {
        v.as_array().unwrap().iter().map(|x| f32::from_bits(x.as_u64().unwrap() as u32)).collect()
    }

    fn bf16_to_f32s(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
    }

    fn buf_f32(dev: &Device, v: &[f32]) -> MtlBuffer<f32> {
        let mut b = MtlBuffer::with_len(dev, v.len().max(1));
        b.as_mut_slice()[..v.len()].copy_from_slice(v);
        b
    }

    fn buf_u8(dev: &Device, v: &[u8]) -> MtlBuffer<u8> {
        let mut b = MtlBuffer::with_len(dev, v.len().max(1));
        b.as_mut_slice()[..v.len()].copy_from_slice(v);
        b
    }

    /// The Metal PLE ops ([`PleGpuKernels`]) and the CPU ones
    /// ([`gate_host`] / [`conv_host`]) against `ple_reference.py`'s
    /// fixture (`tests/fixtures/qwen38_ple/ple_tiny.json`): the n-gram
    /// embedding is rebuilt from the `MFNG` table on the host and
    /// projected on the host (`matvec_4bit_tokens`); the rest runs on
    /// each path, the conv state carried across the fixture's calls
    /// (prefill chunks and decode steps of two sequences). The output
    /// is the fixture's `hidden + out`.
    #[test]
    fn ple_ops_match_reference_fixture() {
        use super::super::ngram::{NgramConfig, NgramTable};
        use super::super::ple::PleLayer;
        let dir = format!("{}/tests/fixtures/qwen38_ple", env!("CARGO_MANIFEST_DIR"));
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/ple_tiny.json")).unwrap()).unwrap();
        let get = |k: &str| fx[k].as_u64().unwrap() as usize;
        let cfg = PleConfig {
            hidden: get("hidden"),
            hc: get("hc"),
            embed_dim: get("ple_embed_dim"),
            conv_kernel: 4,
            conv_dilation: 3,
            group_size: get("group"),
            eps: 1e-6,
        };
        let ncfg = NgramConfig {
            vocab_size: get("vocab") as u64,
            ngram_size: 3,
            heads_per_ngram: get("heads_per_ngram"),
            ngram_vocab_size_base: get("ngram_vocab_size_base") as u64,
            make_ngram_vocab_size_divisible_by: 128,
            seed: 1234,
            eos_token_id: get("eos") as i32,
            ple_layer_index: 0,
        };
        let table = NgramTable::open(std::path::Path::new(&format!("{dir}/ple_tiny.mfng")), 5).unwrap();
        let layer = PleLayer::new(cfg, ncfg, table).unwrap();
        let b = |k: &str| unhex(&fx[k]);
        let (h, hc, hh, ed) = (cfg.hidden, cfg.hc, cfg.hc_hidden(), cfg.embed_dim);
        let (kk, d) = (cfg.conv_kernel, cfg.conv_dilation);
        let s_rows = cfg.conv_state_rows();
        let (gk, gq, gc, cw) = (b("norm_key"), b("norm_query"), b("norm_conv"), b("conv"));
        let (gkf, gqf, gcf, cwf) = (bf16_to_f32s(&gk), bf16_to_f32s(&gq), bf16_to_f32s(&gc), bf16_to_f32s(&cw));
        let metal = MetalContext::new().ok();
        let kern = metal.as_ref().map(|m| PleGpuKernels::new(m.device()).unwrap());
        let (mut worst_cpu, mut worst_gpu, mut n_tok) = (0f32, 0f32, 0usize);
        for seq in fx["seqs"].as_array().unwrap() {
            let mut st = layer.new_state();
            let mut cpu_state = vec![0f32; s_rows * hh];
            let mut gpu_state = metal.as_ref().map(|m| buf_f32(m.device(), &cpu_state));
            for step in seq.as_array().unwrap() {
                let toks: Vec<i32> = step["tokens"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
                let t = toks.len();
                let hidden = f32s_json(&step["hidden"]);
                let want: Vec<f32> = hidden.iter().zip(f32s_json(&step["out"])).map(|(a, o)| a + o).collect();
                let e = layer.embedding(&mut st, &toks, None).unwrap();
                let (mut k, mut v) = (vec![0f32; t * hh], vec![0f32; t * h]);
                matvec_4bit_tokens(&b("key_w"), &b("key_s"), &b("key_b"), cfg.group_size, ed, hh, &e, &mut k).unwrap();
                matvec_4bit_tokens(&b("value_w"), &b("value_s"), &b("value_b"), cfg.group_size, ed, h, &e, &mut v).unwrap();
                // CPU path.
                let mut u = vec![0f32; t * hh];
                gate_host(&k, &hidden, &v, &mut u, &gkf, &gqf, h, hc, cfg.eps);
                let mut n = vec![0f32; t * hh];
                let mut x = hidden.clone();
                conv_host(&u, &mut n, &mut cpu_state, &mut x, &gcf, &cwf, h, hc, kk, d, cfg.eps);
                for (a, w) in x.iter().zip(&want) {
                    worst_cpu = worst_cpu.max((a - w).abs() / (1.0 + w.abs()));
                }
                // Metal path.
                if let (Some(m), Some(kern), Some(gs)) = (metal.as_ref(), kern.as_ref(), gpu_state.as_ref()) {
                    let dev = m.device();
                    let (kb, vb) = (buf_f32(dev, &k), buf_f32(dev, &v));
                    let xb = buf_f32(dev, &hidden);
                    let (ub, nb) = (buf_f32(dev, &vec![0.0; t * hh]), buf_f32(dev, &vec![0.0; t * hh]));
                    let (gkb, gqb, gcb, cwb) = (buf_u8(dev, &gk), buf_u8(dev, &gq), buf_u8(dev, &gc), buf_u8(dev, &cw));
                    let cb = m.queue_clone().new_command_buffer().to_owned();
                    let enc = cb.new_compute_command_encoder();
                    let (hu, cu, tu) = (h as u32, hc as u32, t as u32);
                    kern.encode_gate(
                        enc,
                        (kb.raw(), 0),
                        (xb.raw(), 0),
                        (vb.raw(), 0),
                        (ub.raw(), 0),
                        (gkb.raw(), 0),
                        (gqb.raw(), 0),
                        hu,
                        cu,
                        tu,
                        cfg.eps,
                    );
                    enc.memory_barrier_with_resources(&[ub.raw()]);
                    kern.encode_conv(
                        enc,
                        (ub.raw(), 0),
                        (nb.raw(), 0),
                        (gs.raw(), 0),
                        (xb.raw(), 0),
                        (gcb.raw(), 0),
                        (cwb.raw(), 0),
                        hu,
                        cu,
                        kk as u32,
                        d as u32,
                        tu,
                        cfg.eps,
                    );
                    enc.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    for (a, w) in xb.as_slice()[..t * hh].iter().zip(&want) {
                        worst_gpu = worst_gpu.max((a - w).abs() / (1.0 + w.abs()));
                    }
                    let ds = gs.as_slice()[..s_rows * hh].iter().zip(&cpu_state).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                    assert!(ds < 1e-4, "conv state GPU vs CPU {ds}");
                }
                n_tok += t;
            }
        }
        eprintln!("[ple ops] {n_tok} tokens vs reference: cpu {worst_cpu:.2e}, metal {worst_gpu:.2e}");
        assert!(n_tok >= 25);
        assert!(worst_cpu < 2e-5, "cpu {worst_cpu}");
        assert!(worst_gpu < 2e-5, "metal {worst_gpu}");
    }
}
