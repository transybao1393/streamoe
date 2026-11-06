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
//!
//! **Norm gains.** Every `1 + w` gain is applied in the kernels, from
//! the checkpoint's zero-centred bf16 values in the weight file. The
//! converter and the loader leave them as stored.
//!
//! A pass is a list of [`Step`]s encoded into one command buffer
//! ([`run_steps`]).

use std::cell::RefCell;

use metal::{CompileOptions, ComputePipelineState, MTLSize, NSUInteger};

use super::gated_residual_gpu::{Q38GpuError, Q38GpuKernels};
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
    HcNorm { x: FBuf, gain_off: u64, out: FBuf, group: u32, n_groups: u32, eps: f32 },
    /// `v[i] = silu(v[i] · scale)`, `n` values.
    SiluScale { v: FBuf, scale: f32, n: u32 },
    /// `out[j] = mean_c sigmoid(mix[c·H + j]) · normed[c·H + j]`.
    HcMix { mix: FBuf, normed: FBuf, out: FBuf, hidden: u32, hc: u32 },
    /// `inj[c] = 2 · sigmoid(inj[c] / hc)`.
    InjWeights { inj: FBuf, hc: u32 },
    /// `hyper[c·H + j] += inj[c] · block[j]`.
    HcInject { hyper: FBuf, block: FBuf, inj: FBuf, hidden: u32, hc: u32 },
    /// `hyper[c·H + j] = h[j]`.
    HcExpand { h: FBuf, hyper: FBuf, hidden: u32, hc: u32 },
    /// In place: each `dim` slice of `x` (`n_groups` of them) RMS-
    /// normalized and scaled by `1 + gain[i]` (`gain` `[dim]`, shared).
    HeadNormOnePlus { x: FBuf, gain_off: u64, dim: u32, n_groups: u32, eps: f32 },
    /// `out = v̂ · w · sigmoid(z)` per `dim` slice (`n_groups` of them;
    /// plain gain `w` `[dim]`, shared).
    GatedRmsNormSigmoid { values: FBuf, z: FBuf, weight_off: u64, out: FBuf, dim: u32, n_groups: u32, eps: f32 },
    /// `dst[i] += src[i]`, `n` values.
    Add { dst: FBuf, src: FBuf, n: u32 },
}

/// One entry of a pass.
#[derive(Debug)]
pub enum Step {
    G(Op),
    F(FamOp),
}

/// A backend that also runs [`FamOp`]s.
pub trait Q38Exec: Backend<Error = GraphError> {
    /// Compiled family kernels (Metal) or nothing (CPU).
    type Fam;
    fn fam_new(&self) -> Result<Self::Fam, Q38GpuError>;
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
"#;

/// The family's Metal pipelines.
pub struct Q38MetalFam {
    pub gr: Q38GpuKernels,
    pub head_norm: ComputePipelineState,
    pub gated_norm_sigmoid: ComputePipelineState,
    pub add: ComputePipelineState,
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

impl Q38Exec for MetalBackend {
    type Fam = Q38MetalFam;

    fn fam_new(&self) -> Result<Q38MetalFam, Q38GpuError> {
        let device = self.metal().device();
        let gr = Q38GpuKernels::new(device)?;
        let lib = device
            .new_library_with_source(Q38_ENGINE_SHADERS, &CompileOptions::new())
            .map_err(Q38GpuError::Compile)?;
        let p = |name: &'static str| -> Result<ComputePipelineState, Q38GpuError> {
            let f = lib.get_function(name, None).map_err(|msg| Q38GpuError::Pipeline { name, msg })?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|msg| Q38GpuError::Pipeline { name, msg })
        };
        Ok(Q38MetalFam {
            gr,
            head_norm: p("q38e_head_norm_one_plus")?,
            gated_norm_sigmoid: p("q38e_gated_rms_norm_sigmoid")?,
            add: p("q38e_add")?,
        })
    }

    fn encode_fam(&self, fam: &Q38MetalFam, op: &FamOp, ctx: &mut Self::EncodeCtx) {
        let cmd = ctx.cmdbuf();
        let pool = self.pool();
        let wbuf = self.weight_buf().buffer();
        let enc = cmd.new_compute_command_encoder();
        match *op {
            FamOp::HcNorm { x, gain_off, out, group, n_groups, eps } => {
                enc.set_compute_pipeline_state(&fam.gr.hc_norm);
                enc.set_buffer(0, Some(pool.handle(x)), 0);
                enc.set_buffer(1, Some(wbuf), gain_off as NSUInteger);
                enc.set_buffer(2, Some(pool.handle(out)), 0);
                set_u32(enc, 3, group);
                set_f32(enc, 4, eps);
                // gain_len: one token's `hc · H` gains.
                set_u32(enc, 5, group * n_groups);
                enc.dispatch_thread_groups(MTLSize::new(n_groups as NSUInteger, 1, 1), MTLSize::new(256, 1, 1));
            }
            FamOp::SiluScale { v, scale, n } => {
                enc.set_compute_pipeline_state(&fam.gr.silu_scale);
                enc.set_buffer(0, Some(pool.handle(v)), 0);
                set_f32(enc, 1, scale);
                set_u32(enc, 2, n);
                let (g, t) = grid1(n, 64);
                enc.dispatch_thread_groups(g, t);
            }
            FamOp::HcMix { mix, normed, out, hidden, hc } => {
                enc.set_compute_pipeline_state(&fam.gr.hc_mix);
                enc.set_buffer(0, Some(pool.handle(mix)), 0);
                enc.set_buffer(1, Some(pool.handle(normed)), 0);
                enc.set_buffer(2, Some(pool.handle(out)), 0);
                set_u32(enc, 3, hidden);
                set_u32(enc, 4, hc);
                set_u32(enc, 5, hidden); // total = T · H, T = 1
                let (g, t) = grid1(hidden, 256);
                enc.dispatch_thread_groups(g, t);
            }
            FamOp::InjWeights { inj, hc } => {
                enc.set_compute_pipeline_state(&fam.gr.inj_weights);
                enc.set_buffer(0, Some(pool.handle(inj)), 0);
                set_u32(enc, 1, hc);
                set_u32(enc, 2, hc); // n = T · hc, T = 1
                let (g, t) = grid1(hc, 32);
                enc.dispatch_thread_groups(g, t);
            }
            FamOp::HcInject { hyper, block, inj, hidden, hc } => {
                let total = hidden * hc;
                enc.set_compute_pipeline_state(&fam.gr.hc_inject);
                enc.set_buffer(0, Some(pool.handle(hyper)), 0);
                enc.set_buffer(1, Some(pool.handle(block)), 0);
                enc.set_buffer(2, Some(pool.handle(inj)), 0);
                set_u32(enc, 3, hidden);
                set_u32(enc, 4, total);
                set_u32(enc, 5, hc);
                let (g, t) = grid1(total, 256);
                enc.dispatch_thread_groups(g, t);
            }
            FamOp::HcExpand { h, hyper, hidden, hc } => {
                let total = hidden * hc;
                enc.set_compute_pipeline_state(&fam.gr.hc_expand);
                enc.set_buffer(0, Some(pool.handle(h)), 0);
                enc.set_buffer(1, Some(pool.handle(hyper)), 0);
                set_u32(enc, 2, hidden);
                set_u32(enc, 3, total);
                set_u32(enc, 4, hc);
                let (g, t) = grid1(total, 256);
                enc.dispatch_thread_groups(g, t);
            }
            FamOp::HeadNormOnePlus { x, gain_off, dim, n_groups, eps } => {
                enc.set_compute_pipeline_state(&fam.head_norm);
                enc.set_buffer(0, Some(pool.handle(x)), 0);
                enc.set_buffer(1, Some(wbuf), gain_off as NSUInteger);
                set_u32(enc, 2, dim);
                set_f32(enc, 3, eps);
                enc.dispatch_thread_groups(MTLSize::new(n_groups as NSUInteger, 1, 1), MTLSize::new(128, 1, 1));
            }
            FamOp::GatedRmsNormSigmoid { values, z, weight_off, out, dim, n_groups, eps } => {
                enc.set_compute_pipeline_state(&fam.gated_norm_sigmoid);
                enc.set_buffer(0, Some(pool.handle(values)), 0);
                enc.set_buffer(1, Some(pool.handle(z)), 0);
                enc.set_buffer(2, Some(wbuf), weight_off as NSUInteger);
                enc.set_buffer(3, Some(pool.handle(out)), 0);
                set_u32(enc, 4, dim);
                set_f32(enc, 5, eps);
                enc.dispatch_thread_groups(MTLSize::new(n_groups as NSUInteger, 1, 1), MTLSize::new(128, 1, 1));
            }
            FamOp::Add { dst, src, n } => {
                enc.set_compute_pipeline_state(&fam.add);
                enc.set_buffer(0, Some(pool.handle(dst)), 0);
                enc.set_buffer(1, Some(pool.handle(src)), 0);
                set_u32(enc, 2, n);
                let (g, t) = grid1(n, 256);
                enc.dispatch_thread_groups(g, t);
            }
        }
        enc.end_encoding();
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

impl Q38Exec for CpuBackend {
    type Fam = ();

    fn fam_new(&self) -> Result<(), Q38GpuError> {
        Ok(())
    }

    fn encode_fam(&self, _: &(), op: &FamOp, _ctx: &mut ()) {
        let pool = self.pool();
        match *op {
            FamOp::HcNorm { x, gain_off, out, group, n_groups, eps } => {
                let (group, n) = (group as usize, (group * n_groups) as usize);
                let gain = bf16_gain(self, gain_off, n);
                let xv = f32s(pool.handle(x));
                let mut ov = f32s_mut(pool.handle(out));
                rms_groups(&xv[..n], &mut ov[..n], group, eps, |g, i| 1.0 + gain[g * group + i]);
            }
            FamOp::SiluScale { v, scale, n } => {
                let mut vv = f32s_mut(pool.handle(v));
                for x in vv[..n as usize].iter_mut() {
                    let t = *x * scale;
                    *x = t / (1.0 + (-t).exp());
                }
            }
            FamOp::HcMix { mix, normed, out, hidden, hc } => {
                let (h, hc) = (hidden as usize, hc as usize);
                let (m, nv) = (f32s(pool.handle(mix)), f32s(pool.handle(normed)));
                let mut ov = f32s_mut(pool.handle(out));
                for j in 0..h {
                    let mut acc = 0.0f32;
                    for c in 0..hc {
                        acc += sigmoid(m[c * h + j]) * nv[c * h + j];
                    }
                    ov[j] = acc / hc as f32;
                }
            }
            FamOp::InjWeights { inj, hc } => {
                let mut iv = f32s_mut(pool.handle(inj));
                for v in iv[..hc as usize].iter_mut() {
                    *v = 2.0 * sigmoid(*v / hc as f32);
                }
            }
            FamOp::HcInject { hyper, block, inj, hidden, hc } => {
                let (h, hc) = (hidden as usize, hc as usize);
                let (bv, iv) = (f32s(pool.handle(block)), f32s(pool.handle(inj)));
                let mut hv = f32s_mut(pool.handle(hyper));
                for c in 0..hc {
                    for j in 0..h {
                        hv[c * h + j] = bv[j].mul_add(iv[c], hv[c * h + j]);
                    }
                }
            }
            FamOp::HcExpand { h, hyper, hidden, hc } => {
                let hd = hidden as usize;
                let src = f32s(pool.handle(h))[..hd].to_vec();
                let mut hv = f32s_mut(pool.handle(hyper));
                for c in 0..hc as usize {
                    hv[c * hd..(c + 1) * hd].copy_from_slice(&src);
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
        }
    }
}
