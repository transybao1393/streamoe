//! MLX affine group size 32 (Qwen3.8-Flash-Next) on the engine's own
//! Metal kernels, against the CPU dequant reference.
//!
//! Every quantized kernel in `shaders.metal` takes `group_size` as a
//! runtime argument; Qwen3.8-Flash-Next's MLX build uses 32 for its
//! 4-bit weights and 64 for the 8-bit router gate / shared_expert_gate.
//! Each test runs the kernel at group 64 and group 32 on synthetic
//! weights at that model's shapes (hidden 2560, expert intermediate
//! 640, attention q 24 x 256 + gate / kv 2 x 256, linear-attention
//! 48 V x 128 / 16 QK x 128 heads, vocab 248320) and compares with
//! `dequant_matvec_{4,8}bit_gs_cpu`. Sums run in another order on the
//! GPU, so the contract is a relative error, not bit equality. No model
//! files; the tests run under any variant feature.
//!
//! ```bash
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b --test group32_kernels
//! # group-32 vs group-64 speed at the 2560 x 640 expert shape:
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b --release \
//!     --test group32_kernels -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::time::Instant;

use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize, NSUInteger};
use streammoe::riir::MetalContext;
use streammoe::riir::backend::cpu::cpu_matvec::{dequant_matvec_4bit_gs_cpu, dequant_matvec_8bit_gs_cpu};
use streammoe::riir::backend::gpu::gpu_matvec::{MatvecPipelines, encode_matvec_n_tokens};
use streammoe::riir::variants::group_size_for_bits;

/// Max |gpu - cpu| over the rows, relative to the largest |cpu| value
/// (or, for a row that cancels to near zero, the typical magnitude of a
/// sum of its terms — see [`Mat::floor`]).
const REL_TOL: f64 = 2e-5;

/// xorshift64*, deterministic.
struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f32(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(((b >> 16) & 1) + 0x7fff) >> 16) as u16
}

fn shared_buf(metal: &MetalContext, bytes: usize) -> Buffer {
    metal.device().new_buffer(bytes.max(16) as NSUInteger, MTLResourceOptions::StorageModeShared)
}

fn slice_mut<T>(buf: &Buffer, off: usize, n: usize) -> &mut [T] {
    unsafe { std::slice::from_raw_parts_mut((buf.contents() as *mut u8).add(off) as *mut T, n) }
}

fn slice<T>(buf: &Buffer, off: usize, n: usize) -> &[T] {
    unsafe { std::slice::from_raw_parts((buf.contents() as *const u8).add(off) as *const T, n) }
}

/// One `[out_dim, in_dim]` affine matrix (`bits` 4 or 8, group `gs`) in
/// a shared buffer laid out like the weight file: packed u32 words,
/// then bf16 scales, then bf16 biases, each 16-byte aligned. Filled in
/// place (the LM head is ~400 MB; no host copy).
struct Mat {
    out_dim: usize,
    in_dim: usize,
    bits: u32,
    gs: usize,
    buf: Buffer,
    s_off: usize,
    b_off: usize,
}

impl Mat {
    fn new(metal: &MetalContext, rng: &mut Rng, out_dim: usize, in_dim: usize, bits: u32, gs: usize) -> Self {
        let words = out_dim * in_dim * bits as usize / 32;
        let groups = out_dim * (in_dim / gs);
        let s_off = (words * 4).next_multiple_of(16);
        let b_off = (s_off + groups * 2).next_multiple_of(16);
        let buf = shared_buf(metal, b_off + groups * 2);
        for w in slice_mut::<u32>(&buf, 0, words) {
            *w = rng.u64() as u32;
        }
        for s in slice_mut::<u16>(&buf, s_off, groups) {
            *s = bf16(rng.f32() * 0.05);
        }
        for b in slice_mut::<u16>(&buf, b_off, groups) {
            *b = bf16(rng.f32() * 0.02);
        }
        Mat { out_dim, in_dim, bits, gs, buf, s_off, b_off }
    }

    fn packed(&self) -> &[u32] {
        slice(&self.buf, 0, self.out_dim * self.in_dim * self.bits as usize / 32)
    }
    fn scales(&self) -> &[u16] {
        slice(&self.buf, self.s_off, self.out_dim * (self.in_dim / self.gs))
    }
    fn biases(&self) -> &[u16] {
        slice(&self.buf, self.b_off, self.out_dim * (self.in_dim / self.gs))
    }

    /// Typical magnitude of row 0's dot product, `Σ|w·x| / sqrt(in_dim)`:
    /// the error scale when an output cancels to near zero (a 1-row
    /// matrix such as shared_expert_gate).
    fn floor(&self, x: &[f32]) -> f64 {
        let per = 32 / self.bits as usize;
        let mask = (1u32 << self.bits) - 1;
        let (p, s, b) = (self.packed(), self.scales(), self.biases());
        let mut l1 = 0f64;
        for c in 0..self.in_dim {
            let q = (p[c / per] >> (self.bits as usize * (c % per))) & mask;
            let sc = f32::from_bits((s[c / self.gs] as u32) << 16) as f64;
            let bi = f32::from_bits((b[c / self.gs] as u32) << 16) as f64;
            l1 += ((q as f64 * sc + bi) * x[c] as f64).abs();
        }
        l1 / (self.in_dim as f64).sqrt()
    }

    /// CPU reference for one input row.
    fn reference(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.out_dim];
        let (p, s, b) = (self.packed(), self.scales(), self.biases());
        if self.bits == 4 {
            dequant_matvec_4bit_gs_cpu(p, s, b, self.in_dim, self.out_dim, self.gs, x, &mut out).unwrap();
        } else {
            dequant_matvec_8bit_gs_cpu(p, s, b, self.in_dim, self.out_dim, self.gs, x, &mut out).unwrap();
        }
        out
    }
}

fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    rel_err_floor(got, want, 1e-6)
}

fn rel_err_floor(got: &[f32], want: &[f32], floor: f64) -> f64 {
    let scale = want.iter().fold(floor, |m, &v| m.max(v.abs() as f64));
    let diff = got.iter().zip(want).fold(0f64, |m, (&g, &w)| m.max((g as f64 - w as f64).abs()));
    assert!(got.iter().all(|v| v.is_finite()), "non-finite GPU output");
    diff / scale
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// 256 threads, 8 rows per threadgroup (v3 / 8-bit v3 layout).
    V3,
    /// 64 threads, one row per threadgroup.
    Fast,
    /// 64 threads, 8 rows per threadgroup.
    Qmv,
    /// `*_n_tokens` v3 layout: one threadgroup per (row tile, token).
    V3N,
    /// `*_n_tokens` fast layout: one threadgroup per (row, token).
    FastN,
}

#[allow(clippy::too_many_arguments)]
fn encode(cmd: &metal::CommandBufferRef, pso: &ComputePipelineState, kind: Kind, m: &Mat, x: &Buffer, out: &Buffer, n_tokens: u32) {
    let (o, i, g) = (m.out_dim as u32, m.in_dim as u32, m.gs as u32);
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(&m.buf), 0);
    enc.set_buffer(1, Some(&m.buf), m.s_off as u64);
    enc.set_buffer(2, Some(&m.buf), m.b_off as u64);
    enc.set_buffer(3, Some(x), 0);
    enc.set_buffer(4, Some(out), 0);
    enc.set_bytes(5, 4, (&o as *const u32).cast());
    enc.set_bytes(6, 4, (&i as *const u32).cast());
    enc.set_bytes(7, 4, (&g as *const u32).cast());
    let tiles = o.div_ceil(8);
    enc.set_bytes(8, 4, (&n_tokens as *const u32).cast());
    enc.set_bytes(9, 4, (&tiles as *const u32).cast());
    let (tgs, threads) = match kind {
        Kind::V3 => (tiles, 256),
        Kind::Fast => (o, 64),
        Kind::Qmv => (tiles, 64),
        Kind::V3N => (tiles * n_tokens, 256),
        Kind::FastN => (o * n_tokens, 64),
    };
    enc.dispatch_thread_groups(MTLSize::new(tgs as u64, 1, 1), MTLSize::new(threads, 1, 1));
    enc.end_encoding();
}

fn run(metal: &MetalContext, f: impl FnOnce(&metal::CommandBufferRef)) {
    let cmd = metal.queue().new_command_buffer();
    f(cmd);
    cmd.commit();
    cmd.wait_until_completed();
}

/// `(out_dim, in_dim, name)` — Qwen3.8-Flash-Next 4-bit projections.
const SHAPES_4BIT: &[(usize, usize, &str)] = &[
    (24 * 256 * 2, 2560, "attn q + gate (24 x 256 x 2)"),
    (2 * 256, 2560, "attn k / v (2 x 256)"),
    (16 * 128 * 2 + 48 * 128, 2560, "linear qkv (16 QK + 48 V x 128)"),
    (48 * 128, 2560, "linear z (48 x 128)"),
    (48, 2560, "linear beta / alpha (48)"),
    (2560, 24 * 256, "attn o_proj (6144)"),
    (2560, 48 * 128, "linear out_proj (6144)"),
    (640, 2560, "expert / shared gate|up"),
    (2560, 640, "expert / shared down"),
];

/// Single-token and N-token 4-bit matvec kernels (decode, and the
/// per-token prefill fallback) at group 64 and 32 on every projection
/// shape.
#[test]
fn matvec_4bit_group32_matches_cpu() {
    let mut metal = MetalContext::new().expect("Metal");
    let kernels = [
        ("dequant_matvec_4bit_v3", Kind::V3),
        ("dequant_matvec_4bit_fast", Kind::Fast),
        ("dequant_matvec_4bit_qmv", Kind::Qmv),
        ("dequant_matvec_4bit_v3_n_tokens", Kind::V3N),
        ("dequant_matvec_4bit_fast_n_tokens", Kind::FastN),
    ];
    let psos: Vec<ComputePipelineState> =
        kernels.iter().map(|(n, _)| metal.pipeline(n).expect(n).to_owned()).collect();
    let mut rng = Rng(0x0038_4b17);
    const N: usize = 3;
    for &gs in &[64usize, 32] {
        for &(out_dim, in_dim, name) in SHAPES_4BIT {
            let m = Mat::new(&metal, &mut rng, out_dim, in_dim, 4, gs);
            let xv: Vec<f32> = (0..N * in_dim).map(|_| rng.f32()).collect();
            let x = shared_buf(&metal, xv.len() * 4);
            slice_mut::<f32>(&x, 0, xv.len()).copy_from_slice(&xv);
            let want: Vec<Vec<f32>> = (0..N).map(|t| m.reference(&xv[t * in_dim..][..in_dim])).collect();
            let out = shared_buf(&metal, N * out_dim * 4);
            for (pso, &(kname, kind)) in psos.iter().zip(&kernels) {
                // v3 stages x in 4096 floats of threadgroup memory.
                if in_dim > 4096 && matches!(kind, Kind::V3 | Kind::V3N) {
                    continue;
                }
                let n = if matches!(kind, Kind::V3N | Kind::FastN) { N } else { 1 };
                run(&metal, |cmd| encode(cmd, pso, kind, &m, &x, &out, n as u32));
                for t in 0..n {
                    let got = slice::<f32>(&out, t * out_dim * 4, out_dim);
                    let err = rel_err(got, &want[t]);
                    assert!(err < REL_TOL, "{kname} gs{gs} {name} token {t}: rel err {err:.3e}");
                }
            }
            eprintln!("gs{gs} {name} ({out_dim} x {in_dim}): v3 / fast / qmv / v3_n / fast_n match CPU");
        }
    }
}

/// The LM head (vocab 248320 x 2560) at group 32 on the decode kernels
/// the engine picks for it (qmv; fast is the fallback).
#[test]
fn matvec_4bit_group32_lm_head_matches_cpu() {
    let mut metal = MetalContext::new().expect("Metal");
    let mut rng = Rng(0x0038_1a4d);
    let m = Mat::new(&metal, &mut rng, 248_320, 2560, 4, 32);
    let xv: Vec<f32> = (0..2560).map(|_| rng.f32()).collect();
    let x = shared_buf(&metal, 2560 * 4);
    slice_mut::<f32>(&x, 0, 2560).copy_from_slice(&xv);
    let want = m.reference(&xv);
    let out = shared_buf(&metal, 248_320 * 4);
    for (kname, kind) in [("dequant_matvec_4bit_qmv", Kind::Qmv), ("dequant_matvec_4bit_fast", Kind::Fast)] {
        let pso = metal.pipeline(kname).expect(kname).to_owned();
        run(&metal, |cmd| encode(cmd, &pso, kind, &m, &x, &out, 1));
        let err = rel_err(slice::<f32>(&out, 0, 248_320), &want);
        assert!(err < REL_TOL, "{kname} gs32 lm_head: rel err {err:.3e}");
        eprintln!("gs32 lm_head {kname}: rel err {err:.2e}");
    }
}

/// 8-bit kernels: the router gate (512 x 2560) and shared_expert_gate
/// (1 x 2560) at group 64 — Qwen3.8-Flash-Next's 8-bit tensors — and 32.
#[test]
fn matvec_8bit_router_gate_matches_cpu() {
    let mut metal = MetalContext::new().expect("Metal");
    let v3 = metal.pipeline("dequant_matvec_8bit_v3").expect("8bit v3").to_owned();
    let v3n = metal.pipeline("dequant_matvec_8bit_v3_n_tokens").expect("8bit v3 n").to_owned();
    let mut rng = Rng(0x0038_0008);
    const N: usize = 3;
    for &gs in &[64usize, 32] {
        for (out_dim, name) in [(512usize, "router gate"), (1, "shared_expert_gate")] {
            let m = Mat::new(&metal, &mut rng, out_dim, 2560, 8, gs);
            let xv: Vec<f32> = (0..N * 2560).map(|_| rng.f32()).collect();
            let x = shared_buf(&metal, xv.len() * 4);
            slice_mut::<f32>(&x, 0, xv.len()).copy_from_slice(&xv);
            let out = shared_buf(&metal, N * out_dim * 4);
            run(&metal, |cmd| encode(cmd, &v3, Kind::V3, &m, &x, &out, 1));
            let err = rel_err_floor(slice::<f32>(&out, 0, out_dim), &m.reference(&xv[..2560]), m.floor(&xv[..2560]));
            assert!(err < REL_TOL, "8-bit v3 gs{gs} {name}: rel err {err:.3e}");
            run(&metal, |cmd| encode(cmd, &v3n, Kind::V3N, &m, &x, &out, N as u32));
            for t in 0..N {
                let xt = &xv[t * 2560..][..2560];
                let err = rel_err_floor(slice::<f32>(&out, t * out_dim * 4, out_dim), &m.reference(xt), m.floor(xt));
                assert!(err < REL_TOL, "8-bit v3_n gs{gs} {name} token {t}: rel err {err:.3e}");
            }
        }
    }
}

/// The engine's own encoder (`encode_matvec_n_tokens`) picks the group
/// from the weight's bits ([`group_size_for_bits`]): 4-bit at the
/// variant's `GROUP_SIZE`, 8-bit at `GROUP_SIZE_8BIT`. Under
/// `model-qwen3-8-flash-next` that is 32 / 64; elsewhere 64 / 64.
#[test]
fn engine_matvec_encoder_uses_group_for_bits() {
    let mut metal = MetalContext::new().expect("Metal");
    let pipes = MatvecPipelines::fetch(&mut metal).expect("matvec pipelines");
    let mut rng = Rng(0x0038_e9c0);
    for (bits, out_dim, in_dim) in [(4u32, 640usize, 2560usize), (4, 2560, 6144), (8, 512, 2560)] {
        let gs = group_size_for_bits(bits);
        let m = Mat::new(&metal, &mut rng, out_dim, in_dim, bits, gs);
        for n in [1usize, 3] {
            let xv: Vec<f32> = (0..n * in_dim).map(|_| rng.f32()).collect();
            let x = shared_buf(&metal, xv.len() * 4);
            slice_mut::<f32>(&x, 0, xv.len()).copy_from_slice(&xv);
            let out = shared_buf(&metal, n * out_dim * 4);
            run(&metal, |cmd| {
                encode_matvec_n_tokens(
                    cmd, &pipes, &m.buf, 0, m.s_off as u64, m.b_off as u64, &x, 0, &out, 0,
                    in_dim as u32, out_dim as u32, n as u32, bits,
                )
            });
            for t in 0..n {
                let xt = &xv[t * in_dim..][..in_dim];
                let err = rel_err_floor(slice::<f32>(&out, t * out_dim * 4, out_dim), &m.reference(xt), m.floor(xt));
                assert!(err < REL_TOL, "encode_matvec_n_tokens bits {bits} gs{gs} n={n} token {t}: {err:.3e}");
            }
        }
    }
}

// --- decode expert kernels ---------------------------------------------------

const HIDDEN: usize = 2560;
const INTER: usize = 640;

/// Expert block layout at group `gs`: `[gate | up | down]`, each
/// `[weights | scales | biases]` (the engine's `ExpertLayout`).
struct Experts {
    blobs: Vec<Buffer>,
    /// Packed weight bytes of one block (4-bit or packed 3-bit).
    w_bytes: usize,
    s_bytes: usize,
    gs: usize,
}

impl Experts {
    fn block(&self) -> usize {
        self.w_bytes + 2 * self.s_bytes
    }
    /// (w, s, b) offsets of block `i` (0 gate, 2 down).
    fn offs(&self, i: usize) -> (usize, usize, usize) {
        let w = i * self.block();
        (w, w + self.w_bytes, w + self.w_bytes + self.s_bytes)
    }
}

/// `n` experts at group `gs`. With `codes3`, every code is 0..7 and the
/// experts come in two layouts: 4-bit nibbles and packed 3-bit.
fn make_experts(metal: &MetalContext, rng: &mut Rng, n: usize, gs: usize, codes3: bool) -> (Experts, Option<Experts>) {
    let w4 = INTER * HIDDEN / 2;
    let w3 = INTER * HIDDEN * 3 / 8;
    let s = INTER * (HIDDEN / gs) * 2;
    let mut e4 = Experts { blobs: vec![], w_bytes: w4, s_bytes: s, gs };
    let mut e3 = Experts { blobs: vec![], w_bytes: w3, s_bytes: s, gs };
    for _ in 0..n {
        let b4 = shared_buf(metal, 3 * e4.block());
        let b3 = shared_buf(metal, 3 * e3.block());
        for blk in 0..3 {
            let (w, so, bo) = e4.offs(blk);
            let (w3o, s3o, b3o) = e3.offs(blk);
            let words = slice_mut::<u32>(&b4, w, w4 / 4);
            for word in words.iter_mut() {
                *word = rng.u64() as u32 & if codes3 { 0x7777_7777 } else { u32::MAX };
            }
            let scales: Vec<u16> = (0..s / 2).map(|_| bf16(rng.f32() * 0.05)).collect();
            let biases: Vec<u16> = (0..s / 2).map(|_| bf16(rng.f32() * 0.02)).collect();
            slice_mut::<u16>(&b4, so, s / 2).copy_from_slice(&scales);
            slice_mut::<u16>(&b4, bo, s / 2).copy_from_slice(&biases);
            if codes3 {
                let packed = slice_mut::<u8>(&b3, w3o, w3);
                for (i, &word) in slice::<u32>(&b4, w, w4 / 4).iter().enumerate() {
                    let mut p = 0u32;
                    for k in 0..8 {
                        p |= ((word >> (4 * k)) & 0x7) << (3 * k);
                    }
                    packed[3 * i..3 * i + 3].copy_from_slice(&p.to_le_bytes()[..3]);
                }
                slice_mut::<u16>(&b3, s3o, s / 2).copy_from_slice(&scales);
                slice_mut::<u16>(&b3, b3o, s / 2).copy_from_slice(&biases);
            }
        }
        e4.blobs.push(b4);
        e3.blobs.push(b3);
    }
    (e4, codes3.then_some(e3))
}

/// One `dequant_matvec_{4,3}bit_v3_experts` dispatch over all experts
/// (at most 8, like the engine's chunks) for block `blk`.
#[allow(clippy::too_many_arguments)]
fn encode_experts(
    cmd: &metal::CommandBufferRef,
    pso: &ComputePipelineState,
    e: &Experts,
    blk: usize,
    x: &Buffer,
    out: &Buffer,
    out_dim: u32,
    in_dim: u32,
    input_stride: u32,
) {
    let k = e.blobs.len() as u32;
    let (w, s, b) = e.offs(blk);
    let (w, s, b, gs, tiles, base) = (w as u32, s as u32, b as u32, e.gs as u32, out_dim.div_ceil(8), 0u32);
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    for i in 0..8 {
        enc.set_buffer(i, Some(&e.blobs[(i as usize).min(e.blobs.len() - 1)]), 0);
    }
    enc.set_buffer(8, Some(x), 0);
    enc.set_buffer(9, Some(out), 0);
    for (idx, v) in [(10u64, out_dim), (11, in_dim), (12, gs), (13, w), (14, s), (15, b), (16, tiles), (17, k), (18, input_stride), (19, base)] {
        enc.set_bytes(idx, 4, (&v as *const u32).cast());
    }
    enc.dispatch_thread_groups(MTLSize::new((tiles * k) as u64, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
}

/// CPU reference for expert `ei`, block `blk` (4-bit nibble layout).
fn expert_reference(e: &Experts, ei: usize, blk: usize, out_dim: usize, in_dim: usize, x: &[f32]) -> Vec<f32> {
    let (w, s, b) = e.offs(blk);
    let blob = &e.blobs[ei];
    let groups = out_dim * (in_dim / e.gs);
    let mut out = vec![0f32; out_dim];
    dequant_matvec_4bit_gs_cpu(
        slice(blob, w, out_dim * in_dim / 8),
        slice(blob, s, groups),
        slice(blob, b, groups),
        in_dim,
        out_dim,
        e.gs,
        x,
        &mut out,
    )
    .unwrap();
    out
}

/// The decode expert kernels at group 64 and 32: 4-bit against the CPU
/// reference on the gate (640 x 2560, shared input) and down (2560 x
/// 640, per-expert input) blocks, and packed 3-bit bit-identical to
/// the same codes in 4-bit nibbles.
#[test]
fn experts_kernels_group32_match_cpu() {
    let mut metal = MetalContext::new().expect("Metal");
    let k4 = metal.pipeline("dequant_matvec_4bit_v3_experts").expect("v3 experts").to_owned();
    let k3 = metal.pipeline("dequant_matvec_3bit_v3_experts").expect("3bit experts").to_owned();
    let mut rng = Rng(0x0038_e4e7);
    const K: usize = 8;
    for &gs in &[64usize, 32] {
        for codes3 in [false, true] {
            let (e4, e3) = make_experts(&metal, &mut rng, K, gs, codes3);
            for (blk, out_dim, in_dim, stride) in [(0usize, INTER, HIDDEN, 0usize), (2, HIDDEN, INTER, INTER)] {
                let xv: Vec<f32> = (0..K * in_dim).map(|_| rng.f32()).collect();
                let x = shared_buf(&metal, xv.len() * 4);
                slice_mut::<f32>(&x, 0, xv.len()).copy_from_slice(&xv);
                let out4 = shared_buf(&metal, K * out_dim * 4);
                run(&metal, |cmd| {
                    encode_experts(cmd, &k4, &e4, blk, &x, &out4, out_dim as u32, in_dim as u32, stride as u32)
                });
                for ei in 0..K {
                    let xi = &xv[ei * stride..][..in_dim];
                    let want = expert_reference(&e4, ei, blk, out_dim, in_dim, xi);
                    let err = rel_err(slice::<f32>(&out4, ei * out_dim * 4, out_dim), &want);
                    assert!(err < REL_TOL, "v3_experts gs{gs} block {blk} expert {ei}: rel err {err:.3e}");
                }
                if let Some(e3) = &e3 {
                    let out3 = shared_buf(&metal, K * out_dim * 4);
                    run(&metal, |cmd| {
                        encode_experts(cmd, &k3, e3, blk, &x, &out3, out_dim as u32, in_dim as u32, stride as u32)
                    });
                    let (a, b) = (slice::<u32>(&out4, 0, K * out_dim), slice::<u32>(&out3, 0, K * out_dim));
                    let differ = a.iter().zip(b).filter(|(x, y)| x != y).count();
                    assert_eq!(differ, 0, "3bit_v3_experts gs{gs} block {blk}: {differ} outputs differ from 4-bit");
                }
            }
            eprintln!("gs{gs} experts ({}): gate + down match CPU", if codes3 { "3-bit codes, both layouts" } else { "4-bit" });
        }
    }
}

/// The token-embedding gather at group 32: exact per element (no sum).
#[test]
fn embed_gather_group32_matches_cpu() {
    let mut metal = MetalContext::new().expect("Metal");
    let pso = metal.pipeline("embed_gather_4bit").expect("embed gather").to_owned();
    let mut rng = Rng(0x0038_e3b0);
    let (vocab, hidden) = (1024usize, 2560usize);
    for &gs in &[64usize, 32] {
        let m = Mat::new(&metal, &mut rng, vocab, hidden, 4, gs);
        let ids: Vec<i32> = (0..7).map(|_| (rng.u64() % vocab as u64) as i32).collect();
        let idb = shared_buf(&metal, ids.len() * 4);
        slice_mut::<i32>(&idb, 0, ids.len()).copy_from_slice(&ids);
        let out = shared_buf(&metal, ids.len() * hidden * 4);
        let (n, h, g) = (ids.len() as u32, hidden as u32, gs as u32);
        run(&metal, |cmd| {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&m.buf), 0);
            enc.set_buffer(1, Some(&m.buf), m.s_off as u64);
            enc.set_buffer(2, Some(&m.buf), m.b_off as u64);
            enc.set_buffer(3, Some(&idb), 0);
            enc.set_buffer(4, Some(&out), 0);
            enc.set_bytes(5, 4, (&n as *const u32).cast());
            enc.set_bytes(6, 4, (&h as *const u32).cast());
            enc.set_bytes(7, 4, (&g as *const u32).cast());
            enc.dispatch_thread_groups(MTLSize::new((n * h).div_ceil(256) as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        });
        let got = slice::<f32>(&out, 0, ids.len() * hidden);
        let (p, s, b) = (m.packed(), m.scales(), m.biases());
        let groups = hidden / gs;
        for (t, &id) in ids.iter().enumerate() {
            for d in 0..hidden {
                let row = id as usize;
                let q = (p[row * hidden / 8 + d / 8] >> (4 * (d % 8))) & 0xF;
                let sc = f32::from_bits((s[row * groups + d / gs] as u32) << 16);
                let bi = f32::from_bits((b[row * groups + d / gs] as u32) << 16);
                let want = q as f32 * sc + bi;
                let g = got[t * hidden + d];
                assert!((g - want).abs() <= 1e-6 * want.abs().max(1e-3), "embed gs{gs} t{t} d{d}: {g} vs {want}");
            }
        }
    }
}

// --- speed -------------------------------------------------------------------

fn time_ms(metal: &MetalContext, f: &dyn Fn(&metal::CommandBufferRef)) -> f64 {
    let once = |k: u32| {
        let cmd = metal.queue().new_command_buffer();
        for _ in 0..k {
            f(cmd);
        }
        let t = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        t.elapsed().as_secs_f64() * 1e3
    };
    once(4);
    let k = ((100.0 / once(1).max(1e-3)).round() as u32).clamp(1, 2000);
    let mut v: Vec<f64> = (0..7).map(|_| once(k) / k as f64).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[3]
}

/// Group 32 vs group 64 at the expert shape (gate|up 640 x 2560, down
/// 2560 x 640): one matrix on the single-matrix kernels, and an
/// 8-expert decode dispatch on `dequant_matvec_4bit_v3_experts`. Warm
/// GPU, many dispatches per command buffer; bandwidth counts weights +
/// scales + biases.
#[test]
#[ignore = "GPU microbench"]
fn group32_vs_group64_expert_shape_speed() {
    let mut metal = MetalContext::new().expect("Metal");
    let mut rng = Rng(0x0038_5eed);
    eprintln!("\nper dispatch, median of 7 (µs, GB/s incl. scales+biases)");
    for (kname, kind) in [
        ("dequant_matvec_4bit_v3", Kind::V3),
        ("dequant_matvec_4bit_qmv", Kind::Qmv),
        ("dequant_matvec_4bit_fast", Kind::Fast),
    ] {
        let pso = metal.pipeline(kname).expect(kname).to_owned();
        for (out_dim, in_dim, name) in [(INTER, HIDDEN, "gate|up 640x2560"), (HIDDEN, INTER, "down 2560x640")] {
            let mut line = format!("  {:26} {:18}", kname.trim_start_matches("dequant_matvec_"), name);
            for gs in [64usize, 32] {
                let m = Mat::new(&metal, &mut rng, out_dim, in_dim, 4, gs);
                let x = shared_buf(&metal, in_dim * 4);
                for v in slice_mut::<f32>(&x, 0, in_dim) {
                    *v = rng.f32();
                }
                let out = shared_buf(&metal, out_dim * 4);
                let ms = time_ms(&metal, &|cmd| encode(cmd, &pso, kind, &m, &x, &out, 1));
                let bytes = (out_dim * in_dim / 2 + 2 * out_dim * (in_dim / gs) * 2) as f64;
                line += &format!("  gs{gs}: {:6.1} µs {:5.1} GB/s", ms * 1e3, bytes / ms / 1e6);
            }
            eprintln!("{line}");
        }
    }
    let k4 = metal.pipeline("dequant_matvec_4bit_v3_experts").expect("v3 experts").to_owned();
    for (blk, out_dim, in_dim, stride, name) in
        [(0usize, INTER, HIDDEN, 0usize, "gate|up 640x2560"), (2, HIDDEN, INTER, INTER, "down 2560x640")]
    {
        let mut line = format!("  {:26} {:18}", "4bit_v3_experts x8", name);
        for gs in [64usize, 32] {
            let (e, _) = make_experts(&metal, &mut rng, 8, gs, false);
            let x = shared_buf(&metal, 8 * in_dim * 4);
            for v in slice_mut::<f32>(&x, 0, 8 * in_dim) {
                *v = rng.f32();
            }
            let out = shared_buf(&metal, 8 * out_dim * 4);
            let ms = time_ms(&metal, &|cmd| {
                encode_experts(cmd, &k4, &e, blk, &x, &out, out_dim as u32, in_dim as u32, stride as u32)
            });
            let bytes = 8.0 * (e.w_bytes + 2 * e.s_bytes) as f64;
            line += &format!("  gs{gs}: {:6.1} µs {:5.1} GB/s", ms * 1e3, bytes / ms / 1e6);
        }
        eprintln!("{line}");
    }
    let (q64, q32) = (
        streammoe::riir::variants::expert_bytes(HIDDEN, INTER, 4, 64),
        streammoe::riir::variants::expert_bytes(HIDDEN, INTER, 4, 32),
    );
    eprintln!(
        "  bytes per expert (4-bit): gs64 {q64} ({:.2} MiB), gs32 {q32} ({:.2} MiB), +{:.1}%",
        q64 as f64 / (1 << 20) as f64,
        q32 as f64 / (1 << 20) as f64,
        (q32 as f64 / q64 as f64 - 1.0) * 100.0
    );
}
