//! Per-layer embedding (PLE) layer, CPU.
//!
//! Port of transformers' `Qwen4ExpTextPLELayer`. It runs at the start
//! of decoder layer index 1 (`ple_layer_ids = [2]` is one-indexed:
//! `Qwen4ExpTextDecoderLayer` builds it where `layer_idx + 1` is in the
//! list), before that layer's attention Gated Residual, and adds its
//! output to the 4-stream residual:
//! `hidden (T × hc·H) += ple(hidden, tokens)`.
//!
//! Per token, with `e` the n-gram embedding (`ngram.rs`, 2560 wide)
//! and `x` the residual (`hc` = 4 streams of `H` = 2560):
//!
//! ```text
//! k = norm_key(key_proj · e)            hc·H, RMS per stream, × (1 + w)
//! v = value_proj · e                    H
//! q = norm_query(x)                     hc·H, RMS per stream, × (1 + w)
//! g_c = (k_c · q_c) / √H;  g_c = sign(g_c) · √max(|g_c|, 1e-6)
//! u_c = sigmoid(g_c) · v                hc·H
//! n = norm_conv(u)                      hc·H, RMS per stream, × (1 + w)
//! y_t = silu(Σ_k w[ch, k] · n_{t - (K-1-k)·D}[ch])    depthwise, K 4, dilation D = ngram_size 3
//! out = u + y
//! ```
//!
//! The conv sees zeros before the first token and keeps the last
//! `(K-1)·D` = 9 rows of `n` between calls ([`PleState`]). The
//! reference's `conv_mask` (padding in a batch) is all ones for a
//! single unpadded sequence and is not modelled; the `ple_input_ids`
//! eos substitution under it is a no-op then too.
//!
//! Tested against the reference at a tiny size
//! (`tests/fixtures/qwen38_ple/ple_tiny.*`, from
//! `tools/model_layer/qwen3_8_flash_next/ple_reference.py`).
//!
//! ## Paths
//!
//! - [`PleLayer::forward`]: all on the CPU, any number of tokens; a
//!   chunk equals its tokens one at a time bit for bit.
//! - [`PleLayer::forward_gpu`] / [`PleGpuProj`]: `key_proj` and
//!   `value_proj` (4-bit group 32) on the GPU — `_n_tokens` matvec, or
//!   the MLX `qmm` GEMM from 16 tokens. The norms, the signed-sqrt
//!   gate and the dilated conv fit no existing kernel and stay on the
//!   CPU ([`forward_with_projections`]), so a GPU engine reads `k`,
//!   `v` and the residual back at this layer (once per prefill chunk,
//!   and once per decode token).
//! - [`PleBlock`]: layer + weights + state as q38-core's `PleHook`,
//!   one token or a chunk, CPU or GPU projections.

use rayon::prelude::*;

use super::gated_residual_gpu::{GpuLinear, GpuQ4Linear};
use super::layer::{PleHook, Q38Error};
use super::ngram::{NgramConfig, NgramHasher, NgramHistory, NgramTable, NgramTableError};
use crate::riir::backend::gpu::metal::{MetalContext, MetalError, MtlBuffer};
use crate::riir::io::embedding::bf16_to_f32;

/// Layer shape (from the text config).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PleConfig {
    pub hidden: usize,
    /// `hc_count`: residual streams.
    pub hc: usize,
    pub embed_dim: usize,
    pub conv_kernel: usize,
    /// Conv dilation (= `ngram_size`).
    pub conv_dilation: usize,
    /// Quantization group of `key_proj` / `value_proj`.
    pub group_size: usize,
    pub eps: f32,
}

impl PleConfig {
    pub const QWEN3_8_FLASH_NEXT: PleConfig = PleConfig {
        hidden: 2560,
        hc: 4,
        embed_dim: 2560,
        conv_kernel: 4,
        conv_dilation: 3,
        group_size: 32,
        eps: 1e-6,
    };

    /// Width of the residual stream (`hc · hidden`).
    pub fn hc_hidden(&self) -> usize {
        self.hc * self.hidden
    }

    /// Rows of conv history kept between calls (`(K-1) · D`).
    pub fn conv_state_rows(&self) -> usize {
        (self.conv_kernel - 1) * self.conv_dilation
    }
}

/// Decoder layer index (0-based) that runs the PLE for each entry of
/// `ple_layer_ids` (one-indexed ids): `id - 1`.
pub fn ple_decoder_layer(ple_layer_id: usize) -> usize {
    ple_layer_id - 1
}

/// The layer's weights as raw little-endian bytes (MLX layout).
/// Tensor names (manifest, `model.layers.1.ple.` prefix):
/// `key_proj.{weight,scales,biases}` (4-bit, group 32,
/// `[hc·H, embed]`), `value_proj.{…}` (`[H, embed]`),
/// `norm_key.weight` / `norm_query.weight` / `norm_conv.weight`
/// (bf16, `hc·H`, stored zero-centered — the layer applies `1 + w`),
/// `conv1d.weight` (bf16, `hc·H × K` with `k` fastest — torch
/// `[C, 1, K]` and MLX `[C, K, 1]` flatten the same).
#[derive(Debug, Clone, Copy)]
pub struct PleWeights<'a> {
    pub key_w: &'a [u8],
    pub key_s: &'a [u8],
    pub key_b: &'a [u8],
    pub value_w: &'a [u8],
    pub value_s: &'a [u8],
    pub value_b: &'a [u8],
    pub norm_key: &'a [u8],
    pub norm_query: &'a [u8],
    pub norm_conv: &'a [u8],
    pub conv: &'a [u8],
}

/// Manifest names of the layer's tensors, for decoder layer `layer`.
pub fn tensor_names(layer: usize) -> [String; 10] {
    let p = format!("model.layers.{layer}.ple.");
    [
        format!("{p}key_proj.weight"),
        format!("{p}key_proj.scales"),
        format!("{p}key_proj.biases"),
        format!("{p}value_proj.weight"),
        format!("{p}value_proj.scales"),
        format!("{p}value_proj.biases"),
        format!("{p}norm_key.weight"),
        format!("{p}norm_query.weight"),
        format!("{p}norm_conv.weight"),
        format!("{p}conv1d.weight"),
    ]
}

#[derive(Debug, thiserror::Error)]
pub enum PleError {
    #[error(transparent)]
    Table(#[from] NgramTableError),
    #[error("ple: {field} is {got} bytes / values, expected {expected}")]
    Shape { field: &'static str, got: usize, expected: usize },
}

/// Per-sequence state of the PLE layer: the n-gram token history and
/// the conv history. Save it with the rest of the sequence state.
/// Like the linear-attention state it cannot be truncated to an
/// earlier position (the conv rows are activations): a rewind must
/// restore a snapshot or replay from the start.
#[derive(Debug, Clone, PartialEq)]
pub struct PleState {
    pub history: NgramHistory,
    /// Last `conv_state_rows` rows of `norm_conv(u)`, oldest first,
    /// `rows × hc·H` (zeros at the start of a sequence).
    pub conv: Vec<f32>,
}

impl PleState {
    pub fn new(cfg: &PleConfig, ncfg: &NgramConfig) -> Self {
        Self { history: NgramHistory::new(ncfg), conv: vec![0.0; cfg.conv_state_rows() * cfg.hc_hidden()] }
    }

    /// Back to the start of a sequence.
    pub fn reset(&mut self) {
        self.history.reset();
        self.conv.fill(0.0);
    }

    /// Bytes of [`Self::save`] (8 + 368,640 on the real model).
    pub fn state_bytes(&self) -> usize {
        self.history.state_bytes() + 4 * self.conv.len()
    }

    /// Append the state to `out`: history (i32 LE), then conv rows
    /// (f32 LE).
    pub fn save(&self, out: &mut Vec<u8>) {
        self.history.save(out);
        for v in &self.conv {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// Load from [`Self::save`] bytes.
    pub fn load(&mut self, bytes: &[u8]) -> Result<(), PleError> {
        if bytes.len() != self.state_bytes() {
            return Err(PleError::Shape { field: "state", got: bytes.len(), expected: self.state_bytes() });
        }
        let (h, c) = bytes.split_at(self.history.state_bytes());
        self.history.load(h)?;
        for (v, b) in self.conv.iter_mut().zip(c.chunks_exact(4)) {
            *v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        Ok(())
    }
}

/// `out[t·out_dim + r] = Σ_c dequant(W[r, c]) · x[t·in_dim + c]` for an
/// MLX 4-bit matrix with any group size. Each row is dequantized once
/// for all `T` tokens; rows run in parallel.
pub fn matvec_4bit_tokens(
    w: &[u8],
    s: &[u8],
    b: &[u8],
    group: usize,
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), PleError> {
    let groups = in_dim / group;
    let check = |field, got: usize, expected: usize| {
        if got == expected { Ok(()) } else { Err(PleError::Shape { field, got, expected }) }
    };
    check("proj in_dim % group", in_dim % group, 0)?;
    check("proj weight", w.len(), out_dim * in_dim / 2)?;
    check("proj scales", s.len(), out_dim * groups * 2)?;
    check("proj biases", b.len(), out_dim * groups * 2)?;
    check("proj x", x.len() % in_dim, 0)?;
    let t_n = x.len() / in_dim;
    check("proj out", out.len(), t_n * out_dim)?;
    let mut cols = vec![0f32; out_dim * t_n]; // [r][t]
    cols.par_chunks_mut(t_n).enumerate().for_each_init(
        || vec![0f32; in_dim],
        |row, (r, dst)| {
            let wr = &w[r * in_dim / 2..(r + 1) * in_dim / 2];
            for g in 0..groups {
                let o = 2 * (r * groups + g);
                let sc = bf16_to_f32(u16::from_le_bytes([s[o], s[o + 1]]));
                let bi = bf16_to_f32(u16::from_le_bytes([b[o], b[o + 1]]));
                for j in g * group..(g + 1) * group {
                    let q = (wr[j / 2] >> (4 * (j % 2))) & 0xF;
                    row[j] = (q as f32) * sc + bi;
                }
            }
            for (t, d) in dst.iter_mut().enumerate() {
                let xt = &x[t * in_dim..(t + 1) * in_dim];
                let mut acc = 0f32;
                for (a, c) in row.iter().zip(xt) {
                    acc += a * c;
                }
                *d = acc;
            }
        },
    );
    for r in 0..out_dim {
        for t in 0..t_n {
            out[t * out_dim + r] = cols[r * t_n + t];
        }
    }
    Ok(())
}

/// `Qwen4ExpTextRMSNorm` with `group_size`: each `group` of `x` is
/// normalized on its own, then scaled by `1 + w` (w bf16 bytes).
fn rms_norm_groups_1p(x: &[f32], w: &[u8], group: usize, eps: f32, out: &mut [f32]) {
    for (gi, (xg, og)) in x.chunks(group).zip(out.chunks_mut(group)).enumerate() {
        let mut ss = 0f32;
        for v in xg {
            ss += v * v;
        }
        let inv = 1.0 / (ss / group as f32 + eps).sqrt();
        for (i, (o, v)) in og.iter_mut().zip(xg).enumerate() {
            let k = 2 * (gi * group + i);
            *o = v * inv * (1.0 + bf16_to_f32(u16::from_le_bytes([w[k], w[k + 1]])));
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One PLE layer: hashing, the row table and the shape. Weights and
/// per-sequence state are passed per call.
#[derive(Debug)]
pub struct PleLayer {
    pub cfg: PleConfig,
    pub hasher: NgramHasher,
    pub table: NgramTable,
}

impl PleLayer {
    /// Wrap an opened table; checks it against the hashing.
    pub fn new(cfg: PleConfig, ncfg: NgramConfig, table: NgramTable) -> Result<Self, PleError> {
        let hasher = NgramHasher::new(ncfg);
        table.check_hasher(&hasher, cfg.embed_dim)?;
        Ok(Self { cfg, hasher, table })
    }

    /// A fresh per-sequence state.
    pub fn new_state(&self) -> PleState {
        PleState::new(&self.cfg, self.hasher.config())
    }

    /// The n-gram embedding of `tokens` (`T × embed_dim`), advancing
    /// `state.history`.
    pub fn embedding(
        &self,
        state: &mut PleState,
        tokens: &[i32],
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<Vec<f32>, PleError> {
        let mut ids = vec![0u64; tokens.len() * self.hasher.n_heads()];
        self.hasher.ids(&mut state.history, tokens, &mut ids);
        let mut e = vec![0f32; tokens.len() * self.cfg.embed_dim];
        self.table.embed(&ids, &mut e, pool)?;
        Ok(e)
    }

    /// PLE output for `tokens` (`T`) at residual `hidden`
    /// (`T × hc·H`) into `out` (`T × hc·H`); the caller adds it to
    /// the residual. Advances `state`. `pool` runs the row reads.
    pub fn forward(
        &self,
        w: &PleWeights,
        state: &mut PleState,
        tokens: &[i32],
        hidden: &[f32],
        out: &mut [f32],
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<(), PleError> {
        let e = self.embedding(state, tokens, pool)?;
        forward_from_embedding(&self.cfg, w, state, &e, hidden, out, None)
    }

    /// [`Self::forward`] with `key_proj` / `value_proj` on the GPU
    /// (`gpu`); the n-gram rows, norms, gate and dilated conv stay on
    /// the CPU. Within f32 rounding of [`Self::forward`] (the GPU sums
    /// the projections in another order).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_gpu(
        &self,
        w: &PleWeights,
        state: &mut PleState,
        tokens: &[i32],
        hidden: &[f32],
        out: &mut [f32],
        pool: Option<&rayon::ThreadPool>,
        gpu: &mut PleGpuProj,
    ) -> Result<(), PleError> {
        let e = self.embedding(state, tokens, pool)?;
        forward_from_embedding(&self.cfg, w, state, &e, hidden, out, Some(gpu))
    }

    /// The rest of [`Self::forward`] from the projections
    /// `k = key_proj · e` (`T × hc·H`, before `norm_key`) and
    /// `v = value_proj · e` (`T × H`) — for a caller that runs them on
    /// the GPU ([`PleGpuProj::encode`]). Call [`Self::embedding`] first
    /// (it advances the token history); this advances the conv state.
    pub fn forward_with_embedding(
        &self,
        w: &PleWeights,
        state: &mut PleState,
        k: &[f32],
        v: &[f32],
        hidden: &[f32],
        out: &mut [f32],
    ) -> Result<(), PleError> {
        forward_with_projections(&self.cfg, w, state, k, v, hidden, out)
    }
}

/// PLE from the n-gram embedding `e` (`T × embed_dim`): the two
/// projections — on the CPU, or on the GPU with `gpu` — then
/// [`forward_with_projections`]. Advances the conv state.
pub fn forward_from_embedding(
    c: &PleConfig,
    w: &PleWeights,
    state: &mut PleState,
    e: &[f32],
    hidden: &[f32],
    out: &mut [f32],
    gpu: Option<&mut PleGpuProj>,
) -> Result<(), PleError> {
    let t_n = e.len() / c.embed_dim;
    let mut k = vec![0f32; t_n * c.hc_hidden()];
    let mut v = vec![0f32; t_n * c.hidden];
    match gpu {
        Some(g) => g.project(e, &mut k, &mut v)?,
        None => {
            matvec_4bit_tokens(w.key_w, w.key_s, w.key_b, c.group_size, c.embed_dim, c.hc_hidden(), e, &mut k)?;
            matvec_4bit_tokens(w.value_w, w.value_s, w.value_b, c.group_size, c.embed_dim, c.hidden, e, &mut v)?;
        }
    }
    forward_with_projections(c, w, state, &k, &v, hidden, out)
}

/// The PLE after the projections ([`PleLayer::forward_with_embedding`]
/// without the table): `norm_key`, `norm_query`, the gate, `norm_conv`
/// and the dilated conv, on the CPU. Each token's arithmetic does not
/// depend on how the tokens are split into calls, so a chunk equals
/// its tokens one at a time bit for bit.
pub fn forward_with_projections(
    c: &PleConfig,
    w: &PleWeights,
    state: &mut PleState,
    k: &[f32],
    v: &[f32],
    hidden: &[f32],
    out: &mut [f32],
) -> Result<(), PleError> {
    let (hh, h) = (c.hc_hidden(), c.hidden);
    let t_n = v.len() / h;
    let check = |field, got: usize, expected: usize| {
        if got == expected { Ok(()) } else { Err(PleError::Shape { field, got, expected }) }
    };
    check("v", v.len(), t_n * h)?;
    check("k", k.len(), t_n * hh)?;
    check("hidden", hidden.len(), t_n * hh)?;
    check("out", out.len(), t_n * hh)?;
    for (f, n) in [("norm_key", w.norm_key), ("norm_query", w.norm_query), ("norm_conv", w.norm_conv)] {
        check(f, n.len(), 2 * hh)?;
    }
    check("conv1d", w.conv.len(), 2 * hh * c.conv_kernel)?;
    check("state.conv", state.conv.len(), c.conv_state_rows() * hh)?;

    let s_rows = c.conv_state_rows();
    // Conv input history: state rows, then this call's rows.
    let mut hist = Vec::with_capacity((s_rows + t_n) * hh);
    hist.extend_from_slice(&state.conv);
    let mut kn = vec![0f32; hh];
    let mut qn = vec![0f32; hh];
    let scale = (h as f32).sqrt();
    for t in 0..t_n {
        rms_norm_groups_1p(&k[t * hh..(t + 1) * hh], w.norm_key, h, c.eps, &mut kn);
        rms_norm_groups_1p(&hidden[t * hh..(t + 1) * hh], w.norm_query, h, c.eps, &mut qn);
        let u = &mut out[t * hh..(t + 1) * hh];
        for s in 0..c.hc {
            let mut g = 0f32;
            for i in s * h..(s + 1) * h {
                g += kn[i] * qn[i];
            }
            g /= scale;
            let sign = if g > 0.0 { 1.0 } else if g < 0.0 { -1.0 } else { 0.0 };
            let g = g.abs().max(1e-6).sqrt() * sign;
            let sg = sigmoid(g);
            for (o, &vi) in u[s * h..(s + 1) * h].iter_mut().zip(&v[t * h..(t + 1) * h]) {
                *o = sg * vi;
            }
        }
        let base = hist.len();
        hist.resize(base + hh, 0.0);
        rms_norm_groups_1p(u, w.norm_conv, h, c.eps, &mut hist[base..]);
    }
    // Depthwise dilated conv + SiLU, added to u (already in out).
    let (kk, d) = (c.conv_kernel, c.conv_dilation);
    out.par_chunks_mut(hh).enumerate().for_each(|(t, o)| {
        let p = s_rows + t;
        for ch in 0..hh {
            let mut acc = 0f32;
            for j in 0..kk {
                let src = p - (kk - 1 - j) * d;
                let wi = 2 * (ch * kk + j);
                acc += bf16_to_f32(u16::from_le_bytes([w.conv[wi], w.conv[wi + 1]])) * hist[src * hh + ch];
            }
            o[ch] += acc * sigmoid(acc);
        }
    });
    let keep = hist.len() - s_rows * hh;
    state.conv.copy_from_slice(&hist[keep..]);
    Ok(())
}

// =====================================================================
// GPU projections
// =====================================================================

/// `key_proj` and `value_proj` on the GPU: two [`GpuQ4Linear`]s at the
/// layer's group (32), `_n_tokens` matvec for short chunks and the MLX
/// `qmm` GEMM from `gemm_min_tokens` rows. The norms, the gate and the
/// dilated conv have no existing kernel and stay on the CPU
/// ([`forward_with_projections`]).
///
/// Two ways to use it: [`Self::encode`] onto the caller's command
/// buffer (the engine: `e` already in a GPU buffer, `k` / `v` read
/// back after the commit), or [`Self::project`], which uploads `e`,
/// runs, waits and copies `k` / `v` out through its own buffers.
pub struct PleGpuProj {
    pub key: GpuQ4Linear,
    pub value: GpuQ4Linear,
    queue: metal::CommandQueue,
    device: metal::Device,
    /// `[cap, embed]`, `[cap, hc·H]`, `[cap, H]`; regrown on demand.
    e: MtlBuffer<f32>,
    k: MtlBuffer<f32>,
    v: MtlBuffer<f32>,
}

impl PleGpuProj {
    /// From two linears whose weights live where the caller put them
    /// (the engine's weight buffer: [`GpuQ4Linear::new`] with the
    /// tensors' byte offsets).
    pub fn new(metal: &MetalContext, key: GpuQ4Linear, value: GpuQ4Linear) -> Self {
        assert_eq!(key.in_dim, value.in_dim, "PleGpuProj: key / value in_dim");
        let device = metal.device().clone();
        Self {
            e: MtlBuffer::with_len(&device, key.in_dim),
            k: MtlBuffer::with_len(&device, key.out_dim),
            v: MtlBuffer::with_len(&device, value.out_dim),
            key,
            value,
            queue: metal.queue_clone(),
            device,
        }
    }

    /// Copy the two projections out of `w` into GPU buffers of their
    /// own (tests, or a caller without the weight file on the GPU).
    pub fn from_weights(metal: &mut MetalContext, cfg: &PleConfig, w: &PleWeights) -> Result<Self, MetalError> {
        let key = GpuQ4Linear::from_bytes(metal, w.key_w, w.key_s, w.key_b, cfg.hc_hidden(), cfg.embed_dim, cfg.group_size)?;
        let value = GpuQ4Linear::from_bytes(metal, w.value_w, w.value_s, w.value_b, cfg.hidden, cfg.embed_dim, cfg.group_size)?;
        Ok(Self::new(metal, key, value))
    }

    /// Encode `k = key_proj · e` and `v = value_proj · e` for
    /// `n_tokens` rows: `e` `[n, embed]` at `e_off`, `k` `[n, hc·H]`,
    /// `v` `[n, H]` (f32, byte offsets).
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        cmdbuf: &metal::CommandBufferRef,
        e: &metal::BufferRef,
        e_off: u64,
        k: &metal::BufferRef,
        k_off: u64,
        v: &metal::BufferRef,
        v_off: u64,
        n_tokens: usize,
    ) {
        self.key.encode_n(cmdbuf, e, e_off, k, k_off, n_tokens);
        self.value.encode_n(cmdbuf, e, e_off, v, v_off, n_tokens);
    }

    /// Upload `e` (`T × embed`), run both projections, wait, and write
    /// `k` (`T × hc·H`) and `v` (`T × H`).
    pub fn project(&mut self, e: &[f32], k: &mut [f32], v: &mut [f32]) -> Result<(), PleError> {
        let (ed, kd, vd) = (self.key.in_dim, self.key.out_dim, self.value.out_dim);
        let t_n = e.len() / ed;
        let check = |field, got: usize, expected: usize| {
            if got == expected { Ok(()) } else { Err(PleError::Shape { field, got, expected }) }
        };
        check("gpu e", e.len(), t_n * ed)?;
        check("gpu k", k.len(), t_n * kd)?;
        check("gpu v", v.len(), t_n * vd)?;
        if t_n == 0 {
            return Ok(());
        }
        if self.e.len() < t_n * ed {
            self.e = MtlBuffer::with_len(&self.device, t_n * ed);
            self.k = MtlBuffer::with_len(&self.device, t_n * kd);
            self.v = MtlBuffer::with_len(&self.device, t_n * vd);
        }
        self.e.as_mut_slice()[..e.len()].copy_from_slice(e);
        let cb = self.queue.new_command_buffer();
        self.encode(cb, self.e.raw(), 0, self.k.raw(), 0, self.v.raw(), 0, t_n);
        cb.commit();
        cb.wait_until_completed();
        k.copy_from_slice(&self.k.as_slice()[..k.len()]);
        v.copy_from_slice(&self.v.as_slice()[..v.len()]);
        Ok(())
    }
}

// =====================================================================
// PleBlock: the PleHook
// =====================================================================

/// [`PleWeights`] with owned bytes.
#[derive(Debug, Clone, Default)]
pub struct PleWeightsOwned {
    pub key_w: Vec<u8>,
    pub key_s: Vec<u8>,
    pub key_b: Vec<u8>,
    pub value_w: Vec<u8>,
    pub value_s: Vec<u8>,
    pub value_b: Vec<u8>,
    pub norm_key: Vec<u8>,
    pub norm_query: Vec<u8>,
    pub norm_conv: Vec<u8>,
    pub conv: Vec<u8>,
}

impl PleWeightsOwned {
    pub fn from_weights(w: &PleWeights) -> Self {
        Self {
            key_w: w.key_w.to_vec(),
            key_s: w.key_s.to_vec(),
            key_b: w.key_b.to_vec(),
            value_w: w.value_w.to_vec(),
            value_s: w.value_s.to_vec(),
            value_b: w.value_b.to_vec(),
            norm_key: w.norm_key.to_vec(),
            norm_query: w.norm_query.to_vec(),
            norm_conv: w.norm_conv.to_vec(),
            conv: w.conv.to_vec(),
        }
    }
}

/// Where a [`PleBlock`] gets its weights: owned bytes, borrowed
/// slices, or the engine's mapped weight file.
pub trait PleWeightSource {
    fn ple_weights(&self) -> PleWeights<'_>;
}

impl PleWeightSource for PleWeightsOwned {
    fn ple_weights(&self) -> PleWeights<'_> {
        PleWeights {
            key_w: &self.key_w,
            key_s: &self.key_s,
            key_b: &self.key_b,
            value_w: &self.value_w,
            value_s: &self.value_s,
            value_b: &self.value_b,
            norm_key: &self.norm_key,
            norm_query: &self.norm_query,
            norm_conv: &self.norm_conv,
            conv: &self.conv,
        }
    }
}

impl PleWeightSource for PleWeights<'_> {
    fn ple_weights(&self) -> PleWeights<'_> {
        *self
    }
}

/// The PLE layer with its weights and per-sequence state: q38-core's
/// `PleHook` (`layer.rs`), one token ([`PleHook::forward`]) or a
/// chunk ([`PleHook::forward_n`]). A chunk equals its tokens one at a
/// time bit for bit on the CPU, and on the GPU when the projections
/// run the `_n_tokens` matvec without qmv (`gpu.key.qmv = false`,
/// `gemm_min_tokens = usize::MAX` on both linears); with the defaults
/// (qmv for one row, GEMM from 16) the GPU projections differ from
/// per-token ones by f32 rounding.
pub struct PleBlock<W: PleWeightSource> {
    /// Decoder layer index this block answers for (1 on the real
    /// model, [`ple_decoder_layer`]).
    pub layer_index: usize,
    pub ple: PleLayer,
    pub weights: W,
    pub state: PleState,
    /// Position of the next token. [`PleState`] does not carry it:
    /// set it after [`PleState::load`].
    pub next_pos: usize,
    /// Runs the n-gram row reads (`None`: the global rayon pool).
    pub pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    /// `Some`: projections on the GPU.
    pub gpu: Option<PleGpuProj>,
}

impl<W: PleWeightSource> PleBlock<W> {
    pub fn new(layer_index: usize, ple: PleLayer, weights: W) -> Self {
        let state = ple.new_state();
        Self { layer_index, ple, weights, state, next_pos: 0, pool: None, gpu: None }
    }

    /// `tokens` (`T`) at positions `pos..pos + T`; `hidden` / `out`
    /// `[T, hc·H]`. `pos` must be [`Self::next_pos`].
    pub fn forward_tokens(&mut self, pos: usize, tokens: &[i32], hidden: &[f32], out: &mut [f32]) -> Result<(), PleError> {
        if pos != self.next_pos {
            return Err(PleError::Shape { field: "position", got: pos, expected: self.next_pos });
        }
        let w = self.weights.ple_weights();
        let pool = self.pool.as_deref();
        match self.gpu.as_mut() {
            Some(g) => self.ple.forward_gpu(&w, &mut self.state, tokens, hidden, out, pool, g)?,
            None => self.ple.forward(&w, &mut self.state, tokens, hidden, out, pool)?,
        }
        self.next_pos += tokens.len();
        Ok(())
    }

    /// Back to the start of a sequence.
    pub fn reset(&mut self) {
        self.state.reset();
        self.next_pos = 0;
    }
}

impl<W: PleWeightSource> PleHook for PleBlock<W> {
    fn forward(&mut self, layer: usize, pos: usize, token: u32, hyper: &[f32], out: &mut [f32]) -> Result<(), Q38Error> {
        PleHook::forward_n(self, layer, pos, &[token], hyper, out)
    }
    fn forward_n(&mut self, layer: usize, pos: usize, tokens: &[u32], hyper: &[f32], out: &mut [f32])
        -> Result<(), Q38Error> {
        if layer != self.layer_index {
            return Err(Q38Error::Hook { layer, msg: format!("PLE block runs at layer {}", self.layer_index) });
        }
        let toks: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        self.forward_tokens(pos, &toks, hyper, out).map_err(|e| Q38Error::Hook { layer, msg: e.to_string() })
    }
    fn reset(&mut self) {
        PleBlock::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(v: &serde_json::Value) -> Vec<u8> {
        let s = v.as_str().unwrap();
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    fn f32s(v: &serde_json::Value) -> Vec<f32> {
        v.as_array().unwrap().iter().map(|x| f32::from_bits(x.as_u64().unwrap() as u32)).collect()
    }

    /// The tiny reference layer (hidden 32, 4 streams, 4 heads of 32,
    /// table read from an `MFNG` file) across prefill and decode steps
    /// of two sequences, with a save / load of the state in between.
    #[test]
    fn forward_matches_reference_tiny() {
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
        let bytes: Vec<Vec<u8>> = ["key_w", "key_s", "key_b", "value_w", "value_s", "value_b", "norm_key", "norm_query", "norm_conv", "conv"]
            .iter()
            .map(|k| unhex(&fx[*k]))
            .collect();
        let w = PleWeights {
            key_w: &bytes[0],
            key_s: &bytes[1],
            key_b: &bytes[2],
            value_w: &bytes[3],
            value_s: &bytes[4],
            value_b: &bytes[5],
            norm_key: &bytes[6],
            norm_query: &bytes[7],
            norm_conv: &bytes[8],
            conv: &bytes[9],
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let (mut worst, mut n) = (0f32, 0usize);
        for (si, seq) in fx["seqs"].as_array().unwrap().iter().enumerate() {
            let mut state = layer.new_state();
            for (step_i, step) in seq.as_array().unwrap().iter().enumerate() {
                if si == 0 && step_i == 4 {
                    // Snapshot round trip mid-sequence.
                    let mut buf = Vec::new();
                    state.save(&mut buf);
                    assert_eq!(buf.len(), state.state_bytes());
                    let mut fresh = layer.new_state();
                    fresh.load(&buf).unwrap();
                    assert_eq!(fresh, state);
                    state = fresh;
                }
                let toks: Vec<i32> = step["tokens"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
                let hs = f32s(&step["hidden"]);
                let want = f32s(&step["out"]);
                let mut got = vec![0f32; want.len()];
                let p = if step_i % 2 == 0 { Some(&pool) } else { None };
                layer.forward(&w, &mut state, &toks, &hs, &mut got, p).unwrap();
                for (a, b) in got.iter().zip(&want) {
                    worst = worst.max((a - b).abs() / (1.0 + b.abs()));
                }
                n += toks.len();
            }
        }
        assert!(n >= 25, "{n}");
        assert!(worst < 2e-5, "worst relative error {worst}");
    }

    #[test]
    fn real_config_shapes() {
        let c = PleConfig::QWEN3_8_FLASH_NEXT;
        assert_eq!(ple_decoder_layer(2), 1);
        assert_eq!(c.conv_state_rows(), 9);
        let s = PleState::new(&c, &NgramConfig::QWEN3_8_FLASH_NEXT);
        assert_eq!(s.state_bytes(), 8 + 9 * 10240 * 4);
        assert_eq!(NgramConfig::QWEN3_8_FLASH_NEXT.n_heads() * 160, c.embed_dim);
    }

    // --- batched and GPU --------------------------------------------

    /// The tiny fixture layer and its weights.
    fn tiny_layer() -> (PleLayer, PleWeightsOwned, usize) {
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
        let b = |k: &str| unhex(&fx[k]);
        let w = PleWeightsOwned {
            key_w: b("key_w"),
            key_s: b("key_s"),
            key_b: b("key_b"),
            value_w: b("value_w"),
            value_s: b("value_s"),
            value_b: b("value_b"),
            norm_key: b("norm_key"),
            norm_query: b("norm_query"),
            norm_conv: b("norm_conv"),
            conv: b("conv"),
        };
        (PleLayer::new(cfg, ncfg, table).unwrap(), w, get("vocab"))
    }

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
        fn bf16_bytes(&mut self, n: usize, scale: f32) -> Vec<u8> {
            (0..n)
                .flat_map(|_| {
                    let b = (self.f32() * scale).to_bits();
                    (((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16).to_le_bytes()
                })
                .collect()
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.u64() as u8).collect()
        }
    }

    const CHUNKS: [usize; 6] = [3, 1, 6, 2, 8, 1];

    /// Run `tokens` through `block`: one by one (`chunks = None`) or in
    /// `chunks`, through the `PleHook` trait.
    fn run_block<W: PleWeightSource>(block: &mut PleBlock<W>, tokens: &[u32], hidden: &[f32], chunks: Option<&[usize]>) -> Vec<f32> {
        let hh = block.ple.cfg.hc_hidden();
        let mut out = vec![0f32; hidden.len()];
        let one: Vec<usize> = vec![1; tokens.len()];
        let mut t0 = 0;
        for &n in chunks.unwrap_or(&one) {
            let (a, b) = (t0 * hh, (t0 + n) * hh);
            if n == 1 && chunks.is_none() {
                PleHook::forward(block, 1, t0, tokens[t0], &hidden[a..b], &mut out[a..b]).unwrap();
            } else {
                PleHook::forward_n(block, 1, t0, &tokens[t0..t0 + n], &hidden[a..b], &mut out[a..b]).unwrap();
            }
            t0 += n;
        }
        assert_eq!(t0, tokens.len());
        out
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
    }

    /// `PleHook::forward_n` on [`PleBlock`] (CPU) equals token-by-token
    /// `PleHook::forward` bit for bit; a wrong position or layer is
    /// refused; `reset` restarts the sequence.
    #[test]
    fn ple_block_forward_n_bit_identical_cpu() {
        let (layer, w, vocab) = tiny_layer();
        let hh = layer.cfg.hc_hidden();
        let n: usize = CHUNKS.iter().sum();
        let mut r = Rng(0x504c_4501);
        let tokens: Vec<u32> = (0..n).map(|_| (r.u64() % vocab as u64) as u32).collect();
        let hidden = r.vec(n * hh, 1.5);
        let mut block = PleBlock::new(1, layer, w);
        let want = run_block(&mut block, &tokens, &hidden, None);
        block.reset();
        let got = run_block(&mut block, &tokens, &hidden, Some(&CHUNKS));
        assert!(same_bits(&got, &want), "chunked PLE != token by token");
        assert!(PleHook::forward(&mut block, 1, 3, tokens[0], &hidden[..hh], &mut vec![0.0; hh]).is_err());
        assert!(PleHook::forward(&mut block, 2, n, tokens[0], &hidden[..hh], &mut vec![0.0; hh]).is_err());
        PleHook::reset(&mut block);
        assert_eq!(block.next_pos, 0);
        let again = run_block(&mut block, &tokens, &hidden, Some(&[n]));
        assert!(same_bits(&again, &want), "one-chunk PLE != token by token");
    }

    /// The tiny layer with the projections on the GPU: against the CPU
    /// block (default kernels, and the GEMM forced), and chunked vs
    /// token by token on the GPU bit for bit (qmv and GEMM off).
    #[test]
    fn ple_block_gpu_tiny_matches_cpu() {
        let Ok(mut metal) = MetalContext::new() else {
            eprintln!("[ple gpu] skipping: no Metal");
            return;
        };
        let (layer, w, vocab) = tiny_layer();
        let cfg = layer.cfg;
        let hh = cfg.hc_hidden();
        let n: usize = CHUNKS.iter().sum();
        let mut r = Rng(0x504c_4502);
        let tokens: Vec<u32> = (0..n).map(|_| (r.u64() % vocab as u64) as u32).collect();
        let hidden = r.vec(n * hh, 1.5);
        let mut block = PleBlock::new(1, layer, w);
        let cpu = run_block(&mut block, &tokens, &hidden, Some(&CHUNKS));
        block.reset();
        block.gpu = Some(PleGpuProj::from_weights(&mut metal, &cfg, &block.weights.ple_weights()).unwrap());
        for (label, gemm_min) in [("default", 16usize), ("gemm", 1)] {
            let g = block.gpu.as_mut().unwrap();
            g.key.gemm_min_tokens = gemm_min;
            g.value.gemm_min_tokens = gemm_min;
            let got = run_block(&mut block, &tokens, &hidden, Some(&CHUNKS));
            block.reset();
            let e = max_rel(&got, &cpu);
            eprintln!("[ple gpu tiny] {label}: max|diff| / max|ref| {e:.2e}");
            assert!(e < 2e-5, "{label}: {e}");
        }
        {
            let g = block.gpu.as_mut().unwrap();
            for lin in [&mut g.key, &mut g.value] {
                lin.qmv = false;
                lin.gemm_min_tokens = usize::MAX;
            }
        }
        let one = run_block(&mut block, &tokens, &hidden, None);
        block.reset();
        let many = run_block(&mut block, &tokens, &hidden, Some(&CHUNKS));
        assert!(same_bits(&one, &many), "GPU chunked PLE != GPU token by token");
    }

    /// The real shape (`hc·H` 10240, embed 2560, group 32) from a random
    /// embedding: GPU projections vs the CPU ones through
    /// [`forward_from_embedding`], at 1, 3 and 20 tokens (qmv, matvec_n,
    /// GEMM), carrying the conv state across calls.
    #[test]
    fn ple_gpu_real_shape_matches_cpu() {
        let Ok(mut metal) = MetalContext::new() else {
            eprintln!("[ple gpu] skipping: no Metal");
            return;
        };
        let c = PleConfig::QWEN3_8_FLASH_NEXT;
        let (hh, h, ed, g) = (c.hc_hidden(), c.hidden, c.embed_dim, c.group_size);
        let mut r = Rng(0x504c_4503);
        let w = PleWeightsOwned {
            key_w: r.bytes(hh * ed / 2),
            key_s: r.bf16_bytes(hh * ed / g, 0.05),
            key_b: r.bf16_bytes(hh * ed / g, 0.02),
            value_w: r.bytes(h * ed / 2),
            value_s: r.bf16_bytes(h * ed / g, 0.05),
            value_b: r.bf16_bytes(h * ed / g, 0.02),
            norm_key: r.bf16_bytes(hh, 0.2),
            norm_query: r.bf16_bytes(hh, 0.2),
            norm_conv: r.bf16_bytes(hh, 0.2),
            conv: r.bf16_bytes(hh * c.conv_kernel, 0.5),
        };
        let pw = w.ple_weights();
        let mut gpu = PleGpuProj::from_weights(&mut metal, &c, &pw).unwrap();
        let mut st_cpu = PleState::new(&c, &NgramConfig::QWEN3_8_FLASH_NEXT);
        let mut st_gpu = st_cpu.clone();
        for n in [1usize, 3, 20] {
            let e = r.vec(n * ed, 0.1);
            let hidden = r.vec(n * hh, 1.0);
            let (mut a, mut b) = (vec![0f32; n * hh], vec![0f32; n * hh]);
            forward_from_embedding(&c, &pw, &mut st_cpu, &e, &hidden, &mut a, None).unwrap();
            forward_from_embedding(&c, &pw, &mut st_gpu, &e, &hidden, &mut b, Some(&mut gpu)).unwrap();
            let (eo, es) = (max_rel(&b, &a), max_rel(&st_gpu.conv, &st_cpu.conv));
            eprintln!("[ple gpu real] {n} tokens: out {eo:.2e}, conv state {es:.2e}");
            assert!(eo < 1e-4 && es < 1e-4, "{n} tokens: out {eo}, state {es}");
        }
    }
}
