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
//! `tools/model_layer/qwen3_8_flash_next/ple_reference.py`). The two
//! projections run on the CPU here; the Metal group-32 matvec can take
//! them over ([`PleLayer::forward_with_embedding`] takes precomputed
//! `k` / `v` for that).

use rayon::prelude::*;

use super::ngram::{NgramConfig, NgramHasher, NgramHistory, NgramTable, NgramTableError};
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
        let c = &self.cfg;
        let t_n = tokens.len();
        let e = self.embedding(state, tokens, pool)?;
        let mut k = vec![0f32; t_n * c.hc_hidden()];
        let mut v = vec![0f32; t_n * c.hidden];
        matvec_4bit_tokens(w.key_w, w.key_s, w.key_b, c.group_size, c.embed_dim, c.hc_hidden(), &e, &mut k)?;
        matvec_4bit_tokens(w.value_w, w.value_s, w.value_b, c.group_size, c.embed_dim, c.hidden, &e, &mut v)?;
        self.forward_with_embedding(w, state, &k, &v, hidden, out)
    }

    /// The rest of [`Self::forward`] from the projections
    /// `k = key_proj · e` (`T × hc·H`, before `norm_key`) and
    /// `v = value_proj · e` (`T × H`) — for a caller that runs them on
    /// the GPU. Call [`Self::embedding`] first (it advances the token
    /// history); this advances the conv state.
    pub fn forward_with_embedding(
        &self,
        w: &PleWeights,
        state: &mut PleState,
        k: &[f32],
        v: &[f32],
        hidden: &[f32],
        out: &mut [f32],
    ) -> Result<(), PleError> {
        let c = &self.cfg;
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
}
