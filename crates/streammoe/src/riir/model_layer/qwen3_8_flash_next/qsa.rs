//! Qwen Sparse Attention (QSA) — the attention of every 4th layer of
//! Qwen3.8-Flash-Next (`Qwen4ExpTextAttention` +
//! `Qwen4ExpTextQSAIndexer` in transformers `models/qwen4_exp`).
//!
//! ## What it computes
//!
//! Dense part — the same math as the Qwen3.5 / 3.6 full attention in
//! `attn/full_attn_forward.rs`:
//!
//! - `q_proj` gives `2 * heads * head_dim` per token, laid out per head
//!   as `[q_h (head_dim) | gate_h (head_dim)]` (the reference's
//!   `view(.., heads, 2 * head_dim).chunk(2)`), so [`Op::SplitQGate`]
//!   splits it as for Qwen3.5.
//! - per-head `q_norm` / `k_norm` (zero-centred: the multiplier is
//!   `1 + w`), then RoPE on the first `rotary_dim` = 64 channels of
//!   each head, GPT-NeoX half split (`x[i]`, `x[i + 32]`), theta 1e7,
//!   interleaved M-RoPE section `[11, 11, 10]` (pair `i` uses the h
//!   position when `i % 3 == 1 && i < 33`, w when `i % 3 == 2 && i <
//!   30`, else t — [`crate::riir::variants::mrope_axis`]).
//! - GQA SDPA, scale `1 / sqrt(head_dim)`, then `out *= sigmoid(gate)`
//!   and `o_proj`.
//!
//! Sparse part — the indexer. Per token it projects `index_qk_proj`
//! (`(idx_heads + 1) * idx_head_dim` = 640 wide): 4 query heads and one
//! raw key. The raw keys are cached. For each query, the visible
//! tokens (`0..n_visible`, causal) are cut into blocks of
//! `compress_ratio` = 4 consecutive tokens; a complete block's key is
//! `rope(k_layernorm(mean of its 4 raw keys), position of its first
//! token)`. The query (`q_layernorm`, then RoPE at its own position)
//! scores every block as `sum_h relu(q_h · k_b) / sqrt(idx_head_dim)`;
//! the top `budget / compress_ratio` = 512 blocks are kept, plus the
//! tail of an incomplete block, and attention runs over those tokens
//! only.
//!
//! ## When QSA equals dense attention
//!
//! A query with `n_visible` visible tokens has `n_visible /
//! compress_ratio` complete blocks (floor). `topk(min(512, n_blocks))`
//! keeps every block while `n_blocks <= 512`, and the tail is always
//! kept, so every visible token is selected — QSA is exactly dense
//! attention while `n_visible <= budget + compress_ratio - 1` (2051 on
//! the real model; [`QsaShape::dense_is_exact`]). From 2052 visible
//! tokens one or more blocks drop. The reference confirms it
//! bit for bit (`tools/model_layer/qwen3_8_flash_next/qsa_reference.py
//! --check`, and [`tests`] here).
//!
//! The indexer's raw keys must be cached for every token from the
//! start whatever the mode, or a context that grows over the budget
//! later cannot be scored.
//!
//! ## Paths
//!
//! - [`qsa_forward_cpu`]: the whole layer on the CPU with f32 weights
//!   — the reference port the tests compare against transformers.
//! - [`push_qsa_pre_sdpa`] / [`push_qsa_sdpa`] / [`push_qsa_o_proj`]:
//!   the dense layer as existing graph [`Op`]s (CPU and Metal
//!   backends), prefill and decode alike (`n_tokens` = 1 is decode).
//! - [`QsaIndexerCache::append`] + [`qsa_attend_rows`]: the host side
//!   of the indexer and the block-selected attention, run after
//!   [`push_qsa_pre_sdpa`] when [`qsa_needs_sparse`] says the chunk goes
//!   over the budget (CPU only for now).
//!
//! Only the f32 KV format is handled here.

use crate::riir::backend::buftype::{
    AttnInputBuf, AttnOutBuf, KProjOutBuf, KvCacheKBuf, KvCacheVBuf, MatvecOut, OProjOutBuf, QBuf,
    QGateBuf, QProjOutBuf, RopeInvFreqBuf, RopePosBuf, VProjOutBuf,
};
use crate::riir::backend::{BufId, Graph, Op, WeightRef};

unsafe extern "C" {
    fn expf(x: f32) -> f32;
}

/// Attention + indexer shape. Runtime values (not the `VARIANT`
/// constants) so the tests can run a tiny model under any variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QsaShape {
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Channels per head RoPE rotates (`head_dim * partial_rotary_factor`).
    pub rotary_dim: usize,
    pub rope_theta: f32,
    /// Interleaved M-RoPE section over the `rotary_dim / 2` pairs.
    pub mrope_section: [usize; 3],
    pub eps: f32,
    pub idx_heads: usize,
    pub idx_head_dim: usize,
    pub compress_ratio: usize,
    /// Token budget; `budget / compress_ratio` blocks are kept.
    pub budget: usize,
}

impl QsaShape {
    /// Qwen3.8-Flash-Next (`text_config` of the HF config).
    pub const QWEN3_8_FLASH_NEXT: QsaShape = QsaShape {
        hidden: 2560,
        num_heads: 24,
        num_kv_heads: 2,
        head_dim: 256,
        rotary_dim: 64,
        rope_theta: 10_000_000.0,
        mrope_section: [11, 11, 10],
        eps: 1e-6,
        idx_heads: 4,
        idx_head_dim: 128,
        compress_ratio: 4,
        budget: 2048,
    };

    pub const fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
    /// `q_proj` output: query + per-head gate.
    pub const fn q_proj_dim(&self) -> usize {
        2 * self.q_dim()
    }
    pub const fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
    /// `index_qk_proj` output: `idx_heads` query heads + one key.
    pub const fn idx_proj_dim(&self) -> usize {
        (self.idx_heads + 1) * self.idx_head_dim
    }
    pub const fn idx_q_dim(&self) -> usize {
        self.idx_heads * self.idx_head_dim
    }
    /// Blocks kept per query.
    pub const fn block_topk(&self) -> usize {
        self.budget / self.compress_ratio
    }
    /// Whether a query that sees `n_visible` tokens keeps all of them,
    /// i.e. QSA equals dense attention: `n_visible / compress_ratio <=
    /// block_topk`, so up to `budget + compress_ratio - 1` tokens.
    pub const fn dense_is_exact(&self, n_visible: usize) -> bool {
        n_visible / self.compress_ratio <= self.block_topk()
    }
    /// RoPE `inv_freq[i] = 1 / theta^(2i / rotary_dim)`, `rotary_dim / 2`
    /// entries — as `Qwen4ExpTextRotaryEmbedding` and
    /// `FullAttnGraphScratch::new` compute it.
    pub fn inv_freq(&self) -> Vec<f32> {
        let half = self.rotary_dim / 2;
        (0..half)
            .map(|i| 1.0f32 / self.rope_theta.powf((2 * i) as f32 / self.rotary_dim as f32))
            .collect()
    }
}

/// Dense attention everywhere, or the indexer's block selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsaMode {
    /// Attend over every visible token. Exact while
    /// [`QsaShape::dense_is_exact`]; above the budget it is not the
    /// model's attention.
    Dense,
    /// The reference behaviour: block selection once a query sees more
    /// than `budget + compress_ratio - 1` tokens (dense below that).
    Sparse,
}

impl QsaMode {
    /// `STREAMMOE_QSA=dense|sparse` (default sparse), read once.
    pub fn from_env() -> QsaMode {
        static MODE: std::sync::OnceLock<QsaMode> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| match std::env::var("STREAMMOE_QSA").as_deref() {
            Ok("dense") => QsaMode::Dense,
            _ => QsaMode::Sparse,
        })
    }
}

/// Whether a chunk whose last query sees `kv_len_total` tokens needs
/// the block-selected attention: `mode` is [`QsaMode::Sparse`] and the
/// dense SDPA is not exact for it. When `false` the dense graph
/// ([`push_qsa_sdpa`]) gives the model's output.
pub fn qsa_needs_sparse(shape: &QsaShape, mode: QsaMode, kv_len_total: usize) -> bool {
    mode == QsaMode::Sparse && !shape.dense_is_exact(kv_len_total)
}

// =====================================================================
// CPU building blocks
// =====================================================================

/// RMS-normalize each of `heads` `dim`-slices of `x` in place and scale
/// by `w` (the effective `1 + weight` multiplier).
pub fn rms_norm_heads(x: &mut [f32], heads: usize, dim: usize, w: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), heads * dim);
    debug_assert_eq!(w.len(), dim);
    for xh in x.chunks_mut(dim) {
        let mut sum_sq = 0.0f32;
        for &v in xh.iter() {
            sum_sq += v * v;
        }
        let inv = 1.0f32 / (sum_sq / dim as f32 + eps).sqrt();
        for (v, &wi) in xh.iter_mut().zip(w) {
            *v = *v * inv * wi;
        }
    }
}

/// RoPE on the first `rotary_dim` channels of each `head_dim` head of
/// `x`, at the `[t, h, w]` position `pos` with interleaved M-RoPE.
/// Same arithmetic as the CPU backend's `rope_n_tokens_cpu`.
pub fn rope_heads(
    x: &mut [f32],
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &[f32],
    mrope_section: [usize; 3],
    pos: [i32; 3],
) {
    let half = rotary_dim / 2;
    debug_assert_eq!(inv_freq.len(), half);
    for xh in x.chunks_mut(head_dim) {
        for i in 0..half {
            let p = pos[crate::riir::variants::mrope_axis(mrope_section, i)] as f32;
            let angle = p * inv_freq[i];
            let (c, s) = (angle.cos(), angle.sin());
            let x0 = xh[i];
            let x1 = xh[i + half];
            xh[i] = x0 * c - x1 * s;
            xh[i + half] = x0 * s + x1 * c;
        }
    }
}

/// `out[r] = sum_c w[r, c] * x[c]` for a row-major `[rows, x.len()]`.
fn matvec_f32(w: &[f32], x: &[f32], out: &mut [f32]) {
    let cols = x.len();
    debug_assert_eq!(w.len(), out.len() * cols);
    for (o, row) in out.iter_mut().zip(w.chunks(cols)) {
        let mut acc = 0.0f32;
        for (a, b) in row.iter().zip(x) {
            acc += a * b;
        }
        *o = acc;
    }
}

/// Split one token's `q_proj` row (`[q_h | gate_h]` per head) into the
/// query and gate stacks.
pub fn split_q_gate(shape: &QsaShape, q_proj: &[f32], q: &mut [f32], gate: &mut [f32]) {
    let hd = shape.head_dim;
    for h in 0..shape.num_heads {
        q[h * hd..(h + 1) * hd].copy_from_slice(&q_proj[2 * h * hd..(2 * h + 1) * hd]);
        gate[h * hd..(h + 1) * hd].copy_from_slice(&q_proj[(2 * h + 1) * hd..(2 * h + 2) * hd]);
    }
}

// =====================================================================
// Indexer cache
// =====================================================================

/// Per-layer indexer state: the raw key of every cached token, their
/// positions, and the finished key of every complete block.
///
/// Lives next to the layer's KV cache and has the same length. For a
/// snapshot, save `len`, [`Self::raw_keys`] and [`Self::positions`];
/// [`Self::restore`] rebuilds the block keys from them (or save
/// [`Self::block_keys`] too and skip the rebuild — they depend only on
/// the raw keys, positions, `k_layernorm` and RoPE). A rewind is
/// [`Self::truncate`].
#[derive(Debug, Clone)]
pub struct QsaIndexerCache {
    shape: QsaShape,
    /// Effective `k_layernorm` multiplier (`1 + weight`), `idx_head_dim`.
    k_norm: Vec<f32>,
    inv_freq: Vec<f32>,
    /// `[len, idx_head_dim]` raw keys (no norm, no RoPE).
    raw_k: Vec<f32>,
    /// `[t, h, w]` RoPE position of every cached token.
    pos: Vec<[i32; 3]>,
    /// `[len / compress_ratio, idx_head_dim]` block keys: mean of the
    /// block's raw keys, `k_layernorm`, RoPE at its first token.
    block_k: Vec<f32>,
}

impl QsaIndexerCache {
    /// `k_norm` is the effective `k_layernorm` multiplier (`1 + weight`).
    pub fn new(shape: &QsaShape, k_norm: Vec<f32>) -> Self {
        assert_eq!(k_norm.len(), shape.idx_head_dim);
        assert!(shape.rotary_dim <= shape.idx_head_dim);
        Self {
            shape: *shape,
            k_norm,
            inv_freq: shape.inv_freq(),
            raw_k: Vec::new(),
            pos: Vec::new(),
            block_k: Vec::new(),
        }
    }

    /// Cached tokens.
    pub fn len(&self) -> usize {
        self.pos.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }
    /// Complete blocks.
    pub fn n_blocks(&self) -> usize {
        self.block_k.len() / self.shape.idx_head_dim
    }
    pub fn raw_keys(&self) -> &[f32] {
        &self.raw_k
    }
    pub fn positions(&self) -> &[[i32; 3]] {
        &self.pos
    }
    pub fn block_keys(&self) -> &[f32] {
        &self.block_k
    }

    /// Append tokens: `raw_k` is `[n, idx_head_dim]`, `pos` their RoPE
    /// positions. Every block the new tokens complete gets its key.
    pub fn append(&mut self, raw_k: &[f32], pos: &[[i32; 3]]) {
        let d = self.shape.idx_head_dim;
        assert_eq!(raw_k.len(), pos.len() * d);
        self.raw_k.extend_from_slice(raw_k);
        self.pos.extend_from_slice(pos);
        self.finish_blocks();
    }

    fn finish_blocks(&mut self) {
        let (d, cr) = (self.shape.idx_head_dim, self.shape.compress_ratio);
        let mut b = self.n_blocks();
        while (b + 1) * cr <= self.len() {
            // `key_groups.float().mean(dim=1)`.
            let mut key = vec![0.0f32; d];
            for t in b * cr..(b + 1) * cr {
                for (k, &r) in key.iter_mut().zip(&self.raw_k[t * d..(t + 1) * d]) {
                    *k += r;
                }
            }
            for k in key.iter_mut() {
                *k /= cr as f32;
            }
            rms_norm_heads(&mut key, 1, d, &self.k_norm, self.shape.eps);
            rope_heads(
                &mut key,
                d,
                self.shape.rotary_dim,
                &self.inv_freq,
                self.shape.mrope_section,
                self.pos[b * cr],
            );
            self.block_k.extend_from_slice(&key);
            b += 1;
        }
    }

    /// Keep the first `len` tokens (rewind / sequence removal).
    pub fn truncate(&mut self, len: usize) {
        let d = self.shape.idx_head_dim;
        let len = len.min(self.len());
        self.raw_k.truncate(len * d);
        self.pos.truncate(len);
        self.block_k.truncate(len / self.shape.compress_ratio * d);
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Replace the state with saved raw keys and positions; rebuilds
    /// the block keys.
    pub fn restore(&mut self, raw_k: &[f32], pos: &[[i32; 3]]) {
        self.clear();
        self.append(raw_k, pos);
    }

    /// Token indices a query sees under QSA: `q_idx` is its
    /// `[idx_heads, idx_head_dim]` indexer query after `q_layernorm`
    /// and RoPE, `n_visible` the tokens it sees (`0..n_visible`).
    /// `None` when every visible token is kept
    /// ([`QsaShape::dense_is_exact`]); otherwise the kept tokens in
    /// increasing order — the top `block_topk` blocks by score (ties
    /// to the lower block) and the incomplete tail.
    pub fn select(&self, q_idx: &[f32], n_visible: usize) -> Option<Vec<u32>> {
        let s = &self.shape;
        let (d, cr, topk) = (s.idx_head_dim, s.compress_ratio, s.block_topk());
        assert!(n_visible <= self.len(), "query sees {n_visible} tokens, cache holds {}", self.len());
        assert_eq!(q_idx.len(), s.idx_q_dim());
        let n_blocks = n_visible / cr;
        if n_blocks <= topk {
            return None;
        }
        let scale = 1.0f32 / (d as f32).sqrt();
        let mut scored: Vec<(f32, u32)> = (0..n_blocks)
            .map(|b| {
                let kb = &self.block_k[b * d..(b + 1) * d];
                let mut sum = 0.0f32;
                for qh in q_idx.chunks(d) {
                    let mut dot = 0.0f32;
                    for (a, k) in qh.iter().zip(kb) {
                        dot += a * k;
                    }
                    sum += dot.max(0.0);
                }
                (sum * scale, b as u32)
            })
            .collect();
        scored.select_nth_unstable_by(topk - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut blocks: Vec<u32> = scored[..topk].iter().map(|&(_, b)| b).collect();
        blocks.sort_unstable();
        let mut tokens = Vec::with_capacity(topk * cr + cr);
        for b in blocks {
            tokens.extend((b as usize * cr..(b as usize + 1) * cr).map(|t| t as u32));
        }
        tokens.extend((n_blocks * cr..n_visible).map(|t| t as u32));
        Some(tokens)
    }
}

/// The indexer half of a chunk, on the host: split each token's
/// `index_qk_proj` row (`idx_qk`, `[n, idx_proj_dim]`), append the raw
/// keys to `cache`, and return the indexer queries after `q_layernorm`
/// (`q_norm`, effective multiplier) and RoPE at `pos`,
/// `[n, idx_heads * idx_head_dim]`. Run it for every chunk whatever the
/// mode — the raw keys must be cached from the first token.
pub fn qsa_indexer_step(
    shape: &QsaShape,
    idx_qk: &[f32],
    pos: &[[i32; 3]],
    q_norm: &[f32],
    cache: &mut QsaIndexerCache,
) -> Vec<f32> {
    let (pd, qd, d) = (shape.idx_proj_dim(), shape.idx_q_dim(), shape.idx_head_dim);
    let n = pos.len();
    assert_eq!(idx_qk.len(), n * pd);
    let inv_freq = shape.inv_freq();
    let mut q = Vec::with_capacity(n * qd);
    let mut raw = Vec::with_capacity(n * d);
    for (t, row) in idx_qk.chunks(pd).enumerate() {
        let mut qt = row[..qd].to_vec();
        rms_norm_heads(&mut qt, shape.idx_heads, d, q_norm, shape.eps);
        rope_heads(&mut qt, d, shape.rotary_dim, &inv_freq, shape.mrope_section, pos[t]);
        q.extend_from_slice(&qt);
        raw.extend_from_slice(&row[qd..]);
    }
    cache.append(&raw, pos);
    q
}

/// GQA attention of one query over the cached rows `rows` (all of
/// `0..n_visible` when `None`), gated: `out = sigmoid(gate) *
/// softmax(q·k / sqrt(head_dim)) v`. `k_cache` / `v_cache` are
/// `[>= n_visible, kv_dim]` f32.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_rows(
    shape: &QsaShape,
    q: &[f32],
    gate: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_visible: usize,
    rows: Option<&[u32]>,
    out: &mut [f32],
) {
    let (hd, kvd) = (shape.head_dim, shape.kv_dim());
    let hpk = shape.num_heads / shape.num_kv_heads;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let all: Vec<u32>;
    let rows = match rows {
        Some(r) => r,
        None => {
            all = (0..n_visible as u32).collect();
            &all
        }
    };
    let mut scores = vec![0.0f32; rows.len()];
    for h in 0..shape.num_heads {
        let kvh = h / hpk;
        let qh = &q[h * hd..(h + 1) * hd];
        for (s, &r) in scores.iter_mut().zip(rows) {
            let kp = &k_cache[r as usize * kvd + kvh * hd..][..hd];
            let mut dot = 0.0f32;
            for (a, b) in qh.iter().zip(kp) {
                dot += a * b;
            }
            *s = dot * scale;
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            // SAFETY: scalar libm `expf` on a finite f32.
            *s = unsafe { expf(*s - max) };
            sum += *s;
        }
        let inv = 1.0f32 / sum;
        let oh = &mut out[h * hd..(h + 1) * hd];
        oh.fill(0.0);
        for (&s, &r) in scores.iter().zip(rows) {
            let w = s * inv;
            let vp = &v_cache[r as usize * kvd + kvh * hd..][..hd];
            for (o, v) in oh.iter_mut().zip(vp) {
                *o += w * v;
            }
        }
    }
    for (o, &g) in out.iter_mut().zip(gate) {
        // SAFETY: as above.
        *o *= 1.0f32 / (1.0f32 + unsafe { expf(-g) });
    }
}

/// Attention for a chunk of `n` queries whose rows are already in the
/// caches (token `t` is cache row `kv_start + t` and sees rows
/// `0..=kv_start + t`): `q` / `gate` are `[n, q_dim]` (after norm and
/// RoPE), `q_idx` the indexer queries from [`qsa_indexer_step`], and
/// `indexer` already holds the chunk. Writes the gated attention output
/// (the `o_proj` input), `[n, q_dim]`. Queries go over `pool` when
/// given; the result does not depend on it.
#[allow(clippy::too_many_arguments)]
pub fn qsa_attend_rows(
    shape: &QsaShape,
    mode: QsaMode,
    pool: Option<&rayon::ThreadPool>,
    q: &[f32],
    gate: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_start: usize,
    q_idx: &[f32],
    indexer: &QsaIndexerCache,
    out: &mut [f32],
) {
    let (qd, iqd) = (shape.q_dim(), shape.idx_q_dim());
    let n = q.len() / qd;
    assert_eq!(out.len(), n * qd);
    assert!(indexer.len() >= kv_start + n, "indexer cache behind the KV cache");
    let one = |t: usize, ot: &mut [f32]| {
        let n_visible = kv_start + t + 1;
        let rows = match mode {
            QsaMode::Dense => None,
            QsaMode::Sparse => indexer.select(&q_idx[t * iqd..(t + 1) * iqd], n_visible),
        };
        sdpa_rows(
            shape,
            &q[t * qd..(t + 1) * qd],
            &gate[t * qd..(t + 1) * qd],
            k_cache,
            v_cache,
            n_visible,
            rows.as_deref(),
            ot,
        );
    };
    match pool {
        Some(pool) => pool.install(|| {
            use rayon::prelude::*;
            out.par_chunks_mut(qd).enumerate().for_each(|(t, ot)| one(t, ot));
        }),
        None => out.chunks_mut(qd).enumerate().for_each(|(t, ot)| one(t, ot)),
    }
}

// =====================================================================
// Whole layer on the CPU (reference port)
// =====================================================================

/// f32 weights of one QSA layer, row-major `[out, in]`; norms are the
/// effective multipliers (`1 + weight`).
#[derive(Debug, Clone, Copy)]
pub struct QsaWeightsF32<'a> {
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    pub o_proj: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub idx_qk_proj: &'a [f32],
    pub idx_q_norm: &'a [f32],
    pub idx_k_norm: &'a [f32],
}

/// KV + indexer cache of one layer for [`qsa_forward_cpu`].
#[derive(Debug, Clone)]
pub struct QsaCpuCache {
    /// `[len, kv_dim]` keys after `k_norm` and RoPE.
    pub k: Vec<f32>,
    /// `[len, kv_dim]` values.
    pub v: Vec<f32>,
    pub indexer: QsaIndexerCache,
}

impl QsaCpuCache {
    pub fn new(shape: &QsaShape, idx_k_norm: &[f32]) -> Self {
        Self { k: Vec::new(), v: Vec::new(), indexer: QsaIndexerCache::new(shape, idx_k_norm.to_vec()) }
    }
    pub fn len(&self) -> usize {
        self.indexer.len()
    }
    pub fn is_empty(&self) -> bool {
        self.indexer.is_empty()
    }
    /// Keep the first `len` tokens.
    pub fn truncate(&mut self, len: usize) {
        let kvd = self.k.len() / self.len().max(1);
        let len = len.min(self.len());
        self.k.truncate(len * kvd);
        self.v.truncate(len * kvd);
        self.indexer.truncate(len);
    }
}

/// One QSA layer on the CPU for a chunk of `pos.len()` tokens: `x` is
/// the attention input (`[n, hidden]`, after the Gated Residual's
/// norm), `out` gets the `o_proj` output (`[n, hidden]`). Appends the
/// chunk to `cache`. Mirrors `Qwen4ExpTextAttention.forward`.
pub fn qsa_forward_cpu(
    shape: &QsaShape,
    mode: QsaMode,
    w: &QsaWeightsF32<'_>,
    x: &[f32],
    pos: &[[i32; 3]],
    cache: &mut QsaCpuCache,
    out: &mut [f32],
) {
    let proj = |p: QsaProj, x: &[f32], out: &mut [f32]| {
        let m = match p {
            QsaProj::Q => w.q_proj,
            QsaProj::K => w.k_proj,
            QsaProj::V => w.v_proj,
            QsaProj::O => w.o_proj,
            QsaProj::IdxQk => w.idx_qk_proj,
        };
        matvec_f32(m, x, out)
    };
    let norms = QsaNorms { q: w.q_norm, k: w.k_norm, idx_q: w.idx_q_norm };
    qsa_forward_with(shape, mode, &proj, &norms, x, pos, cache, out)
}

/// Which projection [`qsa_forward_with`] asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsaProj {
    /// `q_proj`, `hidden -> q_proj_dim`.
    Q,
    /// `k_proj`, `hidden -> kv_dim`.
    K,
    /// `v_proj`, `hidden -> kv_dim`.
    V,
    /// `o_proj`, `q_dim -> hidden`.
    O,
    /// `index_qk_proj`, `hidden -> idx_proj_dim`.
    IdxQk,
}

/// Effective norm multipliers (`1 + weight`) for [`qsa_forward_with`];
/// the indexer key norm lives in the [`QsaIndexerCache`].
#[derive(Debug, Clone, Copy)]
pub struct QsaNorms<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub idx_q: &'a [f32],
}

/// [`qsa_forward_cpu`] with the projections supplied by `proj(which,
/// x, out)` (any matvec: f32, quantized, GPU).
#[allow(clippy::too_many_arguments)]
pub fn qsa_forward_with(
    shape: &QsaShape,
    mode: QsaMode,
    proj: &dyn Fn(QsaProj, &[f32], &mut [f32]),
    norms: &QsaNorms<'_>,
    x: &[f32],
    pos: &[[i32; 3]],
    cache: &mut QsaCpuCache,
    out: &mut [f32],
) {
    let (hid, qd, kvd, hd) = (shape.hidden, shape.q_dim(), shape.kv_dim(), shape.head_dim);
    let n = pos.len();
    assert_eq!(x.len(), n * hid);
    assert_eq!(out.len(), n * hid);
    let kv_start = cache.len();
    let inv_freq = shape.inv_freq();
    let mut q = vec![0.0f32; n * qd];
    let mut gate = vec![0.0f32; n * qd];
    let mut idx_qk = vec![0.0f32; n * shape.idx_proj_dim()];
    let mut q_proj = vec![0.0f32; shape.q_proj_dim()];
    let mut k = vec![0.0f32; kvd];
    let mut v = vec![0.0f32; kvd];
    for t in 0..n {
        let xt = &x[t * hid..(t + 1) * hid];
        proj(QsaProj::Q, xt, &mut q_proj);
        let (qt, gt) = (&mut q[t * qd..(t + 1) * qd], &mut gate[t * qd..(t + 1) * qd]);
        split_q_gate(shape, &q_proj, qt, gt);
        rms_norm_heads(qt, shape.num_heads, hd, norms.q, shape.eps);
        rope_heads(qt, hd, shape.rotary_dim, &inv_freq, shape.mrope_section, pos[t]);
        proj(QsaProj::K, xt, &mut k);
        rms_norm_heads(&mut k, shape.num_kv_heads, hd, norms.k, shape.eps);
        rope_heads(&mut k, hd, shape.rotary_dim, &inv_freq, shape.mrope_section, pos[t]);
        proj(QsaProj::V, xt, &mut v);
        cache.k.extend_from_slice(&k);
        cache.v.extend_from_slice(&v);
        proj(QsaProj::IdxQk, xt, &mut idx_qk[t * shape.idx_proj_dim()..(t + 1) * shape.idx_proj_dim()]);
    }
    let q_idx = qsa_indexer_step(shape, &idx_qk, pos, norms.idx_q, &mut cache.indexer);
    let mut attn = vec![0.0f32; n * qd];
    qsa_attend_rows(shape, mode, None, &q, &gate, &cache.k, &cache.v, kv_start, &q_idx, &cache.indexer, &mut attn);
    for t in 0..n {
        proj(QsaProj::O, &attn[t * qd..(t + 1) * qd], &mut out[t * hid..(t + 1) * hid]);
    }
}

// =====================================================================
// QsaBlock: every QSA layer of a model, for q38-core's `AttnBlock`
// =====================================================================

/// A matvec, the shape of q38-core's `Linear` trait
/// (`gated_residual::Linear`): `out[r] = sum_c W[r, c] x[c]`. A blanket
/// `impl<L: Linear> QsaLinear for L` adapts it at integration.
pub trait QsaLinear {
    fn in_dim(&self) -> usize;
    fn out_dim(&self) -> usize;
    fn forward(&self, x: &[f32], out: &mut [f32]);
}

/// Weights of one QSA layer for [`QsaBlock`]. The norm gains are the
/// checkpoint's zero-centred values (applied as `1 + w`), as the MLX
/// build stores them.
#[derive(Debug, Clone)]
pub struct QsaLayer<L> {
    pub q_proj: L,
    pub k_proj: L,
    pub v_proj: L,
    pub o_proj: L,
    pub idx_qk_proj: L,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub idx_q_norm: Vec<f32>,
    pub idx_k_norm: Vec<f32>,
}

struct QsaBlockLayer<L> {
    w: QsaLayer<L>,
    /// Effective `1 + w` multipliers of `q_norm`, `k_norm`, `idx_q_norm`.
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    idx_q_norm: Vec<f32>,
    cache: QsaCpuCache,
}

/// Errors from [`QsaBlock`].
#[derive(Debug, thiserror::Error)]
pub enum QsaError {
    #[error("layer {0} has no QSA weights")]
    NoLayer(usize),
    #[error("layer {layer}: cache holds {len} tokens, got position {pos}")]
    Position { layer: usize, len: usize, pos: usize },
    #[error("layer {layer}: {what} is {got} wide, want {want}")]
    Shape { layer: usize, what: &'static str, got: usize, want: usize },
}

/// Every QSA layer of a model on the CPU, with its KV and indexer
/// caches: the block q38-core's `layer.rs` calls through `AttnBlock`
/// (one token at a time, text positions) and the chunked prefill entry
/// [`Self::forward_chunk`] (any `[t, h, w]` positions).
///
/// Integration (in `layer.rs`, where `AttnBlock`, `Linear` and
/// `Q38Error` live):
///
/// ```ignore
/// impl<L: Linear> QsaLinear for L { /* in_dim, out_dim, forward → Linear's */ }
/// impl<L: Linear> AttnBlock for QsaBlock<L> {
///     fn forward(&mut self, layer: usize, pos: usize, x: &[f32], out: &mut [f32]) -> Result<(), Q38Error> {
///         QsaBlock::forward_checked(self, layer, pos, x, out)
///             .map_err(|e| Q38Error::Hook { layer, msg: e.to_string() })
///     }
///     fn reset(&mut self) { QsaBlock::reset(self) }
/// }
/// ```
pub struct QsaBlock<L> {
    shape: QsaShape,
    mode: QsaMode,
    layers: Vec<Option<QsaBlockLayer<L>>>,
}

impl<L: QsaLinear> QsaBlock<L> {
    /// `layers[i]` is `Some` on the QSA layers (every 4th).
    pub fn new(shape: &QsaShape, mode: QsaMode, layers: Vec<Option<QsaLayer<L>>>) -> Result<Self, QsaError> {
        let one_plus = |w: &[f32]| w.iter().map(|v| 1.0 + v).collect::<Vec<f32>>();
        let mut out = Vec::with_capacity(layers.len());
        for (i, l) in layers.into_iter().enumerate() {
            let Some(w) = l else {
                out.push(None);
                continue;
            };
            let s = shape;
            for (what, got, want) in [
                ("q_proj out", w.q_proj.out_dim(), s.q_proj_dim()),
                ("q_proj in", w.q_proj.in_dim(), s.hidden),
                ("k_proj out", w.k_proj.out_dim(), s.kv_dim()),
                ("v_proj out", w.v_proj.out_dim(), s.kv_dim()),
                ("o_proj in", w.o_proj.in_dim(), s.q_dim()),
                ("o_proj out", w.o_proj.out_dim(), s.hidden),
                ("idx_qk_proj out", w.idx_qk_proj.out_dim(), s.idx_proj_dim()),
                ("q_norm", w.q_norm.len(), s.head_dim),
                ("k_norm", w.k_norm.len(), s.head_dim),
                ("idx_q_norm", w.idx_q_norm.len(), s.idx_head_dim),
                ("idx_k_norm", w.idx_k_norm.len(), s.idx_head_dim),
            ] {
                if got != want {
                    return Err(QsaError::Shape { layer: i, what, got, want });
                }
            }
            out.push(Some(QsaBlockLayer {
                q_norm: one_plus(&w.q_norm),
                k_norm: one_plus(&w.k_norm),
                idx_q_norm: one_plus(&w.idx_q_norm),
                cache: QsaCpuCache::new(shape, &one_plus(&w.idx_k_norm)),
                w,
            }));
        }
        Ok(Self { shape: *shape, mode, layers: out })
    }

    pub fn shape(&self) -> &QsaShape {
        &self.shape
    }
    pub fn mode(&self) -> QsaMode {
        self.mode
    }
    pub fn set_mode(&mut self, mode: QsaMode) {
        self.mode = mode;
    }

    /// One text token at absolute position `pos` (M-RoPE `t = h = w`).
    pub fn forward(&mut self, layer: usize, pos: usize, x: &[f32], out: &mut [f32]) -> Result<(), QsaError> {
        let p = pos as i32;
        self.forward_chunk(layer, &[[p, p, p]], x, out)
    }

    /// A chunk of `pos.len()` tokens (`x`, `out`: `[n, hidden]`) that
    /// continue the layer's cache; `pos` are their `[t, h, w]` RoPE
    /// positions. The cache length is the next token's index, so
    /// positions may run ahead of it (images).
    pub fn forward_chunk(&mut self, layer: usize, pos: &[[i32; 3]], x: &[f32], out: &mut [f32]) -> Result<(), QsaError> {
        let (shape, mode) = (self.shape, self.mode);
        let l = self.layers.get_mut(layer).and_then(|l| l.as_mut()).ok_or(QsaError::NoLayer(layer))?;
        let n = pos.len();
        for (what, got, want) in [("x", x.len(), n * shape.hidden), ("out", out.len(), n * shape.hidden)] {
            if got != want {
                return Err(QsaError::Shape { layer, what, got, want });
            }
        }
        let w = &l.w;
        let proj = |p: QsaProj, x: &[f32], out: &mut [f32]| match p {
            QsaProj::Q => w.q_proj.forward(x, out),
            QsaProj::K => w.k_proj.forward(x, out),
            QsaProj::V => w.v_proj.forward(x, out),
            QsaProj::O => w.o_proj.forward(x, out),
            QsaProj::IdxQk => w.idx_qk_proj.forward(x, out),
        };
        let norms = QsaNorms { q: &l.q_norm, k: &l.k_norm, idx_q: &l.idx_q_norm };
        qsa_forward_with(&shape, mode, &proj, &norms, x, pos, &mut l.cache, out);
        Ok(())
    }

    /// [`Self::forward`] with the cache-length check q38-core expects:
    /// a text token at `pos` must be the cache's next row.
    pub fn forward_checked(&mut self, layer: usize, pos: usize, x: &[f32], out: &mut [f32]) -> Result<(), QsaError> {
        let len = self.cache(layer).map(|c| c.len()).ok_or(QsaError::NoLayer(layer))?;
        if len != pos {
            return Err(QsaError::Position { layer, len, pos });
        }
        self.forward(layer, pos, x, out)
    }

    /// The layer's KV + indexer cache (for snapshots).
    pub fn cache(&self, layer: usize) -> Option<&QsaCpuCache> {
        self.layers.get(layer)?.as_ref().map(|l| &l.cache)
    }
    pub fn cache_mut(&mut self, layer: usize) -> Option<&mut QsaCpuCache> {
        self.layers.get_mut(layer)?.as_mut().map(|l| &mut l.cache)
    }

    /// Keep the first `len` tokens of every layer (rewind).
    pub fn truncate(&mut self, len: usize) {
        for l in self.layers.iter_mut().flatten() {
            l.cache.truncate(len);
        }
    }

    /// Drop all cached context (new sequence).
    pub fn reset(&mut self) {
        self.truncate(0);
    }
}

// =====================================================================
// Graph ops (CPU and Metal backends)
// =====================================================================

/// Weight-file references of one QSA layer for the graph path.
/// `q_norm_off` / `k_norm_off` are bf16 `[head_dim]` effective
/// multipliers (`1 + weight`, as the existing per-head norm reads
/// them). The indexer norms run on the host
/// ([`qsa_indexer_step`], [`QsaIndexerCache::new`]).
#[derive(Debug, Clone, Copy)]
pub struct QsaLayerWeights {
    pub q_proj: WeightRef,
    pub k_proj: WeightRef,
    pub v_proj: WeightRef,
    pub o_proj: WeightRef,
    pub idx_qk_proj: WeightRef,
    pub q_norm_off: u64,
    pub k_norm_off: u64,
}

/// Buffers of one QSA layer for the graph path. All `[chunk, dim]`
/// f32 at the prefill-chunk width except the caches (`[n_ctx,
/// kv_dim]`) and `inv_freq` ([`QsaShape::inv_freq`]).
#[derive(Debug, Clone, Copy)]
pub struct QsaGraphBufs {
    /// Attention input `[n, hidden]` (the Gated Residual's output).
    pub input: BufId<AttnInputBuf>,
    /// `[n, q_proj_dim]`.
    pub q_proj: BufId<QProjOutBuf>,
    /// `[n, kv_dim]`, normed and rotated in place, then appended.
    pub k: BufId<KProjOutBuf>,
    pub v: BufId<VProjOutBuf>,
    /// `[n, idx_proj_dim]` indexer projection, read back by the host.
    pub idx_qk: BufId<KProjOutBuf>,
    /// `[n, q_dim]` query after norm + RoPE.
    pub q: BufId<QBuf>,
    pub gate: BufId<QGateBuf>,
    /// `[n, q_dim]` gated attention output (the `o_proj` input).
    pub attn_out: BufId<AttnOutBuf>,
    /// `[n, hidden]` `o_proj` output.
    pub o_out: BufId<OProjOutBuf>,
    pub inv_freq: BufId<RopeInvFreqBuf>,
    /// `[n, 3]` i32 `[t, h, w]` positions for a multimodal chunk;
    /// `None` rotates token `t` by `start_pos + t`. The backends take
    /// the M-RoPE section from `variants::MROPE_SECTION`, which must be
    /// `[11, 11, 10]` for this family.
    pub positions: Option<BufId<RopePosBuf>>,
    pub k_cache: BufId<KvCacheKBuf>,
    pub v_cache: BufId<KvCacheVBuf>,
}

/// Projections, q / gate split, per-head norms, RoPE and the KV append
/// for `n` tokens at cache rows `kv_start..kv_start + n` (RoPE
/// positions `start_pos + t` unless `bufs.positions`). After it runs,
/// `bufs.q` / `bufs.gate` hold the queries, the caches hold the chunk
/// and `bufs.idx_qk` the indexer projection for
/// [`qsa_indexer_step`].
pub fn push_qsa_pre_sdpa(
    g: &mut Graph,
    shape: &QsaShape,
    w: &QsaLayerWeights,
    bufs: &QsaGraphBufs,
    n_tokens: usize,
    kv_start: usize,
    start_pos: i32,
) {
    let n = n_tokens as u32;
    let hid = shape.hidden as u32;
    let (nh, nkv, hd) = (shape.num_heads as u32, shape.num_kv_heads as u32, shape.head_dim as u32);
    let mut proj = |label, weight, output: BufId<MatvecOut>, out_dim: usize| {
        g.push(Op::MatvecNTokens {
            label,
            weight,
            input: bufs.input.into(),
            input_off: 0,
            output,
            output_off: 0,
            in_dim: hid,
            out_dim: out_dim as u32,
            n_tokens: n,
        });
    };
    proj("qsa.q_proj", w.q_proj, bufs.q_proj.into(), shape.q_proj_dim());
    proj("qsa.k_proj", w.k_proj, bufs.k.into(), shape.kv_dim());
    proj("qsa.v_proj", w.v_proj, bufs.v.into(), shape.kv_dim());
    proj("qsa.idx_qk_proj", w.idx_qk_proj, bufs.idx_qk.into(), shape.idx_proj_dim());
    g.push(Op::SplitQGate {
        label: "qsa.split_q_gate",
        q_proj: bufs.q_proj,
        q_out: bufs.q,
        gate_out: bufs.gate,
        num_heads: nh,
        head_dim: hd,
        n_tokens: n,
    });
    for (label, x, off, heads) in [
        ("qsa.q_norm", bufs.q.into(), w.q_norm_off, nh),
        ("qsa.k_norm", bufs.k.into(), w.k_norm_off, nkv),
    ] {
        g.push(Op::RmsNormPerHeadNTokens {
            label,
            x,
            weight_off: off,
            num_heads: heads,
            head_dim: hd,
            n_tokens: n,
            eps: shape.eps,
        });
    }
    for (label, x, heads) in [("qsa.q_rope", bufs.q.into(), nh), ("qsa.k_rope", bufs.k.into(), nkv)] {
        g.push(Op::RopeNTokens {
            label,
            x,
            inv_freq: bufs.inv_freq,
            positions: bufs.positions,
            n_tokens: n,
            num_heads: heads,
            head_dim: hd,
            rotary_dim: shape.rotary_dim as u32,
            start_pos,
        });
    }
    g.push(Op::KvCacheAppendNTokens {
        label: "qsa.kv_append",
        k_src: bufs.k,
        v_src: bufs.v,
        k_cache: bufs.k_cache,
        v_cache: bufs.v_cache,
        kv_dim: shape.kv_dim() as u32,
        n_tokens: n,
        kv_start: kv_start as u32,
    });
}

/// Dense causal SDPA over the cache and the sigmoid gate, into
/// `bufs.attn_out`. Exact (the model's attention) when
/// [`qsa_needs_sparse`] is false for `kv_start + n_tokens`; otherwise
/// fill `bufs.attn_out` with [`qsa_attend_rows`] instead.
pub fn push_qsa_sdpa(g: &mut Graph, shape: &QsaShape, bufs: &QsaGraphBufs, n_tokens: usize, kv_start: usize) {
    let n = n_tokens as u32;
    g.push(Op::SdpaCausalTiled {
        label: "qsa.sdpa",
        q: bufs.q,
        k: bufs.k_cache,
        v: bufs.v_cache,
        attn_out: bufs.attn_out,
        n_tokens: n,
        num_heads: shape.num_heads as u32,
        heads_per_kv: (shape.num_heads / shape.num_kv_heads) as u32,
        head_dim: shape.head_dim as u32,
        kv_dim: shape.kv_dim() as u32,
        kv_start: kv_start as u32,
        kv_len_total: (kv_start + n_tokens) as u32,
        softmax_scale: 1.0f32 / (shape.head_dim as f32).sqrt(),
    });
    g.push(Op::SigmoidGateNTokens {
        label: "qsa.sigmoid_gate",
        x: bufs.attn_out,
        gate: bufs.gate,
        dim: shape.q_dim() as u32,
        n_tokens: n,
    });
}

/// `o_proj` of the gated attention output into `bufs.o_out`.
pub fn push_qsa_o_proj(g: &mut Graph, shape: &QsaShape, w: &QsaLayerWeights, bufs: &QsaGraphBufs, n_tokens: usize) {
    g.push(Op::MatvecNTokens {
        label: "qsa.o_proj",
        weight: w.o_proj,
        input: bufs.attn_out.into(),
        input_off: 0,
        output: bufs.o_out.into(),
        output_off: 0,
        in_dim: shape.q_dim() as u32,
        out_dim: shape.hidden as u32,
        n_tokens: n_tokens as u32,
    });
}

#[cfg(test)]
mod tests;
