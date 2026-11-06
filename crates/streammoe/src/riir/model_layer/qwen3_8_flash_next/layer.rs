//! Decoder-layer orchestration for Qwen3.8-Flash-Next, CPU.
//!
//! `Qwen4ExpTextDecoderLayer.forward`, one token:
//!
//! ```text
//! if PLE layer:  hyper += ple(hyper, token)                 // [hc · H]
//! x, inj = attn_hyper_connection(hyper)                     // gated residual
//! y      = linear_attn(x)  |  self_attn(x)  (QSA)           // [H]
//! hyper += y ⊗ inj
//! x, inj = mlp_hyper_connection(hyper)
//! y      = moe(x)                                           // routed + shared
//! hyper += y ⊗ inj
//! ```
//!
//! There are no input / post-attention RMSNorms: the Gated Residual's
//! `hc_norm` replaces them, and the mixed `x` goes straight into the
//! block. The CPU primitives this reuses take an already-normalized
//! input, so nothing needs bypassing on this path
//! ([`gated_delta_recurrence_supplied`], [`moe_router_cpu_with`]); see
//! the module docs of [`super`] for the GPU paths, which fuse the norm.
//!
//! The QSA block and PLE belong to other modules and plug in through
//! [`AttnBlock`] and [`PleHook`]. [`DenseAttention`] is the QSA block
//! with every KV block selected — exact while the context is within
//! the indexer budget (2,048 tokens) — and serves the tiny tests.

use super::config::{GateAct, Q38Config, Q38LayerKind};
use super::gated_residual::{
    gated_residual_mix, hc_expand, inject, rms_norm_grouped, GatedResidualScratch, GatedResidualWeights,
    Linear, NormGain,
};
use crate::riir::attn::linear_attn::gated_delta_recurrence_supplied;
use crate::riir::moe::moe_router::moe_router_cpu_with;

/// Errors from the family's CPU path.
#[derive(Debug, thiserror::Error)]
pub enum Q38Error {
    #[error("shape: {0}")]
    Shape(String),
    #[error("linear attention: {0}")]
    LinearAttn(#[from] crate::riir::attn::linear_attn::LinearAttnError),
    #[error("router: {0}")]
    Router(#[from] crate::riir::moe::moe_router::MoeRouterError),
    #[error("layer {layer}: {msg}")]
    Hook { layer: usize, msg: String },
}

// --- Call points for the other modules ------------------------------

/// The attention block of the `Qsa` layers (Qwen Sparse Attention).
/// Implemented by the QSA module; [`DenseAttention`] is the dense
/// stand-in.
///
/// `x` is the block input — the attention Gated Residual's mixed
/// output, **not** normalized by any input layernorm (this family has
/// none). `out` receives the block output after `o_proj` (`[H]`); the
/// caller injects it into the stream. The block owns its KV cache and
/// indexer state for every QSA layer; `pos` is the token's absolute
/// position (a text token's MRoPE `t = h = w = pos`). Calls come in
/// order of `pos`, layer by layer.
pub trait AttnBlock {
    fn forward(&mut self, layer: usize, pos: usize, x: &[f32], out: &mut [f32]) -> Result<(), Q38Error>;
    /// Drop all cached context (new sequence).
    fn reset(&mut self);
}

/// Per-layer embedding (n-gram PLE). Implemented by the PLE module.
///
/// Called at the top of each layer in [`Q38Config::ple_layers`]
/// (0-based; index 1 for the real model), before the attention Gated
/// Residual, with the residual stream `hyper` (`[hc · H]`) as the
/// query. Writes the addition to `out` (`[hc · H]`, overwritten); the
/// caller adds it to `hyper`. `token` is this position's input id; the
/// hook keeps its own n-gram history and its dilated short-conv state,
/// so calls come in order of `pos`.
pub trait PleHook {
    fn forward(&mut self, layer: usize, pos: usize, token: u32, hyper: &[f32], out: &mut [f32])
        -> Result<(), Q38Error>;
    /// Drop the n-gram history and conv state (new sequence).
    fn reset(&mut self);
}

/// No PLE: errors if a PLE layer is reached. For configs with
/// `ple_layers = []`.
pub struct NoPle;

impl PleHook for NoPle {
    fn forward(&mut self, layer: usize, _: usize, _: u32, _: &[f32], _: &mut [f32]) -> Result<(), Q38Error> {
        Err(Q38Error::Hook {
            layer,
            msg: "PLE layer reached without a PleHook".into(),
        })
    }
    fn reset(&mut self) {}
}

/// The routed experts of one MoE layer. The engine streams them; the
/// tiny tests hold them in memory.
pub trait ExpertSet {
    /// `out = down(silu(gate(x)) ⊙ up(x))` for expert `e` (`[H]`).
    fn expert_forward(&self, e: usize, x: &[f32], out: &mut [f32]);
}

// --- Weights ---------------------------------------------------------

/// One Gated Residual's weights, owned.
#[derive(Debug, Clone)]
pub struct HcWeights<L: Linear> {
    /// Raw gain `[hc · H]` (applied as `1 + w`).
    pub hc_norm: Vec<f32>,
    pub mix_down: L,
    pub mix_up: L,
    pub block_inject: Option<L>,
}

impl<L: Linear> HcWeights<L> {
    pub fn weights(&self) -> GatedResidualWeights<'_, L> {
        GatedResidualWeights {
            hc_norm: &self.hc_norm,
            mix_down: &self.mix_down,
            mix_up: &self.mix_up,
            block_inject: self.block_inject.as_ref(),
        }
    }
}

/// Gated DeltaNet weights (`linear_attn.*`).
#[derive(Debug, Clone)]
pub struct GdnWeights<L: Linear> {
    pub in_qkv: L,
    pub in_z: L,
    pub in_b: L,
    pub in_a: L,
    /// Depthwise conv, `[conv_dim, kernel]` (channel, time); time
    /// `kernel - 1` multiplies the newest input.
    pub conv1d: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// Plain gain `[value_dim]`, shared by the value heads.
    pub norm: Vec<f32>,
    pub out_proj: L,
}

/// Gated DeltaNet recurrent state of one layer.
#[derive(Debug, Clone)]
pub struct GdnState {
    /// `[(kernel - 1) · conv_dim]`, time-major (oldest first) — the
    /// engine's `LinearAttnState` conv layout.
    pub conv: Vec<f32>,
    /// `[v_heads, value_dim, key_dim]` — the engine's `ssm_state`
    /// layout (the reference keeps `[v_heads, key_dim, value_dim]`).
    pub ssm: Vec<f32>,
}

impl GdnState {
    pub fn new(cfg: &Q38Config) -> Self {
        Self {
            conv: vec![0.0; (cfg.conv_kernel - 1) * cfg.linear_conv_dim()],
            ssm: vec![0.0; cfg.linear_v_heads * cfg.linear_value_dim * cfg.linear_key_dim],
        }
    }
    pub fn reset(&mut self) {
        self.conv.fill(0.0);
        self.ssm.fill(0.0);
    }
}

/// SwiGLU MLP (`shared_expert`, or one routed expert).
#[derive(Debug, Clone)]
pub struct MlpWeights<L: Linear> {
    pub gate: L,
    pub up: L,
    pub down: L,
}

impl<L: Linear> MlpWeights<L> {
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        let n = self.gate.out_dim();
        let mut g = vec![0.0; n];
        let mut u = vec![0.0; n];
        self.gate.forward(x, &mut g);
        self.up.forward(x, &mut u);
        for (gv, uv) in g.iter_mut().zip(&u) {
            *gv = *gv / (1.0 + (-*gv).exp()) * uv;
        }
        self.down.forward(&g, out);
    }
}

impl<L: Linear> ExpertSet for Vec<MlpWeights<L>> {
    fn expert_forward(&self, e: usize, x: &[f32], out: &mut [f32]) {
        self[e].forward(x, out);
    }
}

/// MoE weights of one layer (`mlp.*`).
#[derive(Debug, Clone)]
pub struct MoeWeights<L: Linear, E: ExpertSet> {
    /// `mlp.gate`: `[num_experts, H]`.
    pub router: L,
    pub experts: E,
    pub shared: MlpWeights<L>,
    /// `mlp.shared_expert_gate`: `[1, H]`, applied as a sigmoid.
    pub shared_gate: L,
}

/// One decoder layer. The QSA layers' attention weights live in the
/// [`AttnBlock`]; PLE's in the [`PleHook`].
#[derive(Debug, Clone)]
pub struct LayerWeights<L: Linear, E: ExpertSet> {
    pub attn_hc: HcWeights<L>,
    pub mlp_hc: HcWeights<L>,
    /// `Some` exactly on linear-attention layers.
    pub gdn: Option<GdnWeights<L>>,
    pub moe: MoeWeights<L, E>,
}

// --- Blocks ----------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `x / sqrt(Σx² + eps)` — the FLA-style `l2norm` the reference
/// applies to Q and K (`use_qk_l2norm_in_kernel`).
fn l2norm(x: &mut [f32], eps: f32) {
    let mut s = 0.0f32;
    for &v in x.iter() {
        s += v * v;
    }
    let inv = 1.0 / (s + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `Qwen4ExpTextGatedDeltaNet` for one cached token
/// (`causal_conv1d_update` + `torch_recurrent_gated_delta_rule`).
/// `x` is the mixed block input; writes `out` (`[H]`).
pub fn gdn_step<L: Linear>(cfg: &Q38Config, w: &GdnWeights<L>, st: &mut GdnState, x: &[f32], out: &mut [f32])
    -> Result<(), Q38Error> {
    let (hk, hv) = (cfg.linear_k_heads, cfg.linear_v_heads);
    let (dk, dv) = (cfg.linear_key_dim, cfg.linear_value_dim);
    let c = cfg.linear_conv_dim();
    let kk = cfg.conv_kernel;
    let mut qkv = vec![0.0; c];
    let mut z = vec![0.0; hv * dv];
    let mut b = vec![0.0; hv];
    let mut a = vec![0.0; hv];
    w.in_qkv.forward(x, &mut qkv);
    w.in_z.forward(x, &mut z);
    w.in_b.forward(x, &mut b);
    w.in_a.forward(x, &mut a);

    // Depthwise causal conv over [state | new], then SiLU; shift state.
    let mut conv_out = vec![0.0; c];
    for ch in 0..c {
        let mut acc = 0.0f32;
        for k in 0..kk - 1 {
            acc = st.conv[k * c + ch].mul_add(w.conv1d[ch * kk + k], acc);
        }
        acc = qkv[ch].mul_add(w.conv1d[ch * kk + kk - 1], acc);
        conv_out[ch] = acc / (1.0 + (-acc).exp());
    }
    if kk > 1 {
        st.conv.copy_within(c.., 0);
        let last = (kk - 2) * c;
        st.conv[last..last + c].copy_from_slice(&qkv);
    }

    let key_total = hk * dk;
    let (q, rest) = conv_out.split_at_mut(key_total);
    let (k, v) = rest.split_at_mut(key_total);
    let q_scale = 1.0 / (dk as f32).sqrt();
    for h in 0..hk {
        l2norm(&mut q[h * dk..(h + 1) * dk], 1e-6);
        l2norm(&mut k[h * dk..(h + 1) * dk], 1e-6);
        for qv in &mut q[h * dk..(h + 1) * dk] {
            *qv *= q_scale;
        }
    }
    // g = -exp(A_log) · softplus(a + dt_bias); decay = exp(g).
    let mut decay = vec![0.0; hv];
    let mut beta = vec![0.0; hv];
    for h in 0..hv {
        let sp = (1.0f32 + (a[h] + w.dt_bias[h]).exp()).ln();
        decay[h] = (-w.a_log[h].exp() * sp).exp();
        beta[h] = sigmoid(b[h]);
    }
    let mut core = vec![0.0; hv * dv];
    // v-head h reads k-head h / (hv / hk): the reference's
    // `repeat_interleave`.
    gated_delta_recurrence_supplied(&decay, &beta, q, k, v, hv, hk, dk, dv, &mut st.ssm, &mut core)?;

    // Gated RMSNorm per value head: x̂ · w · act(z).
    for h in 0..hv {
        let o = &mut core[h * dv..(h + 1) * dv];
        let mut s = 0.0f32;
        for &vv in o.iter() {
            s += vv * vv;
        }
        let inv = 1.0 / (s / dv as f32 + cfg.rms_eps).sqrt();
        for i in 0..dv {
            let zi = z[h * dv + i];
            let gate = match cfg.gdn_gate {
                GateAct::Sigmoid => sigmoid(zi),
                GateAct::Silu => zi * sigmoid(zi),
            };
            o[i] = o[i] * inv * w.norm[i] * gate;
        }
    }
    w.out_proj.forward(&core, out);
    Ok(())
}

/// `Qwen4ExpTextSparseMoeBlock` for one token: softmax router, top-K
/// (renormalized when `norm_topk_prob`), routed experts, plus
/// `sigmoid(shared_expert_gate(x)) · shared_expert(x)`. Returns the
/// chosen expert ids (router order).
pub fn moe_forward<L: Linear, E: ExpertSet>(cfg: &Q38Config, w: &MoeWeights<L, E>, x: &[f32], out: &mut [f32])
    -> Result<Vec<i32>, Q38Error> {
    let h = cfg.hidden;
    let mut scores = vec![0.0; cfg.num_experts];
    w.router.forward(x, &mut scores);
    let mut idx = vec![0i32; cfg.top_k];
    let mut wts = vec![0.0f32; cfg.top_k];
    moe_router_cpu_with(&mut scores, cfg.top_k, cfg.norm_topk_prob, &mut idx, &mut wts)?;
    out.fill(0.0);
    let mut e_out = vec![0.0; h];
    for (&e, &wt) in idx.iter().zip(&wts) {
        w.experts.expert_forward(e as usize, x, &mut e_out);
        for (o, &y) in out.iter_mut().zip(&e_out) {
            *o = y.mul_add(wt, *o);
        }
    }
    w.shared.forward(x, &mut e_out);
    let mut sg = [0.0f32];
    w.shared_gate.forward(x, &mut sg);
    let g = sigmoid(sg[0]);
    for (o, &y) in out.iter_mut().zip(&e_out) {
        *o = y.mul_add(g, *o);
    }
    Ok(idx)
}

/// Per-layer state of the CPU path.
#[derive(Debug, Clone)]
pub enum LayerState {
    LinearAttn(GdnState),
    /// QSA layers keep their state in the [`AttnBlock`].
    Qsa,
}

/// Reusable buffers for [`decoder_layer_forward`].
#[derive(Debug, Clone)]
pub struct LayerScratch {
    pub gr: GatedResidualScratch,
    pub mixed: Vec<f32>,
    pub inj: Vec<f32>,
    pub block_out: Vec<f32>,
    pub ple_out: Vec<f32>,
}

impl LayerScratch {
    pub fn new(cfg: &Q38Config) -> Self {
        Self {
            gr: GatedResidualScratch::new(cfg.hc_hidden(), cfg.hc_lowrank),
            mixed: vec![0.0; cfg.hidden],
            inj: vec![0.0; cfg.hc_count],
            block_out: vec![0.0; cfg.hidden],
            ple_out: vec![0.0; cfg.hc_hidden()],
        }
    }
}

/// `Qwen4ExpTextDecoderLayer.forward` for one token. Updates `hyper`
/// (`[hc · H]`) in place; returns the MoE's chosen experts.
#[allow(clippy::too_many_arguments)]
pub fn decoder_layer_forward<L, E, A, P>(
    cfg: &Q38Config,
    layer: usize,
    w: &LayerWeights<L, E>,
    state: &mut LayerState,
    attn: &mut A,
    ple: &mut P,
    pos: usize,
    token: u32,
    hyper: &mut [f32],
    s: &mut LayerScratch,
) -> Result<Vec<i32>, Q38Error>
where
    L: Linear,
    E: ExpertSet,
    A: AttnBlock + ?Sized,
    P: PleHook + ?Sized,
{
    let (h, hc) = (cfg.hidden, cfg.hc_count);
    if hyper.len() != cfg.hc_hidden() {
        return Err(Q38Error::Shape(format!("hyper is {} wide, want {}", hyper.len(), cfg.hc_hidden())));
    }
    if cfg.has_ple(layer) {
        ple.forward(layer, pos, token, hyper, &mut s.ple_out)?;
        for (x, &p) in hyper.iter_mut().zip(&s.ple_out) {
            *x += p;
        }
    }

    gated_residual_mix(&w.attn_hc.weights(), h, hc, cfg.rms_eps, hyper, &mut s.gr, &mut s.mixed, Some(&mut s.inj));
    match (cfg.layer_kind(layer), state) {
        (Q38LayerKind::LinearAttn, LayerState::LinearAttn(st)) => {
            let gw = w.gdn.as_ref().ok_or_else(|| Q38Error::Shape(format!("layer {layer}: no GDN weights")))?;
            gdn_step(cfg, gw, st, &s.mixed, &mut s.block_out)?;
        }
        (Q38LayerKind::Qsa, LayerState::Qsa) => attn.forward(layer, pos, &s.mixed, &mut s.block_out)?,
        _ => return Err(Q38Error::Shape(format!("layer {layer}: state does not match the layer kind"))),
    }
    inject(hyper, &s.block_out, &s.inj);

    gated_residual_mix(&w.mlp_hc.weights(), h, hc, cfg.rms_eps, hyper, &mut s.gr, &mut s.mixed, Some(&mut s.inj));
    let routes = moe_forward(cfg, &w.moe, &s.mixed, &mut s.block_out)?;
    inject(hyper, &s.block_out, &s.inj);
    Ok(routes)
}

// --- Dense attention (QSA with every block selected) ------------------

/// Attention weights of one QSA layer, without the indexer.
#[derive(Debug, Clone)]
pub struct AttnWeights<L: Linear> {
    /// `[2 · heads · head_dim, H]`, per head `[q | gate]`.
    pub q_proj: L,
    pub k_proj: L,
    pub v_proj: L,
    pub o_proj: L,
    /// Raw gains `[head_dim]` (applied as `1 + w`).
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// `Qwen4ExpTextAttention` with every KV block selected: gated q_proj,
/// per-head Q/K norms (`1 + w`), partial NeoX RoPE on the first
/// `rotary_dim` channels, GQA softmax attention, sigmoid output gate,
/// `o_proj`. Equals QSA while the context is within the indexer
/// budget. Text positions only (MRoPE `t = h = w`).
pub struct DenseAttention<L: Linear> {
    cfg: Q38Config,
    /// Indexed by layer; `Some` on QSA layers.
    pub layers: Vec<Option<AttnWeights<L>>>,
    /// Per layer: K and V rows, `[t, kv_heads · head_dim]`.
    kv: Vec<(Vec<f32>, Vec<f32>)>,
}

impl<L: Linear> DenseAttention<L> {
    pub fn new(cfg: &Q38Config, layers: Vec<Option<AttnWeights<L>>>) -> Self {
        let n = layers.len();
        Self {
            cfg: cfg.clone(),
            layers,
            kv: vec![(Vec::new(), Vec::new()); n],
        }
    }
}

/// NeoX half-split RoPE on the first `rotary_dim` channels of each
/// `head_dim` head, text position `pos`.
fn rope_heads(x: &mut [f32], head_dim: usize, rotary_dim: usize, theta: f32, pos: usize) {
    let half = rotary_dim / 2;
    for head in x.chunks_mut(head_dim) {
        for i in 0..half {
            let inv_freq = 1.0 / theta.powf((2 * i) as f32 / rotary_dim as f32);
            let ang = pos as f32 * inv_freq;
            let (sn, cs) = ang.sin_cos();
            let (x0, x1) = (head[i], head[i + half]);
            head[i] = x0 * cs - x1 * sn;
            head[i + half] = x1 * cs + x0 * sn;
        }
    }
}

impl<L: Linear> AttnBlock for DenseAttention<L> {
    fn forward(&mut self, layer: usize, pos: usize, x: &[f32], out: &mut [f32]) -> Result<(), Q38Error> {
        let c = &self.cfg;
        let w = self.layers.get(layer).and_then(|l| l.as_ref()).ok_or(Q38Error::Hook {
            layer,
            msg: "no attention weights".into(),
        })?;
        let (hq, hkv, d) = (c.attn_heads, c.kv_heads, c.head_dim);
        let mut qg = vec![0.0; 2 * hq * d];
        w.q_proj.forward(x, &mut qg);
        let mut q = vec![0.0; hq * d];
        let mut gate = vec![0.0; hq * d];
        for hh in 0..hq {
            q[hh * d..(hh + 1) * d].copy_from_slice(&qg[hh * 2 * d..hh * 2 * d + d]);
            gate[hh * d..(hh + 1) * d].copy_from_slice(&qg[hh * 2 * d + d..(hh + 1) * 2 * d]);
        }
        let mut k = vec![0.0; hkv * d];
        let mut v = vec![0.0; hkv * d];
        w.k_proj.forward(x, &mut k);
        w.v_proj.forward(x, &mut v);
        let qn = w.q_norm.repeat(hq);
        let kn = w.k_norm.repeat(hkv);
        let q_in = q.clone();
        rms_norm_grouped(&q_in, &qn, d, c.rms_eps, NormGain::OnePlus, &mut q);
        let k_in = k.clone();
        rms_norm_grouped(&k_in, &kn, d, c.rms_eps, NormGain::OnePlus, &mut k);
        rope_heads(&mut q, d, c.rotary_dim, c.rope_theta, pos);
        rope_heads(&mut k, d, c.rotary_dim, c.rope_theta, pos);
        let (kc, vc) = &mut self.kv[layer];
        kc.extend_from_slice(&k);
        vc.extend_from_slice(&v);
        let t_len = kc.len() / (hkv * d);
        if t_len != pos + 1 {
            return Err(Q38Error::Hook {
                layer,
                msg: format!("KV holds {t_len} rows at position {pos}"),
            });
        }
        let scale = 1.0 / (d as f32).sqrt();
        let rep = hq / hkv;
        let mut attn_out = vec![0.0; hq * d];
        let mut scores = vec![0.0f32; t_len];
        for hh in 0..hq {
            let kvh = hh / rep;
            let qh = &q[hh * d..(hh + 1) * d];
            let mut mx = f32::NEG_INFINITY;
            for t in 0..t_len {
                let kr = &kc[(t * hkv + kvh) * d..(t * hkv + kvh + 1) * d];
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot = qh[i].mul_add(kr[i], dot);
                }
                scores[t] = dot * scale;
                mx = mx.max(scores[t]);
            }
            let mut sum = 0.0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - mx).exp();
                sum += *sc;
            }
            let o = &mut attn_out[hh * d..(hh + 1) * d];
            for t in 0..t_len {
                let p = scores[t] / sum;
                let vr = &vc[(t * hkv + kvh) * d..(t * hkv + kvh + 1) * d];
                for i in 0..d {
                    o[i] = p.mul_add(vr[i], o[i]);
                }
            }
        }
        for (o, &g) in attn_out.iter_mut().zip(&gate) {
            *o *= sigmoid(g);
        }
        w.o_proj.forward(&attn_out, out);
        Ok(())
    }

    fn reset(&mut self) {
        for (k, v) in &mut self.kv {
            k.clear();
            v.clear();
        }
    }
}

// --- Model ------------------------------------------------------------

/// Token embedding lookup.
pub trait Embedding {
    fn lookup(&self, token: u32, out: &mut [f32]);
}

/// Row-major `[vocab, H]` f32 table.
#[derive(Debug, Clone)]
pub struct DenseEmbedding {
    pub hidden: usize,
    pub table: Vec<f32>,
}

impl Embedding for DenseEmbedding {
    fn lookup(&self, token: u32, out: &mut [f32]) {
        let t = token as usize;
        out.copy_from_slice(&self.table[t * self.hidden..(t + 1) * self.hidden]);
    }
}

/// Whole-model weights of the CPU path.
pub struct ModelWeights<L: Linear, E: ExpertSet, M: Embedding> {
    pub embed: M,
    pub layers: Vec<LayerWeights<L, E>>,
    /// `hyper_connection_mixer` (`block_inject = None`).
    pub mixer: HcWeights<L>,
    pub lm_head: L,
}

/// CPU end-to-end forward of the text model, one token at a time.
pub struct Q38CpuModel<L: Linear, E: ExpertSet, M: Embedding, A: AttnBlock, P: PleHook> {
    pub cfg: Q38Config,
    pub weights: ModelWeights<L, E, M>,
    pub attn: A,
    pub ple: P,
    states: Vec<LayerState>,
    scratch: LayerScratch,
    pos: usize,
}

impl<L: Linear, E: ExpertSet, M: Embedding, A: AttnBlock, P: PleHook> Q38CpuModel<L, E, M, A, P> {
    pub fn new(cfg: Q38Config, weights: ModelWeights<L, E, M>, attn: A, ple: P) -> Self {
        let states = (0..cfg.num_layers)
            .map(|l| match cfg.layer_kind(l) {
                Q38LayerKind::LinearAttn => LayerState::LinearAttn(GdnState::new(&cfg)),
                Q38LayerKind::Qsa => LayerState::Qsa,
            })
            .collect();
        let scratch = LayerScratch::new(&cfg);
        Self {
            cfg,
            weights,
            attn,
            ple,
            states,
            scratch,
            pos: 0,
        }
    }

    /// Tokens consumed so far.
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) {
        for s in &mut self.states {
            if let LayerState::LinearAttn(g) = s {
                g.reset();
            }
        }
        self.attn.reset();
        self.ple.reset();
        self.pos = 0;
    }

    /// Run one token at the next position; writes `logits` (`[vocab]`)
    /// and returns each layer's chosen experts.
    pub fn forward_token(&mut self, token: u32, logits: &mut [f32]) -> Result<Vec<Vec<i32>>, Q38Error> {
        let c = &self.cfg;
        let mut emb = vec![0.0; c.hidden];
        self.weights.embed.lookup(token, &mut emb);
        let mut hyper = vec![0.0; c.hc_hidden()];
        hc_expand(&emb, c.hc_count, &mut hyper);
        let mut routes = Vec::with_capacity(c.num_layers);
        for l in 0..c.num_layers {
            routes.push(decoder_layer_forward(
                c,
                l,
                &self.weights.layers[l],
                &mut self.states[l],
                &mut self.attn,
                &mut self.ple,
                self.pos,
                token,
                &mut hyper,
                &mut self.scratch,
            )?);
        }
        // Final mixer, then the LM head — no final norm in this family.
        let mut out = vec![0.0; c.hidden];
        gated_residual_mix(&self.weights.mixer.weights(), c.hidden, c.hc_count, c.rms_eps, &hyper, &mut self.scratch.gr, &mut out, None);
        self.weights.lm_head.forward(&out, logits);
        self.pos += 1;
        Ok(routes)
    }
}
