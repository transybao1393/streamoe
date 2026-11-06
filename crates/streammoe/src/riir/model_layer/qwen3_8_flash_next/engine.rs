//! Qwen3.8-Flash-Next in the engine: weights from the converted
//! model, streamed experts, decode on the GPU (or the CPU backend, the
//! same steps op for op).
//!
//! [`Q38Engine`] is what [`crate::riir::RsCtx`] runs for this family
//! (`model-qwen3-8-flash-next`): `RsCtx::open` builds it in place of
//! the qwen3_5_moe layer caches, and `eval_prompt` / `eval_token` /
//! `state_save` / `state_load` / `memory_clear` route to it. It is
//! generic over the backend ([`Q38Exec`]), so the integration tests run
//! the Metal and the CPU backends through the same code.
//!
//! ## One chunk (one token in decode, up to `prefill_chunk` in prefill)
//!
//! ```text
//! hyper = expand(embed(tokens))                        [T, hc · H]
//! per layer l:
//!   if PLE layer: hyper += ple(hyper, tokens)          n-gram rows (host) → GPU (ple_gpu)
//!   x, inj = gated_residual(attn_hc, hyper)            FamOp + matvecs
//!   y = gated_deltanet(x)                              graph ops (the conv / recurrence walk
//!                                                      the chunk in order), sigmoid-gated norm
//!     | qsa(x)                                         push_qsa_pre_sdpa + dense SDPA, or over
//!                                                      the budget: host block selection, GPU
//!                                                      sparse rows (FamOp::SparseRows)
//!   hyper += y ⊗ inj
//!   x, inj = gated_residual(mlp_hc, hyper)
//!   router (host softmax top-k), shared expert + sigmoid gate (graph)
//!   routed experts: read (hot tier / pread) into the top_k staging
//!     slots, rows gathered on the GPU, Op::MoeBatchedPermuteFuse
//!     (a chunk: tokens grouped by expert, top_k experts per group)
//!   y = experts + sigmoid(g) · shared                  combine against zero (no residual)
//!   hyper += y ⊗ inj
//! x = gated_residual(mixer, hyper) → lm_head           last row; no final norm
//! ```
//!
//! ## Expert reads
//!
//! The routed experts come from the hot tier (a copy from RAM), a
//! speculative prefetch slot, or a pread, into one staging buffer the
//! permute-fuse op reads at `slot · expert_size`:
//!
//! ```text
//! slots [0, K)              demand reads and hot-tier copies, bucket i in slot i
//! slots [K + s·P, K + s·P + P)   prefetch set s = layer % 2, P = min(prefetch_experts, K)
//! ```
//!
//! The prefetch is `io::prefetch::PrefetchState`, the a3b decode
//! path's state machine (previous-token guess, strongest pick first,
//! hot-tier experts masked out): layer L's reads are issued at the top
//! of layer L, before its first command buffer, so they run under the
//! GPU work up to its router; with `prefetch_lookahead` layer L+1's go
//! out right after layer L's router readback. A routed expert found in
//! the layer's set binds that slot (no copy). Wrong guesses finish in
//! the background; everything in flight is drained at the end of the
//! token. Off with `prefetch_experts = 0` (then only `K` slots are
//! allocated).
//!
//! Safety of the set reuse rests on the per-layer router readback
//! waiting for every command buffer queued before it: set `L % 2` was
//! last bound by layer L-2's experts, which are queued before layer
//! L-1's router readback, which completes before layer L starts.
//!
//! **Command buffers.** One per layer: everything from the previous
//! layer's experts to this layer's router logits, which the host reads
//! to pick the experts to load. On top of that: a QSA layer past the budget (the host
//! scores blocks from the chunk's indexer projection), each extra
//! expert group of a chunk, and the tail. A QSA layer within the budget
//! adds none: its indexer keys are read back after the router's submit
//! (`drain_pending_index`). Tiny model, decode: 5 per token
//! within the budget, 6 past it ([`Q38Engine::submits`]). The QSA Q / K
//! norms and the Gated DeltaNet output norm are family kernels
//! ([`super::engine_exec`]); every `1 + w` gain is applied in a kernel
//! from the stored bf16 value.
//!
//! ## Prefill
//!
//! [`Q38Engine::eval`] runs the prompt in chunks of
//! [`Q38Engine::prefill_chunk`] tokens through the same ops at `T`
//! rows ([`Q38Engine::forward_chunk`]; a token is `T = 1`). Within f32
//! rounding of a step per token (bit-identical on the CPU backend);
//! `batched_prefill_matches_per_token` checks logits, state and the
//! decode after it on both backends, past the QSA budget.
//!
//! ## Not supported yet
//!
//! - Layer ranges (`OpenConfig::layer_range` other than the whole
//!   model) and `eval_stage` with hidden rows: the residual crossing a
//!   stage boundary is `hc · H` wide here. `RsCtx` rejects them.
//! - Rewinding (`memory_seq_rm` to a middle position): the linear
//!   attention and PLE conv states cannot be truncated. Checkpoints
//!   (`state_save` / `state_load`) work.
//! - KV formats other than f32, multimodal positions, multi-request
//!   pools.

use std::path::Path;

use rayon::prelude::*;

use super::config::{
    Q38Config, Q38LayerKind, HC_COUNT, INDEXER_BUDGET, INDEXER_COMPRESS_RATIO, INDEXER_HEAD_DIM, INDEXER_N_HEADS,
    NGRAM_SEED, NGRAM_SIZE, NGRAM_VOCAB_DIVISIBLE_BY, NGRAM_VOCAB_SIZE_BASE, HEADS_PER_NGRAM, PLE_CONV_KERNEL,
    PLE_EMBED_DIM,
};
use super::engine_exec::{retag, run_steps, FBuf, FamOp, Q38Exec, Step};
use super::ngram::{NgramConfig, NgramTable};
use super::ple::{PleConfig, PleError, PleLayer, PleState, PleWeights};
use super::qsa::gpu::qsa_select_rows;
use super::ple_gpu::PleOp;
use super::qsa::{
    push_qsa_o_proj, push_qsa_pre_sdpa, push_qsa_sdpa, qsa_indexer_step, qsa_needs_sparse,
    QsaGraphBufs, QsaIndexerCache, QsaLayerWeights, QsaMode, QsaShape,
};
use super::weights::{final_mixer_names, gated_residual_names, GatedResidualNames, HcSite};
use crate::riir::backend::buftype::Buf;
use crate::riir::backend::{BufId, BufferPool, Graph, GraphError, Op, WeightRef};
use crate::riir::io::embedding::{bf16_to_f32, embed_lookup};
use crate::riir::io::expert_io::{ExpertFiles, ExpertIoError};
use crate::riir::io::io_stats::{guess_overlap, Guesser};
use crate::riir::io::prefetch::{mask_hot, PrefetchState};
use crate::riir::moe::expert_forward::MAX_K;
use crate::riir::io::weight_file::WeightFile;
use crate::riir::moe::moe_router::{build_expert_buckets, moe_router_cpu_with, ExpertBuckets};
use crate::riir::variants::{MROPE_SECTION, RMS_NORM_EPS, ROPE_THETA, VARIANT};

/// Errors of the family engine.
#[derive(Debug, thiserror::Error)]
pub enum Q38EngineError {
    #[error("graph: {0}")]
    Graph(#[from] GraphError),
    #[error("weights: tensor {0} missing from the manifest")]
    Missing(String),
    #[error("experts: {0}")]
    Experts(#[from] ExpertIoError),
    #[error("ple: {0}")]
    Ple(#[from] PleError),
    #[error("n-gram table: {0}")]
    Ngram(#[from] super::ngram::NgramTableError),
    #[error("kernels: {0}")]
    Kernels(#[from] super::gated_residual_gpu::Q38GpuError),
    #[error("router: {0}")]
    Router(#[from] crate::riir::moe::moe_router::MoeRouterError),
    #[error("snapshot: {0}")]
    Snapshot(#[from] SnapErr),
    #[error("{0}")]
    Other(String),
}

use crate::riir::snapshot::state_snapshot::StateSnapshotError as SnapErr;

type Result<T> = std::result::Result<T, Q38EngineError>;

// --- Shapes --------------------------------------------------------------

/// The family's runtime config ([`Q38Config::REAL`] for the build).
pub fn q38_config() -> Q38Config {
    Q38Config::REAL
}

/// QSA shape of the build.
pub fn qsa_shape() -> QsaShape {
    let v = VARIANT;
    QsaShape {
        hidden: v.hidden_dim,
        num_heads: v.num_attn_heads,
        num_kv_heads: v.num_kv_heads,
        head_dim: v.head_dim,
        rotary_dim: v.gqa_rotary_dim,
        rope_theta: ROPE_THETA,
        mrope_section: MROPE_SECTION.unwrap_or([0; 3]),
        eps: RMS_NORM_EPS,
        idx_heads: INDEXER_N_HEADS,
        idx_head_dim: INDEXER_HEAD_DIM,
        compress_ratio: INDEXER_COMPRESS_RATIO,
        budget: INDEXER_BUDGET,
    }
}

/// PLE layer shape of the build.
pub fn ple_config() -> PleConfig {
    PleConfig {
        hidden: VARIANT.hidden_dim,
        hc: HC_COUNT,
        embed_dim: PLE_EMBED_DIM,
        conv_kernel: PLE_CONV_KERNEL,
        conv_dilation: NGRAM_SIZE,
        group_size: crate::riir::variants::GROUP_SIZE,
        eps: RMS_NORM_EPS,
    }
}

/// N-gram hashing of the build's one PLE layer.
pub fn ngram_config() -> NgramConfig {
    NgramConfig {
        vocab_size: VARIANT.vocab_size as u64,
        ngram_size: NGRAM_SIZE,
        heads_per_ngram: HEADS_PER_NGRAM,
        ngram_vocab_size_base: NGRAM_VOCAB_SIZE_BASE as u64,
        make_ngram_vocab_size_divisible_by: NGRAM_VOCAB_DIVISIBLE_BY as u64,
        seed: NGRAM_SEED,
        // `text_config.eos_token_id`.
        eos_token_id: VARIANT.eos_token_1 as i32,
        ple_layer_index: 0,
    }
}

/// Rows the n-gram table keeps in RAM (`STREAMMOE_NGRAM_CACHE_ROWS`,
/// default 16384: 1.6 MB of 100-byte rows on the real model).
pub fn ngram_cache_rows() -> usize {
    std::env::var("STREAMMOE_NGRAM_CACHE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16384)
}

/// Tokens per prefill chunk for an `n_ctx` context
/// (`STREAMMOE_Q38_CHUNK`, default 64, at most `n_ctx`): the row count
/// of every per-token buffer.
pub fn prefill_chunk(n_ctx: usize) -> usize {
    std::env::var("STREAMMOE_Q38_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64usize)
        .min(n_ctx)
        .max(1)
}

/// The n-gram table file of a converted model (`root/ngram.mfng`).
pub fn ngram_path(root: &Path) -> std::path::PathBuf {
    root.join("ngram.mfng")
}

// --- Weights ---------------------------------------------------------------

fn info<'a>(wf: &'a WeightFile, name: &str) -> Result<&'a crate::riir::io::weight_file::TensorInfo> {
    wf.tensor_info(name).ok_or_else(|| Q38EngineError::Missing(name.to_string()))
}

/// `WeightRef` of the quantized triple `base.{weight,scales,biases}`.
fn qref(wf: &WeightFile, base: &str) -> Result<WeightRef> {
    let w = info(wf, &format!("{base}.weight"))?;
    let s = info(wf, &format!("{base}.scales"))?;
    let b = info(wf, &format!("{base}.biases"))?;
    let bits = if w.bits > 0 { w.bits as u32 } else { 4 };
    Ok(WeightRef { w_off: w.offset, s_off: s.offset, b_off: b.offset, bits })
}

fn off(wf: &WeightFile, name: &str) -> Result<u64> {
    Ok(info(wf, name)?.offset)
}

/// A bf16 gain as `1 + w` f32 values (host-side norms: the indexer's).
fn one_plus(wf: &WeightFile, name: &str) -> Result<Vec<f32>> {
    let bytes = wf.tensor_bytes(name).ok_or_else(|| Q38EngineError::Missing(name.to_string()))?;
    Ok(bytes.chunks_exact(2).map(|c| 1.0 + bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
}

/// One Gated Residual's weights.
#[derive(Debug, Clone, Copy)]
pub struct HcRefs {
    pub gain_off: u64,
    pub down: WeightRef,
    pub up: WeightRef,
    pub inject: Option<WeightRef>,
}

impl HcRefs {
    fn load(wf: &WeightFile, n: &GatedResidualNames) -> Result<Self> {
        Ok(Self {
            gain_off: off(wf, &n.hc_norm)?,
            down: qref(wf, &n.mix_down)?,
            up: qref(wf, &n.mix_up)?,
            inject: n.block_inject.as_deref().map(|b| qref(wf, b)).transpose()?,
        })
    }
}

/// Gated DeltaNet weights of one layer.
#[derive(Debug, Clone, Copy)]
pub struct GdnRefs {
    pub qkv: WeightRef,
    pub z: WeightRef,
    pub b: WeightRef,
    pub a: WeightRef,
    pub out: WeightRef,
    pub conv_off: u64,
    pub a_log_off: u64,
    pub dt_bias_off: u64,
    pub norm_off: u64,
}

/// QSA weights of one layer: the graph refs plus the host indexer's
/// norms (`1 + w`).
#[derive(Debug, Clone)]
pub struct QsaRefs {
    pub w: QsaLayerWeights,
    pub idx_q_norm: Vec<f32>,
    pub idx_k_norm: Vec<f32>,
}

#[derive(Debug, Clone)]
pub enum AttnRefs {
    Gdn(GdnRefs),
    Qsa(QsaRefs),
}

/// MoE weights of one layer (the routed experts are streamed).
#[derive(Debug, Clone, Copy)]
pub struct MoeRefs {
    pub router: WeightRef,
    pub shared_gate: WeightRef,
    pub gate: WeightRef,
    pub up: WeightRef,
    pub down: WeightRef,
}

#[derive(Debug, Clone)]
pub struct LayerRefs {
    pub attn_hc: HcRefs,
    pub mlp_hc: HcRefs,
    pub attn: AttnRefs,
    pub moe: MoeRefs,
}

/// Weight-file offsets of the whole text model — the family's
/// counterpart of `LayerWeightCache` (no input / post-attention
/// norms; Gated Residual offsets from [`gated_residual_names`]).
#[derive(Debug, Clone)]
pub struct Q38Weights {
    pub layers: Vec<LayerRefs>,
    pub mixer: HcRefs,
    pub lm_head: WeightRef,
}

impl Q38Weights {
    pub fn load(wf: &WeightFile) -> Result<Self> {
        let cfg = q38_config();
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            let p = format!("model.layers.{l}");
            let attn = match cfg.layer_kind(l) {
                Q38LayerKind::LinearAttn => AttnRefs::Gdn(GdnRefs {
                    qkv: qref(wf, &format!("{p}.linear_attn.in_proj_qkv"))?,
                    z: qref(wf, &format!("{p}.linear_attn.in_proj_z"))?,
                    b: qref(wf, &format!("{p}.linear_attn.in_proj_b"))?,
                    a: qref(wf, &format!("{p}.linear_attn.in_proj_a"))?,
                    out: qref(wf, &format!("{p}.linear_attn.out_proj"))?,
                    conv_off: off(wf, &format!("{p}.linear_attn.conv1d.weight"))?,
                    a_log_off: off(wf, &format!("{p}.linear_attn.A_log"))?,
                    dt_bias_off: off(wf, &format!("{p}.linear_attn.dt_bias"))?,
                    norm_off: off(wf, &format!("{p}.linear_attn.norm.weight"))?,
                }),
                Q38LayerKind::Qsa => AttnRefs::Qsa(QsaRefs {
                    w: QsaLayerWeights {
                        q_proj: qref(wf, &format!("{p}.self_attn.q_proj"))?,
                        k_proj: qref(wf, &format!("{p}.self_attn.k_proj"))?,
                        v_proj: qref(wf, &format!("{p}.self_attn.v_proj"))?,
                        o_proj: qref(wf, &format!("{p}.self_attn.o_proj"))?,
                        idx_qk_proj: qref(wf, &format!("{p}.self_attn.indexer.index_qk_proj"))?,
                        // Raw (zero-centred) gains: the engine swaps the
                        // graph's plain-gain norm for
                        // `FamOp::HeadNormOnePlus` (see `qsa_steps`).
                        q_norm_off: off(wf, &format!("{p}.self_attn.q_norm.weight"))?,
                        k_norm_off: off(wf, &format!("{p}.self_attn.k_norm.weight"))?,
                    },
                    idx_q_norm: one_plus(wf, &format!("{p}.self_attn.indexer.q_layernorm.weight"))?,
                    idx_k_norm: one_plus(wf, &format!("{p}.self_attn.indexer.k_layernorm.weight"))?,
                }),
            };
            layers.push(LayerRefs {
                attn_hc: HcRefs::load(wf, &gated_residual_names(l, HcSite::Attn))?,
                mlp_hc: HcRefs::load(wf, &gated_residual_names(l, HcSite::Mlp))?,
                attn,
                moe: MoeRefs {
                    router: qref(wf, &format!("{p}.mlp.gate"))?,
                    shared_gate: qref(wf, &format!("{p}.mlp.shared_expert_gate"))?,
                    gate: qref(wf, &format!("{p}.mlp.shared_expert.gate_proj"))?,
                    up: qref(wf, &format!("{p}.mlp.shared_expert.up_proj"))?,
                    down: qref(wf, &format!("{p}.mlp.shared_expert.down_proj"))?,
                },
            });
        }
        Ok(Self {
            layers,
            mixer: HcRefs::load(wf, &final_mixer_names())?,
            lm_head: qref(wf, "lm_head")?,
        })
    }
}

/// The PLE layer's weights, borrowed from the weight file.
fn ple_weights<'a>(wf: &'a WeightFile, layer: usize) -> Result<PleWeights<'a>> {
    let names = super::ple::tensor_names(layer);
    let t = |i: usize| wf.tensor_bytes(&names[i]).ok_or_else(|| Q38EngineError::Missing(names[i].clone()));
    Ok(PleWeights {
        key_w: t(0)?,
        key_s: t(1)?,
        key_b: t(2)?,
        value_w: t(3)?,
        value_s: t(4)?,
        value_b: t(5)?,
        norm_key: t(6)?,
        norm_query: t(7)?,
        norm_conv: t(8)?,
        conv: t(9)?,
    })
}

// --- Memory plan -------------------------------------------------------------

/// Bytes the family engine allocates, by account, for the budget plan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Q38PlanBytes {
    /// QSA K / V caches at `n_ctx` (f32).
    pub kv: u64,
    /// Gated DeltaNet conv + recurrence state (GPU).
    pub linear_state: u64,
    /// The expert staging buffer: `top_k` demand slots plus two
    /// prefetch sets of `min(prefetch_experts, top_k)`.
    pub expert_slots: u64,
    /// Per-token scratch at [`prefill_chunk`] rows: the `hc · H`
    /// residual and Gated Residual buffers, projection / MoE scratch;
    /// plus the logits.
    pub scratch: u64,
    /// Host state: QSA indexer raw + block keys at `n_ctx`, PLE conv /
    /// history state, the n-gram row cache.
    pub family_state: u64,
}

impl Q38PlanBytes {
    pub fn total(&self) -> u64 {
        self.kv + self.linear_state + self.expert_slots + self.scratch + self.family_state
    }
}

/// The plan for `n_ctx` tokens and `expert_size`-byte experts, prefetch
/// off.
pub fn plan_bytes(n_ctx: usize, expert_size: usize, ngram_cache_bytes: u64) -> Q38PlanBytes {
    plan_bytes_with(n_ctx, expert_size, ngram_cache_bytes, 0)
}

/// Prefetch slots per set for `prefetch_experts` experts per layer.
pub fn prefetch_slots(prefetch_experts: usize) -> usize {
    prefetch_experts.min(q38_config().top_k).min(MAX_K)
}

/// [`plan_bytes`] with speculative prefetch reading `prefetch_experts`
/// experts per layer (two sets of [`prefetch_slots`] staging slots).
pub fn plan_bytes_with(n_ctx: usize, expert_size: usize, ngram_cache_bytes: u64, prefetch_experts: usize) -> Q38PlanBytes {
    let c = q38_config();
    let s = qsa_shape();
    let n_qsa = (0..c.num_layers).filter(|&l| c.layer_kind(l) == Q38LayerKind::Qsa).count() as u64;
    let n_lin = c.num_layers as u64 - n_qsa;
    let f = 4u64;
    let kv = n_qsa * 2 * (n_ctx * s.kv_dim()) as u64 * f;
    let lin_state =
        ((c.conv_kernel - 1) * c.linear_conv_dim() + c.linear_v_heads * c.linear_value_dim * c.linear_key_dim) as u64;
    let linear_state = n_lin * lin_state * f;
    let k = c.top_k as u64;
    let expert_slots = (k + 2 * prefetch_slots(prefetch_experts) as u64) * expert_size as u64;
    let (h, hc) = (c.hidden as u64, c.hc_hidden() as u64);
    let vt = (c.linear_v_heads * c.linear_value_dim) as u64;
    let rows = prefill_chunk(n_ctx) as u64;
    let scratch = f
        * (rows * (4 * hc // hyper, ple_out, normed, mix
            + 4 * h + c.hc_lowrank as u64 + c.hc_count as u64 // emb, mixed, block_out, out_sum + low + inj
            + 2 * c.linear_conv_dim() as u64 + 3 * vt + 4 * c.linear_v_heads as u64
            + (s.q_proj_dim() + 3 * s.q_dim() + 2 * s.kv_dim() + s.idx_proj_dim() + s.rotary_dim) as u64
            + c.num_experts as u64 + 4 * c.shared_intermediate as u64 + 2 * h + 2
            + k * (2 * h + 3 * c.moe_intermediate as u64 + 3))
            + c.vocab as u64);
    let pc = ple_config();
    // GPU PLE: the n-gram embedding, k, v, norm_conv rows (one token).
    let scratch = scratch + f * (pc.embed_dim + 2 * pc.hc_hidden() + pc.hidden) as u64;
    // PLE conv state: the GPU copy and the host one (snapshots).
    let family_state = n_qsa * (n_ctx * s.idx_head_dim) as u64 * f * 5 / 4
        + n_qsa * (n_ctx * 12) as u64
        + 2 * (pc.conv_state_rows() * pc.hc_hidden()) as u64 * f
        + ngram_cache_bytes;
    Q38PlanBytes { kv, linear_state, expert_slots, scratch, family_state }
}

// --- Buffers ---------------------------------------------------------------

/// Per-token buffers (all persistent: the pool never aliases them).
struct Bufs {
    hyper: FBuf,
    emb: FBuf,
    ple_out: FBuf,
    normed: FBuf,
    low: FBuf,
    mix: FBuf,
    mixed: FBuf,
    inj: FBuf,
    block_out: FBuf,
    // Gated DeltaNet
    qkv: FBuf,
    z: FBuf,
    beta: FBuf,
    alpha: FBuf,
    conv_out: FBuf,
    g_decay: FBuf,
    beta_gate: FBuf,
    delta_out: FBuf,
    value_out: FBuf,
    // QSA (k / v caches per layer, in `LayerBufs`)
    q_proj: FBuf,
    k: FBuf,
    v: FBuf,
    idx_qk: FBuf,
    q: FBuf,
    gate: FBuf,
    attn_out: FBuf,
    inv_freq: FBuf,
    // MoE
    router_logits: FBuf,
    shared_gate: FBuf,
    s_gate: FBuf,
    s_up: FBuf,
    s_act: FBuf,
    s_down: FBuf,
    zeros: FBuf,
    out_sum: FBuf,
    expert_base: FBuf,
    expert_indices: FBuf,
    bucket_input: FBuf,
    bucket_gate: FBuf,
    bucket_up: FBuf,
    bucket_act: FBuf,
    bucket_out: FBuf,
    bucket_token_idx: FBuf,
    bucket_weights: FBuf,
    logits: FBuf,
}

/// The PLE on the GPU ([`super::ple_gpu`]): weight offsets, buffers
/// for one token, and the conv state (the state of record; the host
/// [`PleState::conv`] is only a snapshot staging copy).
struct PleGpu {
    key: WeightRef,
    value: WeightRef,
    norm_key_off: u64,
    norm_query_off: u64,
    norm_conv_off: u64,
    conv_off: u64,
    /// `[embed]`, `[hc·H]`, `[H]`, `[hc·H]`.
    e: FBuf,
    k: FBuf,
    v: FBuf,
    n: FBuf,
    /// `[(K-1)·D, hc·H]`.
    state: FBuf,
}

/// `STREAMMOE_Q38_PLE=host`: run the PLE on the host (the older path:
/// residual read back, [`PleLayer::forward`], GPU add).
fn ple_on_host_from_env() -> bool {
    std::env::var("STREAMMOE_Q38_PLE").is_ok_and(|v| v == "host")
}

/// Per-layer GPU state.
enum LayerBufs {
    Gdn { conv: FBuf, delta: FBuf, conv_len: usize, delta_len: usize },
    Qsa { k_cache: FBuf, v_cache: FBuf },
}

fn alloc<B: Q38Exec>(b: &mut B, floats: usize, label: &'static str) -> Result<FBuf> {
    b.pool_mut()
        .alloc::<crate::riir::backend::HiddenBuf>(floats.max(1) * 4, label, true)
        .map_err(|e| Q38EngineError::Other(format!("alloc {label}: {e}")))
}

fn upload<B: Q38Exec, T: bytemuck::Pod>(b: &mut B, id: FBuf, data: &[T]) -> Result<()> {
    b.pool_mut()
        .upload(id, bytemuck::cast_slice(data))
        .map_err(|e| Q38EngineError::Other(format!("upload: {e}")))
}

fn download<B: Q38Exec>(b: &B, id: FBuf, n: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; n];
    b.pool()
        .download(id, bytemuck::cast_slice_mut(&mut out))
        .map_err(|e| Q38EngineError::Other(format!("download: {e}")))?;
    Ok(out)
}

fn t<A: Buf>(id: FBuf) -> BufId<A> {
    retag(id)
}

// --- Engine ------------------------------------------------------------------

/// The family's forward state on backend `B`.
pub struct Q38Engine<B: Q38Exec> {
    fam: B::Fam,
    cfg: Q38Config,
    qsa: QsaShape,
    mode: QsaMode,
    w: Q38Weights,
    bufs: Bufs,
    layers: Vec<LayerBufs>,
    /// Host indexer caches (QSA layers).
    indexer: Vec<Option<QsaIndexerCache>>,
    /// `(decoder layer, layer, state)` of the PLE.
    ple: Option<(usize, PleLayer, PleState)>,
    /// The PLE's GPU weights and buffers (with a PLE layer).
    ple_gpu: Option<PleGpu>,
    /// Run the PLE on the host instead ([`Self::set_ple_on_host`]).
    ple_host: bool,
    n_ctx: usize,
    expert_size: usize,
    /// Rows of the per-token buffers ([`prefill_chunk`]).
    max_chunk: usize,
    /// Tokens per prefill chunk, `<= max_chunk` ([`Self::set_prefill_chunk`]).
    chunk: usize,
    /// Tokens consumed (the next token's position).
    pos: usize,
    /// A dense QSA layer's indexer step not run yet: `(layer, pos,
    /// n_tokens)` ([`Self::drain_pending_index`]).
    pending_index: Option<(usize, usize, usize)>,
    /// Command buffers submitted so far ([`Self::submits`]).
    submits: std::sync::atomic::AtomicU64,
    /// Speculative expert prefetch (`None`: off).
    pf: Option<PrefetchState>,
    /// Staging slots per prefetch set ([`prefetch_slots`]; 0 = off).
    pf_slots: usize,
    /// Host address of `bufs.expert_base` (shared storage / the CPU
    /// backend's vector: allocated once, never moved). Slots are
    /// handed out as disjoint slices of it ([`Self::slot`]), so
    /// prefetch workers and demand reads never alias.
    staging_addr: usize,
}

impl<B: Q38Exec> Q38Engine<B> {
    /// Build the engine on `b`: weight offsets from `wf`, the n-gram
    /// table from `root/ngram.mfng`, buffers for `n_ctx` tokens and
    /// `expert_size`-byte experts. Allocates and zeroes everything
    /// now. No speculative prefetch ([`Self::with_prefetch`]).
    pub fn new(b: &mut B, wf: &WeightFile, root: &Path, n_ctx: usize, expert_size: usize) -> Result<Self> {
        Self::with_prefetch(b, wf, root, n_ctx, expert_size, None)
    }

    /// [`Self::new`] with speculative expert prefetch driven by `pf`
    /// (its `max_experts` and `lookahead` settings; the cross-layer
    /// guesser is not used here). `None` or `max_experts() == 0`: off.
    pub fn with_prefetch(
        b: &mut B,
        wf: &WeightFile,
        root: &Path,
        n_ctx: usize,
        expert_size: usize,
        pf: Option<PrefetchState>,
    ) -> Result<Self> {
        let pf_slots = pf.as_ref().map_or(0, |p| prefetch_slots(p.max_experts()));
        let pf = pf.filter(|_| pf_slots > 0);
        let cfg = q38_config();
        let qsa = qsa_shape();
        let w = Q38Weights::load(wf)?;
        let fam = b.fam_new(&qsa)?;
        let (h, hc) = (cfg.hidden, cfg.hc_hidden());
        let k = cfg.top_k;
        // Rows of every per-token buffer: one prefill chunk.
        let ch = prefill_chunk(n_ctx);
        let vt = cfg.linear_v_heads * cfg.linear_value_dim;
        let bufs = Bufs {
            hyper: alloc(b, ch * hc, "q38.hyper")?,
            emb: alloc(b, ch * h, "q38.emb")?,
            ple_out: alloc(b, ch * hc, "q38.ple_out")?,
            normed: alloc(b, ch * hc, "q38.hc_normed")?,
            low: alloc(b, ch * cfg.hc_lowrank, "q38.hc_low")?,
            mix: alloc(b, ch * hc, "q38.hc_mix")?,
            mixed: alloc(b, ch * h, "q38.mixed")?,
            inj: alloc(b, ch * cfg.hc_count, "q38.inj")?,
            block_out: alloc(b, ch * h, "q38.block_out")?,
            qkv: alloc(b, ch * cfg.linear_conv_dim(), "q38.qkv")?,
            z: alloc(b, ch * vt, "q38.z")?,
            beta: alloc(b, ch * cfg.linear_v_heads, "q38.beta")?,
            alpha: alloc(b, ch * cfg.linear_v_heads, "q38.alpha")?,
            conv_out: alloc(b, ch * cfg.linear_conv_dim(), "q38.conv_out")?,
            g_decay: alloc(b, ch * cfg.linear_v_heads, "q38.g_decay")?,
            beta_gate: alloc(b, ch * cfg.linear_v_heads, "q38.beta_gate")?,
            delta_out: alloc(b, ch * vt, "q38.delta_out")?,
            value_out: alloc(b, ch * vt, "q38.value_out")?,
            q_proj: alloc(b, ch * qsa.q_proj_dim(), "q38.q_proj")?,
            k: alloc(b, ch * qsa.kv_dim(), "q38.k")?,
            v: alloc(b, ch * qsa.kv_dim(), "q38.v")?,
            idx_qk: alloc(b, ch * qsa.idx_proj_dim(), "q38.idx_qk")?,
            q: alloc(b, ch * qsa.q_dim(), "q38.q")?,
            gate: alloc(b, ch * qsa.q_dim(), "q38.gate")?,
            attn_out: alloc(b, ch * qsa.q_dim(), "q38.attn_out")?,
            inv_freq: alloc(b, qsa.rotary_dim / 2, "q38.inv_freq")?,
            router_logits: alloc(b, ch * cfg.num_experts, "q38.router_logits")?,
            shared_gate: alloc(b, ch, "q38.shared_gate")?,
            s_gate: alloc(b, ch * cfg.shared_intermediate, "q38.shared_gate_proj")?,
            s_up: alloc(b, ch * cfg.shared_intermediate, "q38.shared_up")?,
            s_act: alloc(b, ch * cfg.shared_intermediate, "q38.shared_act")?,
            s_down: alloc(b, ch * h, "q38.shared_down")?,
            zeros: alloc(b, ch * h, "q38.zeros")?,
            out_sum: alloc(b, ch * h, "q38.out_sum")?,
            expert_base: alloc(b, (k + 2 * pf_slots) * expert_size / 4 + 1, "q38.expert_staging")?,
            expert_indices: alloc(b, ch * k, "q38.expert_indices")?,
            bucket_input: alloc(b, ch * k * h, "q38.bucket_input")?,
            bucket_gate: alloc(b, ch * k * cfg.moe_intermediate, "q38.bucket_gate")?,
            bucket_up: alloc(b, ch * k * cfg.moe_intermediate, "q38.bucket_up")?,
            bucket_act: alloc(b, ch * k * cfg.moe_intermediate, "q38.bucket_act")?,
            bucket_out: alloc(b, ch * k * h, "q38.bucket_out")?,
            bucket_token_idx: alloc(b, ch * k, "q38.bucket_token_idx")?,
            bucket_weights: alloc(b, ch * k, "q38.bucket_weights")?,
            logits: alloc(b, cfg.vocab, "q38.logits")?,
        };
        upload(b, bufs.inv_freq, &qsa.inv_freq())?;
        let staging_addr = b
            .pool_mut()
            .with_bytes_mut(bufs.expert_base, |s: &mut [u8]| s.as_mut_ptr() as usize)
            .map_err(|e| Q38EngineError::Other(format!("staging: {e}")))?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        let mut indexer = Vec::with_capacity(cfg.num_layers);
        for (l, lr) in w.layers.iter().enumerate() {
            match &lr.attn {
                AttnRefs::Gdn(_) => {
                    let conv_len = (cfg.conv_kernel - 1) * cfg.linear_conv_dim();
                    let delta_len = vt * cfg.linear_key_dim;
                    layers.push(LayerBufs::Gdn {
                        conv: alloc(b, conv_len, "q38.conv_state")?,
                        delta: alloc(b, delta_len, "q38.delta_state")?,
                        conv_len,
                        delta_len,
                    });
                    indexer.push(None);
                }
                AttnRefs::Qsa(q) => {
                    layers.push(LayerBufs::Qsa {
                        k_cache: alloc(b, n_ctx * qsa.kv_dim(), "q38.k_cache")?,
                        v_cache: alloc(b, n_ctx * qsa.kv_dim(), "q38.v_cache")?,
                    });
                    indexer.push(Some(QsaIndexerCache::new(&qsa, q.idx_k_norm.clone())));
                }
            }
            let _ = l;
        }
        let ple = match cfg.ple_layers.first() {
            Some(&pl) => {
                let table = NgramTable::open(&ngram_path(root), ngram_cache_rows())?;
                let layer = PleLayer::new(ple_config(), ngram_config(), table)?;
                let state = layer.new_state();
                ple_weights(wf, pl)?; // all ten tensors present
                Some((pl, layer, state))
            }
            None => None,
        };
        let ple_gpu = match cfg.ple_layers.first() {
            Some(&pl) => {
                let pc = ple_config();
                let p = format!("model.layers.{pl}.ple");
                let state = alloc(b, pc.conv_state_rows() * pc.hc_hidden(), "q38.ple_conv_state")?;
                upload(b, state, &vec![0.0f32; pc.conv_state_rows() * pc.hc_hidden()])?;
                Some(PleGpu {
                    key: qref(wf, &format!("{p}.key_proj"))?,
                    value: qref(wf, &format!("{p}.value_proj"))?,
                    norm_key_off: off(wf, &format!("{p}.norm_key.weight"))?,
                    norm_query_off: off(wf, &format!("{p}.norm_query.weight"))?,
                    norm_conv_off: off(wf, &format!("{p}.norm_conv.weight"))?,
                    conv_off: off(wf, &format!("{p}.conv1d.weight"))?,
                    e: alloc(b, ch * pc.embed_dim, "q38.ple_e")?,
                    k: alloc(b, ch * pc.hc_hidden(), "q38.ple_k")?,
                    v: alloc(b, ch * pc.hidden, "q38.ple_v")?,
                    n: alloc(b, ch * pc.hc_hidden(), "q38.ple_n")?,
                    state,
                })
            }
            _ => None,
        };
        Ok(Self {
            fam,
            cfg,
            qsa,
            mode: QsaMode::from_env(),
            w,
            bufs,
            layers,
            indexer,
            ple,
            ple_gpu,
            ple_host: ple_on_host_from_env(),
            n_ctx,
            expert_size,
            max_chunk: ch,
            chunk: ch,
            pos: 0,
            pending_index: None,
            submits: std::sync::atomic::AtomicU64::new(0),
            pf,
            pf_slots,
            staging_addr,
        })
    }

    /// Tokens consumed so far.
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// Command buffers (Metal) / passes (CPU) submitted since the
    /// engine was built.
    pub fn submits(&self) -> u64 {
        self.submits.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Tokens per prefill chunk (clamped to `1..=` the buffers' rows,
    /// [`prefill_chunk`] at build).
    pub fn set_prefill_chunk(&mut self, n: usize) {
        self.chunk = n.clamp(1, self.max_chunk);
    }

    pub fn prefill_chunk(&self) -> usize {
        self.chunk
    }

    /// QSA mode (`STREAMMOE_QSA`, default sparse).
    pub fn set_qsa_mode(&mut self, mode: QsaMode) {
        self.mode = mode;
    }

    /// Speculative prefetch `(hits, misses)` in expert slots (hot-tier
    /// hits count as neither), `(0, 0)` when off.
    pub fn prefetch_stats(&self) -> (u64, u64) {
        self.pf.as_ref().map_or((0, 0), |p| p.stats())
    }

    /// Zero [`Self::prefetch_stats`].
    pub fn reset_prefetch_stats(&self) {
        if let Some(p) = &self.pf {
            p.reset_stats();
        }
    }

    /// Staging slots per prefetch set (0: prefetch off).
    pub fn prefetch_slots(&self) -> usize {
        self.pf_slots
    }

    /// Wait for every prefetch read still in flight (a token that
    /// failed midway can leave some). [`Self::step`] drains on success.
    pub fn drain_prefetch(&mut self) {
        if let Some(p) = self.pf.as_mut() {
            p.drain();
        }
    }

    /// Staging slot `i` (`expert_size` bytes).
    ///
    /// # Safety
    /// No other live slice of slot `i` (each read or prefetch task gets
    /// its own slot), and no GPU work reading it in flight.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slot(&self, i: usize) -> &mut [u8] {
        debug_assert!(i < self.cfg.top_k + 2 * self.pf_slots);
        // SAFETY: forwarded.
        unsafe { slot_at(self.staging_addr, self.expert_size, i) }
    }

    /// First staging slot of `layer`'s prefetch set.
    fn pf_first_slot(&self, layer: usize) -> usize {
        self.cfg.top_k + (layer % 2) * self.pf_slots
    }

    /// Issue `layer`'s speculative reads (previous-token guess, hot-tier
    /// experts masked) into its prefetch set, unless already in flight
    /// or there is no guess yet. Set `layer % 2` was last bound by
    /// layer `layer - 2`'s experts, complete by now (module docs).
    fn prefetch_dispatch(&mut self, layer: usize, experts: &ExpertFiles, io_pool: &rayon::ThreadPool) {
        let p = self.pf_slots;
        let first = self.pf_first_slot(layer);
        let (addr, es) = (self.staging_addr, self.expert_size);
        let Some(pf) = self.pf.as_mut() else { return };
        if layer >= self.cfg.num_layers || pf.in_flight_for(layer) {
            return;
        }
        let Some(mut predicted) = pf.predict_for(layer) else { return };
        mask_hot(experts.hot_tier(), layer, &mut predicted[..p]);
        let mut spare = [0u8; MAX_K];
        let mut spare = spare.chunks_mut(1);
        // SAFETY: the set's slots are disjoint from each other and from
        // the demand slots; the set's previous prefetch is drained by
        // `dispatch` before it writes, and its last GPU reader is done.
        let dsts: [&mut [u8]; MAX_K] = std::array::from_fn(|i| {
            if i < p { unsafe { slot_at(addr, es, first + i) } } else { spare.next().expect("MAX_K spares") }
        });
        pf.dispatch(layer, predicted, p, dsts, io_pool, experts);
    }

    /// Back to an empty sequence.
    pub fn reset(&mut self, b: &mut B) -> Result<()> {
        if let Some(p) = self.pf.as_mut() {
            p.invalidate_all();
        }
        for lb in &self.layers {
            if let LayerBufs::Gdn { conv, delta, conv_len, delta_len } = *lb {
                upload(b, conv, &vec![0.0f32; conv_len])?;
                upload(b, delta, &vec![0.0f32; delta_len])?;
            }
        }
        for ix in self.indexer.iter_mut().flatten() {
            ix.clear();
        }
        if let Some((_, _, st)) = self.ple.as_mut() {
            st.reset();
            if let Some(g) = &self.ple_gpu {
                upload(b, g.state, &st.conv)?;
            }
        }
        self.pending_index = None;
        self.pos = 0;
        Ok(())
    }

    /// `true` if the PLE runs on the host (`STREAMMOE_Q38_PLE=host` or
    /// [`Self::set_ple_on_host`]).
    pub fn ple_on_host(&self) -> bool {
        self.ple_host
    }

    /// Run the PLE on the host (`true`) or the backend (`false`, the
    /// default); the conv state moves with it.
    pub fn set_ple_on_host(&mut self, b: &mut B, on: bool) -> Result<()> {
        if on == self.ple_host {
            return Ok(());
        }
        if let (Some((_, _, st)), Some(g)) = (self.ple.as_mut(), &self.ple_gpu) {
            if on {
                st.conv = download(b, g.state, st.conv.len())?;
            } else {
                upload(b, g.state, &st.conv)?;
            }
        }
        self.ple_host = on;
        Ok(())
    }

    /// The PLE inject for `tokens` at the PLE layer: `hyper += ple(…)`.
    ///
    /// GPU (default): the n-gram rows are looked up on the host and
    /// uploaded; `key_proj` / `value_proj` (graph matvecs) and the
    /// gate / conv ([`PleOp`]) are queued on `steps` — no readback.
    /// The upload is safe while `steps` is pending: only these steps
    /// read `ple_e`, and the previous chunk's are complete. A chunk of
    /// `T` tokens is the same ops at `T` rows (the buffers hold a
    /// prefill chunk). The host path: flush, read the residual back, [`PleLayer::forward`] on the host (the conv
    /// state moved down and back), GPU add.
    fn ple_layer(
        &mut self,
        b: &mut B,
        wf: &WeightFile,
        io_pool: &rayon::ThreadPool,
        tokens: &[i32],
        steps: &mut Vec<Step>,
    ) -> Result<()> {
        let pc = ple_config();
        let (hh, t_n) = (pc.hc_hidden(), tokens.len());
        let (_, layer, state) = self.ple.as_mut().expect("PLE layer");
        let gpu_state = self.ple_gpu.as_ref().filter(|_| !self.ple_host);
        if let Some(g) = gpu_state {
            let e = layer.embedding(state, tokens, Some(io_pool))?;
            upload(b, g.e, &e)?;
            let (ed, h) = (pc.embed_dim as u32, pc.hidden as u32);
            let tn = t_n as u32;
            steps.push(matvec_n("q38.ple.key_proj", g.key, g.e, g.k, ed, hh as u32, tn));
            steps.push(matvec_n("q38.ple.value_proj", g.value, g.e, g.v, ed, h, tn));
            steps.push(Step::F(FamOp::Ple(PleOp::Gate {
                k: g.k,
                x: self.bufs.hyper,
                v: g.v,
                u: self.bufs.ple_out,
                norm_key_off: g.norm_key_off,
                norm_query_off: g.norm_query_off,
                hidden: h,
                hc: pc.hc as u32,
                n_tokens: tn,
                eps: pc.eps,
            })));
            steps.push(Step::F(FamOp::Ple(PleOp::Conv {
                u: self.bufs.ple_out,
                n: g.n,
                state: g.state,
                x: self.bufs.hyper,
                norm_conv_off: g.norm_conv_off,
                conv_off: g.conv_off,
                hidden: h,
                hc: pc.hc as u32,
                kernel: pc.conv_kernel as u32,
                dilation: pc.conv_dilation as u32,
                n_tokens: tn,
                eps: pc.eps,
            })));
            return Ok(());
        }
        self.flush(b, steps, "q38.layer")?;
        let hyper = download(b, self.bufs.hyper, t_n * hh)?;
        let mut out = vec![0.0f32; t_n * hh];
        let (pl, layer, state) = self.ple.as_mut().expect("PLE layer");
        let gpu_state = self.ple_gpu.as_ref().filter(|_| !self.ple_host);
        if let Some(g) = gpu_state {
            state.conv = download(b, g.state, state.conv.len())?;
        }
        let w = ple_weights(wf, *pl)?;
        layer.forward(&w, state, tokens, &hyper, &mut out, Some(io_pool))?;
        if let Some(g) = gpu_state {
            upload(b, g.state, &state.conv)?;
        }
        upload(b, self.bufs.ple_out, &out)?;
        steps.push(Step::F(FamOp::Add { dst: self.bufs.hyper, src: self.bufs.ple_out, n: (t_n * hh) as u32 }));
        Ok(())
    }

    /// The Gated Residual mix of `hc` from `hyper` into `mixed` (and
    /// the injection weights into `inj` when it has a `block_inject`),
    /// `n` token rows.
    fn hc_steps(&self, hc: &HcRefs, n_tok: u32, steps: &mut Vec<Step>) {
        let c = &self.cfg;
        let bu = &self.bufs;
        let (h, n) = (c.hidden as u32, c.hc_count as u32);
        let width = h * n;
        let lr = c.hc_lowrank as u32;
        steps.push(Step::F(FamOp::HcNorm {
            x: bu.hyper,
            gain_off: hc.gain_off,
            out: bu.normed,
            group: h,
            gain_len: width,
            rows: n_tok,
            eps: c.rms_eps,
        }));
        steps.push(matvec_n("q38.hc_mix_down", hc.down, bu.normed, bu.low, width, lr, n_tok));
        steps.push(Step::F(FamOp::SiluScale { v: bu.low, scale: 1.0 / n as f32, n: lr * n_tok }));
        steps.push(matvec_n("q38.hc_mix_up", hc.up, bu.low, bu.mix, lr, width, n_tok));
        steps.push(Step::F(FamOp::HcMix { mix: bu.mix, normed: bu.normed, out: bu.mixed, hidden: h, hc: n, n_tokens: n_tok }));
        if let Some(inj) = hc.inject {
            steps.push(matvec_n("q38.hc_block_inject", inj, bu.normed, bu.inj, width, n, n_tok));
            steps.push(Step::F(FamOp::InjWeights { inj: bu.inj, hc: n, n_tokens: n_tok }));
        }
    }

    fn inject_step(&self, n_tok: u32) -> Step {
        let c = &self.cfg;
        Step::F(FamOp::HcInject {
            hyper: self.bufs.hyper,
            block: self.bufs.block_out,
            inj: self.bufs.inj,
            hidden: c.hidden as u32,
            hc: c.hc_count as u32,
            n_tokens: n_tok,
        })
    }

    /// Gated DeltaNet on `mixed` into `block_out`, `n` tokens. The conv
    /// and the recurrence ops walk the chunk's tokens in order inside
    /// the kernel (the state carries from row to row), so a chunk
    /// equals its tokens one at a time.
    fn gdn_steps(&self, g: &GdnRefs, lb: &LayerBufs, n: u32, steps: &mut Vec<Step>) {
        let LayerBufs::Gdn { conv, delta, .. } = *lb else { unreachable!() };
        let c = &self.cfg;
        let bu = &self.bufs;
        let h = c.hidden as u32;
        let (hk, hv) = (c.linear_k_heads as u32, c.linear_v_heads as u32);
        let (dk, dv) = (c.linear_key_dim as u32, c.linear_value_dim as u32);
        let conv_dim = c.linear_conv_dim() as u32;
        steps.push(matvec_n("q38.gdn.in_proj_qkv", g.qkv, bu.mixed, bu.qkv, h, conv_dim, n));
        steps.push(matvec_n("q38.gdn.in_proj_z", g.z, bu.mixed, bu.z, h, hv * dv, n));
        steps.push(matvec_n("q38.gdn.in_proj_b", g.b, bu.mixed, bu.beta, h, hv, n));
        steps.push(matvec_n("q38.gdn.in_proj_a", g.a, bu.mixed, bu.alpha, h, hv, n));
        steps.push(Step::G(Op::Conv1dStepNTokens {
            label: "q38.gdn.conv1d",
            qkv_in: t(bu.qkv),
            conv_state: t(conv),
            weight_off: g.conv_off,
            conv_out: t(bu.conv_out),
            conv_dim,
            n_tokens: n,
        }));
        steps.push(Step::G(Op::RmsNormQkNTokens {
            label: "q38.gdn.qk_norm",
            x: t(bu.conv_out),
            num_k_heads: hk,
            key_dim: dk,
            key_offset_per_token: hk * dk,
            per_token_total: conv_dim,
            n_tokens: n,
        }));
        steps.push(Step::G(Op::ComputeDecayBetaNTokens {
            label: "q38.gdn.decay_beta",
            alpha_in: t(bu.alpha),
            beta_in: t(bu.beta),
            a_log_off: g.a_log_off,
            dt_bias_off: g.dt_bias_off,
            g_decay_out: t(bu.g_decay),
            beta_gate_out: t(bu.beta_gate),
            num_v_heads: hv,
            n_tokens: n,
        }));
        steps.push(Step::G(Op::GatedDeltaNetStepNTokens {
            label: "q38.gdn.recurrence",
            state: t(delta),
            conv_out: t(bu.conv_out),
            g_decay: t(bu.g_decay),
            beta_gate: t(bu.beta_gate),
            output: t(bu.delta_out),
            num_v_heads: hv,
            value_dim: dv,
            k_heads_per_v: hv / hk,
            n_tokens: n,
        }));
        steps.push(Step::F(FamOp::GatedRmsNormSigmoid {
            values: bu.delta_out,
            z: bu.z,
            weight_off: g.norm_off,
            out: bu.value_out,
            dim: dv,
            n_groups: hv * n,
            eps: c.rms_eps,
        }));
        steps.push(matvec_n("q38.gdn.out_proj", g.out, bu.value_out, bu.block_out, hv * dv, h, n));
    }

    fn qsa_bufs(&self, lb: &LayerBufs) -> QsaGraphBufs {
        let LayerBufs::Qsa { k_cache, v_cache } = *lb else { unreachable!() };
        let bu = &self.bufs;
        QsaGraphBufs {
            input: t(bu.mixed),
            q_proj: t(bu.q_proj),
            k: t(bu.k),
            v: t(bu.v),
            idx_qk: t(bu.idx_qk),
            q: t(bu.q),
            gate: t(bu.gate),
            attn_out: t(bu.attn_out),
            o_out: t(bu.block_out),
            inv_freq: t(bu.inv_freq),
            positions: None,
            k_cache: t(k_cache),
            v_cache: t(v_cache),
        }
    }

    /// Run `steps` (cleared) as one pass.
    fn flush(&self, b: &B, steps: &mut Vec<Step>, label: &'static str) -> Result<()> {
        if !steps.is_empty() {
            run_steps(b, &self.fam, steps, label)?;
            self.submits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            steps.clear();
        }
        Ok(())
    }

    /// Run the QSA indexer step a dense QSA layer left for later
    /// ([`Self::pending_index`]): read its `idx_qk` rows back (the
    /// submit that computed them has completed) and cache the raw
    /// keys.
    fn drain_pending_index(&mut self, b: &B) -> Result<()> {
        let Some((l, pos, n)) = self.pending_index.take() else { return Ok(()) };
        let AttnRefs::Qsa(q) = &self.w.layers[l].attn else { unreachable!() };
        let shape = self.qsa;
        let idx_qk = download(b, self.bufs.idx_qk, n * shape.idx_proj_dim())?;
        let positions: Vec<[i32; 3]> = (pos..pos + n).map(|p| [p as i32; 3]).collect();
        let cache = self.indexer[l].as_mut().expect("QSA layer has an indexer cache");
        qsa_indexer_step(&shape, &idx_qk, &positions, &q.idx_q_norm, cache);
        Ok(())
    }

    /// QSA on `mixed` into `block_out` for the `n` tokens at
    /// `self.pos..`.
    fn qsa_layer(&mut self, b: &mut B, l: usize, n: usize, steps: &mut Vec<Step>) -> Result<()> {
        self.drain_pending_index(b)?;
        let AttnRefs::Qsa(q) = &self.w.layers[l].attn else { unreachable!() };
        let qw = q.w;
        let idx_q_norm = q.idx_q_norm.clone();
        let bufs = self.qsa_bufs(&self.layers[l]);
        let shape = self.qsa;
        let pos = self.pos;
        let mut g = Graph::new();
        push_qsa_pre_sdpa(&mut g, &shape, &qw, &bufs, n, pos, pos as i32);
        for op in g.ops.drain(..) {
            // The graph's per-head norm applies a plain bf16 gain; this
            // family's Q / K norms are `1 + w` on the stored gain.
            match op {
                Op::RmsNormPerHeadNTokens { x, weight_off, num_heads, head_dim, n_tokens, eps, .. } => {
                    steps.push(Step::F(FamOp::HeadNormOnePlus {
                        x: retag(x),
                        gain_off: weight_off,
                        dim: head_dim,
                        n_groups: num_heads * n_tokens,
                        eps,
                    }));
                }
                op => steps.push(Step::G(op)),
            }
        }
        if !qsa_needs_sparse(&shape, self.mode, pos + n) {
            // Dense is exact: the indexer only has to cache the chunk's
            // raw keys, which can wait for the router's submit (no extra
            // command buffer).
            let mut g = Graph::new();
            push_qsa_sdpa(&mut g, &shape, &bufs, n, pos);
            steps.extend(g.ops.drain(..).map(Step::G));
            self.pending_index = Some((l, pos, n));
        } else {
            // Over the budget: the host scores the blocks and picks each
            // query's rows (it needs this chunk's indexer projection, so
            // one submit here); the GPU gathers them from the caches
            // (`FamOp::SparseRows`, `qsa::gpu`). Queries still within the
            // budget get all their rows.
            self.flush(b, steps, "q38.qsa_pre")?;
            let idx_qk = download(b, self.bufs.idx_qk, n * shape.idx_proj_dim())?;
            let positions: Vec<[i32; 3]> = (pos..pos + n).map(|p| [p as i32; 3]).collect();
            let cache = self.indexer[l].as_mut().expect("QSA layer has an indexer cache");
            let q_idx = qsa_indexer_step(&shape, &idx_qk, &positions, &idx_q_norm, cache);
            let lists = qsa_select_rows(&shape, self.mode, None, &q_idx, pos, n, cache);
            b.fam_set_rows(&mut self.fam, &lists);
            let LayerBufs::Qsa { k_cache, v_cache } = self.layers[l] else { unreachable!() };
            steps.push(Step::F(FamOp::SparseRows {
                q: self.bufs.q,
                gate: self.bufs.gate,
                k_cache,
                v_cache,
                out: self.bufs.attn_out,
                n_tokens: n as u32,
            }));
        }
        let mut g = Graph::new();
        push_qsa_o_proj(&mut g, &shape, &qw, &bufs, n);
        steps.extend(g.ops.drain(..).map(Step::G));
        Ok(())
    }

    /// MoE of a chunk (`n > 1` tokens) on `mixed` into `block_out`,
    /// then inject. The routing reads back every token's router logits
    /// (one submit), then groups the tokens by expert like the batched
    /// prefill of the other families (`build_expert_buckets`, one
    /// `Op::MoeBatchedPermuteFuse` per group): the experts the chunk
    /// picked are read into the `top_k` staging slots `top_k` at a time,
    /// each group adding its buckets into `out_sum`. Every group but
    /// the last is one more submit (the staging slots are reused).
    fn moe_layer_n(
        &mut self,
        b: &mut B,
        l: usize,
        n: usize,
        experts: &ExpertFiles,
        io_pool: &rayon::ThreadPool,
        steps: &mut Vec<Step>,
    ) -> Result<()> {
        let c = self.cfg.clone();
        let m = self.w.layers[l].moe;
        let bu = &self.bufs;
        let (h, si, k, ne) = (c.hidden as u32, c.shared_intermediate as u32, c.top_k, c.num_experts);
        let nt = n as u32;
        steps.push(matvec_n("q38.moe.router", m.router, bu.mixed, bu.router_logits, h, ne as u32, nt));
        steps.push(matvec_n("q38.moe.shared_expert_gate", m.shared_gate, bu.mixed, bu.shared_gate, h, 1, nt));
        steps.push(matvec_n("q38.moe.shared_gate_proj", m.gate, bu.mixed, bu.s_gate, h, si, nt));
        steps.push(matvec_n("q38.moe.shared_up_proj", m.up, bu.mixed, bu.s_up, h, si, nt));
        steps.push(Step::G(Op::SwigluFusedBatched {
            label: "q38.moe.shared_swiglu",
            gate: t(bu.s_gate),
            up: t(bu.s_up),
            out: t(bu.s_act),
            total: si * nt,
        }));
        steps.push(matvec_n("q38.moe.shared_down_proj", m.down, bu.s_act, bu.s_down, si, h, nt));
        self.flush(b, steps, "q38.layer")?;

        // Router (host): softmax, top-k, renormalized, per token.
        let mut scores = download(b, bu.router_logits, n * ne)?;
        let mut idx = vec![0i32; n * k];
        let mut wts = vec![0.0f32; n * k];
        for ((s, i), w) in scores.chunks_mut(ne).zip(idx.chunks_mut(k)).zip(wts.chunks_mut(k)) {
            moe_router_cpu_with(s, k, c.norm_topk_prob, i, w)?;
        }
        let buckets = build_expert_buckets(&idx, &wts, n, k, ne);
        let nb = buckets.expert_ids.len();
        for bi in 0..nb {
            let cnt = buckets.offsets[bi + 1] - buckets.offsets[bi];
            experts.record_pick(l, buckets.expert_ids[bi] as usize, cnt);
        }
        steps.push(Step::G(Op::ZeroBuffer { label: "q38.moe.out_sum_zero", buf: t(bu.out_sum), n_bytes: nt * h * 4 }));
        if let Some(pf) = self.pf.as_mut() {
            // The next decode token's guess: the chunk's last token's
            // picks, strongest first.
            let (li, lw) = (&idx[(n - 1) * k..], &wts[(n - 1) * k..]);
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_by(|&a, &b| lw[b].total_cmp(&lw[a]));
            let mut actual = [0i32; MAX_K];
            for (d, &i) in actual.iter_mut().zip(&order) {
                *d = li[i];
            }
            pf.record_actual(l, actual);
        }
        let es = self.expert_size;
        let mut start = 0;
        while start < nb {
            let end = (start + k).min(nb);
            let ids = &buckets.expert_ids[start..end];
            {
                let this = &*self;
                // SAFETY: slot `i` (< top_k) for the group's bucket `i`
                // only; no GPU work is in flight (the router flush or the
                // previous group's flush waited), and prefill issues no
                // prefetch into the demand slots.
                let dsts: Vec<&mut [u8]> = (0..ids.len()).map(|i| unsafe { this.slot(i) }).collect();
                io_pool.install(|| {
                    dsts.into_par_iter().zip(ids.par_iter()).try_for_each(|(dst, &e)| experts.read_expert(l, e as usize, dst))
                })?;
            }
            let bu = &self.bufs;
            let (a0, a1) = (buckets.offsets[start] as usize, buckets.offsets[end] as usize);
            let group = ExpertBuckets {
                expert_ids: ids.to_vec(),
                offsets: buckets.offsets[start..=end].iter().map(|&o| o - a0 as u32).collect(),
                token_idx: buckets.token_idx[a0..a1].to_vec(),
                weights: buckets.weights[a0..a1].to_vec(),
            };
            let mut expert_indices = vec![0u32; a1 - a0];
            for bi in 0..ids.len() {
                let (s, e) = (group.offsets[bi] as usize, group.offsets[bi + 1] as usize);
                expert_indices[s..e].fill(bi as u32);
            }
            upload(b, bu.bucket_token_idx, &group.token_idx)?;
            upload(b, bu.bucket_weights, &group.weights)?;
            upload(b, bu.expert_indices, &expert_indices)?;
            steps.push(Step::F(FamOp::GatherRows {
                src: bu.mixed,
                idx: bu.bucket_token_idx,
                dst: bu.bucket_input,
                width: h,
                n_rows: (a1 - a0) as u32,
            }));
            steps.push(Step::G(Op::MoeBatchedPermuteFuse {
                label: "q38.moe.permute_fuse",
                expert_base: t(bu.expert_base),
                expert_stride: es as u64,
                expert_indices: t(bu.expert_indices),
                expert_slots: (0..ids.len() as u32).collect(),
                bucket_input: t(bu.bucket_input),
                bucket_gate: t(bu.bucket_gate),
                bucket_up: t(bu.bucket_up),
                bucket_act: t(bu.bucket_act),
                bucket_out: t(bu.bucket_out),
                bucket_token_idx: t(bu.bucket_token_idx),
                bucket_weights: t(bu.bucket_weights),
                out_sum: t(bu.out_sum),
                buckets: group,
            }));
            if end < nb {
                self.flush(b, steps, "q38.moe_group")?;
            }
            start = end;
        }
        steps.push(Step::G(Op::MoeCombineResidualNTokens {
            label: "q38.moe.combine",
            h_mid: t(bu.zeros),
            moe_sum: t(bu.out_sum),
            shared_out: t(bu.s_down),
            shared_gate: t(bu.shared_gate),
            hidden_out: t(bu.block_out),
            n_tokens: nt,
            dim: h,
        }));
        steps.push(self.inject_step(nt));
        Ok(())
    }

    /// MoE on `mixed` (already mixed by the MoE Gated Residual, steps
    /// queued) into `block_out`, then inject.
    fn moe_layer(
        &mut self,
        b: &mut B,
        l: usize,
        experts: &ExpertFiles,
        io_pool: &rayon::ThreadPool,
        steps: &mut Vec<Step>,
    ) -> Result<()> {
        let c = self.cfg.clone();
        let m = self.w.layers[l].moe;
        let bu = &self.bufs;
        let (h, si, k) = (c.hidden as u32, c.shared_intermediate as u32, c.top_k);
        steps.push(matvec("q38.moe.router", m.router, bu.mixed, bu.router_logits, h, c.num_experts as u32));
        steps.push(matvec("q38.moe.shared_expert_gate", m.shared_gate, bu.mixed, bu.shared_gate, h, 1));
        steps.push(matvec("q38.moe.shared_gate_proj", m.gate, bu.mixed, bu.s_gate, h, si));
        steps.push(matvec("q38.moe.shared_up_proj", m.up, bu.mixed, bu.s_up, h, si));
        steps.push(Step::G(Op::SwigluFusedBatched {
            label: "q38.moe.shared_swiglu",
            gate: t(bu.s_gate),
            up: t(bu.s_up),
            out: t(bu.s_act),
            total: si,
        }));
        steps.push(matvec("q38.moe.shared_down_proj", m.down, bu.s_act, bu.s_down, si, h));
        self.flush(b, steps, "q38.layer")?;
        if self.pf.as_ref().is_some_and(|p| p.lookahead()) {
            // Layer l+1's set was last bound by layer l-1's experts,
            // which the flush above completed.
            self.prefetch_dispatch(l + 1, experts, io_pool);
        }
        let bu = &self.bufs;

        // Router (host): softmax, top-k, renormalized.
        let mut scores = download(b, bu.router_logits, c.num_experts)?;
        let mut idx = vec![0i32; k];
        let mut wts = vec![0.0f32; k];
        moe_router_cpu_with(&mut scores, k, c.norm_topk_prob, &mut idx, &mut wts)?;
        let buckets = build_expert_buckets(&idx, &wts, 1, k, c.num_experts);

        // Routed experts: bucket i's expert from the hot tier or disk
        // into staging slot i, or bound in place from the layer's
        // prefetch set.
        let ids = buckets.expert_ids.clone();
        for &e in &ids {
            experts.record_pick(l, e as usize, 1);
        }
        let io_t0 = std::time::Instant::now();
        let first_pf = self.pf_first_slot(l);
        let status = self.pf.as_mut().and_then(|p| p.begin_layer(l));
        let hot = experts.hot_tier();
        let mut expert_slots: Vec<u32> = (0..ids.len() as u32).collect();
        let mut pf_buf: Vec<Option<usize>> = vec![None; ids.len()];
        let (mut hot_n, mut hit_n) = (0u64, 0u64);
        for (bi, &e) in ids.iter().enumerate() {
            if hot.is_some_and(|h| h.contains(l, e as usize)) {
                hot_n += 1;
                continue;
            }
            if let Some(st) = &status {
                if let Some(j) = st.loaded_indices[..st.k].iter().position(|&x| x == e) {
                    pf_buf[bi] = Some(j);
                    expert_slots[bi] = (first_pf + j) as u32;
                    hit_n += 1;
                }
            }
        }
        {
            let this = &*self;
            // SAFETY: slot `bi` (< top_k) for bucket `bi` only; no GPU
            // work is in flight (the flush above waited).
            let dsts: Vec<&mut [u8]> = (0..ids.len()).map(|bi| unsafe { this.slot(bi) }).collect();
            io_pool.install(|| {
                dsts.into_par_iter().zip(ids.par_iter()).zip(pf_buf.par_iter()).try_for_each(|((dst, &e), pb)| {
                    if pb.is_none() { experts.read_expert(l, e as usize, dst) } else { Ok(()) }
                })
            })?;
        }
        for bi in 0..ids.len() {
            let Some(j) = pf_buf[bi] else { continue };
            let ok = self.pf.as_mut().is_some_and(|p| p.wait_slot(l, j));
            if !ok {
                // SAFETY: as above.
                experts.read_expert_disk(l, ids[bi] as usize, unsafe { self.slot(bi) })?;
                expert_slots[bi] = bi as u32;
                hit_n -= 1;
            }
        }
        experts.counters().record_io_wait(io_t0.elapsed().as_nanos() as u64);
        if let Some(pf) = self.pf.as_mut() {
            let n = ids.len() as u64;
            pf.record_outcome(hit_n, n - hit_n - hot_n);
            experts.counters().record_decode_layer(l, hot_n, hit_n);
            if let Some(prev) = pf.predict_for(l) {
                experts.counters().record_guess(Guesser::PrevToken, k as u64, guess_overlap(&prev[..k], &idx));
            }
            // Strongest pick first: a prefetch of the top few reads the
            // likeliest repeats.
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_by(|&a, &b| wts[b].total_cmp(&wts[a]));
            let mut actual = [0i32; MAX_K];
            for (d, &i) in actual.iter_mut().zip(&order) {
                *d = idx[i];
            }
            pf.record_actual(l, actual);
        }
        let bu = &self.bufs;
        let n_assign = buckets.token_idx.len();
        let mut expert_indices = vec![0u32; n_assign];
        for bi in 0..ids.len() {
            let (s, e) = (buckets.offsets[bi] as usize, buckets.offsets[bi + 1] as usize);
            expert_indices[s..e].fill(expert_slots[bi]);
        }
        upload(b, bu.bucket_token_idx, &buckets.token_idx)?;
        upload(b, bu.bucket_weights, &buckets.weights)?;
        upload(b, bu.expert_indices, &expert_indices)?;
        // Each assignment row gets its token's MoE input, on the GPU.
        steps.push(Step::F(FamOp::GatherRows {
            src: bu.mixed,
            idx: bu.bucket_token_idx,
            dst: bu.bucket_input,
            width: h,
            n_rows: n_assign as u32,
        }));
        steps.push(Step::G(Op::ZeroBuffer { label: "q38.moe.out_sum_zero", buf: t(bu.out_sum), n_bytes: h * 4 }));
        steps.push(Step::G(Op::MoeBatchedPermuteFuse {
            label: "q38.moe.permute_fuse",
            expert_base: t(bu.expert_base),
            expert_stride: self.expert_size as u64,
            expert_indices: t(bu.expert_indices),
            expert_slots,
            bucket_input: t(bu.bucket_input),
            bucket_gate: t(bu.bucket_gate),
            bucket_up: t(bu.bucket_up),
            bucket_act: t(bu.bucket_act),
            bucket_out: t(bu.bucket_out),
            bucket_token_idx: t(bu.bucket_token_idx),
            bucket_weights: t(bu.bucket_weights),
            out_sum: t(bu.out_sum),
            buckets,
        }));
        // Combine against zero: this family's MoE output carries no
        // residual term — the inject adds it to each copy.
        steps.push(Step::G(Op::MoeCombineResidualNTokens {
            label: "q38.moe.combine",
            h_mid: t(bu.zeros),
            moe_sum: t(bu.out_sum),
            shared_out: t(bu.s_down),
            shared_gate: t(bu.shared_gate),
            hidden_out: t(bu.block_out),
            n_tokens: 1,
            dim: h,
        }));
        steps.push(self.inject_step(1));
        Ok(())
    }

    /// One token at the next position. Writes the logits when `logits`
    /// is given (`[vocab]`).
    pub fn step(
        &mut self,
        b: &mut B,
        wf: &WeightFile,
        experts: &ExpertFiles,
        io_pool: &rayon::ThreadPool,
        token: i32,
        logits: Option<&mut [f32]>,
    ) -> Result<()> {
        self.forward_chunk(b, wf, experts, io_pool, &[token], logits)
    }

    /// `tokens` (at most the buffers' rows) at the next positions in
    /// one pass of every layer; the last token's logits into `logits`
    /// when given. One token is the decode step.
    pub fn forward_chunk(
        &mut self,
        b: &mut B,
        wf: &WeightFile,
        experts: &ExpertFiles,
        io_pool: &rayon::ThreadPool,
        tokens: &[i32],
        logits: Option<&mut [f32]>,
    ) -> Result<()> {
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }
        if n > self.max_chunk {
            return Err(Q38EngineError::Other(format!("chunk of {n} tokens, buffers hold {}", self.max_chunk)));
        }
        if self.pos + n > self.n_ctx {
            return Err(Q38EngineError::Other(format!("context full ({} tokens)", self.n_ctx)));
        }
        let c = self.cfg.clone();
        let nt = n as u32;
        let mut emb = vec![0.0f32; n * c.hidden];
        for (&tok, row) in tokens.iter().zip(emb.chunks_mut(c.hidden)) {
            embed_lookup(wf, tok, row).map_err(|e| Q38EngineError::Other(format!("embedding: {e}")))?;
        }
        upload(b, self.bufs.emb, &emb)?;
        let mut steps = Vec::new();
        steps.push(Step::F(FamOp::HcExpand {
            h: self.bufs.emb,
            hyper: self.bufs.hyper,
            hidden: c.hidden as u32,
            hc: c.hc_count as u32,
            n_tokens: nt,
        }));
        for l in 0..c.num_layers {
            if n == 1 {
                // Speculative reads are a decode thing: a chunk reads
                // every expert it routes to anyway.
                self.prefetch_dispatch(l, experts, io_pool);
            }
            if self.ple.as_ref().is_some_and(|p| p.0 == l) {
                self.ple_layer(b, wf, io_pool, tokens, &mut steps)?;
            }
            let lr = self.w.layers[l].clone();
            self.hc_steps(&lr.attn_hc, nt, &mut steps);
            match &lr.attn {
                AttnRefs::Gdn(g) => self.gdn_steps(g, &self.layers[l], nt, &mut steps),
                AttnRefs::Qsa(_) => self.qsa_layer(b, l, n, &mut steps)?,
            }
            steps.push(self.inject_step(nt));
            self.hc_steps(&lr.mlp_hc, nt, &mut steps);
            if n == 1 {
                self.moe_layer(b, l, experts, io_pool, &mut steps)?;
            } else {
                self.moe_layer_n(b, l, n, experts, io_pool, &mut steps)?;
            }
        }
        // Final mixer → LM head on the last row; no final norm in this
        // family. Without logits the mixer is skipped (it changes no
        // state).
        if logits.is_some() {
            let mixer = self.w.mixer;
            self.hc_steps(&mixer, nt, &mut steps);
            steps.push(Step::G(Op::MatvecNTokens {
                label: "q38.lm_head",
                weight: self.w.lm_head,
                input: t(self.bufs.mixed),
                input_off: ((n - 1) * c.hidden * 4) as u64,
                output: t(self.bufs.logits),
                output_off: 0,
                in_dim: c.hidden as u32,
                out_dim: c.vocab as u32,
                n_tokens: 1,
            }));
        }
        self.flush(b, &mut steps, "q38.tail")?;
        self.drain_pending_index(b)?;
        self.drain_prefetch();
        if let Some(out) = logits {
            let v = download(b, self.bufs.logits, c.vocab)?;
            out.copy_from_slice(&v);
        }
        self.pos += n;
        Ok(())
    }

    /// `tokens` at the next positions in chunks of
    /// [`Self::prefill_chunk`] tokens ([`Self::forward_chunk`]); the
    /// last one's logits into `logits` when given.
    pub fn eval(
        &mut self,
        b: &mut B,
        wf: &WeightFile,
        experts: &ExpertFiles,
        io_pool: &rayon::ThreadPool,
        tokens: &[i32],
        mut logits: Option<&mut [f32]>,
    ) -> Result<()> {
        if self.pos + tokens.len() > self.n_ctx {
            return Err(Q38EngineError::Other(format!("context full ({} tokens)", self.n_ctx)));
        }
        let chunks = tokens.chunks(self.chunk);
        let last = chunks.len().saturating_sub(1);
        for (i, ch) in chunks.enumerate() {
            let out = if i == last { logits.as_deref_mut() } else { None };
            self.forward_chunk(b, wf, experts, io_pool, ch, out)?;
        }
        Ok(())
    }

    // --- Snapshots -----------------------------------------------------------

    /// Bytes of [`Self::state_save`] at the current position.
    pub fn state_size(&self) -> usize {
        self.state_size_at(self.pos)
    }

    /// Bytes of a snapshot at position `pos` (`n_ctx` for the largest).
    pub fn state_size_at(&self, pos: usize) -> usize {
        state_size_at(pos)
    }

    /// Serialize the sequence state (`Q38S` v1): position, per layer
    /// the Gated DeltaNet conv + recurrence state or the QSA K / V rows
    /// and indexer raw keys, then the PLE state.
    pub fn state_save(&self, b: &B, buf: &mut [u8]) -> Result<usize> {
        let need = self.state_size();
        if buf.len() < need {
            return Err(SnapErr::BufferTooSmall { need, got: buf.len() }.into());
        }
        let mut out: Vec<u8> = Vec::with_capacity(need);
        out.extend_from_slice(&Q38_STATE_MAGIC.to_le_bytes());
        out.extend_from_slice(&Q38_STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.pos as u32).to_le_bytes());
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        let kvd = self.qsa.kv_dim();
        for (l, lb) in self.layers.iter().enumerate() {
            match *lb {
                LayerBufs::Gdn { conv, delta, conv_len, delta_len } => {
                    out.extend_from_slice(bytemuck::cast_slice(&download(b, conv, conv_len)?));
                    out.extend_from_slice(bytemuck::cast_slice(&download(b, delta, delta_len)?));
                }
                LayerBufs::Qsa { k_cache, v_cache } => {
                    out.extend_from_slice(bytemuck::cast_slice(&download(b, k_cache, self.pos * kvd)?));
                    out.extend_from_slice(bytemuck::cast_slice(&download(b, v_cache, self.pos * kvd)?));
                    let ix = self.indexer[l].as_ref().expect("indexer");
                    out.extend_from_slice(bytemuck::cast_slice(ix.raw_keys()));
                }
            }
        }
        if let Some((_, _, st)) = &self.ple {
            let mut ps = Vec::new();
            match self.ple_gpu.as_ref().filter(|_| !self.ple_host) {
                Some(g) => {
                    let mut st = st.clone();
                    st.conv = download(b, g.state, st.conv.len())?;
                    st.save(&mut ps);
                }
                None => st.save(&mut ps),
            }
            out.extend_from_slice(&(ps.len() as u64).to_le_bytes());
            out.extend_from_slice(&ps);
        }
        debug_assert_eq!(out.len(), need);
        buf[..out.len()].copy_from_slice(&out);
        Ok(out.len())
    }

    /// Restore a [`Self::state_save`] snapshot. Checks the whole buffer
    /// before changing anything.
    pub fn state_load(&mut self, b: &mut B, buf: &[u8]) -> Result<()> {
        let u32_at = |o: usize| -> Result<u32> {
            buf.get(o..o + 4)
                .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
                .ok_or(Q38EngineError::Snapshot(SnapErr::Truncated { layer: 0, need: o + 4, got: buf.len() }))
        };
        let magic = u32_at(0)?;
        if magic != Q38_STATE_MAGIC {
            return Err(SnapErr::BadMagic { got: magic, want: Q38_STATE_MAGIC }.into());
        }
        let version = u32_at(4)?;
        if version != Q38_STATE_VERSION {
            return Err(SnapErr::BadVersion { got: version, want: Q38_STATE_VERSION }.into());
        }
        let pos = u32_at(8)? as usize;
        let n_layers = u32_at(12)?;
        if n_layers as usize != self.layers.len() {
            return Err(SnapErr::ShapeMismatch { field: "num_layers", got: n_layers, want: self.layers.len() as u32 }
                .into());
        }
        if pos > self.n_ctx {
            return Err(SnapErr::LenOverflow { layer: 0, len: pos as i32, max: self.n_ctx }.into());
        }
        let need = self.state_size_at(pos);
        if buf.len() < need {
            return Err(SnapErr::Truncated { layer: 0, need, got: buf.len() }.into());
        }
        let kvd = self.qsa.kv_dim();
        let idd = self.qsa.idx_head_dim;
        let mut o = 16;
        let mut take = |n_f32: usize| -> Vec<f32> {
            let v: Vec<f32> = buf[o..o + 4 * n_f32]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            o += 4 * n_f32;
            v
        };
        let mut layer_data = Vec::with_capacity(self.layers.len());
        for lb in &self.layers {
            layer_data.push(match *lb {
                LayerBufs::Gdn { conv_len, delta_len, .. } => (take(conv_len), take(delta_len), Vec::new()),
                LayerBufs::Qsa { .. } => (take(pos * kvd), take(pos * kvd), take(pos * idd)),
            });
        }
        if let Some((_, _, st)) = self.ple.as_mut() {
            let n = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap()) as usize;
            if n != st.state_bytes() {
                return Err(SnapErr::ShapeMismatch { field: "ple_state", got: n as u32, want: st.state_bytes() as u32 }
                    .into());
            }
            // Fixed size and inside `need`: cannot fail after the checks.
            st.load(&buf[o + 8..o + 8 + n])?;
            if let Some(g) = &self.ple_gpu {
                upload(b, g.state, &st.conv)?;
            }
        }
        for (l, (lb, (a, bb, ix))) in self.layers.iter().zip(layer_data).enumerate() {
            match *lb {
                LayerBufs::Gdn { conv, delta, .. } => {
                    upload(b, conv, &a)?;
                    upload(b, delta, &bb)?;
                }
                LayerBufs::Qsa { k_cache, v_cache } => {
                    upload(b, k_cache, &a)?;
                    upload(b, v_cache, &bb)?;
                    let positions: Vec<[i32; 3]> = (0..pos as i32).map(|p| [p, p, p]).collect();
                    self.indexer[l].as_mut().expect("indexer").restore(&ix, &positions);
                }
            }
        }
        self.pos = pos;
        Ok(())
    }
}

impl<B: Q38Exec> Drop for Q38Engine<B> {
    fn drop(&mut self) {
        // No prefetch worker may outlive the staging it writes.
        self.drain_prefetch();
    }
}

/// Bytes of a [`Q38Engine::state_save`] snapshot at position `pos`
/// (`n_ctx` for the largest; sizes the checkpoint buffers at open).
pub fn state_size_at(pos: usize) -> usize {
    let c = q38_config();
    let s = qsa_shape();
    let mut n = 16;
    for l in 0..c.num_layers {
        n += match c.layer_kind(l) {
            Q38LayerKind::LinearAttn => {
                4 * ((c.conv_kernel - 1) * c.linear_conv_dim()
                    + c.linear_v_heads * c.linear_value_dim * c.linear_key_dim)
            }
            Q38LayerKind::Qsa => 4 * pos * (2 * s.kv_dim() + s.idx_head_dim),
        };
    }
    if !c.ple_layers.is_empty() {
        n += 8 + PleState::new(&ple_config(), &ngram_config()).state_bytes();
    }
    n
}

/// `Q38S` little-endian.
pub const Q38_STATE_MAGIC: u32 = u32::from_le_bytes(*b"Q38S");
pub const Q38_STATE_VERSION: u32 = 1;

/// Staging slot `i` of `es` bytes at host address `addr`.
///
/// # Safety
/// `addr` is the start of the engine's staging buffer (alive, never
/// moved, at least `(i + 1) · es` bytes); no other live slice of slot
/// `i` and no GPU work reading it in flight.
unsafe fn slot_at<'a>(addr: usize, es: usize, i: usize) -> &'a mut [u8] {
    // SAFETY: the caller's contract.
    unsafe { std::slice::from_raw_parts_mut((addr + i * es) as *mut u8, es) }
}

/// A one-token matvec step.
fn matvec(label: &'static str, w: WeightRef, input: FBuf, output: FBuf, in_dim: u32, out_dim: u32) -> Step {
    matvec_n(label, w, input, output, in_dim, out_dim, 1)
}

/// A matvec step over `n_tokens` rows (`[n, in_dim] → [n, out_dim]`).
fn matvec_n(label: &'static str, w: WeightRef, input: FBuf, output: FBuf, in_dim: u32, out_dim: u32, n_tokens: u32) -> Step {
    Step::G(Op::MatvecNTokens {
        label,
        weight: w,
        input: t(input),
        input_off: 0,
        output: t(output),
        output_off: 0,
        in_dim,
        out_dim,
        n_tokens,
    })
}
