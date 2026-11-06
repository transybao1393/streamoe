//! Shapes of the Qwen3.8-Flash-Next family that [`Variant`] cannot
//! carry, and a runtime shape struct for the family's CPU path.
//!
//! [`VARIANT`] holds what the family shares with qwen3_5_moe. The
//! consts here are the rest, from `text_config` in
//! `~/models/qwen3-8-flash-next/meta/config.json`. [`Q38Config`] puts
//! both in one runtime struct so the family's layer code also runs at
//! the tiny random-weight sizes the tests use
//! ([`Q38Config::REAL`] is the real model).
//!
//! [`Variant`]: crate::riir::variants::Variant

use crate::riir::variants::{MROPE_SECTION, RMS_NORM_EPS, ROPE_THETA, VARIANT};

/// Copies of the hidden state in the residual stream (`hc_count`).
pub const HC_COUNT: usize = 4;
/// Rank of the Gated Residual's input-mix gate (`hc_lowrank`).
#[cfg(not(feature = "model-qwen3-8-flash-next-tiny"))]
pub const HC_LOWRANK: usize = 320;
/// Test-only tiny build (`model-qwen3-8-flash-next-tiny`): the
/// `make_tiny.py --shapes engine` config. Every tiny value below
/// follows it.
#[cfg(feature = "model-qwen3-8-flash-next-tiny")]
pub const HC_LOWRANK: usize = 32;
/// Width of the residual stream: `hc_count * hidden_size` = 10240.
pub const HC_HIDDEN: usize = HC_COUNT * VARIANT.hidden_dim;

/// Gated DeltaNet head dims and conv width (`linear_key_head_dim`,
/// `linear_value_head_dim`, `linear_conv_kernel_dim`) — the same as
/// `Variant::LINEAR_KEY_DIM` / `LINEAR_VALUE_DIM` / `CONV_KERNEL_SIZE`.
pub const LINEAR_KEY_DIM: usize = 128;
pub const LINEAR_VALUE_DIM: usize = 128;
pub const LINEAR_CONV_KERNEL: usize = 4;

/// Activation on `z` in the Gated DeltaNet output norm
/// (`Qwen4ExpTextRMSNormGated`). The family config sets
/// `output_gate_type = "sigmoid"`; qwen3_5_moe used SiLU (the engine's
/// `rms_norm_gated` / `gated_rms_norm` kernels).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAct {
    Sigmoid,
    Silu,
}
pub const GDN_OUTPUT_GATE: GateAct = GateAct::Sigmoid;

/// 0-based indices of the layers that add the n-gram per-layer
/// embedding. The config's `ple_layer_ids = [2]` is one-based
/// (`Qwen4ExpTextDecoderLayer` tests `layer_idx + 1 in ple_layer_ids`),
/// so PLE runs at layer index 1; the MLX weights are under
/// `language_model.model.layers.1.ple.*`.
pub const PLE_LAYERS: &[usize] = &[1];
/// PLE shapes (`ngram_size`, `heads_per_ngram`, `ple_embed_dim`,
/// `ple_conv_kernel_size`, `split_ngram_parts`,
/// `ngram_vocab_size_base`). Owned by the PLE module; listed here so
/// the whole config is in one place.
pub const NGRAM_SIZE: usize = 3;
#[cfg(not(feature = "model-qwen3-8-flash-next-tiny"))]
pub const HEADS_PER_NGRAM: usize = 8;
#[cfg(not(feature = "model-qwen3-8-flash-next-tiny"))]
pub const PLE_EMBED_DIM: usize = 2560;
pub const PLE_CONV_KERNEL: usize = 4;
#[cfg(not(feature = "model-qwen3-8-flash-next-tiny"))]
pub const SPLIT_NGRAM_PARTS: usize = 128;
#[cfg(not(feature = "model-qwen3-8-flash-next-tiny"))]
pub const NGRAM_VOCAB_SIZE_BASE: usize = 20_000_000;
#[cfg(feature = "model-qwen3-8-flash-next-tiny")]
pub const HEADS_PER_NGRAM: usize = 2;
#[cfg(feature = "model-qwen3-8-flash-next-tiny")]
pub const PLE_EMBED_DIM: usize = 128;
#[cfg(feature = "model-qwen3-8-flash-next-tiny")]
pub const SPLIT_NGRAM_PARTS: usize = 4;
#[cfg(feature = "model-qwen3-8-flash-next-tiny")]
pub const NGRAM_VOCAB_SIZE_BASE: usize = 1000;
/// `make_ngram_vocab_size_divisible_by` and `seed` (the released config
/// leaves `seed` at its default, 1234).
pub const NGRAM_VOCAB_DIVISIBLE_BY: usize = 128;
pub const NGRAM_SEED: u64 = 1234;

/// QSA indexer (`indexer_n_heads`, `indexer_head_dim`,
/// `indexer_kv_heads`, `indexer_budget`, `indexer_compress_ratio`).
/// Owned by the QSA module. Up to `INDEXER_BUDGET` tokens of context
/// every block is selected and QSA equals dense attention.
pub const INDEXER_N_HEADS: usize = 4;
#[cfg(not(feature = "model-qwen3-8-flash-next-tiny"))]
pub const INDEXER_HEAD_DIM: usize = 128;
#[cfg(not(feature = "model-qwen3-8-flash-next-tiny"))]
pub const INDEXER_BUDGET: usize = 2048;
#[cfg(feature = "model-qwen3-8-flash-next-tiny")]
pub const INDEXER_HEAD_DIM: usize = 32;
#[cfg(feature = "model-qwen3-8-flash-next-tiny")]
pub const INDEXER_BUDGET: usize = 8;
pub const INDEXER_KV_HEADS: usize = 1;
pub const INDEXER_COMPRESS_RATIO: usize = 4;

/// MLX quantization: everything 4-bit with group size 32, except the
/// router (`mlp.gate`) and `mlp.shared_expert_gate`, 8-bit group 64.
pub const QUANT_GROUP_SIZE: usize = 32;
pub const ROUTER_BITS: usize = 8;
pub const ROUTER_GROUP_SIZE: usize = 64;

/// Layer kind of this family. `Qsa` is the config's `full_attention`
/// (`indexed_attention` in transformers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q38LayerKind {
    LinearAttn,
    Qsa,
}

/// Runtime shapes of the family's text model. The real model is
/// [`Q38Config::REAL`]; tests build tiny ones.
#[derive(Debug, Clone, PartialEq)]
pub struct Q38Config {
    pub hidden: usize,
    pub num_layers: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub full_attn_interval: usize,
    pub linear_k_heads: usize,
    pub linear_v_heads: usize,
    pub linear_key_dim: usize,
    pub linear_value_dim: usize,
    pub conv_kernel: usize,
    pub attn_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Channels RoPE rotates per head (`head_dim * partial_rotary_factor`).
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
    pub num_experts: usize,
    pub top_k: usize,
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    pub norm_topk_prob: bool,
    pub vocab: usize,
    pub rms_eps: f32,
    pub gdn_gate: GateAct,
    /// 0-based layers with PLE (see [`PLE_LAYERS`]).
    pub ple_layers: &'static [usize],
}

impl Q38Config {
    /// Qwen3.8-Flash-Next as built ([`VARIANT`] + the consts above).
    pub const REAL: Q38Config = Q38Config {
        hidden: VARIANT.hidden_dim,
        num_layers: VARIANT.num_layers,
        hc_count: HC_COUNT,
        hc_lowrank: HC_LOWRANK,
        full_attn_interval: VARIANT.full_attn_interval,
        linear_k_heads: VARIANT.linear_num_k_heads,
        linear_v_heads: VARIANT.linear_num_v_heads,
        linear_key_dim: LINEAR_KEY_DIM,
        linear_value_dim: LINEAR_VALUE_DIM,
        conv_kernel: LINEAR_CONV_KERNEL,
        attn_heads: VARIANT.num_attn_heads,
        kv_heads: VARIANT.num_kv_heads,
        head_dim: VARIANT.head_dim,
        rotary_dim: VARIANT.gqa_rotary_dim,
        rope_theta: ROPE_THETA,
        mrope_section: match MROPE_SECTION {
            Some(s) => s,
            None => [0; 3],
        },
        num_experts: VARIANT.num_experts,
        top_k: VARIANT.num_experts_per_tok,
        moe_intermediate: VARIANT.moe_intermediate,
        shared_intermediate: VARIANT.shared_intermediate,
        norm_topk_prob: VARIANT.norm_topk_prob,
        vocab: VARIANT.vocab_size,
        rms_eps: RMS_NORM_EPS,
        gdn_gate: GDN_OUTPUT_GATE,
        ple_layers: PLE_LAYERS,
    };

    /// `hc_count * hidden`.
    pub const fn hc_hidden(&self) -> usize {
        self.hc_count * self.hidden
    }
    pub const fn linear_key_total(&self) -> usize {
        self.linear_k_heads * self.linear_key_dim
    }
    pub const fn linear_value_total(&self) -> usize {
        self.linear_v_heads * self.linear_value_dim
    }
    /// Gated DeltaNet conv channels: Q + K + V.
    pub const fn linear_conv_dim(&self) -> usize {
        2 * self.linear_key_total() + self.linear_value_total()
    }
    /// `layer_types`: `(i + 1) % full_attention_interval == 0` is QSA.
    pub const fn layer_kind(&self, layer: usize) -> Q38LayerKind {
        if (layer + 1) % self.full_attn_interval == 0 {
            Q38LayerKind::Qsa
        } else {
            Q38LayerKind::LinearAttn
        }
    }
    pub fn has_ple(&self, layer: usize) -> bool {
        self.ple_layers.contains(&layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real config against `text_config` in the MLX `config.json`.
    #[cfg(not(feature = "model-qwen3-8-flash-next-tiny"))]
    #[test]
    fn real_config_matches_text_config() {
        let c = Q38Config::REAL;
        assert_eq!((c.hidden, c.num_layers, c.hc_count, c.hc_lowrank), (2560, 48, 4, 320));
        assert_eq!(c.hc_hidden(), 10240);
        assert_eq!((c.linear_k_heads, c.linear_v_heads, c.linear_key_dim, c.linear_value_dim), (16, 48, 128, 128));
        assert_eq!(c.linear_conv_dim(), 2 * 2048 + 6144);
        assert_eq!((c.attn_heads, c.kv_heads, c.head_dim, c.rotary_dim), (24, 2, 256, 64));
        assert_eq!(c.rope_theta, 1e7);
        assert_eq!(c.mrope_section, [11, 11, 10]);
        assert_eq!((c.num_experts, c.top_k, c.moe_intermediate, c.shared_intermediate), (512, 10, 640, 640));
        assert_eq!((c.vocab, c.rms_eps), (248320, 1e-6));
        assert_eq!(c.gdn_gate, GateAct::Sigmoid);
        // layer_types: 3 linear + 1 full, × 12.
        let kinds: Vec<_> = (0..48).map(|i| c.layer_kind(i)).collect();
        for (i, k) in kinds.iter().enumerate() {
            let want = if i % 4 == 3 { Q38LayerKind::Qsa } else { Q38LayerKind::LinearAttn };
            assert_eq!(*k, want, "layer {i}");
        }
        // PLE at one-based layer 2 = index 1, a linear-attention layer.
        assert!(c.has_ple(1) && !c.has_ple(2));
        assert_eq!(c.layer_kind(1), Q38LayerKind::LinearAttn);
        // rotary dim fits the QSA index head (config validation).
        assert!(c.rotary_dim <= INDEXER_HEAD_DIM);
        assert_eq!(INDEXER_BUDGET % INDEXER_COMPRESS_RATIO, 0);
        // PLE: ple_embed_dim divisible by (ngram_size - 1) * heads_per_ngram.
        assert_eq!(PLE_EMBED_DIM % ((NGRAM_SIZE - 1) * HEADS_PER_NGRAM), 0);
    }
}
