//! Weight names: the MLX checkpoint's tensors and what the engine
//! does with each.
//!
//! The MLX build (`model.safetensors.index.json`, 3767 tensors) names
//! the text model `language_model.*`; the engine's manifest drops that
//! prefix, as for the other Qwen variants (`model.layers.N.…`,
//! `model.embed_tokens.…`, `lm_head.…`). `vision_tower.*` (333) is not
//! loaded yet; the MLX build has no `mtp.*` tensors.
//!
//! Every quantized tensor is a `(weight, scales, biases)` triple:
//! U32-packed codes plus BF16 scales and biases per group.
//!
//! | MLX name (after `language_model.`) | Destination | Quant |
//! | --- | --- | --- |
//! | `model.embed_tokens` | resident (or streamed rows, `embed_stream`) | 4-bit g32 |
//! | `lm_head` | resident | 4-bit g32 |
//! | `model.hyper_connection_mixer.hc_norm.weight` | resident | BF16, gain `1 + w` |
//! | `model.hyper_connection_mixer.input_mix_weight_{down,up}` | resident | 4-bit g32 |
//! | `model.layers.N.{attn,mlp}_hyper_connection.hc_norm.weight` | resident | BF16, gain `1 + w` |
//! | `model.layers.N.{attn,mlp}_hyper_connection.input_mix_weight_{down,up}` | resident | 4-bit g32 |
//! | `model.layers.N.{attn,mlp}_hyper_connection.block_inject_weight` | resident | 4-bit g32 |
//! | `model.layers.N.linear_attn.in_proj_{qkv,z,a,b}`, `out_proj` | resident | 4-bit g32 |
//! | `model.layers.N.linear_attn.{conv1d.weight,A_log,dt_bias}` | resident | unquantized |
//! | `model.layers.N.linear_attn.norm.weight` | resident | unquantized, plain gain |
//! | `model.layers.N.self_attn.{q,k,v,o}_proj` | resident | 4-bit g32 |
//! | `model.layers.N.self_attn.{q,k}_norm.weight` | resident | gain `1 + w` |
//! | `model.layers.N.self_attn.indexer.index_qk_proj` | resident | 4-bit g32 (QSA) |
//! | `model.layers.N.self_attn.indexer.{q,k}_layernorm.weight` | resident | gain `1 + w` (QSA) |
//! | `model.layers.N.mlp.gate` | resident | 8-bit g64 |
//! | `model.layers.N.mlp.shared_expert_gate` | resident | 8-bit g64 |
//! | `model.layers.N.mlp.shared_expert.{gate,up,down}_proj` | resident | 4-bit g32 |
//! | `model.layers.N.mlp.switch_mlp.{gate,up,down}_proj` | packed expert files | 4-bit g32 |
//! | `model.layers.1.ple.ple_embedding.ngram_embedding.shards.S` (128) | streamed rows (PLE) | 4-bit g32 |
//! | `model.layers.1.ple.{key_proj,value_proj}` | resident (PLE) | 4-bit g32 |
//! | `model.layers.1.ple.{norm_key,norm_query,norm_conv}.weight` | resident (PLE) | gain `1 + w` |
//! | `model.layers.1.ple.conv1d.weight` | resident (PLE) | unquantized |
//! | `model.layers.1.ple.ple_embedding.{layer_multipliers,ngram_heads_offsets,ngram_heads_vocab_sizes}` | resident (PLE) | integer tables |
//!
//! Norm gains: the checkpoint stores `Qwen4ExpTextRMSNorm` gains
//! centered at zero and the model applies `1 + w` (148 tensors: 96
//! `hc_norm`, the mixer's, 24 `q_norm` / `k_norm`, 24 indexer norms and
//! PLE's 3). The MLX build keeps them that way (its README: "norm
//! tensors bit-identical 148 / 148" with the bf16 source). The engine
//! adds the 1 at run time ([`super::gated_residual::NormGain::OnePlus`]).
//! The Gated DeltaNet's `linear_attn.norm.weight`
//! (`Qwen4ExpTextRMSNormGated`) is a plain gain, as in qwen3_5_moe.
//!
//! No `input_layernorm`, `post_attention_layernorm` or `model.norm`
//! exists in this family: the Gated Residual's `hc_norm` takes their
//! place, and the final mixer's output goes straight to `lm_head`.

/// Where a tensor goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dest {
    /// `model_weights.bin`, mapped whole.
    Resident,
    /// Packed per-layer expert files (`switch_mlp`), streamed.
    Experts,
    /// PLE n-gram table shards, read row by row from disk.
    PleRows,
    /// Not loaded (vision tower, MTP) yet.
    Skip,
}

/// How a tensor is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    /// MLX affine 4-bit, group 32.
    Q4G32,
    /// MLX affine 8-bit, group 64 (router, shared-expert gate).
    Q8G64,
    /// Not quantized (BF16 / F32 / integer tables).
    Raw,
}

/// What the engine needs to know about one MLX tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightEntry {
    /// Name in the engine manifest (MLX name without `language_model.`
    /// and without the `.weight` / `.scales` / `.biases` suffix for
    /// quantized tensors' triples — see [`classify`]).
    pub engine_name: String,
    pub dest: Dest,
    pub quant: Quant,
    /// A `Qwen4ExpTextRMSNorm` gain: apply `1 + w`.
    pub one_plus_gain: bool,
}

const PREFIX: &str = "language_model.";

/// Classify one MLX tensor name. Returns `None` for a name this table
/// does not know — a new tensor in a newer checkpoint, which the
/// converter should refuse rather than drop.
pub fn classify(mlx_name: &str) -> Option<WeightEntry> {
    if mlx_name.starts_with("vision_tower.") || mlx_name.starts_with("mtp.") {
        return Some(WeightEntry {
            engine_name: mlx_name.to_string(),
            dest: Dest::Skip,
            quant: Quant::Raw,
            one_plus_gain: false,
        });
    }
    let name = mlx_name.strip_prefix(PREFIX)?;
    let entry = |dest, quant, one_plus_gain| {
        Some(WeightEntry {
            engine_name: name.to_string(),
            dest,
            quant,
            one_plus_gain,
        })
    };
    // Base name of a quantized triple.
    let base = name
        .strip_suffix(".weight")
        .or_else(|| name.strip_suffix(".scales"))
        .or_else(|| name.strip_suffix(".biases"));

    if let Some(b) = base {
        match b {
            "model.embed_tokens" | "lm_head" => return entry(Dest::Resident, Quant::Q4G32, false),
            "model.hyper_connection_mixer.input_mix_weight_down"
            | "model.hyper_connection_mixer.input_mix_weight_up" => {
                return entry(Dest::Resident, Quant::Q4G32, false);
            }
            _ => {}
        }
    }
    if name == "model.hyper_connection_mixer.hc_norm.weight" {
        return entry(Dest::Resident, Quant::Raw, true);
    }

    let rest = name.strip_prefix("model.layers.")?;
    let (layer, tail) = rest.split_once('.')?;
    layer.parse::<usize>().ok()?;

    // Unquantized per-layer tensors first (exact names).
    match tail {
        "attn_hyper_connection.hc_norm.weight"
        | "mlp_hyper_connection.hc_norm.weight"
        | "self_attn.q_norm.weight"
        | "self_attn.k_norm.weight"
        | "self_attn.indexer.q_layernorm.weight"
        | "self_attn.indexer.k_layernorm.weight"
        | "ple.norm_key.weight"
        | "ple.norm_query.weight"
        | "ple.norm_conv.weight" => return entry(Dest::Resident, Quant::Raw, true),
        "linear_attn.conv1d.weight"
        | "linear_attn.A_log"
        | "linear_attn.dt_bias"
        | "linear_attn.norm.weight"
        | "ple.conv1d.weight"
        | "ple.ple_embedding.layer_multipliers"
        | "ple.ple_embedding.ngram_heads_offsets"
        | "ple.ple_embedding.ngram_heads_vocab_sizes" => {
            return entry(Dest::Resident, Quant::Raw, false);
        }
        _ => {}
    }

    let tail_base = tail
        .strip_suffix(".weight")
        .or_else(|| tail.strip_suffix(".scales"))
        .or_else(|| tail.strip_suffix(".biases"))?;
    if tail_base.starts_with("ple.ple_embedding.ngram_embedding.shards.") {
        return entry(Dest::PleRows, Quant::Q4G32, false);
    }
    match tail_base {
        "mlp.gate" | "mlp.shared_expert_gate" => entry(Dest::Resident, Quant::Q8G64, false),
        "mlp.switch_mlp.gate_proj" | "mlp.switch_mlp.up_proj" | "mlp.switch_mlp.down_proj" => {
            entry(Dest::Experts, Quant::Q4G32, false)
        }
        "attn_hyper_connection.input_mix_weight_down"
        | "attn_hyper_connection.input_mix_weight_up"
        | "attn_hyper_connection.block_inject_weight"
        | "mlp_hyper_connection.input_mix_weight_down"
        | "mlp_hyper_connection.input_mix_weight_up"
        | "mlp_hyper_connection.block_inject_weight"
        | "linear_attn.in_proj_qkv"
        | "linear_attn.in_proj_z"
        | "linear_attn.in_proj_a"
        | "linear_attn.in_proj_b"
        | "linear_attn.out_proj"
        | "self_attn.q_proj"
        | "self_attn.k_proj"
        | "self_attn.v_proj"
        | "self_attn.o_proj"
        | "self_attn.indexer.index_qk_proj"
        | "mlp.shared_expert.gate_proj"
        | "mlp.shared_expert.up_proj"
        | "mlp.shared_expert.down_proj"
        | "ple.key_proj"
        | "ple.value_proj" => entry(Dest::Resident, Quant::Q4G32, false),
        _ => None,
    }
}

/// Engine names of one Gated Residual's tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatedResidualNames {
    /// BF16 gain, applied as `1 + w`.
    pub hc_norm: String,
    /// Quantized-triple base names (append `.weight` / `.scales` /
    /// `.biases`).
    pub mix_down: String,
    pub mix_up: String,
    /// `None` for the final mixer (`use_combine = False`).
    pub block_inject: Option<String>,
}

/// Which Gated Residual of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcSite {
    /// Before the attention block (`attn_hyper_connection`).
    Attn,
    /// Before the MoE (`mlp_hyper_connection`).
    Mlp,
}

/// Tensor names of the Gated Residual at `site` of `layer`.
pub fn gated_residual_names(layer: usize, site: HcSite) -> GatedResidualNames {
    let s = match site {
        HcSite::Attn => "attn_hyper_connection",
        HcSite::Mlp => "mlp_hyper_connection",
    };
    let p = format!("model.layers.{layer}.{s}");
    GatedResidualNames {
        hc_norm: format!("{p}.hc_norm.weight"),
        mix_down: format!("{p}.input_mix_weight_down"),
        mix_up: format!("{p}.input_mix_weight_up"),
        block_inject: Some(format!("{p}.block_inject_weight")),
    }
}

/// Tensor names of the final `hyper_connection_mixer`.
pub fn final_mixer_names() -> GatedResidualNames {
    let p = "model.hyper_connection_mixer";
    GatedResidualNames {
        hc_norm: format!("{p}.hc_norm.weight"),
        mix_down: format!("{p}.input_mix_weight_down"),
        mix_up: format!("{p}.input_mix_weight_up"),
        block_inject: None,
    }
}

/// Every engine tensor name layer `layer` carries (quantized triples
/// expanded), for the converter's completeness check and the loader.
/// Expert tensors (`switch_mlp`) and PLE are included; PLE only at
/// [`super::config::PLE_LAYERS`].
pub fn layer_tensor_names(layer: usize) -> Vec<String> {
    use super::config::{Q38Config, Q38LayerKind};
    let cfg = Q38Config::REAL;
    let p = format!("model.layers.{layer}");
    let mut out = Vec::new();
    let mut q = |base: String| {
        for s in ["weight", "scales", "biases"] {
            out.push(format!("{base}.{s}"));
        }
    };
    for site in ["attn_hyper_connection", "mlp_hyper_connection"] {
        for t in ["input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"] {
            q(format!("{p}.{site}.{t}"));
        }
    }
    match cfg.layer_kind(layer) {
        Q38LayerKind::LinearAttn => {
            for t in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"] {
                q(format!("{p}.linear_attn.{t}"));
            }
        }
        Q38LayerKind::Qsa => {
            for t in ["q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj"] {
                q(format!("{p}.self_attn.{t}"));
            }
        }
    }
    for t in [
        "mlp.gate",
        "mlp.shared_expert_gate",
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
        "mlp.switch_mlp.gate_proj",
        "mlp.switch_mlp.up_proj",
        "mlp.switch_mlp.down_proj",
    ] {
        q(format!("{p}.{t}"));
    }
    if cfg.has_ple(layer) {
        q(format!("{p}.ple.key_proj"));
        q(format!("{p}.ple.value_proj"));
        for s in 0..super::config::SPLIT_NGRAM_PARTS {
            q(format!("{p}.ple.ple_embedding.ngram_embedding.shards.{s}"));
        }
    }
    out.push(format!("{p}.attn_hyper_connection.hc_norm.weight"));
    out.push(format!("{p}.mlp_hyper_connection.hc_norm.weight"));
    match cfg.layer_kind(layer) {
        Q38LayerKind::LinearAttn => {
            for t in ["conv1d.weight", "A_log", "dt_bias", "norm.weight"] {
                out.push(format!("{p}.linear_attn.{t}"));
            }
        }
        Q38LayerKind::Qsa => {
            for t in ["q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight"] {
                out.push(format!("{p}.self_attn.{t}"));
            }
        }
    }
    if cfg.has_ple(layer) {
        for t in [
            "norm_key.weight",
            "norm_query.weight",
            "norm_conv.weight",
            "conv1d.weight",
            "ple_embedding.layer_multipliers",
            "ple_embedding.ngram_heads_offsets",
            "ple_embedding.ngram_heads_vocab_sizes",
        ] {
            out.push(format!("{p}.ple.{t}"));
        }
    }
    out
}

/// Engine names outside the layers.
pub fn global_tensor_names() -> Vec<String> {
    let mut out = Vec::new();
    for base in [
        "model.embed_tokens",
        "lm_head",
        "model.hyper_connection_mixer.input_mix_weight_down",
        "model.hyper_connection_mixer.input_mix_weight_up",
    ] {
        for s in ["weight", "scales", "biases"] {
            out.push(format!("{base}.{s}"));
        }
    }
    out.push("model.hyper_connection_mixer.hc_norm.weight".to_string());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_examples() {
        let c = |n: &str| classify(n).unwrap();
        let e = c("language_model.model.layers.7.attn_hyper_connection.hc_norm.weight");
        assert_eq!(e.engine_name, "model.layers.7.attn_hyper_connection.hc_norm.weight");
        assert!(e.one_plus_gain && e.dest == Dest::Resident && e.quant == Quant::Raw);
        assert_eq!(c("language_model.model.layers.0.mlp.gate.scales").quant, Quant::Q8G64);
        assert_eq!(c("language_model.model.layers.0.mlp.switch_mlp.down_proj.weight").dest, Dest::Experts);
        assert_eq!(
            c("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.127.biases").dest,
            Dest::PleRows
        );
        assert!(!c("language_model.model.layers.0.linear_attn.norm.weight").one_plus_gain);
        assert!(c("language_model.model.layers.3.self_attn.q_norm.weight").one_plus_gain);
        assert_eq!(c("vision_tower.blocks.0.attn.qkv.weight").dest, Dest::Skip);
        assert_eq!(c("language_model.lm_head.weight").engine_name, "lm_head.weight");
        assert!(classify("language_model.model.norm.weight").is_none());
        assert!(classify("language_model.model.layers.0.input_layernorm.weight").is_none());
    }

    /// Against the real MLX index when it is on this host: every
    /// `language_model.*` tensor is classified, the norm-gain count is
    /// the README's 148, and the per-layer / global name lists are
    /// exactly the index's text tensors.
    #[test]
    fn index_is_fully_classified() {
        let path = std::path::Path::new(&std::env::var("HOME").unwrap_or_default())
            .join("models/qwen3-8-flash-next/meta/model.safetensors.index.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("[q38 weights] skipping: {} not found", path.display());
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let map = v["weight_map"].as_object().unwrap();
        let mut gains = 0;
        let mut text_names = std::collections::BTreeSet::new();
        for name in map.keys() {
            let e = classify(name).unwrap_or_else(|| panic!("unclassified tensor {name}"));
            if e.one_plus_gain {
                gains += 1;
            }
            if e.dest != Dest::Skip {
                text_names.insert(e.engine_name);
            }
        }
        assert_eq!(gains, 148);
        let mut want = std::collections::BTreeSet::new();
        want.extend(global_tensor_names());
        for l in 0..48 {
            want.extend(layer_tensor_names(l));
        }
        let missing: Vec<_> = want.difference(&text_names).collect();
        let extra: Vec<_> = text_names.difference(&want).collect();
        assert!(missing.is_empty(), "expected but not in index: {missing:?}");
        assert!(extra.is_empty(), "in index but not expected: {extra:?}");
    }
}
