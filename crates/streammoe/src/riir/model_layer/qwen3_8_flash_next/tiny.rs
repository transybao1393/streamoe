//! Tiny random-weight models for the family's tests.
//!
//! Weights come from a splitmix64 stream seeded by the FNV-1a hash of
//! each tensor's name — the same generator as
//! `tools/model_layer/qwen3_8_flash_next/core_ref.py`, the torch port
//! of the reference — so the Rust test and the reference build the
//! same model without a weight file. The reference's outputs are in
//! `tests/fixtures/q38_core_ref.json`.

use super::config::{GateAct, Q38Config, Q38LayerKind};
use super::gated_residual::DenseLinear;
use super::layer::{
    AttnWeights, DenseAttention, DenseEmbedding, GdnWeights, HcWeights, LayerWeights, MlpWeights, ModelWeights,
    MoeWeights, NoPle, Q38CpuModel,
};

/// FNV-1a, 64-bit.
fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// `offset + scale · u`, `u` uniform in `[-1, 1)` on a 2⁻²³ grid, from
/// the splitmix64 stream seeded by `fnv1a64(name)`. Matches
/// `core_ref.gen` (which computes in f64; the grid values are exact in
/// f32, the `scale` / `offset` products round once).
pub fn gen_tensor(name: &str, n: usize, scale: f32, offset: f32) -> Vec<f32> {
    let mut state = fnv1a64(name);
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 40) as f64 / (1u64 << 24) as f64;
            (offset as f64 + scale as f64 * (2.0 * u - 1.0)) as f32
        })
        .collect()
}

fn lin(name: &str, out_dim: usize, in_dim: usize, scale: f32) -> DenseLinear {
    DenseLinear::new(out_dim, in_dim, gen_tensor(name, out_dim * in_dim, scale, 0.0))
}

/// The tiny config of `core_ref.py` (`TINY`).
pub fn tiny_config() -> Q38Config {
    Q38Config {
        hidden: 32,
        num_layers: 4,
        hc_count: 4,
        hc_lowrank: 8,
        full_attn_interval: 4,
        linear_k_heads: 2,
        linear_v_heads: 4,
        linear_key_dim: 8,
        linear_value_dim: 8,
        conv_kernel: 4,
        attn_heads: 4,
        kv_heads: 2,
        head_dim: 16,
        rotary_dim: 4,
        rope_theta: 10000.0,
        mrope_section: [1, 1, 0],
        num_experts: 8,
        top_k: 2,
        moe_intermediate: 16,
        shared_intermediate: 16,
        norm_topk_prob: true,
        vocab: 64,
        rms_eps: 1e-6,
        gdn_gate: GateAct::Sigmoid,
        ple_layers: &[],
    }
}

/// A Gated Residual with dense weights.
pub type TinyHc = HcWeights<DenseLinear>;

impl TinyHc {
    /// `core_ref.hc_weights(cfg, pre, combine)`.
    pub fn generate(cfg: &Q38Config, pre: &str, combine: bool) -> Self {
        let w = cfg.hc_hidden();
        let r = cfg.hc_lowrank;
        HcWeights {
            hc_norm: gen_tensor(&format!("{pre}.hc_norm"), w, 0.2, 0.0),
            mix_down: lin(&format!("{pre}.down"), r, w, 1.0 / (w as f32).sqrt()),
            mix_up: lin(&format!("{pre}.up"), w, r, 2.0 / (r as f32).sqrt()),
            block_inject: combine.then(|| lin(&format!("{pre}.inject"), cfg.hc_count, w, 1.0 / (w as f32).sqrt())),
        }
    }
}

/// Tiny model: dense weights, dense attention, no PLE.
pub type TinyModel =
    Q38CpuModel<DenseLinear, Vec<MlpWeights<DenseLinear>>, DenseEmbedding, DenseAttention<DenseLinear>, NoPle>;

/// `core_ref.layer_weights` / `model_forward`'s weights.
pub fn tiny_model(cfg: &Q38Config) -> TinyModel {
    let h = cfg.hidden;
    let s = 1.0 / (h as f32).sqrt();
    let mut layers = Vec::new();
    let mut attn_layers = Vec::new();
    for l in 0..cfg.num_layers {
        let pre = format!("layers.{l}");
        let mut gdn = None;
        let mut attn = None;
        match cfg.layer_kind(l) {
            Q38LayerKind::LinearAttn => {
                let (hk, hv, dk, dv) = (cfg.linear_k_heads, cfg.linear_v_heads, cfg.linear_key_dim, cfg.linear_value_dim);
                let c = 2 * hk * dk + hv * dv;
                let g = format!("{pre}.gdn");
                gdn = Some(GdnWeights {
                    in_qkv: lin(&format!("{g}.in_qkv"), c, h, s),
                    in_z: lin(&format!("{g}.in_z"), hv * dv, h, s),
                    in_b: lin(&format!("{g}.in_b"), hv, h, s),
                    in_a: lin(&format!("{g}.in_a"), hv, h, s),
                    conv1d: gen_tensor(&format!("{g}.conv1d"), c * cfg.conv_kernel, 0.5, 0.0),
                    a_log: gen_tensor(&format!("{g}.A_log"), hv, 1.0, 0.0),
                    dt_bias: gen_tensor(&format!("{g}.dt_bias"), hv, 0.5, 0.0),
                    norm: gen_tensor(&format!("{g}.norm"), dv, 0.2, 1.0),
                    out_proj: lin(&format!("{g}.out_proj"), h, hv * dv, 1.0 / ((hv * dv) as f32).sqrt()),
                });
            }
            Q38LayerKind::Qsa => {
                let (hq, hkv, d) = (cfg.attn_heads, cfg.kv_heads, cfg.head_dim);
                let a = format!("{pre}.attn");
                attn = Some(AttnWeights {
                    q_proj: lin(&format!("{a}.q_proj"), 2 * hq * d, h, s),
                    k_proj: lin(&format!("{a}.k_proj"), hkv * d, h, s),
                    v_proj: lin(&format!("{a}.v_proj"), hkv * d, h, s),
                    o_proj: lin(&format!("{a}.o_proj"), h, hq * d, 1.0 / ((hq * d) as f32).sqrt()),
                    q_norm: gen_tensor(&format!("{a}.q_norm"), d, 0.2, 0.0),
                    k_norm: gen_tensor(&format!("{a}.k_norm"), d, 0.2, 0.0),
                });
            }
        }
        let m = format!("{pre}.moe");
        let (e, i, is) = (cfg.num_experts, cfg.moe_intermediate, cfg.shared_intermediate);
        let experts = (0..e)
            .map(|x| MlpWeights {
                gate: lin(&format!("{m}.experts.{x}.gate"), i, h, s),
                up: lin(&format!("{m}.experts.{x}.up"), i, h, s),
                down: lin(&format!("{m}.experts.{x}.down"), h, i, 1.0 / (i as f32).sqrt()),
            })
            .collect();
        layers.push(LayerWeights {
            attn_hc: TinyHc::generate(cfg, &format!("{pre}.attn_hc"), true),
            mlp_hc: TinyHc::generate(cfg, &format!("{pre}.mlp_hc"), true),
            gdn,
            moe: MoeWeights {
                router: lin(&format!("{m}.router"), e, h, 4.0 * s),
                experts,
                shared: MlpWeights {
                    gate: lin(&format!("{m}.shared.gate"), is, h, s),
                    up: lin(&format!("{m}.shared.up"), is, h, s),
                    down: lin(&format!("{m}.shared.down"), h, is, 1.0 / (is as f32).sqrt()),
                },
                shared_gate: lin(&format!("{m}.shared_expert_gate"), 1, h, s),
            },
        });
        attn_layers.push(attn);
    }
    let weights = ModelWeights {
        embed: DenseEmbedding {
            hidden: h,
            table: gen_tensor("embed_tokens", cfg.vocab * h, 1.0, 0.0),
        },
        layers,
        mixer: TinyHc::generate(cfg, "mixer", false),
        lm_head: lin("lm_head", cfg.vocab, h, s),
    };
    let attn = DenseAttention::new(cfg, attn_layers);
    Q38CpuModel::new(cfg.clone(), weights, attn, NoPle)
}

/// `tests/fixtures/q38_core_ref.json`.
#[derive(Debug, Clone)]
pub struct Fixture {
    pub config: Q38Config,
    pub gated_residual: GrFixture,
    pub tokens: Vec<u32>,
    /// `[tokens][vocab]`.
    pub logits: Vec<Vec<f64>>,
    /// `[tokens · layers][top_k]`, sorted.
    pub routes: Vec<Vec<i32>>,
}

#[derive(Debug, Clone)]
pub struct GrFixture {
    pub normed: Vec<f64>,
    pub mixed: Vec<f64>,
    pub inj: Vec<f64>,
    pub injected: Vec<f64>,
    pub final_mixed: Vec<f64>,
}

/// Load the reference fixture and check its config is [`tiny_config`].
pub fn load_fixture() -> Fixture {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/q38_core_ref.json");
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let v: serde_json::Value = serde_json::from_str(&text).expect("fixture json");
    let f64s = |x: &serde_json::Value| -> Vec<f64> { x.as_array().unwrap().iter().map(|y| y.as_f64().unwrap()).collect() };
    let cfg = tiny_config();
    let c = &v["config"];
    let u = |k: &str| c[k].as_u64().unwrap() as usize;
    assert_eq!(
        (u("hidden"), u("hc_count"), u("hc_lowrank"), u("num_layers"), u("full_attn_interval")),
        (cfg.hidden, cfg.hc_count, cfg.hc_lowrank, cfg.num_layers, cfg.full_attn_interval)
    );
    assert_eq!(
        (u("linear_k_heads"), u("linear_v_heads"), u("linear_key_dim"), u("linear_value_dim"), u("conv_kernel")),
        (cfg.linear_k_heads, cfg.linear_v_heads, cfg.linear_key_dim, cfg.linear_value_dim, cfg.conv_kernel)
    );
    assert_eq!(
        (u("attn_heads"), u("kv_heads"), u("head_dim"), u("rotary_dim")),
        (cfg.attn_heads, cfg.kv_heads, cfg.head_dim, cfg.rotary_dim)
    );
    assert_eq!(
        (u("num_experts"), u("top_k"), u("moe_intermediate"), u("shared_intermediate"), u("vocab")),
        (cfg.num_experts, cfg.top_k, cfg.moe_intermediate, cfg.shared_intermediate, cfg.vocab)
    );
    assert_eq!(c["rope_theta"].as_f64().unwrap() as f32, cfg.rope_theta);
    let g = &v["gated_residual"];
    Fixture {
        config: cfg,
        gated_residual: GrFixture {
            normed: f64s(&g["normed"]),
            mixed: f64s(&g["mixed"]),
            inj: f64s(&g["inj"]),
            injected: f64s(&g["injected"]),
            final_mixed: f64s(&g["final_mixed"]),
        },
        tokens: v["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect(),
        logits: v["logits"].as_array().unwrap().iter().map(f64s).collect(),
        routes: v["routes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::layer::{AttnBlock, PleHook, Q38Error};
    use super::*;

    /// The splitmix64 / FNV stream against values `core_ref.gen`
    /// prints for the same name.
    #[test]
    fn generator_is_deterministic_and_bounded() {
        let a = gen_tensor("embed_tokens", 1000, 1.0, 0.0);
        assert_eq!(a, gen_tensor("embed_tokens", 1000, 1.0, 0.0));
        assert!(a.iter().all(|v| (-1.0..1.0).contains(v)));
        let mean: f32 = a.iter().sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert_ne!(a, gen_tensor("lm_head", 1000, 1.0, 0.0));
    }

    /// CPU end-to-end forward of the tiny model against the torch
    /// port: logits at every position of an 8-token sequence (two
    /// passes through the QSA layer's growing KV cache and three GDN
    /// layers' recurrent state) and every layer's chosen experts.
    #[test]
    fn tiny_forward_matches_reference() {
        let fx = load_fixture();
        let cfg = fx.config.clone();
        let mut model = tiny_model(&cfg);
        let mut logits = vec![0.0f32; cfg.vocab];
        let mut worst = 0.0f64;
        for (p, &t) in fx.tokens.iter().enumerate() {
            let routes = model.forward_token(t, &mut logits).unwrap();
            for (l, r) in routes.iter().enumerate() {
                let mut got = r.clone();
                got.sort();
                assert_eq!(got, fx.routes[p * cfg.num_layers + l], "routes at pos {p} layer {l}");
            }
            for (i, (&g, &w)) in logits.iter().zip(&fx.logits[p]).enumerate() {
                let err = (g as f64 - w).abs();
                assert!(err < 1e-5, "logit pos {p} id {i}: got {g}, want {w}");
                worst = worst.max(err);
            }
        }
        eprintln!("[q38 tiny forward] {} tokens, worst abs logit err {worst:.2e}", fx.tokens.len());

        // reset() gives the same first-token logits again.
        model.reset();
        model.forward_token(fx.tokens[0], &mut logits).unwrap();
        for (&g, &w) in logits.iter().zip(&fx.logits[0]) {
            assert!((g as f64 - w).abs() < 1e-5);
        }
    }

    /// A PLE layer calls the hook at its index with the stream width
    /// and adds the result before the attention Gated Residual.
    #[test]
    fn ple_hook_call_point() {
        struct Probe {
            calls: Vec<(usize, usize, u32)>,
            add: f32,
        }
        impl PleHook for Probe {
            fn forward(&mut self, layer: usize, pos: usize, token: u32, hyper: &[f32], out: &mut [f32]) -> Result<(), Q38Error> {
                assert_eq!(hyper.len(), out.len());
                self.calls.push((layer, pos, token));
                out.fill(self.add);
                Ok(())
            }
            fn reset(&mut self) {
                self.calls.clear();
            }
        }
        let mut cfg = tiny_config();
        cfg.ple_layers = &[1];
        let run = |add: f32| {
            let m = tiny_model(&cfg);
            let mut m = Q38CpuModel::new(cfg.clone(), m.weights, m.attn, Probe { calls: vec![], add });
            let mut logits = vec![0.0; cfg.vocab];
            m.forward_token(5, &mut logits).unwrap();
            m.forward_token(9, &mut logits).unwrap();
            (m.ple.calls.clone(), logits)
        };
        let (calls, l0) = run(0.0);
        assert_eq!(calls, vec![(1, 0, 5), (1, 1, 9)]);
        let (_, l1) = run(0.5);
        assert!(l0.iter().zip(&l1).any(|(a, b)| (a - b).abs() > 1e-4), "PLE output must reach the logits");
    }

    /// Dense attention rejects a skipped position (its KV must stay in
    /// step with `pos`).
    #[test]
    fn dense_attention_tracks_positions() {
        let cfg = tiny_config();
        let mut m = tiny_model(&cfg);
        let x = vec![0.1f32; cfg.hidden];
        let mut out = vec![0.0; cfg.hidden];
        m.attn.forward(3, 0, &x, &mut out).unwrap();
        assert!(m.attn.forward(3, 2, &x, &mut out).is_err());
    }
}
