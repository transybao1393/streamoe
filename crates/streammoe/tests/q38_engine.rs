//! Qwen3.8-Flash-Next through the public context: `Ctx::open_with`
//! under a memory cap, `eval_prompt` / `eval_token`, checkpoints,
//! against the reference goldens of the tiny random-weight model.
//!
//! Needs `--features model-qwen3-8-flash-next-tiny` and
//! `STREAMMOE_Q38_TINY` = the `tiny/` directory written by
//! `tools/model_layer/qwen3_8_flash_next/make_tiny.py --shapes engine`
//! (with `streammoe/` converted by `convert.py`). Skips when unset.
//!
//! ```text
//! STREAMMOE_Q38_TINY=…/tiny cargo test -p streammoe \
//!     --features model-qwen3-8-flash-next-tiny --test q38_engine
//! ```

#![cfg(all(target_os = "macos", feature = "model-qwen3-8-flash-next-tiny"))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use streammoe::{Ctx, OpenConfig};

/// `max |Δ| ≤ TOL · max |golden|` per logits row (the reference runs
/// the same 4-bit weights in f32; measured ~6e-5).
const TOL: f32 = 2e-3;
const GIB: u64 = 1 << 30;

fn tiny_dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var_os("STREAMMOE_Q38_TINY")?);
    d.join("goldens.safetensors").is_file().then_some(d)
}

struct Golden {
    ids: Vec<i32>,
    rows: Vec<Vec<f32>>,
    decode: Vec<i32>,
}

fn goldens(dir: &Path) -> Vec<Golden> {
    let bytes = std::fs::read(dir.join("goldens.safetensors")).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];
    let mut t: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for (k, v) in header.as_object().unwrap() {
        let Some(dtype) = v.get("dtype").and_then(|d| d.as_str()) else { continue };
        let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let o = v["data_offsets"].as_array().unwrap();
        let s = &data[o[0].as_u64().unwrap() as usize..o[1].as_u64().unwrap() as usize];
        let vals: Vec<f32> = match dtype {
            "F32" => s.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "I64" => s.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            _ => continue,
        };
        t.insert(k.clone(), (shape, vals));
    }
    let mut out = Vec::new();
    for i in 0.. {
        let k = |f: &str| format!("q4.p{i}.{f}");
        let Some((_, ids)) = t.get(&k("input_ids")) else { break };
        let (ps, pl) = &t[&k("prefill_logits")];
        let (_, dl) = &t[&k("decode_logits")];
        let v = ps[1];
        out.push(Golden {
            ids: ids.iter().map(|&x| x as i32).collect(),
            rows: pl.chunks(v).chain(dl.chunks(v)).map(|r| r.to_vec()).collect(),
            decode: t[&k("decode_tokens")].1.iter().map(|&x| x as i32).collect(),
        });
    }
    assert!(!out.is_empty());
    out
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-6);
    a.iter().zip(b).fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs())) / scale
}

fn open(dir: &Path, cap: Option<u64>) -> Ctx {
    let m = dir.join("streammoe");
    let a = m.join("artifacts");
    let cfg = OpenConfig { n_ctx: 64, mem_cap: cap, ..OpenConfig::default() };
    Ctx::open_with(
        &a.join("model_weights.bin"),
        &a.join("model_weights.json"),
        &a.join("vocab.bin"),
        &m.join("root"),
        false,
        &cfg,
    )
    .expect("open the tiny model")
}

/// Every golden prompt: `eval_token` row by row (prefill positions,
/// then the golden decode tokens), and `eval_prompt` over the whole
/// prompt for its last row. Under a 2 GiB cap.
#[test]
fn ctx_matches_goldens_under_cap() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38_engine] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let gs = goldens(&dir);
    let mut ctx = open(&dir, Some(2 * GIB));
    let v = ctx.n_vocab();
    let mut logits = vec![0.0f32; v];
    let mut worst = 0.0f32;
    for (pi, g) in gs.iter().enumerate() {
        ctx.memory_clear();
        for (r, &tok) in g.ids.iter().chain(&g.decode).enumerate() {
            ctx.eval_token(tok, r, 0, &mut logits).unwrap();
            let e = rel_err(&logits, &g.rows[r]);
            worst = worst.max(e);
            assert!(e <= TOL, "prompt {pi} row {r}: rel err {e}");
        }
        assert_eq!(ctx.memory_seq_pos_max(0), (g.ids.len() + g.decode.len()) as i32 - 1);
        // The prompt path.
        ctx.memory_clear();
        ctx.eval_prompt(&g.ids, 0, 0, &mut logits).unwrap();
        let e = rel_err(&logits, &g.rows[g.ids.len() - 1]);
        assert!(e <= TOL, "prompt {pi} eval_prompt: rel err {e}");
        // Decode continues from the prompt.
        ctx.eval_token(g.decode[0], g.ids.len(), 0, &mut logits).unwrap();
        assert!(rel_err(&logits, &g.rows[g.ids.len()]) <= TOL);
    }
    // A position that does not continue the sequence is refused.
    assert!(ctx.eval_token(1, 3, 0, &mut logits).is_err());
    eprintln!("[q38_engine] worst rel err vs goldens {worst:.3e}");
}

/// `checkpoint_pos` / `restore_to` and `state_save` / `state_load`
/// reproduce the continuation exactly; `memory_seq_rm` refuses a
/// middle truncation.
#[test]
fn ctx_checkpoints_round_trip() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38_engine] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let gs = goldens(&dir);
    let g = &gs[0];
    let mut ctx = open(&dir, Some(2 * GIB));
    let v = ctx.n_vocab();
    let mut logits = vec![0.0f32; v];
    let cut = g.ids.len() - 2;
    ctx.memory_clear();
    ctx.eval_prompt(&g.ids[..cut], 0, 0, &mut logits).unwrap();
    ctx.checkpoint_pos(cut as i32).unwrap();
    let mut snap = vec![0u8; ctx.state_size()];
    ctx.state_save(&mut snap).unwrap();
    let tail: Vec<i32> = g.ids[cut..].iter().chain(&g.decode).copied().collect();
    let mut run = |ctx: &mut Ctx| -> Vec<Vec<f32>> {
        tail.iter()
            .enumerate()
            .map(|(i, &t)| {
                let mut l = vec![0.0f32; v];
                ctx.eval_token(t, cut + i, 0, &mut l).unwrap();
                l
            })
            .collect()
    };
    let first = run(&mut ctx);
    for (i, row) in first.iter().enumerate() {
        assert!(rel_err(row, &g.rows[cut + i]) <= TOL);
    }
    assert!(!ctx.memory_seq_rm(0, 5, -1), "middle truncation must be refused");
    ctx.restore_to(cut as i32).unwrap();
    assert_eq!(ctx.memory_seq_pos_max(0), cut as i32 - 1);
    assert_eq!(run(&mut ctx), first, "restore_to");
    ctx.state_load(&snap).unwrap();
    assert_eq!(run(&mut ctx), first, "state_load");
    assert!(ctx.memory_seq_rm(0, 0, -1));
    assert_eq!(ctx.memory_seq_pos_max(0), -1);
}

/// What the family does not support is refused at open.
#[test]
fn ctx_rejects_unsupported_config() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38_engine] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let m = dir.join("streammoe");
    let a = m.join("artifacts");
    for cfg in [
        OpenConfig { n_ctx: 64, layer_range: Some(0..2), ..OpenConfig::default() },
        OpenConfig { n_ctx: 64, max_decode_tokens: 4, ..OpenConfig::default() },
    ] {
        let r = Ctx::open_with(
            &a.join("model_weights.bin"),
            &a.join("model_weights.json"),
            &a.join("vocab.bin"),
            &m.join("root"),
            false,
            &cfg,
        );
        assert!(r.is_err());
    }
}

/// Speculative prefetch and the hot tier through the context, under a
/// cap: the logits of every arm equal prefetch-off / no-tier bit for
/// bit, the counters move, and the footprint stays under the cap.
#[test]
fn ctx_prefetch_and_hot_tier_under_cap() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38_engine] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let gs = goldens(&dir);
    let m = dir.join("streammoe");
    let a = m.join("artifacts");
    let es = 30720u64; // the tiny model's expert size (layout.json)
    let cap = 2 * GIB;
    // (label, prefetch_experts, lookahead, hot_tier_bytes)
    let arms: [(&str, Option<usize>, bool, Option<u64>); 4] = [
        ("off", Some(0), false, Some(0)),
        ("default prefetch (capped)", None, false, Some(0)),
        ("prefetch 1 + lookahead, 8-expert tier", Some(1), true, Some(8 * es)),
        ("default prefetch, tier takes the rest", None, false, None),
    ];
    let mut base: Option<Vec<Vec<f32>>> = None;
    for (label, n, ahead, hot) in arms {
        let cfg = OpenConfig {
            n_ctx: 64,
            mem_cap: Some(cap),
            prefetch_experts: n,
            prefetch_lookahead: ahead,
            hot_tier_bytes: hot,
            ..OpenConfig::default()
        };
        let mut ctx = Ctx::open_with(
            &a.join("model_weights.bin"),
            &a.join("model_weights.json"),
            &a.join("vocab.bin"),
            &m.join("root"),
            false,
            &cfg,
        )
        .expect("open the tiny model");
        let v = ctx.n_vocab();
        let mut rows = Vec::new();
        for g in &gs {
            ctx.memory_clear();
            for (r, &tok) in g.ids.iter().chain(&g.decode).enumerate() {
                let mut l = vec![0.0f32; v];
                ctx.eval_token(tok, r, 0, &mut l).unwrap();
                rows.push(l);
            }
        }
        let (hits, misses) = ctx.prefetch_stats();
        let st = ctx.expert_io_stats();
        let h = st.hot.unwrap_or_default();
        let fp = ctx.footprint().expect("footprint");
        eprintln!(
            "[q38_engine] {label}: prefetch hits {hits} misses {misses}, prefetch reads {}, demand reads {}, \
             hot hits {} / misses {}, tier {} slots; footprint {:.1} MiB of cap {} MiB",
            st.prefetch.reads,
            st.demand.reads,
            h.hits,
            h.misses,
            h.capacity,
            fp.current as f64 / (1u64 << 20) as f64,
            cap >> 20
        );
        assert!(fp.current <= cap, "{label}: footprint over the cap");
        match n {
            Some(0) => assert_eq!((hits, st.prefetch.reads), (0, 0), "{label}"),
            _ if hot.is_none() => {} // every expert fits the tier: little left to prefetch
            _ => assert!(hits > 0, "{label}: no prefetch hits"),
        }
        if hot != Some(0) {
            assert!(h.hits > 0, "{label}: no hot-tier hits");
        }
        match &base {
            None => base = Some(rows),
            Some(want) => {
                let same = rows.iter().zip(want).all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()));
                assert!(same, "{label}: logits differ from prefetch off / no tier");
            }
        }
    }
}
