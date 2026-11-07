//! Qwen3-Next through the public context: `Ctx::open_with` under a
//! memory cap, `eval_prompt` / `eval_token`, against the goldens of the
//! tiny random-weight model from the pure-torch reference.
//!
//! Needs `--features model-qwen3-next-tiny` and `STREAMMOE_QNEXT_TINY`
//! = the `tiny/` directory written by
//! `tools/model_layer/qwen3_next/make_tiny.py`, with `streammoe/`
//! converted by `convert.py` and `streammoe-naive/` by
//! `convert.py --layout naive` (the fused projections split without
//! the per-key-head de-interleave). Skips when unset.
//!
//! ```text
//! STREAMMOE_QNEXT_TINY=…/tiny cargo test -p streammoe \
//!     --features model-qwen3-next-tiny --test qnext_engine
//! ```

#![cfg(all(target_os = "macos", feature = "model-qwen3-next-tiny"))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use streammoe::{Ctx, OpenConfig};

/// `max |Δ| ≤ TOL · max |golden|` per logits row (the reference runs
/// the same 4-bit weights in f32). `eval_token` measures ≤ 1.7e-5.
const TOL: f32 = 2e-3;
/// The same after the batched prefill (`eval_prompt`): its MoE runs the
/// gather-id GEMM (`gather_mm_id.metal`, default on for every variant),
/// which is less exact than the per-token path. Measured 4.7e-4 to
/// 7.5e-4 on the prompt's last row and up to 2.5e-3 on decode rows
/// after the 600-token prompt; with `STREAMMOE_MOE_GATHER_ID=0` every
/// row is within 5e-6.
const TOL_PREFILL: f32 = 5e-3;
const GIB: u64 = 1 << 30;

fn tiny_dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var_os("STREAMMOE_QNEXT_TINY")?);
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

fn open(model: &Path, n_ctx: usize, cap: Option<u64>) -> Ctx {
    let a = model.join("artifacts");
    let cfg = OpenConfig { n_ctx, mem_cap: cap, ..OpenConfig::default() };
    Ctx::open_with(
        &a.join("model_weights.bin"),
        &a.join("model_weights.json"),
        &a.join("vocab.bin"),
        &model.join("root"),
        false,
        &cfg,
    )
    .expect("open the tiny model")
}

/// Every golden prompt (7, 40 and 600 tokens), under a 2 GiB cap:
/// `eval_token` row by row over the prompt and the golden decode
/// tokens, then `eval_prompt` over the whole prompt (the batched
/// prefill; 600 tokens run as a 512-token chunk plus 88) for its last
/// row, and decode continuing from it.
#[test]
fn ctx_matches_goldens_under_cap() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[qnext_engine] STREAMMOE_QNEXT_TINY unset: skipped");
        return;
    };
    let gs = goldens(&dir);
    let mut ctx = open(&dir.join("streammoe"), 640, Some(2 * GIB));
    let v = ctx.n_vocab();
    assert_eq!(v, gs[0].rows[0].len());
    let mut logits = vec![0.0f32; v];
    for (pi, g) in gs.iter().enumerate() {
        let mut worst = 0.0f32;
        let mut worst_row = 0;
        ctx.memory_clear();
        for (r, &tok) in g.ids.iter().chain(&g.decode).enumerate() {
            ctx.eval_token(tok, r, 0, &mut logits).unwrap();
            let e = rel_err(&logits, &g.rows[r]);
            if e > worst {
                worst = e;
                worst_row = r;
            }
        }
        eprintln!("[qnext_engine] prompt {pi} ({} tokens): eval_token worst rel err {worst:.3e} at row {worst_row}",
                  g.ids.len());
        assert!(worst <= TOL, "prompt {pi} eval_token row {worst_row}: rel err {worst}");
        // The qwen3_5_moe path reports the KV length (the C oracle's
        // `mf_memory_seq_pos_max`), not the last position.
        assert_eq!(ctx.memory_seq_pos_max(0), (g.ids.len() + g.decode.len()) as i32);
        // The prompt path.
        ctx.memory_clear();
        ctx.eval_prompt(&g.ids, 0, 0, &mut logits).unwrap();
        let e = rel_err(&logits, &g.rows[g.ids.len() - 1]);
        eprintln!("[qnext_engine] prompt {pi}: eval_prompt last row rel err {e:.3e}");
        assert!(e <= TOL_PREFILL, "prompt {pi} eval_prompt: rel err {e}");
        // Decode continues from the prompt.
        let mut worst = 0.0f32;
        for (k, &tok) in g.decode.iter().enumerate() {
            ctx.eval_token(tok, g.ids.len() + k, 0, &mut logits).unwrap();
            let e = rel_err(&logits, &g.rows[g.ids.len() + k]);
            worst = worst.max(e);
            assert!(e <= TOL_PREFILL, "prompt {pi} decode {k} after eval_prompt: rel err {e}");
        }
        eprintln!("[qnext_engine] prompt {pi}: decode after eval_prompt worst rel err {worst:.3e}");
    }
    let fp = ctx.footprint().expect("footprint");
    assert!(fp.current <= 2 * GIB, "footprint over the cap");
}

/// The fused-projection de-interleave matters and the goldens see it:
/// the same tiny model converted without the per-key-head regrouping
/// (`convert.py --layout naive`, rows taken as contiguous `[q|k|v|z]`
/// and `[b|a]` blocks) is far from the goldens, while the real
/// conversion is within `TOL` (`ctx_matches_goldens_under_cap`).
#[test]
fn naive_fused_split_is_far_from_goldens() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[qnext_engine] STREAMMOE_QNEXT_TINY unset: skipped");
        return;
    };
    let naive = dir.join("streammoe-naive");
    assert!(naive.join("artifacts/model_weights.json").is_file(), "convert.py --layout naive first");
    let gs = goldens(&dir);
    let g = &gs[1];
    let mut good = open(&dir.join("streammoe"), 64, Some(2 * GIB));
    let mut bad = open(&naive, 64, Some(2 * GIB));
    let v = good.n_vocab();
    let (mut lg, mut lb) = (vec![0.0f32; v], vec![0.0f32; v]);
    for (i, &t) in g.ids.iter().enumerate() {
        good.eval_token(t, i, 0, &mut lg).unwrap();
        bad.eval_token(t, i, 0, &mut lb).unwrap();
    }
    let last = &g.rows[g.ids.len() - 1];
    let (eg, eb) = (rel_err(&lg, last), rel_err(&lb, last));
    eprintln!("[qnext_engine] de-interleaved split rel err {eg:.3e}; naive split {eb:.3e}");
    assert!(eg <= TOL);
    assert!(eb > 50.0 * TOL, "a mis-ordered split should be far off (got {eb})");
}

/// Prompt cut mid-way, state saved, continuation reproduced after
/// `state_load`, against the goldens.
#[test]
fn state_save_load_continues_on_goldens() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[qnext_engine] STREAMMOE_QNEXT_TINY unset: skipped");
        return;
    };
    let gs = goldens(&dir);
    let g = &gs[1];
    let mut ctx = open(&dir.join("streammoe"), 64, Some(2 * GIB));
    let v = ctx.n_vocab();
    let mut logits = vec![0.0f32; v];
    let cut = 20;
    ctx.eval_prompt(&g.ids[..cut], 0, 0, &mut logits).unwrap();
    assert!(rel_err(&logits, &g.rows[cut - 1]) <= TOL_PREFILL);
    let mut snap = vec![0u8; ctx.state_size()];
    ctx.state_save(&mut snap).unwrap();
    let tail: Vec<i32> = g.ids[cut..].iter().chain(&g.decode).copied().collect();
    let run = |ctx: &mut Ctx| -> Vec<Vec<f32>> {
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
        assert!(rel_err(row, &g.rows[cut + i]) <= TOL_PREFILL, "row {}", cut + i);
    }
    ctx.memory_clear();
    ctx.state_load(&snap).unwrap();
    assert_eq!(run(&mut ctx), first, "state_load");
}

/// Layer-major prefill (`OpenConfig::layer_major`) is bit-identical to
/// chunk-by-chunk prefill on this family too: the 600-token golden
/// prompt in chunks of 64 (windows of 4 + 4 + 2 chunks), full staging
/// and bounded staging (16 of the 32 experts, double-buffered) — the
/// prefill logits, the saved state and the decode after it — and it
/// stages fewer experts.
#[test]
fn layer_major_matches_chunk_major() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[qnext_engine] STREAMMOE_QNEXT_TINY unset: skipped");
        return;
    };
    let gs = goldens(&dir);
    let g = gs.iter().max_by_key(|g| g.ids.len()).unwrap();
    let model = dir.join("streammoe");
    let a = model.join("artifacts");
    let run = |staging: Option<usize>, overlap: usize, lm: bool| -> (Vec<Vec<f32>>, Vec<u8>, u64) {
        let cfg = OpenConfig {
            n_ctx: 640,
            mem_cap: Some(2 * GIB),
            max_checkpoints: 0,
            prefill_chunk: Some(64),
            prefill_staging_experts: staging,
            prefill_overlap_groups: Some(overlap),
            layer_major: Some(lm),
            layer_major_chunks: lm.then_some(4),
            ..OpenConfig::default()
        };
        let mut ctx = Ctx::open_with(
            &a.join("model_weights.bin"),
            &a.join("model_weights.json"),
            &a.join("vocab.bin"),
            &model.join("root"),
            false,
            &cfg,
        )
        .expect("open the tiny model");
        let mut logits = vec![0.0f32; ctx.n_vocab()];
        ctx.reset_expert_io_stats();
        ctx.eval_prompt(&g.ids, 0, 0, &mut logits).unwrap();
        let st = ctx.expert_io_stats();
        let staged: u64 = st.per_layer.iter().map(|l| l.prefill_reads + l.prefill_hot).sum();
        let mut state = vec![0u8; ctx.state_size()];
        let len = ctx.state_save(&mut state).unwrap();
        state.truncate(len);
        let mut rows = vec![logits.clone()];
        for (k, &tok) in g.decode.iter().enumerate() {
            ctx.eval_token(tok, g.ids.len() + k, 0, &mut logits).unwrap();
            rows.push(logits.clone());
        }
        (rows, state, staged)
    };
    for (staging, overlap) in [(None, 0), (Some(16), 1)] {
        let (base_rows, base_state, base_staged) = run(staging, overlap, false);
        let (rows, state, staged) = run(staging, overlap, true);
        eprintln!("[qnext_engine] layer-major, staging {staging:?}: staged {staged} experts, chunk-major {base_staged}");
        for (i, (x, y)) in rows.iter().zip(&base_rows).enumerate() {
            let diff = x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            assert_eq!(diff, 0, "staging {staging:?} row {i}: {diff} logits differ");
        }
        assert!(state == base_state, "staging {staging:?}: state differs");
        assert!(staged < base_staged, "staging {staging:?}: staged {staged} vs {base_staged}");
    }
}
