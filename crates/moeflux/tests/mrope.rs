//! Multimodal RoPE ([`Ctx::eval_stage_mrope`]) on the real a3b model,
//! under a 3 GiB cap.
//!
//! Text rows (`[p, p, p]` with `p` = the KV slot) must give exactly the
//! logits of the plain path, through batched prefill and per-token
//! decode; shifted positions must change them.
#![cfg(all(target_os = "macos", feature = "model-qwen3-6-35b-a3b"))]

use std::path::PathBuf;

use moeflux::{Ctx, OpenConfig, StageInput, StageOutput};

fn dir(var: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or(default.into()))
}

fn open() -> Ctx {
    let art = dir("MOEFLUX_SMOKE_ARTIFACTS", "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts");
    let root = dir("MOEFLUX_SMOKE_ROOT", "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-root");
    let cfg = OpenConfig {
        n_ctx: 512,
        mem_cap: Some(3 << 30),
        max_checkpoints: 0,
        ..OpenConfig::default()
    };
    Ctx::open_with(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
        &art.join("vocab.bin"),
        &root,
        false,
        &cfg,
    )
    .expect("open a3b under 3 GiB")
}

fn argmax(l: &[f32]) -> i32 {
    (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as i32
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn text_rows_match_plain_rope_bitwise() {
    let mut ctx = open();
    let prompt: Vec<i32> = vec![760, 3841, 13477, 37550, 34208, 916, 279, 15678, 5562, 13];
    let n = prompt.len();
    let mut plain = vec![0f32; ctx.n_vocab()];
    let mut mrope = vec![0f32; ctx.n_vocab()];

    // Batched prefill.
    ctx.eval_stage(StageInput::Tokens(&prompt), 0, StageOutput::Logits(&mut plain)).unwrap();
    let tok = argmax(&plain);
    let mut plain_next = vec![0f32; ctx.n_vocab()];
    ctx.eval_token(tok, n, 0, &mut plain_next).unwrap();

    ctx.memory_clear();
    let rows: Vec<[i32; 3]> = (0..n as i32).map(|p| [p; 3]).collect();
    ctx.eval_stage_mrope(StageInput::Tokens(&prompt), 0, &rows, StageOutput::Logits(&mut mrope))
        .unwrap();
    assert!(plain == mrope, "prefill logits differ with [p, p, p] rows");

    // Per-token decode.
    let mut mrope_next = vec![0f32; ctx.n_vocab()];
    ctx.eval_stage_mrope(
        StageInput::Tokens(&[tok]),
        n,
        &[[n as i32; 3]],
        StageOutput::Logits(&mut mrope_next),
    )
    .unwrap();
    assert!(plain_next == mrope_next, "decode logits differ with a [p, p, p] row");

    // Uneven per-axis positions change the result. (A uniform shift
    // would not: RoPE scores depend on relative positions only.)
    ctx.memory_clear();
    let shifted: Vec<[i32; 3]> = (0..n as i32).map(|p| [p, p + p % 3, p + 2 * (p % 2)]).collect();
    ctx.eval_stage_mrope(StageInput::Tokens(&prompt), 0, &shifted, StageOutput::Logits(&mut mrope))
        .unwrap();
    let diff = plain.iter().zip(&mrope).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("[mrope] shifted-position max |dlogit| = {diff:.3}, argmax {} vs {}", argmax(&plain), argmax(&mrope));
    assert!(diff > 0.05, "shifted M-RoPE positions did not change the logits");
}

/// Image-style hidden rows at uneven `[t, h, w]` positions: batched
/// prefill (Metal `rope_n_tokens_mrope`) and row-by-row decode (CPU
/// `apply_mrope`) must agree, as must the following text token.
#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn image_rows_batched_match_per_row() {
    let mut ctx = open();
    let hd = ctx.hidden_dim();
    let prefix: Vec<i32> = vec![760, 3841, 13477];
    let (gh, gw) = (4usize, 6usize);
    let n_img = gh * gw;
    // Deterministic pseudo-embeddings at a token embedding's scale.
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let rows: Vec<f32> = (0..n_img * hd)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.06
        })
        .collect();
    let p0 = prefix.len() as i32;
    let img_pos: Vec<[i32; 3]> =
        (0..n_img as i32).map(|i| [p0, p0 + i / gw as i32, p0 + i % gw as i32]).collect();
    let next = p0 + gh.max(gw) as i32;
    let after = n_img + prefix.len();
    let tok = 13;

    // Plain positions (one slot per row) give the batched-vs-per-token
    // float noise floor to compare against.
    let plain_pos: Vec<[i32; 3]> = (0..n_img as i32).map(|i| [p0 + i; 3]).collect();
    let run = |ctx: &mut Ctx, batched: bool, img_pos: &[[i32; 3]], next: i32| -> Vec<f32> {
        ctx.memory_clear();
        let pre: Vec<[i32; 3]> = (0..p0).map(|p| [p; 3]).collect();
        ctx.eval_stage_mrope(StageInput::Tokens(&prefix), 0, &pre, StageOutput::Discard).unwrap();
        if batched {
            ctx.eval_stage_mrope(StageInput::Hidden(&rows), prefix.len(), &img_pos, StageOutput::Discard)
                .unwrap();
        } else {
            for i in 0..n_img {
                ctx.eval_stage_mrope(
                    StageInput::Hidden(&rows[i * hd..(i + 1) * hd]),
                    prefix.len() + i,
                    &img_pos[i..i + 1],
                    StageOutput::Discard,
                )
                .unwrap();
            }
        }
        let mut l = vec![0f32; ctx.n_vocab()];
        ctx.eval_stage_mrope(StageInput::Tokens(&[tok]), after, &[[next; 3]], StageOutput::Logits(&mut l))
            .unwrap();
        l
    };
    let maxdiff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let a = run(&mut ctx, true, &img_pos, next);
    let b = run(&mut ctx, false, &img_pos, next);
    let diff = maxdiff(&a, &b);
    let pa = run(&mut ctx, true, &plain_pos, after as i32);
    let pb = run(&mut ctx, false, &plain_pos, after as i32);
    let floor = maxdiff(&pa, &pb);
    let cross = maxdiff(&a, &pa);
    eprintln!(
        "[mrope] image rows batched vs per-row: max |dlogit| {diff:.4} (plain-position floor {floor:.4}; \
         M-RoPE vs plain {cross:.4}), argmax {} vs {}",
        argmax(&a),
        argmax(&b)
    );
    assert_eq!(argmax(&a), argmax(&b));
    assert!(diff < 2.0 * floor + 0.02, "batched and per-row M-RoPE disagree: {diff} vs floor {floor}");
    assert!(cross > 5.0 * diff, "M-RoPE positions barely matter here ({cross}); the check is weak");
}

/// A prompt prefilled in one batched call, in three batched calls (as
/// the chat does around an image), and token by token must agree.
#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn split_batched_prefill_matches_whole() {
    let mut ctx = open();
    let base: Vec<i32> = vec![760, 3841, 13477, 37550, 34208, 916, 279, 15678, 5562, 13];
    let prompt: Vec<i32> = base.iter().cycle().take(40).copied().collect();
    let maxdiff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let mut whole = vec![0f32; ctx.n_vocab()];
    ctx.eval_stage(StageInput::Tokens(&prompt), 0, StageOutput::Logits(&mut whole)).unwrap();

    ctx.memory_clear();
    let mut split = vec![0f32; ctx.n_vocab()];
    ctx.eval_stage(StageInput::Tokens(&prompt[..13]), 0, StageOutput::Discard).unwrap();
    ctx.eval_stage(StageInput::Tokens(&prompt[13..27]), 13, StageOutput::Discard).unwrap();
    ctx.eval_stage(StageInput::Tokens(&prompt[27..]), 27, StageOutput::Logits(&mut split)).unwrap();

    ctx.memory_clear();
    let mut per_tok = vec![0f32; ctx.n_vocab()];
    for (i, &t) in prompt.iter().enumerate() {
        ctx.eval_token(t, i, 0, &mut per_tok).unwrap();
    }
    let (ds, dt) = (maxdiff(&whole, &split), maxdiff(&whole, &per_tok));
    eprintln!("[mrope] whole vs 3-call prefill {ds:.4}; whole vs per-token {dt:.4}");
    assert!(ds < 2.0 * dt + 0.02, "split prefill diverges: {ds} (per-token floor {dt})");
}
