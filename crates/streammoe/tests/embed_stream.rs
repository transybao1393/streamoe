//! Streamed token embedding ([`OpenConfig::embed_stream`]): a context
//! that reads embedding rows on demand from `model_weights.bin` must
//! give **bit-identical** logits to one that keeps the table resident —
//! through a multi-chunk batched prefill (GPU gather over uploaded
//! rows), greedy per-token decode (CPU lookup), `state_save` /
//! `state_load`, a second prefill (cache hits), and `Ctx::embed`. The
//! streamed context's hot tier must get the table's bytes.
//!
//! Needs the a3b model; check the host first. Runs under a 4 GiB cap
//! (`STREAMMOE_TEST_CAP_GIB` overrides), one context at a time.
//! ```text
//! scripts/check_test_host.sh
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b \
//!     --test embed_stream -- --include-ignored --test-threads=1
//! ```

#![cfg(all(target_os = "macos", feature = "model-qwen3-6-35b-a3b"))]

#[allow(dead_code)]
mod common;

use common::diff_helpers::default_a3b_paths;
use streammoe::{Ctx, OpenConfig};

const GIB: u64 = 1 << 30;
const DECODE: usize = 8;

fn open(stream: bool) -> Ctx {
    let p = default_a3b_paths();
    let gib: f64 = std::env::var("STREAMMOE_TEST_CAP_GIB").map_or(4.0, |g| g.parse().expect("GiB"));
    let cfg = OpenConfig {
        n_ctx: 2048,
        mem_cap: Some((gib * GIB as f64) as u64),
        max_checkpoints: 0,
        // Three chunks for the prompt below.
        prefill_chunk: Some(64),
        embed_stream: stream,
        ..OpenConfig::default()
    };
    Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &p.root, false, &cfg).expect("open")
}

fn argmax(l: &[f32]) -> i32 {
    l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as i32).unwrap()
}

/// Every logits vector in order, plus a few embeddings.
fn run(ctx: &mut Ctx) -> Vec<Vec<f32>> {
    // Repeats inside the prompt exercise in-chunk duplicates; ids span
    // the vocabulary.
    let prompt: Vec<i32> = (0..150).map(|i| (1000 + 1637 * (i % 97)) % 248_000).collect();
    let mut logits = vec![0f32; ctx.n_vocab()];
    ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
    let mut all = vec![logits.clone()];
    let mut pos = prompt.len();
    for _ in 0..DECODE / 2 {
        ctx.eval_token(argmax(&logits), pos, 0, &mut logits).expect("decode");
        pos += 1;
        all.push(logits.clone());
    }
    // Save, decode on, then load and decode the same token again.
    let mut snap = vec![0u8; ctx.state_size()];
    let n = ctx.state_save(&mut snap).expect("state_save");
    let next = argmax(&logits);
    for _ in 0..DECODE / 2 {
        ctx.eval_token(argmax(&logits), pos, 0, &mut logits).expect("decode");
        pos += 1;
        all.push(logits.clone());
    }
    ctx.state_load(&snap[..n]).expect("state_load");
    ctx.eval_token(next, prompt.len() + DECODE / 2, 0, &mut logits).expect("decode after load");
    all.push(logits.clone());
    ctx.memory_clear();
    ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill again");
    all.push(logits.clone());
    for tok in [0, 1, 151_643, 248_000] {
        let mut e = vec![0f32; ctx.hidden_dim()];
        ctx.embed(tok, &mut e).expect("embed");
        all.push(e);
    }
    all
}

fn assert_bit_identical(a: &[Vec<f32>], b: &[Vec<f32>]) {
    assert_eq!(a.len(), b.len());
    for (step, (x, y)) in a.iter().zip(b).enumerate() {
        let diff = x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let max = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        assert!(diff == 0, "step {step}: {diff} values differ (max |d| {max:e})");
    }
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn streamed_embedding_matches_resident() {
    let (resident, hot_off) = {
        let mut ctx = open(false);
        let got = run(&mut ctx);
        (got, ctx.expert_io_stats().hot.expect("hot tier").capacity)
    };
    let mut ctx = open(true);
    let got = run(&mut ctx);
    let hot_on = ctx.expert_io_stats().hot.expect("hot tier").capacity;
    eprintln!("[embed_stream] hot tier slots: resident {hot_off}, streamed {hot_on}");
    assert_bit_identical(&resident, &got);
    // 0.27 GiB of table minus a few MiB of cache: ≥ 100 more slots of
    // ~1.7 MiB experts.
    assert!(hot_on >= hot_off + 100, "hot tier {hot_off} → {hot_on}");
}
