//! Bounded prefill staging ([`OpenConfig::prefill_staging_experts`]):
//! a context that stages only a few experts of a layer at a time, and
//! so runs each batched MoE layer in expert groups, must give
//! **bit-identical** logits to one that stages every expert — through
//! a multi-chunk prefill, the decode after it (which borrows the
//! smaller staging buffer as hot-tier slots), and a second prefill
//! that reclaims them.
//!
//! Needs the a3b model; check the host first. Runs under a 4 GiB cap
//! (`STREAMMOE_TEST_CAP_GIB` overrides). `STREAMMOE_TEST_ROOT` points
//! at another expert root (e.g. the packed 3-bit one).
//! ```text
//! scripts/check_test_host.sh
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b \
//!     --test bounded_staging -- --include-ignored --test-threads=1
//! ```

#![cfg(all(target_os = "macos", feature = "model-qwen3-6-35b-a3b"))]

#[allow(dead_code)]
mod common;

use std::path::PathBuf;

use common::diff_helpers::default_a3b_paths;
use streammoe::{Ctx, OpenConfig};

const GIB: u64 = 1 << 30;
const DECODE: usize = 8;

fn open(staging: Option<usize>) -> Ctx {
    let p = default_a3b_paths();
    let root = std::env::var_os("STREAMMOE_TEST_ROOT").map_or(p.root.clone(), PathBuf::from);
    let gib: f64 = std::env::var("STREAMMOE_TEST_CAP_GIB").map_or(4.0, |g| g.parse().expect("GiB"));
    let cfg = OpenConfig {
        n_ctx: 2048,
        mem_cap: Some((gib * GIB as f64) as u64),
        max_checkpoints: 0,
        // Three chunks for the prompt below.
        prefill_chunk: Some(64),
        prefill_staging_experts: staging,
        ..OpenConfig::default()
    };
    Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &root, false, &cfg).expect("open")
}

fn argmax(l: &[f32]) -> i32 {
    l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as i32).unwrap()
}

/// Prefill, greedy decode, clear, prefill again. Every logits vector
/// in order.
fn run(ctx: &mut Ctx) -> Vec<Vec<f32>> {
    let prompt: Vec<i32> = (0..150).map(|i| 1000 + 7 * i).collect();
    let mut logits = vec![0f32; ctx.n_vocab()];
    ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
    let mut all = vec![logits.clone()];
    for pos in prompt.len()..prompt.len() + DECODE {
        ctx.eval_token(argmax(&logits), pos, 0, &mut logits).expect("decode");
        all.push(logits.clone());
    }
    ctx.memory_clear();
    ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill again");
    all.push(logits.clone());
    all
}

fn assert_bit_identical(label: &str, a: &[Vec<f32>], b: &[Vec<f32>]) {
    assert_eq!(a.len(), b.len());
    for (step, (x, y)) in a.iter().zip(b).enumerate() {
        let diff = x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let max = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        assert!(diff == 0, "[{label}] step {step}: {diff} logits differ (max |d| {max:e})");
    }
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn bounded_staging_matches_full_staging() {
    let full = {
        let mut ctx = open(None);
        run(&mut ctx)
    };
    // 16 slots: ~a dozen groups per layer chunk; 40 → 40 (a multiple
    // of 8) with a short last group.
    for staging in [16, 40] {
        let mut ctx = open(Some(staging));
        let got = run(&mut ctx);
        let hot = ctx.expert_io_stats().hot.expect("hot tier");
        eprintln!("[bounded_staging] {staging} slots: hot tier {hot:?}");
        assert_bit_identical(&format!("{staging} slots"), &full, &got);
    }
}
