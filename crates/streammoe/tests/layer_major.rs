//! Layer-major prefill ([`OpenConfig::layer_major`]): a prompt longer
//! than one prefill chunk runs layer by layer over a window of chunks,
//! so each layer's experts are read once per window instead of once
//! per chunk. It must give **bit-identical** results to chunk-major
//! prefill — the prefill logits, the state it leaves (`state_save`
//! bytes), the greedy decode after it, and a second prefill after
//! that decode (which reclaims the buffers lent to the hot tier) —
//! with full staging (staging first and overlapped) and bounded
//! staging (staging first and double-buffered), and over several
//! windows.
//!
//! Needs the a3b model; check the host first. Runs under a 4 GiB cap
//! (`STREAMMOE_TEST_CAP_GIB` overrides). `STREAMMOE_TEST_ROOT` points
//! at another expert root (e.g. the packed 3-bit one).
//! ```text
//! scripts/check_test_host.sh
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b \
//!     --test layer_major -- --include-ignored --test-threads=1
//! ```

#![cfg(all(target_os = "macos", feature = "model-qwen3-6-35b-a3b"))]

#[allow(dead_code)]
mod common;

use std::path::PathBuf;

use common::diff_helpers::default_a3b_paths;
use streammoe::{Ctx, OpenConfig};

const GIB: u64 = 1 << 30;
const DECODE: usize = 6;
/// Four chunks of 64: one window at the default 4 chunks per window.
const PROMPT: usize = 230;

#[derive(Clone, Copy, Debug)]
struct Setup {
    staging: Option<usize>,
    overlap: usize,
    /// `None`: chunk-major. `Some(c)`: layer-major, `c` chunks a window.
    layer_major: Option<usize>,
}

fn open(s: Setup) -> Ctx {
    let p = default_a3b_paths();
    let root = std::env::var_os("STREAMMOE_TEST_ROOT").map_or(p.root.clone(), PathBuf::from);
    let gib: f64 = std::env::var("STREAMMOE_TEST_CAP_GIB").map_or(4.0, |g| g.parse().expect("GiB"));
    let cfg = OpenConfig {
        n_ctx: 2048,
        mem_cap: Some((gib * GIB as f64) as u64),
        max_checkpoints: 0,
        prefill_chunk: Some(64),
        prefill_staging_experts: s.staging,
        prefill_overlap_groups: Some(s.overlap),
        layer_major: Some(s.layer_major.is_some()),
        layer_major_chunks: s.layer_major,
        ..OpenConfig::default()
    };
    Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &root, false, &cfg).expect("open")
}

fn argmax(l: &[f32]) -> i32 {
    l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as i32).unwrap()
}

struct Run {
    /// Prefill, each decode step, the second prefill.
    logits: Vec<Vec<f32>>,
    /// `state_save` right after the first prefill.
    state: Vec<u8>,
    /// Experts the first prefill staged (from disk or the hot tier).
    staged: u64,
}

/// Prefill, save the state, greedy decode, clear, prefill again.
fn run(ctx: &mut Ctx) -> Run {
    let prompt: Vec<i32> = (0..PROMPT as i32).map(|i| 1000 + 7 * i).collect();
    let mut logits = vec![0f32; ctx.n_vocab()];
    ctx.reset_expert_io_stats();
    ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
    let st = ctx.expert_io_stats();
    let staged: u64 = st.per_layer.iter().map(|l| l.prefill_reads + l.prefill_hot).sum();
    let mut state = vec![0u8; ctx.state_size()];
    let len = ctx.state_save(&mut state).expect("state_save");
    state.truncate(len);
    let mut all = vec![logits.clone()];
    for pos in prompt.len()..prompt.len() + DECODE {
        ctx.eval_token(argmax(&logits), pos, 0, &mut logits).expect("decode");
        all.push(logits.clone());
    }
    ctx.memory_clear();
    ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill again");
    all.push(logits.clone());
    Run { logits: all, state, staged }
}

fn assert_bit_identical(label: &str, a: &Run, b: &Run) {
    assert_eq!(a.logits.len(), b.logits.len());
    for (step, (x, y)) in a.logits.iter().zip(&b.logits).enumerate() {
        let diff = x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let max = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        assert!(diff == 0, "[{label}] step {step}: {diff} logits differ (max |d| {max:e})");
    }
    assert_eq!(a.state.len(), b.state.len(), "[{label}] state size");
    let first = a.state.iter().zip(&b.state).position(|(p, q)| p != q);
    assert!(first.is_none(), "[{label}] state differs from byte {first:?}");
}

/// Chunk-major vs layer-major for each staging setup.
fn check(setups: &[(Option<usize>, usize)]) {
    for &(staging, overlap) in setups {
        let base = {
            let mut ctx = open(Setup { staging, overlap, layer_major: None });
            run(&mut ctx)
        };
        // One window of 4 chunks; windows of 2 + 2; 3 + a 1-chunk tail.
        for chunks in [4, 2, 3] {
            let s = Setup { staging, overlap, layer_major: Some(chunks) };
            let mut ctx = open(s);
            let got = run(&mut ctx);
            eprintln!(
                "[layer_major] staging {staging:?}, overlap {overlap}, {chunks} chunks a window: \
                 staged {} experts, chunk-major {}",
                got.staged, base.staged
            );
            assert_bit_identical(&format!("{s:?}"), &base, &got);
            // Each layer's experts are read once per window, not per chunk.
            assert!(got.staged < base.staged, "{s:?}: staged {} vs {}", got.staged, base.staged);
        }
    }
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn layer_major_matches_chunk_major_full_staging() {
    check(&[(None, 0), (None, 8)]);
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn layer_major_matches_chunk_major_bounded_staging() {
    check(&[(Some(64), 0), (Some(64), 1), (Some(16), 1)]);
}
