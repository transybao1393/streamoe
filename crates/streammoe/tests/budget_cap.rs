//! Hard memory cap ([`OpenConfig::mem_cap`]) behavior.
//!
//! The first group runs without model files: config validation and
//! the open-time budget plan both fail before any model file is
//! mapped. The `#[ignore]`d group loads the a3b model and checks the
//! measured footprint stays under the cap through open + eval.
//!
//! Run the model group (check the host first with
//! `scripts/check_test_host.sh`):
//! ```text
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b \
//!     --test budget_cap -- --include-ignored --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use streammoe::{Ctx, Error, ExpertIoMode, OpenConfig};

#[allow(dead_code)]
mod common;

const GIB: u64 = 1 << 30;

fn scratch_file(name: &str, len: u64) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("streammoe-budget-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(name);
    // Sparse: set_len reserves no disk blocks.
    std::fs::File::create(&p).unwrap().set_len(len).unwrap();
    p
}

fn open_with(weights: &Path, cfg: &OpenConfig) -> Result<Ctx, Error> {
    let none = Path::new("/nonexistent/streammoe");
    Ctx::open_with(weights, none, none, none, false, cfg)
}

#[test]
fn invalid_n_ctx_rejected_before_io() {
    let cfg = OpenConfig { n_ctx: 0, ..OpenConfig::default() };
    let err = open_with(Path::new("/nonexistent/w.bin"), &cfg).err().expect("must fail");
    assert!(matches!(err, Error::InvalidConfig { .. }), "{err:?}");
}

#[test]
fn mmap_under_cap_rejected_before_io() {
    let cfg = OpenConfig {
        mem_cap: Some(10 * GIB),
        expert_io: Some(ExpertIoMode::Mmap),
        ..OpenConfig::default()
    };
    let err = open_with(Path::new("/nonexistent/w.bin"), &cfg).err().expect("must fail");
    assert!(matches!(err, Error::InvalidConfig { .. }), "{err:?}");
}

#[test]
fn plan_over_cap_fails_before_mapping() {
    // A 4 GiB (sparse) weights file can't fit a 2 GiB cap.
    let w = scratch_file("weights_4g.bin", 4 * GIB);
    let cfg = OpenConfig {
        n_ctx: 1024,
        mem_cap: Some(2 * GIB),
        ..OpenConfig::default()
    };
    let err = open_with(&w, &cfg).err().expect("must fail");
    match err {
        Error::OverBudget { detail } => {
            assert!(detail.contains("weights="), "{detail}");
        }
        other => panic!("expected OverBudget, got {other:?}"),
    }
}

#[test]
fn full_context_kv_does_not_fit_a_small_cap() {
    // The historical 1M-token default can't fit 10 GiB once KV is
    // accounted (a3b: 40 KiB/token → 40 GiB).
    let w = scratch_file("weights_small.bin", 1 << 20);
    let cfg = OpenConfig { mem_cap: Some(10 * GIB), ..OpenConfig::default() };
    let err = open_with(&w, &cfg).err().expect("must fail");
    assert!(matches!(err, Error::OverBudget { .. }), "{err:?}");
}

/// The application's fixed buffers (`streammoe-net` frames) are planned
/// inside the cap: buffers alone larger than the cap fail the plan.
#[test]
fn host_buffers_count_against_the_cap() {
    let w = scratch_file("weights_host.bin", 1 << 20);
    let cfg = OpenConfig {
        n_ctx: 64,
        mem_cap: Some(2 * GIB),
        runtime_overhead: 0,
        hot_tier_bytes: Some(0),
        host_buffers: 3 * GIB,
        ..OpenConfig::default()
    };
    match open_with(&w, &cfg).err().expect("must fail") {
        Error::OverBudget { detail } => assert!(detail.contains("host_buffers="), "{detail}"),
        other => panic!("expected OverBudget, got {other:?}"),
    }
}

#[cfg(feature = "model-qwen3-6-35b-a3b")]
mod model {
    use super::*;

    use crate::common::diff_helpers;

    fn open_capped(cap: u64, n_ctx: usize) -> Result<Ctx, Error> {
        let p = diff_helpers::default_a3b_paths();
        let cfg = OpenConfig { n_ctx, mem_cap: Some(cap), ..OpenConfig::default() };
        Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &p.root, false, &cfg)
    }

    #[test]
    #[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
    fn a3b_under_10gib_cap_stays_under_cap() {
        let cap = 10 * GIB;
        let mut ctx = open_capped(cap, 4096).expect("open under 10 GiB");
        let mut logits = vec![0f32; ctx.n_vocab()];
        let prompt: Vec<i32> = (0..64).map(|i| 1000 + i).collect();
        ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
        let mut tok = 1000;
        for pos in prompt.len()..prompt.len() + 32 {
            ctx.eval_token(tok, pos, 0, &mut logits).expect("decode");
            tok = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as i32)
                .unwrap();
        }
        let fp = ctx.footprint().expect("footprint");
        eprintln!("[budget_cap] peak {:.2} GiB", fp.peak as f64 / GIB as f64);
        assert!(fp.peak <= cap, "peak footprint {} over cap {cap}", fp.peak);
    }

    /// A long decode at the default `runtime_overhead`: a 130-token
    /// prompt, then 128 tokens one at a time under a 4 GiB cap. Every
    /// `eval_token` re-checks the footprint, so going over fails the
    /// call, and the footprint must stay flat once decode is warm (it
    /// grew ~0.33 MiB per token while metal-rs' autoreleased command
    /// buffers piled up). `STREAMMOE_TEST_ROOT` swaps in another expert
    /// root (e.g. packed 3-bit).
    #[test]
    #[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
    fn a3b_long_decode_stays_under_4gib_cap() {
        let cap = 4 * GIB;
        let p = diff_helpers::default_a3b_paths();
        let root = std::env::var_os("STREAMMOE_TEST_ROOT").map_or(p.root.clone(), PathBuf::from);
        let cfg = OpenConfig { n_ctx: 4096, mem_cap: Some(cap), max_checkpoints: 0, ..OpenConfig::default() };
        let mut ctx = Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &root, false, &cfg).expect("open");
        let open = ctx.footprint().expect("footprint").current;
        let mut logits = vec![0f32; ctx.n_vocab()];
        let prompt: Vec<i32> = (0..130).map(|i| 1000 + 7 * i).collect();
        ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
        let gib = |b: u64| b as f64 / GIB as f64;
        let mut warm = 0;
        for i in 0..128 {
            let tok = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as i32)
                .unwrap();
            if let Err(e) = ctx.eval_token(tok, prompt.len() + i, 0, &mut logits) {
                panic!("decode token {i}: {e:?}");
            }
            if i % 16 == 15 {
                let fp = ctx.footprint().expect("footprint");
                if i == 15 {
                    warm = fp.current;
                }
                eprintln!(
                    "[budget_cap] token {:3}: footprint {:.3} GiB (+{:.0} MiB since open), peak {:.3}",
                    i + 1,
                    gib(fp.current),
                    (fp.current as f64 - open as f64) / (1 << 20) as f64,
                    gib(fp.peak)
                );
            }
        }
        let fp = ctx.footprint().expect("footprint");
        assert!(fp.peak <= cap, "peak footprint {} over cap {cap}", fp.peak);
        let growth = fp.current.saturating_sub(warm);
        assert!(growth < 8 << 20, "footprint grew {growth} bytes over 112 warm decode tokens");
    }

    /// Decode lends the prefill staging and scratch to the hot tier; the
    /// next prefill takes them back. A prefill after a decode that
    /// filled the lent slots must give the same logits as the first
    /// one, and so must the decode after it (`STREAMMOE_HOT_LEND=0` /
    /// `STREAMMOE_HOT_LEND_SCRATCH=0` turn the lending off).
    #[test]
    #[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
    fn a3b_prefill_after_lent_decode_matches_first_prefill() {
        let cap = 4 * GIB;
        let p = diff_helpers::default_a3b_paths();
        let root = std::env::var_os("STREAMMOE_TEST_ROOT").map_or(p.root.clone(), PathBuf::from);
        let cfg = OpenConfig { n_ctx: 4096, mem_cap: Some(cap), max_checkpoints: 0, ..OpenConfig::default() };
        let mut ctx = Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &root, false, &cfg).expect("open");
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as i32).unwrap();
        let prompt: Vec<i32> = (0..40).map(|i| 2000 + 13 * i).collect();
        let n = prompt.len();
        let mut first = vec![0f32; ctx.n_vocab()];
        ctx.eval_prompt(&prompt, 0, 0, &mut first).expect("prefill");
        let own = ctx.expert_io_stats().hot.expect("hot tier").capacity;
        let mut first_decode = vec![0f32; ctx.n_vocab()];
        ctx.eval_token(argmax(&first), n, 0, &mut first_decode).expect("decode");
        let mut logits = first_decode.clone();
        for pos in n + 1..n + 48 {
            ctx.eval_token(argmax(&logits), pos, 0, &mut logits).expect("decode");
        }
        let hot = ctx.expert_io_stats().hot.expect("hot tier");
        eprintln!("[budget_cap] hot tier {own} own slots; while lent {} slots, {} resident", hot.capacity, hot.resident);
        if std::env::var("STREAMMOE_HOT_LEND").map_or(true, |v| v != "0") {
            assert!(hot.capacity > own && hot.resident == hot.capacity, "lent slots unused: {hot:?}");
        }
        ctx.memory_clear();
        let mut again = vec![0f32; ctx.n_vocab()];
        ctx.eval_prompt(&prompt, 0, 0, &mut again).expect("prefill again");
        assert_eq!(ctx.expert_io_stats().hot.unwrap().capacity, own, "prefill reclaims the lent slots");
        assert!(first == again, "prefill after a lent decode differs");
        let mut again_decode = vec![0f32; ctx.n_vocab()];
        ctx.eval_token(argmax(&again), n, 0, &mut again_decode).expect("decode again");
        assert!(first_decode == again_decode, "decode after reclaim differs");
    }

    /// What `runtime_overhead` has to cover, over a mixed workload under
    /// a 4 GiB cap: a three-chunk prompt, a long decode, multi-token
    /// passes with rewinds, two requests decoded together, checkpoints
    /// saved and restored. Logs the footprint beyond the itemized plan
    /// (the plan fills the cap, so that is `peak - (cap - overhead)`,
    /// within one hot-tier slot) after each phase; every eval re-checks
    /// the cap.
    #[test]
    #[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
    fn a3b_mixed_workload_unplanned_footprint() {
        use streammoe::ReqId;
        let cap = 4 * GIB;
        let p = diff_helpers::default_a3b_paths();
        let root = std::env::var_os("STREAMMOE_TEST_ROOT").map_or(p.root.clone(), PathBuf::from);
        let cfg = OpenConfig {
            n_ctx: 4096,
            mem_cap: Some(cap),
            max_checkpoints: 2,
            max_decode_tokens: 4,
            max_requests: 2,
            ..OpenConfig::default()
        };
        let overhead = cfg.runtime_overhead;
        let mut ctx = Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &root, false, &cfg).expect("open");
        let vocab = ctx.n_vocab();
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as i32).unwrap();
        let report = |ctx: &Ctx, phase: &str| {
            let fp = ctx.footprint().expect("footprint");
            let mib = |b: u64| b as f64 / (1 << 20) as f64;
            eprintln!(
                "[reserve] {phase:<28} current {:.3} GiB, peak {:.3} GiB, unplanned at peak {:+.0} MiB of {:.0}",
                fp.current as f64 / GIB as f64,
                fp.peak as f64 / GIB as f64,
                mib(fp.peak) - mib(cap - overhead),
                mib(overhead),
            );
        };
        report(&ctx, "open");
        let mut logits = vec![0f32; vocab];
        let prompt: Vec<i32> = (0..1500).map(|i| 1000 + (i * 37) % 20000).collect();
        ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
        report(&ctx, "prefill 1500 (3 chunks)");
        let mut pos = prompt.len();
        for _ in 0..160 {
            ctx.eval_token(argmax(&logits), pos, 0, &mut logits).expect("decode");
            pos += 1;
        }
        report(&ctx, "decode 160");
        let mut multi = vec![0f32; 4 * vocab];
        for i in 0..16 {
            let t = argmax(&logits);
            let toks = [t, t + 1, t + 2, t + 3];
            ctx.eval_tokens(&toks, pos, &mut multi).expect("eval_tokens");
            let keep = if i % 2 == 0 { 2 } else { 4 };
            ctx.rewind_tokens(keep).expect("rewind");
            pos += keep;
            logits.copy_from_slice(&multi[(keep - 1) * vocab..keep * vocab]);
        }
        report(&ctx, "eval_tokens x16 + rewind");
        ctx.checkpoint_pos(pos as i32).expect("checkpoint");
        let mark = pos;
        for _ in 0..16 {
            ctx.eval_token(argmax(&logits), pos, 0, &mut logits).expect("decode");
            pos += 1;
        }
        ctx.checkpoint_pos(pos as i32).expect("checkpoint 2");
        ctx.restore_to(mark as i32).expect("restore");
        pos = mark;
        for _ in 0..16 {
            ctx.eval_token(argmax(&logits), pos, 0, &mut logits).expect("decode");
            pos += 1;
        }
        report(&ctx, "checkpoint / restore");
        let other = ReqId(1);
        ctx.admit_request(other).expect("admit");
        let mut one = vec![0f32; vocab];
        let short: Vec<i32> = (0..200).map(|i| 3000 + i).collect();
        ctx.eval_request(other, streammoe::StageInput::Tokens(&short), 0, streammoe::StageOutput::Logits(&mut one))
            .expect("prefill request 1");
        let mut pos1 = short.len();
        let mut two = vec![0f32; 2 * vocab];
        for _ in 0..32 {
            let batch = [(ReqId::DEFAULT, argmax(&logits), pos), (other, argmax(&one), pos1)];
            ctx.eval_requests(&batch, &mut two).expect("eval_requests");
            logits.copy_from_slice(&two[..vocab]);
            one.copy_from_slice(&two[vocab..]);
            pos += 1;
            pos1 += 1;
        }
        report(&ctx, "eval_requests x32");
        let fp = ctx.footprint().expect("footprint");
        assert!(fp.peak <= cap, "peak footprint {} over cap {cap}", fp.peak);
    }

    #[test]
    #[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
    fn a3b_tiny_cap_fails_at_open() {
        let err = open_capped(2 * GIB, 4096).err().expect("must fail");
        assert!(matches!(err, Error::OverBudget { .. }), "{err:?}");
    }
}
