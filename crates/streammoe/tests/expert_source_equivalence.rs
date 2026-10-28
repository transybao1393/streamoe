//! Expert-source equivalence: the bytes an expert dispatch reads must
//! not depend on where they came from. mmap, pread with no hot tier,
//! pread with a hot tier, and a profile-seeded hot tier all run the
//! same prompt and must produce **bit-identical** logits and the same
//! greedy tokens.
//!
//! Every run here forces the CPU decode attention
//! (`STREAMMOE_DECODE_SDPA=cpu`). By default mmap mode takes the GPU
//! SDPA once the KV cache holds 32 rows and pread mode the CPU one;
//! the two sum in a different order, so mmap and pread would agree to
//! ~1e-5 on the logits (same greedy tokens), not bit for bit, for a
//! reason that has nothing to do with the expert bytes.
//!
//! Needs the a3b model; check the host first:
//! ```text
//! scripts/check_test_host.sh
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b \
//!     --test expert_source_equivalence -- --include-ignored --test-threads=1
//! ```

#![cfg(all(target_os = "macos", feature = "model-qwen3-6-35b-a3b"))]

#[allow(dead_code)]
mod common;

use common::diff_helpers::default_a3b_paths;
use streammoe::{Ctx, ExpertIoMode, OpenConfig};

const GIB: u64 = 1 << 30;
const N_CTX: usize = 2048;
const DECODE: usize = 16;

fn open(cfg: OpenConfig) -> Ctx {
    // SAFETY: set before the first open in this process reads it, and
    // always to the same value; nothing else touches the environment.
    unsafe { std::env::set_var("STREAMMOE_DECODE_SDPA", "cpu") };
    let p = default_a3b_paths();
    Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &p.root, false, &cfg).expect("open")
}

/// Prefill a fixed prompt, then greedy-decode. Returns every step's
/// logits and the chosen tokens.
fn run(ctx: &mut Ctx) -> (Vec<Vec<f32>>, Vec<i32>) {
    let prompt: Vec<i32> = (0..48).map(|i| 1000 + 7 * i).collect();
    let mut logits = vec![0f32; ctx.n_vocab()];
    ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
    let mut all = vec![logits.clone()];
    let mut toks = Vec::new();
    for pos in prompt.len()..prompt.len() + DECODE {
        let tok = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as i32)
            .unwrap();
        toks.push(tok);
        ctx.eval_token(tok, pos, 0, &mut logits).expect("decode");
        all.push(logits.clone());
    }
    (all, toks)
}

fn assert_same(label: &str, a: &(Vec<Vec<f32>>, Vec<i32>), b: &(Vec<Vec<f32>>, Vec<i32>)) {
    assert_eq!(a.1, b.1, "[{label}] greedy tokens differ");
    for (step, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
        let bits_equal = x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
        assert!(bits_equal, "[{label}] logits differ at step {step}");
    }
}

fn cfg(mode: ExpertIoMode, hot: u64) -> OpenConfig {
    OpenConfig {
        n_ctx: N_CTX,
        expert_io: Some(mode),
        hot_tier_bytes: Some(hot),
        ..OpenConfig::default()
    }
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn pread_hot_tier_sizes_match_pread_without_tier() {
    let base = run(&mut open(cfg(ExpertIoMode::Pread, 0)));
    for hot in [GIB / 2, 2 * GIB] {
        let mut ctx = open(cfg(ExpertIoMode::Pread, hot));
        let got = run(&mut ctx);
        let st = ctx.expert_io_stats();
        eprintln!("[equiv] hot={hot} stats={st:?}");
        assert!(st.hot.expect("tier attached").admissions > 0);
        assert_same(&format!("hot={hot}"), &base, &got);
    }
}

#[test]
#[ignore = "needs a3b weights and RAM for mmap mode; run scripts/check_test_host.sh first"]
fn mmap_matches_pread() {
    let pread = run(&mut open(cfg(ExpertIoMode::Pread, 0)));
    let mmap = run(&mut open(OpenConfig {
        n_ctx: N_CTX,
        expert_io: Some(ExpertIoMode::Mmap),
        ..OpenConfig::default()
    }));
    assert_same("mmap vs pread", &pread, &mmap);
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn profile_seeded_tier_matches_and_starts_warm() {
    let mut first = open(cfg(ExpertIoMode::Pread, 0));
    let base = run(&mut first);
    let profile = std::env::temp_dir().join(format!("streammoe-a3b-{}.prof", std::process::id()));
    first.save_routing_profile(&profile).expect("save profile");
    drop(first);

    let mut seeded = open(OpenConfig {
        hot_tier_profile: Some(profile.clone()),
        ..cfg(ExpertIoMode::Pread, 2 * GIB)
    });
    let resident = seeded.expert_io_stats().hot.unwrap().resident;
    assert!(resident > 0, "profile seeded nothing");
    let got = run(&mut seeded);
    assert_same("profile-seeded", &base, &got);
    let _ = std::fs::remove_file(&profile);
}
