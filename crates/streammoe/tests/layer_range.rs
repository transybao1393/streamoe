//! Layer-range execution ([`OpenConfig::layer_range`], plan phase 3).
//!
//! The first group runs without model files, against a fake manifest
//! ([`common::fake_model`]): range validation, the stage-call checks
//! and the range-scoped budget plan, all of which fail or pass before
//! any GPU resource is built.
//!
//! The `#[ignore]`d group loads the a3b model:
//! - a whole-model `eval_stage` matches `eval_prompt` / `eval_token`
//!   bit for bit;
//! - 40 layers vs 20 + 10 + 10 in one process match within the oracle
//!   tolerance (the plan's phase 3 exit criterion);
//! - per-range `state_save` / `state_load` round-trips;
//! - a mode B range under a 10 GiB cap holds all its experts in the
//!   hot tier and never reads an expert from disk after open.
//!
//! ```text
//! scripts/check_test_host.sh
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b \
//!     --test layer_range -- --include-ignored --test-threads=1
//! ```

#![cfg(target_os = "macos")]

#[allow(dead_code)]
mod common;

use common::fake_model::FakeModel;
use streammoe::riir::budget::{Account, BudgetPlan};
use streammoe::riir::io::weight_file::filtered_bytes;
use streammoe::riir::{VARIANT, range};
use streammoe::{Error, ExpertIoMode, OpenConfig, StageInput, StageOutput};

fn cfg(range: std::ops::Range<usize>) -> OpenConfig {
    OpenConfig {
        n_ctx: 64,
        layer_range: Some(range),
        ..OpenConfig::default()
    }
}

#[test]
fn bad_ranges_rejected_before_io() {
    let n = VARIANT.num_layers;
    let fake = FakeModel::new("badrange");
    for r in [3..3, 0..n + 1] {
        let err = fake.open(&cfg(r.clone())).err().expect("must fail");
        assert!(matches!(err, Error::InvalidConfig { .. }), "{r:?}: {err:?}");
    }
}

/// A partial range opens on the whole-model manifest (the variant
/// guard reads the unfiltered manifest) and reports its ends.
#[test]
fn partial_ranges_open_and_report_ends() {
    let n = VARIANT.num_layers;
    let fake = FakeModel::new("ends");
    let first = fake.open(&cfg(0..n / 2)).expect("first range");
    assert!(first.has_embed() && !first.has_head());
    assert_eq!(first.layer_range(), 0..n / 2);
    let mid = fake.open(&cfg(n / 2..n - 1)).expect("mid range");
    assert!(!mid.has_embed() && !mid.has_head());
    let last = fake.open(&cfg(n - 1..n)).expect("last range");
    assert!(!last.has_embed() && last.has_head());
    let full = fake.open(&OpenConfig { n_ctx: 64, ..OpenConfig::default() }).expect("full");
    assert!(full.has_embed() && full.has_head());
    assert_eq!(full.layer_range(), 0..n);
}

/// Calls that don't fit the range fail with `StageMismatch` before
/// any GPU work.
#[test]
fn stage_calls_checked_against_range() {
    let n = VARIANT.num_layers;
    let hd = VARIANT.hidden_dim;
    let fake = FakeModel::new("stage");
    let mut mid = fake.open(&cfg(1..n - 1)).expect("mid range");
    let mut logits = vec![0f32; mid.n_vocab()];
    let mut hidden = vec![0f32; 2 * hd];

    let stage_err = |r: Result<(), Error>| match r {
        Err(Error::StageMismatch { detail }) => detail,
        other => panic!("expected StageMismatch, got {other:?}"),
    };
    // Whole-model entry points need the whole model.
    stage_err(mid.eval_prompt(&[1, 2], 0, 0, &mut logits));
    stage_err(mid.eval_token(1, 0, 0, &mut logits));
    // No embedding mid-model.
    stage_err(mid.eval_stage(StageInput::Tokens(&[1, 2]), 0, StageOutput::Hidden(&mut hidden)));
    // No LM head mid-model.
    let h_in = vec![0f32; 2 * hd];
    stage_err(mid.eval_stage(StageInput::Hidden(&h_in), 0, StageOutput::Logits(&mut logits)));
    // Ragged hidden input, and an output of the wrong size.
    stage_err(mid.eval_stage(StageInput::Hidden(&h_in[1..]), 0, StageOutput::Discard));
    let detail = stage_err(mid.eval_stage(
        StageInput::Hidden(&h_in),
        0,
        StageOutput::Hidden(&mut hidden[..hd]),
    ));
    assert!(detail.contains("hidden output"), "{detail}");
    // Empty input is a no-op, like an empty `eval_prompt`.
    mid.eval_stage(StageInput::Hidden(&[]), 0, StageOutput::Discard).expect("empty");
}

/// The cap plan counts only the range: a cap the whole model's KV
/// can't fit passes the plan for a short range (the fake model may
/// then fail to build GPU resources, but not on the budget).
#[test]
fn range_budget_plans_only_its_layers() {
    let n = VARIANT.num_layers;
    let fake = FakeModel::new("budget");
    // One full-attn layer's worth of range.
    let first_full = (0..n)
        .find(|&i| VARIANT.layer_kind(i) == streammoe::riir::variants::LayerKind::FullAttn)
        .unwrap();
    let short = first_full..first_full + 1;
    let n_ctx = 4096;
    let overhead = 0;
    let plan = |r: &std::ops::Range<usize>| {
        let keep = |name: &str| range::keeps_tensor(r, name);
        let keep: Option<&dyn Fn(&str) -> bool> = (!range::is_full(r)).then_some(&keep);
        let w = filtered_bytes(&fake.weights, &fake.manifest, keep).unwrap();
        let chunk = streammoe::riir::config::default_prefill_chunk(n_ctx, true);
        BudgetPlan::for_range(n_ctx, ExpertIoMode::Pread, w, overhead, r, 1, chunk)
    };
    let full_plan = plan(&(0..n));
    let short_plan = plan(&short);
    assert!(short_plan.get(Account::Kv) < full_plan.get(Account::Kv));
    let cap = (short_plan.total() + full_plan.total()) / 2;

    let capped = |r: std::ops::Range<usize>| OpenConfig {
        n_ctx,
        mem_cap: Some(cap),
        runtime_overhead: overhead,
        hot_tier_bytes: Some(0),
        layer_range: Some(r),
        ..OpenConfig::default()
    };
    match fake.open(&capped(0..n)).err().expect("full must fail") {
        Error::OverBudget { detail } => assert!(detail.contains("kv="), "{detail}"),
        other => panic!("full range: expected OverBudget, got {other:?}"),
    }
    if let Err(Error::OverBudget { detail }) = fake.open(&capped(short)) {
        panic!("short range over budget: {detail}");
    }
}

#[cfg(feature = "model-qwen3-6-35b-a3b")]
mod model {
    use super::*;

    use crate::common::diff_helpers::{argmax, assert_logits_close, default_a3b_paths};
    use streammoe::Ctx;

    const GIB: u64 = 1 << 30;
    const N_CTX: usize = 2048;
    const DECODE: usize = 12;

    fn open(range: Option<std::ops::Range<usize>>, extra: OpenConfig) -> Ctx {
        let p = default_a3b_paths();
        let cfg = OpenConfig { n_ctx: N_CTX, layer_range: range, ..extra };
        Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &p.root, false, &cfg).expect("open")
    }

    fn prompt() -> Vec<i32> {
        (0..40).map(|i| 1000 + 13 * i).collect()
    }

    /// Whole model via `eval_prompt` + greedy `eval_token`. Returns the
    /// logits after the prompt and after each decode step, and the
    /// decoded tokens.
    fn run_whole() -> (Vec<Vec<f32>>, Vec<i32>) {
        let mut ctx = open(None, OpenConfig::default());
        let prompt = prompt();
        let mut logits = vec![0f32; ctx.n_vocab()];
        ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
        let mut all = vec![logits.clone()];
        let mut toks = Vec::new();
        for pos in prompt.len()..prompt.len() + DECODE {
            let tok = argmax(&logits);
            toks.push(tok);
            ctx.eval_token(tok, pos, 0, &mut logits).expect("decode");
            all.push(logits.clone());
        }
        (all, toks)
    }

    /// One pass through a chain of stages: tokens in, logits out.
    fn chain(stages: &mut [Ctx], input: StageInput<'_>, pos: usize, logits: &mut [f32]) {
        let n = input.n_tokens().unwrap();
        let hd = VARIANT.hidden_dim;
        let mut cur: Vec<f32> = Vec::new();
        let last = stages.len() - 1;
        for (i, st) in stages.iter_mut().enumerate() {
            let inp = if i == 0 { input } else { StageInput::Hidden(&cur) };
            if i == last {
                st.eval_stage(inp, pos, StageOutput::Logits(logits)).expect("last stage");
            } else {
                let mut next = vec![0f32; n * hd];
                st.eval_stage(inp, pos, StageOutput::Hidden(&mut next)).expect("stage");
                cur = next;
            }
        }
    }

    /// Feed the whole model's token sequence through `stages`; returns
    /// the logits at the same steps as [`run_whole`].
    fn run_chain(stages: &mut [Ctx], toks: &[i32]) -> Vec<Vec<f32>> {
        let prompt = prompt();
        let mut logits = vec![0f32; VARIANT.vocab_size];
        chain(stages, StageInput::Tokens(&prompt), 0, &mut logits);
        let mut all = vec![logits.clone()];
        for (i, &tok) in toks.iter().enumerate() {
            chain(stages, StageInput::Tokens(&[tok]), prompt.len() + i, &mut logits);
            all.push(logits.clone());
        }
        all
    }

    #[test]
    #[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
    fn whole_model_eval_stage_matches_eval_prompt_bitwise() {
        let (want, toks) = run_whole();
        let mut whole = vec![open(None, OpenConfig::default())];
        let got = run_chain(&mut whole, &toks);
        for (step, (a, b)) in want.iter().zip(&got).enumerate() {
            let same = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same, "logits differ at step {step}");
        }
    }

    /// Plan phase 3 exit: 40 layers vs 20 + 10 + 10.
    #[test]
    #[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
    fn split_20_10_10_matches_whole_model() {
        let (want, toks) = run_whole();
        let mut stages = vec![
            open(Some(0..20), OpenConfig::default()),
            open(Some(20..30), OpenConfig::default()),
            open(Some(30..40), OpenConfig::default()),
        ];
        let got = run_chain(&mut stages, &toks);
        for (step, (a, b)) in want.iter().zip(&got).enumerate() {
            assert_logits_close(&format!("split step {step}"), a, b);
        }
    }

    /// A mid-model range snapshot restores its own state exactly.
    #[test]
    #[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
    fn range_state_save_load_roundtrip() {
        let hd = VARIANT.hidden_dim;
        let mut first = open(Some(0..20), OpenConfig::default());
        let mut mid = open(Some(20..40), OpenConfig::default());
        let prompt = prompt();
        let mut h = vec![0f32; prompt.len() * hd];
        first.eval_stage(StageInput::Tokens(&prompt), 0, StageOutput::Hidden(&mut h)).unwrap();
        let mut logits = vec![0f32; VARIANT.vocab_size];
        mid.eval_stage(StageInput::Hidden(&h), 0, StageOutput::Logits(&mut logits)).unwrap();
        let mut snap_a = vec![0u8; first.state_size()];
        let mut snap_b = vec![0u8; mid.state_size()];
        first.state_save(&mut snap_a).unwrap();
        mid.state_save(&mut snap_b).unwrap();
        // A range snapshot doesn't load into a different range.
        assert!(mid.state_load(&snap_a).is_err());

        let step = |first: &mut Ctx, mid: &mut Ctx, out: &mut [f32]| {
            let mut h1 = vec![0f32; hd];
            let p = prompt.len();
            first.eval_stage(StageInput::Tokens(&[77]), p, StageOutput::Hidden(&mut h1)).unwrap();
            mid.eval_stage(StageInput::Hidden(&h1), p, StageOutput::Logits(out)).unwrap();
        };
        let mut a = vec![0f32; VARIANT.vocab_size];
        step(&mut first, &mut mid, &mut a);
        first.state_load(&snap_a).unwrap();
        mid.state_load(&snap_b).unwrap();
        let mut b = vec![0f32; VARIANT.vocab_size];
        step(&mut first, &mut mid, &mut b);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    /// Mode B: a range under a 10 GiB cap holds all its experts; decode
    /// reads nothing from disk after open and stays under the cap.
    #[test]
    #[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
    fn mode_b_range_holds_all_experts_under_cap() {
        let cap = 10 * GIB;
        let mut ctx = open(
            Some(0..14),
            // A stage takes no checkpoints (like `streammoe-node`).
            OpenConfig { mem_cap: Some(cap), max_checkpoints: 0, ..OpenConfig::default() },
        );
        let before = ctx.expert_io_stats();
        let hot = before.hot.expect("hot tier attached");
        assert_eq!(hot.resident, 14 * VARIANT.num_experts, "{hot:?}");
        let prompt = prompt();
        let mut h = vec![0f32; prompt.len() * VARIANT.hidden_dim];
        ctx.eval_stage(StageInput::Tokens(&prompt), 0, StageOutput::Hidden(&mut h)).unwrap();
        let mut h1 = vec![0f32; VARIANT.hidden_dim];
        for i in 0..DECODE {
            let pos = prompt.len() + i;
            ctx.eval_stage(StageInput::Tokens(&[500 + i as i32]), pos, StageOutput::Hidden(&mut h1))
                .unwrap();
        }
        let after = ctx.expert_io_stats();
        assert_eq!(after.disk_reads, before.disk_reads, "{after:?}");
        let fp = ctx.footprint().expect("footprint");
        assert!(fp.peak <= cap, "peak {} over cap {cap}", fp.peak);
    }
}
