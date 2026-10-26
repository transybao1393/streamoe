//! Per-request sequence state ([`OpenConfig::max_requests`], plan
//! phase 4).
//!
//! The first group runs without model files, against a fake manifest
//! ([`common::fake_model`]): admission, refusal when the pool is full,
//! `drop_request`, unknown-request errors and the pool's budget
//! accounting.
//!
//! The `#[ignore]`d group loads the a3b model: two requests decoded
//! interleaved, token by token, match each request run alone (the
//! plan's phase 4 exit criterion), bit for bit.
//!
//! ```text
//! scripts/check_test_host.sh
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b \
//!     --test requests -- --include-ignored --test-threads=1
//! ```

#![cfg(target_os = "macos")]

#[allow(dead_code)]
mod common;

use common::fake_model::FakeModel;
use streammoe::riir::VARIANT;
use streammoe::riir::budget::{Account, BudgetPlan};
use streammoe::{Error, ExpertIoMode, OpenConfig, ReqId, StageInput, StageOutput};

fn cfg(max_requests: usize) -> OpenConfig {
    OpenConfig {
        n_ctx: 64,
        max_requests,
        ..OpenConfig::default()
    }
}

#[test]
fn zero_requests_rejected() {
    let fake = FakeModel::new("req0");
    let err = fake.open(&cfg(0)).err().expect("must fail");
    assert!(matches!(err, Error::InvalidConfig { .. }), "{err:?}");
}

#[test]
fn admission_fills_the_pool_then_refuses() {
    let fake = FakeModel::new("reqadmit");
    let mut ctx = fake.open(&cfg(3)).expect("open");
    assert_eq!(ctx.max_requests(), 3);
    assert_eq!(ctx.requests_in_use(), 1);
    assert!(ctx.has_request(ReqId::DEFAULT));

    ctx.admit_request(ReqId(10)).expect("first");
    ctx.admit_request(ReqId(11)).expect("second");
    match ctx.admit_request(ReqId(10)) {
        Err(Error::RequestExists { req: 10 }) => {}
        other => panic!("expected RequestExists, got {other:?}"),
    }
    match ctx.admit_request(ReqId(12)) {
        Err(Error::PoolFull { max_requests: 3 }) => {}
        other => panic!("expected PoolFull, got {other:?}"),
    }
    assert_eq!(ctx.requests_in_use(), 3);

    // Dropping frees the slot for a new request.
    assert!(ctx.drop_request(ReqId(10)));
    assert!(!ctx.drop_request(ReqId(10)));
    assert!(!ctx.has_request(ReqId(10)));
    ctx.admit_request(ReqId(12)).expect("reuse freed slot");
    // The default request can be dropped (= cleared) but never leaves.
    assert!(ctx.drop_request(ReqId::DEFAULT));
    assert!(ctx.has_request(ReqId::DEFAULT));
    assert_eq!(ctx.requests_in_use(), 3);
}

#[test]
fn unknown_requests_rejected() {
    let fake = FakeModel::new("requnknown");
    let mut ctx = fake.open(&cfg(2)).expect("open");
    let unknown = |r: Result<(), Error>| match r {
        Err(Error::UnknownRequest { req: 99 }) => {}
        other => panic!("expected UnknownRequest, got {other:?}"),
    };
    unknown(ctx.eval_request(ReqId(99), StageInput::Tokens(&[1]), 0, StageOutput::Discard));
    unknown(ctx.state_size_for(ReqId(99)).map(drop));
    unknown(ctx.state_save_for(ReqId(99), &mut []).map(drop));
    unknown(ctx.state_load_for(ReqId(99), &[]));
    // An admitted request's empty state has the same size as the
    // default request's.
    ctx.admit_request(ReqId(5)).unwrap();
    assert_eq!(ctx.state_size_for(ReqId(5)).unwrap(), ctx.state_size());
}

#[test]
fn drop_all_requests_empties_the_pool() {
    let fake = FakeModel::new("reqdropall");
    let mut ctx = fake.open(&cfg(3)).expect("open");
    ctx.admit_request(ReqId(1)).unwrap();
    ctx.admit_request(ReqId(2)).unwrap();
    ctx.drop_all_requests();
    assert_eq!(ctx.requests_in_use(), 1);
    assert!(ctx.has_request(ReqId::DEFAULT));
    assert!(!ctx.has_request(ReqId(1)) && !ctx.has_request(ReqId(2)));
    ctx.admit_request(ReqId(1)).expect("slots free again");
}

/// Positions are checked before the forward narrows them to i32.
#[test]
fn positions_past_i32_rejected() {
    let fake = FakeModel::new("reqpos");
    let mut ctx = fake.open(&cfg(1)).expect("open");
    for start in [i32::MAX as usize, (1usize << 32) + 5] {
        match ctx.eval_request(ReqId::DEFAULT, StageInput::Tokens(&[1]), start, StageOutput::Discard) {
            Err(Error::PositionOutOfRange { start_pos, n_tokens: 1 }) => assert_eq!(start_pos, start),
            other => panic!("expected PositionOutOfRange at {start}, got {other:?}"),
        }
    }
}

/// Every slot's state is in the plan: a cap that fits one request's
/// KV doesn't fit eight.
#[test]
fn request_pool_counted_against_the_cap() {
    let fake = FakeModel::new("reqbudget");
    let n_ctx = 4096;
    let w = std::fs::metadata(&fake.weights).unwrap().len();
    let r = 0..VARIANT.num_layers;
    let chunk = streammoe::riir::config::default_prefill_chunk(n_ctx, true);
    let one = BudgetPlan::for_range(n_ctx, ExpertIoMode::Pread, w, 0, &r, 1, chunk);
    let eight = BudgetPlan::for_range(n_ctx, ExpertIoMode::Pread, w, 0, &r, 8, chunk);
    assert_eq!(eight.get(Account::Kv), 8 * one.get(Account::Kv));
    let cap = (one.total() + eight.total()) / 2;
    let capped = |max_requests| OpenConfig {
        n_ctx,
        mem_cap: Some(cap),
        runtime_overhead: 0,
        hot_tier_bytes: Some(0),
        max_requests,
        // Request state only; checkpoint buffers are their own account.
        max_checkpoints: 0,
        ..OpenConfig::default()
    };
    match fake.open(&capped(8)).err().expect("eight must fail") {
        Error::OverBudget { detail } => assert!(detail.contains("kv="), "{detail}"),
        other => panic!("eight requests: expected OverBudget, got {other:?}"),
    }
    if let Err(Error::OverBudget { detail }) = fake.open(&capped(1)) {
        panic!("one request over budget: {detail}");
    }
    // Checkpoint buffers are planned too: 64 full-context snapshots
    // (each ≥ one request's KV) can't fit in what one request leaves.
    let with_checkpoints = OpenConfig { max_checkpoints: 64, ..capped(1) };
    match fake.open(&with_checkpoints).err().expect("checkpoints must count") {
        Error::OverBudget { detail } => assert!(detail.contains("checkpoints="), "{detail}"),
        other => panic!("checkpoints: expected OverBudget, got {other:?}"),
    }
}

#[cfg(feature = "model-qwen3-6-35b-a3b")]
mod model {
    use super::*;

    use crate::common::diff_helpers::{argmax, default_a3b_paths};
    use streammoe::Ctx;

    const N_CTX: usize = 1024;
    const DECODE: usize = 10;

    fn open(max_requests: usize) -> Ctx {
        let p = default_a3b_paths();
        let cfg = OpenConfig { n_ctx: N_CTX, max_requests, ..OpenConfig::default() };
        Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &p.root, false, &cfg).expect("open")
    }

    fn prompt(seed: i32) -> Vec<i32> {
        (0..24).map(|i| 1000 + seed * 101 + 13 * i).collect()
    }

    /// Prefill + greedy decode of one request; logits per step.
    struct Run {
        logits: Vec<Vec<f32>>,
        next: i32,
        pos: usize,
    }

    fn start(ctx: &mut Ctx, req: ReqId, seed: i32) -> Run {
        let p = prompt(seed);
        let mut logits = vec![0f32; ctx.n_vocab()];
        ctx.eval_request(req, StageInput::Tokens(&p), 0, StageOutput::Logits(&mut logits))
            .expect("prefill");
        Run { next: argmax(&logits), logits: vec![logits], pos: p.len() }
    }

    fn step(ctx: &mut Ctx, req: ReqId, run: &mut Run) {
        let mut logits = vec![0f32; ctx.n_vocab()];
        ctx.eval_request(req, StageInput::Tokens(&[run.next]), run.pos, StageOutput::Logits(&mut logits))
            .expect("decode");
        run.next = argmax(&logits);
        run.pos += 1;
        run.logits.push(logits);
    }

    fn alone(seed: i32) -> Vec<Vec<f32>> {
        let mut ctx = open(1);
        let mut run = start(&mut ctx, ReqId::DEFAULT, seed);
        for _ in 0..DECODE {
            step(&mut ctx, ReqId::DEFAULT, &mut run);
        }
        run.logits
    }

    fn assert_bitwise(what: &str, want: &[Vec<f32>], got: &[Vec<f32>]) {
        assert_eq!(want.len(), got.len());
        for (s, (a, b)) in want.iter().zip(got).enumerate() {
            let same = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same, "{what}: logits differ at step {s}");
        }
    }

    /// Plan phase 4 exit: interleaved requests match sequential runs.
    #[test]
    #[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
    fn interleaved_requests_match_sequential_runs() {
        let want_a = alone(1);
        let want_b = alone(2);

        let mut ctx = open(3);
        let (a, b) = (ReqId(1), ReqId(2));
        ctx.admit_request(a).unwrap();
        ctx.admit_request(b).unwrap();
        let mut ra = start(&mut ctx, a, 1);
        let mut rb = start(&mut ctx, b, 2);
        for _ in 0..DECODE {
            step(&mut ctx, a, &mut ra);
            step(&mut ctx, b, &mut rb);
        }
        assert_bitwise("request a", &want_a, &ra.logits);
        assert_bitwise("request b", &want_b, &rb.logits);

        // A dropped-and-readmitted slot starts from empty state.
        assert!(ctx.drop_request(a));
        ctx.admit_request(ReqId(3)).unwrap();
        let mut rc = start(&mut ctx, ReqId(3), 1);
        for _ in 0..DECODE {
            step(&mut ctx, ReqId(3), &mut rc);
        }
        assert_bitwise("readmitted", &want_a, &rc.logits);
    }
}
