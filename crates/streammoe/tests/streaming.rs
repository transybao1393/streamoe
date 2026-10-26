//! Streaming improvements (plan phase 6): every knob changes where
//! expert bytes come from or when they are read, never the math, so
//! each must give **bit-identical** logits to the default path. The
//! stats test prints the telemetry the phase is judged on (read
//! bytes per token, guess accuracy, I/O wait vs. decode time) as CSV.
//!
//! Needs the a3b model; check the host first:
//! ```text
//! scripts/check_test_host.sh
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b \
//!     --test streaming -- --include-ignored --test-threads=1 --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "model-qwen3-6-35b-a3b"))]

#[allow(dead_code)]
mod common;

use common::diff_helpers::default_a3b_paths;
use streammoe::{Ctx, ExpertIoMode, ExpertIoStats, OpenConfig, PrefetchGuess};

const GIB: u64 = 1 << 30;
const N_CTX: usize = 2048;
const DECODE: usize = 24;

fn open(cfg: &OpenConfig) -> Ctx {
    let p = default_a3b_paths();
    Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &p.root, false, cfg).expect("open")
}

/// Prefill a fixed prompt, then greedy-decode. Returns every step's
/// logits and the chosen tokens.
fn run(ctx: &mut Ctx) -> (Vec<Vec<f32>>, Vec<i32>) {
    let prompt: Vec<i32> = (0..64).map(|i| 1000 + 7 * i).collect();
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

/// Pread with a small hot tier, so decode both hits and misses.
fn base_cfg() -> OpenConfig {
    OpenConfig {
        n_ctx: N_CTX,
        expert_io: Some(ExpertIoMode::Pread),
        hot_tier_bytes: Some(GIB),
        ..OpenConfig::default()
    }
}

fn print_stats(label: &str, st: &ExpertIoStats) {
    eprintln!("[streaming] {label}");
    eprintln!("label,{}", ExpertIoStats::CSV_HEADER);
    eprintln!("{label},{}", st.csv_row());
    eprintln!("{}", ExpertIoStats::LAYER_CSV_HEADER);
    eprint!("{}", st.layer_csv());
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn streaming_stats_report() {
    let mut ctx = open(&base_cfg());
    ctx.reset_expert_io_stats();
    let _ = run(&mut ctx);
    let st = ctx.expert_io_stats();
    print_stats("default", &st);
    assert_eq!(st.decode_tokens, DECODE as u64);
    assert!(st.prefill_tokens > 0);
    assert!(st.decode_bytes_per_token().is_some());
    assert!(st.io_wait_ns <= st.decode_ns, "{st:?}");
    // Every pread decode layer after the first token is scored.
    assert!(st.guess_prev.layers > 0, "{st:?}");
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn bounded_io_matches_and_respects_rate() {
    let base = run(&mut open(&base_cfg()));
    let rate = 400u64 << 20; // 400 MiB/s
    let mut ctx = open(&OpenConfig {
        io_queue_depth: Some(2),
        io_bytes_per_sec: Some(rate),
        ..base_cfg()
    });
    ctx.reset_expert_io_stats();
    let t0 = std::time::Instant::now();
    let got = run(&mut ctx);
    let secs = t0.elapsed().as_secs_f64();
    let st = ctx.expert_io_stats();
    print_stats("bounded", &st);
    assert_same("bounded io", &base, &got);
    // Bytes over wall time can exceed the rate by at most the burst.
    let burst = rate as f64 * streammoe::riir::io::io_limit::RATE_BURST.as_secs_f64();
    assert!(
        st.disk_bytes as f64 <= rate as f64 * secs + burst,
        "{} bytes in {secs:.2}s over {rate} B/s",
        st.disk_bytes
    );
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn prefill_reads_each_expert_once_per_layer() {
    let v = streammoe::riir::VARIANT;
    // No hot tier: every staged expert comes from disk.
    let mut ctx = open(&OpenConfig { hot_tier_bytes: Some(0), ..base_cfg() });
    let prompt: Vec<i32> = (0..64).map(|i| 1000 + 7 * i).collect();
    let mut logits = vec![0f32; ctx.n_vocab()];
    ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
    let st = ctx.expert_io_stats();
    print_stats("prefill", &st);
    // One chunk: each distinct routed expert of a layer is read once.
    let counts = ctx.routing_counts();
    for (l, row) in st.per_layer.iter().enumerate() {
        let distinct = counts[l * v.num_experts..(l + 1) * v.num_experts]
            .iter()
            .filter(|&&c| c > 0)
            .count() as u64;
        assert_eq!(row.prefill_reads, distinct, "layer {l}: {row:?}");
    }
    assert_eq!(st.prefill.bytes, st.prefill.reads * v.expert_size_4bit() as u64);
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn cross_layer_guess_matches_and_reports_accuracy() {
    let base = run(&mut open(&base_cfg()));
    let mut ctx = open(&OpenConfig {
        prefetch_guess: PrefetchGuess::CrossLayer,
        ..base_cfg()
    });
    ctx.reset_expert_io_stats();
    let got = run(&mut ctx);
    let st = ctx.expert_io_stats();
    print_stats("cross_layer", &st);
    assert_same("cross-layer guess", &base, &got);
    eprintln!(
        "[streaming] guess accuracy: prev-token {:?}, cross-layer {:?}",
        st.guess_prev.accuracy(),
        st.guess_cross.accuracy()
    );
    // Every layer after the first of each decode token is guessed.
    let v = streammoe::riir::VARIANT;
    assert_eq!(st.guess_cross.layers, DECODE as u64 * (v.num_layers as u64 - 1), "{st:?}");
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn lookahead_prefetch_matches() {
    let base = run(&mut open(&base_cfg()));
    let mut ctx = open(&OpenConfig { prefetch_lookahead: true, ..base_cfg() });
    ctx.reset_expert_io_stats();
    let got = run(&mut ctx);
    let st = ctx.expert_io_stats();
    print_stats("lookahead", &st);
    assert_same("lookahead prefetch", &base, &got);
    assert!(st.prefetch.reads > 0, "{st:?}");
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn lookahead_with_bounded_io_and_no_hot_tier_matches() {
    // The hardest timing case: every pick streams, reads are throttled
    // to one at a time, prefetches overlap the previous layer.
    let base = run(&mut open(&OpenConfig { hot_tier_bytes: Some(0), ..base_cfg() }));
    let got = run(&mut open(&OpenConfig {
        hot_tier_bytes: Some(0),
        prefetch_lookahead: true,
        io_queue_depth: Some(1),
        ..base_cfg()
    }));
    assert_same("lookahead + depth 1", &base, &got);
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn fast_experts_dir_matches_and_is_used() {
    let base = run(&mut open(&base_cfg()));
    // A "fast tier" holding layers 0 and 1 (symlinks: same bytes, no
    // copy; a real deployment copies them to the internal SSD).
    let p = default_a3b_paths();
    let fast = std::env::temp_dir().join(format!("streammoe-fast-{}", std::process::id()));
    let sub = fast.join("packed_experts");
    std::fs::create_dir_all(&sub).unwrap();
    for l in 0..2 {
        let name = format!("layer_{l:02}.bin");
        let _ = std::fs::remove_file(sub.join(&name));
        std::os::unix::fs::symlink(p.root.join("packed_experts").join(&name), sub.join(&name))
            .unwrap();
    }
    let mut ctx = open(&OpenConfig { fast_experts_dir: Some(fast.clone()), ..base_cfg() });
    ctx.reset_expert_io_stats();
    let got = run(&mut ctx);
    let st = ctx.expert_io_stats();
    print_stats("fast_dir", &st);
    assert_same("fast experts dir", &base, &got);
    assert!(st.fast_dir_bytes > 0, "{st:?}");
    assert_eq!(
        st.fast_dir_bytes,
        st.per_layer[0].disk_bytes + st.per_layer[1].disk_bytes,
        "{st:?}"
    );
    let _ = std::fs::remove_dir_all(&fast);
}
