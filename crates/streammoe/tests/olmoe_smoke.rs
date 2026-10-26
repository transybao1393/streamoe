//! End-to-end smoke for the OLMoE-1B-7B variant: open under a hard
//! memory cap, batched prefill of a chat prompt, greedy per-token
//! decode, compared against an MLX reference greedy run of the same
//! 4-bit weights.
//!
//! `#[ignore]` because it needs the converted model on disk. Every
//! test opens under a hard `mem_cap`. Paths
//! default to `$HOME/models/olmoe-1b-7b/{artifacts,root}`; override
//! the parent with `STREAMMOE_OLMOE_DIR`. Run with:
//!
//! ```text
//! cargo test -p streammoe --features model-olmoe-1b-7b --release \
//!     --test olmoe_smoke -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(all(target_os = "macos", feature = "model-olmoe-1b-7b"))]

use std::path::PathBuf;

use streammoe::riir::budget::footprint;
use streammoe::{Ctx, Error, KvFormat, OpenConfig};

const VOCAB: usize = 50_304;
const EOS: i32 = 50_279;

/// "What is the capital of France? Answer in one sentence." in the
/// OLMoE chat template (`<|endoftext|><|user|>\n…\n<|assistant|>\n`).
const PROMPT: &[i32] = &[
    50279, 29, 93, 4537, 49651, 187, 1276, 310, 253, 5347, 273, 6181, 32, 37741, 275, 581, 6197,
    15, 187, 29, 93, 515, 5567, 49651, 187,
];
/// MLX greedy continuation: "The capital of France is Paris." + EOS.
const REF_TOKENS: &[i32] = &[510, 5347, 273, 6181, 310, 7785, 15, 50279];

fn model_dir() -> PathBuf {
    std::env::var_os("STREAMMOE_OLMOE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME")).join("models/olmoe-1b-7b")
        })
}

/// 0.35 GiB of hot tier, in whole experts.
const HOT_TIER_3G: u64 = 106 * 3_538_944;

fn open() -> Result<Ctx, Error> {
    open_capped(3 << 30, Some(HOT_TIER_3G))
}

fn open_capped(cap: u64, hot_tier_bytes: Option<u64>) -> Result<Ctx, Error> {
    open_with(cap, hot_tier_bytes, KvFormat::F32)
}

fn open_with(cap: u64, hot_tier_bytes: Option<u64>, kv_format: KvFormat) -> Result<Ctx, Error> {
    open_cfg(OpenConfig {
        n_ctx: 2048,
        mem_cap: Some(cap),
        hot_tier_bytes,
        kv_format,
        // These tests take no checkpoints; at the default 4 the plan
        // would hold 4 × 0.5 GiB of snapshot buffers.
        max_checkpoints: 0,
        ..OpenConfig::default()
    })
}

fn open_cfg(cfg: OpenConfig) -> Result<Ctx, Error> {
    let dir = model_dir();
    let art = dir.join("artifacts");
    Ctx::open_with(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
        &art.join("vocab.bin"),
        &dir.join("root"),
        /* use_2bit = */ false,
        &cfg,
    )
}

fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    d / (na.sqrt() * nb.sqrt())
}

#[test]
#[ignore = "needs the OLMoE-1B-7B model on disk"]
fn olmoe_greedy_decode_matches_mlx_reference() -> Result<(), Error> {
    let mut ctx = open()?;
    let mut logits = vec![0.0f32; VOCAB];

    let t0 = std::time::Instant::now();
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)?;
    eprintln!(
        "[olmoe-smoke] prefill {} tokens in {:.2}s",
        PROMPT.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut generated = Vec::new();
    let mut pos = PROMPT.len();
    let t1 = std::time::Instant::now();
    for _ in 0..REF_TOKENS.len() {
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        let tok = argmax(&logits);
        eprintln!("[olmoe-smoke] step {}: token {tok} logit {:.3}", generated.len(), logits[tok as usize]);
        generated.push(tok);
        if tok == EOS {
            break;
        }
        ctx.eval_token(tok, pos, 0, &mut logits)?;
        pos += 1;
    }
    eprintln!(
        "[olmoe-smoke] decoded {:?} in {:.2}s",
        generated,
        t1.elapsed().as_secs_f64()
    );
    assert_eq!(generated, REF_TOKENS, "greedy tokens diverge from the MLX reference");
    Ok(())
}

/// Batched prefill (graph path) and token-by-token decode (per-token
/// path) of the same prompt must agree on the last position's logits.
#[test]
#[ignore = "needs the OLMoE-1B-7B model on disk"]
fn olmoe_batched_prefill_matches_per_token() -> Result<(), Error> {
    let mut batched = vec![0.0f32; VOCAB];
    {
        let mut ctx = open()?;
        ctx.eval_prompt(PROMPT, 0, 0, &mut batched)?;
    }
    let mut per_token = vec![0.0f32; VOCAB];
    {
        let mut ctx = open()?;
        for (pos, &tok) in PROMPT.iter().enumerate() {
            ctx.eval_token(tok, pos, 0, &mut per_token)?;
        }
    }
    let cos = cosine(&batched, &per_token);
    eprintln!(
        "[olmoe-smoke] batched vs per-token: cos={cos:.6} argmax {} / {}",
        argmax(&batched),
        argmax(&per_token)
    );
    assert_eq!(argmax(&batched), argmax(&per_token));
    assert!(cos > 0.999, "cosine {cos}");
    Ok(())
}

/// Child-process body of the cap tests: open, prefill the prompt,
/// decode 32 tokens; the process's lifetime peak must stay under the
/// cap, and the footprint right after open must be ≥ 95 % of it —
/// nothing big may be allocated after the open-time check.
fn peak_stays_at_open(cap: u64, hot_tier_bytes: Option<u64>) {
    let mut ctx = open_capped(cap, hot_tier_bytes).expect("open");
    let at_open = footprint().expect("footprint").current;
    let mut logits = vec![0.0f32; VOCAB];
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits).expect("prefill");
    let mut pos = PROMPT.len();
    for _ in 0..32 {
        ctx.eval_token(argmax(&logits), pos, 0, &mut logits).expect("decode");
        pos += 1;
    }
    let peak = footprint().expect("footprint").peak;
    eprintln!(
        "[olmoe-cap] cap {cap} B: open footprint {at_open} B, lifetime peak {peak} B ({:.1} %)",
        100.0 * at_open as f64 / peak as f64
    );
    assert!(peak <= cap, "lifetime peak {peak} B over cap {cap} B");
    assert!(at_open * 100 >= peak * 95, "open footprint {at_open} B < 95 % of peak {peak} B");
}

/// The lifetime peak is per process, so each cap case runs in a fresh
/// child (this test binary, re-run on one test with
/// `OLMOE_CAP_CHILD` set).
fn run_in_child(test: &str, body: impl FnOnce()) {
    if std::env::var_os("OLMOE_CAP_CHILD").is_some() {
        body();
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().expect("current_exe"))
        .args([test, "--exact", "--ignored", "--nocapture", "--test-threads=1"])
        .env("OLMOE_CAP_CHILD", "1")
        .status()
        .expect("spawn child");
    assert!(status.success(), "{test} child failed: {status}");
}

#[test]
#[ignore = "needs the OLMoE-1B-7B model on disk; holds ~4.8 GiB"]
fn olmoe_cap_6gib_auto_hot_tier_peak_is_planned_at_open() {
    run_in_child("olmoe_cap_6gib_auto_hot_tier_peak_is_planned_at_open", || {
        peak_stays_at_open(6 << 30, None)
    });
}

#[test]
#[ignore = "needs the OLMoE-1B-7B model on disk"]
fn olmoe_cap_3gib_peak_is_planned_at_open() {
    run_in_child("olmoe_cap_3gib_peak_is_planned_at_open", || {
        peak_stays_at_open(3 << 30, Some(HOT_TIER_3G))
    });
}

/// Prefill + greedy decode of `steps` tokens; returns the tokens and
/// every step's logits.
fn greedy(ctx: &mut Ctx, steps: usize) -> Result<(Vec<i32>, Vec<Vec<f32>>), Error> {
    let mut logits = vec![0.0f32; VOCAB];
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)?;
    let (mut toks, mut all) = (Vec::new(), Vec::new());
    let mut pos = PROMPT.len();
    for _ in 0..steps {
        let tok = argmax(&logits);
        toks.push(tok);
        all.push(logits.clone());
        ctx.eval_token(tok, pos, 0, &mut logits)?;
        pos += 1;
    }
    Ok((toks, all))
}

/// Each quantized KV format decodes the MLX reference answer and
/// keeps its logits close to the f32 cache's.
#[test]
#[ignore = "needs the OLMoE-1B-7B model on disk"]
fn olmoe_kv_formats_track_f32() -> Result<(), Error> {
    let (ref_toks, ref_logits) = greedy(&mut open()?, REF_TOKENS.len())?;
    assert_eq!(ref_toks, REF_TOKENS);
    for fmt in [KvFormat::F16, KvFormat::Turbo4, KvFormat::Turbo3] {
        let (toks, logits) = greedy(&mut open_with(3 << 30, Some(HOT_TIER_3G), fmt)?, REF_TOKENS.len())?;
        let cos: Vec<f64> = logits.iter().zip(&ref_logits).map(|(a, b)| cosine(a, b)).collect();
        let min = cos.iter().copied().fold(1.0, f64::min);
        eprintln!(
            "[olmoe-kv] {fmt:?}: tokens {toks:?}, logits cos vs f32: prefill {:.6}, min {min:.6}",
            cos[0]
        );
        assert_eq!(toks, REF_TOKENS, "{fmt:?} diverges from the reference answer");
        // Measured (2026-10-26): F16 0.999999, Turbo4 0.9928.
        let bound = match fmt {
            KvFormat::F16 => 0.9999,
            KvFormat::Turbo4 => 0.98,
            _ => 0.95,
        };
        assert!(min > bound, "{fmt:?}: logits cos {min} ≤ {bound}");
    }
    Ok(())
}

/// Batched prefill (GPU quantize) and per-token decode (CPU
/// quantize) agree for a quantized cache.
#[test]
#[ignore = "needs the OLMoE-1B-7B model on disk"]
fn olmoe_kv_quant_batched_matches_per_token() -> Result<(), Error> {
    for fmt in [KvFormat::F16, KvFormat::Turbo4, KvFormat::Turbo3] {
        let mut batched = vec![0.0f32; VOCAB];
        open_with(3 << 30, Some(HOT_TIER_3G), fmt)?.eval_prompt(PROMPT, 0, 0, &mut batched)?;
        let mut per_token = vec![0.0f32; VOCAB];
        let mut ctx = open_with(3 << 30, Some(HOT_TIER_3G), fmt)?;
        for (pos, &tok) in PROMPT.iter().enumerate() {
            ctx.eval_token(tok, pos, 0, &mut per_token)?;
        }
        let cos = cosine(&batched, &per_token);
        eprintln!("[olmoe-kv] {fmt:?} batched vs per-token: cos={cos:.6}");
        assert_eq!(argmax(&batched), argmax(&per_token), "{fmt:?}");
        // The two paths quantize K/V that differ in the last bits, so
        // a few codes land on the other side of a level boundary.
        // Measured (2026-10-26): F16 1.000000, Turbo4 0.9975.
        let bound = if fmt == KvFormat::F16 { 0.999 } else { 0.99 };
        assert!(cos > bound, "{fmt:?}: cosine {cos}");
    }
    Ok(())
}

/// A quantized cache snapshots its stored rows (v4) and restores to
/// bit-identical decode; an f32 context refuses the snapshot.
#[test]
#[ignore = "needs the OLMoE-1B-7B model on disk"]
fn olmoe_kv_quant_snapshot_round_trips() -> Result<(), Error> {
    let fmt = KvFormat::Turbo4;
    let mut a = open_with(3 << 30, Some(HOT_TIER_3G), fmt)?;
    let mut logits = vec![0.0f32; VOCAB];
    a.eval_prompt(PROMPT, 0, 0, &mut logits)?;
    let mut snap = vec![0u8; a.state_size()];
    let n = a.state_save(&mut snap).expect("save");
    assert_eq!(n, snap.len());
    assert_eq!(u32::from_le_bytes(snap[4..8].try_into().unwrap()), 4, "v4 header");
    let tok = argmax(&logits);
    let mut after_a = vec![0.0f32; VOCAB];
    a.eval_token(tok, PROMPT.len(), 0, &mut after_a)?;
    drop(a);

    let mut b = open_with(3 << 30, Some(HOT_TIER_3G), fmt)?;
    b.memory_clear();
    b.state_load(&snap).expect("load");
    let mut after_b = vec![0.0f32; VOCAB];
    b.eval_token(tok, PROMPT.len(), 0, &mut after_b)?;
    assert_eq!(after_a, after_b, "restored decode differs");
    drop(b);

    let mut f = open()?;
    f.memory_clear();
    let err = f.state_load(&snap).expect_err("f32 context must refuse a Turbo4 snapshot");
    eprintln!("[olmoe-kv] f32 load of a Turbo4 snapshot: {err}");
    Ok(())
}

/// Splitting decode attention across threads changes nothing: every
/// step's logits are bit-identical to the serial path.
#[test]
#[ignore = "needs the OLMoE-1B-7B model on disk"]
fn olmoe_attn_threads_are_bit_identical() -> Result<(), Error> {
    for kv_format in [KvFormat::F32, KvFormat::Turbo4] {
        let run = |attn_threads| {
            let cfg = OpenConfig {
                n_ctx: 2048,
                mem_cap: Some(3 << 30),
                hot_tier_bytes: Some(HOT_TIER_3G),
                kv_format,
                attn_threads,
                max_checkpoints: 0,
                ..OpenConfig::default()
            };
            greedy(&mut open_cfg(cfg)?, REF_TOKENS.len())
        };
        let (t1, l1) = run(1)?;
        let (t4, l4) = run(4)?;
        assert_eq!(t1, t4, "{kv_format:?}");
        assert!(l1 == l4, "{kv_format:?}: logits differ between 1 and 4 attention threads");
        eprintln!("[olmoe-attn] {kv_format:?}: 1 vs 4 threads bit-identical over {} steps", l1.len());
    }
    Ok(())
}

/// Checkpoints under a cap use the buffers planned and touched at
/// open: save / evict / restore never grow the footprint, and a
/// restore resumes bit-identically.
fn checkpoints_stay_inside_the_cap(cap: u64) {
    let cfg = OpenConfig {
        n_ctx: 1024,
        mem_cap: Some(cap),
        hot_tier_bytes: Some(HOT_TIER_3G),
        max_checkpoints: 2,
        ..OpenConfig::default()
    };
    let mut ctx = open_cfg(cfg).expect("open");
    let at_open = footprint().expect("footprint").current;
    let mut logits = vec![0.0f32; VOCAB];
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits).expect("prefill");
    let p0 = PROMPT.len();
    ctx.checkpoint_pos(p0 as i32).expect("checkpoint at prompt end");
    let tok = argmax(&logits);
    let mut first = vec![0.0f32; VOCAB];
    ctx.eval_token(tok, p0, 0, &mut first).expect("decode");
    // A second checkpoint further on (and an overwrite of it): both
    // planned slots in use.
    let mut pos = p0 + 1;
    let mut l = first.clone();
    for _ in 0..8 {
        ctx.eval_token(argmax(&l), pos, 0, &mut l).expect("decode");
        pos += 1;
    }
    ctx.checkpoint_pos(pos as i32).expect("checkpoint 2");
    ctx.checkpoint_pos(pos as i32).expect("overwrite checkpoint 2");
    assert_eq!(ctx.checkpoint_count(), 2);
    // Raising past the planned count is clamped under a cap.
    ctx.set_max_checkpoints(16);
    ctx.restore_to(p0 as i32).expect("restore");
    assert_eq!(ctx.checkpoint_count(), 1, "later checkpoint dropped");
    let mut again = vec![0.0f32; VOCAB];
    ctx.eval_token(tok, p0, 0, &mut again).expect("decode after restore");
    assert!(first == again, "restored decode differs");
    // Fill past the planned slots: the oldest non-zero one is evicted
    // into its buffer; nothing is allocated.
    for k in 0..4 {
        ctx.checkpoint_pos((p0 + 1 + k) as i32).expect("checkpoint");
        assert!(ctx.checkpoint_count() <= 2);
    }
    ctx.memory_clear();
    assert_eq!(ctx.checkpoint_count(), 0);
    let peak = footprint().expect("footprint").peak;
    eprintln!(
        "[olmoe-ckpt] cap {cap} B: open footprint {at_open} B, lifetime peak {peak} B ({:.1} %)",
        100.0 * at_open as f64 / peak as f64
    );
    assert!(peak <= cap, "lifetime peak {peak} B over cap {cap} B");
    assert!(at_open * 100 >= peak * 99, "open footprint {at_open} B < 99 % of peak {peak} B");
}

#[test]
#[ignore = "needs the OLMoE-1B-7B model on disk"]
fn olmoe_checkpoints_stay_inside_the_cap() {
    run_in_child("olmoe_checkpoints_stay_inside_the_cap", || {
        checkpoints_stay_inside_the_cap(3 << 30)
    });
}
