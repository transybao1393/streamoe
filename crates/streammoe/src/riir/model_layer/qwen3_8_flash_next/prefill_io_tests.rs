//! [`super::engine::Q38Engine`]'s prefill expert reads on the tiny
//! model ([`super::prefill_io`]): chunk size, whole-layer reads, the
//! reads in flight (staging slots, preads per run), the hot tier and
//! prefetch change how experts reach the staging, never the results —
//! bit for bit on the CPU backend, within [`TOL_CPU_METAL`] of a step
//! per token on Metal. Also the prefill buffers against the plan, and
//! the IO counters.
//!
//! Same setup as `engine_tests` (`STREAMMOE_Q38_TINY`). The tiny
//! model's experts are padded (30,720 B at a 32,768 B stride), so the
//! padded span reads are exercised.

use std::path::Path;

use super::engine::{plan_bytes_io, Q38Engine, Q38PrefillIo};
use super::engine_exec::Q38Exec;
use super::engine_tests::{rel_err, tiny_dir, TOL_CPU_METAL};
use super::prefill_io::PrefillIoStats;
use super::qsa::QsaMode;
use crate::riir::backend::cpu::CpuBackend;
use crate::riir::backend::gpu::MetalBackend;
use crate::riir::backend::Backend;
use crate::riir::io::expert_io::ExpertFiles;
use crate::riir::io::expert_io_mode::ExpertIoMode;
use crate::riir::io::hot_tier::HotTier;
use crate::riir::io::prefetch::PrefetchState;
use crate::riir::io::weight_file::WeightFile;
use crate::riir::variants::VARIANT;
use crate::riir::{MetalContext, MtlWeightBuf};

/// Context and prompt: 560 tokens run as one 512-token chunk plus 48,
/// far past the tiny QSA budget (8 tokens).
const N_CTX: usize = 576;
const PROMPT: usize = 560;

fn open_wf(dir: &Path) -> WeightFile {
    let a = dir.join("streammoe/artifacts");
    WeightFile::open(&a.join("model_weights.bin"), &a.join("model_weights.json")).expect("weights")
}

fn root(dir: &Path) -> std::path::PathBuf {
    dir.join("streammoe/root")
}

fn metal(dir: &Path) -> MetalBackend {
    let wf = open_wf(dir);
    let metal = MetalContext::new().expect("MetalContext::new");
    let buf = MtlWeightBuf::wrap(&wf, metal.device());
    MetalBackend::new(metal, buf).expect("MetalBackend::new")
}

fn experts(dir: &Path, hot_slots: usize) -> ExpertFiles {
    let mut e = ExpertFiles::open_with(&root(dir), false, false).expect("experts");
    if hot_slots > 0 {
        let dev = metal::Device::system_default().expect("Metal device");
        let hot = HotTier::new(&dev, hot_slots, e.expert_size(), VARIANT.num_layers, VARIANT.num_experts).unwrap();
        e.attach_hot_tier(hot);
    }
    e
}

fn prefetch(on: bool) -> Option<PrefetchState> {
    on.then(|| {
        let mut p = PrefetchState::new(VARIANT.num_layers, ExpertIoMode::Pread);
        p.set_max_experts(16);
        p.set_lookahead(true);
        p
    })
}

fn prompt() -> Vec<i32> {
    let v = VARIANT.vocab_size;
    (0..PROMPT).map(|i| ((i * 37 + 5) % v) as i32).collect()
}

/// Build-time settings: buffers for a 512-token chunk and 64 slots.
fn io(dir: &Path, es: usize) -> Q38PrefillIo {
    let mut io = Q38PrefillIo::for_root(&root(dir), N_CTX, es);
    io.chunk = 512;
    io.slots = 64;
    io
}

/// Last logits, snapshot, three decode tokens' logits.
type Run = (Vec<f32>, Vec<u8>, Vec<Vec<f32>>);

fn run<B: Q38Exec>(
    e: &mut Q38Engine<B>,
    b: &mut B,
    wf: &WeightFile,
    ex: &ExpertFiles,
    pool: &rayon::ThreadPool,
    per_token: bool,
) -> Run {
    let v = VARIANT.vocab_size;
    let p = prompt();
    e.reset(b).unwrap();
    let mut last = vec![0.0f32; v];
    if per_token {
        for (i, &t) in p.iter().enumerate() {
            let out = if i + 1 == p.len() { Some(&mut last[..]) } else { None };
            e.step(b, wf, ex, pool, t, out).unwrap();
        }
    } else {
        e.eval(b, wf, ex, pool, &p, Some(&mut last)).unwrap();
    }
    let mut snap = vec![0u8; e.state_size()];
    e.state_save(b, &mut snap).unwrap();
    let decode = [5, 17, 300]
        .iter()
        .map(|&t| {
            let mut l = vec![0.0f32; v];
            e.step(b, wf, ex, pool, t, Some(&mut l)).unwrap();
            l
        })
        .collect();
    (last, snap, decode)
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Worst of last logits / decode logits rel err, and whether the
/// snapshots are equal byte for byte.
fn compare(a: &Run, b: &Run) -> (f32, bool) {
    let e = a.2.iter().zip(&b.2).map(|(x, y)| rel_err(x, y)).fold(rel_err(&a.0, &b.0), f32::max);
    (e, a.1 == b.1)
}

fn same_bits(a: &Run, b: &Run) -> bool {
    bits_eq(&a.0, &b.0) && a.1 == b.1 && a.2.iter().zip(&b.2).all(|(x, y)| bits_eq(x, y))
}

/// (label, chunk, staging slots in use, layer-read frac, max read).
type Arm = (&'static str, usize, usize, f64, usize);

/// The read arms: whole-layer off (frac > 1) / forced (frac 0); one
/// expert per pread / runs; one slot (no overlap, a batch per expert),
/// two (one expert in flight per half), six, all 64.
const ARMS: [Arm; 10] = [
    ("chunk 512, layer read, 64 slots", 512, 64, 0.0, 16),
    ("chunk 512, expert reads, 1 slot, 1/pread", 512, 1, 2.0, 1),
    ("chunk 512, expert reads, 2 slots", 512, 2, 2.0, 16),
    ("chunk 512, layer read, 2 slots", 512, 2, 0.0, 16),
    ("chunk 512, layer read, 6 slots, 2/pread", 512, 6, 0.0, 2),
    ("chunk 512, expert reads, 6 slots", 512, 6, 2.0, 16),
    ("chunk 64, layer read, 64 slots", 64, 64, 0.0, 16),
    ("chunk 5, expert reads, 4 slots", 5, 4, 2.0, 16),
    ("chunk 5, layer read (gaps read through), 8 slots, 3/pread", 5, 8, 0.0, 3),
    ("chunk 1 (decode path)", 1, 64, 0.0, 16),
];

/// The arms the CPU test also runs with the hot tier and prefetch on
/// (the CPU backend is slow at 560 tokens).
const CPU_HOT_ARMS: [usize; 5] = [0, 1, 3, 4, 8];

fn set_arm<B: Q38Exec>(e: &mut Q38Engine<B>, arm: &Arm) {
    let (_, chunk, slots, frac, max_read) = *arm;
    e.set_prefill_chunk(chunk);
    e.set_prefill_slots(slots);
    e.set_layer_read_frac(frac);
    e.set_max_read(max_read);
}

/// CPU backend: every arm, hot tier and prefetch off and on, equals a
/// step per token bit for bit (logits, state, the decode after it).
#[test]
fn prefill_reads_bit_identical_cpu() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 prefill io] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let wf = open_wf(&dir);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let mut want: Option<Run> = None;
    for (hot, pf) in [(0usize, false), (12, true)] {
        let ex = experts(&dir, hot);
        let es = ex.expert_size();
        let mut b = CpuBackend::new(open_wf(&dir));
        let mut e = Q38Engine::with_options(&mut b, &wf, &root(&dir), N_CTX, es, prefetch(pf), io(&dir, es)).unwrap();
        e.set_qsa_mode(QsaMode::Sparse);
        let reference = run(&mut e, &mut b, &wf, &ex, &pool, true);
        match &want {
            None => want = Some(reference),
            Some(w) => assert!(same_bits(&reference, w), "hot {hot} prefetch {pf}: per-token run differs"),
        }
        let w = want.as_ref().unwrap();
        for (ai, arm) in ARMS.iter().enumerate() {
            if hot > 0 && !CPU_HOT_ARMS.contains(&ai) {
                continue;
            }
            set_arm(&mut e, arm);
            e.reset_prefill_io_stats();
            let got = run(&mut e, &mut b, &wf, &ex, &pool, false);
            let st = e.prefill_io_stats();
            eprintln!("[q38 prefill io] cpu hot {hot} pf {pf} {}: {}", arm.0, st.line());
            assert!(same_bits(&got, w), "cpu hot {hot} prefetch {pf} {}: differs from per-token", arm.0);
            check_counters(arm, hot > 0, &st);
        }
    }
}

/// What the counters must show for `arm` (hot tier on: `hot`).
fn check_counters(arm: &Arm, hot: bool, st: &PrefillIoStats) {
    let (label, chunk, slots, frac, max_read) = *arm;
    if chunk == 1 {
        assert_eq!(st.chunks, 0, "{label}: chunk 1 is the decode path");
        return;
    }
    assert!(st.chunks > 0 && st.layers == st.chunks * VARIANT.num_layers as u64, "{label}");
    if frac == 0.0 {
        assert_eq!(st.layer_reads, st.layers, "{label}: every layer read whole");
        if chunk == 5 {
            // 5 tokens × top 3 of 8 experts leave gaps.
            assert!(st.read_through > 0, "{label}: no gap read through");
        }
    } else {
        assert_eq!((st.layer_reads, st.read_through), (0, 0), "{label}: no whole-layer read");
    }
    // Every routed expert came from the hot tier or a read.
    assert_eq!(st.experts, st.hot + st.experts_read - st.read_through, "{label}");
    if max_read == 1 {
        assert_eq!(st.preads, st.experts_read, "{label}: one expert per pread");
    }
    let cap = if slots >= 2 { slots / 2 } else { 1 };
    assert!(st.batches * cap as u64 >= st.hot + st.experts_read, "{label}: batch over its slots");
    if !hot {
        assert_eq!(st.hot, 0, "{label}");
    }
}

/// Metal: every arm within [`TOL_CPU_METAL`] of a step per token, hot
/// tier and prefetch off and on; the arms against each other too.
#[test]
fn prefill_reads_match_per_token_metal() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 prefill io] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let wf = open_wf(&dir);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let mut worst = (0.0f32, 0.0f32);
    for (hot, pf) in [(0usize, false), (12, true)] {
        let ex = experts(&dir, hot);
        let es = ex.expert_size();
        let mut b = metal(&dir);
        let mut e = Q38Engine::with_options(&mut b, &wf, &root(&dir), N_CTX, es, prefetch(pf), io(&dir, es)).unwrap();
        e.set_qsa_mode(QsaMode::Sparse);
        let reference = run(&mut e, &mut b, &wf, &ex, &pool, true);
        let mut first: Option<Run> = None;
        for arm in &ARMS {
            set_arm(&mut e, arm);
            e.reset_prefill_io_stats();
            let s0 = e.submits();
            let got = run(&mut e, &mut b, &wf, &ex, &pool, false);
            let st = e.prefill_io_stats();
            let (err, _) = compare(&got, &reference);
            let (err_arm, _) = first.as_ref().map_or((0.0, true), |f| compare(&got, f));
            eprintln!(
                "[q38 prefill io] metal hot {hot} pf {pf} {}: vs per-token {err:.3e}, vs first arm {err_arm:.3e}, \
                 {} submits; {}",
                arm.0,
                e.submits() - s0,
                st.line()
            );
            assert!(err <= TOL_CPU_METAL, "metal hot {hot} prefetch {pf} {}: rel err {err}", arm.0);
            assert!(err_arm <= TOL_CPU_METAL, "metal {}: vs the first arm {err_arm}", arm.0);
            check_counters(arm, hot > 0, &st);
            worst = (worst.0.max(err), worst.1.max(err_arm));
            if first.is_none() {
                first = Some(got);
            }
        }
    }
    eprintln!("[q38 prefill io] metal worst: vs per-token {:.3e}, between arms {:.3e}", worst.0, worst.1);
}

/// The prefill buffers the engine allocates are what the plan counts:
/// the staging exactly (slots × the padded stride), every other
/// chunk-sized buffer within the scratch account.
#[test]
fn prefill_buffers_match_plan() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 prefill io] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let wf = open_wf(&dir);
    let ex = experts(&dir, 0);
    let es = ex.expert_size();
    for (chunk, slots) in [(512usize, 64usize), (64, 8), (5, 1)] {
        let mut io = io(&dir, es);
        io.chunk = chunk;
        io.slots = slots;
        assert_eq!(io.stride, ex.expert_stride(), "staging at the file stride");
        let mut b = metal(&dir);
        let e = Q38Engine::with_options(&mut b, &wf, &root(&dir), N_CTX, es, None, io).unwrap();
        assert_eq!(e.prefill_io(), io);
        let plan = plan_bytes_io(N_CTX, es, 0, 0, &io);
        let by = b.pool().bytes_by_label();
        let sum = |f: &dyn Fn(&str) -> bool| -> u64 { by.iter().filter(|x| f(x.0)).map(|x| x.1).sum() };
        let staging = sum(&|l| l == "q38.prefill_staging");
        assert_eq!(plan.prefill_staging, (slots * ex.expert_stride()) as u64);
        assert!(staging >= plan.prefill_staging && staging <= plan.prefill_staging + 16 * 1024, "staging {staging}");
        let state = [
            "q38.prefill_staging",
            "q38.expert_staging",
            "q38.k_cache",
            "q38.v_cache",
            "q38.conv_state",
            "q38.delta_state",
            "q38.ple_conv_state",
        ];
        let scratch = sum(&|l| l.starts_with("q38.") && !state.contains(&l));
        eprintln!(
            "[q38 prefill io] chunk {chunk}, {slots} slots: staging {staging} B (plan {}), pool scratch {scratch} B \
             (plan scratch {} B, incl. QSA row lists and host copies)",
            plan.prefill_staging, plan.scratch
        );
        assert!(scratch <= plan.scratch, "chunk {chunk}: pool scratch {scratch} B over the plan's {}", plan.scratch);
    }
    // Shrinking for a cap: staging to 16, then the chunk to 64, then the
    // staging to 2.
    let mut io = io(&dir, es);
    let mut seen = vec![(io.chunk, io.slots)];
    while io.shrink() {
        seen.push((io.chunk, io.slots));
    }
    assert_eq!(seen, vec![(512, 64), (512, 32), (512, 16), (256, 16), (128, 16), (64, 16), (64, 8), (64, 4), (64, 2)]);
}
