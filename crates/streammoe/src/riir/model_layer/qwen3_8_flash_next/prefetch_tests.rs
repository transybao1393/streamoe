//! [`super::engine::Q38Engine`]'s expert reads on the tiny model:
//! speculative prefetch and the hot tier change where an expert comes
//! from, never the logits — bit for bit, on Metal and on the CPU
//! backend. Also the staging buffer against the memory plan, and the
//! PLE on the backend ([`super::ple_gpu`]) against the host PLE.
//!
//! Same setup as `engine_tests` (`STREAMMOE_Q38_TINY`).

use std::path::Path;

use super::engine::{plan_bytes_with, prefetch_slots, q38_config, Q38Engine};
use super::engine_exec::Q38Exec;
use super::engine_tests::{goldens, rel_err, tiny_dir, Golden, TOL_GOLDEN};
use super::qsa::QsaMode;
use crate::riir::backend::cpu::CpuBackend;
use crate::riir::backend::Backend;
use crate::riir::backend::gpu::MetalBackend;
use crate::riir::io::expert_io::ExpertFiles;
use crate::riir::io::expert_io_mode::ExpertIoMode;
use crate::riir::io::hot_tier::HotTier;
use crate::riir::io::prefetch::{PrefetchGuess, PrefetchState};
use crate::riir::io::weight_file::WeightFile;
use crate::riir::variants::VARIANT;
use crate::riir::{MetalContext, MtlWeightBuf};

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

/// Expert files with a hot tier of `hot_slots` (0: none).
fn experts(dir: &Path, hot_slots: usize) -> ExpertFiles {
    let mut e = ExpertFiles::open_with(&root(dir), false, false).expect("experts");
    if hot_slots > 0 {
        let dev = metal::Device::system_default().expect("Metal device");
        let hot = HotTier::new(&dev, hot_slots, e.expert_size(), VARIANT.num_layers, VARIANT.num_experts).unwrap();
        e.attach_hot_tier(hot);
    }
    e
}

/// Prefetch reading `n` experts per layer (`None`: off), guessed the
/// previous token's way.
fn prefetch(n: Option<usize>, lookahead: bool) -> Option<PrefetchState> {
    prefetch_with(n, lookahead, PrefetchGuess::PrevToken)
}

/// [`prefetch`] with the guess `guess`.
fn prefetch_with(n: Option<usize>, lookahead: bool, guess: PrefetchGuess) -> Option<PrefetchState> {
    n.map(|n| {
        let mut p = PrefetchState::new(VARIANT.num_layers, ExpertIoMode::Pread);
        p.set_max_experts(n);
        p.set_lookahead(lookahead);
        p.set_guess(guess);
        p
    })
}

/// Every prompt, every position: logits rows.
fn run_all<B: Q38Exec>(e: &mut Q38Engine<B>, b: &mut B, wf: &WeightFile, ex: &ExpertFiles, gs: &[Golden]) -> Vec<Vec<f32>> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    let v = VARIANT.vocab_size;
    let mut rows = Vec::new();
    for g in gs {
        e.reset(b).unwrap();
        for &tok in g.input_ids.iter().chain(&g.decode_tokens) {
            let mut l = vec![0.0f32; v];
            e.step(b, wf, ex, &pool, tok, Some(&mut l)).unwrap();
            rows.push(l);
        }
    }
    rows
}

fn same_bits(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()))
}

const PREV: PrefetchGuess = PrefetchGuess::PrevToken;
const CROSS: PrefetchGuess = PrefetchGuess::CrossLayer;

/// The arms: (label, prefetch experts, lookahead, hot-tier slots, guess).
const ARMS: [(&str, Option<usize>, bool, usize, PrefetchGuess); 9] = [
    ("off", None, false, 0, PREV),
    ("prefetch all", Some(16), false, 0, PREV),
    ("prefetch 1 + lookahead", Some(1), true, 0, PREV),
    ("hot tier", None, false, 12, PREV),
    ("prefetch all + hot tier", Some(16), false, 12, PREV),
    ("prefetch 1 + lookahead + hot tier", Some(1), true, 12, PREV),
    ("cross 2", Some(2), false, 0, CROSS),
    ("cross all", Some(16), false, 0, CROSS),
    ("cross 2 + hot tier", Some(2), false, 12, CROSS),
];

/// Metal: every arm's logits equal the "off" arm's bit for bit; the
/// prefetch and the hot tier both get hits. Prints the counters.
#[test]
fn prefetch_and_hot_tier_do_not_change_logits_metal() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 prefetch] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let gs = goldens(&dir);
    let wf = open_wf(&dir);
    let mut base: Option<Vec<Vec<f32>>> = None;
    for (label, n, ahead, hot, guess) in ARMS {
        let ex = experts(&dir, hot);
        let mut b = metal(&dir);
        let pf = prefetch_with(n, ahead, guess);
        let mut e = Q38Engine::with_prefetch(&mut b, &wf, &root(&dir), 64, ex.expert_size(), pf).unwrap();
        e.set_qsa_mode(QsaMode::Sparse);
        let rows = run_all(&mut e, &mut b, &wf, &ex, &gs);
        let (hits, misses) = e.prefetch_stats();
        let st = ex.io_stats();
        let h = st.hot.unwrap_or_default();
        eprintln!(
            "[q38 prefetch] {label}: prefetch hits {hits} misses {misses} (slots/set {}), prefetch reads {} \
             (used {}), demand reads {}, hot hits {} misses {} admissions {} resident {}/{}, guess prev {}/{} \
             cross {}/{}",
            e.prefetch_slots(),
            st.prefetch.reads,
            st.prefetch_hits,
            st.demand.reads,
            h.hits,
            h.misses,
            h.admissions,
            h.resident,
            h.capacity,
            st.guess_prev.correct,
            st.guess_prev.guessed,
            st.guess_cross.correct,
            st.guess_cross.guessed,
        );
        if guess == CROSS {
            assert!(st.guess_cross.guessed > 0, "{label}: cross guess never scored");
        } else {
            assert_eq!(st.guess_cross.guessed, 0, "{label}");
        }
        if n.is_some() {
            assert!(hits > 0, "{label}: no prefetch hits");
            assert!(st.prefetch.reads > 0, "{label}: no prefetch reads");
        } else {
            assert_eq!((hits, misses, st.prefetch.reads), (0, 0, 0), "{label}");
        }
        if hot > 0 {
            assert!(h.hits > 0, "{label}: no hot-tier hits");
        }
        match &base {
            None => base = Some(rows),
            Some(want) => assert!(same_bits(&rows, want), "{label}: logits differ from prefetch/hot off"),
        }
    }
}

/// The CPU backend, prefetch and hot tier on vs off: bit for bit.
#[test]
fn prefetch_and_hot_tier_do_not_change_logits_cpu() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 prefetch] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let gs: Vec<Golden> = goldens(&dir).into_iter().take(1).collect();
    let wf = open_wf(&dir);
    let mut out = Vec::new();
    for (n, hot, ahead, guess) in [(None, 0, true, PREV), (Some(16), 12, true, PREV), (Some(2), 12, false, CROSS)] {
        let ex = experts(&dir, hot);
        let mut b = CpuBackend::new(open_wf(&dir));
        let pf = prefetch_with(n, ahead, guess);
        let mut e = Q38Engine::with_prefetch(&mut b, &wf, &root(&dir), 64, ex.expert_size(), pf).unwrap();
        e.set_qsa_mode(QsaMode::Sparse);
        out.push(run_all(&mut e, &mut b, &wf, &ex, &gs));
        eprintln!("[q38 prefetch] cpu {n:?} {guess:?}: prefetch {:?}", e.prefetch_stats());
    }
    assert!(same_bits(&out[0], &out[1]), "CPU: prefetch + hot tier changed the logits");
    assert!(same_bits(&out[0], &out[2]), "CPU: the cross-layer guess changed the logits");
}

/// The staging buffer the engine allocates is what the plan counts:
/// `top_k` demand slots plus two sets of `min(prefetch_experts,
/// top_k)`, at the expert size (the padded file stride is only an
/// offset).
#[test]
fn staging_matches_plan() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 prefetch] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let wf = open_wf(&dir);
    let ex = experts(&dir, 0);
    let es = ex.expert_size();
    assert!(ex.expert_stride() >= es);
    let k = q38_config().top_k;
    for n in [0usize, 1, 4, 16] {
        let mut b = metal(&dir);
        let pf = prefetch(Some(n), false);
        let e = Q38Engine::with_prefetch(&mut b, &wf, &root(&dir), 64, es, pf).unwrap();
        let p = prefetch_slots(n);
        assert_eq!(e.prefetch_slots(), p);
        let plan = plan_bytes_with(64, es, 0, n).expert_slots;
        assert_eq!(plan, ((k + 2 * p) * es) as u64);
        let got: u64 = b.pool().bytes_by_label().iter().filter(|x| x.0 == "q38.expert_staging").map(|x| x.1).sum();
        // One f32 of slack, rounded up to the pool's alignment.
        assert!(got >= plan && got <= plan + 16 * 1024, "prefetch {n}: staging {got} B, plan {plan} B");
    }
}

/// The PLE on the backend vs the host PLE (`set_ple_on_host`), every
/// position of every prompt, on Metal and on the CPU backend; the
/// backend PLE also against the goldens. Switching mid-sequence moves
/// the conv state along.
#[test]
fn ple_on_backend_matches_host_ple() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 ple] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let gs = goldens(&dir);
    let wf = open_wf(&dir);
    let ex = experts(&dir, 0);
    let es = ex.expert_size();
    let want: Vec<Vec<f32>> = gs
        .iter()
        .flat_map(|g| {
            let v = VARIANT.vocab_size;
            g.prefill_logits.chunks(v).chain(g.decode_logits.chunks(v)).map(|r| r.to_vec()).collect::<Vec<_>>()
        })
        .collect();
    let worst = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max(rel_err(x, y)));
    // Metal.
    let mut b = metal(&dir);
    let mut e = Q38Engine::new(&mut b, &wf, &root(&dir), 64, es).unwrap();
    e.set_qsa_mode(QsaMode::Sparse);
    assert!(!e.ple_on_host(), "the PLE runs on the backend by default");
    let gpu = run_all(&mut e, &mut b, &wf, &ex, &gs);
    e.set_ple_on_host(&mut b, true).unwrap();
    let host = run_all(&mut e, &mut b, &wf, &ex, &gs);
    let (eg, eh) = (worst(&gpu, &host), worst(&gpu, &want));
    // Switch paths every token: the conv state follows.
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    let g0 = &gs[0];
    e.reset(&mut b).unwrap();
    let mut mixed = Vec::new();
    for (i, &tok) in g0.input_ids.iter().chain(&g0.decode_tokens).enumerate() {
        e.set_ple_on_host(&mut b, i % 2 == 0).unwrap();
        let mut l = vec![0.0f32; VARIANT.vocab_size];
        e.step(&mut b, &wf, &ex, &pool, tok, Some(&mut l)).unwrap();
        mixed.push(l);
    }
    let em = worst(&mixed, &gpu[..mixed.len()]);
    // CPU backend, same ops.
    let mut cb = CpuBackend::new(open_wf(&dir));
    let mut ce = Q38Engine::new(&mut cb, &wf, &root(&dir), 64, es).unwrap();
    ce.set_qsa_mode(QsaMode::Sparse);
    let cpu = run_all(&mut ce, &mut cb, &wf, &ex, &gs);
    ce.set_ple_on_host(&mut cb, true).unwrap();
    let cpu_host = run_all(&mut ce, &mut cb, &wf, &ex, &gs);
    let (ec, ech) = (worst(&cpu, &gpu), worst(&cpu, &cpu_host));
    eprintln!(
        "[q38 ple] metal backend-PLE vs host-PLE {eg:.2e}, vs goldens {eh:.2e}, switching {em:.2e}; \
         cpu backend-PLE vs metal {ec:.2e}, vs cpu host-PLE {ech:.2e}"
    );
    assert!(eg <= 1e-4 && em <= 1e-4 && ech <= 1e-4, "backend PLE vs host PLE: {eg} {em} {ech}");
    assert!(ec <= 1e-3, "cpu vs metal {ec}");
    assert!(eh <= TOL_GOLDEN, "vs goldens {eh}");
}
