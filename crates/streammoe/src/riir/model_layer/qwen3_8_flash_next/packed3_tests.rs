//! Packed 3-bit experts on the tiny model: the converted 4-bit experts
//! requantized to 3-bit codes in 4-bit nibbles
//! (`tools/requant_experts.py`), and the same codes packed 8 per 3
//! bytes (`tools/pack_experts.py`). The packed experts must compute
//! exactly what the nibble ones do — bit for bit, prefill (chunks,
//! padded prefill staging, the 3-bit gather GEMM) and decode, on the
//! CPU and the Metal backend, hot tier on or off — and the 3-bit model
//! must stay close to the 4-bit one.
//!
//! Same setup as `engine_tests` (`STREAMMOE_Q38_TINY`), plus `python3`
//! with NumPy: the test writes `streammoe/root-3nib` and
//! `streammoe/root-3bit` next to `streammoe/root` (~1 MB each). The
//! tiny experts are 30,720 B at 4-bit and 24,576 B packed, both at the
//! converter's 32,768 B padded stride, so the stride alone does not
//! tell the layouts apart (`layout.json` does).

use std::path::{Path, PathBuf};
use std::process::Command;

use super::engine::{Q38Engine, Q38PrefillIo};
use super::engine_exec::Q38Exec;
use super::engine_tests::{goldens, rel_err, tiny_dir, TOL_CPU_METAL};
use super::qsa::QsaMode;
use crate::riir::backend::cpu::CpuBackend;
use crate::riir::backend::gpu::MetalBackend;
use crate::riir::io::expert_io::ExpertFiles;
use crate::riir::io::hot_tier::HotTier;
use crate::riir::io::weight_file::WeightFile;
use crate::riir::variants::{ExpertLayout, VARIANT};
use crate::riir::{MetalContext, MtlWeightBuf};

/// 560 tokens: one 512-token chunk plus 48, past the QSA budget.
const N_CTX: usize = 576;
const PROMPT: usize = 560;

fn open_wf(dir: &Path) -> WeightFile {
    let a = dir.join("streammoe/artifacts");
    WeightFile::open(&a.join("model_weights.bin"), &a.join("model_weights.json")).expect("weights")
}

fn metal(dir: &Path) -> MetalBackend {
    let wf = open_wf(dir);
    let metal = MetalContext::new().expect("MetalContext::new");
    let buf = MtlWeightBuf::wrap(&wf, metal.device());
    MetalBackend::new(metal, buf).expect("MetalBackend::new")
}

fn run_tool(args: &[&Path]) {
    let st = Command::new("python3").arg("-I").args(args).status().expect("run python3");
    assert!(st.success(), "{args:?} failed");
}

/// `(nibble 3-bit root, packed 3-bit root)`, written by the tools from
/// `streammoe/root` (the n-gram table copied along), once per run.
fn roots_3bit(dir: &Path) -> (PathBuf, PathBuf) {
    static ROOTS: std::sync::OnceLock<(PathBuf, PathBuf)> = std::sync::OnceLock::new();
    ROOTS.get_or_init(|| write_roots_3bit(dir)).clone()
}

fn write_roots_3bit(dir: &Path) -> (PathBuf, PathBuf) {
    let tools = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools");
    let m = dir.join("streammoe");
    let (src, nib, packed) = (m.join("root"), m.join("root-3nib"), m.join("root-3bit"));
    for d in [&nib, &packed] {
        let _ = std::fs::remove_dir_all(d);
    }
    let out = format!("3={}", nib.display());
    run_tool(&[&tools.join("requant_experts.py"), &src, Path::new("--out"), Path::new(&out)]);
    run_tool(&[&tools.join("pack_experts.py"), &nib, &packed]);
    for d in [&nib, &packed] {
        std::fs::copy(src.join("ngram.mfng"), d.join("ngram.mfng")).expect("copy ngram.mfng");
    }
    (nib, packed)
}

fn experts(root: &Path, hot_slots: usize) -> ExpertFiles {
    let mut e = ExpertFiles::open_with(root, false, false).expect("experts");
    if hot_slots > 0 {
        let dev = metal::Device::system_default().expect("Metal device");
        let hot = HotTier::new(&dev, hot_slots, e.expert_size(), VARIANT.num_layers, VARIANT.num_experts).unwrap();
        e.attach_hot_tier(hot);
    }
    e
}

fn prompt() -> Vec<i32> {
    let v = VARIANT.vocab_size;
    (0..PROMPT).map(|i| ((i * 37 + 5) % v) as i32).collect()
}

/// Every logits row: the prompt (`eval` in chunks of `chunk`, or a
/// step per token when `chunk` is 0), then three decode steps.
/// Rows: the prompt's last, then the decode ones.
fn run<B: Q38Exec>(
    b: &mut B,
    wf: &WeightFile,
    root: &Path,
    hot: usize,
    chunk: usize,
    slots: usize,
) -> (Vec<Vec<f32>>, Vec<u8>) {
    let ex = experts(root, hot);
    let es = ex.expert_size();
    let mut io = Q38PrefillIo::for_root(root, N_CTX, es);
    io.chunk = 512;
    io.slots = 64;
    let mut e = Q38Engine::with_options(b, wf, root, N_CTX, es, None, io).unwrap();
    e.set_qsa_mode(QsaMode::Sparse);
    if chunk > 0 {
        e.set_prefill_chunk(chunk);
        e.set_prefill_slots(slots);
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let v = VARIANT.vocab_size;
    let p = prompt();
    let mut rows = vec![vec![0.0f32; v]];
    if chunk > 0 {
        e.eval(b, wf, &ex, &pool, &p, Some(&mut rows[0])).unwrap();
    } else {
        for (i, &t) in p.iter().enumerate() {
            let out = if i + 1 == p.len() { Some(&mut rows[0][..]) } else { None };
            e.step(b, wf, &ex, &pool, t, out).unwrap();
        }
    }
    let mut snap = vec![0u8; e.state_size()];
    e.state_save(b, &mut snap).unwrap();
    for t in [5, 17, 300] {
        let mut l = vec![0.0f32; v];
        e.step(b, wf, &ex, &pool, t, Some(&mut l)).unwrap();
        rows.push(l);
    }
    (rows, snap)
}

fn same_bits(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()))
}

/// (label, hot slots, prefill chunk (0 = a step per token), staging slots).
const ARMS: [(&str, usize, usize, usize); 4] = [
    ("chunk 512, 64 slots", 0, 512, 64),
    ("chunk 5, 4 slots (overlapped halves)", 0, 5, 4),
    ("step per token (decode staging)", 0, 0, 0),
    ("chunk 512, 64 slots, hot tier 12", 12, 512, 64),
];

/// The tools' layouts, and the engine reading each.
#[test]
fn tools_write_detected_layouts() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 packed 3-bit] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let (nib, packed) = roots_3bit(&dir);
    let src = experts(&dir.join("streammoe/root"), 0);
    let (n, p) = (experts(&nib, 0), experts(&packed, 0));
    assert_eq!((src.layout(), src.expert_stride()), (ExpertLayout::Q4, 32768));
    assert_eq!((n.layout(), n.expert_stride()), (ExpertLayout::Q4, 32768));
    assert_eq!((p.layout(), p.expert_stride()), (ExpertLayout::Q3, 32768));
    assert_eq!((n.expert_size(), p.expert_size()), (30720, 24576));
}

/// Packed 3-bit vs nibble 3-bit: bit-identical logits and state on the
/// CPU and the Metal backend, every arm; CPU vs Metal within
/// [`TOL_CPU_METAL`].
#[test]
fn packed_3bit_bit_identical_to_nibbles() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 packed 3-bit] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let (nib, packed) = roots_3bit(&dir);
    let wf = open_wf(&dir);
    let mut metal_rows = Vec::new();
    for (label, hot, chunk, slots) in ARMS {
        let mut b = metal(&dir);
        let (a, sa) = run(&mut b, &wf, &nib, 0, chunk, slots);
        let (p, sp) = run(&mut b, &wf, &packed, hot, chunk, slots);
        assert!(same_bits(&a, &p), "metal {label}: packed 3-bit logits differ from nibble 3-bit");
        assert!(sa == sp, "metal {label}: states differ");
        eprintln!("[q38 packed 3-bit] metal {label}: {} rows bit-identical", a.len());
        metal_rows.push(p);
    }
    // The CPU backend (slow at 560 tokens): the chunked arms.
    for (ai, (label, hot, chunk, slots)) in ARMS.iter().copied().enumerate() {
        if chunk == 0 {
            continue;
        }
        let mut b = CpuBackend::new(open_wf(&dir));
        let (a, sa) = run(&mut b, &wf, &nib, 0, chunk, slots);
        let (p, sp) = run(&mut b, &wf, &packed, hot, chunk, slots);
        assert!(same_bits(&a, &p), "cpu {label}: packed 3-bit logits differ from nibble 3-bit");
        assert!(sa == sp, "cpu {label}: states differ");
        let e = p.iter().zip(&metal_rows[ai]).map(|(c, g)| rel_err(c, g)).fold(0.0f32, f32::max);
        eprintln!("[q38 packed 3-bit] cpu {label}: bit-identical; cpu vs metal rel err {e:.3e}");
        assert!(e <= TOL_CPU_METAL, "cpu {label}: cpu vs metal rel err {e}");
    }
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
    let lse = x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
    x.iter().map(|&v| v as f64 - lse).collect()
}

/// KL(p ‖ q) of two logits rows.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (lp, lq) = (log_softmax(p), log_softmax(q));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |bi, (i, &x)| if x > v[bi] { i } else { bi })
}

/// Packed 3-bit vs 4-bit on Metal, every position of every golden
/// prompt (a step per token) and the 560-token prompt's rows: mean /
/// max KL(4-bit ‖ 3-bit), max relative logit difference, argmax
/// agreement. Measured: KL mean 0.0013, max 0.0031 over 69 rows,
/// argmax 57 / 69 (the tiny random model's logits are flat); the
/// bounds leave ~10x room and catch a broken 3-bit path.
#[test]
fn packed_3bit_close_to_4bit() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 packed 3-bit] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let (_, packed) = roots_3bit(&dir);
    let q4 = dir.join("streammoe/root");
    let wf = open_wf(&dir);
    let v = VARIANT.vocab_size;
    let mut b = metal(&dir);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    let mut rows: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    let (e4, e3) = (experts(&q4, 0), experts(&packed, 0));
    let mut eng = |b: &mut MetalBackend, ex: &ExpertFiles, root: &Path| {
        let mut e = Q38Engine::new(b, &wf, root, 64, ex.expert_size()).unwrap();
        e.set_qsa_mode(QsaMode::Sparse);
        e
    };
    let (mut g4, mut g3) = (eng(&mut b, &e4, &q4), eng(&mut b, &e3, &packed));
    for g in goldens(&dir) {
        g4.reset(&mut b).unwrap();
        g3.reset(&mut b).unwrap();
        for &t in g.input_ids.iter().chain(&g.decode_tokens) {
            let (mut a, mut c) = (vec![0.0f32; v], vec![0.0f32; v]);
            g4.step(&mut b, &wf, &e4, &pool, t, Some(&mut a)).unwrap();
            g3.step(&mut b, &wf, &e3, &pool, t, Some(&mut c)).unwrap();
            rows.push((a, c));
        }
    }
    let (r4, _) = run(&mut b, &wf, &q4, 0, 512, 64);
    let (r3, _) = run(&mut b, &wf, &packed, 0, 512, 64);
    rows.extend(r4.into_iter().zip(r3));
    let kls: Vec<f64> = rows.iter().map(|(a, c)| kl(a, c)).collect();
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    let max = kls.iter().fold(0.0f64, |m, &x| m.max(x));
    let rel = rows.iter().map(|(a, c)| rel_err(c, a)).fold(0.0f32, f32::max);
    let agree = rows.iter().filter(|(a, c)| argmax(a) == argmax(c)).count();
    eprintln!(
        "[q38 packed 3-bit] vs 4-bit over {} rows: KL mean {mean:.4} max {max:.4}, max rel logit diff {rel:.3}, \
         argmax agree {agree}/{}",
        rows.len(),
        rows.len()
    );
    assert!(rows.iter().all(|(_, c)| c.iter().all(|x| x.is_finite())));
    assert!(mean < 0.01 && max < 0.03, "3-bit too far from 4-bit: KL mean {mean} max {max}");
    assert!(agree * 2 > rows.len(), "argmax agrees on {agree} of {} rows", rows.len());
}
