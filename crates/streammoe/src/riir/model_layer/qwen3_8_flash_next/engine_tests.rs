//! [`super::engine::Q38Engine`] against the reference: the converted
//! tiny model of `make_tiny.py --shapes engine` and its goldens.
//!
//! Needs the `model-qwen3-8-flash-next-tiny` feature and
//! `STREAMMOE_Q38_TINY` pointing at the `make_tiny.py --out` directory's
//! `tiny/` (with `streammoe/` converted by `convert.py` and
//! `goldens.safetensors`). Skips with a message when it is unset.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::engine::Q38Engine;
use super::engine_exec::Q38Exec;
use super::qsa::QsaMode;
use crate::riir::backend::cpu::CpuBackend;
use crate::riir::backend::gpu::MetalBackend;
use crate::riir::io::expert_io::ExpertFiles;
use crate::riir::{MetalContext, MtlWeightBuf};
use crate::riir::io::weight_file::WeightFile;
use crate::riir::variants::VARIANT;

/// Logit tolerance against the q4 goldens: `max |Δ| ≤ TOL · max |golden|`
/// per row. The reference runs the same 4-bit weights in f32 (torch);
/// the engine differs in accumulation order and in the Metal kernels'
/// fast-math, ~1e-5 relative in practice.
pub const TOL_GOLDEN: f32 = 2e-3;
/// CPU backend vs Metal backend, same metric.
pub const TOL_CPU_METAL: f32 = 1e-3;

pub fn tiny_dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var_os("STREAMMOE_Q38_TINY")?);
    if d.join("goldens.safetensors").is_file() && d.join("streammoe/artifacts/model_weights.bin").is_file() {
        Some(d)
    } else {
        eprintln!("[q38 engine] STREAMMOE_Q38_TINY={} lacks goldens / converted model", d.display());
        None
    }
}

/// A safetensors tensor: dtype, shape, data as f32 (I64 cast).
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Minimal safetensors reader (F32 / I64 / I32).
pub fn read_safetensors(path: &Path) -> HashMap<String, Tensor> {
    let bytes = std::fs::read(path).expect("read goldens");
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).expect("header");
    let data = &bytes[8 + n..];
    let mut out = HashMap::new();
    for (k, v) in header.as_object().unwrap() {
        if k == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().unwrap();
        let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let off = v["data_offsets"].as_array().unwrap();
        let s = &data[off[0].as_u64().unwrap() as usize..off[1].as_u64().unwrap() as usize];
        let data: Vec<f32> = match dtype {
            "F32" => s.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "I64" => s.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            "I32" => s.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            _ => continue,
        };
        out.insert(k.clone(), Tensor { shape, data });
    }
    out
}

/// One golden prompt.
pub struct Golden {
    pub input_ids: Vec<i32>,
    pub prefill_logits: Vec<f32>,
    pub decode_tokens: Vec<i32>,
    pub decode_logits: Vec<f32>,
}

pub fn goldens(dir: &Path) -> Vec<Golden> {
    let t = read_safetensors(&dir.join("goldens.safetensors"));
    let mut out = Vec::new();
    for i in 0.. {
        let k = |f: &str| format!("q4.p{i}.{f}");
        let Some(ids) = t.get(&k("input_ids")) else { break };
        let ints = |x: &Tensor| x.data.iter().map(|&v| v as i32).collect::<Vec<_>>();
        assert_eq!(t[&k("prefill_logits")].shape[1], VARIANT.vocab_size);
        out.push(Golden {
            input_ids: ints(ids),
            prefill_logits: t[&k("prefill_logits")].data.clone(),
            decode_tokens: ints(&t[&k("decode_tokens")]),
            decode_logits: t[&k("decode_logits")].data.clone(),
        });
    }
    assert!(!out.is_empty(), "no q4 goldens");
    out
}

/// `max |a − b| / max |b|`.
pub fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-6);
    a.iter().zip(b).fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs())) / scale
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |bi, (i, &x)| if x > v[bi] { i } else { bi })
}

fn io_pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap()
}

struct Model {
    wf: WeightFile,
    experts: ExpertFiles,
    root: PathBuf,
}

fn open_model(dir: &Path) -> Model {
    let a = dir.join("streammoe/artifacts");
    let root = dir.join("streammoe/root");
    Model {
        wf: open_wf(dir),
        experts: ExpertFiles::open_with(&root, false, false).expect("experts"),
        root: a.parent().unwrap().join("root"),
    }
}

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

fn cpu(dir: &Path) -> CpuBackend {
    CpuBackend::new(open_wf(dir))
}

/// Every logits row of `g`: prefill positions, then each decode token.
fn run_prompt<B: Q38Exec>(e: &mut Q38Engine<B>, b: &mut B, m: &Model, pool: &rayon::ThreadPool, g: &Golden) -> Vec<Vec<f32>> {
    e.reset(b).unwrap();
    let v = VARIANT.vocab_size;
    let mut rows = Vec::new();
    for &tok in g.input_ids.iter().chain(&g.decode_tokens) {
        let mut l = vec![0.0f32; v];
        e.step(b, &m.wf, &m.experts, pool, tok, Some(&mut l)).unwrap();
        rows.push(l);
    }
    rows
}

fn golden_rows(g: &Golden) -> Vec<&[f32]> {
    let v = VARIANT.vocab_size;
    g.prefill_logits.chunks(v).chain(g.decode_logits.chunks(v)).collect()
}

/// Metal engine vs the q4 goldens, every position of every prompt;
/// the CPU backend vs Metal op for op.
#[test]
fn engine_matches_goldens_cpu_and_metal() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 engine] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let gs = goldens(&dir);
    let m = open_model(&dir);
    let es = m.experts.expert_size();
    let pool = io_pool();
    let mut gb = metal(&dir);
    let mut ge = Q38Engine::new(&mut gb, &m.wf, &m.root, 64, es).unwrap();
    ge.set_qsa_mode(QsaMode::Sparse);
    let mut cb = cpu(&dir);
    let mut ce = Q38Engine::new(&mut cb, &m.wf, &m.root, 64, es).unwrap();
    ce.set_qsa_mode(QsaMode::Sparse);
    let (mut worst_g, mut worst_c) = (0.0f32, 0.0f32);
    for (pi, g) in gs.iter().enumerate() {
        let want = golden_rows(g);
        let got = run_prompt(&mut ge, &mut gb, &m, &pool, g);
        let gotc = run_prompt(&mut ce, &mut cb, &m, &pool, g);
        for (r, w) in want.iter().enumerate() {
            let eg = rel_err(&got[r], w);
            let ec = rel_err(&gotc[r], &got[r]);
            worst_g = worst_g.max(eg);
            worst_c = worst_c.max(ec);
            assert!(eg <= TOL_GOLDEN, "prompt {pi} row {r}: metal vs golden rel err {eg}");
            assert!(ec <= TOL_CPU_METAL, "prompt {pi} row {r}: cpu vs metal rel err {ec}");
            assert_eq!(argmax(&got[r]), argmax(w), "prompt {pi} row {r}: argmax");
        }
    }
    eprintln!("[q38 engine] worst rel err: metal vs golden {worst_g:.3e}, cpu vs metal {worst_c:.3e}");
}

/// `state_save` mid-sequence, run on, `state_load`, run the same
/// tokens again: identical logits (Metal).
#[test]
fn engine_state_round_trip() {
    let Some(dir) = tiny_dir() else {
        eprintln!("[q38 engine] STREAMMOE_Q38_TINY unset: skipped");
        return;
    };
    let gs = goldens(&dir);
    let g = &gs[0];
    let m = open_model(&dir);
    let pool = io_pool();
    let mut b = metal(&dir);
    let mut e = Q38Engine::new(&mut b, &m.wf, &m.root, 64, m.experts.expert_size()).unwrap();
    e.set_qsa_mode(QsaMode::Sparse);
    let v = VARIANT.vocab_size;
    // Save past the QSA budget so the sparse path and the indexer keys
    // are in the snapshot.
    let cut = g.input_ids.len() - 3;
    e.eval(&mut b, &m.wf, &m.experts, &pool, &g.input_ids[..cut], None).unwrap();
    let mut snap = vec![0u8; e.state_size()];
    let n = e.state_save(&b, &mut snap).unwrap();
    assert_eq!(n, snap.len());
    let tail: Vec<i32> = g.input_ids[cut..].iter().chain(&g.decode_tokens).copied().collect();
    let run = |e: &mut Q38Engine<MetalBackend>, b: &mut MetalBackend| {
        tail.iter()
            .map(|&t| {
                let mut l = vec![0.0f32; v];
                e.step(b, &m.wf, &m.experts, &pool, t, Some(&mut l)).unwrap();
                l
            })
            .collect::<Vec<_>>()
    };
    let first = run(&mut e, &mut b);
    // Clobber, then restore.
    e.reset(&mut b).unwrap();
    e.eval(&mut b, &m.wf, &m.experts, &pool, &g.decode_tokens, None).unwrap();
    e.state_load(&mut b, &snap).unwrap();
    assert_eq!(e.pos(), cut);
    let second = run(&mut e, &mut b);
    assert_eq!(first, second, "logits after state_load differ");
    // And they are the goldens' rows.
    let want = golden_rows(g);
    for (i, row) in first.iter().enumerate() {
        assert!(rel_err(row, want[cut + i]) <= TOL_GOLDEN);
    }
    // A corrupt snapshot is rejected without touching the state.
    let mut bad = snap.clone();
    bad[0] ^= 0xFF;
    assert!(e.state_load(&mut b, &bad).is_err());
    assert_eq!(e.pos(), g.input_ids.len() + g.decode_tokens.len());
}
