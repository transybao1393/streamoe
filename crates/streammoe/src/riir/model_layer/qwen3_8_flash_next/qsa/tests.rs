//! QSA against the transformers reference
//! (`tools/model_layer/qwen3_8_flash_next/qsa_reference.py`, fixture
//! `tests/fixtures/qsa_reference.*`), dense vs indexed, and the graph
//! path on the CPU and Metal backends against the CPU path.

use super::*;

/// The reference script's tiny shape: the real attention with hidden
/// 128 and a 16-token budget (dense is exact up to 19 visible tokens).
const TINY: QsaShape = QsaShape { hidden: 128, budget: 16, ..QsaShape::QWEN3_8_FLASH_NEXT };
const SEED: u64 = 0x5153_4131;
const N_TOTAL: usize = 40;
const N_PREFILL: usize = 12;

/// xorshift64, bit-identical to the reference script's `Rng`.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    }
    fn mat(&mut self, rows: usize, cols: usize, scale: f32) -> Vec<f32> {
        (0..rows * cols).map(|_| self.unit() * scale).collect()
    }
    fn norm_w(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| 1.0 + (((self.next_u64() >> 40) % 17) as f32 - 8.0) / 64.0).collect()
    }
}

struct Model {
    x: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    idx_qk_proj: Vec<f32>,
    idx_q_norm: Vec<f32>,
    idx_k_norm: Vec<f32>,
}

impl Model {
    /// Weights and inputs in the reference script's order.
    fn reference(s: &QsaShape) -> Model {
        let mut r = Rng(SEED);
        let h = s.hidden;
        Model {
            x: r.mat(N_TOTAL, h, 1.0),
            q_proj: r.mat(s.q_proj_dim(), h, 1.0 / 16.0),
            k_proj: r.mat(s.kv_dim(), h, 1.0 / 16.0),
            v_proj: r.mat(s.kv_dim(), h, 1.0 / 16.0),
            o_proj: r.mat(h, s.q_dim(), 1.0 / 64.0),
            q_norm: r.norm_w(s.head_dim),
            k_norm: r.norm_w(s.head_dim),
            idx_qk_proj: r.mat(s.idx_proj_dim(), h, 1.0 / 16.0),
            idx_q_norm: r.norm_w(s.idx_head_dim),
            idx_k_norm: r.norm_w(s.idx_head_dim),
        }
    }
    fn weights(&self) -> QsaWeightsF32<'_> {
        QsaWeightsF32 {
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            o_proj: &self.o_proj,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            idx_qk_proj: &self.idx_qk_proj,
            idx_q_norm: &self.idx_q_norm,
            idx_k_norm: &self.idx_k_norm,
        }
    }
}

fn chunks(split: &str) -> Vec<usize> {
    match split {
        "prefill" => vec![N_TOTAL],
        _ => std::iter::once(N_PREFILL).chain(std::iter::repeat(1).take(N_TOTAL - N_PREFILL)).collect(),
    }
}

/// `[t, h, w]` per token, as the reference's `positions(kind)`.
fn positions(kind: &str) -> Vec<[i32; 3]> {
    let json: serde_json::Value = serde_json::from_str(FIXTURE_JSON).unwrap();
    let p = &json["positions"][kind];
    (0..N_TOTAL)
        .map(|i| [0, 1, 2].map(|a| p[a][i].as_i64().unwrap() as i32))
        .collect()
}

/// The whole sequence through [`qsa_forward_cpu`] in `chunks`.
fn run_cpu(s: &QsaShape, m: &Model, mode: QsaMode, pos: &[[i32; 3]], chunks: &[usize]) -> Vec<f32> {
    let w = m.weights();
    let mut cache = QsaCpuCache::new(s, &m.idx_k_norm);
    let mut out = vec![0.0f32; N_TOTAL * s.hidden];
    let mut t0 = 0;
    for &n in chunks {
        let (a, b) = (t0 * s.hidden, (t0 + n) * s.hidden);
        qsa_forward_cpu(s, mode, &w, &m.x[a..b], &pos[t0..t0 + n], &mut cache, &mut out[a..b]);
        t0 += n;
    }
    out
}

const FIXTURE_JSON: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qsa_reference.json"));
const FIXTURE_BIN: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qsa_reference.bin"));

fn fixture_case(name: &str) -> Vec<f32> {
    let json: serde_json::Value = serde_json::from_str(FIXTURE_JSON).unwrap();
    let c = &json["cases"][name];
    let off = c["offset"].as_u64().unwrap() as usize;
    FIXTURE_BIN[off..off + N_TOTAL * TINY.hidden * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn max_abs(a: &[f32]) -> f32 {
    a.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}
/// Max |a - b| of each row.
fn row_diffs(s: &QsaShape, a: &[f32], b: &[f32]) -> Vec<f32> {
    a.chunks(s.hidden).zip(b.chunks(s.hidden)).map(|(x, y)| max_diff(x, y)).collect()
}

#[test]
fn fixture_matches_the_shape() {
    let json: serde_json::Value = serde_json::from_str(FIXTURE_JSON).unwrap();
    let c = &json["config"];
    assert_eq!(c["hidden_size"], TINY.hidden);
    assert_eq!(c["num_attention_heads"], TINY.num_heads);
    assert_eq!(c["num_key_value_heads"], TINY.num_kv_heads);
    assert_eq!(c["head_dim"], TINY.head_dim);
    assert_eq!(c["indexer_n_heads"], TINY.idx_heads);
    assert_eq!(c["indexer_head_dim"], TINY.idx_head_dim);
    assert_eq!(c["indexer_budget"], TINY.budget);
    assert_eq!(c["indexer_compress_ratio"], TINY.compress_ratio);
    assert_eq!(json["seed"], SEED);
    assert_eq!(json["exact_until"], TINY.budget + TINY.compress_ratio - 1);
    assert_eq!(TINY.rotary_dim, (TINY.head_dim as f64 * c["partial_rotary_factor"].as_f64().unwrap()) as usize);
}

/// The CPU path against transformers, every case: text and M-RoPE
/// positions, dense and indexed, one 40-token prefill and a 12-token
/// prefill + 28 decode steps.
#[test]
fn cpu_matches_reference() {
    let m = Model::reference(&TINY);
    for kind in ["text", "mrope"] {
        let pos = positions(kind);
        for (mode_name, mode) in [("indexed", QsaMode::Sparse), ("dense", QsaMode::Dense)] {
            for split in ["prefill", "decode"] {
                let name = format!("{kind}.{mode_name}.{split}");
                let want = fixture_case(&name);
                let got = run_cpu(&TINY, &m, mode, &pos, &chunks(split));
                let (d, scale) = (max_diff(&got, &want), max_abs(&want));
                eprintln!("[qsa] {name}: max|diff| {d:.3e} (max|ref| {scale:.3e})");
                assert!(d <= 1e-5 * scale.max(1.0), "{name}: max|diff| {d} vs max|ref| {scale}");
            }
        }
    }
}

/// Dense and indexed attention agree bit for bit while every query sees
/// at most `budget + compress_ratio - 1` tokens and differ above it —
/// in the reference fixture and in the CPU path.
#[test]
fn dense_equals_qsa_up_to_budget_plus_ratio_minus_one() {
    let exact_until = TINY.budget + TINY.compress_ratio - 1;
    assert!(TINY.dense_is_exact(exact_until) && !TINY.dense_is_exact(exact_until + 1));
    let m = Model::reference(&TINY);
    for kind in ["text", "mrope"] {
        let pos = positions(kind);
        for split in ["prefill", "decode"] {
            let reference = (
                fixture_case(&format!("{kind}.dense.{split}")),
                fixture_case(&format!("{kind}.indexed.{split}")),
            );
            let ours = (
                run_cpu(&TINY, &m, QsaMode::Dense, &pos, &chunks(split)),
                run_cpu(&TINY, &m, QsaMode::Sparse, &pos, &chunks(split)),
            );
            for (who, (dense, sparse)) in [("reference", &reference), ("cpu", &ours)] {
                let rows = row_diffs(&TINY, dense, sparse);
                // Row r is the query that sees r + 1 tokens.
                assert!(
                    rows[..exact_until].iter().all(|&d| d == 0.0),
                    "{who} {kind}.{split}: dense != indexed within the budget: {:?}",
                    &rows[..exact_until]
                );
                let min_above = rows[exact_until..].iter().copied().fold(f32::INFINITY, f32::min);
                assert!(min_above > 1e-3, "{who} {kind}.{split}: dense == indexed above the budget ({min_above})");
            }
        }
    }
}

/// At the real shape: every token is kept up to 2051 visible tokens,
/// 512 whole blocks plus the tail from 2052 on.
#[test]
fn select_keeps_every_token_iff_within_budget() {
    let s = QsaShape::QWEN3_8_FLASH_NEXT;
    let mut r = Rng(7);
    let n = 2070;
    let mut cache = QsaIndexerCache::new(&s, r.norm_w(s.idx_head_dim));
    let pos: Vec<[i32; 3]> = (0..n as i32).map(|p| [p; 3]).collect();
    cache.append(&r.mat(n, s.idx_head_dim, 1.0), &pos);
    assert_eq!(cache.n_blocks(), n / 4);
    let q = r.mat(1, s.idx_q_dim(), 1.0);
    for nv in [1, 4, 2047, 2048, 2051] {
        assert!(cache.select(&q, nv).is_none(), "n_visible {nv} should keep every token");
    }
    for nv in 2052..=n {
        let sel = cache.select(&q, nv).expect("over the budget");
        assert_eq!(sel.len(), 2048 + nv % 4, "n_visible {nv}");
        assert!(sel.windows(2).all(|w| w[0] < w[1]) && *sel.last().unwrap() < nv as u32);
        // Whole blocks, then the tail.
        let tail = nv / 4 * 4;
        assert!(sel.iter().filter(|&&t| (t as usize) < tail).count() % 4 == 0);
        assert_eq!(sel.iter().filter(|&&t| t as usize >= tail).count(), nv - tail);
    }
}

/// `select` keeps the top-scoring blocks: brute force on the tiny shape.
#[test]
fn select_takes_the_top_blocks() {
    let s = TINY;
    let mut r = Rng(11);
    let n = 37;
    let mut cache = QsaIndexerCache::new(&s, r.norm_w(s.idx_head_dim));
    let pos: Vec<[i32; 3]> = (0..n as i32).map(|p| [p; 3]).collect();
    cache.append(&r.mat(n, s.idx_head_dim, 1.0), &pos);
    let q = r.mat(1, s.idx_q_dim(), 1.0);
    let d = s.idx_head_dim;
    let score = |b: usize| -> f32 {
        let kb = &cache.block_keys()[b * d..(b + 1) * d];
        q.chunks(d).map(|qh| qh.iter().zip(kb).map(|(a, k)| a * k).sum::<f32>().max(0.0)).sum()
    };
    let sel = cache.select(&q, n).unwrap();
    let kept: Vec<usize> = sel.iter().filter(|&&t| (t as usize) < 36).map(|&t| t as usize / 4).collect();
    let min_kept = kept.iter().map(|&b| score(b)).fold(f32::INFINITY, f32::min);
    for b in 0..9 {
        if !kept.contains(&b) {
            assert!(score(b) <= min_kept, "dropped block {b} outscores a kept one");
        }
    }
    assert_eq!(sel.iter().filter(|&&t| t == 36).count(), 1, "tail kept");
}

/// Truncating and appending again gives the same state as one append;
/// `restore` from the saved raw keys rebuilds the block keys.
#[test]
fn indexer_truncate_and_restore() {
    let s = TINY;
    let mut r = Rng(5);
    let n = 30;
    let k_norm = r.norm_w(s.idx_head_dim);
    let raw = r.mat(n, s.idx_head_dim, 1.0);
    let pos: Vec<[i32; 3]> = (0..n as i32).map(|p| [p, p + 1, p + 2]).collect();
    let mut full = QsaIndexerCache::new(&s, k_norm.clone());
    full.append(&raw, &pos);
    let mut cut = QsaIndexerCache::new(&s, k_norm.clone());
    cut.append(&raw[..21 * s.idx_head_dim], &pos[..21]);
    cut.truncate(13);
    assert_eq!((cut.len(), cut.n_blocks()), (13, 3));
    cut.append(&raw[13 * s.idx_head_dim..], &pos[13..]);
    assert_eq!(cut.block_keys(), full.block_keys());
    let mut restored = QsaIndexerCache::new(&s, k_norm);
    restored.restore(full.raw_keys(), full.positions());
    assert_eq!(restored.block_keys(), full.block_keys());
}

// =====================================================================
// Graph path: CpuBackend and MetalBackend against the CPU path
// =====================================================================

mod graph {
    use super::*;
    use crate::riir::backend::{Backend, BufferPool, CpuBackend, MetalBackend};
    use crate::riir::io::weight_file::WeightFile;
    use crate::riir::variants::GROUP_SIZE;
    use crate::riir::{MetalContext, MtlWeightBuf};

    fn bf16(v: f32) -> u16 {
        let b = v.to_bits();
        assert_eq!(b & 0xFFFF, 0, "{v} is not exact in bf16");
        (b >> 16) as u16
    }

    /// A random MLX 4-bit `[out, in]` matrix: packed u32 nibbles, bf16
    /// scales and biases per `GROUP_SIZE` inputs, and its f32 values.
    struct Q4 {
        rows: usize,
        cols: usize,
        packed: Vec<u8>,
        scales: Vec<u8>,
        biases: Vec<u8>,
        dense: Vec<f32>,
    }

    fn q4(r: &mut Rng, rows: usize, cols: usize) -> Q4 {
        let groups = cols / GROUP_SIZE;
        let mut sc = Vec::with_capacity(rows * groups);
        for _ in 0..rows * groups {
            sc.push((8 + (r.next_u64() >> 40) % 8) as f32 / 2048.0);
        }
        let mut packed = Vec::with_capacity(rows * cols / 2);
        let mut dense = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for c8 in 0..cols / 8 {
                let mut word = 0u32;
                for k in 0..8 {
                    let q = (r.next_u64() >> 40) as u32 & 0xF;
                    word |= q << (4 * k);
                    let s = sc[row * groups + (c8 * 8 + k) / GROUP_SIZE];
                    dense.push(q as f32 * s - 8.0 * s);
                }
                packed.extend_from_slice(&word.to_le_bytes());
            }
        }
        let scales = sc.iter().flat_map(|&s| bf16(s).to_le_bytes()).collect();
        let biases = sc.iter().flat_map(|&s| bf16(-8.0 * s).to_le_bytes()).collect();
        Q4 { rows, cols, packed, scales, biases, dense }
    }

    /// Synthetic weight file: the four projections + indexer
    /// projection as 4-bit, the per-head norms as bf16.
    struct Fixture {
        dir: std::path::PathBuf,
        w: QsaLayerWeights,
        dense: Model,
    }

    impl Fixture {
        fn build(s: &QsaShape, tag: &str) -> Fixture {
            let mut r = Rng(0xC0FFEE);
            let h = s.hidden;
            let x = r.mat(N_TOTAL, h, 1.0);
            let mats = [
                ("q_proj", q4(&mut r, s.q_proj_dim(), h)),
                ("k_proj", q4(&mut r, s.kv_dim(), h)),
                ("v_proj", q4(&mut r, s.kv_dim(), h)),
                ("o_proj", q4(&mut r, h, s.q_dim())),
                ("idx_qk_proj", q4(&mut r, s.idx_proj_dim(), h)),
            ];
            let (q_norm, k_norm) = (r.norm_w(s.head_dim), r.norm_w(s.head_dim));
            let (idx_q_norm, idx_k_norm) = (r.norm_w(s.idx_head_dim), r.norm_w(s.idx_head_dim));

            let mut blob: Vec<u8> = Vec::new();
            let mut tensors = serde_json::Map::new();
            let mut put = |name: String, dtype: &str, bits: i32, shape: Vec<usize>, bytes: &[u8]| -> u64 {
                while blob.len() % 64 != 0 {
                    blob.push(0);
                }
                let off = blob.len() as u64;
                blob.extend_from_slice(bytes);
                tensors.insert(
                    name,
                    serde_json::json!({ "offset": off, "size": bytes.len(), "shape": shape, "dtype": dtype, "bits": bits }),
                );
                off
            };
            let mut refs = Vec::new();
            for (name, m) in &mats {
                let (out_dim, in_dim) = (m.rows, m.cols);
                let w_off = put(format!("{name}.weight"), "U32", 4, vec![out_dim, in_dim / 8], &m.packed);
                let s_off = put(format!("{name}.scales"), "BF16", 0, vec![out_dim, in_dim / GROUP_SIZE], &m.scales);
                let b_off = put(format!("{name}.biases"), "BF16", 0, vec![out_dim, in_dim / GROUP_SIZE], &m.biases);
                refs.push(WeightRef { w_off, s_off, b_off, bits: 4 });
            }
            let norm_bytes = |w: &[f32]| -> Vec<u8> { w.iter().flat_map(|&v| bf16(v).to_le_bytes()).collect() };
            let q_norm_off = put("q_norm.weight".into(), "BF16", 0, vec![s.head_dim], &norm_bytes(&q_norm));
            let k_norm_off = put("k_norm.weight".into(), "BF16", 0, vec![s.head_dim], &norm_bytes(&k_norm));

            let dir = std::env::temp_dir().join(format!("streammoe-qsa-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model_weights.bin"), &blob).unwrap();
            std::fs::write(
                dir.join("model_weights.json"),
                serde_json::json!({ "tensors": tensors }).to_string(),
            )
            .unwrap();
            let [q_proj, k_proj, v_proj, o_proj, idx_qk_proj] = mats.map(|(_, m)| m.dense);
            Fixture {
                dir,
                w: QsaLayerWeights {
                    q_proj: refs[0],
                    k_proj: refs[1],
                    v_proj: refs[2],
                    o_proj: refs[3],
                    idx_qk_proj: refs[4],
                    q_norm_off,
                    k_norm_off,
                },
                dense: Model { x, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, idx_qk_proj, idx_q_norm, idx_k_norm },
            }
        }
        fn open(&self) -> WeightFile {
            WeightFile::open(&self.dir.join("model_weights.bin"), &self.dir.join("model_weights.json")).unwrap()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn bytes_f32(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// What q38-core's layer does per chunk: pre-SDPA graph, the
    /// indexer on the host, then the dense SDPA graph or the host
    /// block-selected attention, then `o_proj`.
    fn run_graph<B: Backend>(
        b: &mut B,
        s: &QsaShape,
        f: &Fixture,
        mode: QsaMode,
        pos: &[[i32; 3]],
        mrope: bool,
        chunks: &[usize],
    ) -> (Vec<f32>, usize)
    where
        B::Pool: BufferPool,
    {
        // Per-chunk buffers sized to the chunk: the CPU backend's
        // per-head norm and RoPE want exact lengths (Metal does not).
        let p = b.pool_mut();
        let inv_freq = p.alloc(s.rotary_dim / 2 * 4, "inv_freq", true).unwrap();
        let k_cache = p.alloc(N_TOTAL * s.kv_dim() * 4, "k_cache", true).unwrap();
        let v_cache = p.alloc(N_TOTAL * s.kv_dim() * 4, "v_cache", true).unwrap();
        p.upload(inv_freq, &f32_bytes(&s.inv_freq())).unwrap();
        let m = &f.dense;
        let mut indexer = QsaIndexerCache::new(s, m.idx_k_norm.clone());
        let mut out = vec![0.0f32; N_TOTAL * s.hidden];
        let mut sparse_chunks = 0;
        let mut t0 = 0;
        for &n in chunks {
            let p = b.pool_mut();
            macro_rules! alloc {
                ($n:expr, $l:expr) => {
                    p.alloc($n * 4, $l, false).unwrap()
                };
            }
            let bufs = QsaGraphBufs {
                input: alloc!(n * s.hidden, "input"),
                q_proj: alloc!(n * s.q_proj_dim(), "q_proj"),
                k: alloc!(n * s.kv_dim(), "k"),
                v: alloc!(n * s.kv_dim(), "v"),
                idx_qk: alloc!(n * s.idx_proj_dim(), "idx_qk"),
                q: alloc!(n * s.q_dim(), "q"),
                gate: alloc!(n * s.q_dim(), "gate"),
                attn_out: alloc!(n * s.q_dim(), "attn_out"),
                o_out: alloc!(n * s.hidden, "o_out"),
                inv_freq,
                positions: if mrope { Some(p.alloc(n * 12, "positions", false).unwrap()) } else { None },
                k_cache,
                v_cache,
            };
            p.upload(bufs.input, &f32_bytes(&m.x[t0 * s.hidden..(t0 + n) * s.hidden])).unwrap();
            if let Some(pb) = bufs.positions {
                let flat: Vec<u8> = pos[t0..t0 + n].iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
                p.upload(pb, &flat).unwrap();
            }
            let mut g = Graph::new();
            push_qsa_pre_sdpa(&mut g, s, &f.w, &bufs, n, t0, pos[t0][0]);
            b.execute(&g, "qsa.pre_sdpa").unwrap();

            let mut idx = vec![0u8; n * s.idx_proj_dim() * 4];
            b.pool().download(bufs.idx_qk, &mut idx).unwrap();
            let q_idx = qsa_indexer_step(s, &bytes_f32(&idx), &pos[t0..t0 + n], &m.idx_q_norm, &mut indexer);

            let mut g = Graph::new();
            if qsa_needs_sparse(s, mode, t0 + n) {
                sparse_chunks += 1;
                let q = download(b, bufs.q, n * s.q_dim());
                let gate = download(b, bufs.gate, n * s.q_dim());
                let kc = download(b, bufs.k_cache, (t0 + n) * s.kv_dim());
                let vc = download(b, bufs.v_cache, (t0 + n) * s.kv_dim());
                let mut attn = vec![0.0f32; n * s.q_dim()];
                qsa_attend_rows(s, mode, None, &q, &gate, &kc, &vc, t0, &q_idx, &indexer, &mut attn);
                b.pool_mut().upload(bufs.attn_out, &f32_bytes(&attn)).unwrap();
            } else {
                push_qsa_sdpa(&mut g, s, &bufs, n, t0);
            }
            push_qsa_o_proj(&mut g, s, &f.w, &bufs, n);
            b.execute(&g, "qsa.post").unwrap();
            let mut o = vec![0u8; n * s.hidden * 4];
            b.pool().download(bufs.o_out, &mut o).unwrap();
            out[t0 * s.hidden..(t0 + n) * s.hidden].copy_from_slice(&bytes_f32(&o));
            t0 += n;
            b.pool_mut().reset_transient();
        }
        (out, sparse_chunks)
    }

    fn download<B: Backend, T: crate::riir::backend::buftype::Buf>(b: &B, id: BufId<T>, len: usize) -> Vec<f32> {
        let mut v = vec![0u8; len * 4];
        b.pool().download(id, &mut v).unwrap();
        bytes_f32(&v)
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

    /// The graph path on both backends, dense and sparse, prefill and
    /// decode, text and M-RoPE positions, against [`qsa_forward_cpu`]
    /// on the dequantized weights.
    #[test]
    fn graph_cpu_and_metal_match_cpu_path() {
        let s = TINY;
        let fx = Fixture::build(&s, "graph");
        let cpu_wf = fx.open();
        let mut cpu = CpuBackend::new(cpu_wf);
        let gpu_wf = fx.open();
        let metal = MetalContext::new().expect("MetalContext::new");
        let wf_buf = MtlWeightBuf::wrap(&gpu_wf, metal.device());
        let mut gpu = MetalBackend::new(metal, wf_buf).expect("MetalBackend::new");
        let mut kinds = vec!["text"];
        if crate::riir::variants::MROPE_SECTION == Some(s.mrope_section) {
            kinds.push("mrope");
        } else {
            eprintln!("[qsa] variants::MROPE_SECTION is not [11, 11, 10]: M-RoPE graph case skipped");
        }
        for kind in kinds {
            let pos = positions(kind);
            for mode in [QsaMode::Dense, QsaMode::Sparse] {
                for split in ["prefill", "decode"] {
                    let ch = chunks(split);
                    let want = run_cpu(&s, &fx.dense, mode, &pos, &ch);
                    let mrope = kind == "mrope";
                    let (c, cs) = run_graph(&mut cpu, &s, &fx, mode, &pos, mrope, &ch);
                    let (g, gs) = run_graph(&mut gpu, &s, &fx, mode, &pos, mrope, &ch);
                    let scale = max_abs(&want);
                    let (dc, dg) = (max_diff(&c, &want), max_diff(&g, &want));
                    let (cc, cg) = (cosine(&c, &want), cosine(&g, &want));
                    eprintln!(
                        "[qsa graph] {kind} {mode:?} {split}: sparse chunks {cs}/{gs}; \
                         CpuBackend max|diff| {dc:.3e} cos {cc:.9}; Metal max|diff| {dg:.3e} cos {cg:.9} (max|ref| {scale:.3e})"
                    );
                    let want_sparse = if mode == QsaMode::Sparse { if split == "prefill" { 1 } else { N_TOTAL - 19 } } else { 0 };
                    assert_eq!((cs, gs), (want_sparse, want_sparse));
                    assert!(dc <= 2e-5 * scale && cc >= 0.99999, "CpuBackend graph vs CPU path");
                    assert!(dg <= 2e-5 * scale && cg >= 0.99999, "Metal graph vs CPU path");
                }
            }
        }
        drop(gpu);
        drop(gpu_wf);
    }
}

/// [`QsaBlock`] (q38-core's `AttnBlock` entry): token by token from raw
/// zero-centred gains, through a `Linear`-shaped matvec, against the
/// reference; two QSA layers keep separate caches; `reset` and the
/// position check.
#[test]
fn qsa_block_token_by_token_matches_reference() {
    struct Dense {
        w: Vec<f32>,
        rows: usize,
        cols: usize,
    }
    impl QsaLinear for Dense {
        fn in_dim(&self) -> usize {
            self.cols
        }
        fn out_dim(&self) -> usize {
            self.rows
        }
        fn forward(&self, x: &[f32], out: &mut [f32]) {
            matvec_f32(&self.w, x, out)
        }
    }
    let s = TINY;
    let m = Model::reference(&s);
    let lin = |w: &Vec<f32>, rows: usize, cols: usize| Dense { w: w.clone(), rows, cols };
    let raw = |w: &[f32]| w.iter().map(|v| v - 1.0).collect::<Vec<f32>>();
    let layer = || QsaLayer {
        q_proj: lin(&m.q_proj, s.q_proj_dim(), s.hidden),
        k_proj: lin(&m.k_proj, s.kv_dim(), s.hidden),
        v_proj: lin(&m.v_proj, s.kv_dim(), s.hidden),
        o_proj: lin(&m.o_proj, s.hidden, s.q_dim()),
        idx_qk_proj: lin(&m.idx_qk_proj, s.idx_proj_dim(), s.hidden),
        q_norm: raw(&m.q_norm),
        k_norm: raw(&m.k_norm),
        idx_q_norm: raw(&m.idx_q_norm),
        idx_k_norm: raw(&m.idx_k_norm),
    };
    let mut block = QsaBlock::new(&s, QsaMode::Sparse, vec![None, None, None, Some(layer()), None, None, None, Some(layer())]).unwrap();
    let want = fixture_case("text.indexed.prefill");
    for round in 0..2 {
        let mut got = vec![0.0f32; N_TOTAL * s.hidden];
        for t in 0..N_TOTAL {
            let x = &m.x[t * s.hidden..(t + 1) * s.hidden];
            for layer in [3, 7] {
                let o = &mut got[t * s.hidden..(t + 1) * s.hidden];
                block.forward_checked(layer, t, x, o).unwrap();
            }
        }
        let d = max_diff(&got, &want);
        eprintln!("[qsa block] round {round}: max|diff| {d:.3e}");
        assert!(d <= 1e-5 * max_abs(&want));
        assert!(matches!(block.forward_checked(3, 5, &m.x[..s.hidden], &mut vec![0.0; s.hidden]), Err(QsaError::Position { .. })));
        assert!(matches!(block.forward(2, 0, &m.x[..s.hidden], &mut vec![0.0; s.hidden]), Err(QsaError::NoLayer(2))));
        block.reset();
        assert_eq!(block.cache(7).unwrap().len(), 0);
    }
}

/// `AttnBlock::forward_n` on [`QsaBlock`] (chunks of a prefill, some
/// crossing the 19-token dense limit of the tiny budget) is
/// bit-identical to `AttnBlock::forward` token by token, in both
/// modes, through q38-core's `Linear` (the blanket `QsaLinear`).
#[test]
fn qsa_block_forward_n_bit_identical_to_token_by_token() {
    use super::super::gated_residual::DenseLinear;
    let s = TINY;
    let m = Model::reference(&s);
    let lin = |w: &Vec<f32>, rows: usize, cols: usize| DenseLinear::new(rows, cols, w.clone());
    let raw = |w: &[f32]| w.iter().map(|v| v - 1.0).collect::<Vec<f32>>();
    let layer = || QsaLayer {
        q_proj: lin(&m.q_proj, s.q_proj_dim(), s.hidden),
        k_proj: lin(&m.k_proj, s.kv_dim(), s.hidden),
        v_proj: lin(&m.v_proj, s.kv_dim(), s.hidden),
        o_proj: lin(&m.o_proj, s.hidden, s.q_dim()),
        idx_qk_proj: lin(&m.idx_qk_proj, s.idx_proj_dim(), s.hidden),
        q_norm: raw(&m.q_norm),
        k_norm: raw(&m.k_norm),
        idx_q_norm: raw(&m.idx_q_norm),
        idx_k_norm: raw(&m.idx_k_norm),
    };
    let h = s.hidden;
    for mode in [QsaMode::Sparse, QsaMode::Dense] {
        let mk = || QsaBlock::new(&s, mode, vec![None, None, None, Some(layer())]).unwrap();
        let (mut one, mut many) = (mk(), mk());
        let mut want = vec![0.0f32; N_TOTAL * h];
        for t in 0..N_TOTAL {
            AttnBlock::forward(&mut one, 3, t, &m.x[t * h..(t + 1) * h], &mut want[t * h..(t + 1) * h]).unwrap();
        }
        let mut got = vec![0.0f32; N_TOTAL * h];
        let mut t0 = 0;
        for n in [5usize, 9, 1, 25] {
            let (a, b) = (t0 * h, (t0 + n) * h);
            AttnBlock::forward_n(&mut many, 3, t0, n, &m.x[a..b], &mut got[a..b]).unwrap();
            t0 += n;
        }
        assert_eq!(t0, N_TOTAL);
        assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{mode:?}: chunked != token by token");
        // Wrong start position is refused.
        assert!(AttnBlock::forward_n(&mut many, 3, 3, 1, &m.x[..h], &mut vec![0.0; h]).is_err());
    }
}
