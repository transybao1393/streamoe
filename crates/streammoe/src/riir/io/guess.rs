//! Cross-layer expert guesses for speculative prefetch (plan phase 6,
//! "Strengthening streaming" item 3).
//!
//! The default guess for layer L+1 is the previous token's picks for
//! L+1 ([`super::prefetch::PrefetchState::predict_for`]), which flash-
//! moe measured at ~25% hits. The residual stream changes slowly from
//! layer to layer, so layer L's post-attention residual (`h_mid`) run
//! through layer L+1's post-attention norm and router gate is a much
//! closer stand-in for L+1's actual router input. [`CrossLayerGuesser`]
//! does exactly that on the CPU, after layer L's routing and before
//! L+1's prefetch is issued:
//!
//! 1. [`CrossLayerGuesser::snapshot`] copies `h_mid` of layer L into a
//!    fixed host buffer (before L's expert dispatch can touch it).
//! 2. [`CrossLayerGuesser::guess`] computes `rms_norm(h) ⊙ w_norm` of
//!    L+1, the gate logits `W_gate · x`, and their top-k.
//!
//! All buffers are allocated once at open (a few tens of KB, inside the
//! runtime-overhead account); a guess allocates nothing and runs
//! serially on the calling thread, so it adds no threads. The guess
//! only chooses which experts to read early — the forward pass always
//! routes on its own logits — so it can never change the math.

use crate::riir::io::embedding::bf16_to_f32;
use crate::riir::io::weight_file::WeightFile;
use crate::riir::moe::expert_forward::MAX_K;
use crate::riir::variants::{GROUP_SIZE, RMS_NORM_EPS, VARIANT};

/// Byte ranges in the weight file for one layer's router gate and the
/// post-attention norm that feeds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateRef {
    /// `post_attention_layernorm.weight` (bf16, `hidden_dim`).
    pub norm: (u64, usize),
    /// `mlp.gate.weight` (u32-packed, `[num_experts, hidden_dim * bits / 32]`).
    pub weight: (u64, usize),
    /// `mlp.gate.scales` (bf16, `[num_experts, hidden_dim / GROUP_SIZE]`).
    pub scales: (u64, usize),
    /// `mlp.gate.biases` (bf16, same shape as scales).
    pub biases: (u64, usize),
    /// Quantization bits of the gate (4 or 8).
    pub bits: u32,
}

impl GateRef {
    /// Resolve layer `layer`'s tensors. `None` when a tensor is
    /// missing, the gate is not 4/8-bit quantized, or the sizes do not
    /// match the variant (e.g. a bf16 noaux router on MLA variants).
    pub fn resolve(wf: &WeightFile, layer: usize) -> Option<Self> {
        let v = VARIANT;
        let info = |suffix: &str| {
            wf.tensor_info(&format!("model.layers.{layer}.{suffix}"))
                .map(|t| (t.offset, t.size as usize, t.bits))
        };
        let (no, ns, _) = info("post_attention_layernorm.weight")?;
        let (wo, ws, bits) = info("mlp.gate.weight")?;
        let (so, ss, _) = info("mlp.gate.scales")?;
        let (bo, bs, _) = info("mlp.gate.biases")?;
        // Same rule as the forward pass's `bits_of`: unset means 4.
        let bits = (bits as u32).max(4);
        if bits != 4 && bits != 8 {
            return None;
        }
        let groups = v.hidden_dim / GROUP_SIZE;
        let ok = ns == v.hidden_dim * 2
            && ws == v.num_experts * v.hidden_dim * bits as usize / 8
            && ss == v.num_experts * groups * 2
            && bs == ss;
        ok.then_some(Self {
            norm: (no, ns),
            weight: (wo, ws),
            scales: (so, ss),
            biases: (bo, bs),
            bits,
        })
    }
}

/// `out = rms_norm(x) ⊙ w` with a bf16 weight given as little-endian
/// bytes. Same formula as [`crate::riir::attn::rms_norm::rms_norm_cpu`].
pub fn rms_norm_bf16(x: &[f32], w: &[u8], eps: f32, out: &mut [f32]) {
    let dim = x.len();
    let mut sum_sq = 0.0f32;
    for &xi in x {
        sum_sq += xi * xi;
    }
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        let wb = bf16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]]));
        out[i] = x[i] * inv_rms * wb;
    }
}

/// Serial `out[r] = Σ_c dequant(W[r, c]) · x[c]` for a `bits`-bit
/// (4 or 8) MLX-packed matrix: `packed` holds `32 / bits` values per
/// u32, low bits first; `scales` / `biases` are bf16 per
/// [`GROUP_SIZE`] inputs. Matches `dequant_matvec_4bit_cpu` /
/// `dequant_matvec_8bit_v3_cpu` without the thread pool.
pub fn gate_logits(
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    bits: u32,
    x: &[f32],
    out: &mut [f32],
) {
    let in_dim = x.len();
    let per_word = (32 / bits) as usize;
    let mask = (1u32 << bits) - 1;
    let in_packed = in_dim / per_word;
    let in_groups = in_dim / GROUP_SIZE;
    let words_per_group = GROUP_SIZE / per_word;
    for (r, o) in out.iter_mut().enumerate() {
        let row = &packed[r * in_packed..(r + 1) * in_packed];
        let mut acc = 0.0f32;
        for g in 0..in_groups {
            let scale = bf16_to_f32(scales[r * in_groups + g]);
            let bias = bf16_to_f32(biases[r * in_groups + g]);
            for p in 0..words_per_group {
                let word = row[g * words_per_group + p];
                let xs = &x[g * GROUP_SIZE + p * per_word..][..per_word];
                for (n, &xv) in xs.iter().enumerate() {
                    let q = ((word >> (n as u32 * bits)) & mask) as f32;
                    acc = q.mul_add(scale, bias).mul_add(xv, acc);
                }
            }
        }
        *o = acc;
    }
}

/// The `k` largest entries of `scores` (strict `>`, lowest index wins
/// ties) into `out[..k]`, in descending score order; `out[k..]` is
/// set to `-1`.
pub fn top_k_into(scores: &[f32], k: usize, out: &mut [i32; MAX_K]) {
    let k = k.min(MAX_K).min(scores.len());
    let mut best = [f32::NEG_INFINITY; MAX_K];
    *out = [-1; MAX_K];
    let mut n = 0;
    for (i, &s) in scores.iter().enumerate() {
        if k == 0 {
            break;
        }
        if n == k && !(s > best[k - 1]) {
            continue;
        }
        // Insert, keeping best[..n] sorted descending.
        let mut j = if n < k {
            n += 1;
            n - 1
        } else {
            k - 1
        };
        while j > 0 && s > best[j - 1] {
            best[j] = best[j - 1];
            out[j] = out[j - 1];
            j -= 1;
        }
        best[j] = s;
        out[j] = i as i32;
    }
}

/// Fixed-buffer cross-layer guesser. See the module docs.
#[derive(Debug)]
pub struct CrossLayerGuesser {
    /// Per model layer; `None` where the gate can't be guessed from.
    gates: Vec<Option<GateRef>>,
    h: Vec<f32>,
    normed: Vec<f32>,
    scores: Vec<f32>,
    /// Layer whose residual is in `h`.
    from_layer: Option<usize>,
    /// Last computed guess and the layer it is for.
    guess: Option<(usize, [i32; MAX_K])>,
}

impl CrossLayerGuesser {
    /// Resolve the gates of every layer in `range` and allocate the
    /// scratch.
    pub fn new(wf: &WeightFile, range: std::ops::Range<usize>) -> Self {
        let v = VARIANT;
        let gates = (0..v.num_layers)
            .map(|l| if range.contains(&l) { GateRef::resolve(wf, l) } else { None })
            .collect();
        Self {
            gates,
            h: vec![0.0; v.hidden_dim],
            normed: vec![0.0; v.hidden_dim],
            scores: vec![0.0; v.num_experts],
            from_layer: None,
            guess: None,
        }
    }

    /// `true` if layer `layer` has a gate to guess from.
    pub fn can_guess(&self, layer: usize) -> bool {
        self.gates.get(layer).is_some_and(Option::is_some)
    }

    /// Keep layer `layer`'s post-attention residual for the next
    /// layer's guess. `h_mid` must be `hidden_dim` long.
    pub fn snapshot(&mut self, layer: usize, h_mid: &[f32]) {
        self.h.copy_from_slice(h_mid);
        self.from_layer = Some(layer);
    }

    /// If the snapshot is from `layer - 1`, compute the top-`k` guess
    /// for `layer`, store it and return it. Consumes the snapshot.
    pub fn guess(&mut self, wf: &WeightFile, layer: usize, k: usize) -> Option<[i32; MAX_K]> {
        let from = self.from_layer.take()?;
        if from + 1 != layer {
            return None;
        }
        let g = (*self.gates.get(layer)?)?;
        let bytes = |(off, len): (u64, usize)| wf.bytes_at(off, len);
        let packed: &[u32] = bytemuck::try_cast_slice(bytes(g.weight)?).ok()?;
        let scales: &[u16] = bytemuck::try_cast_slice(bytes(g.scales)?).ok()?;
        let biases: &[u16] = bytemuck::try_cast_slice(bytes(g.biases)?).ok()?;
        rms_norm_bf16(&self.h, bytes(g.norm)?, RMS_NORM_EPS, &mut self.normed);
        gate_logits(packed, scales, biases, g.bits, &self.normed, &mut self.scores);
        let mut out = [-1i32; MAX_K];
        top_k_into(&self.scores, k, &mut out);
        self.guess = Some((layer, out));
        Some(out)
    }

    /// Take the stored guess if it is for `layer` (scoring it against
    /// the actual routing). Clears it either way.
    pub fn take_guess_for(&mut self, layer: usize) -> Option<[i32; MAX_K]> {
        match self.guess.take() {
            Some((l, g)) if l == layer => Some(g),
            _ => None,
        }
    }

    /// Forget the snapshot and any stored guess (request switch,
    /// memory clear).
    pub fn clear(&mut self) {
        self.from_layer = None;
        self.guess = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::riir::backend::cpu::cpu_matvec::{dequant_matvec_4bit_cpu, dequant_matvec_8bit_v3_cpu};

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn bf16(x: f32) -> u16 {
        (x.to_bits() >> 16) as u16
    }

    #[test]
    fn gate_logits_matches_threaded_kernels() {
        let (in_dim, out_dim) = (4 * GROUP_SIZE, 5);
        let mut seed = 7u64;
        let x: Vec<f32> = (0..in_dim).map(|_| (lcg(&mut seed) % 200) as f32 / 100.0 - 1.0).collect();
        let groups = in_dim / GROUP_SIZE;
        let scales: Vec<u16> = (0..out_dim * groups).map(|i| bf16(0.01 + 0.001 * i as f32)).collect();
        let biases: Vec<u16> = (0..out_dim * groups).map(|i| bf16(-0.05 + 0.002 * i as f32)).collect();
        for bits in [4u32, 8] {
            let words = out_dim * in_dim * bits as usize / 32;
            let packed: Vec<u32> = (0..words).map(|_| lcg(&mut seed)).collect();
            let mut want = vec![0f32; out_dim];
            if bits == 4 {
                dequant_matvec_4bit_cpu(&packed, &scales, &biases, in_dim, out_dim, &x, &mut want)
                    .unwrap();
            } else {
                dequant_matvec_8bit_v3_cpu(&packed, &scales, &biases, in_dim, out_dim, &x, &mut want)
                    .unwrap();
            }
            let mut got = vec![0f32; out_dim];
            gate_logits(&packed, &scales, &biases, bits, &x, &mut got);
            assert_eq!(got, want, "bits={bits}");
        }
    }

    #[test]
    fn top_k_picks_largest_in_order() {
        let scores = [0.1, 0.9, -1.0, 0.5, 0.9, 0.3];
        let mut out = [0i32; MAX_K];
        top_k_into(&scores, 3, &mut out);
        assert_eq!(&out[..3], &[1, 4, 3]);
        assert!(out[3..].iter().all(|&e| e == -1));
        top_k_into(&[f32::NEG_INFINITY, -5.0], 2, &mut out);
        assert_eq!(&out[..2], &[1, 0]);
        top_k_into(&scores, 0, &mut out);
        assert!(out.iter().all(|&e| e == -1));
    }

    #[test]
    fn rms_norm_bf16_matches_formula() {
        let x = [3.0f32, 4.0];
        let w: Vec<u8> = [bf16(1.0), bf16(2.0)].iter().flat_map(|b| b.to_le_bytes()).collect();
        let mut out = [0f32; 2];
        rms_norm_bf16(&x, &w, 0.0, &mut out);
        let inv = 1.0 / (12.5f32).sqrt();
        assert_eq!(out, [3.0 * inv * 1.0, 4.0 * inv * 2.0]);
    }
}
