//! RMS normalization — Rust port of `infer.m`'s `cpu_rms_norm`.
//!
//! `out[i] = x[i] / sqrt(mean(x*x) + eps) * bf16_to_f32(w[i])`.
//!
//! ## Bit-exact contract
//!
//! Bit-equal against the C path on the same hardware: same
//! sequential accumulation order in the sum-of-squares loop, same
//! `f32` arithmetic for `sqrt` / division, same bf16-as-shift weight
//! decode. The diff oracle asserts byte-equal output.
//!
//! Floating-point determinism note: this hold for the CPU path on
//! Apple silicon's `expf`/`sqrtf` (libm). Once the Metal `rms_norm_*`
//! kernels land in a future slice, bit-exactness moves to a per-
//! kernel loose-tolerance comparison (cosine / Jaccard floors), since
//! Metal's `fast_math` and reduction order do diverge from the CPU
//! reference.

use crate::riir::io::embedding::bf16_to_f32;
use crate::riir::io::weight_file::WeightFile;
use crate::riir::variants::{RMS_NORM_EPS, VARIANT};

/// Errors specific to the RMSNorm port.
#[derive(Debug, thiserror::Error)]
pub enum RmsNormError {
    #[error("RMSNorm weight tensor '{name}' missing from manifest")]
    MissingTensor { name: String },
    #[error("RMSNorm weight '{name}' size {got} bytes, expected {expected} ({dim} bf16 elements)")]
    WeightSize {
        name: String,
        got: u64,
        expected: u64,
        dim: usize,
    },
    #[error("input length {got} != expected {expected}")]
    InputLen { got: usize, expected: usize },
    #[error("output length {got} != expected {expected}")]
    OutputLen { got: usize, expected: usize },
    #[error("non-positive shape: num_heads={num_heads}, head_dim={head_dim}")]
    NonPositiveShape { num_heads: usize, head_dim: usize },
}

/// Apply RMSNorm with the bf16 weight tensor named `weight_name`.
/// `x` and `out` must both be `HIDDEN_DIM` long; they may not alias.
pub fn rms_norm_cpu(
    wf: &WeightFile,
    weight_name: &str,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), RmsNormError> {
    let dim = VARIANT.hidden_dim;
    if x.len() != dim {
        return Err(RmsNormError::InputLen {
            got: x.len(),
            expected: dim,
        });
    }
    if out.len() != dim {
        return Err(RmsNormError::OutputLen {
            got: out.len(),
            expected: dim,
        });
    }

    let bytes = wf
        .tensor_bytes(weight_name)
        .ok_or_else(|| RmsNormError::MissingTensor {
            name: weight_name.to_string(),
        })?;
    let expected_bytes = (dim * 2) as u64;
    if bytes.len() as u64 != expected_bytes {
        return Err(RmsNormError::WeightSize {
            name: weight_name.to_string(),
            got: bytes.len() as u64,
            expected: expected_bytes,
            dim,
        });
    }

    let mut sum_sq: f32 = 0.0;
    for &xi in x.iter() {
        sum_sq += xi * xi;
    }
    let rms = (sum_sq / dim as f32 + RMS_NORM_EPS).sqrt();
    let inv_rms = 1.0f32 / rms;

    for i in 0..dim {
        let w_bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        let weight = bf16_to_f32(w_bits);
        out[i] = x[i] * inv_rms * weight;
    }

    Ok(())
}

/// Per-head CPU RMSNorm, mutating in place. `x_inout` is
/// `num_heads * head_dim` floats laid out contiguously per head; each
/// head's `head_dim`-long slice is independently RMS-normalized and
/// scaled by the same shared bf16 weight tensor of length `head_dim`
/// loaded from `weight_name`. Bit-exact against
/// `mf_rms_norm_per_head_cpu` on the same hardware.
pub fn rms_norm_per_head_cpu(
    wf: &WeightFile,
    weight_name: &str,
    num_heads: usize,
    head_dim: usize,
    x_inout: &mut [f32],
) -> Result<(), RmsNormError> {
    if num_heads == 0 || head_dim == 0 {
        return Err(RmsNormError::NonPositiveShape {
            num_heads,
            head_dim,
        });
    }
    let expected_len = num_heads * head_dim;
    if x_inout.len() != expected_len {
        return Err(RmsNormError::InputLen {
            got: x_inout.len(),
            expected: expected_len,
        });
    }

    let bytes = wf
        .tensor_bytes(weight_name)
        .ok_or_else(|| RmsNormError::MissingTensor {
            name: weight_name.to_string(),
        })?;
    let expected_bytes = (head_dim * 2) as u64;
    if bytes.len() as u64 != expected_bytes {
        return Err(RmsNormError::WeightSize {
            name: weight_name.to_string(),
            got: bytes.len() as u64,
            expected: expected_bytes,
            dim: head_dim,
        });
    }

    rms_norm_heads_bf16(bytes, head_dim, x_inout);
    Ok(())
}

/// In-place RMSNorm of each `head_dim`-wide chunk of `x_inout` with
/// one shared bf16 weight row (`w_bytes`, `head_dim * 2` bytes). A
/// single chunk spanning the whole slice is the full-projection
/// norm ([`crate::riir::variants::QkNorm::Full`]).
fn rms_norm_heads_bf16(w_bytes: &[u8], head_dim: usize, x_inout: &mut [f32]) {
    let head_dim_f = head_dim as f32;
    for xh in x_inout.chunks_exact_mut(head_dim) {
        let mut sum_sq: f32 = 0.0;
        for &xi in xh.iter() {
            sum_sq += xi * xi;
        }
        let inv_rms = 1.0f32 / (sum_sq / head_dim_f + RMS_NORM_EPS).sqrt();
        for i in 0..head_dim {
            let w_bits = u16::from_le_bytes([w_bytes[i * 2], w_bytes[i * 2 + 1]]);
            let weight = bf16_to_f32(w_bits);
            xh[i] = xh[i] * inv_rms * weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-only sanity. Real bit-exact diff lives in the harness.
    #[cfg(not(feature = "model-olmoe-1b-7b"))]
    #[test]
    fn rms_norm_eps_is_1e6() {
        assert!((RMS_NORM_EPS - 1e-6).abs() < 1e-12);
    }

    #[cfg(feature = "model-olmoe-1b-7b")]
    #[test]
    fn rms_norm_eps_is_1e5() {
        assert!((RMS_NORM_EPS - 1e-5).abs() < 1e-12);
    }

    fn bf16_bytes(w: &[f32]) -> Vec<u8> {
        w.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// One chunk over the whole projection (QkNorm::Full) normalizes
    /// by the RMS of all channels, unlike the per-head split.
    #[test]
    fn full_dim_norm_uses_one_rms_over_projection() {
        let n = 8;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 3.0) * 0.5).collect();
        let w: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.25).collect();
        let wb = bf16_bytes(&w);

        let mut full = x.clone();
        rms_norm_heads_bf16(&wb, n, &mut full);
        let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv = 1.0 / (ms + RMS_NORM_EPS).sqrt();
        for i in 0..n {
            assert!((full[i] - x[i] * inv * w[i]).abs() < 1e-6, "i={i}");
        }

        // Per-head (2 heads of 4) re-uses the first 4 weights per head
        // and its own RMS per head — differs from the full norm.
        let mut per_head = x.clone();
        rms_norm_heads_bf16(&wb[..8], 4, &mut per_head);
        for h in 0..2 {
            let xh = &x[h * 4..h * 4 + 4];
            let ms = xh.iter().map(|v| v * v).sum::<f32>() / 4.0;
            let inv = 1.0 / (ms + RMS_NORM_EPS).sqrt();
            for i in 0..4 {
                assert!((per_head[h * 4 + i] - xh[i] * inv * w[i]).abs() < 1e-6);
            }
        }
        assert_ne!(full, per_head);
    }
}
