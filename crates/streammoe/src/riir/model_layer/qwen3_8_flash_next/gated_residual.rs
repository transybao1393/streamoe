//! Gated Residual (hyper-connections), CPU.
//!
//! Port of `Qwen4ExpTextGatedResidual` and the stream handling around
//! it in `Qwen4ExpTextDecoderLayer` / `Qwen4ExpTextModel`:
//!
//! ```text
//! hyper   = repeat(embed, hc_count)                 // [hc · H], model input
//! normed  = hc_norm(hyper)                          // RMS per copy (group H), gain 1 + w
//! mix     = sigmoid(up(silu(down(normed) / hc)))    // [hc · H]
//! x       = mean_c(mix[c] ⊙ normed[c])              // [H], block input
//! inj     = 2 · sigmoid(block_inject(normed) / hc)  // [hc]
//! y       = block(x)                                // attention / MoE
//! hyper   = hyper + y ⊗ inj                         // hyper[c] += inj[c] · y
//! ...
//! out     = mean_c(mix ⊙ normed) of the final mixer // no inject, no final norm
//! logits  = lm_head(out)
//! ```
//!
//! `hc_norm` is `Qwen4ExpTextRMSNorm(hc · H, group_size = H)`: the
//! 10240-wide input is reshaped to `[hc, H]` and each copy gets its own
//! RMS over its H values; the 10240 gains apply elementwise, as
//! `1 + w` ([`NormGain::OnePlus`]). The residual added back is the
//! *un-normalized* `hyper`.

/// A linear map `out = W · x` (no bias). Implemented by dense f32
/// weights for the tiny tests ([`DenseLinear`]) and, in the engine, by
/// the quantized matvecs (4-bit group 32, router 8-bit group 64).
pub trait Linear {
    fn in_dim(&self) -> usize;
    fn out_dim(&self) -> usize;
    /// `out[r] = Σ_c W[r, c] · x[c]`; `x.len() == in_dim`,
    /// `out.len() == out_dim`.
    fn forward(&self, x: &[f32], out: &mut [f32]);
}

/// Row-major `[out_dim, in_dim]` f32 weights — torch `nn.Linear`
/// layout.
#[derive(Debug, Clone)]
pub struct DenseLinear {
    pub out_dim: usize,
    pub in_dim: usize,
    pub w: Vec<f32>,
}

impl DenseLinear {
    pub fn new(out_dim: usize, in_dim: usize, w: Vec<f32>) -> Self {
        assert_eq!(w.len(), out_dim * in_dim, "DenseLinear: weight length");
        Self { out_dim, in_dim, w }
    }
}

impl Linear for DenseLinear {
    fn in_dim(&self) -> usize {
        self.in_dim
    }
    fn out_dim(&self) -> usize {
        self.out_dim
    }
    fn forward(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert_eq!(out.len(), self.out_dim);
        for (r, o) in out.iter_mut().enumerate() {
            let row = &self.w[r * self.in_dim..(r + 1) * self.in_dim];
            let mut acc = 0.0f32;
            for (w, xv) in row.iter().zip(x) {
                acc = w.mul_add(*xv, acc);
            }
            *o = acc;
        }
    }
}

/// How a norm's stored gain applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormGain {
    /// `x̂ · (1 + w)` — `Qwen4ExpTextRMSNorm`, gains stored centered
    /// at zero (the checkpoint and the MLX build both store them so).
    OnePlus,
    /// `x̂ · w` — `Qwen4ExpTextRMSNormGated` and the engine's other
    /// norms.
    Plain,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Grouped RMSNorm: for each `group`-wide slice of `x`,
/// `out = x / sqrt(mean(x²) + eps) · gain(w)`. `w.len() == x.len()`.
/// `group == x.len()` is a plain RMSNorm.
pub fn rms_norm_grouped(x: &[f32], w: &[f32], group: usize, eps: f32, gain: NormGain, out: &mut [f32]) {
    assert!(group > 0 && x.len() % group == 0, "rms_norm_grouped: {} not a multiple of {group}", x.len());
    assert_eq!(w.len(), x.len());
    assert_eq!(out.len(), x.len());
    for g in 0..x.len() / group {
        let xs = &x[g * group..(g + 1) * group];
        let mut sum_sq = 0.0f32;
        for &v in xs {
            sum_sq += v * v;
        }
        let inv = 1.0 / (sum_sq / group as f32 + eps).sqrt();
        for i in 0..group {
            let j = g * group + i;
            let gain_v = match gain {
                NormGain::OnePlus => 1.0 + w[j],
                NormGain::Plain => w[j],
            };
            out[j] = x[j] * inv * gain_v;
        }
    }
}

/// One Gated Residual's weights. `hc_norm` is the raw (zero-centered)
/// gain, `[hc · H]`.
pub struct GatedResidualWeights<'a, L: Linear> {
    pub hc_norm: &'a [f32],
    /// `input_mix_weight_down`: `[lowrank, hc · H]`.
    pub mix_down: &'a L,
    /// `input_mix_weight_up`: `[hc · H, lowrank]`.
    pub mix_up: &'a L,
    /// `block_inject_weight`: `[hc, hc · H]`; `None` for the final
    /// mixer.
    pub block_inject: Option<&'a L>,
}

/// Scratch for [`gated_residual_mix`].
#[derive(Debug, Clone)]
pub struct GatedResidualScratch {
    pub normed: Vec<f32>,
    pub low: Vec<f32>,
    pub mix: Vec<f32>,
}

impl GatedResidualScratch {
    pub fn new(hc_hidden: usize, lowrank: usize) -> Self {
        Self {
            normed: vec![0.0; hc_hidden],
            low: vec![0.0; lowrank],
            mix: vec![0.0; hc_hidden],
        }
    }
}

/// `Qwen4ExpTextGatedResidual.forward` for one token.
///
/// `hyper` is the residual stream `[hc · H]` (read only); writes the
/// block input `x` (`[H]`) to `mixed`, and — when the weights carry
/// `block_inject` — the injection weights `2 · sigmoid(…/hc)` (`[hc]`)
/// to `inj`. The caller keeps `hyper` and later calls [`inject`].
pub fn gated_residual_mix<L: Linear>(
    w: &GatedResidualWeights<'_, L>,
    hidden: usize,
    hc: usize,
    eps: f32,
    hyper: &[f32],
    scratch: &mut GatedResidualScratch,
    mixed: &mut [f32],
    inj: Option<&mut [f32]>,
) {
    let width = hc * hidden;
    assert_eq!(hyper.len(), width, "gated_residual_mix: hyper width");
    assert_eq!(mixed.len(), hidden);
    assert_eq!(w.mix_down.in_dim(), width);
    assert_eq!(w.mix_up.out_dim(), width);
    let lowrank = w.mix_down.out_dim();
    scratch.normed.resize(width, 0.0);
    scratch.low.resize(lowrank, 0.0);
    scratch.mix.resize(width, 0.0);

    rms_norm_grouped(hyper, w.hc_norm, hidden, eps, NormGain::OnePlus, &mut scratch.normed);
    w.mix_down.forward(&scratch.normed, &mut scratch.low);
    let inv_hc = 1.0 / hc as f32;
    for v in scratch.low.iter_mut() {
        *v = silu(*v * inv_hc);
    }
    w.mix_up.forward(&scratch.low, &mut scratch.mix);
    for j in 0..hidden {
        let mut acc = 0.0f32;
        for c in 0..hc {
            let k = c * hidden + j;
            acc += sigmoid(scratch.mix[k]) * scratch.normed[k];
        }
        mixed[j] = acc * inv_hc;
    }
    if let Some(inj) = inj {
        let bi = w.block_inject.expect("gated_residual_mix: inj requested without block_inject");
        assert_eq!(inj.len(), hc);
        bi.forward(&scratch.normed, inj);
        for v in inj.iter_mut() {
            *v = 2.0 * sigmoid(*v * inv_hc);
        }
    }
}

/// `hyper[c · H + j] += inj[c] · block_out[j]` — the stream update
/// after a block (`hyper_input + (block_out ⊗ injection).flatten()`).
pub fn inject(hyper: &mut [f32], block_out: &[f32], inj: &[f32]) {
    let hidden = block_out.len();
    assert_eq!(hyper.len(), inj.len() * hidden, "inject: widths");
    for (c, &wc) in inj.iter().enumerate() {
        let row = &mut hyper[c * hidden..(c + 1) * hidden];
        for (h, &y) in row.iter_mut().zip(block_out) {
            *h = y.mul_add(wc, *h);
        }
    }
}

/// The model input stream: `hidden.repeat(hc)` → `[hc · H]`.
pub fn hc_expand(hidden: &[f32], hc: usize, hyper: &mut [f32]) {
    let h = hidden.len();
    assert_eq!(hyper.len(), hc * h);
    for c in 0..hc {
        hyper[c * h..(c + 1) * h].copy_from_slice(hidden);
    }
}

#[cfg(test)]
mod tests {
    use super::super::tiny::{gen_tensor, load_fixture, TinyHc};
    use super::*;

    fn assert_close(name: &str, got: &[f32], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{name}: length");
        let mut worst = 0.0f64;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let err = (g as f64 - w).abs() / (1.0 + w.abs());
            assert!(err <= tol, "{name}[{i}]: got {g}, want {w} (rel err {err:.3e})");
            worst = worst.max(err);
        }
        eprintln!("[q38 {name}] {} values, worst rel err {worst:.2e}", got.len());
    }

    /// Against the torch port (`core_ref.py`): `hc_norm`, the mix, the
    /// injection weights, the inject and the final mixer.
    #[test]
    fn gated_residual_matches_reference() {
        let fx = load_fixture();
        let cfg = fx.config.clone();
        let w = cfg.hc_hidden();
        let hcw = TinyHc::generate(&cfg, "gr_case", true);
        let hyper = gen_tensor("gr_case.hyper", w, 1.5, 0.0);
        let block = gen_tensor("gr_case.block", cfg.hidden, 1.0, 0.0);
        let gr = &fx.gated_residual;

        let mut normed = vec![0.0; w];
        rms_norm_grouped(&hyper, &hcw.hc_norm, cfg.hidden, cfg.rms_eps, NormGain::OnePlus, &mut normed);
        assert_close("hc_norm", &normed, &gr.normed, 1e-5);

        let mut scratch = GatedResidualScratch::new(w, cfg.hc_lowrank);
        let mut mixed = vec![0.0; cfg.hidden];
        let mut inj = vec![0.0; cfg.hc_count];
        gated_residual_mix(&hcw.weights(), cfg.hidden, cfg.hc_count, cfg.rms_eps, &hyper, &mut scratch, &mut mixed, Some(&mut inj));
        assert_close("mixed", &mixed, &gr.mixed, 1e-5);
        assert_close("inj", &inj, &gr.inj, 1e-5);

        let mut h = hyper.clone();
        inject(&mut h, &block, &inj);
        assert_close("injected", &h, &gr.injected, 1e-5);

        let mut fin = vec![0.0; cfg.hidden];
        let final_w = GatedResidualWeights { block_inject: None, ..hcw.weights() };
        gated_residual_mix(&final_w, cfg.hidden, cfg.hc_count, cfg.rms_eps, &hyper, &mut scratch, &mut fin, None);
        assert_close("final_mixed", &fin, &gr.final_mixed, 1e-5);
    }

    #[test]
    fn grouped_norm_groups_are_independent() {
        // Scaling one copy must not change the others' output.
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let w = vec![0.1f32; 16];
        let mut a = vec![0.0; 16];
        rms_norm_grouped(&x, &w, 4, 1e-6, NormGain::OnePlus, &mut a);
        let mut x2 = x.clone();
        for v in &mut x2[4..8] {
            *v *= 10.0;
        }
        let mut b = vec![0.0; 16];
        rms_norm_grouped(&x2, &w, 4, 1e-6, NormGain::OnePlus, &mut b);
        for i in (0..4).chain(8..16) {
            assert_eq!(a[i], b[i]);
        }
        for i in 4..8 {
            assert!((a[i] - b[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn expand_then_inject() {
        let h = [1.0f32, 2.0];
        let mut hyper = vec![0.0; 6];
        hc_expand(&h, 3, &mut hyper);
        assert_eq!(hyper, [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        inject(&mut hyper, &[10.0, 20.0], &[0.5, 1.0, 2.0]);
        assert_eq!(hyper, [6.0, 12.0, 11.0, 22.0, 21.0, 42.0]);
    }
}
