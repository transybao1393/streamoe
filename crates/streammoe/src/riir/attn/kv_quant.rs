//! Quantized storage for the GQA KV cache ([`KvFormat`]).
//!
//! The KV cache is `[n_ctx, num_kv_heads * head_dim]` per side and
//! full-attn layer — f32 by default, which under a memory cap is
//! budget the hot tier could use. A [`KvFormat`] stores each row more
//! compactly; every attention path reads it through this module:
//!
//! - **Append** quantizes rows: [`quantize_row`] on the CPU decode
//!   path, the `kv_append_*` Metal kernels on batched prefill.
//! - **Decode SDPA** on the CPU reads the stored rows directly
//!   ([`sdpa_cpu_quant`]), never materializing f32.
//! - **Prefill SDPA** dequantizes the attended prefix into one shared
//!   f32 scratch ([`dequantize_row`], the `kv_dequant_*` kernels), so
//!   the flash kernel is unchanged.
//!
//! Formats:
//!
//! - [`KvFormat::F16`] — IEEE half, round to nearest even. 2×.
//! - [`KvFormat::Turbo4`] / [`KvFormat::Turbo3`] — TurboQuant's MSE
//!   quantizer (Zandieh et al., 2025): each head vector is rotated by
//!   a fixed random orthogonal matrix `R = H·D / √d` (`H` the
//!   Walsh–Hadamard matrix, `D` fixed random signs), which makes its
//!   coordinates near-Gaussian and outlier-free; each coordinate of
//!   `√d · Rx / ‖x‖` is then rounded to the 16 / 8 Lloyd–Max levels
//!   of N(0, 1). A head stores one f32 scale (the least-squares scale
//!   of its codes, `⟨y, c⟩ / ⟨c, c⟩` for rotated `y`) and `d · b`
//!   bits of codes: 68 B (4-bit) / 52 B (3-bit) per 128-dim head vs
//!   512 B f32. Attention runs in the rotated space — `q·k = Rq·Rk` —
//!   so decode rotates `q` once per head and the weighted value sum
//!   once at the end.
//!
//! `head_dim` must be a power of two for the Turbo formats (every GQA
//! variant here: 128 or 256). MLA caches stay f32.

use crate::riir::variants::{AttnKind, Variant};

/// Storage format of the GQA KV cache rows
/// ([`crate::riir::OpenConfig::kv_format`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvFormat {
    /// f32 rows — the historical layout; bit-identical to before.
    #[default]
    F32,
    /// IEEE half rows.
    F16,
    /// TurboQuant MSE, 4 bits per coordinate + an f32 scale per head.
    Turbo4,
    /// TurboQuant MSE, 3 bits per coordinate + an f32 scale per head.
    Turbo3,
}

/// Positive Lloyd–Max levels of N(0, 1), 8 levels (3-bit).
const LEVELS3: [f32; 4] = [0.2450942, 0.7560053, 1.3439093, 2.1519457];
/// Positive Lloyd–Max levels of N(0, 1), 16 levels (4-bit).
const LEVELS4: [f32; 8] = [
    0.1283950, 0.3880483, 0.6567591, 0.9423405, 1.2562312, 1.6180464, 2.0690172, 2.7325896,
];

impl KvFormat {
    /// Stable wire code (snapshots).
    pub fn code(self) -> u32 {
        match self {
            KvFormat::F32 => 0,
            KvFormat::F16 => 1,
            KvFormat::Turbo4 => 2,
            KvFormat::Turbo3 => 3,
        }
    }

    /// Inverse of [`Self::code`].
    pub fn from_code(c: u32) -> Option<Self> {
        Some(match c {
            0 => KvFormat::F32,
            1 => KvFormat::F16,
            2 => KvFormat::Turbo4,
            3 => KvFormat::Turbo3,
            _ => return None,
        })
    }

    /// Bits per coordinate of a Turbo format, `None` otherwise.
    pub fn turbo_bits(self) -> Option<u32> {
        match self {
            KvFormat::Turbo4 => Some(4),
            KvFormat::Turbo3 => Some(3),
            _ => None,
        }
    }

    /// Bytes one head vector of `head_dim` takes.
    pub fn head_bytes(self, head_dim: usize) -> usize {
        match self {
            KvFormat::F32 => head_dim * 4,
            KvFormat::F16 => head_dim * 2,
            KvFormat::Turbo4 => 4 + head_dim / 2,
            KvFormat::Turbo3 => 4 + head_dim * 3 / 8,
        }
    }

    /// Bytes one cache row (one token, all KV heads) takes for `v`.
    pub fn row_bytes(self, v: &Variant) -> usize {
        v.num_kv_heads * self.head_bytes(v.head_dim)
    }

    /// Whether `v` can store its KV in this format.
    pub fn validate(self, v: &Variant) -> Result<(), String> {
        if self == KvFormat::F32 {
            return Ok(());
        }
        if v.attn_kind != AttnKind::Gqa {
            return Err(format!("kv_format {self:?} needs a GQA variant (MLA caches stay f32)"));
        }
        if self.turbo_bits().is_some() && (!v.head_dim.is_power_of_two() || v.head_dim < 8) {
            return Err(format!(
                "kv_format {self:?} needs a power-of-two head_dim ≥ 8 (got {})",
                v.head_dim
            ));
        }
        Ok(())
    }

    fn levels(self) -> &'static [f32] {
        match self {
            KvFormat::Turbo4 => &LEVELS4,
            KvFormat::Turbo3 => &LEVELS3,
            _ => &[],
        }
    }
}

/// Fixed random sign of coordinate `i` (the `D` of `R = H·D / √d`).
/// Same hash in the Metal kernels (`kv_sign`).
#[inline]
pub fn sign(i: usize) -> f32 {
    let mut h = (i as u32).wrapping_mul(0x9E37_79B1);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    if h & 1 == 0 { 1.0 } else { -1.0 }
}

/// In-place unnormalized fast Walsh–Hadamard transform (`len` a power
/// of two). Same butterfly order as the Metal kernels.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// `y = R x = H·D·x / √d`.
pub fn rotate(x: &[f32], y: &mut [f32]) {
    for (i, (yi, &xi)) in y.iter_mut().zip(x).enumerate() {
        *yi = xi * sign(i);
    }
    fwht(y);
    let s = 1.0 / (x.len() as f32).sqrt();
    for yi in y.iter_mut() {
        *yi *= s;
    }
}

/// `x = Rᵀ y = D·H·y / √d`, in place.
pub fn unrotate(y: &mut [f32]) {
    fwht(y);
    let s = 1.0 / (y.len() as f32).sqrt();
    for (i, yi) in y.iter_mut().enumerate() {
        *yi *= s * sign(i);
    }
}

/// f32 → IEEE half bits, round to nearest even (Metal's `half(x)`).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut h = m >> shift;
        if rest > half || (rest == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// [`f16_to_f32`] without branches, so the decode SDPA's conversion
/// loop vectorizes: the magnitude bits shifted into an f32 and scaled
/// by 2^112 give every finite half exactly (subnormals included);
/// inf / NaN are selected separately.
#[inline(always)]
pub fn f16_to_f32_fast(h: u16) -> f32 {
    let m = (h & 0x7fff) as u32;
    let finite = (f32::from_bits(m << 13) * f32::from_bits(0x7780_0000)).to_bits();
    let special = 0x7f80_0000 | ((m & 0x3ff) << 13);
    let mag = if m >= 0x7c00 { special } else { finite };
    f32::from_bits(mag | ((h as u32 & 0x8000) << 16))
}

/// Convert one stored f16 head into `out`. On aarch64, eight halves
/// at a time with the base-ARMv8 `fcvtl` / `fcvtl2` (exact, like
/// [`f16_to_f32`]); elsewhere and for any tail, [`f16_to_f32_fast`].
#[inline]
fn f16_head(rec: &[u8], out: &mut [f32]) {
    let mut done = 0;
    #[cfg(target_arch = "aarch64")]
    {
        let n8 = out.len().min(rec.len() / 2) / 8 * 8;
        for i in (0..n8).step_by(8) {
            // SAFETY: reads 16 bytes at `rec[2i..]` and writes 8 f32 at
            // `out[i..]`, both in bounds (`i + 8 ≤ n8`); `ldr` / `stp`
            // of q registers need no alignment on aarch64.
            unsafe {
                std::arch::asm!(
                    "ldr q0, [{s}]",
                    "fcvtl v1.4s, v0.4h",
                    "fcvtl2 v2.4s, v0.8h",
                    "stp q1, q2, [{d}]",
                    s = in(reg) rec.as_ptr().add(2 * i),
                    d = in(reg) out.as_mut_ptr().add(i),
                    out("v0") _, out("v1") _, out("v2") _,
                    options(nostack),
                );
            }
        }
        done = n8;
    }
    for (o, b) in out[done..].iter_mut().zip(rec[2 * done..].chunks_exact(2)) {
        *o = f16_to_f32_fast(u16::from_le_bytes([b[0], b[1]]));
    }
}

/// IEEE half bits → f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Index of the level nearest `u` among the signed levels
/// `-L[n-1] … -L[0], L[0] … L[n-1]` (code `0..2n`).
#[inline]
fn nearest(levels: &[f32], u: f32) -> u32 {
    let n = levels.len();
    let a = u.abs();
    let mut k = 0;
    while k + 1 < n && a > 0.5 * (levels[k] + levels[k + 1]) {
        k += 1;
    }
    if u < 0.0 { (n - 1 - k) as u32 } else { (n + k) as u32 }
}

#[inline]
fn level(levels: &[f32], code: u32) -> f32 {
    let n = levels.len() as u32;
    if code < n { -levels[(n - 1 - code) as usize] } else { levels[(code - n) as usize] }
}

#[inline]
fn put_code(bytes: &mut [u8], bits: u32, j: usize, c: u32) {
    if bits == 4 {
        bytes[j / 2] |= (c as u8) << ((j % 2) * 4);
    } else {
        let bit = j * 3;
        let w = bit / 8;
        let s = bit % 8;
        bytes[w] |= ((c << s) & 0xff) as u8;
        if s > 5 {
            bytes[w + 1] |= (c >> (8 - s)) as u8;
        }
    }
}

#[inline]
fn get_code(bytes: &[u8], bits: u32, j: usize) -> u32 {
    if bits == 4 {
        ((bytes[j / 2] >> ((j % 2) * 4)) & 0xf) as u32
    } else {
        let bit = j * 3;
        let w = bit / 8;
        let s = bit % 8;
        let mut v = (bytes[w] as u32) >> s;
        if s > 5 {
            v |= (bytes[w + 1] as u32) << (8 - s);
        }
        v & 7
    }
}

/// Quantize one head vector `x` into `out` (`fmt.head_bytes(x.len())`).
/// `scratch` holds `x.len()` floats.
pub fn quantize_head(fmt: KvFormat, x: &[f32], out: &mut [u8], scratch: &mut [f32]) {
    let d = x.len();
    match fmt {
        KvFormat::F32 => {
            for (o, &v) in out.chunks_exact_mut(4).zip(x) {
                o.copy_from_slice(&v.to_le_bytes());
            }
        }
        KvFormat::F16 => {
            for (o, &v) in out.chunks_exact_mut(2).zip(x) {
                o.copy_from_slice(&f32_to_f16(v).to_le_bytes());
            }
        }
        KvFormat::Turbo4 | KvFormat::Turbo3 => {
            let bits = fmt.turbo_bits().unwrap();
            let levels = fmt.levels();
            let y = &mut scratch[..d];
            rotate(x, y);
            let norm = y.iter().map(|v| v * v).sum::<f32>().sqrt();
            let (codes, rest) = out.split_at_mut(4);
            let _ = codes;
            rest.fill(0);
            let mut yc = 0.0f32;
            let mut cc = 0.0f32;
            if norm > 0.0 {
                let to_unit = (d as f32).sqrt() / norm;
                for (j, &yj) in y.iter().enumerate() {
                    let c = nearest(levels, yj * to_unit);
                    put_code(rest, bits, j, c);
                    let l = level(levels, c);
                    yc += yj * l;
                    cc += l * l;
                }
            }
            let scale = if cc > 0.0 { yc / cc } else { 0.0 };
            out[..4].copy_from_slice(&scale.to_le_bytes());
        }
    }
}

/// Dequantize one head (`fmt.head_bytes(out.len())` bytes) into `out`,
/// in the original (unrotated) space.
pub fn dequantize_head(fmt: KvFormat, bytes: &[u8], out: &mut [f32]) {
    match fmt {
        KvFormat::F32 => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = f32::from_le_bytes(b.try_into().unwrap());
            }
        }
        KvFormat::F16 => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        KvFormat::Turbo4 | KvFormat::Turbo3 => {
            dequantize_head_rotated(fmt, bytes, out);
            unrotate(out);
        }
    }
}

/// Turbo formats: the head's reconstruction in the rotated space
/// (`scale · level(code_j)`).
pub fn dequantize_head_rotated(fmt: KvFormat, bytes: &[u8], out: &mut [f32]) {
    let bits = fmt.turbo_bits().expect("turbo format");
    let levels = fmt.levels();
    let scale = f32::from_le_bytes(bytes[..4].try_into().unwrap());
    let codes = &bytes[4..];
    for (j, o) in out.iter_mut().enumerate() {
        *o = scale * level(levels, get_code(codes, bits, j));
    }
}

/// Quantize one cache row (`num_kv_heads * head_dim` floats) into
/// `out` (`fmt.row_bytes(v)` bytes).
pub fn quantize_row(fmt: KvFormat, v: &Variant, x: &[f32], out: &mut [u8]) {
    let hd = v.head_dim;
    let hb = fmt.head_bytes(hd);
    let mut scratch = vec![0.0f32; hd];
    for h in 0..v.num_kv_heads {
        quantize_head(fmt, &x[h * hd..(h + 1) * hd], &mut out[h * hb..(h + 1) * hb], &mut scratch);
    }
}

/// Dequantize one cache row into `out` (`num_kv_heads * head_dim`).
pub fn dequantize_row(fmt: KvFormat, v: &Variant, bytes: &[u8], out: &mut [f32]) {
    let hd = v.head_dim;
    let hb = fmt.head_bytes(hd);
    for h in 0..v.num_kv_heads {
        dequantize_head(fmt, &bytes[h * hb..(h + 1) * hb], &mut out[h * hd..(h + 1) * hd]);
    }
}

/// Decode SDPA over `kv_len` stored rows of `k_rows` / `v_rows`
/// (`fmt.row_bytes(v)` each) — the quantized counterpart of
/// [`super::sdpa::sdpa_cpu_with`], same contract: `q` / `out` are
/// `[num_attn_heads * head_dim]`, `q_gate` the optional sigmoid
/// output gate.
pub fn sdpa_cpu_quant(
    v: &Variant,
    fmt: KvFormat,
    kv_len: usize,
    q: &[f32],
    q_gate: Option<&[f32]>,
    k_rows: &[u8],
    v_rows: &[u8],
    out: &mut [f32],
) {
    sdpa_cpu_quant_pool(None, v, fmt, kv_len, q, q_gate, k_rows, v_rows, out)
}

/// [`sdpa_cpu_quant`] with the heads split across `pool`; bit-identical
/// to the serial path.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_cpu_quant_pool(
    pool: Option<&rayon::ThreadPool>,
    v: &Variant,
    fmt: KvFormat,
    kv_len: usize,
    q: &[f32],
    q_gate: Option<&[f32]>,
    k_rows: &[u8],
    v_rows: &[u8],
    out: &mut [f32],
) {
    let hd = v.head_dim;
    let hb = fmt.head_bytes(hd);
    let rb = fmt.row_bytes(v);
    let heads_per_kv = v.num_attn_heads / v.num_kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let bits = fmt.turbo_bits();
    // Signed levels by code, so a Turbo record is read without
    // materializing the head: k·q' = s · Σ_j q'_j · lut[c_j].
    let mut lut = [0.0f32; 16];
    if bits.is_some() {
        let levels = fmt.levels();
        for (c, l) in lut.iter_mut().enumerate().take(2 * levels.len()) {
            *l = level(levels, c as u32);
        }
    }
    struct Scratch {
        scores: Vec<f32>,
        qr: Vec<f32>,
        row: Vec<f32>,
        acc: Vec<f32>,
    }
    let scratch = || Scratch {
        scores: vec![0.0; kv_len],
        qr: vec![0.0; hd],
        row: vec![0.0; hd],
        acc: vec![0.0; hd],
    };
    let head = |h: usize, o: &mut [f32], sc: &mut Scratch| {
        let Scratch { scores, qr, row, acc } = sc;
        let kvh = h / heads_per_kv;
        let qh = &q[h * hd..(h + 1) * hd];
        if bits.is_some() {
            rotate(qh, qr);
        } else {
            qr.copy_from_slice(qh);
        }
        for (p, s) in scores.iter_mut().enumerate() {
            let rec = &k_rows[p * rb + kvh * hb..p * rb + (kvh + 1) * hb];
            let dot = match bits {
                Some(b) => {
                    let sc = f32::from_le_bytes(rec[..4].try_into().unwrap());
                    sc * turbo_dot(b, &rec[4..], &lut, qr)
                }
                None => {
                    f16_head(rec, row);
                    qr.iter().zip(row.iter()).map(|(a, b)| a * b).sum::<f32>()
                }
            };
            *s = dot * scale;
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        acc.fill(0.0);
        for (p, &w) in scores.iter().enumerate() {
            let rec = &v_rows[p * rb + kvh * hb..p * rb + (kvh + 1) * hb];
            let w = w / sum;
            match bits {
                Some(b) => {
                    let sc = f32::from_le_bytes(rec[..4].try_into().unwrap());
                    turbo_axpy(b, &rec[4..], &lut, w * sc, acc);
                }
                None => {
                    f16_head(rec, row);
                    for (a, &r) in acc.iter_mut().zip(row.iter()) {
                        *a += w * r;
                    }
                }
            }
        }
        if bits.is_some() {
            unrotate(acc);
        }
        match q_gate {
            Some(g) => {
                for (j, oj) in o.iter_mut().enumerate() {
                    *oj = acc[j] / (1.0 + (-g[h * hd + j]).exp());
                }
            }
            None => o.copy_from_slice(acc),
        }
    };
    let out = &mut out[..v.num_attn_heads * hd];
    match pool {
        Some(pool) => pool.install(|| {
            use rayon::prelude::*;
            out.par_chunks_mut(hd).enumerate().for_each_init(scratch, |sc, (h, o)| head(h, o, sc));
        }),
        None => {
            let mut sc = scratch();
            for (h, o) in out.chunks_mut(hd).enumerate() {
                head(h, o, &mut sc);
            }
        }
    }
}

/// `Σ_j x_j · lut[code_j]` over a Turbo record's packed codes.
#[inline]
fn turbo_dot(bits: u32, codes: &[u8], lut: &[f32; 16], x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    if bits == 4 {
        for (b, x2) in codes.iter().zip(x.chunks_exact(2)) {
            s += x2[0] * lut[(b & 0xf) as usize] + x2[1] * lut[(b >> 4) as usize];
        }
    } else {
        for (b3, x8) in codes.chunks_exact(3).zip(x.chunks_exact(8)) {
            let w = b3[0] as u32 | (b3[1] as u32) << 8 | (b3[2] as u32) << 16;
            for (k, &xk) in x8.iter().enumerate() {
                s += xk * lut[((w >> (3 * k)) & 7) as usize];
            }
        }
    }
    s
}

/// `acc_j += a · lut[code_j]` over a Turbo record's packed codes.
#[inline]
fn turbo_axpy(bits: u32, codes: &[u8], lut: &[f32; 16], a: f32, acc: &mut [f32]) {
    if bits == 4 {
        for (b, y2) in codes.iter().zip(acc.chunks_exact_mut(2)) {
            y2[0] += a * lut[(b & 0xf) as usize];
            y2[1] += a * lut[(b >> 4) as usize];
        }
    } else {
        for (b3, y8) in codes.chunks_exact(3).zip(acc.chunks_exact_mut(8)) {
            let w = b3[0] as u32 | (b3[1] as u32) << 8 | (b3[2] as u32) << 16;
            for (k, yk) in y8.iter_mut().enumerate() {
                *yk += a * lut[((w >> (3 * k)) & 7) as usize];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::riir::variants::VARIANT;

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            // Box–Muller over xorshift.
            let mut u = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
            };
            let (a, b) = (u(), u());
            ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
        }
    }

    #[test]
    fn rotation_is_orthogonal() {
        let mut g = rng(1);
        let x: Vec<f32> = (0..128).map(|_| g()).collect();
        let mut y = vec![0.0; 128];
        rotate(&x, &mut y);
        let nx: f32 = x.iter().map(|v| v * v).sum();
        let ny: f32 = y.iter().map(|v| v * v).sum();
        assert!((nx - ny).abs() / nx < 1e-5);
        unrotate(&mut y);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn f16_round_trip_matches_ieee() {
        for &x in &[0.0f32, -0.0, 1.0, -2.5, 65504.0, 1e-5, 6.1e-5, 3.3e-8, 1e9, 0.1] {
            let back = f16_to_f32(f32_to_f16(x));
            let rel = if x == 0.0 { back.abs() } else { ((back - x) / x).abs() };
            assert!(rel <= 1.0 / 1024.0 || x.abs() < 6.2e-5 || x > 65504.0, "{x} -> {back}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(65520.0), 0x7c00, "rounds to inf");
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        // Ties to even: 1 + 2^-11 lies halfway between 1 and 1 + 2^-10.
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00);
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let bytes: Vec<u8> = all.iter().flat_map(|h| h.to_le_bytes()).collect();
        let mut hw = vec![0.0f32; all.len()];
        f16_head(&bytes, &mut hw);
        for (&h, &c) in all.iter().zip(&hw) {
            let (a, b) = (f16_to_f32(h), f16_to_f32_fast(h));
            assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{h:#x}");
            assert!(a.to_bits() == c.to_bits() || (a.is_nan() && c.is_nan()), "fcvtl {h:#x}");
        }
    }

    #[test]
    fn three_bit_packing_round_trips() {
        let mut b = vec![0u8; 3 * 16 / 8];
        for j in 0..16 {
            put_code(&mut b, 3, j, (j as u32 * 5) % 8);
        }
        for j in 0..16 {
            assert_eq!(get_code(&b, 3, j), (j as u32 * 5) % 8);
        }
    }

    #[test]
    fn turbo_error_matches_lloyd_max() {
        // Relative MSE of a Gaussian head vector ≈ the scalar
        // quantizer's distortion (0.0095 at 4 bits, 0.0345 at 3).
        let mut g = rng(7);
        for (fmt, bound) in [(KvFormat::Turbo4, 0.012), (KvFormat::Turbo3, 0.04), (KvFormat::F16, 1e-6)] {
            let mut err = 0.0f64;
            let mut tot = 0.0f64;
            let mut scratch = vec![0.0; 128];
            for _ in 0..200 {
                let x: Vec<f32> = (0..128).map(|_| g() * 3.0).collect();
                let mut buf = vec![0u8; fmt.head_bytes(128)];
                quantize_head(fmt, &x, &mut buf, &mut scratch);
                let mut xr = vec![0.0; 128];
                dequantize_head(fmt, &buf, &mut xr);
                err += x.iter().zip(&xr).map(|(a, b)| ((a - b) * (a - b)) as f64).sum::<f64>();
                tot += x.iter().map(|a| (a * a) as f64).sum::<f64>();
            }
            assert!(err / tot < bound, "{fmt:?}: {}", err / tot);
        }
    }

    #[test]
    fn outlier_channels_are_spread_by_the_rotation() {
        // Real K vectors have a few huge channels; the rotation keeps
        // the error relative to the whole vector.
        let mut g = rng(3);
        let mut x: Vec<f32> = (0..128).map(|_| g()).collect();
        x[5] = 40.0;
        x[77] = -25.0;
        let mut buf = vec![0u8; KvFormat::Turbo4.head_bytes(128)];
        let mut s = vec![0.0; 128];
        quantize_head(KvFormat::Turbo4, &x, &mut buf, &mut s);
        let mut xr = vec![0.0; 128];
        dequantize_head(KvFormat::Turbo4, &buf, &mut xr);
        let err: f32 = x.iter().zip(&xr).map(|(a, b)| (a - b) * (a - b)).sum();
        let tot: f32 = x.iter().map(|a| a * a).sum();
        assert!(err / tot < 0.015, "{}", err / tot);
    }

    #[test]
    fn quant_sdpa_matches_f32_sdpa_on_dequantized_rows() {
        let v = VARIANT;
        if v.attn_kind != AttnKind::Gqa {
            return;
        }
        let mut g = rng(11);
        let kv_dim = v.num_kv_heads * v.head_dim;
        let q_dim = v.num_attn_heads * v.head_dim;
        let n = 37;
        let k: Vec<f32> = (0..n * kv_dim).map(|_| g()).collect();
        let vv: Vec<f32> = (0..n * kv_dim).map(|_| g()).collect();
        let q: Vec<f32> = (0..q_dim).map(|_| g()).collect();
        for fmt in [KvFormat::F16, KvFormat::Turbo4, KvFormat::Turbo3] {
            if fmt.validate(&v).is_err() {
                continue;
            }
            let rb = fmt.row_bytes(&v);
            let mut kq = vec![0u8; n * rb];
            let mut vq = vec![0u8; n * rb];
            let mut kd = vec![0.0f32; n * kv_dim];
            let mut vd = vec![0.0f32; n * kv_dim];
            for p in 0..n {
                quantize_row(fmt, &v, &k[p * kv_dim..(p + 1) * kv_dim], &mut kq[p * rb..(p + 1) * rb]);
                quantize_row(fmt, &v, &vv[p * kv_dim..(p + 1) * kv_dim], &mut vq[p * rb..(p + 1) * rb]);
                dequantize_row(fmt, &v, &kq[p * rb..(p + 1) * rb], &mut kd[p * kv_dim..(p + 1) * kv_dim]);
                dequantize_row(fmt, &v, &vq[p * rb..(p + 1) * rb], &mut vd[p * kv_dim..(p + 1) * kv_dim]);
            }
            let mut a = vec![0.0f32; q_dim];
            let mut b = vec![0.0f32; q_dim];
            sdpa_cpu_quant(&v, fmt, n, &q, None, &kq, &vq, &mut a);
            super::super::sdpa::sdpa_cpu_with(n as i32, &q, None, &kd, &vd, &mut b).unwrap();
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-4, "{fmt:?}: {x} vs {y}");
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::riir::variants::VARIANT;

    /// `cargo test --release ... kv_quant::bench -- --ignored --nocapture`
    #[test]
    #[ignore = "timing only"]
    fn decode_sdpa_timing() {
        let v = VARIANT;
        let n = 2700;
        let kv_dim = v.num_kv_heads * v.head_dim;
        let q_dim = v.num_attn_heads * v.head_dim;
        let x: Vec<f32> = (0..n * kv_dim).map(|i| ((i * 7919 % 1000) as f32 - 500.0) / 300.0).collect();
        let q: Vec<f32> = (0..q_dim).map(|i| ((i * 31 % 100) as f32 - 50.0) / 50.0).collect();
        let reps = 40;
        let mut stores = vec![];
        for fmt in [KvFormat::F16, KvFormat::Turbo4, KvFormat::Turbo3] {
            let rb = fmt.row_bytes(&v);
            let mut st = vec![0u8; n * rb];
            for p in 0..n {
                quantize_row(fmt, &v, &x[p * kv_dim..(p + 1) * kv_dim], &mut st[p * rb..(p + 1) * rb]);
            }
            stores.push((fmt, st));
        }
        let mut serial = vec![vec![0.0f32; q_dim]; 4];
        for threads in [0usize, 1, 2, 4, 6, 8] {
            let pool = (threads > 0)
                .then(|| rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap());
            let p = pool.as_ref();
            let mut line = format!("threads {threads}:");
            let mut out = vec![0.0f32; q_dim];
            let t = std::time::Instant::now();
            for _ in 0..reps {
                super::super::sdpa::sdpa_cpu_pool(p, n as i32, &q, None, &x, &x, &mut out).unwrap();
            }
            line += &format!(" F32 {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
            if threads == 0 { serial[0] = out.clone() } else { assert_eq!(out, serial[0], "F32 differs") }
            for (k, (fmt, st)) in stores.iter().enumerate() {
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    sdpa_cpu_quant_pool(p, &v, *fmt, n, &q, None, st, st, &mut out);
                }
                line += &format!(" {fmt:?} {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
                if threads == 0 { serial[k + 1] = out.clone() } else { assert_eq!(out, serial[k + 1], "{fmt:?} differs") }
            }
            eprintln!("{line}");
        }
    }
}
