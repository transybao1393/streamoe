//! Vision encoder for Qwen3.5 / Qwen3.6 (`qwen3_5_moe`'s Qwen3-VL-style
//! ViT), on the CPU.
//!
//! Turns an RGB image into input embeddings for the language model: one
//! `out_hidden_size` row per 2 × 2 block of 16-px patches. The caller
//! feeds them with [`crate::riir::RsCtx::eval_stage_mrope`] in place of
//! the `<|image_pad|>` tokens, at the positions [`image_rope_positions`]
//! gives. Mirrors `Qwen3_5MoeVisionModel` (transformers 5.x):
//!
//! - patch embedding (a `[t=2, 16, 16, 3]` conv; a still image repeats
//!   its one frame, so the two temporal taps are summed once at open),
//! - learned 48 × 48 position embedding, bilinearly resampled
//!   (`align_corners = True`) to the image's patch grid,
//! - `depth` pre-norm blocks: LayerNorm → attention with 2-D axial RoPE
//!   over the whole image → LayerNorm → `gelu_pytorch_tanh` MLP,
//! - merger: LayerNorm per patch → 4 patches concatenated → Linear →
//!   GELU (erf) → Linear.
//!
//! ## Memory
//!
//! The ~0.85 GiB of BF16 weights stay on disk: each matrix is streamed
//! with `F_NOCACHE` reads (no page cache) into a fixed buffer, in row
//! chunks of at most [`CHUNK_ELEMS`], just before its GEMM. Everything is
//! allocated (and touched) at [`VisionEncoder::open`] for the largest
//! image `max_pixels` allows, so [`VisionEncoder::bytes`] is the
//! encoder's whole footprint — pass it as `OpenConfig::host_buffers` to
//! keep a capped context's plan exact. Per image the encoder reads the
//! weight file once (~0.85 GiB of SSD reads).
//!
//! GEMMs run on Accelerate's `cblas_sgemm`; normalization, activations,
//! softmax and RoPE on a dedicated `threads`-wide rayon pool.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::MmapMut;
use rayon::prelude::*;

const PAGE: usize = 16384;
/// Largest weight slice converted to f32 at once (elements): the
/// MLP's `fc1` / `fc2` (4304 × 1152) fit whole; the merger's 4608 × 4608
/// `fc1` streams in row chunks.
pub const CHUNK_ELEMS: usize = 4304 * 1152;
const LN_EPS: f32 = 1e-6;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    fn erff(x: f32) -> f32;
}
const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

/// Errors from [`VisionEncoder`].
#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("vision weights: {0}")]
    Io(#[from] std::io::Error),
    #[error("vision manifest: {0}")]
    Manifest(String),
    #[error("image {width}x{height}: {why}")]
    Image { width: usize, height: usize, why: String },
}

/// `vision_weights.json`'s `config` (see `tools/export_vision.py`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub num_position_embeddings: usize,
    pub image_token_id: i32,
    pub vision_start_token_id: i32,
    pub vision_end_token_id: i32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub min_pixels: usize,
    pub max_pixels: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TensorEntry {
    offset: u64,
    size: usize,
    shape: Vec<usize>,
}

#[derive(serde::Deserialize)]
struct Manifest {
    config: VisionConfig,
    tensors: HashMap<String, TensorEntry>,
}

/// One encoded image.
#[derive(Debug, Clone)]
pub struct VisionOutput {
    /// `n_tokens × out_hidden_size` input embeddings, row-major, in
    /// the order of [`image_rope_positions`].
    pub embeds: Vec<f32>,
    pub n_tokens: usize,
    /// Merged token grid (patch grid / `spatial_merge_size`).
    pub grid_h: usize,
    pub grid_w: usize,
}

/// `smart_resize` of the Qwen2-VL image processor: the image size
/// (`height`, `width`) the encoder takes for an input of that size —
/// both multiples of `factor` (= patch × merge, 32), area within
/// `[min_pixels, max_pixels]`, aspect ratio kept as closely as possible.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let (h, w) = (height as f64, width as f64);
    let f = factor as f64;
    let round = |x: f64| (x / f).round_ties_even() * f;
    let (mut hb, mut wb) = (round(h).max(f), round(w).max(f));
    if hb * wb > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = ((h / beta / f).floor() * f).max(f);
        wb = ((w / beta / f).floor() * f).max(f);
    } else if hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    (hb as usize, wb as usize)
}

/// M-RoPE positions of an image's `grid_h × grid_w` tokens placed after
/// text that ends at position `start - 1` (Qwen3.5 `get_rope_index`):
/// token `(r, c)` gets `[start, start + r, start + c]`. Returns the rows
/// and the position the following text continues from,
/// `start + max(grid_h, grid_w)`.
pub fn image_rope_positions(start: i32, grid_h: usize, grid_w: usize) -> (Vec<[i32; 3]>, i32) {
    let mut rows = Vec::with_capacity(grid_h * grid_w);
    for r in 0..grid_h as i32 {
        for c in 0..grid_w as i32 {
            rows.push([start, start + r, start + c]);
        }
    }
    (rows, start + grid_h.max(grid_w) as i32)
}

/// The CPU vision encoder. See the module docs.
pub struct VisionEncoder {
    cfg: VisionConfig,
    file: File,
    tensors: HashMap<String, TensorEntry>,
    max_pixels: usize,
    max_patches: usize,
    pool: rayon::ThreadPool,
    /// Patch kernel with the temporal taps summed: `[hidden, 16·16·3]`
    /// in `(y, x, c)` order, plus its bias.
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    /// `[num_position_embeddings, hidden]`.
    pos_embed: Vec<f32>,
    // Streaming buffers.
    io: MmapMut,
    wbuf: Vec<f32>,
    small: Vec<f32>,
    // Activations, sized for `max_patches`.
    x: Vec<f32>,
    xn: Vec<f32>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    mlp: Vec<f32>,
    scores: Vec<f32>,
    merged: Vec<f32>,
    merged2: Vec<f32>,
}

impl VisionEncoder {
    /// Open `dir/vision_weights.{bin,json}` for images up to `max_pixels`
    /// (after [`smart_resize`]; clamped to the model's own limit) using
    /// a `threads`-wide pool for the non-GEMM work.
    pub fn open(dir: &Path, max_pixels: usize, threads: usize) -> Result<Self, VisionError> {
        let m: Manifest = serde_json::from_str(&std::fs::read_to_string(dir.join("vision_weights.json"))?)
            .map_err(|e| VisionError::Manifest(e.to_string()))?;
        let cfg = m.config;
        let max_pixels = max_pixels.min(cfg.max_pixels).max(cfg.min_pixels);
        let p = cfg.patch_size;
        let max_patches = max_pixels / (p * p);
        let file = File::open(dir.join("vision_weights.bin"))?;
        crate::riir::io::weight_file::set_nocache(&file)?;
        let largest = m.tensors.values().map(|t| t.size).max().unwrap_or(0);
        // Room for one CHUNK_ELEMS-sized BF16 slice, page-rounded.
        let io_len = (CHUNK_ELEMS * 2).max(largest.min(CHUNK_ELEMS * 2)).div_ceil(PAGE) * PAGE + PAGE;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("moeflux-vision-{i}"))
            .build()
            .map_err(|e| VisionError::Manifest(e.to_string()))?;
        let (h, inter, out) = (cfg.hidden_size, cfg.intermediate_size, cfg.out_hidden_size);
        let mh = h * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let max_tokens = max_patches / (cfg.spatial_merge_size * cfg.spatial_merge_size);
        let mut enc = Self {
            file,
            tensors: m.tensors,
            max_pixels,
            max_patches,
            pool,
            patch_w: Vec::new(),
            patch_b: Vec::new(),
            pos_embed: Vec::new(),
            io: MmapMut::map_anon(io_len)?,
            wbuf: vec![0f32; CHUNK_ELEMS],
            small: vec![0f32; mh.max(3 * h).max(inter)],
            x: vec![0f32; max_patches * h],
            xn: vec![0f32; max_patches * h],
            qkv: vec![0f32; max_patches * 3 * h],
            attn: vec![0f32; max_patches * h],
            mlp: vec![0f32; max_patches * inter],
            scores: vec![0f32; max_patches * max_patches],
            merged: vec![0f32; max_tokens * mh],
            merged2: vec![0f32; max_tokens * mh.max(out)],
            cfg,
        };
        enc.load_resident()?;
        Ok(enc)
    }

    /// Bytes the encoder holds after [`Self::open`], plus the largest
    /// [`VisionOutput::embeds`] it returns — everything an
    /// [`Self::encode`] needs beyond a few small tables.
    pub fn bytes(&self) -> u64 {
        let f = |v: &Vec<f32>| (v.len() * 4) as u64;
        let m2 = self.cfg.spatial_merge_size * self.cfg.spatial_merge_size;
        let out = (self.max_patches / m2 * self.cfg.out_hidden_size * 4) as u64;
        out + self.io.len() as u64
            + [&self.patch_w, &self.patch_b, &self.pos_embed, &self.wbuf, &self.small, &self.x, &self.xn]
                .iter()
                .map(|v| f(v))
                .sum::<u64>()
            + [&self.qkv, &self.attn, &self.mlp, &self.scores, &self.merged, &self.merged2]
                .iter()
                .map(|v| f(v))
                .sum::<u64>()
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// The largest image area (pixels) [`Self::encode`] takes.
    pub fn max_pixels(&self) -> usize {
        self.max_pixels
    }

    /// [`smart_resize`] for this encoder: the size to resize an image of
    /// `height × width` to before [`Self::encode`].
    pub fn target_size(&self, height: usize, width: usize) -> (usize, usize) {
        let f = self.cfg.patch_size * self.cfg.spatial_merge_size;
        smart_resize(height, width, f, self.cfg.min_pixels, self.max_pixels)
    }

    fn entry(&self, name: &str) -> Result<TensorEntry, VisionError> {
        self.tensors.get(name).cloned().ok_or_else(|| VisionError::Manifest(format!("missing tensor {name}")))
    }

    /// Read BF16 elements `[from, from + n)` of tensor `name` into `dst`.
    fn read_bf16(&mut self, name: &str, from: usize, dst: &mut [f32]) -> Result<(), VisionError> {
        use std::os::unix::fs::FileExt;
        let e = self.entry(name)?;
        let n = dst.len();
        if (from + n) * 2 > e.size {
            return Err(VisionError::Manifest(format!("{name}: read past end")));
        }
        // Page-aligned window covering the bytes.
        let start = e.offset + (from * 2) as u64;
        let aligned = start / PAGE as u64 * PAGE as u64;
        let lead = (start - aligned) as usize;
        let want = (lead + n * 2).div_ceil(PAGE) * PAGE;
        if want > self.io.len() {
            return Err(VisionError::Manifest(format!("{name}: slice of {n} larger than the read buffer")));
        }
        let mut got = 0;
        while got < want {
            let r = self.file.read_at(&mut self.io[got..want], aligned + got as u64)?;
            if r == 0 {
                break; // the file ends inside the last page; the bytes we need are in
            }
            got += r;
        }
        if got < lead + n * 2 {
            return Err(VisionError::Manifest(format!("{name}: short read")));
        }
        let src = &self.io[lead..lead + n * 2];
        self.pool.install(|| {
            dst.par_chunks_mut(4096).enumerate().for_each(|(ci, d)| {
                let s = &src[ci * 8192..];
                for (i, o) in d.iter_mut().enumerate() {
                    let b = u16::from_le_bytes([s[2 * i], s[2 * i + 1]]);
                    *o = f32::from_bits((b as u32) << 16);
                }
            })
        });
        Ok(())
    }

    fn read_vec(&mut self, name: &str) -> Result<Vec<f32>, VisionError> {
        let n = self.entry(name)?.shape.iter().product();
        let mut v = vec![0f32; n];
        self.read_bf16(name, 0, &mut v)?;
        Ok(v)
    }

    /// Patch kernel (temporal taps summed), its bias, the position table.
    fn load_resident(&mut self) -> Result<(), VisionError> {
        let w = self.read_vec("patch_embed.proj.weight")?; // [o, t, y, x, c]
        let (h, p, t) = (self.cfg.hidden_size, self.cfg.patch_size, self.cfg.temporal_patch_size);
        let k = p * p * 3;
        let mut pw = vec![0f32; h * k];
        for o in 0..h {
            for j in 0..k {
                pw[o * k + j] = (0..t).map(|ti| w[(o * t + ti) * k + j]).sum();
            }
        }
        self.patch_w = pw;
        self.patch_b = self.read_vec("patch_embed.proj.bias")?;
        self.pos_embed = self.read_vec("pos_embed.weight")?;
        // Commit the activation buffers now, so the footprint is what
        // `bytes` reports from the start.
        for v in [&mut self.x, &mut self.xn, &mut self.qkv, &mut self.attn, &mut self.mlp, &mut self.scores] {
            v.fill(0.0);
        }
        self.merged.fill(0.0);
        self.merged2.fill(0.0);
        self.io.fill(0);
        Ok(())
    }

    /// `y[n, out] = x[n, in] · W^T + b`, `W` = tensor `{name}.weight`
    /// (`[out, in]`) streamed in row chunks. `x` / `y` are the encoder's
    /// own buffers, passed by role.
    fn linear(&mut self, name: &str, n: usize, x: Buf, y: Buf) -> Result<(), VisionError> {
        let e = self.entry(&format!("{name}.weight"))?;
        let (out, inp) = (e.shape[0], e.shape[1]);
        let rows = (CHUNK_ELEMS / inp).max(1);
        let mut wbuf = std::mem::take(&mut self.wbuf);
        let mut r0 = 0;
        while r0 < out {
            let r1 = (r0 + rows).min(out);
            self.read_bf16(&format!("{name}.weight"), r0 * inp, &mut wbuf[..(r1 - r0) * inp])?;
            let (xp, yp) = (self.buf(x).as_ptr(), self.buf_mut(y).as_mut_ptr());
            // SAFETY: x is n × inp, y is n × out, wbuf holds (r1-r0) × inp;
            // x and y are distinct buffers.
            unsafe {
                cblas_sgemm(
                    ROW_MAJOR, NO_TRANS, TRANS, n as i32, (r1 - r0) as i32, inp as i32, 1.0, xp,
                    inp as i32, wbuf.as_ptr(), inp as i32, 0.0, yp.add(r0), out as i32,
                );
            }
            r0 = r1;
        }
        self.wbuf = wbuf;
        let mut b = std::mem::take(&mut self.small);
        self.read_bf16(&format!("{name}.bias"), 0, &mut b[..out])?;
        let bias = &b[..out];
        let (yb, pool) = self.buf_mut_pool(y);
        let yb = &mut yb[..n * out];
        pool.install(|| {
            yb.par_chunks_mut(out).for_each(|row| row.iter_mut().zip(bias).for_each(|(v, b)| *v += b))
        });
        self.small = b;
        Ok(())
    }

    fn buf(&self, b: Buf) -> &[f32] {
        match b {
            Buf::X => &self.x,
            Buf::Xn => &self.xn,
            Buf::Qkv => &self.qkv,
            Buf::Attn => &self.attn,
            Buf::Mlp => &self.mlp,
            Buf::Merged => &self.merged,
            Buf::Merged2 => &self.merged2,
        }
    }

    fn buf_mut(&mut self, b: Buf) -> &mut [f32] {
        self.buf_mut_pool(b).0
    }

    fn vec_mut(&mut self, b: Buf) -> &mut Vec<f32> {
        match b {
            Buf::X => &mut self.x,
            Buf::Xn => &mut self.xn,
            Buf::Qkv => &mut self.qkv,
            Buf::Attn => &mut self.attn,
            Buf::Mlp => &mut self.mlp,
            Buf::Merged => &mut self.merged,
            Buf::Merged2 => &mut self.merged2,
        }
    }

    fn buf_mut_pool(&mut self, b: Buf) -> (&mut [f32], &rayon::ThreadPool) {
        let v = match b {
            Buf::X => &mut self.x,
            Buf::Xn => &mut self.xn,
            Buf::Qkv => &mut self.qkv,
            Buf::Attn => &mut self.attn,
            Buf::Mlp => &mut self.mlp,
            Buf::Merged => &mut self.merged,
            Buf::Merged2 => &mut self.merged2,
        };
        (v, &self.pool)
    }

    /// LayerNorm (`{name}.weight` / `.bias`) of `n` rows of `src` into `dst`.
    fn layer_norm(&mut self, name: &str, n: usize, dim: usize, src: Buf, dst: Buf) -> Result<(), VisionError> {
        let mut wb = std::mem::take(&mut self.small);
        self.read_bf16(&format!("{name}.weight"), 0, &mut wb[..dim])?;
        let mut bb = vec![0f32; dim];
        self.read_bf16(&format!("{name}.bias"), 0, &mut bb)?;
        // Move dst out so src can be borrowed beside it (src != dst).
        let mut d = std::mem::take(self.vec_mut(dst));
        let s = &self.buf(src)[..n * dim];
        let (w, pool) = (&wb[..dim], &self.pool);
        pool.install(|| {
            d[..n * dim].par_chunks_mut(dim).zip(s.par_chunks(dim)).for_each(|(o, r)| {
                let mean = r.iter().sum::<f32>() / dim as f32;
                let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
                let inv = 1.0 / (var + LN_EPS).sqrt();
                for i in 0..dim {
                    o[i] = (r[i] - mean) * inv * w[i] + bb[i];
                }
            })
        });
        *self.vec_mut(dst) = d;
        self.small = wb;
        Ok(())
    }

    /// Encode an RGB image (`height × width × 3` bytes, already resized
    /// to [`Self::target_size`]).
    pub fn encode(&mut self, rgb: &[u8], height: usize, width: usize) -> Result<VisionOutput, VisionError> {
        let c = self.cfg.clone();
        let (p, m, hid, heads) = (c.patch_size, c.spatial_merge_size, c.hidden_size, c.num_heads);
        let bad = |why: String| VisionError::Image { width, height, why };
        if height % (p * m) != 0 || width % (p * m) != 0 {
            return Err(bad(format!("sides must be multiples of {}; use target_size", p * m)));
        }
        if rgb.len() != height * width * 3 {
            return Err(bad(format!("{} bytes, want {}", rgb.len(), height * width * 3)));
        }
        let (gh, gw) = (height / p, width / p);
        let n = gh * gw;
        if n > self.max_patches {
            return Err(bad(format!("{n} patches > {} (max_pixels {})", self.max_patches, self.max_pixels)));
        }
        // Patch order: 2 × 2 merge blocks row-major, row-major inside.
        let patch_rc: Vec<(usize, usize)> = (0..n)
            .map(|s| {
                let (blk, inner) = (s / (m * m), s % (m * m));
                let (bh, bw) = (blk / (gw / m), blk % (gw / m));
                (bh * m + inner / m, bw * m + inner % m)
            })
            .collect();

        // ── patch embedding + position embedding → x ─────────────────
        let k = p * p * 3;
        // The MLP buffer (n × intermediate ≥ n × k) is free until block 0.
        let mut patches = std::mem::take(&mut self.mlp);
        for (s, &(r, cl)) in patch_rc.iter().enumerate() {
            let dst = &mut patches[s * k..(s + 1) * k];
            for y in 0..p {
                for x in 0..p {
                    let px = &rgb[((r * p + y) * width + cl * p + x) * 3..][..3];
                    for ch in 0..3 {
                        dst[(y * p + x) * 3 + ch] = (px[ch] as f32 / 255.0 - c.image_mean[ch]) / c.image_std[ch];
                    }
                }
            }
        }
        // SAFETY: patches n × k, patch_w hid × k, x n × hid.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR, NO_TRANS, TRANS, n as i32, hid as i32, k as i32, 1.0, patches.as_ptr(),
                k as i32, self.patch_w.as_ptr(), k as i32, 0.0, self.x.as_mut_ptr(), hid as i32,
            );
        }
        self.mlp = patches;
        let side = (c.num_position_embeddings as f64).sqrt() as usize;
        let taps = |i: usize, len: usize| -> [(usize, f32); 2] {
            let src = if len > 1 { i as f32 * (side - 1) as f32 / (len - 1) as f32 } else { 0.0 };
            let f = src.floor();
            let t0 = (f as usize).min(side - 1);
            let t1 = (f as usize + 1).min(side - 1);
            [(t0, (1.0 - (src - f)).max(0.0)), (t1, (1.0 - (f + 1.0 - src)).max(0.0))]
        };
        {
            let (x, pe, pb) = (&mut self.x, &self.pos_embed, &self.patch_b);
            self.pool.install(|| {
                x[..n * hid].par_chunks_mut(hid).enumerate().for_each(|(s, row)| {
                    let (r, cl) = patch_rc[s];
                    for (i, b) in row.iter_mut().zip(pb) {
                        *i += b;
                    }
                    for (ti, wi) in taps(r, gh) {
                        for (tj, wj) in taps(cl, gw) {
                            let w = wi * wj;
                            if w != 0.0 {
                                let e = &pe[(ti * side + tj) * hid..][..hid];
                                row.iter_mut().zip(e).for_each(|(v, e)| *v += w * e);
                            }
                        }
                    }
                })
            });
        }

        // ── vision RoPE tables: head_dim/4 frequencies per axis ───────
        let hd = hid / heads;
        let quarter = hd / 4;
        let inv: Vec<f32> = (0..quarter).map(|j| 1.0 / 10000f32.powf((2 * j) as f32 / (hd / 2) as f32)).collect();
        let mut cos = vec![0f32; n * hd / 2];
        let mut sin = vec![0f32; n * hd / 2];
        for (s, &(r, cl)) in patch_rc.iter().enumerate() {
            for i in 0..hd / 2 {
                let a = if i < quarter { r as f32 * inv[i] } else { cl as f32 * inv[i - quarter] };
                cos[s * hd / 2 + i] = a.cos();
                sin[s * hd / 2 + i] = a.sin();
            }
        }

        // ── blocks ───────────────────────────────────────────────────
        let scale = (hd as f32).powf(-0.5);
        for b in 0..c.depth {
            let pre = format!("blocks.{b}");
            self.layer_norm(&format!("{pre}.norm1"), n, hid, Buf::X, Buf::Xn)?;
            self.linear(&format!("{pre}.attn.qkv"), n, Buf::Xn, Buf::Qkv)?;
            {
                let (qkv, pool) = (&mut self.qkv, &self.pool);
                let (cos, sin) = (&cos, &sin);
                pool.install(|| {
                    qkv[..n * 3 * hid].par_chunks_mut(3 * hid).enumerate().for_each(|(s, row)| {
                        let (cs, sn) = (&cos[s * hd / 2..][..hd / 2], &sin[s * hd / 2..][..hd / 2]);
                        for part in 0..2 {
                            for h in 0..heads {
                                let v = &mut row[part * hid + h * hd..][..hd];
                                for i in 0..hd / 2 {
                                    let (x0, x1) = (v[i], v[i + hd / 2]);
                                    v[i] = x0 * cs[i] - x1 * sn[i];
                                    v[i + hd / 2] = x1 * cs[i] + x0 * sn[i];
                                }
                            }
                        }
                    })
                });
            }
            for h in 0..heads {
                let q = self.qkv[h * hd..].as_ptr();
                let kk = self.qkv[hid + h * hd..].as_ptr();
                let v = self.qkv[2 * hid + h * hd..].as_ptr();
                // SAFETY: q/k/v rows stride 3·hid, n rows of hd; scores n × n.
                unsafe {
                    cblas_sgemm(
                        ROW_MAJOR, NO_TRANS, TRANS, n as i32, n as i32, hd as i32, scale, q,
                        (3 * hid) as i32, kk, (3 * hid) as i32, 0.0, self.scores.as_mut_ptr(), n as i32,
                    );
                }
                let (sc, pool) = (&mut self.scores, &self.pool);
                pool.install(|| {
                    sc[..n * n].par_chunks_mut(n).for_each(|row| {
                        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0;
                        for v in row.iter_mut() {
                            *v = (*v - mx).exp();
                            sum += *v;
                        }
                        let inv = 1.0 / sum;
                        row.iter_mut().for_each(|v| *v *= inv);
                    })
                });
                // SAFETY: scores n × n, v n × hd (stride 3·hid), attn n × hid.
                unsafe {
                    cblas_sgemm(
                        ROW_MAJOR, NO_TRANS, NO_TRANS, n as i32, hd as i32, n as i32, 1.0,
                        self.scores.as_ptr(), n as i32, v, (3 * hid) as i32, 0.0,
                        self.attn.as_mut_ptr().add(h * hd), hid as i32,
                    );
                }
            }
            self.linear(&format!("{pre}.attn.proj"), n, Buf::Attn, Buf::Xn)?;
            self.add_into_x(n * hid);
            self.layer_norm(&format!("{pre}.norm2"), n, hid, Buf::X, Buf::Xn)?;
            self.linear(&format!("{pre}.mlp.linear_fc1"), n, Buf::Xn, Buf::Mlp)?;
            {
                let (mlp, pool) = (&mut self.mlp, &self.pool);
                let len = n * c.intermediate_size;
                pool.install(|| mlp[..len].par_chunks_mut(4096).for_each(|ch| ch.iter_mut().for_each(|v| *v = gelu_tanh(*v))));
            }
            self.linear(&format!("{pre}.mlp.linear_fc2"), n, Buf::Mlp, Buf::Xn)?;
            self.add_into_x(n * hid);
        }

        // ── merger ───────────────────────────────────────────────────
        let tokens = n / (m * m);
        let mh = hid * m * m;
        self.layer_norm("merger.norm", n, hid, Buf::X, Buf::Merged)?; // n × hid = tokens × mh
        self.linear("merger.linear_fc1", tokens, Buf::Merged, Buf::Merged2)?;
        {
            let (mg, pool) = (&mut self.merged2, &self.pool);
            pool.install(|| {
                mg[..tokens * mh].par_chunks_mut(4096).for_each(|ch| ch.iter_mut().for_each(|v| *v = gelu_erf(*v)))
            });
        }
        // fc2 reads merged2 → writes merged (now free).
        self.linear("merger.linear_fc2", tokens, Buf::Merged2, Buf::Merged)?;
        let out = c.out_hidden_size;
        Ok(VisionOutput {
            embeds: self.merged[..tokens * out].to_vec(),
            n_tokens: tokens,
            grid_h: gh / m,
            grid_w: gw / m,
        })
    }

    /// `x += xn` over the first `len` elements.
    fn add_into_x(&mut self, len: usize) {
        let (x, xn, pool) = (&mut self.x, &self.xn, &self.pool);
        pool.install(|| {
            x[..len].par_chunks_mut(4096).zip(xn[..len].par_chunks(4096)).for_each(|(a, b)| {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b)
            })
        });
    }
}

#[derive(Clone, Copy)]
enum Buf {
    X,
    Xn,
    Qkv,
    Attn,
    Mlp,
    Merged,
    Merged2,
}

#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn gelu_erf(x: f32) -> f32 {
    // SAFETY: scalar libm.
    0.5 * x * (1.0 + unsafe { erff(x * std::f32::consts::FRAC_1_SQRT_2) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `smart_resize` against the Qwen2-VL processor's arithmetic
    /// (factor 32, min 65 536, max 16 777 216) on the reference images.
    #[test]
    fn smart_resize_matches_processor() {
        let f = |h, w| smart_resize(h, w, 32, 65536, 16_777_216);
        assert_eq!(f(336, 448), (320, 448)); // shapes: grid 20 × 28
        assert_eq!(f(256, 512), (256, 512)); // text: 16 × 32
        assert_eq!(f(210, 300), (224, 320)); // gradient: 14 × 20
        // Over the cap: shrink, keep the aspect ratio.
        assert_eq!(smart_resize(2000, 1000, 32, 65536, 401_408), (896, 448));
    }

    #[test]
    fn image_positions_follow_get_rope_index() {
        let (rows, next) = image_rope_positions(5, 2, 3);
        assert_eq!(rows, vec![[5, 5, 5], [5, 5, 6], [5, 5, 7], [5, 6, 5], [5, 6, 6], [5, 6, 7]]);
        assert_eq!(next, 8);
    }
}
