//! Streamed token embedding (opt-in: [`crate::riir::OpenConfig::embed_stream`]
//! or `STREAMMOE_EMBED_STREAM=1`).
//!
//! The embedding table is `vocab × hidden` and each token needs only its
//! own row, so instead of holding `model.embed_tokens.{weight,scales,
//! biases}` in the resident weights (0.27 GiB on a3b, 0.53 GiB on a17b)
//! the context reads rows from `model_weights.bin` on demand with
//! `pread` and keeps a small direct-mapped cache of recent rows. The
//! bytes the resident copy no longer holds go to the hot tier under a
//! cap.
//!
//! A row is the raw quantized bytes — `weight` row, then `scales` row,
//! then `biases` row — so every consumer dequantizes exactly as it
//! does from the resident table: the CPU path through
//! [`super::embedding::embed_lookup_at`], the batched prefill path by
//! uploading the rows and running the same `embed_gather_4bit` kernel
//! over them. Output is bit-identical to the resident table.
//!
//! Reads go through their own `F_NOCACHE` fd (readahead off), spans
//! rounded out to 4 KiB, into a page-aligned buffer, so they do not
//! grow the page cache.

use std::fs::File;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::riir::io::weight_file::{TensorInfo, WeightFileError, parse_manifest, set_nocache};
use crate::riir::variants::GROUP_SIZE;

/// Embedding tensor names in the manifest.
pub const EMBED_PREFIX: &str = "model.embed_tokens.";
const WEIGHT_NAME: &str = "model.embed_tokens.weight";
const SCALES_NAME: &str = "model.embed_tokens.scales";
const BIASES_NAME: &str = "model.embed_tokens.biases";

/// Rows the cache holds unless `STREAMMOE_EMBED_CACHE_ROWS` says
/// otherwise (4.5 MiB on a3b, 9 MiB on a17b).
pub const DEFAULT_CACHE_ROWS: usize = 4096;

/// File-span alignment of a row read.
const BLOCK: u64 = 4096;
/// Buffer alignment (Apple Silicon VM page).
const PAGE: usize = 16384;

/// Cache rows: `STREAMMOE_EMBED_CACHE_ROWS` or [`DEFAULT_CACHE_ROWS`].
pub fn cache_rows_from_env() -> usize {
    std::env::var("STREAMMOE_EMBED_CACHE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CACHE_ROWS)
}

/// `true` if `STREAMMOE_EMBED_STREAM` is set to `0` / `false` / `off`.
pub fn disabled_from_env() -> bool {
    matches!(
        std::env::var("STREAMMOE_EMBED_STREAM").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

/// Why the embedding cannot be streamed for this manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotStreamable {
    /// No `lm_head.weight`: the LM head reads (ties) the embedding,
    /// which must then stay resident.
    TiedLmHead,
    /// The embedding is missing or not the 4-bit layout the lookup
    /// reads.
    Layout(String),
}

/// Row geometry and file offsets of the embedding tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedGeometry {
    pub vocab: usize,
    pub hidden_dim: usize,
    /// File offsets of the three tensors.
    pub w_off: u64,
    pub s_off: u64,
    pub b_off: u64,
    /// Bytes of one `weight` row (`hidden / 8` u32 words).
    pub w_row: usize,
    /// Bytes of one `scales` (= `biases`) row.
    pub sb_row: usize,
    /// Bytes of the three tensors together (what the resident copy
    /// held).
    pub table_bytes: u64,
}

impl EmbedGeometry {
    /// Bytes of one cached row (`weight` + `scales` + `biases`).
    pub fn row_bytes(&self) -> usize {
        self.w_row + 2 * self.sb_row
    }

    /// Read the geometry from a parsed manifest. Fails if the LM head
    /// is tied or the layout is not the 4-bit one.
    pub fn from_manifest(
        tensors: &std::collections::HashMap<String, TensorInfo>,
        hidden_dim: usize,
    ) -> Result<Self, NotStreamable> {
        if !tensors.contains_key("lm_head.weight") {
            return Err(NotStreamable::TiedLmHead);
        }
        let get = |n: &str| {
            tensors
                .get(n)
                .ok_or_else(|| NotStreamable::Layout(format!("{n} missing")))
        };
        let (w, s, b) = (get(WEIGHT_NAME)?, get(SCALES_NAME)?, get(BIASES_NAME)?);
        let vocab = w.shape.first().copied().unwrap_or(0);
        let packed_cols = hidden_dim / 8;
        let num_groups = hidden_dim / GROUP_SIZE;
        if w.dtype != "U32" || w.bits != 4 || w.shape != [vocab, packed_cols] {
            return Err(NotStreamable::Layout(format!(
                "{WEIGHT_NAME}: {} bits {} shape {:?}",
                w.dtype, w.bits, w.shape
            )));
        }
        for t in [s, b] {
            if t.shape != [vocab, num_groups] {
                return Err(NotStreamable::Layout(format!("scales/biases shape {:?}", t.shape)));
            }
        }
        Ok(Self {
            vocab,
            hidden_dim,
            w_off: w.offset,
            s_off: s.offset,
            b_off: b.offset,
            w_row: packed_cols * 4,
            sb_row: num_groups * 2,
            table_bytes: w.size + s.size + b.size,
        })
    }
}

/// Direct-mapped row cache: slot `token % rows`.
struct RowCache {
    tags: Vec<i32>,
    data: Vec<u8>,
}

/// A page-aligned heap buffer for `F_NOCACHE` reads.
struct AlignedBuf {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let len = len.div_ceil(PAGE).max(1) * PAGE;
        let layout = std::alloc::Layout::from_size_align(len, PAGE).expect("layout");
        // SAFETY: non-zero size, valid alignment.
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null(), "embed stream: aligned alloc of {len} B failed");
        Self { ptr, layout }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` owns `layout.size()` bytes for our lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

/// On-demand embedding rows from `model_weights.bin`. See the module
/// docs.
pub struct EmbedStream {
    file: File,
    geo: EmbedGeometry,
    cache: Mutex<RowCache>,
    cache_rows: usize,
    /// Rows read from disk / served from the cache.
    reads: AtomicU64,
    hits: AtomicU64,
}

impl std::fmt::Debug for EmbedStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedStream")
            .field("geo", &self.geo)
            .field("cache_rows", &self.cache_rows)
            .finish()
    }
}

impl EmbedStream {
    /// Open `bin_path` for row reads with the geometry of `manifest`.
    /// `Ok(Err(_))` when the embedding cannot be streamed (tied LM
    /// head, other layout) — the caller keeps it resident.
    pub fn open(
        bin_path: &Path,
        manifest_path: &Path,
        hidden_dim: usize,
        cache_rows: usize,
    ) -> Result<Result<Self, NotStreamable>, WeightFileError> {
        let tensors = parse_manifest(manifest_path)?;
        let geo = match EmbedGeometry::from_manifest(&tensors, hidden_dim) {
            Ok(g) => g,
            Err(e) => return Ok(Err(e)),
        };
        let file = File::open(bin_path)?;
        set_nocache(&file)?;
        // SAFETY: valid owned fd; F_RDAHEAD has no failure mode that
        // affects correctness.
        unsafe {
            use std::os::unix::io::AsRawFd;
            libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 0i32);
        }
        let len = file.metadata()?.len();
        for (off, rows_bytes) in [
            (geo.w_off, geo.w_row),
            (geo.s_off, geo.sb_row),
            (geo.b_off, geo.sb_row),
        ] {
            let end = off + (geo.vocab * rows_bytes) as u64;
            if end > len {
                return Err(WeightFileError::OutOfBounds {
                    name: EMBED_PREFIX.into(),
                    offset: off,
                    end,
                    file_size: len,
                });
            }
        }
        let cache_rows = cache_rows.max(1);
        Ok(Ok(Self {
            file,
            geo,
            cache: Mutex::new(RowCache {
                tags: vec![-1; cache_rows],
                data: vec![0u8; cache_rows * geo.row_bytes()],
            }),
            cache_rows,
            reads: AtomicU64::new(0),
            hits: AtomicU64::new(0),
        }))
    }

    pub fn geometry(&self) -> EmbedGeometry {
        self.geo
    }

    /// Bytes the cache holds (counted in the budget plan).
    pub fn cache_bytes(&self) -> u64 {
        (self.cache_rows * self.geo.row_bytes()) as u64
    }

    /// `(rows read from disk, rows served from the cache)`.
    pub fn stats(&self) -> (u64, u64) {
        (self.reads.load(Ordering::Relaxed), self.hits.load(Ordering::Relaxed))
    }

    /// Read row `token` from disk into `dst` (`row_bytes` long).
    fn read_row(&self, token: usize, dst: &mut [u8], scratch: &mut AlignedBuf) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        let g = &self.geo;
        let parts = [
            (g.w_off + (token * g.w_row) as u64, 0, g.w_row),
            (g.s_off + (token * g.sb_row) as u64, g.w_row, g.sb_row),
            (g.b_off + (token * g.sb_row) as u64, g.w_row + g.sb_row, g.sb_row),
        ];
        for (off, dst_off, len) in parts {
            let start = off / BLOCK * BLOCK;
            let end = (off + len as u64).div_ceil(BLOCK) * BLOCK;
            let span = (end - start) as usize;
            let buf = &mut scratch.bytes_mut()[..span];
            let mut got = 0;
            while got < span {
                let n = self.file.read_at(&mut buf[got..], start + got as u64)?;
                if n == 0 {
                    // The file's last block may be short.
                    if start + (got as u64) >= off + len as u64 {
                        break;
                    }
                    return Err(std::io::ErrorKind::UnexpectedEof.into());
                }
                got += n;
            }
            let skip = (off - start) as usize;
            dst[dst_off..dst_off + len].copy_from_slice(&buf[skip..skip + len]);
        }
        Ok(())
    }

    fn scratch(&self) -> AlignedBuf {
        let g = &self.geo;
        AlignedBuf::new(g.w_row.max(g.sb_row) + 2 * BLOCK as usize)
    }

    /// Raw rows of `tokens`, in order, into `out` (`tokens.len() ×
    /// row_bytes`, each `[weight | scales | biases]`). Misses are read
    /// on `pool` when given (one task per miss), else on this thread.
    /// Token ids must be in range.
    pub fn rows(
        &self,
        tokens: &[i32],
        out: &mut [u8],
        pool: Option<&rayon::ThreadPool>,
    ) -> std::io::Result<()> {
        let rb = self.geo.row_bytes();
        assert_eq!(out.len(), tokens.len() * rb);
        // Hits straight from the cache; collect the misses (deduped).
        let mut misses: Vec<i32> = Vec::new();
        {
            let cache = self.cache.lock().expect("embed cache");
            for (i, &t) in tokens.iter().enumerate() {
                let slot = t as usize % self.cache_rows;
                if cache.tags[slot] == t {
                    out[i * rb..(i + 1) * rb].copy_from_slice(&cache.data[slot * rb..(slot + 1) * rb]);
                    self.hits.fetch_add(1, Ordering::Relaxed);
                } else if !misses.contains(&t) {
                    misses.push(t);
                }
            }
        }
        if misses.is_empty() {
            return Ok(());
        }
        let mut fetched = vec![0u8; misses.len() * rb];
        match pool {
            Some(pool) if misses.len() > 1 => {
                use rayon::prelude::*;
                pool.install(|| {
                    fetched
                        .par_chunks_mut(rb)
                        .zip(misses.par_iter())
                        .try_for_each_init(
                            || self.scratch(),
                            |scratch, (dst, &t)| self.read_row(t as usize, dst, scratch),
                        )
                })?;
            }
            _ => {
                let mut scratch = self.scratch();
                for (dst, &t) in fetched.chunks_mut(rb).zip(&misses) {
                    self.read_row(t as usize, dst, &mut scratch)?;
                }
            }
        }
        self.reads.fetch_add(misses.len() as u64, Ordering::Relaxed);
        let mut cache = self.cache.lock().expect("embed cache");
        for (row, &t) in fetched.chunks(rb).zip(&misses) {
            let slot = t as usize % self.cache_rows;
            cache.tags[slot] = t;
            cache.data[slot * rb..(slot + 1) * rb].copy_from_slice(row);
        }
        drop(cache);
        for (i, &t) in tokens.iter().enumerate() {
            if let Some(m) = misses.iter().position(|&x| x == t) {
                out[i * rb..(i + 1) * rb].copy_from_slice(&fetched[m * rb..(m + 1) * rb]);
            }
        }
        Ok(())
    }

    /// Dequantize the embedding of `token` into `out` (`hidden_dim`
    /// floats) — bit-identical to [`super::embedding::embed_lookup`]
    /// on the resident table.
    pub fn lookup(&self, token: i32, out: &mut [f32]) -> std::io::Result<()> {
        let g = &self.geo;
        let mut row = vec![0u8; g.row_bytes()];
        self.rows(&[token], &mut row, None)?;
        let (w, sb) = row.split_at(g.w_row);
        let (s, b) = sb.split_at(g.sb_row);
        super::embedding::embed_lookup_at(w, s, b, 0, out);
        Ok(())
    }
}

impl Drop for EmbedStream {
    fn drop(&mut self) {
        let (reads, hits) = self.stats();
        if reads + hits > 0 {
            eprintln!(
                "[embed-stream] rows read {reads} ({:.2} MiB from disk, 4 KiB-rounded), cache hits {hits}",
                (reads * 3 * BLOCK) as f64 / (1u64 << 20) as f64
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Streamed rows and dequantized embeddings equal the resident
    /// table's, with and without a pool, through hits and misses and a
    /// one-row cache (every lookup evicts).
    #[test]
    fn streamed_rows_match_resident_table() {
        use crate::riir::io::weight_file::WeightFile;
        let dir = std::env::temp_dir().join(format!("streammoe-es-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (bin, manifest) = (dir.join("w.bin"), dir.join("w.json"));
        let (vocab, hidden) = (300usize, 128usize);
        let (w_row, sb_row) = (hidden / 8 * 4, hidden / GROUP_SIZE * 2);
        // An odd lead-in so rows straddle nothing in particular.
        let lead = 777usize;
        let (w_off, s_off) = (lead, lead + vocab * w_row);
        let b_off = s_off + vocab * sb_row;
        let len = b_off + vocab * sb_row + 13;
        let mut data: Vec<u8> = (0..len).map(|i| (i * 131 % 251) as u8).collect();
        // Keep bf16 scales/biases finite: clear the exponent's top bit.
        for i in (s_off..b_off + vocab * sb_row).step_by(2) {
            data[i + 1] &= 0x3F;
        }
        std::fs::write(&bin, &data).unwrap();
        std::fs::write(
            &manifest,
            format!(
                r#"{{"tensors": {{
                "lm_head.weight": {{"offset": 0, "size": 4, "shape": [1], "dtype": "U32", "bits": 4}},
                "model.embed_tokens.weight": {{"offset": {w_off}, "size": {}, "shape": [{vocab}, {}], "dtype": "U32", "bits": 4}},
                "model.embed_tokens.scales": {{"offset": {s_off}, "size": {}, "shape": [{vocab}, {}], "dtype": "BF16"}},
                "model.embed_tokens.biases": {{"offset": {b_off}, "size": {}, "shape": [{vocab}, {}], "dtype": "BF16"}}
            }}}}"#,
                vocab * w_row,
                hidden / 8,
                vocab * sb_row,
                hidden / GROUP_SIZE,
                vocab * sb_row,
                hidden / GROUP_SIZE
            ),
        )
        .unwrap();
        let wf = WeightFile::open(&bin, &manifest).unwrap();
        let (w, s, b) = (
            wf.tensor_bytes(WEIGHT_NAME).unwrap(),
            wf.tensor_bytes(SCALES_NAME).unwrap(),
            wf.tensor_bytes(BIASES_NAME).unwrap(),
        );
        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
        for cache_rows in [1, 7, 4096] {
            let es = EmbedStream::open(&bin, &manifest, hidden, cache_rows).unwrap().unwrap();
            assert_eq!(es.geometry().table_bytes, (vocab * (w_row + 2 * sb_row)) as u64);
            let tokens: Vec<i32> = vec![0, 299, 5, 5, 128, 0, 299, 64, 3, 200];
            for p in [None, Some(&pool)] {
                let rb = es.geometry().row_bytes();
                let mut rows = vec![0u8; tokens.len() * rb];
                es.rows(&tokens, &mut rows, p).unwrap();
                for (i, &t) in tokens.iter().enumerate() {
                    let t = t as usize;
                    let r = &rows[i * rb..(i + 1) * rb];
                    assert_eq!(&r[..w_row], &w[t * w_row..(t + 1) * w_row]);
                    assert_eq!(&r[w_row..w_row + sb_row], &s[t * sb_row..(t + 1) * sb_row]);
                    assert_eq!(&r[w_row + sb_row..], &b[t * sb_row..(t + 1) * sb_row]);
                }
            }
            for &t in &tokens {
                let (mut want, mut got) = (vec![0f32; hidden], vec![0f32; hidden]);
                super::super::embedding::embed_lookup_at(w, s, b, t, &mut want);
                es.lookup(t, &mut got).unwrap();
                assert_eq!(
                    want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    got.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
            let (reads, hits) = es.stats();
            assert!(reads > 0);
            if cache_rows == 4096 {
                assert!(hits > 0);
            }
        }
        // No lm_head.weight → tied: not streamable.
        let tied = std::fs::read_to_string(&manifest).unwrap().replace("lm_head.weight", "other");
        std::fs::write(&manifest, tied).unwrap();
        assert_eq!(
            EmbedStream::open(&bin, &manifest, hidden, 8).unwrap().unwrap_err(),
            NotStreamable::TiedLmHead
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
