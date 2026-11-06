//! Hashed n-gram embedding of the per-layer embedding (PLE).
//!
//! Port of transformers' `Qwen4ExpTextNGramEmbedding`. For every token
//! the layer looks up `(ngram_size - 1) × heads_per_ngram` rows (16 on
//! the real model: 8 bigram heads, 8 trigram heads) of a table with
//! ~320M rows of 160 values, and concatenates them into one
//! `ple_embed_dim` (2560) vector.
//!
//! - [`NgramHasher`] computes the row ids, bit-identical to the
//!   reference (`tests/fixtures/qwen38_ple/ngram_ids.json`, made by
//!   `tools/model_layer/qwen3_8_flash_next/ple_reference.py`).
//! - [`NgramHistory`] is the per-sequence state the ids need across
//!   calls: the last `ngram_size - 1` token ids (eos before the first
//!   token).
//! - [`NgramTable`] reads rows from the `MFNG` row-table file on demand
//!   (`pread`, small direct-mapped cache, nothing resident), like
//!   `io/embed_stream.rs`.
//!
//! ## Hashing
//!
//! Position `p` of the token history `[ctx…, tokens…]` sees
//! `shifted[s] = tok[p - s]` for `s < ngram_size`, except that a shift
//! never looks back across an eos: if any of `tok[p - s ..= p - 1]` is
//! eos, `shifted[s] = eos` (`_shift_right_ignore_eos`). For the n-gram
//! of order `n` (2 ..= ngram_size), with the layer multipliers `m`:
//!
//! ```text
//! mixed = shifted[0]·m[0] ^ shifted[1]·m[1] ^ … ^ shifted[n-1]·m[n-1]   (i64)
//! id[head] = mixed mod head_vocab_size[head] + head_offset[head]
//! ```
//!
//! for the `heads_per_ngram` heads of that order. Head sizes are the
//! consecutive primes from `ngram_vocab_size_base` on (global head
//! `ple_layer_index × n_heads + h` takes the `(g+1)`-th), offsets their
//! running sum; the table is padded to a multiple of
//! `make_ngram_vocab_size_divisible_by` rows.
//!
//! ## `MFNG` row table (written by the converter)
//!
//! Little-endian. A 4096-byte header:
//!
//! | Offset | Type | Field |
//! | --- | --- | --- |
//! | 0 | `[u8; 4]` | magic `b"MFNG"` |
//! | 4 | u32 | version = 1 |
//! | 8 | u32 | header_bytes = 4096 (where rows start) |
//! | 12 | u32 | dim: values per row (160) |
//! | 16 | u32 | group_size (32) |
//! | 20 | u32 | bits = 4 |
//! | 24 | u32 | scale dtype: 1 = bf16 (the only one accepted) |
//! | 28 | u32 | row_bytes = dim / 2 + 4 × (dim / group_size) (100) |
//! | 32 | u64 | n_rows: the padded table (320,001,536) |
//! | 40 | u32 | n_heads (16) |
//! | 44 | u32 | ngram_size (3) |
//! | 48 | `u64[8]` | layer multipliers (`ngram_size` used, rest 0) |
//! | 112 | `u64[64]` | head vocab sizes (`n_heads` used) |
//! | 624 | `u64[64]` | head offsets (`n_heads` used) |
//! | 1136 | — | zero up to 4096 |
//!
//! Row `r` is at `header_bytes + r × row_bytes`:
//! `[packed u32 × dim/8 | scales bf16 × dim/group | biases bf16 × dim/group]`,
//! MLX affine 4-bit (`w = q × scale + bias`, element `j` in nibble
//! `j % 8` of word `j / 8`, low nibble first). The file is exactly
//! `header_bytes + n_rows × row_bytes` long. Rows are the checkpoint's
//! `ngram_embedding.shards.{0..127}` concatenated in shard order along
//! dim 0. The multipliers / sizes / offsets are copied from the
//! checkpoint's `layer_multipliers`, `ngram_heads_vocab_sizes` and
//! `ngram_heads_offsets` buffers, so [`NgramTable::check_hasher`] can
//! prove the Rust hashing matches the checkpoint's at open.

use std::fs::File;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::riir::io::embedding::bf16_to_f32;
use crate::riir::io::weight_file::set_nocache;

const MASK_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10007;

/// splitmix64 finalizer (`_splitmix64`).
pub fn splitmix64(value: u64) -> u64 {
    let mut v = value.wrapping_add(MASK_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1);
    v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2);
    v ^ (v >> 31)
}

/// Per-position multipliers (`_build_layer_multipliers`): odd, and
/// small enough that `token × m` never overflows an i64.
pub fn build_layer_multipliers(
    unigram_vocab_size: u64,
    ngram_size: usize,
    ple_layer_index: u64,
    seed: u64,
) -> Vec<u64> {
    let max_long = (1u64 << 63) - 1;
    let multiplier_max = max_long / unigram_vocab_size.max(1);
    let half_bound = (multiplier_max / 2).max(1);
    let base_seed = seed.wrapping_add(PRIME_1.wrapping_mul(ple_layer_index));
    (0..ngram_size as u64)
        .map(|i| {
            let value = base_seed.wrapping_add(MASK_GAMMA.wrapping_mul(i + 1));
            2 * (splitmix64(value) % half_bound) + 1
        })
        .collect()
}

/// Trial-division primality (`_is_prime`).
pub fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut d = 3u64;
    while d * d <= value {
        if value % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// The `count`-th prime after `start` (`_find_nth_prime_after`).
pub fn find_nth_prime_after(start: u64, count: u64) -> u64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

/// Hashing parameters (from the text config).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramConfig {
    /// `vocab_size` (the unigram vocabulary).
    pub vocab_size: u64,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: u64,
    pub make_ngram_vocab_size_divisible_by: u64,
    /// `config.seed` (default 1234; the released config does not set it).
    pub seed: u64,
    /// `text_config.eos_token_id` (first entry if a list).
    pub eos_token_id: i32,
    /// Index of this layer in `ple_layer_ids` (0 for the one PLE layer).
    pub ple_layer_index: u64,
}

impl NgramConfig {
    /// Qwen3.8-Flash-Next: `ple_layer_ids = [2]` (one-indexed), so the
    /// PLE sits at the start of decoder layer index 1, `ple_layer_index` 0.
    pub const QWEN3_8_FLASH_NEXT: NgramConfig = NgramConfig {
        vocab_size: 248_320,
        ngram_size: 3,
        heads_per_ngram: 8,
        ngram_vocab_size_base: 20_000_000,
        make_ngram_vocab_size_divisible_by: 128,
        seed: 1234,
        eos_token_id: 248_044,
        ple_layer_index: 0,
    };

    /// Context tokens the hashing needs before each token.
    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }

    /// Rows looked up per token.
    pub fn n_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }
}

/// Row-id hashing of one PLE layer. See the module docs.
#[derive(Debug, Clone)]
pub struct NgramHasher {
    cfg: NgramConfig,
    multipliers: Vec<u64>,
    head_vocab_sizes: Vec<u64>,
    head_offsets: Vec<u64>,
    total_vocab: u64,
    padded_rows: u64,
}

impl NgramHasher {
    pub fn new(cfg: NgramConfig) -> Self {
        assert!(cfg.ngram_size >= 2 && cfg.heads_per_ngram >= 1);
        let n_heads = cfg.n_heads();
        let mut head_vocab_sizes = Vec::with_capacity(n_heads);
        let mut head_offsets = Vec::with_capacity(n_heads);
        let mut total = 0u64;
        for h in 0..n_heads as u64 {
            let global = cfg.ple_layer_index * n_heads as u64 + h;
            let size = find_nth_prime_after(cfg.ngram_vocab_size_base - 1, global + 1);
            head_vocab_sizes.push(size);
            head_offsets.push(total);
            total += size;
        }
        let div = cfg.make_ngram_vocab_size_divisible_by.max(1);
        let multipliers =
            build_layer_multipliers(cfg.vocab_size, cfg.ngram_size, cfg.ple_layer_index, cfg.seed);
        Self {
            cfg,
            multipliers,
            head_vocab_sizes,
            head_offsets,
            total_vocab: total,
            padded_rows: total.div_ceil(div) * div,
        }
    }

    pub fn config(&self) -> &NgramConfig {
        &self.cfg
    }
    pub fn multipliers(&self) -> &[u64] {
        &self.multipliers
    }
    pub fn head_vocab_sizes(&self) -> &[u64] {
        &self.head_vocab_sizes
    }
    pub fn head_offsets(&self) -> &[u64] {
        &self.head_offsets
    }
    /// Sum of the head sizes (before padding).
    pub fn total_vocab(&self) -> u64 {
        self.total_vocab
    }
    /// Rows of the table (the padded vocabulary).
    pub fn padded_rows(&self) -> u64 {
        self.padded_rows
    }
    pub fn n_heads(&self) -> usize {
        self.cfg.n_heads()
    }

    /// Row ids of `tokens` (continuing `history`) into `out`
    /// (`tokens.len() × n_heads`, token-major), and advance `history`.
    pub fn ids(&self, history: &mut NgramHistory, tokens: &[i32], out: &mut [u64]) {
        let n_heads = self.n_heads();
        let ctx = self.cfg.context_len();
        let ngram = self.cfg.ngram_size;
        let hpn = self.cfg.heads_per_ngram;
        let eos = self.cfg.eos_token_id as i64;
        assert_eq!(out.len(), tokens.len() * n_heads, "ngram ids: out length");
        assert_eq!(history.ctx.len(), ctx, "ngram ids: history length");
        let mut hist: Vec<i64> = Vec::with_capacity(ctx + tokens.len());
        hist.extend(history.ctx.iter().map(|&t| t as i64));
        hist.extend(tokens.iter().map(|&t| t as i64));
        let mut shifted = vec![0i64; ngram];
        for (t, row) in out.chunks_mut(n_heads).enumerate() {
            let p = ctx + t;
            // `_shift_right_ignore_eos`: shift s is valid unless an eos
            // sits in hist[p-s ..= p-1] (p ≥ ctx ≥ s, so p-s ≥ 0).
            let mut crossed = false;
            shifted[0] = hist[p];
            for s in 1..ngram {
                crossed |= hist[p - s] == eos;
                shifted[s] = if crossed { eos } else { hist[p - s] };
            }
            for n in 2..=ngram {
                let mut mixed = shifted[0].wrapping_mul(self.multipliers[0] as i64);
                for pos in 1..n {
                    mixed ^= shifted[pos].wrapping_mul(self.multipliers[pos] as i64);
                }
                let start = (n - 2) * hpn;
                for h in start..start + hpn {
                    let size = self.head_vocab_sizes[h] as i64;
                    row[h] = mixed.rem_euclid(size) as u64 + self.head_offsets[h];
                }
            }
        }
        let keep = hist.len() - ctx;
        for (dst, &src) in history.ctx.iter_mut().zip(&hist[keep..]) {
            *dst = src as i32;
        }
    }
}

/// Per-sequence token context of the n-gram hashing: the last
/// `ngram_size - 1` token ids, eos before the first token. Part of
/// the sequence state (snapshots).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHistory {
    eos: i32,
    ctx: Vec<i32>,
}

impl NgramHistory {
    pub fn new(cfg: &NgramConfig) -> Self {
        Self { eos: cfg.eos_token_id, ctx: vec![cfg.eos_token_id; cfg.context_len()] }
    }

    /// Back to the start of a sequence.
    pub fn reset(&mut self) {
        self.ctx.fill(self.eos);
    }

    /// The context tokens, oldest first.
    pub fn tokens(&self) -> &[i32] {
        &self.ctx
    }

    /// Rebuild from the last tokens of a sequence (e.g. after a
    /// rewind): the history after feeding `seq` from the start.
    pub fn set_from_sequence(&mut self, seq: &[i32]) {
        self.reset();
        let n = self.ctx.len();
        let take = seq.len().min(n);
        self.ctx[n - take..].copy_from_slice(&seq[seq.len() - take..]);
    }

    /// Bytes of [`Self::save`].
    pub fn state_bytes(&self) -> usize {
        4 * self.ctx.len()
    }

    /// Append the state (i32 LE each) to `out`.
    pub fn save(&self, out: &mut Vec<u8>) {
        for t in &self.ctx {
            out.extend_from_slice(&t.to_le_bytes());
        }
    }

    /// Load from [`Self::save`] bytes (`state_bytes` long).
    pub fn load(&mut self, bytes: &[u8]) -> Result<(), NgramTableError> {
        if bytes.len() != self.state_bytes() {
            return Err(NgramTableError::Format(format!(
                "ngram history: {} bytes, expected {}",
                bytes.len(),
                self.state_bytes()
            )));
        }
        for (t, b) in self.ctx.iter_mut().zip(bytes.chunks_exact(4)) {
            *t = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        Ok(())
    }
}

// ---- MFNG row table -----------------------------------------------------

pub const MFNG_MAGIC: [u8; 4] = *b"MFNG";
pub const MFNG_VERSION: u32 = 1;
pub const MFNG_HEADER_BYTES: usize = 4096;
pub const MFNG_MAX_NGRAM: usize = 8;
pub const MFNG_MAX_HEADS: usize = 64;
/// Header `scale_dtype` code for bf16.
pub const MFNG_SCALE_BF16: u32 = 1;

/// Rows the cache holds unless `STREAMMOE_NGRAM_CACHE_ROWS` says
/// otherwise (100 B each on the real table: 1.6 MiB).
pub const DEFAULT_CACHE_ROWS: usize = 16384;

/// File-span alignment of a row read.
const BLOCK: u64 = 4096;
/// Buffer alignment (Apple Silicon VM page).
const PAGE: usize = 16384;

/// Cache rows: `STREAMMOE_NGRAM_CACHE_ROWS` or [`DEFAULT_CACHE_ROWS`].
pub fn cache_rows_from_env() -> usize {
    std::env::var("STREAMMOE_NGRAM_CACHE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CACHE_ROWS)
}

#[derive(Debug, thiserror::Error)]
pub enum NgramTableError {
    #[error("ngram table i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("ngram table format: {0}")]
    Format(String),
    #[error("ngram table does not match the hashing: {0}")]
    HasherMismatch(String),
    #[error("ngram row {row} out of range (table has {n_rows})")]
    RowOutOfRange { row: u64, n_rows: u64 },
}

/// Parsed `MFNG` header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MfngHeader {
    pub header_bytes: u64,
    pub dim: usize,
    pub group_size: usize,
    pub bits: u32,
    pub row_bytes: usize,
    pub n_rows: u64,
    pub n_heads: usize,
    pub ngram_size: usize,
    pub multipliers: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
    pub head_offsets: Vec<u64>,
}

fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
fn rd_u64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

impl MfngHeader {
    /// Parse and validate the first [`MFNG_HEADER_BYTES`] of a file.
    pub fn parse(b: &[u8]) -> Result<Self, NgramTableError> {
        let bad = |m: String| Err(NgramTableError::Format(m));
        if b.len() < MFNG_HEADER_BYTES {
            return bad(format!("header {} bytes, need {MFNG_HEADER_BYTES}", b.len()));
        }
        if b[0..4] != MFNG_MAGIC {
            return bad(format!("magic {:?}", &b[0..4]));
        }
        let version = rd_u32(b, 4);
        if version != MFNG_VERSION {
            return bad(format!("version {version}"));
        }
        let header_bytes = rd_u32(b, 8) as u64;
        let dim = rd_u32(b, 12) as usize;
        let group_size = rd_u32(b, 16) as usize;
        let bits = rd_u32(b, 20);
        let scale_dtype = rd_u32(b, 24);
        let row_bytes = rd_u32(b, 28) as usize;
        let n_rows = rd_u64(b, 32);
        let n_heads = rd_u32(b, 40) as usize;
        let ngram_size = rd_u32(b, 44) as usize;
        if header_bytes < MFNG_HEADER_BYTES as u64 {
            return bad(format!("header_bytes {header_bytes}"));
        }
        if bits != 4 || scale_dtype != MFNG_SCALE_BF16 {
            return bad(format!("bits {bits} scale dtype {scale_dtype} (want 4-bit, bf16)"));
        }
        if dim == 0 || group_size == 0 || group_size % 8 != 0 || dim % group_size != 0 {
            return bad(format!("dim {dim} group {group_size}"));
        }
        if row_bytes != dim / 2 + 4 * (dim / group_size) {
            return bad(format!("row_bytes {row_bytes} for dim {dim} group {group_size}"));
        }
        if !(2..=MFNG_MAX_NGRAM).contains(&ngram_size) || n_heads == 0 || n_heads > MFNG_MAX_HEADS {
            return bad(format!("ngram_size {ngram_size} n_heads {n_heads}"));
        }
        Ok(Self {
            header_bytes,
            dim,
            group_size,
            bits,
            row_bytes,
            n_rows,
            n_heads,
            ngram_size,
            multipliers: (0..ngram_size).map(|i| rd_u64(b, 48 + 8 * i)).collect(),
            head_vocab_sizes: (0..n_heads).map(|i| rd_u64(b, 112 + 8 * i)).collect(),
            head_offsets: (0..n_heads).map(|i| rd_u64(b, 624 + 8 * i)).collect(),
        })
    }

    /// Serialize (for the converter's tests and synthetic tables).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = vec![0u8; MFNG_HEADER_BYTES];
        b[0..4].copy_from_slice(&MFNG_MAGIC);
        let put32 = |b: &mut Vec<u8>, off: usize, v: u32| b[off..off + 4].copy_from_slice(&v.to_le_bytes());
        let put64 = |b: &mut Vec<u8>, off: usize, v: u64| b[off..off + 8].copy_from_slice(&v.to_le_bytes());
        put32(&mut b, 4, MFNG_VERSION);
        put32(&mut b, 8, self.header_bytes as u32);
        put32(&mut b, 12, self.dim as u32);
        put32(&mut b, 16, self.group_size as u32);
        put32(&mut b, 20, self.bits);
        put32(&mut b, 24, MFNG_SCALE_BF16);
        put32(&mut b, 28, self.row_bytes as u32);
        put64(&mut b, 32, self.n_rows);
        put32(&mut b, 40, self.n_heads as u32);
        put32(&mut b, 44, self.ngram_size as u32);
        for (i, &m) in self.multipliers.iter().enumerate() {
            put64(&mut b, 48 + 8 * i, m);
        }
        for (i, &s) in self.head_vocab_sizes.iter().enumerate() {
            put64(&mut b, 112 + 8 * i, s);
        }
        for (i, &o) in self.head_offsets.iter().enumerate() {
            put64(&mut b, 624 + 8 * i, o);
        }
        b
    }
}

/// Dequantize one raw row (`[packed | scales | biases]`) into `out`
/// (`dim` floats): `q × scale + bias`, as `embed_lookup_at` does.
pub fn dequant_row(row: &[u8], dim: usize, group_size: usize, out: &mut [f32]) {
    let w_bytes = dim / 2;
    let groups = dim / group_size;
    let (w, sb) = row.split_at(w_bytes);
    for g in 0..groups {
        let scale = bf16_to_f32(u16::from_le_bytes([sb[2 * g], sb[2 * g + 1]]));
        let bias = bf16_to_f32(u16::from_le_bytes([sb[2 * (groups + g)], sb[2 * (groups + g) + 1]]));
        for j in g * group_size..(g + 1) * group_size {
            let word = u32::from_le_bytes(w[4 * (j / 8)..4 * (j / 8) + 4].try_into().unwrap());
            let q = (word >> (4 * (j % 8))) & 0xF;
            out[j] = (q as f32) * scale + bias;
        }
    }
}

/// Direct-mapped row cache: slot `row % rows`.
struct RowCache {
    tags: Vec<u64>,
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
        assert!(!ptr.is_null(), "ngram table: aligned alloc of {len} B failed");
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

/// On-demand rows of an `MFNG` n-gram table. Reads go through their
/// own `F_NOCACHE` fd (readahead off), spans rounded out to 4 KiB, so
/// they do not grow the page cache; only the row cache is resident.
pub struct NgramTable {
    file: File,
    hdr: MfngHeader,
    cache: Mutex<RowCache>,
    cache_rows: usize,
    reads: AtomicU64,
    hits: AtomicU64,
}

impl std::fmt::Debug for NgramTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgramTable")
            .field("hdr", &self.hdr)
            .field("cache_rows", &self.cache_rows)
            .finish()
    }
}

impl NgramTable {
    /// Open an `MFNG` file; checks the header and the file length.
    pub fn open(path: &Path, cache_rows: usize) -> Result<Self, NgramTableError> {
        use std::os::unix::fs::FileExt;
        let file = File::open(path)?;
        let mut head = vec![0u8; MFNG_HEADER_BYTES];
        file.read_exact_at(&mut head, 0)?;
        let hdr = MfngHeader::parse(&head)?;
        let want = hdr.header_bytes + hdr.n_rows * hdr.row_bytes as u64;
        let len = file.metadata()?.len();
        if len != want {
            return Err(NgramTableError::Format(format!(
                "file is {len} bytes, header says {want}"
            )));
        }
        set_nocache(&file)?;
        // SAFETY: valid owned fd; F_RDAHEAD has no failure mode that
        // affects correctness.
        unsafe {
            use std::os::unix::io::AsRawFd;
            libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 0i32);
        }
        let cache_rows = cache_rows.max(1);
        Ok(Self {
            file,
            cache: Mutex::new(RowCache {
                tags: vec![u64::MAX; cache_rows],
                data: vec![0u8; cache_rows * hdr.row_bytes],
            }),
            hdr,
            cache_rows,
            reads: AtomicU64::new(0),
            hits: AtomicU64::new(0),
        })
    }

    pub fn header(&self) -> &MfngHeader {
        &self.hdr
    }

    /// Bytes the cache holds (count it in the budget plan).
    pub fn cache_bytes(&self) -> u64 {
        (self.cache_rows * self.hdr.row_bytes) as u64
    }

    /// `(rows read from disk, rows served from the cache)`.
    pub fn stats(&self) -> (u64, u64) {
        (self.reads.load(Ordering::Relaxed), self.hits.load(Ordering::Relaxed))
    }

    /// Check that the table was written for `hasher`: row count, head
    /// count, row width (`ple_embed_dim / n_heads`) and the hashing
    /// constants stored by the converter.
    pub fn check_hasher(&self, hasher: &NgramHasher, embed_dim: usize) -> Result<(), NgramTableError> {
        let h = &self.hdr;
        let mut why = Vec::new();
        if h.n_rows != hasher.padded_rows() {
            why.push(format!("rows {} vs {}", h.n_rows, hasher.padded_rows()));
        }
        if h.n_heads != hasher.n_heads() || h.dim * h.n_heads != embed_dim {
            why.push(format!("heads {} × dim {} vs {} heads, embed {embed_dim}", h.n_heads, h.dim, hasher.n_heads()));
        }
        if h.ngram_size != hasher.config().ngram_size || h.multipliers != hasher.multipliers() {
            why.push(format!("multipliers {:?} vs {:?}", h.multipliers, hasher.multipliers()));
        }
        if h.head_vocab_sizes != hasher.head_vocab_sizes() || h.head_offsets != hasher.head_offsets() {
            why.push("head sizes / offsets differ".into());
        }
        if why.is_empty() { Ok(()) } else { Err(NgramTableError::HasherMismatch(why.join("; "))) }
    }

    fn scratch(&self) -> AlignedBuf {
        AlignedBuf::new(self.hdr.row_bytes + 2 * BLOCK as usize)
    }

    /// Read row `row` from disk into `dst` (`row_bytes` long).
    fn read_row(&self, row: u64, dst: &mut [u8], scratch: &mut AlignedBuf) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        let len = self.hdr.row_bytes as u64;
        let off = self.hdr.header_bytes + row * len;
        let start = off / BLOCK * BLOCK;
        let end = (off + len).div_ceil(BLOCK) * BLOCK;
        let span = (end - start) as usize;
        let buf = &mut scratch.bytes_mut()[..span];
        let mut got = 0;
        while got < span {
            let n = self.file.read_at(&mut buf[got..], start + got as u64)?;
            if n == 0 {
                // The file's last block may be short.
                if start + (got as u64) >= off + len {
                    break;
                }
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            got += n;
        }
        let skip = (off - start) as usize;
        dst.copy_from_slice(&buf[skip..skip + len as usize]);
        Ok(())
    }

    /// Raw rows `ids`, in order, into `out` (`ids.len() × row_bytes`).
    /// Misses are read on `pool` when given (one task per miss), else
    /// on this thread.
    pub fn rows(
        &self,
        ids: &[u64],
        out: &mut [u8],
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<(), NgramTableError> {
        let rb = self.hdr.row_bytes;
        assert_eq!(out.len(), ids.len() * rb, "ngram rows: out length");
        if let Some(&row) = ids.iter().find(|&&r| r >= self.hdr.n_rows) {
            return Err(NgramTableError::RowOutOfRange { row, n_rows: self.hdr.n_rows });
        }
        let mut misses: Vec<u64> = Vec::new();
        {
            let cache = self.cache.lock().expect("ngram cache");
            for (i, &r) in ids.iter().enumerate() {
                let slot = (r % self.cache_rows as u64) as usize;
                if cache.tags[slot] == r {
                    out[i * rb..(i + 1) * rb].copy_from_slice(&cache.data[slot * rb..(slot + 1) * rb]);
                    self.hits.fetch_add(1, Ordering::Relaxed);
                } else if !misses.contains(&r) {
                    misses.push(r);
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
                        .try_for_each_init(|| self.scratch(), |scratch, (dst, &r)| self.read_row(r, dst, scratch))
                })?;
            }
            _ => {
                let mut scratch = self.scratch();
                for (dst, &r) in fetched.chunks_mut(rb).zip(&misses) {
                    self.read_row(r, dst, &mut scratch)?;
                }
            }
        }
        self.reads.fetch_add(misses.len() as u64, Ordering::Relaxed);
        let mut cache = self.cache.lock().expect("ngram cache");
        for (row, &r) in fetched.chunks(rb).zip(&misses) {
            let slot = (r % self.cache_rows as u64) as usize;
            cache.tags[slot] = r;
            cache.data[slot * rb..(slot + 1) * rb].copy_from_slice(row);
        }
        drop(cache);
        for (i, &r) in ids.iter().enumerate() {
            if let Some(m) = misses.iter().position(|&x| x == r) {
                out[i * rb..(i + 1) * rb].copy_from_slice(&fetched[m * rb..(m + 1) * rb]);
            }
        }
        Ok(())
    }

    /// Dequantized rows `ids` into `out` (`ids.len() × dim`, in order).
    /// With `ids` from [`NgramHasher::ids`] this is the n-gram
    /// embedding: `T × (n_heads · dim)` = `T × ple_embed_dim`.
    pub fn embed(
        &self,
        ids: &[u64],
        out: &mut [f32],
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<(), NgramTableError> {
        let (rb, dim) = (self.hdr.row_bytes, self.hdr.dim);
        assert_eq!(out.len(), ids.len() * dim, "ngram embed: out length");
        let mut raw = vec![0u8; ids.len() * rb];
        self.rows(ids, &mut raw, pool)?;
        for (row, dst) in raw.chunks(rb).zip(out.chunks_mut(dim)) {
            dequant_row(row, dim, self.hdr.group_size, dst);
        }
        Ok(())
    }
}

impl Drop for NgramTable {
    fn drop(&mut self) {
        let (reads, hits) = self.stats();
        if reads + hits > 0 {
            eprintln!("[ngram-table] rows read {reads}, cache hits {hits}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> serde_json::Value {
        let path = format!("{}/tests/fixtures/qwen38_ple/{name}", env!("CARGO_MANIFEST_DIR"));
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn u64s(v: &serde_json::Value) -> Vec<u64> {
        v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
    }

    #[test]
    fn splitmix_and_primes() {
        // splitmix64 of 0 (reference value of the standard generator's
        // first output with seed 0).
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
        assert!(is_prime(2) && is_prime(20_000_003) && !is_prime(20_000_001) && !is_prime(1));
        assert_eq!(find_nth_prime_after(19_999_999, 1), 20_000_003);
        assert_eq!(find_nth_prime_after(19_999_999, 2), 20_000_023);
    }

    /// Constants and row ids equal the transformers reference on random
    /// sequences fed in chunks: a first chunk of 1 token (eos padding),
    /// prefill then single-token decode steps, eos tokens mixed in
    /// (also first), and `ple_layer_index` 1.
    #[test]
    fn ids_match_reference() {
        let fx = fixture("ngram_ids.json");
        let mut n_checked = 0;
        for case in fx["cases"].as_array().unwrap() {
            let cfg = NgramConfig {
                ple_layer_index: case["ple_layer_index"].as_u64().unwrap(),
                ..NgramConfig::QWEN3_8_FLASH_NEXT
            };
            let h = NgramHasher::new(cfg);
            assert_eq!(h.multipliers(), u64s(&case["multipliers"]).as_slice());
            assert_eq!(h.head_vocab_sizes(), u64s(&case["head_vocab_sizes"]).as_slice());
            assert_eq!(h.head_offsets(), u64s(&case["head_offsets"]).as_slice());
            assert_eq!(h.total_vocab(), case["total_vocab"].as_u64().unwrap());
            assert_eq!(h.padded_rows(), case["padded_rows"].as_u64().unwrap());
            let mut hist = NgramHistory::new(&cfg);
            let mut all = Vec::new();
            for (chunk, want) in case["chunks"].as_array().unwrap().iter().zip(case["ids"].as_array().unwrap()) {
                let toks: Vec<i32> = chunk.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
                let mut got = vec![0u64; toks.len() * h.n_heads()];
                h.ids(&mut hist, &toks, &mut got);
                let want: Vec<u64> = want.as_array().unwrap().iter().flat_map(u64s).collect();
                assert_eq!(got, want);
                n_checked += toks.len();
                all.extend(toks);
            }
            // The history equals one rebuilt from the sequence.
            let mut rebuilt = NgramHistory::new(&cfg);
            rebuilt.set_from_sequence(&all);
            assert_eq!(rebuilt, hist);
            // Save / load round trip.
            let mut bytes = Vec::new();
            hist.save(&mut bytes);
            let mut loaded = NgramHistory::new(&cfg);
            loaded.load(&bytes).unwrap();
            assert_eq!(loaded, hist);
        }
        assert!(n_checked > 60);
    }

    fn bf16(x: f32) -> u16 {
        let u = x.to_bits();
        ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
    }

    /// Synthetic table: rows come back byte-exact and dequantize to
    /// `q·scale + bias`, through hits, misses, duplicates, a one-row
    /// cache, with and without a pool; bad files are refused.
    #[test]
    fn table_rows_from_synthetic_file() {
        let dir = std::env::temp_dir().join(format!("streammoe-ngram-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.mfng");
        let (dim, group, n_rows) = (64usize, 32usize, 300u64);
        let row_bytes = dim / 2 + 4 * (dim / group);
        let hdr = MfngHeader {
            header_bytes: MFNG_HEADER_BYTES as u64,
            dim,
            group_size: group,
            bits: 4,
            row_bytes,
            n_rows,
            n_heads: 2,
            ngram_size: 2,
            multipliers: vec![3, 5],
            head_vocab_sizes: vec![149, 151],
            head_offsets: vec![0, 149],
        };
        // Row r: codes (r + j) % 16, scale 0.5 + r/1000, bias -r/100.
        let mut file = hdr.to_bytes();
        let mut want = Vec::new();
        for r in 0..n_rows as usize {
            let mut row = vec![0u8; row_bytes];
            let mut vals = vec![0f32; dim];
            let (s, b) = (bf16(0.5 + r as f32 / 1000.0), bf16(-(r as f32) / 100.0));
            for j in 0..dim {
                let q = ((r + j) % 16) as u32;
                let w = 4 * (j / 8);
                let mut word = u32::from_le_bytes(row[w..w + 4].try_into().unwrap());
                word |= q << (4 * (j % 8));
                row[w..w + 4].copy_from_slice(&word.to_le_bytes());
                vals[j] = q as f32 * bf16_to_f32(s) + bf16_to_f32(b);
            }
            for g in 0..dim / group {
                row[dim / 2 + 2 * g..dim / 2 + 2 * g + 2].copy_from_slice(&s.to_le_bytes());
                let bo = dim / 2 + 2 * (dim / group) + 2 * g;
                row[bo..bo + 2].copy_from_slice(&b.to_le_bytes());
            }
            file.extend_from_slice(&row);
            want.push((row, vals));
        }
        std::fs::write(&path, &file).unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
        let ids: Vec<u64> = vec![0, 299, 5, 5, 128, 0, 299, 64, 3, 200, 41, 40];
        for cache_rows in [1, 7, 4096] {
            let t = NgramTable::open(&path, cache_rows).unwrap();
            assert_eq!(t.header(), &hdr);
            for p in [None, Some(&pool)] {
                let mut raw = vec![0u8; ids.len() * row_bytes];
                t.rows(&ids, &mut raw, p).unwrap();
                let mut emb = vec![0f32; ids.len() * dim];
                t.embed(&ids, &mut emb, p).unwrap();
                for (i, &r) in ids.iter().enumerate() {
                    assert_eq!(&raw[i * row_bytes..(i + 1) * row_bytes], want[r as usize].0.as_slice());
                    assert_eq!(&emb[i * dim..(i + 1) * dim], want[r as usize].1.as_slice());
                }
            }
            let (reads, hits) = t.stats();
            assert!(reads > 0);
            if cache_rows == 4096 {
                assert!(hits > 0);
            }
            assert!(matches!(
                t.rows(&[n_rows], &mut vec![0u8; row_bytes], None),
                Err(NgramTableError::RowOutOfRange { .. })
            ));
        }
        // A short file and a bad magic are refused.
        std::fs::write(&path, &file[..file.len() - 1]).unwrap();
        assert!(matches!(NgramTable::open(&path, 8), Err(NgramTableError::Format(_))));
        let mut bad = file.clone();
        bad[0] = b'X';
        std::fs::write(&path, &bad).unwrap();
        assert!(matches!(NgramTable::open(&path, 8), Err(NgramTableError::Format(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
