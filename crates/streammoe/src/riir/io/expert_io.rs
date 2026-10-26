//! Per-layer expert-blob I/O — slice 9c.
//!
//! Mirrors the C path's expert-loading sub-system in
//! `metal_infer/infer.m` (lines 7570–7605 + 3222–3740) at the
//! "plain pread / mmap" level. Caches and exotic decompression
//! modes are deliberately out of scope:
//!
//! - **In:** open `experts_dir/packed_experts/layer_NN.bin` per
//!   layer (RAII, missing files are tolerated mirroring the C
//!   semantics), read `EXPERT_SIZE` bytes at offset
//!   `expert_idx * EXPERT_SIZE` via [`std::os::unix::fs::FileExt::read_at`].
//! - **Out:** LZ4 decompression (`packed_experts_lz4/`), the LRU
//!   metal-buffer expert cache (`g_expert_cache`), the malloc
//!   expert cache (`g_malloc_cache`), and the async pread
//!   thread pool (`g_async_pread`). Those are slice 9f material.
//! - **Out:** 2-bit experts (`packed_experts_2bit/`). The 4-bit
//!   `EXPERT_SIZE` is wired in here; 2-bit lands as a Phase 7
//!   commit alongside the matvec_2bit pipeline.
//!
//! ## Why no thread pool / mmap yet
//!
//! The C async-pread thread pool exists because pread on macOS has
//! per-syscall overhead and parallelizing K preads gives a real win
//! during decode. For the diff oracle we just need byte-equal
//! output; a synchronous `read_at` produces identical bytes faster
//! than is worth measuring. mmap lands in slice 9d/9e if a real-
//! prompt benchmark shows the cache-warm path matters.
//!
//! ## RAII
//!
//! [`ExpertFiles`] owns one [`std::fs::File`] per layer. Files
//! close on drop. No `Arc`, no global state — fits the strategy's
//! "single-`&mut Ctx` discipline" rule.

use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
#[cfg(feature = "model-cogito-v2-671b")]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use std::sync::atomic::{AtomicU32, Ordering};

use memmap2::Mmap;
use metal::{Buffer, MTLResourceOptions, NSUInteger};
use streammoe_metal::ResidencySet;

use crate::riir::backend::buftype::ExpertBaseBuf;
use crate::riir::backend::{BufId, MetalBufferPool};
use crate::riir::variants::{ExpertLayout, VARIANT, Variant};

/// Disable kernel readahead on a successfully-opened layer fd.
/// Mirrors `fcntl(fd, F_RDAHEAD, 0)` in the C path at
/// `metal_infer/infer.m:7593`.
///
/// **Variant-gated to `model-cogito-v2-671b` only.** Theoretical
/// rationale: when the expert working set is *much larger* than
/// physical RAM (Cogito-V2 ≈ 340 GB at 4-bit on a 96 GB UMA Mac),
/// kernel readahead is wasteful — adjacent pages get pulled into
/// the page cache, evicting other expert pages we still need.
/// When the working set fits (Qwen3.5-397B-A17B streams from
/// per-layer files into a ~24 MB pool, comfortably resident),
/// readahead might help warm-token cache hits. The C path
/// applied F_RDAHEAD=0 unconditionally; we gate it because the
/// Rust port spans a wider variant range than the C path was
/// tuned for, and the disable was speculative for Cogito too —
/// not driven by profiling on either model.
///
/// **Empirical note (2026-05-01):** A/B'd Qwen at max_tokens=512
/// with and without this disable. Result was perf-neutral
/// (1.7822 vs 1.7805 tok/s). Suspected as a regression vector
/// vs the 1.96 baseline; turned out not to be. Gate stays as a
/// documented variant difference; the rationale above remains
/// theoretical and untested for Cogito.
#[cfg(feature = "model-cogito-v2-671b")]
fn disable_readahead(file: &File) {
    // SAFETY: `file.as_raw_fd()` is owned by `file` for the duration
    // of this call; `F_RDAHEAD` with arg=0 has no failure modes that
    // affect correctness — if it errors, readahead stays at default
    // and we eat a tiny perf loss. Log for diagnostics but don't
    // propagate.
    #[cfg(target_os = "macos")]
    unsafe {
        let rc = libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 0i32);
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            eprintln!("[experts] fcntl(F_RDAHEAD, 0) failed: {err}");
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = file;
    }
}

/// No-op for variants whose expert working set fits in RAM with
/// headroom — kernel readahead is helpful there. See the
/// cogito-gated sibling for the full rationale.
#[cfg(not(feature = "model-cogito-v2-671b"))]
fn disable_readahead(_file: &File) {}

/// Errors from expert-blob I/O.
#[derive(Debug, thiserror::Error)]
pub enum ExpertIoError {
    #[error("layer_idx {layer} out of range (must be < {num_layers})")]
    BadLayerIdx { layer: i32, num_layers: usize },
    #[error("expert_idx {expert} out of range (must be < {num_experts})")]
    BadExpertIdx { expert: i32, num_experts: usize },
    #[error("layer {layer} file not opened (probably missing on disk)")]
    LayerFileMissing { layer: usize },
    #[error(
        "expert blob read short: expected {expected} bytes at \
         layer={layer} expert={expert}, got {actual}"
    )]
    ShortRead {
        layer: usize,
        expert: usize,
        expected: usize,
        actual: usize,
    },
    #[error("out buffer must be EXPERT_SIZE={expected} bytes, got {actual}")]
    BadOutLen { expected: usize, actual: usize },
    #[error(
        "fast-dir copy of layer {layer} is {fast} bytes but the primary file is \
         {primary} bytes (stale or partial copy?)"
    )]
    FastCopyMismatch { layer: usize, fast: u64, primary: u64 },
    #[error(
        "layer {layer} holds {bits}-bit experts but layer {first_layer} holds \
         {first_bits}-bit ones (mixed experts directory?)"
    )]
    MixedLayouts { layer: usize, bits: u32, first_layer: usize, first_bits: u32 },
    #[error("I/O error reading layer {layer}: {source}")]
    Io {
        layer: usize,
        #[source]
        source: io::Error,
    },
}

// Prefill always uses the mmap path — the 2026-05-20 measurement
// (see `.claude/memory/pread_teardown_landed.md`) showed mmap-direct
// matched pread on wall-clock at 36% less main-thread CPU on a
// workload that was already GPU-bound. The synchronous `pread +
// pool.upload_at` loop the old `ExpertIoMode::Pread` mode used at
// prefill is dead and should not be revived.
//
// Decode is a different story: on variants whose expert working set
// is larger than the page cache (Qwen3-A17B's 60 layers × 512 experts
// don't fit in M2 Max's 96 GB), mmap-direct stalls the GPU on
// demand-fault VM activity. The decode-time pread+prefetch path is
// gated runtime by [`crate::riir::io::expert_io_mode::select`] —
// `auto` (the default) flips to pread when
// `expert_bytes > 0.75 * physical_ram`; `STREAMMOE_DECODE_IO=mmap|pread`
// forces a choice. The state machine lives in [`super::prefetch`];
// the dispatch lives in
// `attn::linear_attn_forward::moe_dispatch_per_token`.

/// Per-layer file handles for the active variant. Slot `i` is `None`
/// if `experts_dir/packed_experts/layer_<i>.bin` was missing at
/// open time — the C path tolerates missing files (zeroes expert
/// outputs in `fused_layer_forward`); we mirror that semantic.
///
/// **Drop order:** `mmap_buffers` (Metal `Buffer`s wrapping the
/// mmap'd regions, deallocator `None`) must drop before
/// `mmap_layers` (the `Mmap`s that own the kernel mappings). Fields
/// drop in declaration order, so `mmap_buffers` is declared above
/// `mmap_layers`. In practice both live for the whole session.
pub struct ExpertFiles {
    /// Pin every `mmap_buffers` allocation as GPU-resident across
    /// cmdbuf boundaries via `MTLResidencySet` (macOS 15+). Populated
    /// by [`Self::attach_to_device`]. `None` if the residency-set API
    /// isn't available on this OS or no buffers were attached yet.
    ///
    /// **Declared first** so it drops first — `endResidency` runs
    /// while the registered allocations are still alive.
    expert_rset: Option<ResidencySet>,
    /// `Some(file)` if the layer file opened, `None` if it was missing.
    layers: Vec<Option<File>>,
    /// Packed layout of the opened files ([`layout_of_len`]); every
    /// opened layer agrees.
    layout: ExpertLayout,
    /// Bytes per expert blob: `layout.expert_size()`.
    expert_size: usize,
    /// Directory the files were opened relative to. Kept for
    /// diagnostics / debug-impl; never re-read.
    experts_dir: PathBuf,
    /// Per-layer Metal buffer wrapping the mmap'd region via
    /// `newBufferWithBytesNoCopy`. Populated by `attach_to_device`
    /// once the Metal backend exists. `None` for missing layers.
    /// Declared before `mmap_layers` so it drops first.
    mmap_buffers: Vec<Option<Buffer>>,
    /// S10b-pre-2 — per-layer `BufId` parallel to `mmap_buffers`.
    /// Populated by [`Self::attach_to_device`] via
    /// `MetalBufferPool::register_borrowed`. Lets graph-mode `Op`s
    /// (`Op::MoeBatchedPermuteFuse`) address mmap'd expert data by
    /// `BufId` rather than `&Buffer`. The pool holds a refcounted
    /// clone of the same `Buffer` that lives in `mmap_buffers`;
    /// drop order is irrelevant (BufId is `Copy`).
    mmap_buf_ids: Vec<Option<BufId<ExpertBaseBuf>>>,
    /// Per-layer mmap. Populated at `open` time when `mode == Mmap`.
    /// `None` for the Pread mode and for missing layers.
    mmap_layers: Vec<Option<Mmap>>,
    /// Optional RAM cache of hot experts ([`Self::attach_hot_tier`]).
    hot: Option<super::hot_tier::HotTier>,
    /// Routing popularity, `layer * num_experts + expert`, never
    /// decayed — the source for [`Self::routing_counts`] profiles.
    picks: Vec<AtomicU32>,
    /// Streaming telemetry (reads by kind, per layer, timings,
    /// guess accuracy) — see [`super::io_stats`].
    counters: IoCounters,
    /// Queue-depth and bytes/s limits every disk read passes through
    /// ([`Self::set_io_limits`]).
    limits: IoLimits,
    /// Per layer: `true` if the file was opened from the fast dir
    /// ([`Self::open_range_tiered`]).
    fast: Vec<bool>,
}

/// Layout of a layer file `len` bytes long: the one whose
/// `num_experts` experts fill it exactly, else 4-bit (the legacy
/// assumption; reads past a short file still fail as short reads).
pub fn layout_of_len(len: u64) -> ExpertLayout {
    let n = VARIANT.num_experts as u64;
    if len % n == 0 {
        ExpertLayout::from_expert_size(len / n).unwrap_or(ExpertLayout::Q4)
    } else {
        ExpertLayout::Q4
    }
}

/// Layout of the experts under `experts_dir/packed_experts/`, from the
/// first layer file in `range` present on disk (4-bit if none is).
/// [`ExpertFiles::open_range_tiered`] checks that the rest agree.
pub fn detect_layout(experts_dir: &Path, range: std::ops::Range<usize>) -> ExpertLayout {
    let dir = experts_dir.join("packed_experts");
    range
        .filter_map(|i| std::fs::metadata(dir.join(format!("layer_{i:02}.bin"))).ok())
        .map(|m| layout_of_len(m.len()))
        .next()
        .unwrap_or(ExpertLayout::Q4)
}

pub use super::io_stats::ExpertIoStats;
use super::io_limit::IoLimits;
use super::io_stats::{IoCounters, ReadKind};

impl ExpertFiles {
    /// Open `experts_dir/packed_experts/layer_NN.bin` for every layer
    /// in the active variant. Missing files leave the slot at `None`.
    /// I/O errors other than `NotFound` propagate.
    ///
    /// Mirrors the C loop at `metal_infer/infer.m:7580..7605`. Each
    /// opened layer file is mmap'd unconditionally; the matching
    /// Metal buffer is built lazily by [`Self::attach_to_device`]
    /// once the Metal backend exists. The mmap is always built
    /// because prefill always reads through it; the per-token
    /// decode path additionally pread-stages experts when the
    /// runtime gate picks pread mode (see
    /// [`crate::riir::io::expert_io_mode::select`]).
    pub fn open(experts_dir: &Path) -> Result<Self, ExpertIoError> {
        Self::open_with(experts_dir, true, false)
    }

    /// [`Self::open`] with explicit page-cache behavior.
    ///
    /// - `mmap = false` skips the per-layer mmaps; only valid when the
    ///   expert-IO mode is `Pread` (every read goes through
    ///   [`Self::read_expert`]).
    /// - `nocache = true` sets `F_NOCACHE` on every layer fd, so expert
    ///   reads never grow the page cache. Used under a hard memory cap.
    pub fn open_with(experts_dir: &Path, mmap: bool, nocache: bool) -> Result<Self, ExpertIoError> {
        Self::open_range(experts_dir, mmap, nocache, 0..VARIANT.num_layers)
    }

    /// [`Self::open_with`] opening only the layer files in `range`
    /// (plan phase 3). Slots outside it stay `None`, exactly like a
    /// missing file; nothing in this context dispatches them.
    pub fn open_range(
        experts_dir: &Path,
        mmap: bool,
        nocache: bool,
        range: std::ops::Range<usize>,
    ) -> Result<Self, ExpertIoError> {
        Self::open_range_tiered(experts_dir, None, mmap, nocache, range)
    }

    /// [`Self::open_range`] with an optional fast tier (plan phase 6,
    /// "disk tier"): a layer file present under
    /// `fast_dir/packed_experts/` (e.g. a copy on the internal SSD) is
    /// opened from there instead of `experts_dir`. Same file format,
    /// per-layer granularity. A fast copy whose size differs from the
    /// primary file is rejected ([`ExpertIoError::FastCopyMismatch`]).
    pub fn open_range_tiered(
        experts_dir: &Path,
        fast_dir: Option<&Path>,
        mmap: bool,
        nocache: bool,
        range: std::ops::Range<usize>,
    ) -> Result<Self, ExpertIoError> {
        let v: Variant = VARIANT;
        let subdir = experts_dir.join("packed_experts");
        let fast_subdir = fast_dir.map(|d| d.join("packed_experts"));
        let mut layers = Vec::with_capacity(v.num_layers);
        let mut fast = vec![false; v.num_layers];
        let mut mmap_layers: Vec<Option<Mmap>> = (0..v.num_layers).map(|_| None).collect();
        // First opened layer's layout; the rest must match it.
        let mut layout: Option<(usize, ExpertLayout)> = None;
        for i in 0..v.num_layers {
            if !range.contains(&i) {
                layers.push(None);
                continue;
            }
            let name = format!("layer_{i:02}.bin");
            let mut path = subdir.join(&name);
            if let Some(fast_path) = fast_subdir.as_ref().map(|d| d.join(&name)) {
                if let Ok(fm) = std::fs::metadata(&fast_path) {
                    if let Ok(pm) = std::fs::metadata(&path) {
                        if pm.len() != fm.len() {
                            return Err(ExpertIoError::FastCopyMismatch {
                                layer: i,
                                fast: fm.len(),
                                primary: pm.len(),
                            });
                        }
                    }
                    path = fast_path;
                    fast[i] = true;
                }
            }
            match File::open(&path) {
                Ok(f) => {
                    let len = f.metadata().map_err(|e| ExpertIoError::Io { layer: i, source: e })?.len();
                    let this = layout_of_len(len);
                    match layout {
                        None => layout = Some((i, this)),
                        Some((first_layer, first)) if first != this => {
                            return Err(ExpertIoError::MixedLayouts {
                                layer: i,
                                bits: this.bits,
                                first_layer,
                                first_bits: first.bits,
                            });
                        }
                        Some(_) => {}
                    }
                    disable_readahead(&f);
                    if nocache {
                        crate::riir::io::weight_file::set_nocache(&f)
                            .map_err(|e| ExpertIoError::Io { layer: i, source: e })?;
                    }
                    if mmap {
                        // SAFETY: file is owned by the closure for the
                        // duration of the mmap call; the resulting mapping
                        // survives the file handle (closed when `f` drops
                        // at end of this match arm). Returned mapping is
                        // read-only, no aliasing concerns.
                        let m = unsafe { Mmap::map(&f) }.map_err(|e| ExpertIoError::Io {
                            layer: i,
                            source: e,
                        })?;
                        mmap_layers[i] = Some(m);
                    }
                    layers.push(Some(f));
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    layers.push(None);
                }
                Err(e) => {
                    return Err(ExpertIoError::Io {
                        layer: i,
                        source: e,
                    });
                }
            }
        }
        let mmap_buffers = (0..v.num_layers).map(|_| None).collect();
        let mmap_buf_ids = (0..v.num_layers).map(|_| None).collect();
        let layout = layout.map_or(ExpertLayout::Q4, |(_, l)| l);
        eprintln!(
            "[experts] {} layer(s) opened ({} from the fast dir), {}-bit, {} mmap'd{} \
             (Metal buffers built on first batched call)",
            layers.iter().filter(|f| f.is_some()).count(),
            fast.iter().filter(|&&f| f).count(),
            layout.bits,
            mmap_layers.iter().filter(|m| m.is_some()).count(),
            if nocache { ", F_NOCACHE" } else { "" }
        );
        Ok(Self {
            expert_rset: None,
            layers,
            layout,
            expert_size: layout.expert_size(),
            experts_dir: experts_dir.to_path_buf(),
            mmap_buffers,
            mmap_buf_ids,
            mmap_layers,
            hot: None,
            picks: (0..v.num_layers * v.num_experts).map(|_| AtomicU32::new(0)).collect(),
            counters: IoCounters::new(v.num_layers),
            limits: IoLimits::default(),
            fast,
        })
    }

    /// Wrap each mmap'd layer as a Metal-shared buffer via
    /// `newBufferWithBytesNoCopy`, and register the buffer with the
    /// graph-mode buffer pool. Idempotent: the second call is a no-op.
    /// Called once from `ensure_linear_resources` after the Metal
    /// backend is created.
    ///
    /// S10b-pre-2: parameter changed from `&Device` to
    /// `&mut MetalBufferPool`. The buffer is stored in both
    /// `mmap_buffers` (existing `&Buffer` accessor still works) AND
    /// `mmap_buf_ids` (new `BufId` accessor for graph-mode `Op`s);
    /// they share an NSObject-refcounted clone of the same backing.
    pub fn attach_to_device(
        &mut self,
        pool: &mut MetalBufferPool,
        mode: super::expert_io_mode::ExpertIoMode,
    ) {
        // Pread mode never has the GPU read from the per-layer mmap —
        // the prefill gather kernel and the decode batched-expert
        // dispatch both read from staging buffers populated by `pread`
        // (see `expert_io_mode.rs` for the rationale). Skipping the
        // `newBufferWithBytesNoCopy` + residency-pin keeps the GPU
        // off the giant working set; for a17b that's ~245 GB of
        // layer files which can't be pinned on M2 Max's 96 GB UMA
        // anyway, and trying to does measurable harm.
        if mode.is_pread() {
            return;
        }
        // Lazy-init the residency set on the first attach call. The
        // capacity hint is the layer count — a slight over-estimate
        // when some layers are missing, but it's a hint, not a cap.
        // Returns `None` on macOS < 15; we then attach buffers without
        // pinning.
        if self.expert_rset.is_none() {
            self.expert_rset = ResidencySet::new(
                pool.device(),
                "streammoe.experts",
                self.mmap_layers.len() as u64,
            );
            if self.expert_rset.is_none() && !streammoe_metal::residency_set::is_available() {
                eprintln!(
                    "[experts] MTLResidencySet unavailable (macOS < 15) \
                     — expert buffers will not be pinned"
                );
            }
        }
        let mut added_any = false;
        for i in 0..self.mmap_layers.len() {
            if self.mmap_buffers[i].is_some() {
                continue;
            }
            let Some(mmap) = self.mmap_layers[i].as_ref() else {
                continue;
            };
            let raw_len = mmap.len();
            // Page-align the length the way `MtlWeightBuf::wrap` does:
            // Metal accepts a length, the kernel maps in page-granular
            // units. The trailing padding (if any) past the file end
            // is OS-zero-filled and never indexed by kernels because
            // expert offsets bound at `(num_experts - 1) * expert_size
            // + expert_size <= raw_len`.
            let page = 16384usize;
            let aligned_len = (raw_len + page - 1) & !(page - 1);
            // SAFETY: `mmap.as_ptr()` is valid for `raw_len` bytes; the
            // trailing pad to `aligned_len` is kernel-zero-filled (mmap
            // contract). The `Mmap` outlives this `Buffer` because both
            // live on `ExpertFiles` and drop order puts `Buffer` first
            // (see struct docs). Deallocator `None` — Metal does NOT
            // own the mapping; `Mmap`'s drop unmaps after the Buffer
            // is released.
            let buf = pool.device().new_buffer_with_bytes_no_copy(
                mmap.as_ptr() as *const c_void,
                aligned_len as NSUInteger,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            // Register a refcounted clone with the pool. The pool's
            // BufId entry holds an NSObject retain; this `mmap_buffers`
            // slot holds another. Both die when `ExpertFiles` drops
            // (Buffer refcount → 0 → newBufferWithBytesNoCopy releases
            // the non-owning Metal reference), after which `Mmap`
            // drops and unmaps the file. Order is enforced by struct
            // field declaration order.
            let id = pool.register_borrowed(
                buf.clone(),
                aligned_len,
                "expert_io.mmap_layer",
                /* persistent = */ true,
            );
            if let Some(rset) = self.expert_rset.as_ref() {
                rset.add_allocation(&buf);
            }
            self.mmap_buffers[i] = Some(buf);
            self.mmap_buf_ids[i] = Some(id);
            added_any = true;
        }
        // Commit + requestResidency once at the end of the call, only
        // if we actually staged new allocations. The second
        // (idempotent) call to `attach_to_device` exits with
        // `added_any == false` and skips this — re-requesting on an
        // unchanged set is harmless but pointless.
        if added_any {
            if let Some(rset) = self.expert_rset.as_ref() {
                rset.commit();
                rset.request_residency();
            }
        }
    }

    /// Pool BufId for the mmap'd layer Buffer + expert byte offset.
    /// Mmap-mode sibling of [`Self::mmap_buffer_for_expert`]. Returns
    /// `None` in Pread mode and for missing layers.
    ///
    /// Graph-mode `Op::MoeBatchedPermuteFuse` uses this to address
    /// the layer's mmap buffer directly as its `expert_base` (the
    /// layer mmap is already `num_experts` blocks at `expert_size`
    /// stride, so `expert_idx = 0` gives the base).
    pub fn mmap_id_for_expert(
        &self,
        layer_idx: usize,
        expert_idx: u32,
    ) -> Option<(BufId<ExpertBaseBuf>, u64)> {
        let id = self.mmap_buf_ids.get(layer_idx)?.as_ref()?;
        let off = (expert_idx as u64) * (self.expert_size as u64);
        Some((*id, off))
    }

    /// Returns `Some((layer_buf, byte_offset))` for `(layer_idx,
    /// expert_idx)` when the layer file opened. Returns `None` for
    /// missing layers (caller treats the expert output as zeros, per
    /// the C-side `fused_layer_forward` semantic).
    pub fn mmap_buffer_for_expert(
        &self,
        layer_idx: usize,
        expert_idx: u32,
    ) -> Option<(&Buffer, u64)> {
        let buf = self.mmap_buffers.get(layer_idx)?.as_ref()?;
        let off = (expert_idx as u64) * (self.expert_size as u64);
        Some((buf, off))
    }

    /// Read one expert's `EXPERT_SIZE` bytes into `out`. `out.len()`
    /// must equal `expert_size`.
    ///
    /// Equivalent to the C `pread(layer_fds[layer_idx], out,
    /// EXPERT_SIZE, expert_idx * EXPERT_SIZE)` call site at
    /// `infer.m:2915`.
    pub fn read_expert(
        &self,
        layer_idx: usize,
        expert_idx: usize,
        out: &mut [u8],
    ) -> Result<(), ExpertIoError> {
        self.read_expert_as(ReadKind::Demand, layer_idx, expert_idx, out)
    }

    /// [`Self::read_expert`] attributing a disk read (or, for
    /// [`ReadKind::Prefill`], a hot-tier copy) to `kind`.
    pub fn read_expert_as(
        &self,
        kind: ReadKind,
        layer_idx: usize,
        expert_idx: usize,
        out: &mut [u8],
    ) -> Result<(), ExpertIoError> {
        self.read_expert_checks(layer_idx, expert_idx, out)?;
        if let Some(hot) = &self.hot {
            if hot.copy_out(layer_idx, expert_idx, out) {
                if kind == ReadKind::Prefill {
                    self.counters.record_prefill_hot(layer_idx, 1);
                }
                return Ok(());
            }
        }
        self.read_expert_disk_as(kind, layer_idx, expert_idx, out)
    }

    /// [`Self::read_expert`] without the hot-tier lookup: always reads
    /// from disk, then offers the bytes to the hot tier. For callers
    /// that already looked the expert up (decode dispatch) or read
    /// speculatively (prefetch). Same argument checks.
    pub fn read_expert_disk(
        &self,
        layer_idx: usize,
        expert_idx: usize,
        out: &mut [u8],
    ) -> Result<(), ExpertIoError> {
        self.read_expert_disk_as(ReadKind::Demand, layer_idx, expert_idx, out)
    }

    /// [`Self::read_expert_disk`] attributing the read to `kind` in
    /// the telemetry.
    pub fn read_expert_disk_as(
        &self,
        kind: ReadKind,
        layer_idx: usize,
        expert_idx: usize,
        out: &mut [u8],
    ) -> Result<(), ExpertIoError> {
        self.read_expert_checks(layer_idx, expert_idx, out)?;
        let Some(file) = self.layers[layer_idx].as_ref() else {
            return Err(ExpertIoError::LayerFileMissing { layer: layer_idx });
        };
        let off = (expert_idx as u64) * (self.expert_size as u64);
        let permit = self.limits.admit(self.expert_size as u64);
        let tr0 = super::io_trace::now();
        let n = file.read_at(out, off).map_err(|e| ExpertIoError::Io {
            layer: layer_idx,
            source: e,
        })?;
        drop(permit);
        super::io_trace::read(tr0, kind, layer_idx, 1, n);
        if n != self.expert_size {
            return Err(ExpertIoError::ShortRead {
                layer: layer_idx,
                expert: expert_idx,
                expected: self.expert_size,
                actual: n,
            });
        }
        self.counters.record_read(kind, layer_idx, 1, n as u64, self.fast[layer_idx]);
        if let Some(hot) = &self.hot {
            hot.offer(layer_idx, expert_idx, out);
        }
        Ok(())
    }

    /// Read experts `first..first + n` of `layer_idx` from disk in one
    /// call into `out` (`n * expert_size` bytes; the file stores a
    /// layer's experts contiguously, so a run of ids is one range),
    /// then offer each to the hot tier. Counted as `n` reads of
    /// `kind`, and as one read against the queue-depth gate.
    pub fn read_run_disk(
        &self,
        kind: ReadKind,
        layer_idx: usize,
        first: usize,
        n: usize,
        out: &mut [u8],
    ) -> Result<(), ExpertIoError> {
        let es = self.expert_size;
        if n == 0 || out.len() != n * es {
            return Err(ExpertIoError::BadOutLen {
                expected: n * es,
                actual: out.len(),
            });
        }
        self.read_expert_checks(layer_idx, first + n - 1, &out[..es])?;
        let Some(file) = self.layers[layer_idx].as_ref() else {
            return Err(ExpertIoError::LayerFileMissing { layer: layer_idx });
        };
        let off = (first as u64) * (es as u64);
        let permit = self.limits.admit(out.len() as u64);
        let tr0 = super::io_trace::now();
        let mut done = 0usize;
        while done < out.len() {
            let got = file
                .read_at(&mut out[done..], off + done as u64)
                .map_err(|e| ExpertIoError::Io {
                    layer: layer_idx,
                    source: e,
                })?;
            if got == 0 {
                return Err(ExpertIoError::ShortRead {
                    layer: layer_idx,
                    expert: first + done / es,
                    expected: out.len(),
                    actual: done,
                });
            }
            done += got;
        }
        drop(permit);
        super::io_trace::read(tr0, kind, layer_idx, n, out.len());
        self.counters
            .record_read(kind, layer_idx, n as u64, out.len() as u64, self.fast[layer_idx]);
        if let Some(hot) = &self.hot {
            for (i, blob) in out.chunks_exact(es).enumerate() {
                hot.offer(layer_idx, first + i, blob);
            }
        }
        Ok(())
    }

    /// Prefill staging for one layer (plan phase 6): put every expert
    /// in `expert_ids` at its `expert_id * expert_size` slot of
    /// `staging` (the pread-mode `expert_base`). Hot-tier experts are
    /// copied from RAM; the rest are read once each, sorted by file
    /// offset and coalesced into runs of up to [`PREFILL_MAX_RUN`]
    /// consecutive experts, in parallel on `pool`. Bytes outside the
    /// listed slots are left untouched.
    pub fn stage_layer(
        &self,
        pool: &rayon::ThreadPool,
        layer_idx: usize,
        expert_ids: &[i32],
        staging: &mut [u8],
    ) -> Result<(), ExpertIoError> {
        use rayon::prelude::*;
        let es = self.expert_size;
        let num_experts = VARIANT.num_experts;
        if staging.len() < num_experts * es {
            return Err(ExpertIoError::BadOutLen {
                expected: num_experts * es,
                actual: staging.len(),
            });
        }
        for &e in expert_ids {
            if e < 0 || e as usize >= num_experts {
                return Err(ExpertIoError::BadExpertIdx {
                    expert: e,
                    num_experts,
                });
            }
        }
        let hot = self.hot.as_ref();
        let tasks = plan_staging(
            expert_ids,
            |e| hot.is_some_and(|h| h.contains(layer_idx, e)),
            PREFILL_MAX_RUN,
        );
        // Disjoint destination slices, in ascending offset order.
        let mut work: Vec<(StageTask, &mut [u8])> = Vec::with_capacity(tasks.len());
        let mut rest: &mut [u8] = staging;
        let mut base = 0usize;
        for t in tasks {
            let start = t.first as usize * es;
            let tail = std::mem::take(&mut rest).split_at_mut(start - base).1;
            let (dst, tail) = tail.split_at_mut(t.n as usize * es);
            work.push((t, dst));
            rest = tail;
            base = start + t.n as usize * es;
        }
        pool.install(|| {
            work.par_iter_mut().try_for_each(|(t, dst)| {
                let first = t.first as usize;
                if t.hot {
                    if hot.is_some_and(|h| h.copy_out(layer_idx, first, dst)) {
                        self.counters.record_prefill_hot(layer_idx, 1);
                        return Ok(());
                    }
                    // Evicted since planning: read it instead.
                }
                self.read_run_disk(ReadKind::Prefill, layer_idx, first, t.n as usize, dst)
            })
        })
    }

    /// Argument validation shared by the read paths; returns the
    /// error for the first failing check (`Ok` only if all pass).
    fn read_expert_checks(
        &self,
        layer_idx: usize,
        expert_idx: usize,
        out: &[u8],
    ) -> Result<(), ExpertIoError> {
        let v: Variant = VARIANT;
        if layer_idx >= v.num_layers {
            return Err(ExpertIoError::BadLayerIdx {
                layer: layer_idx as i32,
                num_layers: v.num_layers,
            });
        }
        if expert_idx >= v.num_experts {
            return Err(ExpertIoError::BadExpertIdx {
                expert: expert_idx as i32,
                num_experts: v.num_experts,
            });
        }
        if out.len() != self.expert_size {
            return Err(ExpertIoError::BadOutLen {
                expected: self.expert_size,
                actual: out.len(),
            });
        }
        Ok(())
    }

    /// Attach a hot tier. Subsequent [`Self::read_expert`] calls are
    /// served from it on a hit and offer misses for admission.
    pub fn attach_hot_tier(&mut self, hot: super::hot_tier::HotTier) {
        self.hot = Some(hot);
    }

    /// The attached hot tier, if any.
    pub fn hot_tier(&self) -> Option<&super::hot_tier::HotTier> {
        self.hot.as_ref()
    }

    /// Record a routed pick of `(layer, expert)` — feeds the routing
    /// profile and the hot tier's popularity counters.
    pub fn record_pick(&self, layer: usize, expert: usize, n: u32) {
        let v = VARIANT;
        if layer >= v.num_layers || expert >= v.num_experts {
            return;
        }
        self.picks[layer * v.num_experts + expert].fetch_add(n, Ordering::Relaxed);
        if let Some(hot) = &self.hot {
            hot.record_picks(layer, expert, n);
        }
    }

    /// Routing popularity since open, `layer * num_experts + expert`.
    pub fn routing_counts(&self) -> Vec<u32> {
        self.picks.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    /// Disk, timing, guess and hot-tier counters.
    pub fn io_stats(&self) -> ExpertIoStats {
        self.counters.snapshot(self.hot.as_ref().map(|h| h.stats()))
    }

    /// Bound every subsequent disk read by `limits` (plan §3 rule 6).
    pub fn set_io_limits(&mut self, limits: IoLimits) {
        self.limits = limits;
    }

    /// The telemetry store, for the forward paths to record into.
    pub fn counters(&self) -> &IoCounters {
        &self.counters
    }

    /// Zero the telemetry and the hot tier's counters (routing
    /// popularity is kept: it drives admission and profiles).
    pub fn reset_io_stats(&self) {
        self.counters.reset();
        if let Some(hot) = &self.hot {
            hot.reset_stats();
        }
    }

    /// Mark a new token for the hot tier's GPU-hazard rule.
    pub fn begin_epoch(&self) {
        if let Some(hot) = &self.hot {
            hot.begin_epoch();
        }
    }

    /// Number of layers (matches [`Variant::num_layers`]).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Packed layout of the opened expert files.
    pub fn layout(&self) -> ExpertLayout {
        self.layout
    }

    /// Bytes per expert blob (`layout().expert_size()`).
    pub fn expert_size(&self) -> usize {
        self.expert_size
    }

    /// `true` iff `layer_idx`'s file was opened from the fast dir.
    pub fn is_fast_layer(&self, layer_idx: usize) -> bool {
        self.fast.get(layer_idx).copied().unwrap_or(false)
    }

    /// `true` iff the file for `layer_idx` opened successfully.
    pub fn has_layer(&self, layer_idx: usize) -> bool {
        self.layers
            .get(layer_idx)
            .map(Option::is_some)
            .unwrap_or(false)
    }
}

/// Longest run of consecutive experts one prefill read covers
/// (~14 MB for a3b). Long enough to amortize syscall and SSD command
/// overhead, short enough to spread a layer over the I/O threads.
pub const PREFILL_MAX_RUN: usize = 8;

/// One unit of prefill staging: experts `first..first + n` of a
/// layer, copied from the hot tier (`hot`, then `n == 1`) or read
/// from disk in one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageTask {
    /// First expert id.
    pub first: u32,
    /// Experts covered.
    pub n: u32,
    /// Copy from the hot tier instead of reading.
    pub hot: bool,
}

/// Plan prefill staging for `expert_ids` (any order, duplicates
/// allowed): each distinct expert appears in exactly one task, tasks
/// ascend by expert id (= file offset), hot experts get their own
/// copy task, and consecutive non-hot experts merge into disk runs of
/// at most `max_run`. Pure, so it is unit-testable without files.
pub fn plan_staging(
    expert_ids: &[i32],
    is_hot: impl Fn(usize) -> bool,
    max_run: usize,
) -> Vec<StageTask> {
    let mut ids: Vec<u32> = expert_ids.iter().filter(|&&e| e >= 0).map(|&e| e as u32).collect();
    ids.sort_unstable();
    ids.dedup();
    let max_run = max_run.max(1) as u32;
    let mut tasks: Vec<StageTask> = Vec::new();
    for e in ids {
        if is_hot(e as usize) {
            tasks.push(StageTask { first: e, n: 1, hot: true });
            continue;
        }
        match tasks.last_mut() {
            Some(t) if !t.hot && t.first + t.n == e && t.n < max_run => t.n += 1,
            _ => tasks.push(StageTask { first: e, n: 1, hot: false }),
        }
    }
    tasks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dirs(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!("streammoe-tier-{tag}-{}", std::process::id()));
        let (p, f) = (root.join("primary"), root.join("fast"));
        for d in [&p, &f] {
            std::fs::create_dir_all(d.join("packed_experts")).unwrap();
        }
        (p, f)
    }

    fn sparse_layer(dir: &Path, layer: usize, len: u64) {
        let path = dir.join("packed_experts").join(format!("layer_{layer:02}.bin"));
        // Sparse: set_len reserves no disk blocks.
        File::create(path).unwrap().set_len(len).unwrap();
    }

    #[test]
    fn fast_dir_overrides_per_layer() {
        let v = VARIANT;
        let full = (v.num_experts * v.expert_size_4bit()) as u64;
        let (primary, fast) = scratch_dirs("ok");
        sparse_layer(&primary, 0, full);
        sparse_layer(&primary, 1, full);
        sparse_layer(&fast, 1, full);
        // A fast copy alone (no primary) is used too.
        sparse_layer(&fast, 2, full);
        let ef = ExpertFiles::open_range_tiered(&primary, Some(&fast), false, false, 0..3).unwrap();
        assert!(ef.has_layer(0) && ef.has_layer(1) && ef.has_layer(2));
        assert_eq!(
            (ef.is_fast_layer(0), ef.is_fast_layer(1), ef.is_fast_layer(2)),
            (false, true, true)
        );
        let mut buf = vec![1u8; v.expert_size_4bit()];
        ef.read_expert_disk(1, 3, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
        ef.read_expert_disk(0, 3, &mut buf).unwrap();
        let st = ef.io_stats();
        assert_eq!(st.demand.reads, 2);
        assert_eq!(st.fast_dir_bytes, v.expert_size_4bit() as u64);
        let _ = std::fs::remove_dir_all(primary.parent().unwrap());
    }

    #[test]
    fn fast_dir_size_mismatch_rejected() {
        let v = VARIANT;
        let full = (v.num_experts * v.expert_size_4bit()) as u64;
        let (primary, fast) = scratch_dirs("bad");
        sparse_layer(&primary, 0, full);
        sparse_layer(&fast, 0, full / 2);
        let err = ExpertFiles::open_range_tiered(&primary, Some(&fast), false, false, 0..1)
            .err()
            .expect("must fail");
        assert!(matches!(err, ExpertIoError::FastCopyMismatch { layer: 0, .. }), "{err}");
        let _ = std::fs::remove_dir_all(primary.parent().unwrap());
    }

    #[test]
    fn plan_staging_sorts_dedupes_and_coalesces() {
        let ids = [7, 3, 4, 5, 3, 9, 6, 20, 21];
        let tasks = plan_staging(&ids, |_| false, 8);
        let t = |first, n| StageTask { first, n, hot: false };
        assert_eq!(tasks, vec![t(3, 5), t(9, 1), t(20, 2)]);
    }

    #[test]
    fn plan_staging_splits_runs_at_hot_experts_and_max_run() {
        let ids: Vec<i32> = (0..12).collect();
        let tasks = plan_staging(&ids, |e| e == 4, 3);
        let d = |first, n| StageTask { first, n, hot: false };
        let h = |first| StageTask { first, n: 1, hot: true };
        assert_eq!(tasks, vec![d(0, 3), d(3, 1), h(4), d(5, 3), d(8, 3), d(11, 1)]);
        // Every expert exactly once.
        let covered: u32 = tasks.iter().map(|t| t.n).sum();
        assert_eq!(covered, 12);
        assert!(plan_staging(&[], |_| false, 8).is_empty());
        assert!(plan_staging(&[-1], |_| false, 8).is_empty());
    }
}

impl std::fmt::Debug for ExpertFiles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let opened = self.layers.iter().filter(|s| s.is_some()).count();
        f.debug_struct("ExpertFiles")
            .field("experts_dir", &self.experts_dir)
            .field("num_layers", &self.layers.len())
            .field("opened", &opened)
            .field("expert_size", &self.expert_size)
            .finish()
    }
}
