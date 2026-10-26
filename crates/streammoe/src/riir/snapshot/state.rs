//! Per-layer KV / recurrent state for the forward pass (Phase 4a).
//!
//! Mirrors the C-side `KVCache` + `LinearAttnState` allocations and the
//! `mf_state_*` truncation primitives from `metal_infer/infer.m`. The
//! Rust port owns these as fields on [`crate::riir::RsCtx`] instead of
//! addressing them through `KVCache **kv_caches` / `void **layer_states`
//! arrays — Rust enums replace the parallel-array + `is_full` modulo
//! pattern.
//!
//! ## Faithful-port note
//!
//! [`truncate`] preserves the C-side "Option A" semantic: any partial
//! truncation of a linear-attn layer resets that layer's recurrence
//! state to empty. This is lossy (the `(conv_state, ssm_state)` pair
//! folds the entire history; you can't unwind it). The bisect findings
//! flagged this as a bug source — drama_llama's prefix-cache reuse
//! silently diverges when truncating a position inside a linear-attn
//! span. The faithful port keeps the lossy semantic and the typed
//! `Result<(), CannotTruncateLinear>` lands as a Phase 7 post-cutover
//! slice, per `riir_streammoe_strategy.md`.
//!
//! FIXME(riir): lossy partial-linear truncation; faithful port of
//! `infer.m:2291 mf_state_truncate`. Phase 7 introduces the typed
//! error; in the meantime the call still resets to empty.

use metal::{Buffer, Device, MTLResourceOptions, NSUInteger};

use crate::riir::attn::kv_quant::KvFormat;
use crate::riir::backend::buftype::{KvCacheKBuf, KvCacheVBuf};
use crate::riir::backend::gpu::MetalBufferPool;
use crate::riir::backend::{BufId, BufferPool};
use crate::riir::range::RangeVec;
use crate::riir::variants::{MAX_SEQ_LEN, VARIANT, Variant};

/// Clamp a requested context length to `[1, MAX_SEQ_LEN]`. Every KV
/// cache is sized to the runtime `n_ctx` (see
/// [`crate::riir::OpenConfig::n_ctx`]); `MAX_SEQ_LEN` stays the
/// ceiling the RoPE configuration supports.
pub fn clamp_n_ctx(n_ctx: usize) -> usize {
    n_ctx.clamp(1, MAX_SEQ_LEN)
}

/// Full-attention key/value cache for one layer. **GPU-resident**:
/// `k_id` and `v_id` are pool [`BufId`]s for `MTLResourceStorageMode
/// Shared` Metal buffers of shape `[capacity, num_kv_heads *
/// head_dim]` row-major f32, where `capacity` is the context's
/// runtime `n_ctx`. Shared storage means the CPU full-attn
/// path reads/writes the same bytes via `contents()` while the GPU
/// SDPA path reads them directly — no host mirror, no per-layer
/// re-upload (the prefill arc's Phase 0: the prior `Box<[f32]>` host
/// cache forced a `MtlBuffer::with_data` of the whole growing prefix
/// every layer).
///
/// Prefill-arc Phase 0b: the buffers live in the [`MetalBufferPool`]
/// like every other graph-mode allocation, so the SDPA `Op` can
/// address them by `BufId`. They are *registered* (not `pool.alloc`'d)
/// because `pool.alloc` zero-fills — touching every page of a multi-GB
/// reservation up front, the exact cold-init memset the MLA path
/// fought to drop. `register_borrowed` keeps the Mach-VM lazy-commit:
/// untouched rows cost no physical RAM. They are `persistent = true`
/// so `commit_plan` lifetime-coloring never aliases them.
///
/// `BufId`s are `Option`-wrapped for lazy allocation: `LayerState`s
/// are constructed at `Ctx::open` before the pool exists;
/// [`Self::ensure_buffers`] populates them once, eagerly, from
/// `ensure_linear_resources` (the one site holding `&mut pool`).
#[derive(Debug)]
pub struct KvCache {
    /// Pool `BufId` for the `[capacity, num_kv_heads * head_dim]`
    /// row-major f32 K cache. `None` until [`Self::ensure_buffers`].
    pub k_id: Option<BufId<KvCacheKBuf>>,
    pub v_id: Option<BufId<KvCacheVBuf>>,
    pub len: i32,
    /// Maximum number of positions the buffers hold (the context's
    /// `n_ctx`). Appends past it fail with a cache-overflow error.
    pub capacity: usize,
    /// Row format of `k_id` / `v_id` ([`crate::riir::OpenConfig::kv_format`]).
    /// Non-F32 rows are `fmt.row_bytes` each; the f32 slice accessors
    /// are F32-only.
    pub fmt: KvFormat,
    /// Non-F32 only: the f32 `[capacity, kv_dim]` attention scratch
    /// batched prefill dequantizes the prefix into for
    /// `SdpaCausalTiled`. One pair shared by every full-attn layer
    /// (layers run in order), set by `ensure_linear_resources`.
    pub f32_scratch: Option<(BufId<KvCacheKBuf>, BufId<KvCacheVBuf>)>,
}

impl KvCache {
    /// Construct an empty cache without GPU buffers. Buffers are
    /// allocated lazily by [`Self::ensure_buffers`] once the pool
    /// exists.
    pub fn new(capacity: usize, fmt: KvFormat) -> Self {
        Self {
            k_id: None,
            v_id: None,
            len: 0,
            capacity: clamp_n_ctx(capacity),
            fmt,
            f32_scratch: None,
        }
    }

    /// Bytes per side (K or V) at this cache's capacity.
    pub fn bytes_per_side(&self) -> usize {
        self.capacity * self.fmt.row_bytes(&VARIANT)
    }

    /// Stored bytes of K rows `[start_row, end_row)` (any format).
    ///
    /// # Safety
    ///
    /// See [`Self::k_slice`].
    pub unsafe fn k_rows_mut<'p>(
        &self,
        pool: &'p MetalBufferPool,
        start_row: usize,
        end_row: usize,
    ) -> &'p mut [u8] {
        let buf = pool.handle(self.k_id.expect("k_rows_mut called before ensure_buffers"));
        let rb = self.fmt.row_bytes(&VARIANT);
        // SAFETY: caller upholds the no-concurrent-GPU-work invariant.
        unsafe {
            let p = (buf.contents() as *mut u8).add(start_row * rb);
            std::slice::from_raw_parts_mut(p, (end_row - start_row) * rb)
        }
    }

    /// V counterpart of [`Self::k_rows_mut`].
    ///
    /// # Safety
    ///
    /// See [`Self::k_slice`].
    pub unsafe fn v_rows_mut<'p>(
        &self,
        pool: &'p MetalBufferPool,
        start_row: usize,
        end_row: usize,
    ) -> &'p mut [u8] {
        let buf = pool.handle(self.v_id.expect("v_rows_mut called before ensure_buffers"));
        let rb = self.fmt.row_bytes(&VARIANT);
        // SAFETY: caller upholds the no-concurrent-GPU-work invariant.
        unsafe {
            let p = (buf.contents() as *mut u8).add(start_row * rb);
            std::slice::from_raw_parts_mut(p, (end_row - start_row) * rb)
        }
    }

    /// Register the shared-storage K/V buffers into `pool` if not
    /// already present. Idempotent. Sized `capacity * num_kv_heads
    /// * head_dim` f32 per side — virtual reservations until pages are
    /// touched. Allocated via `device.new_buffer` + `register_borrowed`
    /// (NOT `pool.alloc`, which would zero-fill the whole multi-GB
    /// reservation): `StorageModeShared` pages are zero-filled on first
    /// touch by the Mach VM, so the explicit memset is redundant — see
    /// the `MlaKvCacheGpu::ensure_buffers` note. `persistent = true`
    /// keeps the buffers out of `commit_plan` lifetime coloring.
    pub fn ensure_buffers(&mut self, pool: &mut MetalBufferPool) {
        let bytes = self.bytes_per_side();
        let device = pool.device().clone();
        if self.k_id.is_none() {
            let buf = device.new_buffer(bytes as NSUInteger, MTLResourceOptions::StorageModeShared);
            // `new_buffer` is infallible at the Rust level: a nil from
            // `newBufferWithLength:` (device OOM / over working-set
            // budget) becomes a null-wrapping `Buffer`, and the later
            // `contents()` deref (KV append / SDPA) would SIGSEGV. Fail
            // with an attributable message instead. (This path's
            // signature isn't fallible; the pool's `new_shared_zeroed`
            // returns `GraphError::AllocFailed` for the recoverable case.)
            assert!(
                !buf.contents().is_null(),
                "kv.k_cache allocation failed: {bytes} bytes (out of device memory?)"
            );
            self.k_id = Some(pool.register_borrowed(buf, bytes, "kv.k_cache", true));
        }
        if self.v_id.is_none() {
            let buf = device.new_buffer(bytes as NSUInteger, MTLResourceOptions::StorageModeShared);
            assert!(
                !buf.contents().is_null(),
                "kv.v_cache allocation failed: {bytes} bytes (out of device memory?)"
            );
            self.v_id = Some(pool.register_borrowed(buf, bytes, "kv.v_cache", true));
        }
    }

    /// Register the shared f32 prefill scratch for a non-F32 cache of
    /// `capacity` rows ([`Self::f32_scratch`]): `capacity * kv_dim`
    /// f32 per side, lazily committed like the cache itself.
    pub fn alloc_f32_scratch(
        pool: &mut MetalBufferPool,
        capacity: usize,
    ) -> (BufId<KvCacheKBuf>, BufId<KvCacheVBuf>) {
        let bytes = capacity * VARIANT.num_kv_heads * VARIANT.head_dim * 4;
        let device = pool.device().clone();
        let new = |label| {
            let buf = device.new_buffer(bytes as NSUInteger, MTLResourceOptions::StorageModeShared);
            assert!(!buf.contents().is_null(), "{label} allocation failed: {bytes} bytes");
            (buf, label)
        };
        let (k, kl) = new("kv.k_f32_scratch");
        let (v, vl) = new("kv.v_f32_scratch");
        (pool.register_borrowed(k, bytes, kl, true), pool.register_borrowed(v, bytes, vl, true))
    }

    /// Commit the f32 prefill scratch ([`Self::touch`] for it).
    pub fn touch_f32_scratch(&self, pool: &MetalBufferPool) {
        let Some((k, v)) = self.f32_scratch else { return };
        for b in [pool.handle(k), pool.handle(v)] {
            // SAFETY: shared-storage buffer of `length()` bytes; caller
            // guarantees no in-flight GPU use.
            unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, b.length() as usize) };
        }
    }

    /// Commit every page of both buffers by writing zeros, so the
    /// cache's full `capacity` is resident now rather than growing
    /// with the sequence. Used under a hard memory cap
    /// ([`crate::riir::OpenConfig::mem_cap`]). No-op before
    /// [`Self::ensure_buffers`]. Call only with no GPU work in flight.
    pub fn touch(&self, pool: &MetalBufferPool) {
        let bytes = self.bytes_per_side();
        for id in [self.k_id.map(|i| pool.handle(i)), self.v_id.map(|i| pool.handle(i))]
            .into_iter()
            .flatten()
        {
            // SAFETY: shared-storage buffer of `bytes` length, allocated
            // in `ensure_buffers`; caller guarantees no in-flight GPU use.
            unsafe { std::ptr::write_bytes(id.contents() as *mut u8, 0, bytes) };
        }
    }

    /// Reset to positions `[0, new_len)`. No-op if already shorter.
    ///
    /// Prefill-arc Phase 0b: the `[new_len, old_len)` zero-window write
    /// is dropped — `truncate` no longer reaches a raw buffer once the
    /// cache lives in the pool, and the zeroing was never load-bearing.
    /// SDPA reads are bounded by `len`, and every appended row is
    /// overwritten before it is read; stale K/V past `len` is
    /// unreachable. Mirrors the MLA-cache + snapshot-restore precedent
    /// (`state_snapshot.rs` restore tail).
    pub fn truncate(&mut self, new_len: i32) {
        if new_len < 0 || new_len > self.len {
            return;
        }
        self.len = new_len;
    }

    /// Host-readable slice over the populated prefix of the K cache
    /// (`[0, len) × num_kv_heads*head_dim` floats). The returned slice
    /// borrows `pool` (where the buffer's storage lives), not `self`.
    ///
    /// # Safety
    ///
    /// Caller must guarantee no GPU work is reading or writing the
    /// underlying buffer concurrently.
    pub unsafe fn k_slice<'p>(&self, pool: &'p MetalBufferPool, len: usize) -> &'p [f32] {
        debug_assert_eq!(self.fmt, KvFormat::F32, "f32 KV accessor on a quantized cache");
        let buf = pool.handle(self.k_id.expect("k_slice called before ensure_buffers"));
        let n = len * VARIANT.num_kv_heads * VARIANT.head_dim;
        // SAFETY: caller upholds the no-concurrent-GPU-work invariant.
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }
    }

    /// Mutable counterpart to [`Self::k_slice`] over the row window
    /// `[start_row, end_row) × num_kv_heads*head_dim`. `&self` (not
    /// `&mut`): the storage lives in `pool`, addressed through Metal's
    /// interior mutability — the same discipline as
    /// `MetalBufferPool::as_mut_slice_u8`.
    ///
    /// # Safety
    ///
    /// See [`Self::k_slice`].
    pub unsafe fn k_slice_mut<'p>(
        &self,
        pool: &'p MetalBufferPool,
        start_row: usize,
        end_row: usize,
    ) -> &'p mut [f32] {
        debug_assert_eq!(self.fmt, KvFormat::F32, "f32 KV accessor on a quantized cache");
        let buf = pool.handle(self.k_id.expect("k_slice_mut called before ensure_buffers"));
        let stride = VARIANT.num_kv_heads * VARIANT.head_dim;
        // SAFETY: caller upholds the no-concurrent-GPU-work invariant.
        unsafe {
            let p = (buf.contents() as *mut f32).add(start_row * stride);
            std::slice::from_raw_parts_mut(p, (end_row - start_row) * stride)
        }
    }

    /// Host-readable slice over the populated prefix of the V cache.
    ///
    /// # Safety
    ///
    /// See [`Self::k_slice`].
    pub unsafe fn v_slice<'p>(&self, pool: &'p MetalBufferPool, len: usize) -> &'p [f32] {
        debug_assert_eq!(self.fmt, KvFormat::F32, "f32 KV accessor on a quantized cache");
        let buf = pool.handle(self.v_id.expect("v_slice called before ensure_buffers"));
        let n = len * VARIANT.num_kv_heads * VARIANT.head_dim;
        // SAFETY: caller upholds the no-concurrent-GPU-work invariant.
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }
    }

    /// Mutable counterpart to [`Self::v_slice`].
    ///
    /// # Safety
    ///
    /// See [`Self::k_slice`].
    pub unsafe fn v_slice_mut<'p>(
        &self,
        pool: &'p MetalBufferPool,
        start_row: usize,
        end_row: usize,
    ) -> &'p mut [f32] {
        debug_assert_eq!(self.fmt, KvFormat::F32, "f32 KV accessor on a quantized cache");
        let buf = pool.handle(self.v_id.expect("v_slice_mut called before ensure_buffers"));
        let stride = VARIANT.num_kv_heads * VARIANT.head_dim;
        // SAFETY: caller upholds the no-concurrent-GPU-work invariant.
        unsafe {
            let p = (buf.contents() as *mut f32).add(start_row * stride);
            std::slice::from_raw_parts_mut(p, (end_row - start_row) * stride)
        }
    }
}

/// Multi-head Latent Attention KV cache for one MLA layer (DeepSeek-V3
/// architecture — Cogito-V2-Preview-671B).
///
/// MLA jointly compresses K and V to a `kv_lora_rank`-dim latent
/// (512 for Cogito-V2) plus a shared `qk_rope_head_dim`-dim rope-K
/// (64 for Cogito-V2) per token. Total cached width per token is
/// `kv_lora_rank + qk_rope_head_dim` (= 576 for Cogito-V2), giving
/// ~28× memory compression vs GQA at long context.
///
/// At use time, K_nope and V are reconstructed from the latent via
/// `kv_b_proj` per cached position; only the rope-K is stored
/// directly.
///
/// Storage is GPU-resident: `MTLResourceStorageModeShared` Metal
/// buffers, sized to `capacity` (the context's `n_ctx`) rows. Shared storage means the
/// CPU MLA path can read/write the same bytes via `contents()`
/// without a host-side mirror. macOS lazy-commits the virtual
/// reservation, so non-touched rows don't consume physical RAM.
///
/// Buffers are `Option`-wrapped for lazy allocation: `LayerState`s
/// are constructed at `Ctx::open` before the Metal device exists;
/// `ensure_mla_gpu_resources` populates the buffers on first use.
/// Tests that don't init Metal can still construct/mutate `len`
/// without hitting the buffers (`truncate` is a no-op on `None`).
#[derive(Debug)]
pub struct MlaKvCacheGpu {
    /// `kv_a_layernorm`-output cache: post-down-projection,
    /// post-norm latent. Shape `[capacity, kv_lora_rank]`
    /// row-major. `None` until populated by
    /// `ensure_mla_gpu_resources` on first eval.
    pub latent_cache: Option<Buffer>,
    /// Pre-RoPE'd rope-K cache (already RoPE-applied at the position
    /// it was stored at — broadcast across all heads at use time).
    /// Shape `[capacity, qk_rope_head_dim]`.
    pub rope_k_cache: Option<Buffer>,
    /// Number of populated positions.
    pub len: i32,
    /// Maximum number of positions the buffers hold (the context's
    /// `n_ctx`).
    pub capacity: usize,
}

impl MlaKvCacheGpu {
    /// Construct an empty cache without any GPU buffers. Buffers are
    /// allocated lazily by [`Self::ensure_buffers`] once a Metal
    /// device is available.
    pub fn new(capacity: usize) -> Self {
        Self {
            latent_cache: None,
            rope_k_cache: None,
            len: 0,
            capacity: clamp_n_ctx(capacity),
        }
    }

    /// Allocate the underlying shared-storage Metal buffers if not
    /// already present. Idempotent.
    ///
    /// Sizes are read from `VARIANT.kv_lora_rank` and
    /// `VARIANT.qk_rope_head_dim`; for Cogito-V2 that's
    /// `128k * 512 * 4 = 256 MB` for the latent cache and
    /// `128k * 64 * 4 = 32 MB` for the rope-K cache, per layer.
    /// Both are virtual reservations until pages are touched.
    pub fn ensure_buffers(&mut self, device: &Device) {
        // Phase 3 (cogito-v2 full-GPU): drop the explicit
        // buffer-zeroing here. macOS Metal `StorageModeShared`
        // buffers come from the Mach VM via the same anonymous-mapping
        // path `mmap(MAP_ANON)` uses — pages are zero-filled on first
        // touch by the kernel. The 17.5 GB explicit memset that
        // dominated cold-eval profiles was redundant: every byte was
        // about to be implicitly zero-paged anyway when the per-layer
        // append wrote into it. Removing it drops cold init from ~23s
        // to ~1s on cogito-v2-671b without changing kernel-visible
        // values (writes still see zero on first read of unwritten
        // pages, matching the explicit-memset behavior bit-for-bit).
        if self.latent_cache.is_none() {
            let bytes =
                (self.capacity * VARIANT.kv_lora_rank * std::mem::size_of::<f32>()) as NSUInteger;
            let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
            // See the Gqa KV note above: guard the infallible alloc so a
            // nil buffer fails attributably instead of a later SIGSEGV.
            assert!(
                !buf.contents().is_null(),
                "mla.latent_cache allocation failed: {bytes} bytes (out of device memory?)"
            );
            self.latent_cache = Some(buf);
        }
        if self.rope_k_cache.is_none() {
            let bytes =
                (self.capacity * VARIANT.qk_rope_head_dim * std::mem::size_of::<f32>()) as NSUInteger;
            let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
            assert!(
                !buf.contents().is_null(),
                "mla.rope_k_cache allocation failed: {bytes} bytes (out of device memory?)"
            );
            self.rope_k_cache = Some(buf);
        }
    }

    /// Commit every page of both buffers by writing zeros (see
    /// [`KvCache::touch`]). No-op before [`Self::ensure_buffers`].
    pub fn touch(&self) {
        for buf in [&self.latent_cache, &self.rope_k_cache].into_iter().flatten() {
            // SAFETY: shared-storage buffer allocated by
            // `ensure_buffers`; `length()` is its byte size; caller
            // guarantees no in-flight GPU use.
            unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, buf.length() as usize) };
        }
    }

    /// Reset to positions `[0, new_len)`. No-op if already shorter or
    /// `new_len` is invalid. When the underlying buffers exist, zeros
    /// the `[new_len, old_len)` window so stale rows can't bleed into
    /// later decodes.
    pub fn truncate(&mut self, new_len: i32) {
        if new_len < 0 || new_len > self.len {
            return;
        }
        let old_len = self.len;
        if new_len < old_len {
            if let Some(buf) = &self.latent_cache {
                let stride_bytes = VARIANT.kv_lora_rank * std::mem::size_of::<f32>();
                let start = (new_len as usize) * stride_bytes;
                let end = (old_len as usize) * stride_bytes;
                // SAFETY: shared-storage buffer; truncate is called
                // outside per-token forward (memory_clear /
                // checkpoint restore), with no GPU work in flight.
                unsafe {
                    let p = buf.contents() as *mut u8;
                    std::ptr::write_bytes(p.add(start), 0, end - start);
                }
            }
            if let Some(buf) = &self.rope_k_cache {
                let stride_bytes = VARIANT.qk_rope_head_dim * std::mem::size_of::<f32>();
                let start = (new_len as usize) * stride_bytes;
                let end = (old_len as usize) * stride_bytes;
                // SAFETY: see above.
                unsafe {
                    let p = buf.contents() as *mut u8;
                    std::ptr::write_bytes(p.add(start), 0, end - start);
                }
            }
        }
        self.len = new_len;
    }

    /// Host-readable slice over the populated prefix of the latent
    /// cache (`[0, len) × kv_lora_rank` floats).
    ///
    /// # Safety
    ///
    /// Caller must guarantee no GPU work is reading or writing the
    /// underlying buffer concurrently. The CPU MLA path holds this
    /// invariant by construction (no GPU dispatch is in flight when
    /// the CPU pipeline runs); the GPU path uses the buffer directly
    /// via Metal kernels.
    pub unsafe fn latent_slice(&self, len: usize) -> &[f32] {
        let buf = self
            .latent_cache
            .as_ref()
            .expect("latent_slice called before ensure_buffers");
        let n = len * VARIANT.kv_lora_rank;
        // SAFETY: caller upholds the no-concurrent-GPU-work invariant
        // documented on this fn.
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }
    }

    /// Mutable counterpart to [`Self::latent_slice`] over the row
    /// window `[start_row, end_row) × kv_lora_rank`.
    ///
    /// # Safety
    ///
    /// See [`Self::latent_slice`].
    pub unsafe fn latent_slice_mut(&mut self, start_row: usize, end_row: usize) -> &mut [f32] {
        let buf = self
            .latent_cache
            .as_ref()
            .expect("latent_slice_mut called before ensure_buffers");
        let stride = VARIANT.kv_lora_rank;
        // SAFETY: caller upholds the no-concurrent-GPU-work invariant
        // documented on this fn.
        unsafe {
            let p = (buf.contents() as *mut f32).add(start_row * stride);
            std::slice::from_raw_parts_mut(p, (end_row - start_row) * stride)
        }
    }

    /// Host-readable slice over the populated prefix of the rope-K
    /// cache (`[0, len) × qk_rope_head_dim` floats).
    ///
    /// # Safety
    ///
    /// See [`Self::latent_slice`].
    pub unsafe fn rope_k_slice(&self, len: usize) -> &[f32] {
        let buf = self
            .rope_k_cache
            .as_ref()
            .expect("rope_k_slice called before ensure_buffers");
        let n = len * VARIANT.qk_rope_head_dim;
        // SAFETY: caller upholds the no-concurrent-GPU-work invariant
        // documented on this fn.
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }
    }

    /// Mutable counterpart to [`Self::rope_k_slice`] over the row
    /// window `[start_row, end_row) × qk_rope_head_dim`.
    ///
    /// # Safety
    ///
    /// See [`Self::latent_slice`].
    pub unsafe fn rope_k_slice_mut(&mut self, start_row: usize, end_row: usize) -> &mut [f32] {
        let buf = self
            .rope_k_cache
            .as_ref()
            .expect("rope_k_slice_mut called before ensure_buffers");
        let stride = VARIANT.qk_rope_head_dim;
        // SAFETY: caller upholds the no-concurrent-GPU-work invariant
        // documented on this fn.
        unsafe {
            let p = (buf.contents() as *mut f32).add(start_row * stride);
            std::slice::from_raw_parts_mut(p, (end_row - start_row) * stride)
        }
    }
}

impl Default for MlaKvCacheGpu {
    /// Full `MAX_SEQ_LEN` capacity — the pre-`OpenConfig` behavior.
    fn default() -> Self {
        Self::new(MAX_SEQ_LEN)
    }
}

/// GatedDeltaNet recurrent state for one linear-attention layer.
/// `conv_state` holds the depthwise conv1d's last `(kernel_size - 1)`
/// inputs; `ssm_state` holds the per-v-head outer-product state of
/// shape `[num_v_heads, value_dim, key_dim]` (flattened).
#[derive(Debug)]
pub struct LinearAttnState {
    pub conv_state: Box<[f32]>,
    pub ssm_state: Box<[f32]>,
}

impl LinearAttnState {
    /// Allocate a zeroed state sized for the active variant.
    pub fn new() -> Self {
        let conv_entries = (Variant::CONV_KERNEL_SIZE - 1) * VARIANT.linear_conv_dim();
        let ssm_entries =
            VARIANT.linear_num_v_heads * Variant::LINEAR_VALUE_DIM * Variant::LINEAR_KEY_DIM;
        Self {
            conv_state: vec![0.0f32; conv_entries].into_boxed_slice(),
            ssm_state: vec![0.0f32; ssm_entries].into_boxed_slice(),
        }
    }

    /// Commit every page of the host state by writing zeros (it is
    /// lazily committed after `new`). Under a memory cap, so the
    /// open-time footprint check sees it.
    pub fn touch(&mut self) {
        for b in [&mut self.conv_state, &mut self.ssm_state] {
            // SAFETY: writes `len` f32 zeros into the slice's own storage.
            unsafe { std::ptr::write_bytes(b.as_mut_ptr(), 0, b.len()) };
        }
    }

    /// Reset to the empty-sequence state. Lossy by construction — see
    /// the module docs. Mirrors `linear_attn_state_reset` at
    /// infer.m:2260.
    pub fn reset(&mut self) {
        self.conv_state.fill(0.0);
        self.ssm_state.fill(0.0);
    }
}

/// Per-layer state. The variant tag matches the C-side
/// `(layer + 1) % FULL_ATTN_INTERVAL == 0` test: every Nth layer is a
/// full-attention layer with a KV cache, the rest are linear-attention
/// layers with a GatedDeltaNet recurrence.
///
/// The full-attention slot has two flavors selected by
/// [`Variant::attn_kind`]: GQA (`FullAttn`) caches per-head K and V
/// directly; MLA (`Mla`) caches a compressed latent + a shared rope-K
/// per token and reconstructs K/V at use time.
#[derive(Debug)]
pub enum LayerState {
    FullAttn(KvCache),
    Mla(MlaKvCacheGpu),
    LinearAttn(LinearAttnState),
}

impl LayerState {
    /// True for any flavor that grows a sequence-length-shaped cache
    /// (GQA `FullAttn` or `Mla`). Used by callers that distinguish
    /// "real" KV layers from the constant-state linear-attn layers.
    pub fn is_full(&self) -> bool {
        matches!(self, Self::FullAttn(_) | Self::Mla(_))
    }

    /// KV capacity in positions for full-attn flavors; `None` for
    /// linear-attn layers (constant-size state).
    pub fn kv_capacity(&self) -> Option<usize> {
        match self {
            Self::FullAttn(kv) => Some(kv.capacity),
            Self::Mla(mla) => Some(mla.capacity),
            Self::LinearAttn(_) => None,
        }
    }
}

/// Allocate the per-layer state vector for the active variant.
/// Dispatched via [`Variant::layer_kind`] — for qwen3_5_moe that's
/// `(i + 1) % full_attn_interval == 0`; for DeepSeek-V3 / Cogito-V2
/// every layer is full-attn (`full_attn_interval = 1`). The flavor
/// of full-attn (GQA vs MLA) is selected by [`Variant::attn_kind`]:
/// GQA gets a [`KvCache`], MLA gets a compressed [`MlaKvCache`].
/// This matters at allocation because [`KvCache::new`] would reserve
/// `num_kv_heads * head_dim * n_ctx * 2 * 4` bytes — for
/// Cogito-V2 that's a ~196 GB virtual reservation per layer, all of
/// it lazy-committed but still wasted address space we never touch.
/// Mirrors the C-side allocation in `mf_init_model` (infer.m:7511+).
///
/// `n_ctx` sizes every KV cache (clamped to `[1, MAX_SEQ_LEN]`).
pub fn alloc_layer_states(n_ctx: usize) -> Vec<LayerState> {
    (0..VARIANT.num_layers).map(|i| new_layer_state(i, n_ctx, KvFormat::F32)).collect()
}

/// Per-layer state of one sequence, indexed by global layer index and
/// holding only the context's layer range (plan phase 3).
pub type LayerStates = RangeVec<LayerState>;

/// [`alloc_layer_states`] for the layers in `range` only, GQA KV
/// caches stored as `fmt`.
pub fn alloc_layer_states_range(
    n_ctx: usize,
    range: std::ops::Range<usize>,
    fmt: KvFormat,
) -> LayerStates {
    RangeVec::from_fn(range, |i| new_layer_state(i, n_ctx, fmt))
}

fn new_layer_state(i: usize, n_ctx: usize, fmt: KvFormat) -> LayerState {
    use crate::riir::variants::{AttnKind, LayerKind};
    match VARIANT.layer_kind(i) {
        LayerKind::FullAttn => match VARIANT.attn_kind {
            AttnKind::Gqa => LayerState::FullAttn(KvCache::new(n_ctx, fmt)),
            AttnKind::Mla => LayerState::Mla(MlaKvCacheGpu::new(n_ctx)),
        },
        LayerKind::LinearAttn => LayerState::LinearAttn(LinearAttnState::new()),
    }
}

/// Reset every layer's state to empty. Mirrors `mf_state_clear_all`
/// (infer.m:2271).
pub fn clear_all(layers: &mut [LayerState]) {
    for layer in layers {
        match layer {
            LayerState::FullAttn(kv) => kv.truncate(0),
            LayerState::Mla(mla) => mla.truncate(0),
            LayerState::LinearAttn(la) => la.reset(),
        }
    }
}

/// Truncate every layer to positions `[0, p0)`. Linear-attn layers
/// reset to empty (lossy — see module docs). `p0 < 0` is treated as 0
/// (full clear); `p1 < 0` means "to end". Mirrors `mf_state_truncate`
/// (infer.m:2291).
pub fn truncate(layers: &mut [LayerState], p0: i32, p1: i32) {
    let new_len = p0.max(0);
    for layer in layers {
        match layer {
            LayerState::FullAttn(kv) => {
                let effective_end = if p1 < 0 || p1 > kv.len { kv.len } else { p1 };
                let truncate_to = new_len.min(effective_end);
                kv.truncate(truncate_to);
            }
            LayerState::Mla(mla) => {
                let effective_end = if p1 < 0 || p1 > mla.len { mla.len } else { p1 };
                let truncate_to = new_len.min(effective_end);
                mla.truncate(truncate_to);
            }
            // FIXME(riir): faithful port of the lossy semantic. A
            // partial truncation of a linear-attn span resets the
            // recurrence to empty. Phase 7 introduces a typed
            // `Result<(), CannotTruncateLinear>` so callers must
            // explicitly handle the unwind failure.
            LayerState::LinearAttn(la) => la.reset(),
        }
    }
}

/// Largest occupied position across full-attn layers (GQA `FullAttn`
/// or `Mla`), or `-1` if no full-attn layer exists at all. Mirrors
/// `mf_state_pos_max` (infer.m:2320).
pub fn pos_max(layers: &[LayerState]) -> i32 {
    let mut max_len = -1;
    for layer in layers {
        let len = match layer {
            LayerState::FullAttn(kv) => kv.len,
            LayerState::Mla(mla) => mla.len,
            LayerState::LinearAttn(_) => continue,
        };
        if len > max_len {
            max_len = len;
        }
    }
    max_len
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A range holds exactly its layers' states, kinds matching the
    /// whole-model allocation.
    #[test]
    fn range_states_match_full_allocation() {
        let n = VARIANT.num_layers;
        let full = alloc_layer_states(64);
        let part = alloc_layer_states_range(64, n / 2..n, KvFormat::F32);
        assert_eq!(part.len(), n - n / 2);
        for (i, s) in part.iter_global() {
            assert_eq!(s.is_full(), full[i].is_full(), "layer {i}");
        }
    }

    /// Every full-attn layer is sized to the requested `n_ctx`;
    /// linear-attn layers carry no KV capacity.
    #[test]
    fn layer_states_sized_to_n_ctx() {
        let layers = alloc_layer_states(4096);
        for l in &layers {
            match l.kv_capacity() {
                Some(cap) => assert_eq!(cap, 4096),
                None => assert!(!l.is_full()),
            }
        }
        assert!(layers.iter().any(|l| l.is_full()));
    }

    /// `n_ctx` is clamped to `[1, MAX_SEQ_LEN]`, and the GQA cache's
    /// byte size scales with it (no `MAX_SEQ_LEN` reservation).
    #[test]
    fn kv_capacity_clamped_and_bytes_scale() {
        assert_eq!(KvCache::new(0, KvFormat::F32).capacity, 1);
        assert_eq!(KvCache::new(MAX_SEQ_LEN + 7, KvFormat::F32).capacity, MAX_SEQ_LEN);
        assert_eq!(MlaKvCacheGpu::new(0).capacity, 1);
        let small = KvCache::new(1024, KvFormat::F32);
        let row = VARIANT.num_kv_heads * VARIANT.head_dim * std::mem::size_of::<f32>();
        assert_eq!(small.bytes_per_side(), 1024 * row);
        assert_eq!(KvCache::new(2048, KvFormat::F32).bytes_per_side(), 2 * small.bytes_per_side());
    }

    /// Empty state on a fresh allocation: pos_max returns 0 (the
    /// largest occupied position across full-attn layers, all of which
    /// start at len=0). The `-1` sentinel only fires when no full-attn
    /// layer exists at all — not the case for any supported variant.
    #[test]
    fn empty_state_pos_max_is_zero() {
        let mut layers = alloc_layer_states(MAX_SEQ_LEN);
        assert_eq!(pos_max(&layers), 0);
        clear_all(&mut layers);
        assert_eq!(pos_max(&layers), 0);
    }

    /// Truncate on empty state is a no-op; pos_max stays at 0
    /// regardless of arguments.
    #[test]
    fn truncate_empty_is_noop() {
        let mut layers = alloc_layer_states(MAX_SEQ_LEN);
        truncate(&mut layers, 0, -1);
        assert_eq!(pos_max(&layers), 0);
        truncate(&mut layers, 5, 10);
        assert_eq!(pos_max(&layers), 0);
        truncate(&mut layers, -1, -1);
        assert_eq!(pos_max(&layers), 0);
    }

    /// Synthetic: hand-set one full-attn layer's len, observe pos_max
    /// pick it up, truncate, observe pos_max drop. Handles both
    /// flavors (`FullAttn` for GQA variants, `Mla` for DeepSeek-V3
    /// variants); the test asserts the same observable behavior on
    /// either side of the dispatch.
    #[test]
    fn truncate_drops_full_attn_len() {
        let mut layers = alloc_layer_states(MAX_SEQ_LEN);
        let injected = layers.iter_mut().find_map(|l| match l {
            LayerState::FullAttn(kv) => {
                kv.len = 7;
                Some(())
            }
            LayerState::Mla(mla) => {
                mla.len = 7;
                Some(())
            }
            LayerState::LinearAttn(_) => None,
        });
        assert!(
            injected.is_some(),
            "variant must have at least one full-attn (GQA or MLA) layer",
        );
        assert_eq!(pos_max(&layers), 7);

        truncate(&mut layers, 3, -1);
        assert_eq!(pos_max(&layers), 3);

        truncate(&mut layers, 10, -1);
        // new_len > len: kv_cache_truncate is a no-op when new_len >
        // self.len. p0=10 with current len=3 → effective_end=3,
        // truncate_to=min(10, 3)=3. Stays at 3.
        assert_eq!(pos_max(&layers), 3);

        clear_all(&mut layers);
        assert_eq!(pos_max(&layers), 0);
    }
}
