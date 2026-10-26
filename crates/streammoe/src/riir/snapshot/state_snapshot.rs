//! Per-Ctx state snapshot — Phase 4g port of `mf_state_save` /
//! `mf_state_load` (infer.m:8525..8700).
//!
//! Wire format (little-endian, fixed by the C side):
//!
//! - **Header** (8 × `u32`):
//!   - `magic = 0x4D464C58` (`'MFLX'`)
//!   - `version = 1`
//!   - `num_layers`
//!   - `full_attn_interval`
//!   - `num_kv_heads`
//!   - `head_dim`
//!   - `linear_conv_bytes` =
//!     `(CONV_KERNEL_SIZE - 1) * linear_conv_dim() * 4`
//!   - `linear_ssm_bytes` =
//!     `linear_num_v_heads * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * 4`
//! - **Per-layer body** (in 0..num_layers order):
//!   - Full-attn layer: `i32 len`, then `len * fa_stride` bytes K
//!     followed by the same many bytes V (where `fa_stride =
//!     num_kv_heads * head_dim * 4`).
//!   - Linear-attn layer: `linear_conv_bytes` of conv_state then
//!     `linear_ssm_bytes` of ssm_state.
//!
//! - **v3 — layer range** (plan phase 3): a context running layers
//!   `a..b` of the model writes the v2 header plus two trailing words
//!   `layer_start = a`, `layer_end = b`, and a body for those layers
//!   only. A whole-model context still writes v2, byte-identical to
//!   before. A range context loads only a v3 snapshot of the same
//!   range.
//!
//! The Rust port uses GPU buffers as the canonical store for the
//! linear-attn recurrence (`linear_buffers.conv_state[i]` /
//! `delta_state[i]`); save reads them back, load memcpys into them.
//! This matches C's Metal-fast-path layout — see the C path's
//! `g_metal->buf_conv_state[linear_idx]` writes / reads in
//! infer.m:8568..8595 / 8678..8693. The CPU-fallback path the C side
//! also handles (infer.m:8579..8593) doesn't apply: there is no CPU
//! fallback in the Rust port.
//!
//! ## Concurrency contract
//!
//! Mirror of streammoe.h:481-487 — call only at token boundaries
//! (after `eval_prompt` / `eval_token` returns). Pending deferred
//! GPU expert compute must have finalized; this is enforced by the
//! `step_internal` orchestrator draining `RsCtx::deferred` before
//! returning, but a defensive `discard_deferred_experts_in` in
//! `state_save` keeps the contract robust against a buggy caller.

use metal::{Buffer, Device};

use crate::riir::attn::kv_quant::KvFormat;
use crate::riir::attn::linear_attn_forward::{LayerForwardBuffers, linear_layer_idx_for};
use crate::riir::moe::deferred;
use crate::riir::snapshot::state::{LayerState, LayerStates, MlaKvCacheGpu};
use crate::riir::variants::{AttnKind, LayerKind, VARIANT, Variant};

/// `'MFLX'` little-endian. Wire-compatible with the C side's
/// `MF_SNAPSHOT_MAGIC` (infer.m:8487).
pub const SNAPSHOT_MAGIC: u32 = 0x4D464C58;
/// Format version. Bumped when wire layout changes; load() rejects
/// any version it doesn't understand.
///
/// - **v1**: full-attn + linear-attn layers. C-compatible.
/// - **v2**: adds MLA layers (`LayerKind::FullAttn` with
///   `AttnKind::Mla`). Per-layer body is `i32 len`, `len ×
///   kv_lora_rank × 4` latent bytes, `len × qk_rope_head_dim × 4`
///   rope-K bytes. Header gains `kv_lora_rank` and `qk_rope_head_dim`
///   trailing the v1 header (so v1 readers in C see the same first
///   8 words). Loader accepts v1 for backward compat with Qwen
///   variants.
pub const SNAPSHOT_VERSION: u32 = 2;
/// Format version written by a layer-range context (v2 + range words).
pub const SNAPSHOT_VERSION_RANGE: u32 = 3;
/// v3 header size: v2 plus `layer_start`, `layer_end`.
pub const SNAPSHOT_HEADER_V3_U32: usize = 12;
/// Format version written when the GQA KV cache is not f32
/// ([`crate::riir::KvFormat`]): the v3 header (range words always
/// present) plus a `kv_format` code word; KV bodies are the stored
/// rows, `len × fmt.row_bytes` per side. F32 contexts keep writing
/// v2 / v3, byte-identical to before.
pub const SNAPSHOT_VERSION_KV_FORMAT: u32 = 4;
/// v4 header size: v3 plus `kv_format`.
pub const SNAPSHOT_HEADER_V4_U32: usize = 13;
/// Header is exactly this many `u32` fields. Mirrors C
/// `MF_SNAPSHOT_HEADER_U32`. **v2 adds two trailing words**:
/// `kv_lora_rank` and `qk_rope_head_dim`.
pub const SNAPSHOT_HEADER_U32: usize = 10;
/// v1 header size — kept for the backward-compat reader path.
pub const SNAPSHOT_HEADER_V1_U32: usize = 8;

/// Errors from the snapshot save/load path.
#[derive(Debug, thiserror::Error)]
pub enum StateSnapshotError {
    /// Caller-supplied buffer too small for the snapshot.
    #[error("buffer too small: need {need} bytes, got {got}")]
    BufferTooSmall { need: usize, got: usize },
    /// Header magic doesn't match.
    #[error("snapshot magic mismatch (got 0x{got:08X}, want 0x{want:08X})")]
    BadMagic { got: u32, want: u32 },
    /// Header version doesn't match.
    #[error("snapshot version mismatch (got {got}, want {want})")]
    BadVersion { got: u32, want: u32 },
    /// Header shape constants don't match the active build.
    #[error("snapshot shape mismatch on field '{field}' (got {got}, want {want})")]
    ShapeMismatch {
        field: &'static str,
        got: u32,
        want: u32,
    },
    /// Per-layer length value is negative.
    #[error("snapshot layer {layer} has negative KV length {len}")]
    NegativeLen { layer: usize, len: i32 },
    /// Per-layer length exceeds the architectural maximum.
    #[error("snapshot layer {layer} KV length {len} exceeds this context's n_ctx={max}")]
    LenOverflow { layer: usize, len: i32, max: usize },
    /// Truncated payload (preflight failure during load).
    #[error("snapshot truncated at layer {layer}: need {need} more bytes, got {got}")]
    Truncated {
        layer: usize,
        need: usize,
        got: usize,
    },
    /// linear_buffers must exist before save/load. Set up by
    /// `RsCtx::ensure_linear_resources` on first eval.
    #[error("linear_buffers not initialized — call eval_prompt or memory_clear first")]
    BuffersNotReady,
    /// Snapshot wire format v1 doesn't encode MLA's compressed
    /// latent + rope-K caches. A v2 format that supports MLA layers
    /// is post-cutover work; until then save/load on Cogito-V2 /
    /// DeepSeek-V3 builds returns this error.
    #[error("snapshot v{SNAPSHOT_VERSION} doesn't support MLA layers (layer {layer})")]
    MlaUnsupported { layer: usize },
    /// Under a memory cap every checkpoint slot is in use by a
    /// protected position (0 or the one being overwritten), so there
    /// is no buffer to save into without growing past the plan.
    /// Raise `OpenConfig::max_checkpoints` or `forget_pos` one.
    #[error("all {max} planned checkpoint buffers are in use by protected positions")]
    CheckpointsFull { max: usize },
}

/// Bytes per stored KV row (one side) in `fmt`.
#[inline]
fn full_attn_stride_bytes(v: &Variant, fmt: KvFormat) -> usize {
    fmt.row_bytes(v)
}

/// The GQA KV format of `layer_states` (F32 when it has no GQA layer).
fn kv_format_of(layer_states: &LayerStates) -> KvFormat {
    layer_states
        .iter_global()
        .find_map(|(_, l)| match l {
            LayerState::FullAttn(kv) => Some(kv.fmt),
            _ => None,
        })
        .unwrap_or(KvFormat::F32)
}

#[inline]
fn linear_conv_bytes(v: &Variant) -> usize {
    (Variant::CONV_KERNEL_SIZE - 1) * v.linear_conv_dim() * std::mem::size_of::<f32>()
}

#[inline]
fn linear_ssm_bytes(v: &Variant) -> usize {
    v.linear_num_v_heads
        * Variant::LINEAR_VALUE_DIM
        * Variant::LINEAR_KEY_DIM
        * std::mem::size_of::<f32>()
}

#[inline]
fn mla_latent_bytes(v: &Variant, len: usize) -> usize {
    len * v.kv_lora_rank * std::mem::size_of::<f32>()
}

#[inline]
fn mla_rope_k_bytes(v: &Variant, len: usize) -> usize {
    len * v.qk_rope_head_dim * std::mem::size_of::<f32>()
}

/// Read `n` floats from a Metal shared-storage `Buffer` into the
/// given byte slice. Caller ensures no GPU work is in flight.
fn read_buffer_bytes_n_f32(buf: &Buffer, dst: &mut [u8], n_f32: usize) {
    let bytes = n_f32 * std::mem::size_of::<f32>();
    debug_assert_eq!(dst.len(), bytes);
    // SAFETY: shared-storage; caller honors the no-GPU-in-flight contract.
    unsafe {
        std::ptr::copy_nonoverlapping(buf.contents() as *const u8, dst.as_mut_ptr(), bytes);
    }
}

/// Write the byte slice into the first `bytes` of a Metal shared-
/// storage `Buffer`. Caller ensures no GPU work is in flight.
fn write_buffer_bytes_n_f32(buf: &Buffer, src: &[u8], n_f32: usize) {
    let bytes = n_f32 * std::mem::size_of::<f32>();
    debug_assert_eq!(src.len(), bytes);
    // SAFETY: shared-storage; caller honors the no-GPU-in-flight contract.
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), buf.contents() as *mut u8, bytes);
    }
}

/// Allocate the MLA cache buffers if missing. Mirrors the lazy alloc
/// done by `MlaKvCacheGpu::ensure_buffers`, just from the load path
/// where `state_load` doesn't otherwise hold the device.
fn ensure_mla_buffers(cache: &mut MlaKvCacheGpu, device: &Device) {
    cache.ensure_buffers(device);
}

/// Bytes the caller must allocate to hold the current snapshot.
/// Mirrors C `mf_state_size` (infer.m:8505..8523). Re-query after
/// every evaluation: the value grows with KV length.
///
/// **v2** (cogito-v2 full-GPU plan, Phase 7): MLA layers are now
/// supported — per-layer body is `i32 len + len × kv_lora_rank × 4`
/// latent bytes + `len × qk_rope_head_dim × 4` rope-K bytes.
/// Header words for a snapshot of `layer_states`: v2 for the whole
/// model, v3 for a layer range.
fn header_words_for(layer_states: &LayerStates) -> usize {
    if kv_format_of(layer_states) != KvFormat::F32 {
        SNAPSHOT_HEADER_V4_U32
    } else if crate::riir::range::is_full(&layer_states.span()) {
        SNAPSHOT_HEADER_U32
    } else {
        SNAPSHOT_HEADER_V3_U32
    }
}

pub fn state_size(layer_states: &LayerStates) -> usize {
    state_size_at(layer_states, false)
}

/// [`state_size`] with every full-attn layer at its capacity
/// (`n_ctx`): the largest snapshot this context can produce. Sizes the
/// preallocated checkpoint buffers under a memory cap.
pub fn state_size_full(layer_states: &LayerStates) -> usize {
    state_size_at(layer_states, true)
}

fn state_size_at(layer_states: &LayerStates, full: bool) -> usize {
    let v = VARIANT;
    let mut n = header_words_for(layer_states) * std::mem::size_of::<u32>();
    let fa_stride = full_attn_stride_bytes(&v, kv_format_of(layer_states));
    let la_conv = linear_conv_bytes(&v);
    let la_ssm = linear_ssm_bytes(&v);
    for (i, layer) in layer_states.iter_global() {
        match v.layer_kind(i) {
            LayerKind::FullAttn => {
                n += std::mem::size_of::<i32>();
                let len = match layer {
                    _ if full => layer.kv_capacity().unwrap_or(0),
                    LayerState::FullAttn(kv) => kv.len.max(0) as usize,
                    LayerState::Mla(c) => c.len.max(0) as usize,
                    LayerState::LinearAttn(_) => 0,
                };
                match v.attn_kind {
                    AttnKind::Gqa => {
                        n += 2 * len * fa_stride;
                    }
                    AttnKind::Mla => {
                        n += mla_latent_bytes(&v, len) + mla_rope_k_bytes(&v, len);
                    }
                }
            }
            LayerKind::LinearAttn => {
                n += la_conv + la_ssm;
            }
        }
    }
    n
}

/// Serialize the current state into `buf`. Returns the number of
/// bytes written. Mirrors C `mf_state_save` (infer.m:8525..8597).
///
/// Reads the linear-attn GPU recurrence buffers back to host as
/// part of writing the body. Caller must ensure no GPU work is in
/// flight against those buffers — the public `RsCtx::state_save`
/// wrapper enforces this by draining any pending deferred dispatch
/// first.
pub fn state_save(
    buf: &mut [u8],
    layer_states: &LayerStates,
    linear_buffers: Option<&LayerForwardBuffers>,
    pool: Option<&crate::riir::backend::MetalBufferPool>,
) -> Result<usize, StateSnapshotError> {
    use crate::riir::backend::BufferPool as _;
    let v = VARIANT;
    let need = state_size(layer_states);
    if buf.len() < need {
        return Err(StateSnapshotError::BufferTooSmall {
            need,
            got: buf.len(),
        });
    }

    let fmt = kv_format_of(layer_states);
    let fa_stride = full_attn_stride_bytes(&v, fmt);
    let la_conv = linear_conv_bytes(&v);
    let la_ssm = linear_ssm_bytes(&v);

    let mut off = 0usize;

    // Header (v2 — 10 u32 fields; v3 adds the layer range).
    let span = layer_states.span();
    let words = header_words_for(layer_states);
    let header: [u32; SNAPSHOT_HEADER_V4_U32] = [
        SNAPSHOT_MAGIC,
        match words {
            SNAPSHOT_HEADER_U32 => SNAPSHOT_VERSION,
            SNAPSHOT_HEADER_V3_U32 => SNAPSHOT_VERSION_RANGE,
            _ => SNAPSHOT_VERSION_KV_FORMAT,
        },
        v.num_layers as u32,
        v.full_attn_interval as u32,
        v.num_kv_heads as u32,
        v.head_dim as u32,
        la_conv as u32,
        la_ssm as u32,
        v.kv_lora_rank as u32,
        v.qk_rope_head_dim as u32,
        span.start as u32,
        span.end as u32,
        fmt.code(),
    ];
    for &word in header[..words].iter() {
        buf[off..off + 4].copy_from_slice(&word.to_le_bytes());
        off += 4;
    }

    // Per-layer body.
    for (i, layer) in layer_states.iter_global() {
        match v.layer_kind(i) {
            LayerKind::FullAttn => match (v.attn_kind, layer) {
                (AttnKind::Gqa, LayerState::FullAttn(kv)) => {
                    let len = kv.len.max(0);
                    buf[off..off + 4].copy_from_slice(&len.to_le_bytes());
                    off += 4;
                    if len > 0 {
                        let bytes = (len as usize) * fa_stride;
                        // Phase 0b (prefill arc): KV is GPU-resident
                        // in the pool — resolve the BufId to its
                        // shared-storage buffer and read straight from
                        // it (mirrors the MLA branch below). Raw
                        // stored rows, whatever the format.
                        let p = pool.ok_or(StateSnapshotError::BuffersNotReady)?;
                        let k_buf = p.handle(kv.k_id.ok_or(StateSnapshotError::BuffersNotReady)?);
                        read_buffer_bytes(k_buf, &mut buf[off..off + bytes]);
                        off += bytes;
                        let v_buf = p.handle(kv.v_id.ok_or(StateSnapshotError::BuffersNotReady)?);
                        read_buffer_bytes(v_buf, &mut buf[off..off + bytes]);
                        off += bytes;
                    }
                }
                (AttnKind::Mla, LayerState::Mla(cache)) => {
                    let len = cache.len.max(0);
                    buf[off..off + 4].copy_from_slice(&len.to_le_bytes());
                    off += 4;
                    if len > 0 {
                        let lat_bytes = mla_latent_bytes(&v, len as usize);
                        let rope_bytes = mla_rope_k_bytes(&v, len as usize);
                        let lat_buf = cache
                            .latent_cache
                            .as_ref()
                            .ok_or(StateSnapshotError::BuffersNotReady)?;
                        let rope_buf = cache
                            .rope_k_cache
                            .as_ref()
                            .ok_or(StateSnapshotError::BuffersNotReady)?;
                        let n_lat = (len as usize) * v.kv_lora_rank;
                        let n_rope = (len as usize) * v.qk_rope_head_dim;
                        read_buffer_bytes_n_f32(lat_buf, &mut buf[off..off + lat_bytes], n_lat);
                        off += lat_bytes;
                        read_buffer_bytes_n_f32(rope_buf, &mut buf[off..off + rope_bytes], n_rope);
                        off += rope_bytes;
                    }
                }
                _ => {
                    return Err(StateSnapshotError::ShapeMismatch {
                        field: "layer_state_kind",
                        got: 0,
                        want: 1,
                    });
                }
            },
            LayerKind::LinearAttn => {
                let lb = linear_buffers.ok_or(StateSnapshotError::BuffersNotReady)?;
                let p = pool.ok_or(StateSnapshotError::BuffersNotReady)?;
                let linear_idx = linear_layer_idx_for(i).expect("layer_kind says LinearAttn");
                read_buffer_bytes(
                    p.handle(lb.conv_state[linear_idx]),
                    &mut buf[off..off + la_conv],
                );
                off += la_conv;
                read_buffer_bytes(
                    p.handle(lb.delta_state[linear_idx]),
                    &mut buf[off..off + la_ssm],
                );
                off += la_ssm;
            }
        }
    }

    debug_assert_eq!(off, need, "state_save wrote {off} bytes, expected {need}");
    Ok(off)
}

/// Replace current state with the one encoded in `buf`. Mirrors C
/// `mf_state_load` (infer.m:8599..8700). Two-pass: preflight verifies
/// the header + per-layer lengths fit in `buf` before any state is
/// mutated; restore then performs the writes.
pub fn state_load(
    buf: &[u8],
    layer_states: &mut LayerStates,
    mut linear_buffers: Option<&mut LayerForwardBuffers>,
    pool: Option<&crate::riir::backend::MetalBufferPool>,
    device: &Device,
) -> Result<(), StateSnapshotError> {
    use crate::riir::backend::BufferPool as _;
    let v = VARIANT;

    // Read magic + version up front so we can choose v1 vs v2 header
    // size.
    if buf.len() < 8 {
        return Err(StateSnapshotError::Truncated {
            layer: 0,
            need: 8,
            got: buf.len(),
        });
    }
    let read_u32 =
        |off: usize| -> u32 { u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) };
    let magic = read_u32(0);
    if magic != SNAPSHOT_MAGIC {
        return Err(StateSnapshotError::BadMagic {
            got: magic,
            want: SNAPSHOT_MAGIC,
        });
    }
    let version = read_u32(4);
    let header_words = match version {
        1 => SNAPSHOT_HEADER_V1_U32,
        2 => SNAPSHOT_HEADER_U32,
        3 => SNAPSHOT_HEADER_V3_U32,
        4 => SNAPSHOT_HEADER_V4_U32,
        _ => {
            return Err(StateSnapshotError::BadVersion {
                got: version,
                want: SNAPSHOT_VERSION,
            });
        }
    };
    let header_bytes = header_words * std::mem::size_of::<u32>();
    if buf.len() < header_bytes {
        return Err(StateSnapshotError::Truncated {
            layer: 0,
            need: header_bytes,
            got: buf.len(),
        });
    }

    // v1 snapshots can only be loaded into v1-compatible variants
    // (full-attn / linear-attn). MLA variants need v2.
    if version == 1 && v.attn_kind == AttnKind::Mla {
        return Err(StateSnapshotError::BadVersion {
            got: version,
            want: SNAPSHOT_VERSION,
        });
    }

    let check = |off: usize, field: &'static str, want: u32| -> Result<(), StateSnapshotError> {
        let got = read_u32(off);
        if got != want {
            return Err(StateSnapshotError::ShapeMismatch { field, got, want });
        }
        Ok(())
    };
    check(8, "num_layers", v.num_layers as u32)?;
    check(12, "full_attn_interval", v.full_attn_interval as u32)?;
    check(16, "num_kv_heads", v.num_kv_heads as u32)?;
    check(20, "head_dim", v.head_dim as u32)?;
    let la_conv = linear_conv_bytes(&v);
    let la_ssm = linear_ssm_bytes(&v);
    check(24, "linear_conv_bytes", la_conv as u32)?;
    check(28, "linear_ssm_bytes", la_ssm as u32)?;
    if version >= 2 {
        check(32, "kv_lora_rank", v.kv_lora_rank as u32)?;
        check(36, "qk_rope_head_dim", v.qk_rope_head_dim as u32)?;
    }
    // Layer range: v1/v2 snapshots cover the whole model; v3 names its
    // range, which must be exactly this context's.
    let span = layer_states.span();
    let (snap_start, snap_end) = if version >= 3 {
        (read_u32(40), read_u32(44))
    } else {
        (0, v.num_layers as u32)
    };
    if snap_start != span.start as u32 {
        return Err(StateSnapshotError::ShapeMismatch {
            field: "layer_start",
            got: snap_start,
            want: span.start as u32,
        });
    }
    if snap_end != span.end as u32 {
        return Err(StateSnapshotError::ShapeMismatch {
            field: "layer_end",
            got: snap_end,
            want: span.end as u32,
        });
    }

    // KV format: v4 names it, older versions are f32. It must be
    // this context's (`OpenConfig::kv_format`) — rows are not
    // converted on load.
    let fmt = kv_format_of(layer_states);
    let snap_fmt = if version >= 4 { read_u32(48) } else { KvFormat::F32.code() };
    if snap_fmt != fmt.code() {
        return Err(StateSnapshotError::ShapeMismatch {
            field: "kv_format",
            got: snap_fmt,
            want: fmt.code(),
        });
    }
    let fa_stride = full_attn_stride_bytes(&v, fmt);

    // Preflight: walk the body without mutating state. Any size
    // mismatch is caught here so a partial load doesn't leave the
    // ctx in a half-restored state. Mirrors the C-side preflight at
    // infer.m:8629..8649.
    {
        let mut q = header_bytes;
        for i in span.clone() {
            match v.layer_kind(i) {
                LayerKind::FullAttn => {
                    if buf.len() - q < 4 {
                        return Err(StateSnapshotError::Truncated {
                            layer: i,
                            need: 4,
                            got: buf.len() - q,
                        });
                    }
                    let len = i32::from_le_bytes(buf[q..q + 4].try_into().unwrap());
                    q += 4;
                    if len < 0 {
                        return Err(StateSnapshotError::NegativeLen { layer: i, len });
                    }
                    let cap = layer_states[i].kv_capacity().unwrap_or(0);
                    if (len as usize) > cap {
                        return Err(StateSnapshotError::LenOverflow {
                            layer: i,
                            len,
                            max: cap,
                        });
                    }
                    let bytes = match v.attn_kind {
                        AttnKind::Gqa => 2 * (len as usize) * fa_stride,
                        AttnKind::Mla => {
                            mla_latent_bytes(&v, len as usize) + mla_rope_k_bytes(&v, len as usize)
                        }
                    };
                    if buf.len() - q < bytes {
                        return Err(StateSnapshotError::Truncated {
                            layer: i,
                            need: bytes,
                            got: buf.len() - q,
                        });
                    }
                    q += bytes;
                }
                LayerKind::LinearAttn => {
                    let bytes = la_conv + la_ssm;
                    if buf.len() - q < bytes {
                        return Err(StateSnapshotError::Truncated {
                            layer: i,
                            need: bytes,
                            got: buf.len() - q,
                        });
                    }
                    q += bytes;
                }
            }
        }
    }

    // Preflight ok — restore.
    let mut off = header_bytes;
    for i in span {
        match v.layer_kind(i) {
            LayerKind::FullAttn => {
                let len = i32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
                off += 4;
                // MLA path — populate latent + rope_k Metal buffers.
                if v.attn_kind == AttnKind::Mla {
                    let cache = match &mut layer_states[i] {
                        LayerState::Mla(c) => c,
                        _ => {
                            return Err(StateSnapshotError::ShapeMismatch {
                                field: "layer_state_kind",
                                got: 0,
                                want: 1,
                            });
                        }
                    };
                    ensure_mla_buffers(cache, device);
                    if len > 0 {
                        let lat_bytes = mla_latent_bytes(&v, len as usize);
                        let rope_bytes = mla_rope_k_bytes(&v, len as usize);
                        let n_lat = (len as usize) * v.kv_lora_rank;
                        let n_rope = (len as usize) * v.qk_rope_head_dim;
                        let lat_buf = cache
                            .latent_cache
                            .as_ref()
                            .expect("ensure_mla_buffers just ran");
                        let rope_buf = cache
                            .rope_k_cache
                            .as_ref()
                            .expect("ensure_mla_buffers just ran");
                        write_buffer_bytes_n_f32(lat_buf, &buf[off..off + lat_bytes], n_lat);
                        off += lat_bytes;
                        write_buffer_bytes_n_f32(rope_buf, &buf[off..off + rope_bytes], n_rope);
                        off += rope_bytes;
                    }
                    cache.len = len;
                    continue;
                }
                let kv = match &mut layer_states[i] {
                    LayerState::FullAttn(kv) => kv,
                    LayerState::Mla(_) => {
                        unreachable!("attn_kind branch handled above")
                    }
                    LayerState::LinearAttn(_) => {
                        return Err(StateSnapshotError::ShapeMismatch {
                            field: "layer_state_kind",
                            got: 0,
                            want: 1,
                        });
                    }
                };
                // Phase 0b (prefill arc): KV is GPU-resident in the
                // pool — registered by `ensure_linear_resources`,
                // which `state_load` runs before reaching here. Write
                // the restored prefix straight into the shared-storage
                // buffers. The `[len, capacity)` tail is not zeroed
                // — SDPA reads are bounded by `len` and every appended
                // row is overwritten before it is read (mirrors the
                // MLA branch + `MlaKvCacheGpu`).
                if len > 0 {
                    let p = pool.ok_or(StateSnapshotError::BuffersNotReady)?;
                    let bytes = (len as usize) * fa_stride;
                    let k_buf = p.handle(kv.k_id.ok_or(StateSnapshotError::BuffersNotReady)?);
                    write_buffer_bytes(k_buf, &buf[off..off + bytes]);
                    off += bytes;
                    let v_buf = p.handle(kv.v_id.ok_or(StateSnapshotError::BuffersNotReady)?);
                    write_buffer_bytes(v_buf, &buf[off..off + bytes]);
                    off += bytes;
                }
                kv.len = len;
                // The restored KV lives in the canonical pool buffers
                // the oracle SDPA reads directly (#2 removed the GPU
                // mirror), so the bytes copied above are all the GPU
                // fast path needs — no separate mirror sync.
            }
            LayerKind::LinearAttn => {
                let lb = linear_buffers
                    .as_deref_mut()
                    .ok_or(StateSnapshotError::BuffersNotReady)?;
                let p = pool.ok_or(StateSnapshotError::BuffersNotReady)?;
                let linear_idx = linear_layer_idx_for(i).expect("layer_kind says LinearAttn");
                write_buffer_bytes(
                    p.handle(lb.conv_state[linear_idx]),
                    &buf[off..off + la_conv],
                );
                off += la_conv;
                write_buffer_bytes(
                    p.handle(lb.delta_state[linear_idx]),
                    &buf[off..off + la_ssm],
                );
                off += la_ssm;
            }
        }
    }
    Ok(())
}

/// Public-facing convenience for `RsCtx`-aware callers that don't
/// have direct access to the `linear_buffers` field. Drains EVERY
/// in-flight deferred dispatch (the streammoe.h:481 contract) so the
/// snapshot reflects post-token state, not mid-flight state. Slice
/// 5d-9 widened the deferred slot to a depth-2 ring; this drains
/// the whole ring.
pub(in crate::riir) fn drain_deferred(deferred: &mut deferred::DeferredRing) {
    deferred::discard_deferred_experts_in(deferred);
}

/// Copy `dst.len()` bytes from a shared-storage Metal buffer to host.
///
/// # Safety
///
/// Caller (always `state_save`) drains pending deferred dispatch
/// before reading. No other path reaches this function.
fn read_buffer_bytes(buf: &Buffer, dst: &mut [u8]) {
    let n = dst.len();
    debug_assert!(buf.length() as usize >= n);
    // SAFETY: see fn docs.
    unsafe {
        std::ptr::copy_nonoverlapping(buf.contents() as *const u8, dst.as_mut_ptr(), n);
    }
}

/// Copy `src.len()` bytes from host into a shared-storage Metal
/// buffer.
///
/// # Safety
///
/// Caller (always `state_load`) drains pending deferred dispatch
/// before writing. No other path reaches this function.
fn write_buffer_bytes(buf: &Buffer, src: &[u8]) {
    let n = src.len();
    debug_assert!(buf.length() as usize >= n);
    // SAFETY: see fn docs.
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), buf.contents() as *mut u8, n);
    }
}
