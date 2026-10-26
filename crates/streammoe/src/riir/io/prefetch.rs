//! Speculative-prefetch state machine for K-expert disk reads.
//!
//! Slice 5d-6b. Mirrors the C path's `g_async_pread` + `data_B`
//! prefetch (`metal_infer/infer.m:3240-3282, 4460-4470, 5510-5601`).
//!
//! # Concept
//!
//! Each layer's K active-expert disk reads are normally on the
//! critical path between MoE router (CPU) and CMD3 dispatch (GPU).
//! Slice 5d-6a parallelized them across 8 worker threads but kept
//! them synchronous. This slice (5d-6b) overlaps them with GPU
//! compute by prefetching the next layer's predicted experts into a
//! separate buffer set, asynchronously, so by the time CMD3 of layer
//! N completes and layer N+1 begins its expert dispatch, the experts
//! are already resident.
//!
//! Prediction = "the K indices the router selected for this same
//! layer in the previous token" — same as C
//! (`metal_infer/infer.m:5510-5562`). Token-to-token expert locality
//! is high in practice, so this hit rate is empirically good.
//!
//! # Soundness
//!
//! Async prefetch closures receive raw pointers into the
//! `MoeBuffers.data_prefetch[slot]` byte buffers and into
//! [`ExpertFiles`] via the [`DataPrefetchPtr`] / [`ExpertFilesPtr`]
//! newtypes (both with `unsafe impl Send`). Sound because:
//!
//! - **`MoeBuffers.data_prefetch[slot]`** is allocated once at
//!   `RsCtx` lazy-init and never reallocated. The pointer is valid
//!   for as long as `RsCtx` is alive.
//! - **`ExpertFiles`** is on `RsCtx` and exposes only `&self`
//!   methods; `pread64` is thread-safe per POSIX (per-call offset,
//!   no shared file position).
//! - **No aliasing of `data_prefetch[slot]`**: the K dispatched
//!   tasks each get a *disjoint* slot pointer (one per slot index);
//!   no GPU dispatch reads from `data_prefetch[slot]` until the
//!   prefetch has been [`PrefetchState::wait_for`]ed.
//! - **No use-after-free**: the drain points
//!   ([`PrefetchState::wait_for`], [`PrefetchState::invalidate_all`],
//!   [`PrefetchState::drain`], and [`PrefetchState`]'s `Drop` impl)
//!   complete all in-flight prefetches before the caller can mutate
//!   `MoeBuffers` or drop the `RsCtx`.
//!
//! Violating any of those requires misuse that the type signatures
//! don't enforce — same shape of contract as the deferred-experts
//! state machine in [`super::deferred`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};

use crate::riir::io::expert_io::{ExpertFiles, ExpertIoError};
use crate::riir::io::expert_io_mode::ExpertIoMode;
use crate::riir::io::guess::CrossLayerGuesser;
use crate::riir::io::io_stats::ReadKind;
use crate::riir::moe::expert_forward::MAX_K;

/// Per-slot decision: which buffer the K-expert encoder reads from.
/// Set by the per-slot resolution in
/// `linear_attn_forward::post_attention_tail` (one entry per slot per
/// layer) and consumed by `gpu_batched_experts_*` in
/// [`super::expert_forward`].
///
/// Matching is set-based, not position-locked: an actual expert at
/// slot `s` may have been prefetched into a *different* buffer index
/// than `s`. The `Prefetched(buf_idx)` payload tells the encoder
/// which prefetch buffer holds this slot's expert weights, regardless
/// of the order the prefetch loaded them in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SlotSource {
    /// Slot was sync-pread into `MoeBuffers.data_synced[slot]` (miss
    /// or first-touch). Encoder binds `data_synced[slot]`.
    Synced,
    /// Slot's expert was found in the prefetch buffer at index
    /// `buf_idx`. Encoder binds
    /// `data_prefetch[prefetch_set][buf_idx]`.
    Prefetched(usize),
}

/// How speculative prefetch guesses the experts of the layer about to
/// run ([`crate::riir::OpenConfig::prefetch_guess`], plan phase 6).
/// Either way the forward pass routes on its own logits; the guess
/// only decides which experts are read early.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum PrefetchGuess {
    /// The previous token's picks for the same layer (historical).
    #[default]
    PrevToken,
    /// Layer L's post-attention residual through layer L+1's norm and
    /// router gate, on the CPU ([`super::guess::CrossLayerGuesser`]);
    /// falls back to `PrevToken` for a range's first layer and where
    /// the gate can't be read. GQA variants in pread mode only.
    CrossLayer,
}

/// State machine owned by `RsCtx`. Tracks per-layer predictions and
/// up to two in-flight async prefetches — one per prefetch set, i.e.
/// per target-layer parity (the second only with
/// [`Self::set_lookahead`]).
#[derive(Debug)]
pub struct PrefetchState {
    /// Per-layer last-token K indices, used as the prediction for
    /// the next token's same-layer prefetch. `None` until the layer
    /// has been run at least once (or until `invalidate_all` resets).
    last_token_indices: Vec<Option<[i32; MAX_K]>>,
    /// In-flight prefetch per set (`target_layer % 2`), if any.
    in_flight: [Option<InFlight>; 2],
    /// Issue layer L+1's prefetch during layer L's attention
    /// ([`crate::riir::OpenConfig::prefetch_lookahead`]).
    lookahead: bool,
    /// Slot-level hit counter (per-slot prediction matched the
    /// per-token routing). Accumulates across the lifetime of the
    /// state; reset via [`PrefetchState::reset_stats`] for per-request
    /// scoping. Atomic so the counters can be read from any thread,
    /// though the increment site is single-threaded (the orchestrator
    /// loop in `linear_attn_forward::post_attention_tail`).
    hits: AtomicU64,
    /// Slot-level miss counter (no in-flight prefetch, prefetch was
    /// for a different layer, prediction didn't match, or the slot
    /// index ran past the prefetch's K). Same lifecycle as `hits`.
    misses: AtomicU64,
    /// Which expert-IO path the K-expert dispatch uses — picked once
    /// at [`crate::riir::RsCtx::open`] by
    /// [`crate::riir::io::expert_io_mode::select`]. The orchestrator
    /// loop and `moe_dispatch_per_token` both consult this so the
    /// firing side and the consuming side agree on whether prefetch
    /// I/O is happening at all.
    mode: ExpertIoMode,
    /// Cross-layer guesser when [`PrefetchGuess::CrossLayer`] is on.
    cross: Option<CrossLayerGuesser>,
    /// Experts per layer a prefetch reads, strongest guess first
    /// ([`crate::riir::OpenConfig::prefetch_experts`]); `0` = off.
    max_experts: usize,
}

#[derive(Debug)]
struct InFlight {
    /// Layer this prefetch was launched for.
    target_layer: usize,
    /// The K indices the prefetch loaded into
    /// `data_prefetch[0..k]`. [`PrefetchState::wait_for`] returns
    /// these so the caller can match per-slot against the actual
    /// routing for `target_layer`.
    loaded_indices: [i32; MAX_K],
    /// Number of K tasks dispatched (= K-active for the layer).
    k: usize,
    /// Per-task completion channel.
    rx: Receiver<Result<(), ExpertIoError>>,
}

/// Result of [`PrefetchState::wait_for`] when a prefetch was both
/// in-flight and targeted at the requested layer.
#[derive(Copy, Clone, Debug)]
pub struct PrefetchStatus {
    /// Indices the prefetch loaded — caller matches per-slot vs
    /// actual routing for the layer.
    pub loaded_indices: [i32; MAX_K],
    /// Number of valid entries in `loaded_indices` (= K-active).
    pub k: usize,
}

/// `Send`-able handle to one `data_prefetch[slot]` byte buffer. Stored
/// as a `usize` (the pointer's address) so the closure-capture
/// auto-trait inference doesn't see a `*mut u8`/`NonNull<u8>` that
/// `!Send` propagates from. Reconstituted to `*mut u8` only inside
/// the worker closure under the `unsafe` block.
#[derive(Copy, Clone, Debug)]
struct DataPrefetchPtr {
    ptr_addr: usize,
    len: usize,
}

impl DataPrefetchPtr {
    fn from_slice(s: &mut [u8]) -> Self {
        debug_assert!(!s.is_empty(), "data_prefetch slot must be non-empty");
        Self {
            ptr_addr: s.as_mut_ptr() as usize,
            len: s.len(),
        }
    }

    /// Reconstruct the original `&mut [u8]` from the stored address.
    ///
    /// # Safety
    ///
    /// Caller must uphold the module-level soundness invariants:
    /// - The backing `MoeBuffers.data_prefetch[slot]` allocation must
    ///   still be alive (i.e. its owning `RsCtx` has not been dropped
    ///   and the buffer has not been reallocated). Reallocation is
    ///   forbidden by construction — slots are sized once at lazy
    ///   init.
    /// - No other thread may be reading or writing the same slot
    ///   concurrently. The K worker closures get *disjoint* slot
    ///   pointers, so this holds when called from a worker; for
    ///   anything else, the caller must drain in-flight prefetches
    ///   (`PrefetchState::wait_for` / `drain` / `invalidate_all`)
    ///   first.
    /// - The arbitrary lifetime `'a` chosen by the caller must not
    ///   outlive the underlying allocation.
    unsafe fn as_mut_slice<'a>(self) -> &'a mut [u8] {
        // SAFETY: forwarded — see the safety contract above.
        unsafe { std::slice::from_raw_parts_mut(self.ptr_addr as *mut u8, self.len) }
    }
}

/// `Send + Sync` handle to the `ExpertFiles` instance. Same `usize`
/// trick as [`DataPrefetchPtr`] so the closure's auto-trait
/// derivation doesn't see a raw pointer.
#[derive(Copy, Clone, Debug)]
struct ExpertFilesPtr {
    addr: usize,
}

impl ExpertFilesPtr {
    fn from_ref(r: &ExpertFiles) -> Self {
        Self {
            addr: (r as *const ExpertFiles) as usize,
        }
    }

    /// Reconstruct the original `&ExpertFiles` from the stored
    /// address.
    ///
    /// # Safety
    ///
    /// The `ExpertFiles` lives on `RsCtx` and is never moved or
    /// dropped while prefetch tasks are in flight — the
    /// `PrefetchState` is drained before any teardown path. Callers
    /// must preserve that property:
    /// - Never construct an `ExpertFilesPtr` and let it outlive the
    ///   `RsCtx` that holds the `ExpertFiles`.
    /// - The arbitrary lifetime `'a` chosen by the caller must not
    ///   outlive the underlying `RsCtx`.
    /// - `ExpertFiles`'s public surface is `&self`-only (`pread64` is
    ///   per-call-offset thread-safe per POSIX), so concurrent worker
    ///   calls through this reference are fine.
    unsafe fn as_ref<'a>(self) -> &'a ExpertFiles {
        // SAFETY: forwarded — see the safety contract above.
        unsafe { &*(self.addr as *const ExpertFiles) }
    }
}

impl PrefetchState {
    /// Create a fresh state with `num_layers` slots, all unprimed.
    pub fn new(num_layers: usize, mode: ExpertIoMode) -> Self {
        Self {
            last_token_indices: vec![None; num_layers],
            in_flight: [None, None],
            lookahead: false,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            mode,
            cross: None,
            max_experts: MAX_K,
        }
    }

    /// Read at most `n` experts per layer per prefetch (guesses are
    /// ordered strongest first); `0` turns speculative prefetch off.
    pub fn set_max_experts(&mut self, n: usize) {
        self.max_experts = n.min(MAX_K);
    }

    /// Experts per layer a prefetch reads.
    pub fn max_experts(&self) -> usize {
        self.max_experts
    }

    /// Turn on cross-layer guesses ([`PrefetchGuess::CrossLayer`]).
    pub fn enable_cross_layer(&mut self, guesser: CrossLayerGuesser) {
        self.cross = Some(guesser);
    }

    /// Issue each layer's prefetch one layer early (plan phase 6
    /// overlap). See [`crate::riir::OpenConfig::prefetch_lookahead`].
    pub fn set_lookahead(&mut self, on: bool) {
        self.lookahead = on;
    }

    /// `true` if lookahead prefetch is on.
    pub fn lookahead(&self) -> bool {
        self.lookahead
    }

    /// `true` if a prefetch targeting `layer_idx` is in flight.
    pub fn in_flight_for(&self, layer_idx: usize) -> bool {
        self.in_flight[layer_idx % 2]
            .as_ref()
            .is_some_and(|f| f.target_layer == layer_idx)
    }

    /// The cross-layer guesser, when enabled.
    pub fn cross_mut(&mut self) -> Option<&mut CrossLayerGuesser> {
        self.cross.as_mut()
    }

    /// Expert-IO mode for this run. See [`ExpertIoMode`].
    pub fn mode(&self) -> ExpertIoMode {
        self.mode
    }

    /// Record a per-layer outcome: how many of the K slots were
    /// satisfied by a prefetch hit, and how many fell back to a sync
    /// pread. Called from the orchestrator at the resolution site
    /// (`linear_attn_forward::post_attention_tail`).
    pub fn record_outcome(&self, hits: u64, misses: u64) {
        if hits > 0 {
            self.hits.fetch_add(hits, Ordering::Relaxed);
        }
        if misses > 0 {
            self.misses.fetch_add(misses, Ordering::Relaxed);
        }
    }

    /// Read the accumulated `(hits, misses)` counters.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    /// Zero the counters (e.g. for per-request scoping).
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Drain any in-flight prefetch (wait for all K tasks). Used by
    /// `memory_clear`, `state_save`, `state_load`, and the `Drop`
    /// impl. Per-task errors are dropped — caller of `drain` (as
    /// distinct from `wait_for`) shouldn't be reading
    /// `data_prefetch[slot]` afterward.
    pub fn drain(&mut self) {
        for set in 0..2 {
            self.drain_set(set);
        }
    }

    /// Drain the in-flight prefetch of one set, if any.
    fn drain_set(&mut self, set: usize) {
        if let Some(in_flight) = self.in_flight[set].take() {
            for _ in 0..in_flight.k {
                let _ = in_flight.rx.recv();
            }
        }
    }

    /// Drain any in-flight prefetch AND clear all per-layer
    /// predictions. Used by `memory_clear` so the next token starts
    /// from cold-prediction state (no stale predictions from before
    /// the clear).
    pub fn invalidate_all(&mut self) {
        self.drain();
        for slot in self.last_token_indices.iter_mut() {
            *slot = None;
        }
        if let Some(c) = self.cross.as_mut() {
            c.clear();
        }
    }

    /// Wait for any in-flight prefetch in `layer_idx`'s set,
    /// returning the loaded indices if (a) the target matches, and
    /// (b) all K tasks completed without error. Returns `None` if
    /// no prefetch was in flight, the target was a different layer
    /// (stale prefetch), or any task errored. In all cases, that
    /// set's in-flight slot is drained and consumed; the other set's
    /// (a lookahead for the next layer) is left running.
    pub fn wait_for(&mut self, layer_idx: usize) -> Option<PrefetchStatus> {
        let in_flight = self.in_flight[layer_idx % 2].take()?;
        let mut all_ok = true;
        for _ in 0..in_flight.k {
            match in_flight.rx.recv() {
                Ok(Ok(())) => {}
                _ => all_ok = false,
            }
        }
        if !all_ok || in_flight.target_layer != layer_idx {
            None
        } else {
            Some(PrefetchStatus {
                loaded_indices: in_flight.loaded_indices,
                k: in_flight.k,
            })
        }
    }

    /// Swap the per-layer predictions with `other` — a request switch
    /// (plan phase 4): each request keeps its own last-token routing as
    /// its guesses. Drains any in-flight prefetch first.
    pub fn swap_predictions(&mut self, other: &mut Vec<Option<[i32; MAX_K]>>) {
        self.drain();
        std::mem::swap(&mut self.last_token_indices, other);
        if let Some(c) = self.cross.as_mut() {
            c.clear();
        }
    }

    /// The prediction for `layer_idx`, if one exists. `None` means
    /// no prediction yet (first token, or post-`invalidate_all`).
    pub fn predict_for(&self, layer_idx: usize) -> Option<[i32; MAX_K]> {
        self.last_token_indices.get(layer_idx).copied().flatten()
    }

    /// Record this token's actual routing for `layer_idx`, becoming
    /// the prediction for the next token's same layer.
    pub fn record_actual(&mut self, layer_idx: usize, actual: [i32; MAX_K]) {
        if let Some(slot) = self.last_token_indices.get_mut(layer_idx) {
            *slot = Some(actual);
        }
    }

    /// Fire-and-forget: kick off async prefetch of `predicted[0..k]`
    /// into the K caller-supplied `data_prefetch` slot pointers via
    /// `pool`. Stores the task receiver in `self.in_flight`. Caller
    /// must drain (via `wait_for` / `drain` / `invalidate_all`)
    /// before any subsequent mutation of `data_prefetch[0..k]`,
    /// before any GPU read of `data_prefetch[slot]`, and before
    /// dropping the `RsCtx`.
    ///
    /// `target_layer` is the layer the prefetch is FOR (not the
    /// layer currently running). `k` is the number of valid
    /// entries in `predicted` and the count of slot pointers used
    /// from `data_prefetch`.
    ///
    /// If a prior prefetch of the same set is still in flight, drains
    /// it first. Correct callers won't hit that path; the drain is
    /// defensive. The other set's prefetch keeps running.
    ///
    /// Only the first [`Self::max_experts`] guesses are read (guesses
    /// are ordered strongest first); with `0` nothing is dispatched.
    pub fn dispatch(
        &mut self,
        target_layer: usize,
        mut predicted: [i32; MAX_K],
        k: usize,
        data_prefetch: [&mut [u8]; MAX_K],
        pool: &rayon::ThreadPool,
        expert_files: &ExpertFiles,
    ) {
        if self.max_experts == 0 {
            return;
        }
        for e in predicted[self.max_experts.min(k)..k].iter_mut() {
            *e = -1;
        }
        let set = target_layer % 2;
        self.drain_set(set);
        let (tx, rx) = sync_channel::<Result<(), ExpertIoError>>(MAX_K);
        let efp = ExpertFilesPtr::from_ref(expert_files);
        // Capture the K disjoint slot pointers BEFORE spawning;
        // the borrow checker's `data_prefetch: [&mut [u8]; MAX_K]`
        // input is consumed by this conversion.
        let mut slot_ptrs: [Option<DataPrefetchPtr>; MAX_K] = std::array::from_fn(|_| None);
        for (i, dst) in data_prefetch.into_iter().enumerate() {
            slot_ptrs[i] = Some(DataPrefetchPtr::from_slice(dst));
        }
        for slot in 0..k {
            // Negative entries are slots the caller masked out (expert
            // already resident in the hot tier): nothing to read.
            if predicted[slot] < 0 {
                let _ = tx.send(Ok(()));
                continue;
            }
            let expert_idx = predicted[slot] as usize;
            let dst_ptr = slot_ptrs[slot].expect("slot 0..k populated");
            let tx = tx.clone();
            // SAFETY: see module-level docs. `dst_ptr` and `efp` are
            // valid for the duration of the in-flight prefetch; the
            // drain-before-touch discipline guarantees no aliasing.
            pool.spawn(move || {
                let dst = unsafe { dst_ptr.as_mut_slice() };
                let efs = unsafe { efp.as_ref() };
                let r = efs.read_expert_disk_as(ReadKind::Prefetch, target_layer, expert_idx, dst);
                let _ = tx.send(r);
            });
        }
        self.in_flight[set] = Some(InFlight {
            target_layer,
            loaded_indices: predicted,
            k,
            rx,
        });
    }
}

/// Mask out (set to `-1`) the entries of `predicted` that are already
/// resident in `hot` — they bind from RAM, so reading them would only
/// spend SSD bandwidth.
pub fn mask_hot(
    hot: Option<&crate::riir::io::hot_tier::HotTier>,
    layer_idx: usize,
    predicted: &mut [i32],
) {
    if let Some(hot) = hot {
        for e in predicted.iter_mut() {
            if *e >= 0 && hot.contains(layer_idx, *e as usize) {
                *e = -1;
            }
        }
    }
}

impl Drop for PrefetchState {
    fn drop(&mut self) {
        // Final drain — soundness contract requires no in-flight
        // tasks holding pointers when the underlying `MoeBuffers` /
        // `ExpertFiles` are dropped. `RsCtx`'s field drop order is
        // declaration order; `prefetch` field is declared after
        // `moe_buffers` and `experts` so this drop runs first.
        self.drain();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_for_returns_none_until_recorded() {
        let mut st = PrefetchState::new(8, ExpertIoMode::Pread);
        assert_eq!(st.predict_for(3), None);
        let actual = [0i32; MAX_K];
        st.record_actual(3, actual);
        assert_eq!(st.predict_for(3), Some(actual));
        st.invalidate_all();
        assert_eq!(st.predict_for(3), None);
    }

    #[test]
    fn in_flight_tracked_per_set() {
        let mut st = PrefetchState::new(4, ExpertIoMode::Pread);
        assert!(!st.in_flight_for(1));
        let (tx, rx) = sync_channel(MAX_K);
        tx.send(Ok(())).unwrap();
        st.in_flight[1] = Some(InFlight {
            target_layer: 1,
            loaded_indices: [7; MAX_K],
            k: 1,
            rx,
        });
        assert!(st.in_flight_for(1));
        assert!(!st.in_flight_for(3), "same set, other layer");
        // Waiting for the other set's layer leaves set 1 running.
        assert!(st.wait_for(2).is_none());
        assert!(st.in_flight_for(1));
        let got = st.wait_for(1).expect("prefetch for layer 1");
        assert_eq!((got.k, got.loaded_indices[0]), (1, 7));
        assert!(!st.in_flight_for(1));
    }

    #[test]
    fn predict_for_out_of_range_layer_is_none() {
        let st = PrefetchState::new(2, ExpertIoMode::Pread);
        assert_eq!(st.predict_for(99), None);
    }
}
