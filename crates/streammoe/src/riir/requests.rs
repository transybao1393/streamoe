//! Per-request sequence state (plan phase 4).
//!
//! A context can hold several sequences at once — one per request in
//! flight through a pipeline. Each request owns a [`SeqState`]: its
//! layers' KV caches and linear-attn state (host + GPU recurrence
//! buffers) and its prefetch guesses. Everything else (weights, the
//! hot tier, per-step scratch) is shared.
//!
//! **Fixed pool, no growth.** [`crate::riir::OpenConfig::max_requests`]
//! states are allocated (and, under a cap, touched) at open and
//! counted in the budget plan. Admission takes a free slot or fails
//! with [`crate::riir::RsError::PoolFull`]; nothing is allocated per
//! request at runtime.
//!
//! **State switching, not copying.** The forward pass reads the
//! *active* request's state from the context's usual fields. Switching
//! request swaps those fields with the request's parked [`SeqState`] —
//! a few `Vec` pointer swaps, no copies — and only when the request
//! changes, so single-request decode pays nothing.
//!
//! The deferred K-expert ring is not per request: every eval call
//! drains it before returning, so it is empty at every request
//! switch.
//!
//! Slot 0 is the [`ReqId::DEFAULT`] request, always admitted; the
//! single-sequence API (`eval_prompt`, `eval_token`, `memory_*`,
//! `state_*`, checkpoints) runs on it.

use crate::riir::backend::BufId;
use crate::riir::backend::buftype::{ConvStateBuf, DeltaStateBuf};
use crate::riir::moe::expert_forward::MAX_K;
use crate::riir::range::RangeVec;
use crate::riir::snapshot::state::LayerStates;

/// Identifies one request (sequence) in a context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReqId(pub u64);

impl ReqId {
    /// The always-present request the single-sequence API uses.
    pub const DEFAULT: ReqId = ReqId(0);
}

/// One request's sequence state while it is not the active one.
#[derive(Debug, Default)]
pub struct SeqState {
    /// Per-layer KV / linear-attn host state for the context's range.
    pub layer_states: LayerStates,
    /// GPU linear-attn recurrence buffers (empty until the GPU
    /// resources exist, or for a range without linear-attn layers).
    pub conv_state: RangeVec<BufId<ConvStateBuf>>,
    pub delta_state: RangeVec<BufId<DeltaStateBuf>>,
    /// Last-token routing per layer — the prefetch guesses.
    pub predictions: Vec<Option<[i32; MAX_K]>>,
}

/// Why admission failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmitError {
    /// Every slot holds a request.
    Full,
    /// The request is already admitted.
    Exists,
}

/// Which request owns each pool slot, and which slot is active.
#[derive(Debug, Clone)]
pub struct RequestTable {
    owners: Vec<Option<ReqId>>,
    active: usize,
}

impl RequestTable {
    /// `max_requests` slots (≥ 1); slot 0 holds [`ReqId::DEFAULT`].
    pub fn new(max_requests: usize) -> Self {
        let mut owners = vec![None; max_requests.max(1)];
        owners[0] = Some(ReqId::DEFAULT);
        Self { owners, active: 0 }
    }

    /// Pool size.
    pub fn capacity(&self) -> usize {
        self.owners.len()
    }

    /// Slots holding a request.
    pub fn in_use(&self) -> usize {
        self.owners.iter().filter(|o| o.is_some()).count()
    }

    /// Slot of `req`, if admitted.
    pub fn slot_of(&self, req: ReqId) -> Option<usize> {
        self.owners.iter().position(|&o| o == Some(req))
    }

    /// Slot whose state is in the context's live fields.
    pub fn active(&self) -> usize {
        self.active
    }

    pub fn set_active(&mut self, slot: usize) {
        self.active = slot;
    }

    /// Take a free slot for `req`.
    pub fn admit(&mut self, req: ReqId) -> Result<usize, AdmitError> {
        if self.slot_of(req).is_some() {
            return Err(AdmitError::Exists);
        }
        let slot = self
            .owners
            .iter()
            .position(Option::is_none)
            .ok_or(AdmitError::Full)?;
        self.owners[slot] = Some(req);
        Ok(slot)
    }

    /// Free `req`'s slot. The default request is never released.
    /// Returns the freed slot.
    pub fn release(&mut self, req: ReqId) -> Option<usize> {
        if req == ReqId::DEFAULT {
            return None;
        }
        let slot = self.slot_of(req)?;
        self.owners[slot] = None;
        Some(slot)
    }

    /// Requests currently admitted, in slot order.
    pub fn requests(&self) -> impl Iterator<Item = ReqId> + '_ {
        self.owners.iter().filter_map(|&o| o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_request_always_present() {
        let t = RequestTable::new(1);
        assert_eq!(t.capacity(), 1);
        assert_eq!(t.slot_of(ReqId::DEFAULT), Some(0));
        assert_eq!(t.in_use(), 1);
        // `0` is clamped up: there is always the default slot.
        assert_eq!(RequestTable::new(0).capacity(), 1);
    }

    #[test]
    fn admission_fills_then_refuses() {
        let mut t = RequestTable::new(3);
        assert_eq!(t.admit(ReqId(7)), Ok(1));
        assert_eq!(t.admit(ReqId(7)), Err(AdmitError::Exists));
        assert_eq!(t.admit(ReqId(9)), Ok(2));
        assert_eq!(t.admit(ReqId(11)), Err(AdmitError::Full));
        assert_eq!(t.in_use(), 3);
        let all: Vec<_> = t.requests().collect();
        assert_eq!(all, vec![ReqId::DEFAULT, ReqId(7), ReqId(9)]);
    }

    #[test]
    fn release_frees_a_slot_but_never_the_default() {
        let mut t = RequestTable::new(2);
        assert_eq!(t.admit(ReqId(5)), Ok(1));
        assert_eq!(t.release(ReqId::DEFAULT), None);
        assert_eq!(t.release(ReqId(6)), None);
        assert_eq!(t.release(ReqId(5)), Some(1));
        assert_eq!(t.slot_of(ReqId(5)), None);
        assert_eq!(t.admit(ReqId(6)), Ok(1));
    }
}
