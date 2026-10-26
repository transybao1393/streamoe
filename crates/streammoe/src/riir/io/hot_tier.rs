//! Fixed-size, frequency-managed expert cache in RAM ("hot tier").
//!
//! Under a hard memory cap experts are read with `F_NOCACHE`, so the
//! OS page cache no longer keeps hot experts resident. The hot tier
//! takes over that job inside the budget:
//!
//! - **Fixed size.** `capacity` expert slots in shared-storage Metal
//!   buffers (chunked, each ≤ [`CHUNK_BYTES`]), allocated and touched
//!   once at open. Eviction swaps contents; the total never changes.
//! - **Zero-copy decode hits.** The per-token K-expert dispatch binds
//!   `(chunk buffer, slot offset)` directly ([`HotTier::lookup`]).
//! - **Frequency-based admission.** Every routed pick bumps a per
//!   `(layer, expert)` counter (halved periodically, so old popularity
//!   fades). A miss is admitted only if it is hotter than a sampled
//!   victim (approximate LFU, as Redis does it), so one-off experts
//!   don't churn the tier.
//! - **GPU hazard rule.** A slot bound by the current token's dispatch
//!   may still be read by in-flight GPU work, so it is never chosen as
//!   a victim until [`HotTier::begin_epoch`] marks the next token. All
//!   GPU work of earlier tokens has completed by then (each token ends
//!   by reading back logits on the same queue).
//!
//! Counters can be saved as a routing profile and used to seed the
//! tier at open ([`HotTier::seed_from_counts`]), so a fresh process
//! doesn't start cold.

use std::collections::HashMap;
use std::sync::Mutex;

use metal::{Buffer, Device, MTLResourceOptions, NSUInteger};

/// Upper bound per backing Metal buffer.
pub const CHUNK_BYTES: usize = 1 << 30;

/// Halve every counter after this many recorded picks.
const DECAY_EVERY: u64 = 1 << 16;

/// Victims sampled per admission.
const VICTIM_SAMPLES: usize = 8;

/// Magic for routing-profile files.
const PROFILE_MAGIC: &[u8; 4] = b"MFHP";

/// Errors from hot-tier construction and profiles.
#[derive(Debug, thiserror::Error)]
pub enum HotTierError {
    #[error("hot tier allocation failed: {bytes} bytes (out of device memory?)")]
    AllocFailed { bytes: usize },
    #[error("routing profile I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("routing profile shape {layers}×{experts} does not match model {want_layers}×{want_experts}")]
    ProfileShape {
        layers: u32,
        experts: u32,
        want_layers: usize,
        want_experts: usize,
    },
    #[error("not a routing profile (bad magic or length)")]
    ProfileFormat,
}

/// Point-in-time hot-tier counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HotTierStats {
    /// Slots in the tier.
    pub capacity: usize,
    /// Slots holding an expert.
    pub resident: usize,
    /// Lookups served from the tier.
    pub hits: u64,
    /// Lookups that missed.
    pub misses: u64,
    /// Experts copied into the tier.
    pub admissions: u64,
    /// Admissions that replaced another expert.
    pub evictions: u64,
}

#[derive(Debug)]
struct Inner {
    /// `(layer, expert)` → slot.
    map: HashMap<(u16, u16), u32>,
    /// slot → `(layer, expert)`.
    owner: Vec<Option<(u16, u16)>>,
    /// Epoch in which each slot was last bound for GPU reads.
    last_bound: Vec<u64>,
    /// Popularity per `layer * num_experts + expert`.
    counts: Vec<u32>,
    picks: u64,
    epoch: u64,
    /// xorshift state for victim sampling.
    rng: u64,
    stats: HotTierStats,
}

/// See the module docs.
pub struct HotTier {
    chunks: Vec<Buffer>,
    slots_per_chunk: usize,
    capacity: usize,
    expert_size: usize,
    num_layers: usize,
    num_experts: usize,
    inner: Mutex<Inner>,
}

// SAFETY: `metal::Buffer` handles are immutable references to
// shared-storage memory; every mutation of slot contents and of the
// index happens under `inner`'s lock.
unsafe impl Send for HotTier {}
unsafe impl Sync for HotTier {}

impl std::fmt::Debug for HotTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HotTier")
            .field("capacity", &self.capacity)
            .field("chunks", &self.chunks.len())
            .field("expert_size", &self.expert_size)
            .finish()
    }
}

impl HotTier {
    /// Number of whole expert slots that fit in `bytes`.
    pub fn slots_for_bytes(bytes: u64, expert_size: usize) -> usize {
        (bytes / expert_size as u64) as usize
    }

    /// Allocate `capacity` slots and touch every page, so the tier's
    /// full size is resident (and visible to the footprint) now.
    pub fn new(
        device: &Device,
        capacity: usize,
        expert_size: usize,
        num_layers: usize,
        num_experts: usize,
    ) -> Result<Self, HotTierError> {
        Self::with_chunk_bytes(device, capacity, expert_size, num_layers, num_experts, CHUNK_BYTES)
    }

    /// [`Self::new`] with an explicit per-buffer size limit.
    pub fn with_chunk_bytes(
        device: &Device,
        capacity: usize,
        expert_size: usize,
        num_layers: usize,
        num_experts: usize,
        chunk_bytes: usize,
    ) -> Result<Self, HotTierError> {
        let slots_per_chunk = (chunk_bytes / expert_size).max(1);
        let mut chunks = Vec::new();
        let mut left = capacity;
        while left > 0 {
            let n = left.min(slots_per_chunk);
            let bytes = n * expert_size;
            let buf = device.new_buffer(bytes as NSUInteger, MTLResourceOptions::StorageModeShared);
            if buf.contents().is_null() {
                return Err(HotTierError::AllocFailed { bytes });
            }
            // SAFETY: fresh shared buffer of `bytes` length, not yet
            // visible to the GPU.
            unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes) };
            chunks.push(buf);
            left -= n;
        }
        Ok(Self {
            chunks,
            slots_per_chunk,
            capacity,
            expert_size,
            num_layers,
            num_experts,
            inner: Mutex::new(Inner {
                map: HashMap::with_capacity(capacity),
                owner: vec![None; capacity],
                last_bound: vec![0; capacity],
                counts: vec![0; num_layers * num_experts],
                picks: 0,
                epoch: 1,
                rng: 0x9E37_79B9_7F4A_7C15,
                stats: HotTierStats {
                    capacity,
                    ..HotTierStats::default()
                },
            }),
        })
    }

    /// Slots in the tier.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn slot_location(&self, slot: u32) -> (&Buffer, usize) {
        let slot = slot as usize;
        (
            &self.chunks[slot / self.slots_per_chunk],
            (slot % self.slots_per_chunk) * self.expert_size,
        )
    }

    fn slot_ptr(&self, slot: u32) -> *mut u8 {
        let (buf, off) = self.slot_location(slot);
        // SAFETY: `off + expert_size` is within the chunk by
        // construction.
        unsafe { (buf.contents() as *mut u8).add(off) }
    }

    fn key(&self, layer: usize, expert: usize) -> Option<(u16, u16)> {
        (layer < self.num_layers && expert < self.num_experts).then_some((layer as u16, expert as u16))
    }

    /// Start a new token: slots bound before this call are no longer
    /// read by in-flight GPU work.
    pub fn begin_epoch(&self) {
        self.inner.lock().unwrap().epoch += 1;
    }

    /// Record that `(layer, expert)` was routed (popularity counter).
    pub fn record_pick(&self, layer: usize, expert: usize) {
        self.record_picks(layer, expert, 1);
    }

    /// Record `n` routed picks of `(layer, expert)` at once (prefill
    /// buckets). Decay fires each time the running total crosses a
    /// multiple of the decay period.
    pub fn record_picks(&self, layer: usize, expert: usize, n: u32) {
        if layer >= self.num_layers || expert >= self.num_experts || n == 0 {
            return;
        }
        let mut g = self.inner.lock().unwrap();
        let i = layer * self.num_experts + expert;
        g.counts[i] = g.counts[i].saturating_add(n);
        let before = g.picks;
        g.picks += n as u64;
        for _ in before / DECAY_EVERY..g.picks / DECAY_EVERY {
            for c in g.counts.iter_mut() {
                *c >>= 1;
            }
        }
    }

    /// `true` if `(layer, expert)` is resident. No counters touched.
    pub fn contains(&self, layer: usize, expert: usize) -> bool {
        let Some(k) = self.key(layer, expert) else {
            return false;
        };
        self.inner.lock().unwrap().map.contains_key(&k)
    }

    /// Zero-copy GPU binding for `(layer, expert)`: the chunk buffer
    /// and byte offset of its slot. Pins the slot for this epoch.
    pub fn lookup(&self, layer: usize, expert: usize) -> Option<(&Buffer, u64)> {
        let k = self.key(layer, expert)?;
        let mut g = self.inner.lock().unwrap();
        match g.map.get(&k).copied() {
            Some(slot) => {
                let epoch = g.epoch;
                g.last_bound[slot as usize] = epoch;
                g.stats.hits += 1;
                drop(g);
                let (buf, off) = self.slot_location(slot);
                Some((buf, off as u64))
            }
            None => {
                g.stats.misses += 1;
                None
            }
        }
    }

    /// Copy `(layer, expert)` into `dst` if resident (CPU path, for
    /// staging buffers). Returns `true` on a hit.
    pub fn copy_out(&self, layer: usize, expert: usize, dst: &mut [u8]) -> bool {
        let Some(k) = self.key(layer, expert) else {
            return false;
        };
        let mut g = self.inner.lock().unwrap();
        match g.map.get(&k).copied() {
            Some(slot) if dst.len() == self.expert_size => {
                g.stats.hits += 1;
                // SAFETY: slot holds `expert_size` bytes; the lock
                // excludes concurrent admission into this slot.
                unsafe {
                    std::ptr::copy_nonoverlapping(self.slot_ptr(slot), dst.as_mut_ptr(), dst.len())
                };
                true
            }
            _ => {
                g.stats.misses += 1;
                false
            }
        }
    }

    /// Offer freshly read bytes of `(layer, expert)` for admission.
    /// Fills a free slot if any; otherwise replaces the coldest of a
    /// few sampled slots not bound this epoch, and only if the
    /// candidate is hotter. Returns `true` if admitted.
    pub fn offer(&self, layer: usize, expert: usize, src: &[u8]) -> bool {
        let Some(k) = self.key(layer, expert) else {
            return false;
        };
        if src.len() != self.expert_size || self.capacity == 0 {
            return false;
        }
        let mut g = self.inner.lock().unwrap();
        if g.map.contains_key(&k) {
            return false;
        }
        let cand = g.counts[layer * self.num_experts + expert];
        let slot = if g.map.len() < self.capacity {
            // First free slot. Linear scan only while filling.
            g.owner.iter().position(Option::is_none).map(|s| s as u32)
        } else {
            let epoch = g.epoch;
            let mut best: Option<(u32, u32)> = None; // (slot, count)
            for _ in 0..VICTIM_SAMPLES {
                g.rng ^= g.rng << 13;
                g.rng ^= g.rng >> 7;
                g.rng ^= g.rng << 17;
                let s = (g.rng % self.capacity as u64) as u32;
                if g.last_bound[s as usize] == epoch {
                    continue;
                }
                let (l, e) = g.owner[s as usize].expect("full tier");
                let c = g.counts[l as usize * self.num_experts + e as usize];
                if best.is_none_or(|(_, bc)| c < bc) {
                    best = Some((s, c));
                }
            }
            match best {
                Some((s, c)) if cand > c => Some(s),
                _ => None,
            }
        };
        let Some(slot) = slot else {
            return false;
        };
        if let Some(old) = g.owner[slot as usize].take() {
            g.map.remove(&old);
            g.stats.evictions += 1;
        }
        // SAFETY: slot not bound this epoch (or free), so no GPU work
        // reads it; the lock excludes CPU readers.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.slot_ptr(slot), src.len()) };
        g.owner[slot as usize] = Some(k);
        g.map.insert(k, slot);
        g.stats.admissions += 1;
        g.stats.resident = g.map.len();
        true
    }

    /// Current counters.
    pub fn stats(&self) -> HotTierStats {
        let g = self.inner.lock().unwrap();
        HotTierStats {
            resident: g.map.len(),
            ..g.stats
        }
    }

    /// Zero the hit/miss/admission/eviction counters (capacity and
    /// residency are state, not counters, and stay).
    pub fn reset_stats(&self) {
        let mut g = self.inner.lock().unwrap();
        g.stats = HotTierStats {
            capacity: self.capacity,
            ..HotTierStats::default()
        };
    }

    /// Popularity counters, `layer * num_experts + expert`.
    pub fn counts(&self) -> Vec<u32> {
        self.inner.lock().unwrap().counts.clone()
    }

    /// The `capacity` most popular `(layer, expert)` pairs in `counts`
    /// (non-zero only), most popular first. Also installs `counts` as
    /// the tier's starting popularity. Caller reads each pair and
    /// passes the bytes to [`Self::offer`].
    pub fn seed_from_counts(&self, counts: &[u32]) -> Vec<(usize, usize)> {
        let n = self.num_layers * self.num_experts;
        if counts.len() != n {
            return Vec::new();
        }
        self.inner.lock().unwrap().counts.copy_from_slice(counts);
        let mut order: Vec<usize> = (0..n).filter(|&i| counts[i] > 0).collect();
        order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
        order.truncate(self.capacity);
        order
            .into_iter()
            .map(|i| (i / self.num_experts, i % self.num_experts))
            .collect()
    }
}

/// Write a routing profile: magic, `num_layers`, `num_experts` (u32
/// LE), then the counts (u32 LE).
pub fn save_profile(
    path: &std::path::Path,
    num_layers: usize,
    num_experts: usize,
    counts: &[u32],
) -> Result<(), HotTierError> {
    let mut out = Vec::with_capacity(12 + counts.len() * 4);
    out.extend_from_slice(PROFILE_MAGIC);
    out.extend_from_slice(&(num_layers as u32).to_le_bytes());
    out.extend_from_slice(&(num_experts as u32).to_le_bytes());
    for c in counts {
        out.extend_from_slice(&c.to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Read a routing profile written by [`save_profile`].
pub fn load_profile(
    path: &std::path::Path,
    num_layers: usize,
    num_experts: usize,
) -> Result<Vec<u32>, HotTierError> {
    let b = std::fs::read(path)?;
    if b.len() < 12 || &b[0..4] != PROFILE_MAGIC {
        return Err(HotTierError::ProfileFormat);
    }
    let layers = u32::from_le_bytes(b[4..8].try_into().unwrap());
    let experts = u32::from_le_bytes(b[8..12].try_into().unwrap());
    if layers as usize != num_layers || experts as usize != num_experts {
        return Err(HotTierError::ProfileShape {
            layers,
            experts,
            want_layers: num_layers,
            want_experts: num_experts,
        });
    }
    let n = num_layers * num_experts;
    if b.len() != 12 + n * 4 {
        return Err(HotTierError::ProfileFormat);
    }
    Ok(b[12..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ES: usize = 16384;

    fn tier(cap: usize) -> HotTier {
        let dev = Device::system_default().expect("Metal device");
        HotTier::new(&dev, cap, ES, 4, 8).unwrap()
    }

    fn blob(tag: u8) -> Vec<u8> {
        vec![tag; ES]
    }

    #[test]
    fn fill_then_hit_returns_same_bytes() {
        let t = tier(2);
        assert!(t.offer(0, 1, &blob(7)));
        assert!(t.contains(0, 1));
        let (buf, off) = t.lookup(0, 1).expect("hit");
        let got = unsafe {
            std::slice::from_raw_parts((buf.contents() as *const u8).add(off as usize), ES)
        };
        assert!(got.iter().all(|&b| b == 7));
        let mut dst = vec![0u8; ES];
        assert!(t.copy_out(0, 1, &mut dst));
        assert_eq!(dst, blob(7));
        assert!(t.lookup(3, 3).is_none());
        let s = t.stats();
        assert_eq!((s.hits, s.misses, s.resident), (2, 1, 1));
    }

    #[test]
    fn size_never_changes_and_colder_candidates_rejected() {
        let t = tier(2);
        for _ in 0..5 {
            t.record_pick(0, 0);
            t.record_pick(0, 1);
        }
        assert!(t.offer(0, 0, &blob(1)));
        assert!(t.offer(0, 1, &blob(2)));
        t.begin_epoch();
        // (1, 1) has count 0 < 5: never displaces a hotter expert.
        assert!(!t.offer(1, 1, &blob(3)));
        assert_eq!(t.stats().resident, 2);
        // Make it hotter than everything resident; it must get in.
        for _ in 0..20 {
            t.record_pick(1, 1);
        }
        let mut admitted = false;
        for _ in 0..50 {
            if t.offer(1, 1, &blob(3)) {
                admitted = true;
                break;
            }
        }
        assert!(admitted);
        assert_eq!(t.stats().resident, 2);
        assert_eq!(t.stats().evictions, 1);
    }

    #[test]
    fn slots_bound_this_epoch_are_not_evicted() {
        let t = tier(1);
        t.offer(0, 0, &blob(1));
        let _ = t.lookup(0, 0); // bound this epoch
        for _ in 0..100 {
            t.record_pick(2, 2);
        }
        for _ in 0..50 {
            assert!(!t.offer(2, 2, &blob(9)));
        }
        t.begin_epoch();
        assert!(t.offer(2, 2, &blob(9)));
        assert!(!t.contains(0, 0));
    }

    #[test]
    fn seed_orders_by_count_and_profile_roundtrips() {
        let t = tier(3);
        let mut counts = vec![0u32; 32];
        counts[5] = 10;
        counts[9] = 30;
        counts[31] = 20;
        counts[2] = 1;
        let order = t.seed_from_counts(&counts);
        assert_eq!(order, vec![(1, 1), (3, 7), (0, 5)]);
        let p = std::env::temp_dir().join(format!("streammoe-prof-{}.bin", std::process::id()));
        save_profile(&p, 4, 8, &counts).unwrap();
        assert_eq!(load_profile(&p, 4, 8).unwrap(), counts);
        assert!(matches!(load_profile(&p, 4, 9), Err(HotTierError::ProfileShape { .. })));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn spans_multiple_chunks() {
        assert_eq!(HotTier::slots_for_bytes(10 * ES as u64 + 5, ES), 10);
        let dev = Device::system_default().unwrap();
        // 2 slots per chunk → 5 slots need 3 chunks; slot 4 lives in
        // the last one and must round-trip.
        let t = HotTier::with_chunk_bytes(&dev, 5, ES, 1, 8, 2 * ES).unwrap();
        assert_eq!(t.chunks.len(), 3);
        for e in 0..5 {
            assert!(t.offer(0, e, &blob(e as u8 + 1)));
        }
        let mut dst = vec![0u8; ES];
        assert!(t.copy_out(0, 4, &mut dst));
        assert_eq!(dst, blob(5));
        let (buf, off) = t.lookup(0, 4).unwrap();
        assert_eq!(buf.contents(), t.chunks[2].contents());
        assert_eq!(off, 0);
    }
}
