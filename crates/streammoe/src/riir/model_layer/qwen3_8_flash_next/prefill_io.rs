//! Prefill expert reads of the family engine ([`super::engine`]): which
//! preads a chunk's MoE layer issues, how its experts are packed into
//! the prefill staging, and the counters.
//!
//! A chunk routes its tokens to a set of experts per layer (the buckets
//! of `build_expert_buckets`, ascending expert ids). The ones in the
//! hot tier are copied from RAM; the rest are read from the layer file
//! into the prefill staging, whose slots are `stride` bytes apart (the
//! file's padded expert stride), so a run of adjacent experts is one
//! contiguous pread straight into consecutive slots.
//!
//! **Whole-layer read.** With `N` of a layer's `E` experts to read,
//! `N` separate reads cost `N · (size / BW + c)` and one sequential
//! read of the whole layer `E · stride / BW` (`c`: per-request
//! overhead, [`READ_OVERHEAD_BYTES`] = `c · BW`). The layer read wins
//! once `N / E ≥ stride / (size + c · BW)` ([`default_layer_read_frac`]:
//! 0.86 on the real model, `STREAMMOE_Q38_LAYER_READ_FRAC` overrides).
//! Then every gap between the experts to read is read through, so the
//! layer streams in a few large preads of up to `max_read` experts;
//! otherwise only adjacent experts share a pread. Hot-tier experts
//! inside a read range come with it; outside one they are copied.
//!
//! **Batches.** The staging holds `cap` experts at a time (half of it
//! when reads overlap the GPU). The buckets are cut into batches whose
//! reads fit: one `MoeBatchedPermuteFuse` per batch, all its reads in
//! flight at once on the IO pool (up to `cap` experts).

use crate::riir::io::expert_io::declared_stride;

/// Default prefill staging slots: two halves of 32 experts (~94 MiB
/// each on the real model), so 32 reads are in flight while the GPU
/// runs the other half.
pub const DEFAULT_PREFILL_SLOTS: usize = 64;

/// Default most experts one pread covers (~47 MiB on the real model).
pub const DEFAULT_MAX_READ: usize = 16;

/// Per-request read overhead as bytes of transfer (`c · BW`): ~0.3 ms
/// at the external SSD's ~1.7 GB/s. Sets [`default_layer_read_frac`].
pub const READ_OVERHEAD_BYTES: usize = 512 << 10;

/// The fraction of a layer's experts to read above which one sequential
/// read of the layer beats separate reads (module docs):
/// `stride / (expert_size + READ_OVERHEAD_BYTES)`.
pub fn default_layer_read_frac(expert_size: usize, stride: usize) -> f64 {
    stride as f64 / (expert_size + READ_OVERHEAD_BYTES) as f64
}

/// The staging slot stride for `expert_size`-byte experts under `root`:
/// the padded stride `packed_experts/layout.json` declares for that
/// size, else the size.
pub fn staging_stride(root: &std::path::Path, expert_size: usize) -> usize {
    match declared_stride(&root.join("packed_experts")) {
        Some((size, stride)) if size as usize == expert_size => stride as usize,
        _ => expert_size,
    }
}

pub(crate) fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

pub(crate) fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// One pread: experts `first..first + n` of the layer file into staging
/// slots `slot..slot + n` of the batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanRead {
    pub first: u32,
    pub n: u32,
    pub slot: u32,
}

/// Buckets `b0..b1` of a chunk's layer, whose experts fit the staging
/// together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillBatch {
    pub b0: usize,
    pub b1: usize,
    /// Staging slot (within the batch) of each bucket's expert,
    /// `b1 - b0` entries.
    pub slots: Vec<u32>,
    /// The preads, ascending.
    pub reads: Vec<SpanRead>,
    /// `(expert, slot)` copied from the hot tier.
    pub hot: Vec<(u32, u32)>,
    /// Slots used (`<= cap`).
    pub used: u32,
}

/// Plan a layer's reads for the chunk's bucket experts `ids` (strictly
/// ascending): spans over the experts not in the hot tier — adjacent
/// ones only, or with `merge_gaps` every gap read through — at most
/// `min(max_read, cap)` experts each; then batches of at most `cap`
/// slots, each a contiguous range of buckets. Pure (unit-tested).
pub fn plan_prefill_batches(
    ids: &[i32],
    is_hot: impl Fn(usize) -> bool,
    merge_gaps: bool,
    cap: usize,
    max_read: usize,
) -> Vec<PrefillBatch> {
    plan_prefill_batches_rows(ids, is_hot, merge_gaps, cap, max_read, &[], usize::MAX)
}

/// [`plan_prefill_batches`] with at most `max_rows` assignment rows per
/// batch (the bucket scratch's rows): bucket `i` has `rows[i]` rows
/// (`rows` empty: no row limit). A bucket alone over `max_rows` still
/// gets a batch of its own. A batch cut by rows inside a span reads the
/// span again in the next batch (rare: a layer-major window's batch of
/// 32 experts holds ~1,500 of the 5,120 rows).
pub fn plan_prefill_batches_rows(
    ids: &[i32],
    is_hot: impl Fn(usize) -> bool,
    merge_gaps: bool,
    cap: usize,
    max_read: usize,
    rows: &[u32],
    max_rows: usize,
) -> Vec<PrefillBatch> {
    debug_assert!(rows.is_empty() || rows.len() == ids.len());
    debug_assert!(ids.windows(2).all(|w| w[0] < w[1]), "bucket experts ascend");
    let cap = cap.max(1) as u32;
    let lim = (max_read.max(1) as u32).min(cap);
    let hot: Vec<bool> = ids.iter().map(|&e| is_hot(e as usize)).collect();
    // Pass 1: spans over the experts to read, as (first, n).
    let mut spans: Vec<(u32, u32)> = Vec::new();
    for (&e, &h) in ids.iter().zip(&hot) {
        if h {
            continue;
        }
        let e = e as u32;
        match spans.last_mut() {
            Some((f, n)) if (merge_gaps || *f + *n == e) && e - *f < lim => *n = e - *f + 1,
            _ => spans.push((e, 1)),
        }
    }
    // Pass 2: walk the buckets, placing each span when its first expert
    // comes up, hot experts outside every span on their own slot.
    let mut out = Vec::new();
    let mut cur = PrefillBatch { b0: 0, b1: 0, slots: Vec::new(), reads: Vec::new(), hot: Vec::new(), used: 0 };
    let mut si = 0usize;
    // The span `si` and its slot in `cur`, once placed.
    let mut placed: Option<u32> = None;
    let mut cur_rows = 0usize;
    for (bi, &e) in ids.iter().enumerate() {
        let e = e as u32;
        while si < spans.len() && spans[si].0 + spans[si].1 <= e {
            si += 1;
            placed = None;
        }
        let covered = si < spans.len() && spans[si].0 <= e;
        let r = rows.get(bi).map_or(0, |&r| r as usize);
        let over_rows = bi > cur.b0 && cur_rows + r > max_rows;
        if over_rows {
            // The next batch places the span again.
            placed = None;
        }
        let need = if covered { if placed.is_some() { 0 } else { spans[si].1 } } else { 1 };
        if over_rows || (need > 0 && cur.used + need > cap) {
            cur.b1 = bi;
            let next = PrefillBatch { b0: bi, b1: bi, slots: Vec::new(), reads: Vec::new(), hot: Vec::new(), used: 0 };
            out.push(std::mem::replace(&mut cur, next));
            cur_rows = 0;
        }
        cur_rows += r;
        if covered {
            let base = match placed {
                Some(s) => s,
                None => {
                    let s = cur.used;
                    cur.reads.push(SpanRead { first: spans[si].0, n: spans[si].1, slot: s });
                    cur.used += spans[si].1;
                    placed = Some(s);
                    s
                }
            };
            cur.slots.push(base + (e - spans[si].0));
        } else {
            debug_assert!(hot[bi]);
            cur.hot.push((e, cur.used));
            cur.slots.push(cur.used);
            cur.used += 1;
        }
    }
    cur.b1 = ids.len();
    if cur.b1 > cur.b0 {
        out.push(cur);
    }
    out
}

/// Prefill expert-read counters ([`super::engine::Q38Engine::prefill_io_stats`]),
/// summed over chunks (or one chunk).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefillIoStats {
    pub chunks: u64,
    pub tokens: u64,
    /// MoE layers run.
    pub layers: u64,
    /// Of them, read as a whole layer (gaps read through).
    pub layer_reads: u64,
    /// Routed (layer, expert) pairs: the buckets.
    pub experts: u64,
    /// Of them, copied from the hot tier.
    pub hot: u64,
    /// Experts that came from disk (routed or read through).
    pub experts_read: u64,
    /// Of them, not routed (gaps of a whole-layer read).
    pub read_through: u64,
    /// Preads and their bytes.
    pub preads: u64,
    pub bytes: u64,
    /// Expert batches (one permute-fuse each).
    pub batches: u64,
    /// Experts read early (before the router readback; counted in
    /// `experts_read` too).
    pub early: u64,
    /// Time the host waited on reads / on the GPU, and the chunks' wall
    /// time.
    pub io_wait_ns: u64,
    pub gpu_wait_ns: u64,
    pub wall_ns: u64,
}

impl PrefillIoStats {
    pub fn add(&mut self, o: &Self) {
        self.chunks += o.chunks;
        self.tokens += o.tokens;
        self.layers += o.layers;
        self.layer_reads += o.layer_reads;
        self.experts += o.experts;
        self.hot += o.hot;
        self.experts_read += o.experts_read;
        self.read_through += o.read_through;
        self.preads += o.preads;
        self.bytes += o.bytes;
        self.batches += o.batches;
        self.early += o.early;
        self.io_wait_ns += o.io_wait_ns;
        self.gpu_wait_ns += o.gpu_wait_ns;
        self.wall_ns += o.wall_ns;
    }

    /// Mean bytes per pread.
    pub fn avg_read(&self) -> u64 {
        self.bytes / self.preads.max(1)
    }

    /// One line for the trace.
    pub fn line(&self) -> String {
        let ms = |ns: u64| ns as f64 / 1e6;
        format!(
            "{} tok, {} layers ({} whole-layer), experts {} (hot {}), read {} (+{} through) in {} preads, \
             {:.1} MiB, avg {:.0} KiB, {} batches, {} early; io wait {:.1} ms, gpu wait {:.1} ms, wall {:.1} ms",
            self.tokens,
            self.layers,
            self.layer_reads,
            self.experts,
            self.hot,
            self.experts_read - self.read_through,
            self.read_through,
            self.preads,
            self.bytes as f64 / (1u64 << 20) as f64,
            self.avg_read() as f64 / 1024.0,
            self.batches,
            self.early,
            ms(self.io_wait_ns),
            ms(self.gpu_wait_ns),
            ms(self.wall_ns),
        )
    }
}

/// `STREAMMOE_Q38_IO_TRACE=1`: one line per prefill chunk on stderr.
pub fn trace_from_env() -> bool {
    std::env::var("STREAMMOE_Q38_IO_TRACE").is_ok_and(|v| v != "0" && !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn covered(b: &[PrefillBatch], ids: &[i32]) {
        // Every bucket once, in order; its slot holds its expert.
        let mut next = 0;
        for x in b {
            assert_eq!(x.b0, next);
            assert_eq!(x.slots.len(), x.b1 - x.b0);
            for (i, &s) in x.slots.iter().enumerate() {
                let e = ids[x.b0 + i] as u32;
                let by_read = x.reads.iter().any(|r| r.slot <= s && s < r.slot + r.n && r.first + (s - r.slot) == e);
                let by_hot = x.hot.contains(&(e, s));
                assert!(by_read ^ by_hot, "bucket {} expert {e} slot {s}", x.b0 + i);
            }
            let read_slots: u32 = x.reads.iter().map(|r| r.n).sum();
            assert_eq!(read_slots + x.hot.len() as u32, x.used);
            next = x.b1;
        }
        assert_eq!(next, ids.len());
    }

    #[test]
    fn adjacent_experts_share_a_read() {
        let ids = [1, 2, 3, 7, 9, 10];
        let b = plan_prefill_batches(&ids, |_| false, false, 64, 16);
        assert_eq!(b.len(), 1);
        let r = |first, n, slot| SpanRead { first, n, slot };
        assert_eq!(b[0].reads, vec![r(1, 3, 0), r(7, 1, 3), r(9, 2, 4)]);
        assert_eq!(b[0].slots, vec![0, 1, 2, 3, 4, 5]);
        covered(&b, &ids);
    }

    #[test]
    fn whole_layer_reads_through_gaps_and_hot_experts() {
        let ids = [1, 2, 5, 6, 12];
        // 5 is hot: inside the read range, read through.
        let b = plan_prefill_batches(&ids, |e| e == 5 || e == 0, true, 64, 16);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].reads, vec![SpanRead { first: 1, n: 12, slot: 0 }]);
        assert_eq!(b[0].slots, vec![0, 1, 4, 5, 11]);
        assert!(b[0].hot.is_empty());
        covered(&b, &ids);
        // A hot expert at the edge is copied.
        let ids = [0, 3, 4];
        let b = plan_prefill_batches(&ids, |e| e == 0, true, 64, 16);
        assert_eq!(b[0].hot, vec![(0, 0)]);
        assert_eq!(b[0].reads, vec![SpanRead { first: 3, n: 2, slot: 1 }]);
        covered(&b, &ids);
    }

    #[test]
    fn batches_respect_cap_and_max_read() {
        let ids: Vec<i32> = (0..40).filter(|e| e % 3 != 1).collect();
        for (merge, cap, max_read) in [(false, 1, 16), (false, 5, 2), (true, 8, 16), (true, 7, 3), (true, 64, 64)] {
            let b = plan_prefill_batches(&ids, |e| e % 9 == 0, merge, cap, max_read);
            covered(&b, &ids);
            for x in &b {
                assert!(x.used as usize <= cap, "{merge} {cap} {max_read}");
                assert!(x.reads.iter().all(|r| r.n as usize <= max_read.min(cap)));
            }
        }
        assert!(plan_prefill_batches(&[], |_| false, true, 4, 4).is_empty());
    }

    #[test]
    fn batches_respect_row_cap() {
        let ids: Vec<i32> = (0..40).filter(|e| e % 5 != 2).collect();
        let rows: Vec<u32> = ids.iter().map(|&e| 1 + (e as u32 * 7) % 13).collect();
        for (merge, cap, max_rows) in [(false, 64, 20), (true, 64, 20), (true, 8, 30), (true, 64, 1), (false, 3, 1000)] {
            let b = plan_prefill_batches_rows(&ids, |e| e % 11 == 0, merge, cap, 16, &rows, max_rows);
            covered(&b, &ids);
            for x in &b {
                let r: u32 = rows[x.b0..x.b1].iter().sum();
                assert!(r as usize <= max_rows || x.b1 - x.b0 == 1, "{merge} {cap} {max_rows}: {r} rows");
                assert!(x.used as usize <= cap);
            }
        }
    }
}
