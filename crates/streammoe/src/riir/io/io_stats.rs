//! Streaming telemetry (plan phases 0 and 6): what the expert I/O
//! path did, per token, per layer and per source.
//!
//! [`IoCounters`] is a fixed set of relaxed atomics allocated once at
//! open (one row per layer), so recording from the decode hot path,
//! the prefetch workers and the prefill reader is lock- and
//! allocation-free. [`IoCounters::snapshot`] turns it into an
//! [`ExpertIoStats`] value (the only place that allocates), which
//! prints as CSV for `scripts/profile_smoke.sh`-style collection.

use std::sync::atomic::{AtomicU64, Ordering};

/// What a disk read was for. Attributes bytes to a phase.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReadKind {
    /// Decode miss: read on the critical path.
    Demand,
    /// Speculative decode prefetch.
    Prefetch,
    /// Prefill staging.
    Prefill,
    /// Hot-tier seeding at open.
    Seed,
}

impl ReadKind {
    const COUNT: usize = 4;

    fn idx(self) -> usize {
        match self {
            ReadKind::Demand => 0,
            ReadKind::Prefetch => 1,
            ReadKind::Prefill => 2,
            ReadKind::Seed => 3,
        }
    }
}

/// Which prefetch guesser a [`GuessStats`] row scores.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Guesser {
    /// The previous token's picks for the same layer.
    PrevToken,
    /// The previous layer's residual run through this layer's router.
    CrossLayer,
}

/// Reads and bytes for one [`ReadKind`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReadStats {
    /// Read calls (a coalesced prefill read of n experts counts n).
    pub reads: u64,
    /// Bytes read.
    pub bytes: u64,
}

/// Accuracy of one guesser: of `guessed` predicted experts,
/// `correct` were among the layer's actual picks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GuessStats {
    /// Layers scored.
    pub layers: u64,
    /// Experts guessed (k per scored layer).
    pub guessed: u64,
    /// Guessed experts the router actually picked.
    pub correct: u64,
}

impl GuessStats {
    /// `correct / guessed`, or `None` before anything was scored.
    pub fn accuracy(&self) -> Option<f64> {
        (self.guessed > 0).then(|| self.correct as f64 / self.guessed as f64)
    }
}

/// Per-layer counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerIoStats {
    /// Decode picks served zero-copy from the hot tier.
    pub hot_hits: u64,
    /// Decode picks served from a prefetch slot.
    pub prefetch_hits: u64,
    /// Decode picks read on the critical path.
    pub demand_reads: u64,
    /// Speculative reads issued for this layer.
    pub prefetch_reads: u64,
    /// Prefill experts copied from the hot tier.
    pub prefill_hot: u64,
    /// Prefill experts read from disk.
    pub prefill_reads: u64,
    /// Bytes read from disk for this layer (every kind).
    pub disk_bytes: u64,
}

/// Expert-IO counters since open (or the last reset).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpertIoStats {
    /// Expert blobs read from disk (every kind).
    pub disk_reads: u64,
    /// Bytes read from disk (every kind).
    pub disk_bytes: u64,
    /// Decode misses read on the critical path.
    pub demand: ReadStats,
    /// Speculative prefetch reads.
    pub prefetch: ReadStats,
    /// Prefill staging reads.
    pub prefill: ReadStats,
    /// Hot-tier seed reads at open.
    pub seed: ReadStats,
    /// Of the bytes above, those read from layer files opened from the
    /// fast experts directory
    /// ([`crate::riir::OpenConfig::fast_experts_dir`]).
    pub fast_dir_bytes: u64,
    /// Decode picks served from a prefetch slot. `prefetch.reads -
    /// prefetch_hits` is speculative I/O that was not used.
    pub prefetch_hits: u64,
    /// Decode tokens (per-token forward passes).
    pub decode_tokens: u64,
    /// Prefill tokens (batched forward rows).
    pub prefill_tokens: u64,
    /// Wall time in per-token forward passes.
    pub decode_ns: u64,
    /// Of `decode_ns`, time the forward thread waited on expert I/O
    /// (draining prefetches and reading misses). The rest is GPU
    /// waits and CPU work; per-label GPU waits are in
    /// [`crate::riir::RsCtx::cmdbuf_stats`].
    pub io_wait_ns: u64,
    /// Wall time in batched (prefill) forward passes.
    pub prefill_ns: u64,
    /// Of `prefill_ns`, time spent staging experts.
    pub prefill_io_ns: u64,
    /// Accuracy of the previous-token guess (scored on every pread
    /// decode layer that had a prediction).
    pub guess_prev: GuessStats,
    /// Accuracy of the cross-layer guess (scored when computed).
    pub guess_cross: GuessStats,
    /// Hot-tier counters, when a hot tier is attached.
    pub hot: Option<super::hot_tier::HotTierStats>,
    /// One row per model layer (zeros outside the layer range).
    pub per_layer: Vec<LayerIoStats>,
}

impl ExpertIoStats {
    /// CSV header matching [`Self::csv_row`].
    pub const CSV_HEADER: &'static str = "decode_tokens,prefill_tokens,disk_reads,disk_bytes,\
demand_reads,demand_bytes,prefetch_reads,prefetch_bytes,prefetch_hits,prefill_reads,\
prefill_bytes,seed_reads,seed_bytes,fast_dir_bytes,decode_bytes_per_token,decode_ns,\
io_wait_ns,prefill_ns,prefill_io_ns,guess_prev_acc,guess_cross_acc,hot_capacity,\
hot_resident,hot_hits,hot_misses,hot_admissions,hot_evictions";

    /// Per-layer CSV header matching [`Self::layer_csv`] rows.
    pub const LAYER_CSV_HEADER: &'static str = "layer,hot_hits,prefetch_hits,demand_reads,\
prefetch_reads,prefill_hot,prefill_reads,disk_bytes";

    /// Decode disk bytes (demand + prefetch) per decode token.
    pub fn decode_bytes_per_token(&self) -> Option<f64> {
        (self.decode_tokens > 0).then(|| {
            (self.demand.bytes + self.prefetch.bytes) as f64 / self.decode_tokens as f64
        })
    }

    /// One CSV row (no trailing newline). Empty cells for values that
    /// are undefined (no tokens, no hot tier, nothing scored).
    pub fn csv_row(&self) -> String {
        let opt = |x: Option<f64>| x.map_or(String::new(), |v| format!("{v:.4}"));
        let h = self.hot.unwrap_or_default();
        let hot = |x: u64| if self.hot.is_some() { x.to_string() } else { String::new() };
        [
            self.decode_tokens.to_string(),
            self.prefill_tokens.to_string(),
            self.disk_reads.to_string(),
            self.disk_bytes.to_string(),
            self.demand.reads.to_string(),
            self.demand.bytes.to_string(),
            self.prefetch.reads.to_string(),
            self.prefetch.bytes.to_string(),
            self.prefetch_hits.to_string(),
            self.prefill.reads.to_string(),
            self.prefill.bytes.to_string(),
            self.seed.reads.to_string(),
            self.seed.bytes.to_string(),
            self.fast_dir_bytes.to_string(),
            opt(self.decode_bytes_per_token()),
            self.decode_ns.to_string(),
            self.io_wait_ns.to_string(),
            self.prefill_ns.to_string(),
            self.prefill_io_ns.to_string(),
            opt(self.guess_prev.accuracy()),
            opt(self.guess_cross.accuracy()),
            hot(h.capacity as u64),
            hot(h.resident as u64),
            hot(h.hits),
            hot(h.misses),
            hot(h.admissions),
            hot(h.evictions),
        ]
        .join(",")
    }

    /// Per-layer CSV rows (one per layer, newline-separated, no
    /// header).
    pub fn layer_csv(&self) -> String {
        let mut out = String::new();
        for (i, l) in self.per_layer.iter().enumerate() {
            out.push_str(&format!(
                "{i},{},{},{},{},{},{},{}\n",
                l.hot_hits,
                l.prefetch_hits,
                l.demand_reads,
                l.prefetch_reads,
                l.prefill_hot,
                l.prefill_reads,
                l.disk_bytes
            ));
        }
        out
    }
}

#[derive(Debug, Default)]
struct LayerCounters {
    hot_hits: AtomicU64,
    prefetch_hits: AtomicU64,
    demand_reads: AtomicU64,
    prefetch_reads: AtomicU64,
    prefill_hot: AtomicU64,
    prefill_reads: AtomicU64,
    disk_bytes: AtomicU64,
}

#[derive(Debug, Default)]
struct GuessCounters {
    layers: AtomicU64,
    guessed: AtomicU64,
    correct: AtomicU64,
}

/// Fixed-size telemetry store; see the module docs.
#[derive(Debug)]
pub struct IoCounters {
    reads: [AtomicU64; ReadKind::COUNT],
    bytes: [AtomicU64; ReadKind::COUNT],
    fast_dir_bytes: AtomicU64,
    decode_tokens: AtomicU64,
    prefill_tokens: AtomicU64,
    decode_ns: AtomicU64,
    io_wait_ns: AtomicU64,
    prefill_ns: AtomicU64,
    prefill_io_ns: AtomicU64,
    guess: [GuessCounters; 2],
    layers: Vec<LayerCounters>,
}

fn add(c: &AtomicU64, n: u64) {
    if n > 0 {
        c.fetch_add(n, Ordering::Relaxed);
    }
}

fn get(c: &AtomicU64) -> u64 {
    c.load(Ordering::Relaxed)
}

impl IoCounters {
    /// Counters for a model of `num_layers` layers, all zero.
    pub fn new(num_layers: usize) -> Self {
        Self {
            reads: Default::default(),
            bytes: Default::default(),
            fast_dir_bytes: AtomicU64::new(0),
            decode_tokens: AtomicU64::new(0),
            prefill_tokens: AtomicU64::new(0),
            decode_ns: AtomicU64::new(0),
            io_wait_ns: AtomicU64::new(0),
            prefill_ns: AtomicU64::new(0),
            prefill_io_ns: AtomicU64::new(0),
            guess: Default::default(),
            layers: (0..num_layers).map(|_| LayerCounters::default()).collect(),
        }
    }

    /// Record `n` experts (`bytes` total) read from disk for `layer`.
    pub fn record_read(&self, kind: ReadKind, layer: usize, n: u64, bytes: u64, fast_dir: bool) {
        add(&self.reads[kind.idx()], n);
        add(&self.bytes[kind.idx()], bytes);
        if fast_dir {
            add(&self.fast_dir_bytes, bytes);
        }
        if let Some(l) = self.layers.get(layer) {
            add(&l.disk_bytes, bytes);
            match kind {
                ReadKind::Demand => add(&l.demand_reads, n),
                ReadKind::Prefetch => add(&l.prefetch_reads, n),
                ReadKind::Prefill => add(&l.prefill_reads, n),
                ReadKind::Seed => {}
            }
        }
    }

    /// Record one decode layer's slot sources.
    pub fn record_decode_layer(&self, layer: usize, hot: u64, prefetched: u64) {
        if let Some(l) = self.layers.get(layer) {
            add(&l.hot_hits, hot);
            add(&l.prefetch_hits, prefetched);
        }
    }

    /// Record prefill experts copied from the hot tier.
    pub fn record_prefill_hot(&self, layer: usize, n: u64) {
        if let Some(l) = self.layers.get(layer) {
            add(&l.prefill_hot, n);
        }
    }

    /// Score one layer's guess against its actual picks.
    pub fn record_guess(&self, who: Guesser, guessed: u64, correct: u64) {
        let g = &self.guess[who as usize];
        add(&g.layers, 1);
        add(&g.guessed, guessed);
        add(&g.correct, correct);
    }

    /// One decode token (per-token forward pass) took `ns`.
    pub fn record_decode(&self, ns: u64) {
        add(&self.decode_tokens, 1);
        add(&self.decode_ns, ns);
    }

    /// Time the decode thread waited on expert I/O.
    pub fn record_io_wait(&self, ns: u64) {
        add(&self.io_wait_ns, ns);
    }

    /// One batched forward of `n` rows took `ns`.
    pub fn record_prefill(&self, n: u64, ns: u64) {
        add(&self.prefill_tokens, n);
        add(&self.prefill_ns, ns);
    }

    /// Time a prefill layer spent staging experts.
    pub fn record_prefill_io(&self, ns: u64) {
        add(&self.prefill_io_ns, ns);
    }

    /// Zero every counter.
    pub fn reset(&self) {
        let all = self
            .reads
            .iter()
            .chain(&self.bytes)
            .chain([
                &self.fast_dir_bytes,
                &self.decode_tokens,
                &self.prefill_tokens,
                &self.decode_ns,
                &self.io_wait_ns,
                &self.prefill_ns,
                &self.prefill_io_ns,
            ])
            .chain(self.guess.iter().flat_map(|g| [&g.layers, &g.guessed, &g.correct]))
            .chain(self.layers.iter().flat_map(|l| {
                [
                    &l.hot_hits,
                    &l.prefetch_hits,
                    &l.demand_reads,
                    &l.prefetch_reads,
                    &l.prefill_hot,
                    &l.prefill_reads,
                    &l.disk_bytes,
                ]
            }));
        for c in all {
            c.store(0, Ordering::Relaxed);
        }
    }

    /// Point-in-time copy. `hot` comes from the attached hot tier.
    pub fn snapshot(&self, hot: Option<super::hot_tier::HotTierStats>) -> ExpertIoStats {
        let rs = |k: ReadKind| ReadStats {
            reads: get(&self.reads[k.idx()]),
            bytes: get(&self.bytes[k.idx()]),
        };
        let gs = |w: Guesser| {
            let g = &self.guess[w as usize];
            GuessStats {
                layers: get(&g.layers),
                guessed: get(&g.guessed),
                correct: get(&g.correct),
            }
        };
        let per_layer: Vec<LayerIoStats> = self
            .layers
            .iter()
            .map(|l| LayerIoStats {
                hot_hits: get(&l.hot_hits),
                prefetch_hits: get(&l.prefetch_hits),
                demand_reads: get(&l.demand_reads),
                prefetch_reads: get(&l.prefetch_reads),
                prefill_hot: get(&l.prefill_hot),
                prefill_reads: get(&l.prefill_reads),
                disk_bytes: get(&l.disk_bytes),
            })
            .collect();
        let kinds = [ReadKind::Demand, ReadKind::Prefetch, ReadKind::Prefill, ReadKind::Seed];
        ExpertIoStats {
            disk_reads: kinds.iter().map(|&k| rs(k).reads).sum(),
            disk_bytes: kinds.iter().map(|&k| rs(k).bytes).sum(),
            demand: rs(ReadKind::Demand),
            prefetch: rs(ReadKind::Prefetch),
            prefill: rs(ReadKind::Prefill),
            seed: rs(ReadKind::Seed),
            fast_dir_bytes: get(&self.fast_dir_bytes),
            prefetch_hits: per_layer.iter().map(|l| l.prefetch_hits).sum(),
            decode_tokens: get(&self.decode_tokens),
            prefill_tokens: get(&self.prefill_tokens),
            decode_ns: get(&self.decode_ns),
            io_wait_ns: get(&self.io_wait_ns),
            prefill_ns: get(&self.prefill_ns),
            prefill_io_ns: get(&self.prefill_io_ns),
            guess_prev: gs(Guesser::PrevToken),
            guess_cross: gs(Guesser::CrossLayer),
            hot,
            per_layer,
        }
    }
}

/// How many of `guess[..k]` (negative entries ignored) appear in
/// `actual`. Set overlap; the order of either side does not matter.
pub fn guess_overlap(guess: &[i32], actual: &[i32]) -> u64 {
    guess
        .iter()
        .filter(|&&g| g >= 0 && actual.contains(&g))
        .count() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_attributed_by_kind_and_layer() {
        let c = IoCounters::new(4);
        c.record_read(ReadKind::Demand, 1, 1, 100, false);
        c.record_read(ReadKind::Prefetch, 1, 2, 200, true);
        c.record_read(ReadKind::Prefill, 3, 5, 500, false);
        c.record_read(ReadKind::Seed, 0, 1, 100, false);
        c.record_decode_layer(1, 3, 2);
        c.record_decode(1_000);
        c.record_decode(3_000);
        let s = c.snapshot(None);
        assert_eq!(s.disk_reads, 9);
        assert_eq!(s.disk_bytes, 900);
        assert_eq!(s.demand, ReadStats { reads: 1, bytes: 100 });
        assert_eq!(s.prefetch, ReadStats { reads: 2, bytes: 200 });
        assert_eq!(s.fast_dir_bytes, 200);
        assert_eq!(s.prefetch_hits, 2);
        assert_eq!(s.per_layer[1].hot_hits, 3);
        assert_eq!(s.per_layer[1].disk_bytes, 300);
        assert_eq!(s.per_layer[3].prefill_reads, 5);
        assert_eq!(s.decode_tokens, 2);
        assert_eq!(s.decode_bytes_per_token(), Some(150.0));
        c.reset();
        let zero = ExpertIoStats {
            per_layer: vec![LayerIoStats::default(); 4],
            ..Default::default()
        };
        assert_eq!(c.snapshot(None), zero);
    }

    #[test]
    fn guess_accuracy_and_overlap() {
        assert_eq!(guess_overlap(&[1, 2, 3, -1], &[3, 9, 1, 4]), 2);
        assert_eq!(guess_overlap(&[], &[1]), 0);
        let c = IoCounters::new(1);
        assert_eq!(c.snapshot(None).guess_cross.accuracy(), None);
        c.record_guess(Guesser::CrossLayer, 8, 6);
        c.record_guess(Guesser::CrossLayer, 8, 2);
        c.record_guess(Guesser::PrevToken, 8, 4);
        let s = c.snapshot(None);
        assert_eq!(s.guess_cross.layers, 2);
        assert_eq!(s.guess_cross.accuracy(), Some(0.5));
        assert_eq!(s.guess_prev.accuracy(), Some(0.5));
    }

    #[test]
    fn csv_row_matches_header() {
        let c = IoCounters::new(2);
        c.record_read(ReadKind::Demand, 0, 1, 10, false);
        let s = c.snapshot(None);
        let cols = ExpertIoStats::CSV_HEADER.split(',').count();
        assert_eq!(s.csv_row().split(',').count(), cols);
        let with_hot = ExpertIoStats { hot: Some(Default::default()), ..s.clone() };
        assert_eq!(with_hot.csv_row().split(',').count(), cols);
        let layer_cols = ExpertIoStats::LAYER_CSV_HEADER.split(',').count();
        let csv = s.layer_csv();
        let rows: Vec<&str> = csv.lines().collect();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.split(',').count() == layer_cols));
    }
}
