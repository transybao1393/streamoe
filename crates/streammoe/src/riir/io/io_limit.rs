//! Bounded expert I/O (plan §3 rule 6, phase 6): a queue-depth gate
//! and an optional bytes/s limit, shared by every disk read — decode
//! misses, speculative prefetch, prefill staging and hot-tier seeding
//! — so the shared SSD and the cores are not flooded.
//!
//! - **Threads:** [`crate::riir::OpenConfig::io_threads`] sizes the
//!   I/O worker pool; every pool read is blocking, so it is also an
//!   upper bound on reads in flight.
//! - **Queue depth** ([`IoGate`]): at most `depth` reads in flight at
//!   once, whichever thread issues them. A read waits (spin, then
//!   yield) for a free permit.
//! - **Bytes/s** ([`RateLimiter`]): a GCRA token bucket. Each read
//!   reserves its bytes' share of time and sleeps until its slot;
//!   a burst of [`RATE_BURST`] worth of bytes passes without delay.
//!
//! Both are lock-free (one atomic each) and allocation-free. Limits
//! only delay reads; they never change which bytes are read, so the
//! math is unaffected.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Burst the rate limiter lets through without delay, as time at the
/// configured rate.
pub const RATE_BURST: Duration = Duration::from_millis(20);

/// Spins before a waiting read starts yielding its thread.
const SPINS_BEFORE_YIELD: u32 = 64;

/// At most `limit` holders at once. See the module docs.
#[derive(Debug)]
pub struct IoGate {
    inflight: AtomicUsize,
    limit: usize,
}

/// A held [`IoGate`] permit; released on drop.
#[derive(Debug)]
pub struct IoPermit<'a> {
    gate: Option<&'a IoGate>,
}

impl Drop for IoPermit<'_> {
    fn drop(&mut self) {
        if let Some(g) = self.gate {
            g.inflight.fetch_sub(1, Ordering::Release);
        }
    }
}

impl IoGate {
    /// A gate admitting `limit` (≥ 1) concurrent holders.
    pub fn new(limit: usize) -> Self {
        Self {
            inflight: AtomicUsize::new(0),
            limit: limit.max(1),
        }
    }

    /// Take a permit if one is free.
    pub fn try_acquire(&self) -> Option<IoPermit<'_>> {
        let mut cur = self.inflight.load(Ordering::Relaxed);
        while cur < self.limit {
            match self.inflight.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(IoPermit { gate: Some(self) }),
                Err(now) => cur = now,
            }
        }
        None
    }

    /// Take a permit, waiting for one to free up.
    pub fn acquire(&self) -> IoPermit<'_> {
        let mut spins = 0u32;
        loop {
            if let Some(p) = self.try_acquire() {
                return p;
            }
            if spins < SPINS_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Permits held right now.
    pub fn in_flight(&self) -> usize {
        self.inflight.load(Ordering::Relaxed)
    }
}

/// GCRA step: with theoretical arrival time `tat`, a request at `now`
/// costing `cost` (all ns) may start after `wait` ns; returns
/// `(new_tat, wait)`. Up to `burst` ns of debt passes immediately.
pub fn gcra(tat: u64, now: u64, cost: u64, burst: u64) -> (u64, u64) {
    let start = tat.max(now);
    let wait = start.saturating_sub(burst).saturating_sub(now);
    (start + cost, wait)
}

/// Bytes/s limit. See the module docs.
#[derive(Debug)]
pub struct RateLimiter {
    bytes_per_sec: u64,
    burst_ns: u64,
    origin: Instant,
    /// Theoretical arrival time, ns since `origin`.
    tat: AtomicU64,
}

impl RateLimiter {
    /// A limiter for `bytes_per_sec` (≥ 1).
    pub fn new(bytes_per_sec: u64) -> Self {
        Self {
            bytes_per_sec: bytes_per_sec.max(1),
            burst_ns: RATE_BURST.as_nanos() as u64,
            origin: Instant::now(),
            tat: AtomicU64::new(0),
        }
    }

    /// Time `bytes` take at the configured rate, in ns.
    pub fn cost_ns(&self, bytes: u64) -> u64 {
        ((bytes as u128 * 1_000_000_000) / self.bytes_per_sec as u128) as u64
    }

    /// Reserve `bytes` at `now_ns` (since this limiter's origin);
    /// returns how long the caller must wait before reading.
    pub fn reserve_at(&self, now_ns: u64, bytes: u64) -> Duration {
        let cost = self.cost_ns(bytes);
        let mut tat = self.tat.load(Ordering::Relaxed);
        loop {
            let (next, wait) = gcra(tat, now_ns, cost, self.burst_ns);
            match self
                .tat
                .compare_exchange_weak(tat, next, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => return Duration::from_nanos(wait),
                Err(cur) => tat = cur,
            }
        }
    }

    /// Reserve `bytes` now and sleep until the read may start.
    pub fn throttle(&self, bytes: u64) {
        let now = self.origin.elapsed().as_nanos() as u64;
        let wait = self.reserve_at(now, bytes);
        if !wait.is_zero() {
            std::thread::sleep(wait);
        }
    }
}

/// The limits one context applies to its expert reads.
#[derive(Debug, Default)]
pub struct IoLimits {
    gate: Option<IoGate>,
    rate: Option<RateLimiter>,
}

impl IoLimits {
    /// `queue_depth = None` adds no gate beyond the pool size;
    /// `bytes_per_sec = None` is unlimited.
    pub fn new(queue_depth: Option<usize>, bytes_per_sec: Option<u64>) -> Self {
        Self {
            gate: queue_depth.map(IoGate::new),
            rate: bytes_per_sec.map(RateLimiter::new),
        }
    }

    /// Admit one read of `bytes`: wait for its rate slot, then for a
    /// queue-depth permit (held until the returned value drops).
    pub fn admit(&self, bytes: u64) -> IoPermit<'_> {
        if let Some(r) = &self.rate {
            r.throttle(bytes);
        }
        match &self.gate {
            Some(g) => g.acquire(),
            None => IoPermit { gate: None },
        }
    }

    /// The queue-depth gate, if configured.
    pub fn gate(&self) -> Option<&IoGate> {
        self.gate.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn gate_bounds_holders() {
        let g = IoGate::new(2);
        let a = g.try_acquire().expect("first");
        let _b = g.try_acquire().expect("second");
        assert!(g.try_acquire().is_none());
        assert_eq!(g.in_flight(), 2);
        drop(a);
        assert!(g.try_acquire().is_some());
        assert_eq!(IoGate::new(0).limit, 1);
    }

    #[test]
    fn gate_never_exceeds_limit_under_contention() {
        let g = Arc::new(IoGate::new(3));
        let peak = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (g, peak) = (g.clone(), peak.clone());
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        let _p = g.acquire();
                        peak.fetch_max(g.in_flight(), Ordering::Relaxed);
                        std::hint::spin_loop();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert!(peak.load(Ordering::Relaxed) <= 3);
        assert_eq!(g.in_flight(), 0);
    }

    #[test]
    fn gcra_burst_then_paced() {
        // 10 ns per request, 20 ns burst.
        let (mut tat, mut waits) = (0u64, Vec::new());
        for _ in 0..5 {
            let (t, w) = gcra(tat, 0, 10, 20);
            tat = t;
            waits.push(w);
        }
        assert_eq!(waits, vec![0, 0, 0, 10, 20]);
        // Idle time refills the bucket.
        let (_, w) = gcra(tat, 1_000, 10, 20);
        assert_eq!(w, 0);
    }

    #[test]
    fn rate_limiter_paces_bytes() {
        let r = RateLimiter::new(1_000_000); // 1 MB/s → 1 µs per byte
        assert_eq!(r.cost_ns(1_000), 1_000_000);
        // 20 ms burst: the first 20 KB pass, the next 1 KB waits ~1 ms.
        // (GCRA admits burst / cost + 1 requests at once.)
        let mut total = Duration::ZERO;
        for _ in 0..21 {
            total += r.reserve_at(0, 1_000);
        }
        assert_eq!(total, Duration::ZERO);
        assert_eq!(r.reserve_at(0, 1_000), Duration::from_millis(1));
    }

    #[test]
    fn unlimited_admits_immediately() {
        let l = IoLimits::new(None, None);
        let _p = l.admit(1 << 30);
        assert!(l.gate().is_none());
        let l = IoLimits::new(Some(1), None);
        let p = l.admit(10);
        assert_eq!(l.gate().unwrap().in_flight(), 1);
        drop(p);
        assert_eq!(l.gate().unwrap().in_flight(), 0);
    }
}
