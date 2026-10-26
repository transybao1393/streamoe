//! Expert reads from SSD at decode sizes: how fast do `n` experts
//! arrive when read together, and does splitting each expert into
//! pieces (more reads in flight) help? Also: large adjacent reads,
//! async submission (POSIX AIO, GCD `dispatch_io`) against the
//! engine's thread-pool `pread`, and what the hot-tier copy after
//! each read costs. Results so far are in `docs/known-issues.md`
//! (*Measured dead ends*).
//!
//! Numbers are only meaningful on an idle SSD: check that no model
//! job or other heavy reader is running (`iostat -d -w 1 disk0`).
//!
//! Reads the real a3b expert files (`STREAMMOE_SMOKE_ROOT`, default
//! `~/models/qwen3-6-35b-a3b/root`) with the engine's flags:
//! `F_NOCACHE`, readahead off, page-aligned buffers. Picks are random
//! across all layers, so the page cache does not serve them.
//!
//! ```text
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b --release \
//!   --test ssd_read_bench -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(target_os = "macos")]

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

const EXPERT: usize = 1_769_472;
const LAYERS: usize = 40;
const EXPERTS: usize = 256;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Page-aligned anonymous buffer (what `F_NOCACHE` needs to bypass the
/// page cache).
struct Aligned {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for Aligned {}
unsafe impl Sync for Aligned {}
impl Aligned {
    fn new(len: usize) -> Self {
        // SAFETY: anonymous private mapping; checked for MAP_FAILED.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        Self { ptr: ptr.cast(), len }
    }
    /// SAFETY: callers use disjoint ranges.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, off: usize, len: usize) -> &mut [u8] {
        assert!(off + len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off), len) }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

fn open_layers() -> Vec<File> {
    let root = std::env::var_os("STREAMMOE_SMOKE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join("models/qwen3-6-35b-a3b/root"));
    (0..LAYERS)
        .map(|l| {
            let f = File::open(root.join(format!("packed_experts/layer_{l:02}.bin"))).expect("layer file");
            unsafe {
                libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1i32);
                libc::fcntl(f.as_raw_fd(), libc::F_RDAHEAD, 0i32);
            }
            f
        })
        .collect()
}

/// Read `picks` (layer, expert) together, each split into `pieces`
/// equal reads, all in flight at once on `pool`. Returns wall ms.
fn read_batch(pool: &rayon::ThreadPool, files: &[File], buf: &Aligned, picks: &[(usize, usize)], pieces: usize) -> f64 {
    let piece = EXPERT / pieces;
    let jobs: Vec<(usize, usize, usize)> = picks
        .iter()
        .enumerate()
        .flat_map(|(slot, &(l, e))| (0..pieces).map(move |p| (slot, l, e * EXPERT + p * piece)))
        .collect();
    let t = Instant::now();
    pool.install(|| {
        jobs.par_iter().with_max_len(1).for_each(|&(slot, l, off)| {
            let p = (off % EXPERT) / piece;
            let len = if p + 1 == pieces { EXPERT - p * piece } else { piece };
            // SAFETY: each (slot, piece) range is disjoint.
            let dst = unsafe { buf.slice(slot * EXPERT + p * piece, len) };
            let n = files[l].read_at(dst, off as u64).expect("read");
            assert_eq!(n, len);
        })
    });
    t.elapsed().as_secs_f64() * 1e3
}

#[test]
#[ignore = "SSD bench; needs the a3b expert files"]
fn ssd_expert_reads_by_concurrency() {
    let files = open_layers();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(32).build().unwrap();
    let buf = Aligned::new(16 * EXPERT);
    let mut rng = Rng(0x55d_7e57);
    let mut pick = |n: usize| -> Vec<(usize, usize)> {
        (0..n).map(|_| ((rng.next() % LAYERS as u64) as usize, (rng.next() % EXPERTS as u64) as usize)).collect()
    };
    // warm the pool / page tables
    for _ in 0..4 {
        read_batch(&pool, &files, &buf, &pick(4), 1);
    }
    eprintln!("\nexperts read together (1.77 MB each): median ms per batch, GB/s");
    eprint!("{:>8}", "experts");
    let pieces_list = [1usize, 2, 4, 8];
    for p in pieces_list {
        eprint!("  {:>18}", format!("{p} piece(s)"));
    }
    eprintln!();
    for n in [1usize, 2, 3, 4, 6, 8, 12, 16] {
        eprint!("{n:>8}");
        for &pieces in &pieces_list {
            let mut v: Vec<f64> = (0..30).map(|_| read_batch(&pool, &files, &buf, &pick(n), pieces)).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ms = v[v.len() / 2];
            eprint!("  {:>8.2} ms {:>5.1}", ms, (n * EXPERT) as f64 / ms / 1e6);
        }
        eprintln!();
    }

    // Today's decode layer: ~3 prefetch reads, then ~1.5 misses after
    // they all land, vs all of them issued together.
    eprintln!("\ndecode-layer pattern (3 prefetch + 2 misses), median of 40:");
    for (name, pieces) in [("1 piece", 1usize), ("4 pieces", 4)] {
        let (mut seq, mut tog) = (Vec::new(), Vec::new());
        for _ in 0..40 {
            let (a, b) = (pick(3), pick(2));
            seq.push(read_batch(&pool, &files, &buf, &a, pieces) + read_batch(&pool, &files, &buf, &b, pieces));
            let mut all = pick(3);
            all.extend(pick(2));
            tog.push(read_batch(&pool, &files, &buf, &all, pieces));
        }
        seq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tog.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("  {name:>8}: prefetch then misses {:.2} ms, all together {:.2} ms", seq[20], tog[20]);
    }
}

/// Large blocks and sequential placement: n adjacent experts read as
/// one `pread`, as n parallel reads of the adjacent experts, and as n
/// parallel reads of random experts.
#[test]
#[ignore = "SSD bench; needs the a3b expert files"]
fn ssd_expert_reads_large_and_sequential() {
    let files = open_layers();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(32).build().unwrap();
    let buf = Aligned::new(16 * EXPERT);
    let mut rng = Rng(0x5e9_7e57);
    eprintln!("\nn experts: one large read (adjacent) | n parallel reads (adjacent) | n parallel reads (random); median ms, GB/s");
    for n in [1usize, 2, 4, 8, 16] {
        let (mut big, mut adj, mut rnd) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..30 {
            let l = (rng.next() % LAYERS as u64) as usize;
            let e0 = (rng.next() % (EXPERTS - n) as u64) as usize;
            let t = Instant::now();
            // SAFETY: whole buffer, single reader.
            let dst = unsafe { buf.slice(0, n * EXPERT) };
            assert_eq!(files[l].read_at(dst, (e0 * EXPERT) as u64).unwrap(), n * EXPERT);
            big.push(t.elapsed().as_secs_f64() * 1e3);
            let l = (rng.next() % LAYERS as u64) as usize;
            let e0 = (rng.next() % (EXPERTS - n) as u64) as usize;
            let picks: Vec<_> = (0..n).map(|i| (l, e0 + i)).collect();
            adj.push(read_batch(&pool, &files, &buf, &picks, 1));
            let picks: Vec<_> = (0..n)
                .map(|_| ((rng.next() % LAYERS as u64) as usize, (rng.next() % EXPERTS as u64) as usize))
                .collect();
            rnd.push(read_batch(&pool, &files, &buf, &picks, 1));
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let gb = |ms: f64| (n * EXPERT) as f64 / ms / 1e6;
        let (b, a, r) = (med(&mut big), med(&mut adj), med(&mut rnd));
        eprintln!("{n:>3}: {b:6.2} ms {:4.1} | {a:6.2} ms {:4.1} | {r:6.2} ms {:4.1}", gb(b), gb(a), gb(r));
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// POSIX AIO: one `aiocb` per expert, submitted with one
/// `lio_listio(LIO_WAIT)` (`listio`) or with `aio_read` each and
/// reaped with `aio_suspend`. Returns wall ms.
fn aio_batch(files: &[File], buf: &Aligned, picks: &[(usize, usize)], listio: bool) -> f64 {
    // SAFETY (whole fn): the aiocbs and their destination ranges stay
    // put and disjoint until every request has been reaped.
    let mut cbs: Vec<libc::aiocb> = picks
        .iter()
        .enumerate()
        .map(|(slot, &(l, e))| {
            let mut cb: libc::aiocb = unsafe { std::mem::zeroed() };
            cb.aio_fildes = files[l].as_raw_fd();
            cb.aio_offset = (e * EXPERT) as libc::off_t;
            cb.aio_buf = unsafe { buf.slice(slot * EXPERT, EXPERT) }.as_mut_ptr().cast();
            cb.aio_nbytes = EXPERT;
            cb.aio_lio_opcode = libc::LIO_READ;
            cb
        })
        .collect();
    let t = Instant::now();
    if listio {
        let mut list: Vec<*mut libc::aiocb> = cbs.iter_mut().map(|c| c as *mut _).collect();
        let rc = unsafe { libc::lio_listio(libc::LIO_WAIT, list.as_mut_ptr(), list.len() as i32, std::ptr::null_mut()) };
        assert_eq!(rc, 0, "lio_listio: {}", std::io::Error::last_os_error());
        for cb in cbs.iter_mut() {
            assert_eq!(unsafe { libc::aio_return(cb) }, EXPERT as isize);
        }
    } else {
        for cb in cbs.iter_mut() {
            let rc = unsafe { libc::aio_read(cb) };
            assert_eq!(rc, 0, "aio_read: {}", std::io::Error::last_os_error());
        }
        let mut pending: Vec<usize> = (0..cbs.len()).collect();
        while !pending.is_empty() {
            let list: Vec<*const libc::aiocb> = pending.iter().map(|&i| &cbs[i] as *const _).collect();
            unsafe { libc::aio_suspend(list.as_ptr(), list.len() as i32, std::ptr::null()) };
            pending.retain(|&i| {
                let cb = &mut cbs[i];
                if unsafe { libc::aio_error(cb) } == libc::EINPROGRESS {
                    return true;
                }
                assert_eq!(unsafe { libc::aio_return(cb) }, EXPERT as isize);
                false
            });
        }
    }
    t.elapsed().as_secs_f64() * 1e3
}

/// GCD `dispatch_io` (random-access channels), called through
/// hand-built global blocks so the bench needs no block crate.
mod gcd {
    use std::ffi::{c_int, c_void};
    use std::os::unix::io::AsRawFd;

    type Obj = *mut c_void;
    unsafe extern "C" {
        static _NSConcreteGlobalBlock: c_void;
        fn dispatch_queue_create(label: *const i8, attr: Obj) -> Obj;
        fn dispatch_io_create(ty: usize, fd: c_int, queue: Obj, cleanup: *const c_void) -> Obj;
        fn dispatch_io_set_low_water(ch: Obj, low: usize);
        fn dispatch_io_read(ch: Obj, off: i64, len: usize, queue: Obj, handler: *const c_void);
        fn dispatch_data_create_map(data: Obj, buf: *mut *const c_void, size: *mut usize) -> Obj;
        fn dispatch_release(obj: Obj);
        fn dispatch_semaphore_create(v: isize) -> Obj;
        fn dispatch_semaphore_signal(s: Obj) -> isize;
        fn dispatch_semaphore_wait(s: Obj, timeout: u64) -> isize;
    }
    const BLOCK_IS_GLOBAL: i32 = 1 << 28;

    #[repr(C)]
    struct Desc {
        reserved: usize,
        size: usize,
    }
    static CLEANUP_DESC: Desc = Desc { reserved: 0, size: std::mem::size_of::<CleanupBlock>() };
    static READ_DESC: Desc = Desc { reserved: 0, size: std::mem::size_of::<ReadBlock>() };

    #[repr(C)]
    struct CleanupBlock {
        isa: *const c_void,
        flags: i32,
        reserved: i32,
        invoke: unsafe extern "C" fn(*const CleanupBlock, c_int),
        desc: *const Desc,
    }
    unsafe extern "C" fn cleanup_noop(_: *const CleanupBlock, _: c_int) {}

    /// `^(bool done, dispatch_data_t data, int error)` with the
    /// destination appended after the descriptor. Global blocks are
    /// never copied, so the handler sees this very struct.
    #[repr(C)]
    pub struct ReadBlock {
        isa: *const c_void,
        flags: i32,
        reserved: i32,
        invoke: unsafe extern "C" fn(*mut ReadBlock, bool, Obj, c_int),
        desc: *const Desc,
        dst: *mut u8,
        filled: usize,
        err: c_int,
        sem: Obj,
    }
    unsafe extern "C" fn on_read(b: *mut ReadBlock, done: bool, data: Obj, error: c_int) {
        let b = unsafe { &mut *b };
        if !data.is_null() {
            let (mut p, mut n) = (std::ptr::null(), 0usize);
            let map = unsafe { dispatch_data_create_map(data, &mut p, &mut n) };
            if n > 0 {
                unsafe { std::ptr::copy_nonoverlapping(p.cast::<u8>(), b.dst.add(b.filled), n) };
                b.filled += n;
            }
            unsafe { dispatch_release(map) };
        }
        if error != 0 {
            b.err = error;
        }
        if done {
            unsafe { dispatch_semaphore_signal(b.sem) };
        }
    }

    pub struct Gcd {
        channels: Vec<Obj>,
        queues: Vec<Obj>,
        sem: Obj,
        cleanup: Box<CleanupBlock>,
    }
    impl Gcd {
        pub fn new(files: &[std::fs::File], slots: usize) -> Self {
            let cleanup = Box::new(CleanupBlock {
                isa: unsafe { &_NSConcreteGlobalBlock },
                flags: BLOCK_IS_GLOBAL,
                reserved: 0,
                invoke: cleanup_noop,
                desc: &CLEANUP_DESC,
            });
            // One serial handler queue per slot, so copies of different
            // reads overlap but one read's chunks arrive in order.
            let queues: Vec<Obj> = (0..slots)
                .map(|_| unsafe { dispatch_queue_create(c"bench.io".as_ptr(), std::ptr::null_mut()) })
                .collect();
            let channels = files
                .iter()
                .map(|f| unsafe {
                    let ch = dispatch_io_create(1 /* RANDOM */, f.as_raw_fd(), queues[0], (&*cleanup as *const CleanupBlock).cast());
                    assert!(!ch.is_null());
                    dispatch_io_set_low_water(ch, usize::MAX);
                    ch
                })
                .collect();
            Self { channels, queues, sem: unsafe { dispatch_semaphore_create(0) }, cleanup }
        }

        /// Read each `(file, offset, len, dst)` concurrently; wall ms.
        pub fn read(&self, reqs: &[(usize, i64, usize, *mut u8)]) -> f64 {
            let mut blocks: Vec<ReadBlock> = reqs
                .iter()
                .map(|&(_, _, _, dst)| ReadBlock {
                    isa: unsafe { &_NSConcreteGlobalBlock },
                    flags: BLOCK_IS_GLOBAL,
                    reserved: 0,
                    invoke: on_read,
                    desc: &READ_DESC,
                    dst,
                    filled: 0,
                    err: 0,
                    sem: self.sem,
                })
                .collect();
            let t = std::time::Instant::now();
            for (i, (&(f, off, len, _), b)) in reqs.iter().zip(blocks.iter_mut()).enumerate() {
                let q = self.queues[i % self.queues.len()];
                unsafe { dispatch_io_read(self.channels[f], off, len, q, (b as *mut ReadBlock).cast()) };
            }
            for _ in reqs {
                unsafe { dispatch_semaphore_wait(self.sem, u64::MAX) };
            }
            let ms = t.elapsed().as_secs_f64() * 1e3;
            for (b, r) in blocks.iter().zip(reqs) {
                assert_eq!((b.err, b.filled), (0, r.2), "dispatch_io_read");
            }
            ms
        }
    }
    impl Drop for Gcd {
        fn drop(&mut self) {
            // Channels are left open (the bench process exits next);
            // closing would need the cleanup block to outlive them.
            let _ = &self.cleanup;
        }
    }
}

/// Thread-pool `pread` (the engine's way, 8 I/O threads) vs POSIX AIO
/// vs GCD `dispatch_io`, at decode queue depths. Methods take turns
/// within each trial so drift hits them alike.
#[test]
#[ignore = "SSD bench; needs the a3b expert files"]
fn ssd_async_vs_thread_pool() {
    let files = open_layers();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let gcd = gcd::Gcd::new(&files, 8);
    let buf = Aligned::new(16 * EXPERT);
    let mut rng = Rng(0xa10_7e57);
    let mut pick = |n: usize| -> Vec<(usize, usize)> {
        (0..n).map(|_| ((rng.next() % LAYERS as u64) as usize, (rng.next() % EXPERTS as u64) as usize)).collect()
    };
    let gcd_batch = |picks: &[(usize, usize)]| -> f64 {
        let reqs: Vec<_> = picks
            .iter()
            .enumerate()
            .map(|(slot, &(l, e))| (l, (e * EXPERT) as i64, EXPERT, unsafe { buf.slice(slot * EXPERT, EXPERT) }.as_mut_ptr()))
            .collect();
        gcd.read(&reqs)
    };
    for _ in 0..4 {
        read_batch(&pool, &files, &buf, &pick(4), 1);
        aio_batch(&files, &buf, &pick(4), true);
        gcd_batch(&pick(4));
    }
    eprintln!("\nexperts in flight: median ms per batch (GB/s), 40 trials");
    eprintln!("{:>3}  {:>16}  {:>16}  {:>16}  {:>16}", "n", "pool pread", "aio lio_listio", "aio suspend", "dispatch_io");
    for n in [1usize, 2, 3, 4, 6, 8] {
        let mut r: [Vec<f64>; 4] = Default::default();
        for _ in 0..40 {
            r[0].push(read_batch(&pool, &files, &buf, &pick(n), 1));
            r[1].push(aio_batch(&files, &buf, &pick(n), true));
            r[2].push(aio_batch(&files, &buf, &pick(n), false));
            r[3].push(gcd_batch(&pick(n)));
        }
        eprint!("{n:>3}");
        for v in r.iter_mut() {
            let ms = median(v);
            eprint!("  {:>7.2} ms ({:>4.1})", ms, (n * EXPERT) as f64 / ms / 1e6);
        }
        eprintln!();
    }
    // dispatch_io reads into its own buffers: if they are not page
    // aligned, F_NOCACHE does not apply and a re-read comes from RAM.
    let same = pick(4);
    let first = gcd_batch(&same);
    let again = gcd_batch(&same);
    eprintln!("dispatch_io re-read of the same 4 experts: {first:.2} ms then {again:.2} ms (much faster = page cache)");
}

/// The hot-tier `offer` copies each freshly read expert (1.77 MB)
/// into the tier under one mutex on the read thread, before the read
/// reports done. Cost per batch: read alone, read + copy, read + copy
/// under a shared lock (the engine today).
#[test]
#[ignore = "SSD bench; needs the a3b expert files"]
fn ssd_read_plus_offer_copy() {
    let files = open_layers();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let buf = Aligned::new(16 * EXPERT);
    let hot = Aligned::new(16 * EXPERT);
    // Touch the tier pages once, as a long-lived tier's are.
    unsafe { hot.slice(0, 16 * EXPERT) }.fill(1);
    let lock = std::sync::Mutex::new(());
    let mut rng = Rng(0x0ffe_7e57);
    let mut pick = |n: usize| -> Vec<(usize, usize)> {
        (0..n).map(|_| ((rng.next() % LAYERS as u64) as usize, (rng.next() % EXPERTS as u64) as usize)).collect()
    };
    // 0 = read only, 1 = + copy, 2 = + copy under the shared lock.
    let run = |picks: &[(usize, usize)], mode: u8| -> f64 {
        let t = Instant::now();
        pool.install(|| {
            picks.par_iter().enumerate().with_max_len(1).for_each(|(slot, &(l, e))| {
                // SAFETY: one slot per task.
                let dst = unsafe { buf.slice(slot * EXPERT, EXPERT) };
                assert_eq!(files[l].read_at(dst, (e * EXPERT) as u64).unwrap(), EXPERT);
                if mode > 0 {
                    let _g = (mode == 2).then(|| lock.lock().unwrap());
                    let to = unsafe { hot.slice(slot * EXPERT, EXPERT) };
                    to.copy_from_slice(dst);
                }
            })
        });
        t.elapsed().as_secs_f64() * 1e3
    };
    let mut c = Vec::new();
    for _ in 0..50 {
        let t = Instant::now();
        unsafe { hot.slice(0, EXPERT) }.copy_from_slice(unsafe { buf.slice(0, EXPERT) });
        c.push(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("\n1.77 MB memcpy alone: {:.3} ms", median(&mut c));
    eprintln!("experts read together: median ms per batch, 40 trials");
    eprintln!("{:>3}  {:>9}  {:>11}  {:>16}", "n", "read", "read+copy", "read+copy(lock)");
    for n in [1usize, 2, 3, 4, 6, 8] {
        let mut r: [Vec<f64>; 3] = Default::default();
        for _ in 0..40 {
            for (m, v) in r.iter_mut().enumerate() {
                v.push(run(&pick(n), m as u8));
            }
        }
        let (a, b, l) = (median(&mut r[0]), median(&mut r[1]), median(&mut r[2]));
        eprintln!("{n:>3}  {a:>6.2} ms  {b:>8.2} ms  {l:>13.2} ms   (+{:.0}% / +{:.0}%)", (b / a - 1.0) * 100.0, (l / a - 1.0) * 100.0);
    }
}
