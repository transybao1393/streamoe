//! Opt-in disk timeline trace: when expert reads run and when the SSD
//! sits idle.
//!
//! Set `STREAMMOE_IO_TRACE=<path>` and every expert read appends
//!
//! ```text
//! r <t0_us> <t1_us> <kind> <layer> <experts> <bytes>
//! ```
//!
//! (`kind`: `D` decode miss, `P` prefetch, `F` prefill staging, `S`
//! hot-tier seed), and the forward pass appends markers
//!
//! ```text
//! m <t_us> <tag> <layer>
//! ```
//!
//! `T` / `E` decode token start / end, `L` layer top, `R` layer routed,
//! `W` its expert reads landed, `S` / `s` prefill staging start / end.
//! Times are microseconds since the first event. `tools/io_trace_report.py`
//! turns a trace into busy / idle time per phase. Unset (the default)
//! costs one `OnceLock` load per call; set, the locking and formatting
//! slow decode by ~15%, so don't measure speed with it on.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use super::io_stats::ReadKind;

struct Trace {
    out: BufWriter<File>,
    t0: Instant,
}

fn trace() -> Option<&'static Mutex<Trace>> {
    static T: OnceLock<Option<Mutex<Trace>>> = OnceLock::new();
    T.get_or_init(|| {
        let path = std::env::var_os("STREAMMOE_IO_TRACE")?;
        let f = File::create(&path).ok()?;
        Some(Mutex::new(Trace { out: BufWriter::new(f), t0: Instant::now() }))
    })
    .as_ref()
}

/// Start time for a later [`read`]; `None` (no clock read) when off.
pub fn now() -> Option<Instant> {
    trace().map(|_| Instant::now())
}

/// One read of `experts` experts (`bytes` bytes) of `layer` that began
/// at `start`.
pub fn read(start: Option<Instant>, kind: ReadKind, layer: usize, experts: usize, bytes: usize) {
    let (Some(t), Some(start)) = (trace(), start) else { return };
    let end = Instant::now();
    let kind = match kind {
        ReadKind::Demand => 'D',
        ReadKind::Prefetch => 'P',
        ReadKind::Prefill => 'F',
        ReadKind::Seed => 'S',
    };
    let mut t = t.lock().unwrap();
    let a = start.saturating_duration_since(t.t0).as_micros();
    let b = end.saturating_duration_since(t.t0).as_micros();
    let _ = writeln!(t.out, "r {a} {b} {kind} {layer} {experts} {bytes}");
}

/// A phase marker (see the module docs for the tags). The token-end
/// marker `E` flushes, so a trace is readable while the process runs.
pub fn mark(tag: char, layer: usize) {
    let Some(t) = trace() else { return };
    let mut t = t.lock().unwrap();
    let a = t.t0.elapsed().as_micros();
    let _ = writeln!(t.out, "m {a} {tag} {layer}");
    if tag == 'E' {
        let _ = t.out.flush();
    }
}
