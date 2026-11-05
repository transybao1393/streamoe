//! Command-buffer GPU timestamps and an opt-in GPU timeline trace.
//!
//! metal-rs does not surface `MTLCommandBuffer.GPUStartTime` /
//! `GPUEndTime`, so [`gpu_span`] reads them through the Objective-C
//! runtime. Both are host-clock seconds (the `mach_absolute_time` base),
//! valid once the command buffer has completed.
//!
//! Set `STREAMMOE_GPU_TIMING=<path>` and every instrumented wait point
//! appends one line:
//!
//! ```text
//! <label> <gpu_start> <gpu_end> <cpu_resumed>
//! ```
//!
//! in host-clock seconds, `cpu_resumed` being when the waiting CPU
//! thread got control back. Gaps between one buffer's `gpu_end` and the
//! next one's `gpu_start` are time the GPU sat idle. Command buffers
//! nobody waits on (early expert dispatch) are [`record_later`]ed and
//! written once complete; host-side events (the start and end of a
//! layer's expert reads, a commit) are [`mark`]ed as
//!
//! ```text
//! @<label> <host_time>
//! ```
//!
//! Unset (the default) costs one `OnceLock` load per call.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

use metal::CommandBufferRef;
use objc::{msg_send, sel, sel_impl};

/// `(GPUStartTime, GPUEndTime)` of a completed command buffer, in
/// host-clock seconds. `(0, 0)` if it never ran on the GPU.
pub fn gpu_span(cmdbuf: &CommandBufferRef) -> (f64, f64) {
    unsafe {
        let start: f64 = msg_send![cmdbuf, GPUStartTime];
        let end: f64 = msg_send![cmdbuf, GPUEndTime];
        (start, end)
    }
}

/// Host clock now, in the same seconds base as [`gpu_span`].
pub fn host_now() -> f64 {
    static TIMEBASE: OnceLock<f64> = OnceLock::new();
    let scale = *TIMEBASE.get_or_init(|| {
        let mut tb = libc::mach_timebase_info { numer: 0, denom: 0 };
        // SAFETY: plain out-parameter call.
        unsafe { libc::mach_timebase_info(&mut tb) };
        tb.numer as f64 / tb.denom as f64 * 1e-9
    });
    // SAFETY: no arguments, no side effects.
    unsafe { libc::mach_absolute_time() as f64 * scale }
}

fn trace() -> Option<&'static Mutex<BufWriter<File>>> {
    static T: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
    T.get_or_init(|| {
        let path = std::env::var_os("STREAMMOE_GPU_TIMING")?;
        let f = File::create(&path).ok()?;
        Some(Mutex::new(BufWriter::new(f)))
    })
    .as_ref()
}

/// Append `cmdbuf`'s GPU span to the timeline trace, if enabled. Call
/// right after `wait_until_completed`.
pub fn record(label: &str, cmdbuf: &CommandBufferRef) {
    let Some(t) = trace() else { return };
    let resumed = host_now();
    let (start, end) = gpu_span(cmdbuf);
    let mut out = t.lock().unwrap();
    flush_later(&mut out);
    let _ = writeln!(out, "{label} {start:.9} {end:.9} {resumed:.9}");
    // Flushed per line: servers are often stopped with a signal, and
    // ~80 lines per token cost nothing next to the token.
    let _ = out.flush();
}

/// Command buffers handed to [`record_later`], written by the next
/// [`record`] once they have completed.
static LATER: Mutex<Vec<(&'static str, metal::CommandBuffer)>> = Mutex::new(Vec::new());

/// Trace a committed command buffer nobody waits on: its line is written
/// by a later [`record`] once it has completed (`cpu_resumed` is then
/// that later time). No-op unless tracing is on.
pub fn record_later(label: &'static str, cmdbuf: &CommandBufferRef) {
    if trace().is_none() {
        return;
    }
    LATER.lock().unwrap().push((label, cmdbuf.to_owned()));
}

fn flush_later(out: &mut BufWriter<File>) {
    let mut later = LATER.lock().unwrap();
    later.retain(|(label, cb)| {
        if cb.status() != metal::MTLCommandBufferStatus::Completed {
            return true;
        }
        let (start, end) = gpu_span(cb);
        let _ = writeln!(out, "{label} {start:.9} {end:.9} {:.9}", host_now());
        false
    });
}

/// Mark a host-side event at the current host time. No-op unless
/// tracing is on.
pub fn mark(label: &str) {
    let Some(t) = trace() else { return };
    let now = host_now();
    let mut out = t.lock().unwrap();
    let _ = writeln!(out, "@{label} {now:.9}");
}
