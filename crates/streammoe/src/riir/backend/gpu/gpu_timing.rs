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
//! next one's `gpu_start` are time the GPU sat idle. Unset (the default)
//! costs one `OnceLock` load per wait.

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
    let _ = writeln!(out, "{label} {start:.9} {end:.9} {resumed:.9}");
    // Flushed per line: servers are often stopped with a signal, and
    // ~80 lines per token cost nothing next to the token.
    let _ = out.flush();
}
