//! Opt-in per-token routing trace, for replaying expert-cache and
//! prefetch policies offline (`tools/sim_expert_cache.py`).
//!
//! Set `STREAMMOE_ROUTING_TRACE=<path>` and every per-token MoE dispatch
//! appends one line:
//!
//! ```text
//! d <token> <layer> e0 … e{k-1} : p0 … p{E-1}
//! ```
//!
//! with the router's picks and its full softmax. `<token>` counts
//! dispatches of layer 0, so it is per process, not per position.
//! Unset (the default) costs one `OnceLock` load per layer.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

struct Trace {
    out: BufWriter<File>,
    token: u64,
    probs: Vec<f32>,
}

fn trace() -> Option<&'static Mutex<Trace>> {
    static T: OnceLock<Option<Mutex<Trace>>> = OnceLock::new();
    T.get_or_init(|| {
        let path = std::env::var_os("STREAMMOE_ROUTING_TRACE")?;
        let f = File::create(&path).ok()?;
        Some(Mutex::new(Trace { out: BufWriter::new(f), token: 0, probs: Vec::new() }))
    })
    .as_ref()
}

/// Stash the router's softmax for the next [`record`].
pub fn stash_probs(probs: &[f32]) {
    if let Some(t) = trace() {
        let mut t = t.lock().unwrap();
        t.probs.clear();
        t.probs.extend_from_slice(probs);
    }
}

/// Write one line for `layer`'s picks and the stashed softmax.
pub fn record(layer: usize, picks: &[i32]) {
    let Some(t) = trace() else { return };
    let mut t = t.lock().unwrap();
    if layer == 0 {
        t.token += 1;
    }
    let Trace { out, token, probs } = &mut *t;
    let _ = write!(out, "d {} {layer}", *token - 1);
    for e in picks {
        let _ = write!(out, " {e}");
    }
    let _ = write!(out, " :");
    for p in probs.iter() {
        let _ = write!(out, " {p:.4e}");
    }
    let _ = writeln!(out);
    let _ = out.flush();
}
