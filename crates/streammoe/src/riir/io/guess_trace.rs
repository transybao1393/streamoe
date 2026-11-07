//! Opt-in binary trace of each decode layer's router input, picks and
//! hot-tier residency, for scoring prefetch guessers offline
//! (`tools/sim_prefetch_guess.py`).
//!
//! Set `STREAMMOE_GUESS_TRACE=<path>` and every per-token MoE layer
//! appends one little-endian record:
//!
//! ```text
//! u32 layer, u32 k, u32 hidden, u32 n_experts,
//! f32 x[hidden]          the residual the next layer's guess starts from
//! i32 picks[k]           the router's picks
//! f32 weights[k]         their routing weights
//! u8  hot[n_experts / 8] hot-tier residency of this layer before its reads
//!                        (bit e % 8 of byte e / 8)
//! ```
//!
//! `x` is the qwen3_5_moe path's `h_mid` (pre-norm post-attention
//! residual) and the Flash-Next engine's MoE input `mixed`. Unset (the
//! default) costs one `OnceLock` load per layer.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

fn trace() -> Option<&'static Mutex<BufWriter<File>>> {
    static T: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
    T.get_or_init(|| {
        let path = std::env::var_os("STREAMMOE_GUESS_TRACE")?;
        let f = File::create(&path).ok()?;
        Some(Mutex::new(BufWriter::new(f)))
    })
    .as_ref()
}

/// `true` when the trace is on (callers skip building the record).
pub fn enabled() -> bool {
    trace().is_some()
}

/// Append one record. `hot(e)`: is expert `e` resident for `layer`.
pub fn record(
    layer: usize,
    x: &[f32],
    picks: &[i32],
    weights: &[f32],
    n_experts: usize,
    hot: impl Fn(usize) -> bool,
) {
    let Some(t) = trace() else { return };
    let mut out = t.lock().unwrap();
    let mut buf = Vec::with_capacity(16 + 4 * (x.len() + 2 * picks.len()) + n_experts / 8 + 1);
    for v in [layer, picks.len(), x.len(), n_experts] {
        buf.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for v in x {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in picks {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in weights {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let mut bits = vec![0u8; n_experts.div_ceil(8)];
    for e in 0..n_experts {
        if hot(e) {
            bits[e / 8] |= 1 << (e % 8);
        }
    }
    buf.extend_from_slice(&bits);
    let _ = out.write_all(&buf);
    let _ = out.flush();
}
