//! Pipeline-parallel streammoe over the network (plan phase 5).
//!
//! A model is split into contiguous layer ranges, one per **node**
//! process; each node runs a streammoe context opened with
//! [`streammoe::OpenConfig::layer_range`]. A **coordinator** drives the
//! chain: it sends token ids to the first node, relays each node's f32
//! hidden state to the next, and receives the last node's logits.
//!
//! - [`frame`]: the wire format — a fixed 32-byte versioned header
//!   (kind, request id, position, row count, payload length) and a
//!   length-prefixed payload, read into and written from fixed-size
//!   buffers allocated once.
//! - [`node`]: the node's serve loop over any [`node::Stage`] (a
//!   streammoe context, or a test double).
//! - [`coordinator`]: [`coordinator::Chain`], the client that drives a
//!   chain of nodes and measures each node's service time.
//! - [`planner`]: assigns layer ranges from each node's memory cap and
//!   measured speed.
//!
//! Topology is a star: every hidden state passes through the
//! coordinator. That costs one extra hop per stage but keeps nodes
//! stateless about each other and lets the coordinator time every node
//! on its own; a direct node-to-node ring is future work.
//!
//! Frames carry f32 hidden states (not BF16) so a split over the
//! network computes exactly what the same split computes in-process.

#![warn(missing_docs)]

pub mod coordinator;
pub mod frame;
pub mod node;
pub mod planner;

pub use frame::{FrameBuf, Kind, NetError, NodeInfo};

/// Parse a byte count: a plain number or one with a `K` / `M` / `G`
/// (binary) suffix, e.g. `10G`.
pub fn parse_bytes(s: &str) -> Option<u64> {
    let s = s.trim();
    let (num, mul) = match s.as_bytes().last()? {
        b'K' | b'k' => (&s[..s.len() - 1], 1u64 << 10),
        b'M' | b'm' => (&s[..s.len() - 1], 1 << 20),
        b'G' | b'g' => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    if let Ok(n) = num.parse::<u64>() {
        return n.checked_mul(mul);
    }
    let f = num.parse::<f64>().ok()?;
    (f.is_finite() && f >= 0.0).then(|| (f * mul as f64) as u64)
}

/// Parse a layer range `a..b`.
pub fn parse_range(s: &str) -> Option<std::ops::Range<usize>> {
    let (a, b) = s.split_once("..")?;
    Some(a.trim().parse().ok()?..b.trim().parse().ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sizes_and_ranges() {
        assert_eq!(parse_bytes("10G"), Some(10 << 30));
        assert_eq!(parse_bytes("1.5g"), Some(3 << 29));
        assert_eq!(parse_bytes("512M"), Some(512 << 20));
        assert_eq!(parse_bytes("4096"), Some(4096));
        assert_eq!(parse_bytes("x"), None);
        assert_eq!(parse_range("20..40"), Some(20..40));
        assert_eq!(parse_range("20-40"), None);
    }
}
