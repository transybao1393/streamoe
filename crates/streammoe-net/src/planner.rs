//! Layer-range planner.
//!
//! Splits `0..num_layers` into one contiguous range per node, in node
//! order, so that every node's range fits its memory cap and the
//! slowest stage is as fast as possible (pipeline throughput is set by
//! the slowest stage). Ties go to the split with the smaller total
//! time (single-request latency).
//!
//! Inputs are per-layer byte costs ([`LayerCosts`], built from the
//! model manifest and the same budget formulas a capped
//! [`streammoe::OpenConfig`] open uses) and per-node [`NodeSpec`]s:
//! the cap and a measured speed in layers per second (e.g. from
//! [`speed_from_latency`] over a coordinator run). The plan is an
//! estimate — each node's own capped open is the authoritative check.

use std::ops::Range;

/// One node to plan for.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeSpec {
    /// The node's memory cap in bytes (`u64::MAX` = unlimited).
    pub mem_cap: u64,
    /// Measured speed, in (unit-work) layers per second; only ratios
    /// between nodes matter.
    pub speed: f64,
}

/// Memory and work per layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerCosts {
    /// Bytes each layer adds to a node: its weights plus its KV /
    /// linear-attn state for the whole request pool.
    pub layer_bytes: Vec<u64>,
    /// Embedding bytes (the node holding layer 0).
    pub embed_bytes: u64,
    /// Final norm + LM head bytes (the node holding the last layer).
    pub head_bytes: u64,
    /// Bytes every node pays regardless of range: shared tensors,
    /// expert slot ring, prefill staging, runtime overhead, network
    /// buffers.
    pub fixed_bytes: u64,
    /// Relative compute per layer (1.0 = one MoE layer).
    pub layer_work: Vec<f64>,
}

impl LayerCosts {
    /// Uniform costs, for tests and rough plans.
    pub fn uniform(num_layers: usize, layer_bytes: u64, fixed_bytes: u64) -> Self {
        Self {
            layer_bytes: vec![layer_bytes; num_layers],
            embed_bytes: 0,
            head_bytes: 0,
            fixed_bytes,
            layer_work: vec![1.0; num_layers],
        }
    }

    /// Layers in the model.
    pub fn num_layers(&self) -> usize {
        self.layer_bytes.len()
    }

    /// Bytes a node running `r` needs.
    pub fn range_bytes(&self, r: &Range<usize>) -> u64 {
        let mut b = self.fixed_bytes + self.layer_bytes[r.clone()].iter().sum::<u64>();
        if r.start == 0 {
            b += self.embed_bytes;
        }
        if r.end == self.num_layers() {
            b += self.head_bytes;
        }
        b
    }
}

/// A plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    /// One range per node, in node order, tiling `0..num_layers`.
    pub ranges: Vec<Range<usize>>,
    /// Planned bytes per node.
    pub bytes: Vec<u64>,
    /// Estimated seconds per token per node (work / speed).
    pub times: Vec<f64>,
}

impl Assignment {
    /// The slowest stage's time per token.
    pub fn bottleneck(&self) -> f64 {
        self.times.iter().copied().fold(0.0, f64::max)
    }
}

/// Why no plan exists.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PlanError {
    /// No nodes, or more nodes than layers.
    #[error("{nodes} nodes for {layers} layers")]
    NodeCount {
        /// Nodes given.
        nodes: usize,
        /// Layers in the model.
        layers: usize,
    },
    /// A node speed that isn't a positive finite number.
    #[error("node {0} has a non-positive speed")]
    BadSpeed(usize),
    /// The layers don't fit the nodes' caps under any split.
    #[error("no split fits the caps (smallest node needs {min_bytes} bytes for one layer)")]
    DoesNotFit {
        /// Bytes for the cheapest single-layer range.
        min_bytes: u64,
    },
}

/// Plan ranges for `nodes` (in pipeline order).
pub fn plan(costs: &LayerCosts, nodes: &[NodeSpec]) -> Result<Assignment, PlanError> {
    let l = costs.num_layers();
    let n = nodes.len();
    if n == 0 || n > l {
        return Err(PlanError::NodeCount { nodes: n, layers: l });
    }
    if let Some(i) = nodes.iter().position(|s| !(s.speed.is_finite() && s.speed > 0.0)) {
        return Err(PlanError::BadSpeed(i));
    }
    let mut work = vec![0.0f64; l + 1];
    for i in 0..l {
        work[i + 1] = work[i] + costs.layer_work.get(i).copied().unwrap_or(1.0);
    }
    let time = |k: usize, r: &Range<usize>| (work[r.end] - work[r.start]) / nodes[k].speed;
    // best[k][j]: (bottleneck, total) for layers 0..j on nodes 0..k.
    const INF: (f64, f64) = (f64::INFINITY, f64::INFINITY);
    let mut best = vec![vec![INF; l + 1]; n + 1];
    let mut from = vec![vec![0usize; l + 1]; n + 1];
    best[0][0] = (0.0, 0.0);
    for k in 1..=n {
        // Leave at least one layer for each later node.
        for j in k..=l - (n - k) {
            for i in (k - 1)..j {
                let prev = best[k - 1][i];
                if !prev.0.is_finite() {
                    continue;
                }
                let r = i..j;
                if costs.range_bytes(&r) > nodes[k - 1].mem_cap {
                    continue;
                }
                let t = time(k - 1, &r);
                let cand = (prev.0.max(t), prev.1 + t);
                if cand < best[k][j] {
                    best[k][j] = cand;
                    from[k][j] = i;
                }
            }
        }
    }
    if !best[n][l].0.is_finite() {
        let min_bytes = (0..l).map(|i| costs.range_bytes(&(i..i + 1))).min().unwrap_or(0);
        return Err(PlanError::DoesNotFit { min_bytes });
    }
    let mut ranges = vec![0..0; n];
    let mut j = l;
    for k in (1..=n).rev() {
        let i = from[k][j];
        ranges[k - 1] = i..j;
        j = i;
    }
    let bytes = ranges.iter().map(|r| costs.range_bytes(r)).collect();
    let times = ranges.iter().enumerate().map(|(k, r)| time(k, r)).collect();
    Ok(Assignment { ranges, bytes, times })
}

/// A node's speed in layers per second from a measured per-token
/// latency over `layers` layers (e.g. a coordinator's
/// [`crate::coordinator::NodeStats::decode_latency`]).
pub fn speed_from_latency(layers: usize, latency: std::time::Duration) -> f64 {
    layers as f64 / latency.as_secs_f64().max(1e-9)
}

/// [`LayerCosts`] for this build's model variant, from the weights
/// manifest and the budget formulas a capped open uses: per-layer
/// tensors plus KV / linear-attn state for `max_requests` sequences at
/// `n_ctx`; fixed costs are the shared tensors, the expert slot ring
/// and prefill staging (pread, as under a cap), the prefill scratch at
/// the capped default chunk, `runtime_overhead` and
/// `net_bytes`. The hot tier is not counted — it takes what is left.
#[cfg(target_os = "macos")]
pub fn costs_from_manifest(
    manifest: &std::path::Path,
    n_ctx: usize,
    max_requests: usize,
    runtime_overhead: u64,
    net_bytes: u64,
) -> Result<LayerCosts, String> {
    use streammoe::riir::VARIANT;
    use streammoe::riir::budget::{BudgetPlan, kv_bytes_per_token_in, linear_state_bytes_in};

    let text = std::fs::read_to_string(manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("manifest: {e}"))?;
    let tensors = json
        .get("tensors")
        .and_then(|t| t.as_object())
        .ok_or("manifest has no `tensors` object")?;
    let l = VARIANT.num_layers;
    let mut c = LayerCosts::uniform(l, 0, 0);
    let mut shared = 0u64;
    for (name, t) in tensors {
        let size = t.get("size").and_then(|s| s.as_u64()).ok_or_else(|| format!("{name}: no size"))?;
        if let Some(rest) = name.strip_prefix("model.layers.") {
            let layer = rest.split('.').next().and_then(|s| s.parse::<usize>().ok());
            match layer {
                Some(i) if i < l => c.layer_bytes[i] += size,
                _ => return Err(format!("{name}: layer out of range")),
            }
        } else if name.starts_with("model.embed_tokens.") {
            c.embed_bytes += size;
        } else if name.starts_with("model.norm.") || name.starts_with("lm_head.") {
            c.head_bytes += size;
        } else {
            shared += size;
        }
    }
    let reqs = max_requests.max(1) as u64;
    for i in 0..l {
        let r = i..i + 1;
        c.layer_bytes[i] += reqs * (kv_bytes_per_token_in(VARIANT, &r) * n_ctx as u64 + linear_state_bytes_in(VARIANT, &r));
    }
    let chunk = streammoe::riir::config::default_prefill_chunk(n_ctx, true);
    let base = BudgetPlan::for_range(n_ctx, streammoe::ExpertIoMode::Pread, 0, runtime_overhead, &(0..0), 1, chunk);
    c.fixed_bytes = shared + base.total() + net_bytes;
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn node(cap: u64, speed: f64) -> NodeSpec {
        NodeSpec { mem_cap: cap, speed }
    }

    fn tiles(a: &Assignment, l: usize) {
        let mut next = 0;
        for r in &a.ranges {
            assert_eq!(r.start, next);
            assert!(r.end > r.start);
            next = r.end;
        }
        assert_eq!(next, l);
    }

    #[test]
    fn equal_nodes_split_evenly() {
        let c = LayerCosts::uniform(40, GB / 4, GB);
        let a = plan(&c, &[node(100 * GB, 1.0), node(100 * GB, 1.0)]).unwrap();
        assert_eq!(a.ranges, vec![0..20, 20..40]);
        tiles(&a, 40);
    }

    /// Plan phase 5: ranges follow measured speed — a node twice as
    /// fast gets twice the layers.
    #[test]
    fn faster_node_gets_more_layers() {
        let c = LayerCosts::uniform(40, GB / 4, GB);
        let a = plan(&c, &[node(100 * GB, 2.0), node(100 * GB, 1.0), node(100 * GB, 1.0)]).unwrap();
        assert_eq!(a.ranges, vec![0..20, 20..30, 30..40]);
        assert!((a.bottleneck() - 10.0).abs() < 1e-9);
    }

    /// A small cap limits a node regardless of its speed.
    #[test]
    fn cap_limits_a_fast_node() {
        let c = LayerCosts::uniform(40, GB / 4, GB);
        // 1 GiB fixed + 8 layers × 0.25 GiB = 3 GiB.
        let a = plan(&c, &[node(3 * GB, 10.0), node(100 * GB, 1.0)]).unwrap();
        assert_eq!(a.ranges, vec![0..8, 8..40]);
        assert!(a.bytes[0] <= 3 * GB);
        tiles(&a, 40);
    }

    #[test]
    fn embed_and_head_charged_to_the_ends() {
        let mut c = LayerCosts::uniform(4, 10, 0);
        c.embed_bytes = 100;
        c.head_bytes = 1000;
        assert_eq!(c.range_bytes(&(0..2)), 120);
        assert_eq!(c.range_bytes(&(2..4)), 1020);
        assert_eq!(c.range_bytes(&(1..3)), 20);
        // The head's node can only take one layer.
        let a = plan(&c, &[node(u64::MAX, 1.0), node(1010, 1.0)]).unwrap();
        assert_eq!(a.ranges, vec![0..3, 3..4]);
    }

    #[test]
    fn infeasible_and_bad_inputs_refused() {
        let c = LayerCosts::uniform(10, GB, 0);
        assert!(matches!(
            plan(&c, &[node(4 * GB, 1.0), node(4 * GB, 1.0)]),
            Err(PlanError::DoesNotFit { min_bytes }) if min_bytes == GB
        ));
        assert!(matches!(plan(&c, &[]), Err(PlanError::NodeCount { .. })));
        let many = vec![node(u64::MAX, 1.0); 11];
        assert!(matches!(plan(&c, &many), Err(PlanError::NodeCount { .. })));
        assert_eq!(plan(&c, &[node(u64::MAX, 0.0)]), Err(PlanError::BadSpeed(0)));
        assert_eq!(plan(&c, &[node(u64::MAX, f64::NAN)]), Err(PlanError::BadSpeed(0)));
    }

    #[test]
    fn speed_from_latency_is_layers_per_second() {
        let s = speed_from_latency(20, std::time::Duration::from_millis(40));
        assert!((s - 500.0).abs() < 1e-6);
    }
}
