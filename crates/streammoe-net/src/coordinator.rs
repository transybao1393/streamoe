//! The coordinator: drives a chain of nodes.
//!
//! [`Chain::connect`] opens one connection per node, in layer order,
//! and checks that their ranges tile `0..num_layers` with matching
//! shapes. [`Chain::eval`] runs tokens through every stage — chunked to
//! the smallest node's `max_tokens` — and returns the last row's
//! logits. Two fixed frame buffers are reused for every hop (the reply
//! from one node is forwarded to the next without a copy).

use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use crate::frame::{FrameBuf, Kind, NetError, NodeInfo, max_payload_bytes};

/// Per-node timing, measured by the coordinator.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NodeStats {
    /// Eval round trips.
    pub calls: u64,
    /// Rows sent through the node.
    pub rows: u64,
    /// Round-trip time of those calls: transfer both ways plus the
    /// node's compute.
    pub busy: Duration,
    /// Round-trip time of single-row (decode) calls only.
    pub decode_busy: Duration,
    /// Single-row calls.
    pub decode_calls: u64,
}

impl NodeStats {
    /// Mean single-row round trip.
    pub fn decode_latency(&self) -> Option<Duration> {
        (self.decode_calls > 0).then(|| self.decode_busy / self.decode_calls as u32)
    }
}

struct Conn {
    addr: SocketAddr,
    stream: TcpStream,
    info: NodeInfo,
    stats: NodeStats,
}

/// A connected chain of nodes.
pub struct Chain {
    nodes: Vec<Conn>,
    a: FrameBuf,
    b: FrameBuf,
    max_tokens: usize,
}

impl Chain {
    /// Connect to `addrs`, in pipeline order.
    pub fn connect<A: ToSocketAddrs>(addrs: &[A]) -> Result<Self, NetError> {
        let mut hello = FrameBuf::new(256);
        let mut nodes = Vec::with_capacity(addrs.len());
        for a in addrs {
            let mut stream = TcpStream::connect(a)?;
            stream.set_nodelay(true)?;
            let addr = stream.peer_addr()?;
            hello.read_reply(&mut stream)?;
            let info = hello.node_info()?;
            nodes.push(Conn { addr, stream, info, stats: NodeStats::default() });
        }
        let first = nodes.first().ok_or_else(|| NetError::Protocol("no nodes".into()))?.info;
        let mut next = 0;
        for n in &nodes {
            let i = n.info;
            if i.layer_start != next || i.num_layers != first.num_layers || i.hidden_dim != first.hidden_dim
            {
                return Err(NetError::Protocol(format!(
                    "node {} serves {}..{} of {} (hidden {}); expected a range starting at {next} of {} (hidden {})",
                    n.addr, i.layer_start, i.layer_end, i.num_layers, i.hidden_dim, first.num_layers, first.hidden_dim
                )));
            }
            next = i.layer_end;
        }
        if next != first.num_layers {
            return Err(NetError::Protocol(format!("ranges end at {next}, model has {}", first.num_layers)));
        }
        let max_tokens = nodes.iter().map(|n| n.info.max_tokens as usize).min().unwrap_or(1).max(1);
        let last = nodes.last().unwrap().info;
        let max = max_payload_bytes(max_tokens, first.hidden_dim as usize, last.vocab as usize);
        Ok(Self { nodes, a: FrameBuf::new(max), b: FrameBuf::new(max), max_tokens })
    }

    /// The nodes' infos, in order.
    pub fn infos(&self) -> Vec<NodeInfo> {
        self.nodes.iter().map(|n| n.info).collect()
    }

    /// Per-node timing so far, in order.
    pub fn stats(&self) -> Vec<NodeStats> {
        self.nodes.iter().map(|n| n.stats).collect()
    }

    /// Rows per frame (the smallest node limit).
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Vocabulary size (logits per eval).
    pub fn vocab(&self) -> usize {
        self.nodes.last().map_or(0, |n| n.info.vocab as usize)
    }

    /// Admit `req` on every node. If a node refuses, the nodes that
    /// had already admitted it drop it again, so a retry starts clean.
    pub fn admit(&mut self, req: u64) -> Result<(), NetError> {
        for i in 0..self.nodes.len() {
            if let Err(e) = self.control_one(i, Kind::Admit, req) {
                for j in 0..i {
                    let _ = self.control_one(j, Kind::Drop, req);
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// Drop `req` on every node.
    pub fn drop_request(&mut self, req: u64) -> Result<(), NetError> {
        self.control(Kind::Drop, req)
    }

    /// Stop every node process.
    pub fn shutdown(mut self) -> Result<(), NetError> {
        self.control(Kind::Shutdown, 0)
    }

    fn control(&mut self, kind: Kind, req: u64) -> Result<(), NetError> {
        for i in 0..self.nodes.len() {
            self.control_one(i, kind, req)?;
        }
        Ok(())
    }

    fn control_one(&mut self, node: usize, kind: Kind, req: u64) -> Result<(), NetError> {
        let n = &mut self.nodes[node];
        self.a.set_empty(kind, req, 0);
        self.a.write_to(&mut n.stream)?;
        self.b.read_reply(&mut n.stream)?;
        Ok(())
    }

    /// Run `tokens` for request `req` starting at `pos` through the
    /// chain; `logits` receives the last token's logits.
    ///
    /// On failure, nodes before the failing one have already advanced
    /// `req`, so the chain drops `req` everywhere (the default request
    /// is cleared): no node keeps a partial step. Re-admit to retry.
    pub fn eval(&mut self, req: u64, tokens: &[i32], pos: usize, logits: &mut [f32]) -> Result<(), NetError> {
        if logits.len() != self.vocab() {
            return Err(NetError::Protocol(format!("logits buffer {} != vocab {}", logits.len(), self.vocab())));
        }
        if tokens.is_empty() {
            return Err(NetError::Protocol("no tokens to evaluate".into()));
        }
        let r = self.eval_chunks(req, tokens, pos, logits);
        if r.is_err() {
            for i in 0..self.nodes.len() {
                let _ = self.control_one(i, Kind::Drop, req);
            }
        }
        r
    }

    fn eval_chunks(&mut self, req: u64, tokens: &[i32], pos: usize, logits: &mut [f32]) -> Result<(), NetError> {
        for (c, chunk) in tokens.chunks(self.max_tokens).enumerate() {
            let p = (pos + c * self.max_tokens) as u64;
            self.a.set_words(Kind::Tokens, req, p, chunk.len() as u32, chunk)?;
            for n in &mut self.nodes {
                let t0 = Instant::now();
                self.a.write_to(&mut n.stream)?;
                self.b.read_reply(&mut n.stream)?;
                let dt = t0.elapsed();
                n.stats.calls += 1;
                n.stats.rows += chunk.len() as u64;
                n.stats.busy += dt;
                if chunk.len() == 1 {
                    n.stats.decode_calls += 1;
                    n.stats.decode_busy += dt;
                }
                // The reply is the next node's request.
                std::mem::swap(&mut self.a, &mut self.b);
            }
            let h = *self.a.header();
            if h.kind != Kind::Logits || self.a.f32s().len() != logits.len() {
                return Err(NetError::Protocol(format!(
                    "last node replied {:?} with {} f32s, expected {} logits",
                    h.kind,
                    self.a.f32s().len(),
                    logits.len()
                )));
            }
            logits.copy_from_slice(self.a.f32s());
        }
        Ok(())
    }
}
