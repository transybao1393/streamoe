//! A node: one pipeline stage served over TCP.
//!
//! [`serve`] accepts one coordinator connection at a time, sends a
//! [`Kind::Hello`] with the stage's [`NodeInfo`], then answers frames
//! in order:
//!
//! | request | reply |
//! | --- | --- |
//! | `Tokens` / `Hidden` (req, pos, rows) | `Hidden` rows, or `Logits` from the last stage |
//! | `Admit` / `Drop` (req) | `Ack` |
//! | `Shutdown` | `Ack`, then [`serve`] returns |
//!
//! Any failure is answered with an `Error` frame; the connection stays
//! up. Each new connection starts from a clean pool: the stage drops
//! every request ([`Stage::reset`]) before the `Hello`, so nothing a
//! previous (or crashed) coordinator left behind leaks into the next. Both frame buffers are allocated once, before the stage's
//! context is opened, and counted in its cap
//! ([`streammoe::OpenConfig::host_buffers`]).

use std::net::{TcpListener, TcpStream};

use crate::frame::{FrameBuf, Kind, NetError, NodeInfo, max_payload_bytes};

/// Stage input rows.
#[derive(Debug, Clone, Copy)]
pub enum Input<'a> {
    /// Token ids (the first stage).
    Tokens(&'a [i32]),
    /// `rows × hidden_dim` hidden states from the previous stage.
    Hidden(&'a [f32]),
}

/// One pipeline stage, as a node runs it.
pub trait Stage {
    /// What this stage serves.
    fn info(&self) -> NodeInfo;
    /// Admit request `req` (fresh, empty state).
    fn admit(&mut self, req: u64) -> Result<(), String>;
    /// Release request `req`.
    fn drop_request(&mut self, req: u64) -> Result<(), String>;
    /// Drop every request; the default one comes back empty. Called
    /// when a coordinator connects.
    fn reset(&mut self);
    /// Run `input` for request `req` at `pos`. `out` is the output
    /// buffer: `rows × hidden_dim` for a middle stage, `vocab` logits
    /// of the last row for the last stage.
    fn eval(&mut self, req: u64, input: Input<'_>, pos: usize, out: &mut [f32]) -> Result<(), String>;
}

/// The two frame buffers a node needs for `info`: the size to pass as
/// [`streammoe::OpenConfig::host_buffers`] before opening.
pub fn net_buffer_bytes(max_tokens: usize, hidden_dim: usize, vocab: usize) -> u64 {
    2 * (crate::frame::HEADER_BYTES + max_payload_bytes(max_tokens, hidden_dim, vocab)) as u64
}

/// A node's fixed frame buffers.
pub struct NodeBuffers {
    recv: FrameBuf,
    send: FrameBuf,
}

impl NodeBuffers {
    /// Allocate (and touch) both buffers. Do this before opening the
    /// stage's context so a capped open's footprint check sees them.
    pub fn new(max_tokens: usize, hidden_dim: usize, vocab: usize) -> Self {
        let max = max_payload_bytes(max_tokens, hidden_dim, vocab);
        Self { recv: FrameBuf::new(max), send: FrameBuf::new(max) }
    }

    /// Bytes held by both buffers.
    pub fn footprint(&self) -> u64 {
        (self.recv.footprint() + self.send.footprint()) as u64
    }
}

/// Why [`serve`] returned.
#[derive(Debug, PartialEq, Eq)]
pub enum Exit {
    /// A coordinator sent [`Kind::Shutdown`].
    Shutdown,
}

/// Serve `stage` on `listener` until a coordinator sends `Shutdown`.
/// A coordinator disconnecting just ends that session.
pub fn serve(stage: &mut dyn Stage, listener: &TcpListener, bufs: &mut NodeBuffers) -> Result<Exit, NetError> {
    loop {
        let (mut stream, _) = listener.accept()?;
        stream.set_nodelay(true)?;
        match session(stage, &mut stream, bufs) {
            Ok(Some(exit)) => return Ok(exit),
            // Peer went away (or sent garbage): wait for the next one.
            Ok(None) | Err(NetError::Io(_)) => continue,
            Err(e) => {
                eprintln!("[streammoe-node] session ended: {e}");
                continue;
            }
        }
    }
}

/// One coordinator connection.
fn session(stage: &mut dyn Stage, stream: &mut TcpStream, bufs: &mut NodeBuffers) -> Result<Option<Exit>, NetError> {
    stage.reset();
    let info = stage.info();
    bufs.send.set_hello(info)?;
    bufs.send.write_to(stream)?;
    let hd = info.hidden_dim as usize;
    loop {
        let h = match bufs.recv.read_from(stream) {
            Ok(h) => *h,
            Err(NetError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        };
        let reply: Result<(), String> = match h.kind {
            Kind::Tokens | Kind::Hidden => {
                let rows = h.count as usize;
                let out_len = if info.has_head() { info.vocab as usize } else { rows * hd };
                let input = if h.kind == Kind::Tokens {
                    Input::Tokens(bufs.recv.i32s())
                } else {
                    Input::Hidden(bufs.recv.f32s())
                };
                let n_in = match input {
                    Input::Tokens(t) => t.len(),
                    Input::Hidden(x) => x.len() / hd.max(1),
                };
                if rows == 0 {
                    Err("frame carries no rows".to_string())
                } else if n_in != rows || rows as u32 > info.max_tokens {
                    Err(format!("frame of {rows} rows carries {n_in} (max {})", info.max_tokens))
                } else {
                    match bufs.send.f32s_mut(out_len) {
                        Ok(out) => stage.eval(h.req, input, h.pos as usize, out).and_then(|()| {
                            let kind = if info.has_head() { Kind::Logits } else { Kind::Hidden };
                            let count = if info.has_head() { 1 } else { h.count };
                            let pos = if info.has_head() { h.pos + rows as u64 - 1 } else { h.pos };
                            bufs.send.finish(kind, h.req, pos, count, out_len * 4).map_err(|e| e.to_string())
                        }),
                        Err(e) => Err(e.to_string()),
                    }
                }
            }
            Kind::Admit => stage.admit(h.req).map(|()| bufs.send.set_empty(Kind::Ack, h.req, 0)),
            Kind::Drop => stage.drop_request(h.req).map(|()| bufs.send.set_empty(Kind::Ack, h.req, 0)),
            Kind::Shutdown => {
                bufs.send.set_empty(Kind::Ack, h.req, 0);
                bufs.send.write_to(stream)?;
                return Ok(Some(Exit::Shutdown));
            }
            other => Err(format!("unexpected {other:?} frame")),
        };
        if let Err(msg) = reply {
            bufs.send.set_error(h.req, &msg);
        }
        bufs.send.write_to(stream)?;
    }
}

/// A streammoe context as a stage.
#[cfg(target_os = "macos")]
pub struct CtxStage {
    /// The context (opened with the node's layer range).
    pub ctx: streammoe::Ctx,
    /// Frames' row limit.
    pub max_tokens: usize,
    /// Bytes of the node's frame buffers.
    pub net_bytes: u64,
}

#[cfg(target_os = "macos")]
impl Stage for CtxStage {
    fn info(&self) -> NodeInfo {
        let r = self.ctx.layer_range();
        NodeInfo {
            layer_start: r.start as u32,
            layer_end: r.end as u32,
            num_layers: streammoe::riir::VARIANT.num_layers as u32,
            hidden_dim: self.ctx.hidden_dim() as u32,
            vocab: self.ctx.n_vocab() as u32,
            max_requests: self.ctx.max_requests() as u32,
            max_tokens: self.max_tokens as u32,
            mem_cap: self.ctx.mem_cap().unwrap_or(0),
            net_bytes: self.net_bytes,
        }
    }

    fn admit(&mut self, req: u64) -> Result<(), String> {
        let id = streammoe::ReqId(req);
        if id == streammoe::ReqId::DEFAULT {
            // Always present; admitting it means "start fresh".
            self.ctx.drop_request(id);
            return Ok(());
        }
        self.ctx.admit_request(id).map_err(|e| e.to_string())
    }

    fn drop_request(&mut self, req: u64) -> Result<(), String> {
        self.ctx.drop_request(streammoe::ReqId(req));
        Ok(())
    }

    fn reset(&mut self) {
        self.ctx.drop_all_requests();
    }

    fn eval(&mut self, req: u64, input: Input<'_>, pos: usize, out: &mut [f32]) -> Result<(), String> {
        use streammoe::{StageInput, StageOutput};
        let input = match input {
            Input::Tokens(t) => StageInput::Tokens(t),
            Input::Hidden(h) => StageInput::Hidden(h),
        };
        let output = if self.ctx.has_head() { StageOutput::Logits(out) } else { StageOutput::Hidden(out) };
        self.ctx
            .eval_request(streammoe::ReqId(req), input, pos, output)
            .map_err(|e| e.to_string())
    }
}
