//! Wire format.
//!
//! Every frame is a 32-byte header followed by `len` payload bytes,
//! all little-endian:
//!
//! | offset | size | field |
//! | --- | --- | --- |
//! | 0 | 4 | magic `MFNT` |
//! | 4 | 2 | protocol version ([`VERSION`]) |
//! | 6 | 2 | [`Kind`] |
//! | 8 | 8 | request id |
//! | 16 | 8 | position of the first row |
//! | 24 | 4 | row count (tokens / hidden rows) |
//! | 28 | 4 | payload length in bytes |
//!
//! Payloads are 4-byte words: `i32` token ids, `f32` hidden rows
//! (`count × hidden_dim`), `f32` logits (`vocab`), a [`NodeInfo`], or a
//! UTF-8 error message. Unlike the header, payload words are copied in
//! host byte order — little-endian on every Apple Silicon node, which
//! is the only target; a big-endian peer would need a swap here.
//!
//! A [`FrameBuf`] holds one frame. It is allocated (and touched) once
//! at its maximum size; reading a frame whose payload would not fit
//! fails with [`NetError::TooLarge`] instead of growing. Nothing is
//! allocated per frame.

use std::io::{Read, Write};

/// `MFNT`.
pub const MAGIC: [u8; 4] = *b"MFNT";
/// Protocol version. A peer with another version is refused.
pub const VERSION: u16 = 1;
/// Header bytes.
pub const HEADER_BYTES: usize = 32;

/// Frame kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Kind {
    /// Node → coordinator on connect; payload is a [`NodeInfo`].
    Hello = 1,
    /// Token ids to embed and run (first stage): `count` `i32`s.
    Tokens = 2,
    /// Hidden rows in or out of a stage: `count × hidden_dim` `f32`s.
    Hidden = 3,
    /// Last stage's logits for the last row: `vocab` `f32`s.
    Logits = 4,
    /// Admit the frame's request id.
    Admit = 5,
    /// Drop the frame's request id.
    Drop = 6,
    /// Success reply with no payload.
    Ack = 7,
    /// Failure reply; payload is a UTF-8 message.
    Error = 8,
    /// Stop the node process (replied to with [`Kind::Ack`]).
    Shutdown = 9,
}

impl Kind {
    fn from_u16(v: u16) -> Option<Self> {
        Some(match v {
            1 => Kind::Hello,
            2 => Kind::Tokens,
            3 => Kind::Hidden,
            4 => Kind::Logits,
            5 => Kind::Admit,
            6 => Kind::Drop,
            7 => Kind::Ack,
            8 => Kind::Error,
            9 => Kind::Shutdown,
            _ => return None,
        })
    }
}

/// Network errors.
#[derive(Debug, thiserror::Error)]
pub enum NetError {
    /// Socket I/O.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Not a streammoe-net peer.
    #[error("bad frame magic {0:02x?}")]
    BadMagic([u8; 4]),
    /// Protocol version mismatch.
    #[error("protocol version {got}, expected {VERSION}")]
    Version {
        /// The peer's version.
        got: u16,
    },
    /// Unknown frame kind.
    #[error("unknown frame kind {0}")]
    BadKind(u16),
    /// Payload larger than this side's fixed buffer.
    #[error("frame payload {len} bytes exceeds buffer {cap}")]
    TooLarge {
        /// Payload bytes requested.
        len: usize,
        /// Buffer capacity.
        cap: usize,
    },
    /// The peer replied with an error frame.
    #[error("remote: {0}")]
    Remote(String),
    /// A frame that doesn't fit the protocol state.
    #[error("protocol: {0}")]
    Protocol(String),
}

/// Decoded header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Frame kind.
    pub kind: Kind,
    /// Request id.
    pub req: u64,
    /// Position of the first row.
    pub pos: u64,
    /// Row count.
    pub count: u32,
    /// Payload bytes.
    pub len: u32,
}

impl Header {
    fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut b = [0u8; HEADER_BYTES];
        b[0..4].copy_from_slice(&MAGIC);
        b[4..6].copy_from_slice(&VERSION.to_le_bytes());
        b[6..8].copy_from_slice(&(self.kind as u16).to_le_bytes());
        b[8..16].copy_from_slice(&self.req.to_le_bytes());
        b[16..24].copy_from_slice(&self.pos.to_le_bytes());
        b[24..28].copy_from_slice(&self.count.to_le_bytes());
        b[28..32].copy_from_slice(&self.len.to_le_bytes());
        b
    }

    fn decode(b: &[u8; HEADER_BYTES]) -> Result<Self, NetError> {
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let magic: [u8; 4] = b[0..4].try_into().unwrap();
        if magic != MAGIC {
            return Err(NetError::BadMagic(magic));
        }
        let version = u16_at(4);
        if version != VERSION {
            return Err(NetError::Version { got: version });
        }
        let kind = Kind::from_u16(u16_at(6)).ok_or(NetError::BadKind(u16_at(6)))?;
        Ok(Self {
            kind,
            req: u64_at(8),
            pos: u64_at(16),
            count: u32_at(24),
            len: u32_at(28),
        })
    }
}

/// What a node serves, sent in its [`Kind::Hello`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeInfo {
    /// First layer of the node's range.
    pub layer_start: u32,
    /// One past the last layer.
    pub layer_end: u32,
    /// Layers in the whole model.
    pub num_layers: u32,
    /// Hidden size (f32s per row).
    pub hidden_dim: u32,
    /// Vocabulary size (f32s per logits frame).
    pub vocab: u32,
    /// Request pool size.
    pub max_requests: u32,
    /// Most rows the node accepts in one frame.
    pub max_tokens: u32,
    /// The node's memory cap (0 = none).
    pub mem_cap: u64,
    /// Bytes of the node's frame buffers (inside the cap).
    pub net_bytes: u64,
}

const NODE_INFO_WORDS: usize = 12;

impl NodeInfo {
    fn to_words(self) -> [u32; NODE_INFO_WORDS] {
        let lo = |v: u64| v as u32;
        let hi = |v: u64| (v >> 32) as u32;
        [
            self.layer_start,
            self.layer_end,
            self.num_layers,
            self.hidden_dim,
            self.vocab,
            self.max_requests,
            self.max_tokens,
            0,
            lo(self.mem_cap),
            hi(self.mem_cap),
            lo(self.net_bytes),
            hi(self.net_bytes),
        ]
    }

    fn from_words(w: &[u32]) -> Option<Self> {
        if w.len() != NODE_INFO_WORDS {
            return None;
        }
        let u64_of = |lo: u32, hi: u32| lo as u64 | (hi as u64) << 32;
        Some(Self {
            layer_start: w[0],
            layer_end: w[1],
            num_layers: w[2],
            hidden_dim: w[3],
            vocab: w[4],
            max_requests: w[5],
            max_tokens: w[6],
            mem_cap: u64_of(w[8], w[9]),
            net_bytes: u64_of(w[10], w[11]),
        })
    }

    /// `true` if this node runs the last layer (returns logits).
    pub fn has_head(&self) -> bool {
        self.layer_end == self.num_layers
    }
}

/// Largest payload a stage exchanges: `max_tokens` hidden rows in or
/// out, or one logits row.
pub fn max_payload_bytes(max_tokens: usize, hidden_dim: usize, vocab: usize) -> usize {
    4 * (max_tokens * hidden_dim).max(vocab).max(NODE_INFO_WORDS)
}

/// One frame: header plus a fixed-capacity, word-aligned payload.
pub struct FrameBuf {
    header: Header,
    words: Vec<u32>,
}

impl std::fmt::Debug for FrameBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameBuf")
            .field("header", &self.header)
            .field("capacity", &self.capacity())
            .finish()
    }
}

impl FrameBuf {
    /// A buffer for payloads up to `max_payload` bytes (rounded up to
    /// a word), allocated and written now so its pages are resident.
    pub fn new(max_payload: usize) -> Self {
        let n = max_payload.div_ceil(4);
        let mut words = Vec::with_capacity(n);
        // A non-zero fill writes every page (a zeroed alloc may be
        // lazily mapped).
        words.resize(n, u32::MAX);
        Self {
            header: Header { kind: Kind::Ack, req: 0, pos: 0, count: 0, len: 0 },
            words,
        }
    }

    /// Bytes this buffer holds in memory (header + payload capacity).
    pub fn footprint(&self) -> usize {
        HEADER_BYTES + self.capacity()
    }

    /// Payload capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.words.len() * 4
    }

    /// The current frame's header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Payload bytes.
    pub fn bytes(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.words)[..self.header.len as usize]
    }

    /// Payload as `f32`s (a partial trailing word is dropped).
    pub fn f32s(&self) -> &[f32] {
        bytemuck::cast_slice(&self.words[..self.header.len as usize / 4])
    }

    /// Payload as `i32`s.
    pub fn i32s(&self) -> &[i32] {
        bytemuck::cast_slice(&self.words[..self.header.len as usize / 4])
    }

    /// The payload as UTF-8 (lossy), for error frames.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(self.bytes()).into_owned()
    }

    /// The payload as a [`NodeInfo`] (a [`Kind::Hello`] frame).
    pub fn node_info(&self) -> Result<NodeInfo, NetError> {
        if self.header.kind != Kind::Hello {
            return Err(NetError::Protocol(format!("expected Hello, got {:?}", self.header.kind)));
        }
        let w = &self.words[..self.header.len as usize / 4];
        NodeInfo::from_words(w).ok_or_else(|| NetError::Protocol("malformed Hello".into()))
    }

    /// Writable `f32` payload of `n` elements, to fill in place before
    /// [`Self::finish`].
    pub fn f32s_mut(&mut self, n: usize) -> Result<&mut [f32], NetError> {
        self.check_fits(n * 4)?;
        Ok(bytemuck::cast_slice_mut(&mut self.words[..n]))
    }

    /// Set the header for a payload of `len` bytes already in place.
    pub fn finish(&mut self, kind: Kind, req: u64, pos: u64, count: u32, len: usize) -> Result<(), NetError> {
        self.check_fits(len)?;
        self.header = Header { kind, req, pos, count, len: len as u32 };
        Ok(())
    }

    /// A frame with no payload.
    pub fn set_empty(&mut self, kind: Kind, req: u64, pos: u64) {
        self.header = Header { kind, req, pos, count: 0, len: 0 };
    }

    /// A frame carrying `data` as 4-byte words.
    pub fn set_words<T: bytemuck::Pod>(
        &mut self,
        kind: Kind,
        req: u64,
        pos: u64,
        count: u32,
        data: &[T],
    ) -> Result<(), NetError> {
        let src: &[u8] = bytemuck::cast_slice(data);
        self.check_fits(src.len())?;
        let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut self.words);
        dst[..src.len()].copy_from_slice(src);
        self.finish(kind, req, pos, count, src.len())
    }

    /// An error frame; `msg` is truncated to fit.
    pub fn set_error(&mut self, req: u64, msg: &str) {
        let mut n = msg.len().min(self.capacity());
        while !msg.is_char_boundary(n) {
            n -= 1;
        }
        let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut self.words);
        dst[..n].copy_from_slice(&msg.as_bytes()[..n]);
        self.header = Header { kind: Kind::Error, req, pos: 0, count: 0, len: n as u32 };
    }

    /// A [`Kind::Hello`] frame carrying `info`.
    pub fn set_hello(&mut self, info: NodeInfo) -> Result<(), NetError> {
        self.set_words(Kind::Hello, 0, 0, 0, &info.to_words())
    }

    /// Read one frame from `r`, replacing the current one.
    pub fn read_from<R: Read>(&mut self, r: &mut R) -> Result<&Header, NetError> {
        let mut hb = [0u8; HEADER_BYTES];
        r.read_exact(&mut hb)?;
        let h = Header::decode(&hb)?;
        let len = h.len as usize;
        self.check_fits(len)?;
        let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut self.words);
        r.read_exact(&mut dst[..len])?;
        self.header = h;
        Ok(&self.header)
    }

    /// Write the current frame to `w`.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), NetError> {
        w.write_all(&self.header.encode())?;
        w.write_all(self.bytes())?;
        Ok(())
    }

    /// Read a reply and turn an [`Kind::Error`] frame into
    /// [`NetError::Remote`].
    pub fn read_reply<R: Read>(&mut self, r: &mut R) -> Result<&Header, NetError> {
        self.read_from(r)?;
        if self.header.kind == Kind::Error {
            return Err(NetError::Remote(self.text()));
        }
        Ok(&self.header)
    }

    fn check_fits(&self, len: usize) -> Result<(), NetError> {
        if len > self.capacity() {
            return Err(NetError::TooLarge { len, cap: self.capacity() });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn roundtrip(f: &FrameBuf, cap: usize) -> Result<FrameBuf, NetError> {
        let mut wire = Vec::new();
        f.write_to(&mut wire)?;
        assert_eq!(wire.len(), HEADER_BYTES + f.header().len as usize);
        let mut g = FrameBuf::new(cap);
        g.read_from(&mut Cursor::new(wire))?;
        Ok(g)
    }

    #[test]
    fn hidden_and_tokens_roundtrip() {
        let mut f = FrameBuf::new(64);
        let rows = [1.5f32, -0.0, f32::MIN_POSITIVE, 3.25];
        f.set_words(Kind::Hidden, 7, 40, 2, &rows).unwrap();
        let g = roundtrip(&f, 64).unwrap();
        assert_eq!(*g.header(), Header { kind: Kind::Hidden, req: 7, pos: 40, count: 2, len: 16 });
        let same = g.f32s().iter().zip(&rows).all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same);

        f.set_words(Kind::Tokens, u64::MAX, 1 << 40, 3, &[1i32, -2, 3]).unwrap();
        let g = roundtrip(&f, 64).unwrap();
        assert_eq!(g.header().req, u64::MAX);
        assert_eq!(g.header().pos, 1 << 40);
        assert_eq!(g.i32s(), &[1, -2, 3]);
    }

    #[test]
    fn hello_and_error_roundtrip() {
        let info = NodeInfo {
            layer_start: 20,
            layer_end: 40,
            num_layers: 40,
            hidden_dim: 2048,
            vocab: 248320,
            max_requests: 4,
            max_tokens: 512,
            mem_cap: 10 << 30,
            net_bytes: (1 << 33) + 5,
        };
        let mut f = FrameBuf::new(64);
        f.set_hello(info).unwrap();
        let g = roundtrip(&f, 64).unwrap();
        assert_eq!(g.node_info().unwrap(), info);
        assert!(info.has_head());

        f.set_error(3, "pool full ✓");
        let mut wire = Vec::new();
        f.write_to(&mut wire).unwrap();
        let mut g = FrameBuf::new(64);
        match g.read_reply(&mut Cursor::new(wire)) {
            Err(NetError::Remote(m)) => assert_eq!(m, "pool full ✓"),
            other => panic!("expected Remote, got {other:?}"),
        }
        // Long messages are cut at a char boundary to fit.
        let mut small = FrameBuf::new(4);
        small.set_error(0, "ab✓✓");
        assert_eq!(small.text(), "ab");
    }

    #[test]
    fn oversize_payload_refused_without_growing() {
        let mut f = FrameBuf::new(1024);
        f.set_words(Kind::Hidden, 0, 0, 1, &[0f32; 256]).unwrap();
        match roundtrip(&f, 512) {
            Err(NetError::TooLarge { len: 1024, cap: 512 }) => {}
            other => panic!("expected TooLarge, got {other:?}"),
        }
        assert!(f.set_words(Kind::Hidden, 0, 0, 1, &[0f32; 257]).is_err());
        assert_eq!(f.capacity(), 1024);
    }

    #[test]
    fn bad_magic_version_and_kind_refused() {
        let mut f = FrameBuf::new(8);
        f.set_empty(Kind::Ack, 1, 2);
        let mut wire = Vec::new();
        f.write_to(&mut wire).unwrap();
        let mut g = FrameBuf::new(8);

        let mut w = wire.clone();
        w[0] = b'X';
        assert!(matches!(g.read_from(&mut Cursor::new(w)), Err(NetError::BadMagic(_))));
        let mut w = wire.clone();
        w[4] = 2;
        assert!(matches!(g.read_from(&mut Cursor::new(w)), Err(NetError::Version { got: 2 })));
        let mut w = wire.clone();
        w[6] = 99;
        assert!(matches!(g.read_from(&mut Cursor::new(w)), Err(NetError::BadKind(99))));
        // Truncated stream.
        assert!(matches!(g.read_from(&mut Cursor::new(&wire[..10])), Err(NetError::Io(_))));
    }

    #[test]
    fn payload_size_covers_hidden_and_logits() {
        assert_eq!(max_payload_bytes(512, 2048, 1000), 4 * 512 * 2048);
        assert_eq!(max_payload_bytes(1, 2048, 248320), 4 * 248320);
    }
}
