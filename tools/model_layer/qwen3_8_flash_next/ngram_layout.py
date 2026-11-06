#!/usr/bin/env python3
"""The PLE n-gram table on disk: the MFNG row table (q38-ple's format,
read by `riir/model_layer/qwen3_8_flash_next/ngram.rs`).

Little-endian. 4096-byte header:

    0   magic b"MFNG"            4   u32 version = 1
    8   u32 header_bytes = 4096  12  u32 dim (elements per row; 160)
    16  u32 group_size (32)      20  u32 bits = 4
    24  u32 scale_dtype (1 = bf16)
    28  u32 row_bytes = dim/2 + 4*(dim/group)   (100)
    32  u64 n_rows               40  u32 n_heads (16)
    44  u32 ngram_size (3)
    48  u64[8]  layer multipliers (ngram_size used, rest 0)
    112 u64[64] head vocab sizes (n_heads used)
    624 u64[64] head offsets     (n_heads used)
    rest zero.

Row r at 4096 + r*row_bytes: [packed u32 x dim/8 | scales bf16 x
dim/group | biases bf16 x dim/group], MLX nibble order (element j in
nibble j%8 of word j/8, low nibble first). Rows are the MLX shards
`...ngram_embedding.shards.{0..S-1}` concatenated along dim 0 in shard
order; the file is exactly 4096 + n_rows*row_bytes bytes.

q38-ple's `write_mfng()` (tools/model_layer/qwen3_8_flash_next/
ple_reference.py on its branch) writes the same file from whole
in-memory arrays; the converter cannot hold the real 32 GB table, so
`MfngWriter` writes it shard by shard (and component by component, as
the source shards arrive) into a preallocated file. `mfng_header()`
packs the header field for field like `write_mfng`; the tiny-model
check compares the two writers byte for byte.
"""

import os
import struct

import numpy as np

MFNG_HEADER = 4096
MFNG_MAX_NGRAM = 8
MFNG_MAX_HEADS = 64
COMPONENTS = ("weight", "scales", "biases")


def row_bytes(dim, group, bits=4):
    return dim * bits // 8 + 4 * (dim // group)


def mfng_header(n_rows, dim, group, ngram_size, multipliers, head_sizes, head_offsets):
    assert len(multipliers) <= MFNG_MAX_NGRAM and len(head_sizes) <= MFNG_MAX_HEADS
    hdr = bytearray(MFNG_HEADER)
    struct.pack_into("<4sIIIIIIIQII", hdr, 0, b"MFNG", 1, MFNG_HEADER, dim, group, 4, 1,
                     row_bytes(dim, group), n_rows, len(head_sizes), ngram_size)
    for i, m in enumerate(multipliers):
        struct.pack_into("<Q", hdr, 48 + 8 * i, int(m))
    for i, s in enumerate(head_sizes):
        struct.pack_into("<Q", hdr, 112 + 8 * i, int(s))
    for i, o in enumerate(head_offsets):
        struct.pack_into("<Q", hdr, 624 + 8 * i, int(o))
    return bytes(hdr)


class MfngWriter:
    """Preallocated MFNG file filled one (shard, component) at a time.

    `rows_per_shard` places shard s at row s * rows_per_shard (MLX
    splits the table into equal shards: padded vocab / split parts);
    `put()` asserts each shard's real row count against it, and
    `finish()` asserts every shard was written and the rows sum to
    n_rows."""

    CHUNK_ROWS = 1 << 16

    def __init__(self, path, n_rows, dim, group, n_shards):
        self.path, self.n_rows, self.dim, self.group = path, n_rows, dim, group
        self.n_shards = n_shards
        assert n_rows % n_shards == 0, f"{n_rows} rows not divisible into {n_shards} shards"
        self.rows_per_shard = n_rows // n_shards
        self.rb = row_bytes(dim, group)
        self.widths = {"weight": dim // 2, "scales": 2 * (dim // group), "biases": 2 * (dim // group)}
        self.cols = {"weight": 0, "scales": dim // 2, "biases": dim // 2 + 2 * (dim // group)}
        self.size = MFNG_HEADER + n_rows * self.rb
        if not os.path.exists(path) or os.path.getsize(path) != self.size:
            with open(path, "wb") as f:
                f.truncate(self.size)                     # sparse until written

    def put(self, shard, component, raw, shape):
        """raw: uint8 bytes of one MLX tensor `shards.{shard}.{component}`."""
        rows = shape[0]
        assert rows == self.rows_per_shard, \
            f"n-gram shard {shard} has {rows} rows, expected {self.rows_per_shard}"
        w = self.widths[component]
        src = np.asarray(raw, dtype=np.uint8).reshape(rows, w)
        base = MFNG_HEADER + shard * self.rows_per_shard * self.rb
        c = self.cols[component]
        with open(self.path, "r+b") as f:
            for r0 in range(0, rows, self.CHUNK_ROWS):
                n = min(self.CHUNK_ROWS, rows - r0)
                f.seek(base + r0 * self.rb)
                buf = np.frombuffer(f.read(n * self.rb), dtype=np.uint8).reshape(n, self.rb).copy()
                buf[:, c:c + w] = src[r0:r0 + n]
                f.seek(base + r0 * self.rb)
                f.write(buf.tobytes())

    def read_component(self, shard, component):
        """Read a component of a shard back (for verification)."""
        w, c = self.widths[component], self.cols[component]
        base = MFNG_HEADER + shard * self.rows_per_shard * self.rb
        out = np.empty((self.rows_per_shard, w), np.uint8)
        with open(self.path, "rb") as f:
            for r0 in range(0, self.rows_per_shard, self.CHUNK_ROWS):
                n = min(self.CHUNK_ROWS, self.rows_per_shard - r0)
                f.seek(base + r0 * self.rb)
                buf = np.frombuffer(f.read(n * self.rb), dtype=np.uint8).reshape(n, self.rb)
                out[r0:r0 + n] = buf[:, c:c + w]
        return out

    def finish(self, shards_seen, multipliers, head_sizes, head_offsets, ngram_size):
        """shards_seen: {shard: rows}. Writes the header."""
        assert sorted(shards_seen) == list(range(self.n_shards)), "missing n-gram shards"
        assert sum(shards_seen.values()) == self.n_rows, "n-gram shard rows do not sum to n_rows"
        hdr = mfng_header(self.n_rows, self.dim, self.group, ngram_size, multipliers, head_sizes, head_offsets)
        with open(self.path, "r+b") as f:
            f.write(hdr)
        assert os.path.getsize(self.path) == self.size


def read_mfng(path):
    """(header dict, rows uint8 [n_rows, row_bytes] memmap)."""
    with open(path, "rb") as f:
        h = f.read(MFNG_HEADER)
    magic, ver, hb, dim, group, bits, sdt, rb, n_rows, n_heads, ngram = struct.unpack_from("<4sIIIIIIIQII", h, 0)
    assert magic == b"MFNG" and ver == 1 and hb == MFNG_HEADER
    hdr = {"dim": dim, "group_size": group, "bits": bits, "scale_dtype": sdt, "row_bytes": rb, "n_rows": n_rows,
           "n_heads": n_heads, "ngram_size": ngram,
           "layer_multipliers": list(struct.unpack_from(f"<{ngram}Q", h, 48)),
           "head_vocab_sizes": list(struct.unpack_from(f"<{n_heads}Q", h, 112)),
           "head_offsets": list(struct.unpack_from(f"<{n_heads}Q", h, 624))}
    assert os.path.getsize(path) == MFNG_HEADER + n_rows * rb
    rows = np.memmap(path, dtype=np.uint8, mode="r", offset=MFNG_HEADER, shape=(n_rows, rb))
    return hdr, rows
