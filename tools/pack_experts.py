#!/usr/bin/env python3
"""Pack 3-bit experts stored in the 4-bit layout into the packed 3-bit layout.

    python3 tools/pack_experts.py <src_root> <dst_root>

<src_root> is an experts root whose codes all fit in 3 bits (0..7) but
sit in 4-bit nibbles, e.g. the output of `tools/requant_experts.py
--out 3=...`. Each expert blob is [gate | up | down], each block
[weights | bf16 scales | bf16 biases]. The weights go from 8 nibbles
per u32 (low first) to 8 codes per 3 bytes, a little-endian 24-bit pack
with code i at bits 3i..3i+3 (MLX's 3-bit layout); scales and biases
are copied unchanged. The engine detects the layout from the file size,
and a packed expert computes exactly what the nibble one does, with a
quarter fewer weight bytes to read (1.77 → 1.38 MB per expert on
Qwen3.6-35B-A3B).

Fails if any code is above 7 (the source is not 3-bit). Shape constants
are Qwen3.6-35B-A3B's (hidden 2048, moe_intermediate 512).
"""
import os
import sys
import time

import numpy as np

HIDDEN, INTER, GROUP = 2048, 512, 64
W4 = INTER * HIDDEN // 2               # one matrix's 4-bit weights
W3 = INTER * HIDDEN * 3 // 8           # the same, packed 3-bit
SB = INTER * (HIDDEN // GROUP) * 2     # its bf16 scales (biases the same)
BLOCK4, BLOCK3 = W4 + 2 * SB, W3 + 2 * SB
EXPERT4, EXPERT3 = 3 * BLOCK4, 3 * BLOCK3
CHUNK = 16  # experts per pass


def pack_weights(w4):
    """uint8 [n, W4] nibble weights → uint8 [n, W3] packed 3-bit."""
    words = w4.view("<u4").astype(np.uint32)            # [n, W4/4], 8 codes each
    codes = np.stack([(words >> (4 * i)) & 0xF for i in range(8)], axis=-1)
    if codes.max(initial=0) > 7:
        raise ValueError("a code is above 7: the source is not 3-bit")
    packs = np.zeros(words.shape, np.uint32)
    for i in range(8):
        packs |= codes[..., i] << (3 * i)
    out = np.empty(words.shape + (3,), np.uint8)
    out[..., 0] = packs & 0xFF
    out[..., 1] = (packs >> 8) & 0xFF
    out[..., 2] = (packs >> 16) & 0xFF
    return out.reshape(w4.shape[0], W3)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(dst, "packed_experts"), exist_ok=True)
    layers = sorted(f for f in os.listdir(os.path.join(src, "packed_experts")) if f.endswith(".bin"))
    for name in layers:
        t0 = time.time()
        sp = os.path.join(src, "packed_experts", name)
        size = os.path.getsize(sp)
        assert size % EXPERT4 == 0, f"{sp}: size {size} is not a multiple of {EXPERT4}"
        n_exp = size // EXPERT4
        dp = os.path.join(dst, "packed_experts", name)
        with open(sp, "rb") as fi, open(dp + ".part", "wb") as fo:
            for e0 in range(0, n_exp, CHUNK):
                n = min(CHUNK, n_exp - e0)
                raw = np.frombuffer(fi.read(n * EXPERT4), np.uint8).reshape(n, 3, BLOCK4)
                out = np.empty((n, 3, BLOCK3), np.uint8)
                for m in range(3):
                    out[:, m, :W3] = pack_weights(raw[:, m, :W4])
                    out[:, m, W3:] = raw[:, m, W4:]
                fo.write(out.tobytes())
        os.replace(dp + ".part", dp)
        print(f"{name}: {n_exp} experts, {size / 1e6:.0f} → {n_exp * EXPERT3 / 1e6:.0f} MB, "
              f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
