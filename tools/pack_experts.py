#!/usr/bin/env python3
"""Pack 3-bit experts stored in the 4-bit layout into the packed 3-bit layout.

    python3 tools/pack_experts.py <src_root> <dst_root> [--only L [L ...]]
        [--expert-align BYTES] [--hidden H --inter I --group G]

<src_root> is an experts root whose codes all fit in 3 bits (0..7) but
sit in 4-bit nibbles, e.g. the output of `tools/requant_experts.py
--out 3=...`. Each expert blob is [gate | up | down], each block
[weights | bf16 scales | bf16 biases]. The weights go from 8 nibbles
per u32 (low first) to 8 codes per 3 bytes, a little-endian 24-bit pack
with code i at bits 3i..3i+3 (MLX's 3-bit layout); scales and biases
are copied unchanged. The engine detects the layout from the file size
(and packed_experts/layout.json when the experts are padded), and a
packed expert computes exactly what the nibble one does, with a
quarter fewer weight bytes to read (1.77 → 1.38 MB per expert on
Qwen3.6-35B-A3B; 2.93 → 2.34 MiB on Qwen3.8-Flash-Next, whose group
32 scales do not shrink).

Shapes: `tools/expert_shapes.py` (args, else the source's layout.json,
else Qwen3.6-35B-A3B's). Output stride: the packed size rounded up to
--expert-align (default: 16384 when the source's layout.json declares a
padded stride, else 0 = unpadded); layout.json is written when the
source has one. Without --only every layer_XX.bin is packed.

Fails if any code is above 7 (the source is not 3-bit).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert_shapes  # noqa: E402

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
    return out.reshape(w4.shape[0], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_root")
    ap.add_argument("dst_root")
    ap.add_argument("--only", type=int, nargs="+", help="just these layers (e.g. one at a time when disk is tight)")
    ap.add_argument("--expert-align", type=int, help="pad each packed expert to a multiple of this many bytes")
    expert_shapes.add_args(ap)
    a = ap.parse_args()
    src, dst = a.src_root, a.dst_root
    sh, lay = expert_shapes.shape_for(src, a)
    align = a.expert_align
    if align is None:
        align = 16384 if sh.stride != sh.expert4 else 0
    stride3 = expert_shapes.round_up(sh.expert3, align)
    print(f"{sh}; packed stride {stride3}", flush=True)
    os.makedirs(os.path.join(dst, "packed_experts"), exist_ok=True)
    if lay is not None:
        expert_shapes.write_layout(dst, lay, sh.expert3, stride3, 3, sh)
    if a.only is not None:
        layers = [f"layer_{l:02d}.bin" for l in a.only]
    else:
        layers = sorted(f for f in os.listdir(os.path.join(src, "packed_experts")) if f.endswith(".bin"))
    for name in layers:
        t0 = time.time()
        sp = os.path.join(src, "packed_experts", name)
        size = os.path.getsize(sp)
        assert size % sh.stride == 0, f"{sp}: size {size} is not a multiple of {sh.stride} ({sh})"
        n_exp = size // sh.stride
        dp = os.path.join(dst, "packed_experts", name)
        with open(sp, "rb") as fi, open(dp + ".part", "wb") as fo:
            for e0 in range(0, n_exp, CHUNK):
                n = min(CHUNK, n_exp - e0)
                raw = np.frombuffer(fi.read(n * sh.stride), np.uint8).reshape(n, sh.stride)
                raw = raw[:, :sh.expert4].reshape(n, 3, sh.block4)
                out = np.zeros((n, stride3), np.uint8)
                blocks = np.empty((n, 3, sh.block3), np.uint8)
                for m in range(3):
                    blocks[:, m, :sh.w3] = pack_weights(np.ascontiguousarray(raw[:, m, :sh.w4]))
                    blocks[:, m, sh.w3:] = raw[:, m, sh.w4:]
                out[:, :sh.expert3] = blocks.reshape(n, -1)
                fo.write(out.tobytes())
        os.replace(dp + ".part", dp)
        print(f"{name}: {n_exp} experts, {size / 1e6:.0f} → {n_exp * stride3 / 1e6:.0f} MB, "
              f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
