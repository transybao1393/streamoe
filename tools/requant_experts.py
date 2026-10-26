#!/usr/bin/env python3
"""Requantize packed 4-bit experts to fewer levels, kept in the 4-bit layout.

    python3 tools/requant_experts.py <src_root> --out 3=<dst_root> [--out 2=<dst_root>]

Every expert blob in <src_root>/packed_experts/layer_XX.bin is
[gate | up | down], each block [u32 weights (8 nibbles, low first) |
bf16 scales | bf16 biases], affine groups of 64 along the input dim
(w = q * s + b). For each group this rebuilds the 16-level values,
fits 2^bits evenly spaced levels between the group's min and max
(MLX's affine scheme), and writes the codes back as 4-bit nibbles with
the new scale and bias. The engine runs the result unchanged, so the
output has exactly the rounding of a <bits>-bit model. Same file size,
so no speed change; for 3-bit, `tools/pack_experts.py` then packs the
codes 8 per 3 bytes, which the engine reads directly. Starting from the 4-bit values, not the original weights,
slightly overstates the error of a real <bits>-bit conversion.

Shape constants are Qwen3.6-35B-A3B's (hidden 2048, moe_intermediate 512).
"""
import argparse
import os
import sys
import time

import numpy as np

HIDDEN, INTER, GROUP = 2048, 512, 64
W_BYTES = INTER * HIDDEN // 2          # one matrix's packed weights
S_BYTES = INTER * (HIDDEN // GROUP) * 2  # its bf16 scales (biases the same)
BLOCK = W_BYTES + 2 * S_BYTES
EXPERT = 3 * BLOCK
CHUNK = 8  # experts per pass, bounds RAM to a few hundred MB


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f):
    # round to nearest even
    u = f.astype(np.float32).view(np.uint32)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def requant_block(block, levels):
    """block: uint8 [n, BLOCK] → requantized copy. Groups are 64
    consecutive weights of one row, and rows are contiguous, so the
    flat weight stream splits into groups directly."""
    n = block.shape[0]
    packed = block[:, :W_BYTES]
    scales = bf16_to_f32(block[:, W_BYTES:W_BYTES + S_BYTES].copy().view(np.uint16)).reshape(n, -1, 1)
    biases = bf16_to_f32(block[:, W_BYTES + S_BYTES:].copy().view(np.uint16)).reshape(n, -1, 1)
    q = np.empty((n, W_BYTES * 2), np.uint8)
    q[:, 0::2] = packed & 0xF
    q[:, 1::2] = packed >> 4
    v = q.reshape(n, -1, GROUP).astype(np.float32) * scales + biases
    lo = v.min(axis=2, keepdims=True)
    hi = v.max(axis=2, keepdims=True)
    s_new = f32_to_bf16((hi - lo) / (levels - 1))
    b_new = f32_to_bf16(lo)
    s = bf16_to_f32(s_new)
    b = bf16_to_f32(b_new)
    with np.errstate(divide="ignore", invalid="ignore"):
        qn = np.where(s > 0, np.rint((v - b) / s), 0.0)
    qn = np.clip(qn, 0, levels - 1).astype(np.uint8).reshape(n, -1)
    out = np.empty_like(block)
    out[:, :W_BYTES] = qn[:, 0::2] | (qn[:, 1::2] << 4)
    out[:, W_BYTES:W_BYTES + S_BYTES] = s_new.reshape(n, -1).view(np.uint8)
    out[:, W_BYTES + S_BYTES:] = b_new.reshape(n, -1).view(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_root")
    ap.add_argument("--out", action="append", required=True, help="BITS=DST_ROOT, BITS in 1..3")
    ap.add_argument("--layers", type=int, default=40)
    a = ap.parse_args()
    outs = []
    for o in a.out:
        bits, dst = o.split("=", 1)
        bits = int(bits)
        assert 1 <= bits <= 3, "bits must be 1..3"
        os.makedirs(os.path.join(dst, "packed_experts"), exist_ok=True)
        outs.append((1 << bits, dst))
    for layer in range(a.layers):
        t0 = time.time()
        src = os.path.join(a.src_root, "packed_experts", f"layer_{layer:02d}.bin")
        size = os.path.getsize(src)
        assert size % EXPERT == 0, f"{src}: size {size} not a multiple of {EXPERT}"
        n_exp = size // EXPERT
        files = [open(os.path.join(dst, "packed_experts", f"layer_{layer:02d}.bin"), "wb") for _, dst in outs]
        with open(src, "rb") as f:
            for e0 in range(0, n_exp, CHUNK):
                n = min(CHUNK, n_exp - e0)
                raw = np.frombuffer(f.read(n * EXPERT), np.uint8).reshape(n, 3, BLOCK)
                for (levels, _), fo in zip(outs, files):
                    res = np.empty_like(raw)
                    for m in range(3):
                        res[:, m] = requant_block(raw[:, m], levels)
                    fo.write(res.tobytes())
        for fo in files:
            fo.close()
        print(f"layer {layer:02d}: {n_exp} experts, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
