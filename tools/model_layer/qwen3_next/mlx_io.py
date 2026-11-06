#!/usr/bin/env python3
"""Safetensors I/O, BF16 helpers and MLX affine quantization in NumPy.

A copy of tools/model_layer/qwen3_8_flash_next/mlx_io.py (one folder per
model family), used by the Qwen3-Next tools (reference loader,
tiny-model maker, converter). Depends on NumPy only, so the converter
can run under `python3 -I` without torch.

MLX affine quantization (`mx.quantize(w, group_size, bits)`, mode
"affine"), as the Metal kernel `affine_quantize` computes it
(crates/streammoe-metal/shaders/quantized.h, vendored from MLX):

    per group of `group_size` consecutive weights along the last axis,
    all in float32:
      w_min = min(w)              w_max = max(0, max(w))   # max starts at 0
      scale = max((w_max - w_min) / n_bins, 1e-7)          # n_bins = 2^bits - 1
      side  = |w_min| > |w_max|
      scale = side ? scale : -scale                        # scales may be negative
      edge  = side ? w_min : w_max
      q0    = round(edge / scale)                          # round half away from 0
      scale = q0 == 0 ? scale : edge / q0                  # the edge is exact
      bias  = q0 == 0 ? 0 : edge
      q     = min(round((w - bias) / scale), n_bins)       # float scale, not bf16
      stored scale / bias: cast to the weight dtype (BF16, round to nearest even)

    dequantize: w = q * scale + bias  (stored BF16 scale / bias)

Codes are packed low bits first into little-endian u32 words: 8 codes
per word at 4 bits, 4 at 8 bits. The weight tensor's shape is
`[..., in / (32 / bits)]` (U32); scales and biases `[..., in / group]`.

`tools/requant_experts.py` uses the older "min/max" form (bias = min,
scale = (max - min) / n_bins, codes from the BF16-rounded scale): that
is not what MLX writes, so it is not used here. `check_against_mlx()`
compares this implementation with a real MLX install when one exists.
"""

import json
import struct

import numpy as np

# safetensors dtype names <-> NumPy. BF16 is carried as uint16.
ST_TO_NP = {
    "F32": np.float32, "F16": np.float16, "BF16": np.uint16,
    "U32": np.uint32, "I64": np.int64, "I32": np.int32, "U8": np.uint8,
    "F64": np.float64,
}


# --- BF16 -----------------------------------------------------------------

def bf16_to_f32(u16):
    """uint16 BF16 bits -> float32 (exact)."""
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f):
    """float32 -> uint16 BF16 bits, round to nearest even (MLX / torch)."""
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32).astype(np.uint64)
    nan = np.isnan(np.asarray(f, dtype=np.float32))
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    out = u.astype(np.uint16)
    if nan.any():
        out = np.where(nan, np.uint16(0x7FC0), out)
    return out


def round_bf16(f):
    """float32 -> float32 holding the nearest BF16 value."""
    return bf16_to_f32(f32_to_bf16(f))


# --- safetensors ------------------------------------------------------------

def read_header(path):
    """(data_base_offset, header dict without __metadata__, metadata)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None)
    return 8 + n, header, meta


class SafeFile:
    """Lazy, memory-mapped safetensors reader."""

    def __init__(self, path):
        self.path = str(path)
        self.base, self.header, self.metadata = read_header(path)
        self._mm = None

    @property
    def mm(self):
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self._mm

    def names(self):
        return list(self.header.keys())

    def dtype(self, name):
        return self.header[name]["dtype"]

    def shape(self, name):
        return list(self.header[name]["shape"])

    def span(self, name):
        """(absolute file offset, byte length) of a tensor."""
        a, b = self.header[name]["data_offsets"]
        return self.base + a, b - a

    def raw(self, name):
        """Read-only uint8 view of a tensor's bytes (memory-mapped)."""
        off, n = self.span(name)
        return self.mm[off:off + n]

    def array(self, name):
        """The tensor as a NumPy array (BF16 as uint16), copied."""
        dt = ST_TO_NP[self.dtype(name)]
        return np.frombuffer(bytes(self.raw(name)), dtype=dt).reshape(self.shape(name))

    def close(self):
        if self._mm is not None:
            del self._mm
            self._mm = None


def write_safetensors(path, tensors, metadata=None):
    """tensors: {name: (st_dtype, np.ndarray)} or {name: np.ndarray}
    (float32 / uint32 / int64 / uint8 inferred). Names are written in
    sorted order, 8-byte aligned like the reference implementation."""
    np_to_st = {np.dtype(np.float32): "F32", np.dtype(np.uint32): "U32",
                np.dtype(np.int64): "I64", np.dtype(np.int32): "I32",
                np.dtype(np.uint8): "U8", np.dtype(np.float16): "F16"}
    items = []
    for name in sorted(tensors):
        v = tensors[name]
        if isinstance(v, tuple):
            dt, arr = v
        else:
            arr = v
            dt = np_to_st[np.asarray(arr).dtype]
        arr = np.ascontiguousarray(arr, dtype=ST_TO_NP[dt])
        items.append((name, dt, arr))
    header, off = {}, 0
    for name, dt, arr in items:
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + arr.nbytes]}
        off += arr.nbytes
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, _, arr in items:
            f.write(arr.tobytes())


# --- MLX affine quantization --------------------------------------------------

BF16_MAX = float(bf16_to_f32(np.array([0x7F7F], np.uint16))[0])


def _round_away(x):
    """Metal `round`: halfway cases away from zero."""
    return np.where(x >= 0, np.floor(x + np.float32(0.5)), np.ceil(x - np.float32(0.5))).astype(np.float32)


def affine_quantize(w, group_size, bits):
    """MLX affine quantization of a float array whose values are
    exactly representable in BF16 (the model's dtype).

    Returns (packed u32 [..., in*bits/32], scales u16-BF16 [..., in/g],
    biases u16-BF16 [..., in/g])."""
    w = np.asarray(w, dtype=np.float32)
    *lead, n_in = w.shape
    assert n_in % group_size == 0, f"last dim {n_in} not a multiple of group {group_size}"
    assert bits in (2, 4, 8), "only power-of-two widths are packed here"
    g = w.reshape(-1, group_size)
    n_bins = np.float32((1 << bits) - 1)
    eps = np.float32(1e-7)
    w_min = np.minimum(g.min(axis=1), np.float32(BF16_MAX))
    w_max = np.maximum(g.max(axis=1), np.float32(0.0))
    scale = np.maximum((w_max - w_min) / n_bins, eps).astype(np.float32)
    side = np.abs(w_min) > np.abs(w_max)
    scale = np.where(side, scale, -scale).astype(np.float32)
    edge = np.where(side, w_min, w_max).astype(np.float32)
    q0 = _round_away(edge / scale)
    at_zero = q0 == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(at_zero, scale, edge / np.where(at_zero, 1, q0)).astype(np.float32)
    bias = np.where(at_zero, np.float32(0.0), edge).astype(np.float32)
    q = _round_away((g - bias[:, None]) / scale[:, None])
    q = np.clip(q, 0, n_bins).astype(np.uint32)
    per = 32 // bits
    q = q.reshape(-1, per)
    shifts = (np.arange(per, dtype=np.uint32) * bits)
    packed = np.bitwise_or.reduce(q << shifts, axis=1).astype(np.uint32)
    packed = packed.reshape(*lead, n_in * bits // 32)
    scales = f32_to_bf16(scale).reshape(*lead, n_in // group_size)
    biases = f32_to_bf16(bias).reshape(*lead, n_in // group_size)
    return packed, scales, biases


def unpack_codes(packed, bits):
    """u32 [..., k] -> uint8 codes [..., k * 32 / bits]."""
    packed = np.asarray(packed, dtype=np.uint32)
    per = 32 // bits
    mask = np.uint32((1 << bits) - 1)
    shifts = np.arange(per, dtype=np.uint32) * bits
    codes = (packed[..., None] >> shifts) & mask
    return codes.reshape(*packed.shape[:-1], packed.shape[-1] * per).astype(np.uint8)


def affine_dequantize(packed, scales, biases, group_size, bits):
    """w = q * scale + bias in float32 (scales / biases as BF16 u16 or f32)."""
    codes = unpack_codes(packed, bits).astype(np.float32)
    s = bf16_to_f32(scales) if np.asarray(scales).dtype == np.uint16 else np.asarray(scales, np.float32)
    b = bf16_to_f32(biases) if np.asarray(biases).dtype == np.uint16 else np.asarray(biases, np.float32)
    *lead, n_in = codes.shape
    c = codes.reshape(*lead, n_in // group_size, group_size)
    return (c * s[..., None] + b[..., None]).reshape(*lead, n_in).astype(np.float32)


def check_against_mlx(trials=6, seed=0):
    """Compare affine_quantize with mx.quantize (needs an MLX install,
    e.g. ~/models/mlx-venv/bin/python). Returns a report string."""
    import mlx.core as mx
    rng = np.random.RandomState(seed)
    lines = []
    for t in range(trials):
        for bits, gs in ((4, 32), (8, 64), (4, 64)):
            shape = (37, 320) if t % 2 else (5, 64, 128)
            scale = [0.02, 1.0, 50.0][t % 3]
            w = round_bf16(rng.randn(*shape).astype(np.float32) * scale)
            if t == 3:
                w[..., :gs] = -np.abs(w[..., :gs])   # all-negative groups
            if t == 4:
                w[..., :gs] = 0.0                    # all-zero groups
            q, s, b = affine_quantize(w, gs, bits)
            mw = mx.array(w).astype(mx.bfloat16)
            mq, ms, mb = mx.quantize(mw, group_size=gs, bits=bits)
            mq = np.array(mq)
            ms = f32_to_bf16(np.array(ms.astype(mx.float32)))
            mb = f32_to_bf16(np.array(mb.astype(mx.float32)))
            nq = int((unpack_codes(q, bits) != unpack_codes(mq, bits)).sum())
            ns = int((s != ms).sum())
            nb = int((b != mb).sum())
            lines.append(f"trial {t} bits {bits} g {gs} shape {shape}: code mismatches {nq}/{q.size * 32 // bits}, "
                         f"scale {ns}/{s.size}, bias {nb}/{b.size}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(check_against_mlx())
