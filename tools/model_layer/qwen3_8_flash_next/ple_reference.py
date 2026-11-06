#!/usr/bin/env python3
"""Reference fixtures for the Qwen3.8-Flash-Next per-layer n-gram embedding.

Runs the transformers `qwen4_exp` PLE code (pulled verbatim out of
`modeling_qwen4_exp.py` with `ast`, so the hashing and the layer math
are the reference's, not a port) and writes the fixtures the Rust unit
tests in `riir/model_layer/qwen3_8_flash_next/{ngram,ple}.rs` read:

- `ngram_ids.json`: n-gram row ids for the real config (vocab 248,320,
  20M base, 16 heads) on random token sequences fed in chunks
  (prefill, then decode steps, then another chunk), eos tokens mixed
  in, plus the hashing constants (multipliers, head sizes / offsets).
- `ple_tiny.json` + `ple_tiny.mfng`: a tiny PLE layer (hidden 32,
  4 streams, 4 heads of 32, vocab 64) with MLX-style 4-bit group-32
  weights; the forward output per chunk, f32. `ple_tiny.mfng` is the
  n-gram table in the streamoe row-table format (see `write_mfng`).

Usage: python3 -I ple_reference.py <modeling_qwen4_exp.py> <out_dir>
Needs torch and numpy only (no transformers import).
"""

import ast
import json
import math
import struct
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PICK = [
    "_MASK64", "_SPLITMIX_GAMMA", "_SPLITMIX_M1", "_SPLITMIX_M2", "_PRIME_1",
    "_splitmix64", "_build_layer_multipliers", "_is_prime", "_find_nth_prime_after",
    "apply_mask_to_padding_states", "Qwen4ExpTextRMSNorm",
    "Qwen4ExpTextNGramEmbedding", "Qwen4ExpTextPLELayer",
]


def load_reference(path, nn_ns):
    """exec the PICK definitions of the reference file in a namespace
    whose `nn` is `nn_ns` (lets the full-size table be faked)."""
    src = open(path).read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
            node.decorator_list = []
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in PICK:
            keep.append(node)
    missing = set(PICK) - {getattr(n, "name", None) or n.targets[0].id for n in keep}
    assert not missing, missing
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {
        "torch": torch, "F": F, "nn": nn_ns, "math": math,
        "Cache": object, "Qwen4ExpTextConfig": object,
    }
    exec(compile(mod, path, "exec"), ns)
    return ns


class StubCache:
    """The two conv-state calls the PLE code makes, with transformers'
    semantics: `update_conv_state` returns `cat(previous, new)` (just
    `new` on the first call) and keeps the last `conv_kernel_size`
    columns, zero-padded on the left when shorter."""

    def __init__(self):
        self.states = {}

    def get_conv_state(self, layer_idx, state_idx=0):
        return self.states.get((layer_idx, state_idx))

    def update_conv_state(self, new, layer_idx, state_idx=0, conv_kernel_size=None):
        prev = self.states.get((layer_idx, state_idx))
        full = new if prev is None else torch.cat([prev, new], dim=-1)
        n = full.shape[-1]
        if n >= conv_kernel_size:
            st = full[..., -conv_kernel_size:]
        else:
            st = F.pad(full, (conv_kernel_size - n, 0))
        self.states[(layer_idx, state_idx)] = st.clone()
        return full


class FakeEmbedding(torch.nn.Module):
    """Stands in for the 320M-row table: records the ids it is asked
    for and returns zeros."""

    def __init__(self, num, dim):
        super().__init__()
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.num = num
        self.calls = []

    def forward(self, ids):
        self.calls.append(ids.clone())
        return torch.zeros(*ids.shape, self.dim)


def cfg_ns(**kw):
    return types.SimpleNamespace(**kw)


# ---- bf16 / MLX 4-bit helpers -------------------------------------------

def f32_to_bf16_bits(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def quantize_4bit(w, group):
    """MLX affine 4-bit, `w` [rows, cols]: returns packed u32 [rows,
    cols/8] (element j at nibble j%8 of word j/8, low first), bf16
    scales / biases [rows, cols/group], and the f32 dequantization
    `q * scale + bias` the kernels compute."""
    rows, cols = w.shape
    g = w.reshape(rows, cols // group, group)
    mn, mx = g.min(-1), g.max(-1)
    scale = np.maximum((mx - mn) / 15.0, 1e-4).astype(np.float32)
    sb, bb = f32_to_bf16_bits(scale), f32_to_bf16_bits(mn)
    s, b = bf16_bits_to_f32(sb), bf16_bits_to_f32(bb)
    q = np.clip(np.rint((g - b[..., None]) / s[..., None]), 0, 15).astype(np.uint32)
    deq = (q.astype(np.float32) * s[..., None] + b[..., None]).astype(np.float32)
    q = q.reshape(rows, cols // 8, 8)
    packed = np.zeros((rows, cols // 8), dtype=np.uint32)
    for n in range(8):
        packed |= q[..., n] << (4 * n)
    return packed, sb.reshape(rows, -1), bb.reshape(rows, -1), deq.reshape(rows, cols)


MFNG_HEADER = 4096
MFNG_MAX_NGRAM = 8
MFNG_MAX_HEADS = 64


def write_mfng(path, packed, scales, biases, dim, group, ngram_size,
               multipliers, head_sizes, head_offsets):
    """The streamoe n-gram row table (`ngram.rs`, `NgramTable`).

    Header, 4096 bytes, little-endian:
      0  magic b"MFNG"          4  u32 version = 1
      8  u32 header_bytes=4096  12 u32 dim (elements per row)
      16 u32 group_size         20 u32 bits = 4
      24 u32 scale_dtype (1 = bf16)
      28 u32 row_bytes = dim/2 + 4*(dim/group)
      32 u64 n_rows             40 u32 n_heads
      44 u32 ngram_size
      48   u64[8]  layer multipliers (ngram_size used, rest 0)
      112  u64[64] head vocab sizes  (n_heads used)
      624  u64[64] head offsets      (n_heads used)
      rest zero.
    Rows from byte 4096, row r at 4096 + r*row_bytes:
      [packed u32 x dim/8 | scales bf16 x dim/group | biases bf16 x dim/group]
    """
    n_rows = packed.shape[0]
    n_heads = len(head_sizes)
    row_bytes = dim // 2 + 4 * (dim // group)
    hdr = bytearray(MFNG_HEADER)
    struct.pack_into("<4sIIIIIIIQII", hdr, 0, b"MFNG", 1, MFNG_HEADER, dim, group, 4, 1,
                     row_bytes, n_rows, n_heads, ngram_size)
    for i, m in enumerate(multipliers):
        struct.pack_into("<Q", hdr, 48 + 8 * i, int(m))
    for i, s in enumerate(head_sizes):
        struct.pack_into("<Q", hdr, 112 + 8 * i, int(s))
    for i, o in enumerate(head_offsets):
        struct.pack_into("<Q", hdr, 624 + 8 * i, int(o))
    body = np.concatenate([
        packed.astype("<u4").view(np.uint8).reshape(n_rows, -1),
        scales.astype("<u2").view(np.uint8).reshape(n_rows, -1),
        biases.astype("<u2").view(np.uint8).reshape(n_rows, -1),
    ], axis=1)
    assert body.shape[1] == row_bytes
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(body.tobytes())


def hexb(a):
    return np.ascontiguousarray(a).tobytes().hex()


def f32_bits(a):
    return [int(v) for v in np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).ravel()]


# ---- fixtures -----------------------------------------------------------

def ngram_ids_fixture(ref_path):
    fake_nn = types.SimpleNamespace(Module=torch.nn.Module, Buffer=torch.nn.Buffer,
                                    Embedding=FakeEmbedding)
    ns = load_reference(ref_path, fake_nn)
    eos = 248044
    vocab = 248320
    rng = np.random.default_rng(38)
    cases = []
    for ple_layer_index, schedule, eos_rate in [
        (0, [7, 1, 1, 1, 5, 1, 2], 0.15),
        (0, [1, 1, 1, 3, 1], 0.3),
        (0, [2, 1, 9], 0.0),
        (0, [16], 0.25),
        (1, [4, 1, 3], 0.2),
    ]:
        cfg = cfg_ns(ngram_size=3, heads_per_ngram=8, vocab_size=vocab,
                     ngram_vocab_size_base=20_000_000, seed=1234, eos_token_id=eos,
                     make_ngram_vocab_size_divisible_by=128)
        emb = ns["Qwen4ExpTextNGramEmbedding"](cfg, 2560, 1, ple_layer_index)
        cache = StubCache()
        chunks, ids = [], []
        for n in schedule:
            toks = rng.integers(0, vocab, size=n)
            toks[rng.random(n) < eos_rate] = eos
            if not chunks and eos_rate > 0.2:
                toks[0] = eos
            emb(torch.tensor(toks[None, :], dtype=torch.long), cache)
            got = emb.ngram_embedding.calls[-1][0]
            chunks.append([int(t) for t in toks])
            ids.append([[int(v) for v in row] for row in got])
        # One shot over the whole sequence gives the same ids.
        whole = torch.tensor([sum(chunks, [])], dtype=torch.long)
        emb(whole, None)
        assert emb.ngram_embedding.calls[-1][0].tolist() == sum(ids, [])
        cases.append({
            "ple_layer_index": ple_layer_index,
            "multipliers": [int(v) for v in emb.layer_multipliers],
            "head_vocab_sizes": emb.head_vocab_sizes,
            "head_offsets": emb.head_offsets,
            "total_vocab": emb.total_vocab_size,
            "padded_rows": emb.ngram_embedding.num,
            "chunks": chunks,
            "ids": ids,
        })
    return {"eos": eos, "vocab": vocab, "cases": cases}


def ple_tiny_fixture(ref_path, mfng_path):
    ns = load_reference(ref_path, torch.nn)
    torch.manual_seed(38)
    rng = np.random.default_rng(380)
    hidden, hc, ped, group, eos, vocab = 32, 4, 128, 32, 5, 64
    cfg = cfg_ns(ngram_size=3, heads_per_ngram=2, vocab_size=vocab,
                 ngram_vocab_size_base=97, seed=1234, eos_token_id=eos,
                 make_ngram_vocab_size_divisible_by=128, hidden_size=hidden,
                 hc_count=hc, ple_embed_dim=ped, ple_conv_kernel_size=4, rms_norm_eps=1e-6)
    layer = ns["Qwen4ExpTextPLELayer"](cfg, 1, 0).float()
    emb = layer.ple_embedding
    head_dim = emb.ngram_embedding.weight.shape[1]
    rows = emb.ngram_embedding.weight.shape[0]
    assert head_dim == 32

    tab_p, tab_s, tab_b, tab_d = quantize_4bit(rng.normal(0, 1, (rows, head_dim)).astype(np.float32), group)
    kp, ks, kb, kd = quantize_4bit(rng.normal(0, 0.1, (hc * hidden, ped)).astype(np.float32), group)
    vp, vs, vb, vd = quantize_4bit(rng.normal(0, 0.1, (hidden, ped)).astype(np.float32), group)
    norms = {n: f32_to_bf16_bits(rng.normal(0, 0.2, hc * hidden).astype(np.float32))
             for n in ["norm_key", "norm_query", "norm_conv"]}
    conv = f32_to_bf16_bits(rng.normal(0, 0.3, (hc * hidden, 4)).astype(np.float32))
    with torch.no_grad():
        emb.ngram_embedding.weight.copy_(torch.from_numpy(tab_d))
        layer.key_proj.weight.copy_(torch.from_numpy(kd))
        layer.value_proj.weight.copy_(torch.from_numpy(vd))
        for n, b in norms.items():
            getattr(layer, n).weight.copy_(torch.from_numpy(bf16_bits_to_f32(b)))
        layer.conv1d.weight.copy_(torch.from_numpy(bf16_bits_to_f32(conv)).reshape(hc * hidden, 1, 4))
    write_mfng(mfng_path, tab_p, tab_s, tab_b, head_dim, group, 3,
               [int(v) for v in emb.layer_multipliers], emb.head_vocab_sizes, emb.head_offsets)

    seqs = []
    for schedule in [[5, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2], [1, 1, 4]]:
        cache = StubCache()
        steps = []
        for n in schedule:
            toks = rng.integers(0, vocab, size=n)
            toks[rng.random(n) < 0.2] = eos
            hs = rng.normal(0, 1, (n, hc * hidden)).astype(np.float32)
            with torch.no_grad():
                out = layer(torch.from_numpy(hs)[None], torch.tensor(toks[None], dtype=torch.long), cache)
            steps.append({"tokens": [int(t) for t in toks], "hidden": f32_bits(hs),
                          "out": f32_bits(out[0].numpy())})
        seqs.append(steps)
    return {
        "hidden": hidden, "hc": hc, "ple_embed_dim": ped, "group": group, "eos": eos,
        "vocab": vocab, "ngram_vocab_size_base": 97, "heads_per_ngram": 2,
        "rms_norm_eps": 1e-6,
        "key_w": hexb(kp.astype("<u4")), "key_s": hexb(ks.astype("<u2")), "key_b": hexb(kb.astype("<u2")),
        "value_w": hexb(vp.astype("<u4")), "value_s": hexb(vs.astype("<u2")), "value_b": hexb(vb.astype("<u2")),
        "norm_key": hexb(norms["norm_key"].astype("<u2")),
        "norm_query": hexb(norms["norm_query"].astype("<u2")),
        "norm_conv": hexb(norms["norm_conv"].astype("<u2")),
        "conv": hexb(conv.astype("<u2")),
        "seqs": seqs,
    }


def main():
    ref, out = sys.argv[1], sys.argv[2]
    with open(f"{out}/ngram_ids.json", "w") as f:
        json.dump(ngram_ids_fixture(ref), f)
    with open(f"{out}/ple_tiny.json", "w") as f:
        json.dump(ple_tiny_fixture(ref, f"{out}/ple_tiny.mfng"), f)
    print("wrote", out)


if __name__ == "__main__":
    main()
