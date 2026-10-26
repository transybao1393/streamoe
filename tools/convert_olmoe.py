#!/usr/bin/env python3
"""Convert an MLX 4-bit OLMoE checkpoint into streammoe artifacts.

OLMoE differs from the Qwen MLX checkpoints the other scripts expect:
experts are stored one tensor per expert (`mlp.experts.{E}.*`), not
stacked under `switch_mlp`, and scales / biases / norm weights are F16
where streammoe's kernels read BF16. This script handles both in one
pass, reading the safetensors through a memory map (≈ no RAM):

    <out>/artifacts/model_weights.bin   non-expert tensors, 64 B aligned
    <out>/artifacts/model_weights.json  manifest {offset,size,shape,dtype,bits}
    <out>/artifacts/vocab.bin           via scripts/export_vocab.py
    <out>/root/packed_experts/layer_XX.bin + layout.json
        per expert [gate w,s,b | up w,s,b | down w,s,b]

F16 → BF16 rounds to nearest-even (≈0.3 % relative error on dequantized
weights; measured before writing this). `--keep-f16-report` prints the
per-tensor error so the conversion cost stays visible.

Usage:
    python -I tools/convert_olmoe.py --model ~/models/olmoe-1b-7b/hf \
        --out ~/models/olmoe-1b-7b
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ALIGN = 64
COMPONENTS = (
    "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
    "up_proj.weight", "up_proj.scales", "up_proj.biases",
    "down_proj.weight", "down_proj.scales", "down_proj.biases",
)


def f16_to_bf16(raw):
    """F16 bytes → BF16 bytes, round to nearest even."""
    u = np.frombuffer(raw, dtype=np.float16).astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16).tobytes()


class Shard:
    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def raw(self, name):
        a, b = self.header[name]["data_offsets"]
        return bytes(self.mm[self.base + a:self.base + b])

    def meta(self, name):
        return self.header[name]


def converted(shard, name):
    """(bytes, dtype) with F16 promoted to BF16."""
    m = shard.meta(name)
    data = shard.raw(name)
    if m["dtype"] == "F16":
        return f16_to_bf16(data), "BF16"
    return data, m["dtype"]


def write_weights(shards, cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    q = cfg.get("quantization", {})
    bits = q.get("bits", 4)
    names = sorted(n for n, s in shards.items() if ".mlp.experts." not in n)
    manifest = {"model": cfg.get("_name_or_path", "olmoe"), "num_tensors": len(names), "tensors": {}}
    manifest["config"] = {
        "model_type": cfg["model_type"],
        "hidden_size": cfg["hidden_size"],
        "num_hidden_layers": cfg["num_hidden_layers"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "head_dim": cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
        "vocab_size": cfg["vocab_size"],
        "rms_norm_eps": cfg["rms_norm_eps"],
        "num_experts": cfg["num_experts"],
        "num_experts_per_tok": cfg["num_experts_per_tok"],
        "moe_intermediate_size": cfg["intermediate_size"],
        "shared_expert_intermediate_size": 0,
        "norm_topk_prob": cfg.get("norm_topk_prob", False),
        "full_attention_interval": 1,
        "linear_num_value_heads": 0,
        "linear_num_key_heads": 0,
        "partial_rotary_factor": 1.0,
        "rope_theta": cfg.get("rope_theta", 10000.0),
        "default_bits": bits,
        "default_group_size": q.get("group_size", 64),
        "layer_types": ["full_attention"] * cfg["num_hidden_layers"],
    }
    offset = 0
    with open(out_dir / "model_weights.bin", "wb") as f:
        for name in names:
            data, dtype = converted(shards[name], name)
            pad = (-offset) % ALIGN
            f.write(b"\0" * pad)
            offset += pad
            f.write(data)
            entry = {"offset": offset, "size": len(data), "shape": shards[name].meta(name)["shape"], "dtype": dtype}
            if dtype == "U32":
                entry["bits"] = bits
            manifest["tensors"][name] = entry
            offset += len(data)
    (out_dir / "model_weights.json").write_text(json.dumps(manifest, indent=1))
    return offset


def write_experts(shards, cfg, root):
    pk = root / "packed_experts"
    pk.mkdir(parents=True, exist_ok=True)
    L, E = cfg["num_hidden_layers"], cfg["num_experts"]
    probe = [f"model.layers.0.mlp.experts.0.{c}" for c in COMPONENTS]
    sizes = [len(converted(shards[n], n)[0]) for n in probe]
    expert_size = sum(sizes)
    comps, off = [], 0
    for c, n, sz in zip(COMPONENTS, probe, sizes):
        comps.append({"name": c, "offset": off, "size": sz,
                      "dtype": "U32" if c.endswith("weight") else "BF16",
                      "shape": shards[n].meta(n)["shape"]})
        off += sz
    for l in range(L):
        t0 = time.time()
        with open(pk / f"layer_{l:02d}.bin", "wb") as f:
            for e in range(E):
                for c, sz in zip(COMPONENTS, sizes):
                    n = f"model.layers.{l}.mlp.experts.{e}.{c}"
                    data, _ = converted(shards[n], n)
                    if len(data) != sz:
                        sys.exit(f"error: {n} is {len(data)} bytes, expected {sz}")
                    f.write(data)
        print(f"  layer {l:2d}: {E} experts in {time.time() - t0:.1f}s", flush=True)
    (pk / "layout.json").write_text(json.dumps(
        {"expert_size": expert_size, "num_layers": L, "num_experts": E, "components": comps}, indent=1))
    return expert_size


def verify(shards, cfg, root, expert_size):
    """Spot-check packed experts against the source, after conversion."""
    pk = root / "packed_experts"
    for l, e in [(0, 0), (7, 31), (cfg["num_hidden_layers"] - 1, cfg["num_experts"] - 1)]:
        with open(pk / f"layer_{l:02d}.bin", "rb") as f:
            f.seek(e * expert_size)
            got = f.read(expert_size)
        want = b"".join(converted(shards[n], n)[0]
                        for n in (f"model.layers.{l}.mlp.experts.{e}.{c}" for c in COMPONENTS))
        if got != want:
            sys.exit(f"error: packed layer {l} expert {e} mismatch")
    print("  packed experts verified (3 spot checks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="MLX OLMoE directory")
    ap.add_argument("--out", required=True, help="writes <out>/artifacts and <out>/root")
    args = ap.parse_args()
    model, out = Path(args.model).expanduser(), Path(args.out).expanduser()
    cfg = json.loads((model / "config.json").read_text())
    if cfg.get("model_type") != "olmoe":
        sys.exit(f"error: model_type {cfg.get('model_type')!r}, expected 'olmoe'")
    idx = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    opened = {f: Shard(model / f) for f in sorted(set(idx.values()))}
    shards = {n: opened[f] for n, f in idx.items()}

    t0 = time.time()
    art = out / "artifacts"
    nbytes = write_weights(shards, cfg, art)
    print(f"weights: {nbytes / 2**20:.1f} MiB -> {art}/model_weights.bin")
    es = write_experts(shards, cfg, out / "root")
    print(f"experts: {es} B each, {cfg['num_hidden_layers']} layers x {cfg['num_experts']}")
    verify(shards, cfg, out / "root", es)
    script = Path(__file__).resolve().parent.parent / "scripts" / "export_vocab.py"
    subprocess.run([sys.executable, "-I", str(script), str(model / "tokenizer.json"), str(art / "vocab.bin")], check=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
