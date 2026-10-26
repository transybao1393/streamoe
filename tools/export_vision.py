#!/usr/bin/env python3
"""Export the Qwen3.5 / Qwen3.6 vision tower from an MLX checkpoint.

Writes, next to the other artifacts:

    vision_weights.bin    every `vision_tower.*` tensor as BF16, each at a
                          16 KiB-aligned offset (so the encoder can stream
                          a block's weights with page-aligned F_NOCACHE
                          reads instead of keeping them resident)
    vision_weights.json   {"config": vision_config, "tensors":
                          {name: {offset, size, shape, dtype}}}

Names drop the `vision_tower.` prefix. The patch-embedding kernel stays in
MLX layout `[out, t, h, w, c]`. Reads the safetensors through a memory map
(≈ no RAM).

Usage:
    python -I tools/export_vision.py --model ~/models/qwen3-6-35b-a3b/mlx-4bit \
        --out ~/models/qwen3-6-35b-a3b/artifacts
"""
import argparse
import json
import mmap
import os
import struct

ALIGN = 16384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, out = os.path.expanduser(a.model), os.path.expanduser(a.out)
    cfg = json.load(open(os.path.join(model, "config.json")))
    vcfg = cfg.get("vision_config")
    if not vcfg:
        raise SystemExit("config.json has no vision_config: not a vision model")
    vcfg["image_token_id"] = cfg["image_token_id"]
    vcfg["vision_start_token_id"] = cfg["vision_start_token_id"]
    vcfg["vision_end_token_id"] = cfg["vision_end_token_id"]
    pre = json.load(open(os.path.join(model, "preprocessor_config.json")))
    vcfg["image_mean"], vcfg["image_std"] = pre["image_mean"], pre["image_std"]
    vcfg["min_pixels"] = pre["size"]["shortest_edge"]
    vcfg["max_pixels"] = pre["size"]["longest_edge"]

    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    files = sorted({f for k, f in index.items() if k.startswith("vision_tower.")})
    tensors, off = {}, 0
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "vision_weights.bin"), "wb") as dst:
        for f in files:
            path = os.path.join(model, f)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
                base = 8 + n
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
                for name in sorted(k for k in header if k.startswith("vision_tower.")):
                    e = header[name]
                    if e["dtype"] != "BF16":
                        raise SystemExit(f"{name}: expected BF16, got {e['dtype']}")
                    lo, hi = e["data_offsets"]
                    pad = (-off) % ALIGN
                    dst.write(b"\0" * pad)
                    off += pad
                    dst.write(mm[base + lo:base + hi])
                    tensors[name[len("vision_tower."):]] = {
                        "offset": off, "size": hi - lo, "shape": e["shape"], "dtype": "BF16"}
                    off += hi - lo
                mm.close()
        dst.write(b"\0" * ((-off) % ALIGN))
    json.dump({"config": vcfg, "tensors": tensors},
              open(os.path.join(out, "vision_weights.json"), "w"), indent=1)
    print(f"wrote {len(tensors)} tensors, {off / 2**20:.1f} MiB -> {out}/vision_weights.bin")


if __name__ == "__main__":
    main()
