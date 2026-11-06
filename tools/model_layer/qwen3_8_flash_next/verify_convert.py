#!/usr/bin/env python3
"""Read a converted streamoe directory back and check it.

    python3 verify_convert.py --out <converted dir> --mlx <mlx-4bit dir> [--goldens <dir with goldens.safetensors>]
                              [--mfng-writer <path to q38-ple's ple_reference.py>]

1. Every tensor of the MLX source is rebuilt from the converted files
   (model_weights.bin/json, packed_experts/layer_XX.bin + layout.json,
   ngram.mfng) and compared byte for byte (A_log as BF16 -> F32).
2. The reference (reference.py) is loaded from the converted files
   alone and its logits are compared with the q4 goldens: they must be
   identical (same dequantized weights, same code).
3. With --mfng-writer, q38-ple's whole-table `write_mfng()` is run on
   the source shards and its file compared with ngram.mfng.
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx_io  # noqa: E402
import ngram_layout  # noqa: E402
import reference  # noqa: E402
import schema  # noqa: E402


def streammoe_getter(out_dir, cfg):
    """get(MLX name) -> NumPy array, from the converted files only."""
    art = os.path.join(out_dir, "artifacts")
    man = json.load(open(os.path.join(art, "model_weights.json")))
    wbin = np.memmap(os.path.join(art, "model_weights.bin"), dtype=np.uint8, mode="r")
    lay = json.load(open(os.path.join(out_dir, "root", "packed_experts", "layout.json")))
    comps = {c["name"]: c for c in lay["components"]}
    mfng_path = os.path.join(out_dir, "root", "ngram.mfng")
    hdr, rows = ngram_layout.read_mfng(mfng_path) if os.path.exists(mfng_path) else (None, None)
    tc = schema.text_config(cfg)
    np_dt = {"BF16": np.uint16, "U32": np.uint32, "F32": np.float32, "I64": np.int64}

    def get(name):
        short = name[len(schema.PREFIX):]
        if short in man["tensors"]:
            e = man["tensors"][short]
            a = np.frombuffer(bytes(wbin[e["offset"]:e["offset"] + e["size"]]), np_dt[e["dtype"]])
            return a.reshape(e["shape"])
        if ".switch_mlp." in name:
            layer = int(name.split(".layers.")[1].split(".")[0])
            c = comps[name.split(".switch_mlp.")[1]]
            f = np.memmap(os.path.join(out_dir, "root", "packed_experts", f"layer_{layer:02d}.bin"),
                          dtype=np.uint8, mode="r")
            per = [np.frombuffer(bytes(f[e * lay["expert_stride"] + c["offset"]:
                                         e * lay["expert_stride"] + c["offset"] + c["size"]]), np_dt[c["dtype"]])
                   for e in range(lay["num_experts"])]
            return np.stack(per).reshape([lay["num_experts"]] + c["shape"])
        if ".ngram_embedding.shards." in name:
            shard = int(name.split(".shards.")[1].split(".")[0])
            comp = name.rsplit(".", 1)[1]
            per = hdr["n_rows"] // tc["split_ngram_parts"]
            dim, g = hdr["dim"], hdr["group_size"]
            cols = {"weight": (0, dim // 2, np.uint32), "scales": (dim // 2, dim // 2 + 2 * (dim // g), np.uint16),
                    "biases": (dim // 2 + 2 * (dim // g), hdr["row_bytes"], np.uint16)}[comp]
            block = np.ascontiguousarray(rows[shard * per:(shard + 1) * per, cols[0]:cols[1]])
            return block.view(cols[2]).reshape(per, -1)
        raise KeyError(name)
    return get


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlx", required=True)
    ap.add_argument("--goldens")
    ap.add_argument("--mfng-writer")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    cfg = json.load(open(os.path.join(a.mlx, "config.json")))
    src = reference.mlx_getter(a.mlx)
    got = streammoe_getter(a.out, cfg)
    n = 0
    for t in schema.tensors(cfg):
        for name, dt, shape in schema.stored(cfg, t):
            x, y = src(name), got(name)
            if name.endswith(".linear_attn.A_log") and y.dtype == np.float32:
                x = mlx_io.bf16_to_f32(x)
            assert x.dtype == y.dtype and list(x.shape) == list(y.shape), (name, x.dtype, y.dtype, x.shape, y.shape)
            assert x.tobytes() == y.tobytes(), f"value mismatch: {name}"
            n += 1
    print(f"1. {n} source tensors read back byte-identical from the converted files")

    if a.goldens:
        sd = reference.mlx_to_hf_state_dict(cfg, got)
        model = reference.build_model(cfg, sd)
        g = mlx_io.SafeFile(os.path.join(a.goldens, "goldens.safetensors"))
        worst = 0.0
        i = 0
        while f"q4.p{i}.input_ids" in g.header:
            ids = g.array(f"q4.p{i}.input_ids").tolist()
            toks = g.array(f"q4.p{i}.decode_tokens").tolist()
            logits, _ = model(torch.tensor([ids + toks]))
            want = np.concatenate([g.array(f"q4.p{i}.prefill_logits"), g.array(f"q4.p{i}.decode_logits")])
            worst = max(worst, float(np.abs(logits[0].numpy() - want).max()))
            i += 1
        print(f"2. reference loaded from the converted files: {i} prompts, max |logits - q4 goldens| = {worst:.2e}")
        assert worst < 1e-4, "converted model does not reproduce the goldens"

    if a.mfng_writer:
        spec = importlib.util.spec_from_file_location("ple_reference", a.mfng_writer)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tc = schema.text_config(cfg)
        li = schema.ple_layers(tc)[0]
        pre = f"{schema.PREFIX}model.layers.{li}.ple.ple_embedding."
        parts = range(tc["split_ngram_parts"])
        cat = lambda c: np.concatenate([src(f"{pre}ngram_embedding.shards.{s}.{c}") for s in parts])
        w = cat("weight")
        heads = (tc["ngram_size"] - 1) * tc["heads_per_ngram"]
        dim = tc["ple_embed_dim"] // heads
        path = os.path.join(a.out, "mfng_from_write_mfng.tmp")
        mod.write_mfng(path, w, cat("scales"), cat("biases"), dim, schema.quant_rule(cfg, pre + "ngram_embedding.shards.0")[1],
                       tc["ngram_size"], src(pre + "layer_multipliers"), src(pre + "ngram_heads_vocab_sizes"),
                       src(pre + "ngram_heads_offsets"))
        same = open(path, "rb").read() == open(os.path.join(a.out, "root", "ngram.mfng"), "rb").read()
        os.remove(path)
        print(f"3. ngram.mfng {'byte-identical to' if same else 'DIFFERS from'} q38-ple's write_mfng()")
        assert same


if __name__ == "__main__":
    main()
