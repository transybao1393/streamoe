#!/usr/bin/env python3
"""Convert the MLX 4-bit Qwen3-Next checkpoint into streamoe's layout.

    python3 -I convert.py --source <mlx dir> --out <dir> [--delete-source]

The output is what the engine's qwen3_5_moe path reads (the
`model-qwen3-next-80b-a3b` build): Qwen3-Next's math is that family's,
only two projections are stored differently (schema.py has the full
tensor list). The converter

- splits `linear_attn.in_proj_qkvz` into `in_proj_qkv` (rows
  `[all q | all k | all v]`) and `in_proj_z`, and `in_proj_ba` into
  `in_proj_b` and `in_proj_a`, undoing the per-key-head-group
  interleave transformers' `fix_query_key_value_ordering` reads. Only
  rows move (weight, scales and biases alike); nothing is requantized;
- stores A_log as F32 and every other unquantized tensor as BF16;
- copies everything else as it is, and packs the routed experts into
  root/packed_experts/layer_XX.bin (expert e at e * expert_size:
  [gate w|s|b, up w|s|b, down w|s|b]).

It works one source shard at a time, so the 9 x ~5 GB source never has
to be on disk whole: the output layout is planned from config.json +
model.safetensors.index.json, the output files are preallocated, and
each shard's tensors are written to their planned places, read back and
compared, before the next shard. For every fused projection in the
shard it also checks the split semantically: a random vector through
the dequantized fused rows, regrouped the way transformers does it, has
to equal the same vector through the written `in_proj_qkv` / `in_proj_z`
/ `in_proj_b` / `in_proj_a`. Shards may arrive one by one: rerun after
each; done shards are recorded in <out>/.convert_state.json and
skipped. `--delete-source` deletes each shard once it is verified (peak
disk: the output plus one shard). Once every shard is done the last run
writes the manifest, layout.json and vocab.bin.

Norm convention: the MLX build stores `1 + w` for the zero-centred
norms (mlx-lm adds the 1 when it converts) and the engine applies
`norm(x) * w`. A checkpoint whose input / post-attention norms have a
median below 0.5 looks zero-centred (transformers' raw weights): the
converter stops; `--norm-offset 1` adds the 1.

Output (<out>):

    artifacts/model_weights.bin   non-expert tensors, 64 B aligned
    artifacts/model_weights.json  {"tensors": {name: {offset, size, shape, dtype[, bits, group_size]}},
                                   "config": {...}}
    artifacts/vocab.bin           scripts/export_vocab.py, if the source has tokenizer.json
    root/packed_experts/layer_XX.bin, layout.json

NumPy only; runs under `python3 -I`.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx_io  # noqa: E402
import schema  # noqa: E402

ALIGN = 64
STATE = ".convert_state.json"
SUFFIXES = (".weight", ".scales", ".biases")


def split_base(name):
    for s in SUFFIXES:
        if name.endswith(s):
            return name[:-len(s)], s
    return name, ""


class Plan:
    """Where every source tensor goes."""

    def __init__(self, cfg, layout="hf"):
        self.cfg = cfg
        self.d = d = schema.dims(cfg)
        self.L, self.E = d["L"], d["E"]
        self.layout = layout
        self.resident = {}
        off = 0
        for name, dt, shape, bits, group in schema.engine_tensors(cfg):
            size = schema.nbytes(dt, shape)
            off += (-off) % ALIGN
            e = {"offset": off, "size": size, "shape": shape, "dtype": dt}
            if bits is not None:
                e["bits"], e["group_size"] = bits, group
            self.resident[name] = e
            off += size
        self.weights_size = off
        self.components, self.expert_size = schema.expert_layout(cfg)
        self.comp_by_name = {c["name"]: c for c in self.components}
        # source name -> how to handle it
        self.source = {}
        for t in schema.mlx_tensors(cfg):
            for name, dt, shape in schema.entries(t):
                if t.kind == "e":
                    layer = int(name.split(".layers.")[1].split(".")[0])
                    self.source[name] = ("expert", dt, shape, (layer, name.split(".switch_mlp.")[1]))
                elif t.base.endswith(("in_proj_qkvz", "in_proj_ba")):
                    self.source[name] = ("fused", dt, shape, t)
                else:
                    self.source[name] = ("copy", dt, shape, t)
        q, z = schema.qkvz_rows(d)
        b, a = schema.ba_rows(d)
        if layout == "naive":
            # Test only: the bug the converter must not have (contiguous
            # [q|k|v|z] / [b|a] blocks, no per-group de-interleave).
            n_qkv = 2 * d["key_dim"] + d["value_dim"]
            q, z = list(range(n_qkv)), list(range(n_qkv, n_qkv + d["value_dim"]))
            b, a = list(range(d["nv"])), list(range(d["nv"], 2 * d["nv"]))
        self.rows = {"in_proj_qkvz": [("in_proj_qkv", np.array(q)), ("in_proj_z", np.array(z))],
                     "in_proj_ba": [("in_proj_b", np.array(b)), ("in_proj_a", np.array(a))]}

    def fingerprint(self):
        h = hashlib.sha256(json.dumps([sorted(self.resident.items()), self.expert_size, self.layout],
                                      sort_keys=True, default=str).encode())
        return h.hexdigest()[:16]


def layer_path(out, l):
    return out / "root" / "packed_experts" / f"layer_{l:02d}.bin"


def prealloc(path, size):
    if not path.exists() or path.stat().st_size != size:
        with open(path, "wb") as f:
            f.truncate(size)


def to_bf16(arr, dt):
    if dt == "BF16":
        return arr
    return mlx_io.f32_to_bf16(arr.astype(np.float32))


def outputs_of(name, sf, plan, norm_offset):
    """[(output name, bytes)] for one source tensor."""
    kind = plan.source[name][0]
    dt = sf.dtype(name)
    if kind == "fused":
        base, suf = split_base(name)
        arr = sf.array(name)
        stem, fused = base.rsplit(".", 1)
        return [(f"{stem}.{nm}{suf}", np.ascontiguousarray(arr[rows]).tobytes())
                for nm, rows in plan.rows[fused]]
    # copy
    e = plan.resident[name]
    if e["dtype"] == "U32":
        return [(name, bytes(sf.raw(name)))]
    arr = sf.array(name)
    if e["dtype"] == "F32":                       # A_log
        f = mlx_io.bf16_to_f32(arr) if dt == "BF16" else arr.astype(np.float32)
        return [(name, np.ascontiguousarray(f, np.float32).tobytes())]
    if name.endswith(schema.NORMS_1PW) and norm_offset:
        f = (mlx_io.bf16_to_f32(arr) if dt == "BF16" else arr.astype(np.float32)) + np.float32(norm_offset)
        return [(name, mlx_io.f32_to_bf16(f).tobytes())]
    return [(name, to_bf16(arr, dt).tobytes())]


def check_norms(sf, plan, norm_offset):
    """Stop on zero-centred input / post-attention norms (see the module doc)."""
    meds = []
    for name in sf.names():
        if name.endswith((".input_layernorm.weight", ".post_attention_layernorm.weight")):
            a = sf.array(name)
            f = mlx_io.bf16_to_f32(a) if sf.dtype(name) == "BF16" else a.astype(np.float32)
            meds.append(float(np.median(f)) + norm_offset)
    if meds and np.median(meds) < 0.5:
        sys.exit(f"error: {sf.path}: input / post-attention norm weights have median {np.median(meds):.3f}; "
                 "they look zero-centred (transformers' raw `w`, applied as 1 + w). The engine applies "
                 "norm(x) * w: rerun with --norm-offset 1 if this checkpoint is not mlx-lm's.")
    return meds


def check_split(sf, plan, out, fused_bases):
    """Random vector through the dequantized fused rows, regrouped as
    transformers' fix_query_key_value_ordering does, against the written
    split tensors read back from the output."""
    d = plan.d
    rng = np.random.RandomState(0)
    worst = 0.0
    with open(out / "artifacts" / "model_weights.bin", "rb") as fw:
        def read_q(name):
            parts = []
            for suf in SUFFIXES:
                e = plan.resident[name + suf]
                fw.seek(e["offset"])
                parts.append(np.frombuffer(fw.read(e["size"]), np.uint32 if suf == ".weight" else np.uint16)
                             .reshape(e["shape"]))
            e = plan.resident[name + ".weight"]
            return mlx_io.affine_dequantize(parts[0], parts[1], parts[2], e["group_size"], e["bits"])

        for base in fused_bases:
            t = plan.source[base + ".weight"][3]
            w = mlx_io.affine_dequantize(sf.array(base + ".weight"), sf.array(base + ".scales"),
                                         sf.array(base + ".biases"), t.group, t.bits)
            x = rng.randn(w.shape[1]).astype(np.float64)
            y = w.astype(np.float64) @ x
            stem = base.rsplit(".", 1)[0]
            nk, dk, dv, nv = d["nk"], d["dk"], d["dv"], d["nv"]
            if base.endswith("in_proj_qkvz"):
                g = y.reshape(nk, 2 * dk + 2 * dv * nv // nk)
                q, k = g[:, :dk], g[:, dk:2 * dk]
                v, z = g[:, 2 * dk:2 * dk + nv // nk * dv], g[:, 2 * dk + nv // nk * dv:]
                want = {"in_proj_qkv": np.concatenate([q.ravel(), k.ravel(), v.ravel()]), "in_proj_z": z.ravel()}
            else:
                g = y.reshape(nk, 2 * nv // nk)
                want = {"in_proj_b": g[:, :nv // nk].ravel(), "in_proj_a": g[:, nv // nk:].ravel()}
            for nm, ref in want.items():
                got = read_q(f"{stem}.{nm}").astype(np.float64) @ x
                err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
                worst = max(worst, err)
                if err > 1e-9:
                    sys.exit(f"error: {stem}.{nm}: split check failed (rel err {err:.3e}); rows are mis-ordered")
    return worst


def process_shard(path, plan, out, state, norm_offset):
    sf = mlx_io.SafeFile(path)
    fname = os.path.basename(path)
    n_res = n_exp = n_fused = 0
    check_norms(sf, plan, norm_offset)
    expect = {}            # output name -> sha256 of the bytes written
    with open(out / "artifacts" / "model_weights.bin", "r+b") as fw:
        for name in sf.names():
            if name.startswith("mtp."):
                continue
            if name not in plan.source:
                sys.exit(f"error: {name} in {fname} is not in the plan (config / schema mismatch)")
            kind, dt, shape, info = plan.source[name]
            sdt, sshape = sf.dtype(name), sf.shape(name)
            ok_dt = sdt == dt or (kind == "copy" and dt in ("BF16", "F32") and sdt in ("BF16", "F32", "F16"))
            if not ok_dt or sshape != shape:
                if not (name.endswith("conv1d.weight") and sshape == [shape[0], 1, shape[1]]):
                    sys.exit(f"error: {name}: {sdt} {sshape} in {fname}, planned {dt} {shape}")
            if kind == "expert":
                layer, comp = info
                c = plan.comp_by_name[comp]
                raw = sf.raw(name)
                h = hashlib.sha256()
                with open(layer_path(out, layer), "r+b") as fe:
                    for e in range(plan.E):
                        blob = raw[e * c["size"]:(e + 1) * c["size"]].tobytes()
                        fe.seek(e * plan.expert_size + c["offset"])
                        fe.write(blob)
                        h.update(blob)
                expect[f"expert:{layer}:{comp}"] = h.hexdigest()
                n_exp += 1
                continue
            for oname, data in outputs_of(name, sf, plan, norm_offset):
                e = plan.resident[oname]
                if len(data) != e["size"]:
                    sys.exit(f"error: {oname}: {len(data)} B, planned {e['size']}")
                fw.seek(e["offset"])
                fw.write(data)
                expect[oname] = hashlib.sha256(data).hexdigest()
                n_res += 1
            n_fused += kind == "fused"
    # read back
    with open(out / "artifacts" / "model_weights.bin", "rb") as fw:
        for key, digest in expect.items():
            if key.startswith("expert:"):
                _, layer, comp = key.split(":")
                c = plan.comp_by_name[comp]
                h = hashlib.sha256()
                with open(layer_path(out, int(layer)), "rb") as fe:
                    for e in range(plan.E):
                        fe.seek(e * plan.expert_size + c["offset"])
                        h.update(fe.read(c["size"]))
                got = h.hexdigest()
            else:
                e = plan.resident[key]
                fw.seek(e["offset"])
                got = hashlib.sha256(fw.read(e["size"])).hexdigest()
            if got != digest:
                sys.exit(f"error: readback mismatch for {key}")
    # semantic check of the fused splits whose three parts are in this shard
    names = set(sf.names())
    fused = sorted({split_base(n)[0] for n in names if plan.source.get(n, ("",))[0] == "fused"})
    whole = [b for b in fused if all(b + s in names for s in SUFFIXES)]
    split_worst = check_split(sf, plan, out, whole) if plan.layout == "hf" else None
    state.setdefault("fused_checked", []).extend(whole)
    sf.close()
    return n_res, n_exp, n_fused, len(whole), split_worst


def manifest_config(cfg):
    tc = schema.text_config(cfg)
    q = schema.quant(cfg)
    d = schema.dims(cfg)
    return {
        "model_type": "qwen3_next",
        "hidden_size": d["H"], "num_hidden_layers": d["L"],
        "num_attention_heads": d["nh"], "num_key_value_heads": d["nkv"], "head_dim": d["hd"],
        "vocab_size": d["V"], "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "num_experts": d["E"], "num_experts_per_tok": d["topk"],
        "moe_intermediate_size": d["I"], "shared_expert_intermediate_size": d["S"],
        "full_attention_interval": d["interval"],
        "linear_num_value_heads": d["nv"], "linear_num_key_heads": d["nk"],
        "linear_key_head_dim": d["dk"], "linear_value_head_dim": d["dv"],
        "linear_conv_kernel_dim": d["K"],
        "partial_rotary_factor": tc.get("partial_rotary_factor", 1.0),
        "rope_theta": tc.get("rope_theta", 10000.0),
        "default_bits": q.get("bits", 4), "default_group_size": q.get("group_size", 64),
        "layer_types": ["full_attention" if schema.is_full(cfg, i) else "linear_attention" for i in range(d["L"])],
        "tie_word_embeddings": d["tied"],
    }


def finalize(source, out, cfg, plan, state):
    d = plan.d
    if d["tied"]:
        # lm_head = a copy of the embedding (the engine reads lm_head.*).
        with open(out / "artifacts" / "model_weights.bin", "r+b") as fw:
            for suf in SUFFIXES:
                s, t = plan.resident["model.embed_tokens" + suf], plan.resident["lm_head" + suf]
                fw.seek(s["offset"])
                data = fw.read(s["size"])
                fw.seek(t["offset"])
                fw.write(data)
    unchecked = sorted(set(b for b, _ in (split_base(n) for n, v in plan.source.items() if v[0] == "fused"))
                       - set(state.get("fused_checked", [])))
    manifest = {"model": str(source), "num_tensors": len(plan.resident), "tensors": plan.resident,
                "config": manifest_config(cfg)}
    (out / "artifacts" / "model_weights.json").write_text(json.dumps(manifest, indent=1))
    (out / "root" / "packed_experts" / "layout.json").write_text(json.dumps({
        "expert_size": plan.expert_size, "expert_stride": plan.expert_size, "num_layers": plan.L,
        "num_experts": plan.E, "components": plan.components}, indent=1))
    tok = source / "tokenizer.json"
    if tok.exists():
        script = Path(__file__).resolve().parents[3] / "scripts" / "export_vocab.py"
        subprocess.run([sys.executable, "-I", str(script), str(tok), str(out / "artifacts" / "vocab.bin")],
                       check=True)
    else:
        print("note: no tokenizer.json in the source; vocab.bin not written")
    if unchecked and plan.layout == "hf":
        print(f"note: {len(unchecked)} fused projections had their parts in two shards; byte-checked "
              f"only (e.g. {unchecked[0]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="MLX directory (config.json + index + the shards present so far)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delete-source", action="store_true", help="delete each shard after it is verified")
    ap.add_argument("--norm-offset", type=float, default=0.0,
                    help="add this to the 1 + w norms (1 for a zero-centred checkpoint)")
    ap.add_argument("--layout", choices=("hf", "naive"), default="hf", help=argparse.SUPPRESS)
    a = ap.parse_args()
    source, out = Path(a.source).expanduser(), Path(a.out).expanduser()
    cfg = json.loads((source / "config.json").read_text())
    bad = schema.check_supported(cfg)
    if bad:
        sys.exit("error: unsupported config: " + "; ".join(bad))
    wm = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    missing, extra = schema.check_index(cfg, wm)
    if missing or extra:
        sys.exit(f"error: index does not match the config: missing {missing[:5]}, extra {extra[:5]}")
    plan = Plan(cfg, a.layout)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "root" / "packed_experts").mkdir(parents=True, exist_ok=True)
    sp = out / STATE
    state = json.loads(sp.read_text()) if sp.exists() else {}
    fp = [plan.fingerprint(), a.norm_offset]
    if state and state.get("plan") != fp:
        sys.exit(f"error: {sp} is from a different plan or --norm-offset; remove {out} and start again")
    state = state or {"plan": fp, "done": []}
    prealloc(out / "artifacts" / "model_weights.bin", plan.weights_size)
    for l in range(plan.L):
        prealloc(layer_path(out, l), plan.E * plan.expert_size)
    files = sorted(set(wm.values()))
    todo = [f for f in files if f not in state["done"] and (source / f).exists()]
    t0 = time.time()
    for f in todo:
        t1 = time.time()
        n_res, n_exp, n_fused, n_chk, worst = process_shard(source / f, plan, out, state, a.norm_offset)
        state["done"].append(f)
        sp.write_text(json.dumps(state))
        if a.delete_source:
            os.remove(source / f)
        chk = f", split check on {n_chk} fused (worst {worst:.1e})" if n_chk and worst is not None else ""
        print(f"{f}: {n_res} resident + {n_exp} expert tensors written and read back{chk}, "
              f"{time.time() - t1:.1f}s{' (deleted)' if a.delete_source else ''}", flush=True)
    left = [f for f in files if f not in state["done"]]
    if left:
        print(f"{len(left)} of {len(files)} shards still to convert (e.g. {left[0]}); rerun when they are present")
        return
    finalize(source, out, cfg, plan, state)
    exp = plan.L * plan.E * plan.expert_size
    print(f"done: model_weights.bin {plan.weights_size / 2**30:.3f} GiB, experts {plan.L} x {plan.E} x "
          f"{plan.expert_size:,} B = {exp / 2**30:.2f} GiB, {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
