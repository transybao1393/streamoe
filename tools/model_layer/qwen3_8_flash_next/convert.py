#!/usr/bin/env python3
"""Convert the MLX 4-bit Qwen3.8-Flash-Next checkpoint into streamoe's layout.

    python3 -I convert.py --source <mlx dir> --out <dir> [--delete-source]
                          [--expert-align 16384] [--only SHARD ...]

Works one source shard at a time, so the real ~104 GiB source never has
to sit on disk whole: the whole output layout is planned up front from
config.json + model.safetensors.index.json (schema.py derives every
tensor's shape and dtype from the config), output files are
preallocated (sparse), and each shard's tensors are written to their
planned places, read back and compared, before the next shard. With
`--delete-source` each source shard is deleted once its tensors are
verified on disk, so peak disk use is the output plus one shard; shards
may also be downloaded one at a time and the converter rerun after each
(processed shards are recorded in <out>/.convert_state.json and skipped).
The final step, once every shard of the index is done, writes the
manifests and the n-gram header.

Output (<out>):

    artifacts/model_weights.bin   every non-expert, non-n-gram tensor, 64 B aligned,
                                  stored as in the source (A_log BF16 -> F32,
                                  as scripts/extract_weights.py does)
    artifacts/model_weights.json  {"tensors": {name: {offset, size, shape, dtype
                                  [, bits, group_size]}}, "config": {...}}; names
                                  drop the "language_model." prefix
    artifacts/vocab.bin           scripts/export_vocab.py, if tokenizer.json exists
    root/packed_experts/layer_XX.bin + layout.json
                                  expert e of layer l at e * expert_stride:
                                  [gate w|s|b, up w|s|b, down w|s|b], one contiguous blob
    root/ngram.mfng               the PLE n-gram table (ngram_layout.py: MFNG, q38-ple)
    expert_index.json             tools/gen_expert_index.py's schema, pointing into the
                                  source shards (valid while they exist)

Expert blob at hidden 2560 / intermediate 640, 4-bit group 32: each of
gate / up / down is 819,200 B of u32 codes + 102,400 B scales + 102,400 B
biases = 1,024,000 B; the blob is 3,072,000 B, padded to a 16 KiB
multiple: stride 3,080,192 B (`--expert-align 0` packs them unpadded).
Run `python3 -I schema.py --config <config.json>` for every size.
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
import ngram_layout  # noqa: E402
import schema  # noqa: E402

ALIGN = 64
EXPERT_PROJ = ("gate_proj", "up_proj", "down_proj")
STATE = ".convert_state.json"


def strip(name):
    return name[len(schema.PREFIX):] if name.startswith(schema.PREFIX) else name


class Plan:
    """Where every source tensor goes."""

    def __init__(self, cfg, expert_align):
        self.cfg = cfg
        self.tc = tc = schema.text_config(cfg)
        self.L, self.E = tc["num_hidden_layers"], tc["num_experts"]
        # resident tensors: name -> entry
        self.resident = {}
        off = 0
        self.ngram = {}        # MLX name -> (shard, component)
        self.expert = {}       # MLX name -> (layer, component index)
        self.ngram_shape = None
        for t in schema.tensors(cfg):
            for name, dt, shape in schema.stored(cfg, t):
                if t.role == "expert":
                    layer = int(name.split(".layers.")[1].split(".")[0])
                    proj = name.split(".switch_mlp.")[1]
                    self.expert[name] = (layer, proj)
                    continue
                if t.role == "ngram":
                    shard = int(name.split(".shards.")[1].split(".")[0])
                    self.ngram[name] = (shard, name.rsplit(".", 1)[1])
                    self.ngram_bits = schema.quant_rule(cfg, t.base)
                    continue
                out_dt = "F32" if name.endswith(".linear_attn.A_log") and dt == "BF16" else dt
                size = schema.nbytes(out_dt, shape)
                off += (-off) % ALIGN
                e = {"offset": off, "size": size, "shape": shape, "dtype": out_dt, "source_dtype": dt}
                if dt == "U32":
                    e["bits"], e["group_size"] = schema.quant_rule(cfg, t.base)
                self.resident[name] = e
                off += size
        self.weights_size = off
        comps = schema.expert_components(cfg)
        self.components, coff = [], 0
        for comp, dt, shape, nb in comps:
            self.components.append({"name": comp, "offset": coff, "size": nb, "dtype": dt, "shape": shape})
            coff += nb
        self.comp_by_name = {c["name"]: c for c in self.components}
        self.expert_bits = schema.quant_rule(cfg, f"{schema.PREFIX}model.layers.0.mlp.switch_mlp.gate_proj")
        self.expert_size = coff
        self.expert_stride = schema.round_up(coff, expert_align) if expert_align else coff
        if self.ngram:
            heads = (tc["ngram_size"] - 1) * tc["heads_per_ngram"]
            _, _, _, self.ngram_rows = schema.ngram_vocab(tc, 0)
            self.ngram_dim = tc["ple_embed_dim"] // heads
            self.ngram_shards = tc["split_ngram_parts"]
            assert len(schema.ple_layers(tc)) == 1, "one PLE layer supported (the MFNG file holds one table)"
            self.ple_layer = schema.ple_layers(tc)[0]

    def fingerprint(self):
        h = hashlib.sha256(json.dumps([sorted(self.resident.items()), self.expert_stride], sort_keys=True,
                                      default=str).encode())
        return h.hexdigest()[:16]


class Outputs:
    def __init__(self, out, plan):
        self.plan = plan
        self.art = out / "artifacts"
        self.pk = out / "root" / "packed_experts"
        self.art.mkdir(parents=True, exist_ok=True)
        self.pk.mkdir(parents=True, exist_ok=True)
        self.weights = self.art / "model_weights.bin"
        self._prealloc(self.weights, plan.weights_size)
        for l in range(plan.L):
            self._prealloc(self.layer_path(l), plan.E * plan.expert_stride)
        self.mfng = None
        if plan.ngram:
            g = plan.ngram_bits[1]
            self.mfng = ngram_layout.MfngWriter(str(out / "root" / "ngram.mfng"), plan.ngram_rows,
                                                plan.ngram_dim, g, plan.ngram_shards)

    @staticmethod
    def _prealloc(path, size):
        if not path.exists() or path.stat().st_size != size:
            with open(path, "wb") as f:
                f.truncate(size)

    def layer_path(self, l):
        return self.pk / f"layer_{l:02d}.bin"


def converted(raw, src_dt, entry):
    if entry["dtype"] == "F32" and src_dt == "BF16":
        return mlx_io.bf16_to_f32(np.frombuffer(bytes(raw), np.uint16)).tobytes()
    return bytes(raw)


def process_shard(path, plan, outs, state):
    """Write one source shard's tensors, then read them back and compare."""
    sf = mlx_io.SafeFile(path)
    fname = os.path.basename(path)
    n_res = n_exp = n_ng = 0
    with open(outs.weights, "r+b") as fw:
        for name in sf.names():
            dt, shape = sf.dtype(name), sf.shape(name)
            off, nb = sf.span(name)
            state["source_spans"][name] = [fname, off, nb, shape]
            if name in plan.resident:
                e = plan.resident[name]
                if dt != e["source_dtype"] or shape != e["shape"]:
                    sys.exit(f"error: {name}: {dt} {shape} in {fname}, planned {e['source_dtype']} {e['shape']}")
                data = converted(sf.raw(name), dt, e)
                assert len(data) == e["size"], name
                fw.seek(e["offset"])
                fw.write(data)
                n_res += 1
            elif name in plan.expert:
                layer, proj = plan.expert[name]
                c = plan.comp_by_name[proj]
                if dt != c["dtype"] or shape != [plan.E] + c["shape"]:
                    sys.exit(f"error: {name}: {dt} {shape}, planned {c['dtype']} {[plan.E] + c['shape']}")
                raw = sf.raw(name)
                with open(outs.layer_path(layer), "r+b") as fe:
                    for e in range(plan.E):
                        fe.seek(e * plan.expert_stride + c["offset"])
                        fe.write(raw[e * c["size"]:(e + 1) * c["size"]].tobytes())
                n_exp += 1
            elif name in plan.ngram:
                shard, comp = plan.ngram[name]
                outs.mfng.put(shard, comp, sf.raw(name), shape)
                if comp == "weight":
                    state["ngram_rows"][str(shard)] = shape[0]
                n_ng += 1
            elif "vision" in name or ".visual." in name or name.startswith("mtp"):
                continue
            else:
                sys.exit(f"error: {name} in {fname} is not in the plan (config / schema mismatch)")
    verify_shard(sf, plan, outs)
    sf.close()
    return n_res, n_exp, n_ng


def verify_shard(sf, plan, outs):
    """Read every tensor of `sf` back from the output and compare bytes."""
    with open(outs.weights, "rb") as fw:
        for name in sf.names():
            if name in plan.resident:
                e = plan.resident[name]
                fw.seek(e["offset"])
                if fw.read(e["size"]) != converted(sf.raw(name), sf.dtype(name), e):
                    sys.exit(f"error: readback mismatch for {name}")
            elif name in plan.expert:
                layer, proj = plan.expert[name]
                c = plan.comp_by_name[proj]
                raw = sf.raw(name)
                with open(outs.layer_path(layer), "rb") as fe:
                    for e in range(plan.E):
                        fe.seek(e * plan.expert_stride + c["offset"])
                        if fe.read(c["size"]) != raw[e * c["size"]:(e + 1) * c["size"]].tobytes():
                            sys.exit(f"error: readback mismatch for {name} expert {e}")
            elif name in plan.ngram:
                shard, comp = plan.ngram[name]
                got = outs.mfng.read_component(shard, comp)
                if got.tobytes() != sf.raw(name).tobytes():
                    sys.exit(f"error: readback mismatch for {name}")


def manifest_config(cfg, plan):
    tc = plan.tc
    rp = tc.get("rope_parameters") or {}
    q = cfg.get("quantization") or {}
    c = {
        "model_type": "qwen4_exp",
        "hidden_size": tc["hidden_size"], "num_hidden_layers": tc["num_hidden_layers"],
        "num_attention_heads": tc["num_attention_heads"], "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"], "vocab_size": tc["vocab_size"], "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "num_experts": tc["num_experts"], "num_experts_per_tok": tc["num_experts_per_tok"],
        "moe_intermediate_size": tc["moe_intermediate_size"],
        "shared_expert_intermediate_size": tc["shared_expert_intermediate_size"],
        "full_attention_interval": tc.get("full_attention_interval", 4),
        "linear_num_value_heads": tc["linear_num_value_heads"], "linear_num_key_heads": tc["linear_num_key_heads"],
        "linear_key_head_dim": tc["linear_key_head_dim"], "linear_value_head_dim": tc["linear_value_head_dim"],
        "linear_conv_kernel_dim": tc["linear_conv_kernel_dim"],
        "partial_rotary_factor": rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)),
        "rope_theta": rp.get("rope_theta", 10000000.0),
        "default_bits": q.get("bits", 4), "default_group_size": q.get("group_size", 64),
        "layer_types": tc["layer_types"],
        # Everything else (hyper-connections, QSA indexer, PLE / n-gram).
        "text_config": tc,
    }
    return c


def finalize(source, out, cfg, plan, outs, state):
    tensors = {}
    for name, e in plan.resident.items():
        ent = {k: v for k, v in e.items() if k != "source_dtype"}
        tensors[strip(name)] = ent
    manifest = {"model": str(source), "num_tensors": len(tensors), "tensors": tensors,
                "config": manifest_config(cfg, plan)}
    if plan.ngram:
        manifest["ngram_file"] = "root/ngram.mfng"
    (outs.art / "model_weights.json").write_text(json.dumps(manifest, indent=1))
    (outs.pk / "layout.json").write_text(json.dumps({
        "expert_size": plan.expert_size, "expert_stride": plan.expert_stride, "num_layers": plan.L,
        "num_experts": plan.E, "bits": plan.expert_bits[0], "group_size": plan.expert_bits[1],
        "components": plan.components}, indent=1))
    if plan.ngram:
        pre = f"model.layers.{plan.ple_layer}.ple.ple_embedding."
        buf = {}
        with open(outs.weights, "rb") as fw:
            for b in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
                e = tensors[pre + b]
                fw.seek(e["offset"])
                buf[b] = np.frombuffer(fw.read(e["size"]), np.int64).tolist()
        sizes, offsets, _, _ = schema.ngram_vocab(plan.tc, 0)
        if buf["ngram_heads_vocab_sizes"] != sizes or buf["ngram_heads_offsets"] != offsets:
            sys.exit("error: checkpoint n-gram head sizes / offsets differ from the config-derived ones")
        outs.mfng.finish({int(k): v for k, v in state["ngram_rows"].items()}, buf["layer_multipliers"],
                         buf["ngram_heads_vocab_sizes"], buf["ngram_heads_offsets"], plan.tc["ngram_size"])
    # expert_index.json (gen_expert_index.py schema)
    reads = {}
    for name, (layer, proj) in plan.expert.items():
        fname, off, nb, shape = state["source_spans"][name]
        reads.setdefault(str(layer), {})[proj] = {
            "file": fname, "abs_offset": off, "expert_stride": nb // plan.E, "expert_size": nb // plan.E,
            "total_size": nb, "shape": shape}
    (out / "expert_index.json").write_text(json.dumps({"model_path": str(source), "expert_reads": reads}, indent=1))
    tok = source / "tokenizer.json"
    if tok.exists():
        script = Path(__file__).resolve().parents[3] / "scripts" / "export_vocab.py"
        subprocess.run([sys.executable, "-I", str(script), str(tok), str(outs.art / "vocab.bin")], check=True)
    else:
        print("note: no tokenizer.json in the source; vocab.bin not written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="MLX directory (config.json + index + shards present so far)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expert-align", type=int, default=16384)
    ap.add_argument("--delete-source", action="store_true", help="delete each shard after it is verified")
    ap.add_argument("--only", nargs="*", help="process only these shard files")
    a = ap.parse_args()
    source, out = Path(a.source).expanduser(), Path(a.out).expanduser()
    cfg = json.loads((source / "config.json").read_text())
    if cfg.get("model_type") != "qwen4_exp":
        sys.exit(f"error: model_type {cfg.get('model_type')!r}, expected 'qwen4_exp'")
    wm = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    missing, extra = schema.check_index(cfg, wm)
    if missing or extra:
        sys.exit(f"error: index does not match the config: missing {missing[:5]}, extra {extra[:5]}")
    plan = Plan(cfg, a.expert_align)
    out.mkdir(parents=True, exist_ok=True)
    sp = out / STATE
    state = json.loads(sp.read_text()) if sp.exists() else {}
    if state and state.get("plan") != plan.fingerprint():
        sys.exit(f"error: {sp} is from a different plan; remove {out} and start again")
    state = state or {"plan": plan.fingerprint(), "done": [], "source_spans": {}, "ngram_rows": {}}
    outs = Outputs(out, plan)
    files = sorted(set(wm.values()))
    todo = [f for f in files if f not in state["done"] and (source / f).exists()
            and (not a.only or f in a.only)]
    t0 = time.time()
    for f in todo:
        t1 = time.time()
        n = process_shard(source / f, plan, outs, state)
        state["done"].append(f)
        sp.write_text(json.dumps(state))
        if a.delete_source:
            os.remove(source / f)
        print(f"{f}: {n[0]} resident, {n[1]} expert, {n[2]} n-gram tensors, verified, "
              f"{time.time() - t1:.1f}s{' (deleted)' if a.delete_source else ''}", flush=True)
    left = [f for f in files if f not in state["done"]]
    if left:
        print(f"{len(left)} of {len(files)} shards still to convert (e.g. {left[0]}); rerun when they are present")
        return
    finalize(source, out, cfg, plan, outs, state)
    print(f"done: weights {plan.weights_size / 2**30:.3f} GiB, experts {plan.L} x {plan.E} x "
          f"{plan.expert_stride} B, in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
