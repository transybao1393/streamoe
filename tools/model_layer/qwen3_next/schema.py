#!/usr/bin/env python3
"""Tensor schema of Qwen3-Next (`model_type: qwen3_next`) in MLX and in streamoe.

    python3 -I schema.py --config <config.json> [--index <model.safetensors.index.json>]

prints every size (source, converted, per expert) and, with --index,
checks that the index holds exactly the tensors the config implies.
NumPy is not needed here.

MLX layout (`mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`), names
without a `language_model.` prefix. Quantized tensors are three entries,
`<base>.weight` (U32, codes packed low bits first), `<base>.scales` and
`<base>.biases` (BF16, one per group of the input axis):

    model.embed_tokens                        q  [V, H]
    lm_head                                   q  [V, H]   (absent if tied)
    model.norm.weight                            [H]       1 + w (see below)
    model.layers.{i}.input_layernorm.weight      [H]       1 + w
    model.layers.{i}.post_attention_layernorm.weight [H]   1 + w
    Gated DeltaNet layers ((i + 1) % full_attention_interval != 0):
      linear_attn.in_proj_qkvz                q  [2*nk*dk + 2*nv*dv, H]  fused, per key-head group
      linear_attn.in_proj_ba                  q  [2*nv, H]               fused, per key-head group
      linear_attn.conv1d.weight                  [2*nk*dk + nv*dv, K, 1]
      linear_attn.A_log, linear_attn.dt_bias     [nv]
      linear_attn.norm.weight                    [dv]      plain w (gated norm)
      linear_attn.out_proj                    q  [H, nv*dv]
    full-attention layers:
      self_attn.q_proj                        q  [2*nh*hd, H]   per head [q | gate]
      self_attn.k_proj, self_attn.v_proj      q  [nkv*hd, H]
      self_attn.o_proj                        q  [H, nh*hd]
      self_attn.q_norm.weight, k_norm.weight     [hd]      1 + w
    every layer (MoE):
      mlp.gate                                q8 [E, H]
      mlp.shared_expert_gate                  q8 [1, H]
      mlp.shared_expert.{gate,up}_proj        q  [S, H];  down_proj q [H, S]
      mlp.switch_mlp.{gate,up}_proj           q  [E, I, H]; down_proj q [E, H, I]

The fused rows of `in_proj_qkvz` are grouped per key head g (as
transformers' `fix_query_key_value_ordering` reads them):
`[q_g (dk) | k_g (dk) | v_g (r*dv) | z_g (r*dv)]` with r = nv / nk, and
`in_proj_ba` per group `[b_g (r) | a_g (r)]`.

"1 + w": transformers' Qwen3NextRMSNorm computes `norm(x) * (1 + w)`
with zero-centred `w`; mlx-lm's `sanitize` adds the 1 when it converts,
so the MLX files hold `1 + w` and are applied as `norm(x) * w`, which is
what the engine does for every norm. The gated norm stores plain `w`.

streamoe layout (what the engine's qwen3_5_moe path reads): the same
names, except the two fused projections become four, with their rows
re-laid out (quantization groups run along the input axis, so rows of
weight / scales / biases move together and nothing is requantized):

    linear_attn.in_proj_qkv   q [2*nk*dk + nv*dv, H] = [all q | all k | all v]
    linear_attn.in_proj_z     q [nv*dv, H]
    linear_attn.in_proj_b     q [nv, H]
    linear_attn.in_proj_a     q [nv, H]

A_log is stored F32 (BF16 in the MLX build, promoted exactly) and every
other unquantized tensor BF16. Routed
experts go to root/packed_experts/layer_XX.bin, expert e at
e * expert_size: [gate w|s|b, up w|s|b, down w|s|b].
"""

import argparse
import json
import sys
from collections import namedtuple

# kind: "q" quantized (bits / group in the entry), "f" float vector or
# matrix (dtype is the output dtype), "e" routed-expert component.
Tensor = namedtuple("Tensor", "base kind shape bits group dtype")

NORMS_1PW = (".input_layernorm.weight", ".post_attention_layernorm.weight",
             "model.norm.weight", ".q_norm.weight", ".k_norm.weight")
EXPERT_PROJ = ("gate_proj", "up_proj", "down_proj")


def text_config(cfg):
    return cfg.get("text_config", cfg)


def dims(cfg):
    tc = text_config(cfg)
    d = dict(
        H=tc["hidden_size"], L=tc["num_hidden_layers"], V=tc["vocab_size"],
        nh=tc["num_attention_heads"], nkv=tc["num_key_value_heads"],
        hd=tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"],
        nk=tc["linear_num_key_heads"], nv=tc["linear_num_value_heads"],
        dk=tc["linear_key_head_dim"], dv=tc["linear_value_head_dim"],
        K=tc["linear_conv_kernel_dim"], E=tc["num_experts"], topk=tc["num_experts_per_tok"],
        I=tc["moe_intermediate_size"], S=tc["shared_expert_intermediate_size"],
        interval=tc["full_attention_interval"],
        tied=bool(tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False))),
    )
    assert d["nv"] % d["nk"] == 0, "linear_num_value_heads must be a multiple of linear_num_key_heads"
    d["r"] = d["nv"] // d["nk"]
    d["key_dim"] = d["nk"] * d["dk"]
    d["value_dim"] = d["nv"] * d["dv"]
    d["conv_dim"] = 2 * d["key_dim"] + d["value_dim"]
    return d


def is_full(cfg, i):
    tc = text_config(cfg)
    if "layer_types" in tc:
        return tc["layer_types"][i] == "full_attention"
    return (i + 1) % tc["full_attention_interval"] == 0


def check_supported(cfg):
    """The parts of the config the engine's qwen3_5_moe path assumes."""
    tc = text_config(cfg)
    bad = []
    if tc.get("model_type") != "qwen3_next":
        bad.append(f"model_type {tc.get('model_type')!r} is not qwen3_next")
    if tc.get("mlp_only_layers"):
        bad.append("mlp_only_layers is not empty (dense layers are not supported)")
    if tc.get("decoder_sparse_step", 1) != 1:
        bad.append("decoder_sparse_step != 1")
    if tc.get("attention_bias"):
        bad.append("attention_bias")
    if tc.get("hidden_act", "silu") != "silu":
        bad.append("hidden_act is not silu")
    rs = tc.get("rope_scaling") or tc.get("rope_parameters") or {}
    if rs.get("rope_type", rs.get("type", "default")) not in ("default", None):
        bad.append(f"rope type {rs.get('rope_type')!r}")
    if not tc.get("norm_topk_prob", True):
        bad.append("norm_topk_prob false")
    return bad


def quant(cfg):
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    return q


def bits_group(cfg, base):
    q = quant(cfg)
    o = q.get(base)
    if isinstance(o, dict):
        return int(o["bits"]), int(o["group_size"])
    return int(q.get("bits", 4)), int(q.get("group_size", 64))


def mlx_tensors(cfg):
    """Every tensor of the MLX checkpoint (quantized ones as one entry)."""
    d = dims(cfg)
    H = d["H"]
    out = []

    def q(base, shape):
        b, g = bits_group(cfg, base)
        out.append(Tensor(base, "q", list(shape), b, g, None))

    def f(base, shape, dtype="BF16"):
        out.append(Tensor(base, "f", list(shape), None, None, dtype))

    def e(base, shape):
        b, g = bits_group(cfg, base)
        out.append(Tensor(base, "e", list(shape), b, g, None))

    q("model.embed_tokens", [d["V"], H])
    if not d["tied"]:
        q("lm_head", [d["V"], H])
    f("model.norm.weight", [H])
    for i in range(d["L"]):
        p = f"model.layers.{i}."
        f(p + "input_layernorm.weight", [H])
        f(p + "post_attention_layernorm.weight", [H])
        if is_full(cfg, i):
            a = p + "self_attn."
            q(a + "q_proj", [2 * d["nh"] * d["hd"], H])
            q(a + "k_proj", [d["nkv"] * d["hd"], H])
            q(a + "v_proj", [d["nkv"] * d["hd"], H])
            q(a + "o_proj", [H, d["nh"] * d["hd"]])
            f(a + "q_norm.weight", [d["hd"]])
            f(a + "k_norm.weight", [d["hd"]])
        else:
            a = p + "linear_attn."
            q(a + "in_proj_qkvz", [2 * d["key_dim"] + 2 * d["value_dim"], H])
            q(a + "in_proj_ba", [2 * d["nv"], H])
            f(a + "conv1d.weight", [d["conv_dim"], d["K"], 1])
            f(a + "A_log", [d["nv"]])           # BF16 in the MLX build; F32 also accepted
            f(a + "dt_bias", [d["nv"]])
            f(a + "norm.weight", [d["dv"]])
            q(a + "out_proj", [H, d["value_dim"]])
        m = p + "mlp."
        q(m + "gate", [d["E"], H])
        q(m + "shared_expert_gate", [1, H])
        q(m + "shared_expert.gate_proj", [d["S"], H])
        q(m + "shared_expert.up_proj", [d["S"], H])
        q(m + "shared_expert.down_proj", [H, d["S"]])
        e(m + "switch_mlp.gate_proj", [d["E"], d["I"], H])
        e(m + "switch_mlp.up_proj", [d["E"], d["I"], H])
        e(m + "switch_mlp.down_proj", [d["E"], H, d["I"]])
    return out


def q_entries(t):
    """(name, dtype, shape) of the safetensors entries of a quantized tensor."""
    *lead, n_in = t.shape
    assert n_in % t.group == 0 and (n_in * t.bits) % 32 == 0, f"{t.base}: in {n_in} vs group {t.group}"
    return [(t.base + ".weight", "U32", lead + [n_in * t.bits // 32]),
            (t.base + ".scales", "BF16", lead + [n_in // t.group]),
            (t.base + ".biases", "BF16", lead + [n_in // t.group])]


def entries(t):
    if t.kind in ("q", "e"):
        return q_entries(t)
    return [(t.base, t.dtype, t.shape)]


def nbytes(dtype, shape):
    n = {"U32": 4, "F32": 4, "BF16": 2, "F16": 2}[dtype]
    for s in shape:
        n *= s
    return n


def qkvz_rows(d):
    """Source row indices of the engine's in_proj_qkv and in_proj_z, read
    from the fused per-key-head-group layout of in_proj_qkvz."""
    dk, r, dv, nk = d["dk"], d["r"], d["dv"], d["nk"]
    pg = 2 * dk + 2 * r * dv
    qs = [g * pg + j for g in range(nk) for j in range(dk)]
    ks = [g * pg + dk + j for g in range(nk) for j in range(dk)]
    vs = [g * pg + 2 * dk + j for g in range(nk) for j in range(r * dv)]
    zs = [g * pg + 2 * dk + r * dv + j for g in range(nk) for j in range(r * dv)]
    return qs + ks + vs, zs


def ba_rows(d):
    """Source rows of in_proj_b and in_proj_a in the fused in_proj_ba."""
    r, nk = d["r"], d["nk"]
    bs = [g * 2 * r + j for g in range(nk) for j in range(r)]
    as_ = [g * 2 * r + r + j for g in range(nk) for j in range(r)]
    return bs, as_


def engine_tensors(cfg):
    """Every tensor the converter writes to model_weights.bin, as
    (name, dtype, shape, bits, group), in manifest order. Experts are
    separate (expert_layout)."""
    d = dims(cfg)
    out = []
    for t in mlx_tensors(cfg):
        if t.kind == "e":
            continue
        if t.base.endswith("in_proj_qkvz"):
            stem = t.base[:-len("in_proj_qkvz")]
            H = t.shape[1]
            parts = [("in_proj_qkv", 2 * d["key_dim"] + d["value_dim"]), ("in_proj_z", d["value_dim"])]
        elif t.base.endswith("in_proj_ba"):
            stem = t.base[:-len("in_proj_ba")]
            H = t.shape[1]
            parts = [("in_proj_b", d["nv"]), ("in_proj_a", d["nv"])]
        else:
            for name, dt, shape in entries(t):
                if name.endswith(".A_log"):
                    dt = "F32"   # the engine reads A_log as f32
                bits = t.bits if dt == "U32" else None
                out.append((name, dt, shape, bits, t.group if dt == "U32" else None))
            continue
        for nm, rows in parts:
            tt = Tensor(stem + nm, "q", [rows, H], t.bits, t.group, None)
            for name, dt, shape in q_entries(tt):
                out.append((name, dt, shape, t.bits if dt == "U32" else None, t.group if dt == "U32" else None))
    if d["tied"]:
        # The engine reads lm_head.*: the converter writes a copy of the embedding.
        for name, dt, shape, b, g in list(out):
            if name.startswith("model.embed_tokens."):
                out.append(("lm_head." + name.split(".")[-1], dt, shape, b, g))
    return sorted(out, key=lambda x: x[0])


def expert_layout(cfg):
    """Component (name, offset, size, dtype, shape) list and expert size."""
    d = dims(cfg)
    comps, off = [], 0
    for p in EXPERT_PROJ:
        base = f"model.layers.0.mlp.switch_mlp.{p}"
        b, g = bits_group(cfg, base)
        shape = [d["I"], d["H"]] if p != "down_proj" else [d["H"], d["I"]]
        for name, dt, sh in q_entries(Tensor(base, "e", shape, b, g, None)):
            sz = nbytes(dt, sh)
            comps.append({"name": f"{p}.{name.rsplit('.', 1)[1]}", "offset": off, "size": sz,
                          "dtype": dt, "shape": sh})
            off += sz
    return comps, off


def check_index(cfg, weight_map):
    """Names the config implies vs the index: (missing, unexpected)."""
    want = set()
    for t in mlx_tensors(cfg):
        for name, _, _ in entries(t):
            want.add(name)
    have = set(weight_map)
    # MTP weights are not used (transformers ignores `mtp.*` too).
    unexpected = sorted(n for n in have - want if not n.startswith("mtp."))
    return sorted(want - have), unexpected


def report(cfg):
    d = dims(cfg)
    src = sum(nbytes(dt, sh) for t in mlx_tensors(cfg) for _, dt, sh in entries(t))
    res = sum(nbytes(dt, sh) for _, dt, sh, _, _ in engine_tensors(cfg))
    _, esz = expert_layout(cfg)
    exp = esz * d["E"] * d["L"]
    G = 1 << 30
    lines = [
        f"layers {d['L']} ({sum(is_full(cfg, i) for i in range(d['L']))} full attention), "
        f"hidden {d['H']}, experts {d['E']} top-{d['topk']}, vocab {d['V']}",
        f"source tensors: {src / G:.2f} GiB ({src:,} B)",
        f"artifacts/model_weights.bin: {res / G:.3f} GiB ({res:,} B, before 64 B alignment)",
        f"expert: {esz:,} B; per layer file {esz * d['E'] / G:.3f} GiB; all experts {exp / G:.2f} GiB",
        f"converted total: {(res + exp) / G:.2f} GiB",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--index")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    bad = check_supported(cfg)
    for b in bad:
        print("unsupported:", b)
    print(report(cfg))
    if a.index:
        wm = json.load(open(a.index))["weight_map"]
        missing, unexpected = check_index(cfg, wm)
        print(f"index: {len(wm)} tensors; missing {len(missing)}, unexpected {len(unexpected)}")
        for n in missing[:20]:
            print("  missing", n)
        for n in unexpected[:20]:
            print("  unexpected", n)
        if missing or unexpected:
            sys.exit(1)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
