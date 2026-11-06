#!/usr/bin/env python3
"""Every tensor of a Qwen3.8-Flash-Next (`qwen4_exp`) MLX checkpoint,
derived from config.json alone.

    python3 -I schema.py --config ~/models/qwen3-8-flash-next/meta/config.json \
        --index ~/models/qwen3-8-flash-next/meta/model.safetensors.index.json

prints the size of every part of the streammoe layout for that config
and checks the derived names against the index (names and which
tensors are quantized; the index carries no shapes).

The MLX names are the transformers names with a `language_model.`
prefix, except:

- routed experts: transformers `mlp.experts.gate_up_proj [E, 2I, H]`
  (gate rows first, then up) and `mlp.experts.down_proj [E, H, I]`
  are MLX `mlp.switch_mlp.{gate,up,down}_proj` (`[E, I, H]`,
  `[E, I, H]`, `[E, H, I]`);
- depthwise conv weights: transformers `[C, 1, K]`, MLX `[C, K, 1]`;
- the PLE n-gram table: transformers `ple_embedding.ngram_embedding
  .weight [padded_vocab, D]`, MLX `ple_embedding.ngram_embedding
  .shards.{s}` with `split_ngram_parts` equal row blocks;
- no MTP and no final `model.norm` (the hyper-connection mixer's
  `hc_norm` is the last norm).

Norm weights stay centered at zero (the model applies `1 + w`); see
the meta README: the 4-bit build was checked not to fold the `+1`.
"""

import argparse
import json
import math
import sys

PREFIX = "language_model."


# --- n-gram vocabulary (verbatim math from modeling_qwen4_exp.py) ----------

def is_prime(value):
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def find_nth_prime_after(start, count):
    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


def ngram_vocab(tc, ple_layer_index=0):
    """(head_vocab_sizes, head_offsets, total, padded) for one PLE layer.
    Each head's size is the (global_head_idx + 1)-th prime after
    base - 1 (so consecutive heads take consecutive primes)."""
    heads = (tc["ngram_size"] - 1) * tc["heads_per_ngram"]
    sizes, offsets, total = [], [], 0
    prime = tc["ngram_vocab_size_base"] - 1
    # find_nth_prime_after(base-1, g+1) for g = 0, 1, ... walks the same
    # primes; step incrementally instead of restarting per head.
    first = ple_layer_index * heads
    for _ in range(first):
        prime = find_nth_prime_after(prime, 1)
    for _ in range(heads):
        prime = find_nth_prime_after(prime, 1)
        sizes.append(prime)
        offsets.append(total)
        total += prime
    div = tc["make_ngram_vocab_size_divisible_by"]
    padded = math.ceil(total / div) * div
    return sizes, offsets, total, padded


# --- config ------------------------------------------------------------------

def text_config(cfg):
    """The text config with the defaults the transformers config class
    fills in (seed, norm_topk_prob, split parts) and 0-based PLE layers."""
    tc = dict(cfg.get("text_config", cfg))
    tc.setdefault("seed", 1234)
    tc.setdefault("norm_topk_prob", True)
    tc.setdefault("split_ngram_parts", 512)
    tc.setdefault("make_ngram_vocab_size_divisible_by", 128)
    tc.setdefault("hc_count", 4)
    tc.setdefault("hc_lowrank", 320)
    tc["ple_layer_ids"] = sorted(set(tc.get("ple_layer_ids") or []))
    if tc.get("ple_embed_dim") is None:
        tc["ple_embed_dim"] = tc["hidden_size"]
    if tc.get("layer_types") is None:
        k = tc.get("full_attention_interval", 4)
        tc["layer_types"] = ["linear_attention" if (i + 1) % k else "full_attention"
                             for i in range(tc["num_hidden_layers"])]
    # transformers renames "full_attention" to "indexed_attention"; keep
    # the checkpoint's spelling here and treat both as full attention.
    tc["layer_types"] = ["full_attention" if t in ("full_attention", "indexed_attention") else t
                         for t in tc["layer_types"]]
    return tc


def ple_layers(tc):
    """0-based decoder layers that carry a PLE module. `ple_layer_ids`
    is ONE-indexed (`layer_idx + 1 in ple_layer_ids`): id 2 -> layer 1."""
    return [i - 1 for i in tc["ple_layer_ids"]]


def quant_rule(cfg, base):
    """(bits, group_size) for a quantized tensor's base name (MLX name
    without .weight/.scales/.biases)."""
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    o = q.get(base)
    if isinstance(o, dict):
        return int(o["bits"]), int(o["group_size"])
    return int(q.get("bits", 4)), int(q.get("group_size", 64))


# --- tensors -------------------------------------------------------------------

class T:
    """One logical tensor. kind: "q" (quantized linear / embedding, shape
    is the unquantized [..., out, in]), "bf16", "i64"."""
    __slots__ = ("base", "shape", "kind", "role")

    def __init__(self, base, shape, kind, role):
        self.base, self.shape, self.kind, self.role = base, list(shape), kind, role

    def __repr__(self):
        return f"T({self.base}, {self.shape}, {self.kind}, {self.role})"


def tensors(cfg):
    """All logical tensors, MLX base names (prefix included). role is one
    of resident / expert / ngram (the converter's three outputs)."""
    tc = text_config(cfg)
    H, V = tc["hidden_size"], tc["vocab_size"]
    hc, R = tc["hc_count"], tc["hc_lowrank"]
    HH = hc * H
    out = []
    m = PREFIX + "model."

    def add(base, shape, kind, role="resident"):
        out.append(T(base, shape, kind, role))

    def gated_residual(p, combine=True):
        add(p + ".hc_norm.weight", [HH], "bf16")
        add(p + ".input_mix_weight_down", [R, HH], "q")
        add(p + ".input_mix_weight_up", [HH, R], "q")
        if combine:
            add(p + ".block_inject_weight", [hc, HH], "q")

    add(m + "embed_tokens", [V, H], "q")
    gated_residual(m + "hyper_connection_mixer", combine=False)
    nk, nv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    dk, dv = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    conv_dim = 2 * nk * dk + nv * dv
    nh, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    ni, nik, idd = tc["indexer_n_heads"], tc["indexer_kv_heads"], tc["indexer_head_dim"]
    E, I, Is = tc["num_experts"], tc["moe_intermediate_size"], tc["shared_expert_intermediate_size"]
    ple = set(ple_layers(tc))
    for i, lt in enumerate(tc["layer_types"]):
        p = f"{m}layers.{i}."
        gated_residual(p + "attn_hyper_connection")
        gated_residual(p + "mlp_hyper_connection")
        if lt == "linear_attention":
            a = p + "linear_attn."
            add(a + "A_log", [nv], "bf16")
            add(a + "conv1d.weight", [conv_dim, tc["linear_conv_kernel_dim"], 1], "bf16")
            add(a + "dt_bias", [nv], "bf16")
            add(a + "in_proj_a", [nv, H], "q")
            add(a + "in_proj_b", [nv, H], "q")
            add(a + "in_proj_qkv", [conv_dim, H], "q")
            add(a + "in_proj_z", [nv * dv, H], "q")
            add(a + "norm.weight", [dv], "bf16")
            add(a + "out_proj", [H, nv * dv], "q")
        else:
            a = p + "self_attn."
            add(a + "q_proj", [nh * hd * 2, H], "q")
            add(a + "k_proj", [nkv * hd, H], "q")
            add(a + "v_proj", [nkv * hd, H], "q")
            add(a + "o_proj", [H, nh * hd], "q")
            add(a + "q_norm.weight", [hd], "bf16")
            add(a + "k_norm.weight", [hd], "bf16")
            add(a + "indexer.index_qk_proj", [(ni + nik) * idd, H], "q")
            add(a + "indexer.q_layernorm.weight", [idd], "bf16")
            add(a + "indexer.k_layernorm.weight", [idd], "bf16")
        mp = p + "mlp."
        add(mp + "gate", [E, H], "q")
        add(mp + "shared_expert_gate", [1, H], "q")
        add(mp + "shared_expert.gate_proj", [Is, H], "q")
        add(mp + "shared_expert.up_proj", [Is, H], "q")
        add(mp + "shared_expert.down_proj", [H, Is], "q")
        add(mp + "switch_mlp.gate_proj", [E, I, H], "q", "expert")
        add(mp + "switch_mlp.up_proj", [E, I, H], "q", "expert")
        add(mp + "switch_mlp.down_proj", [E, H, I], "q", "expert")
        if i in ple:
            pp = p + "ple."
            D = tc["ple_embed_dim"]
            heads = (tc["ngram_size"] - 1) * tc["heads_per_ngram"]
            add(pp + "conv1d.weight", [HH, tc["ple_conv_kernel_size"], 1], "bf16")
            add(pp + "key_proj", [HH, D], "q")
            add(pp + "value_proj", [H, D], "q")
            add(pp + "norm_conv.weight", [HH], "bf16")
            add(pp + "norm_key.weight", [HH], "bf16")
            add(pp + "norm_query.weight", [HH], "bf16")
            add(pp + "ple_embedding.layer_multipliers", [tc["ngram_size"]], "i64")
            add(pp + "ple_embedding.ngram_heads_offsets", [heads], "i64")
            add(pp + "ple_embedding.ngram_heads_vocab_sizes", [heads], "i64")
            _, _, _, padded = ngram_vocab(tc, sorted(ple).index(i))
            parts = tc["split_ngram_parts"]
            assert padded % parts == 0, f"padded n-gram vocab {padded} not divisible by {parts} shards"
            for s in range(parts):
                add(pp + f"ple_embedding.ngram_embedding.shards.{s}", [padded // parts, D // heads], "q", "ngram")
    add(PREFIX + "lm_head", [V, H], "q")
    return out


def stored(cfg, t):
    """[(MLX name, safetensors dtype, stored shape)] for a logical tensor."""
    if t.kind == "bf16":
        return [(t.base, "BF16", t.shape)]
    if t.kind == "i64":
        return [(t.base, "I64", t.shape)]
    bits, g = quant_rule(cfg, t.base)
    *lead, n_in = t.shape
    assert n_in % g == 0 and (n_in * bits) % 32 == 0, f"{t.base}: in {n_in} vs group {g}"
    return [(t.base + ".weight", "U32", lead + [n_in * bits // 32]),
            (t.base + ".scales", "BF16", lead + [n_in // g]),
            (t.base + ".biases", "BF16", lead + [n_in // g])]


ST_SIZE = {"BF16": 2, "U32": 4, "I64": 8, "F32": 4}


def nbytes(dtype, shape):
    n = ST_SIZE[dtype]
    for d in shape:
        n *= d
    return n


def stored_bytes(cfg, t):
    return sum(nbytes(d, s) for _, d, s in stored(cfg, t))


def expert_components(cfg, layer=0):
    """[(component, dtype, per-expert shape, bytes)] for one expert in
    streammoe's packed order (gate, up, down) x (weight, scales, biases)."""
    tc = text_config(cfg)
    comps = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        base = f"{PREFIX}model.layers.{layer}.mlp.switch_mlp.{proj}"
        t = next(x for x in tensors(cfg) if x.base == base)
        for name, dt, shape in stored(cfg, t):
            comp = proj + "." + name.rsplit(".", 1)[1]
            per = shape[1:]
            comps.append((comp, dt, per, nbytes(dt, per)))
    return comps


def round_up(n, a):
    return (n + a - 1) // a * a


def size_report(cfg, expert_align=16384):
    tc = text_config(cfg)
    ts = tensors(cfg)
    by = {"resident": 0, "expert": 0, "ngram": 0}
    embed = lm_head = 0
    for t in ts:
        b = stored_bytes(cfg, t)
        by[t.role] += b
        if t.base.endswith("embed_tokens"):
            embed = b
        if t.base.endswith("lm_head"):
            lm_head = b
    comps = expert_components(cfg)
    es = sum(c[3] for c in comps)
    stride = round_up(es, expert_align) if expert_align else es
    L, E = tc["num_hidden_layers"], tc["num_experts"]
    sizes, _, total, padded = ngram_vocab(tc) if tc["ple_layer_ids"] else ([], [], 0, 0)
    gib = 2 ** 30
    lines = [
        f"expert blob: {es} B ({es / 2**20:.3f} MiB), stride {stride} B "
        f"({stride - es} B pad to {expert_align})",
        f"  components: " + ", ".join(f"{c[0]} {c[2]} {c[3]}" for c in comps),
        f"experts: {L} x {E} = {L * E} -> {L * E * stride / gib:.2f} GiB on disk "
        f"({L * E * es / gib:.2f} GiB unpadded)",
        f"resident (model_weights.bin, non-expert, non-ngram): {by['resident'] / gib:.3f} GiB "
        f"(embed_tokens {embed / gib:.3f} GiB, lm_head {lm_head / gib:.3f} GiB; "
        f"{(by['resident'] - embed) / gib:.3f} GiB with the embedding streamed)",
    ]
    if tc["ple_layer_ids"]:
        heads = (tc["ngram_size"] - 1) * tc["heads_per_ngram"]
        D = tc["ple_embed_dim"] // heads
        row = next(stored_bytes(cfg, t) for t in ts if t.role == "ngram") // (padded // tc["split_ngram_parts"])
        lines.append(f"n-gram table: {heads} heads, vocab sizes {sizes[0]}..{sizes[-1]}, total {total} rows "
                     f"(padded {padded}), row dim {D}, {row} B/row quantized -> "
                     f"{by['ngram'] / gib:.2f} GiB ({padded * D * 2 / gib:.2f} GiB in BF16), "
                     f"{tc['split_ngram_parts']} shards of {padded // tc['split_ngram_parts']} rows")
    lines.append(f"total: {(by['resident'] + L * E * stride + by['ngram']) / gib:.2f} GiB")
    return "\n".join(lines)


def check_index(cfg, weight_map):
    """Compare derived names with an index's weight_map (vision dropped)."""
    want = {}
    for t in tensors(cfg):
        for name, dt, shape in stored(cfg, t):
            want[name] = dt
    have = {n for n in weight_map if not (n.startswith("vision_tower") or ".visual." in n or n.startswith("visual"))}
    missing = sorted(set(want) - have)
    extra = sorted(have - set(want))
    return missing, extra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--index")
    ap.add_argument("--expert-align", type=int, default=16384)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    print(size_report(cfg, a.expert_align))
    if a.index:
        wm = json.load(open(a.index))["weight_map"]
        missing, extra = check_index(cfg, wm)
        print(f"index check: {len(missing)} derived names missing from the index, {len(extra)} index names not derived")
        for n in missing[:10]:
            print("  missing:", n)
        for n in extra[:10]:
            print("  extra:", n)
        if missing or extra:
            sys.exit(1)


if __name__ == "__main__":
    main()
