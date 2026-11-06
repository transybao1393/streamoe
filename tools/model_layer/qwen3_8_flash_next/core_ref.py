#!/usr/bin/env python3
"""Torch port of the Qwen3.8-Flash-Next (qwen4_exp) text core, for tests.

A line-by-line port of the parts of transformers' `modeling_qwen4_exp.py`
(main, targets transformers 5.8) the core engine implements: the grouped
`Qwen4ExpTextRMSNorm` (gain `1 + w`), `Qwen4ExpTextGatedResidual`, the
Gated DeltaNet (recurrent form, sigmoid output gate), dense GQA attention
(QSA with every block selected: exact while the context is within the
indexer budget), the sparse MoE block and the model forward (repeat x
hc_count, layers, final `hyper_connection_mixer`, no final norm, LM head).
PLE is left out (`ple_layer_ids = []`): it has its own agent and tests.

Weights are not stored anywhere. Every tensor is generated from its name
with the same splitmix64 stream the Rust test uses
(`qwen3_8_flash_next::tiny::gen_tensor`), so the fixture only carries the
config, the inputs and the expected outputs.

    python3 -I tools/model_layer/qwen3_8_flash_next/core_ref.py \
        crates/streammoe/tests/fixtures/q38_core_ref.json

Runs in float64 so the fixture is the exact math; the Rust side (f32)
compares with a tolerance.
"""
import json
import math
import sys

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

MASK64 = (1 << 64) - 1


def fnv1a64(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode():
        h ^= b
        h = (h * 0x100000001B3) & MASK64
    return h


def splitmix64_next(state: int):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def gen(name: str, shape, scale: float, offset: float = 0.0) -> torch.Tensor:
    """`offset + scale * u`, u uniform in [-1, 1) on a 2^-23 grid (exact in f32)."""
    n = 1
    for d in shape:
        n *= d
    state = fnv1a64(name)
    out = []
    for _ in range(n):
        state, z = splitmix64_next(state)
        u = (z >> 40) / float(1 << 24)  # [0, 1), 24 bits
        out.append(offset + scale * (2.0 * u - 1.0))
    return torch.tensor(out).reshape(shape)


TINY = dict(
    hidden=32, hc_count=4, hc_lowrank=8, num_layers=4, full_attn_interval=4,
    linear_k_heads=2, linear_v_heads=4, linear_key_dim=8, linear_value_dim=8, conv_kernel=4,
    attn_heads=4, kv_heads=2, head_dim=16, rotary_dim=4, rope_theta=10000.0,
    num_experts=8, top_k=2, moe_intermediate=16, shared_intermediate=16,
    vocab=64, rms_eps=1e-6,
)


# --- Ported modules -------------------------------------------------------

def qwen_rms_norm(x, w, group_size, eps):
    """Qwen4ExpTextRMSNorm: per-group RMS, gain (1 + w)."""
    if group_size is not None:
        x = x.reshape(*x.shape[:-1], -1, group_size)
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if group_size is not None:
        out = out.flatten(-2)
    return out * (1.0 + w)


def gated_residual(cfg, p, hyper, use_combine=True):
    """Qwen4ExpTextGatedResidual.forward."""
    hc, H = cfg["hc_count"], cfg["hidden"]
    normed = qwen_rms_norm(hyper, p["hc_norm"], H, cfg["rms_eps"])
    mix = F.silu((normed @ p["down"].T) / hc)
    mix = torch.sigmoid(mix @ p["up"].T)
    mix = mix.unflatten(-1, (hc, H))
    mixed = (mix * normed.unflatten(-1, (hc, H))).mean(dim=-2)
    if not use_combine:
        return mixed
    inj = 2 * torch.sigmoid((normed @ p["inject"].T) / hc)
    return mixed, hyper, inj


def inject(hyper, block_out, inj):
    injection = block_out.unsqueeze(-2) * inj.unsqueeze(-1)
    return hyper + injection.flatten(-2)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class GdnState:
    def __init__(self, cfg):
        conv_dim = 2 * cfg["linear_k_heads"] * cfg["linear_key_dim"] + cfg["linear_v_heads"] * cfg["linear_value_dim"]
        self.conv = torch.zeros(conv_dim, cfg["conv_kernel"] - 1)
        self.rec = torch.zeros(cfg["linear_v_heads"], cfg["linear_key_dim"], cfg["linear_value_dim"])


def gdn_step(cfg, p, x, st: GdnState):
    """Qwen4ExpTextGatedDeltaNet, one token, cached (causal_conv1d_update +
    torch_recurrent_gated_delta_rule), output gate = sigmoid."""
    Hk, Hv = cfg["linear_k_heads"], cfg["linear_v_heads"]
    dk, dv = cfg["linear_key_dim"], cfg["linear_value_dim"]
    key_dim, value_dim = Hk * dk, Hv * dv
    qkv = p["in_qkv"] @ x
    z = (p["in_z"] @ x).reshape(Hv, dv)
    b = p["in_b"] @ x
    a = p["in_a"] @ x
    # causal_conv1d_update: window = [state | new], weight [C, K], silu.
    window = torch.cat([st.conv, qkv[:, None]], dim=-1)
    st.conv = window[:, 1:].clone()
    qkv = F.silu((window * p["conv1d"]).sum(-1))
    q, k, v = torch.split(qkv, [key_dim, key_dim, value_dim])
    q, k, v = q.reshape(Hk, dk), k.reshape(Hk, dk), v.reshape(Hv, dv)
    beta = torch.sigmoid(b)
    g = -p["A_log"].exp() * F.softplus(a + p["dt_bias"])
    rep = Hv // Hk
    q = q.repeat_interleave(rep, dim=0)
    k = k.repeat_interleave(rep, dim=0)
    q = l2norm(q)
    k = l2norm(k)
    q = q / math.sqrt(dk)
    out = torch.zeros(Hv, dv)
    for h in range(Hv):
        S = st.rec[h] * g[h].exp()
        kv_mem = (S * k[h][:, None]).sum(dim=0)
        delta = (v[h] - kv_mem) * beta[h]
        S = S + k[h][:, None] * delta[None, :]
        st.rec[h] = S
        out[h] = (S * q[h][:, None]).sum(dim=0)
    # Qwen4ExpTextRMSNormGated with activation = sigmoid, plain weight.
    var = out.pow(2).mean(-1, keepdim=True)
    out = out * torch.rsqrt(var + cfg["rms_eps"])
    out = p["norm"] * out
    out = out * torch.sigmoid(z)
    return p["out_proj"] @ out.reshape(-1)


def rope_tables(cfg, pos):
    rd = cfg["rotary_dim"]
    inv_freq = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
    freqs = pos * inv_freq
    emb = torch.cat([freqs, freqs])
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    xr = xr * cos + rotate_half(xr) * sin
    return torch.cat([xr, xp], dim=-1)


class AttnState:
    def __init__(self):
        self.k = []
        self.v = []


def attn_step(cfg, p, x, pos, st: AttnState):
    """Qwen4ExpTextAttention with every KV block selected (dense GQA)."""
    Hq, Hkv, D = cfg["attn_heads"], cfg["kv_heads"], cfg["head_dim"]
    qg = (p["q_proj"] @ x).view(Hq, 2 * D)
    q, gate = qg[:, :D], qg[:, D:].reshape(-1)
    q = qwen_rms_norm(q, p["q_norm"], None, cfg["rms_eps"])
    k = qwen_rms_norm((p["k_proj"] @ x).view(Hkv, D), p["k_norm"], None, cfg["rms_eps"])
    v = (p["v_proj"] @ x).view(Hkv, D)
    cos, sin = rope_tables(cfg, float(pos))
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    st.k.append(k)
    st.v.append(v)
    K = torch.stack(st.k, dim=1)  # [Hkv, T, D]
    V = torch.stack(st.v, dim=1)
    rep = Hq // Hkv
    K = K.repeat_interleave(rep, dim=0)
    V = V.repeat_interleave(rep, dim=0)
    scores = torch.einsum("hd,htd->ht", q, K) * (D ** -0.5)
    w = torch.softmax(scores, dim=-1)
    out = torch.einsum("ht,htd->hd", w, V).reshape(-1)
    out = out * torch.sigmoid(gate)
    return p["o_proj"] @ out


def moe(cfg, p, x):
    """Qwen4ExpTextSparseMoeBlock for one token."""
    shared = p["shared_down"] @ (F.silu(p["shared_gate_proj"] @ x) * (p["shared_up"] @ x))
    logits = p["router"] @ x
    probs = torch.softmax(logits, dim=-1)
    top_v, top_i = torch.topk(probs, cfg["top_k"])
    top_v = top_v / top_v.sum()
    out = torch.zeros_like(x)
    for wgt, e in zip(top_v.tolist(), top_i.tolist()):
        h = F.silu(p[f"e{e}_gate"] @ x) * (p[f"e{e}_up"] @ x)
        out = out + wgt * (p[f"e{e}_down"] @ h)
    shared = torch.sigmoid(p["shared_expert_gate"] @ x) * shared
    return out + shared, top_i.tolist()


# --- Weights (names shared with the Rust generator) -----------------------

def hc_weights(cfg, pre, combine=True):
    hc, H, R = cfg["hc_count"], cfg["hidden"], cfg["hc_lowrank"]
    W = hc * H
    p = {
        "hc_norm": gen(pre + ".hc_norm", [W], 0.2),
        "down": gen(pre + ".down", [R, W], 1.0 / math.sqrt(W)),
        "up": gen(pre + ".up", [W, R], 2.0 / math.sqrt(R)),
    }
    if combine:
        p["inject"] = gen(pre + ".inject", [hc, W], 1.0 / math.sqrt(W))
    return p


def layer_weights(cfg, l):
    H = cfg["hidden"]
    pre = f"layers.{l}"
    p = {"attn_hc": hc_weights(cfg, pre + ".attn_hc"), "mlp_hc": hc_weights(cfg, pre + ".mlp_hc")}
    s = 1.0 / math.sqrt(H)
    if (l + 1) % cfg["full_attn_interval"] != 0:
        Hk, Hv = cfg["linear_k_heads"], cfg["linear_v_heads"]
        dk, dv = cfg["linear_key_dim"], cfg["linear_value_dim"]
        C = 2 * Hk * dk + Hv * dv
        g = pre + ".gdn"
        p["gdn"] = {
            "in_qkv": gen(g + ".in_qkv", [C, H], s),
            "in_z": gen(g + ".in_z", [Hv * dv, H], s),
            "in_b": gen(g + ".in_b", [Hv, H], s),
            "in_a": gen(g + ".in_a", [Hv, H], s),
            "conv1d": gen(g + ".conv1d", [C, cfg["conv_kernel"]], 0.5),
            "A_log": gen(g + ".A_log", [Hv], 1.0),
            "dt_bias": gen(g + ".dt_bias", [Hv], 0.5),
            "norm": gen(g + ".norm", [dv], 0.2, 1.0),
            "out_proj": gen(g + ".out_proj", [H, Hv * dv], 1.0 / math.sqrt(Hv * dv)),
        }
    else:
        Hq, Hkv, D = cfg["attn_heads"], cfg["kv_heads"], cfg["head_dim"]
        a = pre + ".attn"
        p["attn"] = {
            "q_proj": gen(a + ".q_proj", [2 * Hq * D, H], s),
            "k_proj": gen(a + ".k_proj", [Hkv * D, H], s),
            "v_proj": gen(a + ".v_proj", [Hkv * D, H], s),
            "o_proj": gen(a + ".o_proj", [H, Hq * D], 1.0 / math.sqrt(Hq * D)),
            "q_norm": gen(a + ".q_norm", [D], 0.2),
            "k_norm": gen(a + ".k_norm", [D], 0.2),
        }
    m = pre + ".moe"
    E, I, Is = cfg["num_experts"], cfg["moe_intermediate"], cfg["shared_intermediate"]
    mp = {
        "router": gen(m + ".router", [E, H], 4.0 * s),
        "shared_gate_proj": gen(m + ".shared.gate", [Is, H], s),
        "shared_up": gen(m + ".shared.up", [Is, H], s),
        "shared_down": gen(m + ".shared.down", [H, Is], 1.0 / math.sqrt(Is)),
        "shared_expert_gate": gen(m + ".shared_expert_gate", [1, H], s),
    }
    for e in range(E):
        mp[f"e{e}_gate"] = gen(f"{m}.experts.{e}.gate", [I, H], s)
        mp[f"e{e}_up"] = gen(f"{m}.experts.{e}.up", [I, H], s)
        mp[f"e{e}_down"] = gen(f"{m}.experts.{e}.down", [H, I], 1.0 / math.sqrt(I))
    p["moe"] = mp
    return p


def model_forward(cfg, tokens):
    H, V = cfg["hidden"], cfg["vocab"]
    embed = gen("embed_tokens", [V, H], 1.0)
    lm_head = gen("lm_head", [V, H], 1.0 / math.sqrt(H))
    mixer = hc_weights(cfg, "mixer", combine=False)
    layers = [layer_weights(cfg, l) for l in range(cfg["num_layers"])]
    states = [GdnState(cfg) if "gdn" in p else AttnState() for p in layers]
    logits_all, routes = [], []
    for pos, t in enumerate(tokens):
        hyper = embed[t].repeat(cfg["hc_count"])
        for l, p in enumerate(layers):
            x, hyper_in, inj = gated_residual(cfg, p["attn_hc"], hyper)
            if "gdn" in p:
                y = gdn_step(cfg, p["gdn"], x, states[l])
            else:
                y = attn_step(cfg, p["attn"], x, pos, states[l])
            hyper = inject(hyper_in, y, inj)
            x, hyper_in, inj = gated_residual(cfg, p["mlp_hc"], hyper)
            y, top = moe(cfg, p["moe"], x)
            routes.append(sorted(top))
            hyper = inject(hyper_in, y, inj)
        h = gated_residual(cfg, mixer, hyper, use_combine=False)
        logits_all.append((lm_head @ h).tolist())
    return logits_all, routes


def main():
    out_path = sys.argv[1]
    cfg = TINY
    W = cfg["hc_count"] * cfg["hidden"]
    # Standalone Gated Residual case.
    gr = hc_weights(cfg, "gr_case")
    hyper = gen("gr_case.hyper", [W], 1.5)
    normed = qwen_rms_norm(hyper, gr["hc_norm"], cfg["hidden"], cfg["rms_eps"])
    mixed, _, inj = gated_residual(cfg, gr, hyper)
    block = gen("gr_case.block", [cfg["hidden"]], 1.0)
    injected = inject(hyper, block, inj)
    final_mixed = gated_residual(cfg, gr, hyper, use_combine=False)

    tokens = [3, 17, 42, 5, 63, 0, 17, 42]
    logits, routes = model_forward(cfg, tokens)
    fixture = {
        "source": "tools/model_layer/qwen3_8_flash_next/core_ref.py",
        "config": cfg,
        "gated_residual": {
            "normed": normed.tolist(),
            "mixed": mixed.tolist(),
            "inj": inj.tolist(),
            "injected": injected.tolist(),
            "final_mixed": final_mixed.tolist(),
        },
        "tokens": tokens,
        "logits": logits,
        "routes": routes,
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f)
        f.write("\n")
    print(f"wrote {out_path}: {len(tokens)} tokens, logits[0][:4] = {logits[0][:4]}")


if __name__ == "__main__":
    main()
