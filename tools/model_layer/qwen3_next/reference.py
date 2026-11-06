#!/usr/bin/env python3
"""Standalone pure-torch reference for the Qwen3-Next text model.

A port of transformers' `models/qwen3_next/modeling_qwen3_next.py`
(main branch) that runs on torch 2.7 without transformers (the host's
transformers 4.52 predates qwen3_next). The math is the original's;
what changed:

- No transformers classes: plain functions over a dict of float32
  weights with transformers' parameter names, a small cache object and
  eager attention. Batch size 1, no padding, no attention mask
  argument.
- Gated DeltaNet uses `torch_recurrent_gated_delta_rule` for prefill
  too (the chunked form is the same recurrence regrouped);
  `use_qk_l2norm_in_kernel=True` as the model passes it.
- `fix_query_key_value_ordering` is kept verbatim: the reference reads
  the fused `in_proj_qkvz` / `in_proj_ba` exactly as transformers does,
  so goldens made from it check the converter's re-layout.
- Rotary: `rope_type == "default"` only, partial (first
  `head_dim * partial_rotary_factor` channels, rotate_half pairing).
- Norms: `Qwen3NextRMSNorm` is `norm(x) * (1 + w)`. The MLX files hold
  `1 + w` (mlx-lm adds the 1 when converting), so `load_mlx` subtracts
  1 from those norms to get transformers' `w` (schema.NORMS_1PW). The
  gated norm (`linear_attn.norm`) is plain `w` in both.
- MoE: the `Qwen3NextExperts` loop per expert over a stacked
  `switch_mlp` tensor; softmax router in float32, top-k, renormalised
  (`norm_topk_prob`), sigmoid-gated shared expert.
- No MTP (the MLX build carries none; transformers ignores `mtp.*`).

Everything runs in float32; quantized MLX tensors are dequantized as
`q * scale + bias` (what the engine computes).

    python3 reference.py <mlx dir> --prompt 1,2,3 --decode 4
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx_io  # noqa: E402
import schema  # noqa: E402


# --- loading -------------------------------------------------------------------

def load_mlx(mlx_dir):
    """(config dict, {transformers-style name: float32 torch tensor}).

    Quantized tensors are dequantized; norms stored as `1 + w` are
    shifted back to `w`. Names are the MLX names (which are the
    transformers names, except `switch_mlp` for the stacked experts and
    the fused projections, which transformers also keeps fused)."""
    cfg = json.load(open(os.path.join(mlx_dir, "config.json")))
    idx_path = os.path.join(mlx_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        shards = sorted(set(json.load(open(idx_path))["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    files = [mlx_io.SafeFile(os.path.join(mlx_dir, s)) for s in shards]
    where = {n: f for f in files for n in f.names()}
    w = {}
    for t in schema.mlx_tensors(cfg):
        if t.kind in ("q", "e"):
            f = where[t.base + ".weight"]
            packed = f.array(t.base + ".weight")
            arr = mlx_io.affine_dequantize(packed, f.array(t.base + ".scales"),
                                           f.array(t.base + ".biases"), t.group, t.bits)
        else:
            f = where[t.base]
            a = f.array(t.base)
            dt = f.dtype(t.base)
            arr = mlx_io.bf16_to_f32(a) if dt == "BF16" else a.astype(np.float32)
            if t.base.endswith(schema.NORMS_1PW):
                arr = arr - np.float32(1.0)
        w[t.base] = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    for f in files:
        f.close()
    return cfg, w


# --- modules (functional) -----------------------------------------------------

def rms_norm_1pw(x, w, eps):
    """Qwen3NextRMSNorm: norm(x) * (1 + w)."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return out * (1.0 + w.float())


def rms_norm_gated(x, gate, w, eps):
    """Qwen3NextRMSNormGated: w * norm(x) * silu(gate)."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf * F.silu(gate.float())


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Partial rotary: the first cos.shape[-1] channels rotate."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rd = cos.shape[-1]
    q_rot, q_pass = q[..., :rd], q[..., rd:]
    k_rot, k_pass = k[..., :rd], k[..., rd:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def rope_cos_sin(c, positions):
    rd = int(c["hd"] * c["partial_rotary_factor"])
    inv_freq = 1.0 / (c["rope_theta"] ** (torch.arange(0, rd, 2, dtype=torch.float) / rd))
    freqs = positions[:, None].float() * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None], emb.sin()[None]          # [1, T, rd]


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state):
    """torch_recurrent_gated_delta_rule with use_qk_l2norm_in_kernel=True.
    Inputs [B, T, heads, dim]; returns ([B, T, nv, dv], state)."""
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)]
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query = query / (query.shape[-1] ** 0.5)
    if initial_state is None:
        state = torch.zeros(batch_size, num_v_heads, k_head_dim, v_head_dim)
    else:
        state = initial_state.clone()
    out = torch.zeros_like(value)
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        state = state * g[:, :, i].exp()[..., None, None]
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    return out.transpose(1, 2).contiguous(), state


def fix_query_key_value_ordering(c, mixed_qkvz, mixed_ba):
    """Verbatim from Qwen3NextGatedDeltaNet."""
    nk, nv, dk, dv = c["nk"], c["nv"], c["dk"], c["dv"]
    new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (nk, 2 * dk + 2 * dv * nv // nk)
    new_tensor_shape_ba = mixed_ba.size()[:-1] + (nk, 2 * nv // nk)
    mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
    mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
    split_arg_list_qkvz = [dk, dk, (nv // nk * dv), (nv // nk * dv)]
    split_arg_list_ba = [nv // nk, nv // nk]
    query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
    b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
    value = value.reshape(value.size(0), value.size(1), -1, dv)
    z = z.reshape(z.size(0), z.size(1), -1, dv)
    b = b.reshape(b.size(0), b.size(1), nv)
    a = a.reshape(a.size(0), a.size(1), nv)
    return query, key, value, z, b, a


def gated_delta_net(c, w, p, x, cache, li, trace=None):
    """Qwen3NextGatedDeltaNet.forward, x [1, T, H]."""
    T = x.shape[1]
    qkvz = x @ w[p + "in_proj_qkvz"].T
    ba = x @ w[p + "in_proj_ba"].T
    query, key, value, z, b, a = fix_query_key_value_ordering(c, qkvz, ba)
    query, key, value = (t.reshape(t.shape[0], t.shape[1], -1) for t in (query, key, value))
    mixed_qkv = torch.cat((query, key, value), dim=-1).transpose(1, 2)     # [1, C, T]
    if trace is not None:
        trace[f"layers.{li}.mixed_qkv"] = mixed_qkv.transpose(1, 2)[0].clone()
        trace[f"layers.{li}.z"] = z.reshape(T, -1).clone()
        trace[f"layers.{li}.b"] = b[0].clone()
        trace[f"layers.{li}.a"] = a[0].clone()
    K = c["K"]
    conv_w = w[p + "conv1d.weight"].reshape(c["conv_dim"], K)      # MLX [C, K, 1]
    prev = cache.conv[li] if cache.conv[li] is not None else torch.zeros(1, c["conv_dim"], K - 1)
    full = torch.cat([prev, mixed_qkv], dim=-1)
    cache.conv[li] = full[:, :, -(K - 1):].clone()
    out = F.conv1d(full, conv_w.unsqueeze(1), None, padding=0, groups=c["conv_dim"])
    mixed_qkv = F.silu(out[:, :, -T:]).transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [c["key_dim"], c["key_dim"], c["value_dim"]], dim=-1)
    query = query.reshape(1, T, -1, c["dk"])
    key = key.reshape(1, T, -1, c["dk"])
    value = value.reshape(1, T, -1, c["dv"])
    beta = b.sigmoid()
    g = -w[p + "A_log"].float().exp() * F.softplus(a.float() + w[p + "dt_bias"])
    r = c["nv"] // c["nk"]
    if r > 1:
        query = query.repeat_interleave(r, dim=2)
        key = key.repeat_interleave(r, dim=2)
    core, state = recurrent_gated_delta_rule(query, key, value, g, beta, cache.rec[li])
    cache.rec[li] = state
    zs = z.shape
    core = rms_norm_gated(core.reshape(-1, core.shape[-1]), z.reshape(-1, z.shape[-1]),
                          w[p + "norm.weight"], c["eps"])
    core = core.reshape(zs).reshape(1, T, -1)
    return core @ w[p + "out_proj"].T


def attention(c, w, p, x, positions, cache, li, trace=None):
    """Qwen3NextAttention.forward (eager), x [1, T, H]."""
    T = x.shape[1]
    hd = c["hd"]
    qg = (x @ w[p + "q_proj"].T).view(1, T, -1, hd * 2)
    query_states, gate = torch.chunk(qg, 2, dim=-1)
    gate = gate.reshape(1, T, -1)
    query_states = rms_norm_1pw(query_states.reshape(1, T, -1, hd), w[p + "q_norm.weight"], c["eps"]).transpose(1, 2)
    key_states = rms_norm_1pw((x @ w[p + "k_proj"].T).view(1, T, -1, hd), w[p + "k_norm.weight"], c["eps"]).transpose(1, 2)
    value_states = (x @ w[p + "v_proj"].T).view(1, T, -1, hd).transpose(1, 2)
    cos, sin = rope_cos_sin(c, positions)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if cache.k[li] is not None:
        key_states = torch.cat([cache.k[li], key_states], dim=2)
        value_states = torch.cat([cache.v[li], value_states], dim=2)
    cache.k[li], cache.v[li] = key_states, value_states
    n_rep = c["nh"] // c["nkv"]
    ks = key_states.repeat_interleave(n_rep, dim=1)
    vs = value_states.repeat_interleave(n_rep, dim=1)
    S = ks.shape[2]
    scores = (query_states @ ks.transpose(2, 3)) * (hd ** -0.5)
    past = S - T
    mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=past)
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1)
    out = (probs @ vs).transpose(1, 2).reshape(1, T, -1)
    out = out * torch.sigmoid(gate)
    return out @ w[p + "o_proj"].T


def moe(c, w, p, x, trace=None, li=None):
    """Qwen3NextSparseMoeBlock.forward, x [1, T, H]."""
    hs = x.reshape(-1, c["H"])
    shared = (F.silu(hs @ w[p + "shared_expert.gate_proj"].T) * (hs @ w[p + "shared_expert.up_proj"].T)) \
        @ w[p + "shared_expert.down_proj"].T
    logits = hs @ w[p + "gate"].T
    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    top_v, top_i = torch.topk(probs, c["topk"], dim=-1)
    top_v = top_v / top_v.sum(dim=-1, keepdim=True)
    if trace is not None:
        srt = torch.sort(probs, dim=-1, descending=True).values
        trace.setdefault("router_margin", []).append(float((srt[:, c["topk"] - 1] - srt[:, c["topk"]]).min()))
    out = torch.zeros_like(hs)
    gw, uw, dw = w[p + "switch_mlp.gate_proj"], w[p + "switch_mlp.up_proj"], w[p + "switch_mlp.down_proj"]
    for t in range(hs.shape[0]):
        for k in range(c["topk"]):
            e = int(top_i[t, k])
            h = F.silu(hs[t] @ gw[e].T) * (hs[t] @ uw[e].T)
            out[t] += top_v[t, k] * (h @ dw[e].T)
    shared = torch.sigmoid(hs @ w[p + "shared_expert_gate"].T) * shared
    return (out + shared).reshape(x.shape)


class Cache:
    def __init__(self, L):
        self.conv = [None] * L
        self.rec = [None] * L
        self.k = [None] * L
        self.v = [None] * L
        self.pos = 0


def model_config(cfg):
    d = schema.dims(cfg)
    tc = schema.text_config(cfg)
    rs = tc.get("rope_parameters") or {}
    d["eps"] = tc.get("rms_norm_eps", 1e-6)
    d["rope_theta"] = float(rs.get("rope_theta", tc.get("rope_theta", 10000.0)))
    d["partial_rotary_factor"] = float(rs.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)))
    d["full"] = [schema.is_full(cfg, i) for i in range(d["L"])]
    return d


def forward(c, w, ids, cache, trace=None):
    """Logits [T, V] for `ids` continuing `cache` (updated in place)."""
    T = len(ids)
    positions = torch.arange(cache.pos, cache.pos + T)
    x = w["model.embed_tokens"][torch.tensor(ids)][None]
    for li in range(c["L"]):
        p = f"model.layers.{li}."
        h = rms_norm_1pw(x, w[p + "input_layernorm.weight"], c["eps"])
        if c["full"][li]:
            h = attention(c, w, p + "self_attn.", h, positions, cache, li, trace)
        else:
            h = gated_delta_net(c, w, p + "linear_attn.", h, cache, li, trace)
        x = x + h
        h = rms_norm_1pw(x, w[p + "post_attention_layernorm.weight"], c["eps"])
        x = x + moe(c, w, p + "mlp.", h, trace, li)
        if trace is not None:
            trace[f"layers.{li}.out"] = x[0].clone()
    cache.pos += T
    x = rms_norm_1pw(x, w["model.norm.weight"], c["eps"])
    head = w["lm_head"] if "lm_head" in w else w["model.embed_tokens"]
    return (x @ head.T)[0]


def generate(c, w, ids, n_decode, trace=None):
    """(prefill logits [T, V], decode tokens [n], decode logits [n, V]).
    Token 0 is the argmax of the last prefill row; decode row k is the
    logits after feeding token k."""
    cache = Cache(c["L"])
    pre = forward(c, w, ids, cache, trace)
    toks, rows = [], []
    nxt = int(pre[-1].argmax())
    for _ in range(n_decode):
        toks.append(nxt)
        dt = {} if trace is not None else None     # decode: router margins only
        row = forward(c, w, [nxt], cache, dt)[-1]
        if dt is not None:
            trace.setdefault("router_margin", []).extend(dt.get("router_margin", []))
        rows.append(row)
        nxt = int(row.argmax())
    return pre, toks, torch.stack(rows) if rows else torch.zeros(0, pre.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlx_dir")
    ap.add_argument("--prompt", default="1,2,3")
    ap.add_argument("--decode", type=int, default=4)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    cfg, w = load_mlx(a.mlx_dir)
    c = model_config(cfg)
    ids = [int(t) for t in a.prompt.split(",")]
    pre, toks, _ = generate(c, w, ids, a.decode)
    print("prefill argmax", pre.argmax(-1).tolist())
    print("decode", toks)


if __name__ == "__main__":
    main()
