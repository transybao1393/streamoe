#!/usr/bin/env python3
"""Independent NumPy reference forward for OLMoE, reading the same
converted artifacts streammoe loads (model_weights.{bin,json} +
packed_experts/), so a streammoe run can be checked against it exactly.

Shares no code with streammoe. Follows mlx-lm's olmoe.py:
  - q_norm / k_norm are RMSNorm over the whole projection, before the
    split into heads;
  - RoPE is rotate-half over the full head dim;
  - router softmax, top-k, *no* renormalisation (norm_topk_prob=False);
  - no shared expert; RMSNorm eps from the manifest (1e-5).

Runs one token at a time with a KV cache; only the current layer's
weights are dequantized, so it stays around 1 GB of RAM.

    python -I tools/olmoe_ref.py --artifacts <dir> --root <dir> \
        --tokens 50279,... --decode 32 --out ref.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def bf16(a):
    return (np.asarray(a, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def dequant(w, s, b, bits=4, group=64):
    per = 32 // bits
    q = (w[..., None] >> (np.arange(per, dtype=np.uint32) * bits)) & ((1 << bits) - 1)
    q = q.reshape(*w.shape[:-1], -1).astype(np.float32)
    q = q.reshape(*q.shape[:-1], q.shape[-1] // group, group)
    return (q * s[..., None] + b[..., None]).reshape(*w.shape[:-1], -1)


class Weights:
    def __init__(self, artifacts):
        art = Path(artifacts)
        self.man = json.loads((art / "model_weights.json").read_text())
        self.cfg = self.man["config"]
        self.mm = np.memmap(art / "model_weights.bin", dtype=np.uint8, mode="r")

    def raw(self, name):
        t = self.man["tensors"][name]
        dt = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32}[t["dtype"]]
        return self.mm[t["offset"]:t["offset"] + t["size"]].view(dt).reshape(t["shape"]), t["dtype"]

    def vec(self, name):
        a, dt = self.raw(name)
        return bf16(a) if dt == "BF16" else np.asarray(a, np.float32)

    def linear(self, prefix, rows=None):
        w, _ = self.raw(prefix + ".weight")
        s, _ = self.raw(prefix + ".scales")
        b, _ = self.raw(prefix + ".biases")
        if rows is not None:
            w, s, b = w[rows], s[rows], b[rows]
        return dequant(np.asarray(w), bf16(s), bf16(b))


class Experts:
    def __init__(self, root):
        pk = Path(root) / "packed_experts"
        self.layout = json.loads((pk / "layout.json").read_text())
        self.size = self.layout["expert_size"]
        self.files = [np.memmap(pk / f"layer_{l:02d}.bin", dtype=np.uint8, mode="r")
                      for l in range(self.layout["num_layers"])]

    def get(self, layer, e):
        blob = self.files[layer][e * self.size:(e + 1) * self.size]
        c = {x["name"]: blob[x["offset"]:x["offset"] + x["size"]].view(
            np.uint32 if x["dtype"] == "U32" else np.uint16).reshape(x["shape"])
             for x in self.layout["components"]}
        m = lambda p: dequant(np.asarray(c[p + ".weight"]), bf16(c[p + ".scales"]), bf16(c[p + ".biases"]))
        return m("gate_proj"), m("up_proj"), m("down_proj")


def rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w


def rope(x, pos, theta):
    d = x.shape[-1]
    half = d // 2
    inv = 1.0 / theta ** (np.arange(half, dtype=np.float64) * 2 / d)
    ang = pos * inv
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


class Model:
    def __init__(self, w, ex):
        self.w, self.ex = w, ex
        c = w.cfg
        self.L, self.H, self.D = c["num_hidden_layers"], c["num_attention_heads"], c["head_dim"]
        self.KV, self.K = c["num_key_value_heads"], c["num_experts_per_tok"]
        self.eps, self.theta = c["rms_norm_eps"], c["rope_theta"]
        self.renorm = c.get("norm_topk_prob", False)
        self.k = [[] for _ in range(self.L)]
        self.v = [[] for _ in range(self.L)]
        self.routing = []

    def step(self, tok, pos):
        w = self.w
        x = w.linear("model.embed_tokens", rows=np.array([tok]))[0]
        picks = []
        for l in range(self.L):
            p = f"model.layers.{l}"
            h = rms(x, w.vec(p + ".input_layernorm.weight"), self.eps)
            q = w.linear(p + ".self_attn.q_proj") @ h
            k = w.linear(p + ".self_attn.k_proj") @ h
            v = w.linear(p + ".self_attn.v_proj") @ h
            q = rms(q, w.vec(p + ".self_attn.q_norm.weight"), self.eps).reshape(self.H, self.D)
            k = rms(k, w.vec(p + ".self_attn.k_norm.weight"), self.eps).reshape(self.KV, self.D)
            q, k = rope(q, pos, self.theta), rope(k, pos, self.theta)
            self.k[l].append(k)
            self.v[l].append(v.reshape(self.KV, self.D))
            K, V = np.stack(self.k[l]), np.stack(self.v[l])         # [S, KV, D]
            rep = self.H // self.KV
            Kr, Vr = np.repeat(K, rep, 1), np.repeat(V, rep, 1)
            sc = np.einsum("hd,shd->hs", q, Kr) / np.sqrt(self.D)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            sc /= sc.sum(-1, keepdims=True)
            attn = np.einsum("hs,shd->hd", sc, Vr).reshape(-1)
            x = x + w.linear(p + ".self_attn.o_proj") @ attn

            h = rms(x, w.vec(p + ".post_attention_layernorm.weight"), self.eps)
            lg = w.linear(p + ".mlp.gate") @ h
            pr = np.exp(lg - lg.max())
            pr /= pr.sum()
            top = np.argsort(-pr, kind="stable")[: self.K]
            wt = pr[top] / (pr[top].sum() if self.renorm else 1.0)
            picks.append([int(e) for e in top])
            out = np.zeros_like(x)
            for e, we in zip(top, wt):
                G, U, Dn = self.ex.get(l, int(e))
                a = G @ h
                out += we * (Dn @ (a / (1 + np.exp(-a)) * (U @ h)))
            x = x + out
        self.routing.append(picks)
        h = rms(x, w.vec("model.norm.weight"), self.eps)
        return w.linear("lm_head") @ h


def top(lg, k):
    i = np.argsort(-lg, kind="stable")[:k]
    return [[int(j), float(lg[j])] for j in i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokens", required=True, help="comma-separated prompt ids")
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--eos", default="50279")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = Model(Weights(a.artifacts), Experts(a.root))
    prompt = [int(t) for t in a.tokens.split(",")]
    eos = {int(t) for t in a.eos.split(",")}
    t0 = time.time()
    for i, t in enumerate(prompt):
        lg = m.step(t, i)
    steps, toks = [{"top": top(lg, 10)}], []
    for i in range(a.decode):
        t = int(np.argmax(lg))
        toks.append(t)
        if t in eos:
            break
        lg = m.step(t, len(prompt) + i)
        steps.append({"top": top(lg, 10)})
    Path(a.out).write_text(json.dumps({"prompt": prompt, "tokens": toks, "steps": steps, "routing": m.routing}))
    print(f"{len(prompt)} prompt + {len(toks)} decoded in {time.time() - t0:.0f}s: {toks}")


if __name__ == "__main__":
    main()
