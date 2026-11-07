#!/usr/bin/env python3
"""Score decode-prefetch guessers offline on a STREAMMOE_GUESS_TRACE.

The trace (crates/streammoe/src/riir/io/guess_trace.rs) holds, per
decode (token, layer): the residual the next layer's guess starts from
(`h_mid` on the qwen3_5_moe path, the MoE input `mixed` on Flash-Next),
the router's picks and weights, and which of the layer's experts were
in the hot tier before its reads. Each guesser ranks layer L's experts;
the engine masks hot-tier experts and reads the top N of the rest, so
a guess is scored only on non-hot experts:

    hits  = |top-N non-hot guesses  ∩  picks|
    reads = N (or fewer if the guesser ranks fewer non-hot experts)
    cover = hits / non-hot picks (the demand reads a perfect guess saves)

Guessers:
    prev        previous token's picks for L, strongest first (PrevToken)
    cross       L's router on layer L-1's residual (CrossLayer; on the
                qwen3_5_moe path through L's post-attention norm first)
    cross2      L's router on layer L-2's residual (two layers ahead)
    trans       online transition table: sum over L-1's picks e' of
                count(e' at L-1 -> e at L), learned from earlier tokens
    cross+prev  cross, with this layer's previous-token picks boosted
    cross>t     cross, only guesses the router gives probability >= t
    oracle      the actual non-hot picks (upper bound)

    python3 -I tools/sim_prefetch_guess.py TRACE MODEL_DIR --norm post_attention_layernorm
    python3 -I tools/sim_prefetch_guess.py TRACE MODEL_DIR            (Flash-Next: no norm)
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_trace(path):
    raw = Path(path).read_bytes()
    recs, off = [], 0
    while off + 16 <= len(raw):
        layer, k, hidden, n_exp = np.frombuffer(raw, np.uint32, 4, off)
        off += 16
        x = np.frombuffer(raw, np.float32, hidden, off).copy()
        off += 4 * hidden
        picks = np.frombuffer(raw, np.int32, k, off).copy()
        off += 4 * k
        w = np.frombuffer(raw, np.float32, k, off).copy()
        off += 4 * k
        nb = (n_exp + 7) // 8
        bits = np.unpackbits(np.frombuffer(raw, np.uint8, nb, off), bitorder="little")[:n_exp].astype(bool)
        off += nb
        recs.append((int(layer), x, picks[np.argsort(-w, kind="stable")], bits))
    # Group into tokens: a token starts at each layer-0 record.
    toks, cur = [], None
    for layer, x, picks, hot in recs:
        if layer == 0 or cur is None:
            cur = {}
            toks.append(cur)
        cur[layer] = (x, picks, hot)
    return toks


def bf16(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


class Weights:
    def __init__(self, model_dir):
        art = Path(model_dir) / "artifacts"
        man = json.loads((art / "model_weights.json").read_text())
        self.t = man.get("tensors", man)
        self.f = open(art / "model_weights.bin", "rb")

    def raw(self, name, dtype):
        info = self.t[name]
        self.f.seek(info["offset"])
        return np.frombuffer(self.f.read(info["size"]), dtype), info

    def gate(self, layer):
        p = f"model.layers.{layer}.mlp.gate."
        w, info = self.raw(p + "weight", np.uint32)
        bits, gs = info.get("bits", 4), info.get("group_size", 64)
        rows = info["shape"][0]
        per = 32 // bits
        q = ((w[:, None] >> (np.arange(per, dtype=np.uint32) * bits)) & ((1 << bits) - 1)).reshape(rows, -1)
        s = bf16(self.raw(p + "scales", np.uint16)[0]).reshape(rows, -1)
        b = bf16(self.raw(p + "biases", np.uint16)[0]).reshape(rows, -1)
        g = q.shape[1] // gs
        return (q.reshape(rows, g, gs) * s[:, :, None] + b[:, :, None]).reshape(rows, -1).astype(np.float32)

    def norm(self, layer, name):
        return bf16(self.raw(f"model.layers.{layer}.{name}.weight", np.uint16)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("model_dir")
    ap.add_argument("--norm", default=None, help="norm applied before the router (qwen3_5_moe: post_attention_layernorm)")
    ap.add_argument("--budgets", default="1,2,3,4,6,10")
    ap.add_argument("--boost", type=float, default=0.05, help="cross+prev: probability added to previous-token picks")
    ap.add_argument("--taus", default="0.005,0.01,0.02", help="cross>t: only guesses the router gives at least t")
    ap.add_argument("--skip", type=int, default=1, help="leading decode tokens left out of the score")
    a = ap.parse_args()
    toks = load_trace(a.trace)
    layers = sorted(toks[0])
    wts = Weights(a.model_dir)
    gates = {l: wts.gate(l) for l in layers}
    norms = {l: wts.norm(l, a.norm) for l in layers} if a.norm else None
    n_exp = gates[layers[0]].shape[0]

    def router_probs(layer, x):
        if norms is not None:
            x = x / np.sqrt(np.mean(x * x) + 1e-6) * norms[layer]
        z = gates[layer] @ x
        z = np.exp(z - z.max())
        return z / z.sum()

    budgets = [int(b) for b in a.budgets.split(",")]
    taus = [float(t) for t in a.taus.split(",")] if a.taus else []
    names = ["prev", "cross", "cross2", "trans", "cross+prev"] + [f"cross>{t:g}" for t in taus] + ["oracle"]
    hits = {(n, b): 0 for n in names for b in budgets}
    reads = {(n, b): 0 for n in names for b in budgets}
    nonhot_total = picks_total = scored = 0
    trans = {l: np.zeros((n_exp, n_exp), np.float32) for l in layers}
    for ti, tok in enumerate(toks):
        prev_tok = toks[ti - 1] if ti > 0 else None
        for l in layers:
            if l not in tok:
                continue
            x, picks, hot = tok[l]
            pickset = set(picks.tolist())
            nonhot = [e for e in picks.tolist() if not hot[e]]
            ranks = {}
            if prev_tok is not None and l in prev_tok:
                ranks["prev"] = prev_tok[l][1].tolist()
            if l - 1 in tok:
                p = router_probs(l, tok[l - 1][0])
                ranks["cross"] = np.argsort(-p, kind="stable").tolist()
                for t in taus:
                    ranks[f"cross>{t:g}"] = [e for e in ranks["cross"][:32] if p[e] >= t]
                if "prev" in ranks:
                    q = p.copy()
                    q[np.array(ranks["prev"])] += a.boost
                    ranks["cross+prev"] = np.argsort(-q, kind="stable").tolist()
                s = trans[l][tok[l - 1][1]].sum(0)
                if s.max() > 0:
                    ranks["trans"] = [int(e) for e in np.argsort(-s, kind="stable") if s[e] > 0]
            if l - 2 in tok:
                ranks["cross2"] = np.argsort(-router_probs(l, tok[l - 2][0]), kind="stable").tolist()
            ranks["oracle"] = nonhot
            if ti >= a.skip and l >= 2 and "prev" in ranks:
                scored += 1
                picks_total += len(picks)
                nonhot_total += len(nonhot)
                for n in names:
                    r = [e for e in ranks.get(n, []) if not hot[e]]
                    for b in budgets:
                        sel = r[:b]
                        reads[(n, b)] += len(sel)
                        hits[(n, b)] += sum(e in pickset for e in sel)
            if l - 1 in tok:
                for e1 in tok[l - 1][1]:
                    trans[l][e1, picks] += 1
    ntok = max(1, len(toks) - a.skip)
    print(f"{a.trace}: {len(toks)} tokens, {len(layers)} layers, {n_exp} experts; scored {scored} (token, layer) "
          f"from layer 2, tokens {a.skip}+")
    print(f"picks/token {picks_total / ntok:.1f}, non-hot (demand reads with no prefetch) {nonhot_total / ntok:.1f} "
          f"({100 * nonhot_total / max(1, picks_total):.1f}%)")
    print(f"{'guesser':<11} {'N':>3} {'reads/tok':>9} {'hits/tok':>8} {'precision':>9} {'cover':>6}")
    for n in names:
        for b in budgets:
            r, h = reads[(n, b)], hits[(n, b)]
            print(f"{n:<11} {b:>3} {r / ntok:>9.1f} {h / ntok:>8.1f} {100 * h / max(1, r):>8.1f}% "
                  f"{100 * h / max(1, nonhot_total):>5.1f}%")


if __name__ == "__main__":
    main()
