#!/usr/bin/env python3
"""Replay streammoe routing traces against expert predictors and caches.

Trace lines (STREAMMOE_ROUTING_TRACE):
    <phase> <token> <layer> e0 … e{k-1} [: p0 … p{E-1}]
phase `p` (prefill: picks only, warms history) or `d` (decode: picks
plus the router's full softmax). Every metric is scored on decode.

Each decode (token, layer) is modelled the way streammoe serves it:
    H = hot tier (resident, fixed capacity)
    P = predicted set, prefetched where not in H
    reads = |P \\ H| (prefetch) + |A \\ (H ∪ P)| (demand misses)
with A the experts the router actually picks.

Predictors:
    prev            previous token's picks (streammoe's PrevToken guess)
    mrs-λ/M         moving router score: S ← λ·S + p(token); predict the
                    top-M of S. Short retention: λ ≤ 0.7 forgets in a few
                    tokens. Uses all E probabilities, so near-misses
                    (ranked just below the top-k) are predicted too.
Hot tiers (capacity C experts across all layers):
    none, lfu (streammoe's frequency tier), lru (bounded page cache), mrs-λ (keep the top-C experts
    by moving router score — refilled every token, refills are reads)
Reference: page cache = unbounded LRU (what an uncapped mmap run gets).

    python -I tools/sim_expert_cache.py trace_*.txt --capacity 106,361
"""

import argparse
import collections
from pathlib import Path

import numpy as np


def load(path, n_exp):
    rows = collections.defaultdict(dict)
    phase = {}
    for line in Path(path).read_text().split("\n"):
        if not line.strip():
            continue
        head, _, probs = line.partition(":")
        f = head.split()
        t, l = int(f[1]), int(f[2])
        phase[t] = f[0]
        p = np.array([float(x) for x in probs.split()], np.float32) if probs.strip() else None
        rows[t][l] = (tuple(int(x) for x in f[3:]), p)
    return [(phase[t], rows[t]) for t in sorted(rows)]


class LFUTier:
    """Frequency-admitted tier, as streammoe's hot tier: admit a missed
    expert only if it has been picked more often than the coldest."""

    def __init__(self, cap):
        self.cap, self.freq, self.res = cap, collections.Counter(), set()

    def contains(self, key):
        return key in self.res

    def after_token(self, picks, probs, t):
        for key in picks:
            self.freq[key] += 1
        reads = 0
        for key in picks:
            if key in self.res or self.cap == 0:
                continue
            if len(self.res) < self.cap:
                self.res.add(key)
            else:
                cold = min(self.res, key=self.freq.__getitem__)
                if self.freq[key] > self.freq[cold]:
                    self.res.remove(cold)
                    self.res.add(key)
        return reads  # admissions reuse the bytes just read


class LRUTier:
    """Bounded LRU: admit every pick, evict the least recently used —
    a page cache with a fixed capacity."""

    def __init__(self, cap):
        self.cap, self.res = cap, collections.OrderedDict()

    def contains(self, key):
        return key in self.res

    def after_token(self, picks, probs, t):
        if self.cap == 0:
            return 0
        for key in picks:
            self.res[key] = None
            self.res.move_to_end(key)
            if len(self.res) > self.cap:
                self.res.popitem(last=False)
        return 0  # admissions reuse the bytes just read


class MRSTier:
    """Keep the top-C experts by moving router score. Experts entering
    the set without having just been read cost a read."""

    def __init__(self, cap, lam, layers, n_exp):
        self.cap, self.lam = cap, lam
        self.S = np.zeros((layers, n_exp), np.float32)
        self.res = set()

    def contains(self, key):
        return key in self.res

    def after_token(self, picks, probs, t):
        self.S *= self.lam
        for l, p in probs.items():
            self.S[l] += p
        if self.cap == 0:
            return 0
        flat = np.argsort(-self.S, axis=None)[: self.cap]
        n_exp = self.S.shape[1]
        new = {(int(i) // n_exp, int(i) % n_exp) for i in flat}
        reads = len(new - self.res - set(picks))
        self.res = new
        return reads


def simulate(traces, layers, n_exp, tier_fn, predictor):
    """Returns coverage, prefetch / demand / tier reads per decode token."""
    covered = picks_n = pre = dem = tier_reads = 0
    ntok = 0
    for rows in traces:
        tier = tier_fn()
        S = np.zeros((layers, n_exp), np.float32)
        prev = {}
        for ph, row in rows:
            if ph == "d":
                ntok += 1
                for l in range(layers):
                    if l not in row:
                        continue
                    A = set(row[l][0])
                    H = {e for e in range(n_exp) if tier.contains((l, e))}
                    P = predictor(l, prev, S)
                    pre += len(P - H)
                    dem += len(A - H - P)
                    covered += len(A & (H | P))
                    picks_n += len(A)
            # Update history after the token.
            probs = {}
            for l, (es, p) in row.items():
                prev[l] = set(es)
                if p is None:  # prefill: picks only, one-hot-ish score
                    p = np.zeros(n_exp, np.float32)
                    p[list(es)] = 1.0 / len(es)
                probs[l] = p
            predictor.update(S, probs)
            r = tier.after_token([(l, e) for l, (es, _) in row.items() for e in es], probs, 0)
            if ph == "d":
                tier_reads += r
    return covered / picks_n, pre / ntok, dem / ntok, tier_reads / ntok, ntok


class Prev:
    name = "prev (today)"

    def __call__(self, l, prev, S):
        return prev.get(l, set())

    def update(self, S, probs):
        pass


class MRS:
    def __init__(self, lam, m):
        self.lam, self.m = lam, m
        self.name = f"mrs λ={lam} top-{m}"

    def __call__(self, l, prev, S):
        if not S[l].any():
            return set()
        return set(int(i) for i in np.argsort(-S[l], kind="stable")[: self.m])

    def update(self, S, probs):
        S *= self.lam
        for l, p in probs.items():
            S[l] += p


class Nothing:
    name = "no prediction"

    def __call__(self, l, prev, S):
        return set()

    def update(self, S, probs):
        pass


def page_cache(traces, layers):
    hits = uses = ntok = 0
    for rows in traces:
        seen = set()
        for ph, row in rows:
            ntok += ph == "d"
            for l, (es, _) in row.items():
                for e in es:
                    if ph == "d":
                        hits += (l, e) in seen
                        uses += 1
                    seen.add((l, e))
    return hits / uses, (uses - hits) / ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--expert-mib", type=float, default=3.375)
    ap.add_argument("--capacity", default="0,106,361")
    ap.add_argument("--lambdas", default="0,0.3,0.5,0.7,0.9")
    ap.add_argument("--m", default="8,12,16")
    a = ap.parse_args()
    traces = [load(p, a.experts) for p in a.traces]
    mib = a.expert_mib
    preds = [Nothing(), Prev()] + [MRS(float(l), int(m)) for l in a.lambdas.split(",") for m in a.m.split(",")]
    ntok = sum(ph == "d" for r in traces for ph, _ in r)
    print(f"{len(traces)} traces, {ntok} decode tokens, {a.layers} layers × top-{len(traces[0][0][1][0][0])} of {a.experts}, "
          f"{mib} MiB per expert\n")
    hr, miss = page_cache(traces, a.layers)
    print(f"Reference — page cache (unbounded LRU, uncapped mmap): hit rate {hr:.1%}, "
          f"{miss * mib:.0f} MiB/token read after warm-up (cold misses only)\n")
    for cap in [int(x) for x in a.capacity.split(",")]:
        tiers = [("lfu (today)", lambda: LFUTier(cap))]
        if cap:
            tiers.append(("lru", lambda: LRUTier(cap)))
            tiers += [(f"mrs λ={lam}", (lambda lam=float(lam): MRSTier(cap, lam, a.layers, a.experts)))
                      for lam in ["0.5", "0.9"]]
        print(f"## Hot tier {cap} experts ({cap * mib / 1024:.2f} GiB)\n")
        print("| Hot tier | Predictor | Coverage (picks in RAM or prefetched) | Prefetch MiB/tok | Demand MiB/tok | Tier refill MiB/tok | **Total read MiB/tok** |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        rows = []
        for tname, tf in (tiers if cap else tiers[:1]):
            for p in preds:
                cov, pre, dem, tr, _ = simulate(traces, a.layers, a.experts, tf, p)
                rows.append((pre + dem + tr, tname if cap else "none", p.name, cov, pre, dem, tr))
        for tot, tname, pname, cov, pre, dem, tr in sorted(rows, key=lambda r: r[0]):
            print(f"| {tname} | {pname} | {cov:.1%} | {pre * mib:.0f} | {dem * mib:.0f} | {tr * mib:.0f} | **{tot * mib:.0f}** |")
        print()


if __name__ == "__main__":
    main()
