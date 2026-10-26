#!/usr/bin/env python3
"""Summarize a disk timeline trace: when the SSD reads and when it sits idle.

    STREAMMOE_IO_TRACE=/path/trace.txt <run the engine, e.g. tools/chat>
    python3 tools/io_trace_report.py /path/trace.txt

Format: `crates/streammoe/src/riir/io/io_trace.rs`. "Busy" means at
least one expert read in flight. Decode tokens are split into phases by
the per-layer markers:

  start      token start -> layer 0 top
  attention  layer top -> routed (prefetch reads + the GPU's attention)
  misses     routed -> the layer's expert reads landed (critical path)
  dispatch   reads landed -> next layer top (last layer: LM head too)

Prefills report their staging time, the GPU time between stages and the
read throughput during staging. Tracing itself slows decode by ~15%:
use it to see where time goes, not to measure speed.
"""
import bisect
import collections
import sys


def load(path):
    reads, marks = [], []
    for ln in open(path):
        p = ln.split()
        if p and p[0] == "r":
            reads.append((int(p[1]), int(p[2]), p[3], int(p[4]), int(p[5]), int(p[6])))
        elif p and p[0] == "m":
            marks.append((int(p[1]), p[2], int(p[3])))
    reads.sort()
    return reads, marks


def merged(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap(busy, t0, t1):
    return sum(max(0, min(b, t1) - max(a, t0)) for a, b in busy)


def decode(reads, marks):
    toks, cur = [], None
    for t, tag, layer in marks:
        if tag == "T":
            cur = {"t0": t, "m": []}
        elif tag == "E" and cur:
            cur["t1"] = t
            toks.append(cur)
            cur = None
        elif cur and tag in "LRW":
            cur["m"].append((t, tag, layer))
    if not toks:
        print("decode: no tokens in the trace")
        return
    starts = [r[0] for r in reads]
    names = {"T": "start", "L": "attention", "R": "misses", "W": "dispatch"}
    phase = collections.Counter()
    wall = busy_total = nreads = nbytes = 0
    for tk in toks:
        a, b = tk["t0"], tk["t1"]
        i = bisect.bisect_left(starts, a - 1_000_000)
        rs = [r for r in reads[i:] if r[0] < b and r[1] > a]
        busy = merged((max(r[0], a), min(r[1], b)) for r in rs)
        wall += b - a
        busy_total += sum(y - x for x, y in busy)
        nreads += len(rs)
        nbytes += sum(r[5] for r in rs)
        ms = [(a, "T", -1)] + tk["m"] + [(b, "E", -1)]
        for (t0, tag, _), (t1, _, _) in zip(ms, ms[1:]):
            phase[tag, "wall"] += t1 - t0
            phase[tag, "busy"] += overlap(busy, t0, t1)
    n = len(toks)
    print(f"decode, {n} tokens: {wall / n / 1e3:.1f} ms/token, disk busy {busy_total / n / 1e3:.1f} ms, "
          f"idle {(wall - busy_total) / n / 1e3:.1f} ms; {nreads / n:.0f} reads, "
          f"{nbytes / n / 2**20:.0f} MiB per token")
    print(f"  {'phase':10s} {'wall ms':>8s} {'busy ms':>8s} {'idle ms':>8s}")
    for tag in "TLRW":
        w, bz = phase[tag, "wall"] / n / 1e3, phase[tag, "busy"] / n / 1e3
        print(f"  {names[tag]:10s} {w:8.1f} {bz:8.1f} {w - bz:8.1f}")
    gaps = [t2["t0"] - t1["t1"] for t1, t2 in zip(toks, toks[1:]) if 0 < t2["t0"] - t1["t1"] < 1e6]
    if gaps:
        print(f"  between tokens: {sum(gaps) / len(gaps) / 1e3:.2f} ms")


def prefill(reads, marks):
    runs, cur = [], []
    for t, tag, layer in marks:
        if tag in "Ss":
            cur.append(t)
        elif tag == "T" and cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    for run in runs:
        t0, t1 = run[0], run[-1]
        stage = sum(b - a for a, b in zip(run[::2], run[1::2]))
        gaps = [b - a for a, b in zip(run[1::2], run[2::2])]
        rs = [r for r in reads if r[2] == "F" and t0 <= r[0] <= t1]
        layers = len(run) // 2
        experts, nbytes = sum(r[4] for r in rs), sum(r[5] for r in rs)
        rate = nbytes / (stage / 1e6) / 1e9 if stage else 0
        print(f"prefill, {layers} layer stages: {(t1 - t0) / 1e3:.0f} ms, staging {stage / 1e3:.0f} ms, "
              f"GPU between stages {sum(gaps) / 1e3:.0f} ms; {experts / max(layers, 1):.1f} experts/layer, "
              f"{rate:.1f} GB/s while staging")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    reads, marks = load(sys.argv[1])
    prefill(reads, marks)
    decode(reads, marks)


if __name__ == "__main__":
    main()
