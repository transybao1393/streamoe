"""tools/gpu_timing_report.py <STREAMMOE_GPU_TIMING file> [skip_tokens] — per decode token:
command buffers, CPU waits, GPU busy vs idle, and what each idle gap sat between.

Format (crates/streammoe/src/riir/backend/gpu/gpu_timing.rs):
  <label> <gpu_start> <gpu_end> <cpu_resumed>     a waited (or record_later'd) command buffer
  @<label> <host_time>                            a host mark (commit:<label>, io_start, io_end, <label>=commit of a waited cmdbuf)
Tokens are delimited by `lm_head` lines; a window counts as a decode token if it holds >= 30 cmd2_3 buffers.
"""
import sys, collections

path = sys.argv[1]
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 2
cbs, marks = [], []
for line in open(path):
    f = line.split()
    if not f:
        continue
    if f[0].startswith("@"):
        marks.append((float(f[1]), f[0][1:]))
    elif len(f) == 4:
        cbs.append((f[0], float(f[1]), float(f[2]), float(f[3])))
marks.sort()

# token windows by lm_head cpu_resumed
lm = sorted(c[3] for c in cbs if c[0] == "lm_head")
wins = list(zip(lm, lm[1:]))
toks = []
for (a, b) in wins:
    inw = sorted([c for c in cbs if c[0] != "lm_head" and a < c[2] <= b] +
                 [c for c in cbs if c[0] == "lm_head" and a < c[3] <= b], key=lambda c: c[1])
    if sum(1 for c in inw if c[0] == "post_attn_tail.cmd2_3") < 30 or any(c[0].startswith("graph_") for c in inw):
        continue
    toks.append((a, b, inw))
toks = toks[skip:]
if not toks:
    sys.exit("no decode tokens")

ios = []
_o = None
for t, m in marks:
    if m == "io_start": _o = t
    elif m == "io_end" and _o is not None: ios.append((_o, t)); _o = None
agg = collections.defaultdict(float)
cnt = collections.Counter()
gpu_by_label = collections.defaultdict(float)
gap_pairs = collections.defaultdict(float)
gap_pair_n = collections.Counter()
gap_io = 0.0
gap_hist = collections.Counter()
for a, b, inw in toks:
    wall = b - a
    agg["wall"] += wall
    waited = [c for c in inw if c[0] != "moe.experts_early"]
    cnt["waited_cmdbufs"] += len(waited)
    cnt["all_cmdbufs"] += len(inw)
    for c in inw:
        cnt[c[0]] += 1
        gpu_by_label[c[0]] += c[2] - c[1]
    # union busy + gaps
    iv = sorted((c[1], c[2], c[0]) for c in inw if c[2] > c[1])
    busy = 0.0
    cur_s, cur_e, cur_l = iv[0]
    for s, e, l in iv[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e, cur_l = e, l
            continue
        busy += cur_e - cur_s
        g = s - cur_e
        key = f"{cur_l} -> {l}"
        gap_pairs[key] += g
        gap_pair_n[key] += 1
        gap_hist[min(int(g * 1e4), 20)] += 1  # 0.1 ms bins
        # io time inside the gap
        for (i0, i1) in ios:
            gap_io += max(0.0, min(i1, s) - max(i0, cur_e))
        cur_s, cur_e, cur_l = s, e, l
    busy += cur_e - cur_s
    agg["busy"] += busy
    agg["span"] += iv[-1][1] - iv[0][0]

n = len(toks)
ms = lambda x: x / n * 1e3
print(f"{n} decode tokens (skipped first {skip})")
print(f"wall {ms(agg['wall']):.1f} ms/tok   GPU busy {ms(agg['busy']):.1f}   GPU idle {ms(agg['wall']-agg['busy']):.1f}  (of which disk reads {ms(gap_io):.1f})")
print(f"command buffers/tok: {cnt['all_cmdbufs']/n:.1f}  (CPU-waited {cnt['waited_cmdbufs']/n:.1f})")
print("per label: count/tok, GPU ms/tok, ms per buffer")
for l in sorted(gpu_by_label, key=lambda l: -gpu_by_label[l]):
    print(f"  {l:28s} {cnt[l]/n:5.1f}  {ms(gpu_by_label[l]):6.2f}  {gpu_by_label[l]/max(cnt[l],1)*1e3:.3f}")
print("idle gaps by (buffer before -> buffer after): ms/tok, gaps/tok, mean ms")
for k in sorted(gap_pairs, key=lambda k: -gap_pairs[k]):
    print(f"  {k:55s} {ms(gap_pairs[k]):6.2f}  {gap_pair_n[k]/n:5.1f}  {gap_pairs[k]/gap_pair_n[k]*1e3:.3f}")

# commit -> GPU start latency when the queue was idle; cpu wake latency after waits
lat = collections.defaultdict(list)
wake = collections.defaultdict(list)
t0, t1 = toks[0][0], toks[-1][1]
starts = collections.defaultdict(list)
for c in cbs:
    if t0 < c[3] <= t1 or t0 < c[2] <= t1:
        starts[c[0]].append(c[1])
        if c[0] != "moe.experts_early":
            wake[c[0]].append(c[3] - c[2])
for k in starts:
    starts[k].sort()
import bisect
for t, m in marks:
    if not (t0 < t <= t1):
        continue
    lab = m[7:] if m.startswith("commit:") else m
    if lab not in starts:
        continue
    i = bisect.bisect_left(starts[lab], t)
    if i < len(starts[lab]):
        lat[lab].append(starts[lab][i] - t)
print("commit -> GPU start (median / mean ms), wait end -> CPU resumed (median / mean ms)")
med = lambda v: sorted(v)[len(v) // 2] * 1e3 if v else float('nan')
mean = lambda v: sum(v) / len(v) * 1e3 if v else float('nan')
for k in sorted(set(lat) | set(wake)):
    print(f"  {k:28s} {med(lat[k]):6.3f} / {mean(lat[k]):6.3f}    {med(wake[k]):6.3f} / {mean(wake[k]):6.3f}")
print("gap histogram (0.1 ms bins, gaps/tok):", " ".join(f"{b/10:.1f}:{gap_hist[b]/n:.1f}" for b in sorted(gap_hist)))

# CPU segments: time from one host mark to the next, by (mark -> next mark)
seg = collections.defaultdict(list)
t0, t1 = toks[0][0], toks[-1][1]
mk = [(t, m) for t, m in marks if t0 < t <= t1]
for (ta, ma), (tb, mb) in zip(mk, mk[1:]):
    seg[f"{ma} -> {mb}"].append(tb - ta)
print("host mark -> next host mark: per tok count, ms/tok, median ms")
for k in sorted(seg, key=lambda k: -sum(seg[k]))[:14]:
    v = seg[k]
    print(f"  {k:60s} {len(v)/n:5.1f} {sum(v)/n*1e3:6.2f}  {med(v):.3f}")
