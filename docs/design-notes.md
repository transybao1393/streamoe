# Design notes

Measurements, replays and estimates from the 2026-11-05 work on the
hot tier, prefill and prefetch, plus feasibility notes on new hardware
and models. Each section says whether its numbers are **measured**,
**replayed** (a logged routing trace run through a policy in Python)
or **estimated** (arithmetic from measured rates, nothing run).
Settled results also live in [`known-issues.md`](known-issues.md) and
[`benchmarks.md`](benchmarks.md); this file keeps the reasoning.

Unless stated otherwise, 397B means Qwen3.5-397B-A17B at 4-bit from the
external SSD (Kingston NV3 in a USB4 enclosure, ~2.3–2.6 GB/s) under a
9 GiB cap, and 35B means Qwen3.6-35B-A3B at 4-bit from the internal SSD.

## Contents

- [397B with the new defaults](#397b-with-the-new-defaults)
- [Hot tier: LFU with decay](#hot-tier-lfu-with-decay)
- [Experts a prompt drags in](#experts-a-prompt-drags-in)
- [Prefill lifecycle](#prefill-lifecycle)
- [Right and wrong guesses](#right-and-wrong-guesses)
- [Splitting expert math between CPU and GPU](#splitting-expert-math-between-cpu-and-gpu)
- [Expert layout on disk](#expert-layout-on-disk)
- [On a 128 GB DGX Spark](#on-a-128-gb-dgx-spark)
- [Qwen3.8-Flash-Next](#qwen38-flash-next)

## 397B with the new defaults

**Measured.** *Setup: 397B, cap 9 GiB, n_ctx 4096, 32 tokens per
answer, greedy, thinking off, the three standard prompts typed like the
chat page, a fresh server per row, main at `07c75ec`, 2026-11-05.*
All four rows gave identical answers; swap stayed at 0.99 GiB.

| Setup | First word (3 prompts) | tok/s (3 prompts) | Hot tier | Hits | Peak |
| --- | --- | --- | --- | --- | --- |
| Baseline (prefill overlap and embedding streaming off) | 23.1 / 22.4 / 22.3 s | 0.537 / 0.524 / 0.520 | 1.49 GiB | 21–24% | 8.60 GiB |
| **Defaults** (both on, as of `07c75ec`) | 21.8 / 20.9 / 20.8 s (−6%) | 0.553 / 0.540 / 0.535 (+3%) | 2.01 GiB | 23–26% | 8.62 GiB |
| Defaults + prefill while typing | 21.7 / **13.9 / 14.5** s | 0.542 / 0.533 / 0.589 | 1.60 GiB | 23–25% | 8.62 GiB |
| Defaults + `--spec` | 19.3 / 21.1 / 20.7 s | 0.557 / 0.516 / 0.498 | 1.27 GiB | 19–23% | 8.54 GiB |

- Embedding streaming takes 0.53 GiB out of the resident weights
  (5.14 → 4.60 GiB); the plan gives it to the hot tier. Prefill overlap
  takes ~1.4 s off the first word.
- Prefill while typing had 13–14 of the 26–27 prompt tokens done before
  Enter on prompts 2 and 3: −32% first word against the defaults row.
  Prompt 1 gained nothing: the draft prefill was still running at Enter.
  It became the chat page default on 2026-11-05
  (`--no-prefill-while-typing` turns it off).
- Speculative decoding doesn't pay on normal chat: guesses rarely match,
  and its check-pass buffers (`multi_decode`, 0.74 GiB) shrink the hot
  tier.

Evidence: `docs/dev/a17b-evidence/features-20261105-2039/` (local,
gitignored).

## Hot tier: LFU with decay

How it works (`io/hot_tier.rs`): each router pick adds 1 to a
(layer, expert) counter; every 4,096 picks all counters are halved; a
miss is admitted only if its count beats the coldest of 8 sampled
residents (Redis-style), skipping slots bound this token. Idle prefill
staging and scratch are lent to the tier during decode.

On the 35B, 4,096 picks is ~13 tokens (320 picks per token). The 397B
makes 600 picks per token, so the same setting forgets in ~7 tokens.

### Decay sweep on the 397B

**Measured.** *Setup: as above, defaults on, 305 slots (2.01 GiB),
2026-11-05.* Longer memory is worse:

| `STREAMMOE_HOT_DECAY` | tok/s (3 prompts) |
| --- | --- |
| 4,096 (default) | 0.6 / 0.5 / 0.5 (64 tokens) |
| 16,384 | 0.5 / 0.5 / 0.5 |
| 65,536 | 0.5 / 0.5 / 0.5 |
| 262,144 | 0.5 / 0.4 / 0.4 |

### Replay against other policies

**Replayed.** A 192-token decode trace of the same 397B prompts
(`STREAMMOE_ROUTING_TRACE`), through the tier's own rule at several
decay lengths, plain LRU, the best fixed set, and Belady's optimum
(perfect knowledge of the future, with bypass). The replay was a
scratch script; `tools/sim_hot_tier.py` on the hot-prefill branch
(below) does the same for prefill admission policies.

| Policy | 429 slots (the run's tier + lent slots) | 1,000 slots | 3,000 slots |
| --- | --- | --- | --- |
| LFU, decay 512 | 8.8% | 29.8% | 52.1% |
| LFU, decay 2,048 | **25.7%** | **37.0%** | 55.7% |
| LFU, decay 4,096 (default) | 25.5% | 36.7% | **56.2%** |
| LFU, decay 16,384 | 21.2% | 31.8% | 50.6% |
| LFU, decay 262,144 | 15.3% | 25.8% | 45.2% |
| LRU | 0.0% | 36.7% | 57.0% |
| Best fixed set | 17.2% | 29.5% | 52.9% |
| Belady (optimum) | 44.7% | 58.4% | 73.9% |

- The default decay is at the top of the curve; nothing to tune there.
- At the real size LRU scores zero: 600 picks per token cycle through
  429 slots before any repeat.
- A perfect policy would cut reads from 447 to 332 per token (−26%),
  about 0.54 → 0.7 tok/s. That gap is the most any policy change could
  win at this cap; only a bigger tier or smaller experts move the
  ceiling itself.

Evidence: `docs/dev/a17b-evidence/hot-20261105-2132/`.

## Experts a prompt drags in

The problem: prefill adds one count per token per expert, so an expert
serving 20 prompt tokens gains 20 at once, while decode adds 1 per
token. A 25-token 397B prompt makes ~15,000 picks, crossing the decay
3–4 times. Right after a prompt the tier is mostly prompt experts, and
decode's regulars come back slowly. That is fine when the answer stays
on the prompt's topic and pure churn after a long paste.

The idea tested: a probation / protected split (the SLRU or 2Q family).
Prompt experts go to a small LRU probation segment and are promoted
into the LFU tier on their first decode hit; prefill counts at most
once per expert per chunk.

**Measured and replayed** (branch `worktree-agent-ad49b03d6621548f5`,
commits `f46788b`, `35bf4b9`, `869dc5f`; `OpenConfig::hot_prefill` /
`STREAMMOE_HOT_PREFILL` = `count` (today) | `once` | `ignore`):

- **The probation segment loses.** In the lent slots it can't work,
  because those are emptied at every prefill. Given its own slots, it
  costs 4–6 points of hits, because those slots come out of the tier.
- **`once`** (count each expert once per prefill chunk) gains 1–3
  points on long prompts and is a wash on chat.
- Gate with `once`: unit 179/0, `budget_cap`, `streaming` 7/7,
  `mlx_regression` 1/1. `a3b_long_decode_stays_under_4gib_cap`
  fails now and then with a one-time 9–66 MiB footprint step (not a
  leak); that needs explaining before `once` can merge.
- **397B, measured** (2026-11-06, 9 GiB cap, 305 slots, 64 tokens, the
  three standard prompts): `count` first word 24.4 / 23.6 / 22.7 s,
  `once` 23.9 / 23.4 / 22.3 s; 0.5 tok/s either way; peak 8.62–8.63 GiB.
- **397B, replayed** (78 prefill + 192 decode tokens, 305 own + 124
  lent slots, `tools/sim_hot_tier.py`): today's rule 26.1% hits, prefill
  ignored 25.8%, `once` 26.5%, probation in 124 own slots 23.1%,
  probation sharing the lent slots 26.5%, Belady 45.6%. All prefill
  policies are within half a point; on short chat prompts how prefill
  counts doesn't matter. Evidence:
  `docs/dev/a17b-evidence/hotprefill-20261106-1432/`.

## Prefill lifecycle

Under a cap, a prompt is cut into chunks of ≤ 512 tokens. For each
chunk, for each layer:

```
1. Attention (GPU): batched projections; linear-attention layers
   (3 of 4) step the recurrence token by token in one command buffer.
2. Wait; copy the hidden rows and the router's picks to the CPU.   ← sync
3. CPU: group tokens by expert (buckets); add picks to the hot tier.
4. Stage the layer's experts: hits copied from RAM, misses read in
   merged runs. On the 397B 64 fit at once; groups of 32 are
   double-buffered so the GPU runs one while the next loads
   (prefill overlap, default since 07c75ec).
5. Grouped gather-id MoE, last group adds the shared expert and combine.
6. Wait.                                                           ← sync
```

On the 397B a 25-token prompt takes ~21 s (~0.84 s per token): each
layer reads every distinct expert the prompt picked, 100+ × 6.75 MiB,
at ~2.5 GB/s. The syncs and attention are milliseconds.

Ideas, biggest first:

| # | Idea | Helps | Status |
| --- | --- | --- | --- |
| 1 | **Layer-major prefill**: run each layer over every chunk before the next layer, so its experts are read once per prompt instead of once per chunk | Long prompts. Measured, 2,062 tokens: −45% prefill on the 35B, **2.7× faster on the 397B** (below) | Built, opt-in, bit-identical, branch `worktree-agent-a75d0afb8b2779274` at `98a6263` |
| 2 | Prefill while typing | Chat | Done; chat default since 2026-11-05 |
| 3 | Probation for prompt experts | Decode after a long prompt | Measured: loses (above) |
| 4 | Chunkwise batched linear attention | Long prompts on the 35B (compute-bound at 108–128 tok/s) | Not started |
| 5 | Routing and buckets on the GPU, removing sync 2 | The 35B, a few percent | Not started |
| 6 | Bind hot-tier hits directly instead of copying into staging | Both, ~1 ms per hit | Not started |

Layer-major needs only the same layer's earlier-chunk state (KV,
linear-attention state), so the reorder is exact; the extra memory is
the prompt's hidden rows (2,000 × 4,096 × 4 B ≈ 32 MB) plus a running
sum. Prompts under 512 tokens are one chunk and gain nothing.

**Layer-major, measured.** Switches: `OpenConfig::layer_major` /
`STREAMMOE_LAYER_MAJOR=1`, and `layer_major_chunks` /
`STREAMMOE_LAYER_MAJOR_CHUNKS` (window, default 4 chunks). Within a
window each layer runs all its chunks before the next layer. With full
staging (35B) later chunks stage only experts not already staged. With
bounded staging (397B) every chunk runs attention first, then the
union of the window's experts is staged once in groups, each group
applied to every chunk that routes to it. The window buffers are
planned (`Account::LayerMajor`; 35B 16 MiB of hidden rows; 397B
320 MiB with bounded staging) and lent to the hot tier during decode;
a capped open that can't fit them shrinks the window or turns it off.

- `tests/layer_major.rs`: logits, `state_save` bytes, 6 decode steps
  and a second prefill bit-identical to chunk-major for full staging
  (overlap 0 and 8) and bounded staging (64 and 16 slots), windows of
  4, 2+2 and 3+1 chunks; experts staged 4,720–7,329 vs 11,012.
- Model suites with it off and on: `bounded_staging` 2/2,
  `batched_diff_oracle` 33/33, `mlx_regression` 1/1, `streaming` 7/7,
  `budget_cap` 11/11.

*Setup: 35B 4-bit, 4 GiB cap, n_ctx 4096, F32 KV, internal SSD, a
2,062-token prompt + 16 tokens, a fresh chat server per run, 2 runs
each, alternating. Output identical in all runs.*

| | Chunk-major | Layer-major |
| --- | --- | --- |
| Prefill time | 14.0 s | 7.7 s (−45%) |
| Prefill speed | 147 tok/s | 269 tok/s |
| Expert staging I/O | 10.6 s | 3.5 s |
| Prefill disk reads | 55,890 MiB | 18,887 MiB |
| Peak memory | 3.63 GiB | 3.63 GiB |

*Setup: 397B, external SSD, 9 GiB cap, n_ctx 4096, the same 2,062-token
prompt + 4 tokens, one fresh server per arm, 2026-11-06. Same answer.*

| | Chunk-major | Layer-major |
| --- | --- | --- |
| First word | 368 s | **135 s (−63%)** |
| Prefill speed | 5.6 tok/s | 15.3 tok/s |
| Hot tier | 305 slots (2.01 GiB) | 257 slots (1.69 GiB): the 0.31 GiB window comes out of the plan |
| Decode (4 tokens) | 0.43 tok/s, 14% hits | 0.42 tok/s, 13% hits |
| Peak memory | 8.64 GiB | 8.61 GiB |

The window costs the hot tier 48 slots for the whole session, although
it is only used during prefill (it is lent back during decode, but
lent slots are emptied at every prefill). On short chat the price is a
slightly smaller tier for no prefill gain; layer-major pays only for
prompts over 512 tokens. Evidence:
`docs/dev/a17b-evidence/layermajor-20261106-1449/`.

## Right and wrong guesses

Prefill itself is exact. "Right or wrong" applies only to guesses, and
none of them can change the output:

| Guess | Right when | Wrong when | Effect of a wrong one |
| --- | --- | --- | --- |
| Decode prefetch (next layer's experts before the router picks) | The router picks the guessed expert | Read but not picked | Disk time; before the fix below, also a hot-tier offer |
| Prefill while typing | The sent message starts with the prefilled text | Prefilled text was edited | Rolled back to a checkpoint and redone |
| Speculative decoding | The check pass accepts the tokens | It rejects them | Tokens thrown away |

**Measured:** on the 397B (cross-layer guess, `--prefetch 6`), 86–87%
of prefetches are picked; ~35 reads per token (~7%) are wasted.

Three opt-in switches handle wrong prefetches (branch
`worktree-agent-a65836efdce6e6fad`, commits `41a31af`, `511582f`; all
off by default; output bit-identical in every arm):

| Switch | Env var | What it does |
| --- | --- | --- |
| A | `STREAMMOE_PREFETCH_OFFER_CONFIRM=1` | Offer a prefetched blob to the hot tier only once the router confirms it |
| B | `STREAMMOE_PREFETCH_CANCEL=1` | Skip wrong guesses that haven't started reading |
| C | `STREAMMOE_PREFETCH_PIECE_KIB=1024` | Read in pieces and stop a wrong read between pieces |

**Measured** on the 35B. *Setup: tools/chat, n_ctx 4096, F32 KV,
thinking off, `--prefetch 6`, cross-layer guess, three standard prompts
× 120 tokens, 2 fresh-server runs per arm, interleaved; C = 1 MiB.*

| Cap | Arm | tok/s | Disk MiB / token | Wrong reads cut / token | Hits |
| --- | --- | --- | --- | --- | --- |
| 3 GiB | off | 9.74 | 296.8 | 0 | 52.8% |
| 3 GiB | A | 9.83 | 295.2 | 0 | 52.4% |
| 3 GiB | A+B | 9.64 | 295.0 | 0 | 52.5% |
| 3 GiB | A+B+C | 10.00 | 288.1 | 10.4 | 52.4% |
| 4 GiB | off | 11.88 | 207.7 | 0 | 67.7% |
| 4 GiB | A | 12.03 | 205.7 | 0 | 67.4% |
| 4 GiB | A+B | 11.59 | 205.0 | 0 | 67.6% |
| 4 GiB | A+B+C | 12.07 | 198.8 | 9.6 | 67.5% |

- A: wrong guesses weren't costing hits. B: skips nothing in chat,
  since all six reads have started by routing time. C: −3% disk bytes,
  +1.6 to +2.7% tok/s, inside the ±3% noise.
- Piece size (internal SSD, F_NOCACHE): with 4–8 reads in flight, 1 MiB
  pieces keep full speed (4.6–4.8 GB/s for 1.77 MB experts, 5.0–5.2 for
  6.75 MiB); 512 KiB loses 5–15%, 256 KiB 20–30%. A lone read loses
  even at 1 MiB.
- 397B reads take ~4× longer, so B and C may matter more there.
  Unmeasured.
- Model suites with all three switches on: `streaming` 7/7,
  `budget_cap` 11/11, `mlx_regression` 1/1, `batched_diff_oracle` 33/33
  (2026-11-06), as with them off.

## Splitting expert math between CPU and GPU

**Estimated.** Not worth building for the 397B. Of a ~1.9 s token,
~1.6–1.75 s is waiting on ~490 expert reads (~3.3 GiB); the math is
~0.15–0.3 s. A perfect, free split would give ~0.53 → ~0.57 tok/s, and
the real one less:

- CPU and GPU share one memory system, and 4-bit matmuls are
  bandwidth-bound, so running both mostly splits the same bandwidth.
- The CPU already runs the pread and attention threads; taking it
  slows the reads, which are the bottleneck.
- Each of 60 layers would add a CPU/GPU join.

On the 35B, where the math is a larger share, expect single digits at
best. Worth a microbench, not a rewrite.

## Expert layout on disk

The proposed layout (gate, up, down next to each other, 16 KiB padded,
one read per expert) is what streamoe already does:

- One file per layer; each expert is one contiguous blob with its
  scales and biases, read with one pread.
- The blobs are already 16 KiB multiples: 397B 7,077,888 B = 432 ×
  16 KiB, 35B 1.6875 MiB = 108 × 16 KiB.
- Prefill merges adjacent needed experts into longer runs
  (`PREFILL_MAX_RUN`).

A 6.75 MiB read already runs the drive at full speed, so the cost is
bytes, not read count. Ideas that cut bytes:

| Idea | Expected |
| --- | --- |
| Co-activation layout (experts picked together sit together) | Small; the SSD is flat once ≥ 4 reads are in flight (see known-issues dead ends) |
| Read part of an expert (store each up row with its down column, read predicted-active neurons only) | Large if activations are sparse; SwiGLU is much less sparse than ReLU, so likely small. Measure near-zero activations on the 35B first |
| Packed 3-bit experts (supported) | −22–25% bytes; est. 397B 0.53 → ~0.7 tok/s |
| Dynamic top-k (skip low-weight experts) | Fewer reads, but changes answers; the 35B's gate weights are flat (known-issues) |

## On a 128 GB DGX Spark

**Estimated** from public specs (GB10, 128 GB unified memory, ~273 GB/s,
internal NVMe assumed ~6 GB/s) and this Mac's measured rates. streamoe
is macOS / Metal only: a Spark needs a CUDA backend behind the
`Backend` trait, Linux I/O (`O_DIRECT` or io_uring instead of
`F_NOCACHE`) and Linux memory accounting. Multi-week work.

**35B:** fits whole (~19 GB); nothing streams. ~2.0–2.3 GB read per
token gives a bandwidth ceiling of ~120 tok/s at peak, ~80–90 at a
realistic ~180 GB/s. This Mac's GPU spends ~39 ms per token (~50 GB/s
effective; many small kernels), and a straight port would carry that
over.

| Stage | Decode |
| --- | --- |
| First working port | ~40–60 tok/s |
| Tuned (CUDA graphs, fused MoE) | ~70–90 tok/s |
| Ceiling | ~120 tok/s |

Prefill is compute-bound: est. 1,000–3,000 tok/s. But tools with CUDA
already (llama.cpp, vLLM) run a model that fits; streamoe's edge there
is models that don't.

**397B:** ~202 GiB of experts still don't fit; the hot tier would be
~110 GiB (~54% of experts, against 0.7–1% here). Every-expert-in-memory
ceiling ~20 tok/s.

| Hit rate | NVMe read / token | tok/s |
| --- | --- | --- |
| 80% | 0.8 GiB | ~5–7 |
| 90% | 0.4 GiB | ~8–12 |
| 95% | 0.2 GiB | ~12–15 |

The replay above can narrow this: at 3,000 slots (~10% of experts) LFU
already reaches 56%. 397B 3-bit est. ~8–14 tok/s; 2-bit fits in memory,
~18–22, with a clear quality loss.

## Qwen3.8-Flash-Next

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
architecture `qwen4_exp` (preview). From the model card:

- 125B main model, 6B active per token; plus a 51B n-gram embedding
  (20M rows, applied at layer 2) and a 4B MTP module.
- Hidden 2560; 48 layers as 12 × (3 Gated DeltaNet → MoE, 1 Qwen Sparse
  Attention → MoE).
- 512 experts, 10 routed + 1 shared, intermediate 640.
- Gated DeltaNet 48 V / 16 QK heads, dim 128. QSA: 24 Q / 2 KV heads,
  head dim 256, RoPE 64, an MQA indexer (4 heads, dim 128), budget 512
  blocks = 2,048 tokens.
- Gated Residual: 4 branches, rank 320. Vocab 248,320; context 262K.

**Estimated cost on this Mac (4-bit):**

| | 397B | **3.8-Flash-Next** | 35B |
| --- | --- | --- | --- |
| One expert | 6.75 MiB | ~2.64 MiB | ~1.69 MiB |
| All experts | ~202 GiB | ~63 GiB | ~17 GiB |
| Expert bytes / token, uncached | 3.96 GiB | ~1.27 GiB | 0.53 GiB |
| Resident weights | 5.1 GiB | ~2.5 GiB | ~2 GiB |

At a 6–8 GiB cap the tier holds ~800–1,500 of 24,576 experts; guess
40–50% hits, ~260 reads (~0.7 GiB) per token: **~2.5–3.5 tok/s** from
the external SSD, ~4–6 from the internal one, maybe 1.3–1.6× more with
MTP speculative decoding. The n-gram table is a per-token row lookup
(a few KB) and can stream from disk like the embedding.

**What streamoe needs:**

| Piece | Effort |
| --- | --- |
| Variant shapes | Small |
| Gated DeltaNet at these head counts (kernels are parameterised) | Small to check |
| Qwen Sparse Attention: up to 2,048 tokens it selects every block, so dense attention is exact; past that, the indexer and block selection | Medium |
| Gated Residual | Small to medium |
| N-gram embedding lookup, streamed | Medium |
| Converter | Medium |
| MTP | Later |
| Reference for tests (MLX or transformers `qwen4_exp`) | To check |

Roughly one to two weeks of agent work plus verification.

**Quantization.** streamoe reads its own formats only (MLX-style 4-bit,
group 64, and packed 3-bit); GGUF types such as IQ3_XXS or Q2_K would
each need a converter and a dequant kernel (IQ3_XXS is a codebook
lookup; llama.cpp's MIT Metal kernels could be ported). Per expert
(3 × 640 × 2560 ≈ 4.9M weights), estimated:

| Format | ~bits / weight | One expert | All experts | Est. decode, external SSD | Quality |
| --- | --- | --- | --- | --- | --- |
| streamoe 4-bit | 4.5 | 2.64 MiB | ~63 GiB | ~2.5–3.5 tok/s | Reference |
| **streamoe packed 3-bit** | ~3.5 | ~2.06 MiB | ~49 GiB | ~3–4.3 tok/s | On the 35B: KL 0.048, 10/10 tasks |
| IQ3_XXS | ~3.06 | ~1.79 MiB | ~43 GiB | ~3.3–4.6 tok/s | Lowest bit width usually considered good |
| 2-bit (Q2_K / IQ2) | ~2.5 | ~1.46 MiB | ~35 GiB | ~3.8–5 tok/s | Clear loss; on the 35B all-2-bit failed 4 of 10 tasks |

Recommendation: start with packed 3-bit (already built and measured);
try IQ3_XXS later as an experiment; skip 2-bit.

**Blocker: disk.** ~95 GB for a 4-bit copy with the n-gram table
quantized (~170 GB with it in BF16), plus room for the source during
conversion. Free on 2026-11-05: 75 GiB internal, 17 GiB external.

### Expert reads and PLE in the engine

**Expert reads** (`Q38Engine::moe_layer`). One staging buffer, read
by the permute-fuse op at `slot · expert_size` (the padded file
stride, 3,080,192 B on the real model and 32,768 B on the tiny one, is
only a file offset):

| Slots | Holds |
| --- | --- |
| `0 .. K` | Bucket i's expert: a hot-tier copy or a pread |
| `K + s·P .. K + s·P + P` | Prefetch set `s = layer % 2`, `P = min(prefetch_experts, K)` |

The prefetch is the a3b decode path's `PrefetchState`: previous-token
guess, strongest pick first, hot-tier experts masked out. Layer L's
reads go out at the top of layer L, so they run under the GPU work up
to its router; with `prefetch_lookahead`, layer L+1's go out right
after layer L's router readback. A routed expert found in the set binds
that slot directly. Settings and defaults are the context's (all K
uncapped, 4 under a cap, `prefetch_experts = 0` turns it off); the plan
counts the `2P` extra slots. Not used: the cross-layer guesser (GQA
only) and switches A/B/C (on a separate branch, not merged here).
Everything in flight is drained at the end of each token.

Measured on the tiny model (4 layers × 8 experts, top 3; 13 + 8 tokens
× 3 prompts = 780 picks; logits bit-identical in every arm):

| Arm | Prefetch hits / reads | Demand reads | Hot hits |
| --- | --- | --- | --- |
| Off | – | 780 | – |
| Prefetch all 3 | 283 / 732 | 497 | – |
| Prefetch 1 + lookahead | 95 / 244 | 685 | – |
| Hot tier 12 slots | – | 501 | 279 |
| Prefetch all + hot tier 12 | 173 / 438 | 322 | 285 |

Random weights route close to uniformly, so these say nothing about
the real model's hit rates. Hot-tier hits are still copied into the
staging (~3 MB each); binding them in place needs per-bucket buffers
in the permute-fuse op.

**PLE** (`ple_gpu.rs`). The n-gram rows are a host table lookup and
are uploaded (2,560 floats per token); `key_proj` / `value_proj` are
graph matvecs; the norms, the signed-sqrt gate and the dilated conv
are two family ops (`PleOp::Gate`, `PleOp::Conv`), and the conv state
lives in a GPU buffer. The PLE layer no longer flushes or reads the
residual back. The ops take T rows (the conv walks them in order); the
engine's PLE buffers hold a prefill chunk, so chunked prefill runs them
too. Checked
against `ple_reference.py`'s fixture (Metal 3.6e-7, CPU 2.6e-7) and,
in the engine, against the host PLE (≤ 1.5e-6 on the logits).

### Command buffers and batched prefill in the engine

**One encoder per kernel signature** (`engine_exec.rs`). Every family
op's Metal arm is one call to a shared encoder: the Gated Residual
kernels through `Q38GpuKernels::encode_*` (the functions the T-token
Gated Residual path uses), the engine's own kernels (per-head norm
with `1 + w`, sigmoid-gated norm, add, row gather) through
`Q38EngineKernels`, the PLE through `PleGpuKernels::encode`, QSA sparse
rows through `QsaGpuKernels::encode_sdpa_rows`. Nothing binds kernel
arguments by hand, so the round-2 bug (T-token kernels gained trailing
arguments the engine left unset) cannot recur there.

**Command buffers per token** (`Q38Engine::submits`, tiny model,
Metal, decode):

| | Within the QSA budget | Past it |
| --- | --- | --- |
| Before (round 2) | 7 | 7 |
| Indexer readback deferred, MoE rows gathered on the GPU | 6 | 7 |
| + PLE on the GPU (q38-prefetch-ple) | **5** | **6** |

What is left is one per layer (the router logits: the host must pick
the experts before it reads them), the tail, and past the budget one
per QSA layer (the host scores the chunk's indexer queries against the
block keys before the sparse attention). A QSA layer within the budget
adds none: its raw indexer keys only need caching, so they are read
back after the router's submit. Each layer's router readback still
waits for every earlier command buffer (the prefetch's set reuse
relies on it).

**Batched prefill** (`Q38Engine::eval` → `forward_chunk`). The prompt
runs in chunks of `STREAMMOE_Q38_CHUNK` tokens (default 64, at most
`n_ctx`), the same ops at T rows: the Gated Residual ops and matvecs
(MLX `qmm` at T > 1), the Gated DeltaNet conv / recurrence ops (they
walk the chunk in order, so no chunked GDN kernel is needed),
`push_qsa_pre_sdpa` over the chunk then dense SDPA, or past the budget
host block selection (`qsa_select_rows`) and the GPU sparse-rows kernel,
so no K / V copies to the host in prefill or decode. The MoE groups the
chunk's tokens by expert (`build_expert_buckets`, as the other
families' batched prefill does) and runs `MoeBatchedPermuteFuse` per
group of `top_k` experts through the demand slots; each extra group is
one more submit. The final mixer and LM head run on the last row and
only when logits are wanted. The per-token buffers hold a chunk; the
plan's `PrefillScratch` counts them (~50 MB at 64 rows on the real
model).

Tiny model (debug build, Metal, 48-token prompt, indicative): 336 → 14
command buffers, ~350 → ~5,600 tok/s. Batched vs a step per token
(`batched_prefill_matches_per_token`, chunks 5 / 7 / 64 crossing the
8-token budget): logits 1.3e-6, state ~1e-17, the next decode tokens
1.1e-6 on Metal; bit-identical on the CPU backend.

On the real model a 64-token chunk routes to ~300 distinct experts per
layer, read 10 at a time with the GPU idle meanwhile; overlapping the
reads of the next group with the GPU work of this one is the next step.
