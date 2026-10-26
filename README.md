# streamoe

**Big models. Small machines. Hard limits.**

streamoe is a pure-Rust Mixture-of-Experts inference engine for Apple
Silicon, built on one conviction: *the size of a model should not
decide which machine can run it — your memory budget should.* You set a
cap; streamoe plans every byte at open, streams experts from SSD
through a hot tier it manages itself, and keeps to that plan. The design is called **AEGIS** — Adaptive · Expert ·
Guarded · In-memory · Streaming; see [Architecture](#architecture--aegis).

Today that means a 35-billion-parameter vision-language model —
Qwen3.6-35B-A3B, 19 GiB at 4-bit — answering questions about images
in **4 GiB of RAM** on a shared laptop, at 6–9 tokens per second
([Speed](#speed-at-a-glance)), making the same greedy choices as the MLX reference.

<p align="center">
  <img src="docs/media/chat-image.gif" width="800"
       alt="tools/chat: an image is attached, the vision encoder runs, then Qwen3.6-35B-A3B describes it and counts three red dots at 5.9 tok/s inside a 4 GiB cap">
  <br>
  <sub><code>tools/chat</code> on an M3 Pro · 4-bit experts, 4 GiB cap · an image turn (309-token prompt) ·
  recorded in real time, not sped up · <b>5.9 tok/s</b> · 3.73 GiB peak · <a href="#chat-page">more samples</a></sub>
</p>

> **Lineage.** streamoe grew out of
> [moeflux](https://github.com/mdegans/moeflux), which grew out of
> [flash-moe](https://github.com/danveloper/flash-moe); the revamp
> started on the author's moeflux fork,
> [transybao1393/moeflux](https://github.com/transybao1393/moeflux).
> It keeps their
> engine and Metal kernels and rebuilds everything around them: a
> strict memory contract, cache-free expert streaming with a
> frequency-managed hot tier, layer-range execution, multiple
> requests in flight, compact KV caches and vision input. Git
> history keeps the whole lineage; see [Provenance](#provenance).
>
> **License.** streamoe is source-available: free for personal,
> academic and other noncommercial use; commercial use needs a license
> from the author. Code inherited from moeflux and flash-moe stays MIT.
> See [License](#license).

## Direction

Inference engines usually ask *"how much memory does this model need?"*
and stop there. streamoe turns the question around: *"here is the
memory you may use — make the model fit, and make it fast."* Everything
in the design follows from that inversion.

- **Memory is a contract, not an estimate.** The cap covers weights,
  KV, scratch, checkpoints, driver overhead, the vision encoder,
  network buffers — and the page cache the engine would otherwise
  cause. Every account is planned, allocated and touched at open. If
  the plan does not fit, `open` fails before anything else on the
  machine is affected.
- **Disk is a tier, not a fallback.** A Mac's internal SSD is rated
  ~6.7 GB/s (~4.4 GB/s at decode's read sizes). With `F_NOCACHE`
  streaming, a fixed hot tier and speculative prefetch, a model 4–6×
  larger than its budget decodes at 6–9 tokens per second.
- **Speculation never changes the math.** Prefetch guesses, cross-layer
  predictions and hot-tier admission only decide *what to read early*.
  Routing always uses the model's own logits; output is identical
  across caps.
- **One Mac, done well.** This repository is the single-node engine
  (mode A): one machine runs every layer inside its cap. Splitting a
  model across several Macs (mode B) is handled by a separate
  repository built on this engine — see
  [Mode B](#mode-b--multiple-macs-separate-repository).
- **Every claim is measured.** Correctness is checked against MLX and
  transformers, CPU against Metal, batched against per-token.
  Telemetry counts hits, reads and waits per token, per layer and per
  tier. Speed numbers in this README are measured (the one estimate is
  labelled), and the ideas that did not pay off are written down in
  [`docs/known-issues.md`](docs/known-issues.md) so nobody retries them
  blind.

Where it goes next: per-expert precision chosen at run time (4-bit,
packed 3-bit or skip, by a router policy outside the engine), a faster GPU
side of decode, and more model families — see
[Roadmap](#roadmap).

## Architecture — AEGIS

The engine's design is called **AEGIS**: **A**daptive · **E**xpert ·
**G**uarded · **I**n-memory · **S**treaming. An aegis is a shield. The
memory cap is its rim: nothing inside crosses it, nothing outside it is
disturbed, and everything inside works to run a model several times
larger than the shield — fast.

| | Principle | What does it |
| --- | --- | --- |
| **A** · Adaptive | Fit the work to the cap and to what the model is doing right now | cross-layer expert guesses, per-slot prefetch waits, LFU hot-tier admission, lending idle buffers, 4-bit or packed 3-bit experts |
| **E** · Expert | Make every routed expert cheap to read and to run | packed 3-bit layout (−22% bytes, detected at open), fused gather kernels (`gather_mm_id`), `qmv` for single-token matvecs |
| **G** · Guarded | Memory is a contract, not an estimate | the budget plan at open (`OverBudget` before anything is touched), `F_NOCACHE` so neighbours keep their page cache, vision and KV inside the plan |
| **I** · In-memory | Bytes already in RAM do as much work as they can | the hot tier (zero-copy bind), the slot ring, compact KV (F16, Turbo4/3), prefill staging lent to decode |
| **S** · Streaming | Disk is a tier, and its time is measured | parallel `F_NOCACHE` pread overlapped with GPU work, `STREAMMOE_IO_TRACE` and `STREAMMOE_GPU_TIMING` timelines |

### Data flow

<p align="center">
  <img src="docs/media/aegis-dataflow.svg" width="100%"
       alt="AEGIS data flow, left to right: the application sends a prompt or image to Ctx and the budget plan, then attention, router, expert kernels and the LM head inside the red memory-cap rim, looping over 40 layers; the next token returns to the application. Below, the router's picks go to the adaptive guesser, which prefetches into the slot ring; the slot ring and the hot tier bind experts to the kernels; non-expert weights and the KV cache feed attention. Outside the rim, the SSD streams expert bytes into the slot ring with F_NOCACHE reads.">
</p>

Tokens (indigo) run left to right through each of the 40 layers and
come back as the next token. The red rim is the memory cap: only the
SSD sits outside it, and the only traffic across the rim is expert
bytes coming in through the slot ring (grey).

### G · Guarded: memory planned, not hoped for

At open, streamoe sizes every account for the context you asked for —
`n_ctx`, KV format, request slots, checkpoints, host buffers — and the
expert hot tier takes exactly what is left. Expert reads use
`F_NOCACHE`, so the engine never grows the page cache and nothing else
on the machine loses its file cache. The plan `tools/chat` prints at
open for the [standard setup](docs/benchmarks.md#test-setup) with 4-bit experts and a
4 GiB cap:

```mermaid
pie showData
    title Qwen3.6-35B-A3B · 4-bit experts · 4 GiB cap (GiB)
    "Hot tier (expert cache)" : 1.05
    "Non-expert weights" : 1.29
    "Runtime / Metal overhead" : 0.50
    "Prefill staging" : 0.42
    "Prefill scratch" : 0.23
    "KV cache (F32, n_ctx 4096)" : 0.16
    "Vision encoder buffers" : 0.13
    "Linear-attention state" : 0.12
    "Expert slot ring" : 0.09
```

With packed 3-bit experts the prefill staging shrinks to 0.33 GiB and
the hot tier grows to 1.14 GiB. In the [speed runs](docs/benchmarks.md#speed) the peak
stayed 0.26–0.31 GiB under every cap (2.7 / 3.7 / 4.7 / 5.7 GiB at
3 / 4 / 5 / 6 GiB). One known exception: very long runs can outgrow the
0.5 GiB runtime allowance — open in
[`docs/known-issues.md`](docs/known-issues.md). A Turbo4 KV cache
shrinks the KV slice to 0.02 GiB and the hot tier grows by the
difference; on models with large KV caches (OLMoE-1B-7B at n_ctx 4096)
that trade alone made decode 44% faster.

### I · In-memory: prefill staging serves decode

The prefill staging buffer (0.42 GiB, one slot per expert) is only
written while a prompt is read. During decode it would sit idle, so the
hot tier borrows it: 256 more expert slots without a byte over the
plan. The next prompt takes it back before its first chunk.

```mermaid
stateDiagram-v2
    direction LR
    classDef prefill fill:#475569,stroke:#1e293b,color:#ffffff
    classDef decode fill:#b45309,stroke:#78350f,color:#ffffff

    state "Prefill<br/>staging holds one layer's experts<br/>hot tier: 637 slots" as P
    state "Decode<br/>staging lent to the hot tier<br/>hot tier: 637 + 256 slots" as D

    [*] --> P: prompt arrives
    P --> D: first decode token · lend()
    D --> D: misses fill the lent slots (LFU)
    D --> P: next prompt · reclaim() — lent slots forgotten
    class P prefill
    class D decode
```

In the [standard setup](docs/benchmarks.md#test-setup) (4-bit experts, 4 GiB cap) this
lifts hot-tier hits from 51 / 22 / 42% to 59 / 34 / 54% on the three
test prompts and decode from 6.9 to 7.4 tok/s (+7%).
`STREAMMOE_HOT_LEND=0` turns it off.

### E · Expert: packed 3-bit

Decode is bound by bytes from the SSD, so the cheapest expert is the
smallest one that still gives the same answers. Packed 3-bit stores
8 codes in 3 bytes instead of 4; the kernels keep the 4-bit element
order and `w = scale · q + bias`, so packed and nibble-stored 3-bit
give bit-identical output.

```mermaid
flowchart LR
    classDef e fill:#0f766e,stroke:#134e4a,color:#ffffff
    classDef i fill:#b45309,stroke:#78350f,color:#ffffff
    classDef s fill:#475569,stroke:#1e293b,color:#ffffff

    Q4["<b>4-bit expert</b> · 1.69 MiB<br/>gate | up | down<br/>8 codes → one u32<br/>bf16 scale + bias per 64"]:::s
    Q3["<b>Packed 3-bit expert</b> · 1.31 MiB (−22%)<br/>gate | up | down<br/>8 codes → 3 bytes (24-bit pack)<br/>bf16 scale + bias per 64"]:::s
    OPEN["<b>At open</b><br/>layout read from the<br/>layer file size"]:::e
    K["<b>Kernels</b><br/>dequant_matvec_3bit (decode)<br/>gather_mm_id bits=3 (prefill)"]:::e
    RAM["<b>4 GiB cap</b><br/>hot tier 637 → 893 experts<br/>7.4 → 8.1 tok/s"]:::i

    Q4 -- "requant_experts.py<br/>pack_experts.py" --> Q3 --> OPEN --> K --> RAM
```

Against 4-bit: KL 0.048, the same top token 93% of the time, all 10
scored tasks right (`tools/chat` `quality` bin). 2-bit broke tool
calls, maths, JSON and code, so it is not offered.

### A + S · One layer of decode

Each layer hides as much of its I/O as it can behind GPU work:

```mermaid
sequenceDiagram
    autonumber
    participant C as CPU
    participant IO as I/O pool
    participant G as GPU
    participant H as Hot tier
    participant S as SSD

    Note over C: layer L begins (layer L−1's GPU work drained)
    C->>C: A · guess L's experts — cross-layer from L−1's residual,<br/>or the previous token's picks
    C->>H: drop guesses that are already hot
    C-)IO: S · prefetch the rest into slot set L mod 2
    par overlapped
        G->>G: attention + router logits
        Note right of G: full-attention layers under a cap run<br/>on CPU threads (attn_threads)
    and
        IO->>S: parallel F_NOCACHE pread
        S-->>IO: expert bytes → prefetch slots
    end
    G->>C: router logits
    C->>C: top-k on the model's own logits
    C->>H: I · hot hits bind their RAM slot (zero copy, zero I/O)
    C->>S: misses: parallel pread → synced slots
    C->>IO: A · wait only for the prefetched slots L picked
    Note over C,IO: an unused guess still in flight is not waited for
    C->>G: E · K-expert matvec + combine, chained into L+1's norm
    C->>H: A · count picks, admit misses hotter than a sampled victim (LFU)
```

On Qwen3.6-35B-A3B (40 layers, 256 experts, 8 picked per token) the
cross-layer guess with 6 experts prefetched uses 78–121 prefetched
experts per token instead of 13–35 with the previous-token guess, and
decodes 14% faster (6.5 → 7.4 tok/s, 4-bit, 4 GiB) — with identical
output; see [Prefetch and reuse](docs/benchmarks.md#prefetch-and-reuse). Guesses only decide *what to
read early*; routing always uses the model's own logits.

A decode token at a 4 GiB cap (4-bit experts, 138 ms) spends ~44% of
its time in GPU kernels, ~36% waiting on the SSD and ~20% on CPU work
and GPU wake-up gaps; the CPU uses ~0.7 of 11 cores. So the levers are
fewer bytes from disk and fewer, longer GPU bursts — not more threads
([breakdown](docs/benchmarks.md#where-a-decode-token-goes)).

### S · Streaming: measured, not assumed

`STREAMMOE_IO_TRACE=<file>` records every expert read and the phase of
each layer; `tools/io_trace_report.py` turns it into disk busy / idle
time ([sample report](docs/benchmarks.md#disk-timeline)).
Prefill staging runs at the SSD's limit (~4.1–4.6 GB/s here), and the
disk's idle time in decode comes in ~1 ms pieces too short to fill with
more prefetch — the tries are in
[`docs/known-issues.md`](docs/known-issues.md). What lowers decode time
now is fewer bytes per token (packed 3-bit, a bigger cap) and less GPU
time.

### Mode A — one Mac, streaming

One machine runs every layer. A small hot tier serves the most-used
experts from RAM; the rest stream from disk on demand.

```mermaid
flowchart LR
    classDef mem fill:#b45309,stroke:#78350f,color:#ffffff
    classDef disk fill:#475569,stroke:#1e293b,color:#ffffff
    classDef engine fill:#0f766e,stroke:#134e4a,color:#ffffff

    subgraph MAC["One Mac · e.g. 4–10 GiB cap"]
        direction LR
        HOT["<b>Hot tier</b><br/>most-picked experts<br/>fixed size"]:::mem
        SLOTS["<b>Slot ring</b><br/>prefetched + synced"]:::mem
        ENG["<b>Engine</b><br/>all layers<br/>GPU + CPU"]:::engine
        HOT -- "zero-copy bind" --> ENG
        SLOTS -- "bind" --> ENG
    end
    SSD[("<b>SSD</b><br/>all experts")]:::disk -- "F_NOCACHE pread" --> SLOTS
    PROF["Routing profile<br/>from earlier runs"] -. "seeds at open" .-> HOT
```

### Mode B — multiple Macs (separate repository)

**This repository handles mode A only.** Running one model across
several Macs — nodes, a coordinator, a layer-range planner, network
frames — is handled by a separate repository that builds on this
engine.

What stays here is the per-range engine API a pipeline needs:
`OpenConfig::layer_range` (open, plan and run only layers `a..b` inside
the cap), `Ctx::eval_stage` with f32 hidden states in and out, and
per-range snapshots. `crates/streammoe-net/` is the earlier prototype of
the pipeline (TCP loopback tested); it stays as a reference and is not
developed further here.

### Vision inside the shield

Qwen3.5 / 3.6 take images through a CPU vision encoder that lives inside
the same cap: its weights stream from disk per image, its 137 MiB of
buffers are part of the plan.

```mermaid
flowchart LR
    classDef engine fill:#0f766e,stroke:#134e4a,color:#ffffff
    classDef disk fill:#475569,stroke:#1e293b,color:#ffffff

    IMG["<b>Image</b><br/>resized to multiples of 32<br/>≤ 640 × 640"] --> PE["<b>Patch embed</b><br/>16 px patches<br/>+ learned positions"]:::engine
    PE --> BL["<b>ViT blocks</b><br/>2-D axial RoPE<br/>full-image attention"]:::engine
    BL --> MG["<b>Merger</b><br/>2 × 2 patches → 1 token"]:::engine
    MG --> LM["<b>Language model</b><br/>eval_stage_mrope<br/>tokens at [t, h, w]"]:::engine
    WT[("BF16 weights<br/>0.85 GiB on disk")]:::disk -. "F_NOCACHE, per matrix" .-> PE & BL & MG
```

The encoder matches transformers exactly and mlx_vlm to 0.997; a
640 × 640 image becomes 400 tokens in ~3 s.

### Decode attention across cores

When the GPU attention fast path is not used — streaming or capped
runs, or a quantized `kv_format` — each decode step's attention runs
on the CPU. Its query heads are independent, so they are split across
a small thread pool (`attn_threads`, default 4). Every thread reads
the same KV cache (read-only, shared with the GPU) and writes only its
own heads' slice of the output, so no locks are needed and the result
is bit-identical to one thread.

```mermaid
flowchart LR
    Q["<b>Query</b><br/>one token<br/>16 heads"] --> SPLIT{"split by head"}
    SPLIT --> T1["<b>Thread 1</b><br/>heads 0, 4, …"]
    SPLIT --> T2["<b>Thread 2</b><br/>heads 1, 5, …"]
    SPLIT --> T3["<b>Thread 3</b><br/>heads 2, 6, …"]
    SPLIT --> T4["<b>Thread 4</b><br/>heads 3, 7, …"]
    KV[("<b>KV cache</b><br/>F32 / F16 / Turbo<br/>read-only")] -.-> T1 & T2 & T3 & T4
    T1 & T2 & T3 & T4 --> OUT["<b>Output</b><br/>each head's slice"]
    OUT --> GPU["<b>GPU</b><br/>o_proj → experts"]
```

Per head, a thread scores the query against every cached key,
takes the softmax, and sums the weighted values. Turbo formats work
on the stored 4/3-bit codes directly, without decompressing the cache.
The pool picks up heads dynamically (work stealing), so the head
assignment above is illustrative. *Setup: OLMoE-1B-7B, 2.7k-token
context, F32 and Turbo4 KV:* this made decode 55–58% faster; short contexts gain little, since attention is
then a small part of each step.

Details — memory accounts, the per-token data path, phases — are in
[`docs/fork-upgrade-plan.md`](docs/fork-upgrade-plan.md); the
motivation is in
[`docs/moeflux fork proposed changes.md`](docs/moeflux%20fork%20proposed%20changes.md).

### Repository layout

The project is called streamoe; its crates, binaries and environment
variables are spelled `streammoe` / `STREAMMOE_*`.

- `crates/streammoe/` — the engine. `RsCtx` (re-exported as `Ctx`) is
  the public surface. `src/riir/` holds `backend/{cpu,gpu}`, `attn/`,
  `moe/`, `io/` (expert streaming, prefetch), `snapshot/`
  (state save/load) and `variants.rs` (compile-time model shapes).
  Kernels in `shaders/shaders.metal` are embedded via `include_str!`
  and compiled at runtime.
- `crates/streammoe-metal/` — MLX-derived quantized GEMM and gather
  kernels.
- `crates/streammoe-net/` — earlier mode B prototype (frames,
  `streammoe-node`, `streammoe-coord`, range planner); kept as a reference,
  not developed further here — mode B lives in a separate repository.
- `scripts/` — model-prep pipeline (`extract_weights.py`,
  `export_vocab.py`, `export_tokenizer.py`, `repack_experts.py`), the test-host check
  (`check_test_host.sh`, run before any test phase), profiling
  (`profile_smoke.sh`, `profile_aggregate.py`) and `watch_run.sh`
  (samples a run's memory and CPU).
- `tools/` — model converters (`convert_olmoe.py`,
  `convert_cogito_v2.py`), a NumPy reference forward for OLMoE
  (`olmoe_ref.py`), an expert-cache simulator that replays routing
  traces (`sim_expert_cache.py`), `gen_expert_index.py`,
  `export_vision.py` (the Qwen3.5 / 3.6 vision tower for
  `riir::vision`), `requant_experts.py` / `pack_experts.py` (packed
  3-bit experts), `io_trace_report.py`, `chat/` (a local chat page for
  Qwen3.6-35B-A3B under a cap, with images and a live CPU / RAM bar,
  plus the `quality`, `verify_bench`, `spec_bench` and `batch_bench`
  bins), and
  `mlx_reference/` — the MLX reference diff harness
  (`crates/streammoe/tests/mlx_regression.rs` regenerates its goldens).
- `docs/` — design notes and the upgrade plan.

## Status

**Alpha** — `0.2.0-alpha.1` (unreleased changes since then in the changelog), based on moeflux `0.1.0-pre.4`. APIs may still change; changes are listed in [`CHANGELOG.md`](CHANGELOG.md).

| Area | State |
| --- | --- |
| Engine, Metal kernels, Qwen3.5-A17B / Qwen3.6-35B-A3B / Cogito-V2 variants | Exists |
| OLMoE-1B-7B variant (small test model: converter, NumPy reference, model tests under a cap) | Implemented — the model-level test bed on hosts without a3b weights |
| Expert streaming: mmap mode, pread + speculative prefetch, slot ring | Exists |
| Differential oracles (CPU vs Metal, batched vs per-token, MLX) | Exists |
| `OpenConfig`, `n_ctx`-sized KV, memory cap (plan, reserve at open, footprint checks) | Implemented — Phase 1; tested on a3b by the model suites ([Tests](#tests)): open fails before any I/O when the plan doesn't fit, checkpoints planned inside a 4 GiB cap |
| `F_NOCACHE` reads, fixed hot tier, routing profiles, expert-IO counters | Implemented — Phase 2; tested on a3b by the model suites ([Tests](#tests)): hot-tier sizes and profile seeding give bit-identical logits |
| Layer-range execution (`layer_range`, `eval_stage`, f32 hidden in/out, per-range snapshots) | Implemented — Phase 3; tested on a3b by the model suites ([Tests](#tests)): whole-model `eval_stage` bit-identical to `eval_prompt`, 40 layers vs 20 + 10 + 10 within tolerance, per-range snapshots round-trip |
| Per-request state (`max_requests` pool, `admit_request` / `drop_request` / `eval_request`) | Implemented — Phase 4; tested on a3b by the model suites ([Tests](#tests)): interleaved requests bit-identical to sequential runs |
| `streammoe-net`: frames, `streammoe-node`, `streammoe-coord`, planner (TCP loopback) | Prototype — Phase 5; mode B continues in a separate repository |
| Streaming improvements: telemetry (`ExpertIoStats`, CSV), bounded I/O (`io_queue_depth`, `io_bytes_per_sec`), in-place coalesced prefill reads, cross-layer guesses, lookahead prefetch, `fast_experts_dir` disk tier, `prefetch_experts` (prefetch width) | Implemented — Phase 6; tested on a3b by the model suites ([Tests](#tests)): bounded I/O, prefill reads, cross-layer guess, lookahead and the fast-dir tier all give identical output. Library defaults: previous-token guess, `prefetch_experts` = 4 under a cap; `tools/chat` uses the cross-layer guess with 6, faster on a3b ([Prefetch and reuse](docs/benchmarks.md#prefetch-and-reuse)) |
| KV cache formats (`kv_format`: F32, F16, TurboQuant 4/3-bit; snapshot v4) | Implemented — F32 default; on OLMoE-1B-7B at a 3 GiB cap and n_ctx 4096, F16 / Turbo4 shrink KV 1.00 → 0.56 / 0.19 GiB (incl. prefill scratch) and decode +29% / +44% faster (the freed memory goes to the hot tier); answers correct in every format. MLA (Cogito) stays F32 |
| Decode attention threads (`attn_threads`, default 4) | Implemented — bit-identical output; on OLMoE at 2.7k context decode +55–58% tok/s (F32 and Turbo4) |
| Vision for Qwen3.5 / 3.6 (`riir::vision` CPU encoder, `eval_stage_mrope`, `tools/export_vision.py`) | Implemented — encoder matches transformers exactly and mlx_vlm to 0.997; whole pipeline checked layer by layer against MLX on a3b under a 4 GiB cap (encoder 137 MiB, ~1–3 s per image). Small printed text reads poorly, as it does in MLX |
| Decode I/O: per-slot prefetch waits, prefill staging lent to the hot tier (`STREAMMOE_HOT_LEND`), single-token `qmv` matvec kernel (`STREAMMOE_MATVEC_QMV`), hot experts dispatched before disk reads (`STREAMMOE_EARLY_EXPERTS`) | Implemented — output unchanged; effect of each in [Prefetch and reuse](docs/benchmarks.md#prefetch-and-reuse) |
| Packed 3-bit experts (`tools/requant_experts.py` → `tools/pack_experts.py`, `root-3bit/`; layout detected from the files), quality harness (`tools/chat` `quality` bin: KL, top-1, perplexity, scored tasks), `--precision 4\|3` in the chat page | Implemented — 1.31 MiB per expert vs 1.69 (−22%); vs 4-bit: KL 0.048, top-1 agreement 93%, 10/10 tasks; 9–15% more tok/s at every cap from 3 to 6 GiB ([Speed](#speed-at-a-glance)). 2-bit fails 4 of 10 tasks |
| Several tokens per pass: `eval_tokens` + `rewind_tokens` (one sequence, speculative checks), `eval_requests` (one token from each of up to 4 requests), `OpenConfig::max_decode_tokens` (buffers in the plan) | Implemented — logits bit-identical to one token per pass; parallel decoding 1.3× total throughput, speculative decoding not yet a win ([Several tokens per pass](docs/benchmarks.md#several-tokens-per-pass)) |
| Router policy, per-expert precision | Planned — Phases 7–8 |

Build with exactly one model feature (`model-qwen3-6-35b-a3b`,
`model-qwen3-5-a17b`, `model-cogito-v2-671b`, `model-olmoe-1b-7b`):

```bash
cargo build --features model-qwen3-6-35b-a3b
# the whole workspace, streammoe-net included
cargo build --workspace --features model-qwen3-6-35b-a3b
```

Opening under a cap, with a compact KV cache:

```rust
use streammoe::{Ctx, KvFormat, OpenConfig};

let cfg = OpenConfig {
    n_ctx: 4096,
    mem_cap: Some(10 << 30),        // hard cap, bytes
    kv_format: KvFormat::Turbo4,    // F32 (default) | F16 | Turbo4 | Turbo3
    ..OpenConfig::default()         // hot tier takes what the plan leaves
};
let mut ctx = Ctx::open_with(&weights, &manifest, &vocab, &experts_root, false, &cfg)?;
```

`open_with` fails with `Error::OverBudget` if the plan does not fit the cap.

### Tests

Run `scripts/check_test_host.sh` first; it says which tier (unit /
kernel, or model) this host can run. The model suites need the a3b
files (`STREAMMOE_SMOKE_ARTIFACTS`, `STREAMMOE_SMOKE_ROOT`; see
[Setup](#setup)). Suites that open the model uncapped take
`STREAMMOE_TEST_CAP_GIB=4` to run under a cap on a small host:

```bash
STREAMMOE_TEST_CAP_GIB=4 cargo test --release --features model-qwen3-6-35b-a3b \
    -- --include-ignored --test-threads=1 --skip resuming_prefill_after_seq_rm_matches_full_prefill
```

Last full run (2026-10-26, M3 Pro with 7 GiB free, release build):

| Suite | a3b result |
| --- | --- |
| `mlx_regression` (argmax and top-20 vs the MLX golden), `smoke` | pass |
| `checkpoint_restore` (23), `consecutive_eval_prompt` (3), `mrope` (3) | pass |
| `layer_range` (4), `requests` (1), `streaming` (7), `expert_source_equivalence` (2), `budget_cap` (1) | pass |
| `graph_diff_oracle` (22, CPU vs Metal), `batched_diff_oracle` (33, batched vs per-token kernels) | pass |
| The default tier (169 unit and kernel tests, no model files) | pass |
| `diff_oracle` (opens two contexts at once), the two 10 GiB-cap tests, `mmap_matches_pread` and `engine_op_diff` (need mmap mode), `vision` (needs reference embeddings) | not run — need ≥ 8 GiB free or extra data |
| `resuming_prefill_after_seq_rm_matches_full_prefill` | known failure, skipped ([known issues](docs/known-issues.md)) |
| a17b and Cogito model tests | not run — no weights on this host |

Fixed bugs, open issues and traps are listed in
[`docs/known-issues.md`](docs/known-issues.md).

## Roadmap

- **Faster time to first word.** Expert staging is ~74% of prefill and
  runs one layer at a time while the CPU sits ~95% idle; staging layer
  L+1 while the GPU computes layer L could cut it by a third.
- **Router policy and per-expert precision** (phases 7–8). After
  routing, a policy outside the engine decides per expert whether to
  run it at 4-bit, at 3-bit or skip it — trading quality for I/O one
  expert at a time instead of quantizing the whole model.
- **Fewer, longer GPU bursts.** ~87 ms of every token is GPU and CPU
  work that does not shrink with the cap (constant from 3 to 6 GiB,
  [Speed](docs/benchmarks.md#speed)). GPU timestamps (`STREAMMOE_GPU_TIMING`) show decode
  kernels running 3–4× slower than in isolation: the CPU must see each
  layer's routing before it can fetch that layer's experts, and the
  resulting short bursts leave the GPU in a slow state. Dispatching hot
  experts before the disk reads land was a first step; more of each
  layer's work has to move ahead of that per-layer sync.
- **More model families.** Qwen3-30B-A3B (`qwen3_moe`) as a new
  variant, then runtime variant dispatch instead of one compile-time
  feature per model.
- **Verification at full size.** `diff_oracle`, the mmap-mode tests
  and the a17b and Cogito model runs need a host with ≥ 8 GiB to spare
  (or the weights).

## Running Qwen3.6-35B-A3B on a small Mac

The model is 19 GiB at 4-bit. Under a cap streamoe keeps only the
hottest experts in RAM and streams the rest from the SSD, so it runs in
3–6 GiB.

### What it costs

| Resource | Cost |
| --- | --- |
| Disk | 19.0 GiB download (`mlx-community/Qwen3.6-35B-A3B-4bit`) + 19.0 GiB converted (1.29 GiB non-expert weights, 16.9 GiB 4-bit experts, 0.84 GiB vision weights) ≈ 38 GiB; packed 3-bit experts add 13.1 GiB. The MLX download can be deleted after conversion if you don't need `engine_op_diff` |
| RAM | Whatever you cap it at: peaks of 2.7 / 3.7 / 4.7 / 5.7 GiB at 3 / 4 / 5 / 6 GiB caps. Below ~3 GiB the fixed parts don't fit and `open_with` fails with `OverBudget` |
| Page cache | None from expert reads: under a cap they use `F_NOCACHE`, so the file cache other processes rely on is not evicted |
| CPU | ~0.7 cores while decoding: 4 threads for decode attention (`attn_threads`) plus the I/O threads; the GPU does the rest |
| SSD | 87–477 MiB read per token, depending on cap, expert format and prompt ([per prompt](docs/benchmarks.md#speed)) |

Fixed parts of the plan (n_ctx 4096): weights 1.29 GiB, runtime
overhead 0.50, prefill staging 0.42 (0.33 packed 3-bit), prefill scratch
0.23, KV 0.16 (F32; 0.02 Turbo4), vision buffers 0.13, linear-attention
state 0.12, expert slots 0.09 — 2.95 GiB with 4-bit experts. The hot
tier gets the rest of the cap.

### Speed at a glance

*[Standard setup](docs/benchmarks.md#test-setup): M3 Pro (18 GB RAM,
shared with other services), n_ctx 4096, F32 KV, cross-layer guess with
6 experts prefetched, greedy, three chat prompts, mean of 2 runs.*
Decode tok/s over all three prompts:

| Cap | 4-bit experts | Packed 3-bit experts | Peak footprint |
| --- | --- | --- | --- |
| 3 GiB | 6.5 | 7.5 | 2.7 GiB |
| 4 GiB | 7.4 | 8.1 | 3.7 GiB |
| 5 GiB | 7.9 | 8.6 | 4.7 GiB |
| 6 GiB | 8.3 | 9.2 | 5.7 GiB |

- Packed 3-bit is 9–15% faster at every cap. Against 4-bit: KL 0.048,
  the same top token 93% of the time, all 10 scored tasks right.
- Output is identical across caps; with 4-bit experts it matches the
  MLX reference (argmax equal, top-20 overlap 20/20).
- ~87 ms of each token is GPU and CPU work that a bigger cap does not
  remove, so with every expert in RAM decode would reach about
  11.5 tok/s (*estimated*; this host can't spare 16–20 GiB).
- Several requests at once (`eval_requests`, batch 4) give 1.30–1.33×
  the total throughput; speculative decoding is not a win yet.

Per-prompt numbers, hot-tier hits, disk reads, prefetch and reuse
comparisons, multi-token passes, the per-token breakdown and vision
timings are in [`docs/benchmarks.md`](docs/benchmarks.md).

### Setup

```bash
scripts/check_test_host.sh               # CPU, RAM, headroom, disk

M=~/models/qwen3-6-35b-a3b
hf download mlx-community/Qwen3.6-35B-A3B-4bit --local-dir $M/mlx-4bit

python3 -I scripts/extract_weights.py --model $M/mlx-4bit --output $M/artifacts
python3 -I scripts/export_vocab.py $M/mlx-4bit/tokenizer.json $M/artifacts/vocab.bin
python3 -I tools/gen_expert_index.py --mlx-path $M/mlx-4bit --out $M/expert_index.json
python3 -I scripts/repack_experts.py --index $M/expert_index.json --output-dir $M/root/packed_experts
python3 -I tools/export_vision.py --model $M/mlx-4bit --out $M/artifacts   # optional: images
# optional: packed 3-bit experts (13.1 GiB more disk; 9–15% faster, slightly lower quality)
python3 -I tools/requant_experts.py $M/root --out 3=$M/root-3bit-nibble
python3 -I tools/pack_experts.py $M/root-3bit-nibble $M/root-3bit && rm -r $M/root-3bit-nibble
```

Conversion needs Python with NumPy, peaks under 0.3 GiB of RAM and
takes about 20 s once downloaded.

Then open it under a cap — `OpenConfig { mem_cap: Some(4 << 30),
n_ctx: 4096, max_checkpoints: 0, .. }` — or point the tests at it:

```bash
export STREAMMOE_SMOKE_ARTIFACTS=$M/artifacts STREAMMOE_SMOKE_ROOT=$M/root STREAMMOE_SMOKE_MLX=$M/mlx-4bit
```

`tests/mrope.rs` (multimodal positions) runs the model under a 3 GiB
cap. `tests/vision.rs` checks the vision encoder against reference
embeddings from transformers: set `STREAMMOE_VISION_DIR=$M/artifacts` and
`STREAMMOE_VISION_REF` to a folder of `<name>.rgb` (resized RGB bytes),
`<name>.emb` (f32 rows) and `meta.json` (`{name: {h, w}}`).

The a3b oracle suites (`--include-ignored`) open the model *without* a
cap and need ≥ 8 GiB reclaimable; `check_test_host.sh` says whether this
host qualifies. On a smaller host, run the model only under a cap.

### Chat page

`tools/chat` is a local chat page for trying the model yourself
(standalone crate; the engine gains no dependencies):

```bash
cd tools/chat
cargo run --release -- --cap-gib 4 --n-ctx 8192
# options: --kv f32|f16|turbo4, --attn-threads N, --guess cross|prev, --prefetch N,
#          --precision 4|3 (3 needs $M/root-3bit, see Setup)
# open http://127.0.0.1:8077
```

It listens on 127.0.0.1 only and keeps one conversation in the KV
cache, so each message prefills only the new text. The header shows
memory against the cap and the context used; each answer shows tok/s,
prefill speed, hot-tier hits and disk MiB per token. Thinking can be
toggled. An **Experts** selector in the header switches between
4-bit and packed 3-bit experts (the server restarts itself in ~2 s
and frees the old model first, so memory stays under the cap). Its
defaults are the [standard setup](docs/benchmarks.md#test-setup) with 4-bit experts:
7.4 tok/s over the three test prompts at a 4 GiB cap. Stop the server
(Ctrl-C) to give the memory back.

#### What it looks like

*Setup: standard, 4-bit experts, 4 GiB cap; recorded with headless
Chrome on 2026-10-26, a few engine changes before the [Speed](docs/benchmarks.md#speed)
runs, with other prompts.* Real time — the GIFs are not sped up.

**Text** — 6.7 tok/s, 3.66 GiB of the 4 GiB cap:

<p align="center"><img src="docs/media/chat-text.gif" width="800" alt="tools/chat answering a text question at 6.7 tok/s under a 4 GiB cap"></p>

```text
58 tok · 6.7 tok/s · prompt 39 tok at 13 tok/s · hot-tier hits 46% · 329 MiB/tok from disk · avg CPU 57%
```

**Image** — 280 image tokens encoded and read in ~6 s, then 5.9 tok/s,
3.73 GiB peak. Each stage shows in the page as it happens:

| 1 · Vision encoder running | 2 · Answer streaming | 3 · Done, with the turn's stats |
| :---: | :---: | :---: |
| ![The page shows "looking at the image…" while the CPU vision encoder runs](docs/media/chat-image-encoding.png) | ![The answer streams in; the CPU / RAM bar updates live](docs/media/chat-image-generating.png) | ![The finished answer: three red dots, with tok/s, hot-tier hits and disk reads per token](docs/media/chat-image-done.png) |

```text
162 tok · 5.9 tok/s · prompt 309 tok (280 image) at 52 tok/s · hot-tier hits 35% · 394 MiB/tok from disk · avg CPU 66%
```

The image is a drawing made for this demo (house, tree, sun, one bird
and three red dots). The model names the house, tree and sun, counts
the three dots correctly, and reads the single bird as two.

A bar under the header shows the process's CPU (as cores used), its
memory against the cap with the peak, and the host's reclaimable
memory, sampled every second; it keeps updating while an answer is
generated. Each answer also reports its average CPU.

**Images.** With `artifacts/vision_weights.bin` exported, an *Image…*
button appears (paste works too). The page resizes the picture so both
sides are multiples of 32 and the area is at most `--vision-max-pixels`
(default 640 × 640 = 400 image tokens); the CPU vision encoder
(`streammoe::riir::vision`, Accelerate GEMMs, `--vision-threads` default
4) turns it into embeddings, and the model reads them at their M-RoPE
positions (`Ctx::eval_stage_mrope`). The encoder streams its 0.85 GiB
of BF16 weights from disk for each image and keeps 137 MiB of buffers,
counted in the cap (`OpenConfig::host_buffers`). `--no-vision` turns it
off.
Encoding a 640 × 640 image (400 tokens) takes ~3 s and the process
stays at ~3.7 GiB under a 4 GiB cap
([timings](docs/benchmarks.md#vision)).

Decode after an image runs at the usual speed. Scenes, colors, layout
and counting work (a 2 × 2 color grid, five dots, shapes and their
positions, a photo); reading small or short printed text is
unreliable. MLX makes the same mistakes when given the same image
embeddings (see `docs/known-issues.md`), so this comes from the model,
not the engine. Larger images help: "7780" read correctly at 640 × 320
and not at 512 × 256.

### Qwen3-30B-A3B

Qwen3-30B-A3B (`qwen3_moe`: 48 full-attention layers, 128 experts,
top-8, no linear attention) is **not supported yet**; it needs its own
variant. A size estimate is in
[`docs/benchmarks.md`](docs/benchmarks.md#estimates).

## Provenance

streamoe is built on [mdegans/moeflux](https://github.com/mdegans/moeflux)
(history up to moeflux `0.1.0-pre.4`, commit `a36ae3b`, is upstream's
work). The revamp began on the author's fork,
[transybao1393/moeflux](https://github.com/transybao1393/moeflux), where
it was published under MIT up to commit `7d73350`; development
continues here. moeflux itself began as a fork of
[danveloper/flash-moe](https://github.com/danveloper/flash-moe) and
was rewritten in Rust to the point of being a distinct codebase. The host-side engine is
Rust on `metal-rs`; the original C/Objective-C functions served as
differential oracles during that rewrite. The Metal
streaming-experts kernels were authored by **Claude Opus 4.6**
(Anthropic) for flash-moe and carry over here. streamoe adds the
memory cap, streaming, pipeline, KV-format, vision and tooling work
described above on top of that. The math is the same linear
algebra every inference engine runs; what streamoe contributes is the
systems design around it — the memory contract, the tiering, the
speculation that never changes the output, and the pipeline.

## License

streamoe is licensed in two parts; [`NOTICE`](NOTICE) has the exact
boundary.

- **MIT** ([`LICENSE-MIT`](LICENSE-MIT)) — everything in the repository
  as of commit `7d73350`: the flash-moe and moeflux code it was built
  on, and the streamoe work published under MIT up to that commit on
  [transybao1393/moeflux](https://github.com/transybao1393/moeflux). Also
  `crates/streammoe-metal/` (vendored MLX and llama.cpp kernels; see its
  own `NOTICE`). Check out `7d73350` for the MIT-only code.
- **PolyForm Noncommercial 1.0.0** ([`LICENSE.md`](LICENSE.md)) — every change
  after `7d73350`, starting with vision input, and all future work.

**Free:** personal study, research, experiments, hobby projects, and
use by schools, universities, public research organizations, charities
and government bodies.

**Commercial use** of the current tree — inside a product or service,
or for work done for a business — needs a commercial license. Contact
[@transybao1393](https://github.com/transybao1393).

streamoe is source-available, not open source in the OSI sense. See
also [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

## Acknowledgements

- **@danveloper** — for building the thing the hard way, writing it
  up, and publishing everything openly. moeflux started as flash-moe
  and git history reflects that.
- **@mdegans** — for moeflux, the Rust rewrite streamoe builds on.
- **[transybao1393/moeflux](https://github.com/transybao1393/moeflux)**
  — the author's moeflux fork, where the revamp began: the memory cap,
  `F_NOCACHE` streaming and hot tier, the layer-range pipeline, KV
  formats, decode attention threads and the chat page were built and
  published there under MIT (up to commit `7d73350`) before development
  moved to streamoe.
- **Claude Opus 4.6** — for the Metal streaming-experts kernels and
  the architecture that made all of this run.
- **Anthropic** — for making Claude available to do work like this in
  the first place.
