# moeflux fork: upgrade plan

> Written before the rename: `moeflux` names, crate paths and
> `MOEFLUX_*` env vars here are now `streammoe` / `STREAMMOE_*`.

Oct 25, 2026 · companion to `moeflux fork proposed changes.md`

This plan turns the proposal into an ordered set of changes, corrected
against the moeflux 0.1.0-pre.4 source. Two principles drive it:

1. **The memory cap is a hard ceiling.** The server runs other things that
   must not be touched. Nothing the engine does may push total memory use
   past the configured cap (e.g. 10 GB) — including the OS page cache it
   causes, not just the memory it allocates.
2. **Streaming experts stays the core of the engine, and gets stronger.**
   Mode A (single node) streams experts inside the cap; mode B (pipeline)
   runs the same streaming code with a hot tier big enough to hold its whole
   layer range.

## 1. What the code says (corrections to the proposal)

| Proposal assumption | Actual (pre.4) | Source |
| --- | --- | --- |
| Model is ~17.5 GB at 4-bit | Experts alone are 16.9 GiB (18.1 GB): 1,769,472 B × 256 × 40, plus non-expert weights | `variants.rs` `expert_size_4bit` |
| Unknown full-attention interleave | Every 4th layer: 3, 7, …, 39. A 20+10+10 split holds 5 / 2 / 3 full-attn layers | `variants.rs:251`, `full_attn_interval: 4` |
| Unknown expert I/O path | Both: mmap (GPU reads files directly) or pread + prefetch, chosen at open | `io/expert_io_mode.rs` |
| Router is buried in the forward pass | Decode reads gate logits back to the CPU and runs `moe_router_cpu` — a clean hook point | `attn/linear_attn_forward.rs:1683`, `:1877`, prefill `~:2737` |
| 2-bit is a reference to borrow | `dequant_matvec_2bit` kernel exists; `open(_use_2bit)` is ignored; no 2-bit files or batched 2-bit kernel | `shaders.metal:762`, `mod.rs:441` |
| No telemetry | `cmdbuf_stats`, `prefetch_stats`, `gpu_capture`, `scripts/profile_smoke.sh` exist | `mod.rs`, `riir/gpu_capture.rs` |
| Hidden state as BF16 on the wire | Engine works in f32; BF16 would break exact network equivalence. Use f32 (8 KB/token) | — |
| Multi-request needs batching | Pipeline throughput needs per-request *state switching*, not batched decode | see Phase 4 |

## 2. What breaks a hard cap today

| Hidden growth | Where | Why it escapes the cap |
| --- | --- | --- |
| Expert reads fill the page cache | `io/expert_io.rs:432` (`file.read_at` on a normal fd) | Page cache is not in `phys_footprint` but takes RAM from other services |
| Every expert file is mmapped; mmap mode lets the GPU read it directly | `io/expert_io.rs:221` | File-backed pages grow as touched; up to 16.9 GiB |
| Non-expert weights are mmapped | `io/weight_file.rs:96` | Same: page cache, invisible to footprint |
| KV lazily committed for 1M tokens | `snapshot/state.rs:91`, `variants.rs:372` | 4 GiB virtual per full-attn layer; grows 40 KiB/token without limit |
| Automatic I/O mode check | `io/expert_io_mode.rs:110` | Compares experts to physical RAM, not the cap; picks mmap on a 24 GB M4 (16.9 < 18 GiB) and pread on a 16 GB M1 in mode B even when the node's range fits |
| Prefill staging | pread prefill path | Up to 453 MB per layer, not accounted |
| Metal/driver/shader overhead, thread stacks, heap | everywhere | Real but unaccounted |

## 3. Strict-cap rules

1. **No page cache.** Every model-file read (experts, non-expert weights,
   vocab) goes through an fd opened with `F_NOCACHE`. No mmap of model files
   when a cap is set, so mmap mode is off under a cap.
2. **Peak at open.** Every account is reserved *and touched* at open:
   weights, hot tier, slot ring, prefill staging, KV for the configured
   `n_ctx`, the request-state pool. If it does not fit, `open` fails before
   anything else is affected. After open, generation cannot grow memory.
3. **Admission, not growth.** A request reserves KV + DeltaNet state from the
   pre-reserved pool; when the pool is full, the request is rejected.
4. **Fixed-size hot tier.** Sized at open from what is left; eviction swaps
   contents, never changes total size.
5. **Runtime overhead is an account.** Measured in Phase 0 (Metal, driver,
   shaders, stacks), plus a safety margin, reserved inside the cap.
6. **I/O and CPU are bounded too.** `io_threads`, queue depth and an optional
   bytes/s limit come from config; speculative prefetch counts against the
   I/O limit, so the shared SSD and cores are not flooded.

**Enforcement layers**

- *Prevent:* a ledger refuses any allocation that would cross the cap, with
  an error naming the account.
- *Detect:* a watchdog thread samples `phys_footprint` (`task_info`,
  `TASK_VM_INFO`) every few hundred ms. Footprint above the ledger total, or
  within a margin of the cap, stops new work and reports — it means an
  accounting bug.
- *Verify:* tests check `mincore()` over model files shows ~zero resident
  pages after a long generation.

macOS has no supported per-process hard memory limit (no cgroups; jetsam
limits are private API; a VM gives a hard cap but loses Metal GPU
performance). The guarantee therefore comes from the engine itself: all
reservations at open, all model I/O through `F_NOCACHE`, the watchdog as a
tripwire.

## 4. Streaming design under the cap

### Budget (a3b, mode A, 10 GB)

| Account | Size | Note |
| --- | --- | --- |
| Non-expert weights | ~1.3 GB | Estimate — measure from `model_weights.bin` |
| KV, `n_ctx` = 32k | 1.25 GiB | 40 KiB/token (8k = 320 MB) |
| DeltaNet state per request | ~60 MB | 30 linear-attn layers |
| Slot ring | ~96 MiB | Exists: `MAX_K` × (1 synced + 2 prefetch sets), 2 MiB-aligned (`moe/expert_forward.rs:407`) |
| Prefill staging | 453 MB | One layer of experts |
| Runtime overhead + scratch | ~1 GB | Measured in Phase 0, with margin |
| **Hot tier (remainder)** | **~5.8 GB ≈ 3,300 experts** | ~32% of 10,240 |

### Per-token data path (decode, per layer)

```
GPU  CMD1: attention + router logits ─────────┐
I/O  prefetch guessed experts NOT in hot tier  │  (SSD → prefetch slots, F_NOCACHE pread)
CPU  read logits → top-8 (moe_router_cpu) ◄────┘
     for each of the 8 picks:
       hot tier hit   → bind hot buffer        (zero copy, zero I/O)
       prefetch hit   → bind prefetch slot
       miss           → read SSD → synced slot (parallel F_NOCACHE pread)
GPU  batched expert matvec (existing chunked dispatch)
CPU  off critical path: update frequency counts; maybe promote
     slot → hot tier, evicting the coldest (fixed size)
```

Core code change: add `SlotSource::Hot(id)` next to `Synced` /
`Prefetched(idx)` (`io/prefetch.rs:68`), a fixed hot-tier map, and skip
prefetch for experts already hot.

### Expected speed (external SSD, 2.8 GB/s assumed)

Each token needs 320 expert reads × 1.77 MB = 566 MB if nothing is cached.

| Picks served from RAM | Read per token | I/O ceiling |
| --- | --- | --- |
| 0% | 566 MB | ~5 tok/s |
| 50% | 283 MB | ~10 tok/s |
| 70% | 170 MB | ~16 tok/s |
| 80% | 113 MB | ~25 tok/s (likely compute-bound first) |

Expert use is skewed, so 32% of experts should serve more than 32% of
picks. Phase 0's routing profile measures how much more.

### Strengthening streaming (no extra memory)

1. **Profile-seeded hot tier:** count expert use per layer over
   representative prompts; load the most-used at open (no slow warm-up).
2. **Frequency-based promotion/eviction** at fixed size (recency-only LRU is
   what lost in flash-moe, `docs/io-and-gpu-exploration.md:92-110`).
3. **Better guesses:** run layer N+1's router on layer N's hidden state
   instead of reusing the last token's picks (25–30% hit today); only fetch
   confident, non-hot guesses.
4. **Overlap:** fetch the next layer's guessed cold experts while this layer
   computes (the two prefetch sets already exist for this).
5. **Prefill:** read only non-hot experts into staging; larger chunks spread
   reads over more tokens. A second staging buffer for read/compute overlap
   costs +453 MB — only if the budget allows.
6. **Disk tier (Option C):** keep the most important experts on the internal
   SSD (costs disk, not RAM). First check whether the whole 18 GB of experts
   simply fits on the internal SSD.
7. **Bounded I/O:** threads, queue depth, optional bytes/s from config.

### Mode B

Same engine, hot tier sized to the node's whole layer range, loaded at open
with `F_NOCACHE`, never misses. One code path for both modes keeps streaming
exercised everywhere.

## 5. Phases

Each phase ends with a measurement; the next phase is decided on numbers.

### Progress (Oct 25, 2026)

| Phase | State | Notes |
| --- | --- | --- |
| 0 | Partly done | `scripts/check_test_host.sh`; `RsCtx::expert_io_stats()` (disk reads/bytes, hot-tier hits/misses/admissions/evictions); `routing_counts()` / `save_routing_profile()`. Baselines need the M4 / M1 and the model. |
| 1 | Implemented | `OpenConfig` + `RsCtx::open_with` (`n_ctx`, `expert_io`, `io_threads`, `mem_cap`, `runtime_overhead`); KV sized from `n_ctx`; `budget.rs` plan, reserve-and-touch at open, `phys_footprint` check at open and after every `eval_*`; mmap refused under a cap. `layer_range` moves to Phase 3. |
| 2 | Implemented | Under a cap: weights read with `F_NOCACHE` into owned page-aligned memory, expert fds `F_NOCACHE` and not mmapped. `io/hot_tier.rs`: fixed-size, frequency-admitted, epoch-protected; decode binds hits zero-copy, prefetch skips hot experts, prefill copies hits. Seeding from a routing profile. |
| 3 | Implemented | `OpenConfig::layer_range`: only the range's tensors are loaded (compacted under a cap), planned (`BudgetPlan::for_range`) and run; `RsCtx::eval_stage(StageInput::{Tokens, Hidden}, pos, StageOutput::{Logits, Hidden, Discard})` with f32 hidden in/out; `chain_next` norm fusion broken at the range end; snapshot v3 header records the range. A mode B range seeds every expert of the range into the hot tier. |
| 4 | Implemented | `OpenConfig::max_requests`: a fixed pool of `SeqState`s (KV, linear-attn host + GPU recurrence, prefetch guesses), counted per slot in the plan and allocated at open under a cap; `admit_request` (`PoolFull` when full) / `drop_request` / `eval_request` / `state_{size,save,load}_for`. Switching requests swaps the live state with the parked one (no copies); the single-sequence API runs on `ReqId::DEFAULT`. |
| 5 | Implemented | `crates/moeflux-net`: versioned length-prefixed frames into fixed buffers; `moeflux-node` (one range per process, frame buffers counted in the cap via `OpenConfig::host_buffers`); `moeflux-coord run` (drives the chain, per-node decode latency) and `plan` (ranges from caps + measured speed). TCP loopback. |
| 6 | Implemented | Telemetry: `ExpertIoStats` from fixed per-layer atomics — reads/bytes by kind (demand / prefetch / prefill / seed, fast dir), per-layer hot / prefetch / demand counts, decode and prefill wall time with I/O wait, guess accuracy for both guessers; `csv_row` / `layer_csv`, `RsCtx::reset_expert_io_stats`. `OpenConfig::{io_queue_depth, io_bytes_per_sec}` (lock-free gate + GCRA limiter on every read). Prefill stages experts in place: deduped, sorted, coalesced runs of ≤ 8 on the I/O pool. `prefetch_guess = CrossLayer` (layer L's residual through L+1's router on the CPU). `prefetch_lookahead` (L+1's reads issued during L's attention). `fast_experts_dir` (per-layer disk tier). Every knob defaults off = Phase 2 behavior until measured. |
| 7–8 | Not started | Waiting on model-level verification of 1–6. |

Not yet verified on a model (this dev box has no weights and 18 GB):
the model groups of `tests/budget_cap.rs`, `tests/expert_source_equivalence.rs`,
`tests/layer_range.rs`, `tests/requests.rs`, `tests/streaming.rs` and
`moeflux-net/tests/loopback.rs` are `#[ignore]` until run on a capable
host. The cogito (MLA) decode path does not use the hot tier or
prefetch yet, and layer ranges / request switching are unit-tested but
not model-tested on cogito or a17b. Phase 6's exit numbers (read
bytes/token and tok/s vs Phase 2) come from
`tests/streaming.rs::streaming_stats_report` and the per-knob tests,
which print the stats CSV; flip a default only after it wins there.

### Deviations from the plan (phases 3–5)

- **Split equivalence is within tolerance, not bit-exact.** Breaking the
  `chain_next` norm fusion at a range boundary changes the order of one
  RMSNorm, so 40 vs 20+10+10 is compared with the oracle tolerance
  (`assert_logits_close`). A whole-range `eval_stage` matches
  `eval_prompt` / `eval_token` bit for bit, and network vs in-process
  split is bit for bit.
- **`eval_stage` with one row takes the decode path**; `eval_prompt` with
  one token takes the batched path. Compare like with like.
- **The deferred K-expert ring is not per request.** Every eval drains it
  before returning, so it is empty at every request switch; only the
  prefetch guesses move with the request.
- **Checkpoints (`checkpoint_pos` / `restore_to`) stay on the default
  request.** Per-request snapshots go through `state_save_for` /
  `state_load_for`.
- **Without a cap, the pool's GPU buffers are allocated lazily** at the
  first eval, like every other GPU resource on the uncapped path. Under
  a cap everything is allocated and touched at open.
- **Star topology.** Hidden states return to the coordinator between
  stages (one extra hop per stage). It keeps nodes independent and lets
  the coordinator time each node; a node-to-node ring is future work.
- **Network buffers are planned through `OpenConfig::host_buffers`**
  (new `host_buffers` budget account) rather than inside moeflux: the
  node allocates and touches its two frame buffers before opening.
- **The planner works from per-layer estimates** (manifest tensor sizes +
  the KV / linear-state formulas, fixed costs from `BudgetPlan`), with
  uniform per-layer work. Each node's own capped open stays the
  authoritative check.
- **The weight filter keeps unknown top-level tensors** on every range
  (only `model.layers.N.*`, the embedding and the head are split).

### Deviations from the plan (phase 6)

- **Nothing is on by default.** No speed could be measured here (no
  weights), so every Phase 6 knob ships off and the default path is
  the Phase 2 one; the stats CSV decides each default on the M4.
  Exception, measured since on OLMoE-1B-7B (6 GiB cap, 4 prompts ×
  256 tokens): `prefetch_experts` — speculative reads per layer, the
  previous token's picks by router weight — defaults to 4 under a cap.
  Reading all 8 wasted 42% of decode reads (27% of guesses used) on an
  SSD already at its ~6.7 GB/s ceiling. tok/s, all 8 / top 4 / top 2 /
  off: 7.4 / 8.5 / 8.3 / 8.1 with a 0.35 GiB hot tier, 11.3 / 11.8 /
  11.7 / 11.2 with 1.19 GiB; logits bit-identical. Uncapped keeps all
  K. Routing traces (`MOEFLUX_ROUTING_TRACE`) replayed in
  `tools/sim_expert_cache.py` found no better guess: a moving router
  score ties the previous token (±1 pt), recency-ranked and
  hot-skipping guesses are worse, LFU beats LRU and short-retention
  tiers, and a 64-slot recent-read window got no hits in the engine
  (≈145 slot fills per token).
- **Cross-layer input is layer L's post-attention residual** (`h_mid`),
  not layer L's output: the output only exists once L's experts have
  run on the GPU. It goes through L+1's `post_attention_layernorm` and
  router gate (4- or 8-bit, serial CPU matvec into fixed buffers) at
  the top of L+1, after L's expert dispatch is committed. So the guess
  is better but issued no earlier than today's; the overlap item is
  separate. No confidence threshold: the top-k guess is always
  fetched, minus hot experts.
- **Today's prefetch already overlaps more than the plan assumed:**
  layer L+1's prefetch is issued while L's expert dispatch is still on
  the GPU, and runs through L+1's attention. Lookahead moves it one
  step earlier, into L's attention, using the two existing prefetch
  sets; the cost is that L's demand misses can queue behind L+1's
  speculative reads on the shared I/O pool (watch `io_wait_ns`). It
  needs the previous-token guess: a cross-layer guess isn't known that
  early, so `CrossLayer` + lookahead is a config error.
- **Prefill:** hot-tier experts are still copied into staging (the
  gather kernel needs one base buffer); only non-hot experts are read,
  each once per layer per chunk. Across chunks an expert is re-read,
  since staging holds one layer at a time; chunk size is unchanged and
  the second (+453 MB) staging buffer for read/compute overlap was not
  added. The old path's 1.77 MB host blob per layer is gone.
- **Queue depth** is a lock-free gate on reads in flight; `io_threads`
  already bounds pool reads, so it bites only below `io_threads` (or
  when the main thread reads). A coalesced prefill run is one permit
  and is charged its full bytes by the rate limiter (20 ms burst).
- **Disk tier is per layer, not per expert:** a layer file in
  `fast_experts_dir/packed_experts/` overrides the primary. Per-expert
  placement would need a new file format (per-expert files are
  rejected in §8). Copy the most-used layers (from the routing profile)
  or, if it fits, all 18 GB.
- **GPU time is not measured directly:** metal-rs exposes no GPU
  timestamps (`CmdbufStat::gpu_ns` stays 0). The stats give decode
  wall time and its I/O-wait share; GPU waits per command buffer label
  are in `RsCtx::cmdbuf_stats`.
- **No new budget accounts.** The guesser's buffers (≈ 17 KB on a3b)
  and the per-layer counters (≈ 2 KB) are fixed at open inside
  `runtime_overhead`; the decode additions are atomics and fixed
  arrays (hot-tier lookups still take the tier's existing mutex).

### KV cache formats (`OpenConfig::kv_format`)

The GQA KV cache can be stored as `F32` (default, byte-identical to
before), `F16`, or TurboQuant MSE (`Turbo4` / `Turbo3`: a fixed random
rotation `H·D/√d`, then each coordinate rounded to the Lloyd–Max
levels of N(0,1), plus one f32 scale per head; `attn/kv_quant.rs`).
MLA (Cogito) stays F32; Turbo needs a power-of-two `head_dim`.

- **Decode** quantizes the new row on the CPU and reads stored rows
  directly (`sdpa_cpu_quant`; Turbo works in the rotated space, F16
  converts with `fcvtl`). The GPU SDPA fast path is F32-only.
- **Prefill** writes new f32 rows into one shared f32 scratch,
  quantizes them into the store and dequantizes the attended prefix
  back (`Op::KvConvertRows`, Metal `kv_quant_rows` /
  `kv_dequant_rows`), so the flash SDPA kernel is unchanged and prefill
  attends the same rounded K/V decode will.
- **Budget:** `Account::Kv` per format, plus `Account::KvScratch`
  (2 × n_ctx × kv_dim × 4, one shared), touched at open.
- **Snapshots:** non-F32 writes v4 (v3 header + `kv_format` word, raw
  stored rows); loading into a context of another format is refused
  (`ShapeMismatch { field: "kv_format" }`).

Measured on OLMoE-1B-7B, 3 GiB cap, auto hot tier, n_ctx 4096 (one
process per run; needle = 2,651-token record list, answer 7780;
4 prompts × 256 tokens):

| Format | KV + scratch | Hot tier | Needle | Greedy = F32 until token (story/code/explain/list) | Decode ms/tok (needle) |
| --- | --- | --- | --- | --- | --- |
| F32 | 1.00 GiB | 0.69 GiB | ✓ | — | 148.5 (309.5) |
| F16 | 0.50 + 0.06 GiB | 1.13 GiB | ✓ | 183 / all / all / all | 114.7 (215.9) |
| Turbo4 | 0.13 + 0.06 GiB | 1.49 GiB | ✓ | 19 / 19 / 9 / 3 | 102.9 (202.7) |
| Turbo3 | 0.10 + 0.06 GiB | 1.53 GiB | ✓ | 11 / 6 / 9 / 3 | 106.5 (223.9) |

Logits cosine vs F32 on the short smoke prompt: F16 0.999999, Turbo4
0.9928, Turbo3 0.9780; all three give the MLX reference answer. Turbo
outputs drift from F32's wording early but stay coherent (same
lists/stories, different phrasing). Relative MSE per head matches
Lloyd–Max (4-bit 0.0083 vs 0.0095 theory, 3-bit 0.030 vs 0.0345);
Metal matches the CPU reference to ~1e-13. Per-layer decode SDPA at
2,700 positions: F32 4.45 ms, F16 4.40, Turbo4 4.56, Turbo3 5.76 (CPU,
single-threaded at the time).

**Decode attention threads** (`OpenConfig::attn_threads`, default 4):
the CPU SDPA splits its heads across a small pool, same arithmetic per
head, so logits are bit-identical for any thread count
(`olmoe_attn_threads_are_bit_identical`). Per layer at 2,700
positions: 4.0 ms serial → 2.4 / 1.1 / 1.0 ms on 2 / 4 / 6 threads.
End to end (same matrix, one batch):

| Format | Threads | Needle ms/tok | 256-token prompts ms/tok |
| --- | --- | --- | --- |
| F32 | 1 | 303.6 | 108.1 |
| F32 | 4 | 192.6 (+58% tok/s) | 106.7 |
| Turbo4 | 1 | 205.8 | 77.7 |
| Turbo4 | 4 | 132.8 (+55%) | 74.2 |

Short prompts (~300 positions) gain little; long context gains most.
Thread stacks sit in `runtime_overhead` (both cap peak tests pass).

| # | Phase | Work | Exit criteria |
| --- | --- | --- | --- |
| 0 | Baseline + telemetry | Run a3b on M4 (mmap and pread, with/without `purge`) and one M1. Add streaming counters: bytes read per token/tier, hits per layer, I/O wait vs GPU time, guess accuracy; CSV output from `profile_smoke.sh`. Routing profile pass. Measure runtime overhead, SSD bandwidth (internal and external), 4 KB / 8 KB frame round-trip M4↔M1 | Baseline CSV; overhead number; expert popularity curve |
| 1 | Runtime config + ledger | `RsCtx::open_with(OpenConfig { mem_cap, n_ctx, expert_io, io_threads, layer_range, … })`; `open()` stays as a wrapper (drama_llama unchanged). KV sized from `n_ctx`. Ledger around `MetalBufferPool` and direct `new_buffer` sites. Reserve-and-touch at open. Watchdog | Oracles green; open fails cleanly over cap; footprint ≤ cap |
| 2 | `F_NOCACHE` + hot tier | `F_NOCACHE` fds for all model reads; weights read into owned buffers under a cap; fixed hot tier with `SlotSource::Hot`; profile seeding; frequency eviction | Identical logits at 0 / 32 / 100% hot tier; `mincore` ≈ 0; tok/s vs Phase 0 |
| 3 | Layer range | Load only the range's layers; `embed` / `forward_range` / `head`; break `chain_next` norm fusion at range end; per-range `state_save` / `state_load` | 40 vs 20+10+10 in one process match within oracle tolerance |
| 4 | Per-request state | Move `layer_states`, `LinearAttnBuffers.{conv_state, delta_state}`, deferred ring and prefetch guesses into `SeqState` keyed by `ReqId`; pre-reserved pool; admission; `drop_request` | Interleaved requests match sequential runs |
| 5 | Network + coordinator | `moeflux-net` frames (req id, position, f32 hidden / token id); loopback first; coordinator on M4; planner assigns ranges from budgets and measured speed | 3 processes over loopback match in-process split; latency measured |
| 6 | Streaming improvements | Cross-layer guesses, overlap, prefill non-hot reads, bounded I/O, optional internal-SSD tier | Read bytes/token and tok/s vs Phase 2 |
| 7 | Router policy | `RouterPolicy` hook after `moe_router_cpu`; Skip renormalises remaining weights (variable k already works through chunked matvec) | Task accuracy incl. JSON and tool calls |
| 8 | 2-bit (optional) | 2-bit repack, batched 2-bit experts kernel, mixed precision per layer | Oracle cosine + task accuracy |

## 6. Tests

- **Cap:** at caps of 6 / 8 / 10 GB, long generation at full `n_ctx` with the
  maximum request count: peak `phys_footprint` ≤ cap, `mincore` ≈ 0 on model
  files, identical logits across caps.
- **Over-cap open:** a configuration that cannot fit fails at `open` with an
  error naming the account.
- **Expert source equivalence:** mmap, pread, hot tier at 0 / 32 / 100% give
  identical logits on the same prompt (needs the per-context I/O mode from
  Phase 1; today only `tests/engine_op_diff.rs` exercises pread).
- **Split equivalence:** 40 layers vs 20+10+10, in-process and over loopback.
- **Request interleaving:** interleaved vs sequential requests match.
- **Streaming knobs** (`tests/streaming.rs`): cross-layer guesses,
  lookahead, bounded I/O and the fast experts dir each give
  bit-identical logits to the default path; prefill reads each routed
  expert once per layer; the stats test prints bytes/token, guess
  accuracy and I/O wait as CSV.
- **Policy quality:** skip / 2-bit judged on task accuracy, not perplexity
  alone (2-bit broke JSON and tool calling in flash-moe).
- **Speed regression gate:** per-machine tok/s baselines; a phase that drops
  streaming decode by more than ~5% is fixed or reverted.

The full oracle gate loads a3b and needs a high-RAM Mac; run it on the M4 or
a larger machine, not the 18 GB dev box.

## 7. Guardrails and upstream

- Keep mmap and pread paths; `ExpertSource` wraps them rather than replacing
  them. mmap mode remains available only when no cap is set.
- Budget, source and policy start as feature-gated modules inside `moeflux`
  (they need `MetalBufferPool`, `MoeBuffers`, `PrefetchState`); only
  `moeflux-net` and the coordinator binary are separate crates at first.
- Keep core diffs small so upstream (mdegans/moeflux) streaming work keeps
  merging. Phases 1, 2 and the telemetry are candidates for upstream PRs;
  coordinate before changing the public API.
- Keep `open()` working for drama_llama throughout.

## 8. Rejected or deferred

| Idea | Decision | Why |
| --- | --- | --- |
| Soft page-cache allowance (two fds + `mincore` governor) | Rejected | macOS cannot reliably evict a file range; overshoots the cap |
| Grow hot tier when the machine is idle | Rejected | Violates the fixed cap |
| mmap mode under a cap | Rejected | File-backed pages escape accounting |
| Lazily committed KV | Rejected | Grows after open |
| LZ4 / LZFSE / APFS compression | Rejected | Measured slower (`docs/optimization-experiments-q4.md`) |
| Per-expert files | Rejected | 15% slower (VFS overhead) |
| Recency-only LRU cache | Replaced by frequency-based | Lost to the OS page cache in flash-moe |
| All-2-bit experts | Deferred, opt-in only | Broke JSON and tool calling |

## 9. Measurements still needed

- [ ] Non-expert weight size (`model_weights.bin`) and Metal/runtime overhead
- [ ] Expert popularity per layer (share of picks covered by the top 32%)
- [ ] Internal vs external SSD random-read bandwidth with `F_NOCACHE`
- [ ] Frame round-trip M4↔M1 over Gigabit Ethernet and Thunderbolt bridge
- [ ] moeflux decode speed on an M1 GPU
- [ ] Confirm architecture figures against the model's `config.json`

## 10. Known cap gaps (review of phases 3–5)

- **Checkpoints outside the plan — fixed.** `Account::Checkpoints` =
  `OpenConfig::max_checkpoints` × `state_size_full` (every full-attn
  layer at `n_ctx`); under a cap the buffers are allocated and touched
  at open and reused by `checkpoint_pos` (overwrite → spare → LRU
  victim, pos 0 protected), `restore_to` loads in place (no clone),
  `forget_pos` / `memory_clear` return buffers to the spares.
  drama_llama's API is unchanged; `set_max_checkpoints` is clamped to
  the opened count under a cap, and a save with only protected
  positions left returns `CheckpointsFull`. Uncapped behavior is
  unchanged. `moeflux-node` opens with `max_checkpoints: 0`.
- **Whole-model weights rounding — fixed:** counted at whole 16 KiB
  pages, like the owned copy.
- **Parked slots' host `LinearAttnState` — fixed:** touched at a
  capped open.
- **Prefill scratch outside the plan — fixed.** Every batched-prefill
  buffer (`fags.*`, `mgs.*`, `lags.*`, `hdb.*`, `hts.token_ids`) was
  sized for `BATCHED_CHUNK_SIZE = 8192` tokens whatever `n_ctx` or the
  cap, and no account covered it: OLMoE-1B-7B pinned 2.6 GiB of it
  (a3b 3.6 GiB), so a 3 GiB cap failed at open (plan 3.00 GiB,
  footprint 3.89 GiB). Now `OpenConfig::prefill_chunk` (default 8192
  uncapped, 512 under a cap, clamped to `n_ctx`) sizes it,
  `step_internal` splits prompts by it, and `Account::PrefillScratch`
  plans it exactly (`budget::prefill_scratch_bytes`, checked against a
  real pool by `prefill_scratch_matches_pool`). Per variant:

  | Variant | chunk 512 | chunk 2048 | chunk 8192 |
  | ------- | --------- | ---------- | ---------- |
  | OLMoE-1B-7B | 172,549,120 B (164.6 MiB) | 689,591,296 B (657.6 MiB) | 2,757,760,000 B (2630.0 MiB) |
  | Qwen3.6-35B-A3B | 243,072,128 B (231.8 MiB) | 969,305,216 B (924.4 MiB) | 3,874,237,568 B (3694.8 MiB) |

  Two smaller gaps found while measuring, also fixed: the expert slot
  ring (`alloc_aligned`, heap-backed) was never touched at open, so
  the open-time check missed it (`MetalBufferPool::touch_aligned`);
  and `commit_plan` allocated the colored buffers while still holding
  the old transients, so the first prefill peaked at scratch + colors
  above the open footprint (it now releases them first, and under a
  cap it does not re-color at all — see the next item).
- **Allocations after open, invisible to the tripwire — fixed.** A
  capped run could open at 3.17 GiB and still peak at 4.06 GiB
  (`/usr/bin/time`, 3.4 GiB cap, chunk 8192) with no error: the
  footprint grew in the first eval, and `check_cap` only compared the
  *current* footprint after each eval. Staged sampling (after open /
  prefill / decode 1 / decode 32) plus `footprint -v` region diffs
  found, for OLMoE at cap 6 GiB, hot tier 0.35 GiB, chunk 512:
  - the pool itself does not grow (225 buffers, constant);
  - `commit_plan` lifetime coloring at the first prefill swapped the
    scratch transients for colored buffers, net **+48 MB** (5 × 16 MiB
    allocated, 2 × 16 + 3 × 4 MiB freed) — a capped open now freezes
    the pool layout (`MetalBufferPool::freeze_layout`): the planned,
    touched scratch is used uncolored;
  - the **first GPU submit** makes the driver allocate ≈ **80 MB** of
    IOAccelerator command/encoder state (bisected to the head graph's
    `execute`) — `reserve_at_open` now submits a trivial graph before
    the footprint check;
  - pipelines are compiled when the backend is built (an explicit
    `warm_all` at open changed nothing);
  - what is left grows < 5 MB over 32 decode steps (heap).

  `check_footprint` now also fails when the lifetime peak
  (`ri_lifetime_max_phys_footprint`) is over the cap *and* above the
  peak sampled at the start of `open_with` (`baseline_peak`), so a
  transient breach between checks is reported. Semantics: a peak no
  higher than the baseline predates the context and is ignored, so in
  a process that earlier held a larger context, a transient below that
  old peak is invisible.

  | OLMoE, n_ctx 2048 | open footprint | lifetime peak | open / peak |
  | --- | --- | --- | --- |
  | e4c2003 (chunk 8192), cap 3.4 GiB, n_ctx 512, hot 0, 8 decode | 3.17 GiB | 4,354,861,248 B (`time -l`) | ≈ 78 % |
  | chunk 512, before the `commit_plan` release fix, 8 decode | 1,764,299,904 B | 1,900,074,304 B | 92.9 % |
  | bfd4070, cap 6 GiB, hot 0.35 GiB | 1,764,332,672 B | 1,845,417,088 B | 95.6 % |
  | after, cap 3 GiB, hot 0.35 GiB | 1,844,941,888 B | 1,849,955,456 B | 99.7 % |
  | after, cap 6 GiB, auto hot tier (3.38 GiB, all experts) | 5,095,452,160 B | 5,100,973,568 B | 99.9 % |

  (prefill of the 25-token prompt + 32 decode steps unless noted;
  `olmoe_smoke.rs` `olmoe_cap_*` assert peak ≤ cap and open ≥ 95 % of
  peak, each in a child process.)
- **`runtime_overhead` measured.** A capped open logs `planned
  exactly` (plan minus `runtime_overhead`) and the signed `unplanned`
  remainder of the measured footprint; `MOEFLUX_POOL_DUMP=1` adds the
  GPU pool by label (`MetalBufferPool::bytes_by_label`). OLMoE-1B-7B on
  the M3 Pro (2026-10-25; cap 6 GiB, n_ctx 2048, hot tier 106 slots,
  chunk 512): planned exactly 1,780,708,352 B; with the fixes below
  the open footprint is 1,844,941,888 B and the peak through prefill +
  32 decode steps 1,849,955,456 B. The slot ring is planned
  2 MiB-rounded (~31 MB high), so the unitemized remainder is ≈ 90 MiB,
  nearly all of it the first-submit driver state, now at open.
  `DEFAULT_RUNTIME_OVERHEAD` drops from the
  1.5 GiB guess to 512 MiB (> 5× margin, for full 512-token chunks and
  larger variants — re-measure a3b/a17b the same way). With it, OLMoE
  at `mem_cap` 3 GiB, n_ctx 2048, auto hot tier plans 3.00 GiB (hot
  tier 1.19 GiB, 361 slots) and opens at 2,666,715,968 B (2.48 GiB).
