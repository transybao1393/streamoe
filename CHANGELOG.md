# Changelog

## Unreleased

### Changed

- Decode lends the batched-prefill scratch to the hot tier as well as
  the staging buffer, and the next batched prefill takes it back. a3b
  chat at a 4 GiB cap, greedy output identical, peak unchanged:
  packed 3-bit 8.47 / 7.47 / 8.27 → 8.98 / 7.70 / 8.49 tok/s on the
  three standard prompts (hot-tier hits 64 / 38 / 54% → 67 / 44 /
  59%); 4-bit flat within noise (hits 59 / 34 / 54% → 61 / 38 / 58%).
  Hot tier while lent: 637 → 1012 slots at 4-bit, 893 → 1317 at packed
  3-bit (staging + scratch). `STREAMMOE_HOT_LEND_SCRATCH=0` lends only
  the staging buffer; `STREAMMOE_HOT_LEND=0` still turns all lending
  off.

- The hot tier's popularity counters halve every 4096 recorded picks
  (~13 tokens on a3b) instead of every 65536 (~205 tokens), so the
  previous topic's experts stop holding slots within a sentence. a3b
  chat at a 4 GiB cap, mean of 2 alternating rounds, greedy output
  identical, peak unchanged (3.61–3.65 GiB): packed 3-bit 11.44 /
  8.20 / 9.15 → 12.01 / 10.22 / 11.65 tok/s on the three standard
  prompts (hot-tier hits 67 / 43 / 59% → 73 / 61 / 72%), 4-bit
  9.55 / 7.17 / 8.80 → 10.48 / 8.59 / 10.61 (61 / 38 / 57% → 67 /
  56 / 70%). A prose → code → prose chat in one session, per turn:
  packed 3-bit 11.45 / 8.25 / 9.83 → 11.73 / 9.30 / 11.28 tok/s,
  4-bit 9.50 / 6.68 / 8.35 → 9.93 / 8.08 / 10.01. 16384 and 1024
  gain less. `STREAMMOE_HOT_DECAY=<picks>` sets the period (65536
  restores the old one).

### Added

- `budget_cap::a3b_prefill_after_lent_decode_matches_first_prefill`:
  prefill and decode after the lent slots filled match the first ones
  bit for bit.
- `budget_cap::a3b_mixed_workload_unplanned_footprint`: logs what
  `runtime_overhead` covers over a mixed workload at a 4 GiB cap
  (+104 MiB 4-bit, +76 MiB packed 3-bit at peak, of 512 MiB).

## 0.2.0-beta.2 — 2026-10-28

Second beta: long runs now stay inside the memory cap (a leak of
autoreleased Metal objects is fixed), the mmap-vs-pread test is exact
again, and more of the a3b model suites ran.

### Fixed

- Long decodes went over the memory cap: Metal command buffers and
  encoders are autoreleased and the engine never drained a pool, so
  the footprint grew ~0.33 MiB per token until `OverBudget`. Every
  public `eval_*` and `rewind_tokens` now runs inside an autorelease
  pool. A 130-token prompt + 128-token decode stays flat at a 4 GiB cap
  with the default 512 MiB `runtime_overhead`; `spec_bench` and
  `batch_bench` drop their 768 MiB default for the library's.
- `mmap_matches_pread` failed: mmap mode decodes full attention with
  the GPU SDPA and pread mode with the CPU one (≥ 32 KV rows), which
  round differently (~1e-5). The test now forces the CPU path in both
  modes and is bit-identical.

### Added

- `STREAMMOE_DECODE_SDPA=cpu|gpu`: overrides the expert-I/O-mode gate
  for decode attention (default unchanged).
- `budget_cap::a3b_long_decode_stays_under_4gib_cap`: long decode under
  a 4 GiB cap at the default overhead; fails if warm decode grows.

### Tests

- Run on 2026-10-28 (a3b, M3 Pro, ~7 GiB free): `vision` passes
  (cosine 1.000000 on 3 images), `engine_op_diff` passes with
  `STREAMMOE_EXPERT_IO=mmap`, `mmap_matches_pread` passes after the fix.
  `diff_oracle` cannot run under a cap (two contexts in one process;
  the cap counts the whole process) and needs ~8 GiB free uncapped.

## 0.2.0-beta.1 — 2026-10-27

First beta. Highlights: vision input, packed 3-bit experts, several
tokens per pass (parallel decoding), early expert dispatch, the a3b
model suites run under a cap, and the moeflux → streammoe rename.
Numbers below use the standard setup in `docs/benchmarks.md`.

### Added

- `OpenConfig::max_checkpoints` (default 4): checkpoint buffers planned
  under a cap (`Account::Checkpoints`).
- `tools/chat`: a local chat page for Qwen3.6-35B-A3B under a cap.
- README guide for running Qwen3.6-35B-A3B on a small Mac, with measured
  memory and speed (6.5–9.2 tok/s at 3–6 GiB caps on an M3 Pro), and
  `docs/benchmarks.md` with the full tables and the test setup.
- Multimodal RoPE for Qwen3.5 / 3.6 (`MROPE_SECTION`, interleaved
  `[11, 11, 10]`): `Ctx::eval_stage_mrope` takes per-token `[t, h, w]`
  positions; CPU `apply_mrope` and the Metal `rope_n_tokens_mrope`
  kernel. `[p, p, p]` rows are bit-identical to the plain path.
- `riir::vision`: a CPU vision encoder for Qwen3.5 / 3.6 (Accelerate
  GEMMs, weights streamed per matrix with `F_NOCACHE`, all buffers
  preallocated and reported by `bytes()` for `host_buffers`), plus
  `smart_resize` and `image_rope_positions`. Matches transformers'
  `Qwen3_5MoeVisionModel` (cosine 1.000000 per row) and mlx_vlm (0.997,
  bf16). `tools/export_vision.py` writes `vision_weights.{bin,json}`.
- `tools/chat`: images (upload or paste) and a live CPU / RAM bar
  (`/stats`, served while a turn runs).
- `tools/chat`: `--precision 4|3` and an Experts selector in the header
  (the server re-executes itself to switch); per-turn I/O and CPU stats.
- `tools/requant_experts.py` (3-/2-bit rounding kept in the 4-bit
  layout) and the chat crate's `quality` binary (KL, top-1 agreement,
  perplexity, scored tasks against a reference run).
- `STREAMMOE_GPU_TIMING`: GPU timestamps per command buffer;
  `CmdbufStat::gpu_ns` is now real GPU time.
- `OpenConfig::max_decode_tokens` (default 1): the longest
  `eval_tokens` / `eval_requests` pass. Its buffers are planned under
  a cap (`multi_decode` account, 0.22 GiB at 4 on a3b, exact — a pool
  test keeps the formula in step); a longer pass is refused.
- Early expert dispatch: when some of a layer's experts must come from
  disk, the RAM-resident ones' FFN goes to the GPU before the reads,
  and the main dispatch adds the rest and the unchanged combine.
  Bit-identical; ~5% faster warm decode on a3b at a 4 GiB cap.
  `STREAMMOE_EARLY_EXPERTS=0` turns it off.
- `Ctx::eval_tokens`: decode with T ≤ 4 tokens in one forward pass —
  decode's kernels and expert streaming, each layer's attention for all
  T tokens in one command buffer, the experts of all T read once —
  with logits at every position bit-identical to `eval_token`. The
  verification half of speculative decoding. On a3b, packed 3-bit at a
  4 GiB cap: 176 / 238 / 310 ms per pass for T = 2 / 3 / 4 vs 103–118
  ms for one token.
- `Ctx::rewind_tokens`: keep the first `keep` tokens of the last
  `eval_tokens` pass (the KV drops the rest, the linear-attention
  recurrence goes back to its copy after token `keep - 1`), so
  rejected speculative drafts can be undone.
- `Ctx::eval_requests`: parallel decoding — one token from each of up
  to 4 admitted requests in one pass, each on its own KV and
  recurrence, logits identical to `eval_request` per request. On a3b,
  packed 3-bit, 4 requests × 64 tokens: 1.18–1.25× total tok/s at a
  5 GiB cap, 1.22–1.47× at 7 GiB.
- `tools/chat` benches: `verify_bench` (cost of checking T tokens in
  one pass, bit-identity of `eval_tokens`, `--profile`), `spec_bench`
  (prompt-lookup speculative decoding vs plain decode, end to end) and
  `batch_bench` (parallel decoding vs one request at a time).
- `MetalContext::record_wait`; `Ctx::cmdbuf_stats` now includes the
  deferred expert dispatches and the LM head. An ignored expert-kernel
  GPU-time bench in `tests/packed_3bit.rs`.
- `STREAMMOE_IO_TRACE` + `tools/io_trace_report.py`: a timeline of
  expert reads and forward-pass phases, summarized as disk busy / idle
  time per phase (costs ~15% decode speed while on).
- Ignored benches: `matvec_decode_bench` (decode matvec kernels) and
  `ssd_read_bench` (expert reads at decode sizes, sync vs async).
- `STREAMMOE_TEST_CAP_GIB=<n>`: the model suites that open uncapped
  (`smoke`, `mlx_regression`, `consecutive_eval_prompt`,
  `checkpoint_restore`, `diff_oracle`) open under an n-GiB cap instead,
  so they run on a small host.

### Performance

- Packed 3-bit experts: `tools/pack_experts.py` packs 3-bit codes 8 per
  3 bytes (1.77 → 1.38 MB per expert on a3b); the engine detects the
  layout from the expert files (`variants::ExpertLayout`,
  `ExpertFiles::layout`) and runs 3-bit decode and prefill kernels with
  output bit-identical to the same codes in 4-bit nibbles. a3b at a
  4 GiB cap: 8.1 vs 7.4 tok/s (9–15% faster at every cap from 3 to
  6 GiB). `tools/chat --precision 3` uses it.
- Decode waits per prefetch slot instead of for the whole layer's
  prefetch (−1.5 to −5% per token on a3b at a 4 GiB cap).
- The prefill staging buffer is lent to the hot tier during decode
  (256 more slots on a3b, no new memory; 6.9 → 7.4 tok/s at a 4 GiB
  cap). `STREAMMOE_HOT_LEND=0` turns it off.
- `qmv` kernel for single-token 4-bit matvecs (LM head 8.2 → 5.9 ms).
  `STREAMMOE_MATVEC_QMV=0` turns it off.

### Fixed

- Checkpoints allocated after open under a cap; now preallocated,
  touched at open and reused, with `restore_to` loading in place.
- Whole-model weights counted at raw length instead of whole pages.
- Parked request slots' host linear-attn state not touched at open.
- Intermittent `footprint_is_readable_and_nonzero` failure.
- `dequant_matvec_4bit_n_tokens_v3_n1_matches_single` compared the qmv
  path against the v3 kernel; its reference now follows
  `STREAMMOE_MATVEC_QMV` like the engine.

### Changed

- `MoeGraphScratch::new` takes the expert size; `MoeBuffers::new_for`
  sizes the decode slots for an `ExpertLayout` (`new` stays 4-bit);
  `ExpertIoError` gains `MixedLayouts`; `streammoe-metal`'s
  `MoeGatherIdCall` gains `bits`.
- **Renamed moeflux → streammoe.** Crates `streammoe`, `streammoe-metal`,
  `streammoe-net` (directories `crates/streammoe*`), binaries
  `streammoe-node`, `streammoe-coord`, `streammoe-chat`, and env vars
  `MOEFLUX_*` → `STREAMMOE_*`. Downstream code changes
  `use moeflux::…` to `use streammoe::…`. Snapshot, routing-profile and
  network-frame formats are unchanged (same magics), so existing files
  still load.
- Under a cap, checkpoint buffers take cap memory at open (set
  `max_checkpoints: 0` if unused); `set_max_checkpoints` cannot raise
  the count past the opened value; `checkpoint_pos` can return
  `StateSnapshotError::CheckpointsFull`. `moeflux-node` plans none.
- License: work after commit `7d73350` is under PolyForm Noncommercial
  1.0.0 (`LICENSE.md`); earlier code stays MIT (`LICENSE-MIT`); `NOTICE`
  draws the boundary.
- Removed flash-moe leftovers (C engine results, the paper, a
  model-specific `expert_index.json`); `repack_experts.py` moved to
  `scripts/`.

## 0.2.0-alpha.1 — 2026-10-26

First alpha of the fork revamp (`docs/fork-upgrade-plan.md`), on top
of moeflux `0.1.0-pre.4`. The version is `0.2.0` because, in semver,
`0.1.0-alpha.*` would sort before `0.1.0-pre.4`. APIs may still
change. Known gaps and precautions: [`docs/known-issues.md`](docs/known-issues.md).

### Added

- **Hard memory cap** (phase 1): `RsCtx::open_with(OpenConfig { mem_cap,
  n_ctx, … })`. A budget plan per account, reserve-and-touch at open,
  and footprint checks (current and lifetime peak); `open` fails with
  `Error::OverBudget` before anything else is affected. `open()` is
  unchanged for drama_llama.
- **`F_NOCACHE` streaming and a fixed hot tier** (phase 2):
  page-cache-free reads into aligned buffers, a frequency-based expert
  hot tier sized from what the cap leaves, routing-profile seeding,
  expert-I/O counters.
- **Layer-range execution** (phase 3): `layer_range`, `eval_stage`,
  f32 hidden in/out, per-range snapshots (v3).
- **Per-request state** (phase 4): `max_requests` pool,
  `admit_request` / `drop_request` / `eval_request`.
- **`moeflux-net`** (phase 5): frames, `moeflux-node`,
  `moeflux-coord` and a layer-range planner (TCP).
- **Streaming improvements** (phase 6): `ExpertIoStats` telemetry and
  CSV, bounded I/O (`io_queue_depth`, `io_bytes_per_sec`), coalesced
  prefill reads, cross-layer and lookahead prefetch,
  `fast_experts_dir` disk tier, `prefetch_experts` (default 4 under a
  cap, +4–15% tok/s on OLMoE).
- **OLMoE-1B-7B variant** (`model-olmoe-1b-7b`): converter, NumPy
  reference forward, and model tests under a cap — the test bed on
  hosts without a3b weights.
- **KV cache formats** (`kv_format`): F32 (default), F16, TurboQuant
  4/3-bit; snapshot v4. OLMoE, 3 GiB cap, n_ctx 4096: KV 1.00 → 0.56 /
  0.19 GiB (F16 / Turbo4), decode +29% / +44%.
- **Decode attention threads** (`attn_threads`, default 4):
  bit-identical; +55–58% tok/s at 2.7k context.
- **Tooling**: `scripts/check_test_host.sh`, `scripts/watch_run.sh`,
  `MOEFLUX_ROUTING_TRACE`, `tools/sim_expert_cache.py`.

### Fixed

- Prefill scratch sized for 8192 tokens outside the budget
  (`prefill_chunk`, `Account::PrefillScratch`).
- Allocations after open under a cap: pool re-coloring, first-submit
  driver state, untouched slot ring.
- `moeflux-net` session reset, rollback and input checks.
- `extract_weights.py` on Qwen3-MoE configs.

### Known gaps

- Checkpoints, whole-model weight rounding and parked request slots
  are not fully inside the cap plan (see `docs/known-issues.md`).
- a3b / a17b model-level tests have not run yet; verified on
  OLMoE-1B-7B plus unit / kernel tests on all four variants.
