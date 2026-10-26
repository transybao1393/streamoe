# Changelog

## Unreleased

### Added

- `OpenConfig::max_checkpoints` (default 4): checkpoint buffers planned
  under a cap (`Account::Checkpoints`).
- `tools/chat`: a local chat page for Qwen3.6-35B-A3B under a cap.
- README guide for running Qwen3.6-35B-A3B on a small Mac, with measured
  memory and speed (≈ 5 tok/s at a 4 GiB cap on an M3 Pro).
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
- `STREAMMOE_IO_TRACE` + `tools/io_trace_report.py`: a timeline of
  expert reads and forward-pass phases, summarized as disk busy / idle
  time per phase (costs ~15% decode speed while on).
- Ignored benches: `matvec_decode_bench` (decode matvec kernels) and
  `ssd_read_bench` (expert reads at decode sizes, sync vs async).

### Performance

- Packed 3-bit experts: `tools/pack_experts.py` packs 3-bit codes 8 per
  3 bytes (1.77 → 1.38 MB per expert on a3b); the engine detects the
  layout from the expert files (`variants::ExpertLayout`,
  `ExpertFiles::layout`) and runs 3-bit decode and prefill kernels with
  output bit-identical to the same codes in 4-bit nibbles. a3b at a
  4 GiB cap: −20 to −25% per decode token, −29 to −34% time to first
  word vs 4-bit. `tools/chat --precision 3` uses it.
- Decode waits per prefetch slot instead of for the whole layer's
  prefetch (−1.5 to −5% per token on a3b at a 4 GiB cap).
- The prefill staging buffer is lent to the hot tier during decode
  (637 → 893 slots on a3b at 4 GiB, no new memory; −10 to −12% per
  token on two of three prompts). `STREAMMOE_HOT_LEND=0` turns it off.
- `qmv` kernel for single-token 4-bit matvecs (LM head 8.2 → 5.9 ms).
  `STREAMMOE_MATVEC_QMV=0` turns it off.

### Fixed

- Checkpoints allocated after open under a cap; now preallocated,
  touched at open and reused, with `restore_to` loading in place.
- Whole-model weights counted at raw length instead of whole pages.
- Parked request slots' host linear-attn state not touched at open.
- Intermittent `footprint_is_readable_and_nonzero` failure.

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
