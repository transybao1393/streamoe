# Known issues

What has been fixed, what is still open, and what to watch out for in
the fork revamp (`docs/fork-upgrade-plan.md`). Add an entry when a bug
is fixed or a trap is found; move open items to *Fixed* when they land.
Older engine history (graph mode, prefill, kernels) is in
`.claude/memory/`.

## Open

| Issue | Impact | Plan |
| --- | --- | --- |
| `resuming_prefill_after_seq_rm_matches_full_prefill` fails. | Known: linear-attn recurrence state is not position-truncatable (a partial truncate resets the layer). | `--skip` it; see the fn-level comment and `snapshot/state.rs`. |
| The uncapped a3b oracle suites and all a17b model tests have not run (this host has 5–7 GiB reclaimable; they need ≥ 8 GiB). | a3b runs under a 3–5 GiB cap: answers correct, argmax matches the MLX golden, top-20 overlap 20/20 (F32 KV) / 19/20 (Turbo4), peak within the cap (README, *Running Qwen3.6-35B-A3B on a small Mac*). Phases 1–6, `kv_format`, `attn_threads` and capped checkpoints are otherwise verified on OLMoE-1B-7B, plus unit / kernel tests on all four variants. Checkpoint buffers now come out of the cap, so a3b capped tests have less hot tier than before (≈ 0.9 GiB at n_ctx 4096, default 4); `checkpoint_restore.rs` (uncapped) exercises the rewritten store but has not run. | Run the `--include-ignored` gate on a host with the a3b model. |
| Vision reads small or short printed text unreliably ("STOP" → "SOM" / "Form", "HELLO 2026" → "HILLCO" at 128–160 image tokens); scenes, colors, layout and counting are right. | Not an engine bug: on the same image embeddings, MLX (mlx_vlm, 4-bit) has the same top tokens ("F", "S", "T" for "STOP"); the engine follows MLX layer by layer (cosine ≥ 0.99 at layer 32, 0.96 on the last token at layer 40, bf16 vs f32); positions equal mlx_vlm's `get_rope_index`; the encoder matches transformers exactly and mlx_vlm to 0.997; all 92 160 packed expert tensors are byte-identical to the safetensors. | Use larger images. Re-check against a bf16 / 8-bit checkpoint if text reading matters. |
| GPU time is not measured directly. | metal-rs exposes no GPU timestamps (`CmdbufStat::gpu_ns` stays 0); only wall time and I/O wait. | — |

## Fixed

### Memory cap

- **Prefill scratch outside the plan** (`bfd4070`). Every batched-prefill
  buffer was sized for 8192 tokens whatever `n_ctx` or the cap
  (OLMoE 2.6 GiB, a3b 3.6 GiB), with no account: a 3 GiB cap failed at
  open. Now `OpenConfig::prefill_chunk` (512 under a cap) sizes it and
  `Account::PrefillScratch` plans it exactly.
- **Expert slot ring never touched at open** (`bfd4070`) —
  heap-backed, so the open-time check missed it.
  `MetalBufferPool::touch_aligned`.
- **`commit_plan` re-colored the pool on the first graph** (+48 MB after
  open; `bfd4070`, `6f8cc46`). Under a cap the pool layout is frozen at
  open.
- **First GPU submit allocated ~80 MB of driver state after open**
  (`6f8cc46`). A warm submit now runs at open.
- **Tripwire only compared the current footprint** (`6f8cc46`), so a
  transient peak over the cap went unnoticed. `check_cap` also checks
  the lifetime-max footprint.
- **`DEFAULT_RUNTIME_OVERHEAD` 1.5 GiB → 512 MiB** (`bfd4070`): ~90 MiB
  is actually unitemized at peak on OLMoE.
- **`kv_format` scratch** (`ff6619d`): a quantized KV cache needs an f32
  prefill scratch; it has its own account (`kv_scratch`) and is touched
  at open.
- **Checkpoints outside the plan** (uncommitted). `checkpoint_pos`
  allocated a `state_size()` buffer per checkpoint after open (~1 GiB
  each on OLMoE at n_ctx 4096 F32) and `restore_to` cloned one more.
  Now `Account::Checkpoints` = `max_checkpoints` × the largest state at
  `n_ctx`; under a cap the buffers are allocated and touched at open
  and reused (overwrite, LRU victim, spare); `restore_to` loads in
  place without a copy. Open footprint 99.8% of the lifetime peak
  through save / evict / restore / clear
  (`olmoe_checkpoints_stay_inside_the_cap`).
- **Whole-model weights counted at raw length** (uncommitted): now
  rounded to whole 16 KiB pages like the allocation.
- **Parked request slots' host `LinearAttnState` untouched at open**
  (uncommitted): every slot's linear state is committed at a capped
  open.

### Correctness

- **`extract_weights.py` broke on Qwen3-MoE configs** (`7466141`): no
  linear-attn / shared-expert keys and no `rope_parameters`; now
  defaulted.
- **Metal kernel signature** (`ff6619d`, caught by the CPU-vs-Metal
  test): mixing `uint2` and `uint` thread-position inputs fails to
  compile at runtime. Use the same vector type for both.
- **Flaky `footprint_is_readable_and_nonzero`** (uncommitted): asserted
  `peak >= current`, but the kernel updates the lifetime max lazily,
  so with other test threads allocating it failed intermittently.

### Speed

- **Prefetch read all K guesses** (`b9587b8`): 42% of decode reads were
  wasted on OLMoE at a 6 GiB cap. `prefetch_experts` = 4 under a cap.
- **F16 KV decode was slower than F32** (`ff6619d`): the f16 → f32
  conversion did not vectorize (19 ms vs 4.5 ms per layer). It now uses
  the hardware `fcvtl`; bit-exact over all 65,536 halves.
- **Decode attention ran on one core** (`3eb7f13`): at 2.7k context it
  was ~half of decode time. `OpenConfig::attn_threads` (default 4)
  splits heads across a pool, bit-identical: needle decode
  303.6 → 192.6 ms/tok (F32), 205.8 → 132.8 (Turbo4).

## Precautions

### The shared server

- Other services run on this machine and must not be affected: run
  `scripts/check_test_host.sh` before every test phase, run model tests
  one process at a time (`--test-threads=1`), and keep caps within the
  reclaimable memory it reports. `scripts/watch_run.sh` samples memory,
  swap and pressure during a run.
- Host load changes timings a lot (load 12 vs 2 changed decode by
  > 30%). Compare speeds only within one batch of runs; rerun the
  baseline with every comparison.
- `attn_threads` and `io_threads` use CPU cores other services share;
  lower them if that matters more than speed.

### Configuration rules

- `expert_io = Mmap` is rejected with `mem_cap` (file-backed pages
  escape the cap); a hot tier needs `Pread`.
- `prefetch_lookahead` needs `prefetch_guess = PrevToken`.
- **Checkpoints cost cap memory up front.** Under `mem_cap`,
  `max_checkpoints` (default 4) full-context snapshots are planned
  whether or not they are used: on OLMoE at n_ctx 2048 F32 that is
  4 × 0.5 GiB, enough to fail a 3 GiB cap. Set `max_checkpoints: 0`
  when nothing calls `checkpoint_pos` (pipeline stages: `moeflux-node`
  does), lower it, or use a quantized `kv_format` (Turbo4 snapshots are
  ~5× smaller). Under a cap `set_max_checkpoints` cannot raise it past
  the opened value, and `checkpoint_pos` returns
  `CheckpointsFull` instead of allocating when only protected
  positions (0 and the one being saved) remain.
- `kv_format`: MLA (Cogito) accepts only F32; Turbo needs a
  power-of-two `head_dim`. Non-F32 formats always use the CPU decode
  attention (the GPU SDPA fast path is F32-only).
- Snapshots are not converted between KV formats: loading a snapshot
  into a context with another `kv_format` fails with
  `ShapeMismatch { field: "kv_format" }`. F32 snapshots stay v2 / v3.

### Expected behavior, not bugs

- **Turbo KV changes greedy output wording early** (3–19 tokens on
  OLMoE) while answers stay correct and coherent; F16 matches F32 for
  hundreds of tokens. Logits cosine vs F32: F16 0.999999, Turbo4
  0.9928, Turbo3 0.9780.
- **Turbo CPU and Metal quantizers are not bit-identical**: their norm
  reductions sum in a different order, so a code at a level boundary
  can flip (reconstructions agree to ~1e-13 relative). Batched prefill
  vs per-token decode agree at cosine 0.9975 (Turbo4), not 1.0.
- **Turbo3 is slower than Turbo4**: its 3-bit unpack costs more than
  the 0.04 GiB of extra hot tier saves. Prefer Turbo4.

### Measuring

- **macOS `F_NOCACHE` only bypasses the page cache for page-aligned
  buffers with `F_RDAHEAD` off.** Python reads into unaligned buffers
  still fill the cache. Benchmark SSD reads into an `mmap` buffer with
  readahead off.
- **Files read recently are still in the page cache**: a "disk"
  benchmark on them measures RAM. Use a fresh cold copy, check
  residency, delete it afterwards.
- The SSD tops out at ~6.7 GB/s once ≥ 4 reads are in flight; decode is
  bandwidth-bound, so fewer bytes per token is what speeds it up.

### Tests and tooling

- `graph_diff_oracle.rs` and `checkpoint_restore.rs` compile only with
  `model-qwen3-6-35b-a3b`; the graph-diff kernel tests need no model
  files, so run them under a3b even on a host without a3b weights.
- `checkpoint_restore.rs` fails under a17b — expected.
- `cargo fmt` is not a gate (the crate is ~600 hunks off rustfmt); match
  the surrounding style, don't bulk-reformat.
- zsh does not word-split variables (`$G commit` fails): export
  `GIT_AUTHOR_*` / `GIT_COMMITTER_*` instead. `grep --include=*.rs`
  errors as an unmatched glob; quote it or drop it.

### Measured dead ends — don't retry without new evidence

- Predicting the next experts from a moving router score or recency:
  ties the previous-token guess (±1 pt) or loses.
- LRU or short-retention hot tiers: LFU wins.
- A 64-slot recent-read window: 0 hits in the engine (~145 slot fills
  per token).
- Re-laying-out experts on disk for coalesced reads: no gain (the SSD
  is flat once ≥ 4 reads are in flight).
- More from §8 of the plan: soft page-cache allowance, growing the hot
  tier when idle, lazily committed KV, LZ4 / LZFSE compression,
  per-expert files, all-2-bit experts.
