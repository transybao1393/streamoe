# Known issues

What has been fixed, what is still open, and what to watch out for in
the fork revamp (`docs/fork-upgrade-plan.md`). Add an entry when a bug
is fixed or a trap is found; move open items to *Fixed* when they land.
Older engine history (graph mode, prefill, kernels) is in the git
history of `.claude/memory/`, which is no longer tracked.

## Open

| Issue | Impact | Plan |
| --- | --- | --- |
| Speculative decoding (prompt lookup + `eval_tokens` + `rewind_tokens`) is slower than plain decode: 0.79× / 0.92× on rewrite / code prompts, 1.02× on free text (packed 3-bit, 4 GiB cap, `spec_bench`). Output is token-identical. | A verify pass of ~4 tokens costs ~425 ms in real use (310 ms in `verify_bench`, where every draft is right) and yields 2.4–2.9 tokens; break-even is ~310–370 ms. | Cut the pass cost (it grows with T mostly in disk wait and per-token GPU work — see the multi-column matvec dead end); fewer wrong-draft expert reads; draft only on long matches. |
| `resuming_prefill_after_seq_rm_matches_full_prefill` fails. | Known: linear-attn recurrence state is not position-truncatable (a partial truncate resets the layer). | `--skip` it; see the fn-level comment and `snapshot/state.rs`. |
| `diff_oracle`, the 10 GiB-cap tests and all a17b / Cogito model tests have not run on this host (~7 GiB reclaimable, no a17b / Cogito weights). `diff_oracle` cannot run under a cap at all: the cap check measures the whole process, and the tests hold two contexts in one process, so the second open sees the first's ~2.5 GiB (`OverBudget: footprint 5.09 GiB exceeds cap 3.00 GiB` at a 3 GiB cap with no checkpoints). | Every other a3b model suite passes: on 2026-10-26 `mlx_regression`, `smoke`, `checkpoint_restore`, `consecutive_eval_prompt`, `mrope`, `layer_range`, `requests`, `streaming`, `expert_source_equivalence`, `budget_cap`, `graph_diff_oracle`, `batched_diff_oracle`; on 2026-10-28 `vision` (cosine 1.000000 on 3 images), `engine_op_diff` (with `STREAMMOE_EXPERT_IO=mmap`) and `mmap_matches_pread` (after the fix below). Suites that open uncapped ran under `STREAMMOE_TEST_CAP_GIB=4`. | `diff_oracle` uncapped with ~8 GiB free; the 10 GiB-cap tests with 10 GiB free; a17b / Cogito where their weights are. |
| Vision reads small or short printed text unreliably ("STOP" → "SOM" / "Form", "HELLO 2026" → "HILLCO" at 128–160 image tokens); scenes, colors, layout and counting are right. | Not an engine bug: on the same image embeddings, MLX (mlx_vlm, 4-bit) has the same top tokens ("F", "S", "T" for "STOP"); the engine follows MLX layer by layer (cosine ≥ 0.99 at layer 32, 0.96 on the last token at layer 40, bf16 vs f32); positions equal mlx_vlm's `get_rope_index`; the encoder matches transformers exactly and mlx_vlm to 0.997; all 92 160 packed expert tensors are byte-identical to the safetensors. | Use larger images. Re-check against a bf16 / 8-bit checkpoint if text reading matters. |
| Decode kernels run 3–4× slower than in isolation (a3b, 4 GiB cap: GPU busy ~87 of ~143 ms/token; LM head 9–11 ms vs 2.8 ms). | Reproduced without the model (`tests/matvec_decode_bench.rs`, `matvec_decode_load_pattern`): ~1 ms GPU bursts separated by ~1 ms CPU gaps leave the GPU in a slow state. A spin kernel on a second queue during the gaps restores speed when the GPU is slow but costs 8–15 ms/token when it is not. | Fewer, longer GPU bursts per token; non-blocking keep-alive in the engine; wire `dequant_matvec_4bit_qmv` (+37–55% bandwidth in isolation) into `encode_matvec`. |

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
- **Checkpoints outside the plan** (`cb717b5`). `checkpoint_pos`
  allocated a `state_size()` buffer per checkpoint after open (~1 GiB
  each on OLMoE at n_ctx 4096 F32) and `restore_to` cloned one more.
  Now `Account::Checkpoints` = `max_checkpoints` × the largest state at
  `n_ctx`; under a cap the buffers are allocated and touched at open
  and reused (overwrite, LRU victim, spare); `restore_to` loads in
  place without a copy. Open footprint 99.8% of the lifetime peak
  through save / evict / restore / clear
  (`olmoe_checkpoints_stay_inside_the_cap`).
- **Whole-model weights counted at raw length** (`cb717b5`): now
  rounded to whole 16 KiB pages like the allocation.
- **Parked request slots' host `LinearAttnState` untouched at open**
  (`cb717b5`): every slot's linear state is committed at a capped
  open.
- **Long decodes went over the cap** (a 128-token decode after a
  ~130-token prompt hit `OverBudget` in `eval_token` at 4 and 5 GiB;
  `spec_bench` needed `runtime_overhead` 768 MiB). metal-rs hands out
  command buffers, encoders and labels autoreleased, and the engine
  never drained an autorelease pool, so every one stayed alive until
  the calling thread exited: the footprint grew ~0.33 MiB per decode
  token, without bound (a Metal-only probe: 10,000 command buffers of
  8 encoders kept 16.2 MiB; 0.4 MiB inside a pool). Every public
  `eval_*` (and `rewind_tokens`) now runs its GPU work inside
  `objc::rc::autoreleasepool`, drained before `check_cap`. a3b packed
  3-bit, 4 GiB cap, default 512 MiB overhead
  (`budget_cap::a3b_long_decode_stays_under_4gib_cap`, 130-token
  prompt + 128 tokens): +25 → +66 MiB from token 16 to 128 before,
  +25 → +26 MiB after (the test now fails if warm decode grows 8 MiB).
  `spec_bench` (three prompts, ~1,150 tokens): `OverBudget` at
  4.00 GiB on the third prompt before; peak 3.78 GiB after, with the
  benches back on the default overhead. `verify_bench` peak 3.85 →
  3.77 GiB, decode speed unchanged (eval_token 115.5 / 114.5 ms/tok
  before, 114.4 / 117.6 after, alternating runs); outputs unchanged
  (`spec_bench` / `verify_bench` match plain decode, `mlx_regression`
  passes).

### Correctness

- **`extract_weights.py` broke on Qwen3-MoE configs** (`7466141`): no
  linear-attn / shared-expert keys and no `rope_parameters`; now
  defaulted.
- **Metal kernel signature** (`ff6619d`, caught by the CPU-vs-Metal
  test): mixing `uint2` and `uint` thread-position inputs fails to
  compile at runtime. Use the same vector type for both.
- **`mmap_matches_pread` failed** (step 0 bit-identical; from the first
  `eval_token`, 243,030 logits differ by up to 1.07e-5; greedy tokens
  identical). Not the expert bytes: the decode attention gate in
  `full_attn_forward.rs` sends full-attention layers to the GPU SDPA in
  mmap mode and to the CPU SDPA in pread mode (kept off the GPU for
  a17b's streaming) once the KV cache holds 32 rows, and the two sum
  in a different order. The test's 48-token prompt crosses 32 at its
  first decode step. `STREAMMOE_DECODE_SDPA=cpu|gpu` now overrides the
  gate; `expert_source_equivalence.rs` sets `cpu` for every run, and
  mmap vs pread is bit-identical again (all 17 steps; ~6 min, peak RSS
  1.5 GiB). Default behavior is unchanged in both modes.
- **Flaky `footprint_is_readable_and_nonzero`** (`cb717b5`): asserted
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
- **A routing profile froze the hot tier on its own topics**:
  `seed_from_counts` installed the profile's counts (hundreds) as the
  tier's popularity, so a new conversation's experts (count 1–2) could
  not displace unused seeded ones until decay, ~200 tokens later.
  a3b chat at a 4 GiB cap: first-answer hot-tier hits 51% cold vs 28%
  seeded. Seeding now loads the profile's experts with zero popularity;
  seeded and cold runs match (51 / 22 / 42% on three prompts).
- **Each layer waited for every prefetch read before reading its
  misses**, wrong guesses included. `PrefetchState::begin_layer` /
  `wait_slot` now start the misses right after routing and wait only
  for the prefetch slots the layer binds. a3b, 4 GiB cap, greedy
  output identical: forward 124.0 → 122.2, 157.1 → 149.2 and
  135.6 → 132.6 ms/token on three prompts (most where misses are many).
- **The prefill staging buffer sat idle during decode** (0.42 GiB on
  a3b, one block per expert, already in the budget). It is now lent to
  the hot tier as extra slots for decode (`HotTier::lend`, 637 → 893
  slots at a 4 GiB cap) and reclaimed at the start of every batched
  prefill (`HotTier::reclaim`); `STREAMMOE_HOT_LEND=0` turns it off.
  a3b chat at a 4 GiB cap, greedy output identical, peak unchanged
  (3.67–3.72 GiB): hot-tier hits 51 / 22 / 42% → 58 / 34 / 54%,
  forward 124.3 → 111.8, 150.2 → 149.5, 135.2 → 119.1 ms/token.
- **Single-token 4-bit matvecs use `dequant_matvec_4bit_qmv`**
  (1.3–1.5× v3's bandwidth in isolation; `STREAMMOE_MATVEC_QMV=0`
  restores v3). LM head 8.2 → 5.9 ms, full-attention projections
  3.2 → 2.8 ms per token; greedy output identical. tok/s is unchanged
  within noise because decode waits on disk reads, and `cmd2_3`
  (~15 small dependent dispatches per layer) is bound by dispatch
  latency, not bandwidth.
- **3-bit experts read as much as 4-bit ones** (3-bit codes kept in
  4-bit nibbles, 1.77 MB per expert). Packed 3-bit experts
  (`tools/pack_experts.py`, 8 codes per 3 bytes, MLX's layout) are
  1.38 MB, 22% smaller; the engine detects the layout from the layer
  file size (`ExpertLayout`), sizes slots, staging and hot tier from
  it, and runs `dequant_matvec_3bit_v3_experts` (decode) and
  `streammoe_mm_id` with `bits = 3` (prefill). Both keep the 4-bit
  kernels' element order and per-weight `scale * q + bias`, so packed
  and nibble 3-bit give bit-identical output (`tests/packed_3bit.rs`,
  `moe_mm_id_packed_3bit_matches_nibble_layout`; on the model: same
  perplexity to every digit, KL 5e-12, 10/10 identical texts). a3b
  chat at a 4 GiB cap, 4-bit → packed 3-bit: forward 118 → 89,
  146 → 115, 121 → 97 ms/token; first word 1.57 → 1.11, 1.38 → 0.92,
  1.36 → 0.90 s; hot tier 719 → 997 slots in the same memory.

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
- Packed 3-bit experts need the default prefill path
  (`MoeGatherIdFuse`): it is forced for them, and the
  `STREAMMOE_MOE_GATHER_ID=0` fallbacks and the per-expert decode
  fallback (hidden > 4096, Cogito) have 4-bit kernels only. Every
  layer file must hold the same layout (`ExpertIoError::MixedLayouts`).
- **Checkpoints cost cap memory up front.** Under `mem_cap`,
  `max_checkpoints` (default 4) full-context snapshots are planned
  whether or not they are used: on OLMoE at n_ctx 2048 F32 that is
  4 × 0.5 GiB, enough to fail a 3 GiB cap. Set `max_checkpoints: 0`
  when nothing calls `checkpoint_pos` (pipeline stages: `streammoe-node`
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
- **mmap and pread decode differ by rounding** once a full-attention
  KV cache holds 32 rows: mmap mode uses the GPU SDPA, pread mode the
  CPU one (~1e-5 on the logits, same greedy tokens on the tests). Set
  `STREAMMOE_DECODE_SDPA=cpu` (or `gpu`) on both to compare them bit
  for bit.
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
- **GPU time per command buffer**: set `STREAMMOE_GPU_TIMING=<file>`
  to log `<label> <gpu_start> <gpu_end> <cpu_resumed>` (seconds, host
  clock) for every waited command buffer; `CmdbufStat::gpu_ns` now
  holds GPU time too (it was always 0 — metal-rs has no binding for
  `GPUStartTime` / `GPUEndTime`, so `gpu_timing.rs` calls them through
  `msg_send!`).
- **Disk timeline**: set `STREAMMOE_IO_TRACE=<file>` to log every
  expert read (start, end, kind, layer, bytes) and per-layer phase
  markers, then `python3 tools/io_trace_report.py <file>` prints disk
  busy / idle time per decode phase and prefill staging throughput.
  It slows decode ~15% — use it to see where time goes, never to
  measure speed.
- The SSD tops out at ~6.7 GB/s once ≥ 4 reads are in flight in older
  measurements; at decode sizes on 2026-10-26 (`tests/ssd_read_bench.rs`,
  1.77 MB experts, `F_NOCACHE`) it gave 2.9 GB/s for one read, 4.0 for
  four and 4.4 at most. Decode is bandwidth-bound — a3b at a 4 GiB cap
  reads ~330 MiB per token, ~75 ms of disk at 4.4 GB/s — so fewer bytes
  per token is what speeds it up.
- **`Ctx::cmdbuf_stats` now counts every waited command buffer.**
  It used to miss the deferred expert dispatch (`moe.experts_chained`,
  `moe.experts`) and the LM head, ~37 of ~74 ms GPU per a3b decode
  token, which made ~40 ms per token look unexplained. Their wait is
  ~0 (done by the time the next layer drains them), so wall − waits −
  I/O is CPU time that overlaps expert GPU work, not idle time.
- **Expert kernels are not the decode bottleneck by themselves**
  (`tests/packed_3bit.rs` `decode_experts_gpu_time_4bit_vs_packed_3bit`,
  synthetic, warm GPU): one 8-expert dispatch takes 0.22 ms at 4-bit
  (64 GB/s) and 0.21 ms packed 3-bit (52 GB/s — the byte unpacking
  costs some efficiency, so 3-bit saves disk and RAM, not GPU time).
  Inside decode the same dispatch takes ~0.75 ms: the GPU's slow
  state between short bursts, not the kernel. The first, cold runs
  in the bench take 0.56–0.62 ms.
- **Parallel decoding pays; speculative decoding does not yet.**
  `batch_bench` (4 requests × 64 tokens, packed 3-bit): batches of
  2 / 3 / 4 give 1.18 / 1.12 / 1.25× total tok/s at a 5 GiB cap and
  1.22 / 1.47 / 1.35× at 7 GiB (one run each), 1.16–1.23 / 1.28–1.29 /
  1.30–1.33× at 6 GiB (three runs, early experts on), every output identical
  to one-at-a-time decode; each request then gets 3–5.7 tok/s instead
  of ~9. A bigger cap helps batches more than plain decode, since the
  pass's distinct experts hit RAM more. What a 4-token pass adds over
  one token (`verify_bench --profile`): disk wait 20 → 81 ms, expert
  GPU 31 → 94, attention GPU 33 → 96, LM head 6 → 21.
- **Early expert dispatch** (resident experts' FFN sent before the
  disk reads of the rest; `STREAMMOE_EARLY_EXPERTS`): alternating
  `verify_bench` runs, packed 3-bit, 4 GiB cap — warm decode 107.1 /
  107.2 ms/tok off vs 100.8 / 102.8 on (~5%), cold first decode ~1.5%;
  quality harness bit-identical. The GPU only gets the resident
  experts' share of a layer (~0.5 ms) to fill the disk wait, so it
  does not end the slow state.
- Per-turn decode I/O in the chat log: `io` in each `[chat]` line gives
  forward ms per token, the I/O wait inside it, and demand / prefetch
  reads per token.

### Tests and tooling

- `graph_diff_oracle.rs` and `checkpoint_restore.rs` compile only with
  `model-qwen3-6-35b-a3b`; the graph-diff kernel tests need no model
  files, so run them under a3b even on a host without a3b weights.
- `checkpoint_restore.rs` fails under a17b — expected.
- `dequant_matvec_4bit_n_tokens_v3_n1_matches_single`
  (`batched_diff_oracle.rs`) failed since the qmv kernel landed: one-token
  `encode_matvec_n_tokens` takes qmv, but the test's reference was still
  the v3 kernel, which sums in another order. The reference now follows
  `STREAMMOE_MATVEC_QMV` like the engine; the test passes either way.
- Model suites that open uncapped (`smoke`, `mlx_regression`,
  `consecutive_eval_prompt`, `checkpoint_restore`, `diff_oracle`) take
  `STREAMMOE_TEST_CAP_GIB=<n>` to open under a cap; 4 is the smallest
  that holds the default checkpoints. `diff_oracle` opens two contexts
  at once; the cap check measures the whole process, so it cannot run
  under any cap and needs an uncapped run with ~8 GiB free. Run model
  tests under a memory guard, one binary at a time.
- `engine_op_diff` and `mmap_matches_pread` need mmap mode; the auto
  gate picks pread on an 18 GB host, so `engine_op_diff` needs
  `STREAMMOE_EXPERT_IO=mmap` (`mmap_matches_pread` sets the mode
  itself). `vision` needs `STREAMMOE_VISION_DIR` (the exported
  `vision_weights`) and `STREAMMOE_VISION_REF` (reference embeddings).
- A cap counts the whole process, the caller's buffers included: a
  `state_save` buffer or logits rows the application holds come out
  of `runtime_overhead` unless passed as `host_buffers`.
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
- A multi-column qmv for decode with T tokens (one weight read for
  T ≤ 4 inputs, bit-exact to qmv with `clang fp reassociate(off)` on
  both): 1.15–1.9× at T=4 in a warm microbench, no gain inside
  `eval_tokens` / `eval_requests` (attention GPU 63 / 100 / 96 ms at
  T=2/3/4 vs 52 / 71 / 95 per-token; parallel decoding 1.18–1.25× either
  way). In decode's slow GPU state the per-token arithmetic, not the
  weight reads, sets the cost. `fp contract(off)` on top made T=2/3
  3× slower.
- More from §8 of the plan: soft page-cache allowance, growing the hot
  tier when idle, lazily committed KV, LZ4 / LZFSE compression,
  per-expert files, all-2-bit experts.
- Warm-starting the chat from a routing profile at small caps: at
  4 GiB the tier holds 637 experts and each token routes 320, so the
  tier turns over within the first tokens of any prompt and seeded and
  cold launches measure the same. May help at caps where the tier holds
  a large share of the model — unmeasured.
- Skipping low-weight experts to save reads: a3b's gate weights are
  flat — the top 6 of 8 carry 84% of the weight and only 1.4% of picks
  are under 5% — so dropping picks costs real quality.
- Splitting each expert read into 2–8 pieces for more reads in flight:
  slower (4 experts: 1.76 ms whole, 1.92 ms in 4 pieces).
- Larger or sequential reads: one `pread` of n adjacent experts is only
  5–8% faster than n random ones (8 experts: 3.14 vs 3.39 ms) — a
  1.77 MB expert is already a large block. Worth it only bundled with
  a co-activation file layout, and decode reads ~1.5 misses per layer.
- Using idle disk time during decode (a3b, 4 GiB cap, 2026-10-26; I/O
  timeline trace of 314 tokens): of ~141 ms/token the disk is idle
  ~54 ms, but ~45 ms of that is ~1.1 ms per layer while the GPU runs
  attention — too short to absorb extra reads, which then delay the
  miss reads (the critical path). Wider cross-layer prefetch: 8 guesses
  within noise, 10 / 12 / 16 slower (up to 258 ms/token); misses fall
  54 → 14 but total bytes grow faster. Guessing two layers ahead:
  top-8 overlap 5.75/8 vs 6.37/8 one ahead — weaker. I/O threads at
  user-interactive QoS and `io_threads` 12: no effect. Guessing layer 0
  from the token embedding cuts its misses 6.57 → 4.49 per token but is
  within noise end to end. Decode is bandwidth-bound: read fewer bytes.
- Overlapping prefill staging with GPU work: staging already reads only
  the routed experts at the SSD's peak (~4.6 GB/s); GPU time between
  stages is ~0.33 s per short prefill, so the ceiling is −23% time to
  first word with perfect guesses, realistically −10–15%, for staging
  slots the GPU may still be reading.
- All-2-bit experts, re-measured on a3b (2026-10-26): experts rounded
  to 4 levels (`tools/requant_experts.py`, kept in the 4-bit layout) vs
  the 4-bit model on 314 teacher-forced tokens and 10 scored tasks
  (`tools/chat/src/bin/quality.rs`): KL 0.29, top-1 agreement 82%,
  perplexity 1.15 → 1.64; broke a tool call (no opening `<tool_call>`),
  17 × 23 = 39, a JSON array of objects came back as flat strings, and
  `is_prime` returned True for every n ≤ 7. 3-bit (8 levels) held up:
  KL 0.048, top-1 93%, perplexity 1.21, all 10 tasks right — now the
  packed 3-bit format (see Fixed > Speed).

