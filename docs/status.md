# Status and tests

What exists, what is implemented and how it was tested. The current version and the short summary are in the [README](../README.md#status); changes are listed in [`CHANGELOG.md`](../CHANGELOG.md).

| Area | State |
| --- | --- |
| Engine, Metal kernels, Qwen3.5-A17B / Qwen3.6-35B-A3B / Cogito-V2 variants | Exists |
| OLMoE-1B-7B variant (small test model: converter, NumPy reference, model tests under a cap) | Implemented — the model-level test bed on hosts without a3b weights |
| Expert streaming: mmap mode, pread + speculative prefetch, slot ring | Exists |
| Differential oracles (CPU vs Metal, batched vs per-token, MLX) | Exists |
| `OpenConfig`, `n_ctx`-sized KV, memory cap (plan, reserve at open, footprint checks) | Implemented — Phase 1; tested on a3b by the model suites ([Tests](#tests)): open fails before any I/O when the plan doesn't fit, checkpoints planned inside a 4 GiB cap, a 128-token decode stays flat inside a 4 GiB cap at the default overhead |
| `F_NOCACHE` reads, fixed hot tier, routing profiles, expert-IO counters | Implemented — Phase 2; tested on a3b by the model suites ([Tests](#tests)): hot-tier sizes and profile seeding give bit-identical logits |
| Layer-range execution (`layer_range`, `eval_stage`, f32 hidden in/out, per-range snapshots) | Implemented — Phase 3; tested on a3b by the model suites ([Tests](#tests)): whole-model `eval_stage` bit-identical to `eval_prompt`, 40 layers vs 20 + 10 + 10 within tolerance, per-range snapshots round-trip |
| Per-request state (`max_requests` pool, `admit_request` / `drop_request` / `eval_request`) | Implemented — Phase 4; tested on a3b by the model suites ([Tests](#tests)): interleaved requests bit-identical to sequential runs |
| `streammoe-net`: frames, `streammoe-node`, `streammoe-coord`, planner (TCP loopback) | Prototype — Phase 5; mode B continues in a separate repository |
| Streaming improvements: telemetry (`ExpertIoStats`, CSV), bounded I/O (`io_queue_depth`, `io_bytes_per_sec`), in-place coalesced prefill reads, cross-layer guesses, lookahead prefetch, `fast_experts_dir` disk tier, `prefetch_experts` (prefetch width) | Implemented — Phase 6; tested on a3b by the model suites ([Tests](#tests)): bounded I/O, prefill reads, cross-layer guess, lookahead and the fast-dir tier all give identical output. Library defaults: previous-token guess, `prefetch_experts` = 4 under a cap; `tools/chat` uses the cross-layer guess with 6, faster on a3b ([Prefetch and reuse](benchmarks.md#prefetch-and-reuse)) |
| KV cache formats (`kv_format`: F32, F16, TurboQuant 4/3-bit; snapshot v4) | Implemented — F32 default; on OLMoE-1B-7B at a 3 GiB cap and n_ctx 4096, F16 / Turbo4 shrink KV 1.00 → 0.56 / 0.19 GiB (incl. prefill scratch) and decode +29% / +44% faster (the freed memory goes to the hot tier); answers correct in every format. MLA (Cogito) stays F32 |
| Decode attention threads (`attn_threads`, default 4) | Implemented — bit-identical output; on OLMoE at 2.7k context decode +55–58% tok/s (F32 and Turbo4) |
| Vision for Qwen3.5 / 3.6 (`riir::vision` CPU encoder, `eval_stage_mrope`, `tools/export_vision.py`) | Implemented — encoder matches transformers exactly and mlx_vlm to 0.997; whole pipeline checked layer by layer against MLX on a3b under a 4 GiB cap (encoder 137 MiB, ~1–3 s per image). Small printed text reads poorly, as it does in MLX |
| Decode I/O: per-slot prefetch waits, prefill staging and scratch lent to the hot tier (`STREAMMOE_HOT_LEND`, `STREAMMOE_HOT_LEND_SCRATCH`), hot-tier popularity that fades in ~13 tokens (`STREAMMOE_HOT_DECAY`), single-token `qmv` matvec kernel (`STREAMMOE_MATVEC_QMV`), hot experts dispatched before disk reads (`STREAMMOE_EARLY_EXPERTS`) | Implemented — output unchanged; effect of each in [Prefetch and reuse](benchmarks.md#prefetch-and-reuse) |
| Packed 3-bit experts (`tools/requant_experts.py` → `tools/pack_experts.py`, `root-3bit/`; layout detected from the files), quality harness (`tools/chat` `quality` bin: KL, top-1, perplexity, scored tasks), `--precision 4\|3` in the chat page | Implemented — 1.31 MiB per expert vs 1.69 (−22%); vs 4-bit: KL 0.048, top-1 agreement 93%, 10/10 tasks; 11–14% more tok/s at every cap from 3 to 6 GiB ([Speed](benchmarks.md#speed)). 2-bit fails 4 of 10 tasks |
| Several tokens per pass: `eval_tokens` + `rewind_tokens` (one sequence, speculative checks), `eval_requests` (one token from each of up to 4 requests), `OpenConfig::max_decode_tokens` (buffers in the plan) | Implemented — logits bit-identical to one token per pass; parallel decoding 1.3× total throughput, speculative decoding 1.08–1.12× in `spec_bench` with the match-length guard ([Several tokens per pass](benchmarks.md#several-tokens-per-pass)) |
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

## Tests

Run `scripts/check_test_host.sh` first; it says which tier (unit /
kernel, or model) this host can run. The model suites need the a3b
files (`STREAMMOE_SMOKE_ARTIFACTS`, `STREAMMOE_SMOKE_ROOT`; see
[Setup](../README.md#setup)). Suites that open the model uncapped take
`STREAMMOE_TEST_CAP_GIB=4` to run under a cap on a small host:

```bash
STREAMMOE_TEST_CAP_GIB=4 cargo test --release --features model-qwen3-6-35b-a3b \
    -- --include-ignored --test-threads=1 --skip resuming_prefill_after_seq_rm_matches_full_prefill
```

Last full run 2026-10-26, with the 2026-10-28 additions marked (M3 Pro
with 7 GiB free, release build):

| Suite | a3b result |
| --- | --- |
| `mlx_regression` (argmax and top-20 vs the MLX golden), `smoke` | pass |
| `checkpoint_restore` (23), `consecutive_eval_prompt` (3), `mrope` (3) | pass |
| `layer_range` (4), `requests` (1), `streaming` (7), `expert_source_equivalence` (2) | pass |
| `expert_source_equivalence::mmap_matches_pread`, `budget_cap` (2, with the long decode under a 4 GiB cap) — 2026-10-28 | pass |
| Re-run 2026-10-28 with the prefill scratch lent to the hot tier: `streaming` (7), `requests` (7), `checkpoint_restore` (23), `expert_source_equivalence` (3), `mrope` (3), `mlx_regression`, `vision`; then with the ~13-token hot-tier decay as well: `budget_cap` (10, incl. prefill after a fully lent decode bit-identical to the first), `streaming` (7), the unit tier (151) | pass |
| `vision` (3 images vs reference embeddings, cosine 1.000000), `engine_op_diff` (with `STREAMMOE_EXPERT_IO=mmap`) — 2026-10-28 | pass |
| `graph_diff_oracle` (22, CPU vs Metal), `batched_diff_oracle` (33, batched vs per-token kernels) | pass |
| The default tier (169 unit and kernel tests, no model files) | pass |
| `diff_oracle` (two contexts in one process, so it cannot run under a cap), the two 10 GiB-cap tests | not run — need ~8 / 10 GiB free |
| `resuming_prefill_after_seq_rm_matches_full_prefill` | known failure, skipped ([known issues](known-issues.md)) |
| a17b and Cogito model tests | not run — no weights on this host |

Fixed bugs, open issues and traps are listed in
[`docs/known-issues.md`](known-issues.md).
