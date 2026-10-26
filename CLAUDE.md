# CLAUDE.md

Guidance for Claude Code working in the **streamoe** repository
(https://github.com/transybao1393/streamoe, private), which grew out of
moeflux. The code still uses the `moeflux` names until the rename.

## Project status — read first (as of 2026-10-26)

- **This repo is streamoe.** The user's old fork
  `transybao1393/moeflux` is **ended**: never push to it. Push only to
  streamoe (`git remote -v` — in the original working copy the streamoe
  remote is named `streamoe` and `origin` is the old fork; fix the
  remotes if a clone says otherwise).
- History: everything up to `a36ae3b` (moeflux 0.1.0-pre.4) is
  upstream's (mdegans). The user's commits follow; up to `7d73350` they
  were public on the old fork under MIT. The vision work (`b356c5d`
  onward) was never on the old fork.
- **Open tasks, in order:**
  1. **License.** *Drafted 2026-10-26 with PolyForm Noncommercial —
     `LICENSE`, `LICENSE-MIT`, `NOTICE`, Cargo `license`, README;
     awaiting the user's review.* The user wants personal / academic use free and
     commercial use only by them. Pick with the user: PolyForm
     Noncommercial 1.0.0 (recommended) or BSL 1.1 with a change date.
     The new license covers only commits after `7d73350`; upstream code
     and the already-public commits stay MIT. Add `LICENSE` (new),
     `LICENSE-MIT` (current text, unchanged), a `NOTICE` saying what is
     under which license and crediting flash-moe / moeflux /
     Michael de Gans, and set `Cargo.toml` `license` to
     `MIT AND LicenseRef-PolyForm-Noncommercial-1.0.0` (or the BSL
     equivalent). Mention a lawyer for commercial licensing and the
     open question of copyright in AI-written code. Keep the repo
     private until the user has reviewed the license files.
  2. **Rename the code** moeflux → streamoe: crates (`crates/moeflux*`),
     package names, binaries (`moeflux-node`, `moeflux-coord`), docs,
     `tools/chat` dependency, test env vars only if the user agrees.
     Run the gate on all four variants afterwards.
  3. **Archive the old fork** on GitHub (README note pointing to
     streamoe first; archiving is outward-facing — ask before doing it).
  4. **Tell the drama_llama maintainer** (Michael de Gans) about the new
     repo and license; drama_llama path-points at this crate. The user
     sends that message — offer a draft, don't send anything.
- Details of the vision work and how it was verified:
  `.claude/memory/vision_landed.md`.
- **Known issues** — open bugs, fixed bugs, precautions for the shared
  host, configuration rules and measured dead ends — live in
  `docs/known-issues.md`. Read it before changing the engine; add an
  entry when a bug is fixed or a trap is found.

## What moeflux is (the engine streamoe is built on)

moeflux is a pure-Rust streaming-experts Mixture-of-Experts inference
engine for Apple Silicon. It began as a fork of danveloper's flash-moe
and has been rewritten in Rust on `metal-rs`; the Metal kernels were
authored by Claude Opus 4.6 for flash-moe and carry over. See
`README.md` for provenance.

moeflux is consumed by
[`drama_llama`](https://github.com/mdegans/drama_llama), which is
where this work is usually launched from — drama_llama is the
downstream consumer and is path-pointed at this crate. Treat
drama_llama's needs as the API's north star.

## Build & test

macOS only; on other targets the crate exposes no symbols. Exactly
one model-variant feature must be selected:

```bash
# Build, per variant
cargo build --features model-qwen3-6-35b-a3b
cargo build --features model-qwen3-5-a17b
cargo build --features model-cogito-v2-671b
cargo build --features model-olmoe-1b-7b

# Check the host first: CPU, RAM, headroom, model files present
scripts/check_test_host.sh

# Behavioral gate — the oracle suites load the a3b model
cargo test --no-fail-fast --features model-qwen3-6-35b-a3b -- --include-ignored
```

Run `scripts/check_test_host.sh` before any test phase. Exit 2 means
only the unit/kernel tier can run here (no model files or too little
memory) — run without `--include-ignored` and say so.

`--no-fail-fast` so one model-mismatched test binary doesn't mask the
rest. `checkpoint_restore.rs` is a3b-only and fails under a17b — that
is expected. `resuming_prefill_after_seq_rm_matches_full_prefill` is
a known failure (linear-attn recurrence state is not
position-truncatable — see the fn-level comment); `--skip` it.
"Green" means every other Rust-path test passes.

`cargo fmt` is *not* a gate — the crate predates machine formatting
and is ~600 hunks off default rustfmt with no `rustfmt.toml`. Match
surrounding style by eye; don't bulk-reformat.

## Architecture

- `crates/moeflux/` — the only crate. `RsCtx::open` opens a model;
  `eval_prompt` / `eval_token` / `state_save` / `state_load` are the
  public surface (`RsCtx` is re-exported as `Ctx`).
- `src/riir/` — the engine. `backend/{cpu,gpu}/` is the `Backend`
  trait with CPU and Metal implementations; `attn/`, `moe/`, `io/`
  (weight streaming), `snapshot/` (state save/load) hold the rest.
- `src/riir/variants.rs` — compile-time per-variant shape constants,
  gated by the model features. See `docs/model_variants.md`.
- `crates/moeflux/shaders/shaders.metal` — the Metal kernels, embedded
  via `include_str!` and compiled at runtime.

## Conventions

- **Land work directly on `main`** while drama_llama is the sole
  consumer. Revisit when external users appear.
- **Model shape is compile-time.** One `model-*` feature at a time;
  runtime variant dispatch is future work.
- **The C oracle is retired.** moeflux is C-free; correctness is
  cross-checked Rust-internally — `graph_diff_oracle` (CpuBackend vs
  MetalBackend), `batched_diff_oracle` (batched vs per-token), and
  `mlx_regression` (vs an MLX reference). See
  `.claude/memory/future_work_c_oracle_broken_post_riir.md` for why.

## Durable context

Session-spanning notes live in `.claude/memory/` — graph-mode arc
landings, the cleanup arc, future-work items. Read the most recent
`*_landed.md` when picking up.
