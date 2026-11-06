//! Qwen3.8-Flash-Next (`qwen4_exp`, architecture preview).
//!
//! 48 layers as 12 × (3 Gated DeltaNet + 1 Qwen Sparse Attention),
//! each followed by a 512-expert MoE (top-10 + 1 shared, intermediate
//! 640) on a hidden size of 2560. New against Qwen3.5 / 3.6:
//!
//! - Gated Residual (hyper-connections): the residual stream is
//!   `hc_count` = 4 copies of the hidden state (10240 wide). Before
//!   attention and before the MoE, a low-rank gate (rank 320) mixes the
//!   copies into the block input, and the block output is injected back
//!   into each copy with its own weight. A final mixer folds the four
//!   copies into one before the LM head. There are no per-layer
//!   input / post-attention RMSNorms; each Gated Residual's `hc_norm`
//!   takes their place.
//! - Qwen Sparse Attention: GQA 24 Q / 2 KV heads, head dim 256,
//!   partial (0.25) interleaved MRoPE, sigmoid output gate, plus an MQA
//!   indexer that picks blocks of the KV cache up to a 2,048-token
//!   budget. At or under the budget every block is picked, so dense
//!   attention gives the same result.
//! - Per-layer embedding (PLE) at layer 2: a 20M-row 3-gram embedding,
//!   hashed and split into 128 shards, read from disk row by row.
//!   `ple_layer_ids = [2]` is one-based: it runs at layer index 1
//!   (weights `model.layers.1.ple.*`).
//! - MLX 4-bit weights use group size 32 (router gates 8-bit, group 64).
//! - Norm gains are stored centered at zero and applied as `1 + w`
//!   (`hc_norm`, Q/K norms, indexer and PLE norms); the Gated
//!   DeltaNet's output norm keeps a plain gain but gates with
//!   `sigmoid(z)` instead of qwen3_5_moe's `silu(z)`.
//!
//! Reference: Hugging Face transformers `models/qwen4_exp`. Spec and
//! status: `docs/design-notes.md#qwen38-flash-next`.
//!
//! ## Modules
//!
//! - [`config`] — family shapes `Variant` can't carry; [`config::Q38Config`].
//! - [`weights`] — the MLX tensor names and where each goes.
//! - [`gated_residual`] / [`gated_residual_gpu`] — the Gated Residual,
//!   CPU and Metal.
//! - [`layer`] — decoder-layer orchestration, the call points for QSA
//!   ([`layer::AttnBlock`]) and PLE ([`layer::PleHook`]), and a CPU
//!   end-to-end forward ([`layer::Q38CpuModel`]).
//! - [`ngram`] / [`ple`] — the per-layer n-gram embedding: hashing,
//!   streamed row table (`MFNG` file), PLE layer forward.
//! - [`qsa`] — Qwen Sparse Attention: dense and block-selected
//!   attention, the indexer, graph ops, [`qsa::QsaBlock`].
//! - [`tiny`] — tiny random-weight models matching the torch port
//!   `tools/model_layer/qwen3_8_flash_next/core_ref.py`.
//! - [`engine`] / [`engine_exec`] — the family in the engine:
//!   [`engine::Q38Engine`], the decode forward on Metal or the CPU
//!   backend (same steps op for op), which `RsCtx` runs for this family
//!   (`riir/q38_ctx.rs`).
//!
//! ## Engine integration
//!
//! The qwen3_5_moe GPU layer paths are not reused: they assume one
//! `hidden` stream with input / post-attention RMSNorms and
//! `h = h_mid + Σ experts + shared`. [`engine::Q38Engine`] builds each token from
//! the existing graph ops (matvecs, conv / delta-net, QSA, SDPA, MoE
//! permute-fuse and combine) plus family ops ([`engine_exec::FamOp`]:
//! Gated Residual norm / mix / inject / expand, `1 + w` head norm,
//! sigmoid-gated output norm), with the MoE combined against zero and
//! injected into the `hc · H` stream. Every `1 + w` gain is applied in
//! a kernel from the stored bf16 value. Experts stream through
//! `ExpertFiles` (padded stride from `packed_experts/layout.json`, hot
//! tier); PLE runs on the host. Prefill is token at a time; layer
//! ranges, multi-token decode passes, multi-request pools, non-f32 KV
//! and rewinding to a middle position are refused (`q38_ctx`).

pub mod config;
pub mod engine;
pub mod engine_exec;
#[cfg(all(test, feature = "model-qwen3-8-flash-next-tiny"))]
mod engine_tests;
pub mod gated_residual;
pub mod gated_residual_gpu;
pub mod layer;
pub mod ngram;
pub mod ple;
pub mod ple_gpu;
#[cfg(all(test, feature = "model-qwen3-8-flash-next-tiny"))]
mod prefetch_tests;
pub mod prefill_io;
pub mod qsa;
pub mod tiny;
pub mod weights;
