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
//! - [`tiny`] — tiny random-weight models matching the torch port
//!   `tools/model_layer/qwen3_8_flash_next/core_ref.py`.
//!
//! ## Engine integration (not done yet)
//!
//! The engine's GPU layer paths assume the qwen3_5_moe residual: one
//! `hidden` stream, an input RMSNorm before attention, `h_mid =
//! residual + o_proj`, a post-attention RMSNorm before the router and
//! `h = h_mid + Σ experts + shared`. For this family each of those
//! points changes:
//!
//! - input norm → attention Gated Residual: the norm is fused into the
//!   previous layer's expert command buffer (`chain_next_norm_off`,
//!   `linear_attn_forward.rs`) or CMD1's prelude (`prev_layer_chained`
//!   skips it), and the batched graph pushes
//!   `Op::RmsNormBf16NTokens` with `input_layernorm_w`. Replace with
//!   [`gated_residual_gpu::Q38GpuKernels::encode_gated_residual_mix`]
//!   writing the block input where the norm output went
//!   (`buffers.normed` / `normed_id`).
//! - `h_mid = residual + o_proj` (`post_attention_tail`, the graph's
//!   `ResidualAddNTokens`) → `encode_inject` into the 10240-wide
//!   stream.
//! - post-attention norm → MoE Gated Residual; the router, experts and
//!   shared expert then read its mixed output unchanged.
//! - `combine` adds `h_mid` → the MoE output must come out without the
//!   residual term (combine against zero), then `encode_inject`.
//! - Final `model.norm` → the final mixer (`block_inject = None`),
//!   straight into the LM head.
//! - Gated DeltaNet output norm: the `gated_rms_norm` kernel needs a
//!   sigmoid-gate variant ([`config::GDN_OUTPUT_GATE`]).
//! - `LayerWeightCache::build` requires `input_layernorm.weight`; this
//!   family needs its own cache with the Gated Residual offsets
//!   ([`weights::gated_residual_names`]).
//! - Hidden-state buffers that cross layers (`HiddenBuf`, snapshots,
//!   `StageInput::Hidden` for layer ranges) become `hc · H` wide.

pub mod config;
pub mod gated_residual;
pub mod gated_residual_gpu;
pub mod layer;
pub mod tiny;
pub mod weights;
