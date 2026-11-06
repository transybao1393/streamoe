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
//! - MLX 4-bit weights use group size 32 (router gates 8-bit, group 64).
//!
//! Reference: Hugging Face transformers `models/qwen4_exp`. Spec and
//! status: `docs/design-notes.md#qwen38-flash-next`.

pub mod ngram;
pub mod ple;
