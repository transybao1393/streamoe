//! Per-family model layers.
//!
//! Each model family whose layer structure differs from the
//! qwen3_5_moe / DeepSeek-V3 paths in `attn/` and `moe/` gets its own
//! folder here. A family folder holds what is new for that family
//! (its layer forward, new ops, weight names) and reuses the shared
//! kernels in `attn/`, `moe/` and `backend/` for everything else.
//!
//! The family's tools (converter, reference, tiny test models) live
//! in the matching `tools/model_layer/<family>/` folder.

#[cfg(feature = "model-qwen3-8-flash-next")]
pub mod qwen3_8_flash_next;
