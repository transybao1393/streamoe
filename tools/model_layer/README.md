# Per-family model tools

One folder per model family whose layers live in
`crates/streammoe/src/riir/model_layer/<family>/`: its converter
(Hugging Face / MLX weights → streamoe layout), its reference
implementation for tests, and tiny random-weight test models.

| Family | Folder |
| --- | --- |
| Qwen3.5 / Qwen3.6 MoE (397B-A17B, 35B-A3B) | `qwen3_5_moe/` (wraps the shared scripts) |
| Qwen3.8-Flash-Next | `qwen3_8_flash_next/` |
| Qwen3-Next (80B-A3B) | `qwen3_next/` (converter only; the engine runs it on the qwen3_5_moe path) |
