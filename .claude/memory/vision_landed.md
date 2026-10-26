# Vision for Qwen3.5 / 3.6 — landed 2026-10-26

Commits `b356c5d` (M-RoPE), `27c30c4` (encoder), `0a5d60b` (chat),
`0b485da` (docs), on streamoe main.

## What exists

- `Ctx::eval_stage_mrope(input, start_pos, &[[t, h, w]], output)`:
  per-token multimodal RoPE, interleaved `[11, 11, 10]`
  (`variants::MROPE_SECTION`, a3b only). CPU `apply_mrope` (per-token
  path) and Metal `rope_n_tokens_mrope` (batched). `[p, p, p]` rows are
  bit-identical to plain RoPE.
- `riir::vision::VisionEncoder`: CPU ViT (27 blocks, Accelerate GEMMs),
  BF16 weights streamed per matrix with F_NOCACHE from
  `artifacts/vision_weights.bin` (`tools/export_vision.py`). All
  buffers preallocated; `bytes()` goes into `OpenConfig::host_buffers`
  (137 MiB at the 640×640 default). `smart_resize`,
  `image_rope_positions` (`[s, s+r, s+c]`, text resumes at
  `s + max(gh, gw)`).
- `tools/chat`: image upload / paste, rope-delta bookkeeping, `/stats`
  CPU / RAM bar served by an acceptor thread while a turn runs.

## How it was verified

- Encoder vs transformers `Qwen3_5MoeVisionModel` fp32: cosine
  1.000000 per row (`tests/vision.rs`, needs reference files); vs
  mlx_vlm vision tower (bf16): 0.997 mean.
- Positions equal mlx_vlm's `get_rope_index`.
- Engine vs MLX, layer by layer on an image prompt (MLX streamed one
  layer at a time to stay under the cap): cosine ≥ 0.99 through layer
  32; top tokens overlap.
- `tests/mrope.rs` (a3b, 3 GiB cap): text rows bitwise, image rows
  batched vs per-row within the noise floor, 3-call prefill
  bit-identical to one call.
- All 92 160 packed expert tensors byte-identical to the safetensors.

## Known limitation — don't re-debug

Small printed text reads poorly ("STOP" → "SOM"/"Form"). MLX gives the
same top tokens on the same embeddings, so it is the 4-bit model, not
the engine. Larger images help.

## Useful tooling

LM Studio bundles a Python with `mlx` + `mlx_vlm`:
`~/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-*/bin/python3`.
`mlx_vlm.utils.load_model(path, lazy=True, strict=False)` and running
layers one at a time (drop each after use, `mx.clear_cache()`) gives a
full-model MLX reference at ~1 GiB. Always run heavy jobs under a
memory guard and `scripts/check_test_host.sh` first; the host is shared.
