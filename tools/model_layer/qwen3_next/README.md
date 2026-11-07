# Qwen3-Next tools

Converter, reference and tiny test model for `qwen3_next`
(Qwen3-Next-80B-A3B, MLX build `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`).
The engine runs it on the `qwen3_5_moe` code path with the
`model-qwen3-next-80b-a3b` feature: the math is the same, only two
projections are stored fused (see `docs/model_variants.md`), so there is
no family code under `crates/streammoe/src/riir/model_layer/`.

| File | What |
| --- | --- |
| `schema.py` | Every MLX tensor's name, shape and quantization from config.json; the converted layout; sizes; index check. |
| `mlx_io.py` | Safetensors I/O, BF16, MLX affine quantize / dequantize (a copy of the Qwen3.8 one). |
| `convert.py` | MLX 4-bit → streamoe layout, one source shard at a time, with read-back and split checks. NumPy only (`python3 -I`). |
| `reference.py` | Pure-torch port of transformers' `modeling_qwen3_next.py` (torch 2.7, no transformers). The module doc lists what changed. |
| `make_tiny.py` | Tiny random model in the exact MLX layout (fused `in_proj_qkvz` / `in_proj_ba`, 8-bit router) and goldens from `reference.py`. |

## Real model

```bash
python3 -I tools/model_layer/qwen3_next/schema.py \
    --config "/Volumes/External NVMe M2 PCIe 4.0/models/qwen3-next-80b-a3b/mlx-4bit/config.json" \
    --index "/Volumes/External NVMe M2 PCIe 4.0/models/qwen3-next-80b-a3b/mlx-4bit/model.safetensors.index.json"

python3 -I tools/model_layer/qwen3_next/convert.py \
    --source "/Volumes/External NVMe M2 PCIe 4.0/models/qwen3-next-80b-a3b/mlx-4bit" \
    --out "/Volumes/External NVMe M2 PCIe 4.0/models/qwen3-next-80b-a3b"
```

Output: `artifacts/model_weights.{bin,json}` (1.26 GiB), `artifacts/vocab.bin`
and `root/packed_experts/layer_00..47.bin` (48 × 512 experts of
1,769,472 B, 0.84 GiB per layer, 40.5 GiB in all): 41.8 GiB, about the
size of the 44.8 GB download. Shards can be converted as they arrive
(rerun after each); `--delete-source` deletes each verified shard, so
peak disk is the output plus one shard. The converter reads ~45 GB and
writes ~42 GB on the same drive: run it when nothing else writes there.

Run it with the chat tool (`tools/chat`, `--no-default-features
--features qnext`) or any `Ctx::open_with(artifacts/model_weights.bin,
artifacts/model_weights.json, artifacts/vocab.bin, root, …)` built with
`--features model-qwen3-next-80b-a3b`.

Each shard's line reports the tensors written and read back and the
split check (a random vector through the fused rows, regrouped the way
transformers does, against the written projections; worst relative
error 0 expected). The converter stops on a config it does not support
(dense layers, attention bias, non-default RoPE), on a tensor whose
dtype or shape differs from the plan, and on zero-centred norms
(input / post-attention medians below 0.5: transformers' raw `w`
rather than mlx-lm's `1 + w`; `--norm-offset 1` converts such a
checkpoint).

## Tiny model and tests

```bash
Q=tmp/qwen3-next/qnext           # any scratch dir; ~110 MB
python3 tools/model_layer/qwen3_next/make_tiny.py --out $Q
python3 -I tools/model_layer/qwen3_next/convert.py --source $Q/tiny/mlx-4bit --out $Q/tiny/streammoe
python3 -I tools/model_layer/qwen3_next/convert.py --source $Q/tiny/mlx-4bit --out $Q/tiny/streammoe-naive --layout naive
STREAMMOE_QNEXT_TINY=$PWD/$Q/tiny cargo test -p streammoe --features model-qwen3-next-tiny --test qnext_engine
```

The tiny model: hidden 512, 8 layers (2 × (3 Gated DeltaNet + 1
full attention)), the real per-head dims (linear attention 2 key / 4
value heads of 128, attention 4 / 2 heads of 256, quarter rotary), 32
experts top-10, vocab 1024, three shards cut mid-layer. Goldens
(`goldens.safetensors`, the Qwen3.8 format: `q4.p{i}.input_ids`,
`prefill_logits`, `decode_tokens`, `decode_logits`) are the reference
on the dequantized 4-bit weights for prompts of 7, 40 and 600 tokens
plus 6 greedy decode steps. `--layout naive` writes the fused rows as
contiguous `[q|k|v|z]` / `[b|a]` blocks (the bug the converter must
not have); `qnext_engine` checks that this copy is far from the
goldens.
