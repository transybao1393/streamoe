# Qwen3.5 / Qwen3.6 MoE tools

Converter for the `qwen3_5_moe` family: Qwen3.5-397B-A17B
(`model-qwen3-5-a17b`) and Qwen3.6-35B-A3B (`model-qwen3-6-35b-a3b`).
This family's layers are the engine's original code path in
`crates/streammoe/src/riir/`, so the scripts stay where they are and
`convert.sh` runs them in order.

| Step | Script | Output |
| --- | --- | --- |
| Download | `hf download <repo>` | `mlx-4bit/` |
| Non-expert weights | `scripts/extract_weights.py` | `artifacts/model_weights.{bin,json}` |
| Vocabulary | `scripts/export_vocab.py` | `artifacts/vocab.bin` |
| Expert offsets | `tools/gen_expert_index.py` | `expert_index.json` |
| Expert repack | `scripts/repack_experts.py` | `root/packed_experts/layer_XX.bin` |

```bash
# Qwen3.5-397B-A17B: 209 GiB download, 208 GiB converted
tools/model_layer/qwen3_5_moe/convert.sh mlx-community/Qwen3.5-397B-A17B-4bit \
    "/Volumes/External NVMe M2 PCIe 4.0/models/qwen3-5-397b-a17b"

# Qwen3.6-35B-A3B: 19 GiB download, 19 GiB converted
tools/model_layer/qwen3_5_moe/convert.sh mlx-community/Qwen3.6-35B-A3B-4bit ~/models/qwen3-6-35b-a3b
```

Disk peak is the download plus the converted copy; `mlx-4bit/` can be
deleted once `root/` and `artifacts/` exist. `--no-download` reuses an
existing `mlx-4bit/`. The 35B's optional vision and packed 3-bit steps
are in the top-level README.
