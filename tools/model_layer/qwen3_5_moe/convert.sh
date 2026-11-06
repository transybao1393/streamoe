#!/bin/zsh
# Qwen3.5 / Qwen3.6 MoE (Qwen3.5-397B-A17B, Qwen3.6-35B-A3B): download the
# MLX 4-bit build and convert it to the streamoe layout. Wraps the shared
# scripts in scripts/ and tools/; see README.md next to this file.
#
#   tools/model_layer/qwen3_5_moe/convert.sh <hf-repo> <model-dir> [--no-download]
#
# <model-dir> ends up with mlx-4bit/, artifacts/, expert_index.json and
# root/packed_experts/. mlx-4bit/ can be deleted afterwards.
set -euo pipefail
[[ $# -ge 2 ]] || { sed -n 2,9p $0; exit 2; }
REPO=$1 M=$2
ROOT=${0:A:h:h:h:h}
mkdir -p $M
[[ ${3:-} == --no-download ]] || hf download $REPO --local-dir $M/mlx-4bit
python3 -I $ROOT/scripts/extract_weights.py --model $M/mlx-4bit --output $M/artifacts
python3 -I $ROOT/scripts/export_vocab.py $M/mlx-4bit/tokenizer.json $M/artifacts/vocab.bin
python3 -I $ROOT/tools/gen_expert_index.py --mlx-path $M/mlx-4bit --out $M/expert_index.json
python3 -I $ROOT/scripts/repack_experts.py --index $M/expert_index.json --output-dir $M/root/packed_experts
