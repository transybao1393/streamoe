# Qwen3.8-Flash-Next tools

Reference, tiny test models and converter for `qwen4_exp`. Python 3 with
NumPy; `reference.py`, `make_tiny.py` and `verify_convert.py` also need
torch (2.7 works, no transformers).

| File | What |
| --- | --- |
| `reference.py` | Pure-torch port of the transformers text model (prefill, cached decode, per-layer trace). The module doc lists every change from the original. |
| `schema.py` | Every MLX tensor's name, shape and quantization, from config.json; size report; index check. |
| `mlx_io.py` | Safetensors I/O, BF16, MLX affine quantize / dequantize (bit-exact with MLX 0.32.4). |
| `make_tiny.py` | Tiny random model (BF16, F32, MLX 4-bit) and goldens. |
| `convert.py` | MLX 4-bit → streamoe layout, one source shard at a time. |
| `ngram_layout.py` | The MFNG n-gram row table (q38-ple's format), streamed writer and reader. |
| `verify_convert.py` | Reads a converted directory back; checks bytes, goldens and the MFNG file. |

```bash
python3 make_tiny.py --out DIR                       # DIR/tiny/{bf16,f32,mlx-4bit,goldens.*,trace_*}
python3 -I convert.py --source DIR/tiny/mlx-4bit --out DIR/tiny/streammoe
python3 verify_convert.py --out DIR/tiny/streammoe --mlx DIR/tiny/mlx-4bit --goldens DIR/tiny
python3 -I schema.py --config ~/models/qwen3-8-flash-next/meta/config.json \
    --index ~/models/qwen3-8-flash-next/meta/model.safetensors.index.json
```

Real model: `convert.py --source <mlx dir> --out <dir> --delete-source`,
rerun after each shard arrives if the shards are downloaded one by one.
Peak disk is the output (~103 GiB) plus one source shard.

Goldens: `goldens.safetensors` holds, per variant `bf16` / `q4` and
prompt `i`, `{v}.p{i}.input_ids`, `prefill_logits [T, V]`,
`decode_tokens [N]` and `decode_logits [N, V]` (logits after feeding
`decode_tokens[k]`). Feed the golden tokens rather than your own argmax.
`q4` is what the engine should match: the MLX weights dequantized in f32.
`goldens.json` has the config, prompts and the smallest router / QSA /
argmax margins.
