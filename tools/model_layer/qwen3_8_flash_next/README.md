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

## Packed 3-bit experts

Optional: 20% fewer expert bytes to read and hold (the 4-bit group-32
scales and biases stay the same size, so the saving is smaller than
the 35B-A3B's 22%). Not yet run on the real model: checked on the tiny
model only (below). `tools/requant_experts.py` writes 3-bit codes in
the 4-bit layout, `tools/pack_experts.py` packs them 8 per 3 bytes;
both read the shapes (hidden 2560, moe_intermediate 640, group 32) and
the padded stride from the source's `packed_experts/layout.json`
(`--hidden / --inter / --group` override it).

| Experts | Bytes per expert | Stride | Per layer (512) | 48 layers |
| --- | --- | --- | --- | --- |
| 4-bit (`root/`) | 3,072,000 | 3,080,192 (16 KiB padded) | 1.47 GiB | 70.50 GiB |
| 3-bit in nibbles (temporary) | 3,072,000 | 3,080,192 | 1.47 GiB | 70.50 GiB |
| packed 3-bit (`root-3bit/`) | 2,457,600 | 2,457,600 (= 150 × 16 KiB, no padding) | 1.17 GiB | 56.25 GiB |

Each matrix is 819,200 B of 4-bit codes or 614,400 B packed, plus
102,400 B of scales and 102,400 B of biases. Disk is tight, so convert
one layer at a time: the temporary nibble copy never holds more than
one layer (1.47 GiB). The engine also needs the n-gram table in the
experts directory; a hard link costs no space.

```bash
M=<model dir>                                  # holds root/ (4-bit experts + ngram.mfng)
for L in $(seq 0 47); do
  python3 -I tools/requant_experts.py "$M/root" --only $L --out 3="$M/tmp-3nib"
  python3 -I tools/pack_experts.py "$M/tmp-3nib" "$M/root-3bit" --only $L
  rm "$M/tmp-3nib/packed_experts/layer_$(printf %02d $L).bin"
done
rm -r "$M/tmp-3nib"
ln "$M/root/ngram.mfng" "$M/root-3bit/ngram.mfng"
```

Peak disk on top of the 4-bit model: the packed output (56.25 GiB) plus
one nibble layer, ~57.7 GiB. To end with the 3-bit experts only, add
`rm "$M/root/packed_experts/layer_$(printf %02d $L).bin"` after the pack
step: the peak is then the 4-bit experts plus ~2.6 GiB (~73 GiB), and
14.25 GiB is freed by the end (re-converting 4-bit from the MLX source
is the way back). Time, measured on random data at these shapes: ~0.06 s
an expert to requantize and ~0.04 s to pack (~50 s a layer, ~40 min of
CPU for 48 × 512 experts), plus reading 2 × 70.5 GiB and writing
126.75 GiB on the drive — about an hour; a few hundred MB of RAM.
`pack_experts.py` fails if a code is above 7.

The engine tells the layout from `layout.json` and the files' size;
point `Ctx::open_with`'s experts directory at `root-3bit`, or run the
chat tool with `--precision 3`. Prefill and decode both run the
packed-3-bit gathered GEMM (`affine_gather_qmm_rhs` at gs 32, b 3).

Tiny-model checks (`STREAMMOE_Q38_TINY` as above; the tests run the two
tools themselves, needing `python3` with NumPy):

```bash
STREAMMOE_Q38_TINY=DIR/tiny cargo test -p streammoe --features model-qwen3-8-flash-next-tiny \
    --lib packed3_tests -- --nocapture
STREAMMOE_Q38_TINY=DIR/tiny cargo test -p streammoe --features model-qwen3-8-flash-next-tiny \
    --test q38_engine ctx_packed_3bit -- --nocapture
```

Packed 3-bit gives the nibble 3-bit logits and state bit for bit on
Metal and on the CPU backend (prefill chunks of 512 and 5, a step per
token, hot tier on); against 4-bit over 69 rows, KL mean 0.0013, max
0.0031 (random weights, so this says nothing about real quality — run
`tools/chat`'s `quality` bin against 4-bit after converting).
