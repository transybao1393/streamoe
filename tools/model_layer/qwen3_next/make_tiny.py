#!/usr/bin/env python3
"""Make a tiny random-weight Qwen3-Next in the MLX 4-bit layout, and its goldens.

    python3 make_tiny.py --out DIR [--seed 3] [--name tiny]

Same structure and tensor names as mlx-community/Qwen3-Next-80B-A3B-
Instruct-4bit at a small size: 8 layers as 2 x (3 Gated DeltaNet + 1
gated full attention), the fused `in_proj_qkvz` / `in_proj_ba`, 32
experts top-10 (the real k; the batched MoE kernel has k = 8 and 10
only) plus a sigmoid-gated shared expert, affine 4-bit group 64 with
`mlp.gate` and `mlp.shared_expert_gate` 8-bit group 64, norms stored
as `1 + w`, A_log in BF16 like the real build. The per-head
dims are the real model's (linear attention 128, attention head_dim 256
with a quarter rotary) and hidden is 512, so the
`model-qwen3-next-tiny` engine build runs the same kernels as the real
one. TINY in this file, the `model-qwen3-next-tiny` VARIANT in
crates/streammoe/src/riir/variants.rs: keep them in step.

The weights go into 3 shards split mid-layer, so the converter's
one-shard-at-a-time path sees a layer's tensors in two shards.

Writes, under DIR/<name>/:

    mlx-4bit/   config.json (with `quantization`), model-0000{1,2,3}-of-00003
                .safetensors + model.safetensors.index.json
    goldens.safetensors, goldens.json   reference outputs, see below
    trace.safetensors                   prompt 0 prefill intermediates

Goldens come from reference.py in float32 on the dequantized 4-bit
weights (`w = q * scale + bias`, what the engine computes). For prompt i
(the q38 goldens' format):

    q4.p{i}.input_ids       I64 [T]
    q4.p{i}.prefill_logits  F32 [T, V]   logits of every prompt position
    q4.p{i}.decode_tokens   I64 [N]      greedy; token 0 = argmax of the last prefill row
    q4.p{i}.decode_logits   F32 [N, V]   logits after feeding decode_tokens[k]

Prompts: 7 tokens, 40 (past the engine's 32-row fused-SDPA threshold)
and 600 (past the 512-token capped prefill chunk). Every prompt runs
the two full-attention layers over many RoPE positions.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx_io  # noqa: E402
import reference  # noqa: E402
import schema  # noqa: E402

TINY = {
    "hidden_size": 512, "num_hidden_layers": 8, "vocab_size": 1024,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256,
    "linear_num_key_heads": 2, "linear_num_value_heads": 4,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
    "num_experts": 32, "num_experts_per_tok": 10,
    "moe_intermediate_size": 128, "shared_expert_intermediate_size": 128,
    "full_attention_interval": 4,
}
PROMPT_LENS = (7, 40, 600)
N_DECODE = 6


def tiny_config():
    cfg = {
        "architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
        "attention_dropout": 0.0, "bos_token_id": 1021, "eos_token_id": 1023,
        "decoder_sparse_step": 1, "hidden_act": "silu", "intermediate_size": 1024,
        "max_position_embeddings": 4096, "mlp_only_layers": [], "norm_topk_prob": True,
        "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-6, "rope_scaling": None,
        "rope_theta": 10000000, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "use_sliding_window": False,
    }
    cfg.update(TINY)
    q = {"group_size": 64, "bits": 4, "mode": "affine"}
    for i in range(TINY["num_hidden_layers"]):
        for n in ("mlp.gate", "mlp.shared_expert_gate"):
            q[f"model.layers.{i}.{n}"] = {"group_size": 64, "bits": 8}
    cfg["quantization"] = q
    cfg["quantization_config"] = dict(q)
    return cfg


def random_tensors(cfg, seed):
    """{safetensors name: (dtype, array)} in the MLX layout."""
    rng = np.random.RandomState(seed)
    rb = mlx_io.round_bf16
    out = {}
    for t in schema.mlx_tensors(cfg):
        n, shape = t.base, t.shape
        if t.kind in ("q", "e"):
            fan_in = shape[-1]
            std = 1.0 / np.sqrt(fan_in)
            if n.endswith("embed_tokens"):
                std = 1.0
            elif n.endswith("mlp.gate"):
                std = 3.0 / np.sqrt(fan_in)          # routing with clear margins
            w = rb(rng.randn(*shape).astype(np.float32) * np.float32(std))
            pk, s, b = mlx_io.affine_quantize(w, t.group, t.bits)
            out[n + ".weight"] = ("U32", pk)
            out[n + ".scales"] = ("BF16", s)
            out[n + ".biases"] = ("BF16", b)
            continue
        if n.endswith(schema.NORMS_1PW) or n.endswith("linear_attn.norm.weight"):
            v = 1.0 + 0.1 * rng.randn(*shape)        # stored 1 + w (gated norm: plain w near 1)
        elif n.endswith("conv1d.weight"):
            v = 0.5 * rng.randn(*shape)
        elif n.endswith("A_log"):
            v = np.log(rng.uniform(0.5, 8.0, size=shape))
        elif n.endswith("dt_bias"):
            v = 0.5 * rng.randn(*shape)
        else:
            raise AssertionError(n)
        out[n] = ("BF16", mlx_io.f32_to_bf16(v.astype(np.float32)))
    return out


def write_shards(out_dir, cfg, tensors):
    """3 shards, cut in the middle of layers 2 and 5."""
    os.makedirs(out_dir, exist_ok=True)

    def shard_of(name):
        if not name.startswith("model.layers."):
            return 0 if "embed" in name else 2
        li = int(name.split(".")[2])
        cut = {2: 1, 5: 2}
        if li in cut:
            return cut[li] if ".mlp." in name else cut[li] - 1
        return 0 if li < 2 else (1 if li < 5 else 2)

    groups = [{}, {}, {}]
    for n, v in tensors.items():
        groups[shard_of(n)][n] = v
    wm = {}
    total = 0
    for k, g in enumerate(groups):
        fn = f"model-{k + 1:05d}-of-00003.safetensors"
        mlx_io.write_safetensors(os.path.join(out_dir, fn), g, metadata={"format": "mlx"})
        for n, (_, a) in g.items():
            wm[n] = fn
            total += a.nbytes
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=2)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--name", default="tiny")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    root = os.path.join(a.out, a.name)
    mlx_dir = os.path.join(root, "mlx-4bit")
    cfg = tiny_config()
    write_shards(mlx_dir, cfg, random_tensors(cfg, a.seed))
    print("wrote", mlx_dir)

    cfg2, w = reference.load_mlx(mlx_dir)
    c = reference.model_config(cfg2)
    rng = np.random.RandomState(a.seed + 100)
    gold, info = {}, {"config": cfg, "prompts": [], "router_margin_min": None, "argmax_margin_min": None}
    margins, am = [], []
    trace0 = {}
    for i, T in enumerate(PROMPT_LENS):
        ids = [int(x) for x in rng.randint(0, TINY["vocab_size"] - 4, size=T)]
        tr = trace0 if i == 0 else {}
        pre, toks, dec = reference.generate(c, w, ids, N_DECODE, tr)
        margins += tr.get("router_margin", [])
        for row in torch.cat([pre[-1:], dec]):
            s = torch.sort(row, descending=True).values
            am.append(float(s[0] - s[1]))
        gold[f"q4.p{i}.input_ids"] = np.array(ids, np.int64)
        gold[f"q4.p{i}.prefill_logits"] = pre.numpy().astype(np.float32)
        gold[f"q4.p{i}.decode_tokens"] = np.array(toks, np.int64)
        gold[f"q4.p{i}.decode_logits"] = dec.numpy().astype(np.float32)
        info["prompts"].append({"len": T, "decode": toks})
        print(f"prompt {i}: T={T} decode={toks}")
    info["router_margin_min"] = min(margins) if margins else None
    info["argmax_margin_min"] = min(am)
    mlx_io.write_safetensors(os.path.join(root, "goldens.safetensors"), gold)
    json.dump(info, open(os.path.join(root, "goldens.json"), "w"), indent=1)
    tr = {k: v.numpy().astype(np.float32) for k, v in trace0.items() if torch.is_tensor(v)}
    mlx_io.write_safetensors(os.path.join(root, "trace.safetensors"), tr)
    print(f"smallest top-k router margin (every prompt) {info['router_margin_min']:.3e}, argmax margin {info['argmax_margin_min']:.3e}")


if __name__ == "__main__":
    main()
