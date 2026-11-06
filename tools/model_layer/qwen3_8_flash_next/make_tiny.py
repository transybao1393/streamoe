#!/usr/bin/env python3
"""Make a tiny random-weight Qwen3.8-Flash-Next and its golden outputs.

    python3 make_tiny.py --out DIR [--seed 2] [--name tiny]

Same structure as the real model at a small size: 4 layers as
3 Gated DeltaNet + 1 QSA, 4-copy hyper-connections, PLE at layer id 2
(0-based layer 1) with a 4-head n-gram table in 4 shards, 8 experts
(top 3 + 1 shared). The QSA budget is 8 tokens (2 blocks of 4), so
prompts over 8 tokens exercise the sparse block selection. All
quantized input widths are multiples of 32 (64 for the 8-bit gates).

Writes, under DIR/<name>/:

    bf16/       config.json, model.safetensors   MLX names, unquantized BF16
    f32/        the same values in F32 (exact BF16 values)
    mlx-4bit/   config.json (with `quantization`), 3 safetensors shards +
                model.safetensors.index.json, tokenizer.json — MLX's
                layout: affine 4-bit group 32, mlp.gate and
                mlp.shared_expert_gate 8-bit group 64
    goldens.safetensors, goldens.json        reference outputs, see below
    trace_bf16.safetensors, trace_q4.safetensors
                                             per-layer intermediates

Goldens come from reference.py in float32, once on the BF16 weights
(variant "bf16") and once on the dequantized 4-bit weights (variant
"q4": w = q * scale + bias in f32 — what the engine computes). For
prompt i and variant v:

    {v}.p{i}.input_ids       I64 [T]
    {v}.p{i}.prefill_logits  F32 [T, V]   logits of every prompt position
    {v}.p{i}.decode_tokens   I64 [N]      greedy tokens; token 0 is the
                                          argmax of the last prefill row
    {v}.p{i}.decode_logits   F32 [N, V]   logits after feeding
                                          decode_tokens[k] with the cache

Traces hold the same names as reference.py's trace dict, flattened
with "." (e.g. `prefill.layers.1.ple.ngram_ids`), for prompt 0's
prefill and its first decode step (`decode0.`).
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

V = 1024
EOS = 1023


def tiny_text_config():
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    return {
        "attention_bias": False, "attention_dropout": 0.0,
        "bos_token_id": 1022, "dtype": "bfloat16", "eos_token_id": EOS,
        "full_attention_interval": 4, "hc_count": 4, "hc_lowrank": 32,
        "head_dim": 64, "heads_per_ngram": 2, "hidden_act": "silu", "hidden_size": 128,
        "indexer_budget": 8, "indexer_compress_ratio": 4, "indexer_head_dim": 32,
        "indexer_kv_heads": 1, "indexer_n_heads": 4, "initializer_range": 0.02,
        "layer_types": layer_types, "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32, "linear_num_key_heads": 2,
        "linear_num_value_heads": 6, "linear_value_head_dim": 32,
        "make_ngram_vocab_size_divisible_by": 128, "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 4096, "model_type": "qwen4_exp_text",
        "moe_intermediate_size": 64, "mtp_num_hidden_layers": 0,
        "ngram_size": 3, "ngram_vocab_size_base": 1000, "norm_topk_prob": True,
        "num_attention_heads": 4, "num_experts": 8, "num_experts_per_tok": 3,
        "num_hidden_layers": 4, "num_key_value_heads": 2, "output_gate_type": "sigmoid",
        "output_router_logits": False, "pad_token_id": None,
        "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4, "ple_embed_dim": 128,
        "ple_layer_ids": [2], "rms_norm_eps": 1e-06,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [3, 3, 2],
                            "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
        "router_aux_loss_coef": 0.001, "seed": 1234, "shared_expert_intermediate_size": 64,
        "split_ngram_parts": 4, "tie_word_embeddings": False, "use_cache": True, "vocab_size": V,
    }


def tiny_config(quantized):
    tc = tiny_text_config()
    cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"], "eos_token_id": [1022, EOS],
           "model_type": "qwen4_exp", "language_model_only": True, "tie_word_embeddings": False,
           "text_config": tc}
    if quantized:
        q = {"group_size": 32, "bits": 4, "mode": "affine"}
        for i in range(tc["num_hidden_layers"]):
            for n in ("mlp.gate", "mlp.shared_expert_gate"):
                q[f"language_model.model.layers.{i}.{n}"] = {"group_size": 64, "bits": 8, "mode": "affine"}
        for li in schema.ple_layers(schema.text_config(cfg)):
            for s in range(tc["split_ngram_parts"]):
                q[f"language_model.model.layers.{li}.ple.ple_embedding.ngram_embedding.shards.{s}"] = \
                    {"group_size": 32, "bits": 4, "mode": "affine"}
        cfg["quantization"] = q
    return cfg


def random_weights(cfg, seed):
    """{MLX base name: float32 (BF16-exact) or int64 array}."""
    rng = np.random.RandomState(seed)
    tc = schema.text_config(cfg)
    out = {}
    for t in schema.tensors(cfg):
        n, shape = t.base, t.shape
        if t.kind == "i64":
            continue
        if t.kind == "q":
            fan_in = shape[-1]
            std = 1.0 / np.sqrt(fan_in)
            if n.endswith("embed_tokens") or ".ngram_embedding." in n:
                std = 0.5
            elif n.endswith(".mlp.gate"):
                std = 3.0 / np.sqrt(fan_in)      # spread the router probabilities
            elif n.endswith("block_inject_weight") or "input_mix_weight" in n:
                std = 2.0 / np.sqrt(fan_in)
            w = rng.randn(*shape) * std
        elif n.endswith("linear_attn.norm.weight"):
            w = 1.0 + 0.1 * rng.randn(*shape)                  # RMSNormGated: plain gain
        elif "norm" in n.rsplit(".", 2)[-2]:
            w = 0.1 * rng.randn(*shape)                        # (1 + w) norms
        elif n.endswith("A_log"):
            w = np.log(rng.uniform(0.5, 8.0, size=shape))
        elif n.endswith("dt_bias"):
            w = 1.0 + 0.3 * rng.randn(*shape)
        elif n.endswith("conv1d.weight"):
            w = rng.randn(*shape) * (0.4 if ".linear_attn." in n else 0.25)
        else:
            raise ValueError(f"no init for {n}")
        out[n] = mlx_io.round_bf16(w.astype(np.float32))
    for li in schema.ple_layers(tc):
        p = f"{schema.PREFIX}model.layers.{li}.ple.ple_embedding."
        pli = schema.ple_layers(tc).index(li)
        sizes, offsets, _, _ = schema.ngram_vocab(tc, pli)
        out[p + "layer_multipliers"] = reference._build_layer_multipliers(
            tc["vocab_size"], tc["ngram_size"], pli, tc["seed"]).numpy()
        out[p + "ngram_heads_offsets"] = np.array(offsets, np.int64)
        out[p + "ngram_heads_vocab_sizes"] = np.array(sizes, np.int64)
    return out


def unquantized_tensors(cfg, w, dtype):
    """MLX names -> (st dtype, array) for an unquantized checkpoint."""
    res = {}
    for t in schema.tensors(cfg):
        a = w[t.base]
        name = t.base + ".weight" if t.kind == "q" else t.base
        if t.kind == "i64":
            res[name] = ("I64", a)
        elif dtype == "BF16":
            res[name] = ("BF16", mlx_io.f32_to_bf16(a))
        else:
            res[name] = ("F32", a)
    return res


def quantized_tensors(cfg, w):
    res = {}
    for t in schema.tensors(cfg):
        a = w[t.base]
        if t.kind == "q":
            bits, g = schema.quant_rule(cfg, t.base)
            q, s, b = mlx_io.affine_quantize(a, g, bits)
            res[t.base + ".weight"] = ("U32", q)
            res[t.base + ".scales"] = ("BF16", s)
            res[t.base + ".biases"] = ("BF16", b)
        elif t.kind == "i64":
            res[t.base] = ("I64", a)
        else:
            res[t.base] = ("BF16", mlx_io.f32_to_bf16(a))
    return res


def write_sharded(out_dir, tensors, n_shards=3, force_last=()):
    """MLX-style shards + index. Names in `force_last` go to the last
    shard so one layer's expert components span two shards."""
    names = sorted(tensors)
    size = lambda n: tensors[n][1].nbytes
    total = sum(size(n) for n in names)
    budget = total / n_shards
    assign, acc, shard = {}, 0, 0
    for n in names:
        if any(f in n for f in force_last):
            assign[n] = n_shards - 1
            continue
        if acc > budget * (shard + 1) and shard < n_shards - 1:
            shard += 1
        assign[n] = shard
        acc += size(n)
    weight_map = {}
    for s in range(n_shards):
        fname = f"model-{s + 1:05d}-of-{n_shards:05d}.safetensors"
        part = {n: tensors[n] for n in names if assign[n] == s}
        mlx_io.write_safetensors(os.path.join(out_dir, fname), part, metadata={"format": "mlx"})
        weight_map.update({n: fname for n in part})
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)


def tiny_tokenizer():
    vocab = {f"t{i}": i for i in range(V - 2)}
    added = [{"id": 1022, "content": "<|bos|>", "special": True}, {"id": EOS, "content": "<|eos|>", "special": True}]
    return {"version": "1.0", "added_tokens": added, "model": {"type": "BPE", "vocab": vocab, "merges": []}}


def flatten(prefix, tr, out):
    for k, v in tr.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            flatten(key, v, out)
        elif isinstance(v, list):
            continue                                    # indexer_scores: see indexer_selected_mask
        elif torch.is_tensor(v):
            a = v.detach().cpu()
            if a.dtype == torch.bool:
                out[key] = ("U8", a.numpy().astype(np.uint8))
            elif a.dtype in (torch.int64, torch.int32):
                out[key] = ("I64", a.numpy().astype(np.int64))
            else:
                out[key] = ("F32", a.float().numpy())
    return out


def margins(tr, acc):
    """Smallest gaps that decide discrete choices: router top-k vs the
    next expert (softmax probability), and QSA selected vs unselected
    block score."""
    for k, v in tr.items():
        if isinstance(v, dict):
            margins(v, acc)
        elif k == "router_logits":
            p = torch.softmax(v.float(), -1).sort(-1, descending=True).values
            topk = acc["top_k"]
            acc["router"] = min(acc["router"], float((p[:, topk - 1] - p[:, topk]).min()))
        elif k == "indexer_scores":
            for s in v:
                if s.numel() > acc["block_topk"]:
                    ss = s.sort(descending=True).values
                    acc["indexer"] = min(acc["indexer"], float(ss[acc["block_topk"] - 1] - ss[acc["block_topk"]]))
    return acc


PROMPTS = [
    # 20 tokens: past the 8-token QSA budget, so blocks are selected.
    [17, 512, 3, 999, 64, 64, 128, 700, 5, 6, 901, 33, 250, 251, 12, 800, 77, 431, 2, 1000],
    # EOS inside: the n-gram history restarts after it.
    [40, 41, EOS, 42, 43, 44, 45],
    # One token: shorter than the n-gram context (EOS padding).
    [321],
    # 13 tokens.
    [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
]
DECODE = 6


def run_goldens(model, variant, goldens, meta, tc):
    acc = {"router": 1.0, "indexer": float("inf"), "top_k": tc["num_experts_per_tok"],
           "block_topk": tc["indexer_budget"] // tc["indexer_compress_ratio"]}
    worst = 0.0
    for i, p in enumerate(PROMPTS):
        ids = torch.tensor([p], dtype=torch.long)
        pl, toks, dl = model.generate_greedy(ids, DECODE)
        goldens[f"{variant}.p{i}.input_ids"] = ("I64", np.array(p, np.int64))
        goldens[f"{variant}.p{i}.prefill_logits"] = ("F32", pl.numpy())
        goldens[f"{variant}.p{i}.decode_tokens"] = ("I64", np.array(toks, np.int64))
        goldens[f"{variant}.p{i}.decode_logits"] = ("F32", dl.numpy())
        # Self-check: the cached decode equals a full recompute without cache.
        full = torch.tensor([p + toks], dtype=torch.long)
        tr = {}
        fl, _ = model(full, trace=tr)
        margins(tr, acc)
        ref = torch.cat([pl, dl], 0)
        d = float((fl[0] - ref).abs().max())
        worst = max(worst, d)
        top2 = ref[len(p) - 1:].topk(2, dim=-1).values
        acc["argmax"] = min(acc.get("argmax", float("inf")), float((top2[:, 0] - top2[:, 1]).min()))
    meta["variants"][variant] = {"cache_vs_recompute_max_abs": worst,
                                 "min_router_prob_margin": acc["router"],
                                 "min_indexer_score_margin": acc["indexer"],
                                 "min_greedy_logit_margin": acc["argmax"]}
    if acc["router"] < 1e-4 or acc["indexer"] < 1e-3:
        print(f"warning: {variant}: a discrete choice is close to a tie: {meta['variants'][variant]}")
    return worst


def trace_file(model, path):
    out = {}
    ids = torch.tensor([PROMPTS[0]], dtype=torch.long)
    tr = {}
    logits, cache = model(ids, use_cache=True, trace=tr)
    flatten("prefill", tr, out)
    out["prefill.logits"] = ("F32", logits[0].numpy())
    nxt = int(logits[0, -1].argmax())
    tr = {}
    lg, _ = model(torch.tensor([[nxt]]), past_key_values=cache, use_cache=True, trace=tr)
    flatten("decode0", tr, out)
    out["decode0.input_ids"] = ("I64", np.array([nxt], np.int64))
    out["decode0.logits"] = ("F32", lg[0].numpy())
    mlx_io.write_safetensors(path, out)
    return len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="tiny")
    ap.add_argument("--seed", type=int, default=2)  # 2: widest router / QSA margins of seeds 0-3
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    root = os.path.join(os.path.expanduser(a.out), a.name)
    cfg_u, cfg_q = tiny_config(False), tiny_config(True)
    tc = schema.text_config(cfg_u)
    w = random_weights(cfg_u, a.seed)

    for sub, dt in (("bf16", "BF16"), ("f32", "F32")):
        d = os.path.join(root, sub)
        os.makedirs(d, exist_ok=True)
        json.dump(cfg_u, open(os.path.join(d, "config.json"), "w"), indent=1)
        mlx_io.write_safetensors(os.path.join(d, "model.safetensors"), unquantized_tensors(cfg_u, w, dt),
                                 metadata={"format": "mlx"})
    d4 = os.path.join(root, "mlx-4bit")
    os.makedirs(d4, exist_ok=True)
    json.dump(cfg_q, open(os.path.join(d4, "config.json"), "w"), indent=1)
    write_sharded(d4, quantized_tensors(cfg_q, w), force_last=("layers.0.mlp.switch_mlp.up_proj",))
    json.dump(tiny_tokenizer(), open(os.path.join(d4, "tokenizer.json"), "w"))

    # Reference checks that do not depend on goldens.
    emb = reference.Qwen4ExpTextNGramEmbedding(reference.Qwen4ExpTextConfig.from_dict(cfg_u), tc["ple_embed_dim"], 1, 0)
    orig_sizes = [reference._find_nth_prime_after(tc["ngram_vocab_size_base"] - 1, g + 1)
                  for g in range(emb.ngram_heads)]
    assert orig_sizes == emb.head_vocab_sizes, (orig_sizes, emb.head_vocab_sizes)

    meta = {"config": tc, "prompts": PROMPTS, "decode_steps": DECODE, "variants": {},
            "notes": "decode_logits[k] are the logits after feeding decode_tokens[k]; decode_tokens[0] is the "
                     "argmax of prefill_logits[-1]. Compute in float32; q4 = MLX weights dequantized in f32."}
    goldens = {}
    models = {"bf16": reference.load_mlx(os.path.join(root, "bf16")),
              "q4": reference.load_mlx(d4)}
    # The f32 copy must load to the same model as bf16.
    m32 = reference.load_mlx(os.path.join(root, "f32"))
    for (k, p1), (_, p2) in zip(models["bf16"].state_dict().items(), m32.state_dict().items()):
        assert torch.equal(p1, p2), k
    for v, m in models.items():
        worst = run_goldens(m, v, goldens, meta, tc)
        print(f"{v}: cache vs recompute max |diff| {worst:.2e}; margins {meta['variants'][v]}")
        n = trace_file(m, os.path.join(root, f"trace_{v}.safetensors"))
        print(f"{v}: {n} trace tensors")
    mlx_io.write_safetensors(os.path.join(root, "goldens.safetensors"), goldens)
    json.dump(meta, open(os.path.join(root, "goldens.json"), "w"), indent=1)
    q = goldens["q4.p0.prefill_logits"][1]
    b = goldens["bf16.p0.prefill_logits"][1]
    print(f"q4 vs bf16 prompt 0 logits: max |diff| {np.abs(q - b).max():.3f}, "
          f"argmax agree {(q.argmax(-1) == b.argmax(-1)).mean():.2f}")
    print(f"wrote {root}")


if __name__ == "__main__":
    main()
