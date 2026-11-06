#!/usr/bin/env python3
"""Reference for Qwen Sparse Attention (QSA), Qwen3.8-Flash-Next.

A torch port of `Qwen4ExpTextAttention` + `Qwen4ExpTextQSAIndexer`
(transformers `models/qwen4_exp/modeling_qwen4_exp.py`, main branch),
the rotary embedding with interleaved M-RoPE recomposition, the
zero-centred RMSNorm, the eager attention and a minimal cache with
`update` / `update_indexer`. The functions below are copied from the
reference with only the transformers plumbing removed, so the math is
the reference's.

Weights and inputs come from the same xorshift64 generator the Rust
test (`riir::model_layer::qwen3_8_flash_next::qsa`) uses, so the
fixture holds only the outputs. Run:

    python3 -I tools/model_layer/qwen3_8_flash_next/qsa_reference.py \
        crates/streammoe/tests/fixtures/qsa_reference

writes `<out>.bin` (f32 little-endian) and `<out>.json` (index).
`--check` only runs the dense-vs-indexed checks and prints them.
"""

import json
import math
import sys

import torch
from torch import nn

torch.set_grad_enabled(False)

# Tiny shape: the real attention shape (24 Q / 2 KV heads, head_dim 256,
# rotary 64 with M-RoPE [11, 11, 10], theta 1e7, indexer 4 x 128,
# compress ratio 4) with hidden 128 and a 16-token budget, so a 40-token
# sequence goes over it.
CFG = dict(
    hidden_size=128,
    num_attention_heads=24,
    num_key_value_heads=2,
    head_dim=256,
    partial_rotary_factor=0.25,
    rope_theta=10_000_000.0,
    mrope_section=[11, 11, 10],
    rms_norm_eps=1e-6,
    indexer_n_heads=4,
    indexer_kv_heads=1,
    indexer_head_dim=128,
    indexer_budget=16,
    indexer_compress_ratio=4,
)
SEED = 0x5153_4131  # "QSA1"
N_TOTAL = 40
N_PREFILL = 12


# ---------------------------------------------------------------- rng
class Rng:
    """xorshift64, bit-identical to the Rust test's generator."""

    M = (1 << 64) - 1

    def __init__(self, seed):
        self.s = seed & self.M or 1

    def next_u64(self):
        x = self.s
        x ^= (x << 13) & self.M
        x ^= x >> 7
        x ^= (x << 17) & self.M
        self.s = x
        return x

    def unit(self):
        # Top 24 bits -> [-1, 1); exact in f32 on both sides.
        return ((self.next_u64() >> 40) / 16777216.0) * 2.0 - 1.0

    def mat(self, rows, cols, scale):
        v = [self.unit() * scale for _ in range(rows * cols)]
        return torch.tensor(v, dtype=torch.float32).view(rows, cols)

    def norm_w(self, dim):
        # Effective (1 + w) multiplier 1 + k/64, k in [-8, 8]: exact in bf16.
        return torch.tensor([1.0 + ((self.next_u64() >> 40) % 17 - 8) / 64.0 for _ in range(dim)],
                            dtype=torch.float32)


# ------------------------------------------------- reference functions
class Cfg:
    def __init__(self, d):
        self.__dict__.update(d)
        self.rope_parameters = dict(
            rope_type="default",
            rope_theta=d["rope_theta"],
            partial_rotary_factor=d["partial_rotary_factor"],
            mrope_section=d["mrope_section"],
            mrope_interleaved=True,
        )


class Qwen4ExpTextRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = config.head_dim
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.attention_scaling = 1.0
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    def forward(self, x, position_ids):
        position_ids = position_ids.expand(3, -1, -1)
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        sin = self.recomposition_frequencies(sin)
        cos = self.recomposition_frequencies(cos)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def recomposition_frequencies(self, freq):
        freqs_thw = freq[0].clone()  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)


class Qwen4ExpTextRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = x.float()
        output = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k=None, cos=None, sin=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rope, q_nope = q[..., :rotary_dim], q[..., rotary_dim:]
    q_rope = (q_rope * cos) + (rotate_half(q_rope) * sin)
    q_rotated = torch.cat([q_rope, q_nope], dim=-1)
    if k is not None:
        k_rope, k_nope = k[..., :rotary_dim], k[..., rotary_dim:]
        k_rope = (k_rope * cos) + (rotate_half(k_rope) * sin)
        k_rotated = torch.cat([k_rope, k_nope], dim=-1)
        return q_rotated, k_rotated
    return q_rotated


class Cache:
    """DynamicCache subset: K/V concat on the sequence axis, raw indexer
    keys concat on theirs."""

    def __init__(self):
        self.k = self.v = self.idx = None

    def update(self, k, v, layer_idx):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=-2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=-2)
        return self.k, self.v

    def update_indexer(self, raw_keys, layer_idx):
        self.idx = raw_keys if self.idx is None else torch.cat([self.idx, raw_keys], dim=1)
        return self.idx


class Qwen4ExpTextQSAIndexer(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            config.hidden_size, (self.index_n_heads + self.index_kv_heads) * self.index_head_dim, bias=False
        )
        self.q_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        # Diagnostics: smallest relative gap between the k-th and (k+1)-th
        # block score over every query that dropped a block.
        self.min_gap = math.inf

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
        full_cos, full_sin = position_embeddings
        current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1
        )
        q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = self.q_layernorm(q)
        q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

        visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0

        selected_token_indices = torch.full(
            (batch_size, seq_length, self.token_budget + self.compress_ratio - 1), -1, dtype=torch.int32
        )
        for batch_idx in range(batch_size):
            for query_idx in range(seq_length):
                local_visible_indices = torch.nonzero(
                    visible_token_indices[batch_idx, 0, query_idx], as_tuple=False
                ).flatten()
                num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio
                if num_complete_blocks > 0:
                    block_token_indices = local_visible_indices[: num_complete_blocks * self.compress_ratio].view(
                        num_complete_blocks, self.compress_ratio
                    )
                    key_groups = raw_keys[batch_idx].index_select(0, block_token_indices.flatten())
                    key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
                    pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                    pooled_keys = self.k_layernorm(pooled_keys)
                    group_starts = block_token_indices[:, 0]
                    block_key_states = apply_rotary_pos_emb(
                        pooled_keys.unsqueeze(1),
                        cos=full_cos[batch_idx].index_select(0, group_starts),
                        sin=full_sin[batch_idx].index_select(0, group_starts),
                    ).squeeze(1)
                    scores = torch.matmul(
                        q[batch_idx, query_idx].float(), block_key_states.float().transpose(-1, -2)
                    ).transpose(-1, -2)
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)
                    k = min(self.block_topk, num_complete_blocks)
                    selected_block_indices = scores.topk(k, dim=0).indices
                    if num_complete_blocks > k:
                        s = scores.sort(descending=True).values
                        self.min_gap = min(self.min_gap, float((s[k - 1] - s[k]) / s[0].abs().clamp_min(1e-30)))
                    selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
                else:
                    selected_tokens = torch.tensor([])
                tail = local_visible_indices[num_complete_blocks * self.compress_ratio :]
                selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
                selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

        kv_length = attention_mask.shape[-1]
        selected_token_mask = torch.zeros((*selected_token_indices.shape[:-1], kv_length + 1), dtype=torch.bool)
        scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length).long()
        selected_token_mask = selected_token_mask.scatter(-1, scatter_indices, True)[..., :kv_length].unsqueeze(1)
        return selected_token_mask


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


class Qwen4ExpTextAttention(nn.Module):
    def __init__(self, config, layer_idx=0, use_indexer=True):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = Qwen4ExpTextQSAIndexer(config, layer_idx)
        # Not in the reference: False skips the indexer mask (dense).
        self.use_indexer = use_indexer

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
        selected_token_mask = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
        if self.use_indexer:
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask & selected_token_mask
            else:
                attention_mask = attention_mask.masked_fill(
                    ~selected_token_mask, torch.finfo(attention_mask.dtype).min
                )

        position_embeddings = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, _ = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, scaling=self.scaling
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


def causal_mask(q_len, past, dtype=torch.float32):
    """create_causal_mask's eager form: 0 where visible, min elsewhere."""
    kv = past + q_len
    i = torch.arange(q_len)[:, None] + past
    j = torch.arange(kv)[None, :]
    m = torch.zeros(q_len, kv, dtype=dtype)
    m = m.masked_fill(j > i, torch.finfo(dtype).min)
    return m[None, None]


# ------------------------------------------------------------- model
def build(cfg_d, seed):
    """Weights + inputs in the Rust test's generation order."""
    c = Cfg(cfg_d)
    r = Rng(seed)
    hd, nh, nkv = c.head_dim, c.num_attention_heads, c.num_key_value_heads
    q_dim, kv_dim = nh * hd, nkv * hd
    idx_dim = (c.indexer_n_heads + c.indexer_kv_heads) * c.indexer_head_dim
    x = r.mat(N_TOTAL, c.hidden_size, 1.0)
    w = dict(
        q_proj=r.mat(2 * q_dim, c.hidden_size, 1 / 16),
        k_proj=r.mat(kv_dim, c.hidden_size, 1 / 16),
        v_proj=r.mat(kv_dim, c.hidden_size, 1 / 16),
        o_proj=r.mat(c.hidden_size, q_dim, 1 / 64),
        q_norm=r.norm_w(hd),
        k_norm=r.norm_w(hd),
        idx_qk_proj=r.mat(idx_dim, c.hidden_size, 1 / 16),
        idx_q_norm=r.norm_w(c.indexer_head_dim),
        idx_k_norm=r.norm_w(c.indexer_head_dim),
    )
    return c, x, w


def make_attn(c, w, use_indexer):
    a = Qwen4ExpTextAttention(c, use_indexer=use_indexer)
    a.q_proj.weight.copy_(w["q_proj"])
    a.k_proj.weight.copy_(w["k_proj"])
    a.v_proj.weight.copy_(w["v_proj"])
    a.o_proj.weight.copy_(w["o_proj"])
    a.q_norm.weight.copy_(w["q_norm"] - 1.0)
    a.k_norm.weight.copy_(w["k_norm"] - 1.0)
    a.indexer.index_qk_proj.weight.copy_(w["idx_qk_proj"])
    a.indexer.q_layernorm.weight.copy_(w["idx_q_norm"] - 1.0)
    a.indexer.k_layernorm.weight.copy_(w["idx_k_norm"] - 1.0)
    return a


def positions(kind):
    """[3, N_TOTAL] (t, h, w) positions. "text": all equal. "mrope": an
    8-token 'image' (2 x 4 grid) at tokens 6..13, positions as
    get_rope_index lays them out (t fixed, h / w over the grid), text
    after it continuing from max + 1."""
    if kind == "text":
        p = torch.arange(N_TOTAL)
        return torch.stack([p, p, p])
    t, h, wd = [], [], []
    for i in range(6):
        t.append(i), h.append(i), wd.append(i)
    base = 6
    for gh in range(2):
        for gw in range(4):
            t.append(base), h.append(base + gh), wd.append(base + gw)
    nxt = base + 4
    for i in range(N_TOTAL - 14):
        t.append(nxt + i), h.append(nxt + i), wd.append(nxt + i)
    return torch.tensor([t, h, wd])


def run(c, x, w, pos3, use_indexer, chunks):
    """Feed `x` in `chunks` (list of lengths) through one attention with a
    cache; return the [N_TOTAL, hidden] outputs and the min score gap."""
    a = make_attn(c, w, use_indexer)
    rope = Qwen4ExpTextRotaryEmbedding(c)
    cache = Cache()
    outs, past = [], 0
    for n in chunks:
        # Full position history (the indexer needs it), as the model passes.
        pe = rope(x, pos3[:, None, : past + n])
        y = a(x[None, past : past + n], pe, causal_mask(n, past), cache)
        outs.append(y[0])
        past += n
    return torch.cat(outs), a.indexer.min_gap


def main():
    check_only = "--check" in sys.argv
    args = [s for s in sys.argv[1:] if not s.startswith("--")]
    c, x, w = build(CFG, SEED)
    cr, budget = c.indexer_compress_ratio, c.indexer_budget
    # Dense == QSA while every query sees <= budget + cr - 1 tokens:
    # floor(n_visible / cr) <= budget / cr, so topk keeps every block.
    exact_until = budget + cr - 1
    cases, gaps = {}, []
    for kind in ("text", "mrope"):
        p3 = positions(kind)
        for mode, use in (("indexed", True), ("dense", False)):
            for split, chunks in (("prefill", [N_TOTAL]), ("decode", [N_PREFILL] + [1] * (N_TOTAL - N_PREFILL))):
                y, gap = run(c, x, w, p3, use, chunks)
                cases[f"{kind}.{mode}.{split}"] = y
                if use:
                    gaps.append(gap)
    ok = True
    for kind in ("text", "mrope"):
        for split in ("prefill", "decode"):
            d = cases[f"{kind}.dense.{split}"]
            s = cases[f"{kind}.indexed.{split}"]
            diff = (d - s).abs().amax(dim=-1)
            under = diff[:exact_until].max().item()
            over = diff[exact_until:]
            print(f"{kind}.{split}: dense vs indexed max|diff| rows 1..{exact_until} (visible <= {exact_until}) "
                  f"= {under:.3e}; row {exact_until + 1} = {over[0].item():.3e}; rows above min {over.min().item():.3e}")
            ok &= under == 0.0 and bool((over > 1e-4).all())
        pd = (cases[f"{kind}.indexed.prefill"] - cases[f"{kind}.indexed.decode"]).abs().max().item()
        print(f"{kind}: indexed prefill vs prefill+decode max|diff| = {pd:.3e}")
    print(f"min relative top-k score gap = {min(gaps):.3e}")
    print("dense == QSA at or under budget + cr - 1, different above:", "OK" if ok else "FAILED")
    if check_only or not args:
        return 0 if ok else 1
    out = args[0]
    blob, index, off = bytearray(), {}, 0
    for name, t in cases.items():
        b = t.contiguous().numpy().astype("<f4").tobytes()
        index[name] = dict(offset=off, shape=list(t.shape))
        blob += b
        off += len(b)
    meta = dict(config=CFG, seed=SEED, n_total=N_TOTAL, n_prefill=N_PREFILL, exact_until=exact_until,
                positions={k: positions(k).tolist() for k in ("text", "mrope")}, cases=index)
    with open(out + ".bin", "wb") as f:
        f.write(blob)
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out}.bin ({len(blob)} bytes) and {out}.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
