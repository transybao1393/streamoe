#!/usr/bin/env python3
"""Standalone pure-torch reference for the Qwen3.8-Flash-Next text model.

A port of the text path of transformers' `models/qwen4_exp/
modeling_qwen4_exp.py` (main branch, transformers 5.8; copy at
~/models/qwen3-8-flash-next/reference-transformers/) that runs on
torch 2.7 without transformers. Module names, parameter names and the
math are the original's; class bodies are copied, trimmed only where
noted. What changed:

- Imports, `@use_kernel_*` / `@use_experts_implementation` hub kernel
  decorators, `GradientCheckpointingLayer`, `PreTrainedModel`, output
  recorders, generation, loss, vision and MTP are gone. Only the torch
  fallbacks of the kernels remain (they were the reference math).
- `Qwen4ExpCache` replaces transformers' `DynamicCache` with just the
  methods the model calls: per-layer conv states (state_idx 0 = Gated
  DeltaNet conv, 1 = PLE short conv, 2 = PLE n-gram token context),
  recurrent states, KV, the indexer's raw keys and the indexer's
  position history. `update_conv_state(x, layer, state_idx, K)`
  returns `cat(previous state or K zeros, x)` along the last axis and
  keeps its last K columns; `is_recording_past()` is always False.
- Masks: `create_causal_mask` becomes a boolean causal mask
  `[B, 1, q, past + q]` (the sdpa form); `create_recurrent_attention_mask`
  returns None (no padding: batches must be unpadded, and the
  `attention_mask` argument is not supported). Attention is the eager
  path; the boolean mask is turned into the additive float mask eager
  attention expects after the indexer selection is combined in. The
  indexer's scatter index is cast to int64 (torch requires it).
- `Qwen4ExpTextNGramEmbedding`: the id computation is split out as
  `ngram_ids()` (same code) and the per-head prime sizes come from
  `schema.ngram_vocab`, which walks the same primes as the original's
  per-head `_find_nth_prime_after(base - 1, g + 1)` calls.
- Rotary: only `rope_type == "default"`; `@dynamic_rope_update` is a
  no-op for it and is dropped.
- `config` is a plain namespace built from config.json
  (`Qwen4ExpTextConfig.from_dict`) with the transformers defaults
  (`seed=1234`, `norm_topk_prob=True`, `split_ngram_parts=512`) and
  the `full_attention` -> `indexed_attention` rename.
- Tracing: `model(..., trace={})` fills the dict with per-layer
  intermediates (see `Qwen4ExpForCausalLM.forward`). Tracing only
  copies tensors; it changes no math.
- Loaders: `load_mlx(dir)` reads an MLX checkpoint (quantized or not;
  quantized tensors are dequantized in float32 as `q * scale + bias`)
  and maps MLX names to transformers names (see schema.py).

Run in float32 for goldens (`load_mlx(..., dtype=torch.float32)`): the
original's dtype casts are then no-ops.

    python3 reference.py <mlx dir> --prompt 1,2,3 --decode 4
"""

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx_io  # noqa: E402
import schema  # noqa: E402

ACT2FN = {"silu": F.silu, "sigmoid": torch.sigmoid}


# --- config --------------------------------------------------------------------

class Qwen4ExpTextConfig(SimpleNamespace):
    @classmethod
    def from_dict(cls, cfg):
        tc = dict(cfg.get("text_config", cfg))
        tc.setdefault("seed", 1234)
        tc.setdefault("norm_topk_prob", True)
        tc.setdefault("split_ngram_parts", 512)
        tc.setdefault("make_ngram_vocab_size_divisible_by", 128)
        tc.setdefault("attention_dropout", 0.0)
        tc.setdefault("attention_bias", False)
        tc.setdefault("pad_token_id", None)
        tc["ple_layer_ids"] = [] if tc.get("ple_layer_ids") is None else sorted(set(tc["ple_layer_ids"]))
        if tc.get("ple_embed_dim") is None:
            tc["ple_embed_dim"] = tc["hidden_size"]
        if tc.get("layer_types") is None:
            k = tc.get("full_attention_interval", 4)
            tc["layer_types"] = ["linear_attention" if (i + 1) % k else "indexed_attention"
                                 for i in range(tc["num_hidden_layers"])]
        elif "full_attention" in tc["layer_types"]:
            tc["layer_types"] = ["indexed_attention" if t == "full_attention" else t for t in tc["layer_types"]]
        rp = tc.get("rope_parameters") or {}
        rp.setdefault("rope_type", "default")
        tc["rope_parameters"] = rp
        if rp["rope_type"] != "default":
            raise NotImplementedError("only default rope")
        c = cls(**tc)
        c._attn_implementation = "eager"
        return c


# --- cache -----------------------------------------------------------------------

class Qwen4ExpCache:
    """Minimal stand-in for transformers' DynamicCache (see module doc)."""

    def __init__(self, config):
        n = config.num_hidden_layers
        self.conv_states = [dict() for _ in range(n)]
        self.recurrent_states = [None] * n
        self.key_cache = [None] * n
        self.value_cache = [None] * n
        self.indexer_keys = [None] * n
        self.indexer_position_ids = None
        self.seen_tokens = 0

    def get_seq_length(self):
        return self.seen_tokens

    def is_recording_past(self, layer_idx):
        return False

    def get_conv_state(self, layer_idx, state_idx=0):
        return self.conv_states[layer_idx].get(state_idx)

    def update_conv_state(self, states, layer_idx, state_idx=0, conv_kernel_size=None):
        prev = self.conv_states[layer_idx].get(state_idx)
        if prev is None:
            prev = states.new_zeros(*states.shape[:-1], conv_kernel_size)
        full = torch.cat([prev.to(states.dtype), states], dim=-1)
        self.conv_states[layer_idx][state_idx] = full[..., -conv_kernel_size:].clone()
        return full

    def get_recurrent_state(self, layer_idx, state_idx=0):
        return self.recurrent_states[layer_idx]

    def update_recurrent_state(self, state, layer_idx):
        self.recurrent_states[layer_idx] = state

    def update(self, key_states, value_states, layer_idx):
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx], self.value_cache[layer_idx] = key_states, value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_indexer(self, raw_keys, layer_idx):
        if self.indexer_keys[layer_idx] is None:
            self.indexer_keys[layer_idx] = raw_keys
        else:
            self.indexer_keys[layer_idx] = torch.cat([self.indexer_keys[layer_idx], raw_keys], dim=1)
        return self.indexer_keys[layer_idx]


def _trace(trace, key, value):
    if trace is not None:
        trace[key] = value.detach().clone() if torch.is_tensor(value) else value


# --- modules (copied from modeling_qwen4_exp.py) --------------------------------------

class Qwen4ExpTextRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = self.config.rope_parameters["rope_type"]
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(self.config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(config, device=None, **kwargs):
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        position_ids = position_ids.expand(3, -1, -1)
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        sin = self.recomposition_frequencies(sin)
        cos = self.recomposition_frequencies(cos)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def recomposition_frequencies(self, freq):
        freqs_thw = freq[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)


class Qwen4ExpTextRMSNorm(nn.Module):
    def __init__(self, dim, group_size=None, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")

    def _norm(self, x):
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen4ExpTextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, activation="silu"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(hidden_states.to(weight.dtype), weight=weight.unsqueeze(1), bias=bias,
                   padding=padding, groups=hidden_size)[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                                 output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
    decay = g
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    scaling = query.shape[-1] ** -0.5
    query = query * scaling
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                                     use_qk_l2norm_in_kernel=False, **kwargs):
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query / (query.shape[-1] ** 0.5)
    if initial_state is None:
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen4ExpTextGatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
                                kernel_size=self.conv_kernel_size, groups=self.conv_dim,
                                padding=self.conv_kernel_size - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = Qwen4ExpTextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon,
                                             activation=config.output_gate_type or config.hidden_act)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.layer_type = config.layer_types[layer_idx]
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, trace=None, **kwargs):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        if cache_params is not None:
            conv_state = cache_params.get_conv_state(self.layer_idx, state_idx=0)
            recurrent_state = cache_params.get_recurrent_state(self.layer_idx, state_idx=0)
        else:
            conv_state, recurrent_state = None, None
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        if conv_state is not None and seq_len == 1 and not cache_params.is_recording_past(self.layer_idx):
            mixed_qkv = causal_conv1d_update(mixed_qkv, conv_state, self.conv1d.weight.squeeze(1),
                                             self.conv1d.bias, self.activation)
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(mixed_qkv, self.layer_idx,
                                                           conv_kernel_size=self.conv_kernel_size)
            mixed_qkv = causal_conv1d_fn(mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias,
                                         activation=self.activation)
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]
        mixed_qkv = mixed_qkv.transpose(1, 2)
        _trace(trace, "conv_out", mixed_qkv)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        _trace(trace, "beta", beta)
        _trace(trace, "g", g)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        if recurrent_state is not None and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True)
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True)
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
        _trace(trace, "core_attn_out", core_attn_out)
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)
        return output


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
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


class Qwen4ExpTextQSAIndexer(nn.Module):
    """Select QSA token indices from compressed key blocks."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(config.hidden_size,
                                       (self.index_n_heads + self.index_kv_heads) * self.index_head_dim, bias=False)
        self.q_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, trace=None):
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
        full_cos, full_sin = position_embeddings
        current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]
        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1)
        q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = self.q_layernorm(q)
        q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)
        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
        visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        selected_token_indices = torch.full(
            (batch_size, seq_length, self.token_budget + self.compress_ratio - 1), -1,
            dtype=torch.int32, device=hidden_states.device)
        scores_trace = []
        for batch_idx in range(batch_size):
            for query_idx in range(seq_length):
                local_visible_indices = torch.nonzero(
                    visible_token_indices[batch_idx, 0, query_idx], as_tuple=False).flatten()
                num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio
                if num_complete_blocks > 0:
                    block_token_indices = local_visible_indices[: num_complete_blocks * self.compress_ratio].view(
                        num_complete_blocks, self.compress_ratio)
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
                    if trace is not None:
                        scores_trace.append(scores.detach().clone())
                    selected_block_indices = scores.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
                    selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
                else:
                    selected_tokens = torch.tensor([], device=hidden_states.device)
                    if trace is not None:
                        scores_trace.append(torch.zeros(0))
                tail = local_visible_indices[num_complete_blocks * self.compress_ratio:]
                selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
                selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens
        kv_length = attention_mask.shape[-1]
        selected_token_mask = torch.zeros(
            (*selected_token_indices.shape[:-1], kv_length + 1), device=attention_mask.device, dtype=torch.bool)
        scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
        selected_token_mask = selected_token_mask.scatter(-1, scatter_indices.long(), True)[..., :kv_length].unsqueeze(1)
        if trace is not None:
            trace["indexer_q"] = q.detach().clone()
            trace["indexer_raw_keys"] = raw_keys.detach().clone()
            trace["indexer_scores"] = scores_trace          # list per (batch, query) of [num_blocks]
            trace["indexer_selected_mask"] = selected_token_mask.detach().clone()
        return selected_token_mask


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Qwen4ExpTextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size,
                                bias=config.attention_bias)
        self.q_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = Qwen4ExpTextQSAIndexer(config, layer_idx)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, trace=None,
                **kwargs):
        selected_token_mask = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values,
                                           trace=trace)
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask & selected_token_mask
        else:
            attention_mask = attention_mask.masked_fill(~selected_token_mask, torch.finfo(attention_mask.dtype).min)
        position_embeddings = (x[:, -hidden_states.shape[1]:, :] for x in position_embeddings)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2,
                                         dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        # Port: eager attention wants an additive float mask; the combined
        # boolean mask (sdpa form) is converted here.
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.zeros(attention_mask.shape, dtype=query_states.dtype,
                                         device=query_states.device).masked_fill(
                ~attention_mask, torch.finfo(query_states.dtype).min)
        _trace(trace, "q", query_states)
        _trace(trace, "k", key_states)
        _trace(trace, "v", value_states)
        attn_output, attn_weights = eager_attention_forward(self, query_states, key_states, value_states,
                                                            attention_mask, dropout=0.0, scaling=self.scaling)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        _trace(trace, "sdpa_out", attn_output)
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen4ExpTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen4ExpTextExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class Qwen4ExpTextTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class Qwen4ExpTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen4ExpTextTopKRouter(config)
        self.experts = Qwen4ExpTextExperts(config)
        self.shared_expert = Qwen4ExpTextMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states, trace=None):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        _trace(trace, "router_logits", router_logits)
        _trace(trace, "router_weights", routing_weights)
        _trace(trace, "router_indices", selected_experts)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        _trace(trace, "routed_out", expert_output)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        _trace(trace, "shared_out", shared_expert_output)
        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


class Qwen4ExpTextGatedResidual(nn.Module):
    def __init__(self, config, use_combine=True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = nn.Linear(hc_hidden_size, self.hc_count, bias=False) if use_combine else None

    def forward(self, hyper_input):
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                f"Expected {self.hc_count * self.hidden_size} hyper-connection features, got {hyper_input.shape[-1]}.")
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (input_mix_weight * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(
            dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
        return mixed_input, hyper_input, injection_weights


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value):
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed):
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


_is_prime = schema.is_prime
_find_nth_prime_after = schema.find_nth_prime_after


class Qwen4ExpTextNGramEmbedding(nn.Module):
    def __init__(self, config, embedding_dim, layer_idx, ple_layer_index=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        head_dim_per_ngram = embedding_dim // self.ngram_heads
        self.seed = config.seed
        self.eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
        self.head_vocab_sizes = []
        self.head_offsets = []
        self.total_vocab_size = 0
        # Port: the original calls _find_nth_prime_after(base - 1, g + 1)
        # per head; schema.ngram_vocab walks the same primes once.
        sizes, offsets, total, padded_vocab_size = schema.ngram_vocab(vars(config), self.ple_layer_index)
        self.head_vocab_sizes, self.head_offsets, self.total_vocab_size = sizes, offsets, total
        self.register_buffer("layer_multipliers", _build_layer_multipliers(
            self.unigram_vocab_size, self.ngram_size, self.ple_layer_index, self.seed))
        self.register_buffer("ngram_heads_vocab_sizes", torch.tensor(self.head_vocab_sizes, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets", torch.tensor(self.head_offsets, dtype=torch.long))
        self.ngram_embedding = nn.Embedding(padded_vocab_size, head_dim_per_ngram)

    def _shift_right_ignore_eos(self, token_ids, shift):
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def ngram_ids(self, input_ids, past_key_values):
        """Port: the id computation of forward(), split out so tests can
        compare row ids directly. Same code as the original."""
        input_ids = input_ids.long()
        cached_ctx = (
            past_key_values.get_conv_state(self.layer_idx, state_idx=2) if past_key_values is not None else None
        )
        if cached_ctx is not None:
            previous_context = cached_ctx.clone()
        else:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        if past_key_values is not None:
            input_ids_to_cache = input_ids
            if cached_ctx is None and input_ids.shape[1] < self.context_len:
                input_ids_to_cache = torch.nn.functional.pad(
                    input_ids_to_cache, (self.context_len - input_ids.shape[1], 0), value=self.eos_token_id)
            _ = past_key_values.update_conv_state(
                input_ids_to_cache, self.layer_idx, state_idx=2, conv_kernel_size=self.context_len)
        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(mixed_ids, shifted_tokens[position] * self.layer_multipliers[position])
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))
        return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]

    def forward(self, input_ids, past_key_values, trace=None):
        ngram_ids = self.ngram_ids(input_ids, past_key_values)
        _trace(trace, "ngram_ids", ngram_ids)
        return self.ngram_embedding(ngram_ids).flatten(-2)


class Qwen4ExpTextPLELayer(nn.Module):
    def __init__(self, config, layer_idx, ple_layer_index):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        ple_embed_dim = config.ple_embed_dim
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpTextNGramEmbedding(config, ple_embed_dim, layer_idx, ple_layer_index)
        conv_kernel_size = config.ple_conv_kernel_size
        conv_dilation = config.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
        self.key_proj = nn.Linear(ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_query = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_conv = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(hc_hidden_size, hc_hidden_size, kernel_size=conv_kernel_size, groups=hc_hidden_size,
                                dilation=conv_dilation, bias=False)

    def _short_conv(self, hidden_states, past_key_values):
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)
        if past_key_values is not None:
            hidden_states = past_key_values.update_conv_state(
                hidden_states, self.layer_idx, state_idx=1, conv_kernel_size=self.short_conv_state_len)
        hidden_states = F.pad(hidden_states, (self.short_conv_state_len, 0))
        hidden_states = hidden_states[..., -(self.short_conv_state_len + seq_len):]
        hidden_states = F.silu(self.conv1d(hidden_states))
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

    def forward(self, hidden_states, input_ids, past_key_values, conv_mask=None, trace=None):
        embeddings = self.ple_embedding(input_ids, past_key_values, trace=trace)
        _trace(trace, "embeddings", embeddings)
        key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        _trace(trace, "gate", gate)
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        if conv_mask is not None:
            gated_value = apply_mask_to_padding_states(gated_value, conv_mask)
            gated_value_normed = apply_mask_to_padding_states(gated_value_normed, conv_mask)
        output = gated_value + self._short_conv(gated_value_normed, past_key_values)
        return output


class Qwen4ExpTextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpTextGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen4ExpTextAttention(config, layer_idx)
        self.mlp = Qwen4ExpTextSparseMoeBlock(config)
        ple_layer_index = config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in config.ple_layer_ids else None
        self.ple = Qwen4ExpTextPLELayer(config, layer_idx, ple_layer_index) if ple_layer_index is not None else None
        self.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, conv_mask=None, past_key_values=None,
                ple_input_ids=None, trace=None, **kwargs):
        sub = (lambda name: trace.setdefault(name, {})) if trace is not None else (lambda name: None)
        _trace(trace, "input", hidden_states)
        if self.ple is not None:
            ple_out = self.ple(hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask, trace=sub("ple"))
            _trace(trace, "ple_out", ple_out)
            hidden_states = hidden_states + ple_out
            _trace(trace, "after_ple", hidden_states)
        hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
        _trace(trace, "attn_mixed_input", hidden_states)
        _trace(trace, "attn_injection_weights", injection_weights)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, cache_params=past_key_values, attention_mask=conv_mask,
                                             trace=sub("linear_attn"))
        else:
            hidden_states, _ = self.self_attn(hidden_states, position_embeddings, attention_mask=attention_mask,
                                              past_key_values=past_key_values, trace=sub("self_attn"))
        _trace(trace, "attn_out", hidden_states)
        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)
        _trace(trace, "after_attn", hidden_states)
        hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        _trace(trace, "mlp_mixed_input", hidden_states)
        _trace(trace, "mlp_injection_weights", injection_weights)
        hidden_states = self.mlp(hidden_states, trace=sub("mlp"))
        _trace(trace, "mlp_out", hidden_states)
        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)
        _trace(trace, "output", hidden_states)
        return hidden_states


def create_causal_mask(batch_size, query_length, past_length, device):
    """Port of masking_utils.create_causal_mask for unpadded input:
    [B, 1, q, past + q] booleans, True = visible."""
    q = torch.arange(query_length, device=device)[:, None] + past_length
    k = torch.arange(past_length + query_length, device=device)[None, :]
    return (k <= q)[None, None].expand(batch_size, 1, query_length, past_length + query_length)


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen4ExpTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(config=config)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)

    def forward(self, input_ids, position_ids=None, past_key_values=None, use_cache=False, trace=None):
        inputs_embeds = self.embed_tokens(input_ids)
        ple_input_ids = input_ids if self.config.ple_layer_ids else None
        if use_cache and past_key_values is None:
            past_key_values = Qwen4ExpCache(self.config)
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        elif position_ids.shape[0] == 1:
            text_position_ids = position_ids[0]
            position_ids = position_ids.expand(3, -1, -1)
        else:
            text_position_ids = None
        if past_key_values is not None and "indexed_attention" in self.config.layer_types:
            if past_key_values.indexer_position_ids is not None:
                position_ids = torch.cat([past_key_values.indexer_position_ids, position_ids], dim=-1)
            past_key_values.indexer_position_ids = position_ids
        causal_mask = create_causal_mask(inputs_embeds.shape[0], inputs_embeds.shape[1], past_seen_tokens,
                                         inputs_embeds.device)
        conv_mask = None  # create_recurrent_attention_mask: None without padding
        hidden_states = inputs_embeds
        _trace(trace, "embed", hidden_states)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        _trace(trace, "rope_cos", position_embeddings[0])
        _trace(trace, "rope_sin", position_embeddings[1])
        hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states, position_embeddings=position_embeddings, attention_mask=causal_mask,
                conv_mask=conv_mask, past_key_values=past_key_values, ple_input_ids=ple_input_ids,
                trace=trace.setdefault(f"layers.{layer_idx}", {}) if trace is not None else None)
        hidden_states = self.hyper_connection_mixer(hidden_states)
        _trace(trace, "final_hidden", hidden_states)
        if past_key_values is not None:
            past_key_values.seen_tokens += inputs_embeds.shape[1]
        return hidden_states, past_key_values


class Qwen4ExpForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen4ExpTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @torch.no_grad()
    def forward(self, input_ids, past_key_values=None, use_cache=False, trace=None):
        """input_ids [B, T] (int64). Returns (logits [B, T, V] float, cache).

        trace (a dict) collects: embed, rope_cos/sin, final_hidden and
        per layer `layers.{i}` -> input, (ple_out, after_ple, ple ->
        {ngram_ids, embeddings, gate}), attn_mixed_input,
        attn_injection_weights, linear_attn -> {conv_out, beta, g,
        core_attn_out} or self_attn -> {indexer_q, indexer_raw_keys,
        indexer_scores, indexer_selected_mask, q, k, v, sdpa_out},
        attn_out, after_attn, mlp_mixed_input, mlp_injection_weights,
        mlp -> {router_logits, router_weights, router_indices,
        routed_out, shared_out}, mlp_out, output (the 4H stream)."""
        hidden_states, cache = self.model(input_ids, past_key_values=past_key_values, use_cache=use_cache,
                                          trace=trace)
        logits = self.lm_head(hidden_states)
        return logits.float(), cache

    def generate_greedy(self, input_ids, steps, trace_steps=False):
        """Prefill then `steps` greedy decode steps with the cache.
        Returns (prefill logits [T, V], decode tokens [steps],
        decode logits [steps, V] (logits that produced each token after
        the first: row 0 is the prefill's last row's successor step))."""
        logits, cache = self(input_ids, use_cache=True)
        nxt = int(logits[0, -1].argmax())
        toks, step_logits = [], []
        for _ in range(steps):
            toks.append(nxt)
            lg, cache = self(torch.tensor([[nxt]], dtype=torch.long), past_key_values=cache, use_cache=True)
            step_logits.append(lg[0, -1])
            nxt = int(lg[0, -1].argmax())
        return logits[0], toks, torch.stack(step_logits) if step_logits else torch.zeros(0, self.vocab_size)


# --- loading -------------------------------------------------------------------

def mlx_to_hf_state_dict(cfg, get, dtype=torch.float32):
    """Build a transformers-named state dict from MLX tensors.
    `get(name)` returns the NumPy array of an MLX tensor (BF16 as uint16)
    or raises KeyError. Quantized tensors are dequantized in float32."""
    sd = {}
    tc = schema.text_config(cfg)
    for t in schema.tensors(cfg):
        if t.kind == "q":
            bits, g = schema.quant_rule(cfg, t.base)
            try:
                w = mlx_io.affine_dequantize(get(t.base + ".weight"), get(t.base + ".scales"),
                                             get(t.base + ".biases"), g, bits)
            except KeyError:
                w = get(t.base + ".weight")       # unquantized checkpoint
                w = mlx_io.bf16_to_f32(w) if w.dtype == np.uint16 else w.astype(np.float32)
        else:
            w = get(t.base)
            if t.kind == "bf16":
                w = mlx_io.bf16_to_f32(w) if w.dtype == np.uint16 else w.astype(np.float32)
        name = t.base[len(schema.PREFIX):]
        if name.endswith("conv1d.weight"):
            w = np.ascontiguousarray(np.swapaxes(w, 1, 2))          # [C, K, 1] -> [C, 1, K]
        if t.kind == "q":
            name += ".weight"
        sd[name] = w
    out = {n: w for n, w in sd.items() if ".mlp.switch_mlp." not in n and ".ngram_embedding.shards." not in n}
    for i in range(tc["num_hidden_layers"]):
        p = f"model.layers.{i}.mlp."
        gate, up, down = (sd[p + f"switch_mlp.{n}.weight"] for n in ("gate_proj", "up_proj", "down_proj"))
        out[p + "experts.gate_up_proj"] = np.concatenate([gate, up], axis=1)
        out[p + "experts.down_proj"] = down
    for li in schema.ple_layers(tc):
        p = f"model.layers.{li}.ple.ple_embedding."
        parts = [sd[p + f"ngram_embedding.shards.{s}.weight"] for s in range(tc["split_ngram_parts"])]
        out[p + "ngram_embedding.weight"] = np.concatenate(parts, axis=0)
    return {k: (torch.from_numpy(np.ascontiguousarray(v)).to(dtype) if v.dtype != np.int64
                else torch.from_numpy(v)) for k, v in out.items()}


def mlx_getter(model_dir):
    """get(name) over an MLX directory's safetensors files."""
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    where = {}
    opened = [mlx_io.SafeFile(os.path.join(model_dir, f)) for f in files]
    for sf in opened:
        for n in sf.names():
            where[n] = sf

    def get(name):
        return where[name].array(name)
    return get


def build_model(cfg, state_dict, dtype=torch.float32):
    config = Qwen4ExpTextConfig.from_dict(cfg)
    model = Qwen4ExpForCausalLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Buffers (n-gram multipliers / offsets / sizes) are recomputed by the
    # module; compare them with the checkpoint's copy instead of loading.
    for li in schema.ple_layers(schema.text_config(cfg)):
        emb = model.model.layers[li].ple.ple_embedding
        for b in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
            key = f"model.layers.{li}.ple.ple_embedding.{b}"
            if key in state_dict and not torch.equal(getattr(emb, b), state_dict[key]):
                raise ValueError(f"{key}: checkpoint {state_dict[key].tolist()} != computed "
                                 f"{getattr(emb, b).tolist()}")
    missing = [k for k in missing if not k.endswith(("inv_freq",))]
    if missing or unexpected:
        raise ValueError(f"state dict mismatch: missing {missing[:8]}, unexpected {unexpected[:8]}")
    return model.to(dtype).eval()


def load_mlx(model_dir, dtype=torch.float32):
    """Load an MLX-layout checkpoint (quantized or BF16/F32) into the reference."""
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    sd = mlx_to_hf_state_dict(cfg, mlx_getter(model_dir), dtype)
    return build_model(cfg, sd, dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--prompt", default="1,2,3,4")
    ap.add_argument("--decode", type=int, default=4)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    model = load_mlx(a.model_dir)
    ids = torch.tensor([[int(x) for x in a.prompt.split(",")]], dtype=torch.long)
    pl, toks, _ = model.generate_greedy(ids, a.decode)
    print("prefill argmax:", pl.argmax(-1).tolist())
    print("greedy:", toks)


if __name__ == "__main__":
    main()
