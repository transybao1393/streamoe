# Model variants

streammoe compiles for one model shape at a time. Model shape is
*compile-time*, not runtime: the per-variant constants are folded into
the binary, and Metal shader dispatch dims are simpler to reason about
that way. Runtime model parameterization is future work — for now,
one build per target model.

## How selection works

Select a variant with exactly one Cargo feature:

```bash
cargo build --features model-qwen3-5-a17b      # original target
cargo build --features model-qwen3-6-35b-a3b   # smaller test target
cargo build --features model-cogito-v2-671b    # DeepSeek-V3 architecture
cargo build --features model-olmoe-1b-7b       # OLMoE, small all-full-attn MoE
```

The feature gates the shape constants in `src/riir/variants.rs`.
Selecting no feature, or more than one, is a build error.

## Supported variants

| Variant                | Feature                 | Source                                        |
| ---------------------- | ----------------------- | --------------------------------------------- |
| Qwen3.5-397B-A17B-4bit | `model-qwen3-5-a17b`    | `mlx-community/Qwen3.5-397B-A17B-4bit`        |
| Qwen3.6-35B-A3B-4bit   | `model-qwen3-6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` (convert to MLX 4-bit) |
| Cogito-V2-Preview-671B | `model-cogito-v2-671b`  | DeepSeek-V3 architecture                      |
| OLMoE-1B-7B-0125-Instruct-4bit | `model-olmoe-1b-7b` | `allenai/OLMoE-1B-7B-0125-Instruct` (MLX 4-bit) |

### Shape table — Qwen variants

The Qwen variants are the `qwen3_5_moe` architecture family:
attention-output-gate, mixed linear-attention + full-attention (3:1,
`FULL_ATTN_INTERVAL=4`), MTP head, Qwen tokenizer (vocab 248320).

| Parameter             | A17B    | 35B-A3B |
| --------------------- | ------- | ------- |
| `HIDDEN_DIM`          | 4096    | 2048    |
| `NUM_LAYERS`          | 60      | 40      |
| `NUM_ATTN_HEADS`      | 32      | 16      |
| `NUM_KV_HEADS`        | 2       | 2       |
| `HEAD_DIM`            | 256     | 256     |
| `NUM_EXPERTS`         | 512     | 256     |
| `NUM_EXPERTS_PER_TOK` | 10      | 8       |
| `MOE_INTERMEDIATE`    | 1024    | 512     |
| `SHARED_INTERMEDIATE` | 1024    | 512     |
| `LINEAR_NUM_V_HEADS`  | 64      | 32      |
| `LINEAR_NUM_K_HEADS`  | 16      | 16      |
| `EXPERT_SIZE` (bytes) | 7077888 | 1769472 |

`EXPERT_SIZE` and the `*_OFF` byte offsets are *derived* from
`HIDDEN_DIM`, `MOE_INTERMEDIATE`, `GROUP_SIZE`, and `BITS` — do not
hand-tune them.

Cogito-V2-671B is a separate (DeepSeek-V3) architecture — MLA
attention, not the qwen3_5_moe linear/full mix. Its shape lives in
`variants.rs` under the same feature mechanism.

### OLMoE-1B-7B

`olmoe` architecture: every layer is plain GQA full attention
(`FULL_ATTN_INTERVAL=1`, no linear attention, no MTP) followed by a
routed MoE with no shared expert. It differs from the Qwen GQA path
in four `Variant` flags rather than new layer kinds:

| Flag / constant     | Qwen3.x            | OLMoE                                   |
| ------------------- | ------------------ | --------------------------------------- |
| `attn_output_gate`  | `true` (2× q_proj) | `false` — q_proj is `[q_dim, hidden]`   |
| `qk_norm`           | `PerHead`          | `Full` — one RMSNorm over all of q / k  |
| `gqa_rotary_dim`    | 64 (partial)       | 128 (full head_dim)                     |
| `norm_topk_prob`    | `true`             | `false` — top-k softmax probs as-is     |
| `shared_expert_gate`| `SigmoidGate`      | `None` (no shared expert loaded)        |
| `RMS_NORM_EPS`      | 1e-6               | 1e-5                                    |
| `ROPE_THETA`        | 1e7                | 1e4                                     |

| Parameter             | OLMoE-1B-7B |
| --------------------- | ----------- |
| `HIDDEN_DIM`          | 2048        |
| `NUM_LAYERS`          | 16          |
| `NUM_ATTN_HEADS`      | 16          |
| `NUM_KV_HEADS`        | 16          |
| `HEAD_DIM`            | 128         |
| `NUM_EXPERTS`         | 64          |
| `NUM_EXPERTS_PER_TOK` | 8           |
| `MOE_INTERMEDIATE`    | 1024        |
| `VOCAB_SIZE`          | 50304       |
| `EXPERT_SIZE` (bytes) | 3538944     |

head_dim 128 uses the `attn_sdpa_causal_flash_va_hd128` instance of
the batched flash SDPA kernel (unfolded only; heads_per_kv is 1).

## Adding a new variant

1. Pull the shape constants from the model's HuggingFace
   `config.json`.
2. Confirm the architecture. The `qwen3_5_moe` family drops in by
   constants alone; a different architecture (e.g. plain `qwen3_moe`
   without SSM, or a new MLA model) needs kernel work.
3. Add a `model-your-name` feature in `crates/streammoe/Cargo.toml` and
   a matching shape block in `src/riir/variants.rs`.
4. Run the model-prep pipeline in `scripts/` against your
   MLX-converted weights.
5. Build with the new feature and run the oracle suites against it.
