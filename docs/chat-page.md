# Chat page

`tools/chat` is a local chat page for trying Qwen3.6-35B-A3B under a
memory cap (standalone crate; the engine gains no dependencies). Setup
of the model files is in the [README](../README.md#setup).

```bash
cd tools/chat
cargo run --release -- --cap-gib 4 --n-ctx 8192
# options: --kv f32|f16|turbo4, --attn-threads N, --guess cross|prev, --prefetch N,
#          --precision 4|3 (3 needs $M/root-3bit, see Setup)
# open http://127.0.0.1:8077
```

It listens on 127.0.0.1 only and keeps one conversation in the KV
cache, so each message prefills only the new text. The header shows
memory against the cap and the context used; each answer shows tok/s,
prefill speed, hot-tier hits and disk MiB per token. Thinking can be
toggled. An **Experts** selector in the header switches between 4-bit
and packed 3-bit experts (the server restarts itself in ~2 s and frees
the old model first, so memory stays under the cap). Its defaults are
the [standard setup](benchmarks.md#test-setup) with 4-bit experts:
10.2 tok/s over the three test prompts at a 4 GiB cap. Stop the server
(Ctrl-C) to give the memory back.

A bar under the header shows the process's CPU (as cores used), its
memory against the cap with the peak, and the host's reclaimable
memory, sampled every second; it keeps updating while an answer is
generated. Each answer also reports its average CPU.

## What it looks like

*Setup: standard, 4-bit experts, 4 GiB cap, 2026-10-28 build, recorded
with headless Chrome on a shared M3 Pro with other apps running.* Real
time — the GIFs are not sped up.

**Text** — 8.9 tok/s, 3.65 GiB of the 4 GiB cap:

<p align="center"><img src="media/chat-text.gif" width="800" alt="tools/chat answering a text question at 8.9 tok/s under a 4 GiB cap"></p>

```text
58 tok · 8.9 tok/s · prompt 39 tok at 18 tok/s · hot-tier hits 62% · 238 MiB/tok from disk · avg CPU 48%
```

**Image** — 280 image tokens encoded in 1.7 s, the 309-token prompt
read in 6.4 s, then 9.6 tok/s at a 3.68 GiB peak. Each stage shows in
the page as it happens:

| 1 · Vision encoder running | 2 · Answer streaming | 3 · Done, with the turn's stats |
| :---: | :---: | :---: |
| ![The page shows "looking at the image…" while the CPU vision encoder runs](media/chat-image-encoding.png) | ![The answer streams in; the CPU / RAM bar updates live](media/chat-image-generating.png) | ![The finished answer: three red dots, with tok/s, hot-tier hits and disk reads per token](media/chat-image-done.png) |

```text
162 tok · 9.6 tok/s · prompt 309 tok (280 image) at 49 tok/s · hot-tier hits 63% · 232 MiB/tok from disk · avg CPU 62%
```

The image is a drawing made for this demo (house, tree, sun, one bird
and three red dots). The model names the house, tree and sun, counts
the three dots correctly, and reads the single bird as two. On the
2026-10-26 build the same two turns ran at 6.7 and 5.9 tok/s with the
same answers.

## Images

With `artifacts/vision_weights.bin` exported, an *Image…* button
appears (paste works too). The page resizes the picture so both sides
are multiples of 32 and the area is at most `--vision-max-pixels`
(default 640 × 640 = 400 image tokens); the CPU vision encoder
(`streammoe::riir::vision`, Accelerate GEMMs, `--vision-threads`
default 4) turns it into embeddings, and the model reads them at their
M-RoPE positions (`Ctx::eval_stage_mrope`). The encoder streams its
0.85 GiB of BF16 weights from disk for each image and keeps 137 MiB of
buffers, counted in the cap (`OpenConfig::host_buffers`). `--no-vision`
turns it off. Encoding a 640 × 640 image (400 tokens) takes ~3 s and
the process stays at ~3.7 GiB under a 4 GiB cap
([timings](benchmarks.md#vision)).

Decode after an image runs at the usual speed. Scenes, colors, layout
and counting work (a 2 × 2 color grid, five dots, shapes and their
positions, a photo); reading small or short printed text is
unreliable. MLX makes the same mistakes when given the same image
embeddings (see [`known-issues.md`](known-issues.md)), so this comes
from the model, not the engine. Larger images help: "7780" read
correctly at 640 × 320 and not at 512 × 256.
