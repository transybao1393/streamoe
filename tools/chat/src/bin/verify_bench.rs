//! verify_bench — what checking T draft tokens in one forward pass
//! costs, against decoding them one at a time.
//!
//!   verify_bench --model DIR [--root DIR] [--cap-gib 4] [--tokens 48]
//!                [--prompts FILE] [--profile] [--prompt-path]
//!
//! Per prompt: prefill, save the state, greedy-decode `--tokens` tokens
//! with `eval_token` (the baseline). Then, for T = 1..4, restore the
//! state and feed the same tokens T at a time through `eval_prompt` —
//! the cost of verifying drafts when every draft is accepted. Each
//! group's last logits must pick the next recorded token. The same
//! replay through `eval_tokens` (decode with T tokens) must give
//! bit-identical logits at every position. The `eval_prompt` replay
//! runs with `--prompt-path`. The baseline runs again at the end, since
//! the hot tier keeps warming.

use std::path::PathBuf;
use std::time::Instant;

use streammoe::{Ctx, KvFormat, OpenConfig, PrefetchGuess};
use tokenizers::Tokenizer;

const GIB: f64 = (1u64 << 30) as f64;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

fn template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// One measured pass over the recorded tokens.
struct Run {
    ms_per_tok: f64,
    ms_per_pass: f64,
    io_ms_per_tok: f64,
    reads_per_tok: f64,
    mib_per_tok: f64,
    hot_pct: f64,
    mismatches: usize,
}

/// Per pass: GPU command buffers by label (count, CPU wait, GPU busy),
/// the I/O wait, and what is left (CPU work outside both).
fn profile(ctx: &Ctx, wall_ms: f64, io_ms: f64, passes: usize) {
    let mut rows = ctx.cmdbuf_stats();
    rows.sort_by(|a, b| b.1.cpu_wait_ns.cmp(&a.1.cpu_wait_ns));
    let p = passes as f64;
    let (wait, gpu): (f64, f64) = rows.iter().fold((0.0, 0.0), |(w, g), (_, s)| {
        (w + s.cpu_wait_ns as f64 / 1e6, g + s.gpu_ns as f64 / 1e6)
    });
    println!("    per pass: wall {:.1} ms = cmdbuf waits {:.1} + io wait {:.1} + rest {:.1}; GPU busy {:.1} ms",
             wall_ms / p, wait / p, io_ms / p, (wall_ms - wait - io_ms) / p, gpu / p);
    println!("    {:<34} {:>7} {:>9} {:>9}", "label", "n/pass", "wait ms", "gpu ms");
    for (label, s) in rows.iter().take(12) {
        println!("    {label:<34} {:>7.1} {:>9.2} {:>9.2}",
                 s.count as f64 / p, s.cpu_wait_ns as f64 / 1e6 / p, s.gpu_ns as f64 / 1e6 / p);
    }
}

fn measure(ctx: &mut Ctx, n: usize, f: impl FnOnce(&mut Ctx) -> usize, passes: usize) -> Run {
    ctx.reset_expert_io_stats();
    ctx.reset_cmdbuf_stats();
    let t0 = Instant::now();
    let mismatches = f(ctx);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let s = ctx.expert_io_stats();
    if std::env::args().any(|a| a == "--profile") {
        profile(ctx, ms, (s.io_wait_ns + s.prefill_io_ns) as f64 / 1e6, passes);
    }
    let hot: u64 = s.per_layer.iter().map(|l| l.hot_hits + l.prefill_hot).sum();
    let used = hot + s.prefetch_hits + s.demand.reads + s.prefill.reads;
    let nf = n as f64;
    Run {
        ms_per_tok: ms / nf,
        ms_per_pass: ms / passes as f64,
        io_ms_per_tok: (s.io_wait_ns + s.prefill_io_ns) as f64 / 1e6 / nf,
        reads_per_tok: s.disk_reads as f64 / nf,
        mib_per_tok: s.disk_bytes as f64 / (1u64 << 20) as f64 / nf,
        hot_pct: 100.0 * hot as f64 / used.max(1) as f64,
        mismatches,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base = PathBuf::from(arg(&args, "--model").expect("--model DIR"));
    let cap = arg(&args, "--cap-gib").map_or(4.0, |v| v.parse::<f64>().unwrap());
    let ntok = arg(&args, "--tokens").map_or(48, |v| v.parse::<usize>().unwrap());
    let prompts: Vec<String> = match arg(&args, "--prompts") {
        Some(p) => std::fs::read_to_string(p).unwrap().lines().filter(|l| !l.trim().is_empty()).map(String::from).collect(),
        None => vec!["Explain photosynthesis to a ten-year-old in one paragraph.".into()],
    };

    let cfg = OpenConfig {
        n_ctx: 4096,
        mem_cap: Some((cap * GIB) as u64),
        kv_format: KvFormat::F32,
        max_checkpoints: 0,
        prefetch_guess: PrefetchGuess::CrossLayer,
        prefetch_experts: Some(6),
        max_decode_tokens: 4,
        ..OpenConfig::default()
    };
    let art = base.join("artifacts");
    let tok = Tokenizer::from_file(base.join("mlx-4bit/tokenizer.json")).expect("tokenizer.json");
    let mut ctx = Ctx::open_with(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
        &art.join("vocab.bin"),
        &arg(&args, "--root").map_or(base.join("root"), PathBuf::from),
        false,
        &cfg,
    )
    .unwrap_or_else(|e| {
        eprintln!("[verify] open failed: {e}");
        std::process::exit(2)
    });
    let mut logits = vec![0f32; ctx.n_vocab()];
    let mut state = Vec::new();

    for p in &prompts {
        let prompt: Vec<i32> =
            tok.encode(template(p).as_str(), false).unwrap().get_ids().iter().map(|&i| i as i32).collect();
        let np = prompt.len();
        ctx.memory_clear();
        ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
        let first = argmax(&logits) as i32;
        // Sized after the prefill: the snapshot grows with the position.
        state.resize(ctx.state_size(), 0);
        let saved = ctx.state_save(&mut state).expect("state_save");

        // Baseline: one token per pass. Records the greedy tokens.
        let mut toks = vec![first];
        let mut base_logits: Vec<Vec<f32>> = Vec::with_capacity(ntok);
        let base0 = measure(
            &mut ctx,
            ntok,
            |ctx| {
                for i in 0..ntok {
                    ctx.eval_token(toks[i], np + i, 0, &mut logits).expect("decode");
                    toks.push(argmax(&logits) as i32);
                    base_logits.push(logits.clone());
                }
                0
            },
            ntok,
        );
        let label: String = p.chars().take(40).collect();
        println!("== {label} ({np} prompt tokens, {ntok} decoded)");
        println!("  {:<18} {:>8} {:>9} {:>8} {:>9} {:>8} {:>6} {:>5}",
                 "mode", "ms/tok", "ms/pass", "io ms", "reads", "MiB", "hot%", "bad");
        let show = |name: &str, r: &Run| {
            println!("  {:<18} {:>8.1} {:>9.1} {:>8.1} {:>9.1} {:>8.0} {:>6.0} {:>5}",
                     name, r.ms_per_tok, r.ms_per_pass, r.io_ms_per_tok, r.reads_per_tok, r.mib_per_tok, r.hot_pct, r.mismatches);
        };
        show("eval_token", &base0);

        // eval_tokens: decode with T tokens per pass. `bad` counts
        // positions whose logits are not bit-identical to eval_token's.
        let vocab = ctx.n_vocab();
        for t in 2..=4usize {
            ctx.state_load(&state[..saved]).expect("state_load");
            let groups = ntok.div_ceil(t);
            let mut max_diff = 0f32;
            let mut wrong_pick = 0usize;
            let mut multi = vec![0f32; t * vocab];
            let r = measure(
                &mut ctx,
                ntok,
                |ctx| {
                    let mut bad = 0;
                    for g in 0..groups {
                        let (a, b) = (g * t, ((g + 1) * t).min(ntok));
                        let out = &mut multi[..(b - a) * vocab];
                        ctx.eval_tokens(&toks[a..b], np + a, out).expect("eval_tokens");
                        for (j, row) in out.chunks(vocab).enumerate() {
                            let want = &base_logits[a + j];
                            if row != &want[..] {
                                bad += 1;
                                let d = row.iter().zip(want).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                                max_diff = max_diff.max(d);
                            }
                            wrong_pick += (argmax(row) as i32 != toks[a + j + 1]) as usize;
                        }
                    }
                    bad
                },
                groups,
            );
            show(&format!("eval_tokens T={t}"), &r);
            if r.mismatches > 0 {
                println!("    {} positions differ, max |diff| {max_diff:.3e}, {wrong_pick} greedy picks differ", r.mismatches);
            }
        }

        let prompt_path = std::env::args().any(|a| a == "--prompt-path");
        for t in (1..=4usize).filter(|_| prompt_path) {
            ctx.state_load(&state[..saved]).expect("state_load");
            let groups = ntok.div_ceil(t);
            let r = measure(
                &mut ctx,
                ntok,
                |ctx| {
                    let mut bad = 0;
                    for g in 0..groups {
                        let (a, b) = (g * t, ((g + 1) * t).min(ntok));
                        ctx.eval_prompt(&toks[a..b], np + a, 0, &mut logits).expect("verify");
                        bad += (argmax(&logits) as i32 != toks[b]) as usize;
                    }
                    bad
                },
                groups,
            );
            show(&format!("eval_prompt T={t}"), &r);
        }

        ctx.state_load(&state[..saved]).expect("state_load");
        let base1 = measure(
            &mut ctx,
            ntok,
            |ctx| {
                let mut bad = 0;
                for i in 0..ntok {
                    ctx.eval_token(toks[i], np + i, 0, &mut logits).expect("decode");
                    bad += (argmax(&logits) as i32 != toks[i + 1]) as usize;
                }
                bad
            },
            ntok,
        );
        show("eval_token again", &base1);
    }
    if let Some(f) = ctx.footprint() {
        println!("peak {:.2} GiB", f.peak as f64 / GIB);
    }
}
