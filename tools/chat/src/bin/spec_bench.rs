//! spec_bench — speculative decoding end to end: prompt-lookup drafts
//! checked with `eval_tokens`, rejected drafts undone with
//! `rewind_tokens`, against plain one-token decode on the same prompts.
//!
//!   spec_bench --model DIR [--root DIR] [--cap-gib 4] [--tokens 128]
//!              [--max-draft 3] [--prompts FILE] [--overhead-mib 768]
//!
//! Per prompt: prefill, save the state, greedy-decode `--tokens` tokens
//! with `eval_token` (plain). Restore and decode again speculatively:
//! look the last 3 (then 2) tokens up earlier in the prompt + output,
//! draft up to `--max-draft` tokens that followed, verify `[last,
//! drafts..]` in one pass, keep the agreeing prefix plus the model's own
//! next token. Greedy speculative output must equal plain output token
//! for token. Plain runs again at the end, since the hot tier keeps
//! warming. `--prompts` takes one prompt per line, `\n` for newlines.

use std::path::PathBuf;
use std::time::Instant;

use streammoe::{Ctx, KvFormat, OpenConfig, PrefetchGuess};
use tokenizers::Tokenizer;

const GIB: f64 = (1u64 << 30) as f64;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn argmax(v: &[f32]) -> i32 {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b }) as i32
}

fn template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// Prompt lookup: the tokens that followed the latest earlier match of
/// `hist`'s last 3 (else 2) tokens, at most `max`.
fn lookup(hist: &[i32], max: usize) -> Vec<i32> {
    for n in [3usize, 2] {
        if hist.len() <= n {
            continue;
        }
        let suffix = &hist[hist.len() - n..];
        for i in (0..hist.len() - n).rev() {
            if &hist[i..i + n] == suffix {
                let from = i + n;
                return hist[from..(from + max).min(hist.len())].to_vec();
            }
        }
    }
    Vec::new()
}

const DEFAULT_PROMPTS: [&str; 3] = [
    "Fix the grammar and spelling in this text and return the whole corrected text, nothing else:\\n\\nThe streaming engine load the experts from the disk when it need them. Each layers pick eight expert out of two hundred fifty six, and the engine keep the most used one in memory so it dont have to read them again. When the memory cap is small, more expert has to come from the SSD, which make the decoding more slow. The hot tier remember which experts was used recently and keep them in RAM, and the prefetcher guess which experts the next layer will want and start reading them early.",
    "Add a short docstring to every function in this Python code and return the full code:\\n\\n```python\\ndef load_config(path):\\n    with open(path) as f:\\n        return json.load(f)\\n\\ndef save_config(path, config):\\n    with open(path, 'w') as f:\\n        json.dump(config, f, indent=2)\\n\\ndef merge_configs(base, override):\\n    result = dict(base)\\n    for key, value in override.items():\\n        if isinstance(value, dict) and isinstance(result.get(key), dict):\\n            result[key] = merge_configs(result[key], value)\\n        else:\\n            result[key] = value\\n    return result\\n```",
    "Explain photosynthesis to a ten-year-old in one paragraph.",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base = PathBuf::from(arg(&args, "--model").expect("--model DIR"));
    let cap = arg(&args, "--cap-gib").map_or(4.0, |v| v.parse::<f64>().unwrap());
    let ntok = arg(&args, "--tokens").map_or(128, |v| v.parse::<usize>().unwrap());
    let max_draft = arg(&args, "--max-draft").map_or(3, |v| v.parse::<usize>().unwrap()).clamp(1, 3);
    let prompts: Vec<String> = match arg(&args, "--prompts") {
        Some(p) => std::fs::read_to_string(p).unwrap().lines().filter(|l| !l.trim().is_empty()).map(String::from).collect(),
        None => DEFAULT_PROMPTS.iter().map(|s| s.to_string()).collect(),
    };

    let cfg = OpenConfig {
        n_ctx: 4096,
        mem_cap: Some((cap * GIB) as u64),
        kv_format: KvFormat::F32,
        max_checkpoints: 0,
        prefetch_guess: PrefetchGuess::CrossLayer,
        prefetch_experts: Some(6),
        // Above the 512 MiB default: the T-token buffers (~0.2 GiB) are
        // not in the budget plan yet, and both modes get the same
        // (smaller) hot tier.
        runtime_overhead: arg(&args, "--overhead-mib").map_or(768, |v| v.parse::<u64>().unwrap()) << 20,
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
        eprintln!("[spec] open failed: {e}");
        std::process::exit(2)
    });
    let vocab = ctx.n_vocab();
    let mut logits = vec![0f32; vocab];
    let mut multi = vec![0f32; (max_draft + 1) * vocab];
    let mut state = Vec::new();
    let (mut plain_ms_all, mut spec_ms_all, mut all_tokens) = (0.0, 0.0, 0usize);

    for p in &prompts {
        let p = p.replace("\\n", "\n");
        let prompt: Vec<i32> =
            tok.encode(template(&p).as_str(), false).unwrap().get_ids().iter().map(|&i| i as i32).collect();
        let np = prompt.len();
        ctx.memory_clear();
        ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
        let first = argmax(&logits);
        state.resize(ctx.state_size(), 0);
        let saved = ctx.state_save(&mut state).expect("state_save");

        // Plain: one token per eval_token.
        let plain = |ctx: &mut Ctx, logits: &mut Vec<f32>| {
            let mut out = vec![first];
            let t0 = Instant::now();
            for i in 0..ntok {
                ctx.eval_token(out[i], np + i, 0, logits).expect("decode");
                out.push(argmax(logits));
            }
            (out, t0.elapsed().as_secs_f64() * 1e3)
        };
        let (want, plain_ms) = plain(&mut ctx, &mut logits);

        // Speculative.
        ctx.state_load(&state[..saved]).expect("state_load");
        let mut hist = prompt.clone();
        hist.push(first);
        let mut out = vec![first];
        let (mut passes, mut singles, mut drafted, mut accepted) = (0usize, 0usize, 0usize, 0usize);
        let t0 = Instant::now();
        while out.len() < ntok + 1 {
            let pos = np + out.len() - 1;
            let last = *out.last().unwrap();
            let room = ntok + 1 - out.len();
            let mut draft = lookup(&hist, max_draft);
            draft.truncate(room.saturating_sub(1));
            if draft.is_empty() {
                ctx.eval_token(last, pos, 0, &mut logits).expect("decode");
                let next = argmax(&logits);
                out.push(next);
                hist.push(next);
                singles += 1;
                continue;
            }
            let mut input = vec![last];
            input.extend_from_slice(&draft);
            let rows = &mut multi[..input.len() * vocab];
            ctx.eval_tokens(&input, pos, rows).expect("eval_tokens");
            let mut a = 0;
            while a < draft.len() && argmax(&rows[a * vocab..(a + 1) * vocab]) == draft[a] {
                a += 1;
            }
            let next = argmax(&rows[a * vocab..(a + 1) * vocab]);
            ctx.rewind_tokens(a + 1).expect("rewind_tokens");
            for &t in draft[..a].iter().chain(std::iter::once(&next)) {
                out.push(t);
                hist.push(t);
            }
            passes += 1;
            drafted += draft.len();
            accepted += a;
        }
        let spec_ms = t0.elapsed().as_secs_f64() * 1e3;
        out.truncate(ntok + 1);
        let differ = out.iter().zip(&want).position(|(a, b)| a != b);

        ctx.state_load(&state[..saved]).expect("state_load");
        let (again, plain2_ms) = plain(&mut ctx, &mut logits);
        assert_eq!(again, want, "plain decode is not deterministic");
        let plain_avg = (plain_ms + plain2_ms) / 2.0;

        let label: String = p.chars().take(50).collect::<String>().replace('\n', " ");
        println!("== {label} ({np} prompt tokens, {ntok} decoded)");
        println!("  plain        {:6.1} ms/tok  {:5.2} tok/s   (runs {:.1} / {:.1} ms/tok)",
                 plain_avg / ntok as f64, 1e3 * ntok as f64 / plain_avg,
                 plain_ms / ntok as f64, plain2_ms / ntok as f64);
        println!("  speculative  {:6.1} ms/tok  {:5.2} tok/s   speedup {:.2}x",
                 spec_ms / ntok as f64, 1e3 * ntok as f64 / spec_ms, plain_avg / spec_ms);
        println!("  {passes} verify passes + {singles} single steps; drafted {drafted}, accepted {accepted} ({:.0}%), {:.2} tokens per verify pass",
                 100.0 * accepted as f64 / drafted.max(1) as f64,
                 (accepted + passes) as f64 / passes.max(1) as f64);
        match differ {
            None => println!("  output identical to plain decode"),
            Some(i) => println!("  OUTPUT DIFFERS from plain decode at token {i}"),
        }
        plain_ms_all += plain_avg;
        spec_ms_all += spec_ms;
        all_tokens += ntok;
    }
    println!("== all prompts: plain {:.2} tok/s, speculative {:.2} tok/s, speedup {:.2}x",
             1e3 * all_tokens as f64 / plain_ms_all, 1e3 * all_tokens as f64 / spec_ms_all, plain_ms_all / spec_ms_all);
    if let Some(f) = ctx.footprint() {
        println!("peak {:.2} GiB", f.peak as f64 / GIB);
    }
}
