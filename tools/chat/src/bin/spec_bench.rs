//! spec_bench — speculative decoding end to end: prompt-lookup drafts
//! checked with `eval_tokens`, rejected drafts undone with
//! `rewind_tokens`, against plain one-token decode on the same prompts.
//!
//!   spec_bench --model DIR [--root DIR] [--cap-gib 4] [--tokens 128]
//!              [--max-draft 3] [--prompts FILE] [--overhead-mib 512]
//!              [--arms guard|old|guard,old|old,guard] [--min-match 8]
//!              [--backoff 0] [--after-full] [--trace]
//!
//! Per prompt: prefill, save the state, greedy-decode `--tokens` tokens
//! with `eval_token` (plain). Restore and decode again speculatively:
//! look the last 3 (then 2) tokens up earlier in the prompt + output,
//! draft up to `--max-draft` tokens that followed, verify `[last,
//! drafts..]` in one pass, keep the agreeing prefix plus the model's own
//! next token. Greedy speculative output must equal plain output token
//! for token. Plain runs again at the end, since the hot tier keeps
//! warming. `--prompts` takes one prompt per line, `\n` for newlines.
//!
//! The guard (arm `guard`, the default) drafts only when the match
//! behind the lookup is at least `--min-match` tokens long (with
//! `--after-full`, also right after a pass accepted its whole draft);
//! otherwise it takes a plain step. A pass that accepts nothing skips
//! drafting for `--backoff` steps. Arm `old` drafts on every lookup hit, as before the guard;
//! `STREAMMOE_SPEC_GUARD=0` makes the default arm `old`. `--trace`
//! prints one `[trace]` line per verify pass (match length, drafted,
//! accepted, ms) to stderr.

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

/// Longest match length `lookup` reports; longer matches count as this.
const MATCH_CAP: usize = 64;

/// Prompt lookup: the tokens that followed the latest earlier match of
/// `hist`'s last 3 (else 2) tokens, at most `max`, and how many tokens
/// that match runs back (up to `MATCH_CAP`).
fn lookup(hist: &[i32], max: usize) -> (Vec<i32>, usize) {
    for n in [3usize, 2] {
        if hist.len() <= n {
            continue;
        }
        let suffix = &hist[hist.len() - n..];
        for i in (0..hist.len() - n).rev() {
            if &hist[i..i + n] == suffix {
                let from = i + n;
                let mut len = n;
                while len < MATCH_CAP && len < from && hist[from - len - 1] == hist[hist.len() - len - 1] {
                    len += 1;
                }
                return (hist[from..(from + max).min(hist.len())].to_vec(), len);
            }
        }
    }
    (Vec::new(), 0)
}

/// When to spend a verify pass on a draft. A pass of ~4 tokens costs
/// about three plain steps, so a draft pays only when most of it is
/// accepted. Unguarded passes, packed 3-bit at a 4 GiB cap: matches of
/// 8+ tokens accept 86–96% and gain 1.16–1.25× over plain steps;
/// shorter ones accept 20–67% and gain 0.51–0.93×, and so do short
/// matches right after a fully accepted pass (52%, 0.80×).
#[derive(Clone, Copy)]
struct Guard {
    min_match: usize,
    after_full: bool,
    backoff: usize,
}

#[derive(Default)]
struct Spec {
    out: Vec<i32>,
    ms: f64,
    passes: usize,
    singles: usize,
    skipped: usize,
    drafted: usize,
    accepted: usize,
}

/// Greedy speculative decode of `ntok` tokens after `first`, from the
/// state right after prefill. `guard: None` drafts on every lookup hit.
#[allow(clippy::too_many_arguments)]
fn speculate(ctx: &mut Ctx, prompt: &[i32], first: i32, ntok: usize, max_draft: usize, guard: Option<Guard>,
             trace: bool, logits: &mut [f32], multi: &mut [f32]) -> Spec {
    let vocab = logits.len();
    let np = prompt.len();
    let mut hist = prompt.to_vec();
    hist.push(first);
    let mut s = Spec { out: vec![first], ..Spec::default() };
    let (mut full, mut cool) = (false, 0usize);
    let t0 = Instant::now();
    while s.out.len() < ntok + 1 {
        let pos = np + s.out.len() - 1;
        let last = *s.out.last().unwrap();
        let room = ntok + 1 - s.out.len();
        let (mut draft, len) = lookup(&hist, max_draft);
        draft.truncate(room.saturating_sub(1));
        let skip = match guard {
            Some(g) if !draft.is_empty() => {
                let skip = cool > 0 || (len < g.min_match && !(g.after_full && full));
                cool = cool.saturating_sub(1);
                skip
            }
            _ => false,
        };
        if draft.is_empty() || skip {
            ctx.eval_token(last, pos, 0, logits).expect("decode");
            let next = argmax(logits);
            s.out.push(next);
            hist.push(next);
            s.singles += 1;
            s.skipped += skip as usize;
            full = false;
            continue;
        }
        let tp = Instant::now();
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
            s.out.push(t);
            hist.push(t);
        }
        if trace {
            eprintln!("[trace] match {len} drafted {} accepted {a} ms {:.1}", draft.len(), tp.elapsed().as_secs_f64() * 1e3);
        }
        full = a == draft.len();
        if a == 0 {
            cool = guard.map_or(0, |g| g.backoff);
        }
        s.passes += 1;
        s.drafted += draft.len();
        s.accepted += a;
    }
    s.ms = t0.elapsed().as_secs_f64() * 1e3;
    s.out.truncate(ntok + 1);
    s
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
    let guard = Guard {
        min_match: arg(&args, "--min-match").map_or(8, |v| v.parse::<usize>().unwrap()),
        after_full: args.iter().any(|a| a == "--after-full"),
        backoff: arg(&args, "--backoff").map_or(0, |v| v.parse::<usize>().unwrap()),
    };
    let default_arm = if std::env::var("STREAMMOE_SPEC_GUARD").map_or(true, |v| v != "0") { "guard" } else { "old" };
    let arms: Vec<String> = arg(&args, "--arms").unwrap_or_else(|| default_arm.into()).split(',').map(String::from).collect();
    assert!(arms.iter().all(|a| a == "guard" || a == "old"), "--arms takes guard and/or old");
    let trace = args.iter().any(|a| a == "--trace");

    let cfg = OpenConfig {
        n_ctx: 4096,
        mem_cap: Some((cap * GIB) as u64),
        kv_format: KvFormat::F32,
        max_checkpoints: 0,
        prefetch_guess: PrefetchGuess::CrossLayer,
        prefetch_experts: Some(6),
        max_decode_tokens: max_draft + 1,
        runtime_overhead: arg(&args, "--overhead-mib").map_or(OpenConfig::default().runtime_overhead, |v| v.parse::<u64>().unwrap() << 20),
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
    let (mut plain_ms_all, mut spec_ms_all, mut all_tokens) = (0.0, vec![0.0; arms.len()], 0usize);

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

        // Speculative, one run per arm.
        let mut runs = Vec::new();
        for arm in &arms {
            ctx.state_load(&state[..saved]).expect("state_load");
            if trace {
                eprintln!("[trace] arm {arm}");
            }
            let g = (arm == "guard").then_some(guard);
            runs.push(speculate(&mut ctx, &prompt, first, ntok, max_draft, g, trace, &mut logits, &mut multi));
        }

        ctx.state_load(&state[..saved]).expect("state_load");
        let (again, plain2_ms) = plain(&mut ctx, &mut logits);
        assert_eq!(again, want, "plain decode is not deterministic");
        let plain_avg = (plain_ms + plain2_ms) / 2.0;

        let label: String = p.chars().take(50).collect::<String>().replace('\n', " ");
        println!("== {label} ({np} prompt tokens, {ntok} decoded)");
        println!("  plain        {:6.1} ms/tok  {:5.2} tok/s   (runs {:.1} / {:.1} ms/tok)",
                 plain_avg / ntok as f64, 1e3 * ntok as f64 / plain_avg,
                 plain_ms / ntok as f64, plain2_ms / ntok as f64);
        for (arm, r) in arms.iter().zip(&runs) {
            println!("  spec {:5}   {:6.1} ms/tok  {:5.2} tok/s   speedup {:.2}x",
                     arm, r.ms / ntok as f64, 1e3 * ntok as f64 / r.ms, plain_avg / r.ms);
            println!("    {} verify passes + {} single steps ({} skipped by the guard); drafted {}, accepted {} ({:.0}%), {:.2} tokens per verify pass",
                     r.passes, r.singles, r.skipped, r.drafted, r.accepted,
                     100.0 * r.accepted as f64 / r.drafted.max(1) as f64,
                     (r.accepted + r.passes) as f64 / r.passes.max(1) as f64);
            match r.out.iter().zip(&want).position(|(a, b)| a != b) {
                None => println!("    output identical to plain decode"),
                Some(i) => println!("    OUTPUT DIFFERS from plain decode at token {i}"),
            }
        }
        plain_ms_all += plain_avg;
        for (all, r) in spec_ms_all.iter_mut().zip(&runs) {
            *all += r.ms;
        }
        all_tokens += ntok;
    }
    for (arm, ms) in arms.iter().zip(&spec_ms_all) {
        println!("== all prompts: plain {:.2} tok/s, spec {arm} {:.2} tok/s, speedup {:.2}x",
                 1e3 * all_tokens as f64 / plain_ms_all, 1e3 * all_tokens as f64 / ms, plain_ms_all / ms);
    }
    if let Some(f) = ctx.footprint() {
        println!("peak {:.2} GiB", f.peak as f64 / GIB);
    }
}
