//! batch_bench — parallel decoding end to end: N independent requests
//! decoded together with `eval_requests`, against decoding them one at
//! a time.
//!
//!   batch_bench --model DIR [--root DIR] [--cap-gib 5] [--tokens 64]
//!               [--overhead-mib 768]
//!
//! Four prompts become four admitted requests, each prefilled once and
//! saved. Then, from the saved states, every request greedy-decodes
//! `--tokens` tokens:
//!
//!   sequential   one request after another, `eval_request` per token
//!   round-robin  one token of each request in turn
//!   batch N      one `eval_requests` pass per step over the first N
//!
//! Batched output must equal sequential output token for token. Total
//! tok/s counts every request's tokens; per stream is what one user
//! sees. Sequential runs again at the end, since the hot tier keeps
//! warming.

use std::path::PathBuf;
use std::time::Instant;

use streammoe::{Ctx, KvFormat, OpenConfig, PrefetchGuess, ReqId, StageInput, StageOutput};
use tokenizers::Tokenizer;

const GIB: f64 = (1u64 << 30) as f64;
const N_REQ: usize = 4;

const PROMPTS: [&str; N_REQ] = [
    "Explain photosynthesis to a ten-year-old in one paragraph.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What are the main differences between TCP and UDP? Answer in a few sentences.",
    "Give me three ideas for a weekend trip near the mountains and say why each is fun.",
];

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn argmax(v: &[f32]) -> i32 {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b }) as i32
}

fn template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

struct Req {
    id: ReqId,
    np: usize,
    first: i32,
    state: Vec<u8>,
}

fn restore(ctx: &mut Ctx, reqs: &[Req]) {
    for r in reqs {
        ctx.state_load_for(r.id, &r.state).expect("state_load_for");
    }
}

fn step(ctx: &mut Ctx, r: &Req, tok: i32, i: usize, logits: &mut [f32]) -> i32 {
    ctx.eval_request(r.id, StageInput::Tokens(&[tok]), r.np + i, StageOutput::Logits(logits)).expect("decode");
    argmax(logits)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base = PathBuf::from(arg(&args, "--model").expect("--model DIR"));
    let cap = arg(&args, "--cap-gib").map_or(5.0, |v| v.parse::<f64>().unwrap());
    let ntok = arg(&args, "--tokens").map_or(64, |v| v.parse::<usize>().unwrap());
    let cfg = OpenConfig {
        n_ctx: 2048,
        mem_cap: Some((cap * GIB) as u64),
        kv_format: KvFormat::F32,
        max_checkpoints: 0,
        max_requests: N_REQ,
        prefetch_guess: PrefetchGuess::CrossLayer,
        prefetch_experts: Some(6),
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
        eprintln!("[batch] open failed: {e}");
        std::process::exit(2)
    });
    let vocab = ctx.n_vocab();
    let mut logits = vec![0f32; vocab];

    let mut reqs = Vec::new();
    for (i, p) in PROMPTS.iter().enumerate() {
        let id = ReqId(i as u64);
        if id != ReqId::DEFAULT {
            ctx.admit_request(id).expect("admit_request");
        }
        let prompt: Vec<i32> =
            tok.encode(template(p).as_str(), false).unwrap().get_ids().iter().map(|&t| t as i32).collect();
        ctx.eval_request(id, StageInput::Tokens(&prompt), 0, StageOutput::Logits(&mut logits)).expect("prefill");
        let mut state = vec![0u8; ctx.state_size_for(id).unwrap()];
        let n = ctx.state_save_for(id, &mut state).expect("state_save_for");
        state.truncate(n);
        reqs.push(Req { id, np: prompt.len(), first: argmax(&logits), state });
    }

    // Sequential: the plain baseline, and the reference output.
    let sequential = |ctx: &mut Ctx, logits: &mut Vec<f32>| {
        restore(ctx, &reqs);
        let t0 = Instant::now();
        let outs: Vec<Vec<i32>> = reqs
            .iter()
            .map(|r| {
                let mut out = vec![r.first];
                for i in 0..ntok {
                    let next = step(ctx, r, out[i], i, logits);
                    out.push(next);
                }
                out
            })
            .collect();
        (outs, t0.elapsed().as_secs_f64() * 1e3)
    };
    let (want, seq_ms) = sequential(&mut ctx, &mut logits);

    // Round-robin: one token of each request in turn.
    restore(&mut ctx, &reqs);
    let mut outs: Vec<Vec<i32>> = reqs.iter().map(|r| vec![r.first]).collect();
    let t0 = Instant::now();
    for i in 0..ntok {
        for (r, out) in reqs.iter().zip(outs.iter_mut()) {
            let next = step(&mut ctx, r, out[i], i, &mut logits);
            out.push(next);
        }
    }
    let rr_ms = t0.elapsed().as_secs_f64() * 1e3;
    let rr_ok = outs == want;

    // Batched over the first n requests.
    let mut batched = Vec::new();
    let mut multi = vec![0f32; N_REQ * vocab];
    for n in 2..=N_REQ {
        restore(&mut ctx, &reqs);
        let mut outs: Vec<Vec<i32>> = reqs[..n].iter().map(|r| vec![r.first]).collect();
        let t0 = Instant::now();
        for i in 0..ntok {
            let batch: Vec<(ReqId, i32, usize)> =
                reqs[..n].iter().zip(&outs).map(|(r, out)| (r.id, out[i], r.np + i)).collect();
            let rows = &mut multi[..n * vocab];
            ctx.eval_requests(&batch, rows).expect("eval_requests");
            for (out, row) in outs.iter_mut().zip(rows.chunks(vocab)) {
                out.push(argmax(row));
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let differ = outs.iter().zip(&want).filter(|(a, b)| a != b).count();
        batched.push((n, ms, differ));
    }

    let (_, seq2_ms) = sequential(&mut ctx, &mut logits);
    let seq_avg = (seq_ms + seq2_ms) / 2.0;
    let total = (N_REQ * ntok) as f64;
    let plain_tps = 1e3 * total / seq_avg;
    println!("{N_REQ} requests ({} prompt tokens), {ntok} tokens each", reqs.iter().map(|r| r.np.to_string()).collect::<Vec<_>>().join(" / "));
    println!("  {:<14} {:>12} {:>14} {:>9}  output", "mode", "total tok/s", "per stream", "speedup");
    println!("  {:<14} {:>12.2} {:>14.2} {:>9.2}  reference  (runs {:.2} / {:.2} tok/s)", "sequential",
             plain_tps, plain_tps, 1.0, 1e3 * total / seq_ms, 1e3 * total / seq2_ms);
    println!("  {:<14} {:>12.2} {:>14.2} {:>9.2}  {}", "round-robin", 1e3 * total / rr_ms,
             1e3 * ntok as f64 / rr_ms, seq_avg / rr_ms, if rr_ok { "identical" } else { "DIFFERS" });
    for (n, ms, differ) in batched {
        let tps = 1e3 * (n * ntok) as f64 / ms;
        println!("  {:<14} {:>12.2} {:>14.2} {:>9.2}  {}", format!("batch {n}"), tps, 1e3 * ntok as f64 / ms,
                 tps / plain_tps, if differ == 0 { "identical".to_string() } else { format!("{differ} requests DIFFER") });
    }
    if let Some(f) = ctx.footprint() {
        println!("peak {:.2} GiB", f.peak as f64 / GIB);
    }
}
