//! quality — compare a model variant (e.g. requantized experts) with a
//! reference run of the same prompts.
//!
//!   quality --model DIR --cases cases.json --out out.json
//!           [--cap-gib 4] [--root DIR] [--save-ref ref.bin | --ref ref.bin]
//!
//! `--root` picks the experts directory (default `DIR/root`; e.g.
//! `DIR/root-3bit`).
//!
//! `cases.json` holds `{"forced": [{name, prompt, target}],
//! "gen": [{name, prompt, max_tokens, expect?, system?}]}`.
//!
//! - **forced**: the chat-templated prompt, then `target` fed one token
//!   at a time. Per target token: log-prob of the actual token (mean →
//!   perplexity), the argmax, and against a reference run (`--ref`) the
//!   KL divergence and top-1 agreement. `--save-ref` writes the top
//!   [`REF_K`] log-probs per position for later runs to compare with.
//! - **gen**: greedy answer to the prompt. `expect` scores it: `"json"`
//!   (the answer parses as JSON, code fences allowed), `"tool_call"` (a
//!   `<tool_call>` block holding JSON with `name` and `arguments`), or
//!   any other string, which the answer must contain.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{Value, json};
use streammoe::{Ctx, KvFormat, OpenConfig, PrefetchGuess};
use tokenizers::Tokenizer;

const GIB: f64 = (1u64 << 30) as f64;
/// Reference log-probs kept per position. The tail beyond them is
/// folded into one bucket for the KL estimate.
const REF_K: usize = 256;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = logits.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
    logits.iter().map(|&x| x as f64 - z).collect()
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// One reference position: top-`REF_K` (token, log-prob), best first.
struct RefPos {
    top: Vec<(u32, f32)>,
}

impl RefPos {
    fn from_lp(lp: &[f64]) -> Self {
        let mut idx: Vec<usize> = (0..lp.len()).collect();
        idx.select_nth_unstable_by(REF_K, |&a, &b| lp[b].total_cmp(&lp[a]));
        idx.truncate(REF_K);
        idx.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]));
        Self { top: idx.into_iter().map(|i| (i as u32, lp[i] as f32)).collect() }
    }

    /// KL(ref || this) over the reference's top tokens plus one tail
    /// bucket holding the rest of each distribution's mass.
    fn kl(&self, lp: &[f64]) -> f64 {
        let (mut kl, mut p_top, mut q_top) = (0.0, 0.0, 0.0);
        for &(t, lpr) in &self.top {
            let (p, lq) = ((lpr as f64).exp(), lp[t as usize]);
            kl += p * (lpr as f64 - lq);
            p_top += p;
            q_top += lq.exp();
        }
        let (pt, qt) = ((1.0 - p_top).max(1e-12), (1.0 - q_top).max(1e-12));
        kl + pt * (pt.ln() - qt.ln())
    }
}

fn write_ref(path: &str, refs: &[RefPos]) {
    let mut w = BufWriter::new(File::create(path).expect("create ref"));
    w.write_all(&(refs.len() as u32).to_le_bytes()).unwrap();
    for r in refs {
        for &(t, l) in &r.top {
            w.write_all(&t.to_le_bytes()).unwrap();
            w.write_all(&l.to_le_bytes()).unwrap();
        }
    }
}

fn read_ref(path: &str) -> Vec<RefPos> {
    let mut b = Vec::new();
    BufReader::new(File::open(path).expect("open ref")).read_to_end(&mut b).unwrap();
    let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let n = u(0) as usize;
    (0..n)
        .map(|p| RefPos {
            top: (0..REF_K)
                .map(|k| {
                    let o = 4 + (p * REF_K + k) * 8;
                    (u(o), f32::from_bits(u(o + 4)))
                })
                .collect(),
        })
        .collect()
}

fn template(system: Option<&str>, prompt: &str) -> String {
    let mut s = String::new();
    if let Some(sys) = system {
        s += &format!("<|im_start|>system\n{sys}<|im_end|>\n");
    }
    s + &format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// Score a greedy answer against `expect`.
fn check(expect: &str, text: &str) -> bool {
    let strip = |t: &str| {
        let t = t.trim();
        let t = t.strip_prefix("```json").or_else(|| t.strip_prefix("```")).unwrap_or(t);
        t.strip_suffix("```").unwrap_or(t).trim().to_string()
    };
    match expect {
        "json" => serde_json::from_str::<Value>(&strip(text)).is_ok(),
        "tool_call" => text
            .split("<tool_call>")
            .nth(1)
            .and_then(|r| r.split("</tool_call>").next())
            .and_then(|j| serde_json::from_str::<Value>(j.trim()).ok())
            .is_some_and(|v| v.get("name").is_some() && v.get("arguments").is_some()),
        s => text.to_lowercase().contains(&s.to_lowercase()),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base = PathBuf::from(arg(&args, "--model").expect("--model DIR"));
    let cases: Value = serde_json::from_reader(File::open(arg(&args, "--cases").expect("--cases")).unwrap()).unwrap();
    let out_path = arg(&args, "--out").expect("--out");
    let cap = arg(&args, "--cap-gib").map_or(4.0, |v| v.parse::<f64>().unwrap());
    let refs = arg(&args, "--ref").map(|p| read_ref(&p));
    let save_ref = arg(&args, "--save-ref");

    let cfg = OpenConfig {
        n_ctx: 4096,
        mem_cap: Some((cap * GIB) as u64),
        kv_format: KvFormat::F32,
        max_checkpoints: 0,
        prefetch_guess: PrefetchGuess::CrossLayer,
        prefetch_experts: Some(6),
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
        eprintln!("[quality] open failed: {e}");
        std::process::exit(2)
    });
    let enc = |s: &str| -> Vec<i32> { tok.encode(s, false).unwrap().get_ids().iter().map(|&i| i as i32).collect() };
    let id = |s: &str| tok.token_to_id(s).expect(s) as i32;
    let (im_end, eos) = (id("<|im_end|>"), id("<|endoftext|>"));
    let vocab = ctx.n_vocab();
    let mut logits = vec![0f32; vocab];
    let t0 = Instant::now();

    // ── teacher-forced cases ────────────────────────────────────────
    let mut new_refs = Vec::new();
    let mut ref_i = 0usize;
    let mut forced_out = Vec::new();
    let (mut all_nll, mut all_kl, mut all_agree, mut all_n) = (0.0, 0.0, 0usize, 0usize);
    for c in cases["forced"].as_array().into_iter().flatten() {
        let prompt = enc(&template(None, c["prompt"].as_str().unwrap()));
        let target = enc(c["target"].as_str().unwrap());
        ctx.memory_clear();
        ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
        let (mut nll, mut kl, mut agree) = (0.0, 0.0, 0usize);
        for (i, &t) in target.iter().enumerate() {
            let lp = log_softmax(&logits);
            nll -= lp[t as usize];
            if let Some(r) = &refs {
                let rp = &r[ref_i];
                kl += rp.kl(&lp);
                agree += (rp.top[0].0 as usize == argmax(&logits)) as usize;
            }
            if save_ref.is_some() {
                new_refs.push(RefPos::from_lp(&lp));
            }
            ref_i += 1;
            if i + 1 < target.len() {
                ctx.eval_token(t, prompt.len() + i, 0, &mut logits).expect("decode");
            }
        }
        let n = target.len();
        eprintln!("[quality] forced {}: {n} tok, ppl {:.3}", c["name"], (nll / n as f64).exp());
        forced_out.push(json!({
            "name": c["name"], "tokens": n,
            "ppl": (nll / n as f64).exp(),
            "kl": refs.as_ref().map(|_| kl / n as f64),
            "top1_agree": refs.as_ref().map(|_| agree as f64 / n as f64),
        }));
        (all_nll, all_kl, all_agree, all_n) = (all_nll + nll, all_kl + kl, all_agree + agree, all_n + n);
    }
    if let Some(p) = &save_ref {
        write_ref(p, &new_refs);
    }

    // ── greedy generation cases ─────────────────────────────────────
    let mut gen_out = Vec::new();
    let mut passed = 0usize;
    let mut gen_tokens = 0usize;
    let t_gen = Instant::now();
    for c in cases["gen"].as_array().into_iter().flatten() {
        let prompt = enc(&template(c["system"].as_str(), c["prompt"].as_str().unwrap()));
        let max = c["max_tokens"].as_u64().unwrap_or(200) as usize;
        ctx.memory_clear();
        ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
        let mut out: Vec<u32> = Vec::new();
        while out.len() < max {
            let t = argmax(&logits) as i32;
            if t == im_end || t == eos {
                break;
            }
            out.push(t as u32);
            ctx.eval_token(t, prompt.len() + out.len() - 1, 0, &mut logits).expect("decode");
        }
        gen_tokens += out.len();
        let text = tok.decode(&out, false).unwrap_or_default();
        let ok = c["expect"].as_str().map(|e| check(e, &text));
        passed += ok.unwrap_or(false) as usize;
        eprintln!("[quality] gen {}: {} tok, pass {:?}", c["name"], out.len(), ok);
        gen_out.push(json!({"name": c["name"], "text": text, "pass": ok}));
    }
    let gen_secs = t_gen.elapsed().as_secs_f64();

    let summary = json!({
        "model": base.display().to_string(),
        "forced_tokens": all_n,
        "ppl": (all_nll / all_n.max(1) as f64).exp(),
        "kl": refs.as_ref().map(|_| all_kl / all_n.max(1) as f64),
        "top1_agree": refs.as_ref().map(|_| all_agree as f64 / all_n.max(1) as f64),
        "gen_passed": passed,
        "gen_total": gen_out.len(),
        "gen_tokens": gen_tokens,
        "gen_secs": gen_secs,
        "peak_gib": ctx.footprint().map(|f| f.peak as f64 / GIB),
        "secs": t0.elapsed().as_secs_f64(),
    });
    eprintln!("[quality] {summary}");
    let all = json!({"summary": summary, "forced": forced_out, "gen": gen_out});
    std::fs::write(&out_path, serde_json::to_string_pretty(&all).unwrap()).unwrap();
}
