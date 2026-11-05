//! `--spec`: prompt-lookup speculative decoding for the chat's decode
//! loop (`--max-chats 1`, greedy turns, no image in the conversation).
//!
//! Each sampled token is evaluated together with up to three draft
//! tokens looked up earlier in the conversation (`draft.rs`, shared
//! with `spec_bench`) in one `Ctx::eval_tokens` pass, when the lookup
//! match is long enough to be worth it. The drafts the model agrees
//! with are streamed as they would have been one by one; the rest is
//! undone with `Ctx::rewind_tokens`. The answer is token for token the
//! plain greedy one: every token still comes from `Chat::next_token`
//! on the logits after the token before it.

use serde_json::{Value, json};
use streammoe::riir::MAX_DECODE_TOKENS;
use streammoe::{ReqId, StageOutput};

use crate::lookup::{Drafter, Guard, argmax};
use crate::{Chat, Turn};

/// What `--spec` asked for at startup.
#[derive(Clone, Copy)]
pub struct SpecOpts {
    pub guard: Guard,
}

/// Parses `--spec` and `--spec-min-match N` (default 8). `None` when
/// off, or with `--max-chats` > 1 (each chat's step is already a
/// one-token row of a shared `eval_requests` pass; logged).
pub fn opts(args: &[String], max_chats: usize) -> Option<SpecOpts> {
    if !args.iter().any(|a| a == "--spec") {
        return None;
    }
    if max_chats > 1 {
        eprintln!("[chat] --spec is off: it needs --max-chats 1");
        return None;
    }
    let mut guard = Guard::default();
    if let Some(n) = crate::arg(args, "--spec-min-match") {
        guard.min_match = n.parse().expect("--spec-min-match N");
    }
    Some(SpecOpts { guard })
}

/// `OpenConfig::max_decode_tokens` for `--spec`: a pass of the sampled
/// token plus three drafts.
pub const DECODE_TOKENS: usize = MAX_DECODE_TOKENS;

/// A speculating turn's drafter, and the logits rows of its passes.
pub struct TurnSpec {
    d: Drafter,
    rows: Vec<f32>,
}

/// The turn's counters for the `[chat]` stats line and `done` event.
pub fn stats(s: &TurnSpec) -> Value {
    let d = &s.d;
    json!({
        "passes": d.passes,
        "singles": d.singles,
        "drafted": d.drafted,
        "accepted": d.accepted,
        "accept_rate": d.accepted as f64 / d.drafted.max(1) as f64,
    })
}

impl Chat {
    /// A drafter for turn `t` when `--spec` is on and the turn can
    /// speculate: greedy sampling, the default request, no image in
    /// the conversation (`eval_tokens` uses plain RoPE positions).
    pub(crate) fn spec_for_turn(&self, t: &Turn) -> Option<TurnSpec> {
        let o = self.spec?;
        let c = &self.convs[t.k];
        let why = if t.temp > 0.0 {
            Some(format!("temperature {} (greedy only)", t.temp))
        } else if c.req != ReqId::DEFAULT {
            Some("not the default request".into())
        } else if c.rope_delta != 0 || t.n_image > 0 {
            Some("an image is in the conversation".into())
        } else {
            None
        };
        if let Some(why) = why {
            eprintln!("[chat] spec off for this turn: {why}");
            return None;
        }
        Some(TurnSpec { d: Drafter::new(DECODE_TOKENS - 1, Some(o.guard)), rows: Vec::new() })
    }

    /// Evaluate `tok` (sampled and streamed, not yet in the KV cache)
    /// and leave the logits for the token after it in `t.logits`, like
    /// a plain step. When the lookup drafts, `tok` and the drafts go
    /// through one `eval_tokens` pass: each draft the model agrees with
    /// is taken through `next_token` (streamed, counted, stop-checked)
    /// and the rest is rewound. `t.logits` then holds the logits after
    /// the last kept token, whose next token `next_token` samples — the
    /// model's own correction, or the stop if a kept token ended the
    /// turn (greedy sampling gives the same answer again).
    pub(crate) fn spec_eval(&mut self, t: &mut Turn, tok: i32) -> Result<(), streammoe::Error> {
        let k = t.k;
        let mut hist = self.convs[k].history.clone();
        hist.push(tok);
        let room = t.max_tokens.saturating_sub(t.generated.len());
        let draft = t.spec.as_mut().expect("spec_eval without a drafter").d.draft(&hist, room);
        if draft.is_empty() {
            return self.eval(k, &[tok], StageOutput::Logits(&mut t.logits));
        }
        let vocab = t.logits.len();
        let mut input = vec![tok];
        input.extend_from_slice(&draft);
        let mut rows = std::mem::take(&mut t.spec.as_mut().expect("drafter").rows);
        rows.resize(input.len() * vocab, 0.0);
        let start = self.convs[k].history.len();
        let r = self.ctx.eval_tokens(&input, start, &mut rows);
        let r = r.map(|()| {
            // Row j holds the logits after input[j]; input[j + 1] is
            // draft[j], kept while it is what greedy sampling picks.
            let mut keep = 1;
            loop {
                let j = keep - 1;
                let row = &rows[j * vocab..(j + 1) * vocab];
                t.logits.copy_from_slice(row);
                if j == draft.len() || argmax(row) != draft[j] {
                    break;
                }
                // The draft is the greedy token: emit it as a plain
                // step would. `None` (it ends the turn, or max_tokens
                // is reached) leaves it out of the KV cache.
                if self.next_token(t).is_none() {
                    break;
                }
                keep += 1;
            }
            keep
        });
        let s = t.spec.as_mut().expect("drafter");
        s.rows = rows;
        let keep = r?;
        self.ctx.rewind_tokens(keep)?;
        self.convs[k].history.extend_from_slice(&input[..keep]);
        s.d.verified(draft.len(), keep - 1);
        Ok(())
    }
}
