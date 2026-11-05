//! Prefill while the user types (`--prefill-while-typing`, `--max-chats
//! 1` only).
//!
//! The page posts the message box to `/draft` ~300 ms after each
//! keystroke. The server works out the tokens the turn will start with
//! — the last turn's `<|im_end|>`, the user header and the draft up to
//! the start of its last word (the BPE of a word still being typed can
//! change) — and prefills them on the default request, a few tokens at
//! a time between requests, so a `/chat` or a newer `/draft` waits at
//! most one step. When the message is sent, `begin` checks that the
//! prefilled tokens are a prefix of the turn's real prompt and prefills
//! only the rest; otherwise it rolls back first.
//!
//! Rolling back: the KV cache could be truncated, but the linear-
//! attention recurrence cannot (docs/known-issues.md), so the first
//! draft step of a turn takes a checkpoint (`Ctx::checkpoint_pos`) of
//! the conversation as it stood, and an edit of already-prefilled text
//! restores it (`Ctx::restore_to`) and prefills again. The server opens
//! with one checkpoint buffer planned under the cap (`max_checkpoints:
//! 1`); in the first turn, with nothing before it, a rollback is a
//! clear. Every step evaluates at least two tokens, so it runs on the
//! batched prefill path like the turn's own prefill (one token would
//! take the decode path).
//!
//! The answer is token for token the same as without drafts: the same
//! tokens are evaluated, only in other chunks
//! (`tools/chat/typing_check.py` checks it against a running server).

use std::sync::mpsc::{Receiver, TryRecvError};

use serde_json::{Value, json};
use streammoe::StageOutput;

use super::Chat;

/// Draft state of the default conversation.
pub(crate) struct Draft {
    /// `--prefill-while-typing`, and its checkpoint fitted the cap.
    pub(crate) on: bool,
    /// Most tokens one step prefills (`--draft-chunk`).
    pub(crate) chunk: usize,
    /// The turn being typed, from its first `/draft` until `/chat`.
    cur: Option<Typing>,
    /// `begin` took `cur` and has not settled it yet.
    taken: Option<Typing>,
    /// Tokens the last turn found prefilled (reported in its stats).
    pub(crate) reused: Option<usize>,
}

struct Typing {
    /// `history.len()` before the turn: the checkpoint's position.
    base: usize,
    /// The conversation's `pending` tokens then (the last turn's
    /// `<|im_end|>`), which the draft prefills first.
    pending: Vec<i32>,
    /// Checkpoint at `base` taken (not needed at `base` 0).
    saved: bool,
    /// What the turn's prompt should start with, as far as known; the
    /// tokens prefilled so far (`history[base..]`) are a prefix of it.
    target: Vec<i32>,
}

impl Draft {
    pub(crate) fn new(on: bool, chunk: usize) -> Self {
        Self { on, chunk: chunk.max(2), cur: None, taken: None, reused: None }
    }

    /// The conversation was reset: nothing prefilled is kept.
    pub(crate) fn clear(&mut self) {
        self.cur = None;
        self.taken = None;
    }
}

/// Bytes of `msg` up to the start of its last word (the whitespace in
/// front of it, which Qwen's pre-tokenizer joins to the word): tokens
/// ending there no longer change as typing goes on.
fn stable_end(msg: &str) -> usize {
    let b = msg.as_bytes();
    (1..b.len()).rev().find(|&i| b[i - 1].is_ascii_whitespace() && !b[i].is_ascii_whitespace()).map_or(0, |i| i - 1)
}

impl Chat {
    /// `POST /draft` `{message, system}`: what the user has typed so
    /// far. Sets the target and rolls back at once if it no longer
    /// starts with what is prefilled; the prefill runs in
    /// [`Self::next_request`].
    pub(crate) fn draft_post(&mut self, req: &Value) -> Value {
        if !self.draft.on {
            return json!({"ok": false, "reason": "prefill while typing is off"});
        }
        let msg = req["message"].as_str().unwrap_or("");
        let system = req["system"].as_str().unwrap_or("");
        if self.draft.cur.is_none() {
            let c = &mut self.convs[0];
            let pending = std::mem::take(&mut c.pending);
            self.draft.cur = Some(Typing { base: c.history.len(), pending, saved: false, target: Vec::new() });
        }
        let cur = self.draft.cur.as_ref().expect("set above");
        // The turn's head as `begin_inner` builds it.
        let mut head = String::new();
        if cur.base == 0 && cur.pending.is_empty() && !system.is_empty() {
            head += &format!("<|im_start|>system\n{system}<|im_end|>\n");
        }
        if cur.base > 0 {
            head += "\n";
        }
        head += "<|im_start|>user\n";
        let mut target = cur.pending.clone();
        target.extend(self.encode(&head));
        let end = stable_end(msg);
        if end > 0 {
            let enc = self.tok.encode(msg, false).expect("tokenize");
            target.extend(enc.get_ids().iter().zip(enc.get_offsets())
                .take_while(|(_, o)| o.1 <= end).map(|(&i, _)| i as i32));
        }
        // Leave room for the closing template and an answer.
        target.truncate(self.n_ctx.saturating_sub(cur.base + 256));
        let base = cur.base;
        let done = &self.convs[0].history[base..];
        let common = done.iter().zip(&target).take_while(|(a, b)| a == b).count();
        if common < done.len() {
            let n = done.len();
            if let Err(e) = self.draft_rollback() {
                eprintln!("[draft] {e}");
                return json!({"ok": false, "reason": e});
            }
            eprintln!("[draft] edit: rolled back {n} tokens to {base}");
        }
        let cur = self.draft.cur.as_mut().expect("still typing");
        cur.target = target;
        json!({"ok": true, "prefilled": self.convs[0].history.len() - base,
               "target": self.draft.cur.as_ref().map_or(0, |c| c.target.len())})
    }

    /// Back to the conversation as it was before the turn: restore the
    /// checkpoint (clear in the first turn). `history` is cut to
    /// `base`; `pending` stays with the draft. A failed restore resets
    /// the conversation.
    fn draft_rollback(&mut self) -> Result<(), String> {
        let Some(cur) = self.draft.cur.as_ref().or(self.draft.taken.as_ref()) else { return Ok(()) };
        let base = cur.base;
        if base == 0 {
            self.ctx.memory_clear();
        } else if let Err(e) = self.ctx.restore_to(base as i32) {
            self.reset(0);
            return Err(format!("rollback failed: {e} (conversation reset)"));
        }
        self.convs[0].history.truncate(base);
        Ok(())
    }

    /// Tokens left to prefill, if a step is due (at least two: one
    /// would run on the decode path).
    fn draft_todo(&self) -> Option<usize> {
        let cur = self.draft.cur.as_ref()?;
        let left = cur.target.len().saturating_sub(self.convs[0].history.len() - cur.base);
        (left >= 2).then_some(left)
    }

    /// Prefill the next tokens of the target, at most `chunk`.
    fn draft_step(&mut self) {
        let Some(left) = self.draft_todo() else { return };
        let mut n = left.min(self.draft.chunk);
        if left - n == 1 {
            n -= 1; // keep two for the next step
        }
        let cur = self.draft.cur.as_mut().expect("draft_todo");
        if !cur.saved && cur.base > 0 {
            if let Err(e) = self.ctx.checkpoint_pos(cur.base as i32) {
                // Without a checkpoint an edit could not be undone.
                eprintln!("[draft] checkpoint failed: {e}; prefill while typing off");
                let cur = self.draft.cur.take().expect("typing");
                self.convs[0].pending = cur.pending;
                self.draft.on = false;
                return;
            }
            cur.saved = true;
        }
        let (base, from) = (cur.base, self.convs[0].history.len());
        let toks = cur.target[from - base..from - base + n].to_vec();
        if let Err(e) = self.eval(0, &toks, StageOutput::Discard) {
            eprintln!("[draft] prefill failed: {e}; rolling back");
            self.convs[0].history.truncate(from);
            if let Err(e) = self.draft_rollback() {
                eprintln!("[draft] {e}");
            }
            if let Some(cur) = self.draft.cur.take() {
                self.convs[0].pending = cur.pending;
            }
        }
    }

    /// The model thread's next request. While a draft has tokens to
    /// prefill, steps run between requests (a request waits at most one
    /// step); otherwise it blocks.
    pub(crate) fn next_request<T>(&mut self, rx: &Receiver<T>) -> Option<T> {
        loop {
            if self.draft_todo().is_none() {
                return rx.recv().ok();
            }
            match rx.try_recv() {
                Ok(m) => return Some(m),
                Err(TryRecvError::Empty) => self.draft_step(),
                Err(TryRecvError::Disconnected) => return None,
            }
        }
    }

    /// `begin`, first: put conversation `k`'s bookkeeping back to before
    /// the draft (the prefilled tokens stay in the KV cache until
    /// [`Self::draft_reuse`] or [`Self::draft_drop`]).
    pub(crate) fn draft_take(&mut self, k: usize) {
        self.draft.reused = None;
        if k != 0 {
            return;
        }
        if let Some(cur) = self.draft.cur.take() {
            let c = &mut self.convs[0];
            // Keep the prefilled tokens past `base` aside in `taken`.
            let done = c.history.split_off(cur.base);
            c.pending = cur.pending.clone();
            self.draft.taken = Some(Typing { target: done, ..cur });
        }
    }

    /// Right before the turn's prefill: how many leading tokens of
    /// `prompt` (the turn's whole token list; `None` for an image turn)
    /// are already in the KV cache. Rolls back what does not match.
    pub(crate) fn draft_reuse(&mut self, k: usize, prompt: Option<&[i32]>) -> Result<usize, String> {
        if k != 0 {
            return Ok(0);
        }
        let Some(t) = self.draft.taken.as_ref() else { return Ok(0) };
        let (base, saved, done) = (t.base, t.saved, t.target.len());
        let n = match prompt {
            Some(p) if p.len() >= done + 2 && p.starts_with(&t.target) => done,
            _ => 0,
        };
        if n > 0 {
            let t = self.draft.taken.take().expect("checked");
            self.convs[0].history.extend(t.target);
        } else if done > 0 {
            // The KV cache holds the draft past `history`: undo it.
            let r = self.draft_rollback();
            self.draft.taken = None;
            r?;
            eprintln!("[draft] sent message differs from the draft: rolled back {done} tokens");
        }
        self.draft.taken = None;
        if saved {
            self.ctx.forget_pos(base as i32);
        }
        self.draft.reused = Some(n);
        Ok(n)
    }

    /// `begin`, last: a turn that ended before its prefill (an error)
    /// rolls back what typing prefilled.
    pub(crate) fn draft_drop(&mut self, k: usize) {
        if k == 0 && self.draft.taken.is_some() {
            if let Err(e) = self.draft_reuse(0, None) {
                eprintln!("[draft] {e}");
            }
            self.draft.reused = None;
        }
    }
}
