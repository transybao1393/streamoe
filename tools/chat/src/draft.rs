//! Prompt-lookup drafting for speculative decoding, shared by the chat
//! server (`--spec`, `spec.rs`) and `spec_bench`: look the last 3 (then
//! 2) tokens up earlier in the history, draft the tokens that followed,
//! and decide whether a draft is worth a verify pass. The caller
//! verifies `[last, drafts..]` with `Ctx::eval_tokens` and undoes the
//! rejected tail with `Ctx::rewind_tokens`.

/// Longest match length `lookup` reports; longer matches count as this.
pub const MATCH_CAP: usize = 64;

/// Prompt lookup: the tokens that followed the latest earlier match of
/// `hist`'s last 3 (else 2) tokens, at most `max`, and how many tokens
/// that match runs back (up to `MATCH_CAP`).
pub fn lookup(hist: &[i32], max: usize) -> (Vec<i32>, usize) {
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
pub struct Guard {
    /// Draft only when the lookup match is at least this long.
    pub min_match: usize,
    /// Also draft right after a pass that accepted its whole draft.
    pub after_full: bool,
    /// Plain steps to take after a pass that accepted nothing.
    pub backoff: usize,
}

impl Default for Guard {
    fn default() -> Self {
        Self { min_match: 8, after_full: false, backoff: 0 }
    }
}

/// Drafts one step at a time and counts what happened.
#[derive(Default)]
pub struct Drafter {
    pub max_draft: usize,
    /// `None` drafts on every lookup hit.
    pub guard: Option<Guard>,
    /// Match length behind the last `draft` call's lookup.
    pub last_match: usize,
    full: bool,
    cool: usize,
    /// Verify passes, plain steps, and plain steps the guard chose.
    pub passes: usize,
    pub singles: usize,
    pub skipped: usize,
    /// Draft tokens verified, and how many of them were kept.
    pub drafted: usize,
    pub accepted: usize,
}

impl Drafter {
    pub fn new(max_draft: usize, guard: Option<Guard>) -> Self {
        Self { max_draft, guard, ..Self::default() }
    }

    /// The draft to verify after `hist` (whose last token is the one
    /// about to be evaluated), at most `room` tokens. Empty means take
    /// a plain one-token step (counted here).
    pub fn draft(&mut self, hist: &[i32], room: usize) -> Vec<i32> {
        let (mut draft, len) = lookup(hist, self.max_draft);
        draft.truncate(room);
        self.last_match = len;
        let skip = match self.guard {
            Some(g) if !draft.is_empty() => {
                let skip = self.cool > 0 || (len < g.min_match && !(g.after_full && self.full));
                self.cool = self.cool.saturating_sub(1);
                skip
            }
            _ => false,
        };
        if draft.is_empty() || skip {
            self.singles += 1;
            self.skipped += skip as usize;
            self.full = false;
            draft.clear();
        }
        draft
    }

    /// Record a verify pass of `drafted` draft tokens that kept the
    /// first `accepted`.
    pub fn verified(&mut self, drafted: usize, accepted: usize) {
        self.full = accepted == drafted;
        if accepted == 0 {
            self.cool = self.guard.map_or(0, |g| g.backoff);
        }
        self.passes += 1;
        self.drafted += drafted;
        self.accepted += accepted;
    }
}

/// Index of the largest logit (greedy sampling).
pub fn argmax(v: &[f32]) -> i32 {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b }) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_drafts_what_followed_the_latest_match() {
        // ... 1 2 3 [4 5 6] ... 1 2 3 → draft 4 5 6, match of 3.
        let h = [9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3];
        assert_eq!(lookup(&h, 3), (vec![4, 5, 6], 3));
        assert_eq!(lookup(&h, 2), (vec![4, 5], 3));
        // A longer run back counts in the match length.
        let h = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(lookup(&h, 3), (vec![9, 10, 0], 8));
        // 2-gram fallback, and nothing to find.
        assert_eq!(lookup(&[5, 1, 2, 7, 3, 1, 2], 3), (vec![7, 3, 1], 2));
        assert_eq!(lookup(&[1, 2, 3, 4], 3), (vec![], 0));
    }

    #[test]
    fn guard_drafts_only_on_long_matches() {
        let mut d = Drafter::new(3, Some(Guard::default()));
        let short = [9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3];
        assert!(d.draft(&short, 3).is_empty());
        assert_eq!((d.singles, d.skipped), (1, 1));
        let long = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(d.draft(&long, 3), vec![9, 10, 0]);
        assert_eq!(d.draft(&long, 1), vec![9]);
        assert!(d.draft(&long, 0).is_empty());
        assert_eq!((d.singles, d.skipped), (2, 1));
        let mut old = Drafter::new(3, None);
        assert_eq!(old.draft(&short, 3), vec![4, 5, 6]);
    }
}
