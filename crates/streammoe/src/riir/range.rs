//! Layer ranges — running a contiguous slice of the model's layers in
//! one context (plan phase 3, mode B).
//!
//! A context opened with [`crate::riir::OpenConfig::layer_range`]
//! `= Some(a..b)` loads, plans and runs only layers `a..b`: their
//! weights, experts, KV / linear-attn state and weight caches. The
//! embedding exists only when `a == 0`, the final norm and LM head
//! only when `b == num_layers`. Hidden states cross the range
//! boundary as f32 rows of `hidden_dim` (see
//! [`crate::riir::RsCtx::eval_stage`]).
//!
//! Per-layer collections keep their *global* layer index at the use
//! sites through [`RangeVec`]: `states[layer_idx]` indexes by the
//! model layer, while the vector only holds the range's entries.

use std::ops::{Index, IndexMut, Range};

use crate::riir::variants::{LayerKind, VARIANT};

/// The whole model, `0..num_layers`.
pub fn full_range() -> Range<usize> {
    0..VARIANT.num_layers
}

/// `true` if `range` is the whole model.
pub fn is_full(range: &Range<usize>) -> bool {
    range.start == 0 && range.end == VARIANT.num_layers
}

/// Check `range` against the active variant. Returns a reason on
/// failure.
pub fn validate(range: &Range<usize>) -> Result<(), String> {
    if range.start >= range.end {
        return Err(format!("layer_range {range:?} is empty"));
    }
    if range.end > VARIANT.num_layers {
        return Err(format!(
            "layer_range {range:?} exceeds the model's {} layers",
            VARIANT.num_layers
        ));
    }
    Ok(())
}

/// Linear-attention layer indices (the `linear_layer_idx_for` space)
/// covered by `range`. Contiguous because the linear index is
/// monotonic in the layer index.
pub fn linear_span(range: &Range<usize>) -> Range<usize> {
    let before = |end: usize| {
        (0..end)
            .filter(|&i| VARIANT.layer_kind(i) == LayerKind::LinearAttn)
            .count()
    };
    before(range.start)..before(range.end)
}

/// Whether a weight-manifest tensor belongs to a context running
/// `range`. Layer tensors are kept for in-range layers only; the
/// embedding only on the first range, the final norm and LM head only
/// on the last. Any other top-level tensor is kept.
pub fn keeps_tensor(range: &Range<usize>, name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("model.layers.") {
        return rest
            .split('.')
            .next()
            .and_then(|n| n.parse::<usize>().ok())
            .is_some_and(|l| range.contains(&l));
    }
    if name.starts_with("model.embed_tokens.") {
        return range.start == 0;
    }
    if name.starts_with("model.norm.") || name.starts_with("lm_head.") {
        return range.end == VARIANT.num_layers;
    }
    true
}

/// A per-layer vector indexed by global layer (or linear-layer)
/// index, holding only the entries of one contiguous range.
///
/// Indexing outside the range panics — out-of-range layers have no
/// state, weights or buffers in this context, so reaching one is a
/// dispatch bug.
#[derive(Debug, Clone)]
pub struct RangeVec<T> {
    base: usize,
    items: Vec<T>,
}

impl<T> Default for RangeVec<T> {
    fn default() -> Self {
        Self { base: 0, items: Vec::new() }
    }
}

impl<T> RangeVec<T> {
    /// Entries for indices `base..base + items.len()`.
    pub fn new(base: usize, items: Vec<T>) -> Self {
        Self { base, items }
    }

    /// Build from `f(i)` for every `i` in `range`.
    pub fn from_fn(range: Range<usize>, f: impl FnMut(usize) -> T) -> Self {
        Self {
            base: range.start,
            items: range.map(f).collect(),
        }
    }

    /// First global index held.
    pub fn base(&self) -> usize {
        self.base
    }

    /// Global indices held.
    pub fn span(&self) -> Range<usize> {
        self.base..self.base + self.items.len()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Entry at global index `i`, if held.
    pub fn get(&self, i: usize) -> Option<&T> {
        self.items.get(i.checked_sub(self.base)?)
    }

    /// Mutable entry at global index `i`, if held.
    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        self.items.get_mut(i.checked_sub(self.base)?)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.items.iter_mut()
    }

    /// `(global index, entry)` pairs.
    pub fn iter_global(&self) -> impl Iterator<Item = (usize, &T)> {
        let base = self.base;
        self.items.iter().enumerate().map(move |(i, t)| (base + i, t))
    }

    pub fn as_slice(&self) -> &[T] {
        &self.items
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.items
    }
}

impl<T> Index<usize> for RangeVec<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.items[i - self.base]
    }
}

impl<T> IndexMut<usize> for RangeVec<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.items[i - self.base]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_vec_indexes_globally() {
        let v = RangeVec::from_fn(5..8, |i| i * 10);
        assert_eq!(v[5], 50);
        assert_eq!(v[7], 70);
        assert_eq!(v.get(4), None);
        assert_eq!(v.get(8), None);
        assert_eq!(v.span(), 5..8);
        let pairs: Vec<_> = v.iter_global().map(|(i, &x)| (i, x)).collect();
        assert_eq!(pairs, vec![(5, 50), (6, 60), (7, 70)]);
    }

    #[test]
    #[should_panic]
    fn range_vec_out_of_range_panics() {
        let v = RangeVec::from_fn(5..8, |i| i);
        let _ = v[2];
    }

    #[test]
    fn validate_rejects_empty_and_too_long() {
        let n = VARIANT.num_layers;
        assert!(validate(&(0..n)).is_ok());
        assert!(validate(&(1..1)).is_err());
        assert!(validate(&(0..n + 1)).is_err());
    }

    #[test]
    fn tensor_filter_by_range() {
        let n = VARIANT.num_layers;
        let first = 0..2;
        let last = 2..n;
        assert!(keeps_tensor(&first, "model.layers.1.mlp.gate.weight"));
        assert!(!keeps_tensor(&first, "model.layers.2.mlp.gate.weight"));
        assert!(!keeps_tensor(&first, "model.layers.12x.foo"));
        assert!(keeps_tensor(&first, "model.embed_tokens.weight"));
        assert!(!keeps_tensor(&first, "lm_head.weight"));
        assert!(!keeps_tensor(&first, "model.norm.weight"));
        assert!(!keeps_tensor(&last, "model.embed_tokens.scales"));
        assert!(keeps_tensor(&last, "lm_head.scales"));
        assert!(keeps_tensor(&last, "model.norm.weight"));
        assert!(keeps_tensor(&last, &format!("model.layers.{}.x", n - 1)));
    }

    /// The linear spans of a split partition the full span.
    #[test]
    fn linear_spans_partition() {
        let n = VARIANT.num_layers;
        let cuts = [0, n / 2, n * 3 / 4, n];
        let full = linear_span(&full_range());
        let mut at = full.start;
        for w in cuts.windows(2) {
            let s = linear_span(&(w[0]..w[1]));
            assert_eq!(s.start, at);
            at = s.end;
        }
        assert_eq!(at, full.end);
    }
}
