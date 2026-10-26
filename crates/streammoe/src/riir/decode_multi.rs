//! Decode with T tokens: one forward pass over T consecutive tokens
//! that runs decode's own kernels and expert streaming, layer by layer,
//! with every token of the pass sharing each layer's command buffers
//! and expert reads.
//!
//! Per layer, all T tokens' attention + post-attention work goes into
//! one command buffer (full-attention layers: one for the q / k / v
//! projections, the host attention per token in order, then one for
//! the rest), so the GPU runs one longer burst instead of T short ones.
//! The routed experts of the T tokens are gathered once — hot tier,
//! prefetch, then one parallel read of the misses — and all T K-expert
//! dispatches go into one command buffer on the deferred ring. Each
//! token keeps decode's per-token kernels, order and combine, so its
//! logits are bit-identical to [`RsCtx::eval_token`] run T times.
//!
//! This is the verification half of speculative decoding: checking T
//! draft tokens costs one pass. Rolling back rejected drafts is the
//! caller's job (the linear-attention recurrence is not truncatable).

use rayon::prelude::*;

use super::attn::full_attn_forward::{encode_full_attn_cmd1_steps, full_attn_host_attention};
use super::attn::linear_attn_forward::{
    PostAttnIntermediates, alloc_recurrence, encode_linear_attn_cmd1_steps, encode_post_attention_steps,
    finish_post_attention, linear_layer_idx_for,
};
use super::backend::buftype::{ConvStateBuf, DeltaStateBuf, DeprecatedCogitoBuf};
use super::range::RangeVec;
use super::snapshot::state::{pos_max, truncate};
use super::backend::gpu::gpu_ctx::GpuLayerCtx;
use super::backend::gpu::gpu_norm::RmsNormBf16Pipelines;
use super::backend::{Backend as _, BufId, BufferPool as _, MetalBackend, MetalBufferPool};
use super::io::expert_io::ExpertFiles;
use super::io::prefetch::PrefetchState;
use super::moe::deferred::DeferredRing;
use super::moe::expert_forward::{ChainToNormed, MAX_K, MoeBuffers, encode_experts_mmap_into};
use super::variants::{self, RMS_NORM_EPS, VARIANT};
use super::requests::ReqId;
use super::snapshot::state::LayerStates;
use super::{
    LayerForwardBuffers, MetalContext, MtlWeightBuf, RsCtx, RsError, StageInput, StageOutput, full_kv_mut, io, moe,
};

/// Most tokens one [`RsCtx::eval_tokens`] pass takes.
pub const MAX_DECODE_TOKENS: usize = 4;

/// Buffers decode with T tokens needs beyond the single-token path's.
pub(crate) struct MultiDecodeBuffers {
    /// Step buffers for tokens `1..T`; token 0 uses `linear_buffers`.
    steps: Vec<LayerForwardBuffers>,
    /// Combine weights per token (18 f32s each; see
    /// [`encode_experts_mmap_into`]).
    params: Vec<BufId<DeprecatedCogitoBuf>>,
    /// Miss slots past `MoeBuffers`' `MAX_K` synced slots: T tokens can
    /// route to up to `T × K` distinct experts in a layer.
    extra_synced: Vec<BufId<DeprecatedCogitoBuf>>,
    /// Linear-attention recurrence after token `t` of the last pass,
    /// for `t < n - 1`: what [`RsCtx::rewind_tokens`] restores (~63 MB
    /// each on a3b).
    rollback: Vec<Recurrence>,
    /// `(start_pos, n)` of the last pass, until it is rewound or another
    /// eval runs.
    last_pass: Option<(usize, usize)>,
}

type Recurrence = (RangeVec<BufId<ConvStateBuf>>, RangeVec<BufId<DeltaStateBuf>>);

impl MultiDecodeBuffers {
    /// Everything a pass of up to `tokens` tokens needs, for the
    /// linear-attention layers in `range`. Sized exactly as
    /// [`super::budget::multi_decode_bytes`] plans it.
    pub(crate) fn new(
        pool: &mut MetalBufferPool,
        main: &LayerForwardBuffers,
        expert_size: usize,
        tokens: usize,
        range: &std::ops::Range<usize>,
    ) -> Self {
        const TWO_MIB: usize = 2 * 1024 * 1024;
        let steps = (1..tokens).map(|_| LayerForwardBuffers::new_step_set(pool, main)).collect();
        let params = (0..tokens)
            .map(|_| pool.alloc(18 * 4, "multi.combine_params", true).expect("pool.alloc multi.combine_params"))
            .collect();
        let extra = (tokens * VARIANT.num_experts_per_tok).saturating_sub(MAX_K);
        let extra_synced =
            (0..extra).map(|_| pool.alloc_aligned(expert_size, TWO_MIB, "multi.data_synced", true)).collect();
        let rollback = (1..tokens).map(|_| alloc_recurrence(pool, range)).collect();
        Self { steps, params, extra_synced, rollback, last_pass: None }
    }
}

/// Where one expert of the pass's union is read from.
#[derive(Clone, Copy, PartialEq)]
enum Source {
    Hot,
    Prefetched(usize),
    Synced(usize),
}

impl RsCtx<MetalBackend> {
    /// Decode `tokens` (at most [`MAX_DECODE_TOKENS`]) at positions
    /// `start_pos..start_pos + tokens.len()` in one forward pass and
    /// write every position's logits: `logits[t * vocab..(t + 1) *
    /// vocab]` are the logits after `tokens[t]`, exactly what
    /// [`Self::eval_token`] would give for it. One token is
    /// `eval_token`; MLA variants run tokenwise.
    pub fn eval_tokens(&mut self, tokens: &[i32], start_pos: usize, logits: &mut [f32]) -> Result<(), RsError> {
        let v = VARIANT;
        let n = tokens.len();
        if logits.len() != n * v.vocab_size {
            return Err(RsError::EvalFailed);
        }
        self.check_decode_tokens(n)?;
        if n <= 1 || matches!(v.attn_kind, variants::AttnKind::Mla) {
            for (t, (&tok, row)) in tokens.iter().zip(logits.chunks_mut(v.vocab_size)).enumerate() {
                self.eval_token(tok, start_pos + t, 0, row)?;
            }
            return Ok(());
        }
        self.require_full_range("eval_tokens")?;
        self.activate_slot(0);
        self.experts.begin_epoch();
        let t0 = std::time::Instant::now();
        let positions: Vec<usize> = (start_pos..start_pos + n).collect();
        self.decode_multi_inner(tokens, &positions, None, logits)?;
        if let Some(m) = self.multi_decode.as_mut() {
            m.last_pass = Some((start_pos, n));
        }
        let ns = t0.elapsed().as_nanos() as u64 / n as u64;
        for _ in 0..n {
            self.experts.counters().record_decode(ns);
        }
        self.check_cap("eval_tokens")
    }

    /// A pass of `n` tokens must fit `OpenConfig::max_decode_tokens`:
    /// its buffers are planned (and built) for that many.
    fn check_decode_tokens(&self, n: usize) -> Result<(), RsError> {
        if n > 1 && n > self.max_decode_tokens && !matches!(VARIANT.attn_kind, variants::AttnKind::Mla) {
            return Err(RsError::InvalidConfig {
                detail: format!(
                    "a {n}-token pass needs OpenConfig::max_decode_tokens >= {n} (it is {})",
                    self.max_decode_tokens
                ),
            });
        }
        Ok(())
    }

    /// Keep only the first `keep` tokens of the last [`Self::eval_tokens`]
    /// pass: the full-attention KV drops the rest and the
    /// linear-attention recurrence goes back to its state after token
    /// `keep - 1`. Speculative decoding calls this when a draft is
    /// rejected; the next eval continues at `start_pos + keep`. Must
    /// come right after the pass (any other eval in between, or `keep`
    /// outside `1..=n`, is an error). `keep == n` is a no-op.
    pub fn rewind_tokens(&mut self, keep: usize) -> Result<(), RsError> {
        let Some(multi) = self.multi_decode.as_mut() else {
            return Err(RsError::EvalFailed);
        };
        let Some((start_pos, n)) = multi.last_pass.take() else {
            return Err(RsError::EvalFailed);
        };
        // An eval since the pass moved the KV length on.
        if keep == 0 || keep > n || pos_max(self.layer_states.as_slice()) != (start_pos + n) as i32 {
            return Err(RsError::EvalFailed);
        }
        if keep == n {
            return Ok(());
        }
        truncate(self.layer_states.as_mut_slice(), (start_pos + keep) as i32, -1);
        let Self { backend, linear_buffers, multi_decode, .. } = self;
        let backend = backend.as_mut().ok_or(RsError::EvalFailed)?;
        let (metal, _, buffer_pool) = backend.parts_mut();
        let main = linear_buffers.as_ref().ok_or(RsError::EvalFailed)?;
        let (conv, delta) = &multi_decode.as_ref().expect("checked").rollback[keep - 1];
        let cmdbuf = metal.queue_clone().new_command_buffer().to_owned();
        let blit = cmdbuf.new_blit_command_encoder();
        for (src, dst) in conv.iter().zip(main.conv_state.iter()) {
            let (src, dst) = (buffer_pool.handle(*src), buffer_pool.handle(*dst));
            blit.copy_from_buffer(src, 0, dst, 0, src.length());
        }
        for (src, dst) in delta.iter().zip(main.delta_state.iter()) {
            let (src, dst) = (buffer_pool.handle(*src), buffer_pool.handle(*dst));
            blit.copy_from_buffer(src, 0, dst, 0, src.length());
        }
        blit.end_encoding();
        metal.commit_and_wait_labeled(&cmdbuf, "multi.rewind");
        Ok(())
    }

    /// Parallel decoding: one token from each of up to
    /// [`MAX_DECODE_TOKENS`] different requests, `batch[t] = (req,
    /// token, pos)`, in one forward pass. Every token runs on its own
    /// request's KV and recurrence; `logits[t * vocab..]` are what
    /// [`Self::eval_request`] would give for it alone. The requests must
    /// be admitted and distinct. One request is `eval_request`.
    pub fn eval_requests(&mut self, batch: &[(ReqId, i32, usize)], logits: &mut [f32]) -> Result<(), RsError> {
        let v = VARIANT;
        let n = batch.len();
        if n == 0 || logits.len() != n * v.vocab_size {
            return Err(RsError::EvalFailed);
        }
        self.check_decode_tokens(n)?;
        let mut slots = Vec::with_capacity(n);
        for &(req, _, _) in batch {
            let slot = self.requests.slot_of(req).ok_or(RsError::UnknownRequest { req: req.0 })?;
            if slots.contains(&slot) {
                return Err(RsError::RequestExists { req: req.0 });
            }
            slots.push(slot);
        }
        if n == 1 || matches!(v.attn_kind, variants::AttnKind::Mla) {
            for (&(req, tok, pos), row) in batch.iter().zip(logits.chunks_mut(v.vocab_size)) {
                self.eval_request(req, StageInput::Tokens(&[tok]), pos, StageOutput::Logits(row))?;
            }
            return Ok(());
        }
        self.require_full_range("eval_requests")?;
        self.activate_slot(slots[0]);
        self.experts.begin_epoch();
        let t0 = std::time::Instant::now();
        let tokens: Vec<i32> = batch.iter().map(|b| b.1).collect();
        let positions: Vec<usize> = batch.iter().map(|b| b.2).collect();
        self.decode_multi_inner(&tokens, &positions, Some(&slots), logits)?;
        let ns = t0.elapsed().as_nanos() as u64 / n as u64;
        for _ in 0..n {
            self.experts.counters().record_decode(ns);
        }
        self.check_cap("eval_requests")
    }

    /// One pass over `tokens` at `positions`. `slots: None` — one
    /// sequence (the active request), consecutive positions, recurrence
    /// kept after each token for [`Self::rewind_tokens`]. `Some` — token
    /// `t` belongs to pool slot `slots[t]` (`slots[0]` active, all
    /// distinct).
    fn decode_multi_inner(
        &mut self,
        tokens: &[i32],
        positions: &[usize],
        slots: Option<&[usize]>,
        logits: &mut [f32],
    ) -> Result<(), RsError> {
        let v = VARIANT;
        let n = tokens.len();
        self.ensure_linear_resources()?;
        if self.multi_decode.is_none() {
            let expert_size = self.experts.expert_size();
            let tokens = self.max_decode_tokens;
            let Self { backend, linear_buffers, multi_decode, layer_range, .. } = self;
            let pool = backend.as_mut().expect("ensure_linear_resources").pool_mut();
            let main = linear_buffers.as_ref().expect("ensure_linear_resources");
            *multi_decode = Some(MultiDecodeBuffers::new(pool, main, expert_size, tokens, layer_range));
        }
        self.multi_decode.as_mut().expect("just built").last_pass = None;
        self.lend_staging_to_hot_tier();
        let k_active = self.k_active;
        let expert_io_mode = self.expert_io_mode;
        let range = self.layer_range.clone();
        let rope: Vec<[i32; 3]> = positions.iter().map(|&p| self.rope_pos_at(p as i32)).collect();

        let Self {
            wf,
            backend,
            moe_buffers,
            experts,
            layer_states,
            layer_caches,
            linear_buffers,
            multi_decode,
            deferred,
            lm_head_gpu,
            io_pool,
            attn_pool,
            prefetch,
            parked,
            ..
        } = self;
        let backend = backend.as_mut().expect("ensure_linear_resources");
        let (metal, wf_buf, buffer_pool) = backend.parts_mut();
        let layer_caches = layer_caches.as_ref().expect("ensure_linear_resources");
        let main = linear_buffers.as_ref().expect("ensure_linear_resources");
        let moe_buffers = moe_buffers.as_mut().expect("ensure_linear_resources");
        let multi = multi_decode.as_mut().expect("just built");
        // Each token's recurrence: the live one (one sequence, or token
        // 0), else its parked request's. Refreshed every pass — a
        // request switch swaps them.
        for (t, s) in multi.steps.iter_mut().enumerate() {
            let (conv, delta) = match slots {
                Some(sl) if t + 1 < n => (&parked[sl[t + 1]].conv_state, &parked[sl[t + 1]].delta_state),
                _ => (&main.conv_state, &main.delta_state),
            };
            s.conv_state = conv.clone();
            s.delta_state = delta.clone();
        }
        // Each token's KV: `seq_states[seq_of(t)]`.
        let mut seq_states: Vec<&mut LayerStates> = vec![layer_states];
        if let Some(sl) = slots {
            let mut by_slot: Vec<Option<&mut LayerStates>> =
                parked.iter_mut().map(|p| Some(&mut p.layer_states)).collect();
            for &slot in &sl[1..] {
                seq_states.push(by_slot[slot].take().ok_or(RsError::EvalFailed)?);
            }
        }
        let seq_of = |t: usize| if slots.is_some() { t } else { 0 };
        let multi = &*multi;
        let bufs: Vec<&LayerForwardBuffers> = std::iter::once(main).chain(multi.steps.iter()).take(n).collect();
        let io_pool: &rayon::ThreadPool = &*io_pool;

        moe::deferred::discard_deferred_experts_in(deferred);
        for (t, &tok) in tokens.iter().enumerate() {
            // SAFETY: shared-storage buffer; no GPU work is in flight
            // (the ring was just drained).
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer_pool.handle(bufs[t].input).contents() as *mut f32,
                    v.hidden_dim,
                )
            };
            io::embedding::embed_lookup(wf, tok, dst).map_err(|_| RsError::EvalFailed)?;
        }

        let mut prev_layer_chained = false;
        for layer_idx in range.clone() {
            if deferred.is_full() {
                moe::deferred::complete_deferred_experts_chained(metal, deferred)
                    .map_err(|_| RsError::EvalFailed)?;
            }
            io::io_trace::mark('L', layer_idx);

            // Speculative prefetch for this layer, as the single-token
            // path does it (guessed from the pass's last token).
            let prefetch_set = layer_idx % 2;
            if prefetch.mode().is_pread() && !prefetch.in_flight_for(layer_idx) {
                let cross = prefetch.cross_mut().and_then(|c| c.guess(wf, layer_idx, k_active));
                if let Some(mut predicted) = cross.or_else(|| prefetch.predict_for(layer_idx)) {
                    io::prefetch::mask_hot(experts.hot_tier(), layer_idx, &mut predicted[..k_active]);
                    let data_prefetch = moe_buffers.data_prefetch_slots_mut_array(buffer_pool, prefetch_set);
                    prefetch.dispatch(layer_idx, predicted, k_active, data_prefetch, io_pool, experts);
                }
            }

            let chain_next = layer_idx + 1 < range.end;
            let chain_next_norm_off = chain_next.then(|| layer_caches[layer_idx + 1].input_layernorm_w);
            let layer_cache = &layer_caches[layer_idx];
            let ctx = |t: usize| GpuLayerCtx {
                wf,
                wf_buf,
                layer_cache,
                buffers: bufs[t],
                buffer_pool,
                expert_io_mode,
                attn_pool: attn_pool.as_ref(),
            };
            let fail = |_| RsError::EvalFailed;

            let queue = metal.queue_clone();
            let ctxs: Vec<GpuLayerCtx> = (0..n).map(ctx).collect();
            if v.layer_kind(layer_idx) == variants::LayerKind::FullAttn {
                let cmdbuf = queue.new_command_buffer();
                encode_full_attn_cmd1_steps(metal, cmdbuf, &ctxs, layer_idx, prev_layer_chained).map_err(fail)?;
                metal.commit_and_wait_labeled(cmdbuf, "multi.full_attn.cmd1");
                let mut args = Vec::with_capacity(n);
                let mut o_proj = None;
                for t in 0..n {
                    let kv_state = full_kv_mut(&mut *seq_states[seq_of(t)], layer_idx)?;
                    let (a, o) = full_attn_host_attention(&ctxs[t], layer_idx, rope[t], kv_state).map_err(fail)?;
                    args.push(a);
                    o_proj = Some(o);
                }
                let cmdbuf = queue.new_command_buffer();
                let o_proj = o_proj.ok_or(RsError::EvalFailed)?;
                encode_post_attention_steps(metal, cmdbuf, &ctxs, layer_idx, o_proj, &args).map_err(fail)?;
                metal.commit_and_wait_labeled(cmdbuf, "multi.post_attn");
            } else {
                let cmdbuf = queue.new_command_buffer();
                let lli = linear_layer_idx_for(layer_idx).ok_or(RsError::EvalFailed)?;
                // Keep the recurrence after each token but the last for
                // a rewind (one sequence only).
                let rollback: Vec<_> = match slots {
                    None => multi.rollback[..n - 1].iter().map(|(c, d)| (c[lli], d[lli])).collect(),
                    Some(_) => Vec::new(),
                };
                let o_proj = encode_linear_attn_cmd1_steps(metal, cmdbuf, &ctxs, layer_idx, prev_layer_chained, &rollback)
                    .map_err(fail)?;
                let none: Vec<_> = (0..n).map(|_| None).collect();
                encode_post_attention_steps(metal, cmdbuf, &ctxs, layer_idx, o_proj, &none).map_err(fail)?;
                metal.commit_and_wait_labeled(cmdbuf, "multi.post_attn");
            }
            let mut inters = Vec::with_capacity(n);
            for t in 0..n {
                inters.push(finish_post_attention(&ctx(t), k_active).map_err(fail)?);
            }

            moe_dispatch_multi(
                metal,
                wf_buf,
                &bufs,
                buffer_pool,
                moe_buffers,
                multi,
                deferred,
                layer_idx,
                experts,
                io_pool,
                prefetch,
                prefetch_set,
                &inters,
                chain_next_norm_off,
            )?;
            prev_layer_chained = chain_next;
        }

        // The last layer's dispatch copied each token's final hidden
        // state into its `input`; drain everything, then the LM head
        // per token exactly as `eval_token` runs it.
        while !deferred.is_empty() {
            moe::deferred::complete_deferred_experts_chained(metal, deferred).map_err(|_| RsError::EvalFailed)?;
        }
        let lm_head_gpu = lm_head_gpu.as_ref().ok_or(RsError::EvalFailed)?;
        let mut hidden = vec![0.0f32; v.hidden_dim];
        let mut normed = vec![0.0f32; v.hidden_dim];
        for (t, row) in logits.chunks_mut(v.vocab_size).enumerate() {
            // SAFETY: the dispatch that wrote it has completed.
            let src = unsafe {
                std::slice::from_raw_parts(buffer_pool.handle(bufs[t].input).contents() as *const f32, v.hidden_dim)
            };
            hidden.copy_from_slice(src);
            super::rms_norm_cpu(wf, "model.norm.weight", &hidden, &mut normed).map_err(|_| RsError::EvalFailed)?;
            lm_head_gpu.forward(metal, wf_buf, &normed, row).map_err(|_| RsError::EvalFailed)?;
        }
        Ok(())
    }
}

/// The routed experts of all T tokens of one layer: gather the union
/// once (hot tier, this layer's prefetch, one parallel read of the
/// rest), then encode every token's K-expert dispatch + combine (+ the
/// chained next-layer norm) into one command buffer on the ring.
#[allow(clippy::too_many_arguments)]
fn moe_dispatch_multi(
    metal: &mut MetalContext,
    wf_buf: &MtlWeightBuf,
    bufs: &[&LayerForwardBuffers],
    buffer_pool: &MetalBufferPool,
    moe: &mut MoeBuffers,
    multi: &MultiDecodeBuffers,
    deferred: &mut DeferredRing,
    layer_idx: usize,
    experts: &ExpertFiles,
    pool: &rayon::ThreadPool,
    prefetch: &mut PrefetchState,
    prefetch_set: usize,
    inters: &[PostAttnIntermediates],
    chain_next_norm_off: Option<u64>,
) -> Result<(), RsError> {
    let v = VARIANT;
    let n = inters.len();
    let k = inters[0].routing_indices.len();
    for it in inters {
        for &idx in &it.routing_indices {
            experts.record_pick(layer_idx, idx as usize, 1);
        }
        io::routing_trace::record(layer_idx, &it.routing_indices);
    }
    // Distinct experts of the pass, first-seen order.
    let mut union: Vec<i32> = Vec::with_capacity(n * k);
    for it in inters {
        for &idx in &it.routing_indices {
            if !union.contains(&idx) {
                union.push(idx);
            }
        }
    }

    let binding: Vec<(&metal::Buffer, u64)> = if prefetch.mode().is_mmap() {
        union
            .iter()
            .map(|&e| experts.mmap_buffer_for_expert(layer_idx, e as u32).expect("mmap buffer missing for expert"))
            .collect()
    } else {
        let hot: Vec<Option<(&metal::Buffer, u64)>> = union
            .iter()
            .map(|&e| experts.hot_tier().and_then(|h| h.lookup(layer_idx, e as usize)))
            .collect();
        let io_t0 = std::time::Instant::now();
        io::io_trace::mark('R', layer_idx);
        let status = prefetch.begin_layer(layer_idx);
        let mut source = vec![Source::Hot; union.len()];
        let mut synced = 0usize;
        for (u, &e) in union.iter().enumerate() {
            if hot[u].is_some() {
                continue;
            }
            let pre = status.as_ref().and_then(|s| (0..s.k).find(|&b| s.loaded_indices[b] == e));
            source[u] = match pre {
                Some(b) => Source::Prefetched(b),
                None => {
                    synced += 1;
                    Source::Synced(synced - 1)
                }
            };
        }
        let capacity = MAX_K + multi.extra_synced.len();
        // Synced slot `s`: the first MAX_K live in `MoeBuffers`, the rest
        // in `extra_synced`. A failed prefetch slot takes the next one.
        let slot_id = |s: usize| if s < MAX_K { moe.data_synced_id(s) } else { multi.extra_synced[s - MAX_K] };
        let slot_bytes = |s: usize| -> &mut [u8] {
            let b = buffer_pool.handle(slot_id(s));
            // SAFETY: each synced slot is written by exactly one read
            // below, and no GPU work reading it is in flight (the
            // previous layer's dispatch that read it was drained).
            unsafe { std::slice::from_raw_parts_mut(b.contents() as *mut u8, experts.expert_size()) }
        };
        let mut reads: Vec<(usize, &mut [u8])> = Vec::new();
        for (u, s) in source.iter().enumerate() {
            if let Source::Synced(s) = *s {
                if s >= capacity {
                    return Err(RsError::EvalFailed);
                }
                reads.push((union[u] as usize, slot_bytes(s)));
            }
        }
        pool.install(|| reads.par_iter_mut().try_for_each(|(e, dst)| experts.read_expert_disk(layer_idx, *e, dst)))
            .map_err(|_| RsError::EvalFailed)?;
        for u in 0..union.len() {
            if let Source::Prefetched(b) = source[u] {
                if !prefetch.wait_slot(layer_idx, b) {
                    if synced >= capacity {
                        return Err(RsError::EvalFailed);
                    }
                    experts
                        .read_expert_disk(layer_idx, union[u] as usize, slot_bytes(synced))
                        .map_err(|_| RsError::EvalFailed)?;
                    source[u] = Source::Synced(synced);
                    synced += 1;
                }
            }
        }
        // Telemetry per routed pick, as T single-token steps count it.
        let (mut hot_n, mut pre_n) = (0u64, 0u64);
        for it in inters {
            for idx in &it.routing_indices {
                match source[union.iter().position(|e| e == idx).unwrap()] {
                    Source::Hot => hot_n += 1,
                    Source::Prefetched(_) => pre_n += 1,
                    Source::Synced(_) => {}
                }
            }
        }
        prefetch.record_outcome(pre_n, (n * k) as u64 - pre_n - hot_n);
        experts.counters().record_decode_layer(layer_idx, hot_n, pre_n);
        experts.counters().record_io_wait(io_t0.elapsed().as_nanos() as u64);
        io::io_trace::mark('W', layer_idx);

        // The next layer's cross-layer guess and the next pass's
        // previous-token guess come from the pass's last token.
        let last = &inters[n - 1];
        if let Some(c) = prefetch.cross_mut() {
            if c.can_guess(layer_idx + 1) {
                // SAFETY: shared-storage buffer of `hidden_dim` f32s;
                // the command buffer that wrote it has completed.
                let h_mid = unsafe {
                    std::slice::from_raw_parts(
                        buffer_pool.handle(bufs[n - 1].h_mid).contents() as *const f32,
                        v.hidden_dim,
                    )
                };
                c.snapshot(layer_idx, h_mid);
            }
            let _ = c.take_guess_for(layer_idx);
        }
        let mut order: [usize; MAX_K] = std::array::from_fn(|i| i);
        order[..k].sort_by(|&a, &b| last.routing_weights[b].total_cmp(&last.routing_weights[a]));
        let mut actuals = [0i32; MAX_K];
        for (dst, &i) in actuals[..k].iter_mut().zip(&order[..k]) {
            *dst = last.routing_indices[i];
        }
        prefetch.record_actual(layer_idx, actuals);

        union
            .iter()
            .enumerate()
            .map(|(u, _)| match source[u] {
                Source::Hot => hot[u].expect("hot source has a binding"),
                Source::Prefetched(b) => (buffer_pool.handle(moe.data_prefetch_id(prefetch_set, b)), 0),
                Source::Synced(s) => (buffer_pool.handle(slot_id(s)), 0),
            })
            .collect()
    };

    let chain_pipes = match chain_next_norm_off {
        Some(_) => Some(RmsNormBf16Pipelines {
            sum: metal.pipeline("rms_norm_sum_sq").map_err(|_| RsError::EvalFailed)?.clone(),
            apply: metal.pipeline("rms_norm_apply_bf16").map_err(|_| RsError::EvalFailed)?.clone(),
        }),
        None => None,
    };
    let cmdbuf = metal.queue().new_command_buffer().to_owned();
    for (t, it) in inters.iter().enumerate() {
        let b = bufs[t];
        let bindings: Vec<(&metal::Buffer, u64)> = it
            .routing_indices
            .iter()
            .map(|idx| binding[union.iter().position(|e| e == idx).unwrap()])
            .collect();
        let chain = chain_next_norm_off.zip(chain_pipes.as_ref()).map(|(off, pipes)| ChainToNormed {
            pipes,
            wf_buf: wf_buf.buffer(),
            next_norm_off: off,
            combine_out: buffer_pool.handle(b.input),
            chain_sum_sq: buffer_pool.handle(b.sum_sq),
            chain_normed: buffer_pool.handle(b.normed),
            eps: RMS_NORM_EPS,
        });
        let last_layer = chain.is_none();
        encode_experts_mmap_into(
            &cmdbuf,
            metal,
            moe,
            buffer_pool,
            buffer_pool.handle(multi.params[t]),
            k as i32,
            buffer_pool.handle(b.normed),
            buffer_pool.handle(b.h_mid),
            buffer_pool.handle(b.shared_out),
            &it.routing_weights,
            it.shared_gate_score,
            &bindings,
            chain,
        )
        .map_err(|_| RsError::EvalFailed)?;
        if last_layer {
            // Unchained, the combine wrote the shared `moe_hidden`; keep
            // this token's copy before the next token's dispatch.
            let blit = cmdbuf.new_blit_command_encoder();
            blit.copy_from_buffer(
                buffer_pool.handle(moe.moe_hidden_id()),
                0,
                buffer_pool.handle(b.input),
                0,
                (v.hidden_dim * 4) as u64,
            );
            blit.end_encoding();
        }
    }
    cmdbuf.commit();
    deferred.push_gpu(cmdbuf, layer_idx as i32, k).map_err(|_| RsError::EvalFailed)
}
