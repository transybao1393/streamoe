//! Packed 3-bit experts against the same 3-bit codes in 4-bit nibbles.
//!
//! `tools/requant_experts.py` writes 3-bit codes in the 4-bit layout;
//! `tools/pack_experts.py` repacks them 8 per 3 bytes. The decode
//! kernel `dequant_matvec_3bit_v3_experts` keeps the 4-bit kernel's
//! lanes, element order and per-weight `fma`, so the K-expert decode
//! step must give bit-identical output in either layout. Synthetic
//! experts, no model files.
//!
//! ```bash
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b --test packed_3bit
//! ```

#![cfg(target_os = "macos")]

use streammoe::riir::backend::gpu::MetalBufferPool;
use streammoe::riir::moe::expert_forward::ExpertPayload;
use streammoe::riir::variants::{ExpertLayout, VARIANT};
use streammoe::riir::{MetalContext, MoeBuffers, gpu_batched_experts_forward};

/// xorshift64*, deterministic.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn bf16(x: f32) -> [u8; 2] {
    ((x.to_bits() >> 16) as u16).to_le_bytes()
}

/// One expert with random 3-bit codes: `(nibble layout, packed layout)`.
fn expert_pair(rng: &mut Rng) -> (Vec<u8>, Vec<u8>) {
    let (q4, q3) = (ExpertLayout::Q4, ExpertLayout::Q3);
    let words = q4.weight_bytes() / 4;
    let scale_bytes = VARIANT.expert_scale_bytes();
    let (mut a, mut b) = (Vec::with_capacity(q4.expert_size()), Vec::with_capacity(q3.expert_size()));
    for _block in 0..3 {
        for _ in 0..words {
            let w = rng.next() as u32 & 0x7777_7777;
            a.extend_from_slice(&w.to_le_bytes());
            let mut p = 0u32;
            for i in 0..8 {
                p |= ((w >> (4 * i)) & 0x7) << (3 * i);
            }
            b.extend_from_slice(&p.to_le_bytes()[..3]);
        }
        // scales (small, positive) then biases (around zero)
        let mut sb: Vec<u8> = (0..scale_bytes / 2).flat_map(|_| bf16(0.01 + 0.02 * rng.unit().abs())).collect();
        sb.extend((0..scale_bytes / 2).flat_map(|_| bf16(0.02 * rng.unit())));
        a.extend_from_slice(&sb);
        b.extend_from_slice(&sb);
    }
    assert_eq!((a.len(), b.len()), (q4.expert_size(), q3.expert_size()));
    (a, b)
}

#[test]
fn decode_experts_packed_3bit_bit_identical_to_nibbles() {
    let v = VARIANT;
    let k = v.num_experts_per_tok;
    let mut rng = Rng(0x3B17_DEC0_0001);
    let (mut nib, mut packed) = (Vec::new(), Vec::new());
    for _ in 0..k {
        let (a, b) = expert_pair(&mut rng);
        nib.extend(a);
        packed.extend(b);
    }
    let h_post: Vec<f32> = (0..v.hidden_dim).map(|_| rng.unit()).collect();
    let h_mid: Vec<f32> = (0..v.hidden_dim).map(|_| rng.unit()).collect();
    let shared_out: Vec<f32> = (0..v.hidden_dim).map(|_| 0.1 * rng.unit()).collect();
    let weights: Vec<f32> = (0..k).map(|i| 1.0 / (i + 2) as f32).collect();

    let mut metal = MetalContext::new().expect("open Metal");
    let mut pool = MetalBufferPool::new(metal.device().clone());
    let mut run = |layout: ExpertLayout, data: &[u8]| {
        let mut bufs = MoeBuffers::new_for(&mut pool, layout);
        let mut out = vec![0f32; v.hidden_dim];
        let payload = ExpertPayload {
            h_post: &h_post,
            h_mid: &h_mid,
            shared_out: &shared_out,
            expert_weights: &weights,
            shared_gate_score: 0.3,
        };
        gpu_batched_experts_forward(&mut metal, &mut bufs, &pool, k as i32, data, payload, &mut out)
            .expect("experts forward");
        out
    };
    let a = run(ExpertLayout::Q4, &nib);
    let b = run(ExpertLayout::Q3, &packed);
    assert!(a.iter().any(|&x| x != 0.0) && a.iter().all(|x| x.is_finite()));
    let differ = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert_eq!(differ, 0, "{differ} of {} outputs differ between layouts", a.len());
    // Packed reads a quarter fewer weight bytes.
    assert_eq!(packed.len() as f64 / nib.len() as f64, ExpertLayout::Q3.expert_size() as f64 / ExpertLayout::Q4.expert_size() as f64);
    eprintln!("[packed-3bit] decode: {} outputs bit-identical; {} vs {} bytes per expert",
        a.len(), ExpertLayout::Q3.expert_size(), ExpertLayout::Q4.expert_size());
}
