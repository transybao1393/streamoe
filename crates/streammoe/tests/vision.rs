//! The CPU vision encoder against reference embeddings from
//! transformers' `Qwen3_5MoeVisionModel` (fp32) on the same resized
//! images.
//!
//! `STREAMMOE_VISION_DIR` holds `vision_weights.{bin,json}`
//! (`tools/export_vision.py`); `STREAMMOE_VISION_REF` holds, per image,
//! `<name>.rgb` (resized RGB bytes), `<name>.emb` (f32 `n × 2048`) and
//! `meta.json` (`{name: {h, w}}`).
#![cfg(all(target_os = "macos", feature = "model-qwen3-6-35b-a3b"))]

use std::path::PathBuf;

use streammoe::riir::vision::VisionEncoder;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt())
}

#[test]
#[ignore = "needs vision_weights.bin and reference embeddings; run scripts/check_test_host.sh first"]
fn encoder_matches_reference() {
    let dir = PathBuf::from(std::env::var("STREAMMOE_VISION_DIR").expect("STREAMMOE_VISION_DIR"));
    let refs = PathBuf::from(std::env::var("STREAMMOE_VISION_REF").expect("STREAMMOE_VISION_REF"));
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refs.join("meta.json")).unwrap()).unwrap();
    let mut enc = VisionEncoder::open(&dir, 448 * 512, 4).expect("open encoder");
    eprintln!("[vision] encoder holds {:.1} MiB", enc.bytes() as f64 / (1 << 20) as f64);
    for (name, m) in meta.as_object().unwrap() {
        let (h, w) = (m["h"].as_u64().unwrap() as usize, m["w"].as_u64().unwrap() as usize);
        let rgb = std::fs::read(refs.join(format!("{name}.rgb"))).unwrap();
        let want: Vec<f32> = std::fs::read(refs.join(format!("{name}.emb")))
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let t = std::time::Instant::now();
        let out = enc.encode(&rgb, h, w).expect("encode");
        let ms = t.elapsed().as_millis();
        assert_eq!(out.embeds.len(), want.len(), "{name}: token count");
        let d = enc.config().out_hidden_size;
        let total = cosine(&out.embeds, &want);
        let worst = out
            .embeds
            .chunks(d)
            .zip(want.chunks(d))
            .map(|(a, b)| cosine(a, b))
            .fold(1f64, f64::min);
        eprintln!(
            "[vision] {name}: {} tokens ({}x{}), {ms} ms, cos {total:.6}, worst row {worst:.6}",
            out.n_tokens, out.grid_h, out.grid_w
        );
        assert!(worst > 0.999, "{name}: worst row cosine {worst}");
    }
}
