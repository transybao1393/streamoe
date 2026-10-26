//! A tiny stand-in model directory for tests that need `Ctx::open` to
//! succeed without real weights.
//!
//! The manifest carries just enough to pass the variant guard (one
//! `input_layernorm.weight` per layer, `hidden_dim` long) plus the
//! embedding / final norm / LM head names a layer-range filter keys
//! on. No expert files: missing layer files are tolerated at open.
//! Anything that builds GPU resources (an eval that gets past argument
//! checks, or a capped open) fails — these tests stop before that.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use streammoe::riir::VARIANT;

/// Paths of a fake model on disk; removed on drop.
pub struct FakeModel {
    pub dir: PathBuf,
    pub weights: PathBuf,
    pub manifest: PathBuf,
    pub experts: PathBuf,
}

impl FakeModel {
    /// Write a fake model under the temp dir, unique per `tag`.
    pub fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("streammoe-fake-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let weights = dir.join("model_weights.bin");
        let manifest = dir.join("model_weights.json");
        let hd = VARIANT.hidden_dim;
        let norm_bytes = hd * 2; // bf16
        let mut entries = Vec::new();
        let mut off = 0usize;
        let mut push = |name: String, size: usize, shape: String, dtype: &str| {
            entries.push(format!(
                r#""{name}": {{"offset": {off}, "size": {size}, "shape": {shape}, "dtype": "{dtype}"}}"#
            ));
            off += size;
        };
        push("model.embed_tokens.weight".into(), 64, "[16]".into(), "U32");
        for l in 0..VARIANT.num_layers {
            push(
                format!("model.layers.{l}.input_layernorm.weight"),
                norm_bytes,
                format!("[{hd}]"),
                "BF16",
            );
        }
        push("model.norm.weight".into(), norm_bytes, format!("[{hd}]"), "BF16");
        push("lm_head.weight".into(), 64, "[16]".into(), "U32");
        std::fs::write(&manifest, format!(r#"{{"tensors": {{{}}}}}"#, entries.join(",\n")))
            .unwrap();
        std::fs::File::create(&weights)
            .unwrap()
            .set_len(off as u64)
            .unwrap();
        Self {
            experts: dir.join("experts"),
            dir,
            weights,
            manifest,
        }
    }

    pub fn open(&self, cfg: &streammoe::OpenConfig) -> Result<streammoe::Ctx, streammoe::Error> {
        streammoe::Ctx::open_with(
            &self.weights,
            &self.manifest,
            Path::new("/nonexistent/vocab.bin"),
            &self.experts,
            false,
            cfg,
        )
    }
}

impl Drop for FakeModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
