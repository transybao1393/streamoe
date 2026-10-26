//! `streammoe-node`: serve one layer range of the model over TCP.
//!
//! ```text
//! streammoe-node --layers 0..20 --weights W --manifest M --vocab V --experts DIR
//!              [--listen 127.0.0.1:0] [--n-ctx 4096] [--mem-cap 10G]
//!              [--max-requests 1] [--max-tokens 128] [--hot-tier BYTES]
//! ```
//!
//! Frame buffers are allocated before the context opens and counted in
//! `--mem-cap`. Once the context is open the node prints
//! `LISTEN <addr>` on stdout and serves until a coordinator sends
//! `Shutdown`.

#[cfg(target_os = "macos")]
fn main() {
    if let Err(e) = run() {
        eprintln!("streammoe-node: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("streammoe-node: macOS only");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    use std::io::Write;
    use std::path::PathBuf;

    use streammoe_net::node::{CtxStage, NodeBuffers, serve};
    use streammoe_net::{parse_bytes, parse_range};

    let mut layers = None;
    let mut paths: [Option<PathBuf>; 4] = Default::default();
    let mut listen = "127.0.0.1:0".to_string();
    // A pipeline stage never takes checkpoints: plan no buffers for
    // them, so the cap goes to the hot tier.
    let mut cfg = streammoe::OpenConfig {
        n_ctx: 4096,
        max_checkpoints: 0,
        ..streammoe::OpenConfig::default()
    };
    let mut max_tokens = 128usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut val = || args.next().ok_or(format!("{flag} needs a value"));
        let bad = |v: &str| format!("bad value for {flag}: {v}");
        match flag.as_str() {
            "--layers" => {
                let v = val()?;
                layers = Some(parse_range(&v).ok_or_else(|| bad(&v))?);
            }
            "--weights" => paths[0] = Some(val()?.into()),
            "--manifest" => paths[1] = Some(val()?.into()),
            "--vocab" => paths[2] = Some(val()?.into()),
            "--experts" => paths[3] = Some(val()?.into()),
            "--listen" => listen = val()?,
            "--n-ctx" => {
                let v = val()?;
                cfg.n_ctx = v.parse().map_err(|_| bad(&v))?;
            }
            "--mem-cap" => {
                let v = val()?;
                cfg.mem_cap = Some(parse_bytes(&v).ok_or_else(|| bad(&v))?);
            }
            "--max-requests" => {
                let v = val()?;
                cfg.max_requests = v.parse().map_err(|_| bad(&v))?;
            }
            "--max-tokens" => {
                let v = val()?;
                max_tokens = v.parse().map_err(|_| bad(&v))?;
            }
            "--hot-tier" => {
                let v = val()?;
                cfg.hot_tier_bytes = Some(parse_bytes(&v).ok_or_else(|| bad(&v))?);
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let [Some(weights), Some(manifest), Some(vocab), Some(experts)] = paths else {
        return Err("--weights, --manifest, --vocab and --experts are required".into());
    };
    if max_tokens == 0 {
        return Err("--max-tokens must be ≥ 1".into());
    }
    cfg.layer_range = layers;

    let v = streammoe::riir::VARIANT;
    // Buffers first, so the capped open's footprint check covers them.
    let mut bufs = NodeBuffers::new(max_tokens, v.hidden_dim, v.vocab_size);
    let net_bytes = bufs.footprint();
    cfg.host_buffers = net_bytes;
    let ctx = streammoe::Ctx::open_with(&weights, &manifest, &vocab, &experts, false, &cfg)
        .map_err(|e| format!("open: {e}"))?;
    let mut stage = CtxStage { ctx, max_tokens, net_bytes };

    let listener = std::net::TcpListener::bind(&listen).map_err(|e| format!("bind {listen}: {e}"))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    println!("LISTEN {addr}");
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    serve(&mut stage, &listener, &mut bufs).map_err(|e| e.to_string())?;
    Ok(())
}
