//! `streammoe-coord`: plan layer ranges, or drive a chain of nodes.
//!
//! ```text
//! # Ranges for nodes given as CAP:SPEED (speed in layers/s, or 1 each)
//! streammoe-coord plan --manifest M [--n-ctx 4096] [--max-requests 1]
//!                    [--max-tokens 128] [--overhead 1G] --node 10G:2 --node 6G:1
//!
//! # Greedy decode through nodes (in layer order), with per-node latency
//! streammoe-coord run --node 127.0.0.1:7001 --node 127.0.0.1:7002 \
//!                   --prompt 1,2,3 [--decode 16] [--shutdown]
//! ```

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let res = match args.first().map(String::as_str) {
        Some("run") => run(args.split_off(1)),
        Some("plan") => plan(args.split_off(1)),
        _ => Err("usage: streammoe-coord (run|plan) …".into()),
    };
    if let Err(e) = res {
        eprintln!("streammoe-coord: {e}");
        std::process::exit(1);
    }
}

fn argmax(x: &[f32]) -> i32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as i32
}

fn run(args: Vec<String>) -> Result<(), String> {
    use streammoe_net::coordinator::Chain;
    use streammoe_net::planner::speed_from_latency;

    let mut nodes = Vec::new();
    let mut prompt = Vec::new();
    let mut decode = 16usize;
    let mut shutdown = false;
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--node" => nodes.push(val()?),
            "--prompt" => {
                prompt = val()?
                    .split(',')
                    .map(|t| t.trim().parse::<i32>().map_err(|_| format!("bad token {t}")))
                    .collect::<Result<_, _>>()?
            }
            "--decode" => decode = val()?.parse().map_err(|_| "bad --decode")?,
            "--shutdown" => shutdown = true,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if nodes.is_empty() || prompt.is_empty() {
        return Err("--node and --prompt are required".into());
    }
    let mut chain = Chain::connect(&nodes).map_err(|e| e.to_string())?;
    let mut logits = vec![0f32; chain.vocab()];
    let t0 = std::time::Instant::now();
    chain.eval(0, &prompt, 0, &mut logits).map_err(|e| e.to_string())?;
    let prefill = t0.elapsed();
    let mut toks = Vec::new();
    let t1 = std::time::Instant::now();
    for i in 0..decode {
        let tok = argmax(&logits);
        toks.push(tok);
        chain.eval(0, &[tok], prompt.len() + i, &mut logits).map_err(|e| e.to_string())?;
    }
    let dec = t1.elapsed();
    println!("tokens {toks:?}");
    println!("prefill {} tokens in {:.1} ms", prompt.len(), prefill.as_secs_f64() * 1e3);
    if decode > 0 {
        println!(
            "decode {decode} tokens: {:.2} ms/token ({:.2} tok/s)",
            dec.as_secs_f64() * 1e3 / decode as f64,
            decode as f64 / dec.as_secs_f64()
        );
    }
    for (info, st) in chain.infos().iter().zip(chain.stats()) {
        let layers = (info.layer_end - info.layer_start) as usize;
        match st.decode_latency() {
            Some(lat) => println!(
                "node {}..{}: decode round trip {:.3} ms, speed {:.1} layers/s",
                info.layer_start,
                info.layer_end,
                lat.as_secs_f64() * 1e3,
                speed_from_latency(layers, lat)
            ),
            None => println!("node {}..{}: {} calls", info.layer_start, info.layer_end, st.calls),
        }
    }
    if shutdown {
        chain.shutdown().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn plan(args: Vec<String>) -> Result<(), String> {
    use streammoe_net::parse_bytes;
    use streammoe_net::planner::{NodeSpec, costs_from_manifest, plan};

    let mut manifest = None;
    let (mut n_ctx, mut reqs, mut max_tokens) = (4096usize, 1usize, 128usize);
    let mut overhead = streammoe::OpenConfig::default().runtime_overhead;
    let mut nodes = Vec::new();
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--manifest" => manifest = Some(std::path::PathBuf::from(val()?)),
            "--n-ctx" => n_ctx = val()?.parse().map_err(|_| "bad --n-ctx")?,
            "--max-requests" => reqs = val()?.parse().map_err(|_| "bad --max-requests")?,
            "--max-tokens" => max_tokens = val()?.parse().map_err(|_| "bad --max-tokens")?,
            "--overhead" => overhead = parse_bytes(&val()?).ok_or("bad --overhead")?,
            "--node" => {
                let v = val()?;
                let (cap, speed) = v.split_once(':').unwrap_or((&v, "1"));
                nodes.push(NodeSpec {
                    mem_cap: parse_bytes(cap).ok_or(format!("bad cap in {v}"))?,
                    speed: speed.parse().map_err(|_| format!("bad speed in {v}"))?,
                });
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let manifest = manifest.ok_or("--manifest is required")?;
    let v = streammoe::riir::VARIANT;
    let net = streammoe_net::node::net_buffer_bytes(max_tokens, v.hidden_dim, v.vocab_size);
    let costs = costs_from_manifest(&manifest, n_ctx, reqs, overhead, net)?;
    let a = plan(&costs, &nodes).map_err(|e| e.to_string())?;
    for (k, r) in a.ranges.iter().enumerate() {
        println!(
            "node {k}: --layers {}..{}  planned {:.2} GiB of {:.2} GiB, {:.3} s/token-unit",
            r.start,
            r.end,
            a.bytes[k] as f64 / (1u64 << 30) as f64,
            nodes[k].mem_cap as f64 / (1u64 << 30) as f64,
            a.times[k]
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn plan(_: Vec<String>) -> Result<(), String> {
    Err("plan needs the model variant constants (macOS only)".into())
}
