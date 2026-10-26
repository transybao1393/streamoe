//! Pipeline over TCP loopback (plan phase 5).
//!
//! The first group runs without model files: nodes serving a
//! deterministic stand-in [`Stage`] on threads, driven by a
//! [`Chain`] over real loopback sockets, must reproduce the same
//! stages called in-process bit for bit; errors, admission and
//! shutdown cross the wire.
//!
//! The `#[ignore]`d test is the plan's phase 5 exit: three
//! `streammoe-node` processes (a3b layers 0..20, 20..30, 30..40) over
//! loopback match the same split run in one process, bit for bit, and
//! report per-node decode latency.
//!
//! ```text
//! scripts/check_test_host.sh
//! cargo test -p streammoe-net --features model-qwen3-6-35b-a3b \
//!     --test loopback -- --include-ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::net::TcpListener;
use std::thread::JoinHandle;

use streammoe_net::coordinator::Chain;
use streammoe_net::node::{Exit, Input, NodeBuffers, Stage, net_buffer_bytes, serve};
use streammoe_net::{NetError, NodeInfo};

const HD: usize = 8;
const VOCAB: usize = 32;
const LAYERS: u32 = 12;
/// Positions a middle [`Mock`] stage refuses.
const FAIL_POS: usize = 1000;

/// A stand-in stage: mixes input rows with per-request running state,
/// so wrong routing, wrong positions or lost state change the output.
#[derive(Clone)]
struct Mock {
    range: (u32, u32),
    max_requests: usize,
    state: HashMap<u64, f32>,
}

impl Mock {
    fn new(a: u32, b: u32, max_requests: usize) -> Self {
        let mut state = HashMap::new();
        state.insert(0, 0.0);
        Self { range: (a, b), max_requests, state }
    }
}

impl Stage for Mock {
    fn info(&self) -> NodeInfo {
        NodeInfo {
            layer_start: self.range.0,
            layer_end: self.range.1,
            num_layers: LAYERS,
            hidden_dim: HD as u32,
            vocab: VOCAB as u32,
            max_requests: self.max_requests as u32,
            max_tokens: 4,
            mem_cap: 0,
            net_bytes: net_buffer_bytes(4, HD, VOCAB),
        }
    }

    fn admit(&mut self, req: u64) -> Result<(), String> {
        if self.state.len() >= self.max_requests && !self.state.contains_key(&req) {
            return Err("pool full".into());
        }
        self.state.insert(req, 0.0);
        Ok(())
    }

    fn drop_request(&mut self, req: u64) -> Result<(), String> {
        self.state.remove(&req);
        if req == 0 {
            self.state.insert(0, 0.0);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.state.clear();
        self.state.insert(0, 0.0);
    }

    fn eval(&mut self, req: u64, input: Input<'_>, pos: usize, out: &mut [f32]) -> Result<(), String> {
        let st = self.state.get_mut(&req).ok_or(format!("unknown request {req}"))?;
        // Middle stages refuse far positions, so a chain can fail
        // part-way through.
        if self.range.0 != 0 && self.range.1 != LAYERS && pos >= FAIL_POS {
            return Err(format!("position {pos} too far"));
        }
        let rows: Vec<Vec<f32>> = match input {
            Input::Tokens(t) => t.iter().map(|&t| (0..HD).map(|d| t as f32 + d as f32 * 0.5).collect()).collect(),
            Input::Hidden(h) => h.chunks(HD).map(|r| r.to_vec()).collect(),
        };
        let layers = (self.range.1 - self.range.0) as f32;
        let mut last = vec![0f32; HD];
        for (i, row) in rows.iter().enumerate() {
            let p = (pos + i) as f32;
            for d in 0..HD {
                let v = (row[d] * 1.0001 + layers + 0.01 * p + *st).sin();
                *st += v * 1e-3;
                last[d] = v;
                if self.range.1 != LAYERS {
                    out[i * HD + d] = v;
                }
            }
        }
        if self.range.1 == LAYERS {
            for (k, o) in out.iter_mut().enumerate() {
                *o = last.iter().enumerate().map(|(d, &v)| v * ((k * HD + d) as f32).cos()).sum();
            }
        }
        Ok(())
    }
}

fn spawn(stage: Mock) -> (String, JoinHandle<Result<Exit, NetError>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let h = std::thread::spawn(move || {
        let mut stage = stage;
        let mut bufs = NodeBuffers::new(4, HD, VOCAB);
        serve(&mut stage, &listener, &mut bufs)
    });
    (addr, h)
}

fn split() -> Vec<Mock> {
    vec![Mock::new(0, 6, 2), Mock::new(6, 9, 2), Mock::new(9, LAYERS, 2)]
}

/// In-process reference: the same stages called directly.
fn local_eval(stages: &mut [Mock], req: u64, tokens: &[i32], pos: usize) -> Vec<f32> {
    let mut logits = vec![0f32; VOCAB];
    for (c, chunk) in tokens.chunks(4).enumerate() {
        let p = pos + c * 4;
        let mut cur: Vec<f32> = Vec::new();
        for (i, st) in stages.iter_mut().enumerate() {
            let input = if i == 0 { Input::Tokens(chunk) } else { Input::Hidden(&cur) };
            if st.range.1 == LAYERS {
                st.eval(req, input, p, &mut logits).unwrap();
            } else {
                let mut next = vec![0f32; chunk.len() * HD];
                st.eval(req, input, p, &mut next).unwrap();
                cur = next;
            }
        }
    }
    logits
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn chain_over_loopback_matches_in_process() {
    let mut local = split();
    let (addrs, handles): (Vec<_>, Vec<_>) = split().into_iter().map(spawn).unzip();
    let mut chain = Chain::connect(&addrs).expect("connect");
    assert_eq!(chain.max_tokens(), 4);
    assert_eq!(chain.vocab(), VOCAB);

    // A 10-token prompt crosses three frames per stage (4 + 4 + 2).
    let prompt: Vec<i32> = (0..10).map(|i| 3 + 7 * i).collect();
    let mut net = vec![0f32; VOCAB];
    chain.eval(0, &prompt, 0, &mut net).unwrap();
    let want = local_eval(&mut local, 0, &prompt, 0);
    assert_eq!(bits(&net), bits(&want));

    // Two requests decoded interleaved keep separate state.
    chain.admit(5).unwrap();
    local.iter_mut().for_each(|s| s.admit(5).unwrap());
    for step in 0..6 {
        for req in [0u64, 5] {
            let tok = 11 + step as i32 + req as i32;
            let pos = if req == 0 { prompt.len() + step } else { step };
            chain.eval(req, &[tok], pos, &mut net).unwrap();
            let want = local_eval(&mut local, req, &[tok], pos);
            assert_eq!(bits(&net), bits(&want), "req {req} step {step}");
        }
    }

    // Latency is measured per node.
    for st in chain.stats() {
        assert_eq!(st.decode_calls, 12);
        assert!(st.decode_latency().is_some());
        assert_eq!(st.rows, 10 + 12);
    }

    // Remote errors come back as errors; the connection stays usable.
    match chain.admit(6) {
        Err(NetError::Remote(m)) => assert!(m.contains("pool full"), "{m}"),
        other => panic!("expected pool full, got {other:?}"),
    }
    match chain.eval(42, &[1], 0, &mut net) {
        Err(NetError::Remote(m)) => assert!(m.contains("unknown request 42"), "{m}"),
        other => panic!("expected unknown request, got {other:?}"),
    }
    chain.drop_request(5).unwrap();
    chain.admit(6).unwrap();

    chain.shutdown().unwrap();
    for h in handles {
        assert_eq!(h.join().unwrap().unwrap(), Exit::Shutdown);
    }
}

#[test]
fn ranges_that_do_not_tile_the_model_are_refused() {
    let (a, ha) = spawn(Mock::new(0, 6, 1));
    let (b, hb) = spawn(Mock::new(7, LAYERS, 1));
    match Chain::connect(&[a.clone(), b.clone()]) {
        Err(NetError::Protocol(m)) => assert!(m.contains("starting at 6"), "{m}"),
        Ok(_) => panic!("gap accepted"),
        Err(e) => panic!("unexpected {e}"),
    }
    // Missing the tail.
    match Chain::connect(&[a.clone()]) {
        Err(NetError::Protocol(m)) => assert!(m.contains("ranges end at 6"), "{m}"),
        Ok(_) => panic!("partial chain accepted"),
        Err(e) => panic!("unexpected {e}"),
    }
    // The nodes survive refused sessions; stop them.
    let (c, hc) = spawn(Mock::new(6, 7, 1));
    let chain = Chain::connect(&[a, c, b]).expect("tiling chain");
    chain.shutdown().unwrap();
    for h in [ha, hb, hc] {
        assert_eq!(h.join().unwrap().unwrap(), Exit::Shutdown);
    }
}

/// A new coordinator connection starts from a clean pool: the same
/// prompt gives the same logits, and requests a previous coordinator
/// left admitted are gone.
#[test]
fn reconnecting_starts_from_a_clean_pool() {
    let (addrs, handles): (Vec<_>, Vec<_>) = split().into_iter().map(spawn).unzip();
    let prompt = [5, 9, 2];
    let mut first = vec![0f32; VOCAB];
    {
        let mut chain = Chain::connect(&addrs).expect("connect");
        chain.eval(0, &prompt, 0, &mut first).unwrap();
        chain.admit(5).unwrap();
        // Dropped without shutdown or cleanup, as a crash would.
    }
    let mut chain = Chain::connect(&addrs).expect("reconnect");
    let mut again = vec![0f32; VOCAB];
    chain.eval(0, &prompt, 0, &mut again).unwrap();
    assert_eq!(bits(&first), bits(&again));
    // The pool (2 slots) has room again: 5 was released.
    chain.admit(5).unwrap();
    chain.shutdown().unwrap();
    for h in handles {
        assert_eq!(h.join().unwrap().unwrap(), Exit::Shutdown);
    }
}

/// Failures leave no node holding part of a step or an admission.
#[test]
fn failures_do_not_leave_the_chain_half_applied() {
    // The middle node's pool is smaller than its neighbours'.
    let stages = vec![Mock::new(0, 6, 3), Mock::new(6, 9, 2), Mock::new(9, LAYERS, 3)];
    let (addrs, handles): (Vec<_>, Vec<_>) = stages.into_iter().map(spawn).unzip();
    let mut chain = Chain::connect(&addrs).expect("connect");
    let mut logits = vec![0f32; VOCAB];

    // Admission refused by node 1 is rolled back on node 0.
    chain.admit(5).unwrap();
    assert!(matches!(chain.admit(6), Err(NetError::Remote(_))));
    match chain.eval(6, &[1], 0, &mut logits) {
        Err(NetError::Remote(m)) => assert!(m.contains("unknown request 6"), "{m}"),
        other => panic!("node 0 kept request 6: {other:?}"),
    }

    // An eval failing at node 1 drops the request on every node, so
    // node 0 does not keep the step it already ran.
    match chain.eval(5, &[1], FAIL_POS, &mut logits) {
        Err(NetError::Remote(m)) => assert!(m.contains("too far"), "{m}"),
        other => panic!("expected the middle stage to fail, got {other:?}"),
    }
    match chain.eval(5, &[1], 0, &mut logits) {
        Err(NetError::Remote(m)) => assert!(m.contains("unknown request 5"), "{m}"),
        other => panic!("request 5 survived a failed step: {other:?}"),
    }

    // Empty input is refused before anything is sent ...
    assert!(matches!(chain.eval(0, &[], 0, &mut logits), Err(NetError::Protocol(_))));
    chain.shutdown().unwrap();
    for h in handles {
        assert_eq!(h.join().unwrap().unwrap(), Exit::Shutdown);
    }
}

/// ... and a node answers a zero-row frame with an error, not stale
/// logits.
#[test]
fn node_refuses_a_zero_row_frame() {
    let (addr, h) = spawn(Mock::new(0, LAYERS, 1));
    let mut s = std::net::TcpStream::connect(&addr).unwrap();
    let mut buf = streammoe_net::FrameBuf::new(4096);
    buf.read_reply(&mut s).unwrap(); // Hello
    buf.set_words::<i32>(streammoe_net::Kind::Tokens, 0, 0, 0, &[]).unwrap();
    buf.write_to(&mut s).unwrap();
    match buf.read_reply(&mut s) {
        Err(NetError::Remote(m)) => assert!(m.contains("no rows"), "{m}"),
        other => panic!("expected an error frame, got {other:?}"),
    }
    drop(s);
    let chain = Chain::connect(&[addr]).expect("node still serving");
    chain.shutdown().unwrap();
    assert_eq!(h.join().unwrap().unwrap(), Exit::Shutdown);
}

/// The node's buffers are what it reports and plans for.
#[test]
fn node_buffer_size_matches_plan() {
    let b = NodeBuffers::new(128, 2048, 248320);
    assert_eq!(b.footprint(), net_buffer_bytes(128, 2048, 248320));
}

#[cfg(feature = "model-qwen3-6-35b-a3b")]
mod model {
    use super::*;

    use std::io::{BufRead, BufReader};
    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};

    use streammoe::riir::VARIANT;
    use streammoe::{Ctx, OpenConfig, StageInput, StageOutput};

    const N_CTX: usize = 2048;
    const DECODE: usize = 12;
    const RANGES: [(usize, usize); 3] = [(0, 20), (20, 30), (30, 40)];

    fn artifacts() -> PathBuf {
        let d = "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts";
        PathBuf::from(std::env::var("STREAMMOE_SMOKE_ARTIFACTS").unwrap_or(d.into()))
    }

    fn root() -> PathBuf {
        let d = "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-root";
        PathBuf::from(std::env::var("STREAMMOE_SMOKE_ROOT").unwrap_or(d.into()))
    }

    fn prompt() -> Vec<i32> {
        (0..40).map(|i| 1000 + 13 * i).collect()
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

    fn spawn_node(a: usize, b: usize) -> (Child, String) {
        let art = artifacts();
        let mut child = Command::new(env!("CARGO_BIN_EXE_streammoe-node"))
            .args(["--layers", &format!("{a}..{b}"), "--n-ctx", &N_CTX.to_string()])
            .arg("--weights")
            .arg(art.join("model_weights.bin"))
            .arg("--manifest")
            .arg(art.join("model_weights.json"))
            .arg("--vocab")
            .arg(art.join("vocab.bin"))
            .arg("--experts")
            .arg(root())
            .args(["--max-tokens", "64"])
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn streammoe-node");
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
        let addr = line.trim().strip_prefix("LISTEN ").expect("LISTEN line").to_string();
        (child, addr)
    }

    /// Greedy decode through `eval`; logits per step.
    fn drive(mut eval: impl FnMut(&[i32], usize, &mut [f32])) -> Vec<Vec<f32>> {
        let prompt = prompt();
        let mut logits = vec![0f32; VARIANT.vocab_size];
        eval(&prompt, 0, &mut logits);
        let mut all = vec![logits.clone()];
        for i in 0..DECODE {
            let tok = argmax(&logits);
            eval(&[tok], prompt.len() + i, &mut logits);
            all.push(logits.clone());
        }
        all
    }

    /// Plan phase 5 exit: 3 processes over loopback match the
    /// in-process split; latency measured.
    #[test]
    #[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
    fn three_processes_over_loopback_match_in_process_split() {
        let (mut children, addrs): (Vec<_>, Vec<_>) = RANGES.iter().map(|&(a, b)| spawn_node(a, b)).unzip();
        let mut chain = Chain::connect(&addrs).expect("connect");
        let t0 = std::time::Instant::now();
        let net = drive(|t, p, out| chain.eval(0, t, p, out).expect("net eval"));
        let wall = t0.elapsed();
        for (info, st) in chain.infos().iter().zip(chain.stats()) {
            eprintln!(
                "[loopback] node {}..{}: decode round trip {:?} over {} calls",
                info.layer_start,
                info.layer_end,
                st.decode_latency().unwrap(),
                st.decode_calls
            );
        }
        eprintln!("[loopback] prompt + {DECODE} decode steps in {wall:?}");
        chain.shutdown().expect("shutdown");
        for c in &mut children {
            assert!(c.wait().unwrap().success());
        }

        // The same split in one process (after the nodes exit, to keep
        // peak memory to one copy of the model).
        let art = artifacts();
        let mut stages: Vec<Ctx> = RANGES
            .iter()
            .map(|&(a, b)| {
                let cfg = OpenConfig { n_ctx: N_CTX, layer_range: Some(a..b), ..OpenConfig::default() };
                Ctx::open_with(
                    &art.join("model_weights.bin"),
                    &art.join("model_weights.json"),
                    &art.join("vocab.bin"),
                    &root(),
                    false,
                    &cfg,
                )
                .expect("open")
            })
            .collect();
        let hd = VARIANT.hidden_dim;
        let local = drive(|toks, pos, out| {
            // Same 64-row chunking as the network path.
            for (c, chunk) in toks.chunks(64).enumerate() {
                let p = pos + c * 64;
                let mut cur = Vec::new();
                let last = stages.len() - 1;
                for (i, st) in stages.iter_mut().enumerate() {
                    let input = if i == 0 { StageInput::Tokens(chunk) } else { StageInput::Hidden(&cur) };
                    if i == last {
                        st.eval_stage(input, p, StageOutput::Logits(out)).unwrap();
                    } else {
                        let mut next = vec![0f32; chunk.len() * hd];
                        st.eval_stage(input, p, StageOutput::Hidden(&mut next)).unwrap();
                        cur = next;
                    }
                }
            }
        });
        for (step, (a, b)) in net.iter().zip(&local).enumerate() {
            assert_eq!(bits(a), bits(b), "logits differ at step {step}");
        }
    }
}
