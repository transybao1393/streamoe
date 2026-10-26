//! Decode (one-token) 4-bit matvec: qmv vs the kernels it would replace.
//!
//! The GPU timeline trace (`STREAMMOE_GPU_TIMING`) showed decode GPU time
//! dominated by the 4-bit matvec family running at ~20–25 GB/s on an
//! M3 Pro (150 GB/s). This bench checks each kernel against an f64 CPU
//! reference and times it at a3b decode shapes, reporting achieved
//! weight bandwidth. Synthetic weights, no model files.
//!
//! ```bash
//! cargo test -p streammoe --features model-qwen3-6-35b-a3b --release \
//!     --test matvec_decode_bench -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::time::Instant;

use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize, NSUInteger};
use streammoe::riir::MetalContext;

const GROUP_SIZE: usize = 64;

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(((b >> 16) & 1) + 0x7fff) >> 16) as u16
}
fn from_bf16(h: u16) -> f64 {
    f32::from_bits((h as u32) << 16) as f64
}

fn shared_buf(metal: &MetalContext, bytes: usize) -> Buffer {
    metal.device().new_buffer(bytes.max(16) as NSUInteger, MTLResourceOptions::StorageModeShared)
}
fn write<T: Copy>(buf: &Buffer, off: usize, data: &[T]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            (buf.contents() as *mut u8).add(off),
            std::mem::size_of_val(data),
        )
    }
}
fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
}

struct Mat {
    out_dim: usize,
    in_dim: usize,
    packed: Vec<u32>,
    scales: Vec<u16>,
    biases: Vec<u16>,
    buf: Buffer,
    s_off: u64,
    b_off: u64,
}

fn make_mat(metal: &MetalContext, rng: &mut Rng, out_dim: usize, in_dim: usize) -> Mat {
    let groups = in_dim / GROUP_SIZE;
    let packed: Vec<u32> = (0..out_dim * in_dim / 8).map(|_| rng.u64() as u32).collect();
    let scales: Vec<u16> = (0..out_dim * groups).map(|_| bf16(rng.f32() * 0.05)).collect();
    let biases: Vec<u16> = (0..out_dim * groups).map(|_| bf16(rng.f32() * 0.02)).collect();
    let (w, s) = (packed.len() * 4, scales.len() * 2);
    // Scales / biases start 16-byte aligned, like tensors in the weight file.
    let s_off = w.next_multiple_of(16);
    let b_off = (s_off + s).next_multiple_of(16);
    let buf = shared_buf(metal, b_off + biases.len() * 2);
    write(&buf, 0, &packed);
    write(&buf, s_off, &scales);
    write(&buf, b_off, &biases);
    Mat { out_dim, in_dim, packed, scales, biases, buf, s_off: s_off as u64, b_off: b_off as u64 }
}

fn reference_row(m: &Mat, x: &[f32], row: usize) -> f64 {
    let (pc, groups) = (m.in_dim / 8, m.in_dim / GROUP_SIZE);
    let mut acc = 0.0f64;
    for i in 0..m.in_dim {
        let q = (m.packed[row * pc + i / 8] >> (4 * (i % 8))) & 0xF;
        let g = row * groups + i / GROUP_SIZE;
        acc += (q as f64 * from_bf16(m.scales[g]) + from_bf16(m.biases[g])) * x[i] as f64;
    }
    acc
}

#[derive(Clone, Copy)]
enum Kind {
    /// 256 threads, 8 rows per threadgroup (v3 layout).
    V3,
    /// 64 threads, one row per threadgroup (`fast` layout).
    Fast,
    /// 64 threads, 8 rows per threadgroup (2 SIMD groups × 4 rows).
    Qmv,
}

fn encode(
    cmd: &metal::CommandBufferRef,
    pso: &ComputePipelineState,
    kind: Kind,
    m: &Mat,
    x: &Buffer,
    out: &Buffer,
) {
    let (o, i, g) = (m.out_dim as u32, m.in_dim as u32, GROUP_SIZE as u32);
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(&m.buf), 0);
    enc.set_buffer(1, Some(&m.buf), m.s_off);
    enc.set_buffer(2, Some(&m.buf), m.b_off);
    enc.set_buffer(3, Some(x), 0);
    enc.set_buffer(4, Some(out), 0);
    enc.set_bytes(5, 4, (&o as *const u32).cast());
    enc.set_bytes(6, 4, (&i as *const u32).cast());
    enc.set_bytes(7, 4, (&g as *const u32).cast());
    let (tgs, threads) = match kind {
        Kind::V3 => (o.div_ceil(8), 256),
        Kind::Fast => (o, 64),
        Kind::Qmv => (o.div_ceil(8), 64),
    };
    enc.dispatch_thread_groups(MTLSize::new(tgs as u64, 1, 1), MTLSize::new(threads, 1, 1));
    enc.end_encoding();
}

/// Max error over checked rows, relative to the row's magnitude scale.
fn check(m: &Mat, x: &[f32], got: &[f32]) -> f64 {
    let step = (m.out_dim / 2048).max(1);
    let mut worst = 0.0f64;
    for row in (0..m.out_dim).step_by(step) {
        let want = reference_row(m, x, row);
        let scale = want.abs().max(1.0);
        worst = worst.max((got[row] as f64 - want).abs() / scale);
    }
    worst
}

fn time_ms(metal: &MetalContext, pso: &ComputePipelineState, kind: Kind, m: &Mat, x: &Buffer, out: &Buffer) -> f64 {
    let run = |k: u32| {
        let cmd = metal.queue().new_command_buffer();
        for _ in 0..k {
            encode(cmd, pso, kind, m, x, out);
        }
        let t = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        t.elapsed().as_secs_f64() * 1e3
    };
    run(1);
    let k = ((200.0 / run(1)).round() as u32).clamp(1, 256);
    let mut v: Vec<f64> = (0..5).map(|_| run(k) / k as f64).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[2]
}

#[test]
#[ignore = "GPU microbench"]
fn matvec_decode_qmv_vs_v3() {
    let mut metal = MetalContext::new().expect("Metal");
    // (name, dispatch layout). v4 / v5 are earlier experiments in v3's
    // layout; production uses v3 up to in_dim 4096 and `fast` above.
    let kernels = [
        ("dequant_matvec_4bit_v3", Kind::V3),
        ("dequant_matvec_4bit_fast", Kind::Fast),
        ("dequant_matvec_4bit_v4", Kind::V3),
        ("dequant_matvec_4bit_v5", Kind::V3),
        ("dequant_matvec_4bit_qmv", Kind::Qmv),
    ];
    let psos: Vec<ComputePipelineState> =
        kernels.iter().map(|(n, _)| metal.pipeline(n).expect(n).to_owned()).collect();
    // (out_dim, in_dim, name) at a3b decode shapes.
    let shapes = [
        (248_320, 2048, "lm_head"),
        (12_288, 2048, "linear qkv_proj"),
        (8192, 2048, "linear z_proj"),
        (2048, 4096, "linear out_proj"),
        (512, 2048, "shared/expert gate|up"),
        (2048, 512, "shared/expert down"),
    ];
    let mut rng = Rng(0x5eed_4b17);
    eprint!("\n{:22}", "GB/s (max rel err)");
    for (n, _) in &kernels {
        eprint!(" {:>17}", n.trim_start_matches("dequant_matvec_4bit_"));
    }
    eprintln!();
    for (out_dim, in_dim, name) in shapes {
        let m = make_mat(&metal, &mut rng, out_dim, in_dim);
        let xv: Vec<f32> = (0..in_dim).map(|_| rng.f32()).collect();
        let x = shared_buf(&metal, in_dim * 4);
        write(&x, 0, &xv);
        let out = shared_buf(&metal, out_dim * 4);
        let bytes = (m.packed.len() * 4 + m.scales.len() * 4) as f64;
        eprint!("{:22}", name);
        let mut qmv_err = 0.0;
        for (pso, (kname, kind)) in psos.iter().zip(&kernels) {
            // v3 / v4 / v5 stage x in 4096 floats of threadgroup memory.
            if in_dim > 4096 && !matches!(kind, Kind::Fast | Kind::Qmv) {
                eprint!(" {:>17}", "n/a");
                continue;
            }
            let cmd = metal.queue().new_command_buffer();
            encode(cmd, pso, *kind, &m, &x, &out);
            cmd.commit();
            cmd.wait_until_completed();
            let err = check(&m, &xv, &read_f32(&out, out_dim));
            let ms = time_ms(&metal, pso, *kind, &m, &x, &out);
            eprint!(" {:>7.1} ({:>7.1e})", bytes / ms / 1e6, err);
            if kname.ends_with("qmv") {
                qmv_err = err;
            }
        }
        eprintln!();
        assert!(qmv_err < 1e-4, "{name}: qmv error {qmv_err:.3e}");
    }
}

/// Is the decode slowdown the GPU clock? Submit the LM-head matvec one
/// dispatch per command buffer with idle gaps between submissions, as
/// decode does, and read the GPU's own execution time for each.
#[test]
#[ignore = "GPU microbench"]
fn matvec_gpu_time_vs_idle_gap() {
    use streammoe::riir::backend::gpu::gpu_timing::gpu_span;
    let mut metal = MetalContext::new().expect("Metal");
    let pso = metal.pipeline("dequant_matvec_4bit_v3").expect("v3").to_owned();
    let mut rng = Rng(0x5eed_9a95);
    let m = make_mat(&metal, &mut rng, 248_320, 2048);
    let xv: Vec<f32> = (0..2048).map(|_| rng.f32()).collect();
    let x = shared_buf(&metal, 2048 * 4);
    write(&x, 0, &xv);
    let out = shared_buf(&metal, 248_320 * 4);
    eprintln!("\nidle gap before each submit -> LM-head GPU time (median of 40)");
    for gap_us in [0u64, 250, 500, 1000, 2000, 5000] {
        let mut v = Vec::new();
        for _ in 0..40 {
            if gap_us > 0 {
                std::thread::sleep(std::time::Duration::from_micros(gap_us));
            }
            let cmd = metal.queue().new_command_buffer();
            encode(cmd, &pso, Kind::V3, &m, &x, &out);
            cmd.commit();
            cmd.wait_until_completed();
            let (s, e) = gpu_span(cmd);
            v.push((e - s) * 1e3);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("  gap {:>5} us: {:6.2} ms", gap_us, v[20]);
    }
}

/// Same LM-head matvec, weights in a Metal allocation vs in an mmap'd
/// file wrapped with `newBufferWithBytesNoCopy` (how the engine holds
/// `model_weights.bin`). Separates "the GPU reads file-backed pages
/// slowly" from "the pages were evicted".
#[test]
#[ignore = "GPU microbench"]
fn matvec_metal_buffer_vs_mmap_file() {
    use streammoe::riir::backend::gpu::gpu_timing::gpu_span;
    use std::io::Write as _;
    use std::os::unix::io::AsRawFd;
    let mut metal = MetalContext::new().expect("Metal");
    let pso = metal.pipeline("dequant_matvec_4bit_v3").expect("v3").to_owned();
    let mut rng = Rng(0x5eed_3a7f);
    let m = make_mat(&metal, &mut rng, 248_320, 2048);
    let len = m.buf.length() as usize;
    // Write the same bytes to a file and map it like WeightFile does.
    let path = std::env::temp_dir().join(format!("streammoe-mmap-bench-{}.bin", std::process::id()));
    {
        let mut f = std::fs::File::create(&path).unwrap();
        let bytes = unsafe { std::slice::from_raw_parts(m.buf.contents() as *const u8, len) };
        f.write_all(bytes).unwrap();
    }
    let f = std::fs::File::open(&path).unwrap();
    let page = 16384usize;
    let map_len = len.next_multiple_of(page);
    let ptr = unsafe {
        libc::mmap(std::ptr::null_mut(), map_len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0)
    };
    assert_ne!(ptr, libc::MAP_FAILED);
    let file_buf = metal.device().new_buffer_with_bytes_no_copy(
        ptr,
        map_len as NSUInteger,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let xv: Vec<f32> = (0..2048).map(|_| rng.f32()).collect();
    let x = shared_buf(&metal, 2048 * 4);
    write(&x, 0, &xv);
    let out = shared_buf(&metal, 248_320 * 4);
    let run = |w: &Buffer| {
        let cmd = metal.queue().new_command_buffer();
        let (o, i, g) = (248_320u32, 2048u32, GROUP_SIZE as u32);
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(w), m.s_off);
        enc.set_buffer(2, Some(w), m.b_off);
        enc.set_buffer(3, Some(&x), 0);
        enc.set_buffer(4, Some(&out), 0);
        enc.set_bytes(5, 4, (&o as *const u32).cast());
        enc.set_bytes(6, 4, (&i as *const u32).cast());
        enc.set_bytes(7, 4, (&g as *const u32).cast());
        enc.dispatch_thread_groups(MTLSize::new(o.div_ceil(8) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let (s, e) = gpu_span(cmd);
        (e - s) * 1e3
    };
    eprintln!("\nLM-head v3 GPU time, {:.0} MiB of weights", len as f64 / (1 << 20) as f64);
    eprintln!("  mmap file, first touch : {:6.2} ms", run(&file_buf));
    let mut t: Vec<f64> = (0..20).map(|_| run(&file_buf)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("  mmap file, warm        : {:6.2} ms", t[10]);
    let mut t: Vec<f64> = (0..20).map(|_| run(&m.buf)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("  Metal buffer, warm     : {:6.2} ms", t[10]);
    unsafe { libc::munmap(ptr, map_len) };
    let _ = std::fs::remove_file(&path);
}

/// Reproduce decode's GPU load pattern without the model: per "token",
/// 40 × (≈1 ms of matvec work, then a CPU gap), then the LM head. If
/// the GPU clock drops under this pattern, every kernel slows down.
#[test]
#[ignore = "GPU microbench"]
fn matvec_decode_load_pattern() {
    use streammoe::riir::backend::gpu::gpu_timing::{gpu_span, host_now};
    let mut metal = MetalContext::new().expect("Metal");
    let pso = metal.pipeline("dequant_matvec_4bit_v3").expect("v3").to_owned();
    let mut rng = Rng(0x5eed_77aa);
    let head = make_mat(&metal, &mut rng, 248_320, 2048);
    let layer = make_mat(&metal, &mut rng, 12_288, 2048);
    let x = shared_buf(&metal, 2048 * 4);
    write(&x, 0, &(0..2048).map(|_| rng.f32()).collect::<Vec<f32>>());
    let out = shared_buf(&metal, 248_320 * 4);
    let spin = |us: f64| {
        let t = host_now();
        while (host_now() - t) * 1e6 < us {}
    };
    eprintln!("\nper-token pattern: 40 × (8 qkv-sized matvecs, CPU gap), then LM head");
    for gap_us in [0.0, 500.0, 1000.0, 1500.0] {
        let (mut head_ms, mut layer_ms) = (Vec::new(), Vec::new());
        for _tok in 0..12 {
            for _ in 0..40 {
                let cmd = metal.queue().new_command_buffer();
                for _ in 0..8 {
                    encode(cmd, &pso, Kind::V3, &layer, &x, &out);
                }
                cmd.commit();
                cmd.wait_until_completed();
                let (s, e) = gpu_span(cmd);
                layer_ms.push((e - s) * 1e3);
                spin(gap_us);
            }
            let cmd = metal.queue().new_command_buffer();
            encode(cmd, &pso, Kind::V3, &head, &x, &out);
            cmd.commit();
            cmd.wait_until_completed();
            let (s, e) = gpu_span(cmd);
            head_ms.push((e - s) * 1e3);
        }
        head_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        layer_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "  CPU gap {:>5.0} us: layer cmdbuf {:5.3} ms, LM head {:6.2} ms",
            gap_us, layer_ms[layer_ms.len() / 2], head_ms[head_ms.len() / 2]
        );
    }
}

/// Decode load pattern with and without a keep-alive: while the CPU
/// works between layers, a second queue runs a one-SIMD-group spin
/// kernel so the GPU never looks idle to its clock governor.
#[test]
#[ignore = "GPU microbench"]
fn matvec_decode_pattern_keepalive() {
    use streammoe::riir::backend::gpu::gpu_timing::{gpu_span, host_now};
    let mut metal = MetalContext::new().expect("Metal");
    let pso = metal.pipeline("dequant_matvec_4bit_v3").expect("v3").to_owned();
    let src = "#include <metal_stdlib>\nusing namespace metal;\n\
        kernel void spin(device uint* o [[buffer(0)]], constant uint& n [[buffer(1)]], uint t [[thread_position_in_grid]]) {\n\
            uint a = t; for (uint i = 0; i < n; i++) { a = a * 1664525u + 1013904223u; } o[t] = a; }";
    let lib = metal.device().new_library_with_source(src, &metal::CompileOptions::new()).expect("spin lib");
    let spin_pso = metal.device()
        .new_compute_pipeline_state_with_function(&lib.get_function("spin", None).unwrap())
        .unwrap();
    let side = metal.device().new_command_queue();
    let sink = shared_buf(&metal, 32 * 4);
    let mut rng = Rng(0x5eed_77ab);
    let head = make_mat(&metal, &mut rng, 248_320, 2048);
    let layer = make_mat(&metal, &mut rng, 12_288, 2048);
    let x = shared_buf(&metal, 2048 * 4);
    write(&x, 0, &(0..2048).map(|_| rng.f32()).collect::<Vec<f32>>());
    let out = shared_buf(&metal, 248_320 * 4);
    let spin_iters: u32 = 20_000;
    eprintln!("\n40 × (8 qkv matvecs + CPU gap) + LM head, 12 tokens, median");
    eprintln!("{:>8} {:>10} {:>13} {:>10} {:>12}", "gap us", "keepalive", "layer cmdbuf", "LM head", "wall/token");
    for round in 0..2 {
        for gap_us in [500.0f64, 750.0, 1000.0, 1250.0, 1500.0] {
            for keep in [false, true] {
                let (mut head_ms, mut layer_ms, mut wall) = (Vec::new(), Vec::new(), Vec::new());
                for _tok in 0..12 {
                    let t0 = host_now();
                    for _ in 0..40 {
                        let cmd = metal.queue().new_command_buffer();
                        for _ in 0..8 {
                            encode(cmd, &pso, Kind::V3, &layer, &x, &out);
                        }
                        cmd.commit();
                        cmd.wait_until_completed();
                        let (s, e) = gpu_span(cmd);
                        layer_ms.push((e - s) * 1e3);
                        let t = host_now();
                        while (host_now() - t) * 1e6 < gap_us {
                            if keep {
                                let c = side.new_command_buffer();
                                let enc = c.new_compute_command_encoder();
                                enc.set_compute_pipeline_state(&spin_pso);
                                enc.set_buffer(0, Some(&sink), 0);
                                enc.set_bytes(1, 4, (&spin_iters as *const u32).cast());
                                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
                                enc.end_encoding();
                                c.commit();
                                c.wait_until_completed();
                            }
                        }
                    }
                    let cmd = metal.queue().new_command_buffer();
                    encode(cmd, &pso, Kind::V3, &head, &x, &out);
                    cmd.commit();
                    cmd.wait_until_completed();
                    let (s, e) = gpu_span(cmd);
                    head_ms.push((e - s) * 1e3);
                    wall.push((host_now() - t0) * 1e3);
                }
                let med = |v: &mut Vec<f64>| {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    v[v.len() / 2]
                };
                eprintln!(
                    "{:>8.0} {:>10} {:>10.3} ms {:>7.2} ms {:>9.1} ms{}",
                    gap_us, keep, med(&mut layer_ms), med(&mut head_ms), med(&mut wall),
                    if round == 0 { "" } else { "   (repeat)" }
                );
            }
        }
    }
}
