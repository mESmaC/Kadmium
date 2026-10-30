//! Measures delta links against plain `KDM2` frames over a real network link.
//!
//! On the worker (for example the M4 Mac mini):
//!
//! ```text
//! cargo run --release -p kadmium-network --example delta_link_bench -- serve 0.0.0.0:7870
//! ```
//!
//! On the coordinator:
//!
//! ```text
//! cargo run --release -p kadmium-network --example delta_link_bench -- run 192.168.40.237:7870
//! ```
//!
//! Each scenario sends the same payload sequence twice: once over a plain channel and once over a
//! delta link. The server answers every frame with the XXH64 of the payload it rebuilt, so every
//! delta frame is checked byte for byte against what was sent.

use kadmium_network::delta_link::xxh64;
use kadmium_network::{
    DeltaLinkConfig, FrameReader, FrameWriter, KDM_FRAME_HEADER_LEN_V2, PeerChannel, WireFrame,
    WireFrameKind, accept_delta_hello,
};
use std::time::{Duration, Instant};
use tokio::io::BufReader;
use tokio::net::{TcpListener, TcpStream};

fn noise(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        out.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
    }
    out.truncate(len);
    out
}

/// A unit header that changes a little every step, like a `WorkDispatchHeader`.
fn header(step: usize) -> Vec<u8> {
    format!(
        r#"{{"unit":{{"id":"session-7/stage-03","app":"StageForward","params":{{"start_layer":12,"end_layer":24,"q_star":0.8}}}},"instance_id":"session-7/stage-03#{step}","epoch":{step}}}"#
    )
    .into_bytes()
}

struct Scenario {
    name: &'static str,
    about: &'static str,
    payload: Box<dyn Fn(usize) -> Vec<u8>>,
}

fn scenarios() -> Vec<Scenario> {
    let weights = noise(1, 16 << 20);
    let context = noise(2, 4 << 20);
    let kv = noise(3, 16 << 20);
    let params = noise(4, 16 << 20);
    vec![
        Scenario {
            name: "weights-resend",
            about: "16 MiB stage weights re-sent with a changing header",
            payload: Box::new(move |step| [header(step), weights.clone()].concat()),
        },
        Scenario {
            name: "decode-step",
            about: "header + fresh 28 KiB activations + unchanged 4 MiB context",
            payload: Box::new(move |step| {
                [
                    header(step),
                    noise(100 + step as u64, 28 << 10),
                    context.clone(),
                ]
                .concat()
            }),
        },
        Scenario {
            name: "kv-append",
            about: "KV pages growing 64 KiB per step from 4 MiB",
            payload: Box::new(move |step| kv[..(4 << 20) + step * (64 << 10)].to_vec()),
        },
        Scenario {
            name: "sparse-update",
            about: "16 MiB parameters with 32 scattered 64-byte edits per step",
            payload: Box::new(move |step| {
                let mut p = params.clone();
                for s in 0..=step {
                    let edits = noise(1000 + s as u64, 32 * 8);
                    for e in edits.chunks(8) {
                        let at =
                            (u64::from_le_bytes(e.try_into().unwrap()) as usize) % (p.len() - 64);
                        p[at..at + 64].fill(s as u8);
                    }
                }
                p
            }),
        },
        Scenario {
            name: "fresh-random",
            about: "4 MiB of new random bytes every step (worst case)",
            payload: Box::new(|step| noise(5000 + step as u64, 4 << 20)),
        },
    ]
}

async fn serve(bind: &str) -> Result<(), String> {
    let listener = TcpListener::bind(bind)
        .await
        .map_err(|e| format!("bind {bind}: {e}"))?;
    println!("delta_link_bench serving on {bind}");
    loop {
        let (stream, peer) = listener.accept().await.map_err(|e| e.to_string())?;
        tokio::spawn(async move {
            let _ = stream.set_nodelay(true);
            let (r, w) = stream.into_split();
            let mut reader = FrameReader::new(BufReader::with_capacity(1 << 20, r));
            let mut writer = FrameWriter::new(w);
            let mut frames = 0u64;
            while let Ok(frame) = reader.read_frame().await {
                if frame.kind == WireFrameKind::DeltaHello {
                    let config = DeltaLinkConfig::from_env();
                    if accept_delta_hello(&frame, &config, &mut reader, &mut writer)
                        .await
                        .is_err()
                    {
                        break;
                    }
                    continue;
                }
                frames += 1;
                let digest = xxh64(&frame.payload, 0).to_le_bytes().to_vec();
                let reply = WireFrame::new(WireFrameKind::WorkResult, frame.epoch, digest)
                    .with_request_id(frame.request_id);
                if writer.write_frame(&reply).await.is_err() {
                    break;
                }
            }
            let mode = if reader.delta_active() {
                "delta"
            } else {
                "plain"
            };
            println!("  {peer}: {frames} {mode} frames");
        });
    }
}

struct Measured {
    logical: u64,
    wire: u64,
    wall: Duration,
}

async fn measure(
    addr: &str,
    scenario: &Scenario,
    steps: usize,
    delta: bool,
) -> Result<Measured, String> {
    let timeout = Duration::from_secs(10);
    let channel = if delta {
        PeerChannel::connect_with(addr, timeout, &DeltaLinkConfig::default()).await?
    } else {
        // Dial directly so the plain run does not depend on how `connect` evolves.
        let stream = TcpStream::connect(addr)
            .await
            .map_err(|e| format!("connect {addr}: {e}"))?;
        PeerChannel::from_stream(addr, stream)
    };
    if delta && !channel.delta_link_active() {
        return Err(format!("{addr} did not accept a delta link"));
    }
    let payloads: Vec<Vec<u8>> = (0..steps).map(|s| (scenario.payload)(s)).collect();
    let (mut logical, mut plain_wire) = (0u64, 0u64);
    let t0 = Instant::now();
    for (step, payload) in payloads.into_iter().enumerate() {
        let expected = xxh64(&payload, 0);
        logical += payload.len() as u64;
        plain_wire += (KDM_FRAME_HEADER_LEN_V2 + payload.len()) as u64;
        let reply = channel
            .call(
                WireFrameKind::WorkDispatch,
                step as u32,
                payload,
                Duration::from_secs(120),
            )
            .await?;
        let got = u64::from_le_bytes(
            reply
                .payload
                .as_slice()
                .try_into()
                .map_err(|_| "short digest reply".to_string())?,
        );
        if got != expected {
            return Err(format!(
                "{} step {step}: server rebuilt different bytes",
                scenario.name
            ));
        }
    }
    let wall = t0.elapsed();
    let wire = match channel.delta_link_stats() {
        Some((outbound, _)) => {
            outbound.wire_bytes + outbound.frames * KDM_FRAME_HEADER_LEN_V2 as u64
        }
        None => plain_wire,
    };
    channel.close();
    Ok(Measured {
        logical,
        wire,
        wall,
    })
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1 << 20) as f64
}

async fn run(addr: &str, steps: usize) -> Result<(), String> {
    println!("delta_link_bench -> {addr}, {steps} steps per scenario\n");
    println!(
        "{:<15} {:>10} {:>11} {:>11} {:>10} {:>10} {:>12} {:>12} {:>8}",
        "scenario",
        "sent MiB",
        "plain MiB",
        "delta MiB",
        "plain ms",
        "delta ms",
        "plain MiB/s",
        "delta MiB/s",
        "speedup"
    );
    for scenario in scenarios() {
        let plain = measure(addr, &scenario, steps, false).await?;
        let delta = measure(addr, &scenario, steps, true).await?;
        let rate = |m: &Measured| mib(m.logical) / m.wall.as_secs_f64();
        println!(
            "{:<15} {:>10.1} {:>11.1} {:>11.2} {:>10.0} {:>10.0} {:>12.1} {:>12.1} {:>7.1}x",
            scenario.name,
            mib(plain.logical),
            mib(plain.wire),
            mib(delta.wire),
            plain.wall.as_secs_f64() * 1e3,
            delta.wall.as_secs_f64() * 1e3,
            rate(&plain),
            rate(&delta),
            plain.wall.as_secs_f64() / delta.wall.as_secs_f64(),
        );
        println!("  {}", scenario.about);
    }
    Ok(())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("serve") => serve(args.get(1).map_or("0.0.0.0:7870", String::as_str)).await,
        Some("run") if args.len() >= 2 => {
            let steps = args
                .iter()
                .position(|a| a == "--steps")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(24);
            run(&args[1], steps).await
        }
        _ => Err("usage: delta_link_bench serve [bind] | run <host:port> [--steps N]".into()),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
