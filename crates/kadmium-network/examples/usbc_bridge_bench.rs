//! Measures what a cable unplug costs a delta link, with and without a bridge.
//!
//! On the worker (for example the M4 Mac mini):
//!
//! ```text
//! cargo run --release -p kadmium-network --example usbc_bridge_bench -- serve 0.0.0.0:7880
//! ```
//!
//! This serves plain `KDM2` channels on port 7880 and bridge lanes on port 7881. On the
//! coordinator, give the plain address and the bridge address (one lane per path):
//!
//! ```text
//! cargo run --release -p kadmium-network --example usbc_bridge_bench -- run \
//!     192.168.40.237:7880 kdbr://169.254.78.11:7881@27.6,192.168.40.237:7881@1.2
//! ```
//!
//! Each scenario runs a decode-like loop twice. Partway through, the first path goes away:
//! - **plain**: the channel is dropped and redialed, as the pool does when the connection over
//!   the pulled cable dies. The new connection's delta stores start empty.
//! - **bridge**: lane 0 is disabled, as if its cable were pulled. The session carries on over the
//!   remaining lanes with its delta stores intact.
//!
//! The server answers every frame with the XXH64 of the payload it rebuilt, so every frame is
//! checked byte for byte.

use kadmium_network::delta_link::xxh64;
use kadmium_network::{
    BridgeListenerConfig, DeltaLinkConfig, FrameReader, FrameWriter, PeerChannel,
    UsbCBridgeListener, WireFrame, WireFrameKind, accept_delta_hello,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncWrite, BufReader};
use tokio::net::TcpListener;

const STEPS: usize = 12;
const UNPLUG_AT: usize = 4;

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

fn header(step: usize) -> Vec<u8> {
    format!(
        r#"{{"unit":{{"id":"session-7/stage-03","app":"StageForward"}},"instance_id":"session-7/stage-03#{step}","epoch":{step}}}"#
    )
    .into_bytes()
}

struct Scenario {
    name: &'static str,
    about: &'static str,
    payload: Box<dyn Fn(usize) -> Vec<u8>>,
}

fn scenarios() -> Vec<Scenario> {
    let context = noise(2, 4 << 20);
    let weights = noise(1, 16 << 20);
    vec![
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
            name: "weights-resend",
            about: "16 MiB stage weights re-sent with a changing header",
            payload: Box::new(move |step| [header(step), weights.clone()].concat()),
        },
    ]
}

async fn serve_frames<R, W>(reader: R, writer: W)
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut reader = FrameReader::new(BufReader::with_capacity(1 << 20, reader));
    let mut writer = FrameWriter::new(writer);
    let config = DeltaLinkConfig::default();
    while let Ok(frame) = reader.read_frame().await {
        if frame.kind == WireFrameKind::DeltaHello {
            if accept_delta_hello(&frame, &config, &mut reader, &mut writer)
                .await
                .is_err()
            {
                return;
            }
            continue;
        }
        let digest = xxh64(&frame.payload, 0).to_le_bytes().to_vec();
        let reply = WireFrame::new(WireFrameKind::WorkResult, frame.epoch, digest)
            .with_request_id(frame.request_id);
        if writer.write_frame(&reply).await.is_err() {
            return;
        }
    }
}

async fn serve(bind: &str) -> Result<(), String> {
    let plain = TcpListener::bind(bind)
        .await
        .map_err(|e| format!("bind {bind}: {e}"))?;
    let plain_addr = plain.local_addr().map_err(|e| e.to_string())?;
    let bridge_bind = format!("{}:{}", plain_addr.ip(), plain_addr.port() + 1);
    let mut bridges =
        UsbCBridgeListener::bind(&bridge_bind, BridgeListenerConfig::default()).await?;
    println!(
        "serving plain channels on {plain_addr} and bridge lanes on {}",
        bridges.local_addr()
    );
    tokio::spawn(async move {
        while let Ok((stream, peer)) = plain.accept().await {
            println!("plain channel from {peer}");
            let _ = stream.set_nodelay(true);
            let (r, w) = stream.into_split();
            tokio::spawn(serve_frames(r, w));
        }
    });
    while let Some(session) = bridges.accept().await {
        println!(
            "bridge session {:032x} from {}",
            session.bridge.session_id(),
            session.peer
        );
        tokio::spawn(async move {
            let _bridge = session.bridge;
            serve_frames(session.reader, session.writer).await;
        });
    }
    Ok(())
}

#[derive(Default)]
struct RunTotals {
    wall: Duration,
    after_unplug: Duration,
    wire_after_unplug: u64,
    logical_after_unplug: u64,
}

async fn call_checked(channel: &PeerChannel, step: usize, payload: Vec<u8>) -> Result<(), String> {
    let digest = xxh64(&payload, 0);
    let reply = channel
        .call(
            WireFrameKind::WorkDispatch,
            step as u32,
            payload,
            Duration::from_secs(120),
        )
        .await?;
    if reply.payload != digest.to_le_bytes() {
        return Err(format!("step {step}: payload digest mismatch"));
    }
    Ok(())
}

fn outbound(channel: &PeerChannel) -> (u64, u64) {
    channel
        .delta_link_stats()
        .map(|(tx, _)| (tx.wire_bytes, tx.logical_bytes))
        .unwrap_or_default()
}

async fn run_plain(addr: &str, scenario: &Scenario) -> Result<RunTotals, String> {
    let delta = DeltaLinkConfig::default();
    let timeout = Duration::from_secs(5);
    let mut channel = PeerChannel::connect_with(addr, timeout, &delta).await?;
    let mut totals = RunTotals::default();
    let start = Instant::now();
    let mut unplugged_at = start;
    let mut base = (0, 0);
    for step in 0..STEPS {
        if step == UNPLUG_AT {
            // The connection over the pulled cable is gone; the replacement starts cold.
            channel.close();
            channel = PeerChannel::connect_with(addr, timeout, &delta).await?;
            unplugged_at = Instant::now();
            base = outbound(&channel);
        }
        call_checked(&channel, step, (scenario.payload)(step)).await?;
    }
    totals.wall = start.elapsed();
    totals.after_unplug = unplugged_at.elapsed();
    let end = outbound(&channel);
    totals.wire_after_unplug = end.0 - base.0;
    totals.logical_after_unplug = end.1 - base.1;
    Ok(totals)
}

async fn run_bridge(addr: &str, scenario: &Scenario) -> Result<(RunTotals, String), String> {
    let channel: Arc<PeerChannel> =
        PeerChannel::connect_with(addr, Duration::from_secs(5), &DeltaLinkConfig::default())
            .await?;
    let bridge = channel.bridge().ok_or("not a bridge address")?.clone();
    let lanes = bridge.snapshot().lanes.len();
    if !bridge.wait_for_lanes(lanes, Duration::from_secs(5)).await {
        return Err(format!(
            "only {} of {lanes} lanes connected",
            bridge.snapshot().alive_lanes()
        ));
    }
    let mut totals = RunTotals::default();
    let start = Instant::now();
    let mut unplugged_at = start;
    let mut base = (0, 0);
    for step in 0..STEPS {
        if step == UNPLUG_AT && lanes > 1 {
            bridge.set_lane_enabled(0, false)?;
            unplugged_at = Instant::now();
            base = outbound(&channel);
        }
        call_checked(&channel, step, (scenario.payload)(step)).await?;
    }
    totals.wall = start.elapsed();
    totals.after_unplug = unplugged_at.elapsed();
    let end = outbound(&channel);
    totals.wire_after_unplug = end.0 - base.0;
    totals.logical_after_unplug = end.1 - base.1;
    let snap = bridge.snapshot();
    let lane_bytes: Vec<String> = snap
        .lanes
        .iter()
        .map(|l| {
            format!(
                "{}={:.1}MiB",
                l.label,
                l.bytes_sent as f64 / (1 << 20) as f64
            )
        })
        .collect();
    let detail = format!(
        "lanes [{}], retransmitted {} segments ({:.1} MiB), {} lane failures",
        lane_bytes.join(" "),
        snap.retransmitted_segments,
        snap.retransmitted_bytes as f64 / (1 << 20) as f64,
        snap.lane_failures
    );
    bridge.close();
    Ok((totals, detail))
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1 << 20) as f64
}

async fn run(plain_addr: &str, bridge_addr: &str) -> Result<(), String> {
    println!(
        "{STEPS} steps per run, first path lost before step {UNPLUG_AT}; times include the loss\n"
    );
    for scenario in scenarios() {
        println!("{} ({})", scenario.name, scenario.about);
        let plain = run_plain(plain_addr, &scenario).await?;
        let (bridge, detail) = run_bridge(bridge_addr, &scenario).await?;
        for (label, t) in [("plain ", &plain), ("bridge", &bridge)] {
            println!(
                "  {label}  total {:>8.1} ms   after loss {:>8.1} ms   {:>7.2} MiB sent for {:>7.2} MiB of frames",
                t.wall.as_secs_f64() * 1000.0,
                t.after_unplug.as_secs_f64() * 1000.0,
                mib(t.wire_after_unplug),
                mib(t.logical_after_unplug),
            );
        }
        println!("  bridge  {detail}");
        println!(
            "  after the loss the bridge sent {:.1}x fewer bytes and finished {:.2}x faster\n",
            plain.wire_after_unplug as f64 / bridge.wire_after_unplug.max(1) as f64,
            plain.after_unplug.as_secs_f64() / bridge.after_unplug.as_secs_f64().max(1e-9),
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["serve", bind] => serve(bind).await,
        ["run", plain, bridge] => run(plain, bridge).await,
        _ => Err("usage: usbc_bridge_bench serve <ip:port> | run <ip:port> <kdbr://...>".into()),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
