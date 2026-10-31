//! # `kadmium_network::usbc_bridge`
//!
//! A striped bridge carries one ordered, reliable byte stream between two nodes over several
//! lanes at once: typically a USB4 / Thunderbolt or USB-NCM cable plus a Wi-Fi or Ethernet path.
//! It keeps the stream alive when a cable is unplugged, so anything layered on it keeps its state.
//!
//! [`crate::delta_link`] already sends each frame as only the chunks the peer does not hold, but
//! its chunk stores belong to one TCP connection. Unplugging a USB-C cable kills that connection,
//! the replacement starts with empty stores, and every weight blob, context window and KV page the
//! peer already held crosses the slower backup path again. InferSSD's device staging solved the
//! same problem for source trees: each device keeps the commits it has received, and the next
//! bundle leaves out everything that anchor already holds. A bridge applies that to links. The
//! sequence number of the next segment the receiver lacks is the anchor; a lane that fails, or a
//! cable that is replugged within [`BridgeConfig::resume_grace`], resends only the segments after
//! it. The delta stores on either side never see a gap, so they stay valid.
//!
//! ## How it works
//!
//! The local side writes into a [`BridgeWriter`]. A sender task cuts that stream into segments of
//! up to [`BridgeConfig::segment_bytes`], coalescing small writes the way a CDC-NCM transfer block
//! aggregates datagrams, and numbers them. Each segment goes to the live lane that would finish
//! sending it first: `(bytes queued on the lane + segment) / lane weight`. Small, latency-bound
//! frames therefore stay on the fastest cable, and bulk transfers spread across every lane in
//! proportion to how fast each one drains. The receiver puts segments back in order, hands them
//! to its [`BridgeReader`], and returns a cumulative acknowledgement on every lane.
//!
//! The sender keeps each segment until it is acknowledged, holding at most
//! [`BridgeConfig::window_bytes`]. When a lane fails (a write error, a reset, or
//! [`BridgeConfig::lane_idle_timeout`] without a byte or heartbeat, which is how a pulled cable
//! usually shows up), its unacknowledged segments go out again on the remaining lanes. The receiver
//! drops duplicates. The dialing side redials failed lanes with backoff and resumes the same
//! session; the session ends with an error only after no lane has been up for the resume grace.
//!
//! ## Lane wire format
//!
//! ```text
//! hello (dialer -> listener, 30 bytes)
//!   "KDLN" | u16 version | u16 flags (bit 0: create the session if absent)
//!   | u64 session id | u16 lane index | u32 segment bytes | u32 window bytes | u32 weight (Mbit/s)
//! reply (listener -> dialer, 15 bytes)
//!   "KDLN" | u16 version | u8 status (0 ok, 1 unknown session, 2 bad parameters, 3 bad version)
//!   | u32 segment bytes | u32 window bytes
//! then, in both directions, a sequence of
//!   u8 0 (data) | u64 seq | u32 len | len bytes      (len 0 marks the end of the stream)
//!   u8 1 (ack)  | u64 next seq the receiver lacks    (also sent as a heartbeat)
//! ```
//!
//! All integers are little endian. A dialer only sets the create flag until the session exists, so
//! a listener that restarted mid-session answers a resume with "unknown session" instead of
//! silently starting an empty stream that the peer's delta stores would no longer match.

use crate::delta_link::xxh64;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::io::{
    AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter, DuplexStream, ReadBuf,
};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, mpsc, oneshot, watch};
use tokio::task::AbortHandle;

/// Version of the lane protocol this build speaks.
pub const BRIDGE_VERSION: u16 = 1;

/// Address scheme that makes [`crate::PeerChannel`] dial a bridge instead of one TCP connection:
/// `kdbr://169.254.78.11:7871@27.6,192.168.1.20:7871@1.2`, one `host:port` per lane, each with an
/// optional `@` weight in Gbit/s (the lane's expected goodput).
pub const BRIDGE_ADDR_SCHEME: &str = "kdbr://";

/// Default segment size: a 64 KiB jumbo frame on USB4 / Thunderbolt IP, and the largest transfer
/// block a CDC-NCM link accepts once both sides raise it from the 16 KiB default.
pub const DEFAULT_BRIDGE_SEGMENT_BYTES: usize = 64 << 10;
/// Smallest segment size a listener accepts.
pub const MIN_BRIDGE_SEGMENT_BYTES: usize = 4 << 10;
/// Largest segment size a listener accepts.
pub const MAX_BRIDGE_SEGMENT_BYTES: usize = 1 << 20;

/// Default bound on unacknowledged bytes per direction. It must cover the fast lane's throughput
/// for as long as a segment spends on the slowest lane, or the fast lane idles behind it:
/// 3.45 GB/s on a 30 Gbit/s USB4 cable times ~2 ms for 64 KiB over Wi-Fi is ~7 MB.
pub const DEFAULT_BRIDGE_WINDOW_BYTES: usize = 16 << 20;
/// Largest window a listener accepts.
pub const MAX_BRIDGE_WINDOW_BYTES: usize = 256 << 20;

/// How long a session survives with no live lane before it ends with an error.
pub const DEFAULT_BRIDGE_RESUME_GRACE: Duration = Duration::from_secs(30);
/// A lane that delivers neither data nor a heartbeat for this long is treated as unplugged.
pub const DEFAULT_LANE_IDLE_TIMEOUT: Duration = Duration::from_millis(750);
/// Default bound on dialing one lane, including its hello.
pub const DEFAULT_LANE_CONNECT_TIMEOUT: Duration = Duration::from_millis(2500);

/// Most lanes one bridge may have (and the largest lane index a listener accepts, plus one).
pub const MAX_BRIDGE_LANES: usize = 256;

const LANE_HEARTBEAT: Duration = Duration::from_millis(100);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
const REDIAL_BACKOFF_MIN: Duration = Duration::from_millis(25);
const REDIAL_BACKOFF_MAX: Duration = Duration::from_secs(1);
/// Close a finished session only after this long, so the last acknowledgement reaches the peer.
const FINISH_LINGER: Duration = Duration::from_millis(250);
/// Acknowledge at least every this many delivered segments, and whenever delivery goes idle.
const ACK_EVERY_SEGMENTS: u64 = 8;

const LANE_MAGIC: [u8; 4] = *b"KDLN";
const HELLO_LEN: usize = 30;
const REPLY_LEN: usize = 15;
const FLAG_CREATE: u16 = 1;
const STATUS_OK: u8 = 0;
const STATUS_UNKNOWN_SESSION: u8 = 1;
const STATUS_BAD_PARAMS: u8 = 2;
const STATUS_BAD_VERSION: u8 = 3;
const TAG_DATA: u8 = 0;
const TAG_ACK: u8 = 1;
const DATA_HEADER_LEN: usize = 1 + 8 + 4;
const LANE_IO_BUFFER: usize = 256 << 10;
const DUPLEX_BYTES: usize = 1 << 20;

/// Most segments either side may have unacknowledged; both derive it from the agreed window so
/// the receiver can reject a sequence number the sender could never have used.
fn inflight_limit(window_bytes: usize) -> u64 {
    (window_bytes / 1024).max(64) as u64
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------------

/// One lane of a bridge, as the dialing side sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BridgeLane {
    /// `host:port` of the peer's [`UsbCBridgeListener`] on this path, for example its USB4
    /// link-local address or its Wi-Fi address.
    pub addr: String,
    /// Expected goodput in Gbit/s. Only the ratio between lanes matters: it decides how bulk
    /// transfers split and which lane carries small frames.
    pub weight_gbps: f64,
    pub label: String,
}

impl BridgeLane {
    pub fn new(addr: impl Into<String>, weight_gbps: f64, label: impl Into<String>) -> Self {
        Self {
            addr: addr.into(),
            weight_gbps,
            label: label.into(),
        }
    }
}

/// Settings for dialing a bridge.
#[derive(Debug, Clone, PartialEq)]
pub struct BridgeConfig {
    pub lanes: Vec<BridgeLane>,
    pub segment_bytes: usize,
    pub window_bytes: usize,
    pub resume_grace: Duration,
    pub lane_idle_timeout: Duration,
    pub connect_timeout: Duration,
}

impl BridgeConfig {
    pub fn new(lanes: Vec<BridgeLane>) -> Self {
        Self {
            lanes,
            segment_bytes: DEFAULT_BRIDGE_SEGMENT_BYTES,
            window_bytes: DEFAULT_BRIDGE_WINDOW_BYTES,
            resume_grace: DEFAULT_BRIDGE_RESUME_GRACE,
            lane_idle_timeout: DEFAULT_LANE_IDLE_TIMEOUT,
            connect_timeout: DEFAULT_LANE_CONNECT_TIMEOUT,
        }
    }

    /// Parses a [`BRIDGE_ADDR_SCHEME`] address. Returns `None` when `addr` is not a bridge
    /// address at all, so callers can fall back to a plain TCP dial.
    pub fn parse_addr(addr: &str) -> Option<Result<Self, String>> {
        let body = addr.strip_prefix(BRIDGE_ADDR_SCHEME)?;
        let mut lanes = Vec::new();
        for (i, part) in body.split(',').map(str::trim).enumerate() {
            if part.is_empty() {
                return Some(Err(format!("bridge address {addr} has an empty lane")));
            }
            // A weight suffix is the text after the last '@' when it parses as a number; an IPv6
            // zone or a host name never does.
            let (host, weight) = match part.rsplit_once('@') {
                Some((host, w)) => match w.parse::<f64>() {
                    Ok(w) if w.is_finite() && w > 0.0 => (host, w),
                    _ => return Some(Err(format!("bridge lane {part} has an invalid weight"))),
                },
                None => (part, 1.0),
            };
            if !host.contains(':') {
                return Some(Err(format!("bridge lane {part} needs host:port")));
            }
            lanes.push(BridgeLane::new(host, weight, format!("lane{i}")));
        }
        Some(Ok(Self::new(lanes)))
    }

    /// Formats the lanes back as a [`BRIDGE_ADDR_SCHEME`] address.
    pub fn to_addr(&self) -> String {
        let lanes: Vec<String> = self
            .lanes
            .iter()
            .map(|l| format!("{}@{}", l.addr, l.weight_gbps))
            .collect();
        format!("{BRIDGE_ADDR_SCHEME}{}", lanes.join(","))
    }

    fn validated(mut self) -> Result<Self, String> {
        if self.lanes.is_empty() {
            return Err("a bridge needs at least one lane".into());
        }
        if self.lanes.len() > MAX_BRIDGE_LANES {
            return Err(format!("a bridge has at most {MAX_BRIDGE_LANES} lanes"));
        }
        self.segment_bytes = self
            .segment_bytes
            .clamp(MIN_BRIDGE_SEGMENT_BYTES, MAX_BRIDGE_SEGMENT_BYTES);
        self.window_bytes = self
            .window_bytes
            .clamp(self.segment_bytes, MAX_BRIDGE_WINDOW_BYTES);
        Ok(self)
    }
}

/// Settings for accepting bridges.
#[derive(Debug, Clone, PartialEq)]
pub struct BridgeListenerConfig {
    pub max_segment_bytes: usize,
    pub max_window_bytes: usize,
    pub resume_grace: Duration,
    pub lane_idle_timeout: Duration,
}

impl Default for BridgeListenerConfig {
    fn default() -> Self {
        Self {
            max_segment_bytes: MAX_BRIDGE_SEGMENT_BYTES,
            max_window_bytes: MAX_BRIDGE_WINDOW_BYTES,
            resume_grace: DEFAULT_BRIDGE_RESUME_GRACE,
            lane_idle_timeout: DEFAULT_LANE_IDLE_TIMEOUT,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Handshake
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LaneHello {
    version: u16,
    flags: u16,
    session_id: u64,
    lane: u16,
    segment_bytes: u32,
    window_bytes: u32,
    weight_mbps: u32,
}

impl LaneHello {
    fn to_bytes(self) -> [u8; HELLO_LEN] {
        let mut out = [0u8; HELLO_LEN];
        out[..4].copy_from_slice(&LANE_MAGIC);
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6..8].copy_from_slice(&self.flags.to_le_bytes());
        out[8..16].copy_from_slice(&self.session_id.to_le_bytes());
        out[16..18].copy_from_slice(&self.lane.to_le_bytes());
        out[18..22].copy_from_slice(&self.segment_bytes.to_le_bytes());
        out[22..26].copy_from_slice(&self.window_bytes.to_le_bytes());
        out[26..30].copy_from_slice(&self.weight_mbps.to_le_bytes());
        out
    }

    fn from_bytes(b: &[u8; HELLO_LEN]) -> Result<Self, String> {
        if b[..4] != LANE_MAGIC {
            return Err("not a bridge lane hello".into());
        }
        let u16_at = |at: usize| u16::from_le_bytes([b[at], b[at + 1]]);
        let u32_at = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"));
        Ok(Self {
            version: u16_at(4),
            flags: u16_at(6),
            session_id: u64::from_le_bytes(b[8..16].try_into().expect("8 bytes")),
            lane: u16_at(16),
            segment_bytes: u32_at(18),
            window_bytes: u32_at(22),
            weight_mbps: u32_at(26),
        })
    }
}

fn reply_bytes(status: u8, segment_bytes: u32, window_bytes: u32) -> [u8; REPLY_LEN] {
    let mut out = [0u8; REPLY_LEN];
    out[..4].copy_from_slice(&LANE_MAGIC);
    out[4..6].copy_from_slice(&BRIDGE_VERSION.to_le_bytes());
    out[6] = status;
    out[7..11].copy_from_slice(&segment_bytes.to_le_bytes());
    out[11..15].copy_from_slice(&window_bytes.to_le_bytes());
    out
}

#[derive(Debug)]
enum DialError {
    /// The listener no longer has the session: resuming is impossible.
    UnknownSession,
    Other(String),
}

async fn dial_lane(
    addr: &str,
    hello: LaneHello,
    timeout: Duration,
) -> Result<TcpStream, DialError> {
    let attempt = async {
        let mut stream = TcpStream::connect(addr)
            .await
            .map_err(|e| DialError::Other(format!("connect to {addr}: {e}")))?;
        let _ = stream.set_nodelay(true);
        stream
            .write_all(&hello.to_bytes())
            .await
            .map_err(|e| DialError::Other(format!("hello to {addr}: {e}")))?;
        // Read the reply straight off the socket: the listener's first lane messages may follow
        // it immediately and belong to the lane reader.
        let mut reply = [0u8; REPLY_LEN];
        stream
            .read_exact(&mut reply)
            .await
            .map_err(|e| DialError::Other(format!("reply from {addr}: {e}")))?;
        if reply[..4] != LANE_MAGIC {
            return Err(DialError::Other(format!("{addr} is not a bridge listener")));
        }
        match reply[6] {
            STATUS_OK => {
                let segment = u32::from_le_bytes(reply[7..11].try_into().expect("4 bytes"));
                let window = u32::from_le_bytes(reply[11..15].try_into().expect("4 bytes"));
                if segment != hello.segment_bytes || window != hello.window_bytes {
                    return Err(DialError::Other(format!(
                        "{addr} changed the bridge parameters"
                    )));
                }
                Ok(stream)
            }
            STATUS_UNKNOWN_SESSION => Err(DialError::UnknownSession),
            STATUS_BAD_PARAMS => Err(DialError::Other(format!(
                "{addr} rejected segment {} / window {} bytes",
                hello.segment_bytes, hello.window_bytes
            ))),
            STATUS_BAD_VERSION => Err(DialError::Other(format!(
                "{addr} speaks another bridge version"
            ))),
            other => Err(DialError::Other(format!(
                "{addr} answered with status {other}"
            ))),
        }
    };
    match tokio::time::timeout(timeout, attempt).await {
        Ok(result) => result,
        Err(_) => Err(DialError::Other(format!(
            "lane to {addr} timed out after {timeout:?}"
        ))),
    }
}

fn new_session_id(seed: &str) -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let material = format!(
        "{nanos}|{}|{}|{seed}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    xxh64(material.as_bytes(), 0x4B44_4C4E).max(1)
}

// ---------------------------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Default)]
struct BridgeCounters {
    segments_sent: AtomicU64,
    bytes_sent: AtomicU64,
    retransmitted_segments: AtomicU64,
    retransmitted_bytes: AtomicU64,
    segments_received: AtomicU64,
    duplicate_segments: AtomicU64,
    bytes_delivered: AtomicU64,
    lane_attaches: AtomicU64,
    lane_failures: AtomicU64,
}

/// Point-in-time state of one lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BridgeLaneSnapshot {
    pub index: usize,
    pub label: String,
    pub weight_gbps: f64,
    pub alive: bool,
    pub enabled: bool,
    /// Times the lane connected (the first connection included).
    pub attaches: u64,
    pub failures: u64,
    /// Payload bytes this side wrote on the lane, retransmissions included.
    pub bytes_sent: u64,
    pub queued_bytes: u64,
    pub last_error: Option<String>,
}

/// Point-in-time state of a bridge. Byte counts are segment payload bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BridgeSnapshot {
    pub session_id: u64,
    pub open: bool,
    pub close_reason: Option<String>,
    pub segment_bytes: usize,
    pub window_bytes: usize,
    pub segments_sent: u64,
    pub bytes_sent: u64,
    pub retransmitted_segments: u64,
    pub retransmitted_bytes: u64,
    pub segments_received: u64,
    pub duplicate_segments: u64,
    pub bytes_delivered: u64,
    pub unacked_bytes: u64,
    pub lane_attaches: u64,
    pub lane_failures: u64,
    pub lanes: Vec<BridgeLaneSnapshot>,
}

impl BridgeSnapshot {
    /// Lanes currently connected.
    pub fn alive_lanes(&self) -> usize {
        self.lanes.iter().filter(|l| l.alive).count()
    }
}

// ---------------------------------------------------------------------------------------------
// Session core
// ---------------------------------------------------------------------------------------------

struct InFlight {
    bytes: Arc<Vec<u8>>,
    lane: usize,
    generation: u64,
    sends: u32,
}

#[derive(Default)]
struct SendState {
    next_seq: u64,
    peer_ack: u64,
    unacked: BTreeMap<u64, InFlight>,
    unacked_bytes: usize,
    /// Segments waiting for a lane: new ones and those whose lane failed.
    pending: BTreeSet<u64>,
    fin_seq: Option<u64>,
}

#[derive(Default)]
struct RecvState {
    next_contig: u64,
    reorder: BTreeMap<u64, Arc<Vec<u8>>>,
    reorder_bytes: usize,
    fin_seq: Option<u64>,
}

struct LaneLink {
    tx: mpsc::UnboundedSender<Segment>,
    queued: Arc<AtomicU64>,
    tasks: Vec<AbortHandle>,
    down: Option<oneshot::Sender<String>>,
}

struct LaneSlot {
    label: String,
    weight_gbps: f64,
    enabled: bool,
    generation: u64,
    link: Option<LaneLink>,
    attaches: u64,
    failures: u64,
    bytes_sent: Arc<AtomicU64>,
    last_error: Option<String>,
}

impl LaneSlot {
    fn new(label: String, weight_gbps: f64) -> Self {
        Self {
            label,
            weight_gbps,
            enabled: true,
            generation: 0,
            link: None,
            attaches: 0,
            failures: 0,
            bytes_sent: Arc::new(AtomicU64::new(0)),
            last_error: None,
        }
    }
}

#[derive(Clone)]
struct Segment {
    seq: u64,
    bytes: Arc<Vec<u8>>,
}

/// What the dialing side needs to (re)dial lanes.
struct DialInfo {
    connect_timeout: Duration,
    /// Set once any lane has joined: from then on lanes only resume, never create.
    established: AtomicBool,
}

struct Core {
    session_id: u64,
    segment_bytes: usize,
    window_bytes: usize,
    inflight_limit: u64,
    resume_grace: Duration,
    lane_idle_timeout: Duration,
    dial: Option<DialInfo>,
    send: Mutex<SendState>,
    recv: Mutex<RecvState>,
    lanes: Mutex<Vec<LaneSlot>>,
    /// Wakes the sender task: acknowledgements, requeued segments and lane changes.
    wake_sender: Notify,
    /// Wakes lane supervisors and connect waiters on any lane change.
    lanes_changed: Notify,
    deliver_tx: mpsc::UnboundedSender<Segment>,
    ack_tx: watch::Sender<u64>,
    closed: watch::Sender<Option<String>>,
    send_done: AtomicBool,
    recv_done: AtomicBool,
    finishing: AtomicBool,
    last_alive: Mutex<Instant>,
    counters: BridgeCounters,
}

struct CoreParams {
    session_id: u64,
    segment_bytes: usize,
    window_bytes: usize,
    resume_grace: Duration,
    lane_idle_timeout: Duration,
    dial: Option<DialInfo>,
}

impl Core {
    /// Creates a session and starts its sender, delivery and watchdog tasks.
    fn start(params: CoreParams) -> (Arc<Self>, BridgeReader, BridgeWriter) {
        let (deliver_tx, deliver_rx) = mpsc::unbounded_channel();
        let core = Arc::new(Self {
            session_id: params.session_id,
            segment_bytes: params.segment_bytes,
            window_bytes: params.window_bytes,
            inflight_limit: inflight_limit(params.window_bytes),
            resume_grace: params.resume_grace,
            lane_idle_timeout: params.lane_idle_timeout,
            dial: params.dial,
            send: Mutex::new(SendState::default()),
            recv: Mutex::new(RecvState::default()),
            lanes: Mutex::new(Vec::new()),
            wake_sender: Notify::new(),
            lanes_changed: Notify::new(),
            deliver_tx,
            ack_tx: watch::channel(0).0,
            closed: watch::channel(None).0,
            send_done: AtomicBool::new(false),
            recv_done: AtomicBool::new(false),
            finishing: AtomicBool::new(false),
            last_alive: Mutex::new(Instant::now()),
            counters: BridgeCounters::default(),
        });
        let (user_out, source) = tokio::io::duplex(DUPLEX_BYTES);
        let (sink, user_in) = tokio::io::duplex(DUPLEX_BYTES);
        tokio::spawn(run_sender(Arc::clone(&core), source));
        tokio::spawn(run_delivery(Arc::clone(&core), sink, deliver_rx));
        tokio::spawn(run_watchdog(Arc::clone(&core)));
        (
            core,
            BridgeReader { inner: user_in },
            BridgeWriter { inner: user_out },
        )
    }

    fn is_closed(&self) -> bool {
        self.closed.borrow().is_some()
    }

    /// Ends the session: stops every lane, and lets the sender and delivery tasks exit, which
    /// turns the user's reader into EOF and its writer into a broken pipe.
    fn close(&self, reason: String) {
        let first = self.closed.send_if_modified(|slot| {
            if slot.is_none() {
                *slot = Some(reason);
                true
            } else {
                false
            }
        });
        if !first {
            return;
        }
        let links: Vec<LaneLink> = lock(&self.lanes)
            .iter_mut()
            .filter_map(|slot| slot.link.take())
            .collect();
        for mut link in links {
            for task in &link.tasks {
                task.abort();
            }
            if let Some(down) = link.down.take() {
                let _ = down.send("session closed".into());
            }
        }
        self.wake_sender.notify_one();
        self.lanes_changed.notify_waiters();
    }

    fn any_lane_alive(&self) -> bool {
        lock(&self.lanes).iter().any(|s| s.link.is_some())
    }

    fn window_open(&self) -> bool {
        let s = lock(&self.send);
        s.next_seq - s.peer_ack < self.inflight_limit && s.unacked_bytes < self.window_bytes
    }

    /// Numbers a new segment and queues it for a lane. An empty segment ends the stream.
    fn queue_segment(&self, bytes: Arc<Vec<u8>>) {
        let mut s = lock(&self.send);
        let seq = s.next_seq;
        s.next_seq += 1;
        if bytes.is_empty() {
            s.fin_seq = Some(seq);
        }
        s.unacked_bytes += bytes.len();
        s.unacked.insert(
            seq,
            InFlight {
                bytes,
                lane: usize::MAX,
                generation: 0,
                sends: 0,
            },
        );
        s.pending.insert(seq);
    }

    fn pop_pending(&self) -> Option<u64> {
        lock(&self.send).pending.pop_first()
    }

    /// The live lane that would finish sending `len` more bytes first.
    fn choose_lane(&self, len: usize) -> Option<(usize, u64, LaneHandle)> {
        let lanes = lock(&self.lanes);
        let mut best: Option<(f64, usize)> = None;
        for (idx, slot) in lanes.iter().enumerate() {
            let Some(link) = slot.link.as_ref().filter(|_| slot.enabled) else {
                continue;
            };
            let queued = link.queued.load(Ordering::Relaxed) as f64;
            let finish = (queued + len as f64) / slot.weight_gbps.max(1e-3);
            if best.is_none_or(|(b, _)| finish < b) {
                best = Some((finish, idx));
            }
        }
        best.map(|(_, idx)| {
            let slot = &lanes[idx];
            let link = slot.link.as_ref().expect("chosen lane is live");
            (
                idx,
                slot.generation,
                LaneHandle {
                    tx: link.tx.clone(),
                    queued: Arc::clone(&link.queued),
                },
            )
        })
    }

    fn lane_is(&self, idx: usize, generation: u64) -> bool {
        lock(&self.lanes)
            .get(idx)
            .is_some_and(|s| s.generation == generation && s.link.is_some())
    }

    /// Hands segment `seq` to a lane, waiting for one when none is live. Returns `false` once the
    /// session has closed.
    async fn dispatch(&self, seq: u64, closed: &mut watch::Receiver<Option<String>>) -> bool {
        let (bytes, resend) = match lock(&self.send).unacked.get(&seq) {
            Some(f) => (Arc::clone(&f.bytes), f.sends > 0),
            None => return true, // acknowledged while it waited
        };
        loop {
            if self.is_closed() {
                return false;
            }
            let Some((idx, generation, lane)) = self.choose_lane(bytes.len()) else {
                tokio::select! {
                    _ = closed.changed() => return false,
                    _ = self.wake_sender.notified() => {}
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {}
                }
                continue;
            };
            // Record the lane before sending, so a lane failure from here on requeues it.
            match lock(&self.send).unacked.get_mut(&seq) {
                Some(f) => {
                    f.lane = idx;
                    f.generation = generation;
                    f.sends += 1;
                }
                None => return true,
            }
            let len = bytes.len() as u64;
            lane.queued.fetch_add(len, Ordering::Relaxed);
            let sent = lane
                .tx
                .send(Segment {
                    seq,
                    bytes: Arc::clone(&bytes),
                })
                .is_ok();
            if !sent {
                lane.queued.fetch_sub(len, Ordering::Relaxed);
            }
            // A lane that failed after it was chosen may never write this segment, and its failure
            // handling may have run before the lane was recorded above. Retry here; the failure
            // handling may also have requeued it, so take it back out of the queue. (Should both
            // copies go out, the receiver drops the duplicate.)
            if !sent || !self.lane_is(idx, generation) {
                lock(&self.send).pending.remove(&seq);
                continue;
            }
            self.counters.segments_sent.fetch_add(1, Ordering::Relaxed);
            self.counters.bytes_sent.fetch_add(len, Ordering::Relaxed);
            if resend {
                self.counters
                    .retransmitted_segments
                    .fetch_add(1, Ordering::Relaxed);
                self.counters
                    .retransmitted_bytes
                    .fetch_add(len, Ordering::Relaxed);
            }
            return true;
        }
    }

    /// Applies a cumulative acknowledgement from the peer.
    fn on_ack(self: &Arc<Self>, next: u64) -> Result<(), String> {
        let mut s = lock(&self.send);
        if next > s.next_seq {
            return Err(format!(
                "peer acknowledged segment {next} of {} sent",
                s.next_seq
            ));
        }
        if next <= s.peer_ack {
            return Ok(());
        }
        let keep = s.unacked.split_off(&next);
        let acked = std::mem::replace(&mut s.unacked, keep);
        let freed: usize = acked.values().map(|f| f.bytes.len()).sum();
        s.unacked_bytes -= freed;
        s.pending = s.pending.split_off(&next);
        s.peer_ack = next;
        let fin_acked = s.fin_seq.is_some_and(|f| next > f);
        drop(s);
        if fin_acked {
            self.send_done.store(true, Ordering::SeqCst);
            self.check_finished();
        }
        self.wake_sender.notify_one();
        Ok(())
    }

    /// Accepts a segment from a lane and moves every segment that is now in order to delivery.
    fn on_data(&self, seq: u64, bytes: Arc<Vec<u8>>) -> Result<(), String> {
        self.counters
            .segments_received
            .fetch_add(1, Ordering::Relaxed);
        let mut r = lock(&self.recv);
        if seq < r.next_contig || r.reorder.contains_key(&seq) {
            self.counters
                .duplicate_segments
                .fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        if seq >= r.next_contig + self.inflight_limit {
            return Err(format!(
                "segment {seq} is beyond the window (next expected {})",
                r.next_contig
            ));
        }
        if r.fin_seq.is_some_and(|f| seq > f) {
            return Err(format!("segment {seq} follows the end of the stream"));
        }
        if bytes.is_empty() {
            r.fin_seq = Some(seq);
        }
        r.reorder_bytes += bytes.len();
        if r.reorder_bytes > self.window_bytes + self.segment_bytes {
            return Err("peer overran the reorder window".into());
        }
        r.reorder.insert(seq, bytes);
        loop {
            let next = r.next_contig;
            let Some(bytes) = r.reorder.remove(&next) else {
                break;
            };
            r.reorder_bytes -= bytes.len();
            r.next_contig += 1;
            let _ = self.deliver_tx.send(Segment { seq: next, bytes });
        }
        Ok(())
    }

    fn check_finished(self: &Arc<Self>) {
        if self.send_done.load(Ordering::SeqCst)
            && self.recv_done.load(Ordering::SeqCst)
            && !self.finishing.swap(true, Ordering::SeqCst)
        {
            // Linger so the acknowledgement of the peer's end of stream goes out first.
            let weak = Arc::downgrade(self);
            tokio::spawn(async move {
                tokio::time::sleep(FINISH_LINGER).await;
                if let Some(core) = weak.upgrade() {
                    core.close("finished".into());
                }
            });
        }
    }
}

struct LaneHandle {
    tx: mpsc::UnboundedSender<Segment>,
    queued: Arc<AtomicU64>,
}

// ---------------------------------------------------------------------------------------------
// Lanes
// ---------------------------------------------------------------------------------------------

impl Core {
    /// Makes sure slot `idx` exists (a listener learns lanes as they connect).
    fn ensure_slot(lanes: &mut Vec<LaneSlot>, idx: usize, weight_gbps: f64) {
        while lanes.len() <= idx {
            let i = lanes.len();
            lanes.push(LaneSlot::new(format!("lane{i}"), weight_gbps));
        }
    }

    /// Starts serving `stream` as lane `idx`, replacing any earlier connection of that lane. The
    /// returned receiver fires with the reason once this connection fails.
    fn attach_lane(
        self: &Arc<Self>,
        idx: usize,
        weight_gbps: Option<f64>,
        stream: TcpStream,
    ) -> oneshot::Receiver<String> {
        let _ = stream.set_nodelay(true);
        let (read_half, write_half) = stream.into_split();
        let (tx, rx) = mpsc::unbounded_channel();
        let queued = Arc::new(AtomicU64::new(0));
        let (down_tx, down_rx) = oneshot::channel();
        let (generation, replaced, bytes_sent) = {
            let mut lanes = lock(&self.lanes);
            Self::ensure_slot(&mut lanes, idx, weight_gbps.unwrap_or(1.0));
            let slot = &mut lanes[idx];
            if let Some(w) = weight_gbps {
                slot.weight_gbps = w;
            }
            let replaced = slot.link.take().map(|link| (slot.generation, link));
            slot.generation += 1;
            slot.attaches += 1;
            slot.link = Some(LaneLink {
                tx,
                queued: Arc::clone(&queued),
                tasks: Vec::new(),
                down: Some(down_tx),
            });
            (slot.generation, replaced, Arc::clone(&slot.bytes_sent))
        };
        if let Some((old_generation, link)) = replaced {
            self.retire_link(
                idx,
                old_generation,
                link,
                "replaced by a new connection".into(),
            );
        }
        self.counters.lane_attaches.fetch_add(1, Ordering::Relaxed);
        if let Some(dial) = &self.dial {
            dial.established.store(true, Ordering::SeqCst);
        }
        *lock(&self.last_alive) = Instant::now();

        let writer = tokio::spawn(run_lane_writer(
            Arc::clone(self),
            idx,
            generation,
            write_half,
            rx,
            queued,
            bytes_sent,
        ));
        let reader = tokio::spawn(run_lane_reader(
            Arc::clone(self),
            idx,
            generation,
            read_half,
        ));
        let tasks = [writer.abort_handle(), reader.abort_handle()];
        let mut lanes = lock(&self.lanes);
        let current = lanes[idx].generation == generation;
        match lanes[idx].link.as_mut() {
            Some(link) if current => link.tasks.extend(tasks),
            // The lane failed (or the session closed) before its tasks were recorded.
            _ => tasks.iter().for_each(AbortHandle::abort),
        }
        drop(lanes);
        self.wake_sender.notify_one();
        self.lanes_changed.notify_waiters();
        down_rx
    }

    /// Marks connection `generation` of lane `idx` failed and requeues what it had not delivered.
    fn lane_down(&self, idx: usize, generation: u64, reason: String) {
        let link = {
            let mut lanes = lock(&self.lanes);
            match lanes.get_mut(idx) {
                Some(slot) if slot.generation == generation && slot.link.is_some() => {
                    slot.failures += 1;
                    slot.last_error = Some(reason.clone());
                    slot.link.take()
                }
                _ => None,
            }
        };
        if let Some(link) = link {
            self.counters.lane_failures.fetch_add(1, Ordering::Relaxed);
            self.retire_link(idx, generation, link, reason);
        }
    }

    fn retire_link(&self, idx: usize, generation: u64, mut link: LaneLink, reason: String) {
        for task in &link.tasks {
            task.abort();
        }
        if let Some(down) = link.down.take() {
            let _ = down.send(reason);
        }
        {
            let mut s = lock(&self.send);
            let lost: Vec<u64> = s
                .unacked
                .iter()
                .filter(|(_, f)| f.lane == idx && f.generation == generation)
                .map(|(&seq, _)| seq)
                .collect();
            s.pending.extend(lost);
        }
        self.wake_sender.notify_one();
        self.lanes_changed.notify_waiters();
    }

    fn snapshot(&self) -> BridgeSnapshot {
        let c = &self.counters;
        let lanes = lock(&self.lanes)
            .iter()
            .enumerate()
            .map(|(index, s)| BridgeLaneSnapshot {
                index,
                label: s.label.clone(),
                weight_gbps: s.weight_gbps,
                alive: s.link.is_some(),
                enabled: s.enabled,
                attaches: s.attaches,
                failures: s.failures,
                bytes_sent: s.bytes_sent.load(Ordering::Relaxed),
                queued_bytes: s
                    .link
                    .as_ref()
                    .map_or(0, |l| l.queued.load(Ordering::Relaxed)),
                last_error: s.last_error.clone(),
            })
            .collect();
        let close_reason = self.closed.borrow().clone();
        BridgeSnapshot {
            session_id: self.session_id,
            open: close_reason.is_none(),
            close_reason,
            segment_bytes: self.segment_bytes,
            window_bytes: self.window_bytes,
            segments_sent: c.segments_sent.load(Ordering::Relaxed),
            bytes_sent: c.bytes_sent.load(Ordering::Relaxed),
            retransmitted_segments: c.retransmitted_segments.load(Ordering::Relaxed),
            retransmitted_bytes: c.retransmitted_bytes.load(Ordering::Relaxed),
            segments_received: c.segments_received.load(Ordering::Relaxed),
            duplicate_segments: c.duplicate_segments.load(Ordering::Relaxed),
            bytes_delivered: c.bytes_delivered.load(Ordering::Relaxed),
            unacked_bytes: lock(&self.send).unacked_bytes as u64,
            lane_attaches: c.lane_attaches.load(Ordering::Relaxed),
            lane_failures: c.lane_failures.load(Ordering::Relaxed),
            lanes,
        }
    }
}

/// Writes this side's segments and acknowledgements on one lane connection.
async fn run_lane_writer(
    core: Arc<Core>,
    idx: usize,
    generation: u64,
    write_half: OwnedWriteHalf,
    mut rx: mpsc::UnboundedReceiver<Segment>,
    queued: Arc<AtomicU64>,
    bytes_sent: Arc<AtomicU64>,
) {
    // Room for a whole segment and its header, so both leave in one write.
    let capacity = LANE_IO_BUFFER.max(core.segment_bytes + DATA_HEADER_LEN);
    let mut w = BufWriter::with_capacity(capacity, write_half);
    let mut acks = core.ack_tx.subscribe();
    let mut heartbeat = tokio::time::interval(LANE_HEARTBEAT);
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Announce what this side already holds, so a resumed lane prunes the peer's resend queue.
    acks.mark_changed();
    let result: std::io::Result<()> = async {
        loop {
            tokio::select! {
                biased;
                changed = acks.changed() => {
                    if changed.is_err() {
                        return Ok(());
                    }
                    let next = *acks.borrow_and_update();
                    write_ack(&mut w, next).await?;
                    w.flush().await?;
                }
                segment = rx.recv() => {
                    let Some(segment) = segment else { return Ok(()) };
                    let len = segment.bytes.len();
                    let mut header = [0u8; DATA_HEADER_LEN];
                    header[0] = TAG_DATA;
                    header[1..9].copy_from_slice(&segment.seq.to_le_bytes());
                    header[9..13].copy_from_slice(&(len as u32).to_le_bytes());
                    w.write_all(&header).await?;
                    w.write_all(&segment.bytes).await?;
                    queued.fetch_sub(len as u64, Ordering::Relaxed);
                    bytes_sent.fetch_add(len as u64, Ordering::Relaxed);
                    if rx.is_empty() {
                        w.flush().await?;
                    }
                }
                _ = heartbeat.tick() => {
                    let next = *acks.borrow();
                    write_ack(&mut w, next).await?;
                    w.flush().await?;
                }
            }
        }
    }
    .await;
    let reason = match result {
        Ok(()) => "lane writer stopped".to_string(),
        Err(e) => format!("write failed: {e}"),
    };
    core.lane_down(idx, generation, reason);
}

async fn write_ack<W: AsyncWrite + Unpin>(w: &mut W, next: u64) -> std::io::Result<()> {
    let mut msg = [0u8; 9];
    msg[0] = TAG_ACK;
    msg[1..9].copy_from_slice(&next.to_le_bytes());
    w.write_all(&msg).await
}

/// Fills `buf`, failing if no byte arrives for `idle`. Progress resets the clock, so a slow lane
/// carrying a large segment is not mistaken for an unplugged one.
async fn read_full<R: AsyncRead + Unpin>(
    r: &mut R,
    buf: &mut [u8],
    idle: Duration,
) -> std::io::Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        match tokio::time::timeout(idle, r.read(&mut buf[filled..])).await {
            Ok(Ok(0)) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(Ok(n)) => filled += n,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("no data or heartbeat for {idle:?}"),
                ));
            }
        }
    }
    Ok(())
}

enum LaneFailure {
    Io(std::io::Error),
    Protocol(String),
}

async fn read_lane_messages<R: AsyncRead + Unpin>(
    core: &Arc<Core>,
    r: &mut R,
) -> Result<(), LaneFailure> {
    let idle = core.lane_idle_timeout;
    loop {
        let mut tag = [0u8; 1];
        read_full(r, &mut tag, idle)
            .await
            .map_err(LaneFailure::Io)?;
        match tag[0] {
            TAG_DATA => {
                let mut header = [0u8; 12];
                read_full(r, &mut header, idle)
                    .await
                    .map_err(LaneFailure::Io)?;
                let seq = u64::from_le_bytes(header[..8].try_into().expect("8 bytes"));
                let len = u32::from_le_bytes(header[8..].try_into().expect("4 bytes")) as usize;
                if len > core.segment_bytes {
                    return Err(LaneFailure::Protocol(format!(
                        "{len}-byte segment exceeds the agreed {} bytes",
                        core.segment_bytes
                    )));
                }
                let mut body = vec![0u8; len];
                read_full(r, &mut body, idle)
                    .await
                    .map_err(LaneFailure::Io)?;
                core.on_data(seq, Arc::new(body))
                    .map_err(LaneFailure::Protocol)?;
            }
            TAG_ACK => {
                let mut next = [0u8; 8];
                read_full(r, &mut next, idle)
                    .await
                    .map_err(LaneFailure::Io)?;
                core.on_ack(u64::from_le_bytes(next))
                    .map_err(LaneFailure::Protocol)?;
            }
            other => {
                return Err(LaneFailure::Protocol(format!(
                    "unknown lane message tag {other}"
                )));
            }
        }
    }
}

/// Reads the peer's segments and acknowledgements from one lane connection.
async fn run_lane_reader(core: Arc<Core>, idx: usize, generation: u64, read_half: OwnedReadHalf) {
    let mut r = BufReader::with_capacity(LANE_IO_BUFFER, read_half);
    match read_lane_messages(&core, &mut r).await {
        Ok(()) => {}
        Err(LaneFailure::Io(e)) => core.lane_down(idx, generation, format!("read failed: {e}")),
        // A peer that breaks the protocol cannot be trusted on any lane.
        Err(LaneFailure::Protocol(e)) => core.close(format!("protocol error on lane {idx}: {e}")),
    }
}

// ---------------------------------------------------------------------------------------------
// Session tasks
// ---------------------------------------------------------------------------------------------

/// Cuts the user's outbound stream into segments and hands them, and any requeued ones, to lanes.
async fn run_sender(core: Arc<Core>, mut source: DuplexStream) {
    #[derive(PartialEq)]
    enum Source {
        Open,
        /// The user closed its writer; the end-of-stream segment is still to be queued.
        Ending,
        Done,
    }
    let mut closed = core.closed.subscribe();
    // Each segment is read into its own buffer, which then moves into the segment uncopied.
    let mut buf = vec![0u8; core.segment_bytes];
    let mut state = Source::Open;
    loop {
        if core.is_closed() {
            return;
        }
        if let Some(seq) = core.pop_pending() {
            if !core.dispatch(seq, &mut closed).await {
                return;
            }
            continue;
        }
        let window_open = core.window_open();
        if window_open && state == Source::Ending {
            core.queue_segment(Arc::new(Vec::new()));
            state = Source::Done;
            continue;
        }
        if window_open && state == Source::Open {
            tokio::select! {
                biased;
                _ = closed.changed() => return,
                _ = core.wake_sender.notified() => {}
                read = source.read(&mut buf) => {
                    let mut filled = match read {
                        Ok(0) | Err(_) => {
                            state = Source::Ending;
                            continue;
                        }
                        Ok(n) => n,
                    };
                    // Coalesce whatever else is already buffered into the same segment, the way
                    // an NCM transfer block packs datagrams, without waiting for more.
                    while filled < buf.len() {
                        match read_now(&mut source, &mut buf[filled..]).await {
                            Some(Ok(0)) | Some(Err(_)) => {
                                state = Source::Ending;
                                break;
                            }
                            Some(Ok(n)) => filled += n,
                            None => break,
                        }
                    }
                    buf.truncate(filled);
                    let segment = std::mem::replace(&mut buf, vec![0u8; core.segment_bytes]);
                    core.queue_segment(Arc::new(segment));
                }
            }
        } else {
            tokio::select! {
                _ = closed.changed() => return,
                _ = core.wake_sender.notified() => {}
            }
        }
    }
}

/// Reads what `source` already holds without waiting: `None` when a read would block. (A
/// zero-length `tokio::time::timeout` would wait for the timer driver's next tick instead, which
/// is ~15 ms on Windows.)
async fn read_now(source: &mut DuplexStream, buf: &mut [u8]) -> Option<std::io::Result<usize>> {
    std::future::poll_fn(|cx| {
        let mut read_buf = ReadBuf::new(buf);
        match Pin::new(&mut *source).poll_read(cx, &mut read_buf) {
            Poll::Ready(Ok(())) => Poll::Ready(Some(Ok(read_buf.filled().len()))),
            Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
            Poll::Pending => Poll::Ready(None),
        }
    })
    .await
}

/// Writes in-order segments into the user's reader and acknowledges them once they are there,
/// so a reader that stops reading stalls the peer's sender instead of growing a buffer here.
async fn run_delivery(
    core: Arc<Core>,
    mut sink: DuplexStream,
    mut rx: mpsc::UnboundedReceiver<Segment>,
) {
    let mut closed = core.closed.subscribe();
    let mut sink_open = true;
    let mut last_acked = 0u64;
    loop {
        let segment = tokio::select! {
            _ = closed.changed() => return,
            segment = rx.recv() => match segment {
                Some(segment) => segment,
                None => return,
            },
        };
        let next = segment.seq + 1;
        if segment.bytes.is_empty() {
            let _ = sink.shutdown().await;
            sink_open = false;
            core.ack_tx.send_replace(next);
            core.recv_done.store(true, Ordering::SeqCst);
            core.check_finished();
            continue;
        }
        if sink_open {
            let write = tokio::select! {
                _ = closed.changed() => return,
                write = sink.write_all(&segment.bytes) => write,
            };
            // The user dropped its reader: keep acknowledging so the peer is not stalled.
            sink_open = write.is_ok();
        }
        core.counters
            .bytes_delivered
            .fetch_add(segment.bytes.len() as u64, Ordering::Relaxed);
        if rx.is_empty() || next - last_acked >= ACK_EVERY_SEGMENTS {
            core.ack_tx.send_replace(next);
            last_acked = next;
        }
    }
}

/// Ends the session once no lane has been live for the resume grace.
async fn run_watchdog(core: Arc<Core>) {
    let mut closed = core.closed.subscribe();
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    loop {
        tokio::select! {
            _ = closed.changed() => return,
            _ = tick.tick() => {}
        }
        if core.any_lane_alive() {
            *lock(&core.last_alive) = Instant::now();
        } else if lock(&core.last_alive).elapsed() > core.resume_grace {
            core.close(format!(
                "no lane reconnected within the {:?} resume grace",
                core.resume_grace
            ));
            return;
        }
    }
}

/// Keeps one lane of a dialing bridge connected: dials, serves until the connection fails, backs
/// off, and redials into the same session for as long as the session is open.
async fn run_lane_supervisor(core: Arc<Core>, idx: usize, lane: BridgeLane) {
    let mut closed = core.closed.subscribe();
    let mut backoff = REDIAL_BACKOFF_MIN;
    let dial = core
        .dial
        .as_ref()
        .expect("supervisors run on the dialing side");
    loop {
        if core.is_closed() {
            return;
        }
        let changed = core.lanes_changed.notified();
        let enabled = lock(&core.lanes).get(idx).is_some_and(|s| s.enabled);
        if !enabled {
            tokio::select! {
                _ = closed.changed() => return,
                _ = changed => {}
                _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            }
            continue;
        }
        let established = dial.established.load(Ordering::SeqCst);
        let hello = LaneHello {
            version: BRIDGE_VERSION,
            flags: if established { 0 } else { FLAG_CREATE },
            session_id: core.session_id,
            lane: idx as u16,
            segment_bytes: core.segment_bytes as u32,
            window_bytes: core.window_bytes as u32,
            weight_mbps: (lane.weight_gbps * 1000.0).clamp(1.0, u32::MAX as f64) as u32,
        };
        match dial_lane(&lane.addr, hello, dial.connect_timeout).await {
            Ok(stream) => {
                // Disabled while dialing: drop the connection instead of serving it.
                if !lock(&core.lanes).get(idx).is_some_and(|s| s.enabled) || core.is_closed() {
                    continue;
                }
                backoff = REDIAL_BACKOFF_MIN;
                let down = core.attach_lane(idx, None, stream);
                tokio::select! {
                    _ = closed.changed() => return,
                    _ = down => {}
                }
            }
            Err(DialError::UnknownSession) => {
                core.close(format!(
                    "{} no longer holds session {:016x}; it cannot be resumed",
                    lane.addr, core.session_id
                ));
                return;
            }
            Err(DialError::Other(e)) => {
                if let Some(slot) = lock(&core.lanes).get_mut(idx) {
                    slot.last_error = Some(e);
                }
                core.lanes_changed.notify_waiters();
            }
        }
        tokio::select! {
            _ = closed.changed() => return,
            _ = tokio::time::sleep(backoff) => {}
        }
        backoff = (backoff * 2).min(REDIAL_BACKOFF_MAX);
    }
}

// ---------------------------------------------------------------------------------------------
// Public handles
// ---------------------------------------------------------------------------------------------

/// Read half of a bridge: the peer's stream, in order. EOF when the peer closes its writer or the
/// session ends.
#[derive(Debug)]
pub struct BridgeReader {
    inner: DuplexStream,
}

impl AsyncRead for BridgeReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

/// Write half of a bridge. Dropping or shutting it down ends this side's stream after everything
/// written so far is delivered.
#[derive(Debug)]
pub struct BridgeWriter {
    inner: DuplexStream,
}

impl AsyncWrite for BridgeWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

/// Control handle for a bridge session. Cloning it is cheap; dropping it does not end the session
/// (dropping both stream halves does).
#[derive(Clone)]
pub struct UsbCBridge {
    core: Arc<Core>,
}

impl std::fmt::Debug for UsbCBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsbCBridge")
            .field("session_id", &format_args!("{:016x}", self.core.session_id))
            .field("open", &!self.core.is_closed())
            .finish()
    }
}

/// A connected bridge: its control handle and its two stream halves.
#[derive(Debug)]
pub struct BridgeSession {
    pub bridge: UsbCBridge,
    pub reader: BridgeReader,
    pub writer: BridgeWriter,
    /// The address of the lane that opened the session.
    pub peer: String,
}

impl UsbCBridge {
    /// Dials every lane in `config` and returns once the first one has joined the session; the
    /// others join as they connect. Fails if no lane connects within `config.connect_timeout`.
    pub async fn connect(config: BridgeConfig) -> Result<BridgeSession, String> {
        let config = config.validated()?;
        let session_id = new_session_id(&config.to_addr());
        let (core, reader, writer) = Core::start(CoreParams {
            session_id,
            segment_bytes: config.segment_bytes,
            window_bytes: config.window_bytes,
            resume_grace: config.resume_grace,
            lane_idle_timeout: config.lane_idle_timeout,
            dial: Some(DialInfo {
                connect_timeout: config.connect_timeout,
                established: AtomicBool::new(false),
            }),
        });
        {
            let mut lanes = lock(&core.lanes);
            for lane in &config.lanes {
                lanes.push(LaneSlot::new(lane.label.clone(), lane.weight_gbps));
            }
        }
        for (idx, lane) in config.lanes.iter().enumerate() {
            tokio::spawn(run_lane_supervisor(Arc::clone(&core), idx, lane.clone()));
        }
        let bridge = Self {
            core: Arc::clone(&core),
        };
        let deadline = tokio::time::Instant::now() + config.connect_timeout;
        loop {
            let changed = core.lanes_changed.notified();
            let first_live = lock(&bridge.core.lanes)
                .iter()
                .position(|s| s.link.is_some());
            if let Some(idx) = first_live {
                return Ok(BridgeSession {
                    peer: config.lanes[idx].addr.clone(),
                    bridge,
                    reader,
                    writer,
                });
            }
            if bridge.core.is_closed() || tokio::time::Instant::now() >= deadline {
                let errors: Vec<String> = lock(&bridge.core.lanes)
                    .iter()
                    .filter_map(|s| s.last_error.clone())
                    .collect();
                let reason = bridge.core.closed.borrow().clone();
                bridge.close();
                return Err(format!(
                    "no bridge lane connected within {:?}: {}",
                    config.connect_timeout,
                    reason.unwrap_or_else(|| errors.join("; "))
                ));
            }
            let _ = tokio::time::timeout_at(deadline, changed).await;
        }
    }

    pub fn session_id(&self) -> u64 {
        self.core.session_id
    }

    pub fn is_open(&self) -> bool {
        !self.core.is_closed()
    }

    pub fn snapshot(&self) -> BridgeSnapshot {
        self.core.snapshot()
    }

    /// Ends the session immediately, without draining.
    pub fn close(&self) {
        self.core.close("closed locally".into());
    }

    /// Takes a lane out of service (`false`) or puts it back (`true`). Disabling drops the lane's
    /// connection, as pulling its cable would; its unacknowledged segments move to the other lanes.
    /// On the dialing side an enabled lane is redialed and resumes the session.
    pub fn set_lane_enabled(&self, idx: usize, enabled: bool) -> Result<(), String> {
        let generation = {
            let mut lanes = lock(&self.core.lanes);
            let slot = lanes
                .get_mut(idx)
                .ok_or_else(|| format!("bridge has no lane {idx}"))?;
            slot.enabled = enabled;
            slot.generation
        };
        if !enabled {
            self.core
                .lane_down(idx, generation, "disabled locally".into());
        }
        self.core.lanes_changed.notify_waiters();
        Ok(())
    }

    /// Adds a lane to a dialing bridge, for example when a USB-C cable is plugged in after the
    /// session started. Returns the new lane's index.
    pub fn add_lane(&self, lane: BridgeLane) -> Result<usize, String> {
        if self.core.dial.is_none() {
            return Err("lanes are added by the dialing side".into());
        }
        if self.core.is_closed() {
            return Err("bridge session is closed".into());
        }
        let idx = {
            let mut lanes = lock(&self.core.lanes);
            if lanes.len() >= MAX_BRIDGE_LANES {
                return Err(format!("a bridge has at most {MAX_BRIDGE_LANES} lanes"));
            }
            lanes.push(LaneSlot::new(lane.label.clone(), lane.weight_gbps));
            lanes.len() - 1
        };
        tokio::spawn(run_lane_supervisor(Arc::clone(&self.core), idx, lane));
        Ok(idx)
    }

    /// Waits until at least `count` lanes are live, or `timeout` passes. Returns whether they are.
    pub async fn wait_for_lanes(&self, count: usize, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let changed = self.core.lanes_changed.notified();
            let alive = lock(&self.core.lanes)
                .iter()
                .filter(|s| s.link.is_some())
                .count();
            if alive >= count {
                return true;
            }
            if tokio::time::Instant::now() >= deadline || self.core.is_closed() {
                return false;
            }
            let _ = tokio::time::timeout_at(deadline, changed).await;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Listener
// ---------------------------------------------------------------------------------------------

type SessionRegistry = Arc<Mutex<HashMap<u64, Weak<Core>>>>;

/// Accepts bridge lanes on one TCP port and groups them into sessions. Bind it on all
/// interfaces so lanes arriving over the USB-C link-local address and over Wi-Fi meet.
pub struct UsbCBridgeListener {
    local_addr: SocketAddr,
    sessions: mpsc::Receiver<BridgeSession>,
    accept_task: tokio::task::JoinHandle<()>,
}

impl std::fmt::Debug for UsbCBridgeListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsbCBridgeListener")
            .field("local_addr", &self.local_addr)
            .finish()
    }
}

impl Drop for UsbCBridgeListener {
    fn drop(&mut self) {
        self.accept_task.abort();
    }
}

impl UsbCBridgeListener {
    pub async fn bind(addr: &str, config: BridgeListenerConfig) -> Result<Self, String> {
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| format!("bind bridge listener on {addr}: {e}"))?;
        let local_addr = listener
            .local_addr()
            .map_err(|e| format!("bridge listener address: {e}"))?;
        let (tx, sessions) = mpsc::channel(64);
        let registry: SessionRegistry = Arc::new(Mutex::new(HashMap::new()));
        let config = Arc::new(config);
        let accept_task = tokio::spawn(async move {
            while let Ok((stream, peer)) = listener.accept().await {
                if tx.is_closed() {
                    break;
                }
                tokio::spawn(accept_lane(
                    stream,
                    peer,
                    Arc::clone(&registry),
                    Arc::clone(&config),
                    tx.clone(),
                ));
            }
        });
        Ok(Self {
            local_addr,
            sessions,
            accept_task,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Waits for the next new session. Lanes that resume an existing session are absorbed into
    /// it and never show up here.
    pub async fn accept(&mut self) -> Option<BridgeSession> {
        self.sessions.recv().await
    }
}

async fn accept_lane(
    mut stream: TcpStream,
    peer: SocketAddr,
    registry: SessionRegistry,
    config: Arc<BridgeListenerConfig>,
    sessions: mpsc::Sender<BridgeSession>,
) {
    let mut hello = [0u8; HELLO_LEN];
    match tokio::time::timeout(HANDSHAKE_TIMEOUT, stream.read_exact(&mut hello)).await {
        Ok(Ok(_)) => {}
        _ => return,
    }
    let Ok(hello) = LaneHello::from_bytes(&hello) else {
        return;
    };
    let reply = |status| reply_bytes(status, hello.segment_bytes, hello.window_bytes);
    if hello.version != BRIDGE_VERSION {
        let _ = stream.write_all(&reply(STATUS_BAD_VERSION)).await;
        return;
    }
    let (segment, window) = (hello.segment_bytes as usize, hello.window_bytes as usize);
    if !(MIN_BRIDGE_SEGMENT_BYTES..=config.max_segment_bytes).contains(&segment)
        || !(segment..=config.max_window_bytes).contains(&window)
        || hello.lane as usize >= MAX_BRIDGE_LANES
    {
        let _ = stream.write_all(&reply(STATUS_BAD_PARAMS)).await;
        return;
    }

    enum Lookup {
        Existing(Arc<Core>),
        Created(Arc<Core>, BridgeReader, BridgeWriter),
        Mismatch,
        Unknown,
    }
    let lookup = {
        let mut map = lock(&registry);
        map.retain(|_, weak| weak.upgrade().is_some_and(|c| !c.is_closed()));
        match map.get(&hello.session_id).and_then(Weak::upgrade) {
            Some(core) if core.segment_bytes == segment && core.window_bytes == window => {
                Lookup::Existing(core)
            }
            Some(_) => Lookup::Mismatch,
            None if hello.flags & FLAG_CREATE != 0 => {
                let (core, reader, writer) = Core::start(CoreParams {
                    session_id: hello.session_id,
                    segment_bytes: segment,
                    window_bytes: window,
                    resume_grace: config.resume_grace,
                    lane_idle_timeout: config.lane_idle_timeout,
                    dial: None,
                });
                map.insert(hello.session_id, Arc::downgrade(&core));
                Lookup::Created(core, reader, writer)
            }
            None => Lookup::Unknown,
        }
    };
    let (core, created) = match lookup {
        Lookup::Existing(core) => (core, None),
        Lookup::Created(core, reader, writer) => (core, Some((reader, writer))),
        Lookup::Mismatch => {
            let _ = stream.write_all(&reply(STATUS_BAD_PARAMS)).await;
            return;
        }
        Lookup::Unknown => {
            let _ = stream.write_all(&reply(STATUS_UNKNOWN_SESSION)).await;
            return;
        }
    };
    if stream.write_all(&reply(STATUS_OK)).await.is_err() {
        // A session created for this lane gets no lane; its resume grace ends it.
        return;
    }
    let weight = hello.weight_mbps.max(1) as f64 / 1000.0;
    // The listener side never redials, so it has no use for the lane-down signal.
    drop(core.attach_lane(hello.lane as usize, Some(weight), stream));
    if let Some((reader, writer)) = created {
        let session = BridgeSession {
            bridge: UsbCBridge { core },
            reader,
            writer,
            peer: peer.to_string(),
        };
        // An error means nobody accepts sessions any more; dropping the halves ends it.
        let _ = sessions.send(session).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta_link::{DeltaLinkConfig, FrameReader, FrameWriter, accept_delta_hello};
    use crate::{PeerChannel, WireFrame, WireFrameKind};

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

    async fn listener() -> UsbCBridgeListener {
        UsbCBridgeListener::bind(
            "127.0.0.1:0",
            BridgeListenerConfig {
                lane_idle_timeout: Duration::from_millis(400),
                ..BridgeListenerConfig::default()
            },
        )
        .await
        .unwrap()
    }

    fn config(addr: SocketAddr, weights: &[f64]) -> BridgeConfig {
        let lanes = weights
            .iter()
            .enumerate()
            .map(|(i, &w)| BridgeLane::new(addr.to_string(), w, format!("lane{i}")))
            .collect();
        BridgeConfig {
            segment_bytes: 16 << 10,
            window_bytes: 1 << 20,
            lane_idle_timeout: Duration::from_millis(400),
            ..BridgeConfig::new(lanes)
        }
    }

    async fn wait_closed(bridge: &UsbCBridge) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while bridge.is_open() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("bridge should close");
    }

    #[test]
    fn bridge_addresses_parse_with_and_without_weights() {
        let cfg = BridgeConfig::parse_addr("kdbr://169.254.78.11:7871@27.6, 192.168.1.20:7871")
            .unwrap()
            .unwrap();
        assert_eq!(cfg.lanes.len(), 2);
        assert_eq!(cfg.lanes[0].addr, "169.254.78.11:7871");
        assert_eq!(cfg.lanes[0].weight_gbps, 27.6);
        assert_eq!(cfg.lanes[1].weight_gbps, 1.0);
        let reparsed = BridgeConfig::parse_addr(&cfg.to_addr()).unwrap().unwrap();
        assert_eq!(reparsed.lanes, cfg.lanes);
        assert!(BridgeConfig::parse_addr("127.0.0.1:7871").is_none());
        assert!(BridgeConfig::parse_addr("kdbr://no-port").unwrap().is_err());
        assert!(
            BridgeConfig::parse_addr("kdbr://h:1@fast")
                .unwrap()
                .is_err()
        );
        assert!(
            BridgeConfig::parse_addr("kdbr://h:1,,h:2")
                .unwrap()
                .is_err()
        );
    }

    #[test]
    fn lane_hello_round_trips() {
        let hello = LaneHello {
            version: BRIDGE_VERSION,
            flags: FLAG_CREATE,
            session_id: 0x0123_4567_89AB_CDEF,
            lane: 3,
            segment_bytes: 65_536,
            window_bytes: 16 << 20,
            weight_mbps: 27_600,
        };
        assert_eq!(LaneHello::from_bytes(&hello.to_bytes()).unwrap(), hello);
        let mut bad = hello.to_bytes();
        bad[0] = b'X';
        assert!(LaneHello::from_bytes(&bad).is_err());
    }

    #[tokio::test]
    async fn streams_arrive_in_order_across_lanes_in_both_directions() {
        let mut listener = listener().await;
        let client = UsbCBridge::connect(config(listener.local_addr(), &[3.0, 2.0, 1.0]))
            .await
            .unwrap();
        let mut server = listener.accept().await.unwrap();
        assert!(
            client
                .bridge
                .wait_for_lanes(3, Duration::from_secs(5))
                .await
        );

        let upstream = noise(1, 3 << 20);
        let downstream = noise(2, 2 << 20);
        let (mut c_reader, mut c_writer) = (client.reader, client.writer);
        let up = upstream.clone();
        let send_up = tokio::spawn(async move {
            // Uneven writes exercise coalescing and segment boundaries.
            for chunk in up.chunks(7_919) {
                c_writer.write_all(chunk).await.unwrap();
            }
            c_writer.shutdown().await.unwrap();
        });
        let down = downstream.clone();
        let mut s_writer = server.writer;
        let send_down = tokio::spawn(async move {
            s_writer.write_all(&down).await.unwrap();
            s_writer.shutdown().await.unwrap();
        });
        let mut got_up = Vec::new();
        server.reader.read_to_end(&mut got_up).await.unwrap();
        let mut got_down = Vec::new();
        c_reader.read_to_end(&mut got_down).await.unwrap();
        send_up.await.unwrap();
        send_down.await.unwrap();
        assert!(got_up == upstream, "upstream corrupted");
        assert!(got_down == downstream, "downstream corrupted");

        let snap = client.bridge.snapshot();
        assert!(
            snap.lanes.iter().filter(|l| l.bytes_sent > 0).count() >= 2,
            "bulk data should spread over several lanes: {:?}",
            snap.lanes
        );
        assert_eq!(snap.retransmitted_segments, 0);
        // Each side finishes once both ends of stream are acknowledged.
        wait_closed(&client.bridge).await;
        wait_closed(&server.bridge).await;
        assert_eq!(
            client.bridge.snapshot().close_reason.as_deref(),
            Some("finished")
        );
    }

    #[tokio::test]
    async fn small_frames_stay_on_the_heaviest_lane() {
        let mut listener = listener().await;
        let client = UsbCBridge::connect(config(listener.local_addr(), &[27.6, 1.2]))
            .await
            .unwrap();
        let mut server = listener.accept().await.unwrap();
        assert!(
            client
                .bridge
                .wait_for_lanes(2, Duration::from_secs(5))
                .await
        );
        let mut writer = client.writer;
        for i in 0..50u32 {
            let msg = i.to_le_bytes();
            writer.write_all(&msg).await.unwrap();
            let mut back = [0u8; 4];
            server.reader.read_exact(&mut back).await.unwrap();
            assert_eq!(back, msg);
        }
        let lanes = client.bridge.snapshot().lanes;
        assert_eq!(lanes[0].bytes_sent, 200);
        assert_eq!(lanes[1].bytes_sent, 0);
    }

    #[tokio::test]
    async fn unplugging_a_lane_mid_transfer_loses_nothing() {
        let mut listener = listener().await;
        let client = UsbCBridge::connect(config(listener.local_addr(), &[2.0, 1.0]))
            .await
            .unwrap();
        let mut server = listener.accept().await.unwrap();
        assert!(
            client
                .bridge
                .wait_for_lanes(2, Duration::from_secs(5))
                .await
        );

        let payload = noise(3, 8 << 20);
        let expected = payload.clone();
        let bridge = client.bridge.clone();
        let mut writer = client.writer;
        let sender = tokio::spawn(async move {
            for (i, chunk) in payload.chunks(256 << 10).enumerate() {
                writer.write_all(chunk).await.unwrap();
                if i == 8 {
                    bridge.set_lane_enabled(0, false).unwrap();
                }
            }
            writer.shutdown().await.unwrap();
        });
        let mut got = Vec::new();
        server.reader.read_to_end(&mut got).await.unwrap();
        sender.await.unwrap();
        assert!(got == expected, "stream corrupted across the unplug");
        let snap = client.bridge.snapshot();
        assert!(!snap.lanes[0].alive);
        assert!(snap.lanes[0].failures >= 1);
        assert!(snap.lanes[1].bytes_sent > 0);
    }

    #[tokio::test]
    async fn a_replugged_lane_resumes_the_same_session() {
        let mut listener = listener().await;
        let client = UsbCBridge::connect(config(listener.local_addr(), &[1.0]))
            .await
            .unwrap();
        let mut server = listener.accept().await.unwrap();
        let (mut writer, bridge) = (client.writer, client.bridge);

        writer.write_all(b"before ").await.unwrap();
        let mut buf = [0u8; 7];
        server.reader.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"before ");

        // Pull the only cable, write while it is out, then plug it back in.
        bridge.set_lane_enabled(0, false).unwrap();
        writer.write_all(b"while unplugged").await.unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(bridge.snapshot().alive_lanes(), 0);
        bridge.set_lane_enabled(0, true).unwrap();
        assert!(bridge.wait_for_lanes(1, Duration::from_secs(5)).await);

        let mut buf = [0u8; 15];
        server.reader.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"while unplugged");
        let snap = bridge.snapshot();
        assert_eq!(snap.lanes[0].attaches, 2);
        assert!(snap.open);
        assert!(server.bridge.is_open());
        // The resumed lane joined the existing session instead of announcing a new one.
        assert!(
            tokio::time::timeout(Duration::from_millis(100), listener.accept())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn a_lane_added_mid_session_carries_traffic() {
        let mut listener = listener().await;
        let addr = listener.local_addr();
        let client = UsbCBridge::connect(config(addr, &[1.0])).await.unwrap();
        let mut server = listener.accept().await.unwrap();
        // Plug in a much faster cable.
        let idx = client
            .bridge
            .add_lane(BridgeLane::new(addr.to_string(), 40.0, "usb4"))
            .unwrap();
        assert!(
            client
                .bridge
                .wait_for_lanes(2, Duration::from_secs(5))
                .await
        );
        let mut writer = client.writer;
        writer.write_all(b"over the new cable").await.unwrap();
        let mut buf = [0u8; 18];
        server.reader.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"over the new cable");
        assert_eq!(client.bridge.snapshot().lanes[idx].bytes_sent, 18);
    }

    #[tokio::test]
    async fn a_session_without_lanes_past_its_grace_closes() {
        let mut listener = listener().await;
        let mut cfg = config(listener.local_addr(), &[1.0]);
        cfg.resume_grace = Duration::from_millis(200);
        let client = UsbCBridge::connect(cfg).await.unwrap();
        let _server = listener.accept().await.unwrap();
        client.bridge.set_lane_enabled(0, false).unwrap();
        let mut reader = client.reader;
        let mut rest = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), reader.read_to_end(&mut rest))
            .await
            .expect("reader sees EOF once the session ends")
            .unwrap();
        let reason = client.bridge.snapshot().close_reason.unwrap();
        assert!(reason.contains("resume grace"), "{reason}");
    }

    fn raw_hello(session_id: u64, flags: u16) -> LaneHello {
        LaneHello {
            version: BRIDGE_VERSION,
            flags,
            session_id,
            lane: 0,
            segment_bytes: 16 << 10,
            window_bytes: 1 << 20,
            weight_mbps: 1000,
        }
    }

    #[tokio::test]
    async fn resuming_an_unknown_session_or_bad_parameters_is_refused() {
        let listener = listener().await;
        let addr = listener.local_addr().to_string();
        let err = dial_lane(&addr, raw_hello(42, 0), Duration::from_secs(2))
            .await
            .unwrap_err();
        assert!(matches!(err, DialError::UnknownSession));

        let bad = LaneHello {
            segment_bytes: 1,
            ..raw_hello(43, FLAG_CREATE)
        };
        let err = dial_lane(&addr, bad, Duration::from_secs(2))
            .await
            .unwrap_err();
        assert!(matches!(err, DialError::Other(e) if e.contains("rejected")));
    }

    #[tokio::test]
    async fn a_segment_beyond_the_window_closes_the_session() {
        let mut listener = listener().await;
        let addr = listener.local_addr().to_string();
        let mut raw = dial_lane(&addr, raw_hello(7, FLAG_CREATE), Duration::from_secs(2))
            .await
            .unwrap();
        let server = listener.accept().await.unwrap();
        let mut msg = vec![TAG_DATA];
        msg.extend_from_slice(&(1u64 << 40).to_le_bytes());
        msg.extend_from_slice(&4u32.to_le_bytes());
        msg.extend_from_slice(b"evil");
        raw.write_all(&msg).await.unwrap();
        wait_closed(&server.bridge).await;
        let reason = server.bridge.snapshot().close_reason.unwrap();
        assert!(reason.contains("beyond the window"), "{reason}");
    }

    #[tokio::test]
    async fn a_silent_lane_is_dropped_after_its_idle_timeout() {
        let mut listener = listener().await;
        let addr = listener.local_addr().to_string();
        // A lane that completes the hello and then never sends a byte, like a pulled cable.
        let _raw = dial_lane(&addr, raw_hello(8, FLAG_CREATE), Duration::from_secs(2))
            .await
            .unwrap();
        let server = listener.accept().await.unwrap();
        assert_eq!(server.bridge.snapshot().alive_lanes(), 1);
        tokio::time::sleep(Duration::from_millis(900)).await;
        let snap = server.bridge.snapshot();
        assert_eq!(snap.alive_lanes(), 0);
        let error = snap.lanes[0].last_error.clone().unwrap();
        assert!(error.contains("no data or heartbeat"), "{error}");
        assert!(snap.open, "the session waits out its resume grace");
    }

    /// Answers every frame with the XXH64 of its payload, over a delta link.
    async fn serve_digests(mut listener: UsbCBridgeListener) {
        while let Some(session) = listener.accept().await {
            tokio::spawn(async move {
                let _bridge = session.bridge;
                let mut reader = FrameReader::new(BufReader::new(session.reader));
                let mut writer = FrameWriter::new(session.writer);
                let config = DeltaLinkConfig::default();
                while let Ok(frame) = reader.read_frame().await {
                    if frame.kind == WireFrameKind::DeltaHello {
                        accept_delta_hello(&frame, &config, &mut reader, &mut writer)
                            .await
                            .unwrap();
                        continue;
                    }
                    let digest = xxh64(&frame.payload, 0).to_le_bytes().to_vec();
                    let reply = WireFrame::new(WireFrameKind::WorkResult, frame.epoch, digest)
                        .with_request_id(frame.request_id);
                    writer.write_frame(&reply).await.unwrap();
                }
            });
        }
    }

    #[tokio::test]
    async fn delta_stores_survive_an_unplug_so_only_changes_cross_the_backup() {
        let listener = listener().await;
        let addr = listener.local_addr();
        tokio::spawn(serve_digests(listener));
        let channel = PeerChannel::connect_with(
            &format!("kdbr://{addr}@27.6,{addr}@1.2"),
            Duration::from_secs(5),
            &DeltaLinkConfig::default(),
        )
        .await
        .unwrap();
        assert!(channel.delta_link_active());
        let bridge = channel.bridge().expect("dialed over a bridge").clone();
        assert!(bridge.wait_for_lanes(2, Duration::from_secs(5)).await);

        // A 4 MiB stage context re-sent with a changing header, like a decode loop.
        let context = noise(9, 4 << 20);
        let call = |step: u32| {
            let mut payload = format!("step {step}|").into_bytes();
            payload.extend_from_slice(&context);
            let digest = xxh64(&payload, 0);
            let channel = Arc::clone(&channel);
            async move {
                let reply = channel
                    .call(
                        WireFrameKind::WorkDispatch,
                        step,
                        payload,
                        Duration::from_secs(10),
                    )
                    .await
                    .unwrap();
                assert_eq!(reply.payload, digest.to_le_bytes());
            }
        };
        call(0).await;
        // Unplug the USB4 lane: the session, and with it both delta stores, carry on over the
        // backup lane.
        bridge.set_lane_enabled(0, false).unwrap();
        let before = channel.delta_link_stats().unwrap().0;
        for step in 1..4 {
            call(step).await;
        }
        let after = channel.delta_link_stats().unwrap().0;
        let wire = after.wire_bytes - before.wire_bytes;
        let logical = after.logical_bytes - before.logical_bytes;
        assert!(
            wire * 100 < logical,
            "after the unplug only changed chunks should cross: {wire} wire bytes for {logical} logical"
        );
        assert!(channel.is_healthy());
        assert!(bridge.snapshot().lanes[1].bytes_sent > 0);
    }
}
