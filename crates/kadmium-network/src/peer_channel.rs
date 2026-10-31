//! # `kadmium_network::peer_channel`
//!
//! Persistent, multiplexed request/response channels between Kadmium nodes.
//!
//! Before this module every hop between two nodes opened a fresh TCP connection, wrote one
//! `WireFrame`, read one reply, and closed. On a Wi-Fi link that cost 20 to 150 ms per hop for a
//! 4 KB payload, almost all of it connection setup, and it made concurrent requests to one worker
//! impossible. A [`PeerChannel`] keeps a single TCP connection open per peer, tags every outbound
//! frame with a `u64` request id (the `KDM2` header), and demultiplexes replies by that id, so any
//! number of calls can be in flight on one socket and replies may arrive out of order. A
//! [`PeerChannelPool`] caches channels by address and reconnects transparently when one dies.
//!
//! The API is deliberately generic (a `WireFrameKind` plus opaque payload bytes): phase 3 of the
//! grid plan dispatches `kadmium-grid` work units over exactly this channel.
//!
//! A channel opened with [`PeerChannel::connect_with`] (as a pool built with
//! [`PeerChannelPool::with_delta_link`] does) first negotiates a [`crate::delta_link`], so frames
//! in both directions carry only the chunks the other side does not already hold. A peer from
//! before delta links gets an ordinary channel.
//!
//! An address in the [`crate::BRIDGE_ADDR_SCHEME`] form (`kdbr://usb4-addr@27.6,wifi-addr@1.2`)
//! dials a [`crate::UsbCBridge`] instead of one TCP connection: the channel's frames are striped
//! over every lane, and unplugging a lane neither fails a call nor resets the delta link.

use crate::delta_link::{
    ChunkPlan, DeltaDecoder, DeltaEncoder, DeltaHello, DeltaLinkConfig, DeltaLinkSnapshot,
    DeltaLinkStats, FrameReader, FrameWriter,
};
use crate::usbc_bridge::{BridgeConfig, UsbCBridge};
use crate::{WireFrame, WireFrameKind};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, BufReader};
use tokio::net::TcpStream;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

/// Default bound on establishing a TCP connection to a peer.
pub const DEFAULT_PEER_CONNECT_TIMEOUT: Duration = Duration::from_millis(2500);

/// Default bound on one request/reply round trip once connected.
pub const DEFAULT_PEER_CALL_TIMEOUT: Duration = Duration::from_secs(30);

type ReplySender = oneshot::Sender<Result<WireFrame, String>>;

/// Read and write halves of whatever carries a channel: one TCP connection or a bridge.
type ChannelRead = Box<dyn AsyncRead + Send + Unpin>;
type ChannelWrite = Box<dyn AsyncWrite + Send + Unpin>;

/// State shared between a [`PeerChannel`] and its background reader task.
struct ChannelShared {
    addr: String,
    pending: Mutex<HashMap<u64, ReplySender>>,
    healthy: AtomicBool,
    close_reason: Mutex<Option<String>>,
}

impl ChannelShared {
    fn pending(&self) -> MutexGuard<'_, HashMap<u64, ReplySender>> {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn close_reason(&self) -> String {
        self.close_reason
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .unwrap_or_else(|| "closed".to_string())
    }

    /// Marks the channel unhealthy (keeping the first recorded `reason`) and fails every
    /// in-flight call so no caller is left waiting on a dead socket.
    fn fail_all(&self, reason: String) {
        self.healthy.store(false, Ordering::SeqCst);
        {
            let mut slot = self
                .close_reason
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if slot.is_none() {
                *slot = Some(reason);
            }
        }
        let reason = self.close_reason();
        let drained: Vec<(u64, ReplySender)> = self.pending().drain().collect();
        for (request_id, tx) in drained {
            let _ = tx.send(Err(format!(
                "peer channel to {} failed while request {request_id} was in flight: {reason}",
                self.addr
            )));
        }
    }

    /// Routes an inbound frame to the call waiting on its request id.
    fn dispatch_reply(&self, frame: WireFrame) {
        let waiter = {
            let mut pending = self.pending();
            match pending.remove(&frame.request_id) {
                Some(tx) => Some(tx),
                // A legacy `KDMF` peer echoes no request id. With exactly one call in flight the
                // reply is unambiguous, which keeps strictly serial callers working during a
                // mixed-version rollout.
                None if frame.request_id == 0 && pending.len() == 1 => {
                    let only = *pending.keys().next().expect("one pending request");
                    pending.remove(&only)
                }
                None => None,
            }
        };
        if let Some(tx) = waiter {
            // The caller may already have timed out and dropped its receiver; not an error.
            let _ = tx.send(Ok(frame));
        }
    }
}

/// Runs on its own task for the lifetime of the connection: reads replies and demultiplexes them.
async fn run_reader(mut reader: FrameReader<BufReader<ChannelRead>>, shared: Arc<ChannelShared>) {
    /// Fails pending calls even when this task is cancelled (runtime shutdown, channel drop).
    struct FailPendingOnDrop(Arc<ChannelShared>);
    impl Drop for FailPendingOnDrop {
        fn drop(&mut self) {
            self.0.fail_all("reader task stopped".to_string());
        }
    }
    let _guard = FailPendingOnDrop(Arc::clone(&shared));

    loop {
        match reader.read_frame().await {
            Ok(frame) => shared.dispatch_reply(frame),
            Err(e) => {
                shared.fail_all(format!("connection closed: {e}"));
                break;
            }
        }
    }
}

/// One persistent, multiplexed TCP connection to a peer.
///
/// Create one with [`PeerChannel::connect`] (or [`PeerChannel::from_stream`]), then issue any
/// number of concurrent [`PeerChannel::call`]s. Each call writes a `KDM2` frame tagged with a
/// fresh request id and resolves when the reply carrying that id arrives, whatever order the peer
/// answers in. When the connection ends (EOF, reset, malformed frame, write failure) every
/// in-flight call fails and the channel reports unhealthy; it is never reused after that.
pub struct PeerChannel {
    shared: Arc<ChannelShared>,
    writer: tokio::sync::Mutex<FrameWriter<ChannelWrite>>,
    next_request_id: AtomicU64,
    reader_task: JoinHandle<()>,
    /// Outbound and inbound byte counts, when a delta link was negotiated.
    delta: Option<(Arc<DeltaLinkStats>, Arc<DeltaLinkStats>)>,
    /// The bridge carrying this channel, when it was dialed with a bridge address.
    bridge: Option<UsbCBridge>,
}

/// Fails the channel if a delta-encoded frame is dropped before it is fully written: the encoder
/// already counts its chunks as held by the peer, so the stream cannot continue.
struct UnwrittenFrameGuard<'a> {
    shared: &'a ChannelShared,
    armed: bool,
}

impl Drop for UnwrittenFrameGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.shared
                .fail_all("a delta-encoded frame was cancelled before it was written".into());
        }
    }
}

impl std::fmt::Debug for PeerChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerChannel")
            .field("addr", &self.shared.addr)
            .field("healthy", &self.is_healthy())
            .field("in_flight", &self.in_flight())
            .field("delta_link", &self.delta_link_active())
            .finish()
    }
}

impl PeerChannel {
    /// Connects to `addr` within `connect_timeout` and starts the reply reader task.
    pub async fn connect(addr: &str, connect_timeout: Duration) -> Result<Arc<Self>, String> {
        if BridgeConfig::parse_addr(addr).is_some() {
            return Self::connect_bridge(addr, connect_timeout, &DeltaLinkConfig::disabled()).await;
        }
        let stream = Self::dial(addr, connect_timeout).await?;
        Ok(Self::from_stream(addr, stream))
    }

    /// Connects to `addr` and, when `delta` is enabled, negotiates a delta link first.
    ///
    /// A peer from before delta links closes the connection on the unknown hello frame, and a
    /// peer that does not answer within `connect_timeout` is treated the same way: the channel
    /// reconnects and comes back as an ordinary one. A peer that declines keeps the connection
    /// as an ordinary channel.
    pub async fn connect_with(
        addr: &str,
        connect_timeout: Duration,
        delta: &DeltaLinkConfig,
    ) -> Result<Arc<Self>, String> {
        if BridgeConfig::parse_addr(addr).is_some() {
            return Self::connect_bridge(addr, connect_timeout, delta).await;
        }
        if !delta.enabled {
            return Self::connect(addr, connect_timeout).await;
        }
        let stream = Self::dial(addr, connect_timeout).await?;
        let _ = stream.set_nodelay(true);
        let (read_half, write_half) = stream.into_split();
        let mut reader = FrameReader::new(BufReader::new(Box::new(read_half) as ChannelRead));
        let mut writer = FrameWriter::new(Box::new(write_half) as ChannelWrite);
        match negotiate_delta(&mut reader, &mut writer, connect_timeout, delta).await {
            Ok(()) => Ok(Self::from_halves(addr, reader, writer, None)),
            Err(_) => Self::connect(addr, connect_timeout).await,
        }
    }

    /// Dials a bridge address and, when `delta` is enabled, negotiates a delta link across it.
    /// Bridge listeners are newer than delta links, so a failed hello is an error rather than a
    /// reason to retry without one.
    async fn connect_bridge(
        addr: &str,
        connect_timeout: Duration,
        delta: &DeltaLinkConfig,
    ) -> Result<Arc<Self>, String> {
        let mut config = BridgeConfig::parse_addr(addr).expect("caller checked the scheme")?;
        config.connect_timeout = connect_timeout;
        let session = UsbCBridge::connect(config).await?;
        let mut reader = FrameReader::new(BufReader::new(Box::new(session.reader) as ChannelRead));
        let mut writer = FrameWriter::new(Box::new(session.writer) as ChannelWrite);
        if delta.enabled
            && let Err(e) = negotiate_delta(&mut reader, &mut writer, connect_timeout, delta).await
        {
            session.bridge.close();
            return Err(format!("delta link over bridge {addr}: {e}"));
        }
        Ok(Self::from_halves(
            addr,
            reader,
            writer,
            Some(session.bridge),
        ))
    }

    async fn dial(addr: &str, connect_timeout: Duration) -> Result<TcpStream, String> {
        tokio::time::timeout(connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| format!("TCP connect timeout to peer {addr} after {connect_timeout:?}"))?
            .map_err(|e| format!("TCP connect error to peer {addr}: {e}"))
    }

    /// Wraps an already-connected stream. Must run inside a Tokio runtime: it spawns the reader.
    pub fn from_stream(addr: &str, stream: TcpStream) -> Arc<Self> {
        // Request/reply traffic is latency-bound: never let Nagle hold a small frame back.
        let _ = stream.set_nodelay(true);
        let (read_half, write_half) = stream.into_split();
        Self::from_halves(
            addr,
            FrameReader::new(BufReader::new(Box::new(read_half) as ChannelRead)),
            FrameWriter::new(Box::new(write_half) as ChannelWrite),
            None,
        )
    }

    fn from_halves(
        addr: &str,
        reader: FrameReader<BufReader<ChannelRead>>,
        writer: FrameWriter<ChannelWrite>,
        bridge: Option<UsbCBridge>,
    ) -> Arc<Self> {
        let delta = writer.delta_stats().zip(reader.delta_stats());
        let shared = Arc::new(ChannelShared {
            addr: addr.to_string(),
            pending: Mutex::new(HashMap::new()),
            healthy: AtomicBool::new(true),
            close_reason: Mutex::new(None),
        });
        let reader_task = tokio::spawn(run_reader(reader, Arc::clone(&shared)));
        Arc::new(Self {
            shared,
            writer: tokio::sync::Mutex::new(writer),
            next_request_id: AtomicU64::new(1),
            reader_task,
            delta,
            bridge,
        })
    }

    /// The bridge carrying this channel, when it was dialed with a bridge address.
    pub fn bridge(&self) -> Option<&UsbCBridge> {
        self.bridge.as_ref()
    }

    /// `true` when this channel negotiated a delta link.
    pub fn delta_link_active(&self) -> bool {
        self.delta.is_some()
    }

    /// Byte counts of the delta link as `(outbound, inbound)`, or `None` on an ordinary channel.
    pub fn delta_link_stats(&self) -> Option<(DeltaLinkSnapshot, DeltaLinkSnapshot)> {
        self.delta
            .as_ref()
            .map(|(tx, rx)| (tx.snapshot(), rx.snapshot()))
    }

    /// Address this channel was connected to.
    pub fn addr(&self) -> &str {
        &self.shared.addr
    }

    /// `true` while the connection and its reader task are alive. Once `false` the channel stays
    /// dead; a [`PeerChannelPool`] replaces it on the next call.
    pub fn is_healthy(&self) -> bool {
        self.shared.healthy.load(Ordering::SeqCst) && !self.reader_task.is_finished()
    }

    /// Number of calls currently waiting for a reply.
    pub fn in_flight(&self) -> usize {
        self.shared.pending().len()
    }

    /// Closes the channel: fails in-flight calls, stops the reader, and lets the socket drop.
    pub fn close(&self) {
        self.shared.fail_all("closed locally".to_string());
        self.reader_task.abort();
    }

    /// Sends `payload` as a `kind` frame and awaits the reply carrying the same request id.
    pub async fn call(
        &self,
        kind: WireFrameKind,
        epoch: u32,
        payload: Vec<u8>,
        timeout: Duration,
    ) -> Result<WireFrame, String> {
        self.call_frame(&WireFrame::new(kind, epoch, payload), timeout)
            .await
    }

    /// Like [`PeerChannel::call`] but borrows a prepared frame (its `request_id` is replaced by a
    /// freshly allocated one), so a caller can retry the same frame without cloning the payload.
    pub async fn call_frame(
        &self,
        frame: &WireFrame,
        timeout: Duration,
    ) -> Result<WireFrame, String> {
        let addr = &self.shared.addr;
        if !self.is_healthy() {
            return Err(format!(
                "peer channel to {addr} is {}",
                self.shared.close_reason()
            ));
        }
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        // Chunking hashes the whole payload, so do it before taking the writer: concurrent calls
        // then serialize only on the store lookups and the write itself.
        let plan = self
            .delta_link_active()
            .then(|| ChunkPlan::new(&frame.payload));
        let (tx, rx) = oneshot::channel();
        self.shared.pending().insert(request_id, tx);
        let deadline = tokio::time::Instant::now() + timeout;

        // Taking the writer is cancel-safe: nothing has touched the socket yet.
        let Ok(mut writer) = tokio::time::timeout_at(deadline, self.writer.lock()).await else {
            self.shared.pending().remove(&request_id);
            return Err(format!(
                "timed out after {timeout:?} waiting to write {:?} to {addr}",
                frame.kind
            ));
        };
        let encoded = writer.encode(frame, request_id, plan.as_ref());
        let mut unwritten = UnwrittenFrameGuard {
            shared: &self.shared,
            armed: writer.delta_active(),
        };
        // A partially written frame would desynchronize the stream, so a write that fails or
        // times out ends the whole channel rather than just this call.
        let written = tokio::time::timeout_at(deadline, writer.write_encoded(&encoded)).await;
        unwritten.armed = false;
        match written {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                drop(writer);
                self.shared.fail_all(format!("write error: {e}"));
                return Err(format!("write of {:?} to {addr} failed: {e}", frame.kind));
            }
            Err(_) => {
                drop(writer);
                self.shared.fail_all(format!(
                    "write of request {request_id} timed out after {timeout:?}"
                ));
                return Err(format!(
                    "write of {:?} to {addr} timed out after {timeout:?}",
                    frame.kind
                ));
            }
        }
        drop(writer);

        match tokio::time::timeout_at(deadline, rx).await {
            Ok(Ok(Ok(reply))) => Ok(reply),
            Ok(Ok(Err(channel_failure))) => Err(channel_failure),
            Ok(Err(_)) => Err(format!(
                "peer channel to {addr} dropped request {request_id} without a reply"
            )),
            Err(_) => {
                self.shared.pending().remove(&request_id);
                Err(format!(
                    "timed out after {timeout:?} waiting for the {:?} reply to request {request_id} from {addr}",
                    frame.kind
                ))
            }
        }
    }
}

/// Client side of the delta-link hello on a fresh connection. On success the halves encode and
/// decode delta frames, unless the peer declined (an agreed budget of 0).
async fn negotiate_delta(
    reader: &mut FrameReader<BufReader<ChannelRead>>,
    writer: &mut FrameWriter<ChannelWrite>,
    timeout: Duration,
    delta: &DeltaLinkConfig,
) -> Result<(), String> {
    let hello = async {
        writer.write_plain(&DeltaHello::request(delta)).await?;
        let reply = reader.read_frame().await?;
        if reply.kind != WireFrameKind::DeltaHello {
            return Err(format!("peer answered DeltaHello with {:?}", reply.kind));
        }
        DeltaHello::from_bytes(&reply.payload)
    };
    let agreed = tokio::time::timeout(timeout, hello)
        .await
        .map_err(|_| format!("no DeltaHello answer within {timeout:?}"))??;
    // Never trust a reply that raises the budget this side proposed.
    let budget = agreed.store_bytes.min(delta.store_bytes);
    if budget > 0 {
        reader.enable_delta(DeltaDecoder::new(budget));
        writer.enable_delta(DeltaEncoder::new(budget));
    }
    Ok(())
}

impl Drop for PeerChannel {
    fn drop(&mut self) {
        self.close();
    }
}

type ChannelSlot = Arc<tokio::sync::Mutex<Option<Arc<PeerChannel>>>>;

struct PoolInner {
    connect_timeout: Duration,
    delta: DeltaLinkConfig,
    slots: Mutex<HashMap<String, ChannelSlot>>,
}

/// Address-keyed cache of [`PeerChannel`]s. Cheap to clone; every clone shares the same pool.
///
/// [`PeerChannelPool::call`] is the one-stop entry point: it reuses the live channel for the
/// address, connects when there is none, and if the channel dies underneath the call it
/// reconnects once and retries. Connects to one address are serialized so concurrent first calls
/// share a single connection; different addresses never block each other.
#[derive(Clone)]
pub struct PeerChannelPool {
    inner: Arc<PoolInner>,
}

impl Default for PeerChannelPool {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PeerChannelPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerChannelPool")
            .field("connect_timeout", &self.inner.connect_timeout)
            .field("addresses", &self.len())
            .finish()
    }
}

impl PeerChannelPool {
    /// A pool using [`DEFAULT_PEER_CONNECT_TIMEOUT`] for new connections.
    pub fn new() -> Self {
        Self::with_connect_timeout(DEFAULT_PEER_CONNECT_TIMEOUT)
    }

    /// A pool that bounds every new connection attempt by `connect_timeout`.
    pub fn with_connect_timeout(connect_timeout: Duration) -> Self {
        Self::with_options(connect_timeout, DeltaLinkConfig::disabled())
    }

    /// A pool whose new channels negotiate a delta link under `delta` (see
    /// [`PeerChannel::connect_with`]).
    pub fn with_delta_link(delta: DeltaLinkConfig) -> Self {
        Self::with_options(DEFAULT_PEER_CONNECT_TIMEOUT, delta)
    }

    pub fn with_options(connect_timeout: Duration, delta: DeltaLinkConfig) -> Self {
        Self {
            inner: Arc::new(PoolInner {
                connect_timeout,
                delta,
                slots: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Delta-link settings applied to new channels.
    pub fn delta_link(&self) -> DeltaLinkConfig {
        self.inner.delta
    }

    /// Connect timeout applied by [`PeerChannelPool::get_or_connect`] and [`PeerChannelPool::call`].
    pub fn connect_timeout(&self) -> Duration {
        self.inner.connect_timeout
    }

    fn slot(&self, addr: &str) -> ChannelSlot {
        let mut slots = self
            .inner
            .slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Arc::clone(slots.entry(addr.to_string()).or_default())
    }

    /// Returns the healthy channel for `addr`, connecting (or replacing a dead channel) if needed.
    pub async fn get_or_connect(&self, addr: &str) -> Result<Arc<PeerChannel>, String> {
        self.get_or_connect_with_timeout(addr, self.inner.connect_timeout)
            .await
    }

    /// [`PeerChannelPool::get_or_connect`] with an explicit bound on the connection attempt.
    pub async fn get_or_connect_with_timeout(
        &self,
        addr: &str,
        connect_timeout: Duration,
    ) -> Result<Arc<PeerChannel>, String> {
        let slot = self.slot(addr);
        let mut guard = slot.lock().await;
        if let Some(channel) = guard.as_ref()
            && channel.is_healthy()
        {
            return Ok(Arc::clone(channel));
        }
        let fresh = PeerChannel::connect_with(addr, connect_timeout, &self.inner.delta).await?;
        *guard = Some(Arc::clone(&fresh));
        Ok(fresh)
    }

    /// Sends one request to `addr` over the pooled channel and awaits its reply, reconnecting
    /// transparently once if the channel turns out to be dead.
    pub async fn call(
        &self,
        addr: &str,
        kind: WireFrameKind,
        epoch: u32,
        payload: Vec<u8>,
        timeout: Duration,
    ) -> Result<WireFrame, String> {
        self.call_with_connect_timeout(
            addr,
            self.inner.connect_timeout,
            kind,
            epoch,
            payload,
            timeout,
        )
        .await
    }

    /// [`PeerChannelPool::call`] with an explicit bound on any connection attempt it has to make.
    pub async fn call_with_connect_timeout(
        &self,
        addr: &str,
        connect_timeout: Duration,
        kind: WireFrameKind,
        epoch: u32,
        payload: Vec<u8>,
        timeout: Duration,
    ) -> Result<WireFrame, String> {
        let frame = WireFrame::new(kind, epoch, payload);
        let channel = self
            .get_or_connect_with_timeout(addr, connect_timeout)
            .await?;
        match channel.call_frame(&frame, timeout).await {
            Ok(reply) => Ok(reply),
            // The connection died under this call (peer restart, reset, stale socket): dial once
            // more and resend. A timeout on a healthy channel is not retried, because the peer may
            // still be working on the request.
            Err(first_error) if !channel.is_healthy() => {
                let retry = self
                    .get_or_connect_with_timeout(addr, connect_timeout)
                    .await
                    .map_err(|e| format!("{first_error}; reconnect failed: {e}"))?;
                retry
                    .call_frame(&frame, timeout)
                    .await
                    .map_err(|e| format!("{first_error}; retry failed: {e}"))
            }
            Err(e) => Err(e),
        }
    }

    /// Drops the pooled channel for `addr`, closing it. Returns `false` if none was pooled.
    pub fn evict(&self, addr: &str) -> bool {
        let removed = self
            .inner
            .slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(addr);
        match removed {
            Some(slot) => {
                // Close now unless a connect is mid-flight on this slot; in that case the
                // orphaned slot (and its channel) drops as soon as that attempt finishes.
                if let Ok(mut guard) = slot.try_lock()
                    && let Some(channel) = guard.take()
                {
                    channel.close();
                }
                true
            }
            None => false,
        }
    }

    /// Number of addresses with a pooled slot (healthy or not).
    pub fn len(&self) -> usize {
        self.inner
            .slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// `true` when no address has been pooled.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::net::SocketAddr;
    use std::sync::atomic::AtomicUsize;
    use tokio::net::TcpListener;

    /// Loopback server built on the same split-reader / shared-writer loop as the live worker:
    /// every frame is answered on its own task after a pseudo-random delay, so replies to
    /// pipelined requests come back out of order. `frames_per_connection` makes it hang up after
    /// that many frames to simulate a peer dropping the connection.
    struct JitterServer {
        addr: SocketAddr,
        accepted: Arc<AtomicUsize>,
        reply_order: Arc<Mutex<Vec<u64>>>,
    }

    async fn spawn_jitter_server(frames_per_connection: Option<usize>) -> JitterServer {
        spawn_jitter_server_with(frames_per_connection, None).await
    }

    /// [`spawn_jitter_server`] that accepts delta links under `delta`. With `None` it behaves
    /// like a server from before delta links: a `DeltaHello` makes it hang up.
    async fn spawn_jitter_server_with(
        frames_per_connection: Option<usize>,
        delta: Option<DeltaLinkConfig>,
    ) -> JitterServer {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let accepted = Arc::new(AtomicUsize::new(0));
        let reply_order = Arc::new(Mutex::new(Vec::new()));
        let accepted_ref = Arc::clone(&accepted);
        let order_ref = Arc::clone(&reply_order);

        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                accepted_ref.fetch_add(1, Ordering::SeqCst);
                let order = Arc::clone(&order_ref);
                tokio::spawn(async move {
                    let (read_half, write_half) = stream.into_split();
                    let mut reader = FrameReader::new(BufReader::new(read_half));
                    let writer = Arc::new(tokio::sync::Mutex::new(FrameWriter::new(write_half)));
                    let mut served = 0usize;
                    loop {
                        if frames_per_connection.is_some_and(|limit| served >= limit) {
                            break;
                        }
                        let Ok(frame) = reader.read_frame().await else {
                            break;
                        };
                        if frame.kind == WireFrameKind::DeltaHello {
                            let Some(config) = delta else {
                                break;
                            };
                            let mut w = writer.lock().await;
                            let accepted = crate::delta_link::accept_delta_hello(
                                &frame,
                                &config,
                                &mut reader,
                                &mut w,
                            )
                            .await;
                            if accepted.is_err() {
                                break;
                            }
                            continue;
                        }
                        served += 1;
                        let writer = Arc::clone(&writer);
                        let order = Arc::clone(&order);
                        tokio::spawn(async move {
                            let jitter_ms = (frame.request_id.wrapping_mul(0x9E37_79B9) >> 7) % 23;
                            tokio::time::sleep(Duration::from_millis(jitter_ms)).await;
                            let mut echoed = frame.payload.clone();
                            echoed.reverse();
                            let reply =
                                WireFrame::new(WireFrameKind::WorkResult, frame.epoch, echoed)
                                    .with_request_id(frame.request_id);
                            let mut w = writer.lock().await;
                            let _ = w.write_frame(&reply).await;
                            order.lock().expect("order lock").push(frame.request_id);
                        });
                    }
                });
            }
        });

        JitterServer {
            addr,
            accepted,
            reply_order,
        }
    }

    #[tokio::test]
    async fn test_peer_channel_100_concurrent_calls_demultiplex_out_of_order_replies() {
        let server = spawn_jitter_server(None).await;
        let addr = server.addr.to_string();
        let channel = PeerChannel::connect(&addr, Duration::from_secs(5))
            .await
            .expect("connect");
        assert!(channel.is_healthy());
        assert_eq!(channel.addr(), addr);
        assert_eq!(channel.in_flight(), 0);

        let mut tasks = Vec::with_capacity(100);
        for i in 0..100u32 {
            let ch = Arc::clone(&channel);
            tasks.push(tokio::spawn(async move {
                // Distinct payload per call so a misrouted reply is detected.
                let payload: Vec<u8> = (0..=(i % 7) as u8)
                    .map(|b| (i as u8).wrapping_add(b))
                    .collect();
                let reply = ch
                    .call(
                        WireFrameKind::WorkDispatch,
                        i,
                        payload.clone(),
                        Duration::from_secs(5),
                    )
                    .await
                    .expect("call completes");
                (i, payload, reply)
            }));
        }

        let mut request_ids = HashSet::new();
        for task in tasks {
            let (i, payload, reply) = task.await.expect("join call task");
            assert_eq!(reply.kind, WireFrameKind::WorkResult);
            assert_eq!(reply.epoch, i);
            let mut expected = payload;
            expected.reverse();
            assert_eq!(
                reply.payload, expected,
                "reply for call {i} reached the wrong caller"
            );
            assert_ne!(reply.request_id, 0);
            assert!(request_ids.insert(reply.request_id), "request id reused");
        }
        assert_eq!(request_ids.len(), 100);
        assert_eq!(channel.in_flight(), 0);
        assert!(channel.is_healthy());

        let order = server.reply_order.lock().expect("order lock").clone();
        assert_eq!(order.len(), 100);
        assert!(
            order.windows(2).any(|w| w[0] > w[1]),
            "server replied strictly in order, so out-of-order demultiplexing was not exercised"
        );
        assert_eq!(server.accepted.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_peer_channel_pool_reconnects_after_peer_hangs_up_and_evicts_on_demand() {
        // The server drops every connection after two frames.
        let server = spawn_jitter_server(Some(2)).await;
        let addr = server.addr.to_string();
        let pool = PeerChannelPool::with_connect_timeout(Duration::from_secs(5));
        let timeout = Duration::from_secs(5);
        assert!(pool.is_empty());

        let r1 = pool
            .call(&addr, WireFrameKind::WorkRequest, 1, vec![1, 2, 3], timeout)
            .await
            .expect("call 1");
        assert_eq!(r1.payload, vec![3, 2, 1]);
        let r2 = pool
            .call(&addr, WireFrameKind::WorkRequest, 2, vec![4, 5], timeout)
            .await
            .expect("call 2");
        assert_eq!(r2.payload, vec![5, 4]);
        assert_eq!(server.accepted.load(Ordering::SeqCst), 1);
        assert_eq!(pool.len(), 1);

        // The peer has hung up. Whether the channel already noticed (replaced up front) or notices
        // mid-call (reconnect + retry), the third call still succeeds on a second connection.
        let r3 = pool
            .call(&addr, WireFrameKind::WorkRequest, 3, vec![6], timeout)
            .await
            .expect("call 3 after peer hang-up");
        assert_eq!(r3.payload, vec![6]);
        assert_eq!(server.accepted.load(Ordering::SeqCst), 2);

        // Explicit eviction closes the pooled channel; the next call dials a fresh connection.
        let pooled = pool.get_or_connect(&addr).await.expect("pooled channel");
        assert!(pooled.is_healthy());
        assert!(pool.evict(&addr));
        assert!(!pool.evict(&addr));
        assert!(!pooled.is_healthy());
        assert_eq!(pool.len(), 0);
        let r4 = pool
            .call(&addr, WireFrameKind::WorkRequest, 4, vec![7, 8], timeout)
            .await
            .expect("call 4 after evict");
        assert_eq!(r4.payload, vec![8, 7]);
        assert_eq!(server.accepted.load(Ordering::SeqCst), 3);
        assert_eq!(pool.len(), 1);

        // A call on the evicted channel fails fast instead of hanging.
        let err = pooled
            .call(WireFrameKind::WorkRequest, 5, Vec::new(), timeout)
            .await
            .expect_err("dead channel must fail");
        assert!(err.contains("closed"), "{err}");

        // An unreachable peer fails with a connect error, not a hang.
        let vacant = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let vacant_addr = vacant.local_addr().expect("addr").to_string();
        drop(vacant);
        let err = pool
            .call(
                &vacant_addr,
                WireFrameKind::WorkRequest,
                6,
                Vec::new(),
                Duration::from_secs(2),
            )
            .await
            .expect_err("unreachable peer must fail");
        assert!(err.contains("connect"), "{err}");
    }

    /// 1 MiB payload that differs per call only in its first 8 bytes, like a work unit whose
    /// header changes while its weights do not.
    fn mostly_shared_payload(i: u64) -> Vec<u8> {
        let mut payload: Vec<u8> = (0..1u32 << 20)
            .map(|n| (n.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        payload[..8].copy_from_slice(&i.to_le_bytes());
        payload
    }

    #[tokio::test]
    async fn test_delta_channel_demultiplexes_concurrent_calls_and_reuses_shared_chunks() {
        let server = spawn_jitter_server_with(None, Some(DeltaLinkConfig::default())).await;
        let addr = server.addr.to_string();
        let channel =
            PeerChannel::connect_with(&addr, Duration::from_secs(5), &DeltaLinkConfig::default())
                .await
                .expect("connect");
        assert!(channel.delta_link_active());

        let mut tasks = Vec::new();
        for i in 0..32u64 {
            let ch = Arc::clone(&channel);
            tasks.push(tokio::spawn(async move {
                let payload = mostly_shared_payload(i);
                let reply = ch
                    .call(
                        WireFrameKind::WorkDispatch,
                        i as u32,
                        payload.clone(),
                        Duration::from_secs(10),
                    )
                    .await
                    .expect("call");
                let mut expected = payload;
                expected.reverse();
                assert_eq!(reply.kind, WireFrameKind::WorkResult);
                assert_eq!(reply.epoch, i as u32);
                assert_eq!(reply.payload, expected);
            }));
        }
        for task in tasks {
            task.await.expect("call task");
        }
        assert_eq!(server.reply_order.lock().unwrap().len(), 32);

        let (outbound, inbound) = channel.delta_link_stats().expect("delta stats");
        assert_eq!(outbound.logical_bytes, 32 << 20);
        assert_eq!(inbound.logical_bytes, 32 << 20);
        // Each direction sends about one payload's worth of literal chunks, then references.
        for stats in [outbound, inbound] {
            assert!(
                stats.wire_bytes < 4 << 20,
                "32 MiB took {} wire bytes",
                stats.wire_bytes
            );
        }
        assert_eq!(server.accepted.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_delta_channel_falls_back_to_a_peer_without_delta_links() {
        // A peer from before delta links hangs up on the hello; the channel redials plainly.
        let legacy = spawn_jitter_server(None).await;
        let addr = legacy.addr.to_string();
        let channel =
            PeerChannel::connect_with(&addr, Duration::from_secs(5), &DeltaLinkConfig::default())
                .await
                .expect("connect");
        assert!(!channel.delta_link_active());
        assert!(channel.delta_link_stats().is_none());
        let reply = channel
            .call(
                WireFrameKind::WorkDispatch,
                1,
                vec![1, 2, 3],
                Duration::from_secs(5),
            )
            .await
            .expect("plain call");
        assert_eq!(reply.payload, vec![3, 2, 1]);
        assert_eq!(legacy.accepted.load(Ordering::SeqCst), 2);

        // A peer that understands the hello but declines keeps the first connection.
        let declining = spawn_jitter_server_with(None, Some(DeltaLinkConfig::disabled())).await;
        let channel = PeerChannel::connect_with(
            &declining.addr.to_string(),
            Duration::from_secs(5),
            &DeltaLinkConfig::default(),
        )
        .await
        .expect("connect");
        assert!(!channel.delta_link_active());
        let reply = channel
            .call(
                WireFrameKind::WorkDispatch,
                1,
                vec![4, 5],
                Duration::from_secs(5),
            )
            .await
            .expect("plain call");
        assert_eq!(reply.payload, vec![5, 4]);
        assert_eq!(declining.accepted.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_delta_pool_reconnects_with_fresh_stores_after_the_peer_hangs_up() {
        // The server hangs up after two frames per connection. The redialed channel starts with
        // empty stores on both sides, so the resent payload must go out literally and still
        // arrive intact.
        let server = spawn_jitter_server_with(Some(2), Some(DeltaLinkConfig::default())).await;
        let addr = server.addr.to_string();
        let pool = PeerChannelPool::with_delta_link(DeltaLinkConfig::default());
        let timeout = Duration::from_secs(10);
        for i in 0..5u64 {
            let payload = mostly_shared_payload(i);
            let reply = pool
                .call(
                    &addr,
                    WireFrameKind::WorkDispatch,
                    1,
                    payload.clone(),
                    timeout,
                )
                .await
                .expect("pooled call");
            let mut expected = payload;
            expected.reverse();
            assert_eq!(reply.payload, expected);
        }
        let channel = pool.get_or_connect(&addr).await.expect("channel");
        assert!(channel.delta_link_active());
        assert!(server.accepted.load(Ordering::SeqCst) >= 3);
    }
}
