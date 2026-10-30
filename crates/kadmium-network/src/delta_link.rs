//! # `kadmium_network::delta_link`
//!
//! A delta link sends each frame as only the parts of it the peer does not already hold.
//!
//! The idea comes from InferSSD's ggml RPC transport, which streams a model's compute to a remote
//! GPU without moving bytes the remote already has: `SET_TENSOR_HASH` uploads a tensor by hash
//! when the server has it cached, and `GRAPH_RECOMPUTE` replays a stored graph instead of
//! reserializing it. Together these let a model use a GPU in another machine as if it were local,
//! even over Wi-Fi. A delta link generalizes both to any [`WireFrame`] on a Kadmium connection.
//!
//! The sender cuts every payload into content-defined chunks (a Gear rolling hash picks the
//! boundaries, so an insertion or an in-place edit changes only the chunks around it) and keys
//! each chunk by its XXH64 hash and length. A chunk the receiver already holds goes out as a
//! 13-byte reference, and any other chunk goes out literally and is cached by the receiver. A
//! re-dispatched weight blob or an unchanged work-unit header then costs a few bytes, a growing KV
//! page costs only its new tail, and a sparse update costs only the chunks it touched.
//!
//! Unlike ggml RPC, the sender never asks whether the receiver has a chunk, which would cost a
//! round trip per tensor. Each direction of a connection has its own chunk store with one byte
//! budget and deterministic LRU eviction. The encoder mirrors the decoder's store (keys and sizes
//! only), and both apply the same operations in wire order: a TCP stream is reliable and ordered,
//! [`FrameWriter`] encodes under the writer lock and [`FrameReader`] decodes in the read loop. So
//! the mirror is exact, and a reference the decoder cannot resolve means the two sides diverged.
//! That is treated as a fatal stream error, so the channel closes and a fresh connection starts
//! with empty stores on both sides.
//!
//! ## Negotiation
//!
//! A client opts in by sending a [`WireFrameKind::DeltaHello`] as the first frame on a new
//! connection. The server answers with its own hello, giving the store budget both directions will
//! use (the smaller of the two proposals, or 0 to decline). A worker from before delta links does
//! not recognize the frame kind and closes the connection, and the client then reconnects without
//! one. Any other frame on the connection may then be sent as a [`WireFrameKind::DeltaFrame`]
//! wrapping the original kind. The frame keeps the original epoch and request id, so request
//! multiplexing is unchanged.
//!
//! ## `DeltaFrame` payload
//!
//! ```text
//! u8   version (1)
//! u8   inner WireFrameKind
//! u8   mode: 0 = raw, 1 = chunked
//! raw:     the inner payload, verbatim
//! chunked: u32 LE inner payload length
//!          u32 LE record count
//!          records, each one of
//!            u8 0 (literal) | u32 LE len | len bytes
//!            u8 1 (ref)     | u64 LE XXH64 | u32 LE len
//! ```

use crate::{KDM_FRAME_HEADER_LEN_V2, KDM_FRAME_MAGIC_V2, WireFrame, WireFrameKind};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Version of the delta-link wire format this build speaks.
pub const DELTA_LINK_VERSION: u16 = 1;

/// Default per-direction chunk store budget (128 MiB). The decoder holds this many bytes of
/// chunks; the encoder holds only their keys.
pub const DEFAULT_DELTA_STORE_BYTES: u64 = 128 << 20;

/// Payloads shorter than this go out raw: chunking cannot save enough to pay for its records.
pub const DELTA_RAW_THRESHOLD: usize = 256;

/// Smallest average chunk size (2 KiB), used for payloads up to 2 MiB.
const MIN_AVG_CHUNK_BITS: u32 = 11;
/// Largest average chunk size (1 MiB), reached by 1 GiB payloads.
const MAX_AVG_CHUNK_BITS: u32 = 20;
/// Target number of chunks per payload; larger payloads get proportionally larger chunks so the
/// reference list of an unchanged payload stays around 13 KiB.
const TARGET_CHUNKS_LOG2: u32 = 10;

const HELLO_MAGIC: [u8; 4] = *b"KDLT";
const HELLO_LEN: usize = 4 + 2 + 8;

const FRAME_HEADER_LEN: usize = 3;
const CHUNKED_HEADER_LEN: usize = FRAME_HEADER_LEN + 8;
const MODE_RAW: u8 = 0;
const MODE_CHUNKED: u8 = 1;
const TAG_LITERAL: u8 = 0;
const TAG_REF: u8 = 1;
const LITERAL_RECORD_HEADER: usize = 1 + 4;
const REF_RECORD_LEN: usize = 1 + 8 + 4;

// ---------------------------------------------------------------------------------------------
// Hashing and chunking
// ---------------------------------------------------------------------------------------------

const XXH_P1: u64 = 0x9E37_79B1_85EB_CA87;
const XXH_P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const XXH_P3: u64 = 0x1656_67B1_9E37_79F9;
const XXH_P4: u64 = 0x85EB_CA77_C2B2_AE63;
const XXH_P5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline]
fn read_u64_le(data: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(data[at..at + 8].try_into().expect("8 bytes"))
}

#[inline]
fn read_u32_le(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().expect("4 bytes"))
}

#[inline]
fn xxh_round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(XXH_P2))
        .rotate_left(31)
        .wrapping_mul(XXH_P1)
}

#[inline]
fn xxh_merge_round(acc: u64, val: u64) -> u64 {
    (acc ^ xxh_round(0, val))
        .wrapping_mul(XXH_P1)
        .wrapping_add(XXH_P4)
}

/// XXH64 of `data` with `seed`. The chunk key hash: it runs several times faster than the
/// byte-wise FNV-1a used elsewhere in Kadmium, so keying chunks stays well ahead of a 10GbE or
/// USB4 link.
pub fn xxh64(data: &[u8], seed: u64) -> u64 {
    let len = data.len();
    let mut i = 0;
    let mut h = if len >= 32 {
        let mut v1 = seed.wrapping_add(XXH_P1).wrapping_add(XXH_P2);
        let mut v2 = seed.wrapping_add(XXH_P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(XXH_P1);
        while i + 32 <= len {
            v1 = xxh_round(v1, read_u64_le(data, i));
            v2 = xxh_round(v2, read_u64_le(data, i + 8));
            v3 = xxh_round(v3, read_u64_le(data, i + 16));
            v4 = xxh_round(v4, read_u64_le(data, i + 24));
            i += 32;
        }
        let mut acc = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        acc = xxh_merge_round(acc, v1);
        acc = xxh_merge_round(acc, v2);
        acc = xxh_merge_round(acc, v3);
        xxh_merge_round(acc, v4)
    } else {
        seed.wrapping_add(XXH_P5)
    };
    h = h.wrapping_add(len as u64);
    while i + 8 <= len {
        h ^= xxh_round(0, read_u64_le(data, i));
        h = h.rotate_left(27).wrapping_mul(XXH_P1).wrapping_add(XXH_P4);
        i += 8;
    }
    if i + 4 <= len {
        h ^= (read_u32_le(data, i) as u64).wrapping_mul(XXH_P1);
        h = h.rotate_left(23).wrapping_mul(XXH_P2).wrapping_add(XXH_P3);
        i += 4;
    }
    while i < len {
        h ^= (data[i] as u64).wrapping_mul(XXH_P5);
        h = h.rotate_left(11).wrapping_mul(XXH_P1);
        i += 1;
    }
    h ^= h >> 33;
    h = h.wrapping_mul(XXH_P2);
    h ^= h >> 29;
    h = h.wrapping_mul(XXH_P3);
    h ^ (h >> 32)
}

/// Gear table for the rolling chunk-boundary hash: 256 fixed pseudo-random words (SplitMix64).
/// Only the encoder chunks, so the table is not part of the wire format, but keeping it fixed
/// keeps boundaries stable across restarts and therefore across reconnects.
const GEAR: [u64; 256] = {
    let mut table = [0u64; 256];
    let mut state: u64 = 0x4B44_4D32_444C_5431; // "KDM2DLT1"
    let mut i = 0;
    while i < 256 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        table[i] = z ^ (z >> 31);
        i += 1;
    }
    table
};

/// log2 of the average chunk size for a payload of `len` bytes.
fn avg_chunk_bits(len: usize) -> u32 {
    let per_chunk = (len >> TARGET_CHUNKS_LOG2).max(1);
    (usize::BITS - 1 - per_chunk.leading_zeros()).clamp(MIN_AVG_CHUNK_BITS, MAX_AVG_CHUNK_BITS)
}

/// Content-defined chunk lengths of `data`, in order; they sum to `data.len()`.
///
/// A boundary falls where the top `bits` bits of the Gear hash are zero (the top bits depend on
/// the last 64 bytes; the low bits only on the last few). Chunks are at least a quarter and at
/// most eight times the average.
fn chunk_lengths(data: &[u8]) -> Vec<usize> {
    let bits = avg_chunk_bits(data.len());
    let avg = 1usize << bits;
    let (min, max) = (avg / 4, avg * 8);
    let mask = !(u64::MAX >> bits);
    let mut lengths = Vec::with_capacity(data.len() / avg + 1);
    let mut start = 0;
    while start < data.len() {
        let remaining = data.len() - start;
        if remaining <= min {
            lengths.push(remaining);
            break;
        }
        let end = start + remaining.min(max);
        let mut h: u64 = 0;
        let mut cut = end;
        let scan_from = start + min;
        for (offset, &byte) in data[scan_from..end].iter().enumerate() {
            h = (h << 1).wrapping_add(GEAR[byte as usize]);
            if h & mask == 0 {
                cut = scan_from + offset + 1;
                break;
            }
        }
        lengths.push(cut - start);
        start = cut;
    }
    lengths
}

/// Identity of a chunk in a store: its XXH64 and its length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ChunkKey {
    hash: u64,
    len: u32,
}

/// The chunking of one payload, computed without touching any link state.
///
/// [`ChunkPlan::new`] does all the hashing, so a caller can plan outside the writer lock and then
/// encode under it with [`DeltaEncoder::encode_planned`], which only does store lookups and copies.
#[derive(Debug, Clone)]
pub struct ChunkPlan {
    payload_len: usize,
    chunks: Vec<ChunkKey>,
}

impl ChunkPlan {
    pub fn new(payload: &[u8]) -> Self {
        if payload.len() < DELTA_RAW_THRESHOLD {
            return Self {
                payload_len: payload.len(),
                chunks: Vec::new(),
            };
        }
        let mut at = 0;
        let chunks = chunk_lengths(payload)
            .into_iter()
            .map(|len| {
                let key = ChunkKey {
                    hash: xxh64(&payload[at..at + len], 0),
                    len: len as u32,
                };
                at += len;
                key
            })
            .collect();
        Self {
            payload_len: payload.len(),
            chunks,
        }
    }

    /// `true` when the payload is sent raw (too short to chunk).
    pub fn is_raw(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

// ---------------------------------------------------------------------------------------------
// Chunk stores
// ---------------------------------------------------------------------------------------------

struct StoreSlot<V> {
    tick: u64,
    value: V,
}

/// Byte-budgeted LRU keyed by [`ChunkKey`]. The encoder keeps `V = ()` and the decoder keeps the
/// chunk bytes; both see the same sequence of `touch` and `insert` calls, so they evict the same
/// keys at the same points.
struct ChunkStore<V> {
    budget: u64,
    used: u64,
    tick: u64,
    slots: HashMap<ChunkKey, StoreSlot<V>>,
    order: BTreeMap<u64, ChunkKey>,
}

impl<V> ChunkStore<V> {
    fn new(budget: u64) -> Self {
        Self {
            budget,
            used: 0,
            tick: 0,
            slots: HashMap::new(),
            order: BTreeMap::new(),
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Marks `key` most recently used and returns its value, or `None` when it is not stored.
    fn touch(&mut self, key: ChunkKey) -> Option<&V> {
        let tick = self.next_tick();
        let slot = self.slots.get_mut(&key)?;
        self.order.remove(&slot.tick);
        slot.tick = tick;
        self.order.insert(tick, key);
        Some(&slot.value)
    }

    /// Stores a new key, then evicts least recently used keys until the store fits its budget
    /// (which can evict the new key itself when it alone exceeds the budget).
    fn insert(&mut self, key: ChunkKey, value: V) {
        debug_assert!(!self.slots.contains_key(&key));
        let tick = self.next_tick();
        self.slots.insert(key, StoreSlot { tick, value });
        self.order.insert(tick, key);
        self.used += key.len as u64;
        while self.used > self.budget {
            let Some((_, oldest)) = self.order.pop_first() else {
                break;
            };
            self.slots.remove(&oldest);
            self.used -= oldest.len as u64;
        }
    }

    fn contains(&self, key: &ChunkKey) -> bool {
        self.slots.contains_key(key)
    }

    fn len(&self) -> usize {
        self.slots.len()
    }
}

// ---------------------------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------------------------

/// Running byte counts for one direction of a delta link. Shared by `Arc` so a caller can read
/// them while the link runs.
#[derive(Debug, Default)]
pub struct DeltaLinkStats {
    frames: AtomicU64,
    raw_frames: AtomicU64,
    logical_bytes: AtomicU64,
    wire_bytes: AtomicU64,
    literal_chunks: AtomicU64,
    literal_bytes: AtomicU64,
    reused_chunks: AtomicU64,
    reused_bytes: AtomicU64,
}

impl DeltaLinkStats {
    fn record_raw(&self, logical: usize, wire: usize) {
        self.frames.fetch_add(1, Ordering::Relaxed);
        self.raw_frames.fetch_add(1, Ordering::Relaxed);
        self.logical_bytes
            .fetch_add(logical as u64, Ordering::Relaxed);
        self.wire_bytes.fetch_add(wire as u64, Ordering::Relaxed);
    }

    fn record_chunked(&self, logical: usize, wire: usize, tally: &ChunkTally) {
        self.frames.fetch_add(1, Ordering::Relaxed);
        self.logical_bytes
            .fetch_add(logical as u64, Ordering::Relaxed);
        self.wire_bytes.fetch_add(wire as u64, Ordering::Relaxed);
        self.literal_chunks
            .fetch_add(tally.literal_chunks, Ordering::Relaxed);
        self.literal_bytes
            .fetch_add(tally.literal_bytes, Ordering::Relaxed);
        self.reused_chunks
            .fetch_add(tally.reused_chunks, Ordering::Relaxed);
        self.reused_bytes
            .fetch_add(tally.reused_bytes, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> DeltaLinkSnapshot {
        DeltaLinkSnapshot {
            frames: self.frames.load(Ordering::Relaxed),
            raw_frames: self.raw_frames.load(Ordering::Relaxed),
            logical_bytes: self.logical_bytes.load(Ordering::Relaxed),
            wire_bytes: self.wire_bytes.load(Ordering::Relaxed),
            literal_chunks: self.literal_chunks.load(Ordering::Relaxed),
            literal_bytes: self.literal_bytes.load(Ordering::Relaxed),
            reused_chunks: self.reused_chunks.load(Ordering::Relaxed),
            reused_bytes: self.reused_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time copy of [`DeltaLinkStats`]. `logical_bytes` counts inner payload bytes and
/// `wire_bytes` the `DeltaFrame` payload bytes that carried them (frame headers excluded on both).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeltaLinkSnapshot {
    pub frames: u64,
    pub raw_frames: u64,
    pub logical_bytes: u64,
    pub wire_bytes: u64,
    pub literal_chunks: u64,
    pub literal_bytes: u64,
    pub reused_chunks: u64,
    pub reused_bytes: u64,
}

impl DeltaLinkSnapshot {
    /// Logical bytes per wire byte (1.0 when nothing has been sent).
    pub fn compression_ratio(&self) -> f64 {
        if self.wire_bytes == 0 {
            1.0
        } else {
            self.logical_bytes as f64 / self.wire_bytes as f64
        }
    }
}

#[derive(Default)]
struct ChunkTally {
    literal_chunks: u64,
    literal_bytes: u64,
    reused_chunks: u64,
    reused_bytes: u64,
}

// ---------------------------------------------------------------------------------------------
// Encoder and decoder
// ---------------------------------------------------------------------------------------------

/// Sending half of one direction of a delta link.
pub struct DeltaEncoder {
    mirror: ChunkStore<()>,
    stats: Arc<DeltaLinkStats>,
}

impl std::fmt::Debug for DeltaEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaEncoder")
            .field("budget", &self.mirror.budget)
            .field("mirrored_chunks", &self.mirror.len())
            .field("mirrored_bytes", &self.mirror.used)
            .finish()
    }
}

impl DeltaEncoder {
    /// An encoder whose peer decoder holds at most `store_bytes` of chunks.
    pub fn new(store_bytes: u64) -> Self {
        Self {
            mirror: ChunkStore::new(store_bytes),
            stats: Arc::new(DeltaLinkStats::default()),
        }
    }

    pub fn stats(&self) -> Arc<DeltaLinkStats> {
        Arc::clone(&self.stats)
    }

    /// Wraps `frame` as a `DeltaFrame` with the same epoch and request id.
    pub fn encode(&mut self, frame: &WireFrame) -> WireFrame {
        let plan = ChunkPlan::new(&frame.payload);
        self.encode_planned(frame, &plan)
    }

    /// [`DeltaEncoder::encode`] with a plan computed by [`ChunkPlan::new`] from `frame.payload`.
    pub fn encode_planned(&mut self, frame: &WireFrame, plan: &ChunkPlan) -> WireFrame {
        let mut payload = Vec::new();
        self.encode_into(frame, plan, &mut payload);
        WireFrame {
            kind: WireFrameKind::DeltaFrame,
            epoch: frame.epoch,
            request_id: frame.request_id,
            payload,
        }
    }

    /// Encodes `frame` straight into the bytes of a `KDM2` `DeltaFrame` tagged with
    /// `request_id`, copying each literal chunk once.
    pub fn encode_wire(&mut self, frame: &WireFrame, plan: &ChunkPlan, request_id: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(KDM_FRAME_HEADER_LEN_V2 + CHUNKED_HEADER_LEN);
        out.extend_from_slice(&KDM_FRAME_MAGIC_V2);
        out.push(WireFrameKind::DeltaFrame as u8);
        out.extend_from_slice(&frame.epoch.to_le_bytes());
        out.extend_from_slice(&request_id.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        self.encode_into(frame, plan, &mut out);
        let len = (out.len() - KDM_FRAME_HEADER_LEN_V2) as u32;
        out[KDM_FRAME_HEADER_LEN_V2 - 4..KDM_FRAME_HEADER_LEN_V2]
            .copy_from_slice(&len.to_le_bytes());
        out
    }

    /// Appends the `DeltaFrame` payload for `frame` to `out`, updating the mirror and stats.
    fn encode_into(&mut self, frame: &WireFrame, plan: &ChunkPlan, out: &mut Vec<u8>) {
        assert_eq!(
            plan.payload_len,
            frame.payload.len(),
            "ChunkPlan was computed for a different payload"
        );
        let payload = &frame.payload;
        let start = out.len();
        if plan.is_raw() {
            out.reserve(FRAME_HEADER_LEN + payload.len());
            out.extend_from_slice(&[DELTA_LINK_VERSION as u8, frame.kind as u8, MODE_RAW]);
            out.extend_from_slice(payload);
            self.stats.record_raw(payload.len(), out.len() - start);
            return;
        }
        // Size the buffer for the literals and references this frame will actually carry.
        let wire_estimate: usize = plan
            .chunks
            .iter()
            .map(|key| {
                if self.mirror.contains(key) {
                    REF_RECORD_LEN
                } else {
                    LITERAL_RECORD_HEADER + key.len as usize
                }
            })
            .sum();
        out.reserve(CHUNKED_HEADER_LEN + wire_estimate);
        out.extend_from_slice(&[DELTA_LINK_VERSION as u8, frame.kind as u8, MODE_CHUNKED]);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&(plan.chunks.len() as u32).to_le_bytes());
        let mut tally = ChunkTally::default();
        let mut at = 0usize;
        for &key in &plan.chunks {
            let len = key.len as usize;
            if self.mirror.touch(key).is_some() {
                out.push(TAG_REF);
                out.extend_from_slice(&key.hash.to_le_bytes());
                out.extend_from_slice(&key.len.to_le_bytes());
                tally.reused_chunks += 1;
                tally.reused_bytes += len as u64;
            } else {
                out.push(TAG_LITERAL);
                out.extend_from_slice(&key.len.to_le_bytes());
                out.extend_from_slice(&payload[at..at + len]);
                self.mirror.insert(key, ());
                tally.literal_chunks += 1;
                tally.literal_bytes += len as u64;
            }
            at += len;
        }
        self.stats
            .record_chunked(payload.len(), out.len() - start, &tally);
    }
}

/// Receiving half of one direction of a delta link.
pub struct DeltaDecoder {
    store: ChunkStore<Box<[u8]>>,
    stats: Arc<DeltaLinkStats>,
}

impl std::fmt::Debug for DeltaDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaDecoder")
            .field("budget", &self.store.budget)
            .field("stored_chunks", &self.store.len())
            .field("stored_bytes", &self.store.used)
            .finish()
    }
}

/// Bounds-checked little-endian cursor over a `DeltaFrame` payload.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.bytes.len() - self.at < n {
            return Err(format!(
                "DeltaFrame truncated: wanted {n} bytes at offset {}, {} left",
                self.at,
                self.bytes.len() - self.at
            ));
        }
        let slice = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(read_u32_le(self.take(4)?, 0))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(read_u64_le(self.take(8)?, 0))
    }
}

impl DeltaDecoder {
    /// A decoder that holds at most `store_bytes` of chunks.
    pub fn new(store_bytes: u64) -> Self {
        Self {
            store: ChunkStore::new(store_bytes),
            stats: Arc::new(DeltaLinkStats::default()),
        }
    }

    pub fn stats(&self) -> Arc<DeltaLinkStats> {
        Arc::clone(&self.stats)
    }

    /// Rebuilds the inner frame from a `DeltaFrame`. An error means the stream can no longer be
    /// trusted (the stores diverged or the frame is malformed); the connection must be closed.
    pub fn decode(&mut self, outer: &WireFrame) -> Result<WireFrame, String> {
        if outer.kind != WireFrameKind::DeltaFrame {
            return Err(format!("expected a DeltaFrame, got {:?}", outer.kind));
        }
        let mut cur = Cursor {
            bytes: &outer.payload,
            at: 0,
        };
        let version = cur.u8()?;
        if version as u16 != DELTA_LINK_VERSION {
            return Err(format!("unsupported DeltaFrame version {version}"));
        }
        let kind_byte = cur.u8()?;
        let kind = WireFrameKind::from_u8(kind_byte)
            .filter(|k| !matches!(k, WireFrameKind::DeltaFrame | WireFrameKind::DeltaHello))
            .ok_or_else(|| format!("DeltaFrame wraps invalid kind byte {kind_byte}"))?;
        let payload = match cur.u8()? {
            MODE_RAW => {
                let raw = cur.bytes[cur.at..].to_vec();
                self.stats.record_raw(raw.len(), outer.payload.len());
                raw
            }
            MODE_CHUNKED => {
                let total = cur.u32()? as usize;
                let records = cur.u32()?;
                if total > crate::KDM_FRAME_MAX_PAYLOAD_BYTES {
                    return Err(format!("DeltaFrame declares a {total}-byte payload"));
                }
                let mut out = Vec::with_capacity(total);
                let mut tally = ChunkTally::default();
                for _ in 0..records {
                    match cur.u8()? {
                        TAG_LITERAL => {
                            let len = cur.u32()?;
                            let bytes = cur.take(len as usize)?;
                            if out.len() + bytes.len() > total {
                                return Err("DeltaFrame chunks overrun the declared length".into());
                            }
                            let key = ChunkKey {
                                hash: xxh64(bytes, 0),
                                len,
                            };
                            if self.store.contains(&key) {
                                return Err(format!(
                                    "DeltaFrame resends stored chunk {:016x}: chunk stores diverged",
                                    key.hash
                                ));
                            }
                            out.extend_from_slice(bytes);
                            self.store.insert(key, bytes.into());
                            tally.literal_chunks += 1;
                            tally.literal_bytes += len as u64;
                        }
                        TAG_REF => {
                            let key = ChunkKey {
                                hash: cur.u64()?,
                                len: cur.u32()?,
                            };
                            if out.len() + key.len as usize > total {
                                return Err("DeltaFrame chunks overrun the declared length".into());
                            }
                            let bytes = self.store.touch(key).ok_or_else(|| {
                                format!(
                                    "DeltaFrame references chunk {:016x} ({} bytes) this side does not hold: chunk stores diverged",
                                    key.hash, key.len
                                )
                            })?;
                            out.extend_from_slice(bytes);
                            tally.reused_chunks += 1;
                            tally.reused_bytes += key.len as u64;
                        }
                        tag => return Err(format!("unknown DeltaFrame record tag {tag}")),
                    }
                }
                if out.len() != total || cur.at != cur.bytes.len() {
                    return Err(format!(
                        "DeltaFrame rebuilt {} of {total} bytes with {} trailing bytes",
                        out.len(),
                        cur.bytes.len() - cur.at
                    ));
                }
                self.stats
                    .record_chunked(out.len(), outer.payload.len(), &tally);
                out
            }
            mode => return Err(format!("unknown DeltaFrame mode {mode}")),
        };
        Ok(WireFrame {
            kind,
            epoch: outer.epoch,
            request_id: outer.request_id,
            payload,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Negotiation
// ---------------------------------------------------------------------------------------------

/// Whether, and with how large a chunk store, this node uses delta links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeltaLinkConfig {
    pub enabled: bool,
    /// Largest per-direction chunk store this node proposes (as a client) or accepts (as a
    /// server). A server holds one decoder store per connection.
    pub store_bytes: u64,
}

impl Default for DeltaLinkConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            store_bytes: DEFAULT_DELTA_STORE_BYTES,
        }
    }
}

impl DeltaLinkConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            store_bytes: 0,
        }
    }

    /// The default, adjusted by `KADMIUM_DELTA_LINK` (`0`, `off` or `false` disables delta links)
    /// and `KADMIUM_DELTA_LINK_STORE_MB` (per-direction store budget in MiB).
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(v) = std::env::var("KADMIUM_DELTA_LINK")
            && matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "false"
            )
        {
            return Self::disabled();
        }
        if let Some(mb) = std::env::var("KADMIUM_DELTA_LINK_STORE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
        {
            config.store_bytes = mb << 20;
        }
        config
    }

    /// Budget this node offers or accepts; 0 when disabled.
    fn offered_store_bytes(&self) -> u64 {
        if self.enabled { self.store_bytes } else { 0 }
    }
}

/// Payload of a [`WireFrameKind::DeltaHello`]: `"KDLT"`, `u16` LE version, `u64` LE store budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaHello {
    pub version: u16,
    /// Proposed (request) or agreed (reply) per-direction store budget; 0 declines.
    pub store_bytes: u64,
}

impl DeltaHello {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HELLO_LEN);
        out.extend_from_slice(&HELLO_MAGIC);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.store_bytes.to_le_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != HELLO_LEN || bytes[..4] != HELLO_MAGIC {
            return Err("malformed DeltaHello".into());
        }
        Ok(Self {
            version: u16::from_le_bytes([bytes[4], bytes[5]]),
            store_bytes: read_u64_le(bytes, 6),
        })
    }

    /// The hello a client sends to open a delta link with `config`.
    pub fn request(config: &DeltaLinkConfig) -> WireFrame {
        let hello = Self {
            version: DELTA_LINK_VERSION,
            store_bytes: config.offered_store_bytes(),
        };
        WireFrame::new(WireFrameKind::DeltaHello, 0, hello.to_bytes())
    }

    /// The server's answer to `request` under `config`: the agreed budget (0 when either side
    /// declines or the versions differ) and the reply frame carrying it.
    pub fn answer(request: &WireFrame, config: &DeltaLinkConfig) -> (u64, WireFrame) {
        let agreed = match Self::from_bytes(&request.payload) {
            Ok(hello) if hello.version == DELTA_LINK_VERSION => {
                hello.store_bytes.min(config.offered_store_bytes())
            }
            _ => 0,
        };
        let reply = Self {
            version: DELTA_LINK_VERSION,
            store_bytes: agreed,
        };
        (
            agreed,
            WireFrame::new(WireFrameKind::DeltaHello, request.epoch, reply.to_bytes())
                .with_request_id(request.request_id),
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Framed stream halves
// ---------------------------------------------------------------------------------------------

/// Write half of a connection that delta-encodes frames once a link is negotiated.
///
/// Keep it behind the connection's writer lock: frames must be encoded in the order they are
/// written, or the encoder's mirror stops matching the peer's store.
pub struct FrameWriter<W> {
    stream: W,
    encoder: Option<DeltaEncoder>,
}

impl<W: AsyncWriteExt + Unpin> FrameWriter<W> {
    pub fn new(stream: W) -> Self {
        Self {
            stream,
            encoder: None,
        }
    }

    pub fn enable_delta(&mut self, encoder: DeltaEncoder) {
        self.encoder = Some(encoder);
    }

    pub fn delta_active(&self) -> bool {
        self.encoder.is_some()
    }

    pub fn delta_stats(&self) -> Option<Arc<DeltaLinkStats>> {
        self.encoder.as_ref().map(DeltaEncoder::stats)
    }

    /// Encodes `frame` (as a `DeltaFrame` when a link is active, tagged with `request_id`) into
    /// the bytes to write. `plan` must come from `frame.payload` when given. This updates the
    /// encoder's mirror, so the bytes must then be written in full or the connection dropped.
    pub fn encode(
        &mut self,
        frame: &WireFrame,
        request_id: u64,
        plan: Option<&ChunkPlan>,
    ) -> Vec<u8> {
        match self.encoder.as_mut() {
            Some(encoder) => match plan {
                Some(plan) => encoder.encode_wire(frame, plan, request_id),
                None => encoder.encode_wire(frame, &ChunkPlan::new(&frame.payload), request_id),
            },
            None => frame.encode_with_request_id(request_id),
        }
    }

    /// Writes bytes from [`FrameWriter::encode`] and flushes.
    pub async fn write_encoded(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.stream.write_all(bytes).await?;
        self.stream.flush().await
    }

    /// Encodes and writes one frame, keeping its own request id.
    pub async fn write_frame(&mut self, frame: &WireFrame) -> Result<(), String> {
        let bytes = self.encode(frame, frame.request_id, None);
        self.write_encoded(&bytes)
            .await
            .map_err(|e| format!("Stream write error: {e}"))
    }

    /// Writes a frame verbatim, bypassing the encoder (the hello exchange).
    pub async fn write_plain(&mut self, frame: &WireFrame) -> Result<(), String> {
        frame.write_to_stream(&mut self.stream).await
    }
}

/// Read half of a connection that transparently decodes `DeltaFrame`s once a link is negotiated.
pub struct FrameReader<R> {
    stream: R,
    decoder: Option<DeltaDecoder>,
}

impl<R: AsyncReadExt + Unpin> FrameReader<R> {
    pub fn new(stream: R) -> Self {
        Self {
            stream,
            decoder: None,
        }
    }

    pub fn enable_delta(&mut self, decoder: DeltaDecoder) {
        self.decoder = Some(decoder);
    }

    pub fn delta_active(&self) -> bool {
        self.decoder.is_some()
    }

    pub fn delta_stats(&self) -> Option<Arc<DeltaLinkStats>> {
        self.decoder.as_ref().map(DeltaDecoder::stats)
    }

    /// Reads the next frame, rebuilding it if it arrived as a `DeltaFrame`. Any error, including
    /// a `DeltaFrame` the decoder cannot rebuild, ends the connection.
    pub async fn read_frame(&mut self) -> Result<WireFrame, String> {
        let frame = WireFrame::read_from_stream(&mut self.stream).await?;
        if frame.kind != WireFrameKind::DeltaFrame {
            return Ok(frame);
        }
        match self.decoder.as_mut() {
            Some(decoder) => decoder.decode(&frame),
            None => Err("DeltaFrame received before a delta link was negotiated".into()),
        }
    }
}

/// Server side of the hello exchange: answers `request` under `config` and, when a budget is
/// agreed, turns on decoding for `reader` and encoding for `writer`. Returns the agreed budget.
/// Call it from the read loop before reading the next frame, holding the writer lock, so no
/// reply can be encoded before the hello answer is written.
pub async fn accept_delta_hello<R, W>(
    request: &WireFrame,
    config: &DeltaLinkConfig,
    reader: &mut FrameReader<R>,
    writer: &mut FrameWriter<W>,
) -> Result<u64, String>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    if reader.delta_active() || writer.delta_active() {
        return Err("DeltaHello received twice on one connection".into());
    }
    let (agreed, reply) = DeltaHello::answer(request, config);
    writer.write_plain(&reply).await?;
    if agreed > 0 {
        reader.enable_delta(DeltaDecoder::new(agreed));
        writer.enable_delta(DeltaEncoder::new(agreed));
    }
    Ok(agreed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic filler bytes (SplitMix64 stream).
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

    fn frame(payload: Vec<u8>) -> WireFrame {
        WireFrame::new(WireFrameKind::WorkDispatch, 7, payload).with_request_id(42)
    }

    /// Sends `payload` through `enc` and `dec`, checks it arrives intact, returns wire bytes.
    fn round_trip(enc: &mut DeltaEncoder, dec: &mut DeltaDecoder, payload: Vec<u8>) -> usize {
        let original = frame(payload);
        let wrapped = enc.encode(&original);
        assert_eq!(wrapped.kind, WireFrameKind::DeltaFrame);
        assert_eq!(wrapped.request_id, 42);
        let rebuilt = dec.decode(&wrapped).expect("decode");
        assert_eq!(rebuilt, original);
        wrapped.payload.len()
    }

    #[test]
    fn xxh64_matches_reference_vectors() {
        assert_eq!(xxh64(b"", 0), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxh64(b"abc", 0), 0x44BC_2CF5_AD77_0999);
    }

    #[test]
    fn chunk_lengths_cover_the_payload_and_respect_bounds() {
        for len in [256usize, 5_000, 1 << 20, (8 << 20) + 13] {
            let data = noise(len as u64, len);
            let lengths = chunk_lengths(&data);
            assert_eq!(lengths.iter().sum::<usize>(), len);
            let avg = 1usize << avg_chunk_bits(len);
            for &l in &lengths[..lengths.len() - 1] {
                assert!(
                    l >= avg / 4 && l <= avg * 8,
                    "chunk {l} outside bounds for avg {avg}"
                );
            }
        }
    }

    #[test]
    fn resending_a_payload_costs_only_references() {
        let mut enc = DeltaEncoder::new(DEFAULT_DELTA_STORE_BYTES);
        let mut dec = DeltaDecoder::new(DEFAULT_DELTA_STORE_BYTES);
        let blob = noise(1, 4 << 20);
        let first = round_trip(&mut enc, &mut dec, blob.clone());
        assert!(first > blob.len());
        let second = round_trip(&mut enc, &mut dec, blob.clone());
        assert!(second * 200 < blob.len(), "resend took {second} wire bytes");
        let stats = enc.stats().snapshot();
        assert_eq!(stats, dec.stats().snapshot());
        assert_eq!(stats.reused_bytes, blob.len() as u64);
    }

    #[test]
    fn an_edit_or_append_resends_only_nearby_chunks() {
        let mut enc = DeltaEncoder::new(DEFAULT_DELTA_STORE_BYTES);
        let mut dec = DeltaDecoder::new(DEFAULT_DELTA_STORE_BYTES);
        let mut buf = noise(2, 1 << 20);
        round_trip(&mut enc, &mut dec, buf.clone());

        // An in-place edit in the middle.
        buf[500_000..500_016].copy_from_slice(&[0xAB; 16]);
        let edit = round_trip(&mut enc, &mut dec, buf.clone());
        assert!(edit < 40_000, "16-byte edit cost {edit} wire bytes");

        // An insertion shifts everything after it; content-defined boundaries resync.
        buf.splice(10_000..10_000, noise(3, 100));
        let insert = round_trip(&mut enc, &mut dec, buf.clone());
        assert!(insert < 40_000, "100-byte insert cost {insert} wire bytes");

        // A KV-style append within the same chunk-size bucket.
        buf.extend_from_slice(&noise(4, 64 << 10));
        let append = round_trip(&mut enc, &mut dec, buf.clone());
        assert!(
            append < (64 << 10) + 40_000,
            "64 KiB append cost {append} wire bytes"
        );
    }

    #[test]
    fn single_pass_wire_encoding_matches_the_wrapped_frame() {
        let mut composed = DeltaEncoder::new(1 << 20);
        let mut direct = DeltaEncoder::new(1 << 20);
        let blob = noise(50, 300 << 10);
        for payload in [blob.clone(), blob, b"short".to_vec()] {
            let original = frame(payload);
            let plan = ChunkPlan::new(&original.payload);
            let expected = composed
                .encode_planned(&original, &plan)
                .encode_with_request_id(99);
            assert_eq!(direct.encode_wire(&original, &plan, 99), expected);
        }
        assert_eq!(composed.stats().snapshot(), direct.stats().snapshot());
    }

    #[test]
    fn small_payloads_go_raw_and_leave_the_stores_alone() {
        let mut enc = DeltaEncoder::new(1 << 20);
        let mut dec = DeltaDecoder::new(1 << 20);
        let wire = round_trip(&mut enc, &mut dec, b"{\"ping\":1}".to_vec());
        assert_eq!(wire, FRAME_HEADER_LEN + 10);
        assert_eq!(dec.store.len(), 0);
        round_trip(&mut enc, &mut dec, Vec::new());
    }

    #[test]
    fn eviction_stays_in_lockstep_under_a_tiny_budget() {
        // A 64 KiB store with 1 MiB payloads forces constant eviction, including chunks larger
        // than what remains; every frame must still rebuild exactly.
        let mut enc = DeltaEncoder::new(64 << 10);
        let mut dec = DeltaDecoder::new(64 << 10);
        let blobs: Vec<Vec<u8>> = (0..4).map(|i| noise(10 + i, 1 << 20)).collect();
        for round in 0..12 {
            round_trip(&mut enc, &mut dec, blobs[round % blobs.len()].clone());
            assert_eq!(enc.mirror.used, dec.store.used);
            assert_eq!(enc.mirror.len(), dec.store.len());
            assert!(dec.store.used <= 64 << 10);
        }
    }

    #[test]
    fn a_reference_the_decoder_lacks_is_a_fatal_divergence() {
        let mut enc = DeltaEncoder::new(1 << 20);
        let blob = noise(20, 64 << 10);
        enc.encode(&frame(blob.clone()));
        let wrapped = enc.encode(&frame(blob));
        // A decoder that never saw the first frame.
        let err = DeltaDecoder::new(1 << 20).decode(&wrapped).unwrap_err();
        assert!(err.contains("diverged"), "{err}");
    }

    #[test]
    fn malformed_frames_are_rejected_without_panicking() {
        let mut enc = DeltaEncoder::new(1 << 20);
        let wrapped = enc.encode(&frame(noise(30, 32 << 10)));
        for cut in [0, 1, 2, 3, 7, 11, 20, wrapped.payload.len() - 1] {
            let mut truncated = wrapped.clone();
            truncated.payload.truncate(cut);
            assert!(
                DeltaDecoder::new(1 << 20).decode(&truncated).is_err(),
                "cut {cut}"
            );
        }
        let mut bad_kind = wrapped.clone();
        bad_kind.payload[1] = WireFrameKind::DeltaFrame as u8;
        assert!(DeltaDecoder::new(1 << 20).decode(&bad_kind).is_err());
    }

    #[test]
    fn hello_negotiates_the_smaller_budget_and_declines_when_disabled() {
        let client = DeltaLinkConfig {
            enabled: true,
            store_bytes: 64 << 20,
        };
        let request = DeltaHello::request(&client).with_request_id(5);
        let (agreed, reply) = DeltaHello::answer(&request, &DeltaLinkConfig::default());
        assert_eq!(agreed, 64 << 20);
        assert_eq!(reply.request_id, 5);
        assert_eq!(
            DeltaHello::from_bytes(&reply.payload).unwrap().store_bytes,
            64 << 20
        );

        let (declined, _) = DeltaHello::answer(&request, &DeltaLinkConfig::disabled());
        assert_eq!(declined, 0);
        let (garbage, _) = DeltaHello::answer(
            &WireFrame::new(WireFrameKind::DeltaHello, 0, b"nope".to_vec()),
            &DeltaLinkConfig::default(),
        );
        assert_eq!(garbage, 0);
    }

    #[tokio::test]
    async fn framed_halves_carry_delta_frames_over_a_stream() {
        let (client, server) = tokio::io::duplex(1 << 16);
        let (client_read, client_write) = tokio::io::split(client);
        let (server_read, server_write) = tokio::io::split(server);
        let mut c_writer = FrameWriter::new(client_write);
        let mut c_reader = FrameReader::new(client_read);
        let mut s_reader = FrameReader::new(server_read);
        let mut s_writer = FrameWriter::new(server_write);

        let server = tokio::spawn(async move {
            let hello = s_reader.read_frame().await.unwrap();
            assert_eq!(hello.kind, WireFrameKind::DeltaHello);
            accept_delta_hello(
                &hello,
                &DeltaLinkConfig::default(),
                &mut s_reader,
                &mut s_writer,
            )
            .await
            .unwrap();
            for _ in 0..3 {
                let frame = s_reader.read_frame().await.unwrap();
                // Echo the payload back so the reply direction is exercised too.
                let reply = WireFrame::new(WireFrameKind::WorkResult, frame.epoch, frame.payload)
                    .with_request_id(frame.request_id);
                s_writer.write_frame(&reply).await.unwrap();
            }
            s_reader.delta_stats().unwrap().snapshot()
        });

        c_writer
            .write_plain(&DeltaHello::request(&DeltaLinkConfig::default()))
            .await
            .unwrap();
        let reply = c_reader.read_frame().await.unwrap();
        let agreed = DeltaHello::from_bytes(&reply.payload).unwrap().store_bytes;
        assert_eq!(agreed, DEFAULT_DELTA_STORE_BYTES);
        c_writer.enable_delta(DeltaEncoder::new(agreed));
        c_reader.enable_delta(DeltaDecoder::new(agreed));

        let blob = noise(40, 512 << 10);
        for id in 1..=3u64 {
            let request = frame(blob.clone()).with_request_id(id);
            c_writer.write_frame(&request).await.unwrap();
            let echo = c_reader.read_frame().await.unwrap();
            assert_eq!(echo.kind, WireFrameKind::WorkResult);
            assert_eq!(echo.request_id, id);
            assert_eq!(echo.payload, blob);
        }
        let inbound = server.await.unwrap();
        assert_eq!(inbound.logical_bytes, 3 * blob.len() as u64);
        assert!(inbound.wire_bytes < blob.len() as u64 + 20_000);
        assert_eq!(
            c_reader.delta_stats().unwrap().snapshot().reused_bytes,
            2 * blob.len() as u64
        );
    }
}
