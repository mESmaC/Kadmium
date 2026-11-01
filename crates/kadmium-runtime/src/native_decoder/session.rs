//! Authenticated `KDNS` wire frames and worker session host for [`super::DecoderSlice`].
//!
//! Each frame travels inside a `KDM2` [`kadmium_network::WireFrame`] of kind
//! [`kadmium_network::WireFrameKind::NativeDecoderSession`] over the existing [`kadmium_network::PeerChannel`]
//! worker connection:
//!
//! ```text
//! offset  size  field
//!      0     4  magic "KDNS"
//!      4     1  version (1)
//!      5     1  message kind (1..=9)
//!      6     4  metadata length, u32 LE
//!     10     4  binary payload length, u32 LE
//!     14    32  HMAC-SHA256 tag over (domain || header[0..14] || metadata || payload)
//!     46     m  UTF-8 JSON metadata
//!   46+m     p  exact little-endian f32 or u32 binary payload
//! ```
//!
//! Quality and security contract:
//!
//! - Activations and logits cross the wire as raw IEEE 754 `f32` little-endian bytes
//!   (`f32::to_le_bytes` / `f32::from_le_bytes`). Nothing is quantized, cast, or formatted as
//!   decimal text; every 32-bit float pattern (`f32::to_bits`) arrives unchanged.
//! - Every request and response carries a 32-byte HMAC-SHA256 tag under a shared
//!   [`SessionAuthKey`], with distinct domain labels for requests and responses. The tag and all
//!   declared byte lengths are verified before parsing JSON metadata or allocating tensor buffers.
//! - Frame, metadata and payload byte counts, as well as `rows * width * 4` tensor byte counts,
//!   are checked with overflow-safe arithmetic against [`NativeSessionLimits`] before any buffer
//!   is allocated.

use super::{Activations, DecoderSlice, KvCache, NativeDecoderError, Result, SlicePlan};
use kadmium_network::KDM_FRAME_MAX_PAYLOAD_BYTES;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// 4-byte magic of a native decoder session envelope (`"KDNS"`).
pub const NATIVE_SESSION_MAGIC: [u8; 4] = *b"KDNS";

/// Wire protocol version spoken by this build.
pub const NATIVE_SESSION_VERSION: u8 = 1;

/// Fixed byte length of the `KDNS` envelope header (`4 + 1 + 1 + 4 + 4 + 32`).
pub const NATIVE_SESSION_HEADER_BYTES: usize = 46;

/// Byte length of the HMAC-SHA256 authentication tag in the header.
pub const NATIVE_SESSION_MAC_BYTES: usize = 32;

/// Minimum length in bytes of a [`SessionAuthKey`] secret.
pub const NATIVE_SESSION_MIN_KEY_BYTES: usize = 16;

/// Maximum UTF-8 byte length of a `session_id`.
pub const NATIVE_SESSION_MAX_SESSION_ID_BYTES: usize = 256;

/// Maximum UTF-8 byte length of a `model_id`.
pub const NATIVE_SESSION_MAX_MODEL_ID_BYTES: usize = 1024;

/// Default upper bound on an encoded `KDNS` frame (`1 GiB`, matching `KDM2`).
pub const DEFAULT_MAX_NATIVE_SESSION_FRAME_BYTES: usize = KDM_FRAME_MAX_PAYLOAD_BYTES;

/// Default upper bound on the JSON metadata section of a `KDNS` frame (`1 MiB`).
pub const DEFAULT_MAX_NATIVE_SESSION_METADATA_BYTES: usize = 1 << 20;

/// Default upper bound on the raw binary tensor/token section of a `KDNS` frame.
pub const DEFAULT_MAX_NATIVE_SESSION_PAYLOAD_BYTES: usize = DEFAULT_MAX_NATIVE_SESSION_FRAME_BYTES
    - NATIVE_SESSION_HEADER_BYTES
    - DEFAULT_MAX_NATIVE_SESSION_METADATA_BYTES;

/// Default upper bound on open native decoder sessions held concurrently by one worker.
pub const DEFAULT_MAX_ACTIVE_NATIVE_SESSIONS: usize = 64;

/// Bound on concurrent `Open` / `Forward` frames executing on one worker.
pub const MAX_IN_FLIGHT_NATIVE_SESSION_FRAMES: usize = 16;

const KIND_OPEN_REQUEST: u8 = 1;
const KIND_OPEN_RESPONSE: u8 = 2;
const KIND_FORWARD_REQUEST: u8 = 3;
const KIND_FORWARD_RESPONSE: u8 = 4;
const KIND_CLOSE_REQUEST: u8 = 5;
const KIND_CLOSE_RESPONSE: u8 = 6;
const KIND_CANCEL_REQUEST: u8 = 7;
const KIND_CANCEL_RESPONSE: u8 = 8;
const KIND_ERROR_RESPONSE: u8 = 9;

const REQUEST_DOMAIN: &[u8] = b"KDNS-v1-request";
const RESPONSE_DOMAIN: &[u8] = b"KDNS-v1-response";

/// Size and concurrency bounds enforced on native decoder session frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeSessionLimits {
    pub max_frame_bytes: usize,
    pub max_metadata_bytes: usize,
    pub max_payload_bytes: usize,
    pub max_active_sessions: usize,
}

impl Default for NativeSessionLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: DEFAULT_MAX_NATIVE_SESSION_FRAME_BYTES,
            max_metadata_bytes: DEFAULT_MAX_NATIVE_SESSION_METADATA_BYTES,
            max_payload_bytes: DEFAULT_MAX_NATIVE_SESSION_PAYLOAD_BYTES,
            max_active_sessions: DEFAULT_MAX_ACTIVE_NATIVE_SESSIONS,
        }
    }
}

impl NativeSessionLimits {
    pub fn validate(&self) -> Result<()> {
        if self.max_frame_bytes == 0 || self.max_frame_bytes > KDM_FRAME_MAX_PAYLOAD_BYTES {
            return Err(NativeDecoderError::Wire(format!(
                "max_frame_bytes {} must be in 1..={KDM_FRAME_MAX_PAYLOAD_BYTES}",
                self.max_frame_bytes
            )));
        }
        if self.max_metadata_bytes == 0 || self.max_payload_bytes == 0 {
            return Err(NativeDecoderError::Wire(
                "max_metadata_bytes and max_payload_bytes must be positive".into(),
            ));
        }
        let min_frame = NATIVE_SESSION_HEADER_BYTES
            .checked_add(self.max_metadata_bytes)
            .ok_or_else(|| NativeDecoderError::Wire("limit sum overflowed usize".into()))?;
        if min_frame > self.max_frame_bytes || self.max_payload_bytes > self.max_frame_bytes {
            return Err(NativeDecoderError::Wire(
                "metadata and payload limits must fit within max_frame_bytes".into(),
            ));
        }
        if self.max_active_sessions == 0 {
            return Err(NativeDecoderError::Wire(
                "max_active_sessions must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

/// Shared secret authenticating `KDNS` session frames between coordinator and worker.
#[derive(Clone, PartialEq, Eq)]
pub struct SessionAuthKey {
    secret: Arc<[u8]>,
}

impl std::fmt::Debug for SessionAuthKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SessionAuthKey([REDACTED])")
    }
}

impl SessionAuthKey {
    /// Builds a key from raw secret bytes; requires at least [`NATIVE_SESSION_MIN_KEY_BYTES`].
    pub fn new(secret: impl AsRef<[u8]>) -> Result<Self> {
        let bytes = secret.as_ref();
        if bytes.len() < NATIVE_SESSION_MIN_KEY_BYTES {
            return Err(NativeDecoderError::Authentication(format!(
                "session auth key must be at least {NATIVE_SESSION_MIN_KEY_BYTES} bytes; got {}",
                bytes.len()
            )));
        }
        Ok(Self {
            secret: Arc::from(bytes),
        })
    }

    /// Reads `KADMIUM_WORKER_AUTH_TOKEN` from the process environment when set and non-empty.
    pub fn from_env() -> Option<Result<Self>> {
        let raw = std::env::var("KADMIUM_WORKER_AUTH_TOKEN").ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(Self::new(trimmed.as_bytes()))
    }

    /// Computes the 32-byte HMAC-SHA256 tag over `domain || prefix || metadata || payload`.
    pub fn sign(
        &self,
        domain: &[u8],
        header_prefix: &[u8],
        metadata: &[u8],
        payload: &[u8],
    ) -> [u8; NATIVE_SESSION_MAC_BYTES] {
        hmac_sha256(&self.secret, &[domain, header_prefix, metadata, payload])
    }

    /// Verifies `expected_tag` in constant time against `domain || prefix || metadata || payload`.
    pub fn verify(
        &self,
        domain: &[u8],
        header_prefix: &[u8],
        metadata: &[u8],
        payload: &[u8],
        expected_tag: &[u8; NATIVE_SESSION_MAC_BYTES],
    ) -> bool {
        let actual = self.sign(domain, header_prefix, metadata, payload);
        tags_match(&actual, expected_tag)
    }
}

/// RFC 2104 HMAC-SHA256 over the concatenation of `parts`.
pub fn hmac_sha256(key: &[u8], parts: &[&[u8]]) -> [u8; NATIVE_SESSION_MAC_BYTES] {
    let mut key_block = [0u8; 64];
    if key.len() > 64 {
        let hashed = Sha256::digest(key);
        key_block[..32].copy_from_slice(&hashed);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= key_block[i];
        opad[i] ^= key_block[i];
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    for part in parts {
        inner.update(part);
    }
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner.finalize());
    outer.finalize().into()
}

/// Constant-time comparison of two 32-byte MAC tags.
pub fn tags_match(a: &[u8; NATIVE_SESSION_MAC_BYTES], b: &[u8; NATIVE_SESSION_MAC_BYTES]) -> bool {
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Returns `true` when `payload` starts with a `KDNS` `CloseRequest` or `CancelRequest` header,
/// allowing the worker connection loop to dispatch close and cancel frames without waiting behind
/// in-flight `Forward` permits.
pub fn is_priority_control_frame(payload: &[u8]) -> bool {
    payload.len() >= NATIVE_SESSION_HEADER_BYTES
        && payload[..4] == NATIVE_SESSION_MAGIC
        && payload[4] == NATIVE_SESSION_VERSION
        && matches!(payload[5], KIND_CLOSE_REQUEST | KIND_CANCEL_REQUEST)
}

/// Encodes a `rows x width` matrix of `f32` values into `rows * width * 4` little-endian bytes,
/// checking dimensions and byte limits before allocating.
pub fn encode_f32_tensor(
    rows: usize,
    width: usize,
    data: &[f32],
    max_payload_bytes: usize,
) -> Result<Vec<u8>> {
    if rows == 0 || width == 0 {
        return Err(NativeDecoderError::Shape(
            "tensor rows and width must be positive".into(),
        ));
    }
    let elements = rows
        .checked_mul(width)
        .ok_or_else(|| NativeDecoderError::Wire("tensor element count overflowed usize".into()))?;
    let byte_len = elements
        .checked_mul(4)
        .ok_or_else(|| NativeDecoderError::Wire("tensor byte length overflowed usize".into()))?;
    if data.len() != elements {
        return Err(NativeDecoderError::Shape(format!(
            "{} float values do not form a {rows} x {width} tensor",
            data.len()
        )));
    }
    if byte_len > max_payload_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: byte_len,
            limit: max_payload_bytes,
        });
    }
    let mut out = Vec::with_capacity(byte_len);
    for &value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

/// Decodes a `rows x width` matrix of `f32` values from `rows * width * 4` little-endian bytes,
/// verifying overflow-safe dimensions and exact byte length before allocating.
pub fn decode_f32_tensor(
    rows: usize,
    width: usize,
    bytes: &[u8],
    max_payload_bytes: usize,
) -> Result<Vec<f32>> {
    if bytes.len() > max_payload_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: bytes.len(),
            limit: max_payload_bytes,
        });
    }
    if rows == 0 || width == 0 {
        return Err(NativeDecoderError::Shape(
            "tensor rows and width must be positive".into(),
        ));
    }
    let elements = rows
        .checked_mul(width)
        .ok_or_else(|| NativeDecoderError::Wire("tensor element count overflowed usize".into()))?;
    let expected_bytes = elements
        .checked_mul(4)
        .ok_or_else(|| NativeDecoderError::Wire("tensor byte length overflowed usize".into()))?;
    if expected_bytes > max_payload_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: expected_bytes,
            limit: max_payload_bytes,
        });
    }
    if bytes.len() != expected_bytes {
        return Err(NativeDecoderError::Wire(format!(
            "tensor payload has {} bytes; {rows} x {width} f32 requires {expected_bytes} bytes",
            bytes.len()
        )));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// Encodes a non-empty slice of `u32` token ids as `count * 4` little-endian bytes.
pub fn encode_u32_slice(tokens: &[u32], max_payload_bytes: usize) -> Result<Vec<u8>> {
    if tokens.is_empty() {
        return Err(NativeDecoderError::Shape("no tokens in request".into()));
    }
    let byte_len = tokens
        .len()
        .checked_mul(4)
        .ok_or_else(|| NativeDecoderError::Wire("token byte length overflowed usize".into()))?;
    if byte_len > max_payload_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: byte_len,
            limit: max_payload_bytes,
        });
    }
    let mut out = Vec::with_capacity(byte_len);
    for &token in tokens {
        out.extend_from_slice(&token.to_le_bytes());
    }
    Ok(out)
}

/// Decodes `count` `u32` token ids from `count * 4` little-endian bytes.
pub fn decode_u32_slice(count: usize, bytes: &[u8], max_payload_bytes: usize) -> Result<Vec<u32>> {
    if bytes.len() > max_payload_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: bytes.len(),
            limit: max_payload_bytes,
        });
    }
    if count == 0 {
        return Err(NativeDecoderError::Shape("no tokens in request".into()));
    }
    let expected_bytes = count
        .checked_mul(4)
        .ok_or_else(|| NativeDecoderError::Wire("token byte length overflowed usize".into()))?;
    if expected_bytes > max_payload_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: expected_bytes,
            limit: max_payload_bytes,
        });
    }
    if bytes.len() != expected_bytes {
        return Err(NativeDecoderError::Wire(format!(
            "token payload has {} bytes; {count} u32 tokens require {expected_bytes} bytes",
            bytes.len()
        )));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect())
}

/// Encodes a `KDNS` envelope and signs it with `key` under `domain`.
pub fn encode_envelope(
    kind: u8,
    metadata: &[u8],
    payload: &[u8],
    domain: &[u8],
    key: &SessionAuthKey,
    limits: &NativeSessionLimits,
) -> Result<Vec<u8>> {
    limits.validate()?;
    if metadata.len() > limits.max_metadata_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: metadata.len(),
            limit: limits.max_metadata_bytes,
        });
    }
    if payload.len() > limits.max_payload_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: payload.len(),
            limit: limits.max_payload_bytes,
        });
    }
    let total_len = NATIVE_SESSION_HEADER_BYTES
        .checked_add(metadata.len())
        .and_then(|n| n.checked_add(payload.len()))
        .ok_or_else(|| NativeDecoderError::Wire("frame length overflowed usize".into()))?;
    if total_len > limits.max_frame_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: total_len,
            limit: limits.max_frame_bytes,
        });
    }
    let metadata_len = u32::try_from(metadata.len())
        .map_err(|_| NativeDecoderError::Wire("metadata length exceeds u32".into()))?;
    let payload_len = u32::try_from(payload.len())
        .map_err(|_| NativeDecoderError::Wire("payload length exceeds u32".into()))?;

    let mut header_prefix = [0u8; 14];
    header_prefix[0..4].copy_from_slice(&NATIVE_SESSION_MAGIC);
    header_prefix[4] = NATIVE_SESSION_VERSION;
    header_prefix[5] = kind;
    header_prefix[6..10].copy_from_slice(&metadata_len.to_le_bytes());
    header_prefix[10..14].copy_from_slice(&payload_len.to_le_bytes());

    let tag = key.sign(domain, &header_prefix, metadata, payload);

    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&header_prefix);
    out.extend_from_slice(&tag);
    out.extend_from_slice(metadata);
    out.extend_from_slice(payload);
    Ok(out)
}

/// Verifies size limits, header layout, exact length and HMAC-SHA256 tag before returning
/// `(kind, metadata_slice, payload_slice)`.
pub fn decode_envelope<'a>(
    bytes: &'a [u8],
    expected_domain: &[u8],
    key: &SessionAuthKey,
    limits: &NativeSessionLimits,
) -> Result<(u8, &'a [u8], &'a [u8])> {
    limits.validate()?;
    if bytes.len() > limits.max_frame_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: bytes.len(),
            limit: limits.max_frame_bytes,
        });
    }
    if bytes.len() < NATIVE_SESSION_HEADER_BYTES {
        return Err(NativeDecoderError::Wire(format!(
            "frame is {} bytes; KDNS header requires {NATIVE_SESSION_HEADER_BYTES} bytes",
            bytes.len()
        )));
    }
    if bytes[0..4] != NATIVE_SESSION_MAGIC {
        return Err(NativeDecoderError::Wire(
            "missing KDNS magic in native session frame".into(),
        ));
    }
    if bytes[4] != NATIVE_SESSION_VERSION {
        return Err(NativeDecoderError::Wire(format!(
            "unsupported KDNS protocol version {}",
            bytes[4]
        )));
    }
    let kind = bytes[5];
    let metadata_len = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
    let payload_len = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
    if metadata_len > limits.max_metadata_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: metadata_len,
            limit: limits.max_metadata_bytes,
        });
    }
    if payload_len > limits.max_payload_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: payload_len,
            limit: limits.max_payload_bytes,
        });
    }
    let payload_start = NATIVE_SESSION_HEADER_BYTES
        .checked_add(metadata_len)
        .ok_or_else(|| NativeDecoderError::Wire("metadata offset overflowed usize".into()))?;
    let expected_total = payload_start
        .checked_add(payload_len)
        .ok_or_else(|| NativeDecoderError::Wire("frame length overflowed usize".into()))?;
    if expected_total > limits.max_frame_bytes {
        return Err(NativeDecoderError::FrameTooLarge {
            actual: expected_total,
            limit: limits.max_frame_bytes,
        });
    }
    if bytes.len() != expected_total {
        return Err(NativeDecoderError::Wire(format!(
            "frame length {} does not match header total {expected_total}",
            bytes.len()
        )));
    }
    let header_prefix = &bytes[0..14];
    let tag: &[u8; NATIVE_SESSION_MAC_BYTES] = bytes[14..NATIVE_SESSION_HEADER_BYTES]
        .try_into()
        .expect("32-byte slice");
    let metadata = &bytes[NATIVE_SESSION_HEADER_BYTES..payload_start];
    let payload = &bytes[payload_start..expected_total];
    if !key.verify(expected_domain, header_prefix, metadata, payload, tag) {
        return Err(NativeDecoderError::Authentication(
            "HMAC-SHA256 tag verification failed".into(),
        ));
    }
    Ok((kind, metadata, payload))
}

fn validate_session_id(session_id: &str) -> Result<()> {
    if session_id.is_empty() || session_id.len() > NATIVE_SESSION_MAX_SESSION_ID_BYTES {
        return Err(NativeDecoderError::Session(format!(
            "session_id must contain 1..={NATIVE_SESSION_MAX_SESSION_ID_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_model_id(model_id: &str) -> Result<()> {
    if model_id.is_empty() || model_id.len() > NATIVE_SESSION_MAX_MODEL_ID_BYTES {
        return Err(NativeDecoderError::Session(format!(
            "model_id must contain 1..={NATIVE_SESSION_MAX_MODEL_ID_BYTES} bytes"
        )));
    }
    Ok(())
}

/// Deterministic 64-bit FNV-1a digest over the exact little-endian bytes of `values`.
pub fn fnv1a64_f32_le(values: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

/// Request to open a decoder slice session on a worker and allocate its empty [`KvCache`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionOpenRequest {
    pub session_id: String,
    pub model_id: String,
    pub start_layer: usize,
    pub end_layer: usize,
    pub embedding: bool,
    pub head: bool,
}

impl SessionOpenRequest {
    pub fn new(
        session_id: impl Into<String>,
        model_id: impl Into<String>,
        plan: &SlicePlan,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            model_id: model_id.into(),
            start_layer: plan.layers.start,
            end_layer: plan.layers.end,
            embedding: plan.embedding,
            head: plan.head,
        }
    }

    pub fn plan(&self) -> SlicePlan {
        SlicePlan {
            layers: self.start_layer..self.end_layer,
            embedding: self.embedding,
            head: self.head,
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_session_id(&self.session_id)?;
        validate_model_id(&self.model_id)?;
        if self.start_layer > self.end_layer {
            return Err(NativeDecoderError::Session(format!(
                "invalid slice layer range {}..{}",
                self.start_layer, self.end_layer
            )));
        }
        Ok(())
    }
}

/// Worker acknowledgement after opening a slice session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionOpenResponse {
    pub session_id: String,
    pub model_id: String,
    pub start_layer: usize,
    pub end_layer: usize,
    pub embedding: bool,
    pub head: bool,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub stored_weight_bytes: usize,
}

/// Input to a single forward step on an open slice session.
#[derive(Debug, Clone, PartialEq)]
pub enum ForwardInput {
    /// Token ids to embed at `start_position..` on a slice that holds the token embedding,
    /// before running its layers.
    Tokens {
        start_position: usize,
        tokens: Vec<u32>,
    },
    /// Hidden states from the previous slice (or local embedding), carried as exact `f32` bytes.
    Activations(Activations),
}

/// Request to run a forward step on an open slice session.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionForwardRequest {
    pub session_id: String,
    pub input: ForwardInput,
    /// When `true`, runs the slice's final norm and LM head after its layers and returns
    /// [`ForwardOutput::Logits`] instead of [`ForwardOutput::Activations`].
    pub emit_logits: bool,
}

impl SessionForwardRequest {
    pub fn activations(
        session_id: impl Into<String>,
        activations: Activations,
        emit_logits: bool,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            input: ForwardInput::Activations(activations),
            emit_logits,
        }
    }

    pub fn tokens(
        session_id: impl Into<String>,
        start_position: usize,
        tokens: Vec<u32>,
        emit_logits: bool,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            input: ForwardInput::Tokens {
                start_position,
                tokens,
            },
            emit_logits,
        }
    }
}

/// Output of a forward step on an open slice session.
#[derive(Debug, Clone, PartialEq)]
pub enum ForwardOutput {
    /// Hidden states after this slice's layers, carried as exact `f32` bytes for the next slice.
    Activations(Activations),
    /// Logits (`rows x vocab_size`) after the final norm and LM head, carried as exact `f32`
    /// bytes.
    Logits {
        start_position: usize,
        rows: usize,
        vocab_size: usize,
        data: Vec<f32>,
    },
}

/// Worker response to a [`SessionForwardRequest`].
#[derive(Debug, Clone, PartialEq)]
pub struct SessionForwardResponse {
    pub session_id: String,
    pub output: ForwardOutput,
    pub cache_positions: usize,
    pub kv_cache_bytes: usize,
    pub repops_digest_fnv1a: u64,
}

/// Request to close an open slice session and release its KV cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCloseRequest {
    pub session_id: String,
}

/// Worker acknowledgement after closing a slice session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCloseResponse {
    pub session_id: String,
    pub final_positions: usize,
    pub released_kv_bytes: usize,
}

/// Request to cancel an open slice session immediately, stopping any in-flight forward pass and
/// releasing its KV cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCancelRequest {
    pub session_id: String,
}

/// Worker acknowledgement after cancelling a slice session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCancelResponse {
    pub session_id: String,
    pub was_running: bool,
    pub released_kv_bytes: usize,
}

/// Any request sent in a `WireFrameKind::NativeDecoderSession` frame.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionRequest {
    Open(SessionOpenRequest),
    Forward(SessionForwardRequest),
    Close(SessionCloseRequest),
    Cancel(SessionCancelRequest),
}

/// Any response returned in a `WireFrameKind::NativeDecoderSession` frame.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionResponse {
    Open(SessionOpenResponse),
    Forward(SessionForwardResponse),
    Close(SessionCloseResponse),
    Cancel(SessionCancelResponse),
    Error {
        session_id: Option<String>,
        error: String,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "input_kind", rename_all = "snake_case")]
enum WireForwardInputMeta {
    Tokens {
        start_position: usize,
        token_count: usize,
    },
    Activations {
        start_position: usize,
        rows: usize,
        hidden_size: usize,
    },
}

#[derive(Serialize, Deserialize)]
struct WireForwardRequestMeta {
    session_id: String,
    emit_logits: bool,
    #[serde(flatten)]
    input: WireForwardInputMeta,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "output_kind", rename_all = "snake_case")]
enum WireForwardOutputMeta {
    Activations {
        start_position: usize,
        rows: usize,
        hidden_size: usize,
    },
    Logits {
        start_position: usize,
        rows: usize,
        vocab_size: usize,
    },
}

#[derive(Serialize, Deserialize)]
struct WireForwardResponseMeta {
    session_id: String,
    cache_positions: usize,
    kv_cache_bytes: usize,
    repops_digest_fnv1a: u64,
    #[serde(flatten)]
    output: WireForwardOutputMeta,
}

#[derive(Serialize, Deserialize)]
struct WireErrorMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    error: String,
}

impl SessionRequest {
    pub fn session_id(&self) -> &str {
        match self {
            Self::Open(r) => &r.session_id,
            Self::Forward(r) => &r.session_id,
            Self::Close(r) => &r.session_id,
            Self::Cancel(r) => &r.session_id,
        }
    }

    /// Encodes and signs this request into a `KDNS` frame payload.
    pub fn encode(&self, key: &SessionAuthKey, limits: &NativeSessionLimits) -> Result<Vec<u8>> {
        limits.validate()?;
        match self {
            Self::Open(req) => {
                req.validate()?;
                let metadata = serde_json::to_vec(req)
                    .map_err(|e| NativeDecoderError::Wire(format!("serialize OpenRequest: {e}")))?;
                encode_envelope(
                    KIND_OPEN_REQUEST,
                    &metadata,
                    &[],
                    REQUEST_DOMAIN,
                    key,
                    limits,
                )
            }
            Self::Forward(req) => {
                validate_session_id(&req.session_id)?;
                let (input_meta, payload) = match &req.input {
                    ForwardInput::Tokens {
                        start_position,
                        tokens,
                    } => {
                        let bytes = encode_u32_slice(tokens, limits.max_payload_bytes)?;
                        (
                            WireForwardInputMeta::Tokens {
                                start_position: *start_position,
                                token_count: tokens.len(),
                            },
                            bytes,
                        )
                    }
                    ForwardInput::Activations(act) => {
                        let bytes = encode_f32_tensor(
                            act.rows(),
                            act.hidden_size(),
                            act.data(),
                            limits.max_payload_bytes,
                        )?;
                        (
                            WireForwardInputMeta::Activations {
                                start_position: act.start_position(),
                                rows: act.rows(),
                                hidden_size: act.hidden_size(),
                            },
                            bytes,
                        )
                    }
                };
                let meta = WireForwardRequestMeta {
                    session_id: req.session_id.clone(),
                    emit_logits: req.emit_logits,
                    input: input_meta,
                };
                let metadata = serde_json::to_vec(&meta).map_err(|e| {
                    NativeDecoderError::Wire(format!("serialize ForwardRequest: {e}"))
                })?;
                encode_envelope(
                    KIND_FORWARD_REQUEST,
                    &metadata,
                    &payload,
                    REQUEST_DOMAIN,
                    key,
                    limits,
                )
            }
            Self::Close(req) => {
                validate_session_id(&req.session_id)?;
                let metadata = serde_json::to_vec(req).map_err(|e| {
                    NativeDecoderError::Wire(format!("serialize CloseRequest: {e}"))
                })?;
                encode_envelope(
                    KIND_CLOSE_REQUEST,
                    &metadata,
                    &[],
                    REQUEST_DOMAIN,
                    key,
                    limits,
                )
            }
            Self::Cancel(req) => {
                validate_session_id(&req.session_id)?;
                let metadata = serde_json::to_vec(req).map_err(|e| {
                    NativeDecoderError::Wire(format!("serialize CancelRequest: {e}"))
                })?;
                encode_envelope(
                    KIND_CANCEL_REQUEST,
                    &metadata,
                    &[],
                    REQUEST_DOMAIN,
                    key,
                    limits,
                )
            }
        }
    }

    /// Verifies and decodes a `KDNS` request frame payload.
    pub fn decode(
        bytes: &[u8],
        key: &SessionAuthKey,
        limits: &NativeSessionLimits,
    ) -> Result<Self> {
        let (kind, metadata, payload) = decode_envelope(bytes, REQUEST_DOMAIN, key, limits)?;
        match kind {
            KIND_OPEN_REQUEST => {
                if !payload.is_empty() {
                    return Err(NativeDecoderError::Wire(
                        "OpenRequest carries an unexpected binary payload".into(),
                    ));
                }
                let req: SessionOpenRequest = serde_json::from_slice(metadata)
                    .map_err(|e| NativeDecoderError::Wire(format!("invalid OpenRequest: {e}")))?;
                req.validate()?;
                Ok(Self::Open(req))
            }
            KIND_FORWARD_REQUEST => {
                let meta: WireForwardRequestMeta =
                    serde_json::from_slice(metadata).map_err(|e| {
                        NativeDecoderError::Wire(format!("invalid ForwardRequest: {e}"))
                    })?;
                validate_session_id(&meta.session_id)?;
                let input = match meta.input {
                    WireForwardInputMeta::Tokens {
                        start_position,
                        token_count,
                    } => {
                        let tokens =
                            decode_u32_slice(token_count, payload, limits.max_payload_bytes)?;
                        ForwardInput::Tokens {
                            start_position,
                            tokens,
                        }
                    }
                    WireForwardInputMeta::Activations {
                        start_position,
                        rows,
                        hidden_size,
                    } => {
                        let data = decode_f32_tensor(
                            rows,
                            hidden_size,
                            payload,
                            limits.max_payload_bytes,
                        )?;
                        let act = Activations::new(start_position, hidden_size, data)?;
                        ForwardInput::Activations(act)
                    }
                };
                Ok(Self::Forward(SessionForwardRequest {
                    session_id: meta.session_id,
                    input,
                    emit_logits: meta.emit_logits,
                }))
            }
            KIND_CLOSE_REQUEST => {
                if !payload.is_empty() {
                    return Err(NativeDecoderError::Wire(
                        "CloseRequest carries an unexpected binary payload".into(),
                    ));
                }
                let req: SessionCloseRequest = serde_json::from_slice(metadata)
                    .map_err(|e| NativeDecoderError::Wire(format!("invalid CloseRequest: {e}")))?;
                validate_session_id(&req.session_id)?;
                Ok(Self::Close(req))
            }
            KIND_CANCEL_REQUEST => {
                if !payload.is_empty() {
                    return Err(NativeDecoderError::Wire(
                        "CancelRequest carries an unexpected binary payload".into(),
                    ));
                }
                let req: SessionCancelRequest = serde_json::from_slice(metadata)
                    .map_err(|e| NativeDecoderError::Wire(format!("invalid CancelRequest: {e}")))?;
                validate_session_id(&req.session_id)?;
                Ok(Self::Cancel(req))
            }
            other => Err(NativeDecoderError::Wire(format!(
                "unexpected KDNS request kind {other}"
            ))),
        }
    }
}

impl SessionResponse {
    pub fn error(session_id: Option<String>, error: impl Into<String>) -> Self {
        Self::Error {
            session_id,
            error: error.into(),
        }
    }

    pub fn into_result(self) -> Result<Self> {
        match self {
            Self::Error { error, .. } => Err(NativeDecoderError::Session(error)),
            other => Ok(other),
        }
    }

    /// Encodes and signs this response into a `KDNS` frame payload.
    pub fn encode(&self, key: &SessionAuthKey, limits: &NativeSessionLimits) -> Result<Vec<u8>> {
        limits.validate()?;
        match self {
            Self::Open(resp) => {
                validate_session_id(&resp.session_id)?;
                validate_model_id(&resp.model_id)?;
                let metadata = serde_json::to_vec(resp).map_err(|e| {
                    NativeDecoderError::Wire(format!("serialize OpenResponse: {e}"))
                })?;
                encode_envelope(
                    KIND_OPEN_RESPONSE,
                    &metadata,
                    &[],
                    RESPONSE_DOMAIN,
                    key,
                    limits,
                )
            }
            Self::Forward(resp) => {
                validate_session_id(&resp.session_id)?;
                let (output_meta, payload) = match &resp.output {
                    ForwardOutput::Activations(act) => {
                        let bytes = encode_f32_tensor(
                            act.rows(),
                            act.hidden_size(),
                            act.data(),
                            limits.max_payload_bytes,
                        )?;
                        (
                            WireForwardOutputMeta::Activations {
                                start_position: act.start_position(),
                                rows: act.rows(),
                                hidden_size: act.hidden_size(),
                            },
                            bytes,
                        )
                    }
                    ForwardOutput::Logits {
                        start_position,
                        rows,
                        vocab_size,
                        data,
                    } => {
                        let bytes =
                            encode_f32_tensor(*rows, *vocab_size, data, limits.max_payload_bytes)?;
                        (
                            WireForwardOutputMeta::Logits {
                                start_position: *start_position,
                                rows: *rows,
                                vocab_size: *vocab_size,
                            },
                            bytes,
                        )
                    }
                };
                let meta = WireForwardResponseMeta {
                    session_id: resp.session_id.clone(),
                    cache_positions: resp.cache_positions,
                    kv_cache_bytes: resp.kv_cache_bytes,
                    repops_digest_fnv1a: resp.repops_digest_fnv1a,
                    output: output_meta,
                };
                let metadata = serde_json::to_vec(&meta).map_err(|e| {
                    NativeDecoderError::Wire(format!("serialize ForwardResponse: {e}"))
                })?;
                encode_envelope(
                    KIND_FORWARD_RESPONSE,
                    &metadata,
                    &payload,
                    RESPONSE_DOMAIN,
                    key,
                    limits,
                )
            }
            Self::Close(resp) => {
                validate_session_id(&resp.session_id)?;
                let metadata = serde_json::to_vec(resp).map_err(|e| {
                    NativeDecoderError::Wire(format!("serialize CloseResponse: {e}"))
                })?;
                encode_envelope(
                    KIND_CLOSE_RESPONSE,
                    &metadata,
                    &[],
                    RESPONSE_DOMAIN,
                    key,
                    limits,
                )
            }
            Self::Cancel(resp) => {
                validate_session_id(&resp.session_id)?;
                let metadata = serde_json::to_vec(resp).map_err(|e| {
                    NativeDecoderError::Wire(format!("serialize CancelResponse: {e}"))
                })?;
                encode_envelope(
                    KIND_CANCEL_RESPONSE,
                    &metadata,
                    &[],
                    RESPONSE_DOMAIN,
                    key,
                    limits,
                )
            }
            Self::Error { session_id, error } => {
                if let Some(id) = session_id {
                    validate_session_id(id)?;
                }
                let meta = WireErrorMeta {
                    session_id: session_id.clone(),
                    error: error.clone(),
                };
                let metadata = serde_json::to_vec(&meta).map_err(|e| {
                    NativeDecoderError::Wire(format!("serialize ErrorResponse: {e}"))
                })?;
                encode_envelope(
                    KIND_ERROR_RESPONSE,
                    &metadata,
                    &[],
                    RESPONSE_DOMAIN,
                    key,
                    limits,
                )
            }
        }
    }

    /// Verifies and decodes a `KDNS` response frame payload.
    pub fn decode(
        bytes: &[u8],
        key: &SessionAuthKey,
        limits: &NativeSessionLimits,
    ) -> Result<Self> {
        // When a worker rejects an unauthenticated or oversized frame before signing, it answers
        // with a small JSON object `{"error":"..."}` so the channel fails fast instead of hanging.
        // Never accept such a reply as `Ok`: surface it as an error.
        if bytes.first() == Some(&b'{')
            && bytes.len() <= 4096
            && let Ok(val) = serde_json::from_slice::<serde_json::Value>(bytes)
            && let Some(msg) = val.get("error").and_then(|v| v.as_str())
        {
            if msg.contains("authentication") {
                return Err(NativeDecoderError::Authentication(msg.to_string()));
            }
            return Err(NativeDecoderError::Wire(msg.to_string()));
        }

        let (kind, metadata, payload) = decode_envelope(bytes, RESPONSE_DOMAIN, key, limits)?;
        match kind {
            KIND_OPEN_RESPONSE => {
                if !payload.is_empty() {
                    return Err(NativeDecoderError::Wire(
                        "OpenResponse carries an unexpected binary payload".into(),
                    ));
                }
                let resp: SessionOpenResponse = serde_json::from_slice(metadata)
                    .map_err(|e| NativeDecoderError::Wire(format!("invalid OpenResponse: {e}")))?;
                validate_session_id(&resp.session_id)?;
                validate_model_id(&resp.model_id)?;
                Ok(Self::Open(resp))
            }
            KIND_FORWARD_RESPONSE => {
                let meta: WireForwardResponseMeta =
                    serde_json::from_slice(metadata).map_err(|e| {
                        NativeDecoderError::Wire(format!("invalid ForwardResponse: {e}"))
                    })?;
                validate_session_id(&meta.session_id)?;
                let (output, expected_digest) = match meta.output {
                    WireForwardOutputMeta::Activations {
                        start_position,
                        rows,
                        hidden_size,
                    } => {
                        let data = decode_f32_tensor(
                            rows,
                            hidden_size,
                            payload,
                            limits.max_payload_bytes,
                        )?;
                        let digest = fnv1a64_f32_le(&data);
                        let act = Activations::new(start_position, hidden_size, data)?;
                        (ForwardOutput::Activations(act), digest)
                    }
                    WireForwardOutputMeta::Logits {
                        start_position,
                        rows,
                        vocab_size,
                    } => {
                        let data =
                            decode_f32_tensor(rows, vocab_size, payload, limits.max_payload_bytes)?;
                        let digest = fnv1a64_f32_le(&data);
                        (
                            ForwardOutput::Logits {
                                start_position,
                                rows,
                                vocab_size,
                                data,
                            },
                            digest,
                        )
                    }
                };
                if meta.repops_digest_fnv1a != expected_digest {
                    return Err(NativeDecoderError::Wire(format!(
                        "ForwardResponse digest {:#018x} does not match payload digest {:#018x}",
                        meta.repops_digest_fnv1a, expected_digest
                    )));
                }
                Ok(Self::Forward(SessionForwardResponse {
                    session_id: meta.session_id,
                    output,
                    cache_positions: meta.cache_positions,
                    kv_cache_bytes: meta.kv_cache_bytes,
                    repops_digest_fnv1a: meta.repops_digest_fnv1a,
                }))
            }
            KIND_CLOSE_RESPONSE => {
                if !payload.is_empty() {
                    return Err(NativeDecoderError::Wire(
                        "CloseResponse carries an unexpected binary payload".into(),
                    ));
                }
                let resp: SessionCloseResponse = serde_json::from_slice(metadata)
                    .map_err(|e| NativeDecoderError::Wire(format!("invalid CloseResponse: {e}")))?;
                validate_session_id(&resp.session_id)?;
                Ok(Self::Close(resp))
            }
            KIND_CANCEL_RESPONSE => {
                if !payload.is_empty() {
                    return Err(NativeDecoderError::Wire(
                        "CancelResponse carries an unexpected binary payload".into(),
                    ));
                }
                let resp: SessionCancelResponse =
                    serde_json::from_slice(metadata).map_err(|e| {
                        NativeDecoderError::Wire(format!("invalid CancelResponse: {e}"))
                    })?;
                validate_session_id(&resp.session_id)?;
                Ok(Self::Cancel(resp))
            }
            KIND_ERROR_RESPONSE => {
                if !payload.is_empty() {
                    return Err(NativeDecoderError::Wire(
                        "ErrorResponse carries an unexpected binary payload".into(),
                    ));
                }
                let meta: WireErrorMeta = serde_json::from_slice(metadata)
                    .map_err(|e| NativeDecoderError::Wire(format!("invalid ErrorResponse: {e}")))?;
                Ok(Self::Error {
                    session_id: meta.session_id,
                    error: meta.error,
                })
            }
            other => Err(NativeDecoderError::Wire(format!(
                "unexpected KDNS response kind {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SliceCacheKey {
    model_id: String,
    start_layer: usize,
    end_layer: usize,
    embedding: bool,
    head: bool,
}

impl SliceCacheKey {
    fn new(model_id: &str, plan: &SlicePlan) -> Self {
        Self {
            model_id: model_id.to_string(),
            start_layer: plan.layers.start,
            end_layer: plan.layers.end,
            embedding: plan.embedding,
            head: plan.head,
        }
    }
}

struct ActiveSession {
    slice: Arc<DecoderSlice>,
    cache: Mutex<KvCache>,
    in_flight: AtomicBool,
    cancelled: AtomicBool,
    last_kv_bytes: std::sync::atomic::AtomicUsize,
}

struct ResetInFlight<'a>(&'a AtomicBool);

impl Drop for ResetInFlight<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

struct NativeSessionHostInner {
    auth_key: RwLock<Option<SessionAuthKey>>,
    limits: RwLock<NativeSessionLimits>,
    model_dirs: RwLock<BTreeMap<String, PathBuf>>,
    slices: Mutex<BTreeMap<SliceCacheKey, Arc<DecoderSlice>>>,
    sessions: Mutex<BTreeMap<String, Arc<ActiveSession>>>,
}

fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

fn mutex_lock<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Worker-side host for authenticated native decoder slice sessions.
///
/// Shares loaded [`DecoderSlice`] weights across sessions that run the same `(model_id, plan)`
/// and gives each session its own [`KvCache`] and cancellation flag.
#[derive(Clone)]
pub struct NativeSessionHost {
    inner: Arc<NativeSessionHostInner>,
}

impl Default for NativeSessionHost {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeSessionHost {
    pub fn new() -> Self {
        let auth_key = SessionAuthKey::from_env().and_then(std::result::Result::ok);
        Self {
            inner: Arc::new(NativeSessionHostInner {
                auth_key: RwLock::new(auth_key),
                limits: RwLock::new(NativeSessionLimits::default()),
                model_dirs: RwLock::new(BTreeMap::new()),
                slices: Mutex::new(BTreeMap::new()),
                sessions: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    pub fn set_auth_key(&self, key: SessionAuthKey) {
        *write_lock(&self.inner.auth_key) = Some(key);
    }

    pub fn clear_auth_key(&self) {
        *write_lock(&self.inner.auth_key) = None;
    }

    pub fn auth_key(&self) -> Option<SessionAuthKey> {
        read_lock(&self.inner.auth_key).clone()
    }

    pub fn set_limits(&self, limits: NativeSessionLimits) -> Result<()> {
        limits.validate()?;
        *write_lock(&self.inner.limits) = limits;
        Ok(())
    }

    pub fn limits(&self) -> NativeSessionLimits {
        *read_lock(&self.inner.limits)
    }

    /// Registers a local SafeTensors checkpoint directory under `model_id` so the worker can load
    /// requested layer slices on demand.
    pub fn register_model_dir(&self, model_id: impl Into<String>, model_dir: impl AsRef<Path>) {
        write_lock(&self.inner.model_dirs)
            .insert(model_id.into(), model_dir.as_ref().to_path_buf());
    }

    /// Pre-registers an already loaded [`DecoderSlice`] under `model_id`.
    pub fn register_slice(&self, model_id: impl Into<String>, slice: Arc<DecoderSlice>) {
        let model_id = model_id.into();
        let key = SliceCacheKey::new(&model_id, slice.plan());
        mutex_lock(&self.inner.slices).insert(key, slice);
    }

    /// Number of currently open sessions on this worker.
    pub fn active_session_count(&self) -> usize {
        mutex_lock(&self.inner.sessions).len()
    }

    /// Total KV cache bytes held across all currently open sessions.
    pub fn active_kv_cache_bytes(&self) -> usize {
        mutex_lock(&self.inner.sessions)
            .values()
            .map(|s| s.last_kv_bytes.load(Ordering::Relaxed))
            .sum()
    }

    fn resolve_slice(&self, model_id: &str, plan: &SlicePlan) -> Result<Arc<DecoderSlice>> {
        let key = SliceCacheKey::new(model_id, plan);
        if let Some(existing) = mutex_lock(&self.inner.slices).get(&key).cloned() {
            return Ok(existing);
        }
        let dir = read_lock(&self.inner.model_dirs)
            .get(model_id)
            .cloned()
            .ok_or_else(|| {
                NativeDecoderError::Session(format!(
                    "model `{model_id}` is not registered on this worker"
                ))
            })?;
        let loaded = Arc::new(DecoderSlice::load(&dir, plan.clone())?);
        let mut cache = mutex_lock(&self.inner.slices);
        Ok(Arc::clone(
            cache.entry(key).or_insert_with(|| Arc::clone(&loaded)),
        ))
    }

    /// Executes a decoded [`SessionRequest`] against the worker's slices and open sessions.
    pub fn execute(&self, request: SessionRequest) -> SessionResponse {
        let session_id = Some(request.session_id().to_string());
        match self.execute_inner(request) {
            Ok(response) => response,
            Err(error) => SessionResponse::error(session_id, error.to_string()),
        }
    }

    fn execute_inner(&self, request: SessionRequest) -> Result<SessionResponse> {
        let limits = self.limits();
        match request {
            SessionRequest::Open(req) => {
                req.validate()?;
                let plan = req.plan();
                let slice = self.resolve_slice(&req.model_id, &plan)?;
                let mut sessions = mutex_lock(&self.inner.sessions);
                if sessions.contains_key(&req.session_id) {
                    return Err(NativeDecoderError::Session(format!(
                        "session `{}` is already open",
                        req.session_id
                    )));
                }
                if sessions.len() >= limits.max_active_sessions {
                    return Err(NativeDecoderError::Session(format!(
                        "worker already holds {} active sessions (limit is {})",
                        sessions.len(),
                        limits.max_active_sessions
                    )));
                }
                let cache = slice.new_cache();
                let config = slice.config();
                let resp = SessionOpenResponse {
                    session_id: req.session_id.clone(),
                    model_id: req.model_id,
                    start_layer: plan.layers.start,
                    end_layer: plan.layers.end,
                    embedding: plan.embedding,
                    head: plan.head,
                    hidden_size: config.hidden_size,
                    vocab_size: config.vocab_size,
                    max_position_embeddings: config.max_position_embeddings,
                    stored_weight_bytes: slice.stored_weight_bytes(),
                };
                sessions.insert(
                    req.session_id,
                    Arc::new(ActiveSession {
                        slice,
                        cache: Mutex::new(cache),
                        in_flight: AtomicBool::new(false),
                        cancelled: AtomicBool::new(false),
                        last_kv_bytes: std::sync::atomic::AtomicUsize::new(0),
                    }),
                );
                Ok(SessionResponse::Open(resp))
            }
            SessionRequest::Forward(req) => {
                validate_session_id(&req.session_id)?;
                let session = mutex_lock(&self.inner.sessions)
                    .get(&req.session_id)
                    .cloned()
                    .ok_or_else(|| {
                        NativeDecoderError::Session(format!(
                            "session `{}` is not open",
                            req.session_id
                        ))
                    })?;
                if session.cancelled.load(Ordering::SeqCst) {
                    return Err(NativeDecoderError::Cancelled(req.session_id));
                }
                if session.in_flight.swap(true, Ordering::SeqCst) {
                    return Err(NativeDecoderError::Session(format!(
                        "session `{}` already has a forward step in flight",
                        req.session_id
                    )));
                }
                let _in_flight_guard = ResetInFlight(&session.in_flight);

                let mut cache = mutex_lock(&session.cache);
                if session.cancelled.load(Ordering::SeqCst) {
                    return Err(NativeDecoderError::Cancelled(req.session_id));
                }
                let input_activations = match req.input {
                    ForwardInput::Tokens {
                        start_position,
                        tokens,
                    } => session.slice.embed(&tokens, start_position)?,
                    ForwardInput::Activations(act) => act,
                };
                let hidden = match session.slice.forward_until(
                    &input_activations,
                    &mut cache,
                    &session.cancelled,
                ) {
                    Ok(hidden) => hidden,
                    Err(NativeDecoderError::Cancelled(_)) => {
                        return Err(NativeDecoderError::Cancelled(req.session_id));
                    }
                    Err(other) => return Err(other),
                };
                let (output, digest) = if req.emit_logits {
                    let logits = session.slice.logits(&hidden)?;
                    let digest = fnv1a64_f32_le(&logits);
                    (
                        ForwardOutput::Logits {
                            start_position: hidden.start_position(),
                            rows: hidden.rows(),
                            vocab_size: session.slice.config().vocab_size,
                            data: logits,
                        },
                        digest,
                    )
                } else {
                    let digest = fnv1a64_f32_le(hidden.data());
                    (ForwardOutput::Activations(hidden), digest)
                };
                let cache_positions = cache.positions();
                let kv_cache_bytes = cache.bytes();
                session
                    .last_kv_bytes
                    .store(kv_cache_bytes, Ordering::Relaxed);
                Ok(SessionResponse::Forward(SessionForwardResponse {
                    session_id: req.session_id,
                    output,
                    cache_positions,
                    kv_cache_bytes,
                    repops_digest_fnv1a: digest,
                }))
            }
            SessionRequest::Close(req) => {
                validate_session_id(&req.session_id)?;
                let session = mutex_lock(&self.inner.sessions)
                    .remove(&req.session_id)
                    .ok_or_else(|| {
                        NativeDecoderError::Session(format!(
                            "session `{}` is not open",
                            req.session_id
                        ))
                    })?;
                session.cancelled.store(true, Ordering::SeqCst);
                let cache = mutex_lock(&session.cache);
                Ok(SessionResponse::Close(SessionCloseResponse {
                    session_id: req.session_id,
                    final_positions: cache.positions(),
                    released_kv_bytes: cache.bytes(),
                }))
            }
            SessionRequest::Cancel(req) => {
                validate_session_id(&req.session_id)?;
                let removed = mutex_lock(&self.inner.sessions).remove(&req.session_id);
                let Some(session) = removed else {
                    return Ok(SessionResponse::Cancel(SessionCancelResponse {
                        session_id: req.session_id,
                        was_running: false,
                        released_kv_bytes: 0,
                    }));
                };
                session.cancelled.store(true, Ordering::SeqCst);
                let was_running = session.in_flight.load(Ordering::SeqCst);
                let released_kv_bytes = match session.cache.try_lock() {
                    Ok(cache) => cache.bytes(),
                    Err(_) => session.last_kv_bytes.load(Ordering::Relaxed),
                };
                Ok(SessionResponse::Cancel(SessionCancelResponse {
                    session_id: req.session_id,
                    was_running,
                    released_kv_bytes,
                }))
            }
        }
    }

    /// Handles a raw `WireFrameKind::NativeDecoderSession` payload on the worker and returns the
    /// response payload bytes, along with `true` when a forward step completed.
    pub fn handle_frame_payload(&self, payload: &[u8]) -> (Vec<u8>, bool) {
        let Some(key) = self.auth_key() else {
            return (
                br#"{"error":"native decoder session authentication is not configured on this worker"}"#
                    .to_vec(),
                false,
            );
        };
        let limits = self.limits();
        match SessionRequest::decode(payload, &key, &limits) {
            Ok(request) => {
                let response = self.execute(request);
                let counted_forward = matches!(response, SessionResponse::Forward(_));
                let encoded = response.encode(&key, &limits).unwrap_or_else(|error| {
                    SessionResponse::error(None, error.to_string())
                        .encode(&key, &limits)
                        .unwrap_or_else(|_| {
                            br#"{"error":"failed to encode native session error response"}"#
                                .to_vec()
                        })
                });
                (encoded, counted_forward)
            }
            Err(error) => {
                let message = error.to_string();
                let body = serde_json::json!({ "error": message });
                (
                    serde_json::to_vec(&body).unwrap_or_else(|_| {
                        br#"{"error":"invalid native decoder session frame"}"#.to_vec()
                    }),
                    false,
                )
            }
        }
    }
}
