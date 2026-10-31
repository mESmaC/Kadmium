//! InferSSD-style expert resolution, exact reads, bounded residency, and portable wire packets.
//!
//! Sources keep their files open and read only the selected expert's tensor extents. Payloads
//! remain in their original quantization/encoding. The wire codec puts metadata in JSON and all
//! tensor payloads in one contiguous binary section, avoiding JSON byte-array expansion.
//!
//! File sources require a nonzero, catalog-issued durable checkpoint fingerprint. Source file
//! lengths and modification times are checked before and after every batch to reject ordinary
//! checkpoint replacements. As with InferSSD, fingerprinted files are an immutable-file contract;
//! these generation checks are not content authentication. Packet FNV-1a checksums detect wire
//! corruption, not malicious tampering.

use crate::safetensors_manifest::{SafetensorsManifest, SafetensorsManifestError};
use crate::transcoder::SafetensorsToKtbTranscoder;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::SystemTime;

pub const EXPERT_WIRE_MAGIC: [u8; 4] = *b"KDEX";
pub const EXPERT_WIRE_VERSION: u8 = 1;
pub const EXPERT_WIRE_HEADER_BYTES: usize = 10;
pub const EXPERT_LOAD_MAX_EXPERTS: usize = 256;
pub const EXPERT_LOAD_MAX_TENSORS_PER_EXPERT: usize = 64;
pub const EXPERT_LOAD_MAX_METADATA_BYTES: usize = 16 * 1024 * 1024;
pub const EXPERT_LOAD_MAX_FRAME_BYTES: usize = 1 << 30;
pub const EXPERT_LOAD_MAX_PAYLOAD_BYTES: usize =
    EXPERT_LOAD_MAX_FRAME_BYTES - EXPERT_LOAD_MAX_METADATA_BYTES - EXPERT_WIRE_HEADER_BYTES;
pub const DEFAULT_EXPERT_CACHE_BYTES: usize = 1 << 30;
pub const MAX_CONCURRENT_EXPERT_LOADS: usize = 8;
const REQUEST_KIND: u8 = 1;
const RESPONSE_KIND: u8 = 2;
const MAX_MODEL_ID_BYTES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub const fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertLoadRequest {
    pub model_id: String,
    pub source_fingerprint: u64,
    pub experts: Vec<ExpertKey>,
}

impl ExpertLoadRequest {
    pub fn new(
        model_id: impl Into<String>,
        source_fingerprint: u64,
        experts: Vec<ExpertKey>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            source_fingerprint,
            experts,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_source_identity(&self.model_id, self.source_fingerprint)?;
        if self.experts.is_empty() || self.experts.len() > EXPERT_LOAD_MAX_EXPERTS {
            return Err(format!(
                "expert request must contain 1..={EXPERT_LOAD_MAX_EXPERTS} experts"
            ));
        }
        let unique: BTreeSet<_> = self.experts.iter().copied().collect();
        if unique.len() != self.experts.len() {
            return Err("expert request contains duplicate keys".into());
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let metadata = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        encode_envelope(REQUEST_KIND, &metadata, std::iter::empty::<&[u8]>())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let (metadata, payload) = split_envelope(bytes, REQUEST_KIND)?;
        if !payload.is_empty() {
            return Err("expert request contains an unexpected binary payload".into());
        }
        let request: Self = serde_json::from_slice(metadata)
            .map_err(|e| format!("invalid expert request metadata: {e}"))?;
        request.validate()?;
        Ok(request)
    }
}

pub(crate) fn validate_source_identity(
    model_id: &str,
    source_fingerprint: u64,
) -> Result<(), String> {
    validate_model_id(model_id)?;
    if source_fingerprint == 0 {
        return Err("checkpoint fingerprint 0 is reserved for an unknown checkpoint".into());
    }
    Ok(())
}

fn validate_model_id(model_id: &str) -> Result<(), String> {
    if model_id.is_empty() || model_id.len() > MAX_MODEL_ID_BYTES {
        return Err(format!(
            "model id must contain 1..={MAX_MODEL_ID_BYTES} bytes"
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: Arc<[u8]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPacket {
    pub layer: u32,
    pub expert: u32,
    pub checksum_fnv64: u64,
    pub tensors: Vec<ExpertTensor>,
}

impl ExpertPacket {
    pub fn new(layer: u32, expert: u32, tensors: Vec<ExpertTensor>) -> Result<Self, String> {
        let packet = Self {
            layer,
            expert,
            checksum_fnv64: packet_checksum(layer, expert, &tensors),
            tensors,
        };
        packet.validate()?;
        Ok(packet)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.tensors.is_empty() || self.tensors.len() > EXPERT_LOAD_MAX_TENSORS_PER_EXPERT {
            return Err(format!(
                "expert packet must contain 1..={EXPERT_LOAD_MAX_TENSORS_PER_EXPERT} tensors"
            ));
        }
        let mut names = BTreeSet::new();
        for tensor in &self.tensors {
            if tensor.name.is_empty() || tensor.dtype.is_empty() {
                return Err("expert tensor names and dtypes must not be empty".into());
            }
            if !names.insert(tensor.name.as_str()) {
                return Err(format!(
                    "duplicate tensor `{}` in expert {}/{}",
                    tensor.name, self.layer, self.expert
                ));
            }
        }
        let expected = packet_checksum(self.layer, self.expert, &self.tensors);
        if self.checksum_fnv64 != expected {
            return Err(format!(
                "checksum mismatch for expert {}/{}",
                self.layer, self.expert
            ));
        }
        Ok(())
    }

    pub const fn key(&self) -> ExpertKey {
        ExpertKey::new(self.layer, self.expert)
    }

    pub fn payload_bytes(&self) -> usize {
        self.tensors.iter().map(|tensor| tensor.bytes.len()).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertLoadResponse {
    pub model_id: String,
    pub source_fingerprint: u64,
    pub experts: Vec<ExpertPacket>,
    pub error: Option<String>,
}

impl ExpertLoadResponse {
    pub fn success(
        model_id: impl Into<String>,
        source_fingerprint: u64,
        experts: Vec<ExpertPacket>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            source_fingerprint,
            experts,
            error: None,
        }
    }

    pub fn error(
        model_id: impl Into<String>,
        source_fingerprint: u64,
        error: impl Into<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            source_fingerprint,
            experts: Vec::new(),
            error: Some(error.into()),
        }
    }

    pub fn into_result(self) -> Result<Self, String> {
        match self.error {
            Some(error) => Err(error),
            None => Ok(self),
        }
    }

    pub fn validate_for_request(&self, request: &ExpertLoadRequest) -> Result<(), String> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.model_id != request.model_id
            || self.source_fingerprint != request.source_fingerprint
        {
            return Err(
                "expert response model or checkpoint fingerprint does not match request".into(),
            );
        }
        let actual: Vec<_> = self.experts.iter().map(ExpertPacket::key).collect();
        if actual != request.experts {
            return Err("expert response keys do not match the requested order".into());
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct WireResponseMetadata {
    model_id: String,
    source_fingerprint: u64,
    #[serde(default)]
    experts: Vec<WirePacketMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct WirePacketMetadata {
    layer: u32,
    expert: u32,
    checksum_fnv64: u64,
    tensors: Vec<WireTensorMetadata>,
}

#[derive(Serialize, Deserialize)]
struct WireTensorMetadata {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    byte_len: u64,
}

impl ExpertLoadResponse {
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        validate_model_id(&self.model_id)?;
        if self.error.is_some() && !self.experts.is_empty() {
            return Err("an error response cannot carry expert packets".into());
        }
        if self.experts.len() > EXPERT_LOAD_MAX_EXPERTS {
            return Err("too many experts in response".into());
        }
        let mut keys = BTreeSet::new();
        let metadata = WireResponseMetadata {
            model_id: self.model_id.clone(),
            source_fingerprint: self.source_fingerprint,
            experts: self
                .experts
                .iter()
                .map(|packet| {
                    packet.validate()?;
                    if !keys.insert(packet.key()) {
                        return Err(format!("duplicate response expert {:?}", packet.key()));
                    }
                    Ok(WirePacketMetadata {
                        layer: packet.layer,
                        expert: packet.expert,
                        checksum_fnv64: packet.checksum_fnv64,
                        tensors: packet
                            .tensors
                            .iter()
                            .map(|tensor| WireTensorMetadata {
                                name: tensor.name.clone(),
                                dtype: tensor.dtype.clone(),
                                shape: tensor.shape.clone(),
                                byte_len: tensor.bytes.len() as u64,
                            })
                            .collect(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
            error: self.error.clone(),
        };
        let metadata = serde_json::to_vec(&metadata).map_err(|e| e.to_string())?;
        encode_envelope(
            RESPONSE_KIND,
            &metadata,
            self.experts
                .iter()
                .flat_map(|packet| packet.tensors.iter())
                .map(|tensor| tensor.bytes.as_ref()),
        )
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let (metadata, mut payload) = split_envelope(bytes, RESPONSE_KIND)?;
        let metadata: WireResponseMetadata = serde_json::from_slice(metadata)
            .map_err(|e| format!("invalid expert response metadata: {e}"))?;
        validate_model_id(&metadata.model_id)?;
        if metadata.experts.len() > EXPERT_LOAD_MAX_EXPERTS {
            return Err("too many experts in response".into());
        }
        if metadata.error.is_some() {
            if !metadata.experts.is_empty() || !payload.is_empty() {
                return Err("an error response carries expert data".into());
            }
            return Ok(Self {
                model_id: metadata.model_id,
                source_fingerprint: metadata.source_fingerprint,
                experts: Vec::new(),
                error: metadata.error,
            });
        }

        let mut keys = BTreeSet::new();
        let mut experts = Vec::with_capacity(metadata.experts.len());
        for packet in metadata.experts {
            let key = ExpertKey::new(packet.layer, packet.expert);
            if !keys.insert(key) {
                return Err(format!("duplicate response expert {key:?}"));
            }
            if packet.tensors.is_empty()
                || packet.tensors.len() > EXPERT_LOAD_MAX_TENSORS_PER_EXPERT
            {
                return Err(format!("invalid tensor count for response expert {key:?}"));
            }
            let mut tensors = Vec::with_capacity(packet.tensors.len());
            for tensor in packet.tensors {
                let len = usize::try_from(tensor.byte_len)
                    .map_err(|_| "tensor payload length does not fit this platform".to_string())?;
                if len > payload.len() {
                    return Err(format!("truncated payload for tensor `{}`", tensor.name));
                }
                let (tensor_bytes, rest) = payload.split_at(len);
                payload = rest;
                tensors.push(ExpertTensor {
                    name: tensor.name,
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    bytes: Arc::from(tensor_bytes),
                });
            }
            let decoded = ExpertPacket::new(packet.layer, packet.expert, tensors)?;
            if decoded.checksum_fnv64 != packet.checksum_fnv64 {
                return Err(format!("checksum mismatch for response expert {key:?}"));
            }
            experts.push(decoded);
        }
        if !payload.is_empty() {
            return Err(format!(
                "expert response has {} unclaimed payload bytes",
                payload.len()
            ));
        }
        Ok(Self {
            model_id: metadata.model_id,
            source_fingerprint: metadata.source_fingerprint,
            experts,
            error: None,
        })
    }
}

fn encode_envelope<'a>(
    kind: u8,
    metadata: &[u8],
    payload: impl Iterator<Item = &'a [u8]>,
) -> Result<Vec<u8>, String> {
    if metadata.len() > EXPERT_LOAD_MAX_METADATA_BYTES {
        return Err("expert metadata exceeds its size limit".into());
    }
    let chunks: Vec<&[u8]> = payload.collect();
    let payload_len = chunks.iter().try_fold(0usize, |total, chunk| {
        total
            .checked_add(chunk.len())
            .ok_or_else(|| "expert payload length overflow".to_string())
    })?;
    let total_len = EXPERT_WIRE_HEADER_BYTES
        .checked_add(metadata.len())
        .and_then(|len| len.checked_add(payload_len))
        .ok_or_else(|| "expert frame length overflow".to_string())?;
    if total_len > EXPERT_LOAD_MAX_FRAME_BYTES {
        return Err(format!(
            "expert frame is {total_len} bytes; limit is {EXPERT_LOAD_MAX_FRAME_BYTES}"
        ));
    }
    let metadata_len =
        u32::try_from(metadata.len()).map_err(|_| "metadata too large".to_string())?;
    let mut output = Vec::with_capacity(total_len);
    output.extend_from_slice(&EXPERT_WIRE_MAGIC);
    output.push(EXPERT_WIRE_VERSION);
    output.push(kind);
    output.extend_from_slice(&metadata_len.to_le_bytes());
    output.extend_from_slice(metadata);
    for chunk in chunks {
        output.extend_from_slice(chunk);
    }
    Ok(output)
}

fn split_envelope(bytes: &[u8], expected_kind: u8) -> Result<(&[u8], &[u8]), String> {
    if bytes.len() > EXPERT_LOAD_MAX_FRAME_BYTES {
        return Err("expert frame exceeds its size limit".into());
    }
    if bytes.len() < EXPERT_WIRE_HEADER_BYTES || bytes[..4] != EXPERT_WIRE_MAGIC {
        return Err("invalid or truncated KDEX expert frame".into());
    }
    if bytes[4] != EXPERT_WIRE_VERSION {
        return Err(format!("unsupported expert protocol version {}", bytes[4]));
    }
    if bytes[5] != expected_kind {
        return Err(format!("unexpected expert frame kind {}", bytes[5]));
    }
    let metadata_len = u32::from_le_bytes(bytes[6..10].try_into().expect("four bytes")) as usize;
    if metadata_len > EXPERT_LOAD_MAX_METADATA_BYTES {
        return Err("expert metadata exceeds its size limit".into());
    }
    let payload_start = EXPERT_WIRE_HEADER_BYTES + metadata_len;
    if payload_start > bytes.len() {
        return Err("truncated expert metadata".into());
    }
    Ok((
        &bytes[EXPERT_WIRE_HEADER_BYTES..payload_start],
        &bytes[payload_start..],
    ))
}

pub(crate) struct Fnv1a64(u64);

impl Fnv1a64 {
    pub(crate) fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    pub(crate) fn update_u64(&mut self, value: u64) {
        self.update(&value.to_le_bytes());
    }

    pub(crate) fn finish(self) -> u64 {
        self.0
    }
}

fn packet_checksum(layer: u32, expert: u32, tensors: &[ExpertTensor]) -> u64 {
    let mut hash = Fnv1a64::new();
    hash.update(b"kadmium-expert-packet-v1");
    hash.update_u64(u64::from(layer));
    hash.update_u64(u64::from(expert));
    for tensor in tensors {
        hash.update_u64(tensor.name.len() as u64);
        hash.update(tensor.name.as_bytes());
        hash.update_u64(tensor.dtype.len() as u64);
        hash.update(tensor.dtype.as_bytes());
        hash.update_u64(tensor.shape.len() as u64);
        for dimension in &tensor.shape {
            hash.update_u64(*dimension);
        }
        hash.update_u64(tensor.bytes.len() as u64);
        hash.update(&tensor.bytes);
    }
    hash.finish()
}

pub trait ExpertSource: Send + Sync {
    fn model_id(&self) -> &str;
    /// Durable checkpoint identity supplied by the model/catalog owner, not a layout hash.
    fn source_fingerprint(&self) -> u64;
    fn expert_payload_bytes(&self, layer: u32, expert: u32) -> Result<u64, String>;
    fn load_expert(&self, layer: u32, expert: u32) -> Result<ExpertPacket, String>;

    /// Detects ordinary length/mtime changes to an immutable on-disk checkpoint before cache use.
    fn validate_generation(&self) -> Result<(), String> {
        Ok(())
    }
}

/// A persistent file and the generation metadata captured when its source was opened.
#[derive(Debug)]
pub(crate) struct SourceFile {
    pub(crate) file: File,
    path: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
}

impl SourceFile {
    pub(crate) fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .map_err(|e| format!("failed to open source `{}`: {e}", path.display()))?;
        let metadata = file
            .metadata()
            .map_err(|e| format!("failed to stat source `{}`: {e}", path.display()))?;
        Ok(Self {
            file,
            path,
            len: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    pub(crate) fn validate_generation(&self) -> Result<(), String> {
        let metadata = self.file.metadata().map_err(|e| {
            format!(
                "failed to validate source generation for `{}`: {e}",
                self.path.display()
            )
        })?;
        if metadata.len() != self.len || metadata.modified().ok() != self.modified {
            return Err(format!(
                "checkpoint source `{}` changed after it was opened; rebuild the expert source and cache",
                self.path.display()
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct SafeTensorExpertEntry {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    byte_len: u64,
    absolute_offset: u64,
    shard_index: usize,
}

/// A SafeTensors source with immutable expert indexes and persistent shard handles.
#[derive(Debug)]
pub struct SafeTensorsExpertSource {
    model_id: String,
    source_fingerprint: u64,
    files: Vec<SourceFile>,
    experts: BTreeMap<(u32, u32), Vec<SafeTensorExpertEntry>>,
}

impl SafeTensorsExpertSource {
    /// Opens an immutable checkpoint with its catalog-issued durable content fingerprint.
    pub fn open(
        model_id: impl Into<String>,
        model_dir: impl AsRef<Path>,
        source_fingerprint: u64,
    ) -> Result<Self, String> {
        let model_id = model_id.into();
        validate_source_identity(&model_id, source_fingerprint)?;
        let manifest = SafetensorsManifest::open(model_dir).map_err(manifest_error)?;
        let files = manifest
            .shards()
            .iter()
            .map(|shard| {
                let file = SourceFile::open(&shard.path)?;
                if file.len() != shard.file_size_bytes {
                    return Err(format!(
                        "checkpoint shard `{}` changed while its expert source was being opened",
                        shard.path.display()
                    ));
                }
                Ok(file)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut experts: BTreeMap<(u32, u32), Vec<SafeTensorExpertEntry>> = BTreeMap::new();
        for tensor in manifest.tensors() {
            let (layer, expert) = SafetensorsToKtbTranscoder::parse_layer_and_expert(&tensor.name);
            let Some(expert) = expert else {
                continue;
            };
            let entries = experts.entry((layer, expert)).or_default();
            if entries.iter().any(|entry| entry.name == tensor.name) {
                return Err(format!("duplicate expert tensor `{}`", tensor.name));
            }
            entries.push(SafeTensorExpertEntry {
                name: tensor.name.clone(),
                dtype: tensor.dtype.as_str().to_string(),
                shape: tensor.shape.clone(),
                byte_len: tensor.byte_len,
                absolute_offset: tensor.absolute_offset,
                shard_index: tensor.shard_index,
            });
        }
        Ok(Self {
            model_id,
            source_fingerprint,
            files,
            experts,
        })
    }
}

impl ExpertSource for SafeTensorsExpertSource {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn source_fingerprint(&self) -> u64 {
        self.source_fingerprint
    }

    fn expert_payload_bytes(&self, layer: u32, expert: u32) -> Result<u64, String> {
        let entries = self.experts.get(&(layer, expert)).ok_or_else(|| {
            format!(
                "expert {layer}/{expert} was not found in model `{}`",
                self.model_id
            )
        })?;
        entries.iter().try_fold(0u64, |total, entry| {
            total
                .checked_add(entry.byte_len)
                .ok_or_else(|| format!("expert {layer}/{expert} payload length overflow"))
        })
    }

    fn load_expert(&self, layer: u32, expert: u32) -> Result<ExpertPacket, String> {
        let entries = self.experts.get(&(layer, expert)).ok_or_else(|| {
            format!(
                "expert {layer}/{expert} was not found in model `{}`",
                self.model_id
            )
        })?;
        let mut tensors = Vec::with_capacity(entries.len());
        for entry in entries {
            let file = self
                .files
                .get(entry.shard_index)
                .ok_or_else(|| format!("invalid shard index for tensor `{}`", entry.name))?;
            let mut bytes = allocate_bytes(entry.byte_len, &entry.name)?;
            read_exact_at(&file.file, entry.absolute_offset, &mut bytes)
                .map_err(|e| format!("failed to read tensor `{}`: {e}", entry.name))?;
            tensors.push(ExpertTensor {
                name: entry.name.clone(),
                dtype: entry.dtype.clone(),
                shape: entry.shape.clone(),
                bytes: Arc::from(bytes),
            });
        }
        ExpertPacket::new(layer, expert, tensors)
    }

    fn validate_generation(&self) -> Result<(), String> {
        for file in &self.files {
            file.validate_generation()?;
        }
        Ok(())
    }
}

fn manifest_error(error: SafetensorsManifestError) -> String {
    error.to_string()
}

pub(crate) fn allocate_bytes(len: u64, what: &str) -> Result<Vec<u8>, String> {
    let len = usize::try_from(len)
        .map_err(|_| format!("{what} byte length does not fit this platform"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| format!("cannot allocate {len} bytes for {what}"))?;
    bytes.resize(len, 0);
    Ok(bytes)
}

#[cfg(unix)]
pub(crate) fn read_exact_at(file: &File, mut offset: u64, mut output: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !output.is_empty() {
        let read = match file.read_at(output, offset) {
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short positional read",
            ));
        }
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "file offset overflow"))?;
        output = &mut output[read..];
    }
    Ok(())
}

#[cfg(windows)]
pub(crate) fn read_exact_at(file: &File, mut offset: u64, mut output: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !output.is_empty() {
        let read = match file.seek_read(output, offset) {
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short positional read",
            ));
        }
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "file offset overflow"))?;
        output = &mut output[read..];
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertLoaderStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub source_bytes_read: u64,
    pub cache_evictions: u64,
    pub cached_packets: usize,
    pub cached_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ExpertCacheKey {
    model_id: String,
    source_fingerprint: u64,
    layer: u32,
    expert: u32,
}

struct ExpertCacheEntry {
    packet: Arc<ExpertPacket>,
    payload_bytes: usize,
    last_used: u64,
}

#[derive(Default)]
struct ExpertLoaderState {
    entries: BTreeMap<ExpertCacheKey, ExpertCacheEntry>,
    cached_bytes: usize,
    clock: u64,
    cache_hits: u64,
    cache_misses: u64,
    source_bytes_read: u64,
    cache_evictions: u64,
}

struct ExpertLoaderInner {
    sources: RwLock<BTreeMap<String, Arc<dyn ExpertSource>>>,
    state: Mutex<ExpertLoaderState>,
    active_loads: Mutex<usize>,
    load_cv: Condvar,
    max_cache_bytes: usize,
}

struct ExpertLoadPermit<'a> {
    inner: &'a ExpertLoaderInner,
}

impl Drop for ExpertLoadPermit<'_> {
    fn drop(&mut self) {
        let mut active = self
            .inner
            .active_loads
            .lock()
            .expect("expert load limit lock");
        *active = active.saturating_sub(1);
        drop(active);
        self.inner.load_cv.notify_one();
    }
}

/// Worker-local expert sources and their bounded, checkpoint-aware read-only cache.
#[derive(Clone)]
pub struct ExpertLoader {
    inner: Arc<ExpertLoaderInner>,
}

impl Default for ExpertLoader {
    fn default() -> Self {
        Self::new(DEFAULT_EXPERT_CACHE_BYTES)
    }
}

impl ExpertLoader {
    pub fn new(max_cache_bytes: usize) -> Self {
        Self {
            inner: Arc::new(ExpertLoaderInner {
                sources: RwLock::new(BTreeMap::new()),
                state: Mutex::new(ExpertLoaderState::default()),
                active_loads: Mutex::new(0),
                load_cv: Condvar::new(),
                max_cache_bytes,
            }),
        }
    }

    fn acquire_permit(&self) -> ExpertLoadPermit<'_> {
        let mut active = self
            .inner
            .active_loads
            .lock()
            .expect("expert load limit lock");
        while *active >= MAX_CONCURRENT_EXPERT_LOADS {
            active = self
                .inner
                .load_cv
                .wait(active)
                .expect("expert load limit lock");
        }
        *active += 1;
        ExpertLoadPermit { inner: &self.inner }
    }

    pub fn register_source(&self, source: Arc<dyn ExpertSource>) -> Result<(), String> {
        validate_source_identity(source.model_id(), source.source_fingerprint())?;
        let mut sources = self
            .inner
            .sources
            .write()
            .expect("expert source registry lock");
        if sources.contains_key(source.model_id()) {
            return Err(format!(
                "an expert source is already registered for `{}`",
                source.model_id()
            ));
        }
        sources.insert(source.model_id().to_string(), source);
        Ok(())
    }

    /// Replaces a model source after a checkpoint rotation and invalidates its old cache entries.
    pub fn replace_source(&self, source: Arc<dyn ExpertSource>) -> Result<(), String> {
        validate_source_identity(source.model_id(), source.source_fingerprint())?;
        let model_id = source.model_id().to_string();
        self.inner
            .sources
            .write()
            .expect("expert source registry lock")
            .insert(model_id.clone(), source);
        self.invalidate_model(&model_id);
        Ok(())
    }

    fn invalidate_model(&self, model_id: &str) {
        let mut state = self.inner.state.lock().expect("expert cache lock");
        state.entries.retain(|key, _| key.model_id != model_id);
        state.cached_bytes = state
            .entries
            .values()
            .map(|entry| entry.payload_bytes)
            .sum();
    }

    pub fn contains_model(&self, model_id: &str) -> bool {
        self.inner
            .sources
            .read()
            .expect("expert source registry lock")
            .contains_key(model_id)
    }

    pub fn source_fingerprint(&self, model_id: &str) -> Option<u64> {
        self.source(model_id)
            .map(|source| source.source_fingerprint())
    }

    fn source(&self, model_id: &str) -> Option<Arc<dyn ExpertSource>> {
        self.inner
            .sources
            .read()
            .expect("expert source registry lock")
            .get(model_id)
            .cloned()
    }

    pub fn stats(&self) -> ExpertLoaderStats {
        let state = self.inner.state.lock().expect("expert cache lock");
        ExpertLoaderStats {
            cache_hits: state.cache_hits,
            cache_misses: state.cache_misses,
            source_bytes_read: state.source_bytes_read,
            cache_evictions: state.cache_evictions,
            cached_packets: state.entries.len(),
            cached_bytes: state.cached_bytes,
        }
    }

    pub fn load_model_experts(
        &self,
        model_id: &str,
        experts: Vec<ExpertKey>,
    ) -> Result<ExpertLoadResponse, String> {
        let source_fingerprint = self
            .source_fingerprint(model_id)
            .ok_or_else(|| format!("no expert source registered for `{model_id}`"))?;
        self.load_request(&ExpertLoadRequest::new(
            model_id,
            source_fingerprint,
            experts,
        ))
    }

    pub fn load_request(&self, request: &ExpertLoadRequest) -> Result<ExpertLoadResponse, String> {
        request.validate()?;
        let source = self
            .source(&request.model_id)
            .ok_or_else(|| format!("no expert source registered for `{}`", request.model_id))?;
        if source.source_fingerprint() != request.source_fingerprint {
            return Err(format!(
                "checkpoint fingerprint mismatch for `{}`",
                request.model_id
            ));
        }

        let _permit = self.acquire_permit();
        if let Err(error) = source.validate_generation() {
            self.invalidate_model(&request.model_id);
            return Err(error);
        }
        let mut expected_payload_bytes = BTreeMap::new();
        let mut total_payload_bytes = 0u64;
        for key in &request.experts {
            let bytes = source.expert_payload_bytes(key.layer, key.expert)?;
            total_payload_bytes = total_payload_bytes
                .checked_add(bytes)
                .ok_or_else(|| "expert batch payload length overflow".to_string())?;
            if total_payload_bytes > EXPERT_LOAD_MAX_PAYLOAD_BYTES as u64 {
                return Err(format!(
                    "expert batch payload is {total_payload_bytes} bytes; limit is {EXPERT_LOAD_MAX_PAYLOAD_BYTES}; split the request"
                ));
            }
            expected_payload_bytes.insert(*key, bytes);
        }

        let mut packets = Vec::with_capacity(request.experts.len());
        for key in &request.experts {
            let cache_key = ExpertCacheKey {
                model_id: request.model_id.clone(),
                source_fingerprint: request.source_fingerprint,
                layer: key.layer,
                expert: key.expert,
            };
            if let Some(packet) = self.cache_get(&cache_key) {
                packets.push((*packet).clone());
                continue;
            }
            self.inner
                .state
                .lock()
                .expect("expert cache lock")
                .cache_misses += 1;
            let packet = source.load_expert(key.layer, key.expert)?;
            packet.validate()?;
            if packet.key() != *key {
                return Err(format!(
                    "source returned expert {:?} while {:?} was requested",
                    packet.key(),
                    key
                ));
            }
            let payload_bytes = packet.payload_bytes();
            if payload_bytes as u64 != expected_payload_bytes[key] {
                return Err(format!(
                    "source returned {payload_bytes} bytes for expert {key:?}, after advertising {}",
                    expected_payload_bytes[key]
                ));
            }
            {
                let mut state = self.inner.state.lock().expect("expert cache lock");
                state.source_bytes_read =
                    state.source_bytes_read.saturating_add(payload_bytes as u64);
            }
            let packet = Arc::new(packet);
            self.cache_insert(cache_key, Arc::clone(&packet));
            packets.push((*packet).clone());
        }
        if let Err(error) = source.validate_generation() {
            self.invalidate_model(&request.model_id);
            return Err(error);
        }
        Ok(ExpertLoadResponse::success(
            &request.model_id,
            request.source_fingerprint,
            packets,
        ))
    }

    fn cache_get(&self, key: &ExpertCacheKey) -> Option<Arc<ExpertPacket>> {
        let mut state = self.inner.state.lock().expect("expert cache lock");
        state.clock = state.clock.wrapping_add(1);
        let clock = state.clock;
        let packet = state.entries.get_mut(key).map(|entry| {
            entry.last_used = clock;
            Arc::clone(&entry.packet)
        });
        if packet.is_some() {
            state.cache_hits += 1;
        }
        packet
    }

    fn cache_insert(&self, key: ExpertCacheKey, packet: Arc<ExpertPacket>) {
        let payload_bytes = packet.payload_bytes();
        if payload_bytes > self.inner.max_cache_bytes {
            return;
        }
        let mut state = self.inner.state.lock().expect("expert cache lock");
        if let Some(previous) = state.entries.remove(&key) {
            state.cached_bytes = state.cached_bytes.saturating_sub(previous.payload_bytes);
        }
        while state.cached_bytes.saturating_add(payload_bytes) > self.inner.max_cache_bytes {
            let Some(oldest) = state
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(evicted) = state.entries.remove(&oldest) {
                state.cached_bytes = state.cached_bytes.saturating_sub(evicted.payload_bytes);
                state.cache_evictions += 1;
            }
        }
        state.clock = state.clock.wrapping_add(1);
        let clock = state.clock;
        state.cached_bytes = state.cached_bytes.saturating_add(payload_bytes);
        state.entries.insert(
            key,
            ExpertCacheEntry {
                packet,
                payload_bytes,
                last_used: clock,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors_manifest::SAFETENSORS_SINGLE_FILE_NAME;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos();
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kadmium-expert-{}-{nonce}-{id}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create test directory");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove test directory");
        }
    }

    const FINGERPRINT: u64 = 0x5eed_c0de;
    const UP: &str = "model.layers.2.mlp.experts.7.up_proj.weight";
    const GATE: &str = "model.layers.2.mlp.experts.7.gate_proj.weight";
    const DOWN: &str = "model.layers.2.mlp.experts.7.down_proj.weight";
    const OTHER: &str = "model.layers.3.mlp.experts.8.up_proj.weight";

    fn model() -> (TestDir, Arc<SafeTensorsExpertSource>) {
        let dir = TestDir::new();
        let bytes = SafetensorsToKtbTranscoder::build_safetensors_bytes(&[
            (UP, "BF16", vec![2], &[0, 255, 1, 2]),
            (GATE, "BF16", vec![1], &[3, 4]),
            (DOWN, "BF16", vec![2], &[5, 6, 7, 8]),
            (OTHER, "BF16", vec![1], &[9, 10]),
        ]);
        fs::write(dir.0.join(SAFETENSORS_SINGLE_FILE_NAME), bytes).expect("write model");
        let source = Arc::new(
            SafeTensorsExpertSource::open("test-model", &dir.0, FINGERPRINT)
                .expect("open expert source"),
        );
        (dir, source)
    }

    #[test]
    fn safetensors_source_reads_all_selected_expert_tensors_exactly() {
        let (_dir, source) = model();
        let packet = source.load_expert(2, 7).expect("load expert");
        assert_eq!(packet.payload_bytes(), 10);
        assert_eq!(packet.tensors.len(), 3);
        let up = packet
            .tensors
            .iter()
            .find(|tensor| tensor.name == UP)
            .expect("up tensor");
        assert_eq!(up.bytes.as_ref(), &[0, 255, 1, 2]);
        assert_eq!(up.dtype, "BF16");
        assert_eq!(up.shape, vec![2]);
        assert!(source.load_expert(2, 99).is_err());
    }

    #[test]
    fn loader_caches_hits_and_evicts_within_its_byte_budget() {
        let (_dir, source) = model();
        let loader = ExpertLoader::new(10);
        loader.register_source(source).expect("register source");
        let keys = vec![ExpertKey::new(2, 7)];
        let first = loader
            .load_model_experts("test-model", keys.clone())
            .expect("first load");
        let second = loader
            .load_model_experts("test-model", keys)
            .expect("cached load");
        assert_eq!(first, second);
        let stats = loader.stats();
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.source_bytes_read, 10);

        loader
            .load_model_experts("test-model", vec![ExpertKey::new(3, 8)])
            .expect("load second expert");
        let stats = loader.stats();
        assert_eq!(stats.cached_packets, 1);
        assert_eq!(stats.cached_bytes, 2);
        assert_eq!(stats.cache_evictions, 1);
    }

    #[test]
    fn binary_wire_roundtrip_preserves_raw_payload_and_detects_damage() {
        let (_dir, source) = model();
        let fingerprint = source.source_fingerprint();
        let request = ExpertLoadRequest::new(
            "test-model",
            fingerprint,
            vec![ExpertKey::new(2, 7), ExpertKey::new(3, 8)],
        );
        let encoded_request = request.encode().expect("encode request");
        assert_eq!(
            ExpertLoadRequest::decode(&encoded_request),
            Ok(request.clone())
        );

        let loader = ExpertLoader::default();
        loader.register_source(source).expect("register source");
        let response = loader.load_request(&request).expect("load response");
        let encoded = response.encode().expect("encode response");
        assert!(encoded.ends_with(&[9, 10]));
        let decoded = ExpertLoadResponse::decode(&encoded).expect("decode response");
        assert_eq!(decoded, response);
        decoded
            .validate_for_request(&request)
            .expect("match request");

        let mut damaged = encoded.clone();
        *damaged.last_mut().expect("payload") ^= 1;
        assert!(
            ExpertLoadResponse::decode(&damaged)
                .expect_err("checksum must fail")
                .contains("checksum mismatch")
        );
        assert!(ExpertLoadResponse::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn request_validation_fingerprints_and_error_envelopes_are_explicit() {
        let (_dir, source) = model();
        let fingerprint = source.source_fingerprint();
        let loader = ExpertLoader::default();
        loader.register_source(source).expect("register source");
        let wrong = ExpertLoadRequest::new(
            "test-model",
            fingerprint.wrapping_add(1),
            vec![ExpertKey::new(2, 7)],
        );
        assert!(
            loader
                .load_request(&wrong)
                .expect_err("fingerprint must match")
                .contains("fingerprint mismatch")
        );

        let duplicate = ExpertLoadRequest::new(
            "test-model",
            fingerprint,
            vec![ExpertKey::new(2, 7), ExpertKey::new(2, 7)],
        );
        assert!(duplicate.encode().is_err());
        assert!(
            loader
                .load_model_experts("missing", vec![ExpertKey::new(0, 0)])
                .is_err()
        );

        let error = ExpertLoadResponse::error("test-model", fingerprint, "missing expert");
        let encoded = error.encode().expect("encode error");
        assert_eq!(ExpertLoadResponse::decode(&encoded), Ok(error));
        assert!(
            ExpertLoadRequest::new("test-model", 0, vec![ExpertKey::new(2, 7)])
                .encode()
                .is_err()
        );
    }

    #[test]
    fn changed_source_generation_invalidates_cached_checkpoint_reads() {
        use std::io::{Seek, SeekFrom, Write};

        let (_dir, source) = model();
        let loader = ExpertLoader::default();
        loader.register_source(source).expect("register source");
        loader
            .load_model_experts("test-model", vec![ExpertKey::new(2, 7)])
            .expect("warm cache");
        let path = _dir.0.join(SAFETENSORS_SINGLE_FILE_NAME);
        let bytes = fs::read(&path).expect("read test model");
        let offset = bytes
            .windows(4)
            .position(|window| window == [0, 255, 1, 2])
            .expect("find up payload");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open model for mutation");
        file.seek(SeekFrom::Start(offset as u64))
            .expect("seek payload");
        file.write_all(&[42, 255, 1, 2]).expect("mutate payload");
        file.set_modified(SystemTime::now() + std::time::Duration::from_secs(5))
            .expect("advance modification time");
        file.sync_all().expect("flush mutation");
        drop(file);

        let error = loader
            .load_model_experts(
                "test-model",
                vec![ExpertKey::new(2, 7), ExpertKey::new(3, 8)],
            )
            .expect_err("changed checkpoint must not mix old cache and new disk bytes");
        assert!(error.contains("changed after it was opened"));
        assert_eq!(loader.stats().cached_packets, 0);

        let replacement = Arc::new(
            SafeTensorsExpertSource::open("test-model", &_dir.0, FINGERPRINT + 1)
                .expect("open replacement source"),
        );
        loader
            .replace_source(replacement)
            .expect("replace changed source");
        let response = loader
            .load_model_experts("test-model", vec![ExpertKey::new(2, 7)])
            .expect("load from replacement");
        let up = response.experts[0]
            .tensors
            .iter()
            .find(|tensor| tensor.name == UP)
            .expect("up tensor");
        assert_eq!(up.bytes.as_ref(), &[42, 255, 1, 2]);
    }

    #[test]
    fn batch_budget_is_checked_before_any_source_read() {
        struct Oversized;
        impl ExpertSource for Oversized {
            fn model_id(&self) -> &str {
                "oversized"
            }
            fn source_fingerprint(&self) -> u64 {
                1
            }
            fn expert_payload_bytes(&self, _layer: u32, _expert: u32) -> Result<u64, String> {
                Ok(EXPERT_LOAD_MAX_PAYLOAD_BYTES as u64 / 2 + 1)
            }
            fn load_expert(&self, _layer: u32, _expert: u32) -> Result<ExpertPacket, String> {
                panic!("an over-budget batch must not reach the source")
            }
        }

        let loader = ExpertLoader::default();
        loader
            .register_source(Arc::new(Oversized))
            .expect("register");
        let error = loader
            .load_model_experts(
                "oversized",
                vec![ExpertKey::new(0, 0), ExpertKey::new(0, 1)],
            )
            .expect_err("batch must be split");
        assert!(error.contains("split the request"));
        assert_eq!(loader.stats().cache_misses, 0);
        assert_eq!(loader.stats().source_bytes_read, 0);
    }

    #[test]
    fn malformed_custom_source_packets_are_rejected_at_the_loader_boundary() {
        struct Malformed;
        impl ExpertSource for Malformed {
            fn model_id(&self) -> &str {
                "malformed"
            }
            fn source_fingerprint(&self) -> u64 {
                1
            }
            fn expert_payload_bytes(&self, _layer: u32, _expert: u32) -> Result<u64, String> {
                Ok(0)
            }
            fn load_expert(&self, layer: u32, expert: u32) -> Result<ExpertPacket, String> {
                Ok(ExpertPacket {
                    layer,
                    expert,
                    checksum_fnv64: 0,
                    tensors: Vec::new(),
                })
            }
        }

        let loader = ExpertLoader::default();
        loader
            .register_source(Arc::new(Malformed))
            .expect("register");
        assert!(
            loader
                .load_model_experts("malformed", vec![ExpertKey::new(0, 0)])
                .expect_err("empty packet must fail")
                .contains("must contain")
        );
    }
}
