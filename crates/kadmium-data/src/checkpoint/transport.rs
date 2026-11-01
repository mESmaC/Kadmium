//! Transport normalization for checkpoint packages.
//!
//! Follows InferSSD's streaming layout rules: every streamed extent starts and ends on a
//! 4096-byte boundary (direct I/O on Linux and Windows), padding is zero-filled, expert slices
//! are read expert-major — (layer, expert, role, name) — copies move in 16 MiB chunks, and
//! packages publish atomically. InferSSD carries no payload checksums, so Kadmium adds a SHA-256
//! per chunk and per file: a transfer can resume, skip chunks the receiver already holds, and
//! verify every byte before the package becomes visible. Nothing here rewrites tensor bytes.

use super::{
    CheckpointLibrary, CheckpointManifest, PAGE, PackageFile, Result, io_error, package_id,
    read_json, simple_name,
};
use crate::{GGML_MOE_PACK_ALIGNMENT, SafetensorsToKtbTranscoder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const TRANSPORT_SCHEMA: &str = "kadmium.transport.v1";
/// InferSSD's direct-I/O boundary for offsets, lengths and buffers.
pub const TRANSPORT_ALIGNMENT_BYTES: u64 = GGML_MOE_PACK_ALIGNMENT;
/// InferSSD's copy chunk; also the unit of resumable, verified transfer.
pub const TRANSPORT_CHUNK_BYTES: u64 = 16 * 1024 * 1024;
const _: () = assert!(PAGE.is_multiple_of(TRANSPORT_ALIGNMENT_BYTES));
const _: () = assert!(TRANSPORT_CHUNK_BYTES.is_multiple_of(PAGE));
const TRANSPORT_FILE: &str = "transport.json";
const RECEIVING: &str = ".receiving";
/// Packages `receive_finish` has taken out of staging to verify and publish.
const VERIFYING: &str = ".verifying";
const MAX_FILES: usize = 100_000;
/// Abandoned transfers older than this are removed when a new transfer starts.
const STALE_STAGING: std::time::Duration = std::time::Duration::from_secs(7 * 24 * 60 * 60);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportChunk {
    pub offset: u64,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportFile {
    /// `manifest.json`, `source/<name>`, or `banks/<name>`.
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub chunks: Vec<TransportChunk>,
}

/// One tensor's page-aligned location inside a KTB bank, listed in streaming order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportExtent {
    pub tensor: String,
    pub layer: u32,
    pub expert: Option<u32>,
    pub bank: String,
    pub offset: u64,
    pub bytes: u64,
    pub reserved_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportManifest {
    pub schema: String,
    pub checkpoint_id: String,
    pub name: String,
    pub alignment_bytes: u64,
    pub page_bytes: u64,
    pub chunk_bytes: u64,
    pub total_bytes: u64,
    pub files: Vec<TransportFile>,
    pub read_order: Vec<TransportExtent>,
    pub quality_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportSummary {
    pub checkpoint_id: String,
    pub name: String,
    pub schema: String,
    pub alignment_bytes: u64,
    pub page_bytes: u64,
    pub chunk_bytes: u64,
    pub total_bytes: u64,
    pub file_count: usize,
    pub chunk_count: usize,
    pub extent_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRef {
    pub file: u32,
    pub chunk: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiveStatus {
    pub checkpoint_id: String,
    pub complete: bool,
    pub total_bytes: u64,
    pub received_bytes: u64,
    pub missing: Vec<ChunkRef>,
}

impl TransportManifest {
    pub fn summary(&self) -> TransportSummary {
        TransportSummary {
            checkpoint_id: self.checkpoint_id.clone(),
            name: self.name.clone(),
            schema: self.schema.clone(),
            alignment_bytes: self.alignment_bytes,
            page_bytes: self.page_bytes,
            chunk_bytes: self.chunk_bytes,
            total_bytes: self.total_bytes,
            file_count: self.files.len(),
            chunk_count: self.files.iter().map(|f| f.chunks.len()).sum(),
            extent_count: self.read_order.len(),
        }
    }

    /// Rejects anything a receiver should not stage: unknown layouts, unsafe paths, chunk maps
    /// that do not tile each file, misaligned extents, or an identity that does not match the
    /// content-addressed hash of the source files.
    pub fn validate(&self) -> Result<()> {
        if self.schema != TRANSPORT_SCHEMA
            || self.alignment_bytes != TRANSPORT_ALIGNMENT_BYTES
            || self.page_bytes != PAGE
            || self.chunk_bytes != TRANSPORT_CHUNK_BYTES
            || self.quality_policy != "preserve_source"
        {
            return Err("Unsupported transport layout".into());
        }
        if self.files.len() < 2 || self.files.len() > MAX_FILES {
            return Err("Transport must list a manifest and its model files".into());
        }
        if self.files[0].path != "manifest.json" {
            return Err("Transport must start with manifest.json".into());
        }
        let mut paths = BTreeSet::new();
        let mut total = 0u64;
        let mut sources = Vec::new();
        for file in &self.files {
            if !paths.insert(file.path.as_str()) {
                return Err(format!("Duplicate transport file {}", file.path));
            }
            match file.path.split_once('/') {
                None if file.path == "manifest.json" => {}
                Some(("source", name)) => {
                    simple_name(name)?;
                    sources.push(PackageFile {
                        name: name.into(),
                        bytes: file.bytes,
                        sha256: file.sha256.clone(),
                    });
                }
                Some(("banks", name)) => simple_name(name)?,
                _ => return Err(format!("Invalid transport path {:?}", file.path)),
            }
            hex_digest(&file.sha256)?;
            let expected = file.bytes.div_ceil(self.chunk_bytes);
            if file.chunks.len() as u64 != expected {
                return Err(format!("Chunk map for {} is incomplete", file.path));
            }
            let plan = chunk_plan(file.bytes, self.chunk_bytes);
            for (chunk, (offset, bytes)) in file.chunks.iter().zip(plan) {
                if chunk.offset != offset || chunk.bytes != bytes {
                    return Err(format!("Chunk map for {} is misaligned", file.path));
                }
                hex_digest(&chunk.sha256)?;
            }
            total = total
                .checked_add(file.bytes)
                .ok_or("Transport size overflow")?;
        }
        if total != self.total_bytes {
            return Err("Transport size does not match its files".into());
        }
        sources.sort_by(|a, b| a.name.cmp(&b.name));
        if sources.is_empty() || package_id(&sources)? != self.checkpoint_id {
            return Err("Transport identity does not match its source files".into());
        }
        for extent in &self.read_order {
            let bank = self
                .files
                .iter()
                .find(|f| f.path.strip_prefix("banks/") == Some(extent.bank.as_str()))
                .ok_or("Transport extent names an unknown bank")?;
            check_extent(extent)?;
            if extent.offset.saturating_add(extent.reserved_bytes) > bank.bytes {
                return Err("Transport extent exceeds its bank".into());
            }
        }
        Ok(())
    }
}

impl CheckpointLibrary {
    /// Synchronous library adapter for the existing resumable chunk protocol. The caller must
    /// serialize receives for this destination package and authorize its storage before copying.
    /// A failed copy removes incomplete staging; an already published package is never removed.
    pub fn copy_to(
        &self,
        transport: &TransportManifest,
        destination: &CheckpointLibrary,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<CheckpointManifest> {
        transport.validate()?;
        let id = &transport.checkpoint_id;
        if self.transport(id)? != *transport {
            return Err("The source checkpoint changed after its staging plan was prepared".into());
        }
        let mut incomplete = false;
        let outcome = (|| {
            super::selection::check_cancelled(cancelled)?;
            let status = destination.receive_begin(transport)?;
            incomplete = !status.complete;
            for missing in status.missing {
                super::selection::check_cancelled(cancelled)?;
                let file = transport
                    .files
                    .get(missing.file as usize)
                    .ok_or("Unknown transport file")?;
                let chunk = file
                    .chunks
                    .get(missing.chunk as usize)
                    .ok_or("Unknown transport chunk")?;
                let length = usize::try_from(chunk.bytes).map_err(|_| "Chunk size overflow")?;
                if chunk.bytes > TRANSPORT_CHUNK_BYTES {
                    return Err("Transport chunk exceeds the copy buffer".into());
                }
                let mut bytes = Vec::new();
                bytes.try_reserve_exact(length).map_err(io_error)?;
                bytes.resize(length, 0);
                let mut input =
                    File::open(self.package_dir(id)?.join(&file.path)).map_err(io_error)?;
                input
                    .seek(SeekFrom::Start(chunk.offset))
                    .map_err(io_error)?;
                input.read_exact(&mut bytes).map_err(io_error)?;
                destination.receive_chunk(id, missing.file, missing.chunk, &bytes)?;
            }
            super::selection::check_cancelled(cancelled)?;
            destination.receive_finish(id)?;
            let manifest = destination.verify(id)?;
            if destination.transport(id)? != *transport {
                return Err("Received checkpoint does not match the requested transport".into());
            }
            Ok(manifest)
        })();
        if outcome.is_err() && incomplete {
            let staging = destination.receiving_dir(id)?;
            if staging.exists() {
                fs::remove_dir_all(staging).map_err(io_error)?;
            }
        }
        outcome
    }

    /// Returns the package's transport manifest, building and caching it from the package
    /// bytes on first use. Building re-reads every file and fails on any hash mismatch.
    pub fn transport(&self, id: &str) -> Result<TransportManifest> {
        let dir = self.package_dir(id)?;
        let manifest = self.read(id)?;
        let manifest_hash = hash_bytes(&fs::read(dir.join("manifest.json")).map_err(io_error)?);
        if let Ok(cached) = read_json::<TransportManifest>(&dir.join(TRANSPORT_FILE))
            && cached.validate().is_ok()
            && describes(&cached, &manifest, &manifest_hash)
        {
            return Ok(cached);
        }
        let transport = build(&dir, &manifest)?;
        let temporary = dir.join(format!(".{TRANSPORT_FILE}.{}", std::process::id()));
        let mut file = File::create(&temporary).map_err(io_error)?;
        file.write_all(&serde_json::to_vec_pretty(&transport).map_err(io_error)?)
            .map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
        drop(file);
        fs::rename(&temporary, dir.join(TRANSPORT_FILE)).map_err(io_error)?;
        Ok(transport)
    }

    /// Stages an incoming package and reports the chunks still needed. Calling it again with
    /// the same manifest resumes; a published package needs nothing.
    pub fn receive_begin(&self, transport: &TransportManifest) -> Result<ReceiveStatus> {
        transport.validate()?;
        let id = &transport.checkpoint_id;
        if self.read(id).is_ok() {
            return Ok(ReceiveStatus {
                checkpoint_id: id.clone(),
                complete: true,
                total_bytes: transport.total_bytes,
                received_bytes: transport.total_bytes,
                missing: Vec::new(),
            });
        }
        self.remove_stale_transfers(id);
        let staging = self.receiving_dir(id)?;
        let staged = read_json::<TransportManifest>(&staging.join(TRANSPORT_FILE));
        if staged.as_ref().is_ok_and(|staged| staged != transport) || staged.is_err() {
            if staging.exists() {
                fs::remove_dir_all(&staging).map_err(io_error)?;
            }
            // Files are created as verified chunks arrive; nothing is pre-allocated from sizes
            // the sender merely declares.
            for folder in ["source", "banks", ".chunks"] {
                fs::create_dir_all(staging.join(folder)).map_err(io_error)?;
            }
            fs::write(
                staging.join(TRANSPORT_FILE),
                serde_json::to_vec_pretty(transport).map_err(io_error)?,
            )
            .map_err(io_error)?;
        }
        receive_status(&staging, transport)
    }

    /// Writes one verified chunk into the staged package.
    pub fn receive_chunk(&self, id: &str, file: u32, chunk: u32, data: &[u8]) -> Result<()> {
        let staging = self.receiving_dir(id)?;
        let transport: TransportManifest = read_json(&staging.join(TRANSPORT_FILE))
            .map_err(|_| "Start the transfer before sending chunks".to_string())?;
        let entry = transport
            .files
            .get(file as usize)
            .ok_or("Unknown transport file")?;
        let expected = entry
            .chunks
            .get(chunk as usize)
            .ok_or("Unknown transport chunk")?;
        if data.len() as u64 != expected.bytes || hash_bytes(data) != expected.sha256 {
            return Err(format!(
                "Chunk {chunk} of {} failed verification; resend it",
                entry.path
            ));
        }
        let mut output = File::options()
            .write(true)
            .create(true)
            .truncate(false)
            .open(staging.join(&entry.path))
            .map_err(io_error)?;
        output
            .seek(SeekFrom::Start(expected.offset))
            .map_err(io_error)?;
        output.write_all(data).map_err(io_error)?;
        output.sync_data().map_err(io_error)?;
        File::create(staging.join(".chunks").join(marker(file, chunk))).map_err(io_error)?;
        Ok(())
    }

    /// Verifies the complete staged package and publishes it atomically into the library.
    ///
    /// The staged directory is first renamed into a private `.verifying/<claim>/<id>` directory,
    /// so a concurrent `receive_begin` (which may wipe and recreate staging) can never change
    /// what is verified or published: the bytes that pass verification are the bytes renamed
    /// into the library. An incomplete claim goes back to staging for resume; a package that
    /// fails verification is discarded so the next transfer starts clean.
    pub fn receive_finish(&self, id: &str) -> Result<CheckpointManifest> {
        self.finish_claimed(id, &mut || {})
    }

    fn finish_claimed(
        &self,
        id: &str,
        after_claim: &mut dyn FnMut(),
    ) -> Result<CheckpointManifest> {
        if let Ok(manifest) = self.read(id) {
            return Ok(manifest);
        }
        let staging = self.receiving_dir(id)?;
        if !staging.join(TRANSPORT_FILE).exists() {
            return Err("No transfer is staged for this checkpoint".into());
        }
        static CLAIMS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let claim = self.root().join(VERIFYING).join(format!(
            "{}-{}",
            std::process::id(),
            CLAIMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(&claim).map_err(io_error)?;
        let private = claim.join(id);
        if fs::rename(&staging, &private).is_err() {
            let _ = fs::remove_dir_all(&claim);
            return Err("No transfer is staged for this checkpoint".into());
        }
        after_claim();
        let outcome = verify_claimed(&claim, &private, id);
        match outcome {
            Ok(manifest) => {
                let target = self.package_dir(id)?;
                let published = match fs::rename(&private, &target) {
                    Ok(()) => Ok(manifest),
                    Err(_) if target.exists() => self.read(id),
                    Err(error) => Err(io_error(error)),
                };
                let _ = fs::remove_dir_all(&claim);
                published
            }
            Err(Unfinished::Incomplete(message)) => {
                // Return the partial transfer for resume, unless a new one has started.
                if fs::rename(&private, &staging).is_err() {
                    let _ = fs::remove_dir_all(&claim);
                } else {
                    let _ = fs::remove_dir(&claim);
                }
                Err(message)
            }
            Err(Unfinished::Invalid(message)) => {
                let _ = fs::remove_dir_all(&claim);
                Err(format!(
                    "{message}. The received copy was discarded; send the model again"
                ))
            }
        }
    }

    fn remove_stale_transfers(&self, keep: &str) {
        // A claim outlives `receive_finish` only if the process stopped mid-verification.
        if let Ok(claims) = fs::read_dir(self.root().join(VERIFYING)) {
            for claim in claims.flatten() {
                let stale = claim
                    .metadata()
                    .and_then(|m| m.modified())
                    .is_ok_and(|modified| modified.elapsed().is_ok_and(|age| age > STALE_STAGING));
                if stale {
                    let _ = fs::remove_dir_all(claim.path());
                }
            }
        }
        let Ok(entries) = fs::read_dir(self.root().join(RECEIVING)) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let stale = fs::metadata(path.join(TRANSPORT_FILE))
                .or_else(|_| entry.metadata())
                .and_then(|m| m.modified())
                .is_ok_and(|modified| modified.elapsed().is_ok_and(|age| age > STALE_STAGING));
            // Chunk markers refresh activity: a transfer still receiving chunks is not stale.
            let active = fs::metadata(path.join(".chunks"))
                .and_then(|m| m.modified())
                .is_ok_and(|modified| modified.elapsed().is_ok_and(|age| age <= STALE_STAGING));
            if entry.file_name() != keep && stale && !active {
                let _ = fs::remove_dir_all(path);
            }
        }
    }

    fn receiving_dir(&self, id: &str) -> Result<PathBuf> {
        self.package_dir(id)?;
        Ok(self.root().join(RECEIVING).join(id))
    }
}

enum Unfinished {
    /// Chunks are missing: keep what arrived so the sender can resume.
    Incomplete(String),
    /// Every chunk arrived but the package does not verify: nothing can be kept.
    Invalid(String),
}

/// Checks a package that `receive_finish` has claimed. Only this call reads `private`.
fn verify_claimed(
    claim: &Path,
    private: &Path,
    id: &str,
) -> std::result::Result<CheckpointManifest, Unfinished> {
    let transport: TransportManifest = read_json(&private.join(TRANSPORT_FILE))
        .map_err(|_| Unfinished::Invalid("The staged transfer has no manifest".into()))?;
    let status = receive_status(private, &transport).map_err(Unfinished::Invalid)?;
    if !status.complete {
        return Err(Unfinished::Incomplete(format!(
            "{} chunks are still missing; resume the transfer",
            status.missing.len()
        )));
    }
    let invalid = Unfinished::Invalid;
    for file in transport.files.iter().filter(|f| f.bytes == 0) {
        File::create(private.join(&file.path)).map_err(|e| invalid(io_error(e)))?;
    }
    let manifest_bytes =
        fs::read(private.join("manifest.json")).map_err(|e| invalid(io_error(e)))?;
    if hash_bytes(&manifest_bytes) != transport.files[0].sha256 {
        return Err(invalid("manifest.json failed verification".into()));
    }
    // Full package verification: identity, every byte, metadata and bank layout.
    let manifest = CheckpointLibrary::new(claim.to_path_buf())
        .verify(id)
        .map_err(invalid)?;
    if manifest.banks.len() + manifest.source_files.len() + 1 != transport.files.len() {
        return Err(invalid(
            "Transport lists files that the checkpoint manifest does not".into(),
        ));
    }
    fs::remove_dir_all(private.join(".chunks")).map_err(|e| invalid(io_error(e)))?;
    Ok(manifest)
}

fn receive_status(staging: &Path, transport: &TransportManifest) -> Result<ReceiveStatus> {
    let mut missing = Vec::new();
    let mut received = 0u64;
    for (f, file) in transport.files.iter().enumerate() {
        for (c, chunk) in file.chunks.iter().enumerate() {
            if staging
                .join(".chunks")
                .join(marker(f as u32, c as u32))
                .exists()
            {
                received += chunk.bytes;
            } else {
                missing.push(ChunkRef {
                    file: f as u32,
                    chunk: c as u32,
                });
            }
        }
    }
    Ok(ReceiveStatus {
        checkpoint_id: transport.checkpoint_id.clone(),
        complete: missing.is_empty(),
        total_bytes: transport.total_bytes,
        received_bytes: received,
        missing,
    })
}

fn marker(file: u32, chunk: u32) -> String {
    format!("{file}-{chunk}")
}

fn describes(
    cached: &TransportManifest,
    manifest: &CheckpointManifest,
    manifest_hash: &str,
) -> bool {
    let expected = std::iter::once(("manifest.json".to_string(), None))
        .chain(
            manifest
                .source_files
                .iter()
                .map(|f| (format!("source/{}", f.name), Some(f))),
        )
        .chain(
            manifest
                .banks
                .iter()
                .map(|f| (format!("banks/{}", f.name), Some(f))),
        );
    cached.checkpoint_id == manifest.id
        && cached.files.len() == manifest.source_files.len() + manifest.banks.len() + 1
        && cached
            .files
            .iter()
            .zip(expected)
            .all(|(file, (path, record))| {
                file.path == path
                    && record.map_or(file.sha256 == manifest_hash, |r| {
                        r.bytes == file.bytes && r.sha256 == file.sha256
                    })
            })
}

fn build(dir: &Path, manifest: &CheckpointManifest) -> Result<TransportManifest> {
    let mut files = vec![hash_chunks(dir, "manifest.json", None)?];
    for (folder, records) in [
        ("source", &manifest.source_files),
        ("banks", &manifest.banks),
    ] {
        for record in records {
            files.push(hash_chunks(
                dir,
                &format!("{folder}/{}", record.name),
                Some(record),
            )?);
        }
    }
    let mut read_order = manifest
        .tensors
        .iter()
        .map(|tensor| {
            let (layer, expert) = SafetensorsToKtbTranscoder::parse_layer_and_expert(&tensor.name);
            let extent = TransportExtent {
                tensor: tensor.name.clone(),
                layer,
                expert,
                bank: tensor.bank.clone(),
                offset: tensor.bank_offset,
                bytes: tensor.bytes,
                reserved_bytes: tensor.bytes.max(1).div_ceil(PAGE) * PAGE,
            };
            check_extent(&extent).map(|_| extent)
        })
        .collect::<Result<Vec<_>>>()?;
    read_order.sort_by(|a, b| expert_major(a).cmp(&expert_major(b)));
    Ok(TransportManifest {
        schema: TRANSPORT_SCHEMA.into(),
        checkpoint_id: manifest.id.clone(),
        name: manifest.name.clone(),
        alignment_bytes: TRANSPORT_ALIGNMENT_BYTES,
        page_bytes: PAGE,
        chunk_bytes: TRANSPORT_CHUNK_BYTES,
        total_bytes: files.iter().map(|f| f.bytes).sum(),
        files,
        read_order,
        quality_policy: manifest.quality_policy.clone(),
    })
}

/// InferSSD's expert-major order: dense tensors of a layer, then each expert's up, gate, down.
fn expert_major(extent: &TransportExtent) -> (u32, u64, u8, &str) {
    let name = extent.tensor.as_str();
    let role = if name.contains("gate_up") {
        3
    } else if name.contains("up_proj") || name.contains("ffn_up") || name.contains(".w3") {
        0
    } else if name.contains("gate_proj") || name.contains("ffn_gate") || name.contains(".w1") {
        1
    } else if name.contains("down_proj") || name.contains("ffn_down") || name.contains(".w2") {
        2
    } else {
        4
    };
    (
        extent.layer,
        extent.expert.map_or(0, |e| u64::from(e) + 1),
        role,
        name,
    )
}

fn check_extent(extent: &TransportExtent) -> Result<()> {
    if !extent.offset.is_multiple_of(TRANSPORT_ALIGNMENT_BYTES)
        || !extent
            .reserved_bytes
            .is_multiple_of(TRANSPORT_ALIGNMENT_BYTES)
        || extent.bytes > extent.reserved_bytes
    {
        return Err(format!(
            "Tensor {} is not aligned to {TRANSPORT_ALIGNMENT_BYTES} bytes for direct I/O",
            extent.tensor
        ));
    }
    Ok(())
}

/// The only chunk map `TransportManifest::validate` accepts for a file of `bytes` bytes, as
/// `(offset, bytes)` pairs: chunk `i` starts at `i * chunk_bytes` and holds the rest of the file
/// up to `chunk_bytes`. An empty file, or a zero chunk size, has no chunks.
pub fn chunk_plan(bytes: u64, chunk_bytes: u64) -> Vec<(u64, u64)> {
    if chunk_bytes == 0 {
        return Vec::new();
    }
    (0..bytes.div_ceil(chunk_bytes))
        .map(|index| {
            let offset = index * chunk_bytes;
            (offset, (bytes - offset).min(chunk_bytes))
        })
        .collect()
}

fn hash_chunks(dir: &Path, path: &str, record: Option<&PackageFile>) -> Result<TransportFile> {
    let input = File::open(dir.join(path)).map_err(io_error)?;
    let file = hash_stream(input, path, TRANSPORT_CHUNK_BYTES)?;
    if record.is_some_and(|r| r.bytes != file.bytes || r.sha256 != file.sha256) {
        return Err(format!(
            "Integrity check failed for {path}; restore a backup or prepare the original again"
        ));
    }
    Ok(file)
}

/// Reads `input` to the end in chunks of `chunk_bytes` and hashes each chunk and the whole
/// stream. This is the sender's half of the chunk map; the receiver's check is `chunk_plan`.
pub fn hash_stream(mut input: impl Read, path: &str, chunk_bytes: u64) -> Result<TransportFile> {
    let size = usize::try_from(chunk_bytes)
        .ok()
        .filter(|&size| size > 0)
        .ok_or("Transport chunk size must be positive")?;
    let mut whole = Sha256::new();
    let mut chunks = Vec::new();
    let mut buffer = vec![0u8; size];
    let mut offset = 0u64;
    loop {
        let mut filled = 0;
        while filled < buffer.len() {
            let n = input.read(&mut buffer[filled..]).map_err(io_error)?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        if filled == 0 {
            break;
        }
        whole.update(&buffer[..filled]);
        chunks.push(TransportChunk {
            offset,
            bytes: filled as u64,
            sha256: hash_bytes(&buffer[..filled]),
        });
        offset += filled as u64;
        if filled < buffer.len() {
            break;
        }
    }
    let sha256 = format!("{:x}", whole.finalize());
    Ok(TransportFile {
        path: path.into(),
        bytes: offset,
        sha256,
        chunks,
    })
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hex_digest(value: &str) -> Result<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err("Transport digests must be lowercase SHA-256".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SafetensorsToKtbTranscoder;

    /// The interleaving from the review of PR #62, as `Kadmium.Transport.Publish` models it:
    /// a `receive_begin` with a different manifest for the same checkpoint lands after
    /// `receive_finish` has claimed the staged package and before it verifies and publishes.
    #[test]
    fn begin_during_finish_cannot_publish_a_package_without_its_manifest() {
        let root = std::env::temp_dir().join(format!("transport-race-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let source = root.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
        fs::write(source.join("tokenizer.json"), b"{}").unwrap();
        let payload = [0u8, 0, 128, 63, 1, 0, 128, 63];
        fs::write(
            source.join("model.safetensors"),
            SafetensorsToKtbTranscoder::build_safetensors_bytes(&[(
                "model.layers.0.self_attn.q_proj.weight",
                "F32",
                vec![1, 2],
                payload.as_slice(),
            )]),
        )
        .unwrap();
        let sender = CheckpointLibrary::new(root.join("sender"));
        let id = sender.import(&source).unwrap().id;
        let transport = sender.transport(&id).unwrap();
        let receiver = CheckpointLibrary::new(root.join("receiver"));
        receiver.receive_begin(&transport).unwrap();
        for (f, file) in transport.files.iter().enumerate() {
            let bytes = fs::read(sender.package_dir(&id).unwrap().join(&file.path)).unwrap();
            for (c, chunk) in file.chunks.iter().enumerate() {
                let range = chunk.offset as usize..(chunk.offset + chunk.bytes) as usize;
                receiver
                    .receive_chunk(&id, f as u32, c as u32, &bytes[range])
                    .unwrap();
            }
        }
        let mut other = transport.clone();
        other.name = format!("{} (another build)", other.name);
        assert_ne!(other, transport);

        let manifest = receiver
            .finish_claimed(&id, &mut || {
                // Wipes and recreates staging; the claimed package is out of its reach.
                receiver.receive_begin(&other).unwrap();
            })
            .unwrap();
        assert_eq!(manifest.id, id);
        let published = receiver.package_dir(&id).unwrap();
        assert!(published.join("manifest.json").is_file());
        assert_eq!(receiver.verify(&id).unwrap().id, id);
        assert_eq!(receiver.list().unwrap().len(), 1);
        assert!(
            fs::read_dir(root.join("receiver").join(VERIFYING))
                .unwrap()
                .next()
                .is_none()
        );
        let _ = fs::remove_dir_all(&root);
    }
}
