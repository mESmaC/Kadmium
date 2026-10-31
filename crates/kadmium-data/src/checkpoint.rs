//! Lossless local checkpoint packages. Originals and model metadata remain byte-for-byte
//! intact; KTB banks are storage layouts, not architecture conversions or runnable models.
//! Large payloads are copied with bounded buffers, never loaded as a whole checkpoint.

pub(crate) mod formats;

pub use formats::ggml_type_name;

use crate::{
    KTB_PAGE_ALIGNMENT_BYTES, KtbBinaryArchiveHeader, KtbPageAlignedTensorEntry, TensorEncoding,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub type Result<T> = std::result::Result<T, String>;
const PAGE: u64 = KTB_PAGE_ALIGNMENT_BYTES;
const MAX_MANIFEST: u64 = 64 * 1024 * 1024;
static NEXT_IMPORT: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageFile {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub encoding: TensorEncoding,
    pub source: String,
    pub source_offset: u64,
    pub bytes: u64,
    pub bank: String,
    pub bank_offset: u64,
}

/// A source stored below 16-bit floating point, or one that declares a quantization method.
/// Kadmium never produces these bytes; it copies them unchanged, and only after the operator
/// accepts the source precision for that import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreQuantizedSource {
    /// Method declared by the checkpoint, such as `gptq`, `awq`, `fp8` or `gguf`. `None` when
    /// the tensors are sub-16-bit floats but no method is declared.
    pub method: Option<String>,
    /// Bits per weight, when the checkpoint metadata states one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits: Option<u64>,
    /// Stored tensor formats below 16-bit floating point, such as `Q4_K`, `F8_E4M3` or the
    /// `I32` words that hold packed GPTQ weights.
    pub tensor_formats: Vec<String>,
}

impl PreQuantizedSource {
    /// One-line description for errors, warnings and labels, such as `gptq 4-bit (I32)`.
    pub fn description(&self) -> String {
        let mut text = self
            .method
            .clone()
            .unwrap_or_else(|| "undeclared method".into());
        if let Some(bits) = self.bits {
            text.push_str(&format!(" {bits}-bit"));
        }
        if !self.tensor_formats.is_empty() {
            text.push_str(&format!(" ({})", self.tensor_formats.join(", ")));
        }
        text
    }
}

/// Per-import choices. The default accepts only sources at 16-bit floating point or wider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportOptions {
    /// Import a checkpoint that was quantized before it reached Kadmium. Its bytes are kept
    /// exactly; Kadmium cannot restore detail removed before import.
    #[serde(default)]
    pub accept_pre_quantized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub source_format: String,
    pub architecture: Option<String>,
    pub context_length: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scaled_context_length: Option<u64>,
    pub source_files: Vec<PackageFile>,
    pub banks: Vec<PackageFile>,
    pub tensors: Vec<CheckpointTensor>,
    /// The importer never changes tensor bytes, drops layers, or changes routing.
    pub quality_policy: String,
    /// Present when the operator accepted a pre-quantized source for this package.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_quantized: Option<PreQuantizedSource>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckpointSummary {
    pub id: String,
    pub name: String,
    pub source_format: String,
    pub architecture: Option<String>,
    pub context_length: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scaled_context_length: Option<u64>,
    pub tensor_count: usize,
    pub source_bytes: u64,
    pub bank_bytes: u64,
    pub bank_count: usize,
    pub dtypes: BTreeSet<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_quantized: Option<PreQuantizedSource>,
    pub warnings: Vec<String>,
}

impl CheckpointManifest {
    pub fn summary(&self) -> CheckpointSummary {
        CheckpointSummary {
            id: self.id.clone(),
            name: self.name.clone(),
            source_format: self.source_format.clone(),
            architecture: self.architecture.clone(),
            context_length: self.context_length,
            scaled_context_length: self.scaled_context_length,
            tensor_count: self.tensors.len(),
            source_bytes: self.source_files.iter().map(|f| f.bytes).sum(),
            bank_bytes: self.banks.iter().map(|f| f.bytes).sum(),
            bank_count: self.banks.len(),
            dtypes: self.tensors.iter().map(|t| t.dtype.clone()).collect(),
            pre_quantized: self.pre_quantized.clone(),
            warnings: self.warnings.clone(),
        }
    }
}

#[derive(Clone)]
pub struct CheckpointLibrary {
    root: PathBuf,
}

impl Default for CheckpointLibrary {
    fn default() -> Self {
        Self::new(
            std::env::var_os("KADMIUM_MODEL_LIBRARY")
                .map(PathBuf::from)
                .unwrap_or_else(default_library_directory),
        )
    }
}

fn default_library_directory() -> PathBuf {
    #[cfg(windows)]
    let root = std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("Kadmium"));
    #[cfg(target_os = "macos")]
    let root = std::env::var_os("HOME")
        .map(|p| PathBuf::from(p).join("Library/Application Support/Kadmium"));
    #[cfg(not(any(windows, target_os = "macos")))]
    let root = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .map(|p| p.join("kadmium"))
        .or_else(|| {
            std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".local/share/kadmium"))
        });
    root.filter(|p| p.is_absolute())
        .map(|p| p.join("models"))
        .unwrap_or_else(|| PathBuf::from("target/kadmium-model-library"))
}

impl CheckpointLibrary {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn package_dir(&self, id: &str) -> Result<PathBuf> {
        if id.len() != 64
            || !id
                .bytes()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        {
            return Err("Checkpoint ID must be a lowercase SHA-256 digest".into());
        }
        Ok(self.root.join(id))
    }

    pub fn read(&self, id: &str) -> Result<CheckpointManifest> {
        let path = self.package_dir(id)?.join("manifest.json");
        let manifest: CheckpointManifest = read_json(&path)?;
        if !matches!(manifest.schema_version, 1..=3)
            || manifest.id != id
            || manifest.quality_policy != "preserve_source"
        {
            return Err("Unsupported or inconsistent checkpoint manifest".into());
        }
        for file in manifest.source_files.iter().chain(&manifest.banks) {
            simple_name(&file.name)?;
        }
        Ok(manifest)
    }

    pub fn list(&self) -> Result<Vec<CheckpointSummary>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut found = Vec::new();
        for entry in fs::read_dir(&self.root).map_err(io_error)? {
            let entry = entry.map_err(io_error)?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            if entry.file_type().map_err(io_error)?.is_dir() {
                found.push(self.read(&name)?.summary());
            }
        }
        found.sort_by(|a, b| a.name.cmp(&b.name).then(a.id.cmp(&b.id)));
        Ok(found)
    }

    /// Verify the manifest identity, all originals, and every bank before a backend load.
    pub fn verify(&self, id: &str) -> Result<CheckpointManifest> {
        let mut manifest = self.read(id)?;
        if package_id(&manifest.source_files)? != id {
            return Err("Checkpoint identity mismatch".into());
        }
        let dir = self.package_dir(id)?;
        for (folder, files) in [
            ("source", &manifest.source_files),
            ("banks", &manifest.banks),
        ] {
            for expected in files {
                let actual = hash_file(&dir.join(folder).join(&expected.name), &expected.name)?;
                if actual.sha256 != expected.sha256 || actual.bytes != expected.bytes {
                    return Err(format!(
                        "Integrity check failed for {folder}/{}; restore a backup or normalize the original into another library",
                        expected.name
                    ));
                }
            }
        }
        let formats::InspectedSource {
            format,
            architecture,
            context_length: context,
            scaled_context_length: scaled_context,
            tensors,
            pre_quantized,
        } = formats::inspect(&dir.join("source"), &manifest.source_files)?;
        let recorded: Vec<_> = manifest
            .tensors
            .iter()
            .cloned()
            .map(|mut tensor| {
                tensor.bank.clear();
                tensor.bank_offset = 0;
                tensor
            })
            .collect();
        if format != manifest.source_format || tensors != recorded {
            return Err("Checkpoint manifest does not match its source metadata".into());
        }
        if architecture != manifest.architecture
            || context != manifest.context_length
            || scaled_context != manifest.scaled_context_length
        {
            if manifest.schema_version == 1
                && formats::matches_legacy_v1_metadata(
                    &dir.join("source"),
                    &format,
                    &manifest.architecture,
                    manifest.context_length,
                )
            {
                manifest.schema_version = 2;
                manifest.architecture = architecture;
                manifest.context_length = context;
                manifest.scaled_context_length = scaled_context;
                persist_upgrade(&dir, &mut manifest);
            } else {
                return Err("Checkpoint manifest does not match its source metadata".into());
            }
        }
        if pre_quantized != manifest.pre_quantized {
            // Packages written before schema 3 kept quantized sources without recording them.
            // Record what the source holds so every view labels it; the bytes are unchanged.
            if manifest.schema_version < 3 && manifest.pre_quantized.is_none() {
                manifest.schema_version = 3;
                if let Some(quantized) = &pre_quantized {
                    manifest.warnings.push(pre_quantized_warning(quantized));
                }
                manifest.pre_quantized = pre_quantized;
                persist_upgrade(&dir, &mut manifest);
            } else {
                return Err("Checkpoint manifest does not match its source precision".into());
            }
        }
        if manifest.banks.len() != tensors.len().div_ceil(64) {
            return Err("Checkpoint manifest has an inconsistent bank count".into());
        }
        for (index, batch) in manifest.tensors.chunks(64).enumerate() {
            let bank = &manifest.banks[index];
            let name = format!("bank-{index:05}.ktb");
            let mut offset = PAGE;
            for tensor in batch {
                if tensor.bank != name || tensor.bank_offset != offset || bank.name != name {
                    return Err(
                        "Checkpoint manifest has an inconsistent tensor bank location".into(),
                    );
                }
                let padded = tensor
                    .bytes
                    .max(1)
                    .checked_add(PAGE - 1)
                    .ok_or("Tensor size overflow")?
                    / PAGE
                    * PAGE;
                offset = offset.checked_add(padded).ok_or("Bank size overflow")?;
            }
            if bank.bytes != offset {
                return Err("Checkpoint manifest has an inconsistent bank size".into());
            }
        }
        Ok(manifest)
    }

    /// Import with the default options, which refuse pre-quantized sources.
    pub fn import(&self, source: &Path) -> Result<CheckpointManifest> {
        self.import_with(source, ImportOptions::default())
    }

    pub fn import_with(&self, source: &Path, options: ImportOptions) -> Result<CheckpointManifest> {
        let source = source.canonicalize().map_err(io_error)?;
        let paths = source_paths(&source)?;
        // Validate headers before copying a potentially multi-terabyte checkpoint.
        let descriptors: Vec<_> = paths
            .iter()
            .map(|p| {
                Ok(PackageFile {
                    name: p
                        .file_name()
                        .and_then(|s| s.to_str())
                        .ok_or("Checkpoint filenames must be UTF-8")?
                        .into(),
                    bytes: fs::metadata(p).map_err(io_error)?.len(),
                    sha256: String::new(),
                })
            })
            .collect::<Result<_>>()?;
        let inspected = formats::inspect(
            if source.is_dir() {
                &source
            } else {
                source.parent().ok_or("Checkpoint has no parent")?
            },
            &descriptors,
        )?;
        refuse_unaccepted(&inspected.pre_quantized, options)?;
        fs::create_dir_all(&self.root).map_err(io_error)?;
        let temporary = self.root.join(format!(
            ".import-{}-{}",
            std::process::id(),
            NEXT_IMPORT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&temporary).map_err(io_error)?;
        let staging = Staging(temporary);
        fs::create_dir(staging.0.join("source")).map_err(io_error)?;
        fs::create_dir(staging.0.join("banks")).map_err(io_error)?;
        let mut source_files = Vec::new();
        for path in paths {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or("Checkpoint filenames must be UTF-8")?;
            simple_name(name)?;
            let mut input = File::open(&path).map_err(io_error)?;
            let mut output = File::create(staging.0.join("source").join(name)).map_err(io_error)?;
            let (bytes, sha256, _) = copy_hash(&mut input, &mut output, None)?;
            output.sync_all().map_err(io_error)?;
            source_files.push(PackageFile {
                name: name.into(),
                bytes,
                sha256,
            });
        }
        source_files.sort_by(|a, b| a.name.cmp(&b.name));
        let id = package_id(&source_files)?;
        let target = self.package_dir(&id)?;
        if target.exists() {
            return self.verify(&id);
        }
        let formats::InspectedSource {
            format: source_format,
            architecture,
            context_length,
            scaled_context_length,
            mut tensors,
            pre_quantized,
        } = formats::inspect(&staging.0.join("source"), &source_files)?;
        // The copied bytes decide, in case the source changed after the first inspection.
        refuse_unaccepted(&pre_quantized, options)?;
        let mut banks = Vec::new();
        // A bounded number of descriptors per bank keeps version-1 Page 0 readable everywhere.
        for (index, batch) in tensors.chunks_mut(64).enumerate() {
            let name = format!("bank-{index:05}.ktb");
            let path = staging.0.join("banks").join(&name);
            write_bank(&path, &staging.0.join("source"), &id, &name, batch)?;
            banks.push(hash_file(&path, &name)?);
        }
        let mut warnings = vec![
            "KTB normalization preserves source precision; it cannot recover quality lost before import.".into(),
            "Generation requires a compatible inference backend. KTB banks are not a native language-model runtime or a RepOps parity certification.".into(),
        ];
        if let Some(quantized) = &pre_quantized {
            warnings.push(pre_quantized_warning(quantized));
        } else if tensors.iter().any(|t| {
            !matches!(
                t.encoding,
                TensorEncoding::Fp64
                    | TensorEncoding::Fp32
                    | TensorEncoding::Fp16
                    | TensorEncoding::Bf16
            )
        }) {
            warnings.push("This checkpoint contains integer, boolean or complex tensors. Their original formats are preserved.".into());
        }
        let manifest = CheckpointManifest {
            schema_version: 3,
            id,
            name: source
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            source_format,
            architecture,
            context_length,
            scaled_context_length,
            source_files,
            banks,
            tensors,
            quality_policy: "preserve_source".into(),
            pre_quantized,
            warnings,
        };
        let mut file = File::create(staging.0.join("manifest.json")).map_err(io_error)?;
        file.write_all(&serde_json::to_vec_pretty(&manifest).map_err(io_error)?)
            .map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
        drop(file);
        // Publish only complete packages. Concurrent identical imports may share the result.
        if let Err(error) = fs::rename(&staging.0, &target) {
            if target.exists() {
                return self.verify(&manifest.id);
            }
            return Err(io_error(error));
        }
        Ok(manifest)
    }
}

/// Saves a metadata-only manifest upgrade atomically. Verification already succeeded, so a
/// read-only or full library keeps the upgrade in memory and says it was not saved.
fn persist_upgrade(dir: &Path, manifest: &mut CheckpointManifest) {
    let temporary = dir.join(".manifest.json.upgrade");
    let saved = serde_json::to_vec_pretty(&*manifest)
        .map_err(io_error)
        .and_then(|bytes| {
            let mut file = File::create(&temporary).map_err(io_error)?;
            file.write_all(&bytes).map_err(io_error)?;
            file.sync_all().map_err(io_error)
        })
        .and_then(|()| fs::rename(&temporary, dir.join("manifest.json")).map_err(io_error));
    if let Err(error) = saved {
        let _ = fs::remove_file(&temporary);
        manifest.warnings.push(format!(
            "The upgraded manifest could not be saved ({error}); verification repeats the upgrade until it can."
        ));
    }
}

fn pre_quantized_warning(quantized: &PreQuantizedSource) -> String {
    format!(
        "You accepted a pre-quantized source: {}. Kadmium keeps these bytes exactly; responses reflect this stored precision.",
        quantized.description()
    )
}

fn refuse_unaccepted(found: &Option<PreQuantizedSource>, options: ImportOptions) -> Result<()> {
    match found {
        Some(quantized) if !options.accept_pre_quantized => Err(format!(
            "This checkpoint is stored at a quantized precision: {}. Kadmium never quantizes weights; it imports a quantized source only when you accept that precision for this import, and then copies its bytes exactly. Some models are released at this precision, while others were quantized from a wider original whose removed detail Kadmium cannot restore. Accept this precision to import it.",
            quantized.description()
        )),
        _ => Ok(()),
    }
}

struct Staging(PathBuf);
impl Drop for Staging {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn source_paths(source: &Path) -> Result<Vec<PathBuf>> {
    if source.is_file() {
        if source.extension().is_some_and(|e| e == "gguf") {
            return Ok(vec![source.into()]);
        }
        return Err("Select a GGUF file, or a SafeTensors directory containing config.json and tokenizer files. A lone weight shard is not a complete model.".into());
    }
    if !source.is_dir() {
        return Err("Checkpoint path must be a file or directory".into());
    }
    let mut paths = Vec::new();
    for entry in fs::read_dir(source).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_file()
            && (name.ends_with(".safetensors")
                || name.ends_with(".gguf")
                || matches!(
                    name.as_str(),
                    "config.json"
                        | "generation_config.json"
                        | "tokenizer.json"
                        | "tokenizer.model"
                        | "tokenizer_config.json"
                        | "special_tokens_map.json"
                        | "added_tokens.json"
                        | "vocab.json"
                        | "merges.txt"
                        | "chat_template.jinja"
                        | "chat_template.json"
                        | "preprocessor_config.json"
                        | "processor_config.json"
                        | "model.safetensors.index.json"
                        // Quantization parameters that some pre-quantized exports keep
                        // outside config.json (AutoGPTQ, AutoAWQ, NVIDIA ModelOpt).
                        | "quantize_config.json"
                        | "quant_config.json"
                        | "hf_quant_config.json"
                ))
        {
            paths.push(path);
        }
    }
    if paths.is_empty() {
        return Err("No GGUF or SafeTensors checkpoint found in this directory".into());
    }
    paths.sort();
    Ok(paths)
}

fn package_id(files: &[PackageFile]) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(files).map_err(io_error)?)
    ))
}

pub(crate) fn simple_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 240
        || name == "."
        || name == ".."
        || name.contains(['/', '\\', ':'])
    {
        return Err(format!("Invalid checkpoint filename {name:?}"));
    }
    Ok(())
}

pub(crate) fn io_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).map_err(io_error)?;
    if file.metadata().map_err(io_error)?.len() > MAX_MANIFEST {
        return Err("Checkpoint metadata exceeds 64 MiB".into());
    }
    serde_json::from_reader(file).map_err(io_error)
}

fn copy_hash(
    input: &mut impl Read,
    output: &mut impl Write,
    length: Option<u64>,
) -> Result<(u64, String, u64)> {
    let mut sha = Sha256::new();
    let mut fnv = 0xcbf29ce484222325u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut total = 0;
    loop {
        let size = length.map_or(buffer.len(), |n| {
            (n - total).min(buffer.len() as u64) as usize
        });
        if size == 0 {
            break;
        }
        let n = input.read(&mut buffer[..size]).map_err(io_error)?;
        if n == 0 {
            break;
        }
        output.write_all(&buffer[..n]).map_err(io_error)?;
        sha.update(&buffer[..n]);
        for byte in &buffer[..n] {
            fnv = (fnv ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
        total += n as u64;
    }
    if length.is_some_and(|n| n != total) {
        return Err("Checkpoint payload was truncated".into());
    }
    Ok((total, format!("{:x}", sha.finalize()), fnv))
}

fn hash_file(path: &Path, name: &str) -> Result<PackageFile> {
    let (bytes, sha256, _) = copy_hash(
        &mut File::open(path).map_err(io_error)?,
        &mut std::io::sink(),
        None,
    )?;
    Ok(PackageFile {
        name: name.into(),
        bytes,
        sha256,
    })
}

fn write_bank(
    path: &Path,
    source: &Path,
    id: &str,
    name: &str,
    tensors: &mut [CheckpointTensor],
) -> Result<()> {
    let mut output = File::create(path).map_err(io_error)?;
    output
        .write_all(&vec![0u8; PAGE as usize])
        .map_err(io_error)?;
    let mut entries = Vec::new();
    let mut offset = PAGE;
    for tensor in tensors {
        let mut input = File::open(source.join(&tensor.source)).map_err(io_error)?;
        input
            .seek(SeekFrom::Start(tensor.source_offset))
            .map_err(io_error)?;
        let (_, _, checksum) = copy_hash(&mut input, &mut output, Some(tensor.bytes))?;
        let padded = tensor
            .bytes
            .max(1)
            .checked_add(PAGE - 1)
            .ok_or("Tensor size overflow")?
            / PAGE
            * PAGE;
        let padding = vec![0; (padded - tensor.bytes) as usize];
        output.write_all(&padding).map_err(io_error)?;
        let (layer, expert) =
            crate::SafetensorsToKtbTranscoder::parse_layer_and_expert(&tensor.name);
        tensor.bank = name.into();
        tensor.bank_offset = offset;
        entries.push(KtbPageAlignedTensorEntry {
            tensor_name: tensor.name.clone(),
            layer_index: layer,
            expert_index: expert,
            encoding: tensor.encoding,
            page_offset_bytes: offset,
            payload_bytes: tensor.bytes,
            padded_reservation_bytes: padded,
            checksum_fnv64: checksum,
        });
        offset = offset.checked_add(padded).ok_or("Bank size overflow")?;
    }
    let header = KtbBinaryArchiveHeader {
        model_id: id.into(),
        format_version: 1,
        page_alignment_bytes: PAGE,
        entries,
    };
    let bytes = serde_json::to_vec(&header).map_err(io_error)?;
    if bytes.len() > PAGE as usize - 10 {
        return Err("Tensor names exceed the KTB header budget".into());
    }
    output.seek(SeekFrom::Start(0)).map_err(io_error)?;
    output.write_all(b"KDTB").map_err(io_error)?;
    output.write_all(&1u16.to_le_bytes()).map_err(io_error)?;
    output
        .write_all(&(bytes.len() as u32).to_le_bytes())
        .map_err(io_error)?;
    output.write_all(&bytes).map_err(io_error)?;
    output.sync_all().map_err(io_error)
}

pub(crate) fn unique_tensors(tensors: &[CheckpointTensor]) -> Result<()> {
    let mut names = BTreeSet::new();
    for t in tensors {
        if t.name.is_empty() || t.name.len() > 512 || !names.insert(&t.name) {
            return Err(format!("Invalid or duplicate tensor name {:?}", t.name));
        }
    }
    if tensors.is_empty() {
        return Err("Checkpoint has no tensors".into());
    }
    Ok(())
}
