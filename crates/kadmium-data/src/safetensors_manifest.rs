//! Validated, lazy loading for single- and multi-shard SafeTensors models.
//!
//! Opening a model reads its index (when present) and only the 8-byte length prefix and JSON
//! header of each shard. Tensor payloads remain on disk until explicitly requested.

use crate::batch_read::{
    DEFAULT_COALESCE_GAP_BYTES, READ_ALIGNMENT_BYTES, ReadPlanStats, plan_reads, plan_stats,
    read_planned,
};
use crate::transcoder::SafetensorsToKtbTranscoder;
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

/// Conventional Hugging Face multi-shard index file name.
pub const SAFETENSORS_INDEX_FILE_NAME: &str = "model.safetensors.index.json";
/// Conventional Hugging Face single-shard model file name.
pub const SAFETENSORS_SINGLE_FILE_NAME: &str = "model.safetensors";

/// SafeTensors data types with fixed-width element encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SafetensorsDtype {
    Bool,
    U8,
    I8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    Bf16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
    F8E8M0,
}

impl SafetensorsDtype {
    fn from_code(code: &str) -> Option<Self> {
        match code {
            "BOOL" => Some(Self::Bool),
            "U8" => Some(Self::U8),
            "I8" => Some(Self::I8),
            "I16" => Some(Self::I16),
            "U16" => Some(Self::U16),
            "I32" => Some(Self::I32),
            "U32" => Some(Self::U32),
            "I64" => Some(Self::I64),
            "U64" => Some(Self::U64),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            "F32" => Some(Self::F32),
            "F64" => Some(Self::F64),
            "F8_E4M3" => Some(Self::F8E4M3),
            "F8_E5M2" => Some(Self::F8E5M2),
            "F8_E8M0" => Some(Self::F8E8M0),
            _ => None,
        }
    }

    /// Canonical type code used in a SafeTensors JSON header.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::I64 => "I64",
            Self::U64 => "U64",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E8M0 => "F8_E8M0",
        }
    }

    /// Number of bytes occupied by one element.
    pub const fn element_size_bytes(self) -> u64 {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

impl fmt::Display for SafetensorsDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Metadata and on-disk location for one tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsTensorDescriptor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: SafetensorsDtype,
    pub element_count: u64,
    /// Byte offsets relative to the shard's data section, as stored in its JSON header.
    pub data_offsets: [u64; 2],
    /// Absolute byte position of the tensor payload in `shard_path`.
    pub absolute_offset: u64,
    pub byte_len: u64,
    pub shard_index: usize,
    /// Shard identity from the index, or `model.safetensors` for a single-file model.
    pub shard_name: String,
    pub shard_path: PathBuf,
}

/// Header and payload totals for one referenced shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsShardDescriptor {
    pub name: String,
    pub path: PathBuf,
    pub file_size_bytes: u64,
    pub header_size_bytes: u64,
    /// Absolute file position at which relative tensor `data_offsets` begin.
    pub data_base_offset: u64,
    pub tensor_count: usize,
    pub tensor_bytes: u64,
}

/// Aggregate sizes useful for model planning and validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetensorsManifestTotals {
    pub shard_count: usize,
    pub tensor_count: usize,
    pub element_count: u64,
    pub tensor_bytes: u64,
    pub file_bytes: u64,
    /// `metadata.total_size` from the index, when available.
    pub declared_tensor_bytes: Option<u64>,
}

/// Failures produced while validating a model or reading a tensor payload.
#[derive(Debug, Error)]
pub enum SafetensorsManifestError {
    #[error("neither {SAFETENSORS_INDEX_FILE_NAME} nor {SAFETENSORS_SINGLE_FILE_NAME} was found in `{model_dir}`", model_dir = .model_dir.display())]
    ModelFilesNotFound { model_dir: PathBuf },

    #[error("I/O error for `{path}`: {source}", path = .path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("invalid SafeTensors index JSON in `{path}`: {source}", path = .path.display())]
    InvalidIndexJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid SafeTensors header JSON in `{path}`: {source}", path = .path.display())]
    InvalidHeaderJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid header entry for tensor `{tensor}` in `{path}`: {source}", path = .path.display())]
    InvalidTensorEntry {
        tensor: String,
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("unsafe shard path `{path}`: {reason}")]
    UnsafeShardPath { path: String, reason: &'static str },

    #[error("invalid SafeTensors header in `{path}`: {reason}", path = .path.display())]
    InvalidHeader { path: PathBuf, reason: String },

    #[error("arithmetic overflow while computing {context}")]
    ArithmeticOverflow { context: &'static str },

    #[error("cannot allocate {bytes} bytes for {what}")]
    AllocationFailed { what: &'static str, bytes: u64 },

    #[error("unsupported SafeTensors dtype `{dtype}` for tensor `{tensor}`")]
    UnsupportedDtype { tensor: String, dtype: String },

    #[error(
        "shape/dtype size mismatch for tensor `{tensor}`: shape requires {expected_bytes} bytes, but data offsets describe {actual_bytes}"
    )]
    TensorSizeMismatch {
        tensor: String,
        expected_bytes: u64,
        actual_bytes: u64,
    },

    #[error(
        "tensor `{tensor}` has data offsets [{start}, {end}) outside the {data_bytes}-byte data section of shard `{shard}`"
    )]
    TensorOutOfBounds {
        tensor: String,
        shard: String,
        start: u64,
        end: u64,
        data_bytes: u64,
    },

    #[error("duplicate tensor `{tensor}` in shards `{first_shard}` and `{second_shard}`")]
    DuplicateTensor {
        tensor: String,
        first_shard: String,
        second_shard: String,
    },

    #[error(
        "tensor `{tensor}` is indexed to shard `{indexed_shard}` but was found in `{actual_shard}`"
    )]
    TensorShardMismatch {
        tensor: String,
        indexed_shard: String,
        actual_shard: String,
    },

    #[error("indexed tensor `{tensor}` is missing from shard `{shard}`")]
    MissingIndexedTensor { tensor: String, shard: String },

    #[error("tensor `{tensor}` in shard `{shard}` is not covered by the index")]
    UnindexedTensor { tensor: String, shard: String },

    #[error("index metadata declares {declared} tensor bytes, but shard headers describe {actual}")]
    TotalSizeMismatch { declared: u64, actual: u64 },

    #[error("tensor `{name}` was not found in the manifest")]
    TensorNotFound { name: String },

    #[error(
        "output buffer for tensor `{name}` has {actual} bytes; exactly {expected} are required"
    )]
    BufferSizeMismatch {
        name: String,
        expected: u64,
        actual: usize,
    },
}

/// A fully validated model manifest with lazy, exact tensor payload reads.
#[derive(Debug)]
pub struct SafetensorsManifest {
    model_dir: PathBuf,
    index_path: Option<PathBuf>,
    shards: Vec<SafetensorsShardDescriptor>,
    tensors: Vec<SafetensorsTensorDescriptor>,
    total_elements: u64,
    total_tensor_bytes: u64,
    total_file_bytes: u64,
    declared_tensor_bytes: Option<u64>,
}

impl SafetensorsManifest {
    /// Opens and validates either a multi-shard index or a single `model.safetensors` file.
    ///
    /// Only shard headers are read here. Payload bytes are read exclusively by
    /// [`read_tensor`](Self::read_tensor) and [`read_tensor_into`](Self::read_tensor_into).
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self, SafetensorsManifestError> {
        let supplied_dir = model_dir.as_ref();
        let model_dir =
            std::path::absolute(supplied_dir).map_err(|source| SafetensorsManifestError::Io {
                path: supplied_dir.to_path_buf(),
                source,
            })?;
        let index_path = model_dir.join(SAFETENSORS_INDEX_FILE_NAME);
        match read_index_if_present(&index_path)? {
            Some(index) => Self::open_indexed(model_dir, index_path, index),
            None => {
                let shard_path = model_dir.join(SAFETENSORS_SINGLE_FILE_NAME);
                match File::open(&shard_path) {
                    Ok(file) => {
                        drop(file);
                        Self::open_single(model_dir, shard_path)
                    }
                    Err(source) if source.kind() == io::ErrorKind::NotFound => {
                        Err(SafetensorsManifestError::ModelFilesNotFound { model_dir })
                    }
                    Err(source) => Err(SafetensorsManifestError::Io {
                        path: shard_path,
                        source,
                    }),
                }
            }
        }
    }

    fn open_single(
        model_dir: PathBuf,
        shard_path: PathBuf,
    ) -> Result<Self, SafetensorsManifestError> {
        let (shard, tensors) = load_shard(0, SAFETENSORS_SINGLE_FILE_NAME.to_string(), shard_path)?;
        Self::finish(model_dir, None, vec![shard], tensors, None, None)
    }

    fn open_indexed(
        model_dir: PathBuf,
        index_path: PathBuf,
        index: RawIndex,
    ) -> Result<Self, SafetensorsManifestError> {
        let declared_tensor_bytes = index.metadata.and_then(|metadata| metadata.total_size);
        let mut indexed_tensors = BTreeMap::new();
        let mut shard_names = BTreeMap::<PathBuf, String>::new();

        for (tensor_name, shard_name) in index.weight_map.0 {
            let relative_path = validated_shard_path(&shard_name)?;
            shard_names
                .entry(relative_path.clone())
                .or_insert_with(|| shard_name.clone());
            indexed_tensors.insert(tensor_name, (shard_name, relative_path));
        }

        let mut shards = Vec::with_capacity(shard_names.len());
        let mut tensors = Vec::new();
        let mut seen_tensors = BTreeMap::<String, String>::new();

        for (shard_index, (relative_path, shard_name)) in shard_names.into_iter().enumerate() {
            let shard_path = model_dir.join(relative_path);
            let (shard, shard_tensors) = load_shard(shard_index, shard_name, shard_path)?;
            for tensor in shard_tensors {
                if let Some(first_shard) =
                    seen_tensors.insert(tensor.name.clone(), tensor.shard_name.clone())
                {
                    return Err(SafetensorsManifestError::DuplicateTensor {
                        tensor: tensor.name,
                        first_shard,
                        second_shard: shard.name.clone(),
                    });
                }
                tensors.push(tensor);
            }
            shards.push(shard);
        }

        for tensor in &tensors {
            let Some((indexed_shard, indexed_path)) = indexed_tensors.get(&tensor.name) else {
                return Err(SafetensorsManifestError::UnindexedTensor {
                    tensor: tensor.name.clone(),
                    shard: tensor.shard_name.clone(),
                });
            };
            if tensor.shard_path != model_dir.join(indexed_path) {
                return Err(SafetensorsManifestError::TensorShardMismatch {
                    tensor: tensor.name.clone(),
                    indexed_shard: indexed_shard.clone(),
                    actual_shard: tensor.shard_name.clone(),
                });
            }
        }

        for (tensor_name, (shard_name, _)) in &indexed_tensors {
            if !seen_tensors.contains_key(tensor_name) {
                return Err(SafetensorsManifestError::MissingIndexedTensor {
                    tensor: tensor_name.clone(),
                    shard: shard_name.clone(),
                });
            }
        }

        Self::finish(
            model_dir,
            Some(index_path),
            shards,
            tensors,
            declared_tensor_bytes,
            Some(&seen_tensors),
        )
    }

    fn finish(
        model_dir: PathBuf,
        index_path: Option<PathBuf>,
        shards: Vec<SafetensorsShardDescriptor>,
        mut tensors: Vec<SafetensorsTensorDescriptor>,
        declared_tensor_bytes: Option<u64>,
        indexed_names: Option<&BTreeMap<String, String>>,
    ) -> Result<Self, SafetensorsManifestError> {
        if let Some(indexed_names) = indexed_names
            && tensors.len() != indexed_names.len()
        {
            return Err(SafetensorsManifestError::InvalidHeader {
                path: index_path.clone().unwrap_or_default(),
                reason: "index/header tensor counts conflict".to_string(),
            });
        }

        tensors.sort_by(|left, right| left.name.cmp(&right.name));

        let mut total_elements = 0u64;
        let mut total_tensor_bytes = 0u64;
        let mut total_file_bytes = 0u64;
        for tensor in &tensors {
            total_elements = total_elements.checked_add(tensor.element_count).ok_or(
                SafetensorsManifestError::ArithmeticOverflow {
                    context: "total model element count",
                },
            )?;
            total_tensor_bytes = total_tensor_bytes.checked_add(tensor.byte_len).ok_or(
                SafetensorsManifestError::ArithmeticOverflow {
                    context: "total model tensor bytes",
                },
            )?;
        }
        for shard in &shards {
            total_file_bytes = total_file_bytes.checked_add(shard.file_size_bytes).ok_or(
                SafetensorsManifestError::ArithmeticOverflow {
                    context: "total model file bytes",
                },
            )?;
        }

        if let Some(declared) = declared_tensor_bytes
            && declared != total_tensor_bytes
        {
            return Err(SafetensorsManifestError::TotalSizeMismatch {
                declared,
                actual: total_tensor_bytes,
            });
        }

        Ok(Self {
            model_dir,
            index_path,
            shards,
            tensors,
            total_elements,
            total_tensor_bytes,
            total_file_bytes,
            declared_tensor_bytes,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Returns the index path for a multi-shard model, or `None` for a single-file model.
    pub fn index_path(&self) -> Option<&Path> {
        self.index_path.as_deref()
    }

    pub fn shards(&self) -> &[SafetensorsShardDescriptor] {
        &self.shards
    }

    /// Tensor descriptors sorted lexicographically by tensor name.
    pub fn tensors(&self) -> &[SafetensorsTensorDescriptor] {
        &self.tensors
    }

    /// Deterministic, lexicographically sorted tensor names.
    pub fn tensor_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.tensors.iter().map(|tensor| tensor.name.as_str())
    }

    pub fn tensor(&self, name: &str) -> Option<&SafetensorsTensorDescriptor> {
        self.tensors
            .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.tensors[index])
    }

    /// Selects descriptors in the inclusive layer range using the existing transcode parser.
    /// Names without a recognized layer component retain that parser's layer-0 behavior.
    pub fn tensors_in_layer_range(
        &self,
        start_layer: u32,
        end_layer: u32,
    ) -> Vec<&SafetensorsTensorDescriptor> {
        self.tensors
            .iter()
            .filter(|tensor| {
                let (layer, _) = SafetensorsToKtbTranscoder::parse_layer_and_expert(&tensor.name);
                layer >= start_layer && layer <= end_layer
            })
            .collect()
    }

    /// Selects deterministic tensor names in the inclusive layer range.
    pub fn tensor_names_in_layer_range(&self, start_layer: u32, end_layer: u32) -> Vec<&str> {
        self.tensors_in_layer_range(start_layer, end_layer)
            .into_iter()
            .map(|tensor| tensor.name.as_str())
            .collect()
    }

    pub fn totals(&self) -> SafetensorsManifestTotals {
        SafetensorsManifestTotals {
            shard_count: self.shards.len(),
            tensor_count: self.tensors.len(),
            element_count: self.total_elements,
            tensor_bytes: self.total_tensor_bytes,
            file_bytes: self.total_file_bytes,
            declared_tensor_bytes: self.declared_tensor_bytes,
        }
    }

    pub fn total_elements(&self) -> u64 {
        self.total_elements
    }

    pub fn total_tensor_bytes(&self) -> u64 {
        self.total_tensor_bytes
    }

    /// Reads exactly one tensor payload. BF16 and every other encoding are returned unchanged.
    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>, SafetensorsManifestError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| SafetensorsManifestError::TensorNotFound {
                name: name.to_string(),
            })?;
        let mut bytes = allocate_buffer(tensor.byte_len, "tensor payload")?;
        self.read_descriptor_into(tensor, &mut bytes)?;
        Ok(bytes)
    }

    /// Reads many tensors with as few reads as possible: per shard, the tensors' byte ranges are
    /// merged when contiguous or within one `.ktb` page of each other, aligned to the drive's
    /// sector, and read by a small pool of reader threads with one handle each, instead of an
    /// open, a seek and a read per tensor (InferSSD's expert copy plan and pool reader). The
    /// payloads come back in the order of `names`. The stats say how many reads the plan saved.
    pub fn read_tensors(
        &self,
        names: &[&str],
    ) -> Result<(Vec<Vec<u8>>, ReadPlanStats), SafetensorsManifestError> {
        let mut per_shard: BTreeMap<usize, Vec<(usize, u64, u64)>> = BTreeMap::new();
        for (position, name) in names.iter().enumerate() {
            let tensor =
                self.tensor(name)
                    .ok_or_else(|| SafetensorsManifestError::TensorNotFound {
                        name: (*name).to_string(),
                    })?;
            per_shard.entry(tensor.shard_index).or_default().push((
                position,
                tensor.absolute_offset,
                tensor.byte_len,
            ));
        }
        let mut out: Vec<Vec<u8>> = (0..names.len()).map(|_| Vec::new()).collect();
        let mut stats = ReadPlanStats::default();
        for (shard_index, entries) in per_shard {
            let shard = &self.shards[shard_index];
            let ranges: Vec<(u64, u64)> = entries.iter().map(|e| (e.1, e.2)).collect();
            let reads = plan_reads(&ranges, DEFAULT_COALESCE_GAP_BYTES, READ_ALIGNMENT_BYTES);
            let shard_stats = plan_stats(&ranges, &reads);
            stats.ranges += shard_stats.ranges;
            stats.reads += shard_stats.reads;
            stats.bytes_requested += shard_stats.bytes_requested;
            stats.bytes_read += shard_stats.bytes_read;
            let buffers = read_planned(&shard.path, &reads, reads.len()).map_err(|source| {
                SafetensorsManifestError::Io {
                    path: shard.path.clone(),
                    source,
                }
            })?;
            for (entry, bytes) in entries.iter().zip(buffers) {
                out[entry.0] = bytes;
            }
        }
        Ok((out, stats))
    }

    /// Seeks to a tensor and fills an exactly sized caller-provided buffer.
    pub fn read_tensor_into(
        &self,
        name: &str,
        output: &mut [u8],
    ) -> Result<(), SafetensorsManifestError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| SafetensorsManifestError::TensorNotFound {
                name: name.to_string(),
            })?;
        if output.len() as u64 != tensor.byte_len {
            return Err(SafetensorsManifestError::BufferSizeMismatch {
                name: name.to_string(),
                expected: tensor.byte_len,
                actual: output.len(),
            });
        }
        self.read_descriptor_into(tensor, output)
    }

    fn read_descriptor_into(
        &self,
        tensor: &SafetensorsTensorDescriptor,
        output: &mut [u8],
    ) -> Result<(), SafetensorsManifestError> {
        let mut file =
            File::open(&tensor.shard_path).map_err(|source| SafetensorsManifestError::Io {
                path: tensor.shard_path.clone(),
                source,
            })?;
        file.seek(SeekFrom::Start(tensor.absolute_offset))
            .and_then(|_| file.read_exact(output))
            .map_err(|source| SafetensorsManifestError::Io {
                path: tensor.shard_path.clone(),
                source,
            })
    }
}

fn allocate_buffer(len: u64, what: &'static str) -> Result<Vec<u8>, SafetensorsManifestError> {
    let len_usize =
        usize::try_from(len).map_err(|_| SafetensorsManifestError::ArithmeticOverflow {
            context: "buffer length conversion",
        })?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(len_usize)
        .map_err(|_| SafetensorsManifestError::AllocationFailed { what, bytes: len })?;
    buffer.resize(len_usize, 0);
    Ok(buffer)
}

fn validated_shard_path(shard_name: &str) -> Result<PathBuf, SafetensorsManifestError> {
    if shard_name.is_empty() {
        return Err(SafetensorsManifestError::UnsafeShardPath {
            path: shard_name.to_string(),
            reason: "the path is empty",
        });
    }

    // Check both separator conventions so a hostile index cannot become unsafe when moved
    // between Unix and Windows hosts.
    let portable = shard_name.replace('\\', "/");
    let windows_drive_or_device_path = portable.as_bytes().get(1) == Some(&b':');
    if portable.starts_with('/') || windows_drive_or_device_path {
        return Err(SafetensorsManifestError::UnsafeShardPath {
            path: shard_name.to_string(),
            reason: "absolute and drive-prefixed paths are not allowed",
        });
    }
    if portable.split('/').any(|component| component == "..") {
        return Err(SafetensorsManifestError::UnsafeShardPath {
            path: shard_name.to_string(),
            reason: "parent-directory traversal is not allowed",
        });
    }

    let path = Path::new(shard_name);
    if path.is_absolute() {
        return Err(SafetensorsManifestError::UnsafeShardPath {
            path: shard_name.to_string(),
            reason: "absolute paths are not allowed",
        });
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(SafetensorsManifestError::UnsafeShardPath {
                    path: shard_name.to_string(),
                    reason: "parent-directory traversal is not allowed",
                });
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(SafetensorsManifestError::UnsafeShardPath {
                    path: shard_name.to_string(),
                    reason: "absolute and drive-prefixed paths are not allowed",
                });
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err(SafetensorsManifestError::UnsafeShardPath {
            path: shard_name.to_string(),
            reason: "the path does not name a file",
        });
    }
    Ok(normalized)
}

fn load_shard(
    shard_index: usize,
    shard_name: String,
    shard_path: PathBuf,
) -> Result<(SafetensorsShardDescriptor, Vec<SafetensorsTensorDescriptor>), SafetensorsManifestError>
{
    let mut file = File::open(&shard_path).map_err(|source| SafetensorsManifestError::Io {
        path: shard_path.clone(),
        source,
    })?;
    let file_size_bytes = file
        .metadata()
        .map_err(|source| SafetensorsManifestError::Io {
            path: shard_path.clone(),
            source,
        })?
        .len();
    if file_size_bytes < 8 {
        return Err(SafetensorsManifestError::InvalidHeader {
            path: shard_path,
            reason: "file is shorter than the 8-byte header length prefix".to_string(),
        });
    }

    let mut length_prefix = [0u8; 8];
    file.read_exact(&mut length_prefix)
        .map_err(|source| SafetensorsManifestError::Io {
            path: shard_path.clone(),
            source,
        })?;
    let header_size_bytes = u64::from_le_bytes(length_prefix);
    let data_base_offset = 8u64.checked_add(header_size_bytes).ok_or(
        SafetensorsManifestError::ArithmeticOverflow {
            context: "shard data-section position",
        },
    )?;
    if data_base_offset > file_size_bytes {
        return Err(SafetensorsManifestError::InvalidHeader {
            path: shard_path,
            reason: format!(
                "header ends at byte {data_base_offset}, beyond file size {file_size_bytes}"
            ),
        });
    }

    let mut header_bytes = allocate_buffer(header_size_bytes, "SafeTensors JSON header")?;
    file.read_exact(&mut header_bytes)
        .map_err(|source| SafetensorsManifestError::Io {
            path: shard_path.clone(),
            source,
        })?;
    let header: UniqueMap<serde_json::Value> =
        serde_json::from_slice(&header_bytes).map_err(|source| {
            SafetensorsManifestError::InvalidHeaderJson {
                path: shard_path.clone(),
                source,
            }
        })?;

    let data_bytes = file_size_bytes - data_base_offset;
    let mut tensors = Vec::with_capacity(header.0.len().saturating_sub(1));
    let mut shard_tensor_bytes = 0u64;

    for (tensor_name, value) in header.0 {
        if tensor_name == "__metadata__" {
            continue;
        }
        let spec: RawTensorEntry = serde_json::from_value(value).map_err(|source| {
            SafetensorsManifestError::InvalidTensorEntry {
                tensor: tensor_name.clone(),
                path: shard_path.clone(),
                source,
            }
        })?;
        let dtype = SafetensorsDtype::from_code(&spec.dtype).ok_or_else(|| {
            SafetensorsManifestError::UnsupportedDtype {
                tensor: tensor_name.clone(),
                dtype: spec.dtype.clone(),
            }
        })?;
        let [start, end] = spec.data_offsets;
        if start > end || end > data_bytes {
            return Err(SafetensorsManifestError::TensorOutOfBounds {
                tensor: tensor_name,
                shard: shard_name,
                start,
                end,
                data_bytes,
            });
        }

        let element_count = spec.shape.iter().try_fold(1u64, |count, &dimension| {
            count
                .checked_mul(dimension)
                .ok_or(SafetensorsManifestError::ArithmeticOverflow {
                    context: "tensor shape element count",
                })
        })?;
        let expected_bytes = element_count
            .checked_mul(dtype.element_size_bytes())
            .ok_or(SafetensorsManifestError::ArithmeticOverflow {
                context: "tensor dtype byte size",
            })?;
        let byte_len = end - start;
        if byte_len != expected_bytes {
            return Err(SafetensorsManifestError::TensorSizeMismatch {
                tensor: tensor_name,
                expected_bytes,
                actual_bytes: byte_len,
            });
        }
        let absolute_offset = data_base_offset.checked_add(start).ok_or(
            SafetensorsManifestError::ArithmeticOverflow {
                context: "absolute tensor file position",
            },
        )?;
        data_base_offset
            .checked_add(end)
            .ok_or(SafetensorsManifestError::ArithmeticOverflow {
                context: "absolute tensor end position",
            })?;
        shard_tensor_bytes = shard_tensor_bytes.checked_add(byte_len).ok_or(
            SafetensorsManifestError::ArithmeticOverflow {
                context: "shard tensor bytes",
            },
        )?;

        tensors.push(SafetensorsTensorDescriptor {
            name: tensor_name,
            shape: spec.shape,
            dtype,
            element_count,
            data_offsets: [start, end],
            absolute_offset,
            byte_len,
            shard_index,
            shard_name: shard_name.clone(),
            shard_path: shard_path.clone(),
        });
    }

    // SafeTensors payloads must be contiguous and cover the complete data section. Besides
    // detecting corruption, this rules out overlapping tensor descriptors and hidden gaps.
    let mut coverage = tensors.iter().collect::<Vec<_>>();
    coverage.sort_by(|left, right| {
        left.data_offsets
            .cmp(&right.data_offsets)
            .then_with(|| left.name.cmp(&right.name))
    });
    let mut expected_start = 0u64;
    for tensor in coverage {
        if tensor.data_offsets[0] != expected_start {
            return Err(SafetensorsManifestError::InvalidHeader {
                path: shard_path,
                reason: format!(
                    "tensor `{}` starts at relative byte {}, expected {expected_start}; offsets must be contiguous and non-overlapping",
                    tensor.name, tensor.data_offsets[0]
                ),
            });
        }
        expected_start = tensor.data_offsets[1];
    }
    if expected_start != data_bytes {
        return Err(SafetensorsManifestError::InvalidHeader {
            path: shard_path,
            reason: format!(
                "tensor data ends at relative byte {expected_start}, but the shard data section is {data_bytes} bytes"
            ),
        });
    }

    let shard = SafetensorsShardDescriptor {
        name: shard_name,
        path: shard_path,
        file_size_bytes,
        header_size_bytes,
        data_base_offset,
        tensor_count: tensors.len(),
        tensor_bytes: shard_tensor_bytes,
    };
    Ok((shard, tensors))
}

#[derive(Debug, Deserialize)]
struct RawIndex {
    #[serde(default)]
    metadata: Option<RawIndexMetadata>,
    weight_map: UniqueMap<String>,
}

#[derive(Debug, Deserialize)]
struct RawIndexMetadata {
    #[serde(default)]
    total_size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// A JSON object that rejects duplicate keys instead of silently keeping the last value.
#[derive(Debug)]
struct UniqueMap<V>(BTreeMap<String, V>);

impl<'de, V> Deserialize<'de> for UniqueMap<V>
where
    V: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueMapVisitor<V>(std::marker::PhantomData<V>);

        impl<'de, V> Visitor<'de> for UniqueMapVisitor<V>
        where
            V: Deserialize<'de>,
        {
            type Value = UniqueMap<V>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object with unique string keys")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, V>()? {
                    if values.insert(key.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate object key `{key}`"
                        )));
                    }
                }
                Ok(UniqueMap(values))
            }
        }

        deserializer.deserialize_map(UniqueMapVisitor(std::marker::PhantomData))
    }
}

fn read_index_if_present(index_path: &Path) -> Result<Option<RawIndex>, SafetensorsManifestError> {
    let mut file = match File::open(index_path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(SafetensorsManifestError::Io {
                path: index_path.to_path_buf(),
                source,
            });
        }
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|source| SafetensorsManifestError::Io {
            path: index_path.to_path_buf(),
            source,
        })?;
    let index = serde_json::from_slice(&bytes).map_err(|source| {
        SafetensorsManifestError::InvalidIndexJson {
            path: index_path.to_path_buf(),
            source,
        }
    })?;
    Ok(Some(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcoder::{SafetensorFieldSpec, SafetensorsToKtbTranscoder};
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    const LAYER_0: &str = "model.layers.0.self_attn.q_proj.weight";
    const LAYER_1: &str = "model.layers.1.mlp.experts.3.up_proj.weight";
    const LAYER_2: &str = "model.layers.2.self_attn.q_proj.weight";
    const NORM: &str = "model.norm.weight";

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TestModelDir {
        path: PathBuf,
    }

    impl TestModelDir {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos();
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kadmium-safetensors-manifest-{}-{nonce}-{id}-{label}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create unique temporary model directory");
            Self { path }
        }

        fn write_shard(&self, name: &str, bytes: &[u8]) {
            fs::write(self.path.join(name), bytes).expect("write test shard");
        }

        fn write_index(&self, entries: &[(&str, &str)], total_size: u64) {
            let weight_map = entries
                .iter()
                .map(|(tensor, shard)| ((*tensor).to_string(), (*shard).to_string()))
                .collect::<BTreeMap<_, _>>();
            let index = serde_json::json!({
                "metadata": { "total_size": total_size },
                "weight_map": weight_map,
            });
            fs::write(
                self.path.join(SAFETENSORS_INDEX_FILE_NAME),
                serde_json::to_vec_pretty(&index).expect("serialize test index"),
            )
            .expect("write test index");
        }
    }

    impl Drop for TestModelDir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.path).expect("remove temporary model directory");
        }
    }

    fn build_shard_from_specs(entries: &[(&str, SafetensorFieldSpec, &[u8])]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        for (name, spec, _) in entries {
            header.insert(
                (*name).to_string(),
                serde_json::to_value(spec).expect("serialize test tensor spec"),
            );
        }
        let header = serde_json::to_vec(&header).expect("serialize test shard header");
        let payload_len = entries
            .iter()
            .map(|(_, _, payload)| payload.len())
            .sum::<usize>();
        let mut bytes = Vec::with_capacity(8 + header.len() + payload_len);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        for (_, _, payload) in entries {
            bytes.extend_from_slice(payload);
        }
        bytes
    }

    fn two_shard_model() -> (TestModelDir, BTreeMap<String, Vec<u8>>) {
        let model = TestModelDir::new("multi-shard");
        let payloads = BTreeMap::from([
            (
                LAYER_0.to_string(),
                vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
            ),
            (LAYER_1.to_string(), vec![0x10, 0x11, 0x12, 0x13]),
            (LAYER_2.to_string(), vec![0x20, 0x21, 0x22, 0x23]),
            (NORM.to_string(), vec![0x30, 0x31, 0x32, 0x33]),
        ]);
        let shard_1 = SafetensorsToKtbTranscoder::build_safetensors_bytes(&[
            (LAYER_0, "BF16", vec![2, 2], &payloads[LAYER_0]),
            (LAYER_1, "BF16", vec![2], &payloads[LAYER_1]),
        ]);
        let shard_2 = SafetensorsToKtbTranscoder::build_safetensors_bytes(&[
            (LAYER_2, "BF16", vec![1, 2], &payloads[LAYER_2]),
            (NORM, "BF16", vec![2], &payloads[NORM]),
        ]);
        model.write_shard("model-00001-of-00002.safetensors", &shard_1);
        model.write_shard("model-00002-of-00002.safetensors", &shard_2);
        model.write_index(
            &[
                (LAYER_0, "model-00001-of-00002.safetensors"),
                (LAYER_1, "model-00001-of-00002.safetensors"),
                (LAYER_2, "model-00002-of-00002.safetensors"),
                (NORM, "model-00002-of-00002.safetensors"),
            ],
            20,
        );
        (model, payloads)
    }

    #[test]
    fn batch_reads_merge_per_shard_and_return_payloads_in_request_order() {
        let (model, payloads) = two_shard_model();
        let manifest = SafetensorsManifest::open(&model.path).expect("open multi-shard model");
        let names = [NORM, LAYER_0, LAYER_2, LAYER_1];
        let (bytes, stats) = manifest.read_tensors(&names).expect("batch read");
        for (name, got) in names.iter().zip(&bytes) {
            assert_eq!(got, &payloads[*name], "{name}");
        }
        // Four tensors in two shards, each shard's two tensors adjacent: two reads.
        assert_eq!(stats.ranges, 4);
        assert_eq!(stats.reads, 2);
        assert_eq!(stats.bytes_requested, 20);
        assert!(stats.bytes_read >= 20);
        let missing = manifest.read_tensors(&[LAYER_0, "nope"]).unwrap_err();
        assert!(matches!(
            missing,
            SafetensorsManifestError::TensorNotFound { .. }
        ));
        let (none, empty) = manifest.read_tensors(&[]).unwrap();
        assert!(none.is_empty() && empty.reads == 0);
    }

    #[test]
    fn manifest_preserves_multi_shard_metadata_and_deterministic_listing() {
        let (model, _) = two_shard_model();
        let manifest = SafetensorsManifest::open(&model.path).expect("open multi-shard model");

        assert_eq!(manifest.shards().len(), 2);
        assert_eq!(manifest.tensors().len(), 4);
        assert_eq!(manifest.total_tensor_bytes(), 20);
        assert_eq!(manifest.total_elements(), 10);
        assert_eq!(manifest.totals().declared_tensor_bytes, Some(20));
        assert_eq!(
            manifest.index_path(),
            Some(model.path.join(SAFETENSORS_INDEX_FILE_NAME).as_path())
        );
        assert_eq!(
            manifest.tensor_names().collect::<Vec<_>>(),
            vec![LAYER_0, LAYER_1, LAYER_2, NORM]
        );

        let descriptor = manifest.tensor(LAYER_1).expect("layer 1 descriptor");
        assert_eq!(descriptor.shape, vec![2]);
        assert_eq!(descriptor.dtype, SafetensorsDtype::Bf16);
        assert_eq!(descriptor.element_count, 2);
        assert_eq!(descriptor.data_offsets, [8, 12]);
        assert_eq!(descriptor.byte_len, 4);
        assert_eq!(descriptor.shard_index, 0);
        assert_eq!(descriptor.shard_name, "model-00001-of-00002.safetensors");
        assert_eq!(
            descriptor.shard_path,
            model.path.join("model-00001-of-00002.safetensors")
        );
        assert_eq!(
            descriptor.absolute_offset,
            manifest.shards()[0].data_base_offset + descriptor.data_offsets[0]
        );
    }

    #[test]
    fn exact_reads_seek_to_payload_and_retain_bf16_bytes() {
        let (model, payloads) = two_shard_model();
        let manifest = SafetensorsManifest::open(&model.path).expect("open multi-shard model");

        for (name, expected) in &payloads {
            assert_eq!(
                &manifest.read_tensor(name).expect("exact tensor read"),
                expected
            );
        }

        let mut output = [0xa5u8; 4];
        manifest
            .read_tensor_into(LAYER_1, &mut output)
            .expect("read into exact buffer");
        assert_eq!(output.as_slice(), payloads[LAYER_1]);

        let mut too_large = [0u8; 5];
        assert!(matches!(
            manifest.read_tensor_into(LAYER_1, &mut too_large),
            Err(SafetensorsManifestError::BufferSizeMismatch {
                expected: 4,
                actual: 5,
                ..
            })
        ));
        assert!(matches!(
            manifest.read_tensor("missing.weight"),
            Err(SafetensorsManifestError::TensorNotFound { .. })
        ));
    }

    #[test]
    fn layer_selection_is_inclusive_and_uses_existing_name_semantics() {
        let (model, _) = two_shard_model();
        let manifest = SafetensorsManifest::open(&model.path).expect("open multi-shard model");

        assert_eq!(
            manifest.tensor_names_in_layer_range(1, 2),
            vec![LAYER_1, LAYER_2]
        );
        assert_eq!(
            manifest.tensor_names_in_layer_range(0, 0),
            vec![LAYER_0, NORM]
        );
        assert!(manifest.tensor_names_in_layer_range(3, 2).is_empty());
    }

    #[test]
    fn rejects_corrupt_offsets_and_shape_dtype_mismatch() {
        let corrupt_offsets = TestModelDir::new("bad-offsets");
        let payload = [0x42u8; 8];
        let bytes = build_shard_from_specs(&[(
            LAYER_0,
            SafetensorFieldSpec {
                dtype: "BF16".to_string(),
                shape: vec![4],
                data_offsets: [0, 9],
            },
            &payload,
        )]);
        corrupt_offsets.write_shard(SAFETENSORS_SINGLE_FILE_NAME, &bytes);
        assert!(matches!(
            SafetensorsManifest::open(&corrupt_offsets.path),
            Err(SafetensorsManifestError::TensorOutOfBounds { .. })
        ));

        let shape_mismatch = TestModelDir::new("shape-mismatch");
        let bytes = build_shard_from_specs(&[(
            LAYER_0,
            SafetensorFieldSpec {
                dtype: "BF16".to_string(),
                shape: vec![3],
                data_offsets: [0, 8],
            },
            &payload,
        )]);
        shape_mismatch.write_shard(SAFETENSORS_SINGLE_FILE_NAME, &bytes);
        assert!(matches!(
            SafetensorsManifest::open(&shape_mismatch.path),
            Err(SafetensorsManifestError::TensorSizeMismatch {
                expected_bytes: 6,
                actual_bytes: 8,
                ..
            })
        ));
    }

    #[test]
    fn rejects_absolute_and_parent_traversal_shard_paths() {
        let model = TestModelDir::new("unsafe-paths");
        for unsafe_path in ["../outside.safetensors", "/outside/model.safetensors"] {
            model.write_index(&[(LAYER_0, unsafe_path)], 2);
            assert!(matches!(
                SafetensorsManifest::open(&model.path),
                Err(SafetensorsManifestError::UnsafeShardPath { .. })
            ));
        }
    }

    #[test]
    fn supports_single_file_models() {
        let model = TestModelDir::new("single-file");
        let payload = [0xf0, 0x3f, 0x00, 0x80];
        model.write_shard(
            SAFETENSORS_SINGLE_FILE_NAME,
            &SafetensorsToKtbTranscoder::build_safetensors_bytes(&[(
                LAYER_0,
                "BF16",
                vec![2],
                &payload,
            )]),
        );

        let manifest = SafetensorsManifest::open(&model.path).expect("open single-file model");
        assert!(manifest.index_path().is_none());
        assert_eq!(manifest.shards().len(), 1);
        assert_eq!(manifest.total_tensor_bytes(), 4);
        assert_eq!(manifest.read_tensor(LAYER_0).expect("read BF16"), payload);
    }

    #[test]
    fn rejects_index_header_coverage_conflicts() {
        let missing = TestModelDir::new("missing-indexed-tensor");
        missing.write_shard(
            SAFETENSORS_SINGLE_FILE_NAME,
            &SafetensorsToKtbTranscoder::build_safetensors_bytes(&[]),
        );
        missing.write_index(&[(LAYER_0, SAFETENSORS_SINGLE_FILE_NAME)], 8);
        assert!(matches!(
            SafetensorsManifest::open(&missing.path),
            Err(SafetensorsManifestError::MissingIndexedTensor { .. })
        ));

        let unindexed = TestModelDir::new("unindexed-header-tensor");
        let payload = [0u8; 8];
        unindexed.write_shard(
            SAFETENSORS_SINGLE_FILE_NAME,
            &SafetensorsToKtbTranscoder::build_safetensors_bytes(&[(
                LAYER_0,
                "BF16",
                vec![4],
                &payload,
            )]),
        );
        unindexed.write_index(&[("model.other.weight", SAFETENSORS_SINGLE_FILE_NAME)], 8);
        assert!(matches!(
            SafetensorsManifest::open(&unindexed.path),
            Err(SafetensorsManifestError::UnindexedTensor { .. })
        ));
    }
}
