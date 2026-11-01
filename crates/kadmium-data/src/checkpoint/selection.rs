use super::{CheckpointLibrary, CheckpointManifest, Result, Staging, io_error};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, Ordering};

impl CheckpointLibrary {
    /// Creates a separate package containing the named raw tensors and unchanged model metadata.
    /// The input remains a complete checkpoint; the result is a slice, not a standalone model.
    pub fn select_tensors(
        &self,
        id: &str,
        names: &[String],
        destination: &CheckpointLibrary,
        cancelled: &AtomicBool,
    ) -> Result<CheckpointManifest> {
        check_cancelled(cancelled)?;
        let original = self.verify(id)?;
        if original.source_format != "safetensors" || original.pre_quantized.is_some() {
            return Err("Tensor selection requires an unquantized SafeTensors checkpoint".into());
        }
        let wanted: BTreeSet<_> = names.iter().map(String::as_str).collect();
        if wanted.is_empty() || wanted.len() != names.len() {
            return Err("Tensor selection must be nonempty and contain no duplicate names".into());
        }
        let tensors: BTreeMap<_, _> = original
            .tensors
            .iter()
            .filter(|tensor| wanted.contains(tensor.name.as_str()))
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect();
        if tensors.len() != wanted.len() {
            return Err("Tensor selection names a tensor absent from the checkpoint".into());
        }
        let source = self.package_dir(id)?.join("source");
        fs::create_dir_all(destination.root()).map_err(io_error)?;
        let staging = Staging::create(destination.root())?;
        let mut header = BTreeMap::new();
        let mut offset = 0u64;
        for (name, tensor) in &tensors {
            let end = offset
                .checked_add(tensor.bytes)
                .ok_or("Tensor size overflow")?;
            header.insert(
                *name,
                json!({
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let mut header = serde_json::to_vec(&header).map_err(io_error)?;
        if header.len() as u64 > super::MAX_MANIFEST {
            return Err("Selected SafeTensors header is too large".into());
        }
        header.resize(header.len().div_ceil(8) * 8, b' ');
        let mut output = File::create(staging.0.join("model.safetensors")).map_err(io_error)?;
        output
            .write_all(&(header.len() as u64).to_le_bytes())
            .map_err(io_error)?;
        output.write_all(&header).map_err(io_error)?;
        let mut buffer = vec![0u8; 64 * 1024];
        for tensor in tensors.values() {
            check_cancelled(cancelled)?;
            let mut input = File::open(source.join(&tensor.source)).map_err(io_error)?;
            input
                .seek(SeekFrom::Start(tensor.source_offset))
                .map_err(io_error)?;
            let mut remaining = tensor.bytes;
            while remaining > 0 {
                check_cancelled(cancelled)?;
                let count = remaining.min(buffer.len() as u64) as usize;
                let chunk = buffer
                    .get_mut(..count)
                    .ok_or("Invalid copy buffer length")?;
                input.read_exact(chunk).map_err(io_error)?;
                output.write_all(chunk).map_err(io_error)?;
                remaining -= count as u64;
            }
        }
        output.sync_all().map_err(io_error)?;
        drop(output);
        for metadata in &original.source_files {
            if !metadata.name.ends_with(".safetensors")
                && metadata.name != "model.safetensors.index.json"
            {
                check_cancelled(cancelled)?;
                fs::copy(source.join(&metadata.name), staging.0.join(&metadata.name))
                    .map_err(io_error)?;
                let copied = super::hash_file(&staging.0.join(&metadata.name), &metadata.name)?;
                if copied.bytes != metadata.bytes || copied.sha256 != metadata.sha256 {
                    return Err("Checkpoint metadata changed during tensor selection".into());
                }
            }
        }
        check_cancelled(cancelled)?;
        let selected = destination.import(&staging.0)?;
        check_cancelled(cancelled)?;
        Ok(selected)
    }
}

pub(crate) fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Acquire) {
        Err("Checkpoint staging cancelled".into())
    } else {
        Ok(())
    }
}
