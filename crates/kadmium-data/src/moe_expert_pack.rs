//! Reader for InferSSD's `GGMLMOEPACKv1` raw quantized expert packs.
//!
//! The pack stores byte extents but no tensor dtype/shape or checkpoint identity. This source
//! therefore reports dtype `GGML`, leaves shapes empty (graph metadata remains external), and
//! requires the catalog-issued durable checkpoint fingerprint used by its source model.

use crate::expert::{
    ExpertPacket, ExpertSource, ExpertTensor, SourceFile, allocate_bytes, read_exact_at,
    validate_source_identity,
};
use crate::transcoder::SafetensorsToKtbTranscoder;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const GGML_MOE_PACK_MAGIC: [u8; 16] = *b"GGMLMOEPACKv1\0\0\0";
pub const GGML_MOE_PACK_VERSION: u32 = 1;
pub const GGML_MOE_PACK_HEADER_BYTES: usize = 40;
pub const GGML_MOE_PACK_ENTRY_BYTES: usize = 152;
pub const GGML_MOE_PACK_ALIGNMENT: u64 = 4096;
pub const GGML_MOE_PACK_MAX_ENTRIES: u64 = 10_000_000;
const INDEX_ENTRIES_PER_READ: u64 = 4096;

#[derive(Debug)]
struct PackEntry {
    tensor_name: String,
    offset: u64,
    byte_len: u64,
}

/// Persistent, exact-extent reader for one `GGMLMOEPACKv1` file.
#[derive(Debug)]
pub struct MoePackExpertSource {
    model_id: String,
    source_fingerprint: u64,
    path: PathBuf,
    file: SourceFile,
    experts: BTreeMap<(u32, u32), Vec<PackEntry>>,
}

impl MoePackExpertSource {
    pub fn open(
        model_id: impl Into<String>,
        path: impl AsRef<Path>,
        source_fingerprint: u64,
    ) -> Result<Self, String> {
        let model_id = model_id.into();
        validate_source_identity(&model_id, source_fingerprint)?;
        let path = path.as_ref().to_path_buf();
        let file = SourceFile::open(&path)?;
        let file_size = file.len();
        let mut header = [0u8; GGML_MOE_PACK_HEADER_BYTES];
        read_exact_at(&file.file, 0, &mut header)
            .map_err(|e| format!("failed to read expert pack header: {e}"))?;
        if header[..13] != GGML_MOE_PACK_MAGIC[..13] {
            return Err("invalid GGMLMOEPACK magic".into());
        }
        let version = le_u32(&header[16..20]);
        let header_size = le_u32(&header[20..24]) as usize;
        let entry_count = le_u64(&header[24..32]);
        let data_start = le_u64(&header[32..40]);
        if version != GGML_MOE_PACK_VERSION || header_size < GGML_MOE_PACK_HEADER_BYTES {
            return Err(format!(
                "unsupported expert pack version {version} or header size {header_size}"
            ));
        }
        if entry_count > GGML_MOE_PACK_MAX_ENTRIES {
            return Err(format!("expert pack has too many entries: {entry_count}"));
        }
        let index_bytes = entry_count
            .checked_mul(GGML_MOE_PACK_ENTRY_BYTES as u64)
            .ok_or_else(|| "expert pack index length overflow".to_string())?;
        let index_end = (GGML_MOE_PACK_HEADER_BYTES as u64)
            .checked_add(index_bytes)
            .ok_or_else(|| "expert pack index end overflow".to_string())?;
        if data_start < index_end || data_start < header_size as u64 || data_start > file_size {
            return Err(format!(
                "invalid expert pack data start {data_start}; index ends at {index_end} and header size is {header_size}"
            ));
        }

        let mut experts: BTreeMap<(u32, u32), Vec<PackEntry>> = BTreeMap::new();
        let mut first_entry = 0u64;
        while first_entry < entry_count {
            let count = (entry_count - first_entry).min(INDEX_ENTRIES_PER_READ);
            let byte_len = count as usize * GGML_MOE_PACK_ENTRY_BYTES;
            // InferSSD starts the index at byte 40 even when header_size advertises an extension.
            let offset =
                GGML_MOE_PACK_HEADER_BYTES as u64 + first_entry * GGML_MOE_PACK_ENTRY_BYTES as u64;
            let mut bytes = allocate_bytes(byte_len as u64, "expert pack index chunk")?;
            read_exact_at(&file.file, offset, &mut bytes)
                .map_err(|e| format!("failed to read expert pack index: {e}"))?;
            for index in 0..count as usize {
                let start = index * GGML_MOE_PACK_ENTRY_BYTES;
                let raw = &bytes[start..start + GGML_MOE_PACK_ENTRY_BYTES];
                let tensor_name = parse_tensor_name(&raw[..128])?;
                let expert = le_i32(&raw[128..132]);
                let _reserved = le_u32(&raw[132..136]);
                let payload_offset = le_u64(&raw[136..144]);
                let payload_bytes = le_u64(&raw[144..152]);
                if expert < 0 {
                    return Err(format!("invalid expert index for `{tensor_name}`"));
                }
                if payload_bytes == 0 || payload_offset < data_start {
                    return Err(format!("invalid payload extent for `{tensor_name}`"));
                }
                let payload_end = payload_offset
                    .checked_add(payload_bytes)
                    .ok_or_else(|| format!("payload overflow for `{tensor_name}`"))?;
                if payload_end > file_size {
                    return Err(format!("payload for `{tensor_name}` exceeds pack bounds"));
                }
                let (layer, _) = SafetensorsToKtbTranscoder::parse_layer_and_expert(&tensor_name);
                let expert = expert as u32;
                let entries = experts.entry((layer, expert)).or_default();
                if entries.iter().any(|entry| entry.tensor_name == tensor_name) {
                    return Err(format!(
                        "duplicate pack tensor `{tensor_name}` for expert {layer}/{expert}"
                    ));
                }
                entries.push(PackEntry {
                    tensor_name,
                    offset: payload_offset,
                    byte_len: payload_bytes,
                });
            }
            first_entry += count;
        }
        Ok(Self {
            model_id,
            source_fingerprint,
            path,
            file,
            experts,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl ExpertSource for MoePackExpertSource {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn source_fingerprint(&self) -> u64 {
        self.source_fingerprint
    }

    fn expert_payload_bytes(&self, layer: u32, expert: u32) -> Result<u64, String> {
        let entries = self.experts.get(&(layer, expert)).ok_or_else(|| {
            format!(
                "expert {layer}/{expert} was not found in `{}`",
                self.path.display()
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
                "expert {layer}/{expert} was not found in `{}`",
                self.path.display()
            )
        })?;
        let mut tensors = Vec::with_capacity(entries.len());
        for entry in entries {
            let mut bytes = allocate_bytes(entry.byte_len, &entry.tensor_name)?;
            read_exact_at(&self.file.file, entry.offset, &mut bytes)
                .map_err(|e| format!("failed to read pack tensor `{}`: {e}", entry.tensor_name))?;
            tensors.push(ExpertTensor {
                name: entry.tensor_name.clone(),
                dtype: "GGML".into(),
                shape: Vec::new(),
                bytes: Arc::from(bytes),
            });
        }
        ExpertPacket::new(layer, expert, tensors)
    }

    fn validate_generation(&self) -> Result<(), String> {
        self.file.validate_generation()
    }
}

fn parse_tensor_name(field: &[u8]) -> Result<String, String> {
    // InferSSD force-terminates byte 127, so a completely full field is a 127-byte name.
    let end = field.iter().position(|byte| *byte == 0).unwrap_or(127);
    if end == 0 {
        return Err("expert pack tensor name is empty".into());
    }
    std::str::from_utf8(&field[..end])
        .map(str::to_string)
        .map_err(|e| format!("expert pack tensor name is not UTF-8: {e}"))
}

fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("four-byte field"))
}

fn le_i32(bytes: &[u8]) -> i32 {
    i32::from_le_bytes(bytes.try_into().expect("four-byte field"))
}

fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("eight-byte field"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn test_path() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "kadmium-moe-pack-{}-{nonce}-{}.moe",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn build_pack(entries: &[(&str, i32, &[u8])]) -> Vec<u8> {
        let index_end = GGML_MOE_PACK_HEADER_BYTES + entries.len() * GGML_MOE_PACK_ENTRY_BYTES;
        let data_start =
            index_end.div_ceil(GGML_MOE_PACK_ALIGNMENT as usize) * GGML_MOE_PACK_ALIGNMENT as usize;
        let mut offsets = Vec::with_capacity(entries.len());
        let mut cursor = data_start;
        for (_, _, payload) in entries {
            offsets.push(cursor);
            cursor = (cursor + payload.len()).div_ceil(GGML_MOE_PACK_ALIGNMENT as usize)
                * GGML_MOE_PACK_ALIGNMENT as usize;
        }
        let final_len = entries
            .iter()
            .zip(&offsets)
            .map(|((_, _, payload), offset)| offset + payload.len())
            .max()
            .unwrap_or(data_start);
        let mut output = vec![0u8; final_len];
        output[..16].copy_from_slice(&GGML_MOE_PACK_MAGIC);
        output[16..20].copy_from_slice(&GGML_MOE_PACK_VERSION.to_le_bytes());
        output[20..24].copy_from_slice(&(GGML_MOE_PACK_HEADER_BYTES as u32).to_le_bytes());
        output[24..32].copy_from_slice(&(entries.len() as u64).to_le_bytes());
        output[32..40].copy_from_slice(&(data_start as u64).to_le_bytes());
        for ((_, _, payload), payload_offset) in entries.iter().zip(&offsets) {
            output[*payload_offset..*payload_offset + payload.len()].copy_from_slice(payload);
        }
        for (index, ((name, expert, _), payload_offset)) in entries.iter().zip(offsets).enumerate()
        {
            let start = GGML_MOE_PACK_HEADER_BYTES + index * GGML_MOE_PACK_ENTRY_BYTES;
            output[start..start + name.len()].copy_from_slice(name.as_bytes());
            output[start + 128..start + 132].copy_from_slice(&expert.to_le_bytes());
            output[start + 136..start + 144]
                .copy_from_slice(&(payload_offset as u64).to_le_bytes());
            let payload_len = entries[index].2.len() as u64;
            output[start + 144..start + 152].copy_from_slice(&payload_len.to_le_bytes());
        }
        output
    }

    #[test]
    fn pack_source_reads_grouped_roles_without_conversion() {
        let path = test_path();
        let up = [0, 255, 7, 8];
        let gate = [9, 10];
        let down = [11, 12, 13, 14];
        let bytes = build_pack(&[
            ("blk.4.ffn_up_exps.weight", 3, &up),
            ("blk.4.ffn_gate_exps.weight", 3, &gate),
            ("blk.4.ffn_down_exps.weight", 3, &down),
        ]);
        fs::write(&path, bytes).expect("write pack");
        let source =
            MoePackExpertSource::open("pack-model", &path, 0x5eed_c0de).expect("open pack");
        let packet = source.load_expert(4, 3).expect("load pack expert");
        assert_eq!(packet.tensors.len(), 3);
        assert_eq!(packet.payload_bytes(), 10);
        assert_eq!(packet.tensors[0].bytes.as_ref(), &up);
        assert_eq!(packet.tensors[1].bytes.as_ref(), &gate);
        assert_eq!(packet.tensors[2].bytes.as_ref(), &down);
        assert!(packet.tensors.iter().all(|tensor| tensor.dtype == "GGML"));
        fs::remove_file(path).expect("remove pack");
    }

    #[test]
    fn pack_source_rejects_bad_magic_and_out_of_bounds_extents() {
        let path = test_path();
        let mut bad_magic = build_pack(&[("blk.0.ffn_up_exps.weight", 0, &[1, 2])]);
        bad_magic[0] ^= 1;
        fs::write(&path, bad_magic).expect("write bad pack");
        assert!(MoePackExpertSource::open("bad", &path, 0x5eed_c0de).is_err());

        let mut bad_range = build_pack(&[("blk.0.ffn_up_exps.weight", 0, &[1, 2])]);
        let offset = GGML_MOE_PACK_HEADER_BYTES + 136;
        bad_range[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        fs::write(&path, bad_range).expect("write bad range");
        assert!(MoePackExpertSource::open("bad", &path, 0x5eed_c0de).is_err());
        fs::remove_file(path).expect("remove pack");
    }

    #[test]
    fn pack_source_accepts_inferssd_v1_layout_variants() {
        let path = test_path();
        let mut bytes = build_pack(&[("blk.1.ffn_up_exps.weight", 2, &[1, 2])]);
        bytes[13] = 0x7f; // InferSSD compares only the 13-byte magic string.
        bytes[20..24].copy_from_slice(&48u32.to_le_bytes());
        bytes[GGML_MOE_PACK_HEADER_BYTES + 132..GGML_MOE_PACK_HEADER_BYTES + 136]
            .copy_from_slice(&9u32.to_le_bytes());
        let unaligned_offset = bytes.len() + 1;
        bytes.resize(unaligned_offset + 2, 0);
        bytes[unaligned_offset..].copy_from_slice(&[7, 8]);
        bytes[GGML_MOE_PACK_HEADER_BYTES + 136..GGML_MOE_PACK_HEADER_BYTES + 144]
            .copy_from_slice(&(unaligned_offset as u64).to_le_bytes());
        fs::write(&path, bytes).expect("write variant pack");
        let source = MoePackExpertSource::open("variant", &path, 77).expect("open variant");
        assert_eq!(
            source.load_expert(1, 2).expect("load variant").tensors[0]
                .bytes
                .as_ref(),
            &[7, 8]
        );
        assert_eq!(
            parse_tensor_name(&[b'x'; 128]).expect("full name").len(),
            127
        );
        drop(source);
        fs::remove_file(path).expect("remove pack");
    }

    #[test]
    fn pack_generation_change_is_rejected_before_cached_reads() {
        use crate::expert::{ExpertKey, ExpertLoader};
        use std::io::{Seek, SeekFrom, Write};

        let path = test_path();
        fs::write(
            &path,
            build_pack(&[("blk.0.ffn_up_exps.weight", 0, &[1, 2])]),
        )
        .expect("write pack");
        let source = Arc::new(MoePackExpertSource::open("pack", &path, 99).expect("open pack"));
        let loader = ExpertLoader::default();
        loader.register_source(source.clone()).expect("register");
        loader
            .load_model_experts("pack", vec![ExpertKey::new(0, 0)])
            .expect("warm cache");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for mutation");
        file.seek(SeekFrom::Start(GGML_MOE_PACK_ALIGNMENT))
            .expect("seek payload");
        file.write_all(&[3, 4]).expect("mutate payload");
        file.set_modified(SystemTime::now() + std::time::Duration::from_secs(5))
            .expect("advance modification time");
        file.sync_all().expect("flush mutation");
        drop(file);
        assert!(
            loader
                .load_model_experts("pack", vec![ExpertKey::new(0, 0)])
                .expect_err("mutation must invalidate source")
                .contains("changed after it was opened")
        );
        drop(loader);
        drop(source);
        fs::remove_file(path).expect("remove pack");
    }
}
