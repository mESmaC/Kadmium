use kadmium_data::SafetensorsToKtbTranscoder;
use kadmium_data::checkpoint::CheckpointLibrary;
use kadmium_data::checkpoint::transport::{
    TRANSPORT_ALIGNMENT_BYTES, TRANSPORT_CHUNK_BYTES, TransportManifest,
};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(1);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "checkpoint-transport-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(path.join("source")).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn package(temp: &Scratch) -> (CheckpointLibrary, String) {
    let source = temp.0.join("source");
    fs::write(
        source.join("config.json"),
        br#"{"model_type":"mixtral","max_position_embeddings":32768}"#,
    )
    .unwrap();
    fs::write(source.join("tokenizer.json"), b"{}").unwrap();
    let payload = [0u8, 0, 128, 63, 1, 0, 128, 63];
    // Name order differs from InferSSD's expert-major order on purpose.
    let names = [
        "model.layers.1.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ];
    let tensors: Vec<_> = names
        .iter()
        .map(|n| (*n, "F32", vec![1, 2], payload.as_slice()))
        .collect();
    fs::write(
        source.join("model.safetensors"),
        SafetensorsToKtbTranscoder::build_safetensors_bytes(&tensors),
    )
    .unwrap();
    let library = CheckpointLibrary::new(temp.0.join("sender"));
    let id = library.import(&source).unwrap().id;
    (library, id)
}

#[test]
fn transport_is_aligned_expert_major_and_cached() {
    let temp = Scratch::new();
    let (library, id) = package(&temp);
    let transport = library.transport(&id).unwrap();
    transport.validate().unwrap();
    assert_eq!(transport.alignment_bytes, TRANSPORT_ALIGNMENT_BYTES);
    assert_eq!(transport.chunk_bytes, TRANSPORT_CHUNK_BYTES);
    assert_eq!(transport.files[0].path, "manifest.json");
    assert!(
        transport
            .files
            .iter()
            .any(|f| f.path == "banks/bank-00000.ktb")
    );
    assert!(
        transport
            .read_order
            .iter()
            .all(|e| e.offset % 4096 == 0 && e.reserved_bytes % 4096 == 0)
    );
    let order: Vec<_> = transport
        .read_order
        .iter()
        .map(|e| e.tensor.as_str())
        .collect();
    assert_eq!(
        order,
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.1.up_proj.weight",
            "model.layers.1.mlp.experts.0.down_proj.weight",
        ]
    );
    let dir = library.package_dir(&id).unwrap();
    assert!(dir.join("transport.json").exists());
    assert_eq!(library.transport(&id).unwrap(), transport);
    assert_eq!(library.list().unwrap().len(), 1);

    // A bank changed after normalization is caught when the transport is rebuilt.
    fs::remove_file(dir.join("transport.json")).unwrap();
    let bank = dir.join("banks/bank-00000.ktb");
    let mut bytes = fs::read(&bank).unwrap();
    bytes[65536] ^= 1;
    fs::write(&bank, bytes).unwrap();
    assert!(library.transport(&id).unwrap_err().contains("Integrity"));
}

#[test]
fn receiver_resumes_rejects_corruption_and_publishes_verified_package() {
    let temp = Scratch::new();
    let (sender, id) = package(&temp);
    let transport = sender.transport(&id).unwrap();
    let receiver = CheckpointLibrary::new(temp.0.join("receiver"));
    let chunk = |file: usize, index: usize| {
        let entry = &transport.files[file];
        let bytes = fs::read(sender.package_dir(&id).unwrap().join(&entry.path)).unwrap();
        let c = &entry.chunks[index];
        bytes[c.offset as usize..(c.offset + c.bytes) as usize].to_vec()
    };

    let status = receiver.receive_begin(&transport).unwrap();
    assert!(!status.complete);
    assert_eq!(status.missing.len(), transport.summary().chunk_count);
    assert!(receiver.list().unwrap().is_empty(), "staging stays hidden");

    let mut corrupt = chunk(1, 0);
    corrupt[0] ^= 1;
    assert!(
        receiver
            .receive_chunk(&id, 1, 0, &corrupt)
            .unwrap_err()
            .contains("verification")
    );
    receiver.receive_chunk(&id, 0, 0, &chunk(0, 0)).unwrap();
    assert!(
        receiver
            .receive_finish(&id)
            .unwrap_err()
            .contains("missing")
    );

    // Resuming reports only what is still needed.
    let resumed = receiver.receive_begin(&transport).unwrap();
    assert_eq!(resumed.missing.len(), status.missing.len() - 1);
    for missing in resumed.missing {
        receiver
            .receive_chunk(
                &id,
                missing.file,
                missing.chunk,
                &chunk(missing.file as usize, missing.chunk as usize),
            )
            .unwrap();
    }
    let published = receiver.receive_finish(&id).unwrap();
    assert_eq!(published.id, id);
    assert_eq!(receiver.verify(&id).unwrap().tensors.len(), 5);
    assert_eq!(receiver.transport(&id).unwrap(), transport);
    assert!(receiver.receive_begin(&transport).unwrap().complete);
}

#[test]
fn receiver_rejects_unsafe_or_mismatched_transport() {
    let temp = Scratch::new();
    let (sender, id) = package(&temp);
    let transport = sender.transport(&id).unwrap();
    let receiver = CheckpointLibrary::new(temp.0.join("receiver"));
    let mutate = |change: &dyn Fn(&mut TransportManifest)| {
        let mut changed = transport.clone();
        change(&mut changed);
        receiver.receive_begin(&changed).unwrap_err()
    };
    assert!(mutate(&|t| t.files[1].path = "source/../escape".into()).contains("Invalid"));
    assert!(mutate(&|t| t.files[2].sha256 = "0".repeat(64)).contains("identity"));
    assert!(mutate(&|t| t.chunk_bytes = 4096).contains("Unsupported"));
    assert!(mutate(&|t| t.read_order[0].offset += 512).contains("aligned"));
    assert!(mutate(&|t| t.quality_policy = "quantize".into()).contains("Unsupported"));
}
