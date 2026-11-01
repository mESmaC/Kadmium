use kadmium_core::{HardwareNode, NodeId, PoliteState, ResourceOffer, sample_personal_cluster};
use kadmium_data::SafetensorsManifest;
use kadmium_data::checkpoint::CheckpointLibrary;
use kadmium_grid::{ContentKind, ContentRef, PlanApproval, Replica, ReplicaCatalog, ResidencyTier};
use kadmium_runtime::native_decoder::staging::{PreparedSlice, SliceDestination};
use kadmium_runtime::native_decoder::{DecoderConfig, DecoderSlice, SlicePlan};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    source: CheckpointLibrary,
    packages: CheckpointLibrary,
    receiver: CheckpointLibrary,
    model_id: String,
    node: HardwareNode,
    offer: ResourceOffer,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "kadmium-native-staging-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        let input = root.join("input");
        fs::create_dir_all(&input).unwrap();
        let original = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/native_decoder")
            .join(name);
        for name in ["config.json", "model.safetensors"] {
            fs::copy(original.join(name), input.join(name)).unwrap();
        }
        fs::write(input.join("tokenizer.json"), b"{}").unwrap();
        fs::write(
            input.join("generation_config.json"),
            br#"{"eos_token_id":7}"#,
        )
        .unwrap();
        let source = CheckpointLibrary::new(root.join("source"));
        let model_id = source.import(&input).unwrap().id;
        let mut node = sample_personal_cluster().remove(0);
        node.id = NodeId("slice-worker".into());
        node.polite_state = PoliteState::IdleDedicated;
        node.memory.nvme_cache_gb = 1.0;
        let mut offer = node.default_offer();
        offer.nvme_gb = 1.0;
        offer.requires_approval = true;
        Self {
            source,
            packages: CheckpointLibrary::new(root.join("packages")),
            receiver: CheckpointLibrary::new(root.join("receiver")),
            root,
            model_id,
            node,
            offer,
        }
    }

    fn prepare(&self, plan: SlicePlan) -> PreparedSlice {
        PreparedSlice::prepare(
            &self.source,
            &self.model_id,
            plan,
            self.packages.clone(),
            NodeId("source".into()),
            &AtomicBool::new(false),
        )
        .unwrap()
    }

    fn approval(&self, prepared: &PreparedSlice) -> PlanApproval {
        PlanApproval {
            node: self.node.id.clone(),
            model_id: self.model_id.clone(),
            placement_digest: prepared.placement_digest(&self.node.id),
            approved_unix_ms: 1,
        }
    }

    fn stage(
        &self,
        prepared: &PreparedSlice,
        catalog: &mut ReplicaCatalog,
        approval: Option<&PlanApproval>,
        cancelled: &AtomicBool,
    ) -> Result<String, String> {
        prepared
            .stage_to(
                catalog,
                SliceDestination {
                    node: &self.node,
                    offer: &self.offer,
                    approval,
                    library: &self.receiver,
                },
                cancelled,
                2,
            )
            .map(|manifest| manifest.id)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn middle() -> SlicePlan {
    SlicePlan {
        layers: 1..2,
        embedding: false,
        head: false,
    }
}

#[test]
fn selected_packages_contain_exact_tensors_and_original_bits() {
    for name in ["llama3_bf16", "qwen2_f32", "qwen3_bf16", "mistral_f16"] {
        let fixture = Fixture::new(name);
        let directory = fixture
            .source
            .package_dir(&fixture.model_id)
            .unwrap()
            .join("source");
        let original = SafetensorsManifest::open(&directory).unwrap();
        let config = DecoderConfig::from_model_dir(&directory).unwrap();
        for plan in [
            middle(),
            SlicePlan {
                layers: 0..0,
                embedding: true,
                head: false,
            },
            SlicePlan {
                layers: 3..3,
                embedding: false,
                head: true,
            },
        ] {
            let prepared = fixture.prepare(plan.clone());
            let mut catalog = ReplicaCatalog::new();
            prepared.register_source(&mut catalog, 0).unwrap();
            let approval = fixture.approval(&prepared);
            let id = fixture
                .stage(
                    &prepared,
                    &mut catalog,
                    Some(&approval),
                    &AtomicBool::new(false),
                )
                .unwrap();
            let selected_dir = fixture.receiver.package_dir(&id).unwrap().join("source");
            let selected = SafetensorsManifest::open(&selected_dir).unwrap();
            let expected = plan.tensor_names(&original, &config).unwrap();
            assert_eq!(selected.tensors().len(), expected.len());
            for tensor in selected.tensors() {
                assert!(
                    expected.contains(&tensor.name),
                    "unexpected {}",
                    tensor.name
                );
                let source = original.tensor(&tensor.name).unwrap();
                assert_eq!(tensor.dtype, source.dtype);
                assert_eq!(tensor.shape, source.shape);
                assert_eq!(
                    selected.read_tensor(&tensor.name).unwrap(),
                    original.read_tensor(&tensor.name).unwrap()
                );
            }
            for metadata in ["config.json", "tokenizer.json", "generation_config.json"] {
                assert_eq!(
                    fs::read(selected_dir.join(metadata)).unwrap(),
                    fs::read(directory.join(metadata)).unwrap()
                );
            }
            DecoderSlice::load(&selected_dir, plan).unwrap();
            assert_eq!(
                catalog.bytes_resident_on(&fixture.node.id),
                prepared.transport().total_bytes
            );
        }
    }
}

#[test]
fn chain_loaded_from_staged_packages_matches_original_logits_bit_for_bit() {
    for name in ["llama3_bf16", "qwen2_f32", "qwen3_bf16", "mistral_f16"] {
        let fixture = Fixture::new(name);
        let directory = fixture
            .source
            .package_dir(&fixture.model_id)
            .unwrap()
            .join("source");
        let config = DecoderConfig::from_model_dir(&directory).unwrap();
        let whole = DecoderSlice::load(&directory, SlicePlan::whole(&config)).unwrap();
        let tokens = [1, 2, 3, 4];
        let input = whole.embed(&tokens, 0).unwrap();
        let expected = whole
            .logits(&whole.forward(&input, &mut whole.new_cache()).unwrap())
            .unwrap();
        let mut hidden = input;
        let mut actual = Vec::new();
        for plan in [
            SlicePlan {
                layers: 0..1,
                embedding: true,
                head: false,
            },
            SlicePlan {
                layers: 1..config.num_layers,
                embedding: false,
                head: true,
            },
        ] {
            let prepared = fixture.prepare(plan.clone());
            let mut catalog = ReplicaCatalog::new();
            prepared.register_source(&mut catalog, 0).unwrap();
            let approval = fixture.approval(&prepared);
            let id = fixture
                .stage(
                    &prepared,
                    &mut catalog,
                    Some(&approval),
                    &AtomicBool::new(false),
                )
                .unwrap();
            let staged = DecoderSlice::load(
                &fixture.receiver.package_dir(&id).unwrap().join("source"),
                plan,
            )
            .unwrap();
            hidden = staged.forward(&hidden, &mut staged.new_cache()).unwrap();
            if staged.plan().head {
                actual = staged.logits(&hidden).unwrap();
            }
        }
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{name}"
        );
    }
}

#[test]
fn approval_is_required_for_the_exact_model_node_and_slice() {
    let fixture = Fixture::new("qwen2_f32");
    let prepared = fixture.prepare(middle());
    let mut catalog = ReplicaCatalog::new();
    assert!(
        prepared
            .staging_plan(&catalog, &fixture.node, &fixture.offer, None)
            .is_err()
    );
    prepared.register_source(&mut catalog, 0).unwrap();
    let preview = prepared
        .staging_plan(&catalog, &fixture.node, &fixture.offer, None)
        .unwrap();
    assert_eq!(preview.awaiting_approval.len(), 1);
    assert!(preview.ready.is_empty());
    let approval = fixture.approval(&prepared);
    for approval in [
        None,
        Some(PlanApproval {
            model_id: "other".into(),
            ..approval.clone()
        }),
        Some(PlanApproval {
            node: NodeId("other".into()),
            ..approval.clone()
        }),
        Some(PlanApproval {
            placement_digest: approval.placement_digest ^ 1,
            ..approval.clone()
        }),
    ] {
        assert!(
            fixture
                .stage(
                    &prepared,
                    &mut catalog,
                    approval.as_ref(),
                    &AtomicBool::new(false)
                )
                .unwrap_err()
                .contains("approve")
        );
        assert!(!fixture.receiver.root().exists());
        assert_eq!(catalog.bytes_resident_on(&fixture.node.id), 0);
    }
    let id = fixture
        .stage(
            &prepared,
            &mut catalog,
            Some(&approval),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(id, prepared.transport().checkpoint_id);
    fixture
        .stage(&prepared, &mut catalog, None, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(catalog.resident_on(&fixture.node.id).len(), 1);
}

#[test]
fn storage_offer_accounts_for_existing_replicas_and_polite_state() {
    let mut fixture = Fixture::new("qwen2_f32");
    let prepared = fixture.prepare(middle());
    let mut catalog = ReplicaCatalog::new();
    prepared.register_source(&mut catalog, 0).unwrap();
    let needed = prepared.transport().total_bytes;
    fixture.offer.nvme_gb = (needed - 1) as f64 / 1_073_741_824.0;
    assert!(
        prepared
            .staging_plan(&catalog, &fixture.node, &fixture.offer, None)
            .unwrap_err()
            .contains("resource offer")
    );
    fixture.offer.nvme_gb = needed as f64 / 1_073_741_824.0;
    catalog.record(Replica::new(
        ContentRef::new("existing", ContentKind::Artifact, 1),
        fixture.node.id.clone(),
        ResidencyTier::LocalNvme,
        0,
    ));
    assert!(
        prepared
            .staging_plan(&catalog, &fixture.node, &fixture.offer, None)
            .is_err()
    );
    catalog.remove(&fixture.node.id, "existing");
    assert_eq!(
        prepared
            .staging_plan(&catalog, &fixture.node, &fixture.offer, None)
            .unwrap()
            .awaiting_approval
            .len(),
        1
    );
    fixture.node.polite_state = PoliteState::Draining;
    assert!(
        prepared
            .staging_plan(&catalog, &fixture.node, &fixture.offer, None)
            .is_err()
    );
    assert!(!fixture.receiver.root().exists());
}

#[test]
fn corrupted_source_chunks_never_create_a_destination_replica() {
    let fixture = Fixture::new("qwen2_f32");
    let prepared = fixture.prepare(middle());
    let mut catalog = ReplicaCatalog::new();
    prepared.register_source(&mut catalog, 0).unwrap();
    let approval = fixture.approval(&prepared);
    let file = prepared
        .transport()
        .files
        .iter()
        .find(|file| file.path.starts_with("banks/"))
        .unwrap();
    let path = fixture
        .packages
        .package_dir(&prepared.transport().checkpoint_id)
        .unwrap()
        .join(&file.path);
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] ^= 1;
    fs::write(&path, bytes).unwrap();
    assert!(
        fixture
            .stage(
                &prepared,
                &mut catalog,
                Some(&approval),
                &AtomicBool::new(false)
            )
            .unwrap_err()
            .contains("verification")
    );
    assert_eq!(catalog.bytes_resident_on(&fixture.node.id), 0);
    assert!(fixture.receiver.list().unwrap().is_empty());
    assert_eq!(
        fs::read_dir(fixture.receiver.root().join(".receiving"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn existing_but_corrupt_receiver_is_not_credited_again() {
    let fixture = Fixture::new("qwen2_f32");
    let prepared = fixture.prepare(middle());
    let mut catalog = ReplicaCatalog::new();
    prepared.register_source(&mut catalog, 0).unwrap();
    let approval = fixture.approval(&prepared);
    let id = fixture
        .stage(
            &prepared,
            &mut catalog,
            Some(&approval),
            &AtomicBool::new(false),
        )
        .unwrap();
    catalog.remove(&fixture.node.id, &id);
    let bank = prepared
        .transport()
        .files
        .iter()
        .find(|file| file.path.starts_with("banks/"))
        .unwrap();
    fs::write(
        fixture.receiver.package_dir(&id).unwrap().join(&bank.path),
        b"damaged",
    )
    .unwrap();
    assert!(
        fixture
            .stage(
                &prepared,
                &mut catalog,
                Some(&approval),
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert_eq!(catalog.bytes_resident_on(&fixture.node.id), 0);
}

#[test]
fn resume_uses_chunks_the_receiver_already_verified() {
    let fixture = Fixture::new("qwen2_f32");
    let prepared = fixture.prepare(middle());
    let mut catalog = ReplicaCatalog::new();
    prepared.register_source(&mut catalog, 0).unwrap();
    let approval = fixture.approval(&prepared);
    let transport = prepared.transport();
    fixture.receiver.receive_begin(transport).unwrap();
    let (index, bank) = transport
        .files
        .iter()
        .enumerate()
        .find(|(_, file)| file.path.starts_with("banks/"))
        .unwrap();
    assert_eq!(bank.chunks.len(), 1);
    let path = fixture
        .packages
        .package_dir(&transport.checkpoint_id)
        .unwrap()
        .join(&bank.path);
    let original = fs::read(&path).unwrap();
    fixture
        .receiver
        .receive_chunk(&transport.checkpoint_id, index as u32, 0, &original)
        .unwrap();
    fs::write(path, b"the source chunk is no longer readable").unwrap();
    fixture
        .stage(
            &prepared,
            &mut catalog,
            Some(&approval),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(
        fs::read(
            fixture
                .receiver
                .package_dir(&transport.checkpoint_id)
                .unwrap()
                .join(&bank.path)
        )
        .unwrap(),
        original
    );
    assert_eq!(catalog.resident_on(&fixture.node.id).len(), 1);
}

#[test]
fn changing_a_package_invalidates_the_snapshot_and_its_approval() {
    let fixture = Fixture::new("qwen2_f32");
    let prepared = fixture.prepare(middle());
    let mut catalog = ReplicaCatalog::new();
    prepared.register_source(&mut catalog, 0).unwrap();
    let approval = fixture.approval(&prepared);
    let id = &prepared.transport().checkpoint_id;
    let path = fixture
        .packages
        .package_dir(id)
        .unwrap()
        .join("manifest.json");
    let mut manifest = fixture.packages.read(id).unwrap();
    manifest.name = "z".repeat(manifest.name.len());
    fs::write(path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    assert!(
        fixture
            .stage(
                &prepared,
                &mut catalog,
                Some(&approval),
                &AtomicBool::new(false)
            )
            .unwrap_err()
            .contains("changed")
    );
    assert!(!fixture.receiver.root().exists());
    let updated = fixture.prepare(middle());
    assert_eq!(
        updated.transport().total_bytes,
        prepared.transport().total_bytes
    );
    assert_ne!(
        updated.placement_digest(&fixture.node.id),
        prepared.placement_digest(&fixture.node.id)
    );
    updated.register_source(&mut catalog, 1).unwrap();
    assert!(
        fixture
            .stage(
                &updated,
                &mut catalog,
                Some(&approval),
                &AtomicBool::new(false)
            )
            .unwrap_err()
            .contains("approve")
    );
    assert!(!fixture.receiver.root().exists());
}

#[test]
fn catalog_selected_sources_are_not_silently_replaced() {
    let fixture = Fixture::new("qwen2_f32");
    let prepared = fixture.prepare(middle());
    let mut catalog = ReplicaCatalog::new();
    prepared.register_source(&mut catalog, 0).unwrap();
    catalog.record(Replica::new(
        ContentRef::new(
            &prepared.transport().checkpoint_id,
            ContentKind::KtbWeightBank,
            prepared.transport().total_bytes,
        ),
        NodeId("faster-source".into()),
        ResidencyTier::SystemRam,
        0,
    ));
    let approval = fixture.approval(&prepared);
    let error = fixture
        .stage(
            &prepared,
            &mut catalog,
            Some(&approval),
            &AtomicBool::new(false),
        )
        .unwrap_err();
    assert!(error.contains("selected a source unavailable"));
    assert!(!fixture.receiver.root().exists());
    assert!(!catalog.is_resident(&fixture.node.id, &prepared.transport().checkpoint_id));
}

#[test]
fn malformed_or_overflowing_storage_budgets_are_errors() {
    let mut fixture = Fixture::new("qwen2_f32");
    let prepared = fixture.prepare(middle());
    let mut catalog = ReplicaCatalog::new();
    prepared.register_source(&mut catalog, 0).unwrap();
    for capacity in [f64::NAN, f64::INFINITY, -1.0] {
        fixture.offer.nvme_gb = capacity;
        assert!(
            prepared
                .staging_plan(&catalog, &fixture.node, &fixture.offer, None)
                .is_err()
        );
    }
    fixture.offer.nvme_gb = 1.0;
    fixture.node.memory.vram_gb = -1.0;
    assert!(
        prepared
            .staging_plan(&catalog, &fixture.node, &fixture.offer, None)
            .is_err()
    );
    fixture.node.memory.vram_gb = 0.0;
    for (id, bytes) in [("large", u64::MAX), ("one", 1)] {
        catalog.record(Replica::new(
            ContentRef::new(id, ContentKind::Artifact, bytes),
            fixture.node.id.clone(),
            ResidencyTier::LocalNvme,
            0,
        ));
    }
    assert!(
        prepared
            .staging_plan(&catalog, &fixture.node, &fixture.offer, None)
            .unwrap_err()
            .contains("overflow")
    );
}

#[test]
fn stale_residency_does_not_bypass_approval_or_storage_to_recreate_a_package() {
    let fixture = Fixture::new("qwen2_f32");
    let prepared = fixture.prepare(middle());
    let mut catalog = ReplicaCatalog::new();
    prepared.register_source(&mut catalog, 0).unwrap();
    catalog.record(Replica::new(
        ContentRef::new(
            &prepared.transport().checkpoint_id,
            ContentKind::KtbWeightBank,
            prepared.transport().total_bytes,
        ),
        fixture.node.id.clone(),
        ResidencyTier::LocalNvme,
        0,
    ));
    assert!(
        fixture
            .stage(&prepared, &mut catalog, None, &AtomicBool::new(false))
            .is_err()
    );
    assert!(!fixture.receiver.root().exists());
}

#[test]
fn boundary_weights_are_not_staged_to_middle_only_workers() {
    let mut fixture = Fixture::new("qwen2_f32");
    fixture.node.trust_tier = kadmium_core::NodeTrustTier::UntrustedAuditedMiddleLayersOnly;
    for plan in [
        SlicePlan {
            layers: 0..1,
            embedding: true,
            head: false,
        },
        SlicePlan {
            layers: 2..3,
            embedding: false,
            head: true,
        },
    ] {
        let prepared = fixture.prepare(plan);
        let mut catalog = ReplicaCatalog::new();
        prepared.register_source(&mut catalog, 0).unwrap();
        assert!(
            prepared
                .staging_plan(&catalog, &fixture.node, &fixture.offer, None)
                .unwrap_err()
                .contains("trusted-core")
        );
    }
}

#[test]
fn cancellation_and_invalid_selections_do_not_stage() {
    let fixture = Fixture::new("qwen2_f32");
    let prepared = fixture.prepare(middle());
    let mut catalog = ReplicaCatalog::new();
    prepared.register_source(&mut catalog, 0).unwrap();
    let approval = fixture.approval(&prepared);
    assert!(
        fixture
            .stage(
                &prepared,
                &mut catalog,
                Some(&approval),
                &AtomicBool::new(true)
            )
            .unwrap_err()
            .contains("cancelled")
    );
    assert!(!fixture.receiver.root().exists());
    for plan in [
        SlicePlan {
            layers: std::ops::Range { start: 2, end: 1 },
            embedding: false,
            head: false,
        },
        SlicePlan {
            layers: 0..4,
            embedding: true,
            head: true,
        },
        SlicePlan {
            layers: 1..1,
            embedding: false,
            head: false,
        },
    ] {
        assert!(
            PreparedSlice::prepare(
                &fixture.source,
                &fixture.model_id,
                plan,
                fixture.packages.clone(),
                NodeId("source".into()),
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
    let name = "model.embed_tokens.weight".to_string();
    for names in [vec![], vec![name.clone(), name], vec!["missing".into()]] {
        assert!(
            fixture
                .source
                .select_tensors(
                    &fixture.model_id,
                    &names,
                    &fixture.packages,
                    &AtomicBool::new(false)
                )
                .is_err()
        );
    }
}
