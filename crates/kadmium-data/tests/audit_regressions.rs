use kadmium_data::{
    DecentralizedStoragePoolEngine as Storage, KtbBankPacker, KtbBankReader, TensorEncoding,
};

#[test]
fn malformed_ktb_headers_and_tensor_ranges_never_unwind() {
    let mut packer = KtbBankPacker::new("test");
    packer.add_tensor_bank("tensor", 0, None, TensorEncoding::Bf16, vec![1, 2, 3, 4]);
    let valid = packer.pack_to_bytes();
    let reader = KtbBankReader::from_slice(&valid).unwrap();
    assert_eq!(
        reader
            .get_verified_tensor_slice(&reader.header.entries[0])
            .unwrap(),
        [1, 2, 3, 4]
    );
    for len in [0, 1, 10, 65535, 65536, 131072, u32::MAX] {
        let mut raw = valid.clone();
        raw[6..10].copy_from_slice(&len.to_le_bytes());
        assert!(KtbBankReader::from_slice(&raw).is_err());
    }
    for len in [0, 1, 9, 100, 65535] {
        assert!(KtbBankReader::from_slice(&valid[..len]).is_err());
    }
    for (offset, len) in [
        (0, 4),
        (65537, 4),
        (u64::MAX, 4),
        (65536, u64::MAX),
        (131072, 1),
    ] {
        let mut header = reader.header.clone();
        header.entries[0].page_offset_bytes = offset;
        header.entries[0].payload_bytes = len;
        assert!(
            reader
                .get_verified_tensor_slice(&header.entries[0])
                .is_err()
        );
        let json = serde_json::to_vec(&header).unwrap();
        let mut raw = valid.clone();
        raw[6..10].copy_from_slice(&(json.len() as u32).to_le_bytes());
        raw[10..10 + json.len()].copy_from_slice(&json);
        assert!(KtbBankReader::from_slice(&raw).is_err());
    }
    // Bounded deterministic fuzzing of length fields, without huge allocations.
    let mut seed = 0x12345678u32;
    for _ in 0..256 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let mut raw = valid.clone();
        raw[6..10].copy_from_slice(&seed.to_le_bytes());
        assert!(KtbBankReader::from_slice(&raw).is_err());
    }
}

#[test]
fn live_storage_never_claims_synthetic_recovery_or_implicit_copies() {
    for count in [0, 1, 2, 10] {
        let nodes = kadmium_core::sample_node_pool(count);
        let mut anchors =
            Storage::verify_bucket_grounding_and_mass_dropout_recovery(8, 128).configured_anchors;
        anchors[0].anchor_uri = "nas://unavailable/audit".into();
        for configured in [&[][..], anchors.as_slice()] {
            let object =
                Storage::create_bucket_object("audit", "audit", "test", 0.01, 8, 2, None, &nodes)
                    .unwrap();
            let report = Storage::evaluate_live_cluster_bucket(&nodes, &[], &[object], configured);
            assert_eq!(report.total_participating_nodes, count);
            assert_eq!(report.report_kind, "live_placement_estimate");
            assert!(!report.durability_verified);
            assert!(
                !report
                    .erasure_coding_verification
                    .exact_bit_identical_recovery
            );
            assert!(!report.bucket_grounding.grounded_survives_mass_dropout);
            assert_eq!(
                report.bucket_grounding.configured_anchors.len(),
                configured.len()
            );
            assert!(
                report
                    .bucket_grounding
                    .configured_anchors
                    .iter()
                    .all(|a| a.merkle_grounded_coverage_pct == 0.0)
            );
            assert!(report.protection_status.contains("Unverified"));
            if count < 10 {
                assert!(
                    report
                        .protection_status
                        .contains("insufficient distinct holders")
                );
            }
        }
    }
    let benchmark = Storage::run_pool_storage_benchmark(10);
    assert_eq!(benchmark.report_kind, "synthetic_benchmark");
    assert!(!benchmark.durability_verified);
    assert!(
        benchmark
            .erasure_coding_verification
            .exact_bit_identical_recovery
    );
}
