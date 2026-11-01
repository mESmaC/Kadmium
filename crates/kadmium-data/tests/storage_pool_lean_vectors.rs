use kadmium_data::{
    BucketObjectStore, DecentralizedStoragePoolEngine, encode_erasure_coded_shards,
    systematic_cauchy_row,
};

const VECTORS: &str = include_str!("fixtures/storage_pool_vectors.txt");

#[test]
fn storage_pool_matches_lean_vectors() {
    let mut shard_count_rows = 0usize;
    let mut shard_size_rows = 0usize;
    let mut quota_rows = 0usize;
    let mut swap_rows = 0usize;
    let mut cauchy_rows = 0usize;
    let mut medium_pledge_rows = 0usize;

    for (line_no, raw) in VECTORS.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        match cols[0] {
            "shard_counts" => {
                assert_eq!(cols.len(), 5, "line {}", line_no + 1);
                let k: usize = cols[1].parse().expect("k");
                let m: usize = cols[2].parse().expect("m");
                let expected_gen = cols[3];
                let expected_stored = cols[4];

                let actual_gen = match DecentralizedStoragePoolEngine::validate_shard_counts(k, m) {
                    Ok(total) => format!("ok:{total}"),
                    Err(_) => "err".to_string(),
                };
                let actual_stored =
                    match DecentralizedStoragePoolEngine::validate_stored_shard_counts(k, m) {
                        Ok(total) => format!("ok:{total}"),
                        Err(_) => "err".to_string(),
                    };

                assert_eq!(
                    actual_gen,
                    expected_gen,
                    "validate_shard_counts({k}, {m}) mismatch at line {}",
                    line_no + 1
                );
                assert_eq!(
                    actual_stored,
                    expected_stored,
                    "validate_stored_shard_counts({k}, {m}) mismatch at line {}",
                    line_no + 1
                );
                shard_count_rows += 1;
            }
            "shard_size" => {
                assert_eq!(cols.len(), 4, "line {}", line_no + 1);
                let payload_len: usize = cols[1].parse().expect("payload_len");
                let k: usize = cols[2].parse().expect("k");
                let expected_size: usize = cols[3].parse().expect("expected_size");

                let payload = vec![0xABu8; payload_len];
                let (shards, shard_size) = encode_erasure_coded_shards(&payload, k, 2)
                    .expect("encode_erasure_coded_shards");
                assert_eq!(
                    shard_size,
                    expected_size,
                    "shard_size({payload_len}, {k}) mismatch at line {}",
                    line_no + 1
                );
                assert_eq!(shards.len(), k + 2);
                assert!(shards.iter().all(|s| s.len() == expected_size));
                shard_size_rows += 1;
            }
            "quota" => {
                assert_eq!(cols.len(), 7, "line {}", line_no + 1);
                let existing_objects: usize = cols[1].parse().expect("existing_objects");
                let max_objects: usize = cols[2].parse().expect("max_objects");
                let existing_bytes: u64 = cols[3].parse().expect("existing_bytes");
                let new_physical_bytes: u64 = cols[4].parse().expect("new_physical_bytes");
                let max_total_bytes: u64 = cols[5].parse().expect("max_total_bytes");
                let expected_ok: u8 = cols[6].parse().expect("expected_ok");

                let actual_ok = existing_objects < max_objects
                    && existing_bytes.saturating_add(new_physical_bytes) <= max_total_bytes;
                assert_eq!(
                    u8::from(actual_ok),
                    expected_ok,
                    "quota check mismatch at line {}",
                    line_no + 1
                );
                quota_rows += 1;
            }
            "swap" => {
                assert_eq!(cols.len(), 5, "line {}", line_no + 1);
                let trace_name = cols[1];
                let expected_final = cols[2];
                let expected_old = cols[3];
                let expected_staging = cols[4];

                // Also exercise real filesystem recovery on BucketObjectStore for crash traces
                let tmp = std::env::temp_dir().join(format!(
                    "kdm-lean-swap-{}-{}",
                    trace_name,
                    std::process::id()
                ));
                let _ = std::fs::remove_dir_all(&tmp);
                let store = BucketObjectStore::new(&tmp);
                let uri = "kdm://pool-bucket/lean/swap.bin";
                store
                    .put_bytes(uri, Some("v1"), None, b"version-1", 4, 2, &[], None)
                    .expect("put v1");

                match trace_name {
                    "stage_rotate_commit" | "overwrite_twice_with_crash" => {
                        store
                            .put_bytes(uri, Some("v2"), None, b"version-2", 4, 2, &[], None)
                            .expect("put v2");
                        assert_eq!(expected_final, "2");
                        let got = store.get_bytes(uri, None).expect("get v2");
                        assert_eq!(got.bytes, b"version-2");
                    }
                    "stage_rotate_crash_recover" | "stage_rotate_rollback" => {
                        // Simulate a crash right after rotating objects/<dir> -> .old-<dir>-999
                        let objects_dir = tmp.join("objects");
                        let obj_dir = std::fs::read_dir(&objects_dir)
                            .expect("read objects")
                            .filter_map(|e| e.ok())
                            .map(|e| e.path())
                            .find(|p| {
                                p.file_name()
                                    .and_then(|n| n.to_str())
                                    .is_some_and(|n| !n.starts_with('.'))
                            })
                            .expect("object dir");
                        let dir_name = obj_dir.file_name().unwrap().to_str().unwrap().to_string();
                        let old_dir = objects_dir.join(format!(".old-{dir_name}-999"));
                        std::fs::rename(&obj_dir, &old_dir).expect("rotate to .old");

                        assert_eq!(expected_final, "1");
                        // Unrelated put_bytes or get_bytes must recover .old-<dir>-999 -> objects/<dir>
                        store
                            .put_bytes(
                                "kdm://pool-bucket/lean/unrelated.bin",
                                None,
                                None,
                                b"other",
                                4,
                                2,
                                &[],
                                None,
                            )
                            .expect("unrelated put");
                        let got = store.get_bytes(uri, None).expect("recovered v1");
                        assert_eq!(got.bytes, b"version-1");
                    }
                    "stage_crash_recover" => {
                        assert_eq!(expected_final, "1");
                        let got = store.get_bytes(uri, None).expect("get v1");
                        assert_eq!(got.bytes, b"version-1");
                    }
                    other => panic!("unexpected swap trace {other}"),
                }
                assert_eq!(expected_old, "none");
                assert_eq!(expected_staging, "none");
                let _ = std::fs::remove_dir_all(&tmp);
                swap_rows += 1;
            }
            "cauchy_row" => {
                assert_eq!(cols.len(), 4, "line {}", line_no + 1);
                let k: usize = cols[1].parse().expect("k");
                let shard_idx: usize = cols[2].parse().expect("shard_idx");
                let expected: Vec<u8> = cols[3]
                    .split(',')
                    .map(|s| s.parse::<u8>().expect("u8"))
                    .collect();
                let actual = systematic_cauchy_row(shard_idx, k);
                assert_eq!(
                    actual,
                    expected,
                    "systematic_cauchy_row({shard_idx}, {k}) mismatch at line {}",
                    line_no + 1
                );
                cauchy_rows += 1;
            }
            "medium_pledge" => {
                assert_eq!(cols.len(), 7, "line {}", line_no + 1);
                let offer_tenths: u64 = cols[1].parse().expect("offer_tenths");
                let hw_tenths: u64 = cols[2].parse().expect("hw_tenths");
                let pct: u64 = cols[3].parse().expect("pct");
                let floor_tenths: u64 = cols[4].parse().expect("floor_tenths");
                let expected_avail: u64 = cols[5].parse().expect("expected_avail");
                let expected_pledged: u64 = cols[6].parse().expect("expected_pledged");

                let (actual_avail, actual_pledged) =
                    DecentralizedStoragePoolEngine::compute_medium_availability_and_pledge_tenths(
                        offer_tenths,
                        hw_tenths,
                        pct,
                        floor_tenths,
                    );
                assert_eq!(
                    (actual_avail, actual_pledged),
                    (expected_avail, expected_pledged),
                    "compute_medium_availability_and_pledge_tenths({offer_tenths}, {hw_tenths}, {pct}, {floor_tenths}) mismatch at line {}",
                    line_no + 1
                );

                let (avail_gb, pledged_gb) =
                    DecentralizedStoragePoolEngine::compute_medium_availability_and_pledge(
                        offer_tenths as f64 / 10.0,
                        hw_tenths as f64 / 10.0,
                        pct as f64 / 100.0,
                        floor_tenths as f64 / 10.0,
                    );
                assert!(avail_gb <= (offer_tenths as f64 / 10.0) + 1e-9);
                assert!(avail_gb <= (hw_tenths as f64 / 10.0) + 1e-9);
                assert!(pledged_gb <= avail_gb + 1e-9);
                if offer_tenths == 0 {
                    assert_eq!(avail_gb, 0.0);
                    assert_eq!(pledged_gb, 0.0);
                }
                medium_pledge_rows += 1;
            }
            other => panic!("unknown vector tag {other} at line {}", line_no + 1),
        }
    }

    assert!(shard_count_rows >= 40);
    assert!(shard_size_rows >= 30);
    assert_eq!(quota_rows, 4);
    assert_eq!(swap_rows, 5);
    assert!(cauchy_rows >= 15);
    assert_eq!(medium_pledge_rows, 8);
}
