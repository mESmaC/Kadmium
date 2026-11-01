use kadmium_core::NodeId;
use kadmium_grid::{
    ContentKind, ContentRef, PlacementRule, Replica, ReplicaCatalog, ResidencyTier, StagingJob,
};
use std::collections::BTreeMap;

#[test]
fn staging_storage_lean_vectors() {
    let worker = NodeId("worker".into());
    let source = NodeId("source".into());
    let mut checked = 0;
    for line in include_str!("fixtures/staging_storage_lean_vectors.txt")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 9);
        let numbers: Vec<u64> = fields[..8]
            .iter()
            .map(|field| field.parse().unwrap())
            .collect();
        let [
            resident,
            other,
            payload,
            metadata,
            receipts,
            capacity,
            reserved,
            total,
        ] = numbers[..]
        else {
            panic!("invalid storage vector");
        };
        let accepted: bool = fields[8].parse().unwrap();
        assert_eq!(
            payload
                .checked_add(metadata)
                .unwrap()
                .checked_add(receipts)
                .unwrap(),
            reserved
        );
        let mut catalog = ReplicaCatalog::new();
        catalog.record(Replica::new(
            ContentRef::named(ContentKind::KtbWeightBank, "resident", resident),
            worker.clone(),
            ResidencyTier::LocalNvme,
            0,
        ));
        let pending = ContentRef::named(ContentKind::KtbWeightBank, "pending", reserved);
        catalog.record(Replica::new(
            pending.clone(),
            source.clone(),
            ResidencyTier::LocalNvme,
            0,
        ));
        for (content, bytes) in [
            (
                ContentRef::named(ContentKind::KtbWeightBank, "other", other),
                other,
            ),
            (pending.clone(), reserved),
        ] {
            catalog
                .reserve_staging(&StagingJob {
                    content,
                    source_node: source.clone(),
                    destination_node: worker.clone(),
                    target_tier: ResidencyTier::LocalNvme,
                    bytes,
                    pin: true,
                })
                .unwrap();
        }
        assert_eq!(catalog.staging_bytes_on(&worker, None).unwrap(), total);
        assert_eq!(
            catalog
                .staging_bytes_on(&worker, Some(&pending.content_id))
                .unwrap(),
            resident + other
        );
        let rule = PlacementRule {
            content: pending,
            min_replicas: 2,
            preferred_tier: ResidencyTier::LocalNvme,
            candidate_nodes: vec![worker.clone()],
            pin: true,
        };
        let jobs = catalog
            .plan_staging_with_capacity(&[rule], &BTreeMap::from([(worker.clone(), capacity)]));
        assert_eq!(!jobs.is_empty(), accepted, "{line}");
        if let Some(job) = jobs.first() {
            catalog.apply_staging(job, 1);
            assert_eq!(catalog.staging_bytes_on(&worker, None).unwrap(), total);
        }
        checked += 1;
    }
    assert_eq!(checked, 6);
}
