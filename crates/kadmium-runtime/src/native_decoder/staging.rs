//! Slice packages and synchronous checkpoint-library staging. No worker socket or UI uses
//! this adapter yet. Disk placement uses the caller's replica catalog and storage offer;
//! execution memory, full-context KV reservations and layer assignment remain separate work.

use super::{DecoderConfig, SlicePlan};
use kadmium_core::{HardwareNode, NodeId, ResourceOffer};
use kadmium_data::SafetensorsManifest;
use kadmium_data::checkpoint::transport::TransportManifest;
use kadmium_data::checkpoint::{CheckpointLibrary, CheckpointManifest};
use kadmium_grid::{
    ContentKind, ContentRef, PlacementRule, PlanApproval, Replica, ReplicaCatalog, ResidencyTier,
    StagingPlan, fnv1a_64,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone)]
pub struct PreparedSlice {
    model_id: String,
    plan: SlicePlan,
    num_layers: usize,
    library: CheckpointLibrary,
    transport: TransportManifest,
    storage_bytes: u64,
    source_node: NodeId,
}

pub struct SliceDestination<'a> {
    pub node: &'a HardwareNode,
    pub offer: &'a ResourceOffer,
    pub approval: Option<&'a PlanApproval>,
    pub library: &'a CheckpointLibrary,
}

impl PreparedSlice {
    pub fn prepare(
        source: &CheckpointLibrary,
        model_id: &str,
        plan: SlicePlan,
        library: CheckpointLibrary,
        source_node: NodeId,
        cancelled: &AtomicBool,
    ) -> Result<Self, String> {
        let directory = source.package_dir(model_id)?.join("source");
        let config =
            DecoderConfig::from_model_dir(&directory).map_err(|error| error.to_string())?;
        let manifest = SafetensorsManifest::open(&directory).map_err(|error| error.to_string())?;
        let names = plan
            .tensor_names(&manifest, &config)
            .map_err(|error| error.to_string())?;
        let selected = source.select_tensors(model_id, &names, &library, cancelled)?;
        let transport = library.transport(&selected.id)?;
        let storage_bytes = transport.storage_bytes()?;
        Ok(Self {
            model_id: model_id.into(),
            plan,
            num_layers: config.num_layers,
            library,
            transport,
            storage_bytes,
            source_node,
        })
    }

    pub fn plan(&self) -> &SlicePlan {
        &self.plan
    }

    pub fn transport(&self) -> &TransportManifest {
        &self.transport
    }

    pub fn storage_bytes(&self) -> u64 {
        self.storage_bytes
    }

    pub fn register_source(
        &self,
        catalog: &mut ReplicaCatalog,
        now_unix_ms: u64,
    ) -> Result<(), String> {
        self.library.verify(&self.transport.checkpoint_id)?;
        if self.library.transport(&self.transport.checkpoint_id)? != self.transport {
            return Err("The source slice changed after preparation".into());
        }
        catalog.record(Replica::new(
            self.content(),
            self.source_node.clone(),
            ResidencyTier::LocalNvme,
            now_unix_ms,
        ));
        Ok(())
    }

    pub fn placement_digest(&self, destination: &NodeId) -> u64 {
        let mut bytes = Vec::new();
        for field in [
            &self.model_id,
            &self.transport.checkpoint_id,
            &self.source_node.0,
            &destination.0,
        ] {
            bytes.extend_from_slice(&(field.len() as u64).to_le_bytes());
            bytes.extend_from_slice(field.as_bytes());
        }
        for value in [
            self.plan.layers.start as u64,
            self.plan.layers.end as u64,
            self.transport.total_bytes,
            self.storage_bytes,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[u8::from(self.plan.embedding), u8::from(self.plan.head)]);
        for file in &self.transport.files {
            for field in [&file.path, &file.sha256] {
                bytes.extend_from_slice(&(field.len() as u64).to_le_bytes());
                bytes.extend_from_slice(field.as_bytes());
            }
            bytes.extend_from_slice(&file.bytes.to_le_bytes());
        }
        fnv1a_64(&bytes)
    }

    pub fn staging_plan(
        &self,
        catalog: &ReplicaCatalog,
        destination: &HardwareNode,
        offer: &ResourceOffer,
        approval: Option<&PlanApproval>,
    ) -> Result<StagingPlan, String> {
        if destination.id == self.source_node {
            return Err("A worker slice destination must differ from its source node".into());
        }
        if (self.plan.embedding
            || self.plan.head
            || self.plan.layers.start == 0
            || self.plan.layers.end == self.num_layers)
            && !destination.trust_tier.can_host_boundary_ingress_egress()
        {
            return Err("Boundary slices require a trusted-core worker".into());
        }
        if [
            offer.fast_memory_gb,
            offer.system_ram_gb,
            offer.nvme_gb,
            offer.reserved_ram_gb,
            destination.memory.vram_gb,
            destination.memory.unified_mem_gb,
            destination.memory.vram_gb + destination.memory.unified_mem_gb,
            destination.memory.system_ram_gb,
            destination.memory.nvme_cache_gb,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(
                "Worker resource offers and capacities must be finite and nonnegative".into(),
            );
        }
        let offer = offer
            .clone()
            .clamped_to(destination)
            .under_polite_state(destination, destination.polite_state);
        if !offer.nvme_gb.is_finite() {
            return Err("The worker storage capacity must be finite".into());
        }
        let capacity = (offer.nvme_gb * 1_073_741_824.0) as u64;
        let holders = catalog.holders(&self.transport.checkpoint_id);
        if !holders
            .iter()
            .any(|replica| replica.node_id == self.source_node && replica.content == self.content())
        {
            return Err(
                "Register the verified source slice in the replica catalog before staging".into(),
            );
        }
        if holders
            .iter()
            .any(|replica| replica.node_id == destination.id)
        {
            if holders.iter().any(|replica| {
                replica.node_id == destination.id && replica.content != self.content()
            }) {
                return Err(
                    "The catalog's destination slice metadata does not match this package".into(),
                );
            }
            return Ok(StagingPlan::default());
        }
        let resident =
            catalog.staging_bytes_on(&destination.id, Some(&self.transport.checkpoint_id))?;
        let required = resident
            .checked_add(self.storage_bytes)
            .ok_or("Worker storage size overflow")?;
        if required > capacity {
            return Err(format!(
                "Worker {} needs {required} storage bytes; its resource offer allows {capacity}",
                destination.id.0
            ));
        }
        let approved = approval.is_some_and(|approval| {
            approval.node == destination.id
                && approval.covers(&self.model_id, self.placement_digest(&destination.id))
        });
        let needs = if offer.requires_approval && !approved {
            BTreeSet::from([destination.id.clone()])
        } else {
            BTreeSet::new()
        };
        let replicas = u32::try_from(holders.len())
            .ok()
            .and_then(|count| count.checked_add(1))
            .ok_or("Too many slice replicas")?;
        let rule = PlacementRule {
            content: self.content(),
            min_replicas: replicas,
            preferred_tier: ResidencyTier::LocalNvme,
            candidate_nodes: vec![destination.id.clone()],
            pin: true,
        };
        let plan = catalog.plan_staging_gated(
            &[rule],
            &BTreeMap::from([(destination.id.clone(), capacity)]),
            &needs,
        );
        if plan.ready.len() + plan.awaiting_approval.len() != 1 {
            return Err("The replica catalog could not stage the requested worker slice".into());
        }
        if plan
            .ready
            .iter()
            .chain(&plan.awaiting_approval)
            .any(|job| job.source_node != self.source_node)
        {
            return Err(
                "The catalog selected a source unavailable to this local library adapter".into(),
            );
        }
        Ok(plan)
    }

    pub fn stage_to(
        &self,
        catalog: &mut ReplicaCatalog,
        destination: SliceDestination<'_>,
        cancelled: &AtomicBool,
        now_unix_ms: u64,
    ) -> Result<CheckpointManifest, String> {
        if cancelled.load(Ordering::Acquire) {
            return Err("Checkpoint staging cancelled".into());
        }
        let plan = self.staging_plan(
            catalog,
            destination.node,
            destination.offer,
            destination.approval,
        )?;
        if !plan.awaiting_approval.is_empty() {
            return Err(format!(
                "Worker {} must approve this slice placement before staging",
                destination.node.id.0
            ));
        }
        if plan.ready.is_empty() {
            let received = destination.library.verify(&self.transport.checkpoint_id)?;
            if destination.library.transport(&received.id)? != self.transport {
                return Err(
                    "The catalog's resident slice does not match the worker package".into(),
                );
            }
            return Ok(received);
        }
        for job in &plan.ready {
            catalog.reserve_staging(job)?;
        }
        let received = self
            .library
            .copy_to(&self.transport, destination.library, cancelled)?;
        for job in &plan.ready {
            catalog.apply_staging(job, now_unix_ms);
        }
        Ok(received)
    }

    fn content(&self) -> ContentRef {
        ContentRef::new(
            &self.transport.checkpoint_id,
            ContentKind::KtbWeightBank,
            self.storage_bytes,
        )
    }
}
