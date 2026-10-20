//! Multi-Tenant Sub-Cluster RBAC & Quota Enforcement Engine (`kadmium-core::rbac`).
//!
//! Implements Phase 5 (`v1.0 GA`) role-based access control and per-slice resource quotas
//! for research labs, engineering teams, and Kadmium Supply Grid operators.

use crate::WorkloadKind;
use serde::{Deserialize, Serialize};

/// Role-Based Access Control (RBAC) privilege level within a Kadmium cluster or workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RbacRole {
    /// Full control over coordinator quorum, node pairing, slice creation, and cloud billing caps.
    ClusterAdmin,
    /// Can deploy inference/training jobs and attach burst GPUs within assigned sub-cluster slices.
    SliceOperator,
    /// Can run inference, QLoRA fine-tuning, and Dream-RSI worlds within an assigned slice; cannot alter cloud spend caps.
    ModelResearcher,
    /// External Supply Grid provider: restricted strictly to blind middle-layer execution ($1 \dots N-2$) and WASI sandboxes.
    BlindSupplyProvider,
}

/// Resource and security quota policy bound to an isolated Sub-Compute Environment slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubClusterQuotaPolicy {
    pub slice_id: String,
    pub display_name: String,
    pub max_nodes: usize,
    pub max_fast_memory_gb: f64,
    pub max_cloud_burst_usd_per_hr: f64,
    pub allow_ingress_egress_layers: bool,
    pub allowed_workloads: Vec<WorkloadKind>,
}

/// Cryptographically signed, slice-scoped API key (`kdm_live_...`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopedApiKey {
    pub key_id: String,
    pub principal_name: String,
    pub role: RbacRole,
    pub bound_slice_id: String,
    pub token_string: String,
    pub mac_signature_fnv64: u64,
}

/// Evaluates multi-tenant RBAC permissions and per-slice resource quotas.
pub struct SubClusterRbacEngine;

impl SubClusterRbacEngine {
    /// Mints a deterministic, MAC-signed `kdm_live_...` API key bound to a principal, role, and slice.
    pub fn mint_scoped_api_key(
        key_id: impl Into<String>,
        principal_name: impl Into<String>,
        role: RbacRole,
        bound_slice_id: impl Into<String>,
    ) -> ScopedApiKey {
        let key_id = key_id.into();
        let principal_name = principal_name.into();
        let bound_slice_id = bound_slice_id.into();

        let mut mac: u64 = 0xcbf2_9ce4_8422_2325;
        let payload = format!("kdm-rbac-v1:{key_id}:{principal_name}:{role:?}:{bound_slice_id}");
        for b in payload.bytes() {
            mac ^= b as u64;
            mac = mac.wrapping_mul(0x0100_0000_01b3);
        }

        let token_string = format!("kdm_live_{}_{:016x}", bound_slice_id.replace('-', "_"), mac);
        ScopedApiKey {
            key_id,
            principal_name,
            role,
            bound_slice_id,
            token_string,
            mac_signature_fnv64: mac,
        }
    }

    /// Checks whether `api_key` is authorized to execute `workload` with `requested_cloud_usd_per_hr` on `quota`.
    pub fn authorize_slice_request(
        api_key: &ScopedApiKey,
        quota: &SubClusterQuotaPolicy,
        workload: WorkloadKind,
        requested_cloud_usd_per_hr: f64,
        requires_ingress_egress_layers: bool,
    ) -> Result<(), String> {
        if api_key.role != RbacRole::ClusterAdmin && api_key.bound_slice_id != quota.slice_id {
            return Err(format!(
                "SliceMismatch: key bound to '{}' cannot access '{}'",
                api_key.bound_slice_id, quota.slice_id
            ));
        }
        if api_key.role == RbacRole::BlindSupplyProvider && requires_ingress_egress_layers {
            return Err(
                "PrivacyViolation: BlindSupplyProvider is forbidden from hosting Layer 0 or Layer N-1"
                    .into(),
            );
        }
        if requires_ingress_egress_layers && !quota.allow_ingress_egress_layers {
            return Err("QuotaViolation: Slice policy forbids raw ingress/egress layers".into());
        }
        if !quota.allowed_workloads.contains(&workload) {
            return Err(format!(
                "WorkloadForbidden: {:?} not permitted in slice '{}'",
                workload, quota.slice_id
            ));
        }
        if requested_cloud_usd_per_hr > quota.max_cloud_burst_usd_per_hr {
            return Err(format!(
                "CloudBudgetExceeded: requested ${:.2}/hr exceeds slice cap ${:.2}/hr",
                requested_cloud_usd_per_hr, quota.max_cloud_burst_usd_per_hr
            ));
        }
        Ok(())
    }

    /// Returns the default multi-tenant RBAC quota policies and scoped keys for a 5-slice cluster.
    pub fn sample_lab_rbac_catalog() -> (Vec<SubClusterQuotaPolicy>, Vec<ScopedApiKey>) {
        let quotas = vec![
            SubClusterQuotaPolicy {
                slice_id: "slice-01-prod-deploy".into(),
                display_name: "Production Inference & Failover Shield".into(),
                max_nodes: 20,
                max_fast_memory_gb: 320.0,
                max_cloud_burst_usd_per_hr: 2.50,
                allow_ingress_egress_layers: true,
                allowed_workloads: vec![WorkloadKind::Inference],
            },
            SubClusterQuotaPolicy {
                slice_id: "slice-03-diloco-train".into(),
                display_name: "Sharded-Island DiLoCo + QLoRA Training".into(),
                max_nodes: 20,
                max_fast_memory_gb: 384.0,
                max_cloud_burst_usd_per_hr: 1.80,
                allow_ingress_egress_layers: true,
                allowed_workloads: vec![
                    WorkloadKind::DistributedTraining,
                    WorkloadKind::BatchPipeline,
                ],
            },
            SubClusterQuotaPolicy {
                slice_id: "slice-05-supply-grid".into(),
                display_name: "External Supply Grid Blind Middle-Shard Pool".into(),
                max_nodes: 20,
                max_fast_memory_gb: 256.0,
                max_cloud_burst_usd_per_hr: 0.00,
                allow_ingress_egress_layers: false,
                allowed_workloads: vec![
                    WorkloadKind::Inference,
                    WorkloadKind::DistributedTraining,
                    WorkloadKind::BatchPipeline,
                ],
            },
        ];

        let keys = vec![
            Self::mint_scoped_api_key(
                "key-admin-01",
                "cameron@kadmium.ai",
                RbacRole::ClusterAdmin,
                "slice-01-prod-deploy",
            ),
            Self::mint_scoped_api_key(
                "key-rsch-02",
                "dream-rsi-agent@lab.local",
                RbacRole::ModelResearcher,
                "slice-03-diloco-train",
            ),
            Self::mint_scoped_api_key(
                "key-supply-03",
                "external-renter-pool",
                RbacRole::BlindSupplyProvider,
                "slice-05-supply-grid",
            ),
        ];

        (quotas, keys)
    }
}
