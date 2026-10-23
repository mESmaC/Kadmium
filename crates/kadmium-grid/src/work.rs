//! Work units: the process abstraction of the Kadmium grid.
//!
//! A [`WorkUnit`] is one piece of computation with content-addressed inputs, placement
//! constraints, a cost estimate, an optional deadline and a redundancy policy. It is the same
//! object whether the application is a `.ktb` staging transfer, a WASI reward sandbox, one
//! routed expert applied to a token batch, or a training micro-step. The scheduler never looks
//! inside `params`; the executor registered for the unit's [`AppKind`] does.
//!
//! Lifecycle, with BOINC's names in parentheses: `Unsent` -> `Assigned` (sent) -> `Running` ->
//! `Returned` -> `Validated` or `Invalid` -> `Assimilated`, with `Expired` and `Failed` as the
//! other exits. Each attempt to run a unit is a [`WorkInstance`] (a BOINC result); redundancy
//! means one unit can have several instances on different nodes, and only agreement among them
//! validates it.

use kadmium_core::{ComputeBackend, NodeId, NodeTrustTier};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::GridError;

/// FNV-1a 64-bit hash, the content-addressing and digest primitive used across Kadmium.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Identifier of a work unit, unique within a queue.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WorkUnitId(pub String);

impl WorkUnitId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for WorkUnitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for WorkUnitId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

impl From<String> for WorkUnitId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

/// Identifier of one attempt to run a work unit on one node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct InstanceId(pub String);

impl fmt::Display for InstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a unit asks a worker to do. Generic kinds are executed by the grid worker itself; AI
/// kinds are executed by `kadmium-runtime`, which registers an executor per kind.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AppKind {
    /// Stage a content-addressed object from a holder to a destination (FTS3 third-party transfer).
    Fts3Stage,
    /// Recompute and compare the checksum of resident content.
    Checksum,
    /// Fuel-metered sandbox execution, such as an RL reward check.
    WasiSandbox,
    /// Forward pass over a contiguous layer range, today's pipeline stage.
    StageForward,
    /// Dense trunk (attention and norms) forward for a layer range on its trunk owner.
    TrunkForward,
    /// One or more routed experts of one MoE layer applied to a token batch.
    ExpertSlice,
    /// Fixed-order combine of expert partial sums plus the residual.
    Combine,
    /// Speculative draft generation or verification.
    DraftVerify,
    /// Training forward and backward for one stage and micro-batch.
    TrainForwardBackward,
    /// DeMo DCT-compressed gradient reduce.
    DeMoReduce,
    /// Dream-RSI offline replay batch.
    DreamReplay,
    /// Verde refereed layer-bisection audit.
    VerdeAudit,
    /// Tokenization and `.ktb` packing.
    TokenizePack,
    /// Any other application, by name.
    Custom(String),
}

impl AppKind {
    /// Units on a decode or training critical path. The scheduler pushes these over persistent
    /// channels instead of waiting for a worker to pull them.
    pub fn is_latency_critical(&self) -> bool {
        matches!(
            self,
            Self::StageForward
                | Self::TrunkForward
                | Self::ExpertSlice
                | Self::Combine
                | Self::DraftVerify
                | Self::TrainForwardBackward
                | Self::DeMoReduce
        )
    }

    pub fn label(&self) -> String {
        match self {
            Self::Custom(name) => name.clone(),
            other => format!("{other:?}"),
        }
    }
}

/// What kind of content a [`ContentRef`] points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentKind {
    /// A page-aligned `.ktb` weight bank (dense layers or a layer range).
    KtbWeightBank,
    /// One routed expert, or a bank of experts, of one MoE layer.
    ExpertBank,
    Dataset,
    Checkpoint,
    /// Boundary activations or KV pages produced by one unit and consumed by another.
    Activation,
    /// Any other artifact.
    Artifact,
}

/// A content-addressed reference to bytes the mesh holds somewhere.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentRef {
    pub content_id: String,
    pub kind: ContentKind,
    pub bytes: u64,
}

impl ContentRef {
    pub fn new(content_id: impl Into<String>, kind: ContentKind, bytes: u64) -> Self {
        Self {
            content_id: content_id.into(),
            kind,
            bytes,
        }
    }

    /// Reference whose id is the FNV-1a 64 hash of the bytes, in hex.
    pub fn from_bytes(kind: ContentKind, data: &[u8]) -> Self {
        Self {
            content_id: format!("{:016x}", fnv1a_64(data)),
            kind,
            bytes: data.len() as u64,
        }
    }

    /// Reference for content identified by a stable name such as `ornith-35b/L25/E045`.
    pub fn named(kind: ContentKind, name: &str, bytes: u64) -> Self {
        Self {
            content_id: name.to_string(),
            kind,
            bytes,
        }
    }
}

impl fmt::Display for ContentRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}:{}", self.kind, self.content_id)
    }
}

/// Rank of a trust tier for comparisons: owned hardware is highest.
pub fn trust_rank(tier: NodeTrustTier) -> u8 {
    match tier {
        NodeTrustTier::TrustedCore => 2,
        NodeTrustTier::VerifiedPeerMiddleLayersOnly => 1,
        NodeTrustTier::UntrustedAuditedMiddleLayersOnly => 0,
    }
}

/// True when a worker of `worker` tier may run a unit that requires at least `minimum`.
pub fn trust_satisfies(worker: NodeTrustTier, minimum: NodeTrustTier) -> bool {
    trust_rank(worker) >= trust_rank(minimum)
}

/// Where a unit may run. Everything here is a constraint the matchmaker checks; locality is a
/// preference it scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementConstraints {
    /// Inputs that should already be resident on the chosen node. A node holding none of them
    /// is still eligible; it just scores lower, because the bytes would have to move.
    pub requires_resident: Vec<ContentRef>,
    /// Fast memory (VRAM or unified) the unit needs to run at all, in GiB.
    pub min_fast_memory_gb: f64,
    /// Backends that can run the unit's kernels; empty means any.
    pub allowed_backends: Vec<ComputeBackend>,
    /// Lowest trust tier allowed to run the unit.
    pub min_trust_tier: NodeTrustTier,
    /// True for units that touch layer 0 or the last layer, which must stay on owned hardware.
    pub boundary_layer_access: bool,
    pub preferred_nodes: Vec<NodeId>,
    pub excluded_nodes: Vec<NodeId>,
    /// Upper bound on the hop latency to the node, for latency-critical units.
    pub max_hop_latency_ms: Option<f64>,
}

impl Default for PlacementConstraints {
    fn default() -> Self {
        Self {
            requires_resident: Vec::new(),
            min_fast_memory_gb: 0.0,
            allowed_backends: Vec::new(),
            min_trust_tier: NodeTrustTier::UntrustedAuditedMiddleLayersOnly,
            boundary_layer_access: false,
            preferred_nodes: Vec::new(),
            excluded_nodes: Vec::new(),
            max_hop_latency_ms: None,
        }
    }
}

impl PlacementConstraints {
    /// Constraints for a unit whose inputs should be resident where it runs.
    pub fn resident(inputs: Vec<ContentRef>) -> Self {
        Self {
            requires_resident: inputs,
            ..Self::default()
        }
    }

    /// Constraints for prompt ingress, token egress and anything else that must stay on owned
    /// hardware.
    pub fn trusted_only() -> Self {
        Self {
            min_trust_tier: NodeTrustTier::TrustedCore,
            boundary_layer_access: true,
            ..Self::default()
        }
    }
}

/// What a unit is expected to cost. The scheduler uses `est_wall_ms` for deadlines and, until
/// the worker has been measured for the unit's kind of work, straggler detection; the ledger
/// uses `flops` and bytes for credit.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CostEstimate {
    pub flops: f64,
    pub bytes_read: u64,
    pub bytes_moved: u64,
    pub est_wall_ms: f64,
    /// How much work the unit covers, in the units its result reports as `units_processed`
    /// (rows, rollouts, simulations, layers), so a measured cost per unit of work predicts it.
    /// 0 counts as 1.
    #[serde(default)]
    pub work_units: u64,
}

impl CostEstimate {
    pub fn new(flops: f64, bytes_read: u64, bytes_moved: u64, est_wall_ms: f64) -> Self {
        Self {
            flops,
            bytes_read,
            bytes_moved,
            est_wall_ms,
            work_units: 1,
        }
    }

    pub fn with_work_units(mut self, work_units: u64) -> Self {
        self.work_units = work_units.max(1);
        self
    }

    /// Units of work, never zero.
    pub fn work(&self) -> f64 {
        self.work_units.max(1) as f64
    }
}

/// How returned instances of one unit are compared before the unit is validated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationRule {
    /// The first returned result validates; for owned hardware.
    Trusted,
    /// Results must agree bit-exactly on their RepOps digest.
    BitExactDigest,
    /// Results must agree on their scalar check value within a tolerance; for cross-silicon
    /// drift between CUDA, Metal and CPU nodes.
    ScalarTolerance { max_abs_diff: f64 },
    /// Digests must agree; a disagreement opens a Verde refereed bisection instead of a majority
    /// vote.
    VerdeBisection,
}

/// How many instances a unit gets and how many must agree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RedundancyPolicy {
    /// Agreeing results needed to validate.
    pub min_quorum: u32,
    /// Instances to create up front.
    pub target_replicas: u32,
    /// Ceiling on instances, including straggler duplicates and tie-breakers.
    pub max_replicas: u32,
    pub validation: ValidationRule,
}

impl RedundancyPolicy {
    /// One instance, validated on return; one spare for stragglers.
    pub fn trusted() -> Self {
        Self {
            min_quorum: 1,
            target_replicas: 1,
            max_replicas: 2,
            validation: ValidationRule::Trusted,
        }
    }

    /// One instance whose scalar check must stay within tolerance of a reference when one exists.
    pub fn verified_peer() -> Self {
        Self {
            min_quorum: 1,
            target_replicas: 1,
            max_replicas: 3,
            validation: ValidationRule::ScalarTolerance { max_abs_diff: 1e-3 },
        }
    }

    /// Two agreeing instances, a Verde bisection on disagreement, up to four attempts.
    pub fn untrusted() -> Self {
        Self {
            min_quorum: 2,
            target_replicas: 2,
            max_replicas: 4,
            validation: ValidationRule::VerdeBisection,
        }
    }

    /// Default policy for work that will run on nodes of the given tier.
    pub fn for_trust_tier(tier: NodeTrustTier) -> Self {
        match tier {
            NodeTrustTier::TrustedCore => Self::trusted(),
            NodeTrustTier::VerifiedPeerMiddleLayersOnly => Self::verified_peer(),
            NodeTrustTier::UntrustedAuditedMiddleLayersOnly => Self::untrusted(),
        }
    }

    /// Clamps the counts so that `1 <= min_quorum <= target_replicas <= max_replicas`.
    pub fn normalized(mut self) -> Self {
        self.min_quorum = self.min_quorum.max(1);
        self.target_replicas = self.target_replicas.max(self.min_quorum);
        self.max_replicas = self.max_replicas.max(self.target_replicas);
        self
    }
}

/// Lifecycle state of a unit or an instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkState {
    Unsent,
    Assigned,
    Running,
    Returned,
    Validated,
    Invalid,
    Assimilated,
    Expired,
    Failed,
}

impl WorkState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Assimilated | Self::Invalid | Self::Expired | Self::Failed
        )
    }

    /// States in which an instance occupies a worker or still awaits validation.
    pub fn is_live(self) -> bool {
        matches!(self, Self::Assigned | Self::Running | Self::Returned)
    }
}

/// One piece of computation the grid can schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkUnit {
    pub id: WorkUnitId,
    pub app: AppKind,
    /// Content the executor reads. Usually mirrored into `placement.requires_resident`.
    pub inputs: Vec<ContentRef>,
    /// Opaque parameters for the executor, usually JSON.
    pub params: Vec<u8>,
    pub placement: PlacementConstraints,
    pub cost: CostEstimate,
    pub deadline_unix_ms: Option<u64>,
    pub redundancy: RedundancyPolicy,
    /// Long units that can checkpoint and resume after a yield.
    pub checkpointable: bool,
    /// Higher runs first among units with equal deadlines.
    pub priority: u8,
    pub graph_id: Option<String>,
    pub created_unix_ms: u64,
}

impl WorkUnit {
    pub fn new(id: impl Into<WorkUnitId>, app: AppKind) -> Self {
        Self {
            id: id.into(),
            app,
            inputs: Vec::new(),
            params: Vec::new(),
            placement: PlacementConstraints::default(),
            cost: CostEstimate::default(),
            deadline_unix_ms: None,
            redundancy: RedundancyPolicy::trusted(),
            checkpointable: false,
            priority: 0,
            graph_id: None,
            created_unix_ms: 0,
        }
    }

    /// Sets the inputs and marks them as the content that should be resident where the unit runs.
    pub fn with_inputs(mut self, inputs: Vec<ContentRef>) -> Self {
        self.placement.requires_resident = inputs.clone();
        self.inputs = inputs;
        self
    }

    pub fn with_params_json<T: Serialize>(mut self, params: &T) -> Result<Self, GridError> {
        self.params = serde_json::to_vec(params).map_err(|e| GridError::Params(e.to_string()))?;
        Ok(self)
    }

    pub fn with_params_bytes(mut self, params: Vec<u8>) -> Self {
        self.params = params;
        self
    }

    pub fn with_placement(mut self, placement: PlacementConstraints) -> Self {
        self.placement = placement;
        self
    }

    pub fn with_cost(mut self, cost: CostEstimate) -> Self {
        self.cost = cost;
        self
    }

    pub fn with_deadline(mut self, deadline_unix_ms: u64) -> Self {
        self.deadline_unix_ms = Some(deadline_unix_ms);
        self
    }

    pub fn with_redundancy(mut self, redundancy: RedundancyPolicy) -> Self {
        self.redundancy = redundancy.normalized();
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_created(mut self, created_unix_ms: u64) -> Self {
        self.created_unix_ms = created_unix_ms;
        self
    }

    pub fn with_checkpointable(mut self, checkpointable: bool) -> Self {
        self.checkpointable = checkpointable;
        self
    }

    pub fn params_json<T: DeserializeOwned>(&self) -> Result<T, GridError> {
        serde_json::from_slice(&self.params).map_err(|e| GridError::Params(e.to_string()))
    }

    pub fn input_bytes(&self) -> u64 {
        self.inputs.iter().map(|c| c.bytes).sum()
    }
}

/// Content a unit touched while it ran, with how often: the raw signal behind learned placement
/// (colibri's usage file) and frequency-aware eviction. For a MoE layer the content ids are the
/// experts the router selected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentAccess {
    pub content_id: String,
    pub hits: u64,
}

/// A content access as a run reports it: which node, running which kind of work, touched which
/// content how often. Summed over a run's validated results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedAccess {
    pub node: NodeId,
    /// The unit's app kind label.
    pub app: String,
    pub content_id: String,
    pub hits: u64,
}

/// The RepOps digest of one layer boundary inside a unit: the commitment a Verde bisection
/// compares between disputants, one per layer the unit ran, in layer order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerDigest {
    pub layer: u32,
    pub digest: u64,
}

impl LayerDigest {
    pub fn new(layer: u32, digest: u64) -> Self {
        Self { layer, digest }
    }
}

/// What a worker returns for one instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkResultRecord {
    pub unit_id: WorkUnitId,
    pub instance_id: InstanceId,
    pub node_id: NodeId,
    /// Content the unit produced, now resident on `node_id`.
    pub outputs: Vec<ContentRef>,
    /// RepOps digest of the result, the basis of bit-exact validation.
    pub digest_fnv1a: u64,
    /// The RepOps digest of every layer boundary the unit ran, in layer order: 12 bytes per
    /// layer, enough for a referee to find the first divergent layer without re-running the
    /// unit. Empty for units that do not run layers.
    #[serde(default)]
    pub layer_digests: Vec<LayerDigest>,
    /// Scalar summary of the result (an L2 norm, a loss) for tolerance validation.
    pub check_value: f64,
    pub exec_ms: f64,
    pub bytes_moved: u64,
    /// Opaque result payload for the consumer.
    pub payload: Vec<u8>,
    pub returned_unix_ms: u64,
    /// Rows, rollouts or simulations the result covered, for credit and receipts.
    pub units_processed: u64,
    /// Content the unit touched while it ran, with hit counts.
    #[serde(default)]
    pub accessed: Vec<ContentAccess>,
}

impl WorkResultRecord {
    pub fn new(
        unit_id: WorkUnitId,
        instance_id: InstanceId,
        node_id: NodeId,
        digest_fnv1a: u64,
        exec_ms: f64,
        returned_unix_ms: u64,
    ) -> Self {
        Self {
            unit_id,
            instance_id,
            node_id,
            outputs: Vec::new(),
            digest_fnv1a,
            layer_digests: Vec::new(),
            check_value: 0.0,
            exec_ms,
            bytes_moved: 0,
            payload: Vec::new(),
            returned_unix_ms,
            units_processed: 0,
            accessed: Vec::new(),
        }
    }

    pub fn with_accessed(mut self, accessed: Vec<ContentAccess>) -> Self {
        self.accessed = accessed;
        self
    }

    pub fn with_layer_digests(mut self, layer_digests: Vec<LayerDigest>) -> Self {
        self.layer_digests = layer_digests;
        self
    }

    /// The position of the first entry at which this result's digest chain and `other`'s
    /// disagree, when both carry chains of the same length, starting at the same layer, that
    /// differ somewhere. Verde's first phase: the referee scans the two commitment sequences
    /// for the first difference, which costs comparisons rather than re-execution.
    pub fn first_divergent_index(&self, other: &WorkResultRecord) -> Option<usize> {
        if self.layer_digests.is_empty()
            || self.layer_digests.len() != other.layer_digests.len()
            || self.layer_digests[0].layer != other.layer_digests[0].layer
        {
            return None;
        }
        self.layer_digests
            .iter()
            .zip(&other.layer_digests)
            .position(|(a, b)| a != b)
    }

    /// This result's digest at the first divergent position, see
    /// [`WorkResultRecord::first_divergent_index`].
    pub fn first_divergent_layer(&self, other: &WorkResultRecord) -> Option<LayerDigest> {
        self.first_divergent_index(other)
            .map(|index| self.layer_digests[index])
    }

    pub fn with_units_processed(mut self, units_processed: u64) -> Self {
        self.units_processed = units_processed;
        self
    }

    pub fn with_outputs(mut self, outputs: Vec<ContentRef>) -> Self {
        self.outputs = outputs;
        self
    }

    pub fn with_check_value(mut self, check_value: f64) -> Self {
        self.check_value = check_value;
        self
    }

    pub fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }

    pub fn with_bytes_moved(mut self, bytes_moved: u64) -> Self {
        self.bytes_moved = bytes_moved;
        self
    }
}

/// One attempt to run a unit on one node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkInstance {
    pub instance_id: InstanceId,
    pub unit_id: WorkUnitId,
    pub node_id: NodeId,
    /// 1 for the first attempt of a unit, counting retries and duplicates.
    pub attempt: u32,
    pub state: WorkState,
    pub assigned_unix_ms: u64,
    pub started_unix_ms: Option<u64>,
    pub returned_unix_ms: Option<u64>,
    pub est_wall_ms: f64,
    /// Set when this instance was created to race a straggler.
    pub duplicate_of: Option<InstanceId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_matches_reference_vectors() {
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn content_refs_are_content_addressed() {
        let a = ContentRef::from_bytes(ContentKind::ExpertBank, b"expert-45");
        let b = ContentRef::from_bytes(ContentKind::ExpertBank, b"expert-45");
        let c = ContentRef::from_bytes(ContentKind::ExpertBank, b"expert-46");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.bytes, 9);
        assert_eq!(a.to_string(), format!("ExpertBank:{}", a.content_id));
    }

    #[test]
    fn redundancy_follows_trust_tier_and_normalizes() {
        assert_eq!(
            RedundancyPolicy::for_trust_tier(NodeTrustTier::TrustedCore).min_quorum,
            1
        );
        let untrusted =
            RedundancyPolicy::for_trust_tier(NodeTrustTier::UntrustedAuditedMiddleLayersOnly);
        assert_eq!(untrusted.min_quorum, 2);
        assert_eq!(untrusted.validation, ValidationRule::VerdeBisection);

        let odd = RedundancyPolicy {
            min_quorum: 3,
            target_replicas: 1,
            max_replicas: 0,
            validation: ValidationRule::BitExactDigest,
        }
        .normalized();
        assert_eq!(
            (odd.min_quorum, odd.target_replicas, odd.max_replicas),
            (3, 3, 3)
        );
    }

    #[test]
    fn trust_ranks_order_owned_hardware_highest() {
        assert!(trust_satisfies(
            NodeTrustTier::TrustedCore,
            NodeTrustTier::VerifiedPeerMiddleLayersOnly
        ));
        assert!(!trust_satisfies(
            NodeTrustTier::UntrustedAuditedMiddleLayersOnly,
            NodeTrustTier::VerifiedPeerMiddleLayersOnly
        ));
    }

    #[test]
    fn work_unit_builder_and_params_roundtrip() {
        #[derive(Serialize, serde::Deserialize, PartialEq, Debug)]
        struct ExpertParams {
            layer: u32,
            experts: Vec<u32>,
        }
        let input = ContentRef::named(ContentKind::ExpertBank, "m/L25/E045", 6_300_000);
        let unit = WorkUnit::new("wu-1", AppKind::ExpertSlice)
            .with_inputs(vec![input.clone()])
            .with_params_json(&ExpertParams {
                layer: 25,
                experts: vec![45],
            })
            .unwrap()
            .with_cost(CostEstimate::new(1e9, 6_300_000, 4096, 2.0))
            .with_deadline(10_000)
            .with_priority(5)
            .with_redundancy(RedundancyPolicy::untrusted());

        assert_eq!(unit.placement.requires_resident, vec![input]);
        assert_eq!(unit.input_bytes(), 6_300_000);
        let parsed: ExpertParams = unit.params_json().unwrap();
        assert_eq!(parsed.experts, vec![45]);
        assert!(unit.app.is_latency_critical());
        assert!(!AppKind::DreamReplay.is_latency_critical());
        assert_eq!(AppKind::Custom("fold".into()).label(), "fold");
        assert!(!WorkState::Returned.is_terminal());
        assert!(WorkState::Returned.is_live());
        assert!(WorkState::Expired.is_terminal());
    }
}
