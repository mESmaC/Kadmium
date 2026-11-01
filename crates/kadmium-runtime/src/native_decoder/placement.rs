//! Layer placement for a native SafeTensors checkpoint across heterogeneous machines.
//!
//! Reads the checkpoint's `config.json` and SafeTensors headers (without loading tensor payload
//! bytes), verifies every required tensor's shape and dtype (`F32`, `F16` or `BF16`), computes the
//! exact in-memory weight bytes for the embedding, each decoder layer, the final norm and the LM
//! head (accounting for tied embeddings), adds the `f32` KV cache bytes for the full
//! `max_position_embeddings` context, and delegates contiguous stage assignment to
//! [`kadmium_sharding::LayerPlacementPlanner`].

use super::config::DecoderConfig;
use super::math::MAX_ANGLE;
use super::model::{
    SlicePlan, Spec, block_specs, check_against_checkpoint, inverse_frequencies, stored_spec_bytes,
};
use super::{NativeDecoderError, Result};
use kadmium_core::{HardwareNode, NodeId, NodeTrustTier, ResourceOffer};
use kadmium_data::SafetensorsManifest;
use kadmium_grid::{PlacementRule, ResidencyTier};
use kadmium_sharding::{
    CheckpointMemoryProfile, LayerPlacementPlan, LayerPlacementPlanner, MachineMemoryBudget,
    PlannedLayerStage,
};
use std::collections::BTreeMap;
use std::path::Path;

/// One machine's planned [`SlicePlan`] alongside its verified byte accounting and residency tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedDecoderSlice {
    pub stage_index: usize,
    pub node_id: NodeId,
    pub hostname: String,
    pub trust_tier: NodeTrustTier,
    pub plan: SlicePlan,
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub total_bytes: u64,
    pub capacity_bytes: u64,
    pub free_bytes: u64,
    pub residency_tier: ResidencyTier,
    pub requires_approval: bool,
}

impl From<PlannedLayerStage> for PlannedDecoderSlice {
    fn from(stage: PlannedLayerStage) -> Self {
        Self {
            stage_index: stage.stage_index,
            node_id: stage.node_id,
            hostname: stage.hostname,
            trust_tier: stage.trust_tier,
            plan: SlicePlan {
                layers: stage.layers,
                embedding: stage.embedding,
                head: stage.head,
            },
            weight_bytes: stage.weight_bytes,
            kv_bytes: stage.kv_bytes,
            total_bytes: stage.total_bytes,
            capacity_bytes: stage.capacity_bytes,
            free_bytes: stage.free_bytes,
            residency_tier: stage.residency_tier,
            requires_approval: stage.requires_approval,
        }
    }
}

/// Complete placement of a checkpoint's decoder layers across one or more machines at the model's
/// full native context length.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderPlacementPlan {
    pub config: DecoderConfig,
    pub profile: CheckpointMemoryProfile,
    pub stages: Vec<PlannedDecoderSlice>,
    pub total_weight_bytes: u64,
    pub total_kv_bytes: u64,
    pub total_bytes: u64,
}

impl DecoderPlacementPlan {
    /// The model's full native context length (`max_position_embeddings`), never shortened.
    pub fn context_length(&self) -> usize {
        self.config.max_position_embeddings
    }

    /// Total decoder layers (`num_hidden_layers`).
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// True when the model is split across two or more machines.
    pub fn is_split(&self) -> bool {
        self.stages.len() > 1
    }

    /// The [`SlicePlan`] for each stage in pipeline order.
    pub fn slice_plans(&self) -> Vec<SlicePlan> {
        self.stages.iter().map(|s| s.plan.clone()).collect()
    }

    /// Returns the stage assigned to execute `layer`.
    pub fn stage_for_layer(&self, layer: usize) -> Option<&PlannedDecoderSlice> {
        self.stages.iter().find(|s| s.plan.layers.contains(&layer))
    }

    /// Emits pinned [`PlacementRule`]s for `kadmium-grid` covering each stage's weight slice and
    /// full-context KV cache reservation on its assigned machine.
    pub fn placement_rules(&self, model_id: &str) -> Vec<PlacementRule> {
        let sharding_plan = LayerPlacementPlan {
            num_layers: self.config.num_layers,
            context_length: self.config.max_position_embeddings,
            stages: self
                .stages
                .iter()
                .map(|s| PlannedLayerStage {
                    stage_index: s.stage_index,
                    node_id: s.node_id.clone(),
                    hostname: s.hostname.clone(),
                    trust_tier: s.trust_tier,
                    layers: s.plan.layers.clone(),
                    embedding: s.plan.embedding,
                    head: s.plan.head,
                    weight_bytes: s.weight_bytes,
                    kv_bytes: s.kv_bytes,
                    total_bytes: s.total_bytes,
                    capacity_bytes: s.capacity_bytes,
                    free_bytes: s.free_bytes,
                    residency_tier: s.residency_tier,
                    requires_approval: s.requires_approval,
                })
                .collect(),
            total_weight_bytes: self.total_weight_bytes,
            total_kv_bytes: self.total_kv_bytes,
            total_bytes: self.total_bytes,
        };
        sharding_plan.placement_rules(model_id)
    }
}

/// Inspects a checkpoint's SafeTensors headers and [`DecoderConfig`] to compute the exact
/// in-memory weight bytes for every part of the model and the `f32` KV cache bytes per layer at
/// `config.max_position_embeddings`.
pub fn checkpoint_memory_profile(
    manifest: &SafetensorsManifest,
    config: &DecoderConfig,
) -> Result<CheckpointMemoryProfile> {
    check_against_checkpoint(manifest, config)?;
    let embed_spec = Spec {
        name: "model.embed_tokens.weight".into(),
        shape: vec![config.vocab_size, config.hidden_size],
    };
    let embedding_bytes = stored_spec_bytes(manifest, &embed_spec)?;

    let mut layer_weight_bytes = Vec::with_capacity(config.num_layers);
    for layer in 0..config.num_layers {
        let mut sum = 0u64;
        for spec in &block_specs(config, layer) {
            let bytes = stored_spec_bytes(manifest, spec)?;
            sum = sum.checked_add(bytes).ok_or_else(|| {
                NativeDecoderError::Shape(format!("layer {layer} weight byte sum overflows"))
            })?;
        }
        layer_weight_bytes.push(sum);
    }

    let final_norm_bytes = stored_spec_bytes(
        manifest,
        &Spec {
            name: "model.norm.weight".into(),
            shape: vec![config.hidden_size],
        },
    )?;

    let lm_head_bytes = if config.tie_word_embeddings {
        embedding_bytes
    } else {
        stored_spec_bytes(
            manifest,
            &Spec {
                name: "lm_head.weight".into(),
                shape: vec![config.vocab_size, config.hidden_size],
            },
        )?
    };

    if config.num_layers > 0 {
        let inv = inverse_frequencies(config);
        if let Some(fastest) = inv.iter().copied().reduce(f32::max)
            && f64::from(fastest) * config.max_position_embeddings as f64 > MAX_ANGLE
        {
            return Err(NativeDecoderError::Unsupported(format!(
                "RoPE angles up to {} at the native context; they are reduced accurately only up to {MAX_ANGLE}",
                f64::from(fastest) * config.max_position_embeddings as f64
            )));
        }
    }

    let kv_bytes_per_layer = (config.max_position_embeddings as u64)
        .checked_mul(2)
        .and_then(|v| v.checked_mul(config.num_kv_heads as u64))
        .and_then(|v| v.checked_mul(config.head_dim as u64))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| {
            NativeDecoderError::Shape("full-context KV cache byte size overflows".into())
        })?;

    let profile = CheckpointMemoryProfile {
        num_layers: config.num_layers,
        max_position_embeddings: config.max_position_embeddings,
        embedding_bytes,
        layer_weight_bytes,
        final_norm_bytes,
        lm_head_bytes,
        tie_word_embeddings: config.tie_word_embeddings,
        kv_bytes_per_layer,
    };
    profile.validate()?;
    Ok(profile)
}

/// Plans which machine runs which layers for a checkpoint directory using each node's
/// [`ResourceOffer`] under its current [`kadmium_core::PoliteState`].
pub fn plan_checkpoint_placement(
    model_dir: &Path,
    nodes: &[HardwareNode],
    offers: &BTreeMap<NodeId, ResourceOffer>,
) -> Result<DecoderPlacementPlan> {
    let config = DecoderConfig::from_model_dir(model_dir)?;
    let manifest = SafetensorsManifest::open(model_dir)?;
    plan_placement_from_manifest(&manifest, config, nodes, offers)
}

/// Plans which machine runs which layers from an open [`SafetensorsManifest`] and
/// [`DecoderConfig`] against `nodes` and `offers`.
pub fn plan_placement_from_manifest(
    manifest: &SafetensorsManifest,
    config: DecoderConfig,
    nodes: &[HardwareNode],
    offers: &BTreeMap<NodeId, ResourceOffer>,
) -> Result<DecoderPlacementPlan> {
    let budgets = MachineMemoryBudget::from_cluster(nodes, offers);
    plan_placement_with_budgets(manifest, config, &budgets)
}

/// Plans which machine runs which layers for a checkpoint directory against explicit machine
/// resident memory budgets.
pub fn plan_checkpoint_placement_with_budgets(
    model_dir: &Path,
    budgets: &[MachineMemoryBudget],
) -> Result<DecoderPlacementPlan> {
    let config = DecoderConfig::from_model_dir(model_dir)?;
    let manifest = SafetensorsManifest::open(model_dir)?;
    plan_placement_with_budgets(&manifest, config, budgets)
}

/// Plans which machine runs which layers from an open [`SafetensorsManifest`] and
/// [`DecoderConfig`] against explicit machine resident memory budgets.
pub fn plan_placement_with_budgets(
    manifest: &SafetensorsManifest,
    config: DecoderConfig,
    budgets: &[MachineMemoryBudget],
) -> Result<DecoderPlacementPlan> {
    let profile = checkpoint_memory_profile(manifest, &config)?;
    let plan = LayerPlacementPlanner::plan_with_budgets(&profile, budgets)?;
    Ok(DecoderPlacementPlan {
        config,
        profile,
        stages: plan
            .stages
            .into_iter()
            .map(PlannedDecoderSlice::from)
            .collect(),
        total_weight_bytes: plan.total_weight_bytes,
        total_kv_bytes: plan.total_kv_bytes,
        total_bytes: plan.total_bytes,
    })
}
