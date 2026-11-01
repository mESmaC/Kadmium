//! Multi-machine layer placement inspection and view models for prepared checkpoints.
//!
//! Reads a prepared SafeTensors checkpoint's `source/config.json` and SafeTensors headers without
//! loading weight payloads, computes the exact in-memory weight bytes for the token embedding,
//! each decoder layer, the final norm and the LM head plus the full-context `f32` KV cache bytes at
//! `max_position_embeddings`, and builds stage-by-stage machine views via
//! [`kadmium_sharding::LayerPlacementPlanner`].
//!
//! Quality and safety contract:
//! - Always plans for the model's full native `max_position_embeddings` context window.
//! - If the available machines cannot hold all layers and the full-context KV cache, fails with an
//!   explicit error rather than shrinking the context window or falling back to a smaller model.

use super::{CheckpointLibrary, CheckpointSummary, Result};
use crate::{SafetensorsDtype, SafetensorsManifest};
use kadmium_core::{HardwareNode, NodeId, NodeTrustTier, ResourceOffer};
pub use kadmium_sharding::{
    CheckpointMemoryProfile, LayerPlacementError, LayerPlacementPlan, LayerPlacementPlanner,
    MachineMemoryBudget, PlannedLayerStage,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const CONFIG_FILE_LIMIT: u64 = 4 * 1024 * 1024;
const MAX_ANGLE: f64 = 16_777_216.0;

/// Validated memory profile and metadata for a prepared checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InspectedCheckpointProfile {
    pub checkpoint_id: String,
    pub checkpoint_name: String,
    pub architecture: String,
    pub source_format: String,
    pub profile: CheckpointMemoryProfile,
}

/// One machine's assigned contiguous layer slice and its verified memory accounting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointStageView {
    pub stage_index: usize,
    pub node_id: String,
    pub hostname: String,
    pub trust_tier: NodeTrustTier,
    pub online: bool,
    pub start_layer: usize,
    pub end_layer: usize,
    pub layer_count: usize,
    pub embedding: bool,
    pub head: bool,
    pub shares_tied_embedding: bool,
    pub holdings: Vec<String>,
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub total_bytes: u64,
    pub capacity_bytes: u64,
    pub free_bytes: u64,
    pub residency_tier: String,
    pub requires_approval: bool,
}

impl CheckpointStageView {
    /// Builds a human-readable stage view from a [`PlannedLayerStage`] and its checkpoint profile.
    pub fn from_stage(
        stage: &PlannedLayerStage,
        profile: &CheckpointMemoryProfile,
        online: bool,
    ) -> Self {
        let layer_count = stage.layers.len();
        let shares_tied_embedding = stage.embedding && stage.head && profile.tie_word_embeddings;
        let mut holdings = Vec::new();
        if stage.embedding {
            holdings.push("Token embedding (model.embed_tokens.weight)".into());
        }
        if layer_count == 1 {
            holdings.push(format!("Decoder layer {} (1 layer)", stage.layers.start));
        } else {
            holdings.push(format!(
                "Decoder layers {}..{} (L{}-L{}, {} layers)",
                stage.layers.start,
                stage.layers.end,
                stage.layers.start,
                stage.layers.end.saturating_sub(1),
                layer_count
            ));
        }
        if stage.head {
            if shares_tied_embedding {
                holdings.push(
                    "Final norm + LM head (model.norm.weight, shared tied embed_tokens)".into(),
                );
            } else {
                holdings.push("Final norm + LM head (model.norm.weight, lm_head.weight)".into());
            }
        }
        holdings.push(format!(
            "Full-context f32 KV cache ({} positions)",
            profile.max_position_embeddings
        ));
        Self {
            stage_index: stage.stage_index,
            node_id: stage.node_id.0.clone(),
            hostname: stage.hostname.clone(),
            trust_tier: stage.trust_tier,
            online,
            start_layer: stage.layers.start,
            end_layer: stage.layers.end,
            layer_count,
            embedding: stage.embedding,
            head: stage.head,
            shares_tied_embedding,
            holdings,
            weight_bytes: stage.weight_bytes,
            kv_bytes: stage.kv_bytes,
            total_bytes: stage.total_bytes,
            capacity_bytes: stage.capacity_bytes,
            free_bytes: stage.free_bytes,
            residency_tier: format!("{:?}", stage.residency_tier),
            requires_approval: stage.requires_approval,
        }
    }
}

/// Complete multi-machine placement view for a prepared checkpoint at its full native context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPlacementView {
    pub checkpoint_id: String,
    pub checkpoint_name: String,
    pub architecture: String,
    pub source_format: String,
    pub num_layers: usize,
    pub context_length: usize,
    pub tie_word_embeddings: bool,
    pub is_split: bool,
    pub machine_count: usize,
    pub eligible_machine_count: usize,
    pub total_weight_bytes: u64,
    pub total_kv_bytes: u64,
    pub total_bytes: u64,
    pub embedding_bytes: u64,
    pub final_norm_bytes: u64,
    pub lm_head_bytes: u64,
    pub kv_bytes_per_layer: u64,
    pub stages: Vec<CheckpointStageView>,
}

impl CheckpointPlacementView {
    /// Constructs a [`CheckpointPlacementView`] from an [`InspectedCheckpointProfile`] and a
    /// [`LayerPlacementPlan`].
    pub fn from_plan(
        inspected: &InspectedCheckpointProfile,
        plan: &LayerPlacementPlan,
        eligible_machine_count: usize,
        online_node_ids: &BTreeSet<String>,
    ) -> Self {
        let stages: Vec<CheckpointStageView> = plan
            .stages
            .iter()
            .map(|stage| {
                CheckpointStageView::from_stage(
                    stage,
                    &inspected.profile,
                    online_node_ids.contains(&stage.node_id.0),
                )
            })
            .collect();
        Self {
            checkpoint_id: inspected.checkpoint_id.clone(),
            checkpoint_name: inspected.checkpoint_name.clone(),
            architecture: inspected.architecture.clone(),
            source_format: inspected.source_format.clone(),
            num_layers: plan.num_layers,
            context_length: plan.context_length,
            tie_word_embeddings: inspected.profile.tie_word_embeddings,
            is_split: plan.is_split(),
            machine_count: stages.len(),
            eligible_machine_count,
            total_weight_bytes: plan.total_weight_bytes,
            total_kv_bytes: plan.total_kv_bytes,
            total_bytes: plan.total_bytes,
            embedding_bytes: inspected.profile.embedding_bytes,
            final_norm_bytes: inspected.profile.final_norm_bytes,
            lm_head_bytes: inspected.profile.lm_head_bytes,
            kv_bytes_per_layer: inspected.profile.kv_bytes_per_layer,
            stages,
        }
    }
}

/// Summary of one eligible compute machine's resident memory budget in the cluster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EligibleMachineView {
    pub node_id: String,
    pub hostname: String,
    pub trust_tier: NodeTrustTier,
    pub online: bool,
    pub fast_bytes: u64,
    pub warm_bytes: u64,
    pub capacity_bytes: u64,
    pub preferred_tier: String,
    pub requires_approval: bool,
}

impl EligibleMachineView {
    pub fn from_budget(budget: &MachineMemoryBudget, online: bool) -> Self {
        Self {
            node_id: budget.node_id.0.clone(),
            hostname: budget.hostname.clone(),
            trust_tier: budget.trust_tier,
            online,
            fast_bytes: budget.fast_bytes,
            warm_bytes: budget.warm_bytes,
            capacity_bytes: budget.capacity_bytes,
            preferred_tier: format!("{:?}", budget.preferred_tier),
            requires_approval: budget.requires_approval,
        }
    }
}

/// Catalog of prepared checkpoints, eligible cluster machines, and the active placement plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointPlacementCatalog {
    pub checkpoints: Vec<CheckpointSummary>,
    pub eligible_machines: Vec<EligibleMachineView>,
    pub total_cluster_capacity_bytes: u64,
    pub selected_checkpoint_id: Option<String>,
    pub placement: Option<CheckpointPlacementView>,
    pub placement_error: Option<String>,
}

/// Request payload for `POST /api/v1/checkpoints/placement`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanCheckpointRequest {
    pub checkpoint_id: String,
    #[serde(default)]
    pub checkpoint_name: Option<String>,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub source_format: Option<String>,
    #[serde(default)]
    pub profile: Option<CheckpointMemoryProfile>,
}

/// Plans an [`InspectedCheckpointProfile`] across `nodes` and `offers`.
pub fn plan_inspected_checkpoint(
    inspected: &InspectedCheckpointProfile,
    nodes: &[HardwareNode],
    offers: &BTreeMap<NodeId, ResourceOffer>,
    online_node_ids: &BTreeSet<String>,
) -> std::result::Result<CheckpointPlacementView, LayerPlacementError> {
    let budgets = MachineMemoryBudget::from_cluster(nodes, offers);
    plan_inspected_checkpoint_with_budgets(inspected, &budgets, online_node_ids)
}

/// Plans an [`InspectedCheckpointProfile`] across explicit [`MachineMemoryBudget`]s.
pub fn plan_inspected_checkpoint_with_budgets(
    inspected: &InspectedCheckpointProfile,
    budgets: &[MachineMemoryBudget],
    online_node_ids: &BTreeSet<String>,
) -> std::result::Result<CheckpointPlacementView, LayerPlacementError> {
    let plan = LayerPlacementPlanner::plan_with_budgets(&inspected.profile, budgets)?;
    Ok(CheckpointPlacementView::from_plan(
        inspected,
        &plan,
        budgets.len(),
        online_node_ids,
    ))
}

impl CheckpointLibrary {
    /// Verifies `id` in this library and computes its exact [`InspectedCheckpointProfile`] for
    /// multi-machine layer placement.
    pub fn inspect_placement_profile(&self, id: &str) -> Result<InspectedCheckpointProfile> {
        let manifest = self.verify(id)?;
        if let Some(quantized) = &manifest.pre_quantized {
            return Err(format!(
                "Split across machines runs 16-bit or wider weights only; this checkpoint was quantized before import ({})",
                quantized.description()
            ));
        }
        if manifest.source_format != "safetensors" {
            return Err(format!(
                "Split across machines currently requires a SafeTensors checkpoint with config.json (this checkpoint is {})",
                manifest.source_format
            ));
        }
        let source_dir = self.package_dir(id)?.join("source");
        let (architecture, profile) = inspect_safetensors_dir(&source_dir)?;
        Ok(InspectedCheckpointProfile {
            checkpoint_id: manifest.id,
            checkpoint_name: manifest.name,
            architecture,
            source_format: manifest.source_format,
            profile,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arch {
    Llama,
    Mistral,
    Qwen2,
    Qwen3,
}

impl Arch {
    fn hf_name(self) -> &'static str {
        match self {
            Self::Llama => "LlamaForCausalLM",
            Self::Mistral => "MistralForCausalLM",
            Self::Qwen2 => "Qwen2ForCausalLM",
            Self::Qwen3 => "Qwen3ForCausalLM",
        }
    }
}

#[derive(Clone, Copy)]
enum Rope {
    Default,
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: f64,
    },
}

struct ParsedConfig {
    architecture: Arch,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    rope_scaling: Rope,
    max_position_embeddings: usize,
    tie_word_embeddings: bool,
    qkv_bias: bool,
    o_bias: bool,
    mlp_bias: bool,
    qk_norm: bool,
}

fn positive(json: &Value, key: &str) -> Result<usize> {
    json.get(key)
        .and_then(Value::as_u64)
        .filter(|&n| n > 0)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("invalid model configuration: `{key}` must be a positive integer"))
}

fn optional_positive(json: &Value, key: &str) -> Result<Option<usize>> {
    match json.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => positive(json, key).map(Some),
    }
}

fn number(json: &Value, key: &str) -> Result<f64> {
    json.get(key)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite())
        .ok_or_else(|| format!("invalid model configuration: `{key}` must be a finite number"))
}

fn flag(json: &Value, key: &str, default: bool) -> Result<bool> {
    match json.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(format!(
            "invalid model configuration: `{key}` must be true or false"
        )),
    }
}

fn parse_rope(json: &Value) -> Result<(f64, Rope)> {
    let (theta, scaling) = match json.get("rope_parameters").filter(|v| !v.is_null()) {
        Some(parameters) => {
            if !parameters.is_object() {
                return Err(
                    "invalid model configuration: `rope_parameters` must be an object".into(),
                );
            }
            (number(parameters, "rope_theta")?, Some(parameters))
        }
        None => {
            let theta = match json.get("rope_theta") {
                None | Some(Value::Null) => 10_000.0,
                Some(_) => number(json, "rope_theta")?,
            };
            (theta, json.get("rope_scaling").filter(|v| !v.is_null()))
        }
    };
    if theta <= 1.0 {
        return Err(format!(
            "invalid model configuration: `rope_theta` {theta} must be greater than 1"
        ));
    }
    let Some(scaling) = scaling else {
        return Ok((theta, Rope::Default));
    };
    let kind = scaling
        .get("rope_type")
        .or_else(|| scaling.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("default");
    match kind {
        "default" => Ok((theta, Rope::Default)),
        "llama3" => {
            let factor = number(scaling, "factor")?;
            let low_freq_factor = number(scaling, "low_freq_factor")?;
            let high_freq_factor = number(scaling, "high_freq_factor")?;
            let original = number(scaling, "original_max_position_embeddings")?;
            if factor < 1.0
                || low_freq_factor <= 0.0
                || high_freq_factor <= low_freq_factor
                || original <= 0.0
            {
                return Err(
                    "invalid model configuration: `llama3` RoPE scaling parameters are out of range"
                        .into(),
                );
            }
            Ok((
                theta,
                Rope::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position_embeddings: original,
                },
            ))
        }
        other => Err(format!(
            "the native decoder does not yet support `{other}` RoPE scaling"
        )),
    }
}

fn parse_config(dir: &Path) -> Result<ParsedConfig> {
    let path = dir.join("config.json");
    let length = std::fs::metadata(&path)
        .map_err(|e| {
            format!(
                "invalid model configuration: cannot read {}: {e}",
                path.display()
            )
        })?
        .len();
    if length > CONFIG_FILE_LIMIT {
        return Err(format!(
            "invalid model configuration: {} is {length} bytes, more than a model config can be ({CONFIG_FILE_LIMIT})",
            path.display()
        ));
    }
    let text = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "invalid model configuration: cannot read {}: {e}",
            path.display()
        )
    })?;
    let json: Value = serde_json::from_str(&text)
        .map_err(|e| format!("invalid model configuration: not valid JSON: {e}"))?;

    if json
        .get("quantization_config")
        .is_some_and(|v| !v.is_null())
    {
        return Err(
            "the native decoder does not yet support a checkpoint with `quantization_config`: it was quantized before import, and the native decoder runs 16-bit or wider weights only".into(),
        );
    }
    let name = json
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|list| list.first())
        .and_then(Value::as_str)
        .ok_or_else(|| {
            "invalid model configuration: `architectures` must name the model class".to_string()
        })?;
    let architecture = match name {
        "LlamaForCausalLM" => Arch::Llama,
        "MistralForCausalLM" => Arch::Mistral,
        "Qwen2ForCausalLM" => Arch::Qwen2,
        "Qwen3ForCausalLM" => Arch::Qwen3,
        other => {
            return Err(format!(
                "the native decoder does not yet support the `{other}` architecture"
            ));
        }
    };
    if let Some(act) = json.get("hidden_act").filter(|v| !v.is_null())
        && act.as_str() != Some("silu")
    {
        return Err(format!(
            "the native decoder does not yet support the activation {act}; only SiLU is implemented"
        ));
    }
    let vocab_size = positive(&json, "vocab_size")?;
    let hidden_size = positive(&json, "hidden_size")?;
    let intermediate_size = positive(&json, "intermediate_size")?;
    let num_layers = positive(&json, "num_hidden_layers")?;
    let num_heads = positive(&json, "num_attention_heads")?;
    let num_kv_heads = optional_positive(&json, "num_key_value_heads")?.unwrap_or(num_heads);
    let head_dim = match optional_positive(&json, "head_dim")? {
        Some(dim) => dim,
        None if hidden_size % num_heads == 0 => hidden_size / num_heads,
        None => {
            return Err(format!(
                "invalid model configuration: `hidden_size` {hidden_size} is not divisible by `num_attention_heads` {num_heads}"
            ));
        }
    };
    if num_heads % num_kv_heads != 0 {
        return Err(format!(
            "invalid model configuration: `num_attention_heads` {num_heads} is not a multiple of `num_key_value_heads` {num_kv_heads}"
        ));
    }
    if head_dim % 2 != 0 {
        return Err(format!(
            "invalid model configuration: head dimension {head_dim} must be even for RoPE"
        ));
    }
    for (what, a, b) in [
        ("query width", num_heads, head_dim),
        ("embedding size", vocab_size, hidden_size),
        ("MLP size", intermediate_size, hidden_size),
    ] {
        if a.checked_mul(b).is_none() {
            return Err(format!(
                "invalid model configuration: {what} {a} x {b} overflows"
            ));
        }
    }
    let rms_norm_eps = number(&json, "rms_norm_eps")?;
    if rms_norm_eps <= 0.0 {
        return Err("invalid model configuration: `rms_norm_eps` must be positive".into());
    }
    let max_position_embeddings = positive(&json, "max_position_embeddings")?;
    if max_position_embeddings as f64 > MAX_ANGLE {
        return Err(format!(
            "the native decoder does not yet support a native context of {max_position_embeddings} positions; RoPE angles are exact up to {} positions",
            MAX_ANGLE as u64
        ));
    }
    if let Some(types) = json.get("layer_types").and_then(Value::as_array)
        && types.iter().any(|t| t.as_str() != Some("full_attention"))
    {
        return Err(
            "the native decoder does not yet support sliding-window attention layers (`layer_types`)"
                .into(),
        );
    }
    match architecture {
        Arch::Mistral => {
            if let Some(window) = optional_positive(&json, "sliding_window")?
                && window < max_position_embeddings
            {
                return Err(format!(
                    "the native decoder does not yet support sliding-window attention ({window} of {max_position_embeddings} positions)"
                ));
            }
        }
        Arch::Qwen2 | Arch::Qwen3 => {
            if flag(&json, "use_sliding_window", false)? {
                return Err(
                    "the native decoder does not yet support sliding-window attention (`use_sliding_window`)"
                        .into(),
                );
            }
        }
        Arch::Llama => {}
    }
    let (rope_theta, rope_scaling) = parse_rope(&json)?;
    let attention_bias = flag(&json, "attention_bias", false)?;
    let (qkv_bias, o_bias, mlp_bias, qk_norm) = match architecture {
        Arch::Llama => (
            attention_bias,
            attention_bias,
            flag(&json, "mlp_bias", false)?,
            false,
        ),
        Arch::Mistral => (false, false, false, false),
        Arch::Qwen2 => (true, false, false, false),
        Arch::Qwen3 => (attention_bias, attention_bias, false, true),
    };
    Ok(ParsedConfig {
        architecture,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        rope_scaling,
        max_position_embeddings,
        tie_word_embeddings: flag(&json, "tie_word_embeddings", false)?,
        qkv_bias,
        o_bias,
        mlp_bias,
        qk_norm,
    })
}

fn inverse_frequencies(config: &ParsedConfig) -> Vec<f32> {
    let half = config.head_dim / 2;
    let dim = config.head_dim as f64;
    (0..half)
        .map(|i| {
            let base = config.rope_theta.powf(-2.0 * (i as f64) / dim);
            let scaled = match config.rope_scaling {
                Rope::Default => base,
                Rope::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position_embeddings,
                } => {
                    let wavelength = 2.0 * std::f64::consts::PI / base;
                    let low = original_max_position_embeddings / low_freq_factor;
                    let high = original_max_position_embeddings / high_freq_factor;
                    if wavelength < high {
                        base
                    } else if wavelength > low {
                        base / factor
                    } else {
                        let smooth = (original_max_position_embeddings / wavelength
                            - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * (base / factor) + smooth * base
                    }
                }
            };
            scaled as f32
        })
        .collect()
}

fn stored_spec_bytes(manifest: &SafetensorsManifest, name: &str, shape: &[usize]) -> Result<u64> {
    let descriptor = manifest
        .tensor(name)
        .ok_or_else(|| format!("missing weight tensor `{name}`"))?;
    if !matches!(
        descriptor.dtype,
        SafetensorsDtype::F32 | SafetensorsDtype::F16 | SafetensorsDtype::Bf16
    ) {
        return Err(format!(
            "tensor `{name}` has dtype {}; only F32, F16 and BF16 are executed without loss",
            descriptor.dtype.as_str()
        ));
    }
    let expected: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    if descriptor.shape != expected {
        return Err(format!(
            "tensor `{name}` has shape {:?}, expected {:?}",
            descriptor.shape, shape
        ));
    }
    let elements: u64 = expected
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format!("tensor `{name}` size overflows"))?;
    let width: u64 = if descriptor.dtype == SafetensorsDtype::F32 {
        4
    } else {
        2
    };
    let expected_stored = elements
        .checked_mul(width)
        .ok_or_else(|| format!("tensor `{name}` byte size overflows"))?;
    if descriptor.byte_len != expected_stored {
        return Err(format!(
            "tensor `{name}` has shape {:?}, expected {:?}",
            vec![descriptor.byte_len / width],
            shape
        ));
    }
    if shape.len() == 1 {
        elements
            .checked_mul(4)
            .ok_or_else(|| format!("tensor `{name}` f32 size overflows"))
    } else {
        Ok(expected_stored)
    }
}

fn check_against_checkpoint(manifest: &SafetensorsManifest, config: &ParsedConfig) -> Result<()> {
    for tensor in manifest.tensors() {
        if let Some(index) = tensor
            .name
            .strip_prefix("model.layers.")
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(index, _)| index.parse::<usize>().ok())
            && index >= config.num_layers
        {
            return Err(format!(
                "the checkpoint stores layer {index}; `num_hidden_layers` is {}",
                config.num_layers
            ));
        }
        if tensor.name.contains(".experts.") {
            return Err(format!(
                "the native decoder does not yet support mixture-of-experts tensor `{}`",
                tensor.name
            ));
        }
        if !config.qkv_bias
            && (tensor.name.ends_with(".self_attn.q_proj.bias")
                || tensor.name.ends_with(".self_attn.k_proj.bias")
                || tensor.name.ends_with(".self_attn.v_proj.bias"))
        {
            return Err(format!(
                "invalid model configuration: checkpoint has `{}` while QKV bias is disabled",
                tensor.name
            ));
        }
        if !config.o_bias && tensor.name.ends_with(".self_attn.o_proj.bias") {
            return Err(format!(
                "invalid model configuration: checkpoint has `{}` while `attention_bias` is false",
                tensor.name
            ));
        }
        if !config.mlp_bias
            && (tensor.name.ends_with(".mlp.gate_proj.bias")
                || tensor.name.ends_with(".mlp.up_proj.bias")
                || tensor.name.ends_with(".mlp.down_proj.bias"))
        {
            return Err(format!(
                "invalid model configuration: checkpoint has `{}` while `mlp_bias` is false",
                tensor.name
            ));
        }
    }
    Ok(())
}

fn layer_specs(c: &ParsedConfig, layer: usize) -> Vec<(String, Vec<usize>)> {
    let prefix = format!("model.layers.{layer}.");
    let (h, q, kv, d, i) = (
        c.hidden_size,
        c.num_heads * c.head_dim,
        c.num_kv_heads * c.head_dim,
        c.head_dim,
        c.intermediate_size,
    );
    let mut specs = Vec::new();
    let mut add = |suffix: &str, shape: &[usize]| {
        specs.push((format!("{prefix}{suffix}"), shape.to_vec()));
    };
    add("input_layernorm.weight", &[h]);
    add("post_attention_layernorm.weight", &[h]);
    add("self_attn.q_proj.weight", &[q, h]);
    if c.qkv_bias {
        add("self_attn.q_proj.bias", &[q]);
    }
    add("self_attn.k_proj.weight", &[kv, h]);
    if c.qkv_bias {
        add("self_attn.k_proj.bias", &[kv]);
    }
    add("self_attn.v_proj.weight", &[kv, h]);
    if c.qkv_bias {
        add("self_attn.v_proj.bias", &[kv]);
    }
    add("self_attn.o_proj.weight", &[h, q]);
    if c.o_bias {
        add("self_attn.o_proj.bias", &[h]);
    }
    if c.qk_norm {
        add("self_attn.q_norm.weight", &[d]);
        add("self_attn.k_norm.weight", &[d]);
    }
    add("mlp.gate_proj.weight", &[i, h]);
    if c.mlp_bias {
        add("mlp.gate_proj.bias", &[i]);
    }
    add("mlp.up_proj.weight", &[i, h]);
    if c.mlp_bias {
        add("mlp.up_proj.bias", &[i]);
    }
    add("mlp.down_proj.weight", &[h, i]);
    if c.mlp_bias {
        add("mlp.down_proj.bias", &[h]);
    }
    specs
}

/// Inspects a SafeTensors checkpoint directory (`config.json` and `*.safetensors`) and returns its
/// architecture class name and exact [`CheckpointMemoryProfile`].
pub fn inspect_safetensors_dir(source_dir: &Path) -> Result<(String, CheckpointMemoryProfile)> {
    let config = parse_config(source_dir)?;
    let manifest = SafetensorsManifest::open(source_dir).map_err(|e| e.to_string())?;
    check_against_checkpoint(&manifest, &config)?;

    let embedding_bytes = stored_spec_bytes(
        &manifest,
        "model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
    )?;

    let mut layer_weight_bytes = Vec::with_capacity(config.num_layers);
    for layer in 0..config.num_layers {
        let mut sum = 0u64;
        for (name, shape) in &layer_specs(&config, layer) {
            let bytes = stored_spec_bytes(&manifest, name, shape)?;
            sum = sum
                .checked_add(bytes)
                .ok_or_else(|| format!("layer {layer} weight byte sum overflows"))?;
        }
        layer_weight_bytes.push(sum);
    }

    let final_norm_bytes =
        stored_spec_bytes(&manifest, "model.norm.weight", &[config.hidden_size])?;

    let lm_head_bytes = if config.tie_word_embeddings {
        embedding_bytes
    } else {
        stored_spec_bytes(
            &manifest,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
        )?
    };

    if config.num_layers > 0 {
        let inv = inverse_frequencies(&config);
        if let Some(fastest) = inv.iter().copied().reduce(f32::max)
            && f64::from(fastest) * config.max_position_embeddings as f64 > MAX_ANGLE
        {
            return Err(format!(
                "the native decoder does not yet support RoPE angles up to {} at the native context; they are reduced accurately only up to {MAX_ANGLE}",
                f64::from(fastest) * config.max_position_embeddings as f64
            ));
        }
    }

    let kv_bytes_per_layer = (config.max_position_embeddings as u64)
        .checked_mul(2)
        .and_then(|v| v.checked_mul(config.num_kv_heads as u64))
        .and_then(|v| v.checked_mul(config.head_dim as u64))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| "full-context KV cache byte size overflows".to_string())?;

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
    profile.validate().map_err(|e| e.to_string())?;
    Ok((config.architecture.hf_name().into(), profile))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SafetensorsToKtbTranscoder;
    use kadmium_core::NodeTrustTier;
    use std::collections::BTreeSet;

    fn write_four_layer_llama(dir: &Path, tie_word_embeddings: bool) {
        std::fs::create_dir_all(dir).unwrap();
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "tie_word_embeddings": tie_word_embeddings,
            "hidden_act": "silu"
        });
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();

        let mut specs: Vec<(String, Vec<usize>)> = vec![
            ("model.embed_tokens.weight".into(), vec![16, 8]),
            ("model.norm.weight".into(), vec![8]),
        ];
        if !tie_word_embeddings {
            specs.push(("lm_head.weight".into(), vec![16, 8]));
        }
        for layer in 0..4 {
            let p = format!("model.layers.{layer}.");
            specs.push((format!("{p}input_layernorm.weight"), vec![8]));
            specs.push((format!("{p}post_attention_layernorm.weight"), vec![8]));
            specs.push((format!("{p}self_attn.q_proj.weight"), vec![8, 8]));
            specs.push((format!("{p}self_attn.k_proj.weight"), vec![8, 8]));
            specs.push((format!("{p}self_attn.v_proj.weight"), vec![8, 8]));
            specs.push((format!("{p}self_attn.o_proj.weight"), vec![8, 8]));
            specs.push((format!("{p}mlp.gate_proj.weight"), vec![16, 8]));
            specs.push((format!("{p}mlp.up_proj.weight"), vec![16, 8]));
            specs.push((format!("{p}mlp.down_proj.weight"), vec![8, 16]));
        }
        let payloads: Vec<Vec<u8>> = specs
            .iter()
            .map(|(_, shape)| vec![0u8; shape.iter().product::<usize>() * 2])
            .collect();
        let entries: Vec<(&str, &str, Vec<usize>, &[u8])> = specs
            .iter()
            .zip(payloads.iter())
            .map(|((name, shape), bytes)| (name.as_str(), "BF16", shape.clone(), bytes.as_slice()))
            .collect();
        std::fs::write(
            dir.join("model.safetensors"),
            SafetensorsToKtbTranscoder::build_safetensors_bytes(&entries),
        )
        .unwrap();
    }

    #[test]
    fn inspects_and_splits_a_checkpoint_across_two_machines_with_stage_holdings() {
        let root = std::env::temp_dir().join(format!(
            "kadmium-data-placement-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let source = root.join("source");
        write_four_layer_llama(&source, false);

        let library = CheckpointLibrary::new(root.join("library"));
        let imported = library.import(&source).unwrap();
        let inspected = library.inspect_placement_profile(&imported.id).unwrap();
        assert_eq!(inspected.architecture, "LlamaForCausalLM");
        assert_eq!(inspected.profile.num_layers, 4);
        assert_eq!(inspected.profile.max_position_embeddings, 64);

        let per_layer_total =
            inspected.profile.layer_weight_bytes[0] + inspected.profile.kv_bytes_per_layer;
        let stage0_budget = inspected.profile.embedding_bytes + 2 * per_layer_total;
        let stage1_budget = 2 * per_layer_total
            + inspected.profile.final_norm_bytes
            + inspected.profile.lm_head_bytes;

        let mut alpha = MachineMemoryBudget::new(
            NodeId("node-alpha".into()),
            stage0_budget,
            NodeTrustTier::TrustedCore,
        );
        alpha.hostname = "alpha-workstation".into();
        let mut beta = MachineMemoryBudget::new(
            NodeId("node-beta".into()),
            stage1_budget,
            NodeTrustTier::TrustedCore,
        );
        beta.hostname = "beta-workstation".into();
        let budgets = vec![alpha.clone(), beta];
        let mut live = BTreeSet::new();
        live.insert("node-alpha".to_string());

        let view = plan_inspected_checkpoint_with_budgets(&inspected, &budgets, &live).unwrap();
        assert!(view.is_split);
        assert_eq!(view.context_length, 64);
        assert_eq!(view.stages.len(), 2);

        assert_eq!(view.stages[0].hostname, "alpha-workstation");
        assert_eq!(view.stages[0].start_layer, 0);
        assert_eq!(view.stages[0].end_layer, 2);
        assert!(view.stages[0].embedding);
        assert!(!view.stages[0].head);
        assert!(view.stages[0].online);
        assert_eq!(
            view.stages[0].holdings,
            vec![
                "Token embedding (model.embed_tokens.weight)",
                "Decoder layers 0..2 (L0-L1, 2 layers)",
                "Full-context f32 KV cache (64 positions)"
            ]
        );

        assert_eq!(view.stages[1].hostname, "beta-workstation");
        assert_eq!(view.stages[1].start_layer, 2);
        assert_eq!(view.stages[1].end_layer, 4);
        assert!(!view.stages[1].embedding);
        assert!(view.stages[1].head);
        assert!(!view.stages[1].online);
        assert_eq!(
            view.stages[1].holdings,
            vec![
                "Decoder layers 2..4 (L2-L3, 2 layers)",
                "Final norm + LM head (model.norm.weight, lm_head.weight)",
                "Full-context f32 KV cache (64 positions)"
            ]
        );

        let tight_budgets = vec![alpha];
        let err = plan_inspected_checkpoint_with_budgets(&inspected, &tight_budgets, &live)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("cannot hold all 4 layers")
                && err.contains("refusing to shorten the context or use a smaller model"),
            "{err}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
