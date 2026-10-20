//! Asymmetric 1F1B Pipeline Micro-Batch Scheduler & Sharded-Island DiLoCo Outer Nesterov Optimizer
//! (`kadmium-runtime::scheduler`).

use kadmium_core::NodeId;
use kadmium_sharding::{ActivationCheckpointingPolicy, ClusterShardPlan};
use serde::{Deserialize, Serialize};

/// Micro-batch execution pass kind on a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelinePassKind {
    Forward,
    BackwardCachedActivations,
    BackwardWithSelectiveRecompute,
    BackwardWithFullRecompute,
}

/// Single scheduled micro-batch execution block on a pipeline stage node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStageEvent {
    pub stage_index: usize,
    pub node_id: NodeId,
    pub hostname: String,
    pub micro_batch_id: u32,
    pub pass_kind: PipelinePassKind,
    pub start_ms: f64,
    pub end_ms: f64,
}

/// Complete Asymmetric 1F1B (One-Forward-One-Backward) schedule across heterogeneous stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline1F1BSchedule {
    pub num_stages: usize,
    pub num_micro_batches: u32,
    pub total_step_duration_ms: f64,
    pub pipeline_bubble_pct: f64,
    pub peak_in_flight_activations_per_stage: Vec<u32>,
    pub events: Vec<PipelineStageEvent>,
}

/// Generates an exact 1F1B micro-batch schedule (Warmup -> Steady-State 1F1B -> Cooldown)
/// accounting for per-node selective activation recomputation overhead.
pub struct Pipeline1F1BScheduler;

impl Pipeline1F1BScheduler {
    pub fn build_schedule(plan: &ClusterShardPlan, num_micro_batches: u32) -> Pipeline1F1BSchedule {
        let active_stages: Vec<_> = plan
            .assignments
            .iter()
            .filter(|a| a.layer_count > 0)
            .collect();

        let s_count = active_stages.len().max(1);
        let m_count = num_micro_batches.max(s_count as u32);

        let base_fwd_ms = 18.0;
        let mut events = Vec::new();
        let mut peak_activations = Vec::with_capacity(s_count);
        let mut max_end_ms = 0.0f64;
        let mut total_busy_ms = 0.0f64;

        for (s_idx, assign) in active_stages.iter().enumerate() {
            let warmup_fwds = ((s_count - s_idx) as u32).min(m_count);
            let steady_pairs = m_count.saturating_sub(warmup_fwds);
            peak_activations.push(warmup_fwds);

            let (bwd_pass_kind, bwd_multiplier) = match assign.training_checkpoint_policy {
                ActivationCheckpointingPolicy::CacheActivationsInMemory => {
                    (PipelinePassKind::BackwardCachedActivations, 1.85)
                }
                ActivationCheckpointingPolicy::SelectiveEveryTwoLayers => {
                    (PipelinePassKind::BackwardWithSelectiveRecompute, 2.10)
                }
                _ => (PipelinePassKind::BackwardWithFullRecompute, 2.35),
            };

            let fwd_dur = base_fwd_ms;
            let bwd_dur = base_fwd_ms * bwd_multiplier;

            let mut cursor_ms = (s_idx as f64) * fwd_dur;
            let mut next_fwd_mb = 0u32;
            let mut next_bwd_mb = 0u32;

            // Phase 1: Warmup Forward Passes
            for _ in 0..warmup_fwds {
                let start = cursor_ms;
                let end = start + fwd_dur;
                events.push(PipelineStageEvent {
                    stage_index: s_idx,
                    node_id: assign.node_id.clone(),
                    hostname: assign.hostname.clone(),
                    micro_batch_id: next_fwd_mb,
                    pass_kind: PipelinePassKind::Forward,
                    start_ms: (start * 10.0).round() / 10.0,
                    end_ms: (end * 10.0).round() / 10.0,
                });
                cursor_ms = end;
                total_busy_ms += fwd_dur;
                next_fwd_mb += 1;
            }

            // Phase 2: Steady-State 1F1B (1 Backward followed immediately by 1 Forward)
            for _ in 0..steady_pairs {
                let b_start = cursor_ms;
                let b_end = b_start + bwd_dur;
                events.push(PipelineStageEvent {
                    stage_index: s_idx,
                    node_id: assign.node_id.clone(),
                    hostname: assign.hostname.clone(),
                    micro_batch_id: next_bwd_mb,
                    pass_kind: bwd_pass_kind,
                    start_ms: (b_start * 10.0).round() / 10.0,
                    end_ms: (b_end * 10.0).round() / 10.0,
                });
                cursor_ms = b_end;
                total_busy_ms += bwd_dur;
                next_bwd_mb += 1;

                let f_start = cursor_ms;
                let f_end = f_start + fwd_dur;
                events.push(PipelineStageEvent {
                    stage_index: s_idx,
                    node_id: assign.node_id.clone(),
                    hostname: assign.hostname.clone(),
                    micro_batch_id: next_fwd_mb,
                    pass_kind: PipelinePassKind::Forward,
                    start_ms: (f_start * 10.0).round() / 10.0,
                    end_ms: (f_end * 10.0).round() / 10.0,
                });
                cursor_ms = f_end;
                total_busy_ms += fwd_dur;
                next_fwd_mb += 1;
            }

            // Phase 3: Cooldown Backward Passes
            while next_bwd_mb < m_count {
                let b_start = cursor_ms;
                let b_end = b_start + bwd_dur;
                events.push(PipelineStageEvent {
                    stage_index: s_idx,
                    node_id: assign.node_id.clone(),
                    hostname: assign.hostname.clone(),
                    micro_batch_id: next_bwd_mb,
                    pass_kind: bwd_pass_kind,
                    start_ms: (b_start * 10.0).round() / 10.0,
                    end_ms: (b_end * 10.0).round() / 10.0,
                });
                cursor_ms = b_end;
                total_busy_ms += bwd_dur;
                next_bwd_mb += 1;
            }

            if cursor_ms > max_end_ms {
                max_end_ms = cursor_ms;
            }
        }

        let total_wall_capacity = max_end_ms * (s_count as f64);
        let bubble_pct = if total_wall_capacity > 0.0 {
            (((total_wall_capacity - total_busy_ms) / total_wall_capacity) * 100.0).max(0.0)
        } else {
            0.0
        };

        Pipeline1F1BSchedule {
            num_stages: s_count,
            num_micro_batches: m_count,
            total_step_duration_ms: (max_end_ms * 10.0).round() / 10.0,
            pipeline_bubble_pct: (bubble_pct * 10.0).round() / 10.0,
            peak_in_flight_activations_per_stage: peak_activations,
            events,
        }
    }
}

/// Outer-loop Nesterov momentum optimizer for **Sharded-Island DiLoCo** (`OpenDiLoCo` / `prime-diloco`).
#[derive(Debug, Clone)]
pub struct DiLoCoOuterOptimizer {
    pub outer_lr: f32,
    pub outer_momentum_mu: f32,
    pub velocity: Vec<f32>,
}

impl DiLoCoOuterOptimizer {
    pub fn new(num_params: usize, outer_lr: f32, outer_momentum_mu: f32) -> Self {
        Self {
            outer_lr,
            outer_momentum_mu,
            velocity: vec![0.0; num_params],
        }
    }

    /// Applies one outer-loop DiLoCo synchronization step across $K$ sharded compute islands:
    /// 1. Computes averaged pseudo-gradient $\Delta \theta = \frac{1}{K} \sum_{k=1}^K (\theta^{(0)} - \theta_k^{(H)})$
    /// 2. Updates outer Nesterov velocity $v_t = \mu v_{t-1} + \Delta \theta$
    /// 3. Updates master shard parameters $\theta^{(t+1)} = \theta^{(0)} - \eta_{\text{outer}} (\mu v_t + \Delta \theta)$
    pub fn step_outer_nesterov(
        &mut self,
        master_weights: &mut [f32],
        island_inner_weights: &[Vec<f32>],
    ) {
        if island_inner_weights.is_empty() {
            return;
        }
        let k_islands = island_inner_weights.len() as f32;
        let n = master_weights.len();
        assert_eq!(n, self.velocity.len());

        for i in 0..n {
            let mut sum_pseudo_grad = 0.0f32;
            for island_w in island_inner_weights {
                sum_pseudo_grad += master_weights[i] - island_w[i];
            }
            let avg_pseudo_grad = sum_pseudo_grad / k_islands;

            self.velocity[i] = self.outer_momentum_mu * self.velocity[i] + avg_pseudo_grad;
            let nesterov_step = self.outer_momentum_mu * self.velocity[i] + avg_pseudo_grad;
            master_weights[i] -= self.outer_lr * nesterov_step;
        }
    }
}
