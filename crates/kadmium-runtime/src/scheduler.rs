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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    ///
    /// Pseudo-gradients are ordered canonically by their bits and reduced through the RepOps
    /// tree, so the step is bit-identical however the contributions arrived.
    pub fn step_outer_nesterov(
        &mut self,
        master_weights: &mut [f32],
        island_inner_weights: &[Vec<f32>],
    ) {
        if island_inner_weights.is_empty() {
            return;
        }
        // Preserve the weight-oriented API. Validate every replica before constructing deltas,
        // so even a malformed later replica cannot cause a partially applied update.
        assert!(
            island_inner_weights
                .iter()
                .all(|w| w.len() == master_weights.len()),
            "island parameter dimensions must match the master"
        );
        let pseudo_gradients: Vec<Vec<f32>> = island_inner_weights
            .iter()
            .map(|weights| {
                master_weights
                    .iter()
                    .zip(weights)
                    .map(|(master, local)| master - local)
                    .collect()
            })
            .collect();
        self.try_step_pseudo_gradients(master_weights, &pseudo_gradients)
            .expect("invalid DiLoCo outer update");
    }

    /// Applies already-computed `master_before - local_after` pseudo-gradients from replicas of
    /// the SAME parameter shard. This must not receive parameter values or unrelated layer shards.
    /// The mean is followed by `v = mu * v + g; master -= lr * (g + mu * v)`.
    /// Both parameters and momentum remain unchanged on any validation or arithmetic failure.
    pub fn try_step_pseudo_gradients(
        &mut self,
        master_weights: &mut [f32],
        island_pseudo_gradients: &[Vec<f32>],
    ) -> Result<(), String> {
        if !self.outer_lr.is_finite() || self.outer_lr < 0.0 {
            return Err("outer learning rate must be finite and nonnegative".into());
        }
        if !self.outer_momentum_mu.is_finite() || !(0.0..1.0).contains(&self.outer_momentum_mu) {
            return Err("outer momentum must be finite and in [0, 1)".into());
        }
        let n = master_weights.len();
        if n == 0 || self.velocity.len() != n {
            return Err("master and momentum must have the same nonzero dimension".into());
        }
        if island_pseudo_gradients.is_empty() {
            return Err("outer update requires at least one replica pseudo-gradient".into());
        }
        if master_weights
            .iter()
            .chain(&self.velocity)
            .any(|v| !v.is_finite())
        {
            return Err("master parameters and momentum must be finite".into());
        }
        for (replica, gradient) in island_pseudo_gradients.iter().enumerate() {
            if gradient.len() != n {
                return Err(format!(
                    "replica {replica} has {} parameters; expected {n}",
                    gradient.len()
                ));
            }
            if gradient.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "replica {replica} contains a nonfinite pseudo-gradient"
                ));
            }
        }

        let mut gradients: Vec<&Vec<f32>> = island_pseudo_gradients.iter().collect();
        gradients.sort_by(|a, b| Self::canonical_order(a, b));
        let mut terms = vec![0.0f64; gradients.len()];
        let mut next_weights = Vec::with_capacity(n);
        let mut next_velocity = Vec::with_capacity(n);
        let count = island_pseudo_gradients.len() as f64;
        for i in 0..n {
            // Canonical RepOps tree reduction keeps arrival order irrelevant. f64 avoids
            // overflowing an intermediate f32 sum before the mean is computed.
            for (term, gradient) in terms.iter_mut().zip(&gradients) {
                *term = gradient[i] as f64;
            }
            let gradient = (crate::repops::tree_sum_f64(&terms) / count) as f32;
            let velocity = self.outer_momentum_mu * self.velocity[i] + gradient;
            let weight =
                master_weights[i] - self.outer_lr * (gradient + self.outer_momentum_mu * velocity);
            if !velocity.is_finite() || !weight.is_finite() {
                return Err(format!("outer update overflow at parameter {i}"));
            }
            next_velocity.push(velocity);
            next_weights.push(weight);
        }
        master_weights.copy_from_slice(&next_weights);
        self.velocity = next_velocity;
        Ok(())
    }

    /// The order contributions are reduced in: by length, then by the bits of their values.
    /// Identical contributions are interchangeable, so the order is total on what matters.
    pub fn canonical_order(a: &[f32], b: &[f32]) -> std::cmp::Ordering {
        a.len().cmp(&b.len()).then_with(|| {
            a.iter()
                .map(|v| v.to_bits())
                .cmp(b.iter().map(|v| v.to_bits()))
        })
    }
}

pub(crate) const STATELESS_STAGE_PREVIEW_MODE: &str = "stateless_per_stage_gradient_preview";

/// Legacy diagnostics only: independent stages have independent synthetic baselines. The output
/// norm and digest summarize their concatenation; they are not a replica consensus or saved model.
pub(crate) fn stateless_stage_gradient_preview(
    stage_gradients: &[Vec<f32>],
    outer_lr: f32,
    outer_momentum: f32,
) -> Result<(f32, u64), String> {
    if stage_gradients.is_empty() {
        return Err("preview requires at least one stage gradient".into());
    }
    let mut digest = crate::repops::Digest::new("stateless-stage-preview");
    let mut preview_weights = Vec::new();
    for gradient in stage_gradients {
        let mut weights = vec![0.5; gradient.len()];
        let mut optimizer = DiLoCoOuterOptimizer::new(gradient.len(), outer_lr, outer_momentum);
        optimizer.try_step_pseudo_gradients(&mut weights, std::slice::from_ref(gradient))?;
        digest
            .update_u64(weights.len() as u64)
            .update_f32s(&weights);
        preview_weights.extend_from_slice(&weights);
    }
    let norm = crate::repops::tree_sum_squares_f64(&preview_weights).sqrt() as f32;
    if !norm.is_finite() {
        return Err("stage preview norm overflow".into());
    }
    Ok((norm, digest.finish()))
}

#[cfg(test)]
mod diloco_optimizer_tests {
    use super::*;

    #[test]
    fn pseudo_gradients_follow_two_round_scalar_nesterov_oracle() {
        let mut weights = vec![10.0];
        let mut optimizer = DiLoCoOuterOptimizer::new(1, 0.7, 0.9);
        // Mean delta = 3, v1 = 3, w1 = 10 - 0.7 * (3 + 0.9 * 3) = 6.01.
        optimizer
            .try_step_pseudo_gradients(&mut weights, &[vec![2.0], vec![4.0]])
            .unwrap();
        assert!((weights[0] - 6.01).abs() < 1e-5);
        assert_eq!(optimizer.velocity, vec![3.0]);
        // Mean delta = 2, v2 = 4.7, w2 = 6.01 - 0.7 * (2 + 0.9 * 4.7) = 1.649.
        optimizer
            .try_step_pseudo_gradients(&mut weights, &[vec![1.0], vec![3.0]])
            .unwrap();
        assert!((weights[0] - 1.649).abs() < 1e-5);
        assert!((optimizer.velocity[0] - 4.7).abs() < 1e-5);
    }

    #[test]
    fn weight_compatibility_api_matches_explicit_pseudo_gradients() {
        let mut from_weights = vec![10.0];
        let mut from_deltas = from_weights.clone();
        let mut old = DiLoCoOuterOptimizer::new(1, 0.7, 0.9);
        let mut explicit = old.clone();
        old.step_outer_nesterov(&mut from_weights, &[vec![8.0], vec![6.0]]);
        explicit
            .try_step_pseudo_gradients(&mut from_deltas, &[vec![2.0], vec![4.0]])
            .unwrap();
        assert_eq!(from_weights, from_deltas);
        assert_eq!(old, explicit);
    }

    #[test]
    fn checked_pseudo_gradients_use_canonical_repops_tree_order() {
        let gradients = [vec![1.0e20], vec![1.0], vec![-1.0e20], vec![3.0]];
        // Canonical bit order is [1, 3, 1e20, -1e20]. The fixed tree computes
        // (1 + 3) + (1e20 - 1e20) = 4; a linear sum in that order loses the 4.
        for order in [[0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 3, 0], [2, 0, 3, 1]] {
            let mut weights = vec![10.0];
            let mut optimizer = DiLoCoOuterOptimizer::new(1, 0.5, 0.0);
            let inputs: Vec<_> = order.iter().map(|&i| gradients[i].clone()).collect();
            optimizer
                .try_step_pseudo_gradients(&mut weights, &inputs)
                .unwrap();
            assert_eq!(optimizer.velocity[0].to_bits(), 1.0f32.to_bits());
            assert_eq!(weights[0].to_bits(), 9.5f32.to_bits());
        }
    }

    #[test]
    fn finite_mean_does_not_overflow_the_repops_accumulator() {
        let mut weights = vec![10.0];
        let mut optimizer = DiLoCoOuterOptimizer::new(1, 0.0, 0.0);
        optimizer
            .try_step_pseudo_gradients(&mut weights, &[vec![f32::MAX], vec![f32::MAX]])
            .unwrap();
        assert_eq!(weights, vec![10.0]);
        assert_eq!(optimizer.velocity, vec![f32::MAX]);
    }

    #[test]
    fn invalid_pseudo_gradients_leave_master_and_momentum_unchanged() {
        for gradients in [
            vec![],
            vec![vec![1.0, 2.0], vec![3.0]],
            vec![vec![1.0, f32::NAN]],
            vec![vec![1.0, f32::INFINITY]],
            vec![vec![1.0, f32::MAX]],
        ] {
            let mut weights = vec![10.0, 20.0];
            let mut optimizer = DiLoCoOuterOptimizer::new(2, 0.7, 0.9);
            optimizer.velocity = vec![0.25, 0.5];
            let before = optimizer.clone();
            assert!(
                optimizer
                    .try_step_pseudo_gradients(&mut weights, &gradients)
                    .is_err()
            );
            assert_eq!(weights, vec![10.0, 20.0]);
            assert_eq!(optimizer, before);
        }
        let mut weights = vec![10.0, 20.0];
        let mut optimizer = DiLoCoOuterOptimizer::new(1, 0.7, 0.9);
        let before = optimizer.clone();
        assert!(
            optimizer
                .try_step_pseudo_gradients(&mut weights, &[vec![1.0, 2.0]])
                .is_err()
        );
        assert_eq!(weights, vec![10.0, 20.0]);
        assert_eq!(optimizer, before);
    }

    #[test]
    fn invalid_optimizer_state_is_rejected_atomically() {
        for (lr, momentum, velocity, master) in [
            (-0.1, 0.9, 0.0, 10.0),
            (f32::NAN, 0.9, 0.0, 10.0),
            (0.7, 1.0, 0.0, 10.0),
            (0.7, f32::INFINITY, 0.0, 10.0),
            (0.7, 0.9, f32::NAN, 10.0),
            (0.7, 0.9, 0.0, f32::INFINITY),
        ] {
            let mut weights = vec![master];
            let mut optimizer = DiLoCoOuterOptimizer::new(1, lr, momentum);
            optimizer.velocity[0] = velocity;
            assert!(
                optimizer
                    .try_step_pseudo_gradients(&mut weights, &[vec![1.0]])
                    .is_err()
            );
            assert_eq!(weights[0].to_bits(), master.to_bits());
            assert_eq!(optimizer.velocity[0].to_bits(), velocity.to_bits());
        }
    }

    #[test]
    fn stage_preview_keeps_distinct_shards_independent() {
        // No momentum, unit learning rate: [0.5 - 0.1], [0.5 - 0.2, 0.5 - 0.3].
        let (norm, _) =
            stateless_stage_gradient_preview(&[vec![0.1], vec![0.2, 0.3]], 1.0, 0.0).unwrap();
        assert!((norm - (0.4f32.powi(2) + 0.3f32.powi(2) + 0.2f32.powi(2)).sqrt()).abs() < 1e-6);
        // Opposing gradients must not cancel through replica averaging.
        let (norm, _) =
            stateless_stage_gradient_preview(&[vec![1.0], vec![-1.0]], 1.0, 0.0).unwrap();
        assert!((norm - 2.5f32.sqrt()).abs() < 1e-6);
    }
}
