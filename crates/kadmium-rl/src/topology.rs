//! Disaggregated RL Topology Planner & Group-Relative Policy Optimization (`kadmium-rl::topology`).

use crate::{
    DiscoveryTree, DreamRsiDomain, DreamRsiEvolutionReport, DreamRsiSimulator, RlAlgorithm,
    RlEnvironmentKind, RlNodeRole,
};
use kadmium_core::{DeviceKind, HardwareNode, ModelSpec, NodeId, PoliteState};
use serde::{Deserialize, Serialize};

/// Descriptor of an active RL or Dream-RSI environment suite deployed across the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlEnvironmentSuiteSpec {
    pub suite_id: String,
    pub display_name: String,
    pub kind: RlEnvironmentKind,
    /// Memory footprint per parallel environment slot in MiB.
    pub memory_per_slot_mb: u32,
    /// CPU threads required per parallel environment slot.
    pub threads_per_slot: u32,
    /// Average step or verification duration in milliseconds.
    pub avg_step_ms: f64,
}

impl RlEnvironmentSuiteSpec {
    /// Preset: **Dream-RSI Evolving World Simulator** for GPU Kernel & Algorithm Discovery.
    pub fn dream_rsi_evolving_world() -> Self {
        Self {
            suite_id: "dream-rsi-evolving-world-v1".into(),
            display_name:
                "Dream-RSI Evolving World (Offline Replay Simulator + Online Kernel Discovery)"
                    .into(),
            kind: RlEnvironmentKind::DreamRsiEvolvingWorld {
                domain: DreamRsiDomain::GpuKernelEngineering,
            },
            memory_per_slot_mb: 128,
            threads_per_slot: 1,
            avg_step_ms: 25.0,
        }
    }

    /// Preset: Rust/Python Verifiable Coding & Unit-Test Sandbox Suite (GRPO Post-Training).
    pub fn rust_python_code_sandbox() -> Self {
        Self {
            suite_id: "kadmium-code-sandbox-v1".into(),
            display_name: "Verifiable Code & Unit-Test Sandbox (WASM/Native Isolated Runner)"
                .into(),
            kind: RlEnvironmentKind::VerifiableCodeAndCompilerSandbox,
            memory_per_slot_mb: 256,
            threads_per_slot: 1,
            avg_step_ms: 180.0,
        }
    }

    /// Preset: Vectorized Physics & Robotics Control Environment (Gymnasium / MuJoCo / Brax style).
    pub fn vectorized_physics_control() -> Self {
        Self {
            suite_id: "kadmium-vec-physics-v1".into(),
            display_name: "Vectorized Continuous Control & Physics Sim (64x Sub-Envs/Core)".into(),
            kind: RlEnvironmentKind::VectorizedPhysicsAndControlSim,
            memory_per_slot_mb: 64,
            threads_per_slot: 1,
            avg_step_ms: 4.5,
        }
    }
}

/// Assignment of a node inside the disaggregated Actor-Environment-Learner & Dream-RSI fabric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRlAssignment {
    pub node_id: NodeId,
    pub hostname: String,
    pub device_kind: DeviceKind,
    pub polite_state: PoliteState,
    pub rl_roles: Vec<RlNodeRole>,
    /// Number of parallel vectorized RL environment or Dream-RSI replay slots hosted on this node.
    pub parallel_env_slots: u32,
    /// Policy model layers hosted on this node (for Actor rollout or Learner backward pass).
    pub policy_layer_range: Option<(u32, u32)>,
    /// Estimated trajectories / dream replay rollouts generated or verified per minute by this node.
    pub rollouts_per_min: f64,
}

/// Complete cluster-wide Disaggregated RL & Dream-RSI Execution Plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisaggregatedRlPlan {
    pub epoch: u64,
    pub policy_model_id: String,
    pub env_suite: RlEnvironmentSuiteSpec,
    pub algorithm: RlAlgorithm,
    /// Maximum off-policy staleness (in Learner gradient steps) tolerated before importance clipping.
    pub max_staleness_steps: u32,
    /// Interval (in Learner steps) at which DeMo/LoRA policy weight deltas ($\Delta \theta$) broadcast to Actors.
    pub policy_sync_interval_steps: u32,
    pub total_vectorized_env_slots: u32,
    pub estimated_cluster_episodes_per_min: f64,
    pub node_assignments: Vec<NodeRlAssignment>,
    /// Latest Dream-RSI offline replay evolution report, if active.
    pub dream_rsi_report: Option<DreamRsiEvolutionReport>,
}

/// Planner that disaggregates RL Environment stepping, Dream-RSI Offline Replay Simulation,
/// Actor rollout inference, and Learner updates across heterogeneous hardware.
pub struct RlTopologyPlanner;

impl RlTopologyPlanner {
    pub fn compute_rl_plan(
        epoch: u64,
        policy_model: &ModelSpec,
        env_suite: &RlEnvironmentSuiteSpec,
        nodes: &[HardwareNode],
    ) -> DisaggregatedRlPlan {
        let mut node_assignments = Vec::with_capacity(nodes.len());
        let mut total_env_slots = 0u32;
        let mut total_episodes_per_min = 0.0f64;

        let is_dream_rsi = matches!(
            env_suite.kind,
            RlEnvironmentKind::DreamRsiEvolvingWorld { .. }
        );

        let mut current_layer = 0u32;
        let gpu_nodes: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                n.available_fast_memory_gb() >= 6.0 && n.polite_state != PoliteState::Draining
            })
            .map(|(i, _)| i)
            .collect();

        let layers_per_gpu = if !gpu_nodes.is_empty() {
            (policy_model.num_layers / (gpu_nodes.len() as u32)).max(1)
        } else {
            policy_model.num_layers
        };

        for (idx, node) in nodes.iter().enumerate() {
            let rho = node.polite_state.availability_multiplier();
            let mut rl_roles = Vec::new();
            let mut parallel_env_slots = 0u32;
            let mut policy_layer_range = None;

            let spare_cpu_threads = match node.device_kind {
                DeviceKind::LinuxSbc
                | DeviceKind::MobilePhone
                | DeviceKind::PhoneFarmChassis
                | DeviceKind::LegacyLaptop
                | DeviceKind::HomeNas => {
                    ((node.cpu_threads as f64) * rho).floor() as u32
                }
                _ => ((node.cpu_threads as f64) * 0.35 * rho).floor() as u32,
            };

            if spare_cpu_threads >= env_suite.threads_per_slot {
                let slots_by_cpu = spare_cpu_threads / env_suite.threads_per_slot.max(1);
                parallel_env_slots = slots_by_cpu.max(1);
                if is_dream_rsi
                    || matches!(
                        node.device_kind,
                        DeviceKind::LinuxSbc
                            | DeviceKind::MobilePhone
                            | DeviceKind::PhoneFarmChassis
                            | DeviceKind::LegacyLaptop
                            | DeviceKind::HomeNas
                    )
                {
                    rl_roles.push(RlNodeRole::DreamRsiReplaySimulatorHost);
                }
                rl_roles.push(RlNodeRole::EnvironmentSandboxHost);
            }

            if gpu_nodes.contains(&idx) {
                let start = current_layer.min(policy_model.num_layers.saturating_sub(1));
                let end = if Some(&idx) == gpu_nodes.last() {
                    policy_model.num_layers.saturating_sub(1)
                } else {
                    (start + layers_per_gpu - 1).min(policy_model.num_layers.saturating_sub(1))
                };
                current_layer = end + 1;
                policy_layer_range = Some((start, end));

                match node.device_kind {
                    DeviceKind::DesktopWorkstation | DeviceKind::CloudBurstGpu => {
                        rl_roles.push(RlNodeRole::PolicyGradientLearner);
                        rl_roles.push(RlNodeRole::RolloutActorInference);
                    }
                    DeviceKind::AppleSiliconMac | DeviceKind::GamingLaptop => {
                        rl_roles.push(RlNodeRole::RolloutActorInference);
                        rl_roles.push(RlNodeRole::PolicyGradientLearner);
                    }
                    _ => {
                        rl_roles.push(RlNodeRole::RolloutActorInference);
                    }
                }
            }

            let env_throughput =
                (parallel_env_slots as f64) * (60000.0 / env_suite.avg_step_ms) * 0.05;
            let actor_throughput = if policy_layer_range.is_some() {
                node.effective_fp16_tflops() * 0.85
            } else {
                0.0
            };
            let rollouts_per_min =
                ((env_throughput * 0.35 + actor_throughput) * 10.0).round() / 10.0;

            total_env_slots += parallel_env_slots;
            total_episodes_per_min += rollouts_per_min;

            node_assignments.push(NodeRlAssignment {
                node_id: node.id.clone(),
                hostname: node.hostname.clone(),
                device_kind: node.device_kind,
                polite_state: node.polite_state,
                rl_roles,
                parallel_env_slots,
                policy_layer_range,
                rollouts_per_min,
            });
        }

        let tree = DiscoveryTree::sample_gpu_kernel_discovery_tree();
        let candidates = DreamRsiSimulator::default_candidate_policies();
        let dream_report =
            DreamRsiSimulator::run_dream_cycle(epoch as u32, &tree, &candidates, nodes);

        let algorithm = if is_dream_rsi {
            RlAlgorithm::DreamRsiOfflineReplayEvolution {
                candidate_policies_per_dream: 64,
                waking_budget_evals: 16,
            }
        } else {
            RlAlgorithm::GrpoGroupRelative { group_size: 8 }
        };

        DisaggregatedRlPlan {
            epoch,
            policy_model_id: policy_model.id.clone(),
            env_suite: env_suite.clone(),
            algorithm,
            max_staleness_steps: 2,
            policy_sync_interval_steps: 4,
            total_vectorized_env_slots: total_env_slots,
            estimated_cluster_episodes_per_min: (total_episodes_per_min * 10.0).round() / 10.0,
            node_assignments,
            dream_rsi_report: Some(dream_report),
        }
    }
}

/// Computes normalized Group-Relative Policy Optimization (**GRPO**) advantages $\hat{A}_i$
/// for a group of $G$ rollout rewards $\{r_1, \dots, r_G\}$ sampled for a single prompt/state:
///
/// $$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$
pub fn compute_grpo_group_advantages(rewards: &[f64]) -> Vec<f64> {
    if rewards.is_empty() {
        return Vec::new();
    }
    let n = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / n;
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let eps = 1e-6;

    rewards
        .iter()
        .map(|r| (((r - mean) / (std + eps)) * 1000.0).round() / 1000.0)
        .collect()
}
