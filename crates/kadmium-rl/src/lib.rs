//! # `kadmium-rl`
//!
//! First-class **Distributed Reinforcement Learning (RL) Environment & Dream-RSI Fabric** for Kadmium.
//!
//! ### 1. Disaggregated Actor-Environment-Learner RL
//! Disaggregates RL workloads across heterogeneous hardware into specialized pools:
//! - **Environment & Sandbox Pool (`EnvironmentSandboxHost`):** Runs hundreds of parallel vectorized
//!   RL environments, code/compiler sandboxes, formal math verifiers, or physics/game simulations
//!   on **Linux SBCs, older laptops, and Home NAS CPU threads**—preventing GPUs from sitting idle
//!   waiting on CPU-bound `env.step()` calls.
//! - **Rollout Actor Pool (`RolloutActorInference`):** Uses **Gaming Laptops & Apple Silicon Macs**
//!   to sample high-throughput policy trajectories ($\pi_\theta$) with quantized `.ktb` weights.
//! - **Policy Learner Pool (`PolicyGradientLearner`):** Uses the **Main Desktop PC (and optional
//!   Burst Cloud GPU)** to compute **GRPO / RLOO / PPO / IMPALA V-Trace** policy updates.
//!
//! ### 2. First-Class Dream-RSI Support (`zhengkid/Dream-RSI`, arXiv:2609.14858)
//! Implements **Recursive Self-Improvement through Evolving Worlds**:
//! - **Online Waking Discovery:** Accelerator nodes explore complex domains (GPU kernel engineering,
//!   algorithm design, mathematical optimization), logging full **Discovery Trees** (proposals,
//!   mutations, benchmark scores, and execution costs) to the Home NAS.
//! - **Offline Replay Simulation ("Dreaming"):** Instead of burning expensive live LLM calls to tune
//!   exploration strategies, **Linux SBCs, Legacy Laptops, and NAS CPU threads** treat the accumulated
//!   discovery history as an **Evolving Replay Simulator**. Candidate exploration policies traverse
//!   recorded trees offline (re-ordering branches, tuning parallel wave widths, and pruning thresholds)
//!   at **zero GPU cost** to discover superior exploration policies that drive the next online wave.

use kadmium_core::{DeviceKind, HardwareNode, NodeId};
use serde::{Deserialize, Serialize};

pub mod mcts;
pub mod sandbox_grpo;
pub mod world_registry;

pub use mcts::*;
pub use sandbox_grpo::*;
pub use world_registry::*;

/// Target discovery domain for a **Dream-RSI** (*Recursive Self-Improvement through Evolving Worlds*) loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DreamRsiDomain {
    /// Discovering faster CUDA / Metal / Vulkan tensor & MoE kernels.
    GpuKernelEngineering,
    /// Autonomous synthesis and optimization of combinatorial or systems algorithms.
    AlgorithmDesign,
    /// Symbolic & numerical mathematical bounds, constructions, and theorem proving.
    MathematicalOptimization,
    /// Self-tuning Kadmium's own asymmetric sharding, $q^*$ ratios, and 1F1B pipeline schedules.
    KadmiumSelfTuningAutonomy,
}

/// Category of RL or Recursive Self-Improvement environment hosted on the Kadmium mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RlEnvironmentKind {
    /// **Dream-RSI Evolving World Simulator:** Combines online discovery tree expansion with
    /// offline historical replay simulation ("dreaming") to evolve exploration policies without weight updates.
    DreamRsiEvolvingWorld { domain: DreamRsiDomain },
    /// Isolated Wasmtime / container / process sandbox compiling code & running unit tests (SWE / Coding RL).
    VerifiableCodeAndCompilerSandbox,
    /// Formal theorem proving & symbolic math verifier (Lean 4, SymPy, SMT solver).
    FormalMathAndLogicProver,
    /// Multi-turn interactive agent environment (terminal shell, browser DOM, MCP tool execution).
    MultiTurnAgentToolEnv,
    /// Vectorized classic control, robotics, physics, or game simulation (Gymnasium / MuJoCo / Brax / Godot).
    VectorizedPhysicsAndControlSim,
    /// Competitive or cooperative multi-agent self-play arena (PettingZoo / multi-policy league).
    MultiAgentSelfPlayArena,
}

/// Distributed RL or Recursive Self-Improvement optimization algorithm orchestrated across the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RlAlgorithm {
    /// **Dream-RSI (Recursive Self-Improvement through Evolving Worlds):**
    /// Offline exploration-policy evolution over historical discovery replay trees + online waking waves.
    DreamRsiOfflineReplayEvolution {
        candidate_policies_per_dream: u32,
        waking_budget_evals: u32,
    },
    /// Group-Relative Policy Optimization (eliminates the separate Critic model, saving ~50% memory!).
    GrpoGroupRelative { group_size: u32 },
    /// REINFORCE Leave-One-Out (low-variance critic-free policy gradient for LLM & agent post-training).
    RlooLeaveOneOut { samples_per_prompt: u32 },
    /// Clipped Proximal Policy Optimization with Generalized Advantage Estimation (GAE).
    PpoClippedGae,
    /// Asynchronous off-policy V-Trace (tolerates staleness across slow WAN / heterogeneous actors).
    ImpalaVTraceAsync { max_policy_staleness: u32 },
}

/// Specialized role assigned to a node within the Disaggregated RL & Dream-RSI Topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RlNodeRole {
    /// Executes parallel `env.reset()` / `env.step(action)` instances and deterministic reward verifiers.
    EnvironmentSandboxHost,
    /// **Dream-RSI Offline Replay Host:** Traverses historical discovery trees on CPU/SBC/NAS threads
    /// to evaluate and evolve candidate exploration policies offline at zero live LLM cost.
    DreamRsiReplaySimulatorHost,
    /// Generates trajectories / online discovery proposals from policy $\pi_\theta$ at high throughput.
    RolloutActorInference,
    /// Computes policy gradients ($\nabla_\theta \mathcal{L}_{\text{RL}}$) and streams weight deltas to Actors.
    PolicyGradientLearner,
}

/// Single step transition returned by a Kadmium RL environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvStepOutcome {
    pub env_instance_id: String,
    pub observation: String,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    /// Deterministic verification hash of the sandbox execution state (for Gensyn/BOINC auditability).
    pub verification_hash: String,
    pub step_latency_ms: f64,
}

/// Core trait implemented by native Rust, WASM, or bridged Python/Gymnasium RL environments in Kadmium.
pub trait KadmiumRlEnvironment: Send + Sync {
    /// Unique identifier for the environment suite (e.g., `"dream-rsi-kernel-opt-v1"`).
    fn env_suite_id(&self) -> &str;

    /// Classification of the environment.
    fn kind(&self) -> RlEnvironmentKind;

    /// Resets an environment slot with a deterministic seed and returns the initial observation/prompt.
    fn reset(&mut self, seed: u64) -> String;

    /// Executes an action (token sequence, tool call, or continuous control vector) and returns the transition.
    fn step(&mut self, action: &str) -> EnvStepOutcome;
}

// ================================================================================================
// Dream-RSI: Recursive Self-Improvement through Evolving Worlds (Offline Replay Simulator)
// ================================================================================================

/// A single recorded node inside a Dream-RSI historical **Discovery Tree** ($\mathcal{T}$).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryTreeNode {
    pub node_id: u32,
    pub parent_id: Option<u32>,
    pub depth: u32,
    /// Label of the candidate mutation / idea (e.g., `"Tile-128x64 SharedMem Swizzle"`).
    pub proposal_label: String,
    /// Realized benchmark score / objective value achieved when this node was evaluated online.
    pub realized_score: f64,
    /// Prior heuristic / novelty estimate visible to the exploration policy *before* expanding children.
    pub prior_novelty: f64,
    /// Live compute + evaluation cost (in normalized agent-call units) incurred to evaluate this node.
    pub evaluation_cost: f64,
}

/// A historical discovery tree logged during online waking exploration waves and stored on the Home NAS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryTree {
    pub tree_id: String,
    pub domain: DreamRsiDomain,
    pub baseline_score: f64,
    pub nodes: Vec<DiscoveryTreeNode>,
}

impl DiscoveryTree {
    /// Preset historical discovery tree from a GPU Kernel Engineering campaign
    /// (optimizing a quantized MoE fused dequant-GEMM kernel across consumer GPUs).
    pub fn sample_gpu_kernel_discovery_tree() -> Self {
        Self {
            tree_id: "tree-moe-gemm-kernel-wave-04".into(),
            domain: DreamRsiDomain::GpuKernelEngineering,
            baseline_score: 1.00, // 1.00x baseline throughput
            nodes: vec![
                DiscoveryTreeNode {
                    node_id: 0,
                    parent_id: None,
                    depth: 0,
                    proposal_label: "Root: Baseline FP16xQ4 MoE GEMM Kernel (1.00x)".into(),
                    realized_score: 1.00,
                    prior_novelty: 0.50,
                    evaluation_cost: 1.0,
                },
                // Branch A: Naive unroll (local optimum trap that wastes online calls if not pruned)
                DiscoveryTreeNode {
                    node_id: 1,
                    parent_id: Some(0),
                    depth: 1,
                    proposal_label: "Branch A1: Naive Register Loop Unroll x8".into(),
                    realized_score: 1.04,
                    prior_novelty: 0.25,
                    evaluation_cost: 1.0,
                },
                DiscoveryTreeNode {
                    node_id: 2,
                    parent_id: Some(1),
                    depth: 2,
                    proposal_label: "Branch A2: Unroll x16 (Register Spill Regression)".into(),
                    realized_score: 0.88,
                    prior_novelty: 0.15,
                    evaluation_cost: 1.0,
                },
                DiscoveryTreeNode {
                    node_id: 3,
                    parent_id: Some(2),
                    depth: 3,
                    proposal_label: "Branch A3: Unroll x32 (Severe Occupancy Drop)".into(),
                    realized_score: 0.72,
                    prior_novelty: 0.10,
                    evaluation_cost: 1.0,
                },
                // Branch B: Moderate shared memory staging
                DiscoveryTreeNode {
                    node_id: 4,
                    parent_id: Some(0),
                    depth: 1,
                    proposal_label: "Branch B1: Async Shared-Memory Double Buffering".into(),
                    realized_score: 1.38,
                    prior_novelty: 0.78,
                    evaluation_cost: 1.0,
                },
                DiscoveryTreeNode {
                    node_id: 5,
                    parent_id: Some(4),
                    depth: 2,
                    proposal_label: "Branch B2: Bank-Conflict-Free XOR Shared Swizzle".into(),
                    realized_score: 1.74,
                    prior_novelty: 0.86,
                    evaluation_cost: 1.0,
                },
                DiscoveryTreeNode {
                    node_id: 6,
                    parent_id: Some(5),
                    depth: 3,
                    proposal_label:
                        "Branch B3: Warp-Specialized TensorCore + L2 Prefetch (Breakthrough 2.31x)"
                            .into(),
                    realized_score: 2.31,
                    prior_novelty: 0.94,
                    evaluation_cost: 1.0,
                },
                // Branch C: Sub-optimal threadblock size
                DiscoveryTreeNode {
                    node_id: 7,
                    parent_id: Some(0),
                    depth: 1,
                    proposal_label: "Branch C1: 512-Thread Flat Block Layout".into(),
                    realized_score: 0.92,
                    prior_novelty: 0.30,
                    evaluation_cost: 1.0,
                },
                DiscoveryTreeNode {
                    node_id: 8,
                    parent_id: Some(7),
                    depth: 2,
                    proposal_label: "Branch C2: 1024-Thread Flat Block (Warp Stall)".into(),
                    realized_score: 0.81,
                    prior_novelty: 0.20,
                    evaluation_cost: 1.0,
                },
            ],
        }
    }
}

/// A candidate meta-exploration policy ($\pi_{\text{explore}}$) evaluated inside Dream-RSI's
/// offline replay simulator without modifying base LLM weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationPolicySpec {
    pub policy_id: String,
    pub description: String,
    /// Weight assigned to parent realized score vs. prior novelty when ordering frontier branches.
    pub exploitation_weight: f64,
    pub novelty_weight: f64,
    /// Parallel batch grouping size ($B$) per exploration wave.
    pub parallel_wave_width: u32,
    /// Early-stopping / pruning ratio: prune any child branch whose score falls below
    /// `parent_score * prune_ratio_threshold`.
    pub prune_ratio_threshold: f64,
}

/// Offline replay simulation outcome for a single candidate `ExplorationPolicySpec`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamRsiPolicyEvaluation {
    pub policy: ExplorationPolicySpec,
    /// Traversal order of `node_id`s chosen by this exploration policy inside the replay simulator.
    pub simulated_traversal_order: Vec<u32>,
    /// Number of dead-end branches pruned early by this policy's stopping rule.
    pub pruned_branches: u32,
    /// Highest discovery score reached.
    pub peak_discovery_score: f64,
    /// Cumulative evaluation cost required to reach `peak_discovery_score`.
    pub cost_to_reach_peak: f64,
    /// Percentage reduction in online agent/evaluation calls compared to exhaustive baseline traversal.
    pub discovery_cost_savings_pct: f64,
}

/// Summary report of a **Dream-RSI Offline Dreaming Cycle** executed across SBC/CPU replay hosts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamRsiEvolutionReport {
    pub generation: u32,
    pub tree_id: String,
    pub domain: DreamRsiDomain,
    pub evaluated_policies: Vec<DreamRsiPolicyEvaluation>,
    pub promoted_policy_id: String,
    pub peak_speedup_achieved: f64,
    pub cost_savings_pct: f64,
    pub executed_on_replay_nodes: Vec<NodeId>,
}

/// The **Dream-RSI Replay Simulator** that evaluates and evolves exploration policies offline
/// using recorded discovery history.
pub struct DreamRsiSimulator;

impl DreamRsiSimulator {
    /// Default pool of candidate exploration policies to test during an offline dreaming cycle.
    pub fn default_candidate_policies() -> Vec<ExplorationPolicySpec> {
        vec![
            ExplorationPolicySpec {
                policy_id: "pi-baseline-bfs".into(),
                description: "Fixed Breadth-First Exploration (No Pruning, Uniform Order)".into(),
                exploitation_weight: 1.0,
                novelty_weight: 0.0,
                parallel_wave_width: 1,
                prune_ratio_threshold: 0.0,
            },
            ExplorationPolicySpec {
                policy_id: "pi-greedy-no-novelty".into(),
                description: "Pure Greedy Score Exploitation (Prone to Local Optima)".into(),
                exploitation_weight: 1.0,
                novelty_weight: 0.1,
                parallel_wave_width: 2,
                prune_ratio_threshold: 0.85,
            },
            ExplorationPolicySpec {
                policy_id: "pi-dream-rsi-evolved-v4".into(),
                description:
                    "Dream-RSI Evolved Policy (Novelty-Guided Best-First + Regret Pruning)".into(),
                exploitation_weight: 1.2,
                novelty_weight: 1.6,
                parallel_wave_width: 2,
                prune_ratio_threshold: 0.96,
            },
        ]
    }

    /// Simulates traversing a recorded `DiscoveryTree` under a candidate `ExplorationPolicySpec`.
    pub fn evaluate_policy_on_tree(
        tree: &DiscoveryTree,
        policy: &ExplorationPolicySpec,
    ) -> DreamRsiPolicyEvaluation {
        let total_tree_cost: f64 = tree.nodes.iter().map(|n| n.evaluation_cost).sum();
        let global_peak: f64 = tree
            .nodes
            .iter()
            .map(|n| n.realized_score)
            .fold(tree.baseline_score, f64::max);

        // Frontier priority queue (simulated best-first traversal with pruning rule).
        let mut frontier: Vec<&DiscoveryTreeNode> = tree
            .nodes
            .iter()
            .filter(|n| n.parent_id.is_none())
            .collect();

        let mut visited_order = Vec::new();
        let mut pruned_branches = 0u32;
        let mut cumulative_cost = 0.0;
        let mut cost_to_reach_peak = total_tree_cost;
        let mut best_so_far = tree.baseline_score;

        while !frontier.is_empty() {
            // Sort frontier using ONLY pre-evaluation observables (parent realized score, depth,
            // and prior novelty), preserving causal replay semantics in the Dream-RSI simulator:
            frontier.sort_by(|a, b| {
                let parent_score_a = a
                    .parent_id
                    .and_then(|pid| tree.nodes.iter().find(|n| n.node_id == pid))
                    .map(|p| p.realized_score)
                    .unwrap_or(tree.baseline_score);
                let parent_score_b = b
                    .parent_id
                    .and_then(|pid| tree.nodes.iter().find(|n| n.node_id == pid))
                    .map(|p| p.realized_score)
                    .unwrap_or(tree.baseline_score);

                // Breadth-first policies (novelty_weight == 0.0) prioritize shallow depth first.
                if policy.novelty_weight <= f64::EPSILON {
                    return a
                        .depth
                        .cmp(&b.depth)
                        .then_with(|| a.node_id.cmp(&b.node_id));
                }

                let priority_a = (parent_score_a * policy.exploitation_weight)
                    + (a.prior_novelty * policy.novelty_weight);
                let priority_b = (parent_score_b * policy.exploitation_weight)
                    + (b.prior_novelty * policy.novelty_weight);
                priority_b
                    .partial_cmp(&priority_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let current = frontier.remove(0);
            visited_order.push(current.node_id);
            cumulative_cost += current.evaluation_cost;

            if current.realized_score > best_so_far {
                best_so_far = current.realized_score;
            }
            if (best_so_far - global_peak).abs() < 1e-6 && cost_to_reach_peak == total_tree_cost {
                cost_to_reach_peak = cumulative_cost;
            }

            // Check stopping / pruning condition before expanding children of `current`:
            if let Some(pid) = current.parent_id
                && let Some(parent) = tree.nodes.iter().find(|n| n.node_id == pid)
                && current.realized_score < parent.realized_score * policy.prune_ratio_threshold
            {
                pruned_branches += 1;
                continue;
            }

            // Expand children into the frontier.
            for child in tree
                .nodes
                .iter()
                .filter(|n| n.parent_id == Some(current.node_id))
            {
                frontier.push(child);
            }
        }

        let savings_pct =
            (((total_tree_cost - cost_to_reach_peak) / total_tree_cost.max(1.0)) * 100.0).max(0.0);

        DreamRsiPolicyEvaluation {
            policy: policy.clone(),
            simulated_traversal_order: visited_order,
            pruned_branches,
            peak_discovery_score: best_so_far,
            cost_to_reach_peak,
            discovery_cost_savings_pct: (savings_pct * 10.0).round() / 10.0,
        }
    }

    /// Runs a full offline **Dream-RSI Dreaming Cycle** across the cluster's replay hosts
    /// and promotes the exploration policy that reaches the highest discovery score at minimum cost.
    pub fn run_dream_cycle(
        generation: u32,
        tree: &DiscoveryTree,
        candidates: &[ExplorationPolicySpec],
        cluster_nodes: &[HardwareNode],
    ) -> DreamRsiEvolutionReport {
        let mut evaluations: Vec<DreamRsiPolicyEvaluation> = candidates
            .iter()
            .map(|p| Self::evaluate_policy_on_tree(tree, p))
            .collect();

        // Sort so the best policy (highest peak score, lowest cost_to_reach_peak, most pruned dead ends) is first.
        evaluations.sort_by(|a, b| {
            b.peak_discovery_score
                .partial_cmp(&a.peak_discovery_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.cost_to_reach_peak
                        .partial_cmp(&b.cost_to_reach_peak)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| b.pruned_branches.cmp(&a.pruned_branches))
        });

        let winner = evaluations
            .first()
            .expect("at least one candidate exploration policy");

        let replay_nodes: Vec<NodeId> = cluster_nodes
            .iter()
            .filter(|n| {
                matches!(
                    n.device_kind,
                    DeviceKind::LinuxSbc
                        | DeviceKind::MobilePhone
                        | DeviceKind::PhoneFarmChassis
                        | DeviceKind::LegacyLaptop
                        | DeviceKind::HomeNas
                )
            })
            .map(|n| n.id.clone())
            .collect();

        DreamRsiEvolutionReport {
            generation,
            tree_id: tree.tree_id.clone(),
            domain: tree.domain,
            promoted_policy_id: winner.policy.policy_id.clone(),
            peak_speedup_achieved: winner.peak_discovery_score,
            cost_savings_pct: winner.discovery_cost_savings_pct,
            evaluated_policies: evaluations,
            executed_on_replay_nodes: replay_nodes,
        }
    }
}

pub mod topology;
pub use topology::*;

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_core::{ModelSpec, sample_personal_cluster};

    #[test]
    fn test_disaggregated_rl_plan_assigns_sbcs_and_laptops_to_env_sandboxes() {
        let cluster = sample_personal_cluster();
        let model = ModelSpec::llama_3_70b_lora();
        let env_suite = RlEnvironmentSuiteSpec::rust_python_code_sandbox();

        let plan = RlTopologyPlanner::compute_rl_plan(1, &model, &env_suite, &cluster);
        assert!(plan.total_vectorized_env_slots >= 40);

        let sbc_assign = &plan.node_assignments[5];
        assert!(
            sbc_assign
                .rl_roles
                .contains(&RlNodeRole::EnvironmentSandboxHost)
        );
        assert!(
            sbc_assign
                .rl_roles
                .contains(&RlNodeRole::DreamRsiReplaySimulatorHost)
        );
        assert_eq!(sbc_assign.parallel_env_slots, 24);
    }

    #[test]
    fn test_dream_rsi_offline_replay_identifies_high_savings_exploration_policy() {
        let cluster = sample_personal_cluster();
        let tree = DiscoveryTree::sample_gpu_kernel_discovery_tree();
        let candidates = DreamRsiSimulator::default_candidate_policies();

        let report = DreamRsiSimulator::run_dream_cycle(1, &tree, &candidates, &cluster);
        assert_eq!(report.promoted_policy_id, "pi-dream-rsi-evolved-v4");
        assert!((report.peak_speedup_achieved - 2.31).abs() < 1e-6);
        // Reaching the 2.31x kernel in 4 steps instead of 9 steps saves >50% of online calls!
        assert!(report.cost_savings_pct >= 50.0);

        let catalog = DreamRsiWorldRegistry::public_catalog();
        assert_eq!(catalog.len(), 4);
        assert!(DreamRsiWorldRegistry::lookup_by_slug_or_uri("gpu-kernel-opt-v4").is_some());
    }

    #[test]
    fn test_grpo_group_advantages_zero_mean_unit_variance() {
        let rewards = vec![0.0, 0.25, 0.5, 1.0, 0.0, 1.0, 0.75, 0.5];
        let advantages = compute_grpo_group_advantages(&rewards);
        assert_eq!(advantages.len(), 8);

        let sum: f64 = advantages.iter().sum();
        assert!(sum.abs() < 0.01);
        assert!(advantages[3] > 1.0);
        assert!(advantages[0] < -1.0);
    }

    #[test]
    fn test_wasi_micro_sandbox_and_verified_grpo_step() {
        let step = GrpoEpochExecutor::run_verified_grpo_step();
        assert_eq!(step.group_size, 8);
        assert_eq!(step.best_rollout_id, "rollout-g01");
        assert!(step.rollouts[0].verifiable_reward > step.rollouts[1].verifiable_reward);
        assert!(step.rollouts[4].timed_out_or_trapped);
        assert_eq!(step.rollouts[4].verifiable_reward, 0.0);
        assert!(step.group_advantages[0] > 1.0);
    }

    #[test]
    fn test_dream_rsi_mcts_puct_converges_to_breakthrough_branch() {
        let tree = DiscoveryTree::sample_gpu_kernel_discovery_tree();
        let mcts = DreamRsiMctsEngine::run_puct_search(&tree, 32, 1.414);

        assert_eq!(mcts.best_leaf_node_id, 6);
        assert!((mcts.best_leaf_score - 2.31).abs() < 1e-6);
        assert_eq!(mcts.optimal_path_node_ids, vec![0, 4, 5, 6]);
    }
}
