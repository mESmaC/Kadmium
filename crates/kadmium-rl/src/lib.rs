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

use kadmium_core::{DeviceKind, HardwareNode, ModelSpec, NodeId, PoliteState};
use serde::{Deserialize, Serialize};

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
                    DeviceKind::LinuxSbc | DeviceKind::LegacyLaptop | DeviceKind::HomeNas
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

        // Partition policy layers across capable GPU / Unified Memory nodes.
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

            // 1. Assign Environment Sandbox & Dream-RSI Replay Simulator slots to CPU/SBC/NAS/Legacy Laptop threads.
            let spare_cpu_threads = match node.device_kind {
                DeviceKind::LinuxSbc | DeviceKind::LegacyLaptop | DeviceKind::HomeNas => {
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
                        DeviceKind::LinuxSbc | DeviceKind::LegacyLaptop | DeviceKind::HomeNas
                    )
                {
                    rl_roles.push(RlNodeRole::DreamRsiReplaySimulatorHost);
                }
                rl_roles.push(RlNodeRole::EnvironmentSandboxHost);
            }

            // 2. Assign Policy Learner vs. Rollout Actor roles on Accelerator nodes.
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
/// for a group of $G$ rollout rewards $\{r_1, \dots, r_G\}$ sampled for a single prompt/state,
/// eliminating the need for a separate Critic network on memory-constrained hardware:
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

// ============================================================================
// 5. Deterministic WASI Micro-Sandbox & GRPO Verified Rollout Executor
// ============================================================================

/// Deterministic instruction set executed inside a fuel-metered WASI micro-sandbox
/// on `rk3588-sbc-bramble` or `legacy-thinkpad-t480` to score RL code/math rollouts safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxOp {
    /// Pushes the test-case input integer $x$ onto the sandbox stack.
    PushInputX,
    /// Pushes a literal constant onto the sandbox stack.
    PushConst(i64),
    /// Pops top two values and pushes $a + b$.
    Add,
    /// Pops top two values and pushes $a - b$.
    Sub,
    /// Pops top two values and pushes $a \times b$.
    Mul,
    /// Duplicates the top value on the stack (`x -> x, x`).
    Dup,
}

/// Outcome of executing a candidate RL rollout program inside the fuel-metered sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxVerificationResult {
    pub rollout_id: String,
    pub host_node: String,
    pub passed_assertions: usize,
    pub total_assertions: usize,
    pub fuel_consumed: u32,
    pub timed_out_or_trapped: bool,
    pub verifiable_reward: f64,
}

/// Deterministic, fuel-metered micro-sandbox for verifying RL code/math rollouts on edge CPUs/SBCs.
pub struct WasiMicroSandbox {
    pub max_fuel: u32,
    pub max_stack_depth: usize,
}

impl Default for WasiMicroSandbox {
    fn default() -> Self {
        Self {
            max_fuel: 256,
            max_stack_depth: 64,
        }
    }
}

impl WasiMicroSandbox {
    /// Executes `program` against `input_x` with strict instruction fuel and stack bounds.
    pub fn execute_metered(
        &self,
        program: &[SandboxOp],
        input_x: i64,
    ) -> Result<(i64, u32), String> {
        let mut stack: Vec<i64> = Vec::with_capacity(16);
        let mut fuel = 0u32;

        for &op in program {
            fuel += 1;
            if fuel > self.max_fuel {
                return Err("FuelLimitExceeded".into());
            }
            match op {
                SandboxOp::PushInputX => {
                    if stack.len() >= self.max_stack_depth {
                        return Err("StackOverflow".into());
                    }
                    stack.push(input_x);
                }
                SandboxOp::PushConst(c) => {
                    if stack.len() >= self.max_stack_depth {
                        return Err("StackOverflow".into());
                    }
                    stack.push(c);
                }
                SandboxOp::Dup => {
                    let top = *stack.last().ok_or("StackUnderflow")?;
                    if stack.len() >= self.max_stack_depth {
                        return Err("StackOverflow".into());
                    }
                    stack.push(top);
                }
                SandboxOp::Add => {
                    let b = stack.pop().ok_or("StackUnderflow")?;
                    let a = stack.pop().ok_or("StackUnderflow")?;
                    stack.push(a.wrapping_add(b));
                }
                SandboxOp::Sub => {
                    let b = stack.pop().ok_or("StackUnderflow")?;
                    let a = stack.pop().ok_or("StackUnderflow")?;
                    stack.push(a.wrapping_sub(b));
                }
                SandboxOp::Mul => {
                    let b = stack.pop().ok_or("StackUnderflow")?;
                    let a = stack.pop().ok_or("StackUnderflow")?;
                    stack.push(a.wrapping_mul(b));
                }
            }
        }

        let result = stack.pop().ok_or("EmptyReturnStack")?;
        Ok((result, fuel))
    }

    /// Scores a candidate rollout program against a suite of `(input_x, expected_y)` test cases.
    pub fn verify_candidate_program(
        &self,
        rollout_id: impl Into<String>,
        host_node: impl Into<String>,
        program: &[SandboxOp],
        test_cases: &[(i64, i64)],
    ) -> SandboxVerificationResult {
        let mut passed = 0usize;
        let mut total_fuel = 0u32;
        let mut trapped = false;

        for &(x, expected) in test_cases {
            match self.execute_metered(program, x) {
                Ok((actual, fuel)) => {
                    total_fuel += fuel;
                    if actual == expected {
                        passed += 1;
                    }
                }
                Err(_) => {
                    trapped = true;
                    break;
                }
            }
        }

        let total = test_cases.len().max(1);
        let base_reward = ((passed as f64) / (total as f64)) * 0.90;
        // Conciseness bonus (up to +0.10) when all assertions pass
        let efficiency_bonus = if passed == total && !trapped {
            (0.10 * (1.0 - (program.len() as f64 / 20.0).min(1.0))).max(0.0)
        } else {
            0.0
        };
        let reward = ((base_reward + efficiency_bonus).min(1.0) * 1000.0).round() / 1000.0;

        SandboxVerificationResult {
            rollout_id: rollout_id.into(),
            host_node: host_node.into(),
            passed_assertions: passed,
            total_assertions: total,
            fuel_consumed: total_fuel,
            timed_out_or_trapped: trapped,
            verifiable_reward: reward,
        }
    }
}

/// Complete report for a verified GRPO step ($G = 8$ rollouts) executed across disaggregated RL nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoVerifiedStepReport {
    pub task_description: String,
    pub group_size: usize,
    pub rollouts: Vec<SandboxVerificationResult>,
    pub group_advantages: Vec<f64>,
    pub best_rollout_id: String,
    pub mean_group_reward: f64,
}

/// Executes a complete Disaggregated GRPO verification and advantage calculation step.
pub struct GrpoEpochExecutor;

impl GrpoEpochExecutor {
    /// Runs a group of 8 synthesized candidate programs attempting to compute $f(x) = x^2 + 2x + 1$
    /// across `rk3588-sbc-bramble`, `legacy-thinkpad-t480`, and `home-nas-vault` WASI sandboxes.
    pub fn run_verified_grpo_step() -> GrpoVerifiedStepReport {
        let sandbox = WasiMicroSandbox::default();
        // Target polynomial: f(x) = (x + 1)^2 = x^2 + 2x + 1
        let test_cases = [(0, 1), (1, 4), (2, 9), (3, 16), (-1, 0), (5, 36)];

        let candidates: [(&str, &str, Vec<SandboxOp>); 8] = [
            // Rollout 1: Optimal factored form (x + 1) * (x + 1) -> 4 ops, 100% pass!
            (
                "rollout-g01",
                "rk3588-sbc-bramble",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::PushConst(1),
                    SandboxOp::Add,
                    SandboxOp::Dup,
                    SandboxOp::Mul,
                ],
            ),
            // Rollout 2: Expanded form x*x + 2*x + 1 -> 9 ops, 100% pass
            (
                "rollout-g02",
                "rk3588-sbc-bramble",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::Dup,
                    SandboxOp::Mul,
                    SandboxOp::PushInputX,
                    SandboxOp::PushConst(2),
                    SandboxOp::Mul,
                    SandboxOp::Add,
                    SandboxOp::PushConst(1),
                    SandboxOp::Add,
                ],
            ),
            // Rollout 3: Partial solution x^2 + 1 (passes x=0 only)
            (
                "rollout-g03",
                "legacy-thinkpad-t480",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::Dup,
                    SandboxOp::Mul,
                    SandboxOp::PushConst(1),
                    SandboxOp::Add,
                ],
            ),
            // Rollout 4: Linear approximation 3x + 1 (passes x=0, x=1)
            (
                "rollout-g04",
                "legacy-thinkpad-t480",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::PushConst(3),
                    SandboxOp::Mul,
                    SandboxOp::PushConst(1),
                    SandboxOp::Add,
                ],
            ),
            // Rollout 5: Malformed stack underflow trap
            (
                "rollout-g05",
                "home-nas-vault",
                vec![SandboxOp::PushInputX, SandboxOp::Add],
            ),
            // Rollout 6: Constant 1 (passes x=0)
            (
                "rollout-g06",
                "home-nas-vault",
                vec![SandboxOp::PushConst(1)],
            ),
            // Rollout 7: Exact factored form with redundant +0
            (
                "rollout-g07",
                "rk3588-sbc-bramble",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::PushConst(1),
                    SandboxOp::Add,
                    SandboxOp::Dup,
                    SandboxOp::Mul,
                    SandboxOp::PushConst(0),
                    SandboxOp::Add,
                ],
            ),
            // Rollout 8: Off-by-one x^2 + 2x
            (
                "rollout-g08",
                "legacy-thinkpad-t480",
                vec![
                    SandboxOp::PushInputX,
                    SandboxOp::Dup,
                    SandboxOp::Mul,
                    SandboxOp::PushInputX,
                    SandboxOp::PushConst(2),
                    SandboxOp::Mul,
                    SandboxOp::Add,
                ],
            ),
        ];

        let mut rollouts = Vec::with_capacity(8);
        for (id, host, prog) in &candidates {
            rollouts.push(sandbox.verify_candidate_program(*id, *host, prog, &test_cases));
        }

        let rewards: Vec<f64> = rollouts.iter().map(|r| r.verifiable_reward).collect();
        let advantages = compute_grpo_group_advantages(&rewards);
        let mean_reward =
            ((rewards.iter().sum::<f64>() / (rewards.len() as f64)) * 1000.0).round() / 1000.0;

        let best_rollout_id = rollouts
            .iter()
            .max_by(|a, b| {
                a.verifiable_reward
                    .partial_cmp(&b.verifiable_reward)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.rollout_id.clone())
            .unwrap_or_else(|| "rollout-g01".into());

        GrpoVerifiedStepReport {
            task_description: "Synthesize minimal bytecode for f(x) = x^2 + 2x + 1".into(),
            group_size: rollouts.len(),
            rollouts,
            group_advantages: advantages,
            best_rollout_id,
            mean_group_reward: mean_reward,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_core::sample_personal_cluster;

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
}
