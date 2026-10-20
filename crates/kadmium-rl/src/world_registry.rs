//! Public Dream-RSI World Registry & Evolved Exploration Policy Hub (`kadmium-rl::world_registry`).
//!
//! Implements Phase 5 (`v1.0 GA`) community registry for sharing commit-pinned `gh://` WASI
//! RL verification worlds, recorded causal discovery trees, and evolved PUCT/MCTS exploration
//! policies across Kadmium clusters at zero GPU replay cost.

use crate::DreamRsiDomain;
use serde::{Deserialize, Serialize};

/// A commit-pinned Dream-RSI world and evolved exploration policy published in the public registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredDreamRsiWorld {
    pub world_slug: String,
    pub display_name: String,
    pub github_uri: String,
    pub pinned_commit_sha: String,
    pub domain: DreamRsiDomain,
    pub wasi_sandbox_fuel_limit: u32,
    pub recorded_discovery_nodes: u32,
    pub promoted_policy_id: String,
    pub peak_speedup_or_score: f64,
    pub online_gpu_rollout_savings_pct: f64,
    pub wasi_verifier_digest_fnv1a: u64,
}

/// Community catalog of commit-pinned `gh://` WASI RL worlds and evolved Dream-RSI policies.
pub struct DreamRsiWorldRegistry;

impl DreamRsiWorldRegistry {
    /// Returns the official catalog of commit-pinned Dream-RSI worlds available for offline replay on SBCs & laptops.
    pub fn public_catalog() -> Vec<RegisteredDreamRsiWorld> {
        vec![
            RegisteredDreamRsiWorld {
                world_slug: "gpu-kernel-opt-v4".into(),
                display_name: "Fused Q4_K SwiGLU & Flash-GQA Kernel Discovery".into(),
                github_uri: "gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt".into(),
                pinned_commit_sha: "e4f8a19c7b02d341".into(),
                domain: DreamRsiDomain::GpuKernelEngineering,
                wasi_sandbox_fuel_limit: 512,
                recorded_discovery_nodes: 1_420,
                promoted_policy_id: "pi-dream-rsi-evolved-v4".into(),
                peak_speedup_or_score: 2.31,
                online_gpu_rollout_savings_pct: 71.4,
                wasi_verifier_digest_fnv1a: 0x91a4_77c2_00e1_883b,
            },
            RegisteredDreamRsiWorld {
                world_slug: "kadmium-qstar-autotune".into(),
                display_name: "Self-Tuning FreeToken q* & 1F1B Stage Boundary Optimizer".into(),
                github_uri: "gh://mESmaC/Kadmium@main#worlds/qstar_stage_autotune".into(),
                pinned_commit_sha: "ccd3690f448a0acc".into(),
                domain: DreamRsiDomain::KadmiumSelfTuningAutonomy,
                wasi_sandbox_fuel_limit: 256,
                recorded_discovery_nodes: 860,
                promoted_policy_id: "pi-qstar-waterfill-v3".into(),
                peak_speedup_or_score: 1.48,
                online_gpu_rollout_savings_pct: 64.2,
                wasi_verifier_digest_fnv1a: 0x48cd_2026_f5b7_0048,
            },
            RegisteredDreamRsiWorld {
                world_slug: "lean4-inequality-prover".into(),
                display_name: "Formal Algebraic Inequality & Matrix Bound Synthesis".into(),
                github_uri: "gh://zhengkid/Dream-RSI@main#worlds/formal_math_bounds".into(),
                pinned_commit_sha: "72b09d4418ce3a10".into(),
                domain: DreamRsiDomain::MathematicalOptimization,
                wasi_sandbox_fuel_limit: 1024,
                recorded_discovery_nodes: 2_150,
                promoted_policy_id: "pi-puct-novelty-v6".into(),
                peak_speedup_or_score: 1.94,
                online_gpu_rollout_savings_pct: 68.0,
                wasi_verifier_digest_fnv1a: 0x33f0_9a12_bc44_710e,
            },
            RegisteredDreamRsiWorld {
                world_slug: "riscv-rvv-simd-intrinsics".into(),
                display_name: "ARM64 NEON & RISC-V Vector DotProd Quantized Loop Synthesis".into(),
                github_uri: "gh://mESmaC/Kadmium@main#worlds/sbc_neon_rvv_loops".into(),
                pinned_commit_sha: "af34fc49012bc881".into(),
                domain: DreamRsiDomain::AlgorithmDesign,
                wasi_sandbox_fuel_limit: 384,
                recorded_discovery_nodes: 940,
                promoted_policy_id: "pi-sbc-bramble-v2".into(),
                peak_speedup_or_score: 1.87,
                online_gpu_rollout_savings_pct: 59.5,
                wasi_verifier_digest_fnv1a: 0xd019_55a8_e412_66c0,
            },
        ]
    }

    /// Resolves a world from the registry by slug or `gh://` URI substring.
    pub fn lookup_by_slug_or_uri(query: &str) -> Option<RegisteredDreamRsiWorld> {
        let q = query.trim().to_lowercase();
        Self::public_catalog().into_iter().find(|w| {
            w.world_slug.to_lowercase() == q
                || w.github_uri.to_lowercase().contains(&q)
                || w.promoted_policy_id.to_lowercase() == q
        })
    }
}
