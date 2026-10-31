//! # `kadmium_data::integrations`
//!
//! First-Class Ecosystem Integrations: Hugging Face (`hf://`), GitHub (`gh://`),
//! OCI Container Registries (`oci://`), S3/R2 Object Storage (`s3://`), and Weights & Biases (`wandb://`).

use crate::{HfByteRangeSlicer, SafetensorsIndexManifest};
use kadmium_core::NodeId;
use kadmium_sharding::ClusterShardPlan;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Kind of Hugging Face Hub repository referenced by an `hf://` URI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HfRepoKind {
    /// Model repository containing `model.safetensors.index.json` or `.gguf` shards.
    Model,
    /// Dataset repository (`hf://datasets/org/name`) containing Parquet / JSONL training/RL splits.
    Dataset,
    /// LoRA / DiLoCo checkpoint adapter repository for pushing fine-tuned weights.
    LoraCheckpointAdapter,
}

/// Parsed Hugging Face Hub reference (`hf://deepseek-ai/DeepSeek-V3@main`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfRepoRef {
    pub kind: HfRepoKind,
    pub repo_id: String,
    pub revision: String,
}

impl HfRepoRef {
    /// Parses a Hugging Face URI such as:
    /// - `hf://deepseek-ai/DeepSeek-V3`
    /// - `hf://deepseek-ai/DeepSeek-V3@v1.0`
    /// - `hf://datasets/open-r1/OpenR1-Math-220k@main`
    pub fn parse(uri: &str) -> Result<Self, String> {
        let stripped = uri
            .strip_prefix("hf://")
            .unwrap_or(uri)
            .trim_start_matches('/');
        if stripped.is_empty() {
            return Err("Empty Hugging Face repository URI".into());
        }
        let (path_part, revision) = match stripped.split_once('@') {
            Some((p, r)) if !r.is_empty() => (p, r.to_string()),
            _ => (stripped, "main".to_string()),
        };

        let (kind, repo_id) = if let Some(ds) = path_part.strip_prefix("datasets/") {
            (HfRepoKind::Dataset, ds.to_string())
        } else if path_part.contains("lora") || path_part.contains("adapter") {
            (HfRepoKind::LoraCheckpointAdapter, path_part.to_string())
        } else {
            (HfRepoKind::Model, path_part.to_string())
        };

        if !repo_id.contains('/') {
            return Err(format!(
                "Invalid Hugging Face repo_id '{repo_id}' (expected 'org/name')"
            ));
        }

        Ok(Self {
            kind,
            repo_id,
            revision,
        })
    }

    /// Returns the canonical Hugging Face Hub API URL for `model.safetensors.index.json` or dataset parquet metadata.
    pub fn hub_resolve_url(&self, filename: &str) -> String {
        match self.kind {
            HfRepoKind::Dataset => format!(
                "https://huggingface.co/datasets/{}/resolve/{}/{}",
                self.repo_id, self.revision, filename
            ),
            _ => format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                self.repo_id, self.revision, filename
            ),
        }
    }
}

/// Selective shard download & `.ktb` transcoding assignment for a single node from Hugging Face Hub.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfNodeSelectivePull {
    pub node_id: NodeId,
    pub hostname: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub required_safetensors_files: Vec<String>,
    pub selective_download_gb: f64,
    pub full_repo_avoided_gb: f64,
    pub output_ktb_bank_path: String,
}

/// Cluster-wide selective Hugging Face Hub pull and `.ktb` transcoding plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfSelectivePullPlan {
    pub hf_ref: HfRepoRef,
    pub index_manifest_url: String,
    pub seed_vault_node: String,
    pub total_model_gb: f64,
    pub plan_only: bool,
    pub index_source: String,
    pub files_downloaded: usize,
    pub ktb_banks_compiled: usize,
    pub staged_bytes: u64,
    pub node_pulls: Vec<HfNodeSelectivePull>,
}

/// Dataset split pull plan for `hf://datasets/org/name@revision` URIs (Parquet / JSONL splits, not `.ktb` weight banks).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfDatasetPullPlan {
    pub hf_ref: HfRepoRef,
    pub dataset_info_url: String,
    pub parquet_split_urls: Vec<String>,
    pub plan_only: bool,
    pub files_downloaded: usize,
    pub staged_bytes: u64,
}

/// First-class Hugging Face Hub connector for selective shard pulling, dataset streaming,
/// and DiLoCo/LoRA checkpoint exporting.
pub struct HuggingFaceHubConnector;

impl HuggingFaceHubConnector {
    /// Computes a layer-aware selective shard pull plan from a Hugging Face model repository
    /// (`model.safetensors.index.json`) across a `ClusterShardPlan`.
    /// Rejects `hf://datasets/...` URIs (which must use `plan_dataset_pull`).
    pub fn plan_selective_model_pull(
        hf_uri: &str,
        shard_plan: &ClusterShardPlan,
    ) -> Result<HfSelectivePullPlan, String> {
        Self::plan_selective_model_pull_with_index(hf_uri, shard_plan, None)
    }

    /// Computes a layer-aware selective shard pull plan using a real parsed `SafetensorsIndexManifest`
    /// when provided, or a clearly labelled `unverified_heuristic_estimate` plan when no index manifest is supplied.
    pub fn plan_selective_model_pull_with_index(
        hf_uri: &str,
        shard_plan: &ClusterShardPlan,
        index_manifest: Option<&SafetensorsIndexManifest>,
    ) -> Result<HfSelectivePullPlan, String> {
        let hf_ref = HfRepoRef::parse(hf_uri)?;
        if hf_ref.kind == HfRepoKind::Dataset {
            return Err(format!(
                "URI '{hf_uri}' is a dataset repository (hf://datasets/{}); use plan_dataset_pull instead of model .safetensors/.ktb shard planning",
                hf_ref.repo_id
            ));
        }

        let total_model_gb = match index_manifest {
            Some(idx) if idx.total_size_bytes > 0 => {
                ((idx.total_size_bytes as f64) / 1_000_000_000.0 * 100.0).round() / 100.0
            }
            _ => shard_plan.total_model_weight_gb,
        };
        let per_layer_gb = total_model_gb / (shard_plan.total_layers.max(1) as f64);

        let heuristic_total_shards = 16u32;
        let layers_per_file = (shard_plan.total_layers.max(1))
            .div_ceil(heuristic_total_shards)
            .max(1);

        let mut node_pulls = Vec::new();
        for assign in shard_plan.assignments.iter().filter(|a| a.layer_count > 0) {
            let s_layer = assign.start_layer.unwrap_or(0);
            let e_layer = assign.end_layer.unwrap_or(s_layer);

            let files: Vec<String> = if let Some(manifest) = index_manifest {
                HfByteRangeSlicer::filter_required_shard_files(manifest, s_layer, e_layer)
            } else {
                let first_file_idx = (s_layer / layers_per_file) + 1;
                let last_file_idx = ((e_layer / layers_per_file) + 1).min(heuristic_total_shards);
                (first_file_idx..=last_file_idx)
                    .map(|idx| format!("model-{idx:05}-of-{heuristic_total_shards:05}.safetensors"))
                    .collect()
            };

            let selective_gb = ((assign.layer_count as f64) * per_layer_gb * 100.0).round() / 100.0;
            let avoided_gb = ((total_model_gb - selective_gb).max(0.0) * 100.0).round() / 100.0;
            let slug = hf_ref.repo_id.replace('/', "__");

            node_pulls.push(HfNodeSelectivePull {
                node_id: assign.node_id.clone(),
                hostname: assign.hostname.clone(),
                start_layer: s_layer,
                end_layer: e_layer,
                required_safetensors_files: files,
                selective_download_gb: selective_gb,
                full_repo_avoided_gb: avoided_gb,
                output_ktb_bank_path: format!(
                    "/var/kadmium/ktb/{slug}.L{s_layer:02}-L{e_layer:02}.ktb"
                ),
            });
        }

        let index_source = if index_manifest.is_some() {
            "parsed_safetensors_index".to_string()
        } else {
            "unverified_heuristic_estimate".to_string()
        };

        Ok(HfSelectivePullPlan {
            index_manifest_url: hf_ref.hub_resolve_url("model.safetensors.index.json"),
            hf_ref,
            seed_vault_node: "truenas-core-vault".into(),
            total_model_gb,
            plan_only: true,
            index_source,
            files_downloaded: 0,
            ktb_banks_compiled: 0,
            staged_bytes: 0,
            node_pulls,
        })
    }

    /// Plans a dataset split pull for `hf://datasets/org/name@revision` without fabricating
    /// `model.safetensors.index.json` or `.ktb` weight banks.
    pub fn plan_dataset_pull(hf_uri: &str) -> Result<HfDatasetPullPlan, String> {
        let hf_ref = HfRepoRef::parse(hf_uri)?;
        if hf_ref.kind != HfRepoKind::Dataset {
            return Err(format!(
                "URI '{hf_uri}' is not a dataset repository (expected 'hf://datasets/org/name')"
            ));
        }
        let dataset_info_url = hf_ref.hub_resolve_url("dataset_infos.json");
        let parquet_split_urls = vec![
            hf_ref.hub_resolve_url("data/train-00000-of-00001.parquet"),
            hf_ref.hub_resolve_url("data/test-00000-of-00001.parquet"),
        ];
        Ok(HfDatasetPullPlan {
            hf_ref,
            dataset_info_url,
            parquet_split_urls,
            plan_only: true,
            files_downloaded: 0,
            staged_bytes: 0,
        })
    }
}

/// Parsed GitHub repository reference (`gh://zhengkid/Dream-RSI@main#benchmarks/cuda_kernels`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitHubRepoRef {
    pub owner: String,
    pub repo: String,
    pub git_ref: String,
    pub subpath: Option<String>,
}

impl GitHubRepoRef {
    /// Parses a GitHub URI such as:
    /// - `gh://zhengkid/Dream-RSI`
    /// - `gh://zhengkid/Dream-RSI@a4f91c2#worlds/gpu_kernel_opt`
    /// - `https://github.com/mESmaC/Kadmium`
    pub fn parse(uri: &str) -> Result<Self, String> {
        let stripped = uri
            .strip_prefix("gh://")
            .or_else(|| uri.strip_prefix("https://github.com/"))
            .unwrap_or(uri)
            .trim_start_matches('/')
            .trim_end_matches(".git");

        let (before_hash, subpath) = match stripped.split_once('#') {
            Some((b, s)) if !s.is_empty() => (b, Some(s.to_string())),
            _ => (stripped, None),
        };

        let (repo_part, git_ref) = match before_hash.split_once('@') {
            Some((r, g)) if !g.is_empty() => (r, g.to_string()),
            _ => (before_hash, "main".to_string()),
        };

        let (owner, repo) = repo_part.split_once('/').ok_or_else(|| {
            format!("Invalid GitHub repository '{repo_part}' (expected 'owner/repo')")
        })?;
        if owner.trim().is_empty() || repo.trim().is_empty() {
            return Err(format!(
                "Invalid GitHub repository '{repo_part}' (expected non-empty 'owner/repo')"
            ));
        }

        Ok(Self {
            owner: owner.to_string(),
            repo: repo.to_string(),
            git_ref,
            subpath,
        })
    }
}

/// Automated GitHub Pull Request / Commit CI evaluation dispatched onto a Kadmium Sub-Cluster Slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubSliceCiRun {
    pub gh_ref: GitHubRepoRef,
    pub pinned_commit_sha: String,
    pub assigned_slice_id: String,
    pub sandbox_runtime: String,
    pub mounted: bool,
    pub execution_mode: String,
    pub verified_node_count: usize,
    pub verde_verification_passed: bool,
    pub measured_tokens_per_sec: f64,
    pub dream_rsi_best_speedup: f64,
    pub github_check_conclusion: String,
}

/// First-class GitHub integration for mounting commit-pinned RL / Dream-RSI environments
/// and running Sub-Cluster Slice-on-PR evaluations.
pub struct GitHubConnector;

impl GitHubConnector {
    /// Plans a slice assignment for a GitHub URI without claiming that the repository was fetched,
    /// mounted, or verified when no local checkout was staged.
    pub fn evaluate_repo_on_slice(gh_uri: &str) -> Result<GitHubSliceCiRun, String> {
        let gh_ref = GitHubRepoRef::parse(gh_uri)?;
        let is_dream = gh_ref.repo.to_lowercase().contains("dream")
            || gh_ref.subpath.as_deref().unwrap_or("").contains("kernel");

        let slice_id = if is_dream {
            "slice-04-rl-dream-rsi".to_string()
        } else {
            "slice-02-staging-test".to_string()
        };

        let pinned_commit_sha = gh_ref.git_ref.clone();

        Ok(GitHubSliceCiRun {
            gh_ref,
            pinned_commit_sha,
            assigned_slice_id: slice_id,
            sandbox_runtime: "WasiMicroSandbox + OCI Bridge (unmounted plan)".into(),
            mounted: false,
            execution_mode: "plan_only_unmounted".into(),
            verified_node_count: 0,
            verde_verification_passed: false,
            measured_tokens_per_sec: 0.0,
            dream_rsi_best_speedup: 0.0,
            github_check_conclusion: "unverified_plan_only (repository not fetched or mounted)"
                .into(),
        })
    }

    /// Stages and verifies a GitHub repository from a local checkout path (`local_checkout`),
    /// returning `Err` if no local checkout exists or if the requested `#subpath` is missing.
    pub fn stage_local_repo(
        gh_uri: &str,
        local_checkout: Option<&Path>,
    ) -> Result<GitHubSliceCiRun, String> {
        let mut run = Self::evaluate_repo_on_slice(gh_uri)?;
        let checkout = local_checkout.ok_or_else(|| {
            format!(
                "Cannot mount '{}/{}@{}': no local checkout provided (pass --local-path <dir> or use --plan for an unmounted slice assignment preview)",
                run.gh_ref.owner, run.gh_ref.repo, run.gh_ref.git_ref
            )
        })?;
        if !checkout.exists() || !checkout.is_dir() {
            return Err(format!(
                "Local repository path '{}' does not exist or is not a directory",
                checkout.display()
            ));
        }
        if let Some(sub) = &run.gh_ref.subpath {
            let target = checkout.join(sub);
            if !target.exists() {
                return Err(format!(
                    "Requested repository subpath '{}' was not found inside '{}'",
                    sub,
                    checkout.display()
                ));
            }
        }
        run.mounted = true;
        run.execution_mode = "local_checkout_mounted".into();
        run.sandbox_runtime = format!("Local checkout mounted from {}", checkout.display());
        run.github_check_conclusion = "mounted_unexecuted (awaiting live slice benchmark)".into();
        Ok(run)
    }
}

/// Summary of an ecosystem integration connector supported natively by Kadmium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemIntegrationEntry {
    pub id: String,
    pub provider: String,
    pub uri_scheme: String,
    pub status: String,
    pub placement: String,
    pub example_uri: String,
    pub capabilities: Vec<String>,
}

/// Complete catalog of external integrations (Hugging Face, GitHub, OCI, S3/R2, W&B/OTel)
/// plus a sample selective pull plan and GitHub CI run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemIntegrationsOverview {
    pub connectors: Vec<EcosystemIntegrationEntry>,
    pub sample_hf_pull_plan: HfSelectivePullPlan,
    pub sample_github_ci_run: GitHubSliceCiRun,
}

impl EcosystemIntegrationsOverview {
    pub fn build_for_plan(plan: &ClusterShardPlan) -> Self {
        let sample_hf_pull_plan = HuggingFaceHubConnector::plan_selective_model_pull(
            "hf://deepseek-ai/DeepSeek-V3@main",
            plan,
        )
        .expect("default hf selective pull plan");
        let sample_github_ci_run = GitHubConnector::evaluate_repo_on_slice(
            "gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt",
        )
        .expect("default github slice evaluation");

        let connectors = vec![
            EcosystemIntegrationEntry {
                id: "huggingface-hub".into(),
                provider: "Hugging Face Hub".into(),
                uri_scheme: "hf://".into(),
                status: "connected".into(),
                placement: "local".into(),
                example_uri: "hf://deepseek-ai/DeepSeek-V3@main".into(),
                capabilities: vec![
                    "Selective layer-range Safetensors & GGUF v3 pulling".into(),
                    "On-the-fly 64 KiB .ktb compilation on Home NAS".into(),
                    "Streaming Parquet datasets to SBC/laptop RL workers".into(),
                    "One-command DiLoCo / LoRA adapter push to Hub".into(),
                ],
            },
            EcosystemIntegrationEntry {
                id: "github-repos-actions".into(),
                provider: "GitHub & GitHub Actions".into(),
                uri_scheme: "gh://".into(),
                status: "connected".into(),
                placement: "local".into(),
                example_uri: "gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt".into(),
                capabilities: vec![
                    "Commit-pinned RL & Dream-RSI environment mounting".into(),
                    "Sub-cluster Slice-on-PR evaluation webhooks".into(),
                    "Automated Verde refereed verification check runs".into(),
                    "Multi-arch release binary publishing".into(),
                ],
            },
            EcosystemIntegrationEntry {
                id: "oci-container-bridge".into(),
                provider: "GHCR / Docker OCI Registry".into(),
                uri_scheme: "oci://".into(),
                status: "connected".into(),
                placement: "local".into(),
                example_uri: "oci://ghcr.io/kadmium-ai/gym-cuda-sandbox:latest".into(),
                capabilities: vec![
                    "Rootless OCI container bridge alongside WASI sandboxes".into(),
                    "Deterministic GPU kernel & Gymnasium RL rollouts".into(),
                ],
            },
            EcosystemIntegrationEntry {
                id: "s3-r2-cloudshield-vault".into(),
                provider: "S3 / Cloudflare R2 / MinIO Vault".into(),
                uri_scheme: "s3://".into(),
                status: "standby".into(),
                placement: "cloud".into(),
                example_uri: "s3://kadmium-cloudshield-vault/cameron-lab/deepseek-v3.ktb".into(),
                capabilities: vec![
                    "Pre-warmed .ktb delta chunk Trickle-Sync from Home NAS".into(),
                    "< 400 ms Local <-> Cloud Bidirectional Failover Shield".into(),
                ],
            },
            EcosystemIntegrationEntry {
                id: "wandb-otel-metrics".into(),
                provider: "Weights & Biases + OpenTelemetry".into(),
                uri_scheme: "wandb://".into(),
                status: "connected".into(),
                placement: "local".into(),
                example_uri: "wandb://cameron-lab/kadmium-diloco-70b".into(),
                capabilities: vec![
                    "Live DiLoCo outer-step loss & DeMo DCT compression telemetry".into(),
                    "Dream-RSI discovery tree & compute savings export".into(),
                ],
            },
        ];

        Self {
            connectors,
            sample_hf_pull_plan,
            sample_github_ci_run,
        }
    }
}
