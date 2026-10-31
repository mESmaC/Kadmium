//! End-to-end regression tests for the Claim-Evidence Matrix (`#32`–`#40`).

use kadmium_core::{
    ModelSpec, WorkloadKind, external_hardware_probes_allowed, sample_personal_cluster,
};
use kadmium_data::{GitHubConnector, HuggingFaceHubConnector, SafetensorsIndexManifest};
use kadmium_gateway::{GatewayState, KadmiumMcpServer, McpJsonRpcRequest};
use kadmium_network::UsbCDirectInterconnectScanner;
use kadmium_rl::WasiMicroSandbox;
use kadmium_runtime::{HardwareKernelDispatcher, KernelBenchmarkParticipant};
use kadmium_sharding::{HybridCloudArbitragePlanner, ShardPlanner};
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn issue_32_external_hardware_probes_are_disabled_under_test_except_loopback() {
    assert!(!external_hardware_probes_allowed(None));
    assert!(!external_hardware_probes_allowed(Some(
        "192.168.40.237:7860"
    )));
    assert!(!external_hardware_probes_allowed(Some(
        "192.168.40.75:7859"
    )));
    assert!(external_hardware_probes_allowed(Some("127.0.0.1:7860")));
    assert!(external_hardware_probes_allowed(Some("localhost:7860")));
}

#[test]
fn issue_33_mcp_rl_sandbox_verifier_rejects_broken_and_wrong_code() {
    let sandbox = WasiMicroSandbox {
        max_fuel: 10_000,
        max_stack_depth: 64,
    };
    let tests = vec![(-3, 16), (0, -5), (2, 11), (4, 51)];

    let broken = sandbox.verify_candidate_code(
        "rollout-broken",
        "node-main-pc",
        "THIS IS NOT VALID RUST OR PYTHON {{{{",
        &tests,
    );
    assert!(broken.compile_error.is_some());
    assert_eq!(broken.passed_assertions, 0);
    assert!(!broken.timed_out_or_trapped);
    assert_eq!(broken.verifiable_reward, 0.0);

    let wrong = sandbox.verify_candidate_code(
        "rollout-wrong",
        "node-main-pc",
        "fn solve(x: i64) -> i64 { 0 }",
        &tests,
    );
    assert!(wrong.compile_error.is_none());
    assert_eq!(wrong.passed_assertions, 0);
    assert_eq!(wrong.verifiable_reward, 0.0);

    let correct = sandbox.verify_candidate_code(
        "rollout-correct",
        "node-main-pc",
        "fn solve(x: i64) -> i64 { 3 * x * x + 2 * x - 5 }",
        &tests,
    );
    assert!(correct.compile_error.is_none());
    assert_eq!(correct.passed_assertions, correct.total_assertions);
    assert!(correct.verifiable_reward > 0.9);

    // Also verify via MCP tool dispatch:
    let mut state = GatewayState::new_live_unconnected();
    let req_broken = McpJsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "kadmium_verify_rl_sandbox",
            "arguments": {
                "candidate_code": "THIS IS NOT VALID RUST OR PYTHON {{{{"
            }
        })),
    };
    let resp_broken = KadmiumMcpServer::handle_request(&mut state, req_broken);
    let result = resp_broken.result.unwrap();
    assert_eq!(result["isError"], json!(true));
}

#[test]
fn issue_34_github_pull_verifies_git_checkout_and_rejects_unmounted_or_mcp_local_path() {
    let uri = "gh://nonexistent-owner-404/nonexistent-repo-404@deadbeef";
    let eval = GitHubConnector::evaluate_repo_on_slice(uri).unwrap();
    assert_eq!(eval.gh_ref.git_ref, "deadbeef");
    assert_eq!(eval.pinned_commit_sha, "unverified_unmounted");
    assert_ne!(eval.pinned_commit_sha, "506a788c91d4f02e");
    assert!(!eval.mounted);
    assert_eq!(eval.execution_mode, "plan_only_unmounted");
    assert_eq!(eval.verified_node_count, 0);
    assert!(!eval.verde_verification_passed);
    assert_eq!(eval.measured_tokens_per_sec, 0.0);

    // Staging without a valid local path or on a plain non-.git directory must fail:
    let err = GitHubConnector::stage_local_repo(
        uri,
        Some(std::path::Path::new(
            "target/definitely-missing-checkout-404",
        )),
    )
    .unwrap_err();
    assert!(err.contains("does not exist"));

    let plain_dir = std::env::temp_dir().join(format!("kadmium-non-git-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&plain_dir);
    std::fs::create_dir_all(&plain_dir).unwrap();
    let non_git_err = GitHubConnector::stage_local_repo(uri, Some(&plain_dir)).unwrap_err();
    assert!(non_git_err.contains("missing .git metadata"));

    // Create a minimal .git checkout with matching branch, commit SHA, remote URL, and subpath:
    let git_dir = plain_dir.join(".git");
    std::fs::create_dir_all(git_dir.join("refs/heads")).unwrap();
    std::fs::create_dir_all(plain_dir.join("worlds/gpu_kernel_opt")).unwrap();
    std::fs::write(git_dir.join("HEAD"), "ref: refs/heads/main\n").unwrap();
    std::fs::write(
        git_dir.join("refs/heads/main"),
        "a4f91c20123456789abcdef0123456789abcdef0\n",
    )
    .unwrap();
    std::fs::write(
        git_dir.join("config"),
        "[remote \"origin\"]\n\turl = https://github.com/zhengkid/Dream-RSI.git\n",
    )
    .unwrap();

    let mounted = GitHubConnector::stage_local_repo(
        "gh://zhengkid/Dream-RSI@main#worlds/gpu_kernel_opt",
        Some(&plain_dir),
    )
    .unwrap();
    assert!(mounted.mounted);
    assert_eq!(
        mounted.pinned_commit_sha,
        "a4f91c20123456789abcdef0123456789abcdef0"
    );

    // Mismatched ref or mismatched repo owner must fail:
    assert!(
        GitHubConnector::stage_local_repo("gh://zhengkid/Dream-RSI@other-branch", Some(&plain_dir))
            .is_err()
    );
    assert!(
        GitHubConnector::stage_local_repo("gh://wrong-owner/Dream-RSI@main", Some(&plain_dir))
            .is_err()
    );
    let _ = std::fs::remove_dir_all(&plain_dir);

    // Over MCP, local_path must be rejected and plan_only: false on gh:// must report isError: true
    let mut state = GatewayState::new_live_unconnected();
    let mcp_local_path = KadmiumMcpServer::handle_request(
        &mut state,
        McpJsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(34)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "kadmium_pull_artifact",
                "arguments": {
                    "uri": "gh://zhengkid/Dream-RSI@main",
                    "local_path": "/etc"
                }
            })),
        },
    );
    assert!(mcp_local_path.error.is_some());

    let mcp_unplanned = KadmiumMcpServer::handle_request(
        &mut state,
        McpJsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(35)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "kadmium_pull_artifact",
                "arguments": {
                    "uri": "gh://zhengkid/Dream-RSI@main",
                    "plan_only": false
                }
            })),
        },
    );
    assert_eq!(mcp_unplanned.result.unwrap()["isError"], json!(true));
}

#[test]
fn issue_35_hf_pull_distinguishes_datasets_and_uses_safetensors_index_when_supplied() {
    let ds =
        HuggingFaceHubConnector::plan_dataset_pull("hf://datasets/open-r1/OpenR1-Math-220k@main")
            .unwrap();
    assert!(ds.plan_only);
    assert!(ds.parquet_split_urls.is_empty());

    let ds_explicit = HuggingFaceHubConnector::plan_dataset_pull_with_splits(
        "hf://datasets/open-r1/OpenR1-Math-220k@main",
        &["data/train-00000-of-00001.parquet"],
    )
    .unwrap();
    assert!(
        ds_explicit
            .parquet_split_urls
            .iter()
            .all(|u| u.ends_with(".parquet") && !u.ends_with(".safetensors"))
    );

    let mut weight_map = BTreeMap::new();
    for l in 0..32 {
        weight_map.insert(
            format!("model.layers.{l}.self_attn.q_proj.weight"),
            "model-00001-of-00002.safetensors".to_string(),
        );
    }
    for l in 32..64 {
        weight_map.insert(
            format!("model.layers.{l}.self_attn.q_proj.weight"),
            "model-00002-of-00002.safetensors".to_string(),
        );
    }
    let manifest = SafetensorsIndexManifest {
        total_size_bytes: 4_194_304,
        weight_map,
    };
    let nodes = sample_personal_cluster();
    let model = ModelSpec::qwen2_5_coder_32b();
    let shard_plan = ShardPlanner::compute_plan(1, &model, WorkloadKind::Inference, &nodes);

    // Without an index manifest, required_safetensors_files must be empty (no invented 16 shards):
    let unindexed = HuggingFaceHubConnector::plan_selective_model_pull(
        "hf://org/tiny-2shard@main",
        &shard_plan,
    )
    .unwrap();
    assert_eq!(unindexed.index_source, "unverified_missing_index_manifest");
    assert!(
        unindexed
            .node_pulls
            .iter()
            .all(|np| np.required_safetensors_files.is_empty())
    );

    let plan = HuggingFaceHubConnector::plan_selective_model_pull_with_index(
        "hf://org/tiny-2shard@main",
        &shard_plan,
        Some(&manifest),
    )
    .unwrap();
    assert_eq!(plan.index_source, "parsed_safetensors_index");
    assert!(plan.plan_only);
    assert_eq!(plan.files_downloaded, 0);
    assert_eq!(plan.ktb_banks_compiled, 0);
    assert!(
        plan.node_pulls
            .iter()
            .flat_map(|np| &np.required_safetensors_files)
            .all(|f| f == "model-00001-of-00002.safetensors"
                || f == "model-00002-of-00002.safetensors")
    );

    // Verify 3-shard and 17-shard manifests map to their exact shard filenames:
    for shard_count in [3usize, 17usize] {
        let mut custom_map = BTreeMap::new();
        for l in 0..64usize {
            let shard_idx = (l * shard_count / 64) + 1;
            custom_map.insert(
                format!("model.layers.{l}.self_attn.q_proj.weight"),
                format!("model-{shard_idx:05}-of-{shard_count:05}.safetensors"),
            );
        }
        let custom_manifest = SafetensorsIndexManifest {
            total_size_bytes: 8_388_608,
            weight_map: custom_map,
        };
        let custom_plan = HuggingFaceHubConnector::plan_selective_model_pull_with_index(
            &format!("hf://org/custom-{shard_count}shard@main"),
            &shard_plan,
            Some(&custom_manifest),
        )
        .unwrap();
        let all_files: std::collections::BTreeSet<String> = custom_plan
            .node_pulls
            .iter()
            .flat_map(|np| np.required_safetensors_files.iter().cloned())
            .collect();
        assert_eq!(all_files.len(), shard_count);
        assert!(
            all_files
                .iter()
                .all(|f| f.ends_with(&format!("-of-{shard_count:05}.safetensors")))
        );
    }
}

#[test]
fn issue_36_usb4_discovery_separates_reference_catalog_and_handles_small_topologies() {
    let nodes = sample_personal_cluster();
    let report = UsbCDirectInterconnectScanner::scan_host_usbc_links_with_options(&nodes, false);
    assert!(!report.zero_config_link_local_ready);
    assert!(report.live_interfaces.is_empty());
    assert_eq!(report.reference_tier_catalog.len(), 3);
    assert!(report.delta_bridge_plan.lanes.is_empty());
    assert_eq!(report.delta_bridge_plan.bonded_goodput_gbps, 0.0);
    assert!(!report.handheld_30gbps_dma_benchmark.live_dma_measured);
    assert_eq!(
        report.handheld_30gbps_dma_benchmark.measurement_kind,
        "modelled_arithmetic_estimate"
    );

    let zero = UsbCDirectInterconnectScanner::build_daisy_chain_topology_plan(&[]);
    assert_eq!(zero.total_usb_c_nodes, 0);
    assert!(zero.hops.is_empty());

    let one = UsbCDirectInterconnectScanner::build_daisy_chain_topology_plan(&nodes[..1]);
    assert_eq!(one.total_usb_c_nodes, 1);
    assert!(one.hops.is_empty());
}

#[test]
fn issue_37_kernel_benchmark_distinguishes_unreachable_and_divergent_participants() {
    let nodes = sample_personal_cluster();
    let host_only = HardwareKernelDispatcher::run_cluster_kernel_benchmark(&nodes);
    assert!(!host_only.cross_silicon_bit_identical);
    assert_eq!(host_only.verified_participant_count, 0);
    assert_eq!(host_only.verification_status, "host_reference_only");
    assert!(
        host_only
            .node_profiles
            .iter()
            .all(|p| p.execution_source == "host_reference" && !p.execution_verified)
    );

    let r0 = HardwareKernelDispatcher::measure_local_device_receipt(nodes[0].clone(), "host-a");
    let suite = HardwareKernelDispatcher::run_verified_kernel_benchmark(&[
        r0,
        KernelBenchmarkParticipant::Unreachable(nodes[1].clone()),
        KernelBenchmarkParticipant::VerifiedDevice {
            node: nodes[2].clone(),
            machine_id: "host-c".to_string(),
            repops_digest_fnv1a: host_only.canonical_repops_digest_fnv1a ^ 0x1234,
            elapsed_execution_us: 25.0,
            measured_gflops: 18.0,
        },
    ]);
    assert!(!suite.cross_silicon_bit_identical);
    assert_eq!(suite.verified_participant_count, 2);
    assert!(suite.node_profiles[0].execution_verified);
    assert!(!suite.node_profiles[1].execution_verified);
    assert_ne!(
        suite.node_profiles[2].repops_digest_fnv1a,
        suite.canonical_repops_digest_fnv1a
    );
}

#[test]
fn issue_38_and_39_mcp_server_defaults_to_live_unconnected_and_labels_cloudshield_simulations() {
    let mut state = GatewayState::new_live_unconnected();

    let status_req = McpJsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(10)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "kadmium_cluster_status",
            "arguments": {}
        })),
    };
    let status_resp = KadmiumMcpServer::handle_request(&mut state, status_req);
    let result = status_resp.result.unwrap();
    let structured = &result["structuredContent"];
    assert_eq!(structured["fleet_mode"], json!("live_unconnected"));
    assert_eq!(structured["live_only_mode"], json!(true));
    assert_eq!(structured["node_count"], json!(0));
    assert_eq!(structured["total_vram_gb"], json!(0.0));
    assert_eq!(structured["estimated_throughput"], json!(0.0));

    let infer_req = McpJsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(11)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "kadmium_run_inference",
            "arguments": {
                "prompt": "test empty cluster"
            }
        })),
    };
    let infer_resp = KadmiumMcpServer::handle_request(&mut state, infer_req);
    let infer_result = infer_resp.result.unwrap();
    assert_eq!(infer_result["isError"], json!(true));
    assert_eq!(infer_result["structuredContent"]["stages"], json!([]));
    assert_eq!(
        infer_result["structuredContent"]["effective_tokens_per_sec"],
        json!(0.0)
    );
    assert_eq!(
        infer_result["structuredContent"]["speculative_verification"]["proposed_draft_tokens"],
        json!(0)
    );

    let model = ModelSpec::deepseek_v3_671b_moe();
    let arb = HybridCloudArbitragePlanner::compute_arbitrage_plan(&model);
    assert_eq!(arb.quote_source, "modelled_reference_catalog");
    assert!(!arb.live_provider_quotes);
    assert!(arb.simulation_only);

    let failovers = HybridCloudArbitragePlanner::simulate_bidirectional_failover_shield();
    assert!(failovers.iter().all(|f| f.simulation_only));
}
