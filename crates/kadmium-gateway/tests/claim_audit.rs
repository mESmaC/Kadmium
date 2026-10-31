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
    unsafe {
        std::env::set_var("KADMIUM_DISABLE_LIVE_PROBES", "1");
    }
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
    assert!(broken.timed_out_or_trapped);
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
fn issue_34_github_pull_preserves_caller_ref_and_does_not_fabricate_mounted_verification() {
    let uri = "gh://nonexistent-owner-404/nonexistent-repo-404@deadbeef";
    let eval = GitHubConnector::evaluate_repo_on_slice(uri).unwrap();
    assert_eq!(eval.pinned_commit_sha, "deadbeef");
    assert_ne!(eval.pinned_commit_sha, "506a788c91d4f02e");
    assert!(!eval.mounted);
    assert_eq!(eval.execution_mode, "plan_only_unmounted");
    assert_eq!(eval.verified_node_count, 0);
    assert!(!eval.verde_verification_passed);
    assert_eq!(eval.measured_tokens_per_sec, 0.0);

    // Staging without a valid local path must fail:
    let err = GitHubConnector::stage_local_repo(
        uri,
        Some(std::path::Path::new(
            "target/definitely-missing-checkout-404",
        )),
    )
    .unwrap_err();
    assert!(err.contains("does not exist"));
}

#[test]
fn issue_35_hf_pull_distinguishes_datasets_and_uses_safetensors_index_when_supplied() {
    let ds =
        HuggingFaceHubConnector::plan_dataset_pull("hf://datasets/open-r1/OpenR1-Math-220k@main")
            .unwrap();
    assert!(ds.plan_only);
    assert!(
        ds.parquet_split_urls
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
}

#[test]
fn issue_36_usb4_discovery_separates_reference_catalog_and_handles_small_topologies() {
    let nodes = sample_personal_cluster();
    let report = UsbCDirectInterconnectScanner::scan_host_usbc_links_with_options(&nodes, false);
    assert!(!report.zero_config_link_local_ready);
    assert!(report.live_interfaces.is_empty());
    assert_eq!(report.reference_tier_catalog.len(), 3);
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
    let suite = HardwareKernelDispatcher::run_verified_kernel_benchmark(&[
        KernelBenchmarkParticipant::Reachable(nodes[0].clone()),
        KernelBenchmarkParticipant::Unreachable(nodes[1].clone()),
        KernelBenchmarkParticipant::Divergent {
            node: nodes[2].clone(),
            perturbation: 0.25,
        },
    ]);
    assert!(!suite.cross_silicon_bit_identical);
    assert_eq!(suite.verified_participant_count, 2);
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

    let model = ModelSpec::deepseek_v3_671b_moe();
    let arb = HybridCloudArbitragePlanner::compute_arbitrage_plan(&model);
    assert_eq!(arb.quote_source, "modelled_reference_catalog");
    assert!(!arb.live_provider_quotes);
    assert!(arb.simulation_only);

    let failovers = HybridCloudArbitragePlanner::simulate_bidirectional_failover_shield();
    assert!(failovers.iter().all(|f| f.simulation_only));
}
