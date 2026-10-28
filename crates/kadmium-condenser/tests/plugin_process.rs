use kadmium_condenser::plugin::{PLUGIN_MANIFEST_FILE, PluginManager, PluginManifest};
use kadmium_condenser::{PLUGIN_PROTOCOL_VERSION, ToolDefinition};
use serde_json::json;
use std::path::PathBuf;
use std::time::Duration;

fn plugin_manager(test_name: &str, args: &[&str]) -> (PathBuf, PluginManager) {
    let root = std::env::current_dir()
        .unwrap()
        .join("target")
        .join(format!(
            "condenser-plugin-process-test-{test_name}-{}",
            std::process::id()
        ));
    let plugin_root = root.join("echo");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&plugin_root).unwrap();
    let manifest = PluginManifest {
        protocol_version: PLUGIN_PROTOCOL_VERSION,
        id: "echo".into(),
        name: "Echo".into(),
        version: "0.1.0".into(),
        description: "Integration test plugin".into(),
        command: env!("CARGO_BIN_EXE_condenser-echo-plugin").into(),
        args: args.iter().map(|arg| (*arg).to_string()).collect(),
        tools: vec![ToolDefinition {
            name: "echo".into(),
            description: "Return text".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }),
        }],
    };
    std::fs::write(
        plugin_root.join(PLUGIN_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let manager = PluginManager::new(vec![root.clone()]).with_timeout(Duration::from_secs(5));
    (root, manager)
}

#[tokio::test]
async fn executes_a_real_out_of_process_plugin() {
    let (root, manager) = plugin_manager("echo", &[]);
    let inventory = manager.discover().unwrap();
    assert_eq!(inventory.plugins.len(), 1);
    assert!(inventory.diagnostics.is_empty());
    let result = manager
        .invoke("echo", "echo", json!({"text": "hello from Condenser"}))
        .await
        .unwrap();
    assert_eq!(result.content, "hello from Condenser");
    assert!(!result.is_error);

    let (tools, diagnostics) = manager.registered_tools().unwrap();
    assert!(diagnostics.is_empty());
    assert_eq!(tools[0].qualified_name, "echo__echo");
    let result = manager
        .invoke_registered(&tools[0], json!({"text": "registered"}))
        .await
        .unwrap();
    assert_eq!(result.content, "registered");
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn valid_result_survives_a_plugin_that_ignores_shutdown() {
    let (root, manager) = plugin_manager("hang-after-response", &["hang-after-response"]);
    let result = manager
        .invoke("echo", "echo", json!({"text": "ignored"}))
        .await
        .unwrap();
    assert_eq!(result.content, "completed");
    assert!(!result.is_error);
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn oversized_unterminated_response_is_rejected_before_eof() {
    let (root, manager) = plugin_manager("oversized-line", &["oversized-line"]);
    let error = manager
        .invoke("echo", "echo", json!({"text": "ignored"}))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("plugin response exceeds"),
        "unexpected error: {error}"
    );
    let _ = std::fs::remove_dir_all(root);
}
