//! Runs alone in its own binary because it sets process environment.
use kadmium_condenser::agent::{AgentConfig, AgentTools};
use kadmium_condenser::plugin::PluginManager;
use serde_json::json;

#[tokio::test]
async fn a_malformed_subagent_file_only_affects_subagent_tools() {
    let root = std::env::temp_dir().join(format!("condenser-subagents-{}", std::process::id()));
    std::fs::create_dir_all(root.join("workspace")).unwrap();
    std::fs::write(root.join("subagents.json"), "{ not json").unwrap();
    // SAFETY: this binary has a single test, so no other thread reads the environment.
    unsafe { std::env::set_var("CONDENSER_SUBAGENTS_FILE", root.join("subagents.json")) };
    let tools = AgentTools::in_workspace(
        PluginManager::new(vec![]),
        root.join("workspace"),
        AgentConfig::default(),
    )
    .expect("runs that never delegate still start");
    assert!(
        tools
            .invoke_named("condenser__subagents_list", json!({}))
            .await
            .is_err()
    );
    let _ = std::fs::remove_dir_all(&root);
}
