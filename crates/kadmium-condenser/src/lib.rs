//! Condenser is Kadmium Power Plant's modular inference harness.
//!
//! The crate deliberately separates provider adapters, the durable conversation
//! vocabulary, process-isolated plugins, and the bounded agent/tool harness.
//! Both the CLI and the native UI use this library rather than maintaining
//! separate inference implementations.

mod compute;
pub mod dsv41_mcl4;
pub mod error;
pub mod harness;
pub mod local_models;
pub mod plugin;
pub mod provider;
mod runs;
pub mod server;
pub mod types;
pub mod ui;
pub mod workspace;

pub use dsv41_mcl4::{
    CoeDecisionCandidate, CoeDecisionProbeReport, CompactionStats, DeepSeekV41ArchSpec,
    DsmlParseOutcome, Dsv41ArenaCachePlan, Dsv41Mcl4cPacketLayout, Dsv41PlasticUpperTailHeader,
    Dsv41RouteDecision, EquationVerification, Mcl4CoeKnobs, Mcl4CoeMode, MlaContextBudget,
    ReasoningFormat, SafetensorsSliceMeta, SegmentedReasoning, clamped_swiglu,
    clamped_swiglu_slice, compact_messages_for_dsv41_mcl4, compact_messages_with_tool_pruning,
    detect_tool_doom_loop, dsv41_route_top6_noaux_tc, engram_24prime_xor_hashes,
    engram_signed_sqrt_gate, format_uta_prompt, hc_mix_residual_step, mla_context_budget,
    parse_dsml_tool_calls, parse_safetensors_header, probe_decision_candidates,
    segment_reasoning_and_answer, sinkhorn_knopp_4x4, verify_thought_equations,
};
pub use error::{CondenserError, Result};
pub use harness::{
    Harness, HarnessEvent, HarnessOptions, HarnessReport, HarnessStep, MAX_CALLS_PER_STEP,
    MAX_HARNESS_STEPS, RunControl, RunStatus, ToolExecution,
};
pub use plugin::{PluginInventory, PluginManager, RegisteredTool};
pub use provider::{ProviderDescriptor, ProviderFactory};
pub use types::*;

/// Version spoken by every out-of-process Condenser plugin.
pub const PLUGIN_PROTOCOL_VERSION: u32 = 1;
/// Default loopback port for `condenser serve`.
pub const DEFAULT_HTTP_PORT: u16 = 7856;
