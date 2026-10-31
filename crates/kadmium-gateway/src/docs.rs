//! Official Kadmium Documentation Catalog (`kadmium-gateway::docs`).

use serde::{Deserialize, Serialize};

pub(crate) const DOC_OVERVIEW_MD: &str = include_str!("../../../docs/README.md");
pub(crate) const DOC_GETTING_STARTED_MD: &str = include_str!("../../../docs/getting-started.md");
pub(crate) const DOC_CLI_REFERENCE_MD: &str = include_str!("../../../docs/cli-reference.md");
pub(crate) const DOC_API_REFERENCE_MD: &str = include_str!("../../../docs/api-reference.md");
pub(crate) const DOC_HARDWARE_SHARDING_MD: &str =
    include_str!("../../../docs/guides/hardware-and-sharding.md");
pub(crate) const DOC_TRAINING_DREAM_RSI_MD: &str =
    include_str!("../../../docs/guides/training-and-dream-rsi.md");
pub(crate) const DOC_SUBCLUSTERS_CLOUD_MD: &str =
    include_str!("../../../docs/guides/subclusters-and-cloud-shield.md");
pub(crate) const DOC_INTEGRATIONS_MD: &str =
    include_str!("../../../docs/guides/integrations-hf-github.md");
pub(crate) const DOC_MCP_MD: &str = include_str!("../../../docs/guides/mcp-server.md");
pub(crate) const DOC_CONDENSER_MD: &str =
    include_str!("../../../docs/guides/condenser-workspace.md");
pub(crate) const DOC_LAUNCH_GUIDE_MD: &str = include_str!("../../../docs/guides/cli.md");
pub(crate) const DOC_RELEASE_RUNBOOK_MD: &str = include_str!("../../../docs/guides/release.md");
pub(crate) const DOC_BENCHMARKS_MD: &str = include_str!("../../../docs/BENCHMARKS_2026-10.md");
pub(crate) const DOC_CONTRIBUTING_MD: &str = include_str!("../../../CONTRIBUTING.md");
pub(crate) const DOC_SPECSHEET_ROADMAP_MD: &str =
    include_str!("../../../docs/SPECSHEET_AND_ROADMAP.md");
pub(crate) const DOC_ARCHITECTURE_MD: &str = include_str!("../../../docs/ARCHITECTURE.md");
pub(crate) const DOC_MARKET_RESEARCH_MD: &str =
    include_str!("../../../docs/MARKET_RESEARCH_AND_LAUNCH_PLAN.md");

/// Structured entry for a public documentation page served from `docs/` via `/api/v1/docs` and `/docs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationPage {
    pub slug: String,
    pub title: String,
    pub category: String,
    pub file_path: String,
    pub summary: String,
    pub markdown: String,
}

/// Returns the complete catalog of official Kadmium documentation pages compiled from `docs/`.
pub fn build_documentation_catalog() -> Vec<DocumentationPage> {
    vec![
        DocumentationPage {
            slug: "overview".into(),
            title: "Kadmium Documentation Overview".into(),
            category: "1. Getting Started & Core Reference".into(),
            file_path: "docs/README.md".into(),
            summary: "Sitemap, current reference-execution boundaries, and planned managed services.".into(),
            markdown: DOC_OVERVIEW_MD.into(),
        },
        DocumentationPage {
            slug: "getting-started".into(),
            title: "Quickstart & Installation".into(),
            category: "1. Getting Started & Core Reference".into(),
            file_path: "docs/getting-started.md".into(),
            summary: "Install kadmium, probe local hardware, launch the coordinator, and connect reachable workers.".into(),
            markdown: DOC_GETTING_STARTED_MD.into(),
        },
        DocumentationPage {
            slug: "launch-guide".into(),
            title: "Launch Guide".into(),
            category: "1. Getting Started & Core Reference".into(),
            file_path: "docs/guides/cli.md".into(),
            summary: "Launch planning, configuration preservation, listeners, and demo/live behavior.".into(),
            markdown: DOC_LAUNCH_GUIDE_MD.into(),
        },
        DocumentationPage {
            slug: "cli-reference".into(),
            title: "CLI Command Reference".into(),
            category: "1. Getting Started & Core Reference".into(),
            file_path: "docs/cli-reference.md".into(),
            summary: "Every current CLI subcommand, flag, default, and execution qualification.".into(),
            markdown: DOC_CLI_REFERENCE_MD.into(),
        },
        DocumentationPage {
            slug: "api-reference".into(),
            title: "HTTP & AI Gateway API Reference".into(),
            category: "1. Getting Started & Core Reference".into(),
            file_path: "docs/api-reference.md".into(),
            summary: "OpenAI /v1/chat/completions, Anthropic /v1/messages, and Control Plane + Companion REST APIs.".into(),
            markdown: DOC_API_REFERENCE_MD.into(),
        },
        DocumentationPage {
            slug: "hardware-and-sharding".into(),
            title: "Hardware Setup, Sharding & Polite Governance".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/hardware-and-sharding.md".into(),
            summary: "Hardware profiles, constrained local .ktb import, transfer references, and trust/polite policies.".into(),
            markdown: DOC_HARDWARE_SHARDING_MD.into(),
        },
        DocumentationPage {
            slug: "condenser-workspace".into(),
            title: "Condenser Inference Workspace".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/condenser-workspace.md".into(),
            summary: "Provider tasks, compute placement previews, task forks, drafts and cancellable bounded runs.".into(),
            markdown: DOC_CONDENSER_MD.into(),
        },
        DocumentationPage {
            slug: "training-and-dream-rsi".into(),
            title: "Distributed Training, RL & Dream-RSI".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/training-and-dream-rsi.md".into(),
            summary: "Stateless previews, small stateful DiLoCo sessions, synthetic GRPO, and in-memory Dream-RSI replay.".into(),
            markdown: DOC_TRAINING_DREAM_RSI_MD.into(),
        },
        DocumentationPage {
            slug: "subclusters-and-cloud-shield".into(),
            title: "Sub-Cluster Slicing, Companion App & CloudShield".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/subclusters-and-cloud-shield.md".into(),
            summary: "Pool partitioning, local Companion mode state, and fixed cloud/failover scenarios.".into(),
            markdown: DOC_SUBCLUSTERS_CLOUD_MD.into(),
        },
        DocumentationPage {
            slug: "integrations-hf-github".into(),
            title: "Hugging Face (hf://), GitHub (gh://) & Ecosystem Integrations".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/integrations-hf-github.md".into(),
            summary: "HF pull planning, simulated GitHub evaluation, and unimplemented remote connectors.".into(),
            markdown: DOC_INTEGRATIONS_MD.into(),
        },
        DocumentationPage {
            slug: "mcp-server".into(),
            title: "Model Context Protocol (kadmium-mcp) Server".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/mcp-server.md".into(),
            summary: "Connect Claude Code, Cursor, Antigravity, Windsurf, or any MCP client via kadmium mcp (stdio) or POST /mcp.".into(),
            markdown: DOC_MCP_MD.into(),
        },
        DocumentationPage {
            slug: "specsheet-and-roadmap".into(),
            title: "Technical Specsheet & Product Roadmap".into(),
            category: "3. Deep Specifications".into(),
            file_path: "docs/SPECSHEET_AND_ROADMAP.md".into(),
            summary: "Current formats, algorithms, execution support, and planned production work.".into(),
            markdown: DOC_SPECSHEET_ROADMAP_MD.into(),
        },
        DocumentationPage {
            slug: "architecture".into(),
            title: "Deep Technical Architecture Spec".into(),
            category: "3. Deep Specifications".into(),
            file_path: "docs/ARCHITECTURE.md".into(),
            summary: "Current mechanisms and explicitly labelled target designs across the 12-crate workspace.".into(),
            markdown: DOC_ARCHITECTURE_MD.into(),
        },
        DocumentationPage {
            slug: "benchmarks".into(),
            title: "Measured Benchmark Ledger".into(),
            category: "3. Deep Specifications".into(),
            file_path: "docs/BENCHMARKS_2026-10.md".into(),
            summary: "Dated measurements, derived arithmetic, provenance, and harness limitations.".into(),
            markdown: DOC_BENCHMARKS_MD.into(),
        },
        DocumentationPage {
            slug: "release-runbook".into(),
            title: "Maintainer Release Runbook".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/release.md".into(),
            summary: "Version/tag checks, release artifacts, CI requirements, and installer verification.".into(),
            markdown: DOC_RELEASE_RUNBOOK_MD.into(),
        },
        DocumentationPage {
            slug: "contributing".into(),
            title: "Contributing to Kadmium".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "CONTRIBUTING.md".into(),
            summary: "Development setup, required checks, and documentation conventions.".into(),
            markdown: DOC_CONTRIBUTING_MD.into(),
        },
        DocumentationPage {
            slug: "market-research-and-launch-plan".into(),
            title: "Marketing and Launch Plan".into(),
            category: "3. Deep Specifications".into(),
            file_path: "docs/MARKET_RESEARCH_AND_LAUNCH_PLAN.md".into(),
            summary: "Proposed positioning, outreach experiments, budgets, measurement, and launch controls.".into(),
            markdown: DOC_MARKET_RESEARCH_MD.into(),
        },
    ]
}
