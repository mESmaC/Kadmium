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
pub(crate) const DOC_SPECSHEET_ROADMAP_MD: &str =
    include_str!("../../../docs/SPECSHEET_AND_ROADMAP.md");
pub(crate) const DOC_ARCHITECTURE_MD: &str = include_str!("../../../docs/ARCHITECTURE.md");

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
            summary: "Sitemap, architecture diagram, and Open-Source vs. kadmium.ai Cloud overview.".into(),
            markdown: DOC_OVERVIEW_MD.into(),
        },
        DocumentationPage {
            slug: "getting-started".into(),
            title: "Quickstart & Installation".into(),
            category: "1. Getting Started & Core Reference".into(),
            file_path: "docs/getting-started.md".into(),
            summary: "Install kadmium, probe local hardware, launch the coordinator, and connect LAN/WAN peers.".into(),
            markdown: DOC_GETTING_STARTED_MD.into(),
        },
        DocumentationPage {
            slug: "cli-reference".into(),
            title: "CLI Command Reference".into(),
            category: "1. Getting Started & Core Reference".into(),
            file_path: "docs/cli-reference.md".into(),
            summary: "Complete reference for probe, serve, plan, simulate, slice, dataflow, dream, and cloud-shield.".into(),
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
            summary: "5-tier memory hierarchy, 64 KiB .ktb transcoding, CERN FTS3 P2P staging, <200ms Gaming Auto-Yield, and 3-Tier Trust.".into(),
            markdown: DOC_HARDWARE_SHARDING_MD.into(),
        },
        DocumentationPage {
            slug: "training-and-dream-rsi".into(),
            title: "Distributed Training, RL & Dream-RSI".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/training-and-dream-rsi.md".into(),
            summary: "Sharded-Island DiLoCo + DeMo DCT compression, Disaggregated RL, Dream-RSI offline replay, and TPU/CIM dataflows.".into(),
            markdown: DOC_TRAINING_DREAM_RSI_MD.into(),
        },
        DocumentationPage {
            slug: "subclusters-and-cloud-shield".into(),
            title: "Sub-Cluster Slicing, Companion App & CloudShield".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/subclusters-and-cloud-shield.md".into(),
            summary: "Partitioning 100-node pools into isolated sub-clusters, Desktop Companion OSS/Cloud switching, and <400ms Bidirectional Failover.".into(),
            markdown: DOC_SUBCLUSTERS_CLOUD_MD.into(),
        },
        DocumentationPage {
            slug: "integrations-hf-github".into(),
            title: "Hugging Face (hf://), GitHub (gh://) & Ecosystem Integrations".into(),
            category: "2. Operator & Developer Guides".into(),
            file_path: "docs/guides/integrations-hf-github.md".into(),
            summary: "Layer-aware selective Safetensors/GGUF pulling, dataset streaming, commit-pinned GitHub RL/Dream-RSI worlds, and S3/OCI connectors.".into(),
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
            summary: "Complete Cd-48 engine specifications, 5-tier memory & hardware matrix, algorithmic guarantees, and phased v0.1 -> v1.0 GA roadmap.".into(),
            markdown: DOC_SPECSHEET_ROADMAP_MD.into(),
        },
        DocumentationPage {
            slug: "architecture".into(),
            title: "Deep Technical Architecture Spec".into(),
            category: "3. Deep Specifications".into(),
            file_path: "docs/ARCHITECTURE.md".into(),
            summary: "Full mathematical and systems specification across all 8 Rust workspace crates.".into(),
            markdown: DOC_ARCHITECTURE_MD.into(),
        },
    ]
}
