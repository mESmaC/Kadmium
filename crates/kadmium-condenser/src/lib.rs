//! Condenser is Kadmium Power Plant's modular inference harness.
//!
//! The crate deliberately separates provider adapters, the durable conversation
//! vocabulary, process-isolated plugins, and the bounded agent/tool harness.
//! Both the CLI and the native UI use this library rather than maintaining
//! separate inference implementations.

pub mod error;
pub mod harness;
pub mod plugin;
pub mod provider;
pub mod server;
pub mod types;
pub mod ui;

pub use error::{CondenserError, Result};
pub use harness::{Harness, HarnessOptions, HarnessReport, HarnessStep, ToolExecution};
pub use plugin::{PluginInventory, PluginManager, RegisteredTool};
pub use provider::{ProviderDescriptor, ProviderFactory};
pub use types::*;

/// Version spoken by every out-of-process Condenser plugin.
pub const PLUGIN_PROTOCOL_VERSION: u32 = 1;
/// Default loopback port for `condenser serve`.
pub const DEFAULT_HTTP_PORT: u16 = 7856;
