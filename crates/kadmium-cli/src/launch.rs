//! Shared, validated startup for `serve` and the one-command `launch` path.

use clap::Args;
use kadmium_core::{HardwareNode, KadmiumWorkspaceConfig, LiveHardwareProfiler};
use kadmium_gateway::coordinator::resolve_coordinator_bind;
use kadmium_gateway::inference_api::InferenceConfig;
use kadmium_gateway::{CoordinatorOptions, run_coordinator};
use kadmium_network::DEFAULT_WORKER_TCP_PORT;
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Debug, Clone, Args)]
pub struct StartupArgs {
    /// HTTP port override; retains the configured/default IP address
    #[arg(short, long, conflicts_with = "bind")]
    pub port: Option<u16>,
    /// HTTP IP:PORT override (launch default 127.0.0.1:7848; serve default 0.0.0.0:7848)
    #[arg(long)]
    pub bind: Option<SocketAddr>,
    /// Workspace file; launch defaults to kadmium.toml and creates it only when absent
    #[arg(short, long)]
    pub config: Option<PathBuf>,
    /// Use only live hardware (the default)
    #[arg(long, conflicts_with = "demo")]
    pub live: bool,
    /// Include the simulated example fleet
    #[arg(long)]
    pub demo: bool,
    /// Spawn a local TCP worker test fleet
    #[arg(long)]
    pub spawn_fleet: bool,
    /// Preferred built-in worker TCP port (0 requests an ephemeral port)
    #[arg(long, default_value_t = DEFAULT_WORKER_TCP_PORT)]
    pub worker_port: u16,
    /// Probe optional USB and lab-specific Apple M4/Raspberry Pi CM4 bridges
    #[arg(long)]
    pub discover_bridges: bool,
    /// OpenAI-compatible inference upstream; defaults to KADMIUM_INFERENCE_URL
    #[arg(long)]
    pub inference_url: Option<String>,
    /// Maximum simultaneous upstream inference requests
    #[arg(long, default_value_t = 4)]
    pub inference_concurrency: usize,
    /// Upstream inference timeout in seconds
    #[arg(long, default_value_t = 120)]
    pub inference_timeout_secs: u64,
}

impl Default for StartupArgs {
    fn default() -> Self {
        Self {
            port: None,
            bind: None,
            config: None,
            live: false,
            demo: false,
            spawn_fleet: false,
            worker_port: DEFAULT_WORKER_TCP_PORT,
            discover_bridges: false,
            inference_url: None,
            inference_concurrency: 4,
            inference_timeout_secs: 120,
        }
    }
}

struct PreparedStartup {
    options: CoordinatorOptions,
    config: KadmiumWorkspaceConfig,
    config_path: Option<PathBuf>,
    create_config: bool,
}

fn env_value(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(format!("{name} must contain valid Unicode")),
    }
}

fn inference_config(args: &StartupArgs) -> Result<InferenceConfig, String> {
    let config = InferenceConfig {
        base_url: match &args.inference_url {
            Some(url) => Some(url.clone()),
            None => env_value("KADMIUM_INFERENCE_URL")?,
        },
        // Credentials are environment-only: never a command argument or saved in kadmium.toml.
        api_key: env_value("KADMIUM_INFERENCE_API_KEY")?,
        max_in_flight: args.inference_concurrency,
        timeout_secs: args.inference_timeout_secs,
    };
    config.validate()?;
    Ok(config)
}

fn prepare_with_probe(
    args: &StartupArgs,
    launch: bool,
    probe: impl FnOnce() -> HardwareNode,
) -> Result<PreparedStartup, String> {
    let inference = inference_config(args)?;
    let fallback = if launch {
        SocketAddr::from(([127, 0, 0, 1], 7848))
    } else {
        SocketAddr::from(([0, 0, 0, 0], 7848))
    };
    // Reject invalid overrides before touching files or invoking hardware tools.
    resolve_coordinator_bind(args.bind, args.port, None, fallback)?;
    let config_path = args
        .config
        .clone()
        .or_else(|| launch.then(|| PathBuf::from("kadmium.toml")));
    let text = if let Some(path) = &config_path {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                KadmiumWorkspaceConfig::validate_kadmium_toml(&text)
                    .map_err(|error| format!("configuration {}: {error}", path.display()))?;
                Some(text)
            }
            Err(error) if launch && error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(format!(
                    "could not read configuration {}: {error}",
                    path.display()
                ));
            }
        }
    } else {
        None
    };
    let configured = text
        .as_deref()
        .map(KadmiumWorkspaceConfig::configured_coordinator_bind)
        .transpose()?
        .flatten();
    let bind = resolve_coordinator_bind(args.bind, args.port, configured, fallback)?;
    let config_path_string = config_path
        .as_ref()
        .map(|path| {
            path.to_str()
                .map(str::to_owned)
                .ok_or_else(|| "configuration path must contain valid Unicode".to_string())
        })
        .transpose()?;
    let node = probe();
    let mut config = match &text {
        Some(text) => KadmiumWorkspaceConfig::parse_kadmium_toml_for_node(text, node)?,
        None => {
            let mut cfg = KadmiumWorkspaceConfig::from_local_probe(node);
            cfg.coordinator_bind = fallback.to_string();
            cfg
        }
    };
    config.coordinator_bind = bind.to_string();
    config.planning_model()?;
    Ok(PreparedStartup {
        options: CoordinatorOptions {
            port: bind.port(),
            bind: Some(bind),
            config_path: config_path_string,
            live_only: args.live || !args.demo,
            worker_port: args.worker_port,
            spawn_fleet: args.spawn_fleet,
            discover_bridges: args.discover_bridges,
            inference,
        },
        config,
        config_path,
        create_config: launch && text.is_none(),
    })
}

impl PreparedStartup {
    fn print_plan(&self) {
        let node = &self.config.local_node;
        let offer = node.offered_now();
        println!("Kadmium launch plan");
        println!(
            "  Machine: {} ({:?}, {:?})",
            node.hostname, node.device_kind, node.backend
        );
        println!(
            "  Offered memory: {:.1} GiB fast, {:.1} GiB RAM; {} worker slots",
            offer.fast_memory_gb, offer.system_ram_gb, offer.slots
        );
        println!("  Planning model preset: {}", self.config.default_model);
        println!("  HTTP listener: {}", self.config.coordinator_bind);
        if let Some(bind) = self.options.bind {
            println!(
                "  Built-in worker: {} (preferred TCP port; falls back to an available port)",
                SocketAddr::new(bind.ip(), self.options.worker_port)
            );
            println!(
                "  LAN UDP discovery: {}",
                if bind.ip().is_loopback() {
                    "disabled for loopback binding"
                } else {
                    "enabled"
                }
            );
        }
        println!(
            "  Configuration: {} ({})",
            self.config_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "defaults".into()),
            if self.create_config {
                "will create when launched"
            } else {
                "existing settings"
            }
        );
        if let Some(url) = &self.options.inference.base_url {
            println!("  Real inference: external OpenAI-compatible backend at {url}");
            println!(
                "  Upstream limits: {} simultaneous requests, {} second timeout",
                self.options.inference.max_in_flight, self.options.inference.timeout_secs
            );
        } else {
            println!(
                "  Real inference: unavailable until --inference-url or KADMIUM_INFERENCE_URL is set."
            );
        }
        println!(
            "  Optional hardware bridges: {}",
            if self.options.discover_bridges {
                "enabled"
            } else {
                "disabled"
            }
        );
        println!(
            "  The grid uses reference computation; this plan does not load weights, test the upstream, or prove model fit."
        );
        println!(
            "  Inference uses the upstream's loaded models; the planning preset does not select its model."
        );
    }
}

pub async fn run(args: StartupArgs, launch: bool, plan_only: bool) -> Result<(), String> {
    let prepared = prepare_with_probe(&args, launch, LiveHardwareProfiler::probe_local_host)?;
    if launch {
        prepared.print_plan();
    }
    if plan_only {
        return Ok(());
    }
    if prepared.create_config {
        let path = prepared
            .config_path
            .as_ref()
            .ok_or("missing launch configuration path")?;
        prepared.config.write_to_path(path, false)?;
        println!("  Created {}", path.display());
    }
    run_coordinator(prepared.options).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_path() -> PathBuf {
        static NEXT_PATH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "kadmium-launch-{}-{}-{}.toml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_PATH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    #[test]
    fn plan_creates_nothing_and_new_launch_defaults_to_loopback() {
        let path = unique_path();
        let args = StartupArgs {
            config: Some(path.clone()),
            ..StartupArgs::default()
        };
        let prepared = prepare_with_probe(&args, true, || {
            kadmium_core::sample_personal_cluster().remove(0)
        })
        .unwrap();
        assert!(prepared.create_config);
        assert!(!path.exists());
        assert_eq!(
            prepared.options.bind.unwrap(),
            "127.0.0.1:7848".parse().unwrap()
        );
        assert_eq!(prepared.config.default_model, "qwen2.5-coder-32b-instruct");
    }

    #[test]
    fn invalid_or_missing_explicit_serve_config_fails_before_hardware_probe() {
        let path = unique_path();
        let args = StartupArgs {
            config: Some(path.clone()),
            ..StartupArgs::default()
        };
        assert!(prepare_with_probe(&args, false, || panic!("unexpected probe")).is_err());
        std::fs::write(&path, "[cluster]\ncoordinator_bind='127.0.0.1:0'").unwrap();
        assert!(prepare_with_probe(&args, true, || panic!("unexpected probe")).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn partial_existing_configuration_keeps_launch_loopback_default() {
        let path = unique_path();
        std::fs::write(&path, "[cluster]\nname='local'").unwrap();
        let args = StartupArgs {
            config: Some(path.clone()),
            ..StartupArgs::default()
        };
        let prepared = prepare_with_probe(&args, true, || {
            kadmium_core::sample_personal_cluster().remove(0)
        })
        .unwrap();
        assert_eq!(prepared.config.coordinator_bind, "127.0.0.1:7848");
        assert!(!prepared.create_config);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_overrides_fail_before_hardware_probe() {
        for args in [
            StartupArgs {
                port: Some(0),
                ..StartupArgs::default()
            },
            StartupArgs {
                bind: Some("[::]:7848".parse().unwrap()),
                ..StartupArgs::default()
            },
            StartupArgs {
                inference_concurrency: 0,
                ..StartupArgs::default()
            },
            StartupArgs {
                inference_timeout_secs: 0,
                ..StartupArgs::default()
            },
            StartupArgs {
                inference_url: Some("invalid".into()),
                ..StartupArgs::default()
            },
        ] {
            assert!(prepare_with_probe(&args, true, || panic!("unexpected probe")).is_err());
        }
    }

    #[test]
    fn existing_config_and_explicit_port_resolve_without_rewriting_file() {
        let path = unique_path();
        let text =
            "[cluster]\ncoordinator_bind='127.0.0.2:9000'\ndefault_model='llama-3.2-3b-instruct'";
        std::fs::write(&path, text).unwrap();
        let mut args = StartupArgs {
            config: Some(path.clone()),
            ..StartupArgs::default()
        };
        let probe = || kadmium_core::sample_personal_cluster().remove(0);
        let configured = prepare_with_probe(&args, true, probe).unwrap();
        assert!(!configured.create_config);
        assert_eq!(configured.config.coordinator_bind, "127.0.0.2:9000");
        assert_eq!(configured.config.default_model, "llama-3.2-3b-instruct");
        args.port = Some(9100);
        let explicit = prepare_with_probe(&args, true, probe).unwrap();
        assert_eq!(explicit.config.coordinator_bind, "127.0.0.2:9100");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        std::fs::remove_file(path).unwrap();
    }
}
