//! Additive setup path around the existing coordinator; existing commands keep their behavior.

use clap::Args;
use kadmium_core::{HardwareNode, KadmiumWorkspaceConfig, LiveHardwareProfiler, ModelSpec};
use kadmium_gateway::{CoordinatorOptions, run_coordinator};
use kadmium_network::DEFAULT_WORKER_TCP_PORT;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Args)]
pub struct LaunchArgs {
    /// Show local hardware and startup choices without writing files or starting services
    #[arg(long)]
    plan: bool,
    /// HTTP port on all IPv4 interfaces; configuration does not override this port
    #[arg(short, long, default_value_t = 7848)]
    port: u16,
    /// Preserve this workspace file, or create it if absent
    #[arg(short, long, default_value = "kadmium.toml")]
    config: PathBuf,
    /// Include the simulated example fleet alongside live hardware
    #[arg(long)]
    demo: bool,
    /// Preferred worker TCP port on all IPv4 interfaces (0 requests an ephemeral port)
    #[arg(long, default_value_t = DEFAULT_WORKER_TCP_PORT)]
    worker_port: u16,
    /// Spawn the existing local TCP worker test fleet
    #[arg(long)]
    spawn_fleet: bool,
}

#[derive(Debug)]
struct PreparedLaunch {
    options: CoordinatorOptions,
    path: PathBuf,
    generated_config: Option<String>,
    node: HardwareNode,
    recommendation: ModelSpec,
}

fn prepare(
    args: &LaunchArgs,
    probe: impl FnOnce() -> HardwareNode,
) -> Result<PreparedLaunch, String> {
    if args.port == 0 {
        return Err("HTTP --port must be in 1..=65535".into());
    }
    if args.port == args.worker_port {
        return Err("HTTP --port and --worker-port must differ".into());
    }
    let config_path = args
        .config
        .to_str()
        .ok_or("configuration path must contain valid Unicode")?
        .to_owned();
    if let Ok(metadata) = std::fs::metadata(&args.config)
        && !metadata.is_file()
    {
        return Err(format!(
            "configuration path is not a regular file: {}",
            args.config.display()
        ));
    }
    // Reading checks access only. Parsing remains the existing coordinator's responsibility:
    // its tolerant parser does not validate general TOML or apply bind/model/host overrides.
    let exists = match std::fs::read_to_string(&args.config) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = args.config.parent().filter(|p| !p.as_os_str().is_empty());
            let parent = parent.unwrap_or_else(|| Path::new("."));
            if !parent.is_dir() {
                return Err(format!(
                    "configuration parent directory does not exist: {}",
                    parent.display()
                ));
            }
            false
        }
        Err(error) => {
            return Err(format!(
                "could not read configuration {}: {error}",
                args.config.display()
            ));
        }
    };
    let node = probe();
    let recommendation = ModelSpec::recommended_for_cluster(std::slice::from_ref(&node));
    let generated_config = if exists {
        None
    } else {
        let mut config = KadmiumWorkspaceConfig::from_local_probe(node.clone());
        config.coordinator_bind = format!("0.0.0.0:{}", args.port);
        config.default_model = recommendation.id.clone();
        Some(render_new_config(&config)?)
    };
    Ok(PreparedLaunch {
        options: CoordinatorOptions {
            port: args.port,
            config_path: Some(config_path),
            live_only: !args.demo,
            worker_port: args.worker_port,
            spawn_fleet: args.spawn_fleet,
        },
        path: args.config.clone(),
        generated_config,
        node,
        recommendation,
    })
}

fn render_new_config(config: &KadmiumWorkspaceConfig) -> Result<String, String> {
    // The shared renderer has no TOML escaping. Keep it unchanged, but do not emit unsafe
    // host-derived strings through this new command.
    for (name, value) in [
        ("cluster.name", &config.cluster_name),
        ("cluster.coordinator_bind", &config.coordinator_bind),
        ("cluster.ktb_cache_dir", &config.ktb_cache_dir),
        ("cluster.default_model", &config.default_model),
        ("host.node_id", &config.local_node.id.0),
        ("host.hostname", &config.local_node.hostname),
        (
            "integrations.github_sandbox_slice",
            &config.github_sandbox_slice,
        ),
    ] {
        if value
            .chars()
            .any(|c| c == '"' || c == '\\' || c.is_control())
        {
            return Err(format!(
                "cannot generate configuration: {name} contains a quote, backslash or control character; use an existing configuration with --config"
            ));
        }
    }
    let mut text = String::from(
        "# Planning and hardware values below are observations, not coordinator overrides.\n\
         # The coordinator currently applies cloud-shield settings; choose its HTTP port with --port.\n\
         # The planning recommendation does not load weights or freeze automatic re-recommendation.\n",
    );
    text.push_str(&config.render_kadmium_toml().replacen(
        "# Generated by `kadmium init`",
        "# Generated by `kadmium launch`",
        1,
    ));
    Ok(text)
}

fn write_new_config(path: &Path, text: &str) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("could not create configuration {}: {error}", path.display()))?;
    file.write_all(text.as_bytes())
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("could not write configuration {}: {error}", path.display()))
}

impl PreparedLaunch {
    fn print_plan(&self) {
        let offer = self.node.offered_now();
        println!("Kadmium launch plan");
        println!(
            "  Machine: {} ({:?}, {:?})",
            self.node.hostname, self.node.device_kind, self.node.backend
        );
        println!(
            "  Offered memory: {:.1} GiB fast, {:.1} GiB RAM; {} worker slots",
            offer.fast_memory_gb, offer.system_ram_gb, offer.slots
        );
        println!(
            "  Local planning recommendation: {}",
            self.recommendation.id
        );
        println!(
            "  HTTP listener: 0.0.0.0:{} (all IPv4 interfaces)",
            self.options.port
        );
        println!(
            "  Worker listener: 0.0.0.0:{} (preferred TCP port; falls back to an available port)",
            self.options.worker_port
        );
        println!(
            "  Configuration: {} ({})",
            self.path.display(),
            if self.generated_config.is_some() {
                "will create only when launched"
            } else {
                "existing file preserved; read access checked"
            }
        );
        println!(
            "  Fleet: {}; extra local test workers: {}",
            if self.options.live_only {
                "live hardware"
            } else {
                "live + simulated hardware"
            },
            self.options.spawn_fleet
        );
        println!(
            "  Startup uses the existing LAN announcements and optional hardware bridge discovery."
        );
        println!(
            "  Use a trusted network: HTTP and worker listeners are not limited to this computer."
        );
        println!(
            "  The existing tolerant config parser applies cloud-shield settings; stored bind, model and host values do not override coordinator startup."
        );
        println!(
            "  Planning recommendations may change as workers join and do not prove model fit or load model weights."
        );
        println!(
            "  This command does not download models or add an inference backend; existing diagnostic/reference API behavior is unchanged."
        );
    }
}

pub async fn run(args: LaunchArgs) -> Result<(), String> {
    let prepared = prepare(&args, LiveHardwareProfiler::probe_local_host)?;
    prepared.print_plan();
    if args.plan {
        return Ok(());
    }
    if let Some(text) = &prepared.generated_config {
        write_new_config(&prepared.path, text)?;
        println!("  Created {}", prepared.path.display());
    }
    run_coordinator(prepared.options).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> LaunchArgs {
        static NEXT_PATH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        LaunchArgs {
            plan: true,
            port: 7848,
            config: std::env::temp_dir().join(format!(
                "kadmium-launch-{}-{}-{}.toml",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT_PATH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            )),
            demo: false,
            worker_port: DEFAULT_WORKER_TCP_PORT,
            spawn_fleet: false,
        }
    }

    fn node() -> HardwareNode {
        kadmium_core::sample_personal_cluster().remove(0)
    }

    #[test]
    fn planning_is_read_only_and_records_the_actual_port_and_live_recommendation() {
        let mut args = args();
        args.port = 9010;
        let prepared = prepare(&args, node).unwrap();
        assert!(!args.config.exists());
        let text = prepared.generated_config.unwrap();
        assert!(text.contains("coordinator_bind = \"0.0.0.0:9010\""));
        assert!(text.contains(&format!(
            "default_model = \"{}\"",
            prepared.recommendation.id
        )));
        assert_eq!(prepared.options.port, 9010);
        assert!(prepared.options.live_only);
    }

    #[test]
    fn existing_configuration_is_preserved_without_new_parser_semantics() {
        let mut args = args();
        let original =
            "# keep this\n[cluster]\ncoordinator_bind = \"127.0.0.1:9999\"\nunknown = unparsed\n";
        std::fs::write(&args.config, original).unwrap();
        args.demo = true;
        args.spawn_fleet = true;
        let prepared = prepare(&args, node).unwrap();
        assert!(prepared.generated_config.is_none());
        assert_eq!(prepared.options.port, 7848);
        assert!(!prepared.options.live_only);
        assert!(prepared.options.spawn_fleet);
        assert_eq!(std::fs::read_to_string(&args.config).unwrap(), original);
        std::fs::remove_file(args.config).unwrap();
    }

    #[test]
    fn invalid_ports_and_unreadable_paths_fail_before_probing() {
        let mut args = args();
        args.port = 0;
        assert!(prepare(&args, || panic!("unexpected probe")).is_err());
        args.port = args.worker_port;
        assert!(prepare(&args, || panic!("unexpected probe")).is_err());
        args.port = 7848;
        args.config = args.config.join("missing-parent.toml");
        assert!(prepare(&args, || panic!("unexpected probe")).is_err());
        args.config = std::env::temp_dir();
        assert!(prepare(&args, || panic!("unexpected probe")).is_err());
    }

    #[test]
    fn creation_is_exclusive_even_if_another_writer_wins_after_planning() {
        let args = args();
        let prepared = prepare(&args, node).unwrap();
        let text = prepared.generated_config.unwrap();
        write_new_config(&args.config, "other writer").unwrap();
        assert!(write_new_config(&args.config, &text).is_err());
        assert_eq!(
            std::fs::read_to_string(&args.config).unwrap(),
            "other writer"
        );
        std::fs::remove_file(&args.config).unwrap();
        write_new_config(&args.config, &text).unwrap();
        assert_eq!(std::fs::read_to_string(&args.config).unwrap(), text);
        std::fs::remove_file(args.config).unwrap();
    }

    #[test]
    fn generated_host_strings_cannot_break_the_existing_renderer() {
        let args = args();
        for hostname in ["host\"name", "host\\name", "host\nname"] {
            let mut local_node = node();
            local_node.hostname = hostname.into();
            assert!(
                prepare(&args, || local_node)
                    .unwrap_err()
                    .contains("host.hostname")
            );
        }
        assert!(!args.config.exists());
    }
}
