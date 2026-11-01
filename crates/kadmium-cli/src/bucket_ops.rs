use clap::{Args, Subcommand};
use kadmium_core::LiveHardwareProfiler;
use kadmium_data::{BucketObjectStore, DecentralizedStoragePoolEngine};
use serde_json::json;
use std::path::PathBuf;

#[derive(Args)]
pub struct BucketArgs {
    /// Local bucket root directory (default: OS state directory bucket-store)
    #[arg(long, global = true)]
    pub root: Option<PathBuf>,
    /// Coordinator HTTP URL for generating external access commands
    #[arg(long, global = true, default_value = "http://127.0.0.1:7848")]
    pub coordinator: String,
    #[command(subcommand)]
    pub command: Option<BucketCommand>,
}

#[derive(Subcommand)]
pub enum BucketCommand {
    /// Inspect the local storage bucket store, erasure coding status, and local host storage mediums
    Status,
    /// List all storage mediums detected on the local host (GPU VRAM, Unified Memory, System RAM, NVMe, UFS/eMMC/SD Flash, NAS ZFS/RAID)
    Mediums,
    /// List objects stored in kdm://pool-bucket/...
    #[command(visible_alias = "list")]
    Ls,
    /// Upload and erasure-code a local file into kdm://pool-bucket/...
    #[command(visible_alias = "upload")]
    Put {
        /// Local file path to upload
        source: PathBuf,
        /// Destination URI in kdm://pool-bucket/... format
        object_uri: String,
        /// Object kind / storage class: checkpoint | dataset | rollout | kv
        #[arg(short, long, default_value = "dataset")]
        class: String,
    },
    /// Download or byte-range read an object from kdm://pool-bucket/... (reconstructs missing shards from Reed-Solomon parity)
    #[command(visible_alias = "download")]
    Get {
        /// Object URI in kdm://pool-bucket/... format
        object_uri: String,
        /// Optional local output file path (prints JSON metadata + UTF-8 preview if omitted)
        output: Option<PathBuf>,
        /// Optional byte range start offset
        #[arg(long)]
        offset: Option<u64>,
        /// Optional byte range length
        #[arg(long)]
        length: Option<u64>,
    },
    /// Verify all data and parity shards and Merkle/SHA-256 digests for a bucket object
    Verify {
        /// Object URI in kdm://pool-bucket/... format
        object_uri: String,
    },
    /// Delete an object and its shards from the bucket
    #[command(visible_alias = "delete")]
    Rm {
        /// Object URI in kdm://pool-bucket/... format
        object_uri: String,
    },
    /// Show ways to upload, download, or access a bucket object within Kadmium and outside of Kadmium
    Access {
        /// Object URI in kdm://pool-bucket/... format
        #[arg(default_value = "kdm://pool-bucket/checkpoints/deepseek-v3-671b/epoch-42.ktb")]
        object_uri: String,
    },
}

fn format_storage_class(class: &str) -> &'static str {
    match class.trim().to_ascii_lowercase().as_str() {
        "checkpoint" | "model" | "ktb" | "modelcheckpointktb" => {
            "Distributed Training Checkpoint (.ktb)"
        }
        "rollout" | "rl" | "rltrajectoryrollout" => "RL Trajectory Rollout Bundle",
        "kv" | "kvcache" | "kvcachesnapshot" => "KV-Cache Hot Snapshot",
        _ => "Wire-Carousel Dataset Corpus (.ktb)",
    }
}

pub fn run(args: BucketArgs) -> Result<(), String> {
    let store = match args.root {
        Some(root) => BucketObjectStore::new(root),
        None => BucketObjectStore::default_store(),
    };
    let nodes = vec![LiveHardwareProfiler::probe_local_host()];

    let value = match args.command.unwrap_or(BucketCommand::Status) {
        BucketCommand::Status => {
            let manifests = store.list_objects()?;
            let report =
                DecentralizedStoragePoolEngine::evaluate_live_cluster_bucket(&nodes, &[], &[], &[]);
            let guide = store.access_guide(
                "kdm://pool-bucket/checkpoints/deepseek-v3-671b/epoch-42.ktb",
                &args.coordinator,
            )?;
            json!({
                "storage_mode": "local_coordinator_erasure_store",
                "report": report,
                "persisted_objects": manifests.len(),
                "access_guide": guide,
            })
        }
        BucketCommand::Mediums => {
            let summaries = DecentralizedStoragePoolEngine::summarize_cluster_storage_mediums(
                &nodes,
                &[],
                false,
            );
            let node_profiles: Vec<_> = nodes
                .iter()
                .map(|n| {
                    json!({
                        "node_id": n.id.0,
                        "hostname": n.hostname,
                        "mediums": DecentralizedStoragePoolEngine::inspect_node_storage_mediums(n, false),
                    })
                })
                .collect();
            json!({
                "host_scope": "local_host_probe",
                "cluster_medium_summaries": summaries,
                "node_profiles": node_profiles,
            })
        }
        BucketCommand::Ls => {
            let manifests = store.list_objects()?;
            json!({
                "bucket_root": store.root_dir(),
                "object_count": manifests.len(),
                "objects": manifests,
            })
        }
        BucketCommand::Put {
            source,
            object_uri,
            class,
        } => {
            let kind = format_storage_class(&class);
            let receipt = store.upload_file(
                &source,
                &object_uri,
                None,
                Some(kind),
                8,
                2,
                &[],
                Some(&args.coordinator),
            )?;
            json!({
                "uploaded": receipt.manifest,
                "bucket_entry": receipt.bucket_entry,
                "access_guide": receipt.access_guide,
            })
        }
        BucketCommand::Get {
            object_uri,
            output,
            offset,
            length,
        } => {
            if let Some(ref out_path) = output {
                let mut dl = store.download_to_path(&object_uri, out_path)?;
                dl.bytes.clear();
                json!({
                    "downloaded": dl,
                    "output_path": out_path,
                })
            } else {
                let range = offset.map(|off| (off, length.unwrap_or(u64::MAX - off)));
                let mut dl = store.get_bytes(&object_uri, range)?;
                let content_utf8 = String::from_utf8(dl.bytes.clone()).ok();
                dl.bytes.clear();
                json!({
                    "download": dl,
                    "returned_bytes": dl.returned_bytes_len,
                    "content_utf8": content_utf8,
                })
            }
        }
        BucketCommand::Verify { object_uri } => {
            let verification = store.verify_object(&object_uri, 1)?;
            serde_json::to_value(verification).map_err(|e| e.to_string())?
        }
        BucketCommand::Rm { object_uri } => {
            let removed = store.delete_object(&object_uri)?;
            json!({
                "object_uri": object_uri,
                "removed": removed,
            })
        }
        BucketCommand::Access { object_uri } => {
            let guide = store.access_guide(&object_uri, &args.coordinator)?;
            serde_json::to_value(guide).map_err(|e| e.to_string())?
        }
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_cli_put_verify_get_and_rm() {
        let tmp =
            std::env::temp_dir().join(format!("kadmium-cli-bucket-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("input.txt");
        let dst = tmp.join("output.txt");
        std::fs::write(&src, b"Kadmium CLI distributed bucket test payload").unwrap();

        run(BucketArgs {
            root: Some(tmp.clone()),
            coordinator: "http://127.0.0.1:7848".into(),
            command: Some(BucketCommand::Put {
                source: src.clone(),
                object_uri: "kdm://pool-bucket/cli/test.txt".into(),
                class: "checkpoint".into(),
            }),
        })
        .unwrap();

        run(BucketArgs {
            root: Some(tmp.clone()),
            coordinator: "http://127.0.0.1:7848".into(),
            command: Some(BucketCommand::Verify {
                object_uri: "kdm://pool-bucket/cli/test.txt".into(),
            }),
        })
        .unwrap();

        run(BucketArgs {
            root: Some(tmp.clone()),
            coordinator: "http://127.0.0.1:7848".into(),
            command: Some(BucketCommand::Get {
                object_uri: "kdm://pool-bucket/cli/test.txt".into(),
                output: Some(dst.clone()),
                offset: None,
                length: None,
            }),
        })
        .unwrap();

        assert_eq!(
            std::fs::read(&dst).unwrap(),
            b"Kadmium CLI distributed bucket test payload"
        );

        run(BucketArgs {
            root: Some(tmp.clone()),
            coordinator: "http://127.0.0.1:7848".into(),
            command: Some(BucketCommand::Rm {
                object_uri: "kdm://pool-bucket/cli/test.txt".into(),
            }),
        })
        .unwrap();

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
