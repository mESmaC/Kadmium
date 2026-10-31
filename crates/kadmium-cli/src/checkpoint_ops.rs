use clap::{Args, Subcommand};
use kadmium_data::checkpoint::CheckpointLibrary;
use std::path::PathBuf;

#[derive(Args)]
pub struct CheckpointArgs {
    /// Package directory (default: KADMIUM_MODEL_LIBRARY or target/kadmium-model-library).
    #[arg(long, global = true)]
    library: Option<PathBuf>,
    #[command(subcommand)]
    command: CheckpointCommand,
}

#[derive(Subcommand)]
enum CheckpointCommand {
    /// Preserve a complete local checkpoint and pack its tensors into lossless KTB banks.
    Normalize { path: PathBuf },
    /// List normalized checkpoints; does not load a model.
    List,
    /// Re-read and verify all source files and KTB banks by SHA-256.
    Verify { id: String },
}

pub fn run(args: CheckpointArgs) -> Result<(), String> {
    let library = args.library.map(CheckpointLibrary::new).unwrap_or_default();
    let value = match args.command {
        CheckpointCommand::Normalize { path } => {
            serde_json::to_value(library.import(&path)?.summary())
        }
        CheckpointCommand::List => serde_json::to_value(library.list()?),
        CheckpointCommand::Verify { id } => serde_json::to_value(library.verify(&id)?.summary()),
    }
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?
    );
    Ok(())
}
