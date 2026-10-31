//! Training checkpoints on disk: the `.ktb` files written by `POST /api/v1/training/checkpoint`,
//! listed by `GET /api/v1/training/checkpoints` and resumed with `POST /api/v1/training/resume`.
//!
//! The files are the source of truth, so a checkpoint saved before a restart can still be
//! resumed afterwards. Every file is parsed and its tensor checksums are checked before it is
//! offered, and again before training is pointed at it.

use crate::GatewayState;
use axum::{Json, extract::State, http::StatusCode};
use kadmium_core::ModelSpec;
use kadmium_data::{KtbBankReader, KtbMerkleTree, compute_fnv1a64_checksum};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

/// Most checkpoint files read per listing; the newest files win.
const MAX_LISTED_CHECKPOINTS: usize = 200;

/// Folder the coordinator writes training checkpoints to and reads them back from.
pub fn checkpoint_dir() -> PathBuf {
    std::env::temp_dir().join("kadmium-checkpoints")
}

/// One checkpoint file that training can resume from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingCheckpoint {
    pub checkpoint_id: String,
    pub model_id: String,
    pub step_index: u64,
    pub file_path: String,
    pub file_size_bytes: u64,
    pub tensor_slice_count: usize,
    pub merkle_root_fnv64: u64,
    /// When the file was last written, in seconds since the Unix epoch (0 if unknown).
    pub saved_at_unix_secs: u64,
    /// True when every tensor slice in the file matches the checksum stored in its header.
    pub verified: bool,
}

/// Response of `GET /api/v1/training/checkpoints`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpointList {
    pub directory: String,
    pub training_model_id: String,
    pub current_step: u64,
    /// The checkpoint training was last resumed from, if any.
    pub resumed_from: Option<String>,
    /// Newest step first.
    pub checkpoints: Vec<TrainingCheckpoint>,
}

/// Body of `POST /api/v1/training/resume`.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainingResumeRequest {
    pub checkpoint_id: String,
}

/// Response of `POST /api/v1/training/resume`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResumeReport {
    pub checkpoint: TrainingCheckpoint,
    pub previous_step: u64,
    pub resumed_step: u64,
    pub previous_model_id: String,
    pub model_changed: bool,
}

/// Reads one checkpoint: its header, its step (from the file name) and its slice checksums.
pub fn read_checkpoint(path: &Path) -> Result<TrainingCheckpoint, String> {
    let step_index = step_from_file_name(path)
        .ok_or_else(|| format!("{} is not named like a Kadmium checkpoint", path.display()))?;
    let bytes =
        std::fs::read(path).map_err(|err| format!("could not read {}: {err}", path.display()))?;
    // The header length lives in bytes 6..10; check it before the reader slices with it.
    let header_len = bytes
        .get(6..10)
        .map(|raw| u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize)
        .ok_or_else(|| format!("{} is too short to be a .ktb file", path.display()))?;
    if 10usize.saturating_add(header_len) > bytes.len() {
        return Err(format!("{} has a damaged header", path.display()));
    }
    let reader = KtbBankReader::from_slice(&bytes)?;
    let verified = reader
        .header
        .entries
        .iter()
        .all(|entry| reader.get_verified_tensor_slice(entry).is_ok());
    let saved_at_unix_secs = std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    Ok(TrainingCheckpoint {
        // Same id the export reports, so a freshly saved checkpoint is found under its id.
        checkpoint_id: format!(
            "ckpt-step{step_index:04}-{:08x}",
            compute_fnv1a64_checksum(&bytes) as u32
        ),
        model_id: reader.header.model_id.clone(),
        step_index,
        file_path: path.display().to_string(),
        file_size_bytes: bytes.len() as u64,
        tensor_slice_count: reader.header.entries.len(),
        merkle_root_fnv64: KtbMerkleTree::from_ktb_bytes(&bytes).root_hash,
        saved_at_unix_secs,
        verified,
    })
}

/// `kadmium-<model>-step-<n>.ktb` → `n`.
fn step_from_file_name(path: &Path) -> Option<u64> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_prefix("kadmium-")?.strip_suffix(".ktb")?;
    let (_, step) = stem.rsplit_once("-step-")?;
    step.parse().ok()
}

/// Every readable checkpoint in `dir`, newest step first. A missing folder means none yet.
pub fn list_checkpoints(dir: &Path) -> Vec<TrainingCheckpoint> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<(PathBuf, u64)> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "ktb"))
        .map(|path| {
            let modified = std::fs::metadata(&path)
                .and_then(|meta| meta.modified())
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|elapsed| elapsed.as_secs())
                .unwrap_or(0);
            (path, modified)
        })
        .collect();
    files.sort_by_key(|(_, modified)| std::cmp::Reverse(*modified));
    files.truncate(MAX_LISTED_CHECKPOINTS);
    let mut found: Vec<TrainingCheckpoint> = files
        .iter()
        .filter_map(|(path, _)| read_checkpoint(path).ok())
        .collect();
    found.sort_by(|a, b| {
        b.step_index
            .cmp(&a.step_index)
            .then(b.saved_at_unix_secs.cmp(&a.saved_at_unix_secs))
    });
    found
}

impl GatewayState {
    /// Points training at `checkpoint`: its model, its step and its weights' Merkle root.
    /// The next training step continues from `checkpoint.step_index + 1`.
    pub fn resume_training_from(
        &mut self,
        checkpoint: &TrainingCheckpoint,
    ) -> TrainingResumeReport {
        let previous_step = self.training_step_counter;
        let previous_model_id = self.training_model.id.clone();
        let model_changed = previous_model_id != checkpoint.model_id;
        if model_changed {
            self.training_model = ModelSpec::from_id(&checkpoint.model_id);
            // Keeps the live planner from swapping the recommended model back in.
            self.user_selected_model = true;
        }
        self.training_step_counter = checkpoint.step_index;
        self.last_checkpoint_step = checkpoint.step_index;
        self.training_loss_history
            .retain(|(step, _)| *step <= checkpoint.step_index);
        if let Some((_, loss)) = self.training_loss_history.last() {
            self.last_training_loss = *loss;
        }
        self.active_ktb_merkle_root_fnv64 = checkpoint.merkle_root_fnv64;
        self.resumed_from_checkpoint = Some(checkpoint.checkpoint_id.clone());
        self.epoch += 1;
        self.recompute_plans();
        TrainingResumeReport {
            checkpoint: checkpoint.clone(),
            previous_step,
            resumed_step: checkpoint.step_index,
            previous_model_id,
            model_changed,
        }
    }
}

type ApiError = (StatusCode, Json<serde_json::Value>);

fn api_error(status: StatusCode, message: String) -> ApiError {
    (status, Json(serde_json::json!({ "error": message })))
}

/// `GET /api/v1/training/checkpoints`
pub async fn get_training_checkpoints(
    State(state): State<Arc<Mutex<GatewayState>>>,
) -> Json<TrainingCheckpointList> {
    let dir = checkpoint_dir();
    let scan_dir = dir.clone();
    let checkpoints = tokio::task::spawn_blocking(move || list_checkpoints(&scan_dir))
        .await
        .unwrap_or_default();
    let guard = state.lock().expect("gateway state lock");
    Json(TrainingCheckpointList {
        directory: dir.display().to_string(),
        training_model_id: guard.training_model.id.clone(),
        current_step: guard.training_step_counter,
        resumed_from: guard.resumed_from_checkpoint.clone(),
        checkpoints,
    })
}

/// `POST /api/v1/training/resume` with `{ "checkpoint_id": "ckpt-step0042-…" }`
pub async fn post_training_resume(
    State(state): State<Arc<Mutex<GatewayState>>>,
    Json(request): Json<TrainingResumeRequest>,
) -> Result<Json<TrainingResumeReport>, ApiError> {
    let dir = checkpoint_dir();
    let wanted = request.checkpoint_id.clone();
    let scan_dir = dir.clone();
    let found = tokio::task::spawn_blocking(move || {
        list_checkpoints(&scan_dir)
            .into_iter()
            .find(|checkpoint| checkpoint.checkpoint_id == wanted)
    })
    .await
    .ok()
    .flatten();
    let Some(checkpoint) = found else {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            format!(
                "No checkpoint {} in {}. Save one first, or pick one from the list.",
                request.checkpoint_id,
                dir.display()
            ),
        ));
    };
    if !checkpoint.verified {
        return Err(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "Checkpoint {} failed its checksum check, so training was not resumed from it.",
                checkpoint.checkpoint_id
            ),
        ));
    }
    if ModelSpec::from_id(&checkpoint.model_id).id != checkpoint.model_id {
        return Err(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "Checkpoint {} belongs to {}, which this coordinator has no model profile for.",
                checkpoint.checkpoint_id, checkpoint.model_id
            ),
        ));
    }
    let mut guard = state.lock().expect("gateway state lock");
    let previous = (
        guard.training_model.clone(),
        guard.user_selected_model,
        guard.training_step_counter,
        guard.last_checkpoint_step,
        guard.training_loss_history.clone(),
        guard.last_training_loss,
        guard.active_ktb_merkle_root_fnv64,
        guard.resumed_from_checkpoint.clone(),
        guard.epoch,
    );
    let report = guard.resume_training_from(&checkpoint);
    if let Err(error) = guard.save_persistent_state() {
        (
            guard.training_model,
            guard.user_selected_model,
            guard.training_step_counter,
            guard.last_checkpoint_step,
            guard.training_loss_history,
            guard.last_training_loss,
            guard.active_ktb_merkle_root_fnv64,
            guard.resumed_from_checkpoint,
            guard.epoch,
        ) = previous;
        guard.recompute_plans();
        return Err(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Training was not resumed because its journal could not be saved: {error}"),
        ));
    }
    Ok(Json(report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_runtime::DistributedLanExecutor;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("kadmium-ckpt-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn reads_steps_from_checkpoint_file_names() {
        let step = |name: &str| step_from_file_name(Path::new(name));
        assert_eq!(
            step("kadmium-llama-3_2-3b-instruct-step-0042.ktb"),
            Some(42)
        );
        assert_eq!(step("kadmium-a-step-step-7.ktb"), Some(7));
        assert_eq!(step("kadmium-model.ktb"), None);
        assert_eq!(step("other-model-step-0001.ktb"), None);
        assert_eq!(step("kadmium-model-step-0001.bin"), None);
    }

    #[test]
    fn lists_verifies_and_resumes_from_a_saved_checkpoint() {
        let dir = scratch_dir("resume");
        let export = DistributedLanExecutor::export_checkpoint_ktb_to_disk(
            "llama-3.2-3b-instruct",
            42,
            &dir,
        )
        .expect("export a checkpoint");

        let listed = list_checkpoints(&dir);
        assert_eq!(listed.len(), 1);
        let checkpoint = &listed[0];
        assert_eq!(checkpoint.checkpoint_id, export.checkpoint_id);
        assert_eq!(checkpoint.step_index, 42);
        assert_eq!(checkpoint.model_id, "llama-3.2-3b-instruct");
        assert_eq!(checkpoint.merkle_root_fnv64, export.merkle_root_fnv64);
        assert!(checkpoint.verified);

        let mut state = GatewayState::new();
        state.training_step_counter = 99;
        let report = state.resume_training_from(checkpoint);
        assert_eq!(report.previous_step, 99);
        assert_eq!(report.resumed_step, 42);
        assert!(report.model_changed);
        assert_eq!(state.training_step_counter, 42);
        assert_eq!(state.last_checkpoint_step, 42);
        assert_eq!(state.training_model.id, "llama-3.2-3b-instruct");
        assert_eq!(
            state.resumed_from_checkpoint.as_deref(),
            Some(export.checkpoint_id.as_str())
        );
        assert!(
            state
                .training_loss_history
                .iter()
                .all(|(step, _)| *step <= 42)
        );

        // Damage one byte of the first tensor payload: the file no longer verifies.
        let mut bytes = std::fs::read(&export.file_path).unwrap();
        let first_payload = kadmium_data::KTB_PAGE_ALIGNMENT_BYTES as usize;
        bytes[first_payload] ^= 0xff;
        std::fs::write(&export.file_path, &bytes).unwrap();
        assert!(
            !read_checkpoint(Path::new(&export.file_path))
                .unwrap()
                .verified
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_folder_has_no_checkpoints() {
        assert!(list_checkpoints(&scratch_dir("missing")).is_empty());
    }
}
