use crate::CondenserError;
use crate::server::{ApiError, ServerState};
use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::task::JoinHandle;

const DIRECTORY_ENTRY_LIMIT: usize = 500;
const FILE_READ_LIMIT: usize = 1024 * 1024;
const TERMINAL_OUTPUT_LIMIT: usize = 128 * 1024;
const TERMINAL_COMMAND_LIMIT: usize = 16_000;
const TERMINAL_TIMEOUT: Duration = Duration::from_secs(20);
const TERMINAL_READER_GRACE: Duration = Duration::from_secs(1);

#[derive(Clone, Debug)]
pub(crate) struct WorkspaceState {
    pub(crate) root: Arc<PathBuf>,
}

impl WorkspaceState {
    pub(crate) fn from_current_dir() -> Result<Self, CondenserError> {
        let root = std::env::current_dir()?.canonicalize()?;
        Ok(Self {
            root: Arc::new(root),
        })
    }
}

#[derive(Serialize)]
pub(crate) struct WorkspaceResponse {
    root: String,
    name: String,
    separator: String,
}

pub(crate) async fn workspace(State(state): State<ServerState>) -> Json<WorkspaceResponse> {
    let root = state.workspace.root.as_ref();
    let display = root.to_string_lossy().into_owned();
    let name = root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| display.clone());
    Json(WorkspaceResponse {
        root: display,
        name,
        separator: std::path::MAIN_SEPARATOR.to_string(),
    })
}

#[derive(Debug, Deserialize)]
pub(crate) struct ListFilesRequest {
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub(crate) struct FileEntry {
    name: String,
    path: String,
    kind: &'static str,
    size: u64,
    modified: Option<u64>,
}

#[derive(Serialize)]
pub(crate) struct ListFilesResponse {
    path: String,
    parent: Option<String>,
    entries: Vec<FileEntry>,
    truncated: bool,
}

#[derive(Debug, Eq, PartialEq)]
struct EntryKey {
    kind_rank: u8,
    folded_name: String,
    name: String,
}

impl Ord for EntryKey {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.kind_rank, &self.folded_name, &self.name).cmp(&(
            &other.kind_rank,
            &other.folded_name,
            &other.name,
        ))
    }
}

impl PartialOrd for EntryKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Eq, PartialEq)]
struct RankedEntry {
    key: EntryKey,
    entry: FileEntry,
}

impl Ord for RankedEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for RankedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl FileEntry {
    fn ranked(self) -> RankedEntry {
        RankedEntry {
            key: EntryKey {
                kind_rank: if self.kind == "directory" { 0 } else { 1 },
                folded_name: self.name.to_lowercase(),
                name: self.name.clone(),
            },
            entry: self,
        }
    }
}

pub(crate) async fn list_files(
    State(state): State<ServerState>,
    Json(request): Json<ListFilesRequest>,
) -> Result<Json<ListFilesResponse>, ApiError> {
    let root = state.workspace.root.clone();
    let requested = request.path.unwrap_or_default();
    let response =
        tokio::task::spawn_blocking(move || list_files_blocking(root.as_ref(), &requested))
            .await
            .map_err(|error| {
                CondenserError::Io(std::io::Error::other(format!(
                    "directory listing task failed: {error}"
                )))
            })??;
    Ok(Json(response))
}

fn list_files_blocking(root: &Path, requested: &str) -> Result<ListFilesResponse, CondenserError> {
    let directory = resolve_path(root, requested)?;
    if !directory.is_dir() {
        return Err(bad_request("workspace path is not a directory"));
    }
    let path = relative_path(root, &directory)?;
    let parent = if path.is_empty() {
        None
    } else {
        Some(
            path.rsplit_once('/')
                .map(|(parent, _)| parent.to_string())
                .unwrap_or_default(),
        )
    };
    let iterator = std::fs::read_dir(&directory)
        .map_err(|error| bad_request(format!("cannot list workspace directory: {error}")))?;
    let mut selected = BinaryHeap::with_capacity(DIRECTORY_ENTRY_LIMIT);
    let mut total = 0usize;

    for item in iterator {
        let item = item.map_err(|error| {
            bad_request(format!("cannot inspect workspace directory entry: {error}"))
        })?;
        let name = item.file_name().to_string_lossy().into_owned();
        if should_skip_entry(&name) {
            continue;
        }
        let Ok(metadata) = std::fs::symlink_metadata(item.path()) else {
            continue;
        };
        if metadata.file_type().is_symlink() {
            continue;
        }
        let kind = if metadata.is_dir() {
            "directory"
        } else if metadata.is_file() {
            "file"
        } else {
            continue;
        };
        let entry_path = relative_path(root, &item.path())?;
        let entry = FileEntry {
            name,
            path: entry_path,
            kind,
            size: metadata.len(),
            modified: modified_millis(metadata.modified().ok()),
        };
        total += 1;
        let ranked = entry.ranked();
        if selected.len() < DIRECTORY_ENTRY_LIMIT {
            selected.push(ranked);
        } else if selected
            .peek()
            .is_some_and(|worst| ranked.cmp(worst) == Ordering::Less)
        {
            selected.pop();
            selected.push(ranked);
        }
    }

    let mut selected = selected.into_vec();
    selected.sort();
    Ok(ListFilesResponse {
        path,
        parent,
        entries: selected.into_iter().map(|ranked| ranked.entry).collect(),
        truncated: total > DIRECTORY_ENTRY_LIMIT,
    })
}

#[derive(Debug, Deserialize)]
pub(crate) struct ReadFileRequest {
    path: String,
}

#[derive(Serialize)]
pub(crate) struct ReadFileResponse {
    path: String,
    name: String,
    content: String,
    size: u64,
    truncated: bool,
}

pub(crate) async fn read_file(
    State(state): State<ServerState>,
    Json(request): Json<ReadFileRequest>,
) -> Result<Json<ReadFileResponse>, ApiError> {
    let root = state.workspace.root.as_ref();
    let file = resolve_path(root, &request.path)?;
    let name = file
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| bad_request("workspace file must have a filename"))?;
    if is_sensitive_file(&name) {
        return Err(bad_request("refusing to read a sensitive file").into());
    }
    let metadata = tokio::fs::metadata(&file)
        .await
        .map_err(|error| bad_request(format!("cannot inspect workspace file: {error}")))?;
    if !metadata.is_file() {
        return Err(bad_request("workspace path is not a file").into());
    }

    let mut bytes = Vec::new();
    tokio::fs::File::open(&file)
        .await?
        .take((FILE_READ_LIMIT + 1) as u64)
        .read_to_end(&mut bytes)
        .await?;
    let truncated = metadata.len() > FILE_READ_LIMIT as u64 || bytes.len() > FILE_READ_LIMIT;
    bytes.truncate(FILE_READ_LIMIT);
    if bytes.contains(&0) {
        return Err(bad_request("binary workspace files cannot be read").into());
    }
    let content = match std::str::from_utf8(&bytes) {
        Ok(content) => content.to_string(),
        Err(error) if truncated && error.error_len().is_none() => {
            std::str::from_utf8(&bytes[..error.valid_up_to()])
                .expect("the valid UTF-8 prefix must remain valid")
                .to_string()
        }
        Err(_) => {
            return Err(bad_request("workspace file must contain UTF-8 text").into());
        }
    };
    Ok(Json(ReadFileResponse {
        path: relative_path(root, &file)?,
        name,
        content,
        size: metadata.len(),
        truncated,
    }))
}

#[derive(Debug, Deserialize)]
pub(crate) struct WriteFileRequest {
    path: String,
    content: String,
}

#[derive(Serialize)]
pub(crate) struct WriteFileResponse {
    path: String,
    size: u64,
    modified: Option<u64>,
}

pub(crate) async fn write_file(
    State(state): State<ServerState>,
    Json(request): Json<WriteFileRequest>,
) -> Result<Json<WriteFileResponse>, ApiError> {
    Ok(Json(
        write_text_file(
            state.workspace.root.as_ref(),
            &request.path,
            &request.content,
        )
        .await?,
    ))
}

pub(crate) async fn write_text_file(
    root: &Path,
    requested: &str,
    content: &str,
) -> Result<WriteFileResponse, CondenserError> {
    if content.len() > FILE_READ_LIMIT {
        return Err(bad_request(
            "workspace write content exceeds the 1 MiB limit",
        ));
    }
    if content.as_bytes().contains(&0) {
        return Err(bad_request("binary workspace files cannot be written"));
    }

    let file = resolve_path(root, requested)?;
    let name = file
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| bad_request("workspace file must have a filename"))?;
    if is_sensitive_file(&name) {
        return Err(bad_request("refusing to write a sensitive file"));
    }
    let requested_path = candidate_path(root, requested);
    let requested_metadata = tokio::fs::symlink_metadata(&requested_path)
        .await
        .map_err(|error| bad_request(format!("cannot inspect workspace write target: {error}")))?;
    if requested_metadata.file_type().is_symlink() {
        return Err(bad_request("workspace write target cannot be a symlink"));
    }
    let metadata = tokio::fs::symlink_metadata(&file)
        .await
        .map_err(|error| bad_request(format!("cannot inspect workspace write target: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(bad_request(
            "workspace write target must be an existing regular file",
        ));
    }

    if !existing_file_is_utf8_text(&file).await? {
        return Err(bad_request(
            "existing workspace file must contain UTF-8 text without NUL bytes",
        ));
    }

    let directory = file
        .parent()
        .ok_or_else(|| bad_request("workspace write target must have a parent directory"))?;
    let (temporary, temporary_file) = create_temporary_file(directory)
        .await
        .map_err(|error| write_io("cannot create workspace temporary file", error))?;
    let result = replace_with_content(
        &temporary,
        temporary_file,
        &file,
        &metadata,
        content.as_bytes(),
    )
    .await;
    if let Err(error) = result {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(write_io("cannot replace workspace file", error));
    }

    let metadata = tokio::fs::metadata(&file)
        .await
        .map_err(|error| write_io("cannot inspect replaced workspace file", error))?;
    Ok(WriteFileResponse {
        path: relative_path(root, &file)?,
        size: metadata.len(),
        modified: modified_millis(metadata.modified().ok()),
    })
}

async fn existing_file_is_utf8_text(path: &Path) -> std::io::Result<bool> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut buffer = [0u8; 64 * 1024];
    let mut pending = Vec::with_capacity(4);
    loop {
        let read = file.read(&mut buffer).await?;
        if read == 0 {
            return Ok(pending.is_empty());
        }
        if buffer[..read].contains(&0) {
            return Ok(false);
        }
        let mut bytes = Vec::with_capacity(pending.len() + read);
        bytes.extend_from_slice(&pending);
        bytes.extend_from_slice(&buffer[..read]);
        match std::str::from_utf8(&bytes) {
            Ok(_) => pending.clear(),
            Err(error) if error.error_len().is_none() => {
                pending.clear();
                pending.extend_from_slice(&bytes[error.valid_up_to()..]);
            }
            Err(_) => return Ok(false),
        }
    }
}

async fn create_temporary_file(directory: &Path) -> std::io::Result<(PathBuf, tokio::fs::File)> {
    static NEXT_TEMPORARY_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for _ in 0..32 {
        let id = NEXT_TEMPORARY_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = directory.join(format!(".condenser-write-{}-{id}.tmp", std::process::id()));
        match tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .await
        {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a unique workspace temporary file",
    ))
}

async fn replace_with_content(
    temporary: &Path,
    mut temporary_file: tokio::fs::File,
    target: &Path,
    metadata: &std::fs::Metadata,
    content: &[u8],
) -> std::io::Result<()> {
    temporary_file.write_all(content).await?;
    temporary_file.sync_all().await?;
    drop(temporary_file);
    tokio::fs::set_permissions(temporary, metadata.permissions()).await?;
    replace_file(temporary, target).await
}

#[cfg(not(windows))]
async fn replace_file(temporary: &Path, target: &Path) -> std::io::Result<()> {
    tokio::fs::rename(temporary, target).await
}

#[cfg(windows)]
async fn replace_file(temporary: &Path, target: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }
    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;

    let temporary = temporary
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let target = target
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let replaced = unsafe {
        MoveFileExW(
            temporary.as_ptr(),
            target.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if replaced == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn write_io(context: &str, error: std::io::Error) -> CondenserError {
    CondenserError::Io(std::io::Error::new(
        error.kind(),
        format!("{context}: {error}"),
    ))
}

#[derive(Debug, Deserialize)]
pub(crate) struct CreateFileRequest {
    path: String,
    #[serde(default)]
    content: String,
}

pub(crate) async fn create_file(
    State(state): State<ServerState>,
    Json(request): Json<CreateFileRequest>,
) -> Result<Json<WriteFileResponse>, ApiError> {
    Ok(Json(
        create_text_file(
            state.workspace.root.as_ref(),
            &request.path,
            &request.content,
        )
        .await?,
    ))
}

async fn create_text_file(
    root: &Path,
    requested: &str,
    content: &str,
) -> Result<WriteFileResponse, CondenserError> {
    validate_new_text(content)?;
    let target = new_file_target(root, requested)?;
    let mut file = match tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&target)
        .await
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(bad_request("workspace create target already exists"));
        }
        Err(error) => return Err(write_io("cannot create workspace file", error)),
    };
    let result = async {
        file.write_all(content.as_bytes()).await?;
        file.sync_all().await
    }
    .await;
    if let Err(error) = result {
        drop(file);
        let _ = tokio::fs::remove_file(&target).await;
        return Err(write_io("cannot write new workspace file", error));
    }
    drop(file);
    file_metadata_response(root, &target).await
}

#[derive(Debug, Deserialize)]
pub(crate) struct RenameFileRequest {
    path: String,
    new_path: String,
}

pub(crate) async fn rename_file(
    State(state): State<ServerState>,
    Json(request): Json<RenameFileRequest>,
) -> Result<Json<WriteFileResponse>, ApiError> {
    Ok(Json(
        rename_workspace_file(
            state.workspace.root.as_ref(),
            &request.path,
            &request.new_path,
        )
        .await?,
    ))
}

async fn rename_workspace_file(
    root: &Path,
    requested: &str,
    new_requested: &str,
) -> Result<WriteFileResponse, CondenserError> {
    let (source, _) = existing_regular_file(root, requested, "rename").await?;
    let target = new_file_target(root, new_requested)?;
    require_missing_target(&target).await?;
    move_file_no_replace(&source, &target)
        .await
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                bad_request("workspace rename destination already exists")
            } else {
                write_io("cannot rename workspace file", error)
            }
        })?;
    file_metadata_response(root, &target).await
}

#[derive(Debug, Deserialize)]
pub(crate) struct DeleteFileRequest {
    path: String,
}

#[derive(Serialize)]
pub(crate) struct DeleteFileResponse {
    path: String,
    trash_path: String,
    size: u64,
    modified: Option<u64>,
}

pub(crate) async fn delete_file(
    State(state): State<ServerState>,
    Json(request): Json<DeleteFileRequest>,
) -> Result<Json<DeleteFileResponse>, ApiError> {
    Ok(Json(
        delete_workspace_file(state.workspace.root.as_ref(), &request.path).await?,
    ))
}

async fn delete_workspace_file(
    root: &Path,
    requested: &str,
) -> Result<DeleteFileResponse, CondenserError> {
    let (source, _) = existing_regular_file(root, requested, "delete").await?;
    let original_path = relative_path(root, &source)?;
    let original_name = source
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| bad_request("workspace delete target must have a filename"))?;
    let condenser = ensure_workspace_directory(root, &root.join(".condenser")).await?;
    let trash = ensure_workspace_directory(root, &condenser.join("trash")).await?;
    let timestamp = modified_millis(Some(SystemTime::now())).unwrap_or(0);

    for collision in 0..128u32 {
        let trash_name = if collision == 0 {
            format!("{timestamp}-{original_name}")
        } else {
            format!("{timestamp}-{collision}-{original_name}")
        };
        let target = trash.join(trash_name);
        match move_file_no_replace(&source, &target).await {
            Ok(()) => {
                let metadata = tokio::fs::metadata(&target)
                    .await
                    .map_err(|error| write_io("cannot inspect trashed workspace file", error))?;
                return Ok(DeleteFileResponse {
                    path: original_path,
                    trash_path: relative_path(root, &target)?,
                    size: metadata.len(),
                    modified: modified_millis(metadata.modified().ok()),
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(write_io("cannot move workspace file to trash", error)),
        }
    }
    Err(write_io(
        "cannot allocate a unique trash filename",
        std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "too many trash filename collisions",
        ),
    ))
}

fn validate_new_text(content: &str) -> Result<(), CondenserError> {
    if content.len() > FILE_READ_LIMIT {
        return Err(bad_request(
            "workspace create content exceeds the 1 MiB limit",
        ));
    }
    if content.as_bytes().contains(&0) {
        return Err(bad_request("binary workspace files cannot be created"));
    }
    Ok(())
}

fn new_file_target(root: &Path, requested: &str) -> Result<PathBuf, CondenserError> {
    if requested.is_empty()
        || requested
            .chars()
            .last()
            .is_some_and(std::path::is_separator)
    {
        return Err(bad_request("workspace file path must name a file"));
    }
    let candidate = candidate_path(root, requested);
    let name = candidate
        .file_name()
        .ok_or_else(|| bad_request("workspace file path must have a filename"))?;
    if is_sensitive_file(&name.to_string_lossy()) {
        return Err(bad_request("refusing to target a sensitive file"));
    }
    let parent = candidate
        .parent()
        .ok_or_else(|| bad_request("workspace file path must have a parent directory"))?;
    let parent = parent
        .canonicalize()
        .map_err(|error| bad_request(format!("invalid workspace parent directory: {error}")))?;
    if !parent.starts_with(root) || !parent.is_dir() {
        return Err(bad_request(
            "workspace parent directory escapes the service root or is not a directory",
        ));
    }
    Ok(parent.join(name))
}

async fn existing_regular_file(
    root: &Path,
    requested: &str,
    operation: &str,
) -> Result<(PathBuf, std::fs::Metadata), CondenserError> {
    let file = resolve_path(root, requested)?;
    let name = file
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| bad_request("workspace file must have a filename"))?;
    if is_sensitive_file(&name) {
        return Err(bad_request(format!(
            "refusing to {operation} a sensitive file"
        )));
    }
    let requested_path = candidate_path(root, requested);
    let requested_metadata = tokio::fs::symlink_metadata(&requested_path)
        .await
        .map_err(|error| bad_request(format!("cannot inspect workspace file: {error}")))?;
    if requested_metadata.file_type().is_symlink() {
        return Err(bad_request("workspace file cannot be a symlink"));
    }
    let metadata = tokio::fs::symlink_metadata(&file)
        .await
        .map_err(|error| bad_request(format!("cannot inspect workspace file: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(bad_request(
            "workspace path must be an existing regular file",
        ));
    }
    Ok((file, metadata))
}

async fn require_missing_target(target: &Path) -> Result<(), CondenserError> {
    match tokio::fs::symlink_metadata(target).await {
        Ok(_) => Err(bad_request("workspace destination already exists")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(bad_request(format!(
            "cannot inspect workspace destination: {error}"
        ))),
    }
}

async fn file_metadata_response(
    root: &Path,
    file: &Path,
) -> Result<WriteFileResponse, CondenserError> {
    let metadata = tokio::fs::metadata(file)
        .await
        .map_err(|error| write_io("cannot inspect workspace file metadata", error))?;
    Ok(WriteFileResponse {
        path: relative_path(root, file)?,
        size: metadata.len(),
        modified: modified_millis(metadata.modified().ok()),
    })
}

async fn ensure_workspace_directory(
    root: &Path,
    directory: &Path,
) -> Result<PathBuf, CondenserError> {
    match tokio::fs::create_dir(directory).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(write_io("cannot create workspace trash directory", error)),
    }
    let metadata = tokio::fs::symlink_metadata(directory)
        .await
        .map_err(|error| write_io("cannot inspect workspace trash directory", error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(bad_request(
            "workspace trash path must be a real directory, not a symlink",
        ));
    }
    let canonical = directory
        .canonicalize()
        .map_err(|error| write_io("cannot resolve workspace trash directory", error))?;
    if !canonical.starts_with(root) {
        return Err(bad_request(
            "workspace trash directory escapes the service root",
        ));
    }
    Ok(canonical)
}

#[cfg(not(windows))]
async fn move_file_no_replace(source: &Path, target: &Path) -> std::io::Result<()> {
    tokio::fs::hard_link(source, target).await?;
    if let Err(error) = tokio::fs::remove_file(source).await {
        let _ = tokio::fs::remove_file(target).await;
        return Err(error);
    }
    Ok(())
}

#[cfg(windows)]
async fn move_file_no_replace(source: &Path, target: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let target = target
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let moved = unsafe { MoveFileExW(source.as_ptr(), target.as_ptr(), MOVEFILE_WRITE_THROUGH) };
    if moved == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct TerminalRequest {
    command: String,
    #[serde(default)]
    cwd: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct TerminalResponse {
    cwd: String,
    stdout: String,
    stderr: String,
    stdout_truncated: bool,
    stderr_truncated: bool,
    exit_code: Option<i32>,
    duration_ms: u64,
    timed_out: bool,
}

enum StandaloneCd {
    NotCd,
    Change(String),
}

pub(crate) async fn execute_terminal(
    State(state): State<ServerState>,
    Json(request): Json<TerminalRequest>,
) -> Result<Json<TerminalResponse>, ApiError> {
    run_terminal(state.workspace.root, request).await
}

pub(crate) async fn run_terminal(
    root: Arc<PathBuf>,
    request: TerminalRequest,
) -> Result<Json<TerminalResponse>, ApiError> {
    let command = request.command.trim();
    let command_chars = request.command.chars().count();
    if command.is_empty() || command_chars > TERMINAL_COMMAND_LIMIT {
        return Err(bad_request(format!(
            "terminal command must contain between 1 and {TERMINAL_COMMAND_LIMIT} characters"
        ))
        .into());
    }
    let working = resolve_path(root.as_ref(), request.cwd.as_deref().unwrap_or(""))?;
    if !working.is_dir() {
        return Err(bad_request("terminal cwd is not a directory").into());
    }

    if let StandaloneCd::Change(target) = parse_standalone_cd(command)? {
        let destination = if target.is_empty() {
            root.as_ref().clone()
        } else {
            resolve_path_at(root.as_ref(), &working, &target)?
        };
        if !destination.is_dir() {
            return Err(bad_request("terminal cd target is not a directory").into());
        }
        return Ok(Json(TerminalResponse {
            cwd: relative_path(root.as_ref(), &destination)?,
            stdout: String::new(),
            stderr: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            exit_code: Some(0),
            duration_ms: 0,
            timed_out: false,
        }));
    }

    let started = std::time::Instant::now();
    let mut process = shell_command(command);
    process
        .current_dir(&working)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    process.process_group(0);
    let mut child = process.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("terminal stdout was not piped"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("terminal stderr was not piped"))?;
    let stdout_capture = SharedCapture::default();
    let stderr_capture = SharedCapture::default();
    let mut stdout_task = tokio::spawn(read_capped(stdout, stdout_capture.clone()));
    let mut stderr_task = tokio::spawn(read_capped(stderr, stderr_capture.clone()));

    let (status, timed_out) = match tokio::time::timeout(TERMINAL_TIMEOUT, child.wait()).await {
        Ok(status) => (Some(status?), false),
        Err(_) => {
            kill_process_tree(&mut child).await?;
            (None, true)
        }
    };
    let stdout = collect_reader(&mut stdout_task, &stdout_capture).await?;
    let stderr = collect_reader(&mut stderr_task, &stderr_capture).await?;
    let exit_code = if timed_out {
        None
    } else {
        status.and_then(|status| status.code())
    };
    Ok(Json(TerminalResponse {
        cwd: relative_path(root.as_ref(), &working)?,
        stdout: decode_terminal_output(stdout.bytes),
        stderr: decode_terminal_output(stderr.bytes),
        stdout_truncated: stdout.truncated,
        stderr_truncated: stderr.truncated,
        exit_code,
        duration_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        timed_out,
    }))
}

fn shell_command(command: &str) -> Command {
    #[cfg(windows)]
    {
        let mut process = Command::new("cmd.exe");
        process.arg("/C").arg(command);
        process
    }
    #[cfg(not(windows))]
    {
        let mut process = Command::new("sh");
        process.arg("-c").arg(command);
        process
    }
}

fn parse_standalone_cd(command: &str) -> Result<StandaloneCd, CondenserError> {
    let split_at = command.find(char::is_whitespace).unwrap_or(command.len());
    let (first, remainder) = command.split_at(split_at);
    let is_cd = if cfg!(windows) {
        first.eq_ignore_ascii_case("cd")
    } else {
        first == "cd"
    };
    if !is_cd {
        return Ok(StandaloneCd::NotCd);
    }
    let mut target = remainder.trim();
    if cfg!(windows)
        && target
            .get(..2)
            .is_some_and(|flag| flag.eq_ignore_ascii_case("/d"))
    {
        let after_flag = &target[2..];
        if after_flag.is_empty() {
            target = "";
        } else if after_flag.starts_with(char::is_whitespace) {
            target = after_flag.trim();
        }
    }
    if target.is_empty() {
        return Ok(StandaloneCd::Change(String::new()));
    }

    let parsed = if target.starts_with('"') {
        if target.len() < 2 || !target.ends_with('"') {
            return Err(bad_request("standalone cd has an unterminated quoted path"));
        }
        let inner = &target[1..target.len() - 1];
        if inner.contains('"') {
            return Err(bad_request("standalone cd accepts exactly one quoted path"));
        }
        inner.to_string()
    } else {
        if target.chars().any(char::is_whitespace)
            || target.contains(['&', '|', ';', '<', '>', '"'])
        {
            return Err(bad_request(
                "standalone cd does not allow compound commands or unquoted spaces",
            ));
        }
        target.to_string()
    };
    Ok(StandaloneCd::Change(parsed))
}

async fn kill_process_tree(child: &mut tokio::process::Child) -> std::io::Result<()> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }

    #[cfg(windows)]
    let tree_killed = match child.id() {
        Some(pid) => Command::new("taskkill")
            .arg("/PID")
            .arg(pid.to_string())
            .arg("/T")
            .arg("/F")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_ok_and(|status| status.success()),
        None => false,
    };
    #[cfg(unix)]
    let tree_killed = match child.id() {
        Some(pid) => Command::new("kill")
            .arg("-KILL")
            .arg("--")
            .arg(format!("-{pid}"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_ok_and(|status| status.success()),
        None => false,
    };

    #[cfg(any(windows, unix))]
    {
        if tree_killed {
            child.wait().await?;
            Ok(())
        } else if child.try_wait()?.is_some() {
            Ok(())
        } else {
            child.kill().await
        }
    }
    #[cfg(not(any(windows, unix)))]
    {
        child.kill().await
    }
}

#[derive(Default)]
struct CapturedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

type SharedCapture = Arc<std::sync::Mutex<CapturedOutput>>;

async fn read_capped<R>(mut reader: R, capture: SharedCapture) -> std::io::Result<()>
where
    R: AsyncRead + Unpin,
{
    let mut buffer = [0u8; 8192];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            return Ok(());
        }
        let mut capture = capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let remaining = TERMINAL_OUTPUT_LIMIT.saturating_sub(capture.bytes.len());
        capture
            .bytes
            .extend_from_slice(&buffer[..read.min(remaining)]);
        if capture.bytes.len() == TERMINAL_OUTPUT_LIMIT || read > remaining {
            capture.truncated = true;
        }
    }
}

async fn collect_reader(
    task: &mut JoinHandle<std::io::Result<()>>,
    capture: &SharedCapture,
) -> Result<CapturedOutput, CondenserError> {
    match tokio::time::timeout(TERMINAL_READER_GRACE, &mut *task).await {
        Ok(result) => {
            result.map_err(|error| {
                std::io::Error::other(format!("terminal reader task failed: {error}"))
            })??;
            Ok(capture_snapshot(capture))
        }
        Err(_) => {
            task.abort();
            Ok(capture_snapshot(capture))
        }
    }
}

fn capture_snapshot(capture: &SharedCapture) -> CapturedOutput {
    let capture = capture
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    CapturedOutput {
        bytes: capture.bytes.clone(),
        truncated: capture.truncated,
    }
}

fn decode_terminal_output(bytes: Vec<u8>) -> String {
    match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) => String::from_utf8_lossy(error.as_bytes()).into_owned(),
    }
}

pub(crate) fn resolve_path(root: &Path, requested: &str) -> Result<PathBuf, CondenserError> {
    resolve_path_at(root, root, requested)
}

fn resolve_path_at(root: &Path, base: &Path, requested: &str) -> Result<PathBuf, CondenserError> {
    let candidate = candidate_path(base, requested);
    let canonical = candidate
        .canonicalize()
        .map_err(|error| bad_request(format!("invalid workspace path `{requested}`: {error}")))?;
    if !canonical.starts_with(root) {
        return Err(bad_request("workspace path escapes the service root"));
    }
    Ok(canonical)
}

fn candidate_path(base: &Path, requested: &str) -> PathBuf {
    let requested_path = Path::new(requested);
    if requested.is_empty() {
        base.to_path_buf()
    } else if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        base.join(requested_path)
    }
}

fn relative_path(root: &Path, path: &Path) -> Result<String, CondenserError> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| bad_request("workspace path escapes the service root"))?;
    Ok(relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/"))
}

fn should_skip_entry(name: &str) -> bool {
    matches!(
        name,
        ".git" | "target" | "node_modules" | ".preview" | ".condenser"
    )
}

fn modified_millis(modified: Option<SystemTime>) -> Option<u64> {
    modified
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64)
}

pub(crate) fn is_sensitive_file(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    if name == ".env" || name.starts_with(".env.") {
        return true;
    }
    if matches!(
        name.as_str(),
        "credentials"
            | "credentials.json"
            | "credential"
            | "credential.json"
            | ".credentials"
            | ".netrc"
            | "_netrc"
            | ".git-credentials"
            | ".npmrc"
            | ".pypirc"
            | ".pgpass"
            | ".htpasswd"
            | "id_rsa"
            | "id_dsa"
            | "id_ecdsa"
            | "id_ed25519"
            | "secring.gpg"
    ) {
        return true;
    }
    Path::new(&name)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension,
                "pem" | "key" | "p12" | "pfx" | "jks" | "keystore" | "kdbx"
            )
        })
}

fn bad_request(message: impl Into<String>) -> CondenserError {
    CondenserError::Config(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_paths_are_contained_and_root_relative() {
        let root = std::env::temp_dir().join(format!("condenser-workspace-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("nested")).unwrap();
        let root = root.canonicalize().unwrap();

        let nested = resolve_path(&root, "nested").unwrap();
        assert_eq!(relative_path(&root, &nested).unwrap(), "nested");
        assert_eq!(resolve_path_at(&root, &nested, "..").unwrap(), root);
        assert!(resolve_path(&root, "..").is_err());
        assert!(is_sensitive_file(".env.local"));
        assert!(!is_sensitive_file("notes.txt"));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn write_replaces_existing_text_without_creating_files() {
        let root =
            std::env::temp_dir().join(format!("condenser-write-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        std::fs::write(root.join("notes.txt"), "before").unwrap();

        let replacement = "after ✓";
        let response = write_text_file(&root, "notes.txt", replacement)
            .await
            .unwrap();
        assert_eq!(response.path, "notes.txt");
        assert_eq!(response.size, replacement.len() as u64);
        assert_eq!(
            std::fs::read_to_string(root.join("notes.txt")).unwrap(),
            replacement
        );
        assert!(
            write_text_file(&root, "missing.txt", "no creation")
                .await
                .is_err()
        );
        assert!(!root.join("missing.txt").exists());

        std::fs::write(root.join("binary.txt"), [0xff]).unwrap();
        assert!(write_text_file(&root, "binary.txt", "text").await.is_err());
        assert_eq!(std::fs::read(root.join("binary.txt")).unwrap(), [0xff]);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn lifecycle_creates_renames_and_soft_deletes_without_overwrite() {
        let root =
            std::env::temp_dir().join(format!("condenser-lifecycle-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::create_dir_all(root.join("b")).unwrap();
        let root = root.canonicalize().unwrap();

        let content = "lifecycle ✓";
        let created = create_text_file(&root, "a/note.txt", content)
            .await
            .unwrap();
        assert_eq!(created.path, "a/note.txt");
        assert_eq!(created.size, content.len() as u64);
        assert!(
            create_text_file(&root, "a/note.txt", "overwrite")
                .await
                .is_err()
        );

        let renamed = rename_workspace_file(&root, "a/note.txt", "b/moved.txt")
            .await
            .unwrap();
        assert_eq!(renamed.path, "b/moved.txt");
        assert!(!root.join("a/note.txt").exists());
        assert_eq!(
            std::fs::read_to_string(root.join("b/moved.txt")).unwrap(),
            content
        );

        create_text_file(&root, "a/empty.txt", "").await.unwrap();
        assert!(
            rename_workspace_file(&root, "a/empty.txt", "b/moved.txt")
                .await
                .is_err()
        );
        assert!(root.join("a/empty.txt").exists());

        let deleted = delete_workspace_file(&root, "b/moved.txt").await.unwrap();
        assert_eq!(deleted.path, "b/moved.txt");
        assert!(deleted.trash_path.starts_with(".condenser/trash/"));
        assert!(!root.join("b/moved.txt").exists());
        assert_eq!(
            std::fs::read_to_string(root.join(&deleted.trash_path)).unwrap(),
            content
        );
        let listing = list_files_blocking(&root, "").unwrap();
        assert!(
            listing
                .entries
                .iter()
                .all(|entry| entry.name != ".condenser")
        );

        std::fs::remove_dir_all(root).unwrap();
    }
}
