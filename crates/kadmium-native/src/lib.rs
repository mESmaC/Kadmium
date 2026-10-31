//! Native selection shared by both desktop shells. No HTTP dialog endpoint is exposed.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};

/// Runs before application JavaScript, restricted to the shell's local origin.
pub fn bridge_script(origin: &str) -> String {
    include_str!("bridge.js").replace("__KADMIUM_ORIGIN__", &json!(origin).to_string())
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    ModelFile,
    ModelFolder,
    WorkspaceFile,
}

/// A validated IPC request, dispatched on the window event loop (required by macOS).
#[derive(Debug, Deserialize)]
pub struct PickerRequest {
    id: String,
    kind: Kind,
    path: Option<PathBuf>,
    root: Option<PathBuf>,
}

/// Reject other origins, documents and malformed/unbounded messages before opening a dialog.
pub fn parse_request(origin: &str, source: &str, body: &str) -> Option<PickerRequest> {
    if body.len() > 32 * 1024 {
        return None;
    }
    let base = url::Url::parse(origin).ok()?;
    let source = url::Url::parse(source).ok()?;
    if base.scheme() != "http"
        || !matches!(base.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"))
        || base.origin() != source.origin()
        || !matches!(source.path(), "/" | "/condenser" | "/condenser/")
    {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    if value.get("cmd")?.as_str()? != "pickPath" {
        return None;
    }
    let request: PickerRequest = serde_json::from_value(value).ok()?;
    if request.id.is_empty()
        || request.id.len() > 64
        || !request
            .id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        || matches!(request.kind, Kind::WorkspaceFile) && request.root.is_none()
    {
        return None;
    }
    Some(request)
}

impl PickerRequest {
    /// Open an OS-owned, parented dialog. Cancel returns null; selection never imports a model.
    pub fn show(&self, parent: &(impl HasWindowHandle + HasDisplayHandle)) -> String {
        let mut dialog = rfd::FileDialog::new().set_parent(parent);
        if let Some(path) = self.path.as_ref().or(self.root.as_ref()) {
            let directory = if path.is_file() {
                path.parent()
            } else {
                Some(path.as_path())
            };
            if let Some(directory) = directory.filter(|p| p.is_dir()) {
                dialog = dialog.set_directory(directory);
            }
        }
        let selected = match self.kind {
            Kind::ModelFile => dialog
                .set_title("Select a model file")
                .add_filter("Model files", &["gguf", "safetensors"])
                .pick_file(),
            Kind::ModelFolder => dialog
                .set_title("Select a complete model folder")
                .pick_folder(),
            Kind::WorkspaceFile => dialog.set_title("Open a workspace file").pick_file(),
        };
        self.response(selected)
    }

    fn response(&self, selected: Option<PathBuf>) -> String {
        let result = selected.map(|path| self.selection(&path)).transpose();
        let response = match result {
            Ok(path) => json!({"id": self.id, "path": path}),
            Err(error) => json!({"id": self.id, "error": error}),
        };
        format!("window.__KADMIUM_FILES__?.complete({response});")
    }

    fn selection(&self, path: &Path) -> Result<String, String> {
        let path = match self.kind {
            // SafeTensors must keep config, tokenizer and sibling shards together.
            Kind::ModelFile
                if path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors")) =>
            {
                path.parent()
                    .ok_or("The model file has no parent folder.")?
                    .to_path_buf()
            }
            Kind::WorkspaceFile => {
                let root = self
                    .root
                    .as_ref()
                    .ok_or("The workspace is not ready.")?
                    .canonicalize()
                    .map_err(|e| format!("Cannot open the workspace: {e}"))?;
                let canonical = path
                    .canonicalize()
                    .map_err(|e| format!("Cannot open the selected file: {e}"))?;
                return canonical
                    .strip_prefix(root)
                    .map_err(|_| "Select a file inside the active workspace.".to_string())?
                    .to_str()
                    .map(|text| text.replace(std::path::MAIN_SEPARATOR, "/"))
                    .ok_or_else(|| "The selected path is not valid Unicode.".to_string());
            }
            _ => path.to_path_buf(),
        };
        path.to_str()
            .map(str::to_owned)
            .ok_or_else(|| "The selected path is not valid Unicode.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(kind: &str) -> PickerRequest {
        parse_request(
            "http://127.0.0.1:7856",
            "http://127.0.0.1:7856/",
            &json!({"cmd":"pickPath", "id":"test-1", "kind":kind}).to_string(),
        )
        .unwrap()
    }

    #[test]
    fn ipc_requires_local_application_origin_and_valid_requests() {
        let body = r#"{"cmd":"pickPath","id":"test-1","kind":"model_file"}"#;
        for source in [
            "https://example.com/",
            "http://127.0.0.1:9999/",
            "http://127.0.0.1:7856.evil.example/",
            "about:blank",
            "http://127.0.0.1:7856/plugins/untrusted",
        ] {
            assert!(parse_request("http://127.0.0.1:7856", source, body).is_none());
        }
        assert!(parse_request("https://example.com", "https://example.com/", body).is_none());
        assert!(parse_request("http://127.0.0.1:7856", "http://127.0.0.1:7856/", "{}").is_none());
        assert!(
            parse_request(
                "http://127.0.0.1:7848",
                "http://127.0.0.1:7848/condenser/?key=fixture",
                body
            )
            .is_some()
        );
        assert!(
            parse_request(
                "http://127.0.0.1:7856",
                "http://127.0.0.1:7856/",
                &body.replace("test-1", "bad'id")
            )
            .is_none()
        );
    }

    #[test]
    fn model_selection_preserves_complete_safetensors_packages_and_cancel() {
        let request = request("model_file");
        assert_eq!(
            request
                .selection(Path::new("models/model.safetensors"))
                .unwrap(),
            "models"
        );
        assert_eq!(
            request.selection(Path::new("models/model.gguf")).unwrap(),
            Path::new("models/model.gguf").to_str().unwrap()
        );
        assert!(request.response(None).contains("\"path\":null"));
        let escaped = request.response(Some(PathBuf::from("models/a\"b.gguf")));
        assert!(escaped.contains("a\\\"b.gguf"));
    }

    #[test]
    fn workspace_selection_rejects_files_outside_the_root() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let request = PickerRequest {
            id: "test-1".into(),
            kind: Kind::WorkspaceFile,
            path: None,
            root: Some(root.clone()),
        };
        assert_eq!(
            request.selection(&root.join("src/lib.rs")).unwrap(),
            "src/lib.rs"
        );
        assert!(request.selection(&root.join("../../Cargo.toml")).is_err());
    }
}
