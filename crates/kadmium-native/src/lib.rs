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

/// A trusted picker message either opens a dialog or completes with a validation error.
#[derive(Debug)]
pub enum PickerMessage {
    Request(PickerRequest),
    Rejected { id: String, error: &'static str },
}

impl PickerMessage {
    pub fn show(&self, parent: &(impl HasWindowHandle + HasDisplayHandle)) -> String {
        match self {
            Self::Request(request) => request.show(parent),
            Self::Rejected { id, error } => completion(json!({"id": id, "error": error})),
        }
    }
}

/// Reject other origins, documents and malformed/unbounded messages before opening a dialog.
pub fn parse_request(origin: &str, source: &str, body: &str) -> Option<PickerMessage> {
    if body.len() > 32 * 1024 {
        return None;
    }
    let base = url::Url::parse(origin).ok()?;
    let source = url::Url::parse(source).ok()?;
    if base.scheme() != "http"
        || !matches!(base.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"))
        || base.origin() != source.origin()
    {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    if value.get("cmd")?.as_str()? != "pickPath" {
        return None;
    }
    let id = value.get("id")?.as_str()?;
    if id.is_empty() || id.len() > 64 || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return None;
    }
    let id = id.to_string();
    let reject = |error| {
        Some(PickerMessage::Rejected {
            id: id.clone(),
            error,
        })
    };
    if !matches!(source.path(), "/" | "/condenser" | "/condenser/") {
        return reject("File dialogs are only available from the Condenser workspace.");
    }
    let Ok(request) = serde_json::from_value::<PickerRequest>(value) else {
        return reject(
            "Invalid file dialog request. Choose a model file, model folder, or workspace file.",
        );
    };
    if matches!(request.kind, Kind::WorkspaceFile) && request.root.is_none() {
        return reject("The workspace is not ready. Refresh and try again.");
    }
    Some(PickerMessage::Request(request))
}

fn completion(response: serde_json::Value) -> String {
    format!("window.__KADMIUM_FILES__?.complete({response});")
}

fn initial_directory(path: &Path) -> Option<PathBuf> {
    // Preserve real paths containing quote characters; otherwise accept a pasted shell path.
    let path = if path.exists() {
        path
    } else if let Some(text) = path.to_str() {
        let unquoted = text
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .or_else(|| text.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')));
        unquoted.map(Path::new).unwrap_or(path)
    } else {
        path
    };
    let directory = if path.is_file() { path.parent()? } else { path };
    directory.is_dir().then(|| directory.to_path_buf())
}

impl PickerRequest {
    /// Open an OS-owned, parented dialog. Cancel returns null; selection never imports a model.
    pub fn show(&self, parent: &(impl HasWindowHandle + HasDisplayHandle)) -> String {
        let mut dialog = rfd::FileDialog::new().set_parent(parent);
        if let Some(directory) = self
            .path
            .as_ref()
            .or(self.root.as_ref())
            .and_then(|path| initial_directory(path))
        {
            dialog = dialog.set_directory(directory);
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
        completion(response)
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
        let message = parse_request(
            "http://127.0.0.1:7856",
            "http://127.0.0.1:7856/",
            &json!({"cmd":"pickPath", "id":"test-1", "kind":kind}).to_string(),
        )
        .unwrap();
        let PickerMessage::Request(request) = message else {
            panic!("expected a valid picker request")
        };
        request
    }

    #[test]
    fn ipc_requires_local_application_origin_and_valid_requests() {
        let body = r#"{"cmd":"pickPath","id":"test-1","kind":"model_file"}"#;
        for source in [
            "https://example.com/",
            "http://127.0.0.1:9999/",
            "http://127.0.0.1:7856.evil.example/",
            "about:blank",
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
    fn trusted_invalid_requests_complete_without_opening_a_dialog() {
        struct NoWindow;
        impl HasWindowHandle for NoWindow {
            fn window_handle(
                &self,
            ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError>
            {
                panic!("rejected requests must not open a dialog")
            }
        }
        impl HasDisplayHandle for NoWindow {
            fn display_handle(
                &self,
            ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError>
            {
                panic!("rejected requests must not open a dialog")
            }
        }
        let origin = "http://127.0.0.1:7856";
        for (source, fields) in [
            ("/", json!({"kind": "unknown"})),
            ("/", json!({"kind": "workspace_file"})),
            ("/", json!({"kind": "model_file", "path": 123})),
            ("/plugins/untrusted", json!({"kind": "model_file"})),
        ] {
            let mut body = fields;
            body["cmd"] = json!("pickPath");
            body["id"] = json!("retry-1");
            let message =
                parse_request(origin, &format!("{origin}{source}"), &body.to_string()).unwrap();
            assert!(matches!(message, PickerMessage::Rejected { .. }));
            let script = message.show(&NoWindow);
            let response: serde_json::Value = serde_json::from_str(
                script
                    .strip_prefix("window.__KADMIUM_FILES__?.complete(")
                    .unwrap()
                    .strip_suffix(");")
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(response["id"], "retry-1");
            assert!(
                response["error"]
                    .as_str()
                    .is_some_and(|error| !error.is_empty())
            );
            assert!(response.get("path").is_none());
        }
    }

    #[test]
    fn quoted_initial_paths_find_the_same_directory_without_expanding_shell_text() {
        let fixture =
            std::env::temp_dir().join(format!("kadmium-picker-quotes-{}", std::process::id()));
        let directory = fixture.join("model folder");
        std::fs::create_dir_all(&directory).unwrap();
        let file = directory.join("model.gguf");
        std::fs::write(&file, b"picker fixture").unwrap();
        for path in [&file, &directory] {
            for quote in ['"', '\''] {
                let quoted = PathBuf::from(format!("{quote}{}{quote}", path.display()));
                assert_eq!(initial_directory(&quoted), Some(directory.clone()));
            }
            assert_eq!(initial_directory(path), Some(directory.clone()));
        }
        assert_eq!(initial_directory(Path::new("\"$(not-a-command)\"")), None);
        assert_eq!(
            initial_directory(&PathBuf::from(format!("\"{}'", file.display()))),
            None
        );
        std::fs::remove_dir_all(fixture).unwrap();
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
