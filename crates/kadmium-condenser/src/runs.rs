//! Request-scoped run identities. A handle owns its registry entry so errors,
//! disconnections and completed turns all release the identity.

use crate::{CondenserError, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Default)]
pub(crate) struct ActiveRuns(Arc<Mutex<HashMap<String, CancellationToken>>>);

impl ActiveRuns {
    pub(crate) fn register(&self, id: Option<String>) -> Result<RunHandle> {
        let token = CancellationToken::new();
        if let Some(id) = &id {
            if id.is_empty()
                || id.len() > 80
                || !id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_'))
            {
                return Err(CondenserError::Config(
                    "run_id must contain 1–80 letters, digits, hyphens or underscores".into(),
                ));
            }
            let mut runs = self.0.lock().expect("run registry lock");
            if runs.contains_key(id) {
                return Err(CondenserError::Config("run_id is already active".into()));
            }
            runs.insert(id.clone(), token.clone());
        }
        Ok(RunHandle {
            registry: self.clone(),
            id,
            token,
        })
    }

    pub(crate) fn cancel(&self, id: &str) -> bool {
        let runs = self.0.lock().expect("run registry lock");
        if let Some(token) = runs.get(id) {
            token.cancel();
            true
        } else {
            false
        }
    }
}

pub(crate) struct RunHandle {
    registry: ActiveRuns,
    id: Option<String>,
    pub(crate) token: CancellationToken,
}

impl Drop for RunHandle {
    fn drop(&mut self) {
        self.token.cancel();
        if let Some(id) = &self.id {
            self.registry
                .0
                .lock()
                .expect("run registry lock")
                .remove(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_is_scoped_and_identity_is_released_on_drop() {
        let runs = ActiveRuns::default();
        let first = runs.register(Some("first".into())).unwrap();
        let second = runs.register(Some("second".into())).unwrap();
        assert!(runs.register(Some("first".into())).is_err());
        assert!(runs.cancel("first"));
        assert!(first.token.is_cancelled());
        assert!(!second.token.is_cancelled());
        drop(first);
        assert!(!runs.cancel("first"));
        assert!(runs.register(Some("first".into())).is_ok());
        assert!(runs.register(Some("../bad".into())).is_err());
    }
}
