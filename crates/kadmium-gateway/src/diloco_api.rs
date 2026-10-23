//! Stateful, shard-aligned DiLoCo sessions. Islands train locally and submit only at a
//! synchronization boundary. Checkpoints are explicitly exported/restored by the caller.

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    routing::{get, post},
};
use kadmium_runtime::{
    DiLoCoConfig, DiLoCoProgress, DiLoCoSession, DiLoCoSnapshot, DiLoCoStatus, IslandUpdate,
    ParameterShard, validate_diloco_id,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

const MAX_SESSIONS: usize = 16;
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
type Sessions = Arc<Mutex<BTreeMap<String, DiLoCoSession>>>;
type ApiError = (StatusCode, Json<serde_json::Value>);

/// A separate state store keeps reference training sessions independent of the legacy
/// cluster demo. The enclosing coordinator owns this router's lifetime.
pub fn router<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/api/v1/training/diloco/sessions", get(list).post(create))
        .route("/api/v1/training/diloco/restore", post(restore))
        .route(
            "/api/v1/training/diloco/sessions/{session_id}",
            get(status).delete(remove),
        )
        .route(
            "/api/v1/training/diloco/sessions/{session_id}/updates",
            post(submit),
        )
        .route(
            "/api/v1/training/diloco/sessions/{session_id}/snapshot",
            get(snapshot),
        )
        .route(
            "/api/v1/training/diloco/sessions/{session_id}/checkpoint",
            get(checkpoint),
        )
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(Arc::new(Mutex::new(
            BTreeMap::<String, DiLoCoSession>::new(),
        )))
}

fn error(code: StatusCode, message: impl Into<String>) -> ApiError {
    (code, Json(serde_json::json!({"error": message.into()})))
}

#[derive(Debug, Deserialize)]
pub struct CreateDiLoCoSession {
    pub session_id: String,
    #[serde(default)]
    pub config: DiLoCoConfig,
    pub island_ids: Vec<String>,
    pub shards: Vec<ParameterShard>,
}

#[derive(Debug, Serialize)]
pub struct DiLoCoSessionStatus {
    pub session_id: String,
    #[serde(flatten)]
    pub status: DiLoCoStatus,
}

fn insert(
    sessions: &Sessions,
    id: String,
    session: DiLoCoSession,
) -> Result<DiLoCoSessionStatus, ApiError> {
    let mut guard = sessions.lock().map_err(|_| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "training session lock unavailable",
        )
    })?;
    if guard.contains_key(&id) {
        return Err(error(
            StatusCode::CONFLICT,
            "session already exists; restore under a new session_id",
        ));
    }
    if guard.len() >= MAX_SESSIONS {
        return Err(error(
            StatusCode::CONFLICT,
            "session limit reached; release an unused session first",
        ));
    }
    let result = DiLoCoSessionStatus {
        session_id: id.clone(),
        status: session.status(),
    };
    guard.insert(id, session);
    Ok(result)
}

async fn create(
    State(sessions): State<Sessions>,
    Json(req): Json<CreateDiLoCoSession>,
) -> Result<(StatusCode, Json<DiLoCoSessionStatus>), ApiError> {
    validate_diloco_id(&req.session_id, "session")
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let session = DiLoCoSession::new(req.config, req.island_ids, req.shards)
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    Ok((
        StatusCode::CREATED,
        Json(insert(&sessions, req.session_id, session)?),
    ))
}

async fn list(
    State(sessions): State<Sessions>,
) -> Result<Json<Vec<DiLoCoSessionStatus>>, ApiError> {
    let guard = sessions.lock().map_err(|_| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "training session lock unavailable",
        )
    })?;
    Ok(Json(
        guard
            .iter()
            .map(|(id, session)| DiLoCoSessionStatus {
                session_id: id.clone(),
                status: session.status(),
            })
            .collect(),
    ))
}

async fn status(
    State(sessions): State<Sessions>,
    Path(id): Path<String>,
) -> Result<Json<DiLoCoSessionStatus>, ApiError> {
    let guard = sessions.lock().map_err(|_| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "training session lock unavailable",
        )
    })?;
    let session = guard
        .get(&id)
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "unknown training session"))?;
    Ok(Json(DiLoCoSessionStatus {
        session_id: id,
        status: session.status(),
    }))
}

async fn submit(
    State(sessions): State<Sessions>,
    Path(id): Path<String>,
    Json(update): Json<IslandUpdate>,
) -> Result<Json<DiLoCoProgress>, ApiError> {
    let mut guard = sessions.lock().map_err(|_| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "training session lock unavailable",
        )
    })?;
    let session = guard
        .get_mut(&id)
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "unknown training session"))?;
    // Version, shape and duplicate errors cannot change master parameters or momentum.
    session
        .submit(update)
        .map(Json)
        .map_err(|e| error(StatusCode::CONFLICT, e))
}

async fn snapshot(
    State(sessions): State<Sessions>,
    Path(id): Path<String>,
) -> Result<Json<DiLoCoSnapshot>, ApiError> {
    let guard = sessions.lock().map_err(|_| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "training session lock unavailable",
        )
    })?;
    let session = guard
        .get(&id)
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "unknown training session"))?;
    // An inference consumer sees only a complete committed round, never pending local weights.
    Ok(Json(session.snapshot()))
}

async fn checkpoint(
    State(sessions): State<Sessions>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let guard = sessions.lock().map_err(|_| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "training session lock unavailable",
        )
    })?;
    let session = guard
        .get(&id)
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "unknown training session"))?;
    let json = session
        .checkpoint_json()
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    serde_json::from_str(&json)
        .map(Json)
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

#[derive(Debug, Deserialize)]
pub struct RestoreDiLoCoSession {
    pub session_id: String,
    pub checkpoint: serde_json::Value,
}

async fn restore(
    State(sessions): State<Sessions>,
    Json(req): Json<RestoreDiLoCoSession>,
) -> Result<(StatusCode, Json<DiLoCoSessionStatus>), ApiError> {
    validate_diloco_id(&req.session_id, "session")
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let session = DiLoCoSession::from_checkpoint_json(&req.checkpoint.to_string())
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    Ok((
        StatusCode::CREATED,
        Json(insert(&sessions, req.session_id, session)?),
    ))
}

async fn remove(
    State(sessions): State<Sessions>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let mut guard = sessions.lock().map_err(|_| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "training session lock unavailable",
        )
    })?;
    guard
        .remove(&id)
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "unknown training session"))?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn request(
        addr: std::net::SocketAddr,
        method: &str,
        path: &str,
        body: Value,
    ) -> (u16, Value) {
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let body = body.to_string();
        let text = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(text.as_bytes()).await.unwrap();
        let mut bytes = Vec::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            stream.read_to_end(&mut bytes),
        )
        .await
        .unwrap()
        .unwrap();
        let response = String::from_utf8(bytes).unwrap();
        let (headers, body) = response.split_once("\r\n\r\n").unwrap();
        let code = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
        (code, serde_json::from_str(body).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn session_island_and_parameter_ids_share_the_http_contract() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app: Router = router();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let path = "/api/v1/training/diloco/sessions";
        for id in ["A_09-z".to_owned(), "x".repeat(96)] {
            let (code, _) = request(
                addr,
                "POST",
                path,
                json!({
                    "session_id":id, "island_ids":[id],
                    "shards":[{"shard_id":id,"values":[1.0]}]
                }),
            )
            .await;
            assert_eq!(code, 201);
        }
        for id in ["", " ", "a b", "a/b", "a.b", "é", "a\n", &"x".repeat(97)] {
            for field in ["session", "island", "shard"] {
                let mut body = json!({"session_id":"valid", "island_ids":["valid"],
                    "shards":[{"shard_id":"valid","values":[1.0]}]});
                match field {
                    "session" => body["session_id"] = id.into(),
                    "island" => body["island_ids"][0] = id.into(),
                    _ => body["shards"][0]["shard_id"] = id.into(),
                }
                let (code, _) = request(addr, "POST", path, body).await;
                assert_eq!(code, 400, "accepted invalid {field} ID {id:?}");
            }
        }
        let (_, sessions) = request(addr, "GET", path, Value::Null).await;
        assert_eq!(sessions.as_array().unwrap().len(), 2);
        server.abort();
    }

    #[tokio::test]
    async fn http_round_checkpoint_and_snapshot_are_consistent() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app: Router = router();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = "/api/v1/training/diloco";
        let (code, _) = request(addr, "POST", &format!("{base}/sessions"), json!({
            "session_id":"test", "config":{"inner_steps":2,"outer_learning_rate":0.5,"outer_momentum":0.5},
            "island_ids":["a","b"], "shards":[{"shard_id":"weight","values":[1.0,2.0]},{"shard_id":"bias","values":[0.0]}]
        })).await;
        assert_eq!(code, 201);
        let (_, before) = request(
            addr,
            "GET",
            &format!("{base}/sessions/test/snapshot"),
            Value::Null,
        )
        .await;
        let update = |island: &str| {
            json!({"island_id":island,"round":0,"baseline_id":before["baseline_id"],"local_steps":2,
            "shards":[{"shard_id":"weight","values":[0.6,1.6]},{"shard_id":"bias","values":[0.4]}]})
        };
        let (code, progress) = request(
            addr,
            "POST",
            &format!("{base}/sessions/test/updates"),
            update("a"),
        )
        .await;
        assert_eq!(code, 200);
        assert_eq!(progress["synchronized"], false);
        assert_eq!(
            request(
                addr,
                "GET",
                &format!("{base}/sessions/test/snapshot"),
                Value::Null
            )
            .await
            .1,
            before
        );
        assert_eq!(
            request(
                addr,
                "POST",
                &format!("{base}/sessions/test/updates"),
                update("a")
            )
            .await
            .0,
            409
        );

        let (_, saved) = request(
            addr,
            "GET",
            &format!("{base}/sessions/test/checkpoint"),
            Value::Null,
        )
        .await;
        assert_eq!(
            request(
                addr,
                "POST",
                &format!("{base}/restore"),
                json!({"session_id":"resumed","checkpoint":saved})
            )
            .await
            .0,
            201
        );
        for id in ["test", "resumed"] {
            let (code, progress) = request(
                addr,
                "POST",
                &format!("{base}/sessions/{id}/updates"),
                update("b"),
            )
            .await;
            assert_eq!(code, 200);
            assert_eq!(progress["synchronized"], true);
            assert_eq!(progress["round"], 1);
        }
        let (_, after) = request(
            addr,
            "GET",
            &format!("{base}/sessions/test/snapshot"),
            Value::Null,
        )
        .await;
        let (_, resumed) = request(
            addr,
            "GET",
            &format!("{base}/sessions/resumed/snapshot"),
            Value::Null,
        )
        .await;
        assert_eq!(after, resumed);
        assert_ne!(after["baseline_id"], before["baseline_id"]);
        let shards: Vec<ParameterShard> = serde_json::from_value(after["shards"].clone()).unwrap();
        let weight = shards.iter().find(|s| s.shard_id == "weight").unwrap();
        assert!((weight.values[0] - 0.7).abs() < 1e-6);
        assert!((weight.values[1] - 1.7).abs() < 1e-6);
        let (_, round_one) = request(
            addr,
            "GET",
            &format!("{base}/sessions/test/checkpoint"),
            Value::Null,
        )
        .await;
        assert_eq!(
            request(
                addr,
                "POST",
                &format!("{base}/restore"),
                json!({"session_id":"round-one","checkpoint":round_one})
            )
            .await
            .0,
            201
        );
        for id in ["test", "round-one"] {
            for island in ["a", "b"] {
                let (code, _) = request(addr, "POST", &format!("{base}/sessions/{id}/updates"), json!({
                    "island_id":island, "round":1, "baseline_id":after["baseline_id"], "local_steps":2,
                    "shards":[{"shard_id":"bias","values":[0.5]},{"shard_id":"weight","values":[0.5,1.5]}]
                })).await;
                assert_eq!(code, 200);
            }
        }
        let (_, second) = request(
            addr,
            "GET",
            &format!("{base}/sessions/test/snapshot"),
            Value::Null,
        )
        .await;
        assert_eq!(
            request(
                addr,
                "GET",
                &format!("{base}/sessions/round-one/snapshot"),
                Value::Null
            )
            .await
            .1,
            second
        );
        let shards: Vec<ParameterShard> = serde_json::from_value(second["shards"].clone()).unwrap();
        let weight = shards.iter().find(|s| s.shard_id == "weight").unwrap();
        // The retained velocity is .4; resetting momentum would instead produce .55.
        assert!((weight.values[0] - 0.5).abs() < 1e-6);
        assert_eq!(second["round"], 2);
        assert_eq!(
            request(
                addr,
                "POST",
                &format!("{base}/sessions/test/updates"),
                update("b")
            )
            .await
            .0,
            409
        );
        assert_eq!(
            request(
                addr,
                "POST",
                &format!("{base}/restore"),
                json!({"session_id":"test","checkpoint":saved})
            )
            .await
            .0,
            409
        );
        assert_eq!(
            request(
                addr,
                "POST",
                &format!("{base}/restore"),
                json!({"session_id":"bad","checkpoint":{}})
            )
            .await
            .0,
            400
        );
        assert_eq!(
            request(
                addr,
                "DELETE",
                &format!("{base}/sessions/resumed"),
                Value::Null
            )
            .await
            .0,
            204
        );
        assert_eq!(
            request(
                addr,
                "GET",
                &format!("{base}/sessions/resumed"),
                Value::Null
            )
            .await
            .0,
            404
        );
        server.abort();
    }
}
