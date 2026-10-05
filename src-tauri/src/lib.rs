mod cloud_providers;
mod local_ai;
mod search_index;
mod speech;

use local_ai::LocalAiManager;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path, sync::Mutex};
use tauri::{path::BaseDirectory, AppHandle, Manager, State};

const INITIAL_MIGRATION: &str = include_str!("../migrations/001_initial.sql");
const KEYRING_SERVICE: &str = "com.theaiguide.desktop";

struct AppState {
    db: Mutex<Connection>,
    search: Mutex<Connection>,
    local_ai: LocalAiManager,
    speech: speech::SpeechEngine,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProgressRecord {
    module_id: String,
    current_step: i64,
    checkpoint_score: f64,
    completed: bool,
    updated_at: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SnapshotInput {
    id: String,
    module_id: String,
    name: String,
    state_json: String,
    state_version: i64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SnapshotRecord {
    id: String,
    module_id: String,
    name: String,
    state_json: String,
    state_version: i64,
    created_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatSessionRecord {
    id: String,
    module_id: String,
    backend: String,
    title: String,
    created_at: Option<String>,
    updated_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatMessageRecord {
    id: String,
    session_id: String,
    role: String,
    content: String,
    created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LocalModelRecord {
    id: String,
    display_name: String,
    file_path: String,
    bytes: i64,
    sha256: String,
    context_length: i64,
    status: String,
    updated_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProviderConfigRecord {
    provider: String,
    base_url: Option<String>,
    model: Option<String>,
    enabled: bool,
    updated_at: Option<String>,
}

fn initialize_database(connection: &Connection) -> rusqlite::Result<()> {
    connection.execute_batch(INITIAL_MIGRATION)
}

fn open_database(path: &Path) -> rusqlite::Result<Connection> {
    let connection = Connection::open(path)?;
    connection.pragma_update(None, "journal_mode", "WAL")?;
    connection.pragma_update(None, "foreign_keys", true)?;
    connection.busy_timeout(std::time::Duration::from_secs(3))?;
    initialize_database(&connection)?;
    Ok(connection)
}

const PACKAGED_SEARCH_INDEX: &str = include_str!("../../src/search/content-index.json");

fn open_search_database(resource_root: Option<&Path>) -> Result<Connection, String> {
    let candidates = resource_root
        .map(|root| {
            vec![
                root.join("content-index.sqlite"),
                root.join("resources/content-index.sqlite"),
            ]
        })
        .unwrap_or_default();

    for path in candidates {
        if path.exists() {
            return Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .map_err(|error| error.to_string());
        }
    }

    let connection = Connection::open_in_memory().map_err(|error| error.to_string())?;
    search_index::populate_from_json(&connection, PACKAGED_SEARCH_INDEX)?;
    Ok(connection)
}

#[tauri::command]
fn search_content(
    state: State<'_, AppState>,
    query: String,
    limit: Option<i64>,
) -> Result<Vec<search_index::SearchHit>, String> {
    let db = state
        .search
        .lock()
        .map_err(|_| "Search index lock poisoned".to_string())?;
    search_index::search_documents(&db, &query, limit.unwrap_or(32).clamp(1, 80))
}

#[tauri::command]
fn get_setting(state: State<'_, AppState>, key: String) -> Result<Option<String>, String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.query_row(
        "SELECT value FROM settings WHERE key = ?1",
        params![key],
        |row| row.get(0),
    )
    .optional()
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn set_setting(state: State<'_, AppState>, key: String, value: String) -> Result<(), String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.execute(
        "INSERT INTO settings (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET
           value = excluded.value,
           updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        params![key, value],
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn list_module_progress(state: State<'_, AppState>) -> Result<Vec<ProgressRecord>, String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    let mut statement = db
        .prepare(
            "SELECT module_id, current_step, checkpoint_score, completed, updated_at
             FROM module_progress ORDER BY module_id",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok(ProgressRecord {
                module_id: row.get(0)?,
                current_step: row.get(1)?,
                checkpoint_score: row.get(2)?,
                completed: row.get(3)?,
                updated_at: row.get(4)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn upsert_module_progress(
    state: State<'_, AppState>,
    progress: ProgressRecord,
) -> Result<(), String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.execute(
        "INSERT INTO module_progress (module_id, current_step, checkpoint_score, completed)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(module_id) DO UPDATE SET
           current_step = excluded.current_step,
           checkpoint_score = excluded.checkpoint_score,
           completed = excluded.completed,
           updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        params![
            progress.module_id,
            progress.current_step,
            progress.checkpoint_score,
            progress.completed
        ],
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn create_snapshot(state: State<'_, AppState>, snapshot: SnapshotInput) -> Result<(), String> {
    serde_json::from_str::<serde_json::Value>(&snapshot.state_json)
        .map_err(|_| "Snapshot state must be valid JSON".to_string())?;
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.execute(
        "INSERT INTO snapshots (id, module_id, name, state_json, state_version)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            snapshot.id,
            snapshot.module_id,
            snapshot.name,
            snapshot.state_json,
            snapshot.state_version
        ],
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn list_snapshots(
    state: State<'_, AppState>,
    module_id: Option<String>,
) -> Result<Vec<SnapshotRecord>, String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    let sql = if module_id.is_some() {
        "SELECT id, module_id, name, state_json, state_version, created_at
         FROM snapshots WHERE module_id = ?1 ORDER BY created_at DESC"
    } else {
        "SELECT id, module_id, name, state_json, state_version, created_at
         FROM snapshots ORDER BY created_at DESC"
    };
    let mut statement = db.prepare(sql).map_err(|error| error.to_string())?;
    let map_row = |row: &rusqlite::Row<'_>| {
        Ok(SnapshotRecord {
            id: row.get(0)?,
            module_id: row.get(1)?,
            name: row.get(2)?,
            state_json: row.get(3)?,
            state_version: row.get(4)?,
            created_at: row.get(5)?,
        })
    };
    let records = match module_id {
        Some(id) => statement
            .query_map(params![id], map_row)
            .map_err(|error| error.to_string())?
            .collect::<rusqlite::Result<Vec<_>>>(),
        None => statement
            .query_map([], map_row)
            .map_err(|error| error.to_string())?
            .collect::<rusqlite::Result<Vec<_>>>(),
    };
    records.map_err(|error| error.to_string())
}

#[tauri::command]
fn delete_snapshot(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.execute("DELETE FROM snapshots WHERE id = ?1", params![id])
        .map(|_| ())
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn upsert_chat_session(
    state: State<'_, AppState>,
    session: ChatSessionRecord,
) -> Result<(), String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.execute(
        "INSERT INTO chat_sessions (id, module_id, backend, title)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(id) DO UPDATE SET
           backend = excluded.backend,
           title = excluded.title,
           updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        params![
            session.id,
            session.module_id,
            session.backend,
            session.title
        ],
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn append_chat_message(
    state: State<'_, AppState>,
    message: ChatMessageRecord,
) -> Result<(), String> {
    if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
        return Err("Chat role must be system, user, or assistant".to_string());
    }
    let mut db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    let transaction = db.transaction().map_err(|error| error.to_string())?;
    transaction
        .execute(
            "INSERT INTO chat_messages (id, session_id, role, content)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                message.id,
                message.session_id,
                message.role,
                message.content
            ],
        )
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "UPDATE chat_sessions
             SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?1",
            params![message.session_id],
        )
        .map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())
}

#[tauri::command]
fn list_chat_messages(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<Vec<ChatMessageRecord>, String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    let mut statement = db
        .prepare(
            "SELECT id, session_id, role, content, created_at
             FROM chat_messages WHERE session_id = ?1 ORDER BY created_at",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map(params![session_id], |row| {
            Ok(ChatMessageRecord {
                id: row.get(0)?,
                session_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                created_at: row.get(4)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn delete_chat_session(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.execute("DELETE FROM chat_sessions WHERE id = ?1", params![id])
        .map(|_| ())
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn list_local_models(state: State<'_, AppState>) -> Result<Vec<LocalModelRecord>, String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    let mut statement = db
        .prepare(
            "SELECT id, display_name, file_path, bytes, sha256, context_length, status, updated_at
             FROM local_models ORDER BY display_name",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok(LocalModelRecord {
                id: row.get(0)?,
                display_name: row.get(1)?,
                file_path: row.get(2)?,
                bytes: row.get(3)?,
                sha256: row.get(4)?,
                context_length: row.get(5)?,
                status: row.get(6)?,
                updated_at: row.get(7)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn upsert_local_model(state: State<'_, AppState>, model: LocalModelRecord) -> Result<(), String> {
    if !matches!(
        model.status.as_str(),
        "downloading" | "ready" | "invalid" | "missing"
    ) {
        return Err("Invalid local model status".to_string());
    }
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.execute(
        "INSERT INTO local_models
           (id, display_name, file_path, bytes, sha256, context_length, status)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(id) DO UPDATE SET
           display_name = excluded.display_name,
           file_path = excluded.file_path,
           bytes = excluded.bytes,
           sha256 = excluded.sha256,
           context_length = excluded.context_length,
           status = excluded.status,
           updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        params![
            model.id,
            model.display_name,
            model.file_path,
            model.bytes,
            model.sha256,
            model.context_length,
            model.status
        ],
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

fn read_provider_secret(provider: &str) -> Result<String, String> {
    let entry =
        keyring::Entry::new(KEYRING_SERVICE, provider).map_err(|error| error.to_string())?;
    match entry.get_password() {
        Ok(secret) => Ok(secret),
        Err(keyring::Error::NoEntry) => Err("No key is stored for this provider.".to_string()),
        Err(error) => Err(error.to_string()),
    }
}

fn map_provider_config(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProviderConfigRecord> {
    Ok(ProviderConfigRecord {
        provider: row.get(0)?,
        base_url: row.get(1)?,
        model: row.get(2)?,
        enabled: row.get::<_, i64>(3)? != 0,
        updated_at: row.get(4)?,
    })
}

#[tauri::command]
fn get_provider_config(
    state: State<'_, AppState>,
    provider: String,
) -> Result<Option<ProviderConfigRecord>, String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.query_row(
        "SELECT provider, base_url, model, enabled, updated_at
         FROM provider_config WHERE provider = ?1",
        params![provider],
        map_provider_config,
    )
    .optional()
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn list_provider_configs(
    state: State<'_, AppState>,
) -> Result<Vec<ProviderConfigRecord>, String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    let mut statement = db
        .prepare(
            "SELECT provider, base_url, model, enabled, updated_at
             FROM provider_config ORDER BY provider",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], map_provider_config)
        .map_err(|error| error.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn test_provider_connection(
    request: cloud_providers::ProviderTestRequest,
) -> Result<cloud_providers::ProviderTestResult, String> {
    let mut request = request;
    if request.secret.as_deref().unwrap_or("").trim().is_empty() {
        request.secret = Some(read_provider_secret(&request.provider)?);
    }
    cloud_providers::test_connection(request).await
}

#[tauri::command]
async fn cloud_chat(
    app: AppHandle,
    state: State<'_, AppState>,
    mut request: cloud_providers::CloudChatRequest,
) -> Result<String, String> {
    if request.message.trim().is_empty() {
        return Err("Message cannot be empty".to_string());
    }
    if request.message.chars().count() > 8_000 {
        return Err("Message is too long".to_string());
    }
    let secret = read_provider_secret(&request.provider)?;
    let stored = {
        let db = state
            .db
            .lock()
            .map_err(|_| "Database lock poisoned".to_string())?;
        db.query_row(
            "SELECT provider, base_url, model, enabled, updated_at
             FROM provider_config WHERE provider = ?1",
            params![request.provider],
            map_provider_config,
        )
        .optional()
        .map_err(|error| error.to_string())?
    };
    if request.model.as_deref().unwrap_or("").trim().is_empty() {
        request.model = stored.as_ref().and_then(|row| row.model.clone());
    }
    if request.base_url.as_deref().unwrap_or("").trim().is_empty() {
        request.base_url = stored.as_ref().and_then(|row| row.base_url.clone());
    }
    cloud_providers::chat(app, request, secret).await
}

#[tauri::command]
fn upsert_provider_config(
    state: State<'_, AppState>,
    config: ProviderConfigRecord,
) -> Result<(), String> {
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.execute(
        "INSERT INTO provider_config (provider, base_url, model, enabled)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(provider) DO UPDATE SET
           base_url = excluded.base_url,
           model = excluded.model,
           enabled = excluded.enabled,
           updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        params![
            config.provider,
            config.base_url,
            config.model,
            config.enabled
        ],
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn set_provider_secret(provider: String, secret: String) -> Result<(), String> {
    if provider.trim().is_empty() || secret.trim().is_empty() {
        return Err("Provider and secret are required".to_string());
    }
    let entry =
        keyring::Entry::new(KEYRING_SERVICE, &provider).map_err(|error| error.to_string())?;
    entry
        .set_password(&secret)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn has_provider_secret(provider: String) -> Result<bool, String> {
    let entry =
        keyring::Entry::new(KEYRING_SERVICE, &provider).map_err(|error| error.to_string())?;
    match entry.get_password() {
        Ok(_) => Ok(true),
        Err(keyring::Error::NoEntry) => Ok(false),
        Err(error) => Err(error.to_string()),
    }
}

#[tauri::command]
fn delete_provider_secret(provider: String) -> Result<(), String> {
    let entry =
        keyring::Entry::new(KEYRING_SERVICE, &provider).map_err(|error| error.to_string())?;
    match entry.delete_credential() {
        Ok(_) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[tauri::command]
async fn speak_text(state: State<'_, AppState>, text: String) -> Result<(), String> {
    let engine = state.speech.clone();
    tokio::task::spawn_blocking(move || engine.speak(text))
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
fn stop_speaking(state: State<'_, AppState>) -> Result<(), String> {
    state.speech.stop();
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            fs::create_dir_all(&data_dir)?;
            let database = open_database(&data_dir.join("guide.sqlite3"))?;
            let resource_dir = app.path().resolve("", BaseDirectory::Resource).ok();
            let search = open_search_database(resource_dir.as_deref())?;
            app.manage(AppState {
                db: Mutex::new(database),
                search: Mutex::new(search),
                local_ai: LocalAiManager::new(data_dir.join("local-ai")),
                speech: speech::SpeechEngine::new(),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_setting,
            set_setting,
            search_content,
            list_module_progress,
            upsert_module_progress,
            create_snapshot,
            list_snapshots,
            delete_snapshot,
            upsert_chat_session,
            append_chat_message,
            list_chat_messages,
            delete_chat_session,
            list_local_models,
            upsert_local_model,
            upsert_provider_config,
            get_provider_config,
            list_provider_configs,
            test_provider_connection,
            cloud_chat,
            speak_text,
            stop_speaking,
            set_provider_secret,
            has_provider_secret,
            delete_provider_secret,
            local_ai::local_ai_status,
            local_ai::install_local_ai,
            local_ai::cancel_local_ai_install,
            local_ai::start_local_ai,
            local_ai::stop_local_ai,
            local_ai::remove_local_ai,
            local_ai::local_chat
        ])
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                let state = window.state::<AppState>();
                state.local_ai.stop();
                state.speech.stop();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Discover AI");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_creates_every_domain_table() {
        let connection = Connection::open_in_memory().unwrap();
        initialize_database(&connection).unwrap();
        let expected = [
            "settings",
            "module_progress",
            "snapshots",
            "chat_sessions",
            "chat_messages",
            "local_models",
            "provider_config",
        ];

        for table in expected {
            let exists: bool = connection
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
                    params![table],
                    |row| row.get(0),
                )
                .unwrap();
            assert!(exists, "{table} should exist after migration");
        }
    }

    #[test]
    fn progress_upsert_keeps_module_ids_open_ended() {
        let connection = Connection::open_in_memory().unwrap();
        initialize_database(&connection).unwrap();
        connection
            .execute(
                "INSERT INTO module_progress (module_id, current_step, checkpoint_score, completed)
                 VALUES (?1, 2, 1, 1)",
                params!["third-party-module"],
            )
            .unwrap();

        let id: String = connection
            .query_row("SELECT module_id FROM module_progress", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(id, "third-party-module");
    }

    #[test]
    fn packaged_search_index_finds_attention_and_settings() {
        let connection = Connection::open_in_memory().unwrap();
        let count = search_index::populate_from_json(&connection, PACKAGED_SEARCH_INDEX).unwrap();
        assert!(count >= 800, "expected a full packaged index, found {count}");

        let hits = search_index::search_documents(&connection, "attention softmax", 8).unwrap();
        assert!(
            hits.iter()
                .any(|hit| hit.module_slug.as_deref() == Some("attention")),
            "attention should rank for a core lesson query"
        );

        let settings = search_index::search_documents(&connection, "keychain", 8).unwrap();
        assert!(
            settings
                .iter()
                .any(|hit| hit.settings_tab.as_deref() == Some("cloud")),
            "settings destinations should be searchable"
        );
    }
}
