use crate::AppState;
use flate2::read::GzDecoder;
use futures_util::StreamExt;
use reqwest::{header::RANGE, Client, StatusCode};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};
use tauri::{AppHandle, Emitter, State};

const ENGINE_VERSION: &str = "b10991";
const MODEL_ID: &str = "qwen3.5-2b-q4-k-m";
const MODEL_NAME: &str = "Qwen3.5 2B · Q4_K_M";
const MODEL_FILE: &str = "Qwen3.5-2B-Q4_K_M.gguf";
const MODEL_BYTES: u64 = 1_280_835_840;
const MODEL_SHA256: &str = "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223";
const MODEL_URL: &str = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_K_M.gguf?download=true";
const DOWNLOAD_HEADROOM: u64 = 256 * 1024 * 1024;

#[derive(Clone, Copy)]
struct EngineSpec {
    file_name: &'static str,
    url: &'static str,
    sha256: &'static str,
    bytes: u64,
    archive: &'static str,
}

fn engine_spec() -> Result<EngineSpec, String> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        return Ok(EngineSpec {
            file_name: "llama-b10991-bin-macos-arm64.tar.gz",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b10991/llama-b10991-bin-macos-arm64.tar.gz",
            sha256: "8e91ffb9e150d36035272b9b86e915b48c84b4488faae33dcdfe07d33c4c2b3f",
            bytes: 11_150_256,
            archive: "tar.gz",
        });
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        return Ok(EngineSpec {
            file_name: "llama-b10991-bin-macos-x64.tar.gz",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b10991/llama-b10991-bin-macos-x64.tar.gz",
            sha256: "769b60fc4f828a3389da11039d35512af82acbfb41a2208dbf3b6f4a2ce9ea38",
            bytes: 11_200_128,
            archive: "tar.gz",
        });
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        return Ok(EngineSpec {
            file_name: "llama-b10991-bin-ubuntu-arm64.tar.gz",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b10991/llama-b10991-bin-ubuntu-arm64.tar.gz",
            sha256: "8845b769c3eb4cd09fed133ed1f10ab6534435b0b16666436c3e7b8263eed537",
            bytes: 13_469_041,
            archive: "tar.gz",
        });
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        return Ok(EngineSpec {
            file_name: "llama-b10991-bin-ubuntu-x64.tar.gz",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b10991/llama-b10991-bin-ubuntu-x64.tar.gz",
            sha256: "9058f5d0e9a06939b0a474cc1940c435d45f1246d67e26ba5fdd49e36fca9f86",
            bytes: 16_845_602,
            archive: "tar.gz",
        });
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        return Ok(EngineSpec {
            file_name: "llama-b10991-bin-win-cpu-arm64.zip",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b10991/llama-b10991-bin-win-cpu-arm64.zip",
            sha256: "b19bb14a6a4f80bdeac7ff9379c0d2030642e6dc9b7aea75547e8e1dd3a98575",
            bytes: 11_997_450,
            archive: "zip",
        });
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        return Ok(EngineSpec {
            file_name: "llama-b10991-bin-win-cpu-x64.zip",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b10991/llama-b10991-bin-win-cpu-x64.zip",
            sha256: "8f1b7bcc1df032bb0c5759e3db5c67969f20c6f061aa0efec70de7bfa00fbf37",
            bytes: 18_428_726,
            archive: "zip",
        });
    }

    #[allow(unreachable_code)]
    Err(format!(
        "llama.cpp setup is not available for {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    ))
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LocalAiStatus {
    state: String,
    engine: String,
    engine_version: String,
    platform: String,
    model_id: String,
    model_name: String,
    model_bytes: u64,
    model_sha256: String,
    engine_installed: bool,
    model_installed: bool,
    server_running: bool,
    port: Option<u16>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalAiProgress {
    stage: String,
    downloaded_bytes: u64,
    total_bytes: u64,
    percent: f64,
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LocalChatRequest {
    request_id: String,
    module_title: String,
    mode: String,
    objectives: Vec<String>,
    current_step: String,
    state_json: String,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalChatChunk {
    request_id: String,
    content: String,
    done: bool,
}

pub(crate) struct LocalAiManager {
    root: PathBuf,
    process: Mutex<Option<Child>>,
    port: Mutex<Option<u16>>,
    installing: AtomicBool,
    starting: AtomicBool,
    cancel_install: AtomicBool,
    last_error: Mutex<Option<String>>,
}

impl LocalAiManager {
    pub(crate) fn new(root: PathBuf) -> Self {
        Self {
            root,
            process: Mutex::new(None),
            port: Mutex::new(None),
            installing: AtomicBool::new(false),
            starting: AtomicBool::new(false),
            cancel_install: AtomicBool::new(false),
            last_error: Mutex::new(None),
        }
    }

    fn engine_dir(&self) -> PathBuf {
        self.root.join("engine").join(ENGINE_VERSION)
    }

    fn model_path(&self) -> PathBuf {
        self.root.join("models").join(MODEL_FILE)
    }

    fn model_marker_path(&self) -> PathBuf {
        self.root
            .join("models")
            .join(format!("{MODEL_FILE}.sha256"))
    }

    fn engine_marker_path(&self) -> PathBuf {
        self.engine_dir().join(".verified-sha256")
    }

    fn downloads_dir(&self) -> PathBuf {
        self.root.join("downloads")
    }

    fn set_error(&self, error: Option<String>) {
        if let Ok(mut value) = self.last_error.lock() {
            *value = error;
        }
    }

    pub(crate) fn stop(&self) {
        if let Ok(mut process) = self.process.lock() {
            if let Some(child) = process.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
            *process = None;
        }
        if let Ok(mut port) = self.port.lock() {
            *port = None;
        }
    }

    fn running_port(&self) -> Option<u16> {
        let mut running = false;
        if let Ok(mut process) = self.process.lock() {
            if let Some(child) = process.as_mut() {
                running = matches!(child.try_wait(), Ok(None));
            }
            if !running {
                *process = None;
            }
        }
        if !running {
            if let Ok(mut port) = self.port.lock() {
                *port = None;
            }
            return None;
        }
        self.port.lock().ok().and_then(|port| *port)
    }
}

impl Drop for LocalAiManager {
    fn drop(&mut self) {
        if let Ok(process) = self.process.get_mut() {
            if let Some(child) = process.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

fn emit_progress(
    app: &AppHandle,
    stage: &str,
    downloaded_bytes: u64,
    total_bytes: u64,
    message: impl Into<String>,
) {
    let percent = if total_bytes == 0 {
        0.0
    } else {
        (downloaded_bytes as f64 / total_bytes as f64 * 100.0).clamp(0.0, 100.0)
    };
    let _ = app.emit(
        "local-ai-progress",
        LocalAiProgress {
            stage: stage.to_string(),
            downloaded_bytes,
            total_bytes,
            percent,
            message: message.into(),
        },
    );
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn is_gguf(path: &Path) -> bool {
    let mut magic = [0_u8; 4];
    File::open(path)
        .and_then(|mut file| file.read_exact(&mut magic))
        .is_ok()
        && &magic == b"GGUF"
}

fn marker_matches(path: &Path, expected_sha256: &str) -> bool {
    fs::read_to_string(path)
        .map(|value| value.trim() == expected_sha256)
        .unwrap_or(false)
}

fn engine_is_verified(manager: &LocalAiManager) -> bool {
    let Ok(spec) = engine_spec() else {
        return false;
    };
    find_server_binary(&manager.engine_dir()).is_some()
        && marker_matches(&manager.engine_marker_path(), spec.sha256)
}

fn model_is_verified(manager: &LocalAiManager) -> bool {
    let model_path = manager.model_path();
    fs::metadata(&model_path)
        .map(|metadata| metadata.len() == MODEL_BYTES)
        .unwrap_or(false)
        && is_gguf(&model_path)
        && marker_matches(&manager.model_marker_path(), MODEL_SHA256)
}

fn find_server_binary(directory: &Path) -> Option<PathBuf> {
    let expected = if cfg!(target_os = "windows") {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let entries = fs::read_dir(directory).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_server_binary(&path) {
                return Some(found);
            }
        } else if path.file_name().and_then(|name| name.to_str()) == Some(expected) {
            return Some(path);
        }
    }
    None
}

#[cfg(unix)]
fn ensure_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = fs::metadata(path)
        .map_err(|error| error.to_string())?
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).map_err(|error| error.to_string())
}

#[cfg(not(unix))]
fn ensure_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

async fn download_file(
    app: &AppHandle,
    client: &Client,
    url: &str,
    destination: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    stage: &str,
    cancel: &AtomicBool,
) -> Result<(), String> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let part_path = destination.with_extension(format!(
        "{}.part",
        destination
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("download")
    ));
    let mut existing = fs::metadata(&part_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if existing > expected_bytes {
        fs::remove_file(&part_path).map_err(|error| error.to_string())?;
        existing = 0;
    }

    if existing < expected_bytes {
        let mut request = client.get(url);
        if existing > 0 {
            request = request.header(RANGE, format!("bytes={existing}-"));
        }
        let response = request.send().await.map_err(|error| error.to_string())?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "Download failed with HTTP {status}: {}",
                body.chars().take(180).collect::<String>()
            ));
        }

        let append = existing > 0 && status == StatusCode::PARTIAL_CONTENT;
        if existing > 0 && !append {
            existing = 0;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&part_path)
            .map_err(|error| error.to_string())?;
        let mut downloaded = existing;
        let mut stream = response.bytes_stream();
        let mut last_emit = Instant::now() - Duration::from_secs(1);
        emit_progress(
            app,
            stage,
            downloaded,
            expected_bytes,
            if downloaded > 0 {
                "Resuming download"
            } else {
                "Starting download"
            },
        );

        while let Some(chunk) = stream.next().await {
            if cancel.load(Ordering::Relaxed) {
                return Err(
                    "Setup cancelled. The partial download was kept for resume.".to_string()
                );
            }
            let chunk = chunk.map_err(|error| error.to_string())?;
            file.write_all(&chunk).map_err(|error| error.to_string())?;
            downloaded += chunk.len() as u64;
            if last_emit.elapsed() >= Duration::from_millis(160) {
                emit_progress(app, stage, downloaded, expected_bytes, "Downloading");
                last_emit = Instant::now();
            }
        }
        file.flush().map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        if downloaded != expected_bytes {
            return Err(format!(
                "Download ended at {downloaded} bytes; expected {expected_bytes}. Retry to resume."
            ));
        }
    }

    emit_progress(
        app,
        "verify",
        0,
        expected_bytes,
        format!("Verifying {stage} SHA-256"),
    );
    let actual_hash = sha256_file(&part_path)?;
    if actual_hash != expected_sha256 {
        let _ = fs::remove_file(&part_path);
        return Err(format!(
            "{stage} checksum did not match. The untrusted file was removed."
        ));
    }
    if destination.exists() {
        fs::remove_file(destination).map_err(|error| error.to_string())?;
    }
    fs::rename(&part_path, destination).map_err(|error| error.to_string())?;
    emit_progress(
        app,
        stage,
        expected_bytes,
        expected_bytes,
        "Download verified",
    );
    Ok(())
}

fn extract_tar_gz(archive_path: &Path, destination: &Path) -> Result<(), String> {
    let file = File::open(archive_path).map_err(|error| error.to_string())?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    let entries = archive.entries().map_err(|error| error.to_string())?;
    for entry in entries {
        let mut entry = entry.map_err(|error| error.to_string())?;
        let unpacked = entry
            .unpack_in(destination)
            .map_err(|error| error.to_string())?;
        if !unpacked {
            return Err("Engine archive contained an unsafe path".to_string());
        }
    }
    Ok(())
}

fn extract_zip(archive_path: &Path, destination: &Path) -> Result<(), String> {
    let file = File::open(archive_path).map_err(|error| error.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|error| error.to_string())?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|error| error.to_string())?;
        let relative = entry
            .enclosed_name()
            .ok_or_else(|| "Engine archive contained an unsafe path".to_string())?
            .to_owned();
        let output = destination.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(&output).map_err(|error| error.to_string())?;
            continue;
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut file = File::create(&output).map_err(|error| error.to_string())?;
        std::io::copy(&mut entry, &mut file).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn status_for(manager: &LocalAiManager) -> LocalAiStatus {
    let engine_installed = engine_is_verified(manager);
    let model_installed = model_is_verified(manager);
    let port = manager.running_port();
    let server_running = port.is_some();
    let error = manager
        .last_error
        .lock()
        .ok()
        .and_then(|value| value.clone());
    let state = if manager.installing.load(Ordering::Relaxed) {
        "installing"
    } else if server_running {
        "running"
    } else if engine_installed && model_installed {
        "ready"
    } else if error.is_some() {
        "error"
    } else {
        "notInstalled"
    };

    LocalAiStatus {
        state: state.to_string(),
        engine: "llama.cpp".to_string(),
        engine_version: ENGINE_VERSION.to_string(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        model_id: MODEL_ID.to_string(),
        model_name: MODEL_NAME.to_string(),
        model_bytes: MODEL_BYTES,
        model_sha256: MODEL_SHA256.to_string(),
        engine_installed,
        model_installed,
        server_running,
        port,
        error,
    }
}

async fn install_inner(app: &AppHandle, state: &State<'_, AppState>) -> Result<(), String> {
    let manager = &state.local_ai;
    let spec = engine_spec()?;
    fs::create_dir_all(&manager.root).map_err(|error| error.to_string())?;
    let model_remaining = if manager.model_path().exists() {
        0
    } else {
        MODEL_BYTES
    };
    let engine_remaining = if engine_is_verified(manager) {
        0
    } else {
        spec.bytes
    };
    let required = model_remaining + engine_remaining + DOWNLOAD_HEADROOM;
    let available = fs2::available_space(&manager.root).map_err(|error| error.to_string())?;
    if available < required {
        return Err(format!(
            "Not enough disk space. Setup needs {:.1} GB free.",
            required as f64 / 1_073_741_824.0
        ));
    }

    let client = Client::builder()
        .user_agent("Discover-AI/0.1 local-model-setup")
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(60 * 60 * 6))
        .build()
        .map_err(|error| error.to_string())?;
    fs::create_dir_all(manager.downloads_dir()).map_err(|error| error.to_string())?;

    if !engine_is_verified(manager) {
        let archive_path = manager.downloads_dir().join(spec.file_name);
        download_file(
            app,
            &client,
            spec.url,
            &archive_path,
            spec.bytes,
            spec.sha256,
            "engine",
            &manager.cancel_install,
        )
        .await?;
        emit_progress(app, "extract", 0, spec.bytes, "Installing llama.cpp engine");
        let temporary = manager
            .root
            .join("engine")
            .join(format!("{ENGINE_VERSION}.installing"));
        if temporary.exists() {
            fs::remove_dir_all(&temporary).map_err(|error| error.to_string())?;
        }
        fs::create_dir_all(&temporary).map_err(|error| error.to_string())?;
        match spec.archive {
            "tar.gz" => extract_tar_gz(&archive_path, &temporary)?,
            "zip" => extract_zip(&archive_path, &temporary)?,
            _ => return Err("Unsupported engine archive format".to_string()),
        }
        let server = find_server_binary(&temporary)
            .ok_or_else(|| "llama-server was not found in the verified archive".to_string())?;
        ensure_executable(&server)?;
        fs::write(
            temporary.join(".verified-sha256"),
            format!("{}\n", spec.sha256),
        )
        .map_err(|error| error.to_string())?;
        let final_directory = manager.engine_dir();
        if final_directory.exists() {
            fs::remove_dir_all(&final_directory).map_err(|error| error.to_string())?;
        }
        fs::rename(&temporary, &final_directory).map_err(|error| error.to_string())?;
        let _ = fs::remove_file(archive_path);
    }

    let model_path = manager.model_path();
    if !model_is_verified(manager) && model_path.exists() {
        emit_progress(app, "verify", 0, MODEL_BYTES, "Verifying existing model");
        let existing_hash = sha256_file(&model_path)?;
        if existing_hash == MODEL_SHA256 && is_gguf(&model_path) {
            fs::write(manager.model_marker_path(), format!("{MODEL_SHA256}\n"))
                .map_err(|error| error.to_string())?;
        } else {
            fs::remove_file(&model_path).map_err(|error| error.to_string())?;
            let _ = fs::remove_file(manager.model_marker_path());
        }
    }
    if !model_is_verified(manager) {
        download_file(
            app,
            &client,
            MODEL_URL,
            &model_path,
            MODEL_BYTES,
            MODEL_SHA256,
            "model",
            &manager.cancel_install,
        )
        .await?;
        fs::write(manager.model_marker_path(), format!("{MODEL_SHA256}\n"))
            .map_err(|error| error.to_string())?;
    }
    if !model_is_verified(manager) {
        let _ = fs::remove_file(&model_path);
        let _ = fs::remove_file(manager.model_marker_path());
        return Err("The model could not be marked as verified".to_string());
    }

    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.execute(
        "INSERT INTO local_models
           (id, display_name, file_path, bytes, sha256, context_length, status)
         VALUES (?1, ?2, ?3, ?4, ?5, 4096, 'ready')
         ON CONFLICT(id) DO UPDATE SET
           display_name = excluded.display_name,
           file_path = excluded.file_path,
           bytes = excluded.bytes,
           sha256 = excluded.sha256,
           context_length = excluded.context_length,
           status = excluded.status,
           updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        params![
            MODEL_ID,
            MODEL_NAME,
            model_path.to_string_lossy(),
            MODEL_BYTES as i64,
            MODEL_SHA256
        ],
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

async fn launch_server(app: &AppHandle, manager: &LocalAiManager) -> Result<u16, String> {
    if let Some(port) = manager.running_port() {
        return Ok(port);
    }
    if manager.starting.swap(true, Ordering::SeqCst) {
        for _ in 0..120 {
            if let Some(port) = manager.running_port() {
                return Ok(port);
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        return Err("Local model start is already in progress".to_string());
    }

    let result = async {
        if !engine_is_verified(manager) {
            return Err("Install and verify llama.cpp before starting the local model".to_string());
        }
        let server = find_server_binary(&manager.engine_dir())
            .ok_or_else(|| "Install llama.cpp before starting the local model".to_string())?;
        let model = manager.model_path();
        if !model_is_verified(manager) {
            return Err("Install the Qwen GGUF model before starting".to_string());
        }
        ensure_executable(&server)?;
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|error| error.to_string())?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        drop(listener);
        let threads = std::thread::available_parallelism()
            .map(|value| value.get().saturating_sub(1).max(2))
            .unwrap_or(4);
        let gpu_layers = if cfg!(target_os = "macos") { "99" } else { "0" };
        emit_progress(app, "launch", 0, 1, "Loading Qwen into llama.cpp");

        let child = Command::new(&server)
            .arg("--model")
            .arg(&model)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg("4096")
            .arg("--threads")
            .arg(threads.to_string())
            .arg("--n-gpu-layers")
            .arg(gpu_layers)
            .arg("--no-webui")
            .current_dir(server.parent().unwrap_or(&manager.root))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("Could not launch llama-server: {error}"))?;
        *manager
            .process
            .lock()
            .map_err(|_| "Process lock poisoned".to_string())? = Some(child);
        *manager
            .port
            .lock()
            .map_err(|_| "Port lock poisoned".to_string())? = Some(port);

        let client = Client::builder()
            .connect_timeout(Duration::from_secs(1))
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|error| error.to_string())?;
        for _ in 0..240 {
            if manager.running_port().is_none() {
                return Err("llama-server exited while loading the model".to_string());
            }
            if let Ok(response) = client
                .get(format!("http://127.0.0.1:{port}/health"))
                .send()
                .await
            {
                if response.status().is_success() {
                    emit_progress(app, "ready", 1, 1, "Local model is ready");
                    return Ok(port);
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        manager.stop();
        Err("llama-server did not become healthy within two minutes".to_string())
    }
    .await;
    manager.starting.store(false, Ordering::SeqCst);
    result
}

#[tauri::command]
pub(crate) fn local_ai_status(state: State<'_, AppState>) -> LocalAiStatus {
    status_for(&state.local_ai)
}

#[tauri::command]
pub(crate) async fn install_local_ai(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<LocalAiStatus, String> {
    let manager = &state.local_ai;
    if manager.installing.swap(true, Ordering::SeqCst) {
        return Err("Local model setup is already running".to_string());
    }
    manager.cancel_install.store(false, Ordering::SeqCst);
    manager.set_error(None);
    emit_progress(&app, "prepare", 0, 1, "Checking disk space and platform");

    let result = install_inner(&app, &state).await;
    manager.installing.store(false, Ordering::SeqCst);
    if let Err(error) = result {
        manager.set_error(Some(error.clone()));
        emit_progress(&app, "error", 0, 1, &error);
        return Err(error);
    }
    if let Err(error) = launch_server(&app, manager).await {
        manager.set_error(Some(format!("Installed, but could not start: {error}")));
        return Err(error);
    }
    manager.set_error(None);
    Ok(status_for(manager))
}

#[tauri::command]
pub(crate) fn cancel_local_ai_install(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    if !state.local_ai.installing.load(Ordering::Relaxed) {
        return Err("No local model setup is running".to_string());
    }
    state.local_ai.cancel_install.store(true, Ordering::SeqCst);
    emit_progress(
        &app,
        "cancel",
        0,
        1,
        "Stopping after the current download chunk",
    );
    Ok(())
}

#[tauri::command]
pub(crate) async fn start_local_ai(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<LocalAiStatus, String> {
    state.local_ai.set_error(None);
    if let Err(error) = launch_server(&app, &state.local_ai).await {
        state.local_ai.set_error(Some(error.clone()));
        return Err(error);
    }
    Ok(status_for(&state.local_ai))
}

#[tauri::command]
pub(crate) fn stop_local_ai(state: State<'_, AppState>) -> LocalAiStatus {
    state.local_ai.stop();
    status_for(&state.local_ai)
}

#[tauri::command]
pub(crate) fn remove_local_ai(state: State<'_, AppState>) -> Result<LocalAiStatus, String> {
    let manager = &state.local_ai;
    manager.cancel_install.store(true, Ordering::SeqCst);
    manager.stop();
    for directory in [
        manager.root.join("engine"),
        manager.root.join("models"),
        manager.root.join("downloads"),
    ] {
        if directory.exists() {
            fs::remove_dir_all(directory).map_err(|error| error.to_string())?;
        }
    }
    let db = state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned".to_string())?;
    db.execute("DELETE FROM local_models WHERE id = ?1", params![MODEL_ID])
        .map_err(|error| error.to_string())?;
    manager.set_error(None);
    Ok(status_for(manager))
}

#[tauri::command]
pub(crate) async fn local_chat(
    app: AppHandle,
    state: State<'_, AppState>,
    request: LocalChatRequest,
) -> Result<String, String> {
    if request.message.trim().is_empty() {
        return Err("Message cannot be empty".to_string());
    }
    if request.message.chars().count() > 8_000 {
        return Err("Message is too long".to_string());
    }
    let port = launch_server(&app, &state.local_ai).await?;
    let objectives = request
        .objectives
        .iter()
        .map(|objective| format!("- {objective}"))
        .collect::<Vec<_>>()
        .join("\n");
    let system = format!(
        "You are the private, offline topic guide for the learning module \"{}\".\n\
         Stay within this module. If the learner asks about another subject, briefly redirect them.\n\
         {} Be concise, accurate, and distinguish the toy visualization from real models.\n\
         Learning objectives:\n{}\n\
         Current step: {}\n\
         Current live interactive state: {}\n\
         Refer to the live state when it helps. Do not reveal hidden chain-of-thought; provide only the useful answer.",
        request.module_title,
        crate::cloud_providers::explanation_style(&request.mode),
        objectives,
        request.current_step,
        request.state_json
    );
    let payload = serde_json::json!({
        "model": MODEL_ID,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": request.message }
        ],
        "temperature": 0.6,
        "max_tokens": 500,
        "stream": true,
        "chat_template_kwargs": { "enable_thinking": false }
    });
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|error| error.to_string())?;
    let response = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "Local model returned HTTP {status}: {}",
            body.chars().take(240).collect::<String>()
        ));
    }

    let mut stream = response.bytes_stream();
    let mut pending = Vec::<u8>::new();
    let mut answer = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| error.to_string())?;
        pending.extend_from_slice(&chunk);
        while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
            let line = pending.drain(..=newline).collect::<Vec<_>>();
            let line = String::from_utf8_lossy(&line);
            let data = line.trim().strip_prefix("data:").map(str::trim);
            let Some(data) = data else {
                continue;
            };
            if data == "[DONE]" {
                continue;
            }
            let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            if let Some(content) = value
                .pointer("/choices/0/delta/content")
                .and_then(|content| content.as_str())
            {
                answer.push_str(content);
                let _ = app.emit(
                    "local-chat-chunk",
                    LocalChatChunk {
                        request_id: request.request_id.clone(),
                        content: content.to_string(),
                        done: false,
                    },
                );
            }
        }
    }
    let _ = app.emit(
        "local-chat-chunk",
        LocalChatChunk {
            request_id: request.request_id,
            content: String::new(),
            done: true,
        },
    );
    if answer.trim().is_empty() {
        return Err("The local model returned an empty answer".to_string());
    }
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_model_metadata_is_consistent() {
        assert_eq!(MODEL_SHA256.len(), 64);
        assert!(MODEL_URL.contains("f6d5376be1edb4d416d56da11e5397a961aca8ae"));
        assert!(MODEL_BYTES > 1_000_000_000);
    }

    #[test]
    fn server_search_finds_nested_binary() {
        let root =
            std::env::temp_dir().join(format!("the-ai-guide-server-search-{}", std::process::id()));
        let nested = root.join("build").join("bin");
        fs::create_dir_all(&nested).unwrap();
        let file_name = if cfg!(target_os = "windows") {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        let expected = nested.join(file_name);
        File::create(&expected).unwrap();
        assert_eq!(find_server_binary(&root), Some(expected));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn gguf_validation_rejects_other_files() {
        let path =
            std::env::temp_dir().join(format!("the-ai-guide-gguf-check-{}", std::process::id()));
        fs::write(&path, b"GGUFpayload").unwrap();
        assert!(is_gguf(&path));
        fs::write(&path, b"not a model").unwrap();
        assert!(!is_gguf(&path));
        fs::remove_file(path).unwrap();
    }
}
