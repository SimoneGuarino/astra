use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};

const GENERATED_AUDIO_TTL: Duration = Duration::from_secs(60 * 30);
const POST_PLAYBACK_CLEANUP_DELAY: Duration = Duration::from_secs(5);
const CANCELLATION_CLEANUP_DELAY: Duration = Duration::from_secs(2);

#[derive(Clone)]
pub struct AudioFileRegistry {
    generated_dir: PathBuf,
    files_by_request: Arc<Mutex<HashMap<String, HashSet<PathBuf>>>>,
}

impl AudioFileRegistry {
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            generated_dir: project_root
                .join("python_services")
                .join("tts")
                .join("generated"),
            files_by_request: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn ensure_generated_dir(&self) -> Result<(), String> {
        fs::create_dir_all(&self.generated_dir)
            .map_err(|error| format!("create generated audio dir failed: {error}"))
    }

    pub fn cleanup_stale_files(&self) {
        let generated_dir = self.generated_dir.clone();
        tauri::async_runtime::spawn_blocking(move || {
            if let Err(error) = cleanup_stale_files_in_dir(&generated_dir) {
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "type": "audio_file_cleanup",
                        "event": "stale_cleanup_failed",
                        "error": error,
                    })
                );
            }
        });
    }

    pub fn register(&self, request_id: &str, path: PathBuf) {
        let mut files_by_request = self
            .files_by_request
            .lock()
            .expect("audio file registry mutex poisoned");
        files_by_request
            .entry(request_id.to_string())
            .or_default()
            .insert(path);
    }

    pub fn cleanup_played_file(&self, request_id: &str, path: PathBuf) {
        self.unregister_file(request_id, &path);
        schedule_file_removal(path, POST_PLAYBACK_CLEANUP_DELAY, "post_playback");
    }

    pub fn cleanup_request(&self, request_id: &str) {
        let files = {
            let mut files_by_request = self
                .files_by_request
                .lock()
                .expect("audio file registry mutex poisoned");
            files_by_request.remove(request_id)
        };

        if let Some(files) = files {
            for path in files {
                schedule_file_removal(path, CANCELLATION_CLEANUP_DELAY, "request_cleanup");
            }
        }
    }

    fn unregister_file(&self, request_id: &str, path: &Path) {
        let mut files_by_request = self
            .files_by_request
            .lock()
            .expect("audio file registry mutex poisoned");

        if let Some(files) = files_by_request.get_mut(request_id) {
            files.remove(path);
            if files.is_empty() {
                files_by_request.remove(request_id);
            }
        }
    }
}

fn cleanup_stale_files_in_dir(generated_dir: &Path) -> Result<(), String> {
    if !generated_dir.exists() {
        return Ok(());
    }

    let now = SystemTime::now();
    for entry in fs::read_dir(generated_dir).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("wav") {
            continue;
        }

        let metadata = entry.metadata().map_err(|error| error.to_string())?;
        let modified = metadata.modified().map_err(|error| error.to_string())?;
        let age = now.duration_since(modified).unwrap_or_default();
        if age >= GENERATED_AUDIO_TTL {
            remove_file(&path, "stale_ttl");
        }
    }

    Ok(())
}

fn schedule_file_removal(path: PathBuf, delay: Duration, reason: &'static str) {
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(delay).await;
        remove_file(&path, reason);
    });
}

fn remove_file(path: &Path, reason: &'static str) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            eprintln!(
                "{}",
                serde_json::json!({
                    "type": "audio_file_cleanup",
                    "event": "remove_failed",
                    "reason": reason,
                    "path": path.display().to_string(),
                    "error": error.to_string(),
                })
            );
        }
    }
}
