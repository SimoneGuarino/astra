use crate::desktop_agent_types::{ScreenCaptureResult, ScreenObservationStatus};
use std::{fs, path::PathBuf, process::Command, sync::{Arc, Mutex}};
use uuid::Uuid;

#[derive(Clone)]
pub struct ScreenCaptureRuntime {
    output_dir: PathBuf,
    status: Arc<Mutex<ScreenObservationStatus>>,
}

impl ScreenCaptureRuntime {
    pub fn new(project_root: &PathBuf) -> Self {
        let output_dir = project_root.join(".astra").join("screen");
        let _ = fs::create_dir_all(&output_dir);
        Self {
            output_dir,
            status: Arc::new(Mutex::new(ScreenObservationStatus {
                enabled: false,
                provider: if cfg!(target_os = "windows") {
                    "powershell_gdi".into()
                } else {
                    "not_supported".into()
                },
                last_frame_at: None,
                last_error: None,
                last_capture_path: None,
                capture_count: 0,
                note: "Screen capture foundation is available on demand; continuous observation is disabled by default".into(),
            })),
        }
    }

    pub fn status(&self) -> ScreenObservationStatus {
        self.status.lock().expect("screen capture status mutex poisoned").clone()
    }

    pub fn set_enabled(&self, enabled: bool) -> ScreenObservationStatus {
        let mut status = self.status.lock().expect("screen capture status mutex poisoned");
        status.enabled = enabled;
        if enabled {
            status.note = "Screen observation is enabled in on-demand mode".into();
        } else {
            status.note = "Screen observation is disabled".into();
        }
        status.clone()
    }

    pub fn latest_capture_path(&self) -> Option<String> {
        self.status.lock().expect("screen capture status mutex poisoned").last_capture_path.clone()
    }

    pub fn capture_snapshot(&self) -> Result<ScreenCaptureResult, String> {
        if !cfg!(target_os = "windows") {
            let mut status = self.status.lock().expect("screen capture status mutex poisoned");
            status.last_error = Some("Screen capture is currently implemented only on Windows".into());
            return Err("Screen capture is currently implemented only on Windows".into());
        }

        let capture_id = Uuid::new_v4().to_string();
        let output_path = self.output_dir.join(format!("screen_{capture_id}.png"));
        let script = build_windows_capture_script(&output_path);
        let output = Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .output()
            .map_err(|e| format!("screen capture command failed: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let message = if stderr.is_empty() {
                format!("screen capture command exited with status {:?}", output.status.code())
            } else {
                stderr
            };
            let mut status = self.status.lock().expect("screen capture status mutex poisoned");
            status.last_error = Some(message.clone());
            return Err(message);
        }

        let metadata = fs::metadata(&output_path).map_err(|e| format!("screen capture file missing after capture: {e}"))?;
        let now = now_ms();
        let path_string = output_path.display().to_string();

        let mut status = self.status.lock().expect("screen capture status mutex poisoned");
        status.last_frame_at = Some(now);
        status.last_error = None;
        status.last_capture_path = Some(path_string.clone());
        status.capture_count += 1;

        Ok(ScreenCaptureResult {
            capture_id,
            captured_at: now,
            image_path: path_string,
            width: None,
            height: None,
            bytes: metadata.len(),
            provider: status.provider.clone(),
        })
    }
}

fn build_windows_capture_script(output_path: &PathBuf) -> String {
    let output = output_path.display().to_string().replace('\\', "\\\\");
    format!(
        concat!(
            "Add-Type -AssemblyName System.Windows.Forms; ",
            "Add-Type -AssemblyName System.Drawing; ",
            "$bounds=[System.Windows.Forms.SystemInformation]::VirtualScreen; ",
            "$bmp=New-Object System.Drawing.Bitmap $bounds.Width,$bounds.Height; ",
            "$graphics=[System.Drawing.Graphics]::FromImage($bmp); ",
            "$graphics.CopyFromScreen($bounds.X,$bounds.Y,0,0,$bmp.Size); ",
            "$bmp.Save('{}', [System.Drawing.Imaging.ImageFormat]::Png); ",
            "$graphics.Dispose(); $bmp.Dispose();"
        ),
        output
    )
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
