//! Call detector - process hints only; no process alone confirms an active call.

use super::types::{CallDetectionState, CallInfo};
use std::process::Command;

pub struct CallDetector;

impl CallDetector {
    pub fn detect() -> Option<CallInfo> {
        let processes = Self::get_running_processes();
        Self::get_known_call_apps(&processes).first().cloned()
    }

    pub fn detect_from_process_names(process_names: &[&str]) -> Option<CallInfo> {
        let processes = process_names
            .iter()
            .map(|name| ProcessInfo {
                name: (*name).to_string(),
                window_title: None,
            })
            .collect::<Vec<_>>();
        Self::get_known_call_apps(&processes).first().cloned()
    }

    pub fn detect_from_process_signals(signals: &[(&str, Option<&str>)]) -> Option<CallInfo> {
        let processes = signals
            .iter()
            .map(|(name, title)| ProcessInfo {
                name: (*name).to_string(),
                window_title: title.map(str::to_string),
            })
            .collect::<Vec<_>>();
        Self::get_known_call_apps(&processes).first().cloned()
    }

    fn get_running_processes() -> Vec<ProcessInfo> {
        let mut processes = Vec::new();

        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = Command::new("tasklist").args(["/FO", "CSV"]).output() {
                let text = String::from_utf8_lossy(&output.stdout);
                for line in text.lines().skip(1) {
                    if let Some(name) = line.split(',').next() {
                        processes.push(ProcessInfo {
                            name: name.trim_matches('"').to_string(),
                            window_title: None,
                        });
                    }
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(entries) = std::fs::read_dir("/proc") {
                for entry in entries.filter_map(Result::ok) {
                    let pid_str = entry.file_name().to_string_lossy().to_string();
                    if pid_str.parse::<u32>().is_ok() {
                        if let Ok(exe) = std::fs::read_link(entry.path().join("exe")) {
                            if let Some(name) = exe.file_name() {
                                processes.push(ProcessInfo {
                                    name: name.to_string_lossy().to_string(),
                                    window_title: None,
                                });
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = Command::new("ps").args(["-eo", "pid=,comm="]).output() {
                let text = String::from_utf8_lossy(&output.stdout);
                for line in text.lines() {
                    let parts = line.trim().splitn(2, ' ').collect::<Vec<_>>();
                    if parts.len() == 2 && parts[0].trim().parse::<u32>().is_ok() {
                        processes.push(ProcessInfo {
                            name: parts[1].trim().to_string(),
                            window_title: None,
                        });
                    }
                }
            }
        }

        processes
    }

    fn get_known_call_apps(processes: &[ProcessInfo]) -> Vec<CallInfo> {
        let known_apps = [
            ("teams", "Microsoft Teams", "teams", false),
            ("chrome", "Google Chrome", "chrome", true),
            ("msedge", "Microsoft Edge", "edge", true),
            ("discord", "Discord", "discord", false),
            ("zoom", "Zoom", "zoom", false),
            ("slack", "Slack", "slack", false),
            ("webex", "WebEx", "webex", false),
        ];

        let mut results = Vec::new();

        for (pattern, display_name, process_name, browser_only) in known_apps {
            if let Some(process) = processes
                .iter()
                .find(|process| process.name.to_lowercase().contains(pattern))
            {
                let title = process.window_title.clone().unwrap_or_default();
                let title_lower = title.to_lowercase();
                let strong_meeting_title = title_lower.contains("in call")
                    || title_lower.contains("meeting controls")
                    || title_lower.contains("google meet")
                    || title_lower.contains("zoom meeting")
                    || title_lower.contains("teams meeting");
                let likely_meeting_title = strong_meeting_title
                    || title_lower.contains("meet")
                    || title_lower.contains("meeting")
                    || title_lower.contains("call");
                let detection_state = if strong_meeting_title {
                    CallDetectionState::Confirmed
                } else if likely_meeting_title || !browser_only {
                    CallDetectionState::Likely
                } else {
                    CallDetectionState::Detected
                };
                results.push(CallInfo {
                    platform: display_name.to_string(),
                    window_title: title,
                    process_name: process_name.to_string(),
                    is_active_call: detection_state == CallDetectionState::Confirmed,
                    detection_state,
                    confidence: if strong_meeting_title {
                        0.90
                    } else if likely_meeting_title {
                        0.70
                    } else if browser_only {
                        0.20
                    } else {
                        0.50
                    },
                });
            }
        }

        results
    }
}

#[derive(Clone)]
struct ProcessInfo {
    name: String,
    window_title: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_browser_process_is_not_confirmed_call() {
        let result = CallDetector::detect_from_process_names(&["chrome.exe"])
            .expect("browser process should produce a weak detection");
        assert!(!result.is_active_call);
        assert_eq!(result.detection_state, CallDetectionState::Detected);
        assert!(result.window_title.is_empty());
    }

    #[test]
    fn zoom_process_alone_is_not_confirmed_call() {
        let result = CallDetector::detect_from_process_names(&["Zoom.exe"])
            .expect("zoom process should produce a process-only detection");
        assert!(!result.is_active_call);
        assert_eq!(result.detection_state, CallDetectionState::Likely);
        assert!(result.window_title.is_empty());
    }

    #[test]
    fn strong_meeting_title_can_confirm_call() {
        let result = CallDetector::detect_from_process_signals(&[(
            "chrome.exe",
            Some("Google Meet - In call"),
        )])
        .expect("strong title should produce a detection");
        assert!(result.is_active_call);
        assert_eq!(result.detection_state, CallDetectionState::Confirmed);
        assert_eq!(result.window_title, "Google Meet - In call");
    }
}
