use crate::desktop_agent_types::{Permission, RiskLevel, ToolDescriptor};

#[derive(Debug, Clone)]
pub struct ToolsRegistry {
    tools: Vec<ToolDescriptor>,
}

impl ToolsRegistry {
    pub fn new() -> Self {
        Self {
            tools: vec![
                tool(
                    "filesystem.read_text",
                    "filesystem",
                    "Read a UTF-8 text file from an allowed root",
                    vec![Permission::FilesystemRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "filesystem.write_text",
                    "filesystem",
                    "Create, overwrite, or append UTF-8 text files inside allowed roots",
                    vec![Permission::FilesystemWrite],
                    RiskLevel::High,
                    true,
                ),
                tool(
                    "filesystem.search",
                    "filesystem",
                    "Search files inside an allowed root by filename pattern",
                    vec![Permission::FilesystemSearch],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "terminal.run",
                    "terminal",
                    "Execute an allowlisted terminal command inside an allowed working directory",
                    vec![Permission::TerminalSafe],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "browser.open",
                    "browser",
                    "Open a URL in the system browser",
                    vec![Permission::BrowserAction],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "browser.search",
                    "browser",
                    "Run a web search in the default browser",
                    vec![Permission::BrowserRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "screen.analyze",
                    "screen",
                    "Capture or inspect the current screen and ask Astra Vision what is visible",
                    vec![Permission::DesktopObserve],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "desktop.launch_app",
                    "desktop",
                    "Launch a desktop application or file through the operating system",
                    vec![Permission::DesktopControl],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.detect",
                    "meeting",
                    "Detect meeting software with explicit confidence, without confirming recording",
                    vec![Permission::MeetingDetect],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.consent.read",
                    "meeting",
                    "Read scoped meeting consent state",
                    vec![Permission::MeetingConsentRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.consent.grant",
                    "meeting",
                    "Grant scoped consent for governed meeting capture/transcription",
                    vec![Permission::MeetingConsentWrite],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.consent.revoke",
                    "meeting",
                    "Revoke scoped consent for governed meeting capture/transcription",
                    vec![Permission::MeetingConsentWrite],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.session.read",
                    "meeting",
                    "Read active meeting session metadata and state",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.sessions.list",
                    "meeting",
                    "List local archived meeting sessions from the governed session memory index",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.session.archive.read",
                    "meeting",
                    "Read a local archived meeting session with transcript and intelligence controls",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.session.search",
                    "meeting",
                    "Search local archived meeting sessions lexically without logging raw query text",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.recall.answer",
                    "meeting",
                    "Answer a governed local recall question from archived work-session evidence only",
                    vec![
                        Permission::MeetingSessionRead,
                        Permission::MeetingIntelligenceGenerate,
                    ],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.session.export",
                    "meeting",
                    "Export a local archived meeting session as governed JSON or Markdown",
                    vec![Permission::MeetingSessionRead, Permission::MeetingExport],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.session.reindex",
                    "meeting",
                    "Rebuild the local meeting session memory index from archived session files",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.session.start",
                    "meeting",
                    "Start a governed meeting session; manual mode is supported, real capture is gated separately",
                    vec![Permission::MeetingSessionManage],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.session.manual",
                    "meeting",
                    "Start a governed manual meeting session without audio capture",
                    vec![Permission::MeetingSessionManage],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.session.pause",
                    "meeting",
                    "Pause the active governed meeting session",
                    vec![Permission::MeetingSessionManage],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.session.resume",
                    "meeting",
                    "Resume the active governed meeting session",
                    vec![Permission::MeetingSessionManage],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.session.stop",
                    "meeting",
                    "Stop the active governed meeting session and export notes",
                    vec![Permission::MeetingSessionManage, Permission::MeetingExport],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.session.clear",
                    "meeting",
                    "Clear only the active in-memory meeting session state",
                    vec![Permission::MeetingSessionManage],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.transcript.add",
                    "meeting",
                    "Add a manual transcript entry to an active governed meeting session",
                    vec![Permission::MeetingTranscriptWrite],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.transcript.list",
                    "meeting",
                    "Read active meeting transcript entries with source-channel metadata",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.notes.read",
                    "meeting",
                    "Read transcript-derived meeting notes and summaries",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.action_items.read",
                    "meeting",
                    "Read transcript-derived meeting action items",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.decisions.read",
                    "meeting",
                    "Read transcript-derived meeting decisions",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.diagnostics.read",
                    "meeting",
                    "Read truthful meeting capture/transcription diagnostics",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.screen_context.attach_current",
                    "meeting",
                    "Attach a governed on-demand screen context snapshot to the active work session without desktop control",
                    vec![
                        Permission::DesktopObserve,
                        Permission::MeetingSessionRead,
                        Permission::MeetingNotesWrite,
                    ],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.intelligence.generate",
                    "meeting",
                    "Generate governed transcript-backed meeting intelligence artifacts",
                    vec![
                        Permission::MeetingIntelligenceGenerate,
                        Permission::MeetingTranscriptWrite,
                    ],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.intelligence.read",
                    "meeting",
                    "Read generated transcript-backed meeting intelligence artifacts",
                    vec![Permission::MeetingIntelligenceRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.intelligence.clear",
                    "meeting",
                    "Clear generated meeting intelligence artifacts without changing the raw transcript",
                    vec![Permission::MeetingIntelligenceClear],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.followup.draft",
                    "meeting",
                    "Create a copy-only follow-up draft as a generated meeting intelligence artifact",
                    vec![Permission::MeetingIntelligenceGenerate],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.transcription.file",
                    "meeting",
                    "Transcribe a validated local audio file into the active meeting through the existing SttClient",
                    vec![
                        Permission::MeetingTranscriptionFile,
                        Permission::MeetingTranscriptWrite,
                    ],
                    RiskLevel::High,
                    true,
                ),
                tool(
                    "meeting.action_item.add",
                    "meeting",
                    "Add an action item to an active governed meeting session",
                    vec![Permission::MeetingNotesWrite],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.decision.add",
                    "meeting",
                    "Add a decision to an active governed meeting session",
                    vec![Permission::MeetingNotesWrite],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.speaker.rename",
                    "meeting",
                    "Rename a meeting speaker display label as governed metadata without changing transcript text",
                    vec![Permission::MeetingNotesWrite],
                    RiskLevel::Medium,
                    false,
                ),
                tool(
                    "meeting.audio.devices",
                    "meeting",
                    "Read available audio device names without starting capture",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.audio.backend",
                    "meeting",
                    "Read the platform-preferred audio backend without starting capture",
                    vec![Permission::MeetingSessionRead],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.clear_data.preview",
                    "meeting",
                    "Preview governed meeting runtime state and persisted files that would be cleared",
                    vec![Permission::MeetingClearData],
                    RiskLevel::Low,
                    false,
                ),
                tool(
                    "meeting.clear_data",
                    "meeting",
                    "Clear governed meeting runtime state and persisted meeting files after explicit typed confirmation",
                    vec![Permission::MeetingClearData],
                    RiskLevel::High,
                    true,
                ),
                meeting_audio_capture_tool(),
                meeting_audio_capture_source_tool(
                    "meeting.audio.capture.system",
                    "Start governed Windows WASAPI render loopback capture into managed meeting WAV segments",
                ),
                meeting_audio_capture_source_tool(
                    "meeting.audio.capture.microphone",
                    "Start governed Windows WASAPI microphone capture into managed meeting WAV segments",
                ),
                tool(
                    "meeting.transcription.segment",
                    "meeting",
                    "Transcribe governed managed meeting capture segments through the existing SttClient file bridge",
                    vec![
                        Permission::MeetingTranscriptionSegment,
                        Permission::MeetingTranscriptWrite,
                    ],
                    RiskLevel::High,
                    true,
                ),
                unavailable_tool(
                    "meeting.transcription.live",
                    "Live meeting transcription is not connected to SttClient yet",
                    vec![Permission::MeetingTranscriptionLive],
                    RiskLevel::High,
                    true,
                ),
                unavailable_tool(
                    "meeting.followup.send",
                    "Follow-up sending is disabled until draft-first outbound integrations are governed",
                    vec![Permission::MeetingFollowUpSend],
                    RiskLevel::High,
                    true,
                ),
            ],
        }
    }

    pub fn list(&self) -> Vec<ToolDescriptor> {
        self.tools.clone()
    }

    pub fn get(&self, tool_name: &str) -> Option<ToolDescriptor> {
        self.tools
            .iter()
            .find(|tool| tool.tool_name == tool_name)
            .cloned()
    }
}

fn tool(
    tool_name: &str,
    category: &str,
    description: &str,
    required_permissions: Vec<Permission>,
    default_risk: RiskLevel,
    requires_confirmation: bool,
) -> ToolDescriptor {
    ToolDescriptor {
        tool_name: tool_name.into(),
        category: category.into(),
        description: description.into(),
        required_permissions,
        default_risk,
        requires_confirmation,
        available: true,
        unavailable_reason: None,
    }
}

fn unavailable_tool(
    tool_name: &str,
    reason: &str,
    required_permissions: Vec<Permission>,
    default_risk: RiskLevel,
    requires_confirmation: bool,
) -> ToolDescriptor {
    ToolDescriptor {
        tool_name: tool_name.into(),
        category: "meeting".into(),
        description: reason.into(),
        required_permissions,
        default_risk,
        requires_confirmation,
        available: false,
        unavailable_reason: Some(reason.into()),
    }
}

fn meeting_audio_capture_tool() -> ToolDescriptor {
    #[cfg(target_os = "windows")]
    {
        tool(
            "meeting.audio.capture",
            "meeting",
            "Start governed Windows WASAPI loopback capture into managed meeting WAV segments",
            vec![Permission::MeetingAudioCapture],
            RiskLevel::High,
            true,
        )
    }

    #[cfg(not(target_os = "windows"))]
    {
        unavailable_tool(
            "meeting.audio.capture",
            "Windows WASAPI loopback capture is unavailable on this platform",
            vec![Permission::MeetingAudioCapture],
            RiskLevel::High,
            true,
        )
    }
}

fn meeting_audio_capture_source_tool(tool_name: &str, description: &str) -> ToolDescriptor {
    #[cfg(target_os = "windows")]
    {
        tool(
            tool_name,
            "meeting",
            description,
            vec![Permission::MeetingAudioCapture],
            RiskLevel::High,
            true,
        )
    }

    #[cfg(not(target_os = "windows"))]
    {
        unavailable_tool(
            tool_name,
            "Windows WASAPI capture is unavailable on this platform",
            vec![Permission::MeetingAudioCapture],
            RiskLevel::High,
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meeting_capabilities_are_granular_and_truthful() {
        let registry = ToolsRegistry::new();

        let consent_read = registry.get("meeting.consent.read").expect("consent read");
        let session_start = registry
            .get("meeting.session.start")
            .expect("session start");
        let manual_session = registry
            .get("meeting.session.manual")
            .expect("manual session");
        let capture = registry
            .get("meeting.audio.capture")
            .expect("audio capture");
        let system_capture = registry
            .get("meeting.audio.capture.system")
            .expect("system audio capture");
        let microphone_capture = registry
            .get("meeting.audio.capture.microphone")
            .expect("microphone capture");
        let file_transcription = registry
            .get("meeting.transcription.file")
            .expect("file transcription");
        let transcript_list = registry
            .get("meeting.transcript.list")
            .expect("transcript list");
        let diagnostics_read = registry
            .get("meeting.diagnostics.read")
            .expect("diagnostics read");
        let speaker_rename = registry
            .get("meeting.speaker.rename")
            .expect("speaker rename");
        let intelligence_generate = registry
            .get("meeting.intelligence.generate")
            .expect("intelligence generate");
        let intelligence_read = registry
            .get("meeting.intelligence.read")
            .expect("intelligence read");
        let intelligence_clear = registry
            .get("meeting.intelligence.clear")
            .expect("intelligence clear");
        let followup_draft = registry
            .get("meeting.followup.draft")
            .expect("followup draft");
        let live_transcription = registry
            .get("meeting.transcription.live")
            .expect("live transcription");
        let segment_transcription = registry
            .get("meeting.transcription.segment")
            .expect("segment transcription");
        let followup = registry.get("meeting.followup.send").expect("followup");
        let clear_data = registry.get("meeting.clear_data").expect("clear data");
        let clear_preview = registry
            .get("meeting.clear_data.preview")
            .expect("clear data preview");

        assert!(consent_read.available);
        assert!(session_start.available);
        assert!(manual_session.available);
        assert!(file_transcription.available);
        assert!(transcript_list.available);
        assert!(diagnostics_read.available);
        assert!(speaker_rename.available);
        assert_eq!(speaker_rename.default_risk, RiskLevel::Medium);
        assert!(!speaker_rename.requires_confirmation);
        assert!(intelligence_generate.available);
        assert_eq!(intelligence_generate.default_risk, RiskLevel::Medium);
        assert!(intelligence_read.available);
        assert_eq!(intelligence_read.default_risk, RiskLevel::Low);
        assert!(intelligence_clear.available);
        assert_eq!(intelligence_clear.default_risk, RiskLevel::Medium);
        assert!(followup_draft.available);
        assert!(!followup_draft.requires_confirmation);
        assert_eq!(capture.available, cfg!(target_os = "windows"));
        assert_eq!(system_capture.available, cfg!(target_os = "windows"));
        assert_eq!(microphone_capture.available, cfg!(target_os = "windows"));
        assert!(segment_transcription.available);
        assert!(!live_transcription.available);
        assert_eq!(file_transcription.default_risk, RiskLevel::High);
        assert!(file_transcription.requires_confirmation);
        assert!(!followup.available);
        assert_eq!(capture.default_risk, RiskLevel::High);
        assert!(capture.requires_confirmation);
        assert_eq!(live_transcription.default_risk, RiskLevel::High);
        assert!(live_transcription.requires_confirmation);
        assert!(clear_preview.available);
        assert_eq!(clear_preview.default_risk, RiskLevel::Low);
        assert!(!clear_preview.requires_confirmation);
        assert!(clear_data.available);
        assert_eq!(clear_data.default_risk, RiskLevel::High);
        assert!(clear_data.requires_confirmation);
        if cfg!(target_os = "windows") {
            assert!(capture.unavailable_reason.is_none());
        } else {
            assert_eq!(
                capture.unavailable_reason.as_deref(),
                Some("Windows WASAPI loopback capture is unavailable on this platform")
            );
        }
    }

    #[test]
    fn meeting_screen_context_attach_uses_observe_without_desktop_control() {
        let registry = ToolsRegistry::new();
        let attach_screen = registry
            .get("meeting.screen_context.attach_current")
            .expect("screen context attach");

        assert!(attach_screen.available);
        assert!(attach_screen
            .required_permissions
            .contains(&Permission::DesktopObserve));
        assert!(!attach_screen
            .required_permissions
            .contains(&Permission::DesktopControl));
    }
}
