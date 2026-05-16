use crate::desktop_agent_types::Permission;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionProfile {
    pub allowed: Vec<Permission>,
}

impl PermissionProfile {
    pub fn default_local_agent() -> Self {
        Self {
            allowed: vec![
                Permission::FilesystemRead,
                Permission::FilesystemWrite,
                Permission::FilesystemSearch,
                Permission::TerminalSafe,
                Permission::BrowserRead,
                Permission::BrowserAction,
                Permission::DesktopObserve,
                Permission::DesktopControl,
                Permission::MeetingDetect,
                Permission::MeetingConsentRead,
                Permission::MeetingConsentWrite,
                Permission::MeetingSessionRead,
                Permission::MeetingSessionManage,
                Permission::MeetingTranscriptWrite,
                Permission::MeetingNotesWrite,
                Permission::MeetingExport,
                Permission::MeetingClearData,
                Permission::MeetingTranscriptionFile,
                // DEV-ONLY hardware validation
                Permission::MeetingAudioCapture,
                Permission::MeetingTranscriptionSegment,
            ],
        }
    }

    pub fn allows(&self, permission: &Permission) -> bool {
        self.allowed.contains(permission)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_permission_profile_limits_future_high_risk_meeting_capabilities() {
        let profile = PermissionProfile::default_local_agent();

        assert!(profile.allows(&Permission::MeetingDetect));
        assert!(profile.allows(&Permission::MeetingConsentRead));
        assert!(profile.allows(&Permission::MeetingConsentWrite));
        assert!(profile.allows(&Permission::MeetingSessionRead));
        assert!(profile.allows(&Permission::MeetingSessionManage));
        assert!(profile.allows(&Permission::MeetingTranscriptWrite));
        assert!(profile.allows(&Permission::MeetingNotesWrite));
        assert!(profile.allows(&Permission::MeetingExport));
        assert!(profile.allows(&Permission::MeetingClearData));
        assert!(profile.allows(&Permission::MeetingTranscriptionFile));

        assert!(profile.allows(&Permission::MeetingAudioCapture));
        assert!(profile.allows(&Permission::MeetingTranscriptionSegment));
        assert!(!profile.allows(&Permission::MeetingTranscriptionLive));
        assert!(!profile.allows(&Permission::MeetingFollowUpSend));
    }
}
