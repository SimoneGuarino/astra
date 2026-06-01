#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkSessionChatIntent {
    StartSession,
    StopSession,
    StopAndGenerateRecap,
    AttachScreenContext,
    GenerateIntelligence,
    GenerateTranscriptSummary,
    GenerateDetails,
    GenerateTechnicalRecap,
    GenerateFollowUpDraft,
    RecallSessionMemory,
    SearchSessionMemory,
    ShowEvidence,
    ShowSessionStatus,
    OpenMeetingPanel,
    Unknown,
}

impl WorkSessionChatIntent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StartSession => "start_session",
            Self::StopSession => "stop_session",
            Self::StopAndGenerateRecap => "stop_and_generate_recap",
            Self::AttachScreenContext => "attach_screen_context",
            Self::GenerateIntelligence => "generate_intelligence",
            Self::GenerateTranscriptSummary => "generate_transcript_summary",
            Self::GenerateDetails => "generate_details",
            Self::GenerateTechnicalRecap => "generate_technical_recap",
            Self::GenerateFollowUpDraft => "generate_follow_up_draft",
            Self::RecallSessionMemory => "recall_session_memory",
            Self::SearchSessionMemory => "search_session_memory",
            Self::ShowEvidence => "show_evidence",
            Self::ShowSessionStatus => "show_session_status",
            Self::OpenMeetingPanel => "open_meeting_panel",
            Self::Unknown => "unknown",
        }
    }

    pub fn primary_tool_name(self) -> Option<&'static str> {
        match self {
            Self::StartSession => Some("meeting.session.start"),
            Self::StopSession | Self::StopAndGenerateRecap => Some("meeting.session.stop.request"),
            Self::AttachScreenContext => Some("meeting.screen_context.attach_current"),
            Self::GenerateIntelligence | Self::GenerateTechnicalRecap => {
                Some("meeting.intelligence.generate")
            }
            Self::GenerateTranscriptSummary | Self::GenerateDetails => Some("meeting.session.read"),
            Self::GenerateFollowUpDraft => Some("meeting.followup.draft"),
            Self::RecallSessionMemory => Some("meeting.recall.answer"),
            Self::SearchSessionMemory => Some("meeting.session.search"),
            Self::ShowEvidence => Some("meeting.recall.answer"),
            Self::ShowSessionStatus | Self::OpenMeetingPanel => Some("meeting.session.read"),
            Self::Unknown => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkSessionTargetKind {
    ActiveSession,
    LastReferencedSession,
    LatestArchivedSession,
    LastCompletedSession,
    CurrentScreen,
    ArchivedSessions,
    None,
    Unknown,
}

impl WorkSessionTargetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ActiveSession => "active_session",
            Self::LastReferencedSession => "last_referenced_session",
            Self::LatestArchivedSession => "latest_archived_session",
            Self::LastCompletedSession => "last_completed_session",
            Self::CurrentScreen => "current_screen",
            Self::ArchivedSessions => "archived_sessions",
            Self::None => "none",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkSessionExecutionTarget {
    pub kind: WorkSessionTargetKind,
    pub session_id: Option<String>,
    pub object_type: Option<String>,
    pub object_ids: Vec<String>,
}

impl WorkSessionExecutionTarget {
    pub fn none() -> Self {
        Self {
            kind: WorkSessionTargetKind::None,
            session_id: None,
            object_type: None,
            object_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkSessionChatRoute {
    pub intent: WorkSessionChatIntent,
    pub confidence: f32,
    pub target: Option<WorkSessionExecutionTarget>,
    pub query: Option<String>,
    pub reason_code: Option<String>,
}

#[allow(dead_code)]
pub fn parse_work_session_chat_intent(value: &str) -> WorkSessionChatIntent {
    let normalized = normalize_identifier(value);
    match normalized.as_str() {
        "startsession" | "start_session" => WorkSessionChatIntent::StartSession,
        "stopsession" | "stop_session" => WorkSessionChatIntent::StopSession,
        "stopandgeneraterecap" | "stop_and_generate_recap" => {
            WorkSessionChatIntent::StopAndGenerateRecap
        }
        "attachscreencontext" | "attach_screen_context" => {
            WorkSessionChatIntent::AttachScreenContext
        }
        "generateintelligence" | "generate_intelligence" | "generate_recap" => {
            WorkSessionChatIntent::GenerateIntelligence
        }
        "generatetranscriptsummary"
        | "generate_transcript_summary"
        | "transcript_summary"
        | "analyze_transcript" => WorkSessionChatIntent::GenerateTranscriptSummary,
        "generatedetails" | "generate_details" | "details" => {
            WorkSessionChatIntent::GenerateDetails
        }
        "generatetechnicalrecap" | "generate_technical_recap" => {
            WorkSessionChatIntent::GenerateTechnicalRecap
        }
        "generatefollowupdraft" | "generate_follow_up_draft" => {
            WorkSessionChatIntent::GenerateFollowUpDraft
        }
        "recallsessionmemory" | "recall_session_memory" => {
            WorkSessionChatIntent::RecallSessionMemory
        }
        "searchsessionmemory" | "search_session_memory" => {
            WorkSessionChatIntent::SearchSessionMemory
        }
        "showevidence" | "show_evidence" => WorkSessionChatIntent::ShowEvidence,
        "showsessionstatus" | "show_session_status" => WorkSessionChatIntent::ShowSessionStatus,
        "openmeetingpanel" | "open_meeting_panel" => WorkSessionChatIntent::OpenMeetingPanel,
        _ => WorkSessionChatIntent::Unknown,
    }
}

fn normalize_identifier(value: &str) -> String {
    value
        .trim()
        .chars()
        .flat_map(char::to_lowercase)
        .map(|ch| match ch {
            '-' | ' ' => '_',
            _ => ch,
        })
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .collect()
}

pub fn parse_work_session_target_kind(value: &str) -> WorkSessionTargetKind {
    let normalized = normalize_identifier(value);
    match normalized.as_str() {
        "activesession" | "active_session" => WorkSessionTargetKind::ActiveSession,
        "lastreferencedsession" | "last_referenced_session" => {
            WorkSessionTargetKind::LastReferencedSession
        }
        "latestarchivedsession" | "latest_archived_session" => {
            WorkSessionTargetKind::LatestArchivedSession
        }
        "lastcompletedsession" | "last_completed_session" => {
            WorkSessionTargetKind::LastCompletedSession
        }
        "currentscreen" | "current_screen" => WorkSessionTargetKind::CurrentScreen,
        "archivedsessions" | "archived_sessions" => WorkSessionTargetKind::ArchivedSessions,
        "none" | "" => WorkSessionTargetKind::None,
        _ => WorkSessionTargetKind::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_schema_intent_names_without_natural_language_classification() {
        assert_eq!(
            parse_work_session_chat_intent("generate_transcript_summary"),
            WorkSessionChatIntent::GenerateTranscriptSummary
        );
        assert_eq!(
            parse_work_session_chat_intent("stop-and-generate-recap"),
            WorkSessionChatIntent::StopAndGenerateRecap
        );
        assert_eq!(
            parse_work_session_chat_intent("show_evidence"),
            WorkSessionChatIntent::ShowEvidence
        );
    }

    #[test]
    fn natural_language_is_not_classified_in_work_session_schema_module() {
        assert_eq!(
            parse_work_session_chat_intent("mi fai un recap dell ultima sessione"),
            WorkSessionChatIntent::Unknown
        );
        assert_eq!(
            parse_work_session_chat_intent("di cosa abbiamo parlato nell ultima registrazione"),
            WorkSessionChatIntent::Unknown
        );
        assert_eq!(
            parse_work_session_chat_intent("mi dai piu dettagli al riguardo"),
            WorkSessionChatIntent::Unknown
        );
    }

    #[test]
    fn maps_work_session_intents_to_governed_tool_names() {
        assert_eq!(
            WorkSessionChatIntent::StartSession.primary_tool_name(),
            Some("meeting.session.start")
        );
        assert_eq!(
            WorkSessionChatIntent::AttachScreenContext.primary_tool_name(),
            Some("meeting.screen_context.attach_current")
        );
        assert_eq!(
            WorkSessionChatIntent::RecallSessionMemory.primary_tool_name(),
            Some("meeting.recall.answer")
        );
        assert_eq!(
            WorkSessionChatIntent::GenerateFollowUpDraft.primary_tool_name(),
            Some("meeting.followup.draft")
        );
        assert_eq!(
            WorkSessionChatIntent::GenerateTranscriptSummary.primary_tool_name(),
            Some("meeting.session.read")
        );
    }
}
