#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkSessionChatIntent {
    StartSession,
    StopSession,
    StopAndGenerateRecap,
    AttachScreenContext,
    GenerateIntelligence,
    GenerateTechnicalRecap,
    GenerateFollowUpDraft,
    RecallSessionMemory,
    SearchSessionMemory,
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
            Self::GenerateTechnicalRecap => "generate_technical_recap",
            Self::GenerateFollowUpDraft => "generate_follow_up_draft",
            Self::RecallSessionMemory => "recall_session_memory",
            Self::SearchSessionMemory => "search_session_memory",
            Self::ShowSessionStatus => "show_session_status",
            Self::OpenMeetingPanel => "open_meeting_panel",
            Self::Unknown => "unknown",
        }
    }

    pub fn primary_tool_name(self) -> Option<&'static str> {
        match self {
            Self::StartSession => Some("meeting.session.start"),
            Self::StopSession | Self::StopAndGenerateRecap => Some("meeting.session.stop"),
            Self::AttachScreenContext => Some("meeting.screen_context.attach_current"),
            Self::GenerateIntelligence | Self::GenerateTechnicalRecap => {
                Some("meeting.intelligence.generate")
            }
            Self::GenerateFollowUpDraft => Some("meeting.followup.draft"),
            Self::RecallSessionMemory => Some("meeting.recall.answer"),
            Self::SearchSessionMemory => Some("meeting.session.search"),
            Self::ShowSessionStatus | Self::OpenMeetingPanel => Some("meeting.session.read"),
            Self::Unknown => None,
        }
    }

    pub fn is_actionable(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkSessionChatRoute {
    pub intent: WorkSessionChatIntent,
    pub confidence: f32,
}

pub fn classify_work_session_chat_intent(message: &str) -> WorkSessionChatRoute {
    let normalized = normalize(message);
    let intent = classify_normalized_work_session_intent(&normalized);
    let confidence = if intent.is_actionable() { 0.86 } else { 0.0 };
    WorkSessionChatRoute { intent, confidence }
}

fn classify_normalized_work_session_intent(normalized: &str) -> WorkSessionChatIntent {
    if normalized.is_empty() {
        return WorkSessionChatIntent::Unknown;
    }

    let mentions_session = contains_any(
        normalized,
        &[
            "sessione",
            "work session",
            "session",
            "meeting",
            "registrazione",
            "recap",
            "schermo",
            "screen",
            "transcript",
            "trascritti",
            "segmenti",
        ],
    );

    if contains_any(
        normalized,
        &[
            "stoppa la sessione e genera il recap",
            "ferma la sessione e genera il recap",
            "stop session and generate recap",
            "stop the session and generate recap",
        ],
    ) || ((contains_any(normalized, &["stoppa", "ferma", "stop"]))
        && contains_any(normalized, &["recap", "summary", "riepilogo"]))
    {
        return WorkSessionChatIntent::StopAndGenerateRecap;
    }

    if contains_any(
        normalized,
        &[
            "allega lo schermo",
            "allega quello che sto guardando",
            "salva quello che sto guardando",
            "attach current screen",
            "attach the screen",
            "save what i am looking at",
            "save what i'm looking at",
        ],
    ) {
        return WorkSessionChatIntent::AttachScreenContext;
    }

    if contains_any(
        normalized,
        &[
            "cosa avevamo deciso",
            "cosa abbiamo deciso",
            "quando abbiamo parlato",
            "quando parlavamo",
            "cosa stavamo guardando",
            "cosa c era sullo schermo",
            "cosa c'era sullo schermo",
            "what did we decide",
            "when did we talk",
            "when were we discussing",
            "what were we looking at",
            "what was on screen",
        ],
    ) {
        return WorkSessionChatIntent::RecallSessionMemory;
    }

    if contains_any(
        normalized,
        &[
            "cerca nelle sessioni",
            "cerca nella memoria",
            "search session memory",
            "search sessions",
            "find in work sessions",
        ],
    ) {
        return WorkSessionChatIntent::SearchSessionMemory;
    }

    if contains_any(
        normalized,
        &[
            "la sessione e attiva",
            "la sessione è attiva",
            "stato della sessione",
            "quanti segmenti mancano",
            "segmenti mancano",
            "is the session active",
            "session status",
            "how many segments",
        ],
    ) {
        return WorkSessionChatIntent::ShowSessionStatus;
    }

    if contains_any(
        normalized,
        &[
            "fammi una bozza di follow up",
            "bozza di follow up",
            "follow up draft",
            "follow-up draft",
        ],
    ) {
        return WorkSessionChatIntent::GenerateFollowUpDraft;
    }

    if contains_any(
        normalized,
        &[
            "recap tecnico",
            "riepilogo tecnico",
            "technical recap",
            "technical summary",
        ],
    ) {
        return WorkSessionChatIntent::GenerateTechnicalRecap;
    }

    if contains_any(
        normalized,
        &[
            "genera il recap",
            "generami il recap",
            "genera riassunto",
            "genera meeting intelligence",
            "generate recap",
            "generate summary",
        ],
    ) {
        return WorkSessionChatIntent::GenerateIntelligence;
    }

    if contains_any(
        normalized,
        &[
            "stoppa la sessione",
            "ferma la sessione",
            "ferma la registrazione",
            "stop the session",
            "stop work session",
            "stop recording",
        ],
    ) {
        return WorkSessionChatIntent::StopSession;
    }

    if contains_any(
        normalized,
        &[
            "avvia una sessione",
            "inizia una sessione",
            "inizia una sessione di lavoro",
            "avvia una sessione di lavoro",
            "prendi appunti",
            "start a work session",
            "start work session",
            "start meeting session",
            "take notes",
        ],
    ) {
        return WorkSessionChatIntent::StartSession;
    }

    if mentions_session
        && contains_any(
            normalized,
            &[
                "apri dettagli",
                "apri pannello",
                "open details",
                "open meeting panel",
            ],
        )
    {
        return WorkSessionChatIntent::OpenMeetingPanel;
    }

    WorkSessionChatIntent::Unknown
}

fn normalize(message: &str) -> String {
    message
        .chars()
        .flat_map(char::to_lowercase)
        .map(|ch| {
            if ch.is_alphanumeric() || ch == '\'' || ch == 'à' || ch == 'è' || ch == 'é' {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_italian_work_session_phrases() {
        assert_eq!(
            classify_work_session_chat_intent("Avvia una sessione di lavoro").intent,
            WorkSessionChatIntent::StartSession
        );
        assert_eq!(
            classify_work_session_chat_intent("Stoppa la sessione").intent,
            WorkSessionChatIntent::StopSession
        );
        assert_eq!(
            classify_work_session_chat_intent("Allega quello che sto guardando alla sessione")
                .intent,
            WorkSessionChatIntent::AttachScreenContext
        );
        assert_eq!(
            classify_work_session_chat_intent("Cosa avevamo deciso sullo STT drain?").intent,
            WorkSessionChatIntent::RecallSessionMemory
        );
        assert_eq!(
            classify_work_session_chat_intent("La sessione è attiva?").intent,
            WorkSessionChatIntent::ShowSessionStatus
        );
        assert_eq!(
            classify_work_session_chat_intent("Stoppa la sessione e genera il recap").intent,
            WorkSessionChatIntent::StopAndGenerateRecap
        );
        assert_eq!(
            classify_work_session_chat_intent("Fammi una bozza di follow up").intent,
            WorkSessionChatIntent::GenerateFollowUpDraft
        );
        assert_eq!(
            classify_work_session_chat_intent("Generami il recap tecnico").intent,
            WorkSessionChatIntent::GenerateTechnicalRecap
        );
    }

    #[test]
    fn classifies_english_work_session_phrases() {
        assert_eq!(
            classify_work_session_chat_intent("start a work session").intent,
            WorkSessionChatIntent::StartSession
        );
        assert_eq!(
            classify_work_session_chat_intent("stop the session").intent,
            WorkSessionChatIntent::StopSession
        );
        assert_eq!(
            classify_work_session_chat_intent("attach current screen").intent,
            WorkSessionChatIntent::AttachScreenContext
        );
        assert_eq!(
            classify_work_session_chat_intent("what did we decide about STT drain?").intent,
            WorkSessionChatIntent::RecallSessionMemory
        );
        assert_eq!(
            classify_work_session_chat_intent("is the session active?").intent,
            WorkSessionChatIntent::ShowSessionStatus
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
            WorkSessionChatIntent::ShowSessionStatus.primary_tool_name(),
            Some("meeting.session.read")
        );
    }

    #[test]
    fn unknown_meeting_like_query_does_not_trigger_action() {
        assert_eq!(
            classify_work_session_chat_intent("clicca nel meeting e apri il browser").intent,
            WorkSessionChatIntent::Unknown
        );
    }
}
