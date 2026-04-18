use serde::Serialize;
use std::{
    collections::VecDeque,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use uuid::Uuid;

use crate::vad::{
    normalize_samples, rms, samples_duration_ms, AdaptiveEnergyVad, VadEvent, VadFrameSnapshot,
};

const FOLLOW_UP_WINDOW: Duration = Duration::from_secs(18);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum VoiceSessionMode {
    Passive,
    Conversation,
}

impl VoiceSessionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Passive => "passive",
            Self::Conversation => "conversation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum VoiceTurnState {
    Disabled,
    Passive,
    Armed,
    Listening,
    Processing,
    Speaking,
    Interrupted,
    Cooldown,
}

impl VoiceTurnState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Passive => "passive",
            Self::Armed => "armed",
            Self::Listening => "listening",
            Self::Processing => "processing",
            Self::Speaking => "speaking",
            Self::Interrupted => "interrupted",
            Self::Cooldown => "cooldown",
        }
    }
}

#[derive(Clone)]
pub struct VoiceSessionManager {
    recordings_dir: PathBuf,
    inner: Arc<Mutex<VoiceSessionState>>,
}

#[derive(Debug)]
struct VoiceSessionState {
    session_id: Option<String>,
    mode: VoiceSessionMode,
    turn_state: VoiceTurnState,
    processing_stage: VoiceProcessingStage,
    active_until: Option<Instant>,
    vad: AdaptiveEnergyVad,
    last_vad: VadFrameSnapshot,
    pre_roll: VecDeque<f32>,
    utterance: Vec<f32>,
    current_turn_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VoiceProcessingStage {
    None,
    Stt,
    Response,
}

#[derive(Debug)]
pub enum VoiceSessionAction {
    None,
    StateChanged(VoiceSessionSnapshot),
    BargeIn(VoiceSessionSnapshot),
    UtteranceReady(VoiceUtterance),
}

#[derive(Debug, Clone)]
pub struct VoiceSessionSnapshot {
    pub session_id: Option<String>,
    pub turn_id: Option<String>,
    pub mode: VoiceSessionMode,
    pub state: VoiceTurnState,
    pub reason: String,
    pub conversation_expires_in_ms: Option<u128>,
    pub vad: VadFrameSnapshot,
}

#[derive(Debug)]
pub struct VoiceUtterance {
    pub session_id: String,
    pub turn_id: String,
    pub path: PathBuf,
    pub snapshot: VoiceSessionSnapshot,
}

#[derive(Debug)]
pub enum TranscriptDecision {
    Ignore {
        session_id: String,
        turn_id: String,
        text: String,
        reason: String,
        snapshot: VoiceSessionSnapshot,
    },
    Arm {
        session_id: String,
        turn_id: String,
        text: String,
        reason: String,
        snapshot: VoiceSessionSnapshot,
    },
    Respond {
        session_id: String,
        turn_id: String,
        text: String,
        response_text: String,
        reason: String,
        snapshot: VoiceSessionSnapshot,
    },
}

impl VoiceSessionManager {
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            recordings_dir: project_root
                .join("python_services")
                .join("stt")
                .join("recordings"),
            inner: Arc::new(Mutex::new(VoiceSessionState {
                session_id: None,
                mode: VoiceSessionMode::Passive,
                turn_state: VoiceTurnState::Disabled,
                processing_stage: VoiceProcessingStage::None,
                active_until: None,
                vad: AdaptiveEnergyVad::default(),
                last_vad: VadFrameSnapshot::default(),
                pre_roll: VecDeque::new(),
                utterance: Vec::new(),
                current_turn_id: None,
            })),
        }
    }

    pub fn start(&self) -> VoiceSessionSnapshot {
        let mut state = self.inner.lock().expect("voice session mutex poisoned");
        state.session_id = Some(Uuid::new_v4().to_string());
        state.mode = VoiceSessionMode::Passive;
        state.turn_state = VoiceTurnState::Passive;
        state.processing_stage = VoiceProcessingStage::None;
        state.active_until = None;
        state.reset_audio_state();
        state.snapshot("session_started")
    }

    pub fn stop(&self) -> VoiceSessionSnapshot {
        let mut state = self.inner.lock().expect("voice session mutex poisoned");
        state.session_id = None;
        state.mode = VoiceSessionMode::Passive;
        state.turn_state = VoiceTurnState::Disabled;
        state.processing_stage = VoiceProcessingStage::None;
        state.active_until = None;
        state.reset_audio_state();
        state.snapshot("session_stopped")
    }

    pub fn process_audio_chunk(
        &self,
        session_id: &str,
        sample_rate: u32,
        samples: &[f32],
    ) -> VoiceSessionAction {
        let mut state = self.inner.lock().expect("voice session mutex poisoned");
        if state.session_id.as_deref() != Some(session_id)
            || state.turn_state == VoiceTurnState::Disabled
        {
            return VoiceSessionAction::None;
        }

        state.expire_conversation_window();

        let normalized = normalize_samples(samples);
        state.push_pre_roll(&normalized, sample_rate);
        let rms = rms(&normalized);
        let duration_ms = samples_duration_ms(normalized.len(), sample_rate);
        let vad_update = state.vad.update(rms, duration_ms);
        state.last_vad = vad_update.snapshot;

        match vad_update.event {
            VadEvent::SpeechStarted => {
                state.current_turn_id = Some(Uuid::new_v4().to_string());
                state.utterance.clear();
                let pre_roll = state.pre_roll.iter().copied().collect::<Vec<_>>();
                state.utterance.extend_from_slice(&pre_roll);
                state.utterance.extend_from_slice(&normalized);

                let was_interrupting = matches!(
                    (state.turn_state, state.processing_stage),
                    (VoiceTurnState::Speaking, _)
                        | (VoiceTurnState::Processing, VoiceProcessingStage::Response)
                );
                state.turn_state = if was_interrupting {
                    VoiceTurnState::Interrupted
                } else {
                    VoiceTurnState::Listening
                };
                state.processing_stage = VoiceProcessingStage::None;

                let snapshot = state.snapshot(if was_interrupting {
                    "barge_in"
                } else {
                    "speech_started"
                });
                if was_interrupting {
                    VoiceSessionAction::BargeIn(snapshot)
                } else {
                    VoiceSessionAction::StateChanged(snapshot)
                }
            }
            VadEvent::SpeechContinued => {
                if !state.utterance.is_empty() {
                    state.utterance.extend_from_slice(&normalized);
                }
                VoiceSessionAction::None
            }
            VadEvent::SpeechEnded => {
                if !state.utterance.is_empty() {
                    state.utterance.extend_from_slice(&normalized);
                }

                let config = state.vad.config();
                let utterance_ms =
                    samples_duration_ms(state.utterance.len(), config.target_sample_rate);
                if utterance_ms < config.min_utterance_ms {
                    state.reset_audio_state();
                    state.turn_state = state.passive_or_armed_state();
                    return VoiceSessionAction::StateChanged(
                        state.snapshot("short_utterance_ignored"),
                    );
                }

                let Some(session_id) = state.session_id.clone() else {
                    state.reset_audio_state();
                    return VoiceSessionAction::None;
                };
                let turn_id = state
                    .current_turn_id
                    .clone()
                    .unwrap_or_else(|| Uuid::new_v4().to_string());
                let path = self
                    .recordings_dir
                    .join(format!("voice_{session_id}_{turn_id}.wav"));
                let utterance = std::mem::take(&mut state.utterance);
                state.turn_state = VoiceTurnState::Processing;
                state.processing_stage = VoiceProcessingStage::Stt;
                state.vad.reset();
                let snapshot = state.snapshot("utterance_ready");
                state.current_turn_id = None;

                drop(state);
                if let Err(error) = write_wav_mono_f32(&path, config.target_sample_rate, &utterance)
                {
                    eprintln!(
                        "{}",
                        serde_json::json!({
                            "type": "voice_session",
                            "event": "utterance_write_failed",
                            "path": path.display().to_string(),
                            "error": error,
                        })
                    );
                    return VoiceSessionAction::StateChanged(snapshot);
                }

                VoiceSessionAction::UtteranceReady(VoiceUtterance {
                    session_id,
                    turn_id,
                    path,
                    snapshot,
                })
            }
            VadEvent::None => VoiceSessionAction::None,
        }
    }

    pub fn decide_transcript(
        &self,
        session_id: &str,
        turn_id: &str,
        transcript: &str,
    ) -> TranscriptDecision {
        let mut state = self.inner.lock().expect("voice session mutex poisoned");
        let text = transcript.trim().to_string();

        if state.session_id.as_deref() != Some(session_id) {
            return TranscriptDecision::Ignore {
                session_id: session_id.to_string(),
                turn_id: turn_id.to_string(),
                text,
                reason: "stale_session".to_string(),
                snapshot: state.snapshot("stale_session"),
            };
        }

        if text.is_empty() {
            state.turn_state = state.passive_or_armed_state();
            state.processing_stage = VoiceProcessingStage::None;
            return TranscriptDecision::Ignore {
                session_id: session_id.to_string(),
                turn_id: turn_id.to_string(),
                text,
                reason: "empty_transcript".to_string(),
                snapshot: state.snapshot("empty_transcript"),
            };
        }

        state.expire_conversation_window();
        let active_conversation = state.is_conversation_active();
        let addressed = addressing_decision(&text, active_conversation);

        match addressed {
            AddressingDecision::Ignored(reason) => {
                state.turn_state = if active_conversation {
                    VoiceTurnState::Armed
                } else {
                    VoiceTurnState::Passive
                };
                state.processing_stage = VoiceProcessingStage::None;
                if !active_conversation {
                    state.mode = VoiceSessionMode::Passive;
                    state.active_until = None;
                }

                TranscriptDecision::Ignore {
                    session_id: session_id.to_string(),
                    turn_id: turn_id.to_string(),
                    text,
                    reason: reason.clone(),
                    snapshot: state.snapshot(&reason),
                }
            }
            AddressingDecision::WakeOnly => {
                state.activate_conversation();
                state.turn_state = VoiceTurnState::Armed;
                state.processing_stage = VoiceProcessingStage::None;
                TranscriptDecision::Arm {
                    session_id: session_id.to_string(),
                    turn_id: turn_id.to_string(),
                    text,
                    reason: "wake_word_detected".to_string(),
                    snapshot: state.snapshot("wake_word_detected"),
                }
            }
            AddressingDecision::Respond(response_text) => {
                state.activate_conversation();
                state.turn_state = VoiceTurnState::Processing;
                state.processing_stage = VoiceProcessingStage::Response;
                TranscriptDecision::Respond {
                    session_id: session_id.to_string(),
                    turn_id: turn_id.to_string(),
                    text,
                    response_text,
                    reason: if active_conversation {
                        "conversation_follow_up".to_string()
                    } else {
                        "wake_word_detected".to_string()
                    },
                    snapshot: state.snapshot("accepted"),
                }
            }
        }
    }

    pub fn mark_speaking(&self) -> VoiceSessionSnapshot {
        let mut state = self.inner.lock().expect("voice session mutex poisoned");
        if state.turn_state != VoiceTurnState::Disabled {
            state.turn_state = VoiceTurnState::Speaking;
            state.processing_stage = VoiceProcessingStage::None;
            state.activate_conversation();
        }
        state.snapshot("assistant_speaking")
    }

    pub fn mark_assistant_idle(&self) -> VoiceSessionSnapshot {
        let mut state = self.inner.lock().expect("voice session mutex poisoned");
        if state.turn_state != VoiceTurnState::Disabled {
            state.turn_state = VoiceTurnState::Cooldown;
            state.processing_stage = VoiceProcessingStage::None;
            state.activate_conversation();
            state.turn_state = state.passive_or_armed_state();
        }
        state.snapshot("assistant_idle")
    }
}

impl VoiceSessionState {
    fn reset_audio_state(&mut self) {
        self.vad.reset();
        self.processing_stage = VoiceProcessingStage::None;
        self.pre_roll.clear();
        self.utterance.clear();
        self.current_turn_id = None;
    }

    fn push_pre_roll(&mut self, samples: &[f32], sample_rate: u32) {
        let max_len = (sample_rate as u64 * self.vad.config().pre_roll_ms / 1000) as usize;
        for sample in samples {
            self.pre_roll.push_back(*sample);
        }
        while self.pre_roll.len() > max_len {
            self.pre_roll.pop_front();
        }
    }

    fn activate_conversation(&mut self) {
        self.mode = VoiceSessionMode::Conversation;
        self.active_until = Some(Instant::now() + FOLLOW_UP_WINDOW);
    }

    fn is_conversation_active(&self) -> bool {
        self.active_until
            .map(|deadline| Instant::now() <= deadline)
            .unwrap_or(false)
    }

    fn expire_conversation_window(&mut self) {
        if self
            .active_until
            .is_some_and(|deadline| Instant::now() > deadline)
        {
            self.mode = VoiceSessionMode::Passive;
            self.active_until = None;
            if matches!(
                self.turn_state,
                VoiceTurnState::Armed | VoiceTurnState::Cooldown
            ) {
                self.turn_state = VoiceTurnState::Passive;
            }
        }
    }

    fn passive_or_armed_state(&self) -> VoiceTurnState {
        if self.is_conversation_active() {
            VoiceTurnState::Armed
        } else {
            VoiceTurnState::Passive
        }
    }

    fn snapshot(&self, reason: &str) -> VoiceSessionSnapshot {
        VoiceSessionSnapshot {
            session_id: self.session_id.clone(),
            turn_id: self.current_turn_id.clone(),
            mode: self.mode,
            state: self.turn_state,
            reason: reason.to_string(),
            conversation_expires_in_ms: self.active_until.map(|deadline| {
                deadline
                    .saturating_duration_since(Instant::now())
                    .as_millis()
            }),
            vad: self.last_vad,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum AddressingDecision {
    Ignored(String),
    WakeOnly,
    Respond(String),
}

#[derive(Debug, Clone)]
struct TranscriptQuality {
    normalized: String,
    useful_token_count: usize,
    useful_char_count: usize,
    alphabetic_char_count: usize,
    repeated_run_len: usize,
}

fn addressing_decision(text: &str, active_conversation: bool) -> AddressingDecision {
    let trimmed = text.trim();
    let quality = transcript_quality(trimmed);

    if active_conversation {
        if is_transcript_useful_for_follow_up(&quality) {
            return AddressingDecision::Respond(trimmed.to_string());
        }
        return AddressingDecision::Ignored("low_information_follow_up".to_string());
    }

    let tokens = quality.normalized.split_whitespace().collect::<Vec<_>>();
    let Some(wake_match) = find_wake_match(&tokens) else {
        return AddressingDecision::Ignored("wake_word_required".to_string());
    };

    if wake_match.start > 1 && !tokens[..wake_match.start].iter().all(|token| is_wake_filler(token)) {
        return AddressingDecision::Ignored("wake_word_too_late".to_string());
    }

    let response_text = tokens
        .iter()
        .enumerate()
        .filter_map(|(token_index, token)| {
            if (wake_match.start..wake_match.end).contains(&token_index) {
                return None;
            }

            if token_index < wake_match.start && is_wake_filler(token) {
                return None;
            }

            Some(*token)
        })
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();

    if response_text.is_empty() {
        return AddressingDecision::WakeOnly;
    }

    let command_quality = transcript_quality(&response_text);
    if !is_transcript_useful_for_passive_command(&command_quality) {
        return AddressingDecision::WakeOnly;
    }

    AddressingDecision::Respond(response_text)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WakeMatch {
    start: usize,
    end: usize,
}

fn find_wake_match(tokens: &[&str]) -> Option<WakeMatch> {
    for (index, token) in tokens.iter().enumerate() {
        if is_wake_word(token) {
            return Some(WakeMatch {
                start: index,
                end: index + 1,
            });
        }

        if let Some(next) = tokens.get(index + 1) {
            let joined = format!("{token}{next}");
            if is_wake_word(&joined) {
                return Some(WakeMatch {
                    start: index,
                    end: index + 2,
                });
            }
        }
    }

    None
}

fn is_wake_word(token: &str) -> bool {
    let normalized = token.trim();
    if matches!(
        normalized,
        "astra" | "ashtra" | "astrea" | "austra" | "astro" | "astre" | "asra" | "estra" | "extra"
    ) {
        return true;
    }

    let plausible_stt_variant = normalized.starts_with("ast")
        || normalized.starts_with("asr")
        || normalized.starts_with("ash");

    plausible_stt_variant
        && (4..=6).contains(&normalized.chars().count())
        && edit_distance(normalized, "astra") <= 1
}

fn is_wake_filler(token: &str) -> bool {
    matches!(
        token,
        "hey" | "ehi" | "ciao" | "ok" | "okay" | "pronto" | "eh" | "allora"
    )
}

fn transcript_quality(text: &str) -> TranscriptQuality {
    let normalized = normalize_for_addressing(text);
    let useful_tokens = normalized
        .split_whitespace()
        .filter(|token| !is_low_information_token(token))
        .collect::<Vec<_>>();

    let alphabetic_char_count = normalized.chars().filter(|ch| ch.is_ascii_alphabetic()).count();
    let repeated_run_len = longest_repeated_char_run(&normalized);

    TranscriptQuality {
        useful_char_count: useful_tokens.iter().map(|token| token.chars().count()).sum(),
        useful_token_count: useful_tokens.len(),
        alphabetic_char_count,
        repeated_run_len,
        normalized,
    }
}

fn is_transcript_useful_for_follow_up(quality: &TranscriptQuality) -> bool {
    quality.useful_token_count >= 1
        && quality.useful_char_count >= 2
        && quality.alphabetic_char_count >= 2
        && quality.repeated_run_len < 5
}

fn is_transcript_useful_for_passive_command(quality: &TranscriptQuality) -> bool {
    quality.useful_token_count >= 1
        && quality.useful_char_count >= 3
        && quality.alphabetic_char_count >= 3
        && quality.repeated_run_len < 5
}

fn is_low_information_token(token: &str) -> bool {
    matches!(
        token,
        "uh"
            | "eh"
            | "em"
            | "mm"
            | "mmm"
            | "boh"
            | "cioe"
            | "allora"
            | "pronto"
            | "ok"
            | "okay"
            | "si"
            | "sì"
            | "no"
            | "ciao"
            | "astra"
            | "astro"
            | "extra"
    )
}

fn longest_repeated_char_run(value: &str) -> usize {
    let mut max_run = 0usize;
    let mut current_run = 0usize;
    let mut previous = None;

    for ch in value.chars().filter(|ch| !ch.is_whitespace()) {
        if Some(ch) == previous {
            current_run += 1;
        } else {
            current_run = 1;
            previous = Some(ch);
        }
        max_run = max_run.max(current_run);
    }

    max_run
}

fn normalize_for_addressing(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|ch| if ch.is_alphanumeric() || ch.is_whitespace() { ch } else { ' ' })
        .collect::<String>()
}

fn edit_distance(left: &str, right: &str) -> usize {
    let left_chars = left.chars().collect::<Vec<_>>();
    let right_chars = right.chars().collect::<Vec<_>>();
    let mut previous = (0..=right_chars.len()).collect::<Vec<_>>();
    let mut current = vec![0; right_chars.len() + 1];

    for (left_index, left_char) in left_chars.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_char) in right_chars.iter().enumerate() {
            let insertion = current[right_index] + 1;
            let deletion = previous[right_index + 1] + 1;
            let substitution = previous[right_index] + usize::from(left_char != right_char);
            current[right_index + 1] = insertion.min(deletion).min(substitution);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[right_chars.len()]
}

fn write_wav_mono_f32(path: &PathBuf, sample_rate: u32, samples: &[f32]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }

    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut writer = BufWriter::new(file);
    let data_len = samples.len() as u32 * 2;
    let riff_len = 36 + data_len;

    writer
        .write_all(b"RIFF")
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&riff_len.to_le_bytes())
        .map_err(|error| error.to_string())?;
    writer
        .write_all(b"WAVE")
        .map_err(|error| error.to_string())?;
    writer
        .write_all(b"fmt ")
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&16u32.to_le_bytes())
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&1u16.to_le_bytes())
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&1u16.to_le_bytes())
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&sample_rate.to_le_bytes())
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&(sample_rate * 2).to_le_bytes())
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&2u16.to_le_bytes())
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&16u16.to_le_bytes())
        .map_err(|error| error.to_string())?;
    writer
        .write_all(b"data")
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&data_len.to_le_bytes())
        .map_err(|error| error.to_string())?;

    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_all(&pcm.to_le_bytes())
            .map_err(|error| error.to_string())?;
    }

    writer.flush().map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{addressing_decision, is_wake_word, AddressingDecision};

    #[test]
    fn wake_word_with_command_responds() {
        assert_eq!(
            addressing_decision("Astra ciao come stai?", false),
            AddressingDecision::Respond("ciao come stai".to_string())
        );
    }

    #[test]
    fn greeting_before_wake_word_is_not_sent_to_llm() {
        assert_eq!(
            addressing_decision("ciao Astra come stai?", false),
            AddressingDecision::Respond("come stai".to_string())
        );
    }

    #[test]
    fn common_stt_wake_word_variants_are_accepted() {
        assert!(is_wake_word("astro"));
        assert!(is_wake_word("astre"));
        assert!(is_wake_word("extra"));
    }

    #[test]
    fn split_wake_word_transcript_responds() {
        assert_eq!(
            addressing_decision("A stra come stai?", false),
            AddressingDecision::Respond("come stai".to_string())
        );
    }

    #[test]
    fn passive_mode_ignores_unaddressed_speech() {
        assert_eq!(
            addressing_decision("sto parlando con un'altra persona", false),
            AddressingDecision::Ignored("wake_word_required".to_string())
        );
    }

    #[test]
    fn active_conversation_accepts_follow_up_without_wake_word() {
        assert_eq!(
            addressing_decision("e domani invece?", true),
            AddressingDecision::Respond("e domani invece?".to_string())
        );
    }

    #[test]
    fn active_conversation_rejects_low_information_garbage() {
        assert_eq!(
            addressing_decision("ehm", true),
            AddressingDecision::Ignored("low_information_follow_up".to_string())
        );
    }
}
