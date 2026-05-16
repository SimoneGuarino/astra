//! Transcription stream - buffers audio frames but does not fabricate STT output.

use super::{
    stt_adapter::{ExistingSttClientMeetingAdapter, MeetingSttEngine},
    types::*,
};
use chrono::Utc;
use std::collections::VecDeque;

pub struct TranscriptionConfig {
    pub model: String,
    pub language: String,
    pub vad_threshold: f32,
    pub silence_threshold_ms: u64,
    pub max_utterance_length_ms: u64,
    pub use_diarization: bool,
}

pub struct TranscriptionStream {
    pub config: TranscriptionConfig,
    pub frame_buffer: VecDeque<AudioFrame>,
    pub last_speech_time: Option<chrono::DateTime<Utc>>,
    pub current_utterance: Vec<AudioFrame>,
    pub transcript: Vec<TranscriptEntry>,
    transcriber: ExistingSttClientMeetingAdapter,
}

impl TranscriptionStream {
    pub fn new(model: String, language: Option<String>) -> Self {
        Self {
            config: TranscriptionConfig {
                model,
                language: language.unwrap_or_else(|| "auto".to_string()),
                vad_threshold: 0.01,
                silence_threshold_ms: 1000,
                max_utterance_length_ms: 30000,
                use_diarization: false,
            },
            frame_buffer: VecDeque::new(),
            last_speech_time: None,
            current_utterance: Vec::new(),
            transcript: Vec::new(),
            transcriber: ExistingSttClientMeetingAdapter::new(),
        }
    }

    pub fn push_frame(&mut self, frame: AudioFrame) {
        self.frame_buffer.push_back(frame.clone());
        self.current_utterance.push(frame);
        while self.frame_buffer.len() > 256 {
            self.frame_buffer.pop_front();
        }
    }

    pub fn process_segment(
        &mut self,
        audio: &[AudioFrame],
        speaker: Option<String>,
    ) -> Result<Option<TranscriptEntry>, MeetingRuntimeError> {
        let segment = meeting_segment_from_audio_frames(audio);
        self.transcriber.transcribe_segment(&segment, speaker)
    }

    pub fn finalize_utterance(&mut self) -> Result<Option<TranscriptEntry>, MeetingRuntimeError> {
        let audio = std::mem::take(&mut self.current_utterance);
        self.last_speech_time = Some(Utc::now());
        self.process_segment(&audio, None)
    }

    pub fn get_transcript(&self) -> &[TranscriptEntry] {
        &self.transcript
    }

    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.current_utterance.clear();
        self.last_speech_time = None;
        self.transcript.clear();
    }
}

fn meeting_segment_from_audio_frames(audio: &[AudioFrame]) -> MeetingAudioSegment {
    let chunks = audio
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            let sample_rate = 16_000;
            let frame_count = frame.samples.len();
            let duration_ms = ((frame_count as u64) * 1_000) / sample_rate as u64;
            MeetingAudioChunk {
                sample_rate,
                channels: 1,
                format: AudioSampleFormat::F32Pcm,
                monotonic_timestamp_ms: frame.timestamp.timestamp_millis().max(0) as u64,
                sequence_number: index as u64,
                duration_ms,
                source_backend: CaptureBackend::Default,
                transcript_source: TranscriptSource::Unknown,
                byte_length: frame_count * std::mem::size_of::<f32>(),
                frame_count,
            }
        })
        .collect::<Vec<_>>();
    let total_duration_ms = chunks.iter().map(|chunk| chunk.duration_ms).sum();

    MeetingAudioSegment {
        session_id: None,
        chunks,
        total_duration_ms,
        source_backend: CaptureBackend::Default,
        transcript_source: TranscriptSource::Unknown,
        contains_raw_audio: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_transcription_is_not_returned_as_real_text() {
        let mut stream = TranscriptionStream::new("local".to_string(), None);
        let frame = AudioFrame {
            timestamp: Utc::now(),
            samples: vec![0.1, 0.2],
            device_id: "test".to_string(),
        };
        let result = stream.process_segment(&[frame], None);
        assert!(result.is_err());
        assert!(stream.get_transcript().is_empty());
    }
}
