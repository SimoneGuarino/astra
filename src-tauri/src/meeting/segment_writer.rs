//! Bounded WAV segment writer for managed meeting audio.
//!
//! This module does not capture OS audio. It only writes already-captured PCM
//! samples into generated, managed `.wav` segment files that can be passed to
//! the existing file-based STT bridge.

use super::types::{CaptureBackend, MeetingRuntimeError, TranscriptSource};
use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};
use uuid::Uuid;

pub const DEFAULT_SEGMENT_DURATION_MS: u64 = 15_000;
pub const DEFAULT_MAX_SEGMENT_BYTES: u64 = 50 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentWriterConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub max_duration_ms: u64,
    pub max_bytes: u64,
    pub source_backend: CaptureBackend,
    pub transcript_source: TranscriptSource,
}

impl Default for SegmentWriterConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            channels: 1,
            max_duration_ms: DEFAULT_SEGMENT_DURATION_MS,
            max_bytes: DEFAULT_MAX_SEGMENT_BYTES,
            source_backend: CaptureBackend::Wasapi,
            transcript_source: TranscriptSource::SystemAudio,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedMeetingSegment {
    pub session_id: String,
    pub path: PathBuf,
    pub sequence_number: u64,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub duration_ms: u64,
    pub byte_length: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub source_backend: CaptureBackend,
    pub transcript_source: TranscriptSource,
    pub source_path_redacted: bool,
    pub managed_path_redacted: bool,
    pub capture_metrics_recorded: bool,
}

#[derive(Debug, Clone)]
pub struct SegmentWriter {
    meeting_storage_dir: PathBuf,
    config: SegmentWriterConfig,
}

impl SegmentWriter {
    pub fn new(meeting_storage_dir: PathBuf, config: SegmentWriterConfig) -> Self {
        Self {
            meeting_storage_dir,
            config,
        }
    }

    pub fn write_pcm_i16_segment(
        &self,
        session_id: &str,
        samples: &[i16],
    ) -> Result<CapturedMeetingSegment, MeetingRuntimeError> {
        let session_id = session_id.trim();
        if !is_safe_session_id(session_id) {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "managed segment session_id is invalid".to_string(),
            });
        }
        if self.config.sample_rate == 0 || self.config.channels == 0 {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "segment writer sample_rate and channels are required".to_string(),
            });
        }
        let channels = usize::from(self.config.channels);
        if samples.is_empty() || !samples.len().is_multiple_of(channels) {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "segment samples must contain complete audio frames".to_string(),
            });
        }

        let data_bytes = u64::try_from(samples.len())
            .map_err(|_| MeetingRuntimeError::SegmentWriteFailed {
                reason: "segment sample count overflow".to_string(),
            })?
            .checked_mul(2)
            .ok_or_else(|| MeetingRuntimeError::SegmentWriteFailed {
                reason: "segment byte count overflow".to_string(),
            })?;
        let wav_bytes = data_bytes.checked_add(WAV_HEADER_BYTES).ok_or_else(|| {
            MeetingRuntimeError::SegmentWriteFailed {
                reason: "segment WAV byte count overflow".to_string(),
            }
        })?;
        if wav_bytes > self.config.max_bytes {
            return Err(MeetingRuntimeError::SegmentTooLarge {
                max_bytes: self.config.max_bytes,
                actual_bytes: wav_bytes,
            });
        }

        let frame_count = u64::try_from(samples.len() / channels).map_err(|_| {
            MeetingRuntimeError::SegmentWriteFailed {
                reason: "segment frame count overflow".to_string(),
            }
        })?;
        let duration_ms = frame_count.checked_mul(1_000).ok_or_else(|| {
            MeetingRuntimeError::SegmentWriteFailed {
                reason: "segment duration overflow".to_string(),
            }
        })? / u64::from(self.config.sample_rate);
        if duration_ms > self.config.max_duration_ms {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: format!(
                    "segment duration exceeds {} ms",
                    self.config.max_duration_ms
                ),
            });
        }

        let segment_dir = self.segments_dir(session_id);
        std::fs::create_dir_all(&segment_dir).map_err(|error| {
            MeetingRuntimeError::StorageError {
                message: format!(
                    "create managed meeting segment directory failed: {}",
                    error.kind()
                ),
            }
        })?;

        let segment_id = Uuid::new_v4();
        let final_path = segment_dir.join(format!("{segment_id}.wav"));
        let temp_path = segment_dir.join(format!("{segment_id}.wav.tmp"));
        let write_result = write_wav_i16(
            &temp_path,
            self.config.sample_rate,
            self.config.channels,
            samples,
            data_bytes,
        );
        if let Err(error) = write_result {
            let _ = std::fs::remove_file(&temp_path);
            return Err(error);
        }
        std::fs::rename(&temp_path, &final_path).map_err(|error| {
            let _ = std::fs::remove_file(&temp_path);
            MeetingRuntimeError::StorageError {
                message: format!("finalize managed meeting segment failed: {}", error.kind()),
            }
        })?;

        Ok(CapturedMeetingSegment {
            session_id: session_id.to_string(),
            path: final_path,
            sequence_number: 0,
            start_ms: None,
            end_ms: None,
            duration_ms,
            byte_length: wav_bytes,
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            source_backend: self.config.source_backend,
            transcript_source: self.config.transcript_source,
            source_path_redacted: true,
            managed_path_redacted: true,
            capture_metrics_recorded: false,
        })
    }

    pub fn segments_dir(&self, session_id: &str) -> PathBuf {
        self.meeting_storage_dir.join(session_id).join("segments")
    }
}

const WAV_HEADER_BYTES: u64 = 44;

fn write_wav_i16(
    path: &Path,
    sample_rate: u32,
    channels: u16,
    samples: &[i16],
    data_bytes: u64,
) -> Result<(), MeetingRuntimeError> {
    let data_bytes =
        u32::try_from(data_bytes).map_err(|_| MeetingRuntimeError::SegmentTooLarge {
            max_bytes: u64::from(u32::MAX),
            actual_bytes: data_bytes,
        })?;
    let chunk_size =
        36_u32
            .checked_add(data_bytes)
            .ok_or(MeetingRuntimeError::SegmentTooLarge {
                max_bytes: u64::from(u32::MAX),
                actual_bytes: u64::from(data_bytes) + 36,
            })?;
    let byte_rate = sample_rate
        .checked_mul(u32::from(channels))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| MeetingRuntimeError::SegmentWriteFailed {
            reason: "segment WAV byte_rate overflow".to_string(),
        })?;
    let block_align =
        channels
            .checked_mul(2)
            .ok_or_else(|| MeetingRuntimeError::SegmentWriteFailed {
                reason: "segment WAV block_align overflow".to_string(),
            })?;

    let mut file = File::create(path).map_err(|error| MeetingRuntimeError::StorageError {
        message: format!("create managed meeting segment failed: {}", error.kind()),
    })?;
    file.write_all(b"RIFF")
        .and_then(|()| file.write_all(&chunk_size.to_le_bytes()))
        .and_then(|()| file.write_all(b"WAVE"))
        .and_then(|()| file.write_all(b"fmt "))
        .and_then(|()| file.write_all(&16_u32.to_le_bytes()))
        .and_then(|()| file.write_all(&1_u16.to_le_bytes()))
        .and_then(|()| file.write_all(&channels.to_le_bytes()))
        .and_then(|()| file.write_all(&sample_rate.to_le_bytes()))
        .and_then(|()| file.write_all(&byte_rate.to_le_bytes()))
        .and_then(|()| file.write_all(&block_align.to_le_bytes()))
        .and_then(|()| file.write_all(&16_u16.to_le_bytes()))
        .and_then(|()| file.write_all(b"data"))
        .and_then(|()| file.write_all(&data_bytes.to_le_bytes()))
        .map_err(|error| MeetingRuntimeError::StorageError {
            message: format!(
                "write managed meeting segment header failed: {}",
                error.kind()
            ),
        })?;
    for sample in samples {
        file.write_all(&sample.to_le_bytes()).map_err(|error| {
            MeetingRuntimeError::StorageError {
                message: format!(
                    "write managed meeting segment samples failed: {}",
                    error.kind()
                ),
            }
        })?;
    }
    file.sync_all()
        .map_err(|error| MeetingRuntimeError::StorageError {
            message: format!("sync managed meeting segment failed: {}", error.kind()),
        })
}

fn is_safe_session_id(session_id: &str) -> bool {
    !session_id.is_empty()
        && session_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_meeting_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("astra_segment_writer_{name}_{}", Uuid::new_v4()))
    }

    #[test]
    fn segment_writer_writes_valid_wav_header() {
        let root = temp_meeting_dir("header");
        let writer = SegmentWriter::new(root.clone(), SegmentWriterConfig::default());

        let segment = writer
            .write_pcm_i16_segment("session_1", &[0, 100, -100, 0])
            .expect("write segment");
        let header = std::fs::read(&segment.path).expect("read segment");

        assert_eq!(&header[0..4], b"RIFF");
        assert_eq!(&header[8..12], b"WAVE");
        assert_eq!(&header[12..16], b"fmt ");
        assert_eq!(&header[36..40], b"data");
        assert!(segment.path.starts_with(root));
        assert_eq!(segment.byte_length, 52);
        assert_eq!(segment.transcript_source, TranscriptSource::SystemAudio);
    }

    #[test]
    fn segment_writer_rejects_oversized_segment() {
        let root = temp_meeting_dir("too_large");
        let writer = SegmentWriter::new(
            root,
            SegmentWriterConfig {
                max_bytes: 45,
                ..SegmentWriterConfig::default()
            },
        );

        let result = writer.write_pcm_i16_segment("session_1", &[0]);

        assert!(matches!(
            result,
            Err(MeetingRuntimeError::SegmentTooLarge { .. })
        ));
    }
}
