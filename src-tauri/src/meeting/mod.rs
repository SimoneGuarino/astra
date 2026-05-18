//! Governed Meeting Intelligence Engine.
//!
//! Supported today:
//! - governed manual sessions;
//! - scoped consent;
//! - lifecycle validation;
//! - manual transcript, action item, and decision entry;
//! - file-based transcription through the existing SttClient boundary;
//! - bounded managed WAV segment writing for future capture backends;
//! - Windows WASAPI loopback capture into managed WAV segments;
//! - persisted meeting export and confirmed clear-data;
//! - conservative call detection.
//!
//! Unsupported today:
//! - CoreAudio/PipeWire capture;
//! - live STT streaming protocol;
//! - diarization;
//! - live summarization;
//! - follow-up sending.
//!
//! Future real capture must go through the persistent CaptureController owned
//! by MeetingRuntime so capture can be stopped, paused, resumed, health-checked,
//! restarted with bounds, backpressured, and shut down cleanly.

pub mod action_item_tracker;
pub mod audio_capture;
pub mod audio_quality;
pub mod call_detector;
pub mod capture_controller;
pub mod decision_log;
pub mod follow_up_sender;
pub mod intelligence_engine;
pub mod live_summarizer;
pub mod note_organizer;
pub mod privacy_control;
pub mod runtime;
pub mod segment_writer;
pub mod session_memory;
pub mod session_registry;
pub mod speaker_diarization;
pub mod stt_adapter;
pub mod transcription_stream;
pub mod types;
pub mod wasapi_loopback;
