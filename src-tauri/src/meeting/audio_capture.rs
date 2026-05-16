//! Audio capture boundary for system-audio meeting capture.
//!
//! Windows WASAPI loopback is the first real backend. CoreAudio and PipeWire
//! remain typed unsupported. Captured audio is emitted only as generated,
//! managed WAV segments.

use super::{
    segment_writer::CapturedMeetingSegment,
    types::{
        CaptureBackend, CaptureMetrics, CaptureOverflowPolicy, CapturePipelineConfig,
        MeetingRuntimeError, TranscriptSource,
    },
};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::process::Command;
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};
use tokio::sync::Notify;

#[cfg(target_os = "windows")]
use super::wasapi_loopback;

pub type CapturedSegmentSender = Arc<CapturedSegmentQueue>;

const DEFAULT_CAPTURE_STOP_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Debug)]
pub struct CaptureMetricsReporter {
    inner: Arc<Mutex<CaptureMetrics>>,
}

impl CaptureMetricsReporter {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CaptureMetrics::default())),
        }
    }

    pub fn snapshot(&self) -> CaptureMetrics {
        self.inner
            .lock()
            .map(|metrics| metrics.clone())
            .unwrap_or_default()
    }

    pub fn record_segment_written(&self, byte_length: u64, duration_ms: u64) {
        self.with_metrics(|metrics| {
            metrics.segments_written = metrics.segments_written.saturating_add(1);
            metrics.last_successful_segment_at = Some(chrono::Utc::now());
            metrics.last_segment_status = Some(format!(
                "segment_written:{}:{}",
                duration_bucket(duration_ms),
                size_bucket(byte_length)
            ));
        });
    }

    pub fn record_segment_transcribed(&self) {
        self.with_metrics(|metrics| {
            metrics.segments_transcribed = metrics.segments_transcribed.saturating_add(1);
            metrics.chunks_transcribed = metrics.chunks_transcribed.saturating_add(1);
            metrics.segment_transcription_failures_consecutive = 0;
            metrics.last_segment_status = Some("segment_transcribed".to_string());
            metrics.last_successful_segment_at = Some(chrono::Utc::now());
            metrics.backpressure_active = false;
        });
    }

    pub fn record_segment_write_failure(&self, error_class: &str) {
        self.with_metrics(|metrics| {
            metrics.segment_write_failures = metrics.segment_write_failures.saturating_add(1);
            metrics.last_segment_status = Some(format!("segment_write_failed:{error_class}"));
        });
    }

    pub fn record_segment_transcription_failure(&self, error_class: &str) {
        self.with_metrics(|metrics| {
            metrics.segment_transcription_failures =
                metrics.segment_transcription_failures.saturating_add(1);
            metrics.segment_transcription_failures_total = metrics
                .segment_transcription_failures_total
                .saturating_add(1);
            metrics.segment_transcription_failures_consecutive = metrics
                .segment_transcription_failures_consecutive
                .saturating_add(1);
            metrics.last_segment_status =
                Some(format!("segment_transcription_failed:{error_class}"));
            metrics.last_segment_transcription_error_kind = Some(error_class.to_string());
            metrics.last_segment_transcription_failure_at = Some(chrono::Utc::now());
        });
    }

    pub fn record_consent_revoked(&self) {
        self.with_metrics(|metrics| {
            metrics.last_segment_status = Some("consent_revoked".to_string());
            metrics.backpressure_active = false;
        });
    }

    pub fn record_stop_timed_out(&self) {
        self.with_metrics(|metrics| {
            metrics.last_segment_status = Some("capture_stop_timed_out".to_string());
        });
    }

    pub fn record_wasapi_endpoint_acquired(&self) {
        self.with_metrics(|metrics| {
            metrics.wasapi_endpoint_acquired = true;
        });
    }

    pub fn record_wasapi_mix_format(
        &self,
        sample_rate: u32,
        channel_count: u16,
        sample_format: &str,
    ) {
        self.with_metrics(|metrics| {
            metrics.wasapi_mix_format_detected = true;
            metrics.wasapi_sample_rate = Some(sample_rate);
            metrics.wasapi_channel_count = Some(channel_count);
            metrics.wasapi_sample_format = Some(sanitize_metric_token(sample_format));
        });
    }

    pub fn record_wasapi_buffer_frame_count(&self, frame_count: u32) {
        self.with_metrics(|metrics| {
            metrics.wasapi_buffer_frame_count = Some(frame_count);
        });
    }

    pub fn record_wasapi_stream_initialized(&self) {
        self.with_metrics(|metrics| {
            metrics.wasapi_stream_initialized = true;
        });
    }

    pub fn record_wasapi_stream_started(&self) {
        self.with_metrics(|metrics| {
            metrics.wasapi_stream_started = true;
        });
    }

    pub fn record_wasapi_packet(&self, input_frames: u64) {
        self.with_metrics(|metrics| {
            metrics.wasapi_packets_read = metrics.wasapi_packets_read.saturating_add(1);
            metrics.frames_captured = metrics.frames_captured.saturating_add(input_frames);
        });
    }

    pub fn record_audio_conversion(
        &self,
        output_frames: u64,
        clipped_samples: u64,
        peak_abs: u16,
        rms_bps: u16,
        normalization_gain_bps: u16,
    ) {
        self.with_metrics(|metrics| {
            metrics.frames_converted = metrics.frames_converted.saturating_add(output_frames);
            metrics.audio_clipped_sample_count = metrics
                .audio_clipped_sample_count
                .saturating_add(clipped_samples);
            metrics.audio_peak_abs = metrics.audio_peak_abs.max(peak_abs);
            metrics.audio_rms_bps = rms_bps;
            metrics.audio_normalization_gain_bps = normalization_gain_bps;
        });
    }

    pub fn record_vad_analysis(
        &self,
        speech_frames: u64,
        silence_frames: u64,
        speech_ratio_bps: u16,
        silence_ratio_bps: u16,
    ) {
        self.with_metrics(|metrics| {
            metrics.vad_speech_frames = metrics.vad_speech_frames.saturating_add(speech_frames);
            metrics.vad_silence_frames = metrics.vad_silence_frames.saturating_add(silence_frames);
            metrics.last_speech_ratio_bps = speech_ratio_bps;
            metrics.last_silence_ratio_bps = silence_ratio_bps;
        });
    }

    pub fn record_silence_segment_dropped(
        &self,
        frames_skipped: u64,
        speech_frames: u64,
        silence_frames: u64,
        speech_ratio_bps: u16,
        silence_ratio_bps: u16,
    ) {
        self.with_metrics(|metrics| {
            metrics.dropped_silence_segments = metrics.dropped_silence_segments.saturating_add(1);
            metrics.segments_dropped = metrics.segments_dropped.saturating_add(1);
            metrics.chunks_dropped = metrics.chunks_dropped.saturating_add(1);
            metrics.silence_frames_skipped = metrics
                .silence_frames_skipped
                .saturating_add(frames_skipped);
            metrics.vad_speech_frames = metrics.vad_speech_frames.saturating_add(speech_frames);
            metrics.vad_silence_frames = metrics.vad_silence_frames.saturating_add(silence_frames);
            metrics.last_speech_ratio_bps = speech_ratio_bps;
            metrics.last_silence_ratio_bps = silence_ratio_bps;
            metrics.last_segment_status = Some("segment_dropped:silence".to_string());
        });
    }

    pub fn record_backend_error(&self, error_kind: &str, message: &str) {
        self.with_metrics(|metrics| {
            metrics.last_backend_error_kind = Some(sanitize_metric_token(error_kind));
            metrics.last_backend_error_message = Some(sanitize_backend_message(message));
        });
    }

    pub fn record_queue_depth(&self, depth: usize, bytes_queued: u64, max_depth: usize) {
        self.with_metrics(|metrics| {
            metrics.bytes_queued = bytes_queued;
            metrics.max_queue_depth_seen = metrics.max_queue_depth_seen.max(depth);
            if depth < max_depth {
                metrics.backpressure_active = false;
            }
        });
    }

    pub fn record_segment_queued(&self) {
        self.with_metrics(|metrics| {
            metrics.segments_queued = metrics.segments_queued.saturating_add(1);
        });
    }

    pub fn record_queue_full(&self, policy: CaptureOverflowPolicy) {
        self.with_metrics(|metrics| {
            metrics.queue_full_events = metrics.queue_full_events.saturating_add(1);
            metrics.backpressure_active = true;
            metrics.last_overflow_policy_applied = Some(policy);
        });
    }

    pub fn record_segment_dropped(&self, policy: CaptureOverflowPolicy) {
        self.with_metrics(|metrics| {
            metrics.segments_dropped = metrics.segments_dropped.saturating_add(1);
            metrics.chunks_dropped = metrics.chunks_dropped.saturating_add(1);
            metrics.last_overflow_policy_applied = Some(policy);
            metrics.last_segment_status = Some(
                match policy {
                    CaptureOverflowPolicy::RejectNewest => "segment_dropped:reject_newest",
                    CaptureOverflowPolicy::DropOldestAndReport => {
                        "segment_dropped:drop_oldest_and_report"
                    }
                    CaptureOverflowPolicy::StopCapture => "segment_dropped:stop_capture",
                }
                .to_string(),
            );
        });
    }

    fn with_metrics(&self, update: impl FnOnce(&mut CaptureMetrics)) {
        if let Ok(mut metrics) = self.inner.lock() {
            update(&mut metrics);
        }
    }
}

impl Default for CaptureMetricsReporter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum CapturedSegmentEnqueueOutcome {
    Enqueued {
        depth: usize,
    },
    DroppedNewest {
        segment: CapturedMeetingSegment,
    },
    DroppedOldest {
        dropped_segment: CapturedMeetingSegment,
        depth: usize,
    },
    StopCapture {
        segment: CapturedMeetingSegment,
    },
}

#[derive(Debug)]
pub struct CapturedSegmentQueue {
    inner: Mutex<VecDeque<CapturedMeetingSegment>>,
    notify: Notify,
    max_depth: usize,
    overflow_policy: CaptureOverflowPolicy,
    metrics: CaptureMetricsReporter,
}

impl CapturedSegmentQueue {
    pub fn new(
        max_depth: usize,
        overflow_policy: CaptureOverflowPolicy,
        metrics: CaptureMetricsReporter,
    ) -> CapturedSegmentSender {
        Arc::new(Self {
            inner: Mutex::new(VecDeque::new()),
            notify: Notify::new(),
            max_depth: max_depth.max(1),
            overflow_policy,
            metrics,
        })
    }

    pub fn try_send(&self, segment: CapturedMeetingSegment) -> CapturedSegmentEnqueueOutcome {
        let mut queue = match self.inner.lock() {
            Ok(queue) => queue,
            Err(_) => {
                self.metrics.record_queue_full(self.overflow_policy);
                self.metrics.record_segment_dropped(self.overflow_policy);
                return CapturedSegmentEnqueueOutcome::DroppedNewest { segment };
            }
        };

        if queue.len() < self.max_depth {
            queue.push_back(segment);
            let depth = queue.len();
            let bytes_queued = queue.iter().map(|segment| segment.byte_length).sum();
            self.metrics.record_segment_queued();
            self.metrics
                .record_queue_depth(depth, bytes_queued, self.max_depth);
            self.notify.notify_one();
            return CapturedSegmentEnqueueOutcome::Enqueued { depth };
        }

        self.metrics.record_queue_full(self.overflow_policy);
        match self.overflow_policy {
            CaptureOverflowPolicy::RejectNewest => {
                self.metrics
                    .record_segment_dropped(CaptureOverflowPolicy::RejectNewest);
                CapturedSegmentEnqueueOutcome::DroppedNewest { segment }
            }
            CaptureOverflowPolicy::DropOldestAndReport => {
                let Some(dropped_segment) = queue.pop_front() else {
                    self.metrics
                        .record_segment_dropped(CaptureOverflowPolicy::RejectNewest);
                    return CapturedSegmentEnqueueOutcome::DroppedNewest { segment };
                };
                queue.push_back(segment);
                let depth = queue.len();
                let bytes_queued = queue.iter().map(|segment| segment.byte_length).sum();
                self.metrics
                    .record_segment_dropped(CaptureOverflowPolicy::DropOldestAndReport);
                self.metrics.record_segment_queued();
                self.metrics
                    .record_queue_depth(depth, bytes_queued, self.max_depth);
                self.notify.notify_one();
                CapturedSegmentEnqueueOutcome::DroppedOldest {
                    dropped_segment,
                    depth,
                }
            }
            CaptureOverflowPolicy::StopCapture => {
                self.metrics
                    .record_segment_dropped(CaptureOverflowPolicy::StopCapture);
                CapturedSegmentEnqueueOutcome::StopCapture { segment }
            }
        }
    }

    pub async fn recv(&self) -> Option<CapturedMeetingSegment> {
        loop {
            if let Ok(mut queue) = self.inner.lock() {
                if let Some(segment) = queue.pop_front() {
                    let depth = queue.len();
                    let bytes_queued = queue.iter().map(|segment| segment.byte_length).sum();
                    self.metrics
                        .record_queue_depth(depth, bytes_queued, self.max_depth);
                    return Some(segment);
                }
            }
            self.notify.notified().await;
        }
    }

    pub fn metrics(&self) -> CaptureMetrics {
        self.metrics.snapshot()
    }
}

#[derive(Debug)]
pub struct AudioCaptureStartRequest {
    pub session_id: String,
    pub meeting_storage_dir: PathBuf,
    pub backend: CaptureBackend,
    pub device_id: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub transcript_source: TranscriptSource,
    pub pipeline: CapturePipelineConfig,
    pub emit_segments: bool,
    pub metrics: CaptureMetricsReporter,
}

pub struct AudioCapture {
    pub backend: CaptureBackend,
    pub device: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub capture_running: bool,
    running_capture: Option<RunningAudioCapture>,
}

impl AudioCapture {
    pub fn new(backend: CaptureBackend, device: String, sample_rate: u32) -> Self {
        let channels = if backend == CaptureBackend::CoreAudio {
            1
        } else {
            2
        };
        Self {
            backend,
            device,
            sample_rate,
            channels,
            capture_running: false,
            running_capture: None,
        }
    }

    /// Detect the best available audio backend on this system.
    pub fn auto_detect_backend() -> CaptureBackend {
        #[cfg(target_os = "linux")]
        {
            CaptureBackend::PipeWire
        }

        #[cfg(target_os = "macos")]
        {
            CaptureBackend::CoreAudio
        }

        #[cfg(target_os = "windows")]
        {
            CaptureBackend::Wasapi
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            CaptureBackend::Default
        }
    }

    /// Legacy probe path used by older tests/UI checks. It never starts real
    /// capture because real start requires a managed session and segment sink.
    pub fn start(&mut self) -> Result<(), String> {
        self.capture_running = false;
        let backend = resolve_backend(self.backend);
        self.backend = backend;
        Err(Self::unsupported_message(backend))
    }

    pub fn start_loopback_capture(
        &mut self,
        request: AudioCaptureStartRequest,
        segment_sender: Option<CapturedSegmentSender>,
    ) -> Result<(), MeetingRuntimeError> {
        let backend = resolve_backend(request.backend);
        self.backend = backend;
        self.device = request.device_id.clone();
        self.sample_rate = request.sample_rate;
        self.channels = request.channels;
        self.capture_running = false;

        if backend != CaptureBackend::Wasapi {
            return Err(MeetingRuntimeError::CaptureUnavailable {
                backend,
                reason: Self::unsupported_message(backend),
            });
        }

        let mut request = request;
        request.backend = backend;

        #[cfg(target_os = "windows")]
        {
            let running_capture = RunningAudioCapture::start(request, segment_sender)?;
            self.capture_running = running_capture.is_running();
            self.running_capture = Some(running_capture);
            Ok(())
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = request;
            let _ = segment_sender;
            Err(MeetingRuntimeError::CaptureUnavailable {
                backend,
                reason: wasapi_unavailable_reason(),
            })
        }
    }

    /// Stop capturing system audio.
    pub fn stop(&mut self) -> Result<(), MeetingRuntimeError> {
        if let Some(mut running_capture) = self.running_capture.take() {
            self.capture_running = false;
            return running_capture.stop();
        }
        self.capture_running = false;
        Ok(())
    }

    pub fn pause_loopback_capture(&self) -> Result<(), MeetingRuntimeError> {
        let capture = self.running_capture.as_ref().ok_or_else(|| {
            MeetingRuntimeError::UnsupportedCapability {
                capability: "meeting.audio.capture.pause".to_string(),
                reason: "No active capture handle is available to pause".to_string(),
            }
        })?;
        capture.pause();
        Ok(())
    }

    pub fn resume_loopback_capture(&self) -> Result<(), MeetingRuntimeError> {
        let capture = self.running_capture.as_ref().ok_or_else(|| {
            MeetingRuntimeError::UnsupportedCapability {
                capability: "meeting.audio.capture.resume".to_string(),
                reason: "No active capture handle is available to resume".to_string(),
            }
        })?;
        capture.resume();
        Ok(())
    }

    /// Check if capture is currently running.
    pub fn is_running(&self) -> bool {
        self.running_capture
            .as_ref()
            .is_some_and(RunningAudioCapture::is_running)
            || self.capture_running
    }

    /// Get the list of available audio devices for capture.
    pub fn list_available_devices() -> Result<Vec<String>, String> {
        let mut devices = Vec::new();

        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = Command::new("pactl").args(["list", "sources"]).output() {
                let text = String::from_utf8_lossy(&output.stdout);
                for line in text.lines() {
                    if line.trim().starts_with("Name:") {
                        let name = line.trim().trim_start_matches("Name:").trim().to_string();
                        if !name.is_empty() && !devices.contains(&name) {
                            devices.push(name);
                        }
                    }
                }
            }
            if devices.is_empty() {
                if let Ok(output) = Command::new("pw-cli").args(["ls"]).output() {
                    let text = String::from_utf8_lossy(&output.stdout);
                    for line in text.lines() {
                        if line.contains("PipeWire") {
                            let name = line.trim().to_string();
                            if !name.is_empty() && !devices.contains(&name) {
                                devices.push(name);
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = Command::new("system_profiler")
                .args(["SPAudioDataType"])
                .output()
            {
                let text = String::from_utf8_lossy(&output.stdout);
                for line in text.lines() {
                    if let Some((_, name)) = line.split_once(':') {
                        let name = name.trim().to_string();
                        if !name.is_empty() && !devices.contains(&name) {
                            devices.push(name);
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            devices.push("default_render_loopback".to_string());
            devices.push("default_microphone".to_string());
        }

        if devices.is_empty() {
            devices.push("default".to_string());
        }

        Ok(devices)
    }

    pub fn capture_supported(&self) -> bool {
        resolve_backend(self.backend) == CaptureBackend::Wasapi && wasapi_backend_available()
    }

    #[doc(hidden)]
    pub fn install_fake_running_capture_for_test(&mut self, stop_timeout: Duration) {
        self.backend = CaptureBackend::Wasapi;
        self.capture_running = true;
        self.running_capture = Some(RunningAudioCapture::fake_timed_out_for_test(stop_timeout));
    }

    pub fn unsupported_message(backend: CaptureBackend) -> String {
        match backend {
            CaptureBackend::Wasapi => wasapi_unavailable_reason(),
            CaptureBackend::CoreAudio => {
                "CoreAudio capture is unsupported in this build".to_string()
            }
            CaptureBackend::PipeWire => "PipeWire capture is unsupported in this build".to_string(),
            CaptureBackend::Default => {
                "Default capture backend could not be resolved for this platform".to_string()
            }
        }
    }
}

pub struct RunningAudioCapture {
    running: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    stop_requested: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<()>>,
    stop_timeout: Duration,
    force_stop_timeout: bool,
}

impl RunningAudioCapture {
    #[cfg(target_os = "windows")]
    fn start(
        request: AudioCaptureStartRequest,
        segment_sender: Option<CapturedSegmentSender>,
    ) -> Result<Self, MeetingRuntimeError> {
        let running = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let stop_requested = Arc::new(AtomicBool::new(false));
        let join_handle = wasapi_loopback::start_capture_thread(
            request,
            segment_sender,
            running.clone(),
            paused.clone(),
            stop_requested.clone(),
        )?;

        Ok(Self {
            running,
            paused,
            stop_requested,
            join_handle: Some(join_handle),
            stop_timeout: DEFAULT_CAPTURE_STOP_TIMEOUT,
            force_stop_timeout: false,
        })
    }

    fn fake_timed_out_for_test(stop_timeout: Duration) -> Self {
        Self {
            running: Arc::new(AtomicBool::new(true)),
            paused: Arc::new(AtomicBool::new(false)),
            stop_requested: Arc::new(AtomicBool::new(false)),
            join_handle: None,
            stop_timeout,
            force_stop_timeout: true,
        }
    }

    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }

    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }

    pub fn stop(&mut self) -> Result<(), MeetingRuntimeError> {
        self.stop_requested.store(true, Ordering::SeqCst);
        if self.force_stop_timeout {
            self.running.store(false, Ordering::SeqCst);
            return Err(MeetingRuntimeError::CaptureStopTimedOut {
                backend: CaptureBackend::Wasapi,
                timeout_ms: self.stop_timeout.as_millis() as u64,
            });
        }
        if let Some(join_handle) = self.join_handle.take() {
            let timeout_ms = self.stop_timeout.as_millis() as u64;
            let deadline = Instant::now() + self.stop_timeout;
            while !join_handle.is_finished() {
                if Instant::now() >= deadline {
                    self.running.store(false, Ordering::SeqCst);
                    return Err(MeetingRuntimeError::CaptureStopTimedOut {
                        backend: CaptureBackend::Wasapi,
                        timeout_ms,
                    });
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            join_handle
                .join()
                .map_err(|_| MeetingRuntimeError::CaptureStreamError {
                    backend: CaptureBackend::Wasapi,
                    reason: "WASAPI capture thread panicked during stop".to_string(),
                })?;
        }
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Drop for RunningAudioCapture {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

pub fn wasapi_backend_available() -> bool {
    cfg!(target_os = "windows")
}

pub fn wasapi_unavailable_reason() -> String {
    #[cfg(target_os = "windows")]
    {
        "Windows WASAPI loopback capture failed to initialize; verify an active render endpoint is available".to_string()
    }

    #[cfg(not(target_os = "windows"))]
    {
        "WASAPI loopback capture is Windows-only and unavailable on this platform".to_string()
    }
}

fn resolve_backend(backend: CaptureBackend) -> CaptureBackend {
    if backend == CaptureBackend::Default {
        AudioCapture::auto_detect_backend()
    } else {
        backend
    }
}

fn duration_bucket(duration_ms: u64) -> &'static str {
    match duration_ms {
        0..=30_000 => "duration_0_30s",
        30_001..=60_000 => "duration_30_60s",
        _ => "duration_gt_60s",
    }
}

fn sanitize_metric_token(value: &str) -> String {
    value
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || *character == '_' || *character == '-'
        })
        .take(64)
        .collect::<String>()
}

fn sanitize_backend_message(message: &str) -> String {
    message
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric()
                || matches!(
                    character,
                    ' ' | '_' | '-' | ':' | ';' | ',' | '.' | '(' | ')' | '='
                )
            {
                character
            } else {
                '_'
            }
        })
        .take(160)
        .collect::<String>()
}

fn size_bucket(byte_length: u64) -> &'static str {
    match byte_length {
        0..=1_048_576 => "size_0_1mb",
        1_048_577..=10_485_760 => "size_1_10mb",
        _ => "size_gt_10mb",
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioCapture, CaptureBackend};

    #[test]
    fn unsupported_audio_capture_does_not_mark_capture_running() {
        let mut capture = AudioCapture::new(CaptureBackend::Default, "default".into(), 16_000);

        let result = capture.start();

        assert!(result.is_err());
        assert!(!capture.is_running());
    }
}
