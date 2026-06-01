//! Persistent capture controller boundary for future meeting audio.
//!
//! This module owns the lifecycle shape needed for real capture. It also tracks
//! redacted segment progress for managed segment transcription. The OS backend
//! remains explicitly unsupported until a tested WASAPI integration is added.

use super::{
    audio_capture::{
        AudioCapture, AudioCaptureStartRequest, CaptureMetricsReporter,
        CapturedSegmentDrainOutcome, CapturedSegmentSender,
    },
    types::{
        CaptureBackend, CaptureControllerState, CaptureHealth, CaptureHealthStatus,
        CapturePipelineConfig, MeetingConfig, MeetingRuntimeError, TranscriptSource,
    },
};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct CaptureControllerConfig {
    pub backend: CaptureBackend,
    pub device_id: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub transcript_source: TranscriptSource,
    pub pipeline: CapturePipelineConfig,
}

impl CaptureControllerConfig {
    pub fn from_meeting_config(config: &MeetingConfig) -> Self {
        Self::from_meeting_config_for_source(config, TranscriptSource::SystemAudio)
    }

    pub fn from_meeting_config_for_source(
        config: &MeetingConfig,
        transcript_source: TranscriptSource,
    ) -> Self {
        Self {
            backend: config.capture_backend,
            device_id: "default".to_string(),
            sample_rate: config.sample_rate,
            channels: default_channels_for_backend(config.capture_backend, transcript_source),
            transcript_source,
            pipeline: CapturePipelineConfig::default(),
        }
    }
}

pub struct CaptureController {
    state: CaptureControllerState,
    active_handle: Option<AudioCaptureHandle>,
    config: Option<CaptureControllerConfig>,
    health: CaptureHealth,
    metrics: CaptureMetricsReporter,
}

pub struct CaptureControllerStartRequest {
    pub config: CaptureControllerConfig,
    pub session_id: String,
    pub meeting_storage_dir: PathBuf,
    pub segment_sender: Option<CapturedSegmentSender>,
    pub segment_task: Option<tauri::async_runtime::JoinHandle<()>>,
    pub emit_segments: bool,
    pub metrics: CaptureMetricsReporter,
}

pub struct CaptureSegmentDrainWaiter {
    sender: Option<CapturedSegmentSender>,
    task: Option<tauri::async_runtime::JoinHandle<()>>,
    timeout: Duration,
    metrics: CaptureMetricsReporter,
}

impl CaptureSegmentDrainWaiter {
    pub async fn wait(mut self) -> Option<CapturedSegmentDrainOutcome> {
        let mut drained = true;
        let mut outcome = None;
        if let Some(sender) = self.sender.take() {
            sender.close();
            let drain_outcome = sender.wait_drained(self.timeout).await;
            drained = drain_outcome.drained;
            outcome = Some(drain_outcome);
        }

        if let Some(task) = self.task.take() {
            if drained {
                if task.await.is_err() {
                    self.metrics.record_drain_timed_out(0, 0);
                }
            } else {
                task.abort();
            }
        }

        outcome
    }
}

impl CaptureController {
    pub fn new() -> Self {
        Self {
            state: CaptureControllerState::Idle,
            active_handle: None,
            config: None,
            health: CaptureHealth::default(),
            metrics: CaptureMetricsReporter::new(),
        }
    }

    pub fn prepare(&mut self, config: CaptureControllerConfig) -> CaptureHealth {
        self.config = Some(config);
        self.state = CaptureControllerState::Idle;
        self.refresh_health(CaptureHealthStatus::Idle, None)
    }

    pub fn start(
        &mut self,
        config: CaptureControllerConfig,
    ) -> Result<CaptureHealth, MeetingRuntimeError> {
        self.config = Some(config.clone());
        self.state = CaptureControllerState::Starting;
        self.metrics = CaptureMetricsReporter::new();
        self.refresh_health(CaptureHealthStatus::Idle, None);

        let mut handle = AudioCaptureHandle::new(config.clone(), None, self.metrics.clone());
        match handle.start() {
            Ok(()) if handle.is_running() => {
                self.active_handle = Some(handle);
                self.state = CaptureControllerState::Capturing;
                Ok(self.refresh_health(CaptureHealthStatus::Healthy, None))
            }
            Ok(()) => {
                let reason =
                    "Audio backend returned success without an active capture handle".to_string();
                self.active_handle = None;
                self.state = CaptureControllerState::Failed;
                self.refresh_health(CaptureHealthStatus::Failed, Some(reason.clone()));
                Err(MeetingRuntimeError::CaptureUnavailable {
                    backend: config.backend,
                    reason,
                })
            }
            Err(reason) => {
                self.active_handle = None;
                self.state = CaptureControllerState::Unsupported;
                self.refresh_health(CaptureHealthStatus::Unsupported, Some(reason.clone()));
                Err(MeetingRuntimeError::CaptureUnavailable {
                    backend: config.backend,
                    reason,
                })
            }
        }
    }

    pub fn start_real_capture(
        &mut self,
        mut request: CaptureControllerStartRequest,
    ) -> Result<CaptureHealth, MeetingRuntimeError> {
        let config = request.config.clone();
        self.config = Some(config.clone());
        self.state = CaptureControllerState::Starting;
        self.metrics = request.metrics.clone();
        self.refresh_health(CaptureHealthStatus::Idle, None);

        let segment_task = request.segment_task.take();
        let mut handle =
            AudioCaptureHandle::new(config.clone(), segment_task, self.metrics.clone());
        match handle.start_loopback(request) {
            Ok(()) if handle.is_running() => {
                self.active_handle = Some(handle);
                self.state = CaptureControllerState::Capturing;
                Ok(self.refresh_health(CaptureHealthStatus::Healthy, None))
            }
            Ok(()) => {
                let reason =
                    "Audio backend returned success without an active capture handle".to_string();
                handle.abort_segment_task();
                self.active_handle = None;
                self.state = CaptureControllerState::Failed;
                self.refresh_health(CaptureHealthStatus::Failed, Some(reason.clone()));
                Err(MeetingRuntimeError::CaptureUnavailable {
                    backend: config.backend,
                    reason,
                })
            }
            Err(error) => {
                let reason = error.to_string();
                handle.abort_segment_task();
                self.active_handle = None;
                self.state = if matches!(error, MeetingRuntimeError::CaptureUnavailable { .. }) {
                    CaptureControllerState::Unsupported
                } else {
                    CaptureControllerState::Failed
                };
                let status = if matches!(self.state, CaptureControllerState::Unsupported) {
                    CaptureHealthStatus::Unsupported
                } else {
                    CaptureHealthStatus::Failed
                };
                self.refresh_health(status, Some(reason));
                Err(error)
            }
        }
    }

    pub fn pause(&mut self) -> Result<CaptureHealth, MeetingRuntimeError> {
        if !matches!(self.state, CaptureControllerState::Capturing) {
            return Err(MeetingRuntimeError::UnsupportedCapability {
                capability: "meeting.audio.capture.pause".to_string(),
                reason: "No active capture handle is available to pause".to_string(),
            });
        }

        let handle = self.active_handle.as_mut().ok_or_else(|| {
            MeetingRuntimeError::UnsupportedCapability {
                capability: "meeting.audio.capture.pause".to_string(),
                reason: "Capture state is active, but no durable handle is owned".to_string(),
            }
        })?;
        handle.pause()?;
        self.state = CaptureControllerState::Paused;
        Ok(self.refresh_health(CaptureHealthStatus::Healthy, None))
    }

    pub fn pause_capture(&mut self) -> Result<CaptureHealth, MeetingRuntimeError> {
        self.pause()
    }

    pub fn resume(&mut self) -> Result<CaptureHealth, MeetingRuntimeError> {
        if !matches!(self.state, CaptureControllerState::Paused) {
            return Err(MeetingRuntimeError::UnsupportedCapability {
                capability: "meeting.audio.capture.resume".to_string(),
                reason: "No paused capture handle is available to resume".to_string(),
            });
        }

        let handle = self.active_handle.as_mut().ok_or_else(|| {
            MeetingRuntimeError::UnsupportedCapability {
                capability: "meeting.audio.capture.resume".to_string(),
                reason: "Capture state is paused, but no durable handle is owned".to_string(),
            }
        })?;
        handle.resume()?;
        self.state = CaptureControllerState::Capturing;
        Ok(self.refresh_health(CaptureHealthStatus::Healthy, None))
    }

    pub fn resume_capture(&mut self) -> Result<CaptureHealth, MeetingRuntimeError> {
        self.resume()
    }

    pub fn stop(&mut self) -> Result<CaptureHealth, MeetingRuntimeError> {
        let Some(mut handle) = self.active_handle.take() else {
            if matches!(
                self.state,
                CaptureControllerState::Failed | CaptureControllerState::Unsupported
            ) {
                return Ok(self.health_snapshot());
            }
            self.state = CaptureControllerState::Idle;
            return Ok(self.refresh_health(CaptureHealthStatus::Idle, None));
        };

        self.state = CaptureControllerState::Stopping;
        self.refresh_health(CaptureHealthStatus::Idle, None);
        if let Err(error) = handle.stop() {
            let message = error.to_string();
            self.state = CaptureControllerState::Failed;
            let status = if matches!(error, MeetingRuntimeError::CaptureStopTimedOut { .. }) {
                self.metrics.record_stop_timed_out();
                CaptureHealthStatus::StopTimedOut
            } else {
                CaptureHealthStatus::Failed
            };
            self.refresh_health(status, Some(message));
            return Err(error);
        }
        self.state = CaptureControllerState::Idle;
        Ok(self.refresh_health(CaptureHealthStatus::Idle, None))
    }

    pub fn stop_for_background_finalization(
        &mut self,
        drain_timeout: Duration,
    ) -> (
        CaptureHealth,
        Option<CaptureSegmentDrainWaiter>,
        Option<MeetingRuntimeError>,
    ) {
        let Some(mut handle) = self.active_handle.take() else {
            if matches!(
                self.state,
                CaptureControllerState::Failed | CaptureControllerState::Unsupported
            ) {
                return (self.health_snapshot(), None, None);
            }
            self.state = CaptureControllerState::Idle;
            return (
                self.refresh_health(CaptureHealthStatus::Idle, None),
                None,
                None,
            );
        };

        self.state = CaptureControllerState::Stopping;
        self.refresh_health(CaptureHealthStatus::Idle, None);
        let (stop_result, drain_waiter) = handle.stop_for_background_finalization(drain_timeout);
        if let Err(error) = stop_result {
            let message = error.to_string();
            self.state = CaptureControllerState::Failed;
            let status = if matches!(error, MeetingRuntimeError::CaptureStopTimedOut { .. }) {
                self.metrics.record_stop_timed_out();
                CaptureHealthStatus::StopTimedOut
            } else {
                CaptureHealthStatus::Failed
            };
            let health = self.refresh_health(status, Some(message));
            return (health, drain_waiter, Some(error));
        }
        self.state = CaptureControllerState::Idle;
        (
            self.refresh_health(CaptureHealthStatus::Idle, None),
            drain_waiter,
            None,
        )
    }

    pub fn stop_capture(&mut self) -> Result<CaptureHealth, MeetingRuntimeError> {
        self.stop()
    }

    pub fn abort(&mut self, reason: String) -> Result<CaptureHealth, MeetingRuntimeError> {
        let mut final_reason = reason.clone();
        if let Some(mut handle) = self.active_handle.take() {
            if let Err(error) = handle.stop() {
                final_reason = format!("{reason}; abort stop failed: {error}");
            }
        }
        self.state = CaptureControllerState::Failed;
        Ok(self.refresh_health(CaptureHealthStatus::Failed, Some(final_reason)))
    }

    /// Mark capture as failed and request the backend/segment pipeline to stop
    /// without draining or joining the managed STT task.
    ///
    /// This is intentionally separate from `abort()`: segment transcription
    /// failure handling may run inside the segment STT task itself. Calling the
    /// blocking abort path from that context can wait on the same task that is
    /// currently executing and can poison the capture-controller mutex if the
    /// runtime panics while the lock is held. The finalization path remains
    /// responsible for joining and archiving through the governed lifecycle.
    pub fn request_stop_after_segment_transcription_failure(
        &mut self,
        reason: String,
    ) -> CaptureHealth {
        if let Some(handle) = self.active_handle.as_mut() {
            handle.request_stop_nonblocking();
        }
        self.state = CaptureControllerState::Failed;
        self.refresh_health(CaptureHealthStatus::Failed, Some(reason))
    }

    pub fn abort_capture(&mut self, reason: String) -> Result<CaptureHealth, MeetingRuntimeError> {
        self.abort(reason)
    }

    pub fn quarantine_after_poison(&mut self, component: &str) -> CaptureHealth {
        let mut reason = format!("{component}_poisoned");
        if let Some(mut handle) = self.active_handle.take() {
            if let Err(error) = handle.stop() {
                reason = format!("{reason}; poison recovery stop failed: {error}");
            }
        }
        self.state = CaptureControllerState::Failed;
        self.refresh_health(CaptureHealthStatus::Failed, Some(reason))
    }

    pub fn record_segment_written(&mut self, byte_length: u64, duration_ms: u64) -> CaptureHealth {
        self.metrics
            .record_segment_written(byte_length, duration_ms);
        self.refresh_health(self.health.status.clone(), self.health.last_error.clone())
    }

    pub fn record_segment_transcribed(&mut self) -> CaptureHealth {
        self.metrics.record_segment_transcribed();
        self.refresh_health(self.health.status.clone(), self.health.last_error.clone())
    }

    pub fn record_segment_transcribed_with_id(
        &mut self,
        segment_id: Option<&str>,
    ) -> CaptureHealth {
        self.metrics.record_segment_transcribed_with_id(segment_id);
        self.refresh_health(self.health.status.clone(), self.health.last_error.clone())
    }

    pub fn record_segment_write_failure(&mut self, error_class: &str) -> CaptureHealth {
        self.metrics.record_segment_write_failure(error_class);
        self.state = CaptureControllerState::Failed;
        self.refresh_health(
            CaptureHealthStatus::Failed,
            Some("segment_write_failed".to_string()),
        )
    }

    pub fn record_segment_transcription_failure(&mut self, error_class: &str) -> CaptureHealth {
        self.metrics
            .record_segment_transcription_failure(error_class);
        self.refresh_health(self.health.status.clone(), self.health.last_error.clone())
    }

    pub fn record_segment_transcription_failure_with_id(
        &mut self,
        error_class: &str,
        segment_id: Option<&str>,
    ) -> CaptureHealth {
        self.metrics
            .record_segment_transcription_failure_with_id(error_class, segment_id);
        self.refresh_health(self.health.status.clone(), self.health.last_error.clone())
    }

    pub fn record_terminal_segment_transcription_failure(
        &mut self,
        error_class: &str,
    ) -> CaptureHealth {
        self.metrics
            .record_segment_transcription_failure(error_class);
        self.state = CaptureControllerState::Failed;
        self.refresh_health(
            CaptureHealthStatus::Failed,
            Some("segment_transcription_failed".to_string()),
        )
    }

    pub fn record_consent_revoked(&mut self, platform: &str) -> CaptureHealth {
        let mut stop_message = None;
        if let Some(mut handle) = self.active_handle.take() {
            if let Err(error) = handle.stop() {
                stop_message = Some(format!("capture_stop_after_consent_revoked:{error}"));
            }
        }
        self.metrics.record_consent_revoked();
        self.state = CaptureControllerState::Failed;
        self.refresh_health(
            CaptureHealthStatus::ConsentRevoked,
            Some(stop_message.unwrap_or_else(|| format!("consent_revoked:{platform}"))),
        )
    }

    pub fn record_consent_revoked_from_segment_worker(&mut self, platform: &str) -> CaptureHealth {
        if let Some(handle) = self.active_handle.as_mut() {
            handle.request_stop_nonblocking();
        }
        self.metrics.record_consent_revoked();
        self.state = CaptureControllerState::Failed;
        self.refresh_health(
            CaptureHealthStatus::ConsentRevoked,
            Some(format!("consent_revoked:{platform}")),
        )
    }

    pub fn health_snapshot(&self) -> CaptureHealth {
        let mut health = self.health.clone();
        health.state = self.state.clone();
        health.backend = self.config.as_ref().map(|config| config.backend);
        health.active_handle_present = self.active_handle.is_some();
        health.metrics = self.metrics.snapshot();
        health.backpressure_active = health.metrics.backpressure_active;
        health.last_segment_status = health.metrics.last_segment_status.clone();
        health.last_overflow_policy_applied = health.metrics.last_overflow_policy_applied;
        health
    }

    pub fn has_active_handle(&self) -> bool {
        self.active_handle.is_some()
    }

    pub fn state(&self) -> &CaptureControllerState {
        &self.state
    }

    fn refresh_health(
        &mut self,
        status: CaptureHealthStatus,
        last_error: Option<String>,
    ) -> CaptureHealth {
        let backend = self.config.as_ref().map(|config| config.backend);
        let pipeline = self
            .config
            .as_ref()
            .map(|config| config.pipeline.clone())
            .unwrap_or_default();
        let metrics = self.metrics.snapshot();
        self.health = CaptureHealth {
            state: self.state.clone(),
            status,
            backend,
            active_handle_present: self.active_handle.is_some(),
            backpressure_active: metrics.backpressure_active,
            last_error,
            last_segment_status: metrics
                .last_segment_status
                .clone()
                .or_else(|| self.health.last_segment_status.clone()),
            last_overflow_policy_applied: metrics
                .last_overflow_policy_applied
                .or(self.health.last_overflow_policy_applied),
            effective_pipeline: pipeline.effective(),
            pipeline,
            metrics,
        };
        self.health.clone()
    }

    #[cfg(test)]
    fn install_failing_stop_handle_for_test(&mut self, config: CaptureControllerConfig) {
        self.config = Some(config.clone());
        self.metrics = CaptureMetricsReporter::new();
        self.active_handle = Some(AudioCaptureHandle::failing_stop_for_test(
            config,
            self.metrics.clone(),
        ));
        self.state = CaptureControllerState::Capturing;
        self.refresh_health(CaptureHealthStatus::Healthy, None);
    }

    #[doc(hidden)]
    pub fn install_fake_active_capture_for_test(
        &mut self,
        config: CaptureControllerConfig,
        stop_acknowledges: bool,
        stop_timeout: Duration,
    ) -> CaptureHealth {
        self.config = Some(config.clone());
        self.metrics = CaptureMetricsReporter::new();
        self.active_handle = Some(AudioCaptureHandle::fake_for_test(
            config,
            self.metrics.clone(),
            stop_acknowledges,
            stop_timeout,
        ));
        self.state = CaptureControllerState::Capturing;
        self.refresh_health(CaptureHealthStatus::Healthy, None)
    }
}

impl Default for CaptureController {
    fn default() -> Self {
        Self::new()
    }
}

struct AudioCaptureHandle {
    inner: AudioCaptureHandleInner,
    segment_sender: Option<CapturedSegmentSender>,
    segment_task: Option<tauri::async_runtime::JoinHandle<()>>,
    metrics: CaptureMetricsReporter,
    #[cfg(test)]
    stop_error: Option<MeetingRuntimeError>,
}

enum AudioCaptureHandleInner {
    Real(AudioCapture),
    Fake(FakeAudioCaptureHandle),
}

struct FakeAudioCaptureHandle {
    backend: CaptureBackend,
    running: bool,
    stop_acknowledges: bool,
    stop_timeout: Duration,
}

impl AudioCaptureHandle {
    fn new(
        config: CaptureControllerConfig,
        segment_task: Option<tauri::async_runtime::JoinHandle<()>>,
        metrics: CaptureMetricsReporter,
    ) -> Self {
        Self {
            inner: AudioCaptureHandleInner::Real(AudioCapture::new(
                config.backend,
                config.device_id,
                config.sample_rate,
            )),
            segment_sender: None,
            segment_task,
            metrics,
            #[cfg(test)]
            stop_error: None,
        }
    }

    #[cfg(test)]
    fn failing_stop_for_test(
        config: CaptureControllerConfig,
        metrics: CaptureMetricsReporter,
    ) -> Self {
        Self {
            inner: AudioCaptureHandleInner::Real(AudioCapture::new(
                config.backend,
                config.device_id,
                config.sample_rate,
            )),
            segment_sender: None,
            segment_task: None,
            metrics,
            stop_error: Some(MeetingRuntimeError::CaptureStreamError {
                backend: config.backend,
                reason: "test stop failure".to_string(),
            }),
        }
    }

    fn fake_for_test(
        config: CaptureControllerConfig,
        metrics: CaptureMetricsReporter,
        stop_acknowledges: bool,
        stop_timeout: Duration,
    ) -> Self {
        Self {
            inner: AudioCaptureHandleInner::Fake(FakeAudioCaptureHandle {
                backend: config.backend,
                running: true,
                stop_acknowledges,
                stop_timeout,
            }),
            segment_sender: None,
            segment_task: None,
            metrics,
            #[cfg(test)]
            stop_error: None,
        }
    }

    fn start(&mut self) -> Result<(), String> {
        match &mut self.inner {
            AudioCaptureHandleInner::Real(capture) => capture.start(),
            AudioCaptureHandleInner::Fake(_) => Ok(()),
        }
    }

    fn start_loopback(
        &mut self,
        request: CaptureControllerStartRequest,
    ) -> Result<(), MeetingRuntimeError> {
        let config = request.config.clone();
        let AudioCaptureHandleInner::Real(capture) = &mut self.inner else {
            return Ok(());
        };
        self.segment_sender = request.segment_sender.clone();
        capture.start_loopback_capture(
            AudioCaptureStartRequest {
                session_id: request.session_id,
                meeting_storage_dir: request.meeting_storage_dir,
                backend: config.backend,
                device_id: config.device_id,
                sample_rate: config.sample_rate,
                channels: config.channels,
                transcript_source: config.transcript_source,
                pipeline: config.pipeline,
                emit_segments: request.emit_segments,
                metrics: self.metrics.clone(),
            },
            request.segment_sender,
        )
    }

    fn pause(&mut self) -> Result<(), MeetingRuntimeError> {
        match &mut self.inner {
            AudioCaptureHandleInner::Real(capture) => capture.pause_loopback_capture(),
            AudioCaptureHandleInner::Fake(fake) => {
                if fake.running {
                    Ok(())
                } else {
                    Err(MeetingRuntimeError::UnsupportedCapability {
                        capability: "meeting.audio.capture.pause".to_string(),
                        reason: "No active capture handle is available to pause".to_string(),
                    })
                }
            }
        }
    }

    fn resume(&mut self) -> Result<(), MeetingRuntimeError> {
        match &mut self.inner {
            AudioCaptureHandleInner::Real(capture) => capture.resume_loopback_capture(),
            AudioCaptureHandleInner::Fake(fake) => {
                if fake.running {
                    Ok(())
                } else {
                    Err(MeetingRuntimeError::UnsupportedCapability {
                        capability: "meeting.audio.capture.resume".to_string(),
                        reason: "No active capture handle is available to resume".to_string(),
                    })
                }
            }
        }
    }

    fn stop(&mut self) -> Result<(), MeetingRuntimeError> {
        #[cfg(test)]
        if let Some(error) = self.stop_error.take() {
            return Err(error);
        }

        let result = match &mut self.inner {
            AudioCaptureHandleInner::Real(capture) => capture.stop(),
            AudioCaptureHandleInner::Fake(fake) => {
                if fake.stop_acknowledges {
                    fake.running = false;
                    Ok(())
                } else {
                    std::thread::sleep(fake.stop_timeout);
                    fake.running = false;
                    Err(MeetingRuntimeError::CaptureStopTimedOut {
                        backend: fake.backend,
                        timeout_ms: fake.stop_timeout.as_millis() as u64,
                    })
                }
            }
        };
        self.close_and_drain_segment_task();
        result
    }

    fn stop_for_background_finalization(
        &mut self,
        drain_timeout: Duration,
    ) -> (
        Result<(), MeetingRuntimeError>,
        Option<CaptureSegmentDrainWaiter>,
    ) {
        #[cfg(test)]
        if let Some(error) = self.stop_error.take() {
            return (Err(error), None);
        }

        let result = match &mut self.inner {
            AudioCaptureHandleInner::Real(capture) => capture.stop(),
            AudioCaptureHandleInner::Fake(fake) => {
                if fake.stop_acknowledges {
                    fake.running = false;
                    Ok(())
                } else {
                    std::thread::sleep(fake.stop_timeout);
                    fake.running = false;
                    Err(MeetingRuntimeError::CaptureStopTimedOut {
                        backend: fake.backend,
                        timeout_ms: fake.stop_timeout.as_millis() as u64,
                    })
                }
            }
        };
        (result, self.take_segment_drain_waiter(drain_timeout))
    }

    fn is_running(&self) -> bool {
        match &self.inner {
            AudioCaptureHandleInner::Real(capture) => capture.is_running(),
            AudioCaptureHandleInner::Fake(fake) => fake.running,
        }
    }

    fn abort_segment_task(&mut self) {
        if let Some(sender) = self.segment_sender.take() {
            sender.close();
        }
        if let Some(task) = self.segment_task.take() {
            task.abort();
        }
    }

    fn request_stop_nonblocking(&mut self) {
        match &mut self.inner {
            AudioCaptureHandleInner::Real(capture) => capture.request_stop_nonblocking(),
            AudioCaptureHandleInner::Fake(fake) => {
                fake.running = false;
            }
        }
        if let Some(sender) = self.segment_sender.as_ref() {
            sender.close();
        }
    }

    fn close_and_drain_segment_task(&mut self) {
        let timeout = meeting_stt_drain_timeout();
        let mut drained = true;
        if let Some(sender) = self.segment_sender.take() {
            sender.close();
            let outcome =
                tauri::async_runtime::block_on(async { sender.wait_drained(timeout).await });
            drained = outcome.drained;
        }

        if let Some(task) = self.segment_task.take() {
            if drained {
                let join_result = tauri::async_runtime::block_on(async { task.await });
                if join_result.is_err() {
                    self.metrics.record_drain_timed_out(0, 0);
                }
            } else {
                task.abort();
            }
        }
    }

    fn take_segment_drain_waiter(
        &mut self,
        timeout: Duration,
    ) -> Option<CaptureSegmentDrainWaiter> {
        let sender = self.segment_sender.take();
        let task = self.segment_task.take();
        if sender.is_none() && task.is_none() {
            return None;
        }
        Some(CaptureSegmentDrainWaiter {
            sender,
            task,
            timeout,
            metrics: self.metrics.clone(),
        })
    }
}

fn meeting_stt_drain_timeout() -> Duration {
    let foreground = std::env::var("ASTRA_MEETING_STT_FOREGROUND_DRAIN_TIMEOUT_SECS").ok();
    let legacy = std::env::var("ASTRA_MEETING_STT_DRAIN_TIMEOUT_SECS").ok();
    let seconds = meeting_stt_drain_timeout_secs_from_env(foreground.as_deref(), legacy.as_deref());
    Duration::from_secs(seconds)
}

pub fn meeting_stt_background_drain_timeout() -> Duration {
    let background = std::env::var("ASTRA_MEETING_STT_BACKGROUND_DRAIN_TIMEOUT_SECS").ok();
    let seconds = background
        .as_deref()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(60)
        .clamp(1, 600);
    Duration::from_secs(seconds)
}

fn meeting_stt_drain_timeout_secs_from_env(
    foreground_override: Option<&str>,
    legacy_override: Option<&str>,
) -> u64 {
    foreground_override
        .or(legacy_override)
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(5)
        .clamp(1, 300)
}

fn default_channels_for_backend(
    backend: CaptureBackend,
    transcript_source: TranscriptSource,
) -> u16 {
    if backend == CaptureBackend::CoreAudio || transcript_source == TranscriptSource::Microphone {
        1
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MeetingConfig {
        MeetingConfig {
            platform: "teams".to_string(),
            capture_backend: CaptureBackend::CoreAudio,
            transcription_model: "local".to_string(),
            sample_rate: 16_000,
            diarization_enabled: false,
            privacy_mode: "default".to_string(),
            session_mode: super::super::types::MeetingSessionMode::RealCapture,
            live_transcription_enabled: false,
            capture_options: super::super::types::MeetingCaptureOptions::default(),
        }
    }

    #[test]
    fn capture_controller_starts_idle() {
        let controller = CaptureController::new();

        let health = controller.health_snapshot();

        assert_eq!(health.state, CaptureControllerState::Idle);
        assert_eq!(health.status, CaptureHealthStatus::Idle);
        assert!(!health.active_handle_present);
    }

    #[test]
    fn meeting_stt_drain_timeout_default_is_foreground_safe() {
        assert_eq!(meeting_stt_drain_timeout_secs_from_env(None, None), 5);
    }

    #[test]
    fn meeting_stt_drain_timeout_honors_foreground_override_first() {
        assert_eq!(
            meeting_stt_drain_timeout_secs_from_env(Some("4"), Some("60")),
            4
        );
        assert_eq!(
            meeting_stt_drain_timeout_secs_from_env(Some("999"), Some("60")),
            300
        );
    }

    #[test]
    fn meeting_stt_drain_timeout_honors_legacy_override() {
        assert_eq!(
            meeting_stt_drain_timeout_secs_from_env(None, Some("12")),
            12
        );
    }

    #[test]
    fn capture_controller_unsupported_start_returns_typed_error() {
        let mut controller = CaptureController::new();
        let controller_config = CaptureControllerConfig::from_meeting_config(&config());

        let result = controller.start(controller_config);

        assert!(matches!(
            result,
            Err(MeetingRuntimeError::CaptureUnavailable { .. })
        ));
        let health = controller.health_snapshot();
        assert_eq!(health.state, CaptureControllerState::Unsupported);
        assert_eq!(health.status, CaptureHealthStatus::Unsupported);
        assert!(!health.active_handle_present);
    }

    #[test]
    fn capture_controller_stop_is_idempotent_when_idle() {
        let mut controller = CaptureController::new();

        let first = controller.stop().expect("first stop");
        let second = controller.stop().expect("second stop");

        assert_eq!(first.state, CaptureControllerState::Idle);
        assert_eq!(second.state, CaptureControllerState::Idle);
        assert!(!second.active_handle_present);
    }

    #[test]
    fn capture_controller_abort_without_handle_is_terminal() {
        let mut controller = CaptureController::new();

        let health = controller
            .abort("test abort".to_string())
            .expect("abort without handle");

        assert_eq!(health.state, CaptureControllerState::Failed);
        assert_eq!(health.status, CaptureHealthStatus::Failed);
        assert!(!health.active_handle_present);
        assert_eq!(health.last_error.as_deref(), Some("test abort"));
    }

    #[test]
    fn capture_controller_abort_is_terminal_even_if_stop_fails() {
        let mut controller = CaptureController::new();
        let controller_config = CaptureControllerConfig::from_meeting_config(&config());
        controller.install_failing_stop_handle_for_test(controller_config);

        let health = controller
            .abort("test abort".to_string())
            .expect("abort returns terminal health");

        assert_eq!(health.state, CaptureControllerState::Failed);
        assert_eq!(health.status, CaptureHealthStatus::Failed);
        assert!(!health.active_handle_present);
        assert!(health
            .last_error
            .as_deref()
            .is_some_and(|value| value.contains("abort stop failed")));
    }


    #[test]
    fn segment_transcription_threshold_stop_is_nonblocking_and_preserves_handle_for_finalization() {
        let mut controller = CaptureController::new();
        let controller_config = CaptureControllerConfig::from_meeting_config(&config());
        controller.install_fake_active_capture_for_test(
            controller_config,
            false,
            Duration::from_millis(1_000),
        );

        let health = controller.request_stop_after_segment_transcription_failure(
            "system_segment_transcription_failure_threshold".to_string(),
        );

        assert_eq!(health.state, CaptureControllerState::Failed);
        assert_eq!(health.status, CaptureHealthStatus::Failed);
        assert!(health.active_handle_present);
        assert_eq!(
            health.last_error.as_deref(),
            Some("system_segment_transcription_failure_threshold")
        );
    }

    #[test]
    fn segment_worker_consent_revoked_stop_is_nonblocking_and_preserves_handle_for_finalization() {
        let mut controller = CaptureController::new();
        let controller_config = CaptureControllerConfig::from_meeting_config(&config());
        controller.install_fake_active_capture_for_test(
            controller_config,
            false,
            Duration::from_millis(1_000),
        );

        let health = controller.record_consent_revoked_from_segment_worker("teams");

        assert_eq!(health.state, CaptureControllerState::Failed);
        assert_eq!(health.status, CaptureHealthStatus::ConsentRevoked);
        assert!(health.active_handle_present);
        assert_eq!(health.last_error.as_deref(), Some("consent_revoked:teams"));
    }

    #[test]
    fn capture_controller_pause_without_capture_is_rejected() {
        let mut controller = CaptureController::new();

        let result = controller.pause();

        assert!(matches!(
            result,
            Err(MeetingRuntimeError::UnsupportedCapability { .. })
        ));
    }
}
