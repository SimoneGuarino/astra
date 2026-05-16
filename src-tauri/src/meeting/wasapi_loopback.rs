//! Windows WASAPI loopback capture backend.

use super::{
    audio_capture::{
        AudioCaptureStartRequest, CapturedSegmentEnqueueOutcome, CapturedSegmentSender,
    },
    audio_quality::{
        analyze_mono_i16_final_flush, analyze_mono_i16_segment,
        convert_interleaved_f32_to_i16_mono, downmix_interleaved_i16_to_mono, AudioQualityResult,
    },
    segment_writer::{CapturedMeetingSegment, SegmentWriter, SegmentWriterConfig},
    types::{CaptureBackend, CapturePipelineConfig, MeetingRuntimeError, TranscriptSource},
};
use std::{
    cmp,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, RecvTimeoutError},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

#[cfg(target_os = "windows")]
use std::{ffi::c_void, ptr, slice};

#[cfg(target_os = "windows")]
use windows::Win32::{
    Media::{
        Audio::{
            eCapture, eConsole, eRender, IAudioCaptureClient, IAudioClient, IMMDeviceEnumerator,
            MMDeviceEnumerator, AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK, WAVEFORMATEX, WAVEFORMATEXTENSIBLE, WAVE_FORMAT_PCM,
        },
        KernelStreaming::{KSDATAFORMAT_SUBTYPE_PCM, WAVE_FORMAT_EXTENSIBLE},
        Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
    },
    System::Com::{
        CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
        COINIT_MULTITHREADED,
    },
};

#[cfg(target_os = "windows")]
const WAVE_FORMAT_IEEE_FLOAT: u32 = 3;
#[cfg(target_os = "windows")]
const REFTIME_PER_SEC: i64 = 10_000_000;

#[cfg(target_os = "windows")]
macro_rules! startup_try {
    ($startup_tx:expr, $metrics:expr, $expr:expr) => {
        match $expr {
            Ok(value) => value,
            Err(error) => {
                $metrics.record_backend_error(backend_error_kind(&error), &error.to_string());
                report_startup_error(&mut $startup_tx, &error);
                return Err(error);
            }
        }
    };
}

pub fn start_capture_thread(
    request: AudioCaptureStartRequest,
    segment_sender: Option<CapturedSegmentSender>,
    running: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    stop_requested: Arc<AtomicBool>,
) -> Result<JoinHandle<()>, MeetingRuntimeError> {
    #[cfg(target_os = "windows")]
    {
        start_windows_capture_thread(request, segment_sender, running, paused, stop_requested)
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = request;
        let _ = segment_sender;
        let _ = running;
        let _ = paused;
        let _ = stop_requested;
        Err(MeetingRuntimeError::CaptureUnavailable {
            backend: CaptureBackend::Wasapi,
            reason: "WASAPI loopback capture is Windows-only and unavailable on this platform"
                .to_string(),
        })
    }
}

#[cfg(target_os = "windows")]
fn start_windows_capture_thread(
    request: AudioCaptureStartRequest,
    segment_sender: Option<CapturedSegmentSender>,
    running: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    stop_requested: Arc<AtomicBool>,
) -> Result<JoinHandle<()>, MeetingRuntimeError> {
    let (startup_tx, startup_rx) = mpsc::channel();
    let startup_backend = request.backend;
    let thread_running = running.clone();

    let join_handle = thread::Builder::new()
        .name(format!(
            "astra-wasapi-{}",
            request.transcript_source.as_str()
        ))
        .spawn(move || {
            let result = run_wasapi_capture(
                request,
                segment_sender,
                thread_running.clone(),
                paused,
                stop_requested,
                startup_tx,
            );
            if let Err(error) = result {
                log::warn!("WASAPI loopback capture stopped: {}", error);
            }
            thread_running.store(false, Ordering::SeqCst);
        })
        .map_err(|error| MeetingRuntimeError::CaptureStartFailed {
            backend: startup_backend,
            reason: format!("spawn WASAPI capture thread failed: {error}"),
        })?;

    match wait_for_startup_result(&startup_rx, Duration::from_secs(10), startup_backend) {
        Ok(()) => Ok(join_handle),
        Err(error) => {
            running.store(false, Ordering::SeqCst);
            if join_handle.is_finished() {
                let _ = join_handle.join();
            }
            Err(error)
        }
    }
}

pub fn wait_for_startup_result(
    startup_rx: &mpsc::Receiver<Result<(), MeetingRuntimeError>>,
    timeout: Duration,
    backend: CaptureBackend,
) -> Result<(), MeetingRuntimeError> {
    match startup_rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(RecvTimeoutError::Timeout) => Err(MeetingRuntimeError::CaptureStartupTimeout {
            backend,
            timeout_ms: timeout.as_millis() as u64,
        }),
        Err(RecvTimeoutError::Disconnected) => {
            Err(MeetingRuntimeError::CaptureStartupChannelClosed { backend })
        }
    }
}

#[cfg(target_os = "windows")]
fn run_wasapi_capture(
    request: AudioCaptureStartRequest,
    segment_sender: Option<CapturedSegmentSender>,
    running: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    stop_requested: Arc<AtomicBool>,
    startup_tx: mpsc::Sender<Result<(), MeetingRuntimeError>>,
) -> Result<(), MeetingRuntimeError> {
    let mut startup_tx = Some(startup_tx);
    let metrics = request.metrics.clone();
    let _com = startup_try!(startup_tx, metrics, ComApartment::initialize());
    let enumerator: IMMDeviceEnumerator = unsafe {
        startup_try!(
            startup_tx,
            metrics,
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|error| {
                MeetingRuntimeError::CaptureDeviceUnavailable {
                    backend: CaptureBackend::Wasapi,
                    reason: sanitize_windows_error("create MMDeviceEnumerator", error),
                }
            })
        )
    };
    let endpoint = if request.transcript_source == TranscriptSource::Microphone {
        eCapture
    } else {
        eRender
    };
    let endpoint_label = if request.transcript_source == TranscriptSource::Microphone {
        "default capture endpoint"
    } else {
        "default render endpoint"
    };
    let device = unsafe {
        startup_try!(
            startup_tx,
            metrics,
            enumerator
                .GetDefaultAudioEndpoint(endpoint, eConsole)
                .map_err(|error| MeetingRuntimeError::CaptureDeviceUnavailable {
                    backend: CaptureBackend::Wasapi,
                    reason: sanitize_windows_error(&format!("open {endpoint_label}"), error),
                })
        )
    };
    metrics.record_wasapi_endpoint_acquired();
    let audio_client: IAudioClient = unsafe {
        startup_try!(
            startup_tx,
            metrics,
            device.Activate(CLSCTX_ALL, None).map_err(|error| {
                MeetingRuntimeError::CaptureStartFailed {
                    backend: CaptureBackend::Wasapi,
                    reason: sanitize_windows_error("activate audio client", error),
                }
            })
        )
    };
    let mix_format = startup_try!(startup_tx, metrics, MixFormat::load(&audio_client));
    metrics.record_wasapi_mix_format(
        mix_format.sample_rate,
        mix_format.channels,
        mix_format.sample_format.as_metric_str(),
    );
    let writer_config = writer_config_for_mix(&request, &mix_format);
    let writer = SegmentWriter::new(request.meeting_storage_dir.clone(), writer_config);
    let effective_pipeline = request.pipeline.effective();
    let mut accumulator = SegmentAccumulator::new(SegmentAccumulatorConfig {
        writer,
        session_id: request.session_id,
        emit_segments: request.emit_segments,
        sender: segment_sender,
        max_segments: request
            .pipeline
            .effective()
            .effective_max_segments_per_session,
        sample_rate: mix_format.sample_rate,
        channels: 1,
        max_samples: max_segment_samples(
            mix_format.sample_rate,
            1,
            effective_pipeline.effective_segment_duration_ms,
            effective_pipeline.max_segment_bytes,
        ),
        pipeline: request.pipeline.clone(),
        metrics: request.metrics.clone(),
        stop_requested: stop_requested.clone(),
    });

    let stream_flags = if request.transcript_source == TranscriptSource::Microphone {
        0
    } else {
        AUDCLNT_STREAMFLAGS_LOOPBACK
    };

    unsafe {
        startup_try!(
            startup_tx,
            metrics,
            audio_client
                .Initialize(
                    AUDCLNT_SHAREMODE_SHARED,
                    stream_flags,
                    REFTIME_PER_SEC,
                    0,
                    mix_format.raw,
                    None,
                )
                .map_err(|error| MeetingRuntimeError::CaptureStartFailed {
                    backend: CaptureBackend::Wasapi,
                    reason: sanitize_windows_error("initialize loopback audio client", error),
                })
        );
    }
    metrics.record_wasapi_stream_initialized();
    let buffer_frame_count = startup_try!(
        startup_tx,
        metrics,
        unsafe { audio_client.GetBufferSize() }.map_err(|error| {
            MeetingRuntimeError::CaptureStartFailed {
                backend: CaptureBackend::Wasapi,
                reason: sanitize_windows_error("read audio client buffer size", error),
            }
        })
    );
    metrics.record_wasapi_buffer_frame_count(buffer_frame_count);
    let capture_client: IAudioCaptureClient = startup_try!(
        startup_tx,
        metrics,
        unsafe { audio_client.GetService() }.map_err(|error| {
            MeetingRuntimeError::CaptureStartFailed {
                backend: CaptureBackend::Wasapi,
                reason: sanitize_windows_error("open audio capture client", error),
            }
        })
    );

    unsafe {
        startup_try!(
            startup_tx,
            metrics,
            audio_client
                .Start()
                .map_err(|error| MeetingRuntimeError::CaptureStartFailed {
                    backend: CaptureBackend::Wasapi,
                    reason: sanitize_windows_error("start audio client", error),
                })
        );
    }
    metrics.record_wasapi_stream_started();
    running.store(true, Ordering::SeqCst);
    report_startup_ok(&mut startup_tx);

    let loop_result = capture_loop(
        &audio_client,
        &capture_client,
        &mix_format,
        &mut accumulator,
        &paused,
        &stop_requested,
    );
    let _ = unsafe { audio_client.Stop() };
    loop_result
}

#[cfg(target_os = "windows")]
fn capture_loop(
    audio_client: &IAudioClient,
    capture_client: &IAudioCaptureClient,
    mix_format: &MixFormat,
    accumulator: &mut SegmentAccumulator,
    paused: &AtomicBool,
    stop_requested: &AtomicBool,
) -> Result<(), MeetingRuntimeError> {
    while !stop_requested.load(Ordering::SeqCst) {
        if paused.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(20));
            continue;
        }

        let mut packet_frames = unsafe { capture_client.GetNextPacketSize() }.map_err(|error| {
            let error = MeetingRuntimeError::CaptureStreamError {
                backend: CaptureBackend::Wasapi,
                reason: sanitize_windows_error("read next packet size", error),
            };
            accumulator
                .metrics
                .record_backend_error(backend_error_kind(&error), &error.to_string());
            error
        })?;

        if packet_frames == 0 {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        while packet_frames > 0 && !stop_requested.load(Ordering::SeqCst) {
            let packet = match read_packet(capture_client, mix_format) {
                Ok(packet) => packet,
                Err(error) => {
                    accumulator
                        .metrics
                        .record_backend_error(backend_error_kind(&error), &error.to_string());
                    return Err(error);
                }
            };
            accumulator
                .metrics
                .record_wasapi_packet(packet.input_frames);
            accumulator.metrics.record_audio_conversion(
                packet.output_frames,
                packet.clipped_samples,
                packet.peak_abs,
                packet.rms_bps,
                packet.normalization_gain_bps,
            );
            accumulator.push_samples(&packet.samples)?;
            packet_frames = unsafe { capture_client.GetNextPacketSize() }.map_err(|error| {
                let error = MeetingRuntimeError::CaptureStreamError {
                    backend: CaptureBackend::Wasapi,
                    reason: sanitize_windows_error("read next packet size", error),
                };
                accumulator
                    .metrics
                    .record_backend_error(backend_error_kind(&error), &error.to_string());
                error
            })?;
        }
    }

    accumulator.flush_final()?;
    let _ = unsafe { audio_client.Reset() };
    Ok(())
}

#[cfg(target_os = "windows")]
fn read_packet(
    capture_client: &IAudioCaptureClient,
    mix_format: &MixFormat,
) -> Result<ConvertedPacket, MeetingRuntimeError> {
    let mut data: *mut u8 = ptr::null_mut();
    let mut frame_count = 0_u32;
    let mut flags = 0_u32;
    unsafe {
        capture_client
            .GetBuffer(&mut data, &mut frame_count, &mut flags, None, None)
            .map_err(|error| MeetingRuntimeError::CaptureStreamError {
                backend: CaptureBackend::Wasapi,
                reason: sanitize_windows_error("read capture buffer", error),
            })?;
    }

    let sample_result = if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
        Ok(quality_to_packet(AudioQualityResult {
            samples: vec![0_i16; frame_count as usize],
            input_frames: u64::from(frame_count),
            output_frames: u64::from(frame_count),
            clipped_samples: 0,
            peak_abs: 0,
            rms_bps: 0,
            normalization_gain_bps: 10_000,
        }))
    } else if data.is_null() {
        Err(MeetingRuntimeError::CaptureStreamError {
            backend: CaptureBackend::Wasapi,
            reason: "WASAPI returned a null capture packet".to_string(),
        })
    } else {
        convert_packet_to_i16_mono(data, frame_count, mix_format)
    };

    let release_result = unsafe { capture_client.ReleaseBuffer(frame_count) }.map_err(|error| {
        MeetingRuntimeError::CaptureStreamError {
            backend: CaptureBackend::Wasapi,
            reason: sanitize_windows_error("release capture buffer", error),
        }
    });

    release_result?;
    sample_result
}

#[cfg(target_os = "windows")]
fn convert_packet_to_i16_mono(
    data: *mut u8,
    frame_count: u32,
    mix_format: &MixFormat,
) -> Result<ConvertedPacket, MeetingRuntimeError> {
    let sample_count = frame_count as usize * usize::from(mix_format.channels);
    let quality = match mix_format.sample_format {
        WasapiSampleFormat::Float32 => {
            let samples = unsafe { slice::from_raw_parts(data as *const f32, sample_count) };
            convert_interleaved_f32_to_i16_mono(samples, mix_format.channels)
        }
        WasapiSampleFormat::Pcm16 => {
            let samples = unsafe { slice::from_raw_parts(data as *const i16, sample_count) };
            downmix_interleaved_i16_to_mono(samples, mix_format.channels)
        }
    };
    Ok(quality_to_packet(quality))
}

#[cfg(target_os = "windows")]
struct ConvertedPacket {
    samples: Vec<i16>,
    input_frames: u64,
    output_frames: u64,
    clipped_samples: u64,
    peak_abs: u16,
    rms_bps: u16,
    normalization_gain_bps: u16,
}

#[cfg(target_os = "windows")]
fn quality_to_packet(quality: AudioQualityResult) -> ConvertedPacket {
    ConvertedPacket {
        samples: quality.samples,
        input_frames: quality.input_frames,
        output_frames: quality.output_frames,
        clipped_samples: quality.clipped_samples,
        peak_abs: quality.peak_abs,
        rms_bps: quality.rms_bps,
        normalization_gain_bps: quality.normalization_gain_bps,
    }
}

#[cfg(target_os = "windows")]
struct MixFormat {
    raw: *mut WAVEFORMATEX,
    sample_rate: u32,
    channels: u16,
    sample_format: WasapiSampleFormat,
}

#[cfg(target_os = "windows")]
impl MixFormat {
    fn load(audio_client: &IAudioClient) -> Result<Self, MeetingRuntimeError> {
        let raw = unsafe { audio_client.GetMixFormat() }.map_err(|error| {
            MeetingRuntimeError::CaptureStartFailed {
                backend: CaptureBackend::Wasapi,
                reason: sanitize_windows_error("read mix format", error),
            }
        })?;
        if raw.is_null() {
            return Err(MeetingRuntimeError::CaptureStartFailed {
                backend: CaptureBackend::Wasapi,
                reason: "WASAPI returned a null mix format".to_string(),
            });
        }
        let format = unsafe { *raw };
        let channels = format.nChannels;
        let sample_rate = format.nSamplesPerSec;
        if channels == 0 || sample_rate == 0 {
            return Err(MeetingRuntimeError::CaptureStartFailed {
                backend: CaptureBackend::Wasapi,
                reason: "WASAPI mix format reported zero sample rate or channels".to_string(),
            });
        }
        let sample_format = parse_sample_format(raw)?;
        Ok(Self {
            raw,
            sample_rate,
            channels,
            sample_format,
        })
    }
}

#[cfg(target_os = "windows")]
impl Drop for MixFormat {
    fn drop(&mut self) {
        unsafe {
            CoTaskMemFree(Some(self.raw as *const c_void));
        }
    }
}

#[cfg(target_os = "windows")]
#[derive(Clone, Copy)]
enum WasapiSampleFormat {
    Float32,
    Pcm16,
}

#[cfg(target_os = "windows")]
impl WasapiSampleFormat {
    fn as_metric_str(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Pcm16 => "pcm16",
        }
    }
}

#[cfg(target_os = "windows")]
fn parse_sample_format(
    raw: *const WAVEFORMATEX,
) -> Result<WasapiSampleFormat, MeetingRuntimeError> {
    let format = unsafe { *raw };
    let tag = u32::from(format.wFormatTag);
    let bits = format.wBitsPerSample;
    if tag == WAVE_FORMAT_IEEE_FLOAT && bits == 32 {
        return Ok(WasapiSampleFormat::Float32);
    }
    if tag == WAVE_FORMAT_PCM && bits == 16 {
        return Ok(WasapiSampleFormat::Pcm16);
    }
    if tag == WAVE_FORMAT_EXTENSIBLE {
        let extended = raw as *const WAVEFORMATEXTENSIBLE;
        let sub_format = unsafe { ptr::addr_of!((*extended).SubFormat).read_unaligned() };
        if sub_format == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT && bits == 32 {
            return Ok(WasapiSampleFormat::Float32);
        }
        if sub_format == KSDATAFORMAT_SUBTYPE_PCM && bits == 16 {
            return Ok(WasapiSampleFormat::Pcm16);
        }
    }
    Err(MeetingRuntimeError::CaptureStartFailed {
        backend: CaptureBackend::Wasapi,
        reason: format!("unsupported WASAPI mix format tag={tag} bits_per_sample={bits}"),
    })
}

#[cfg(target_os = "windows")]
struct ComApartment;

#[cfg(target_os = "windows")]
impl ComApartment {
    fn initialize() -> Result<Self, MeetingRuntimeError> {
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED)
                .ok()
                .map_err(|error| MeetingRuntimeError::CaptureStartFailed {
                    backend: CaptureBackend::Wasapi,
                    reason: sanitize_windows_error("initialize COM", error),
                })?;
        }
        Ok(Self)
    }
}

#[cfg(target_os = "windows")]
impl Drop for ComApartment {
    fn drop(&mut self) {
        unsafe {
            CoUninitialize();
        }
    }
}

struct SegmentAccumulator {
    writer: SegmentWriter,
    session_id: String,
    emit_segments: bool,
    sender: Option<CapturedSegmentSender>,
    samples: Vec<i16>,
    sample_rate: u32,
    channels: u16,
    max_samples: usize,
    max_segments: u64,
    segments_emitted: u64,
    timeline_cursor_ms: u64,
    pipeline: CapturePipelineConfig,
    metrics: super::audio_capture::CaptureMetricsReporter,
    stop_requested: Arc<AtomicBool>,
}

struct SegmentAccumulatorConfig {
    writer: SegmentWriter,
    session_id: String,
    emit_segments: bool,
    sender: Option<CapturedSegmentSender>,
    max_segments: u64,
    sample_rate: u32,
    channels: u16,
    max_samples: usize,
    pipeline: CapturePipelineConfig,
    metrics: super::audio_capture::CaptureMetricsReporter,
    stop_requested: Arc<AtomicBool>,
}

impl SegmentAccumulator {
    fn new(config: SegmentAccumulatorConfig) -> Self {
        let min_frame = usize::from(config.channels.max(1));
        let capacity = cmp::max(config.max_samples, config.sample_rate as usize * min_frame);
        Self {
            writer: config.writer,
            session_id: config.session_id,
            emit_segments: config.emit_segments,
            sender: config.sender,
            samples: Vec::with_capacity(capacity),
            sample_rate: config.sample_rate,
            channels: config.channels.max(1),
            max_samples: align_to_frame(config.max_samples, config.channels),
            max_segments: config.max_segments,
            segments_emitted: 0,
            timeline_cursor_ms: 0,
            pipeline: config.pipeline,
            metrics: config.metrics,
            stop_requested: config.stop_requested,
        }
    }

    fn push_samples(&mut self, incoming: &[i16]) -> Result<(), MeetingRuntimeError> {
        let mut offset = 0;
        while offset < incoming.len() {
            if self.segments_emitted >= self.max_segments {
                return Err(MeetingRuntimeError::CaptureStreamError {
                    backend: CaptureBackend::Wasapi,
                    reason: "capture segment limit reached".to_string(),
                });
            }
            let remaining = self.max_samples.saturating_sub(self.samples.len());
            let take = remaining.min(incoming.len() - offset);
            self.samples
                .extend_from_slice(&incoming[offset..offset + take]);
            offset += take;
            if self.samples.len() >= self.max_samples {
                self.flush()?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), MeetingRuntimeError> {
        self.flush_with(FlushPolicy::Normal)
    }

    fn flush_final(&mut self) -> Result<(), MeetingRuntimeError> {
        self.flush_with(FlushPolicy::Final)
    }

    fn flush_with(&mut self, policy: FlushPolicy) -> Result<(), MeetingRuntimeError> {
        if self.samples.is_empty() {
            return Ok(());
        }
        let analysis = match policy {
            FlushPolicy::Normal => {
                analyze_mono_i16_segment(&self.samples, self.sample_rate, &self.pipeline)
            }
            FlushPolicy::Final => {
                analyze_mono_i16_final_flush(&self.samples, self.sample_rate, &self.pipeline)
            }
        };
        if analysis.should_drop {
            let frames_skipped = (self.samples.len() / usize::from(self.channels.max(1))) as u64;
            let skipped_ms = frames_skipped.saturating_mul(1_000) / u64::from(self.sample_rate);
            self.samples.clear();
            self.timeline_cursor_ms = self.timeline_cursor_ms.saturating_add(skipped_ms);
            self.metrics.record_silence_segment_dropped(
                frames_skipped,
                analysis.speech_frames,
                analysis.silence_frames,
                analysis.speech_ratio_bps,
                analysis.silence_ratio_bps,
            );
            return Ok(());
        }
        self.metrics.record_vad_analysis(
            analysis.speech_frames,
            analysis.silence_frames,
            analysis.speech_ratio_bps,
            analysis.silence_ratio_bps,
        );
        let mut segment = match self
            .writer
            .write_pcm_i16_segment(&self.session_id, &self.samples)
        {
            Ok(segment) => segment,
            Err(error) => {
                self.metrics
                    .record_segment_write_failure(backend_error_kind_for_segment(&error));
                self.metrics.record_backend_error(
                    backend_error_kind_for_segment(&error),
                    &error.to_string(),
                );
                return Err(error);
            }
        };
        self.samples.clear();
        let start_ms = self.timeline_cursor_ms;
        let end_ms = start_ms.saturating_add(segment.duration_ms);
        segment.sequence_number = self.segments_emitted.saturating_add(1);
        segment.start_ms = Some(start_ms);
        segment.end_ms = Some(end_ms);
        self.timeline_cursor_ms = end_ms;
        self.segments_emitted = self.segments_emitted.saturating_add(1);
        self.metrics
            .record_segment_written(segment.byte_length, segment.duration_ms);
        segment.capture_metrics_recorded = true;
        if self.emit_segments {
            self.emit_segment(segment)?;
        }
        Ok(())
    }

    fn emit_segment(&self, segment: CapturedMeetingSegment) -> Result<(), MeetingRuntimeError> {
        let Some(sender) = self.sender.as_ref() else {
            return Ok(());
        };
        match sender.try_send(segment) {
            CapturedSegmentEnqueueOutcome::Enqueued { .. } => Ok(()),
            CapturedSegmentEnqueueOutcome::DroppedNewest { segment } => {
                log::warn!(
                    "meeting capture segment queue full or closed; dropping managed segment"
                );
                cleanup_dropped_segment(segment);
                Ok(())
            }
            CapturedSegmentEnqueueOutcome::DroppedOldest {
                dropped_segment, ..
            } => {
                log::warn!("meeting capture segment queue full; dropped oldest managed segment");
                cleanup_dropped_segment(dropped_segment);
                Ok(())
            }
            CapturedSegmentEnqueueOutcome::StopCapture { segment } => {
                log::warn!("meeting capture stopped by segment backpressure policy");
                cleanup_dropped_segment(segment);
                self.stop_requested.store(true, Ordering::SeqCst);
                Err(MeetingRuntimeError::CaptureStreamError {
                    backend: CaptureBackend::Wasapi,
                    reason: "capture stopped by segment backpressure policy".to_string(),
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum FlushPolicy {
    Normal,
    Final,
}

#[cfg(target_os = "windows")]
fn writer_config_for_mix(
    request: &AudioCaptureStartRequest,
    mix_format: &MixFormat,
) -> SegmentWriterConfig {
    let effective = request.pipeline.effective();
    SegmentWriterConfig {
        sample_rate: mix_format.sample_rate,
        channels: 1,
        max_duration_ms: effective.effective_segment_duration_ms,
        max_bytes: effective.max_segment_bytes as u64,
        source_backend: CaptureBackend::Wasapi,
        transcript_source: request.transcript_source,
    }
}

#[cfg(target_os = "windows")]
fn max_segment_samples(
    sample_rate: u32,
    channels: u16,
    chunk_duration_ms: u64,
    max_memory_bytes: usize,
) -> usize {
    let duration_ms = super::types::CapturePipelineConfig {
        chunk_duration_ms,
        max_memory_bytes,
        ..super::types::CapturePipelineConfig::default()
    }
    .effective()
    .effective_segment_duration_ms;
    let by_duration =
        (u128::from(sample_rate) * u128::from(channels) * u128::from(duration_ms)) / 1_000;
    let by_bytes = max_memory_bytes.saturating_sub(44) / 2;
    let bounded = cmp::min(by_duration as usize, by_bytes.max(usize::from(channels)));
    align_to_frame(bounded, channels)
}

fn align_to_frame(samples: usize, channels: u16) -> usize {
    let channels = usize::from(channels.max(1));
    let aligned = samples.saturating_sub(samples % channels);
    cmp::max(aligned, channels)
}

fn cleanup_dropped_segment(segment: CapturedMeetingSegment) {
    let _ = std::fs::remove_file(segment.path);
}

fn backend_error_kind_for_segment(error: &MeetingRuntimeError) -> &'static str {
    match error {
        MeetingRuntimeError::SegmentWriteFailed { .. } => "segment_write_failed",
        MeetingRuntimeError::SegmentTooLarge { .. } => "segment_too_large",
        MeetingRuntimeError::StorageError { .. } => "storage_error",
        MeetingRuntimeError::InvalidConfig { .. } => "invalid_config",
        MeetingRuntimeError::CaptureStreamError { .. } => "capture_stream_error",
        _ => "meeting_runtime_error",
    }
}

#[cfg(target_os = "windows")]
fn report_startup_ok(startup_tx: &mut Option<mpsc::Sender<Result<(), MeetingRuntimeError>>>) {
    if let Some(sender) = startup_tx.take() {
        let _ = sender.send(Ok(()));
    }
}

#[cfg(target_os = "windows")]
fn report_startup_error(
    startup_tx: &mut Option<mpsc::Sender<Result<(), MeetingRuntimeError>>>,
    error: &MeetingRuntimeError,
) {
    if let Some(sender) = startup_tx.take() {
        let _ = sender.send(Err(error.clone()));
    }
}

#[cfg(target_os = "windows")]
fn backend_error_kind(error: &MeetingRuntimeError) -> &'static str {
    match error {
        MeetingRuntimeError::CaptureDeviceUnavailable { .. } => "capture_device_unavailable",
        MeetingRuntimeError::CaptureStartFailed { .. } => "capture_start_failed",
        MeetingRuntimeError::CaptureStreamError { .. } => "capture_stream_error",
        MeetingRuntimeError::CaptureUnavailable { .. } => "capture_unavailable",
        MeetingRuntimeError::CaptureStartupTimeout { .. } => "capture_startup_timeout",
        MeetingRuntimeError::CaptureStartupChannelClosed { .. } => "capture_startup_channel_closed",
        MeetingRuntimeError::SegmentWriteFailed { .. } => "segment_write_failed",
        MeetingRuntimeError::SegmentTooLarge { .. } => "segment_too_large",
        MeetingRuntimeError::StorageError { .. } => "storage_error",
        MeetingRuntimeError::InvalidConfig { .. } => "invalid_config",
        _ => "meeting_runtime_error",
    }
}

#[cfg(target_os = "windows")]
fn sanitize_windows_error(operation: &str, error: windows::core::Error) -> String {
    format!("{operation} failed: 0x{:08X}", error.code().0 as u32)
}

#[cfg(test)]
mod tests {
    use super::super::audio_capture::CaptureMetricsReporter;
    use super::*;
    use std::{
        sync::atomic::AtomicBool,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temp_root(name: &str) -> std::path::PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        std::env::temp_dir().join(format!("astra_wasapi_{name}_{suffix}"))
    }

    fn accumulator(
        root: std::path::PathBuf,
        metrics: CaptureMetricsReporter,
    ) -> SegmentAccumulator {
        accumulator_with_config(
            root,
            metrics,
            CapturePipelineConfig::default(),
            16_000,
            1_000,
        )
    }

    fn accumulator_with_config(
        root: std::path::PathBuf,
        metrics: CaptureMetricsReporter,
        pipeline: CapturePipelineConfig,
        max_samples: usize,
        max_duration_ms: u64,
    ) -> SegmentAccumulator {
        SegmentAccumulator::new(SegmentAccumulatorConfig {
            writer: SegmentWriter::new(
                root,
                SegmentWriterConfig {
                    sample_rate: 16_000,
                    channels: 1,
                    max_duration_ms,
                    max_bytes: 256_000,
                    source_backend: CaptureBackend::Wasapi,
                    transcript_source: TranscriptSource::SystemAudio,
                },
            ),
            session_id: "session_1".to_string(),
            emit_segments: false,
            sender: None,
            max_segments: 10,
            sample_rate: 16_000,
            channels: 1,
            max_samples,
            pipeline,
            metrics,
            stop_requested: Arc::new(AtomicBool::new(false)),
        })
    }

    #[test]
    fn segment_accumulator_drops_silence_without_writing_segment() {
        let root = temp_root("silence_drop");
        let metrics = CaptureMetricsReporter::new();
        let mut accumulator = accumulator(root.clone(), metrics.clone());

        accumulator
            .push_samples(&vec![0_i16; 16_000])
            .expect("silence push");

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.segments_written, 0);
        assert_eq!(snapshot.dropped_silence_segments, 1);
        assert_eq!(snapshot.silence_frames_skipped, 16_000);
        assert!(!root.join("session_1").join("segments").exists());
    }

    #[test]
    fn segment_accumulator_writes_speech_segment() {
        let root = temp_root("speech_write");
        let metrics = CaptureMetricsReporter::new();
        let mut accumulator = accumulator(root.clone(), metrics.clone());

        accumulator
            .push_samples(&vec![3_000_i16; 16_000])
            .expect("speech push");

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.segments_written, 1);
        assert_eq!(snapshot.dropped_silence_segments, 0);
        assert!(snapshot.last_speech_ratio_bps > 0);
        let count = std::fs::read_dir(root.join("session_1").join("segments"))
            .expect("segments dir")
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn segment_accumulator_final_flush_keeps_short_speech() {
        let root = temp_root("final_short_speech");
        let metrics = CaptureMetricsReporter::new();
        let mut accumulator = accumulator_with_config(
            root.clone(),
            metrics.clone(),
            CapturePipelineConfig::default(),
            32_000,
            2_000,
        );
        let mut final_phrase = vec![0_i16; 16_000];
        final_phrase
            .iter_mut()
            .rev()
            .take(800)
            .for_each(|sample| *sample = 300);

        accumulator
            .push_samples(&final_phrase)
            .expect("final phrase push");
        accumulator.flush_final().expect("final flush");

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.segments_written, 1);
        assert_eq!(snapshot.dropped_silence_segments, 0);
        let count = std::fs::read_dir(root.join("session_1").join("segments"))
            .expect("segments dir")
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn segment_accumulator_final_flush_drops_pure_silence() {
        let root = temp_root("final_silence");
        let metrics = CaptureMetricsReporter::new();
        let mut accumulator = accumulator_with_config(
            root.clone(),
            metrics.clone(),
            CapturePipelineConfig::default(),
            32_000,
            2_000,
        );

        accumulator
            .push_samples(&vec![0_i16; 1_600])
            .expect("silence push");
        accumulator.flush_final().expect("final silence flush");

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.segments_written, 0);
        assert_eq!(snapshot.dropped_silence_segments, 1);
        assert_eq!(snapshot.silence_frames_skipped, 1_600);
        assert!(!root.join("session_1").join("segments").exists());
    }
}
