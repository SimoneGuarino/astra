//! Deterministic mono PCM16 conversion and RMS VAD helpers for meeting capture.
//!
//! This module intentionally avoids model calls and large DSP dependencies. It
//! converts WASAPI mix packets into mono PCM16 frames and makes a conservative
//! local silence decision before managed WAV segments reach STT.

use super::types::CapturePipelineConfig;

const BASIS_POINTS: u32 = 10_000;
const FRAME_MS: u64 = 20;
const FINAL_FLUSH_MIN_SPEECH_MS: u64 = 80;
const FINAL_FLUSH_MIN_SPEECH_RATIO_BPS: u16 = 100;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioQualityResult {
    pub samples: Vec<i16>,
    pub input_frames: u64,
    pub output_frames: u64,
    pub clipped_samples: u64,
    pub peak_abs: u16,
    pub rms_bps: u16,
    pub normalization_gain_bps: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VadAnalysis {
    pub speech_detected: bool,
    pub should_drop: bool,
    pub speech_frames: u64,
    pub silence_frames: u64,
    pub speech_ratio_bps: u16,
    pub silence_ratio_bps: u16,
    pub rms_bps: u16,
}

pub fn convert_interleaved_f32_to_i16_mono(samples: &[f32], channels: u16) -> AudioQualityResult {
    let channels = usize::from(channels.max(1));
    let input_frames = (samples.len() / channels) as u64;
    let mut mono = Vec::with_capacity(input_frames as usize);
    let mut clipped_samples = 0_u64;

    for frame in samples.chunks_exact(channels) {
        let mut sum = 0_i32;
        for sample in frame {
            let clamped = sample.clamp(-1.0, 1.0);
            if *sample < -1.0 || *sample > 1.0 {
                clipped_samples = clipped_samples.saturating_add(1);
            }
            sum = sum.saturating_add((clamped * i16::MAX as f32).round() as i32);
        }
        mono.push(clamp_i32_to_i16(sum / channels as i32));
    }

    quality_result(mono, input_frames, clipped_samples)
}

pub fn downmix_interleaved_i16_to_mono(samples: &[i16], channels: u16) -> AudioQualityResult {
    let channels = usize::from(channels.max(1));
    let input_frames = (samples.len() / channels) as u64;
    let mut mono = Vec::with_capacity(input_frames as usize);

    for frame in samples.chunks_exact(channels) {
        let sum = frame.iter().fold(0_i32, |accumulator, sample| {
            accumulator + i32::from(*sample)
        });
        mono.push(clamp_i32_to_i16(sum / channels as i32));
    }

    quality_result(mono, input_frames, 0)
}

pub fn analyze_mono_i16_segment(
    samples: &[i16],
    sample_rate: u32,
    pipeline: &CapturePipelineConfig,
) -> VadAnalysis {
    analyze_mono_i16_with_policy(
        samples,
        sample_rate,
        pipeline,
        VadPolicy::standard(pipeline),
    )
}

pub fn analyze_mono_i16_final_flush(
    samples: &[i16],
    sample_rate: u32,
    pipeline: &CapturePipelineConfig,
) -> VadAnalysis {
    analyze_mono_i16_with_policy(
        samples,
        sample_rate,
        pipeline,
        VadPolicy::final_flush(pipeline),
    )
}

fn analyze_mono_i16_with_policy(
    samples: &[i16],
    sample_rate: u32,
    pipeline: &CapturePipelineConfig,
    policy: VadPolicy,
) -> VadAnalysis {
    if !pipeline.vad_enabled {
        return VadAnalysis {
            speech_detected: !samples.is_empty(),
            should_drop: false,
            speech_frames: if samples.is_empty() { 0 } else { 1 },
            silence_frames: 0,
            speech_ratio_bps: if samples.is_empty() { 0 } else { 10_000 },
            silence_ratio_bps: if samples.is_empty() { 10_000 } else { 0 },
            rms_bps: rms_bps(samples),
        };
    }

    if samples.is_empty() || sample_rate == 0 {
        return VadAnalysis {
            speech_detected: false,
            should_drop: true,
            speech_frames: 0,
            silence_frames: 1,
            speech_ratio_bps: 0,
            silence_ratio_bps: 10_000,
            rms_bps: 0,
        };
    }

    let frame_len = ((u64::from(sample_rate) * FRAME_MS) / 1_000).max(1) as usize;
    let mut speech_frames = 0_u64;
    let mut silence_frames = 0_u64;

    for frame in samples.chunks(frame_len) {
        if rms_pcm(frame) > f64::from(pipeline.vad_silence_threshold_pcm) {
            speech_frames = speech_frames.saturating_add(1);
        } else {
            silence_frames = silence_frames.saturating_add(1);
        }
    }

    let total_frames = speech_frames.saturating_add(silence_frames).max(1);
    let speech_ratio_bps = ratio_bps(speech_frames, total_frames);
    let silence_ratio_bps = ratio_bps(silence_frames, total_frames);
    let speech_ms = speech_frames.saturating_mul(FRAME_MS);
    let silence_ms = silence_frames.saturating_mul(FRAME_MS);
    let segment_rms = rms_pcm(samples);
    let segment_peak = peak_abs(samples);
    let speech_duration_pass = speech_ms >= policy.min_speech_ms;
    let speech_ratio_pass =
        speech_frames > 0 && speech_ratio_bps >= policy.min_speech_ratio_bps.min(10_000);
    let meaningful_audio = segment_rms > f64::from(policy.meaningful_rms_threshold_pcm)
        || segment_peak > policy.meaningful_peak_threshold_pcm;
    let speech_detected = speech_duration_pass || speech_ratio_pass || meaningful_audio;
    let should_drop = !speech_duration_pass
        && !speech_ratio_pass
        && !meaningful_audio
        && silence_ms >= policy.min_silence_ms;

    VadAnalysis {
        speech_detected,
        should_drop,
        speech_frames,
        silence_frames,
        speech_ratio_bps,
        silence_ratio_bps,
        rms_bps: rms_bps(samples),
    }
}

#[derive(Debug, Clone, Copy)]
struct VadPolicy {
    min_speech_ms: u64,
    min_silence_ms: u64,
    min_speech_ratio_bps: u16,
    meaningful_rms_threshold_pcm: u16,
    meaningful_peak_threshold_pcm: u16,
}

impl VadPolicy {
    fn standard(pipeline: &CapturePipelineConfig) -> Self {
        Self {
            min_speech_ms: pipeline.vad_min_speech_ms,
            min_silence_ms: pipeline.vad_min_silence_ms,
            min_speech_ratio_bps: pipeline.vad_min_speech_ratio_bps,
            meaningful_rms_threshold_pcm: pipeline.vad_silence_threshold_pcm,
            meaningful_peak_threshold_pcm: pipeline.vad_silence_threshold_pcm,
        }
    }

    fn final_flush(pipeline: &CapturePipelineConfig) -> Self {
        let final_threshold = (pipeline.vad_silence_threshold_pcm / 2).max(1);
        Self {
            min_speech_ms: pipeline.vad_min_speech_ms.min(FINAL_FLUSH_MIN_SPEECH_MS),
            min_silence_ms: 0,
            min_speech_ratio_bps: pipeline
                .vad_min_speech_ratio_bps
                .min(FINAL_FLUSH_MIN_SPEECH_RATIO_BPS),
            meaningful_rms_threshold_pcm: final_threshold,
            meaningful_peak_threshold_pcm: final_threshold,
        }
    }
}

fn quality_result(
    samples: Vec<i16>,
    input_frames: u64,
    clipped_samples: u64,
) -> AudioQualityResult {
    let output_frames = samples.len() as u64;
    AudioQualityResult {
        peak_abs: peak_abs(&samples),
        rms_bps: rms_bps(&samples),
        normalization_gain_bps: 10_000,
        samples,
        input_frames,
        output_frames,
        clipped_samples,
    }
}

fn clamp_i32_to_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn peak_abs(samples: &[i16]) -> u16 {
    samples
        .iter()
        .map(|sample| sample.unsigned_abs())
        .max()
        .unwrap_or(0)
}

fn rms_bps(samples: &[i16]) -> u16 {
    if samples.is_empty() {
        return 0;
    }
    let rms = rms_pcm(samples);
    ((rms / f64::from(i16::MAX)) * f64::from(BASIS_POINTS))
        .round()
        .clamp(0.0, f64::from(BASIS_POINTS)) as u16
}

fn rms_pcm(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares = samples.iter().fold(0_f64, |accumulator, sample| {
        let value = f64::from(*sample);
        accumulator + (value * value)
    });
    (sum_squares / samples.len() as f64).sqrt()
}

fn ratio_bps(numerator: u64, denominator: u64) -> u16 {
    if denominator == 0 {
        return 0;
    }
    ((numerator.saturating_mul(BASIS_POINTS as u64)) / denominator).min(10_000) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_i16_downmixes_to_mono() {
        let output = downmix_interleaved_i16_to_mono(&[1000, -1000, 500, 1500], 2);

        assert_eq!(output.samples, vec![0, 1000]);
        assert_eq!(output.input_frames, 2);
        assert_eq!(output.output_frames, 2);
        assert_eq!(output.clipped_samples, 0);
    }

    #[test]
    fn f32_conversion_clamps_and_reports_clipping() {
        let output = convert_interleaved_f32_to_i16_mono(&[2.0, -2.0, 0.5, 0.5], 2);

        assert_eq!(output.samples.len(), 2);
        assert_eq!(output.samples[0], 0);
        assert_eq!(output.clipped_samples, 2);
        assert!(output.peak_abs > 0);
    }

    #[test]
    fn vad_drops_pure_silence() {
        let config = CapturePipelineConfig::default();
        let samples = vec![0_i16; 16_000];

        let analysis = analyze_mono_i16_segment(&samples, 16_000, &config);

        assert!(!analysis.speech_detected);
        assert!(analysis.should_drop);
        assert_eq!(analysis.speech_ratio_bps, 0);
    }

    #[test]
    fn vad_detects_sustained_speech() {
        let config = CapturePipelineConfig::default();
        let samples = vec![2_500_i16; 16_000];

        let analysis = analyze_mono_i16_segment(&samples, 16_000, &config);

        assert!(analysis.speech_detected);
        assert!(!analysis.should_drop);
        assert!(analysis.speech_ratio_bps >= config.vad_min_speech_ratio_bps);
    }
}
