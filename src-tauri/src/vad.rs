use serde::Serialize;

pub const VAD_BACKEND_NAME: &str = "adaptive_energy_v3";

#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    pub target_sample_rate: u32,
    pub min_start_rms: f32,
    pub min_end_rms: f32,
    pub max_start_rms: f32,
    pub max_end_rms: f32,
    pub noise_start_multiplier: f32,
    pub noise_end_multiplier: f32,
    pub noise_floor_alpha: f32,
    pub initial_noise_floor: f32,
    pub rms_smoothing_alpha: f32,
    pub min_speech_ms: u64,
    pub min_trailing_silence_ms: u64,
    pub end_silence_ms: u64,
    pub min_utterance_ms: u64,
    pub max_utterance_ms: u64,
    pub pre_roll_ms: u64,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16_000,
            min_start_rms: 0.006,
            min_end_rms: 0.0038,
            max_start_rms: 0.036,
            max_end_rms: 0.022,
            noise_start_multiplier: 2.2,
            noise_end_multiplier: 1.45,
            noise_floor_alpha: 0.045,
            initial_noise_floor: 0.0012,
            rms_smoothing_alpha: 0.35,
            min_speech_ms: 90,
            min_trailing_silence_ms: 180,
            end_silence_ms: 620,
            min_utterance_ms: 240,
            max_utterance_ms: 14_000,
            pre_roll_ms: 280,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    None,
    SpeechStarted,
    SpeechContinued,
    SpeechEnded,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct VadFrameSnapshot {
    pub backend: &'static str,
    pub rms: f32,
    pub smoothed_rms: f32,
    pub noise_floor: f32,
    pub start_threshold: f32,
    pub end_threshold: f32,
    pub start_gate_ms: u64,
    pub speech_ms: u64,
    pub silence_ms: u64,
    pub utterance_ms: u64,
    pub in_speech: bool,
}

impl Default for VadFrameSnapshot {
    fn default() -> Self {
        let config = VadConfig::default();
        Self {
            backend: VAD_BACKEND_NAME,
            rms: 0.0,
            smoothed_rms: 0.0,
            noise_floor: config.initial_noise_floor,
            start_threshold: config.min_start_rms,
            end_threshold: config.min_end_rms,
            start_gate_ms: 0,
            speech_ms: 0,
            silence_ms: 0,
            utterance_ms: 0,
            in_speech: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VadUpdate {
    pub event: VadEvent,
    pub snapshot: VadFrameSnapshot,
}

#[derive(Debug)]
pub struct AdaptiveEnergyVad {
    config: VadConfig,
    speech_ms: u64,
    silence_ms: u64,
    utterance_ms: u64,
    start_gate_ms: u64,
    in_speech: bool,
    noise_floor: f32,
    smoothed_rms: f32,
}

impl Default for AdaptiveEnergyVad {
    fn default() -> Self {
        Self::new(VadConfig::default())
    }
}

impl AdaptiveEnergyVad {
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            speech_ms: 0,
            silence_ms: 0,
            utterance_ms: 0,
            start_gate_ms: 0,
            in_speech: false,
            noise_floor: config.initial_noise_floor,
            smoothed_rms: 0.0,
        }
    }

    pub fn config(&self) -> VadConfig {
        self.config
    }

    pub fn update(&mut self, rms: f32, duration_ms: u64) -> VadUpdate {
        let rms = rms.max(0.0);
        self.smoothed_rms = if self.smoothed_rms <= 0.0 {
            rms
        } else {
            self.smoothed_rms * (1.0 - self.config.rms_smoothing_alpha)
                + rms * self.config.rms_smoothing_alpha
        };

        let start_threshold = self.start_threshold();
        let end_threshold = self.end_threshold();
        let event = if self.in_speech {
            self.update_in_speech(rms, duration_ms, end_threshold)
        } else {
            self.update_idle(rms, duration_ms, start_threshold)
        };

        VadUpdate {
            event,
            snapshot: self.snapshot(rms, start_threshold, end_threshold),
        }
    }

    pub fn reset(&mut self) {
        self.speech_ms = 0;
        self.silence_ms = 0;
        self.utterance_ms = 0;
        self.start_gate_ms = 0;
        self.in_speech = false;
    }

    pub fn snapshot(&self, rms: f32, start_threshold: f32, end_threshold: f32) -> VadFrameSnapshot {
        VadFrameSnapshot {
            backend: VAD_BACKEND_NAME,
            rms,
            smoothed_rms: self.smoothed_rms,
            noise_floor: self.noise_floor,
            start_threshold,
            end_threshold,
            start_gate_ms: self.start_gate_ms,
            speech_ms: self.speech_ms,
            silence_ms: self.silence_ms,
            utterance_ms: self.utterance_ms,
            in_speech: self.in_speech,
        }
    }

    fn update_idle(&mut self, rms: f32, duration_ms: u64, start_threshold: f32) -> VadEvent {
        let speech_like = self.smoothed_rms >= start_threshold || rms >= start_threshold * 1.1;

        if !speech_like {
            self.update_noise_floor(rms);
            self.speech_ms = 0;
            self.start_gate_ms = 0;
            return VadEvent::None;
        }

        self.start_gate_ms += duration_ms;
        self.speech_ms += duration_ms;
        if self.start_gate_ms >= self.config.min_speech_ms {
            self.in_speech = true;
            self.utterance_ms = self.speech_ms;
            self.silence_ms = 0;
            return VadEvent::SpeechStarted;
        }

        VadEvent::None
    }

    fn update_in_speech(&mut self, rms: f32, duration_ms: u64, end_threshold: f32) -> VadEvent {
        self.utterance_ms += duration_ms;

        let silence_like = self.smoothed_rms <= end_threshold && rms <= end_threshold * 1.08;
        if silence_like {
            self.silence_ms += duration_ms;
        } else {
            self.silence_ms = 0;
        }

        if self.silence_ms >= self.config.min_trailing_silence_ms {
            self.update_noise_floor(rms.min(end_threshold));
        }

        if self.silence_ms >= self.config.end_silence_ms
            || self.utterance_ms >= self.config.max_utterance_ms
        {
            self.reset();
            return VadEvent::SpeechEnded;
        }

        VadEvent::SpeechContinued
    }

    fn update_noise_floor(&mut self, rms: f32) {
        let sample = rms.max(0.000_4);
        self.noise_floor = self.noise_floor * (1.0 - self.config.noise_floor_alpha)
            + sample * self.config.noise_floor_alpha;
    }

    fn start_threshold(&self) -> f32 {
        (self.noise_floor * self.config.noise_start_multiplier)
            .max(self.config.min_start_rms)
            .min(self.config.max_start_rms)
    }

    fn end_threshold(&self) -> f32 {
        (self.noise_floor * self.config.noise_end_multiplier)
            .max(self.config.min_end_rms)
            .min(self.config.max_end_rms)
    }
}

pub fn normalize_samples(samples: &[f32]) -> Vec<f32> {
    samples
        .iter()
        .map(|sample| sample.clamp(-1.0, 1.0))
        .collect()
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum = samples.iter().map(|sample| sample * sample).sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}

pub fn samples_duration_ms(sample_count: usize, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        return 0;
    }

    sample_count as u64 * 1000 / sample_rate as u64
}

#[cfg(test)]
mod tests {
    use super::{AdaptiveEnergyVad, VadEvent};

    #[test]
    fn ignores_low_level_noise() {
        let mut vad = AdaptiveEnergyVad::default();
        for _ in 0..20 {
            let update = vad.update(0.0035, 60);
            assert_eq!(update.event, VadEvent::None);
        }
    }

    #[test]
    fn detects_sustained_speech_then_end_silence_faster() {
        let mut vad = AdaptiveEnergyVad::default();
        assert_eq!(vad.update(0.028, 60).event, VadEvent::None);
        assert_eq!(vad.update(0.029, 60).event, VadEvent::SpeechStarted);

        for _ in 0..9 {
            assert_eq!(vad.update(0.0025, 60).event, VadEvent::SpeechContinued);
        }
        assert_eq!(vad.update(0.0025, 60).event, VadEvent::SpeechEnded);
    }
}
