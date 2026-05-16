//! Speaker diarization — voice feature extraction and clustering
//!
//! Extracts voice features (MFCC-like) from audio frames and clusters them
//! into speaker profiles using a simple cosine-similarity approach.

use super::types::*;

pub struct SpeakerDiarization {
    pub speakers: Vec<SpeakerProfile>,
    pub max_speakers: usize,
    pub next_speaker_id: u32,
}

impl SpeakerDiarization {
    pub fn new(max_speakers: usize) -> Self {
        Self {
            speakers: Vec::new(),
            max_speakers,
            next_speaker_id: 1,
        }
    }

    /// Assign a speaker ID to an audio frame based on voice features.
    /// If the frame doesn't match an existing speaker and we have room,
    /// creates a new speaker profile.
    pub fn assign_speaker(&mut self, frame: &AudioFrame) -> Option<String> {
        if frame.samples.is_empty() {
            return None;
        }

        // Extract simple features: RMS energy + zero-crossing rate
        let features = Self::extract_voice_features(&frame.samples);

        // Find best matching speaker
        let best_match = self.find_best_match(&features);

        match best_match {
            Some((idx, similarity)) if similarity > 0.5 => {
                // Update the speaker's features (running average)
                let speaker = &mut self.speakers[idx];
                speaker.sample_count += 1;
                // Blend features
                for (f, n) in speaker.voice_features.iter_mut().zip(features.iter()) {
                    *f = (*f + n) / 2.0;
                }
                Some(speaker.id.clone())
            }
            _ if self.speakers.len() < self.max_speakers => {
                // New speaker
                let id = format!("speaker_{}", self.next_speaker_id);
                self.next_speaker_id += 1;
                self.speakers.push(SpeakerProfile {
                    id: id.clone(),
                    voice_features: features.clone(),
                    sample_count: 1,
                });
                Some(id)
            }
            _ => None,
        }
    }

    /// Get a summary of all identified speakers.
    pub fn get_speakers(&self) -> Vec<SpeakerProfile> {
        self.speakers.clone()
    }

    /// Extract simple voice features for matching.
    fn extract_voice_features(samples: &[f32]) -> Vec<f32> {
        // Simple features for MVP:
        // 1. RMS energy
        // 2. Zero-crossing rate (2 features)
        // 3. First 4 MFCC-like stats (mean, std, min, max of energy bands)

        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt();

        let zcr = samples
            .iter()
            .zip(samples.iter().skip(1))
            .filter(|&(a, b)| (*a >= 0.0) != (*b >= 0.0))
            .count() as f32
            / samples.len().max(1) as f32;

        // Split spectrum into 4 bands (simplified from actual FFT)
        let band_size = samples.len().max(1) / 4;
        let band_stats: Vec<f32> = (0..4)
            .map(|i| {
                let start = i * band_size;
                let end = (start + band_size).min(samples.len());
                let band = &samples[start..end];
                let mean = band.iter().map(|s| *s * *s).sum::<f32>() / band.len().max(1) as f32;
                mean.sqrt()
            })
            .collect();

        // Normalize features to [0, 1]
        let mut features = vec![rms, zcr];
        features.extend(band_stats);
        features
    }

    /// Find best matching speaker by cosine similarity of features.
    fn find_best_match(&self, features: &[f32]) -> Option<(usize, f32)> {
        let mut best_idx: Option<usize> = None;
        let mut best_sim: f32 = 0.0;

        for (idx, speaker) in self.speakers.iter().enumerate() {
            let sim = Self::cosine_similarity(&speaker.voice_features, features);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(idx);
            }
        }

        best_idx.map(|idx| (idx, best_sim))
    }

    /// Compute cosine similarity between two feature vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}
