//! Privacy control — consent flow, data retention policy, and privacy state management
//!
//! Manages user consent for meeting recording, per-app consent, data retention,
//! and privacy mode controls (pausing, redaction).

use super::types::*;
use std::collections::HashMap;

/// Privacy state for the meeting engine.
#[derive(Debug, Clone, PartialEq)]
pub struct PrivacyState {
    pub global_enabled: bool,
    pub consent_given: bool,
    pub per_app_consent: HashMap<String, bool>,
    pub data_retention: DataRetentionPolicy,
    pub privacy_mode: String,
}

/// Default privacy state for a fresh install.
fn default_privacy_state() -> PrivacyState {
    PrivacyState {
        global_enabled: false,
        consent_given: false,
        per_app_consent: HashMap::new(),
        data_retention: DataRetentionPolicy {
            raw_audio_days: 1,
            transcript_days: 30,
            summary_days: 90,
            action_items_days: 0,
            decisions_days: 90,
        },
        privacy_mode: "default".to_string(),
    }
}

impl PrivacyState {
    pub fn new() -> Self {
        default_privacy_state()
    }

    pub fn grant_consent(&mut self, app_name: &str) {
        let app_name = normalize_meeting_app_name(app_name);
        if app_name.is_empty() {
            return;
        }
        self.global_enabled = true;
        self.consent_given = true;
        self.per_app_consent.insert(app_name, true);
    }

    pub fn revoke_consent(&mut self, app_name: &str) {
        let app_name = normalize_meeting_app_name(app_name);
        if app_name.is_empty() {
            return;
        }
        self.per_app_consent.insert(app_name, false);
        self.recalculate_scoped_consent();
    }

    pub fn revoke_all_consent(&mut self) {
        self.consent_given = false;
        self.global_enabled = false;
        self.per_app_consent.clear();
    }

    pub fn can_record(&self, app_name: &str) -> bool {
        if !self.global_enabled || !self.consent_given {
            return false;
        }
        let app_name = normalize_meeting_app_name(app_name);
        if self.per_app_consent.is_empty() {
            return true;
        }
        self.per_app_consent
            .get(&app_name)
            .copied()
            .unwrap_or(false)
    }

    pub fn get_per_app_consent(&self) -> &HashMap<String, bool> {
        &self.per_app_consent
    }

    pub fn set_privacy_mode(&mut self, mode: &str) {
        self.privacy_mode = mode.to_string();
    }

    pub fn is_redaction_mode(&self) -> bool {
        self.privacy_mode == "redact"
    }

    pub fn is_pause_mode(&self) -> bool {
        self.privacy_mode == "pause"
    }

    pub fn get_data_retention(&self) -> &DataRetentionPolicy {
        &self.data_retention
    }

    fn recalculate_scoped_consent(&mut self) {
        let any_allowed = self.per_app_consent.values().any(|allowed| *allowed);
        self.consent_given = any_allowed;
        self.global_enabled = any_allowed;
    }
}

impl Default for PrivacyState {
    fn default() -> Self {
        Self::new()
    }
}

impl PrivacyControl {
    pub fn new() -> Self {
        Self {
            global_enabled: false,
            consent_given: false,
            per_app_consent: HashMap::new(),
            data_retention: DataRetentionPolicy {
                raw_audio_days: 1,
                transcript_days: 30,
                summary_days: 90,
                action_items_days: 0,
                decisions_days: 90,
            },
            mode: "default".to_string(),
        }
    }

    pub fn get_state(&self) -> PrivacyState {
        PrivacyState {
            global_enabled: self.global_enabled,
            consent_given: self.consent_given,
            per_app_consent: self.per_app_consent.clone(),
            data_retention: self.data_retention.clone(),
            privacy_mode: self.mode.clone(),
        }
    }

    pub fn grant_consent(&mut self, app_name: &str) {
        let app_name = normalize_meeting_app_name(app_name);
        if app_name.is_empty() {
            return;
        }
        self.global_enabled = true;
        self.consent_given = true;
        self.per_app_consent.insert(app_name, true);
    }

    pub fn revoke_consent(&mut self, app_name: &str) {
        let app_name = normalize_meeting_app_name(app_name);
        if app_name.is_empty() {
            return;
        }
        self.per_app_consent.insert(app_name, false);
        self.recalculate_scoped_consent();
    }

    pub fn revoke_all_consent(&mut self) {
        self.consent_given = false;
        self.global_enabled = false;
        self.per_app_consent.clear();
    }

    pub fn can_record(&self, app_name: &str) -> bool {
        if !self.global_enabled || !self.consent_given {
            return false;
        }
        let app_name = normalize_meeting_app_name(app_name);
        if self.per_app_consent.is_empty() {
            return true;
        }
        self.per_app_consent
            .get(&app_name)
            .copied()
            .unwrap_or(false)
    }

    pub fn set_privacy_mode(&mut self, mode: &str) {
        self.mode = mode.to_string();
    }

    pub fn get_data_retention(&self) -> &DataRetentionPolicy {
        &self.data_retention
    }

    fn recalculate_scoped_consent(&mut self) {
        let any_allowed = self.per_app_consent.values().any(|allowed| *allowed);
        self.consent_given = any_allowed;
        self.global_enabled = any_allowed;
    }
}

impl Default for PrivacyControl {
    fn default() -> Self {
        Self::new()
    }
}
