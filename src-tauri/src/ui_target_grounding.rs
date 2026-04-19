use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UITargetRole {
    SearchInput,
    RankedResult,
    Button,
    Link,
    TextInput,
    Unknown,
}

impl UITargetRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SearchInput => "search_input",
            Self::RankedResult => "ranked_result",
            Self::Button => "button",
            Self::Link => "link",
            Self::TextInput => "text_input",
            Self::Unknown => "unknown",
        }
    }

    pub fn from_value(value: Option<&str>) -> Self {
        let Some(value) = value else {
            return Self::Unknown;
        };

        match normalize_label(value).as_str() {
            "search_input" | "search_box" | "searchbar" | "search_bar" => Self::SearchInput,
            "ranked_result" | "first_result" | "result" | "search_result" => Self::RankedResult,
            "button" | "submit_button" => Self::Button,
            "link" | "anchor" => Self::Link,
            "text_input" | "input" | "field" | "textbox" => Self::TextInput,
            _ => Self::Unknown,
        }
    }

    fn matches_requested(&self, requested: &Self) -> bool {
        if requested == &Self::Unknown || self == requested {
            return true;
        }

        matches!(
            (self, requested),
            (Self::TextInput, Self::SearchInput)
                | (Self::SearchInput, Self::TextInput)
                | (Self::Link, Self::RankedResult)
                | (Self::RankedResult, Self::Link)
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetGroundingSource {
    WorkflowMetadata,
    ScreenAnalysis,
    RecentContext,
    UserProvided,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetAction {
    Focus,
    Click,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetRegion {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    #[serde(default = "default_coordinate_space")]
    pub coordinate_space: String,
}

impl TargetRegion {
    pub fn center(&self) -> (f64, f64) {
        (self.x + (self.width / 2.0), self.y + (self.height / 2.0))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UITargetCandidate {
    pub candidate_id: String,
    pub role: UITargetRole,
    #[serde(default)]
    pub region: Option<TargetRegion>,
    #[serde(default)]
    pub center_x: Option<f64>,
    #[serde(default)]
    pub center_y: Option<f64>,
    #[serde(default)]
    pub app_hint: Option<String>,
    #[serde(default)]
    pub provider_hint: Option<String>,
    pub confidence: f32,
    pub source: TargetGroundingSource,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub rank: Option<u32>,
    #[serde(default)]
    pub observed_at_ms: Option<u64>,
    #[serde(default = "default_reuse_eligible")]
    pub reuse_eligible: bool,
    pub supports_focus: bool,
    pub supports_click: bool,
    pub rationale: String,
}

impl UITargetCandidate {
    pub fn has_point(&self) -> bool {
        self.center_point().is_some()
    }

    pub fn center_point(&self) -> Option<(f64, f64)> {
        match (self.center_x, self.center_y) {
            (Some(x), Some(y)) if x.is_finite() && y.is_finite() => Some((x, y)),
            _ => self.region.as_ref().map(TargetRegion::center),
        }
    }

    pub fn execution_payload(&self) -> Value {
        let (center_x, center_y) = self
            .center_point()
            .map(|(x, y)| (Some(x), Some(y)))
            .unwrap_or((None, None));
        json!({
            "candidate_id": self.candidate_id,
            "role": self.role.as_str(),
            "region": self.region,
            "center_x": center_x,
            "center_y": center_y,
            "app_hint": self.app_hint,
            "provider_hint": self.provider_hint,
            "confidence": self.confidence,
            "source": self.source,
            "label": self.label,
            "rank": self.rank,
            "observed_at_ms": self.observed_at_ms,
            "reuse_eligible": self.reuse_eligible,
            "supports_focus": self.supports_focus,
            "supports_click": self.supports_click,
            "rationale": self.rationale,
        })
    }

    pub fn with_source(mut self, source: TargetGroundingSource) -> Self {
        self.source = source;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetGroundingRequest {
    pub requested_role: UITargetRole,
    #[serde(default)]
    pub target: Value,
    #[serde(default)]
    pub selection: Value,
    #[serde(default)]
    pub screen_candidates: Vec<UITargetCandidate>,
    #[serde(default)]
    pub recent_candidates: Vec<UITargetCandidate>,
    #[serde(default)]
    pub app_hint: Option<String>,
    #[serde(default)]
    pub provider_hint: Option<String>,
    #[serde(default)]
    pub rank_hint: Option<u32>,
    #[serde(default)]
    pub allow_recent_reuse: bool,
    #[serde(default)]
    pub now_ms: Option<u64>,
    #[serde(default = "default_recent_target_max_age_ms")]
    pub max_recent_age_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetGroundingState {
    pub requested_role: UITargetRole,
    pub candidates: Vec<UITargetCandidate>,
    #[serde(default)]
    pub current_app_hint: Option<String>,
    #[serde(default)]
    pub provider_hint: Option<String>,
    pub sufficient_for_selection: bool,
    pub ambiguous: bool,
    #[serde(default)]
    pub uncertainty: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSelectionPolicy {
    pub min_focus_confidence: f32,
    pub min_click_confidence: f32,
    pub ambiguity_margin: f32,
}

impl Default for TargetSelectionPolicy {
    fn default() -> Self {
        Self {
            min_focus_confidence: 0.78,
            min_click_confidence: 0.86,
            ambiguity_margin: 0.08,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetSelectionStatus {
    Selected,
    NoCandidates,
    LowConfidence,
    Ambiguous,
    UnsupportedTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSelection {
    pub status: TargetSelectionStatus,
    #[serde(default)]
    pub selected_candidate: Option<UITargetCandidate>,
    #[serde(default)]
    pub considered_candidates: Vec<UITargetCandidate>,
    pub required_confidence: f32,
    pub reason: String,
}

impl TargetSelection {
    pub fn selected(
        candidate: UITargetCandidate,
        considered: Vec<UITargetCandidate>,
        required_confidence: f32,
    ) -> Self {
        Self {
            status: TargetSelectionStatus::Selected,
            selected_candidate: Some(candidate),
            considered_candidates: considered,
            required_confidence,
            reason: "A single high-confidence target candidate satisfied the selection policy."
                .into(),
        }
    }

    pub fn rejected(
        status: TargetSelectionStatus,
        considered: Vec<UITargetCandidate>,
        required_confidence: f32,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            status,
            selected_candidate: None,
            considered_candidates: considered,
            required_confidence,
            reason: reason.into(),
        }
    }
}

pub fn structured_candidates_from_value(
    value: &Value,
    requested_role: &UITargetRole,
    app_hint: Option<&str>,
    provider_hint: Option<&str>,
    source: TargetGroundingSource,
) -> Vec<UITargetCandidate> {
    let mut candidates = Vec::new();
    collect_candidates_from_value(
        &mut candidates,
        value,
        requested_role,
        app_hint,
        provider_hint,
        source,
    );
    candidates
}

pub fn ground_targets_for_request(request: &TargetGroundingRequest) -> TargetGroundingState {
    let mut candidates = Vec::new();
    collect_candidates_from_value(
        &mut candidates,
        &request.target,
        &request.requested_role,
        request.app_hint.as_deref(),
        request.provider_hint.as_deref(),
        TargetGroundingSource::WorkflowMetadata,
    );
    collect_candidates_from_value(
        &mut candidates,
        &request.selection,
        &request.requested_role,
        request.app_hint.as_deref(),
        request.provider_hint.as_deref(),
        TargetGroundingSource::WorkflowMetadata,
    );

    candidates.extend(
        request
            .screen_candidates
            .iter()
            .filter(|candidate| candidate_context_matches(candidate, request))
            .cloned()
            .map(|candidate| candidate.with_source(TargetGroundingSource::ScreenAnalysis)),
    );

    if request.allow_recent_reuse {
        candidates.extend(
            request
                .recent_candidates
                .iter()
                .filter(|candidate| candidate_context_matches(candidate, request))
                .filter(|candidate| recent_candidate_is_reusable(candidate, request))
                .cloned()
                .map(|candidate| candidate.with_source(TargetGroundingSource::RecentContext)),
        );
    }

    if let Some(rank_hint) = request.rank_hint {
        for candidate in &mut candidates {
            if candidate.rank.is_none() {
                candidate.rank = Some(rank_hint);
            }
        }
    }

    let mut uncertainty = Vec::new();
    if candidates.is_empty() {
        uncertainty.push("no_structured_target_candidates".into());
    }

    TargetGroundingState {
        requested_role: request.requested_role.clone(),
        sufficient_for_selection: !candidates.is_empty(),
        ambiguous: false,
        candidates,
        current_app_hint: request.app_hint.clone(),
        provider_hint: request.provider_hint.clone(),
        uncertainty,
    }
}

pub fn select_target_candidate(
    state: &TargetGroundingState,
    action: TargetAction,
    policy: &TargetSelectionPolicy,
) -> TargetSelection {
    let required_confidence = match action {
        TargetAction::Focus => policy.min_focus_confidence,
        TargetAction::Click => policy.min_click_confidence,
    };

    if state.candidates.is_empty() {
        return TargetSelection::rejected(
            TargetSelectionStatus::NoCandidates,
            Vec::new(),
            required_confidence,
            "No structured target candidates are available for this step.",
        );
    }

    let mut considered = state
        .candidates
        .iter()
        .filter(|candidate| candidate.role.matches_requested(&state.requested_role))
        .filter(|candidate| match action {
            TargetAction::Focus => candidate.supports_focus,
            TargetAction::Click => candidate.supports_click,
        })
        .filter(|candidate| candidate.has_point())
        .cloned()
        .collect::<Vec<_>>();

    if considered.is_empty() {
        return TargetSelection::rejected(
            TargetSelectionStatus::UnsupportedTarget,
            state.candidates.clone(),
            required_confidence,
            "Candidates exist, but none match the requested role with executable target metadata.",
        );
    }

    considered.sort_by(|left, right| {
        candidate_score(right, state)
            .partial_cmp(&candidate_score(left, state))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.rank
                    .unwrap_or(u32::MAX)
                    .cmp(&right.rank.unwrap_or(u32::MAX))
            })
    });

    let best = considered[0].clone();
    let best_score = candidate_score(&best, state);
    if best_score < required_confidence {
        return TargetSelection::rejected(
            TargetSelectionStatus::LowConfidence,
            considered,
            required_confidence,
            format!(
                "Best target score {:.2} is below the required {:.2}.",
                best_score, required_confidence
            ),
        );
    }

    if let Some(second) = considered.get(1) {
        if (best_score - candidate_score(second, state)).abs() < policy.ambiguity_margin {
            return TargetSelection::rejected(
                TargetSelectionStatus::Ambiguous,
                considered,
                required_confidence,
                "Multiple target candidates are too close in confidence to click safely.",
            );
        }
    }

    TargetSelection::selected(best, considered, required_confidence)
}

fn collect_candidates_from_value(
    candidates: &mut Vec<UITargetCandidate>,
    value: &Value,
    requested_role: &UITargetRole,
    app_hint: Option<&str>,
    provider_hint: Option<&str>,
    source: TargetGroundingSource,
) {
    if let Some(values) = value.as_array() {
        for candidate in values {
            collect_candidates_from_value(
                candidates,
                candidate,
                requested_role,
                app_hint,
                provider_hint,
                source.clone(),
            );
        }
        return;
    }

    if !value.is_object() {
        return;
    }

    for key in ["candidate", "target_candidate", "ui_candidate"] {
        if let Some(candidate) = value.get(key) {
            collect_candidates_from_value(
                candidates,
                candidate,
                requested_role,
                app_hint,
                provider_hint,
                source.clone(),
            );
        }
    }

    for key in ["candidates", "target_candidates", "ui_candidates"] {
        if let Some(values) = value.get(key).and_then(Value::as_array) {
            for candidate in values {
                collect_candidates_from_value(
                    candidates,
                    candidate,
                    requested_role,
                    app_hint,
                    provider_hint,
                    source.clone(),
                );
            }
        }
    }

    if let Some(candidate) = candidate_from_value(
        value,
        candidates.len(),
        requested_role,
        app_hint,
        provider_hint,
        source,
    ) {
        candidates.push(candidate);
    }
}

fn candidate_from_value(
    value: &Value,
    index: usize,
    requested_role: &UITargetRole,
    app_hint: Option<&str>,
    provider_hint: Option<&str>,
    source: TargetGroundingSource,
) -> Option<UITargetCandidate> {
    let region = parse_region(value.get("region").unwrap_or(value));
    let center_x = value
        .get("center_x")
        .or_else(|| value.pointer("/center/x"))
        .and_then(Value::as_f64);
    let center_y = value
        .get("center_y")
        .or_else(|| value.pointer("/center/y"))
        .and_then(Value::as_f64);
    let has_point = center_x.zip(center_y).is_some() || region.is_some();
    let confidence = value
        .get("confidence")
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as f32;

    if !has_point && confidence <= 0.0 {
        return None;
    }

    let role = UITargetRole::from_value(
        value
            .get("role")
            .or_else(|| value.get("target_role"))
            .or_else(|| value.get("element_role"))
            .and_then(Value::as_str),
    );
    let role = if role == UITargetRole::Unknown {
        requested_role.clone()
    } else {
        role
    };
    let supports_focus = value
        .get("supports_focus")
        .and_then(Value::as_bool)
        .unwrap_or(
            matches!(role, UITargetRole::SearchInput | UITargetRole::TextInput) && has_point,
        );
    let supports_click = value
        .get("supports_click")
        .and_then(Value::as_bool)
        .unwrap_or(has_point);
    let app_hint = value
        .get("app_hint")
        .or_else(|| value.get("app"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| app_hint.map(ToOwned::to_owned));
    let provider_hint = value
        .get("provider_hint")
        .or_else(|| value.get("provider"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| provider_hint.map(ToOwned::to_owned));

    Some(UITargetCandidate {
        candidate_id: value
            .get("candidate_id")
            .or_else(|| value.get("id"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("target_candidate_{}", index + 1)),
        role,
        region,
        center_x,
        center_y,
        app_hint,
        provider_hint,
        confidence,
        source,
        label: value
            .get("label")
            .or_else(|| value.get("text"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        rank: value
            .get("rank")
            .or_else(|| value.get("order"))
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        observed_at_ms: value
            .get("observed_at_ms")
            .or_else(|| value.get("captured_at_ms"))
            .or_else(|| value.get("created_at_ms"))
            .and_then(Value::as_u64),
        reuse_eligible: value
            .get("reuse_eligible")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        supports_focus,
        supports_click,
        rationale: value
            .get("rationale")
            .and_then(Value::as_str)
            .unwrap_or("Structured target candidate supplied by workflow metadata.")
            .to_string(),
    })
}

fn candidate_context_matches(
    candidate: &UITargetCandidate,
    request: &TargetGroundingRequest,
) -> bool {
    if let (Some(request_provider), Some(candidate_provider)) = (
        request.provider_hint.as_deref(),
        candidate.provider_hint.as_deref(),
    ) {
        if !labels_match(request_provider, candidate_provider) {
            return false;
        }
    }

    if let (Some(request_app), Some(candidate_app)) =
        (request.app_hint.as_deref(), candidate.app_hint.as_deref())
    {
        if !labels_match(request_app, candidate_app) {
            return false;
        }
    }

    true
}

fn recent_candidate_is_reusable(
    candidate: &UITargetCandidate,
    request: &TargetGroundingRequest,
) -> bool {
    if !candidate.reuse_eligible {
        return false;
    }

    let Some(now_ms) = request.now_ms else {
        return true;
    };
    let Some(observed_at_ms) = candidate.observed_at_ms else {
        return false;
    };

    now_ms.saturating_sub(observed_at_ms) <= request.max_recent_age_ms
}

fn candidate_score(candidate: &UITargetCandidate, state: &TargetGroundingState) -> f32 {
    let mut score = candidate.confidence;
    if let (Some(provider), Some(candidate_provider)) = (
        state.provider_hint.as_deref(),
        candidate.provider_hint.as_deref(),
    ) {
        if labels_match(provider, candidate_provider) {
            score += 0.03;
        }
    }
    if let (Some(app), Some(candidate_app)) = (
        state.current_app_hint.as_deref(),
        candidate.app_hint.as_deref(),
    ) {
        if labels_match(app, candidate_app) {
            score += 0.02;
        }
    }
    if candidate.rank == Some(1) {
        score += 0.02;
    }
    if candidate.source == TargetGroundingSource::WorkflowMetadata {
        score += 0.01;
    }
    if candidate.source == TargetGroundingSource::RecentContext {
        score -= 0.02;
    }

    score.clamp(0.0, 1.0)
}

fn labels_match(left: &str, right: &str) -> bool {
    normalize_label(left) == normalize_label(right)
}

fn parse_region(value: &Value) -> Option<TargetRegion> {
    let x = value
        .get("x")
        .or_else(|| value.get("left"))
        .and_then(Value::as_f64)?;
    let y = value
        .get("y")
        .or_else(|| value.get("top"))
        .and_then(Value::as_f64)?;
    let width = value.get("width").and_then(Value::as_f64).or_else(|| {
        let right = value.get("right")?.as_f64()?;
        Some(right - x)
    })?;
    let height = value.get("height").and_then(Value::as_f64).or_else(|| {
        let bottom = value.get("bottom")?.as_f64()?;
        Some(bottom - y)
    })?;

    if width <= 0.0 || height <= 0.0 {
        return None;
    }

    Some(TargetRegion {
        x,
        y,
        width,
        height,
        coordinate_space: value
            .get("coordinate_space")
            .and_then(Value::as_str)
            .unwrap_or("screen")
            .to_string(),
    })
}

fn default_coordinate_space() -> String {
    "screen".into()
}

fn default_reuse_eligible() -> bool {
    true
}

fn default_recent_target_max_age_ms() -> u64 {
    120_000
}

fn normalize_label(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_search_input_candidate_from_structured_metadata() {
        let state = ground_targets_for_request(&TargetGroundingRequest {
            requested_role: UITargetRole::SearchInput,
            target: json!({
                "candidate": {
                    "role": "search_input",
                    "region": {"x": 10, "y": 20, "width": 200, "height": 40},
                    "confidence": 0.91,
                    "provider": "youtube"
                }
            }),
            selection: json!({}),
            screen_candidates: Vec::new(),
            recent_candidates: Vec::new(),
            app_hint: Some("chrome".into()),
            provider_hint: Some("youtube".into()),
            rank_hint: None,
            allow_recent_reuse: false,
            now_ms: None,
            max_recent_age_ms: default_recent_target_max_age_ms(),
        });

        assert_eq!(state.candidates.len(), 1);
        assert!(state.candidates[0].supports_focus);
        assert_eq!(state.candidates[0].center_point(), Some((110.0, 40.0)));
    }

    #[test]
    fn rejects_low_confidence_target() {
        let state = ground_targets_for_request(&TargetGroundingRequest {
            requested_role: UITargetRole::RankedResult,
            target: json!({}),
            selection: json!({
                "candidate": {
                    "role": "ranked_result",
                    "center_x": 400,
                    "center_y": 260,
                    "confidence": 0.55,
                    "rank": 1
                }
            }),
            screen_candidates: Vec::new(),
            recent_candidates: Vec::new(),
            app_hint: None,
            provider_hint: Some("youtube".into()),
            rank_hint: Some(1),
            allow_recent_reuse: false,
            now_ms: None,
            max_recent_age_ms: default_recent_target_max_age_ms(),
        });
        let selection = select_target_candidate(
            &state,
            TargetAction::Click,
            &TargetSelectionPolicy::default(),
        );

        assert_eq!(selection.status, TargetSelectionStatus::LowConfidence);
        assert!(selection.selected_candidate.is_none());
    }

    #[test]
    fn selects_high_confidence_target() {
        let state = ground_targets_for_request(&TargetGroundingRequest {
            requested_role: UITargetRole::RankedResult,
            target: json!({}),
            selection: json!({
                "candidates": [
                    {
                        "role": "ranked_result",
                        "center_x": 400,
                        "center_y": 260,
                        "confidence": 0.93,
                        "rank": 1
                    },
                    {
                        "role": "ranked_result",
                        "center_x": 400,
                        "center_y": 360,
                        "confidence": 0.70,
                        "rank": 2
                    }
                ]
            }),
            screen_candidates: Vec::new(),
            recent_candidates: Vec::new(),
            app_hint: None,
            provider_hint: Some("youtube".into()),
            rank_hint: Some(1),
            allow_recent_reuse: false,
            now_ms: None,
            max_recent_age_ms: default_recent_target_max_age_ms(),
        });
        let selection = select_target_candidate(
            &state,
            TargetAction::Click,
            &TargetSelectionPolicy::default(),
        );

        assert_eq!(selection.status, TargetSelectionStatus::Selected);
        assert_eq!(selection.selected_candidate.unwrap().rank, Some(1));
    }

    #[test]
    fn rejects_ambiguous_targets() {
        let state = ground_targets_for_request(&TargetGroundingRequest {
            requested_role: UITargetRole::RankedResult,
            target: json!({}),
            selection: json!({
                "candidates": [
                    {
                        "role": "ranked_result",
                        "center_x": 400,
                        "center_y": 260,
                        "confidence": 0.91
                    },
                    {
                        "role": "ranked_result",
                        "center_x": 420,
                        "center_y": 310,
                        "confidence": 0.88
                    }
                ]
            }),
            screen_candidates: Vec::new(),
            recent_candidates: Vec::new(),
            app_hint: None,
            provider_hint: None,
            rank_hint: None,
            allow_recent_reuse: false,
            now_ms: None,
            max_recent_age_ms: default_recent_target_max_age_ms(),
        });
        let selection = select_target_candidate(
            &state,
            TargetAction::Click,
            &TargetSelectionPolicy::default(),
        );

        assert_eq!(selection.status, TargetSelectionStatus::Ambiguous);
    }

    #[test]
    fn extracts_candidates_from_screen_derived_inputs() {
        let state = ground_targets_for_request(&TargetGroundingRequest {
            requested_role: UITargetRole::SearchInput,
            target: json!({}),
            selection: json!({}),
            screen_candidates: vec![UITargetCandidate {
                candidate_id: "screen_search".into(),
                role: UITargetRole::SearchInput,
                region: Some(TargetRegion {
                    x: 100.0,
                    y: 50.0,
                    width: 500.0,
                    height: 40.0,
                    coordinate_space: "screen".into(),
                }),
                center_x: None,
                center_y: None,
                app_hint: Some("chrome".into()),
                provider_hint: Some("youtube".into()),
                confidence: 0.90,
                source: TargetGroundingSource::ScreenAnalysis,
                label: Some("Search".into()),
                rank: None,
                observed_at_ms: Some(1_000),
                reuse_eligible: true,
                supports_focus: true,
                supports_click: true,
                rationale: "structured test candidate".into(),
            }],
            recent_candidates: Vec::new(),
            app_hint: Some("chrome".into()),
            provider_hint: Some("youtube".into()),
            rank_hint: None,
            allow_recent_reuse: false,
            now_ms: Some(1_100),
            max_recent_age_ms: default_recent_target_max_age_ms(),
        });
        let selection = select_target_candidate(
            &state,
            TargetAction::Focus,
            &TargetSelectionPolicy::default(),
        );

        assert_eq!(selection.status, TargetSelectionStatus::Selected);
        assert_eq!(
            selection.selected_candidate.unwrap().source,
            TargetGroundingSource::ScreenAnalysis
        );
    }

    #[test]
    fn reuses_fresh_recent_candidates_when_allowed() {
        let state = ground_targets_for_request(&TargetGroundingRequest {
            requested_role: UITargetRole::SearchInput,
            target: json!({}),
            selection: json!({}),
            screen_candidates: Vec::new(),
            recent_candidates: vec![UITargetCandidate {
                candidate_id: "recent_search".into(),
                role: UITargetRole::SearchInput,
                region: None,
                center_x: Some(250.0),
                center_y: Some(90.0),
                app_hint: Some("chrome".into()),
                provider_hint: Some("youtube".into()),
                confidence: 0.86,
                source: TargetGroundingSource::RecentContext,
                label: Some("YouTube search".into()),
                rank: None,
                observed_at_ms: Some(2_000),
                reuse_eligible: true,
                supports_focus: true,
                supports_click: true,
                rationale: "recent successful focus target".into(),
            }],
            app_hint: Some("chrome".into()),
            provider_hint: Some("youtube".into()),
            rank_hint: None,
            allow_recent_reuse: true,
            now_ms: Some(2_500),
            max_recent_age_ms: default_recent_target_max_age_ms(),
        });
        let selection = select_target_candidate(
            &state,
            TargetAction::Focus,
            &TargetSelectionPolicy::default(),
        );

        assert_eq!(selection.status, TargetSelectionStatus::Selected);
        assert_eq!(
            selection.selected_candidate.unwrap().source,
            TargetGroundingSource::RecentContext
        );
    }

    #[test]
    fn stale_recent_candidates_are_not_reused() {
        let state = ground_targets_for_request(&TargetGroundingRequest {
            requested_role: UITargetRole::SearchInput,
            target: json!({}),
            selection: json!({}),
            screen_candidates: Vec::new(),
            recent_candidates: vec![UITargetCandidate {
                candidate_id: "stale_search".into(),
                role: UITargetRole::SearchInput,
                region: None,
                center_x: Some(250.0),
                center_y: Some(90.0),
                app_hint: Some("chrome".into()),
                provider_hint: Some("youtube".into()),
                confidence: 0.92,
                source: TargetGroundingSource::RecentContext,
                label: None,
                rank: None,
                observed_at_ms: Some(1_000),
                reuse_eligible: true,
                supports_focus: true,
                supports_click: true,
                rationale: "stale recent target".into(),
            }],
            app_hint: Some("chrome".into()),
            provider_hint: Some("youtube".into()),
            rank_hint: None,
            allow_recent_reuse: true,
            now_ms: Some(200_000),
            max_recent_age_ms: 1_000,
        });

        assert!(state.candidates.is_empty());
    }
}
