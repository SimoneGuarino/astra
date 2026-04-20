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
            "ranked_result" | "first_result" | "result" | "search_result" | "video"
            | "video_result" => Self::RankedResult,
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
    pub browser_app_hint: Option<String>,
    #[serde(default)]
    pub provider_hint: Option<String>,
    #[serde(default)]
    pub content_provider_hint: Option<String>,
    #[serde(default)]
    pub page_kind_hint: Option<String>,
    #[serde(default)]
    pub capture_backend: Option<String>,
    #[serde(default)]
    pub observation_source: Option<String>,
    #[serde(default)]
    pub result_kind: Option<String>,
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
            "browser_app_hint": self.browser_app_hint,
            "provider_hint": self.provider_hint,
            "content_provider_hint": self.content_provider_hint,
            "page_kind_hint": self.page_kind_hint,
            "capture_backend": self.capture_backend,
            "observation_source": self.observation_source,
            "result_kind": self.result_kind,
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
    pub result_kind_hint: Option<String>,
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
    #[serde(default)]
    pub rank_hint: Option<u32>,
    #[serde(default)]
    pub result_kind_hint: Option<String>,
    #[serde(default)]
    pub now_ms: Option<u64>,
    #[serde(default = "default_recent_target_max_age_ms")]
    pub max_recent_age_ms: u64,
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
    NoCandidatesPresent,
    CandidatesFilteredOut,
    RankMismatch,
    ProviderMismatch,
    KindMismatch,
    LowConfidence,
    Ambiguous,
    UnsupportedTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetCandidateFilterPhase {
    Presence,
    Role,
    Provider,
    App,
    ResultKind,
    ExecutableMetadata,
    Freshness,
    Rank,
    Confidence,
    Ambiguity,
    Selected,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TargetSelectionDiagnostics {
    pub raw_candidate_count: usize,
    pub role_match_count: usize,
    pub provider_match_count: usize,
    pub app_match_count: usize,
    pub result_kind_match_count: usize,
    pub executable_metadata_count: usize,
    pub freshness_match_count: usize,
    pub rank_match_count: usize,
    pub eligible_candidate_count: usize,
    #[serde(default)]
    pub requested_rank: Option<u32>,
    #[serde(default)]
    pub requested_provider: Option<String>,
    #[serde(default)]
    pub requested_app: Option<String>,
    #[serde(default)]
    pub requested_result_kind: Option<String>,
    #[serde(default)]
    pub rejected_phase: Option<TargetCandidateFilterPhase>,
    #[serde(default)]
    pub rejection_reasons: Vec<String>,
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
    #[serde(default)]
    pub diagnostics: TargetSelectionDiagnostics,
}

impl TargetSelection {
    pub fn selected(
        candidate: UITargetCandidate,
        considered: Vec<UITargetCandidate>,
        required_confidence: f32,
        mut diagnostics: TargetSelectionDiagnostics,
    ) -> Self {
        diagnostics.eligible_candidate_count = considered.len();
        diagnostics.rejected_phase = Some(TargetCandidateFilterPhase::Selected);
        Self {
            status: TargetSelectionStatus::Selected,
            selected_candidate: Some(candidate),
            considered_candidates: considered,
            required_confidence,
            reason: "A single high-confidence target candidate satisfied the selection policy."
                .into(),
            diagnostics,
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
            diagnostics: TargetSelectionDiagnostics::default(),
        }
    }
}

fn rejected_with_diagnostics(
    status: TargetSelectionStatus,
    considered: Vec<UITargetCandidate>,
    required_confidence: f32,
    reason: impl Into<String>,
    mut diagnostics: TargetSelectionDiagnostics,
    phase: TargetCandidateFilterPhase,
    rejection_reason: impl Into<String>,
) -> TargetSelection {
    diagnostics.rejected_phase = Some(phase);
    diagnostics.rejection_reasons.push(rejection_reason.into());
    TargetSelection {
        status,
        selected_candidate: None,
        considered_candidates: considered,
        required_confidence,
        reason: reason.into(),
        diagnostics,
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
            .cloned()
            .map(|candidate| candidate.with_source(TargetGroundingSource::ScreenAnalysis)),
    );

    if request.allow_recent_reuse {
        candidates.extend(
            request
                .recent_candidates
                .iter()
                .cloned()
                .map(|candidate| candidate.with_source(TargetGroundingSource::RecentContext)),
        );
    }

    if let Some(rank_hint) = request.rank_hint {
        for candidate in &mut candidates {
            if candidate.rank.is_none()
                && matches!(candidate.source, TargetGroundingSource::WorkflowMetadata)
            {
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
        rank_hint: request.rank_hint,
        result_kind_hint: request.result_kind_hint.clone(),
        now_ms: request.now_ms,
        max_recent_age_ms: request.max_recent_age_ms,
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
    let mut diagnostics = TargetSelectionDiagnostics {
        raw_candidate_count: state.candidates.len(),
        requested_rank: state.rank_hint,
        requested_provider: state.provider_hint.clone(),
        requested_app: state.current_app_hint.clone(),
        requested_result_kind: state.result_kind_hint.clone(),
        ..Default::default()
    };

    if state.candidates.is_empty() {
        return rejected_with_diagnostics(
            TargetSelectionStatus::NoCandidatesPresent,
            Vec::new(),
            required_confidence,
            "No structured target candidates are available for this step.",
            diagnostics,
            TargetCandidateFilterPhase::Presence,
            "candidate_pool_empty",
        );
    }

    let role_matched = state
        .candidates
        .iter()
        .filter(|candidate| candidate.role.matches_requested(&state.requested_role))
        .cloned()
        .collect::<Vec<_>>();
    diagnostics.role_match_count = role_matched.len();
    if role_matched.is_empty() {
        return rejected_with_diagnostics(
            TargetSelectionStatus::CandidatesFilteredOut,
            state.candidates.clone(),
            required_confidence,
            "Structured candidates are present, but none match the requested target role.",
            diagnostics,
            TargetCandidateFilterPhase::Role,
            "role_mismatch",
        );
    }

    let provider_matched = role_matched
        .iter()
        .filter(|candidate| candidate_provider_matches(candidate, state.provider_hint.as_deref()))
        .cloned()
        .collect::<Vec<_>>();
    diagnostics.provider_match_count = provider_matched.len();
    if provider_matched.is_empty() {
        return rejected_with_diagnostics(
            TargetSelectionStatus::ProviderMismatch,
            role_matched,
            required_confidence,
            "Structured candidates are present, but their provider hints do not match the requested provider.",
            diagnostics,
            TargetCandidateFilterPhase::Provider,
            "provider_mismatch",
        );
    }

    let app_matched = provider_matched
        .iter()
        .filter(|candidate| candidate_app_matches(candidate, state.current_app_hint.as_deref()))
        .cloned()
        .collect::<Vec<_>>();
    diagnostics.app_match_count = app_matched.len();
    if app_matched.is_empty() {
        return rejected_with_diagnostics(
            TargetSelectionStatus::CandidatesFilteredOut,
            provider_matched,
            required_confidence,
            "Structured candidates are present, but their app hints do not match the current app context.",
            diagnostics,
            TargetCandidateFilterPhase::App,
            "app_mismatch",
        );
    }

    let kind_matched = app_matched
        .iter()
        .filter(|candidate| candidate_kind_matches(candidate, state.result_kind_hint.as_deref()))
        .cloned()
        .collect::<Vec<_>>();
    diagnostics.result_kind_match_count = kind_matched.len();
    if kind_matched.is_empty() {
        return rejected_with_diagnostics(
            TargetSelectionStatus::KindMismatch,
            app_matched,
            required_confidence,
            "Structured candidates are present, but their result kind does not match the requested result kind.",
            diagnostics,
            TargetCandidateFilterPhase::ResultKind,
            "result_kind_mismatch",
        );
    }

    let executable_matched = kind_matched
        .iter()
        .filter(|candidate| candidate_is_executable(candidate, &action))
        .cloned()
        .collect::<Vec<_>>();
    diagnostics.executable_metadata_count = executable_matched.len();
    if executable_matched.is_empty() {
        return rejected_with_diagnostics(
            TargetSelectionStatus::UnsupportedTarget,
            kind_matched,
            required_confidence,
            "Structured candidates are present, but none expose executable target metadata for this action.",
            diagnostics,
            TargetCandidateFilterPhase::ExecutableMetadata,
            "missing_executable_target_metadata",
        );
    }

    let freshness_matched = executable_matched
        .iter()
        .filter(|candidate| candidate_is_fresh_for_selection(candidate, state))
        .cloned()
        .collect::<Vec<_>>();
    diagnostics.freshness_match_count = freshness_matched.len();
    if freshness_matched.is_empty() {
        return rejected_with_diagnostics(
            TargetSelectionStatus::CandidatesFilteredOut,
            executable_matched,
            required_confidence,
            "Structured candidates are present, but the reusable candidates are stale or not eligible for reuse.",
            diagnostics,
            TargetCandidateFilterPhase::Freshness,
            "stale_or_non_reusable_candidate",
        );
    }

    let mut considered = freshness_matched;
    if let Some(rank_hint) = state.rank_hint {
        let exact_ranked = considered
            .iter()
            .filter(|candidate| candidate.rank == Some(rank_hint))
            .cloned()
            .collect::<Vec<_>>();
        if !exact_ranked.is_empty() {
            considered = exact_ranked;
        } else if rank_hint == 1 {
            let unranked = considered
                .iter()
                .filter(|candidate| candidate.rank.is_none())
                .cloned()
                .collect::<Vec<_>>();
            if unranked.len() == 1 {
                considered = unranked;
            } else if !unranked.is_empty() {
                diagnostics.rank_match_count = 0;
                return rejected_with_diagnostics(
                    TargetSelectionStatus::Ambiguous,
                    considered,
                    required_confidence,
                    "Multiple result candidates are visible, but none has rank metadata for the requested first result.",
                    diagnostics,
                    TargetCandidateFilterPhase::Rank,
                    "multiple_unranked_candidates_for_first_result",
                );
            } else {
                diagnostics.rank_match_count = 0;
                return rejected_with_diagnostics(
                    TargetSelectionStatus::RankMismatch,
                    considered,
                    required_confidence,
                    format!("No target candidate matches requested result rank {rank_hint}."),
                    diagnostics,
                    TargetCandidateFilterPhase::Rank,
                    "rank_mismatch",
                );
            }
        } else {
            diagnostics.rank_match_count = 0;
            return rejected_with_diagnostics(
                TargetSelectionStatus::RankMismatch,
                considered,
                required_confidence,
                format!("No target candidate matches requested result rank {rank_hint}."),
                diagnostics,
                TargetCandidateFilterPhase::Rank,
                "rank_mismatch",
            );
        }
        diagnostics.rank_match_count = considered.len();
    } else {
        diagnostics.rank_match_count = considered.len();
    }
    diagnostics.eligible_candidate_count = considered.len();

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
        return rejected_with_diagnostics(
            TargetSelectionStatus::LowConfidence,
            considered,
            required_confidence,
            format!(
                "Best target score {:.2} is below the required {:.2}.",
                best_score, required_confidence
            ),
            diagnostics,
            TargetCandidateFilterPhase::Confidence,
            "below_confidence_threshold",
        );
    }

    if let Some(second) = considered.get(1) {
        if (best_score - candidate_score(second, state)).abs() < policy.ambiguity_margin {
            return rejected_with_diagnostics(
                TargetSelectionStatus::Ambiguous,
                considered,
                required_confidence,
                "Multiple target candidates are too close in confidence to click safely.",
                diagnostics,
                TargetCandidateFilterPhase::Ambiguity,
                "candidate_scores_too_close",
            );
        }
    }

    TargetSelection::selected(best, considered, required_confidence, diagnostics)
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
    let raw_app_hint = value
        .get("browser_app_hint")
        .or_else(|| value.get("browser_app"))
        .or_else(|| value.get("browser"))
        .or_else(|| value.get("app_hint"))
        .or_else(|| value.get("app"))
        .and_then(Value::as_str);
    let browser_app_hint = raw_app_hint
        .and_then(normalize_browser_app_hint)
        .or_else(|| app_hint.and_then(normalize_browser_app_hint));
    let app_hint = browser_app_hint
        .clone()
        .or_else(|| app_hint.and_then(normalize_browser_app_hint));
    let raw_provider_hint = value
        .get("content_provider_hint")
        .or_else(|| value.get("site_provider_hint"))
        .or_else(|| value.get("site_provider"))
        .or_else(|| value.get("site_hint"))
        .or_else(|| value.get("site"))
        .or_else(|| value.get("provider_hint"))
        .or_else(|| value.get("provider"))
        .and_then(Value::as_str);
    let content_provider_hint = raw_provider_hint
        .and_then(normalize_content_provider_hint)
        .or_else(|| provider_hint.and_then(normalize_content_provider_hint));
    let provider_hint = content_provider_hint.clone();
    let capture_backend = value
        .get("capture_backend")
        .or_else(|| value.get("observation_backend"))
        .or_else(|| value.get("capture_provider"))
        .and_then(Value::as_str)
        .map(normalize_label)
        .or_else(|| {
            raw_provider_hint
                .filter(|value| is_technical_capture_backend(value))
                .map(normalize_label)
        });
    let observation_source = value
        .get("observation_source")
        .or_else(|| value.get("source_backend"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let page_kind_hint = value
        .get("page_kind_hint")
        .or_else(|| value.get("page_kind"))
        .or_else(|| value.get("page_type"))
        .and_then(Value::as_str)
        .map(normalize_label);

    let app_hint = app_hint.or_else(|| {
        value
            .get("app_hint")
            .or_else(|| value.get("app"))
            .and_then(Value::as_str)
            .and_then(normalize_browser_app_hint)
    });
    let provider_hint = provider_hint.or_else(|| {
        value
            .get("provider_hint")
            .or_else(|| value.get("provider"))
            .and_then(Value::as_str)
            .and_then(normalize_content_provider_hint)
    });
    let content_provider_hint = content_provider_hint.or_else(|| provider_hint.clone());
    let browser_app_hint = browser_app_hint.or_else(|| app_hint.clone());
    let app_hint = app_hint.or_else(|| browser_app_hint.clone());

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
        browser_app_hint,
        provider_hint,
        content_provider_hint,
        page_kind_hint,
        capture_backend,
        observation_source,
        result_kind: value
            .get("result_kind")
            .or_else(|| value.get("kind"))
            .or_else(|| value.get("item_kind"))
            .or_else(|| value.get("content_kind"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
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

fn candidate_provider_matches(candidate: &UITargetCandidate, provider_hint: Option<&str>) -> bool {
    let Some(provider_hint) = provider_hint else {
        return true;
    };
    let Some(candidate_provider) = candidate_content_provider_hint(candidate) else {
        return true;
    };

    labels_match(provider_hint, &candidate_provider)
}

fn candidate_app_matches(candidate: &UITargetCandidate, app_hint: Option<&str>) -> bool {
    let Some(app_hint) = app_hint else {
        return true;
    };
    let Some(candidate_app) = candidate_browser_app_hint(candidate) else {
        return true;
    };

    labels_match(app_hint, &candidate_app)
}

fn candidate_kind_matches(candidate: &UITargetCandidate, result_kind_hint: Option<&str>) -> bool {
    let Some(result_kind_hint) = result_kind_hint else {
        return true;
    };
    if is_generic_result_kind(result_kind_hint) {
        return true;
    }
    let Some(candidate_kind) = candidate.result_kind.as_deref() else {
        return true;
    };
    if is_generic_result_kind(candidate_kind) {
        return true;
    }

    labels_match(result_kind_hint, candidate_kind)
}

fn candidate_is_executable(candidate: &UITargetCandidate, action: &TargetAction) -> bool {
    let supports_action = match action {
        TargetAction::Focus => candidate.supports_focus,
        TargetAction::Click => candidate.supports_click,
    };

    supports_action && candidate.has_point()
}

fn candidate_is_fresh_for_selection(
    candidate: &UITargetCandidate,
    state: &TargetGroundingState,
) -> bool {
    if candidate.source != TargetGroundingSource::RecentContext {
        return true;
    }
    if !candidate.reuse_eligible {
        return false;
    }

    let Some(now_ms) = state.now_ms else {
        return true;
    };
    let Some(observed_at_ms) = candidate.observed_at_ms else {
        return false;
    };

    now_ms.saturating_sub(observed_at_ms) <= state.max_recent_age_ms
}

fn is_generic_result_kind(value: &str) -> bool {
    matches!(
        normalize_label(value).as_str(),
        "result" | "search_result" | "ranked_result" | "item" | "generic" | "unknown"
    )
}

fn candidate_score(candidate: &UITargetCandidate, state: &TargetGroundingState) -> f32 {
    let mut score = candidate.confidence;
    if let (Some(provider), Some(candidate_provider)) = (
        state.provider_hint.as_deref(),
        candidate_content_provider_hint(candidate),
    ) {
        if labels_match(provider, &candidate_provider) {
            score += 0.03;
        }
    }
    if let (Some(app), Some(candidate_app)) = (
        state.current_app_hint.as_deref(),
        candidate_browser_app_hint(candidate),
    ) {
        if labels_match(app, &candidate_app) {
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

pub fn candidate_content_provider_hint(candidate: &UITargetCandidate) -> Option<String> {
    candidate
        .content_provider_hint
        .as_deref()
        .and_then(normalize_content_provider_hint)
        .or_else(|| {
            candidate
                .provider_hint
                .as_deref()
                .and_then(normalize_content_provider_hint)
        })
}

pub fn candidate_browser_app_hint(candidate: &UITargetCandidate) -> Option<String> {
    candidate
        .browser_app_hint
        .as_deref()
        .and_then(normalize_browser_app_hint)
        .or_else(|| {
            candidate
                .app_hint
                .as_deref()
                .and_then(normalize_browser_app_hint)
        })
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

pub fn normalize_browser_app_hint(value: &str) -> Option<String> {
    match normalize_label(value).as_str() {
        "chrome" | "google_chrome" => Some("chrome".into()),
        "firefox" | "mozilla_firefox" => Some("firefox".into()),
        "edge" | "microsoft_edge" => Some("edge".into()),
        "safari" | "apple_safari" => Some("safari".into()),
        "browser" | "web_browser" => Some("browser".into()),
        _ => None,
    }
}

pub fn normalize_content_provider_hint(value: &str) -> Option<String> {
    let normalized = normalize_label(value);
    if is_technical_capture_backend(&normalized) {
        return None;
    }
    match normalized.as_str() {
        "" | "unknown" | "none" | "null" | "n_a" | "na" => None,
        "youtube" | "you_tube" | "youtube_com" | "www_youtube_com" => Some("youtube".into()),
        "google" | "google_search" | "google_com" | "www_google_com" => Some("google".into()),
        "github" | "github_com" => Some("github".into()),
        "amazon" | "amazon_com" => Some("amazon".into()),
        _ => Some(normalized),
    }
}

pub fn is_technical_capture_backend(value: &str) -> bool {
    let normalized = normalize_label(value);
    matches!(
        normalized.as_str(),
        "powershell_gdi"
            | "gdi"
            | "windows_gdi"
            | "windows_capture"
            | "screenshot"
            | "screen_capture"
            | "capture_backend"
            | "observation_backend"
            | "vision_json"
            | "desktop_observation"
    ) || normalized.ends_with("_gdi")
        || normalized.ends_with("_capture")
        || normalized.contains("capture_backend")
        || normalized.contains("observation_backend")
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
            result_kind_hint: None,
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
            result_kind_hint: Some("video".into()),
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
            result_kind_hint: Some("video".into()),
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
    fn reports_rank_mismatch_when_ranked_candidates_are_present() {
        let state = ground_targets_for_request(&TargetGroundingRequest {
            requested_role: UITargetRole::RankedResult,
            target: json!({}),
            selection: json!({
                "candidate": {
                    "role": "ranked_result",
                    "center_x": 400,
                    "center_y": 360,
                    "confidence": 0.95,
                    "provider": "youtube",
                    "result_kind": "video",
                    "rank": 2
                }
            }),
            screen_candidates: Vec::new(),
            recent_candidates: Vec::new(),
            app_hint: None,
            provider_hint: Some("youtube".into()),
            rank_hint: Some(1),
            result_kind_hint: Some("video".into()),
            allow_recent_reuse: false,
            now_ms: None,
            max_recent_age_ms: default_recent_target_max_age_ms(),
        });
        let selection = select_target_candidate(
            &state,
            TargetAction::Click,
            &TargetSelectionPolicy::default(),
        );

        assert_eq!(selection.status, TargetSelectionStatus::RankMismatch);
        assert_eq!(selection.diagnostics.raw_candidate_count, 1);
        assert_eq!(selection.diagnostics.rank_match_count, 0);
        assert_eq!(
            selection.diagnostics.rejected_phase,
            Some(TargetCandidateFilterPhase::Rank)
        );
    }

    #[test]
    fn reports_provider_mismatch_without_collapsing_to_no_candidates() {
        let state = ground_targets_for_request(&TargetGroundingRequest {
            requested_role: UITargetRole::RankedResult,
            target: json!({}),
            selection: json!({
                "candidate": {
                    "role": "ranked_result",
                    "center_x": 400,
                    "center_y": 260,
                    "confidence": 0.95,
                    "provider": "vimeo",
                    "result_kind": "video",
                    "rank": 1
                }
            }),
            screen_candidates: Vec::new(),
            recent_candidates: Vec::new(),
            app_hint: None,
            provider_hint: Some("youtube".into()),
            rank_hint: Some(1),
            result_kind_hint: Some("video".into()),
            allow_recent_reuse: false,
            now_ms: None,
            max_recent_age_ms: default_recent_target_max_age_ms(),
        });
        let selection = select_target_candidate(
            &state,
            TargetAction::Click,
            &TargetSelectionPolicy::default(),
        );

        assert_eq!(selection.status, TargetSelectionStatus::ProviderMismatch);
        assert_eq!(selection.diagnostics.raw_candidate_count, 1);
        assert_eq!(selection.diagnostics.provider_match_count, 0);
        assert_eq!(
            selection.diagnostics.rejected_phase,
            Some(TargetCandidateFilterPhase::Provider)
        );
    }

    #[test]
    fn technical_capture_backend_provider_hint_does_not_create_provider_mismatch() {
        let state = ground_targets_for_request(&TargetGroundingRequest {
            requested_role: UITargetRole::RankedResult,
            target: json!({}),
            selection: json!({
                "candidate": {
                    "role": "ranked_result",
                    "center_x": 400,
                    "center_y": 260,
                    "confidence": 0.95,
                    "provider": "powershell_gdi",
                    "result_kind": "video",
                    "rank": 1
                }
            }),
            screen_candidates: Vec::new(),
            recent_candidates: Vec::new(),
            app_hint: None,
            provider_hint: Some("youtube".into()),
            rank_hint: Some(1),
            result_kind_hint: Some("video".into()),
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
        assert_eq!(selection.diagnostics.provider_match_count, 1);
        assert_eq!(
            selection
                .selected_candidate
                .as_ref()
                .and_then(|candidate| candidate.capture_backend.as_deref()),
            Some("powershell_gdi")
        );
    }

    #[test]
    fn reports_result_kind_mismatch_without_hiding_candidates() {
        let state = ground_targets_for_request(&TargetGroundingRequest {
            requested_role: UITargetRole::RankedResult,
            target: json!({}),
            selection: json!({
                "candidate": {
                    "role": "ranked_result",
                    "center_x": 400,
                    "center_y": 260,
                    "confidence": 0.95,
                    "provider": "youtube",
                    "result_kind": "playlist",
                    "rank": 1
                }
            }),
            screen_candidates: Vec::new(),
            recent_candidates: Vec::new(),
            app_hint: None,
            provider_hint: Some("youtube".into()),
            rank_hint: Some(1),
            result_kind_hint: Some("video".into()),
            allow_recent_reuse: false,
            now_ms: None,
            max_recent_age_ms: default_recent_target_max_age_ms(),
        });
        let selection = select_target_candidate(
            &state,
            TargetAction::Click,
            &TargetSelectionPolicy::default(),
        );

        assert_eq!(selection.status, TargetSelectionStatus::KindMismatch);
        assert_eq!(selection.diagnostics.raw_candidate_count, 1);
        assert_eq!(selection.diagnostics.result_kind_match_count, 0);
        assert_eq!(
            selection.diagnostics.rejected_phase,
            Some(TargetCandidateFilterPhase::ResultKind)
        );
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
            result_kind_hint: None,
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
                browser_app_hint: Some("chrome".into()),
                provider_hint: Some("youtube".into()),
                content_provider_hint: Some("youtube".into()),
                page_kind_hint: None,
                capture_backend: None,
                observation_source: None,
                result_kind: None,
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
            result_kind_hint: None,
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
                browser_app_hint: Some("chrome".into()),
                provider_hint: Some("youtube".into()),
                content_provider_hint: Some("youtube".into()),
                page_kind_hint: None,
                capture_backend: None,
                observation_source: None,
                result_kind: None,
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
            result_kind_hint: None,
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
                browser_app_hint: Some("chrome".into()),
                provider_hint: Some("youtube".into()),
                content_provider_hint: Some("youtube".into()),
                page_kind_hint: None,
                capture_backend: None,
                observation_source: None,
                result_kind: None,
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
            result_kind_hint: None,
            allow_recent_reuse: true,
            now_ms: Some(200_000),
            max_recent_age_ms: 1_000,
        });

        let selection = select_target_candidate(
            &state,
            TargetAction::Focus,
            &TargetSelectionPolicy::default(),
        );

        assert_eq!(
            selection.status,
            TargetSelectionStatus::CandidatesFilteredOut
        );
        assert_eq!(
            selection.diagnostics.rejected_phase,
            Some(TargetCandidateFilterPhase::Freshness)
        );
    }
}
