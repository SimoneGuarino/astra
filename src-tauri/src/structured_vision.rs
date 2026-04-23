use crate::{
    desktop_agent_types::{PageEvidenceSource, PageSemanticEvidence},
    semantic_frame::{semantic_frame_from_candidates, semantic_frame_from_vision_value},
    ui_target_grounding::{
        is_technical_capture_backend, normalize_browser_app_hint, normalize_content_provider_hint,
        structured_candidates_from_value, TargetGroundingSource, UITargetCandidate, UITargetRole,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const MIN_STRUCTURED_VISION_CONFIDENCE: f32 = 0.35;
const MAX_REASONABLE_SCREEN_COORDINATE: f64 = 100_000.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredVisionExtraction {
    pub candidates: Vec<UITargetCandidate>,
    #[serde(default)]
    pub page_evidence: Option<PageSemanticEvidence>,
    #[serde(default)]
    pub semantic_frame: Option<crate::desktop_agent_types::SemanticScreenFrame>,
    pub rejected: Vec<StructuredVisionRejectedCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredVisionRejectedCandidate {
    pub reason: String,
    #[serde(default)]
    pub candidate: Value,
}

pub fn parse_structured_vision_candidates(
    content: &str,
    captured_at_ms: u64,
) -> Result<StructuredVisionExtraction, String> {
    let json_text = extract_json_object(content)
        .ok_or_else(|| "structured vision response did not contain a JSON object".to_string())?;
    let parsed: Value = serde_json::from_str(json_text)
        .map_err(|error| format!("structured vision JSON parse failed: {error}"))?;
    parse_structured_vision_candidates_from_value(&parsed, captured_at_ms)
}

pub fn parse_structured_vision_candidates_from_value(
    parsed: &Value,
    captured_at_ms: u64,
) -> Result<StructuredVisionExtraction, String> {
    let candidates_value = parsed
        .get("ui_candidates")
        .or_else(|| parsed.get("candidates"))
        .ok_or_else(|| "structured vision JSON missing ui_candidates".to_string())?;
    let page_evidence = parse_page_semantic_evidence(parsed);

    let raw_candidates = candidates_value.as_array().cloned().unwrap_or_else(|| {
        if candidates_value.is_object() {
            vec![candidates_value.clone()]
        } else {
            Vec::new()
        }
    });
    let parsed_candidates = structured_candidates_from_value(
        candidates_value,
        &UITargetRole::Unknown,
        None,
        None,
        TargetGroundingSource::ScreenAnalysis,
    );

    let mut candidates = Vec::new();
    let mut rejected = Vec::new();
    for candidate in parsed_candidates {
        match validate_structured_vision_candidate(candidate, captured_at_ms) {
            Ok(candidate) => candidates.push(candidate),
            Err((reason, candidate)) => rejected.push(StructuredVisionRejectedCandidate {
                reason,
                candidate: candidate.execution_payload(),
            }),
        }
    }

    if parsed_candidates_count(&raw_candidates) > candidates.len() + rejected.len() {
        rejected.push(StructuredVisionRejectedCandidate {
            reason: "one or more raw candidates were malformed before typed parsing".into(),
            candidate: Value::Array(raw_candidates),
        });
    }

    let semantic_frame = semantic_frame_from_vision_value(
        &parsed,
        captured_at_ms,
        None,
        page_evidence.clone(),
        candidates.clone(),
    )
    .or_else(|| {
        Some(semantic_frame_from_candidates(
            captured_at_ms,
            None,
            page_evidence.clone(),
            candidates.clone(),
        ))
    });

    Ok(StructuredVisionExtraction {
        candidates,
        page_evidence,
        semantic_frame,
        rejected,
    })
}

fn parse_page_semantic_evidence(parsed: &Value) -> Option<PageSemanticEvidence> {
    let value = parsed
        .get("page_evidence")
        .or_else(|| parsed.get("page_semantics"))
        .or_else(|| parsed.get("page_context"))?;
    let object = value.as_object()?;
    let raw_provider = object
        .get("content_provider_hint")
        .or_else(|| object.get("site_provider_hint"))
        .or_else(|| object.get("site_provider"))
        .or_else(|| object.get("site_hint"))
        .or_else(|| object.get("provider_hint"))
        .or_else(|| object.get("provider"))
        .and_then(Value::as_str);
    let raw_browser = object
        .get("browser_app_hint")
        .or_else(|| object.get("browser_app"))
        .or_else(|| object.get("browser"))
        .or_else(|| object.get("app_hint"))
        .or_else(|| object.get("app"))
        .and_then(Value::as_str);
    let capture_backend = object
        .get("capture_backend")
        .or_else(|| object.get("observation_backend"))
        .or_else(|| object.get("capture_provider"))
        .and_then(Value::as_str)
        .map(normalize_simple_label)
        .or_else(|| {
            raw_provider
                .filter(|value| is_technical_capture_backend(value))
                .map(normalize_simple_label)
        });
    let mut uncertainty = string_array(object.get("uncertainty"));
    if raw_provider.is_some_and(is_technical_capture_backend) {
        uncertainty.push("technical_backend_ignored_as_content_provider".into());
    }

    Some(PageSemanticEvidence {
        browser_app_hint: raw_browser.and_then(normalize_browser_app_hint),
        content_provider_hint: raw_provider.and_then(normalize_content_provider_hint),
        page_kind_hint: object
            .get("page_kind_hint")
            .or_else(|| object.get("page_kind"))
            .or_else(|| object.get("page_type"))
            .and_then(Value::as_str)
            .map(normalize_simple_label),
        query_hint: object
            .get("query_hint")
            .or_else(|| object.get("query"))
            .or_else(|| object.get("search_query"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned),
        result_list_visible: object
            .get("result_list_visible")
            .or_else(|| object.get("results_visible"))
            .and_then(Value::as_bool),
        raw_confidence: object
            .get("confidence")
            .and_then(Value::as_f64)
            .map(|value| value.clamp(0.0, 1.0) as f32),
        confidence: object
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0) as f32,
        evidence_sources: vec![PageEvidenceSource::StructuredVision],
        capture_backend,
        observation_source: object
            .get("observation_source")
            .or_else(|| object.get("source"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        uncertainty,
    })
}

fn string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn normalize_simple_label(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "_")
}

fn validate_structured_vision_candidate(
    mut candidate: UITargetCandidate,
    captured_at_ms: u64,
) -> Result<UITargetCandidate, (String, UITargetCandidate)> {
    if candidate.role == UITargetRole::Unknown {
        return Err(("candidate role is unsupported or unknown".into(), candidate));
    }
    if candidate.confidence < MIN_STRUCTURED_VISION_CONFIDENCE {
        return Err((
            format!(
                "candidate confidence {:.2} is below structured vision retention threshold {:.2}",
                candidate.confidence, MIN_STRUCTURED_VISION_CONFIDENCE
            ),
            candidate,
        ));
    }
    if !candidate.has_point() {
        return Err((
            "candidate has no explicit point or region".into(),
            candidate,
        ));
    }
    if let Some(region) = candidate.region.as_ref() {
        if region.coordinate_space != "screen" {
            return Err((
                "candidate coordinate space is not executable screen coordinates".into(),
                candidate,
            ));
        }
        for value in [region.x, region.y, region.width, region.height] {
            if !value.is_finite() || value.abs() > MAX_REASONABLE_SCREEN_COORDINATE {
                return Err((
                    "candidate region contains invalid coordinates".into(),
                    candidate,
                ));
            }
        }
    }
    if let Some((x, y)) = candidate.center_point() {
        if !x.is_finite()
            || !y.is_finite()
            || x.abs() > MAX_REASONABLE_SCREEN_COORDINATE
            || y.abs() > MAX_REASONABLE_SCREEN_COORDINATE
        {
            return Err((
                "candidate center contains invalid coordinates".into(),
                candidate,
            ));
        }
    }

    candidate.source = TargetGroundingSource::ScreenAnalysis;
    candidate.observed_at_ms = Some(captured_at_ms);
    candidate.reuse_eligible = true;
    if candidate.rationale.trim().is_empty() {
        candidate.rationale = "Structured vision candidate validated by Rust.".into();
    }

    Ok(candidate)
}

fn parsed_candidates_count(raw_candidates: &[Value]) -> usize {
    raw_candidates
        .iter()
        .filter(|value| value.is_object())
        .count()
}

fn extract_json_object(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Some(trimmed);
    }

    if let Some(start) = trimmed.find("```json") {
        let after_fence = &trimmed[start + "```json".len()..];
        if let Some(end) = after_fence.find("```") {
            let body = after_fence[..end].trim();
            if body.starts_with('{') && body.ends_with('}') {
                return Some(body);
            }
        }
    }

    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    (end > start).then_some(trimmed[start..=end].trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_structured_vision_candidate() {
        let extraction = parse_structured_vision_candidates(
            r#"{
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "search_results",
                    "query_hint": "shiva",
                    "result_list_visible": true,
                    "confidence": 0.9
                },
                "ui_candidates": [
                    {
                        "candidate_id": "search",
                        "role": "search_input",
                        "region": {"x": 120, "y": 64, "width": 500, "height": 40, "coordinate_space": "screen"},
                        "confidence": 0.88,
                        "provider_hint": "youtube",
                        "supports_focus": true
                    }
                ]
            }"#,
            1_000,
        )
        .expect("parse");

        assert_eq!(extraction.candidates.len(), 1);
        assert_eq!(extraction.candidates[0].observed_at_ms, Some(1_000));
        assert_eq!(extraction.candidates[0].role, UITargetRole::SearchInput);
        assert_eq!(
            extraction
                .page_evidence
                .as_ref()
                .and_then(|evidence| evidence.content_provider_hint.as_deref()),
            Some("youtube")
        );
    }

    #[test]
    fn separates_capture_backend_from_content_provider() {
        let extraction = parse_structured_vision_candidates(
            r#"{
                "page_evidence": {
                    "provider_hint": "powershell_gdi",
                    "capture_backend": "powershell_gdi",
                    "page_kind_hint": "search_results",
                    "confidence": 0.7
                },
                "ui_candidates": [
                    {
                        "candidate_id": "result",
                        "role": "ranked_result",
                        "region": {"x": 120, "y": 164, "width": 500, "height": 80, "coordinate_space": "screen"},
                        "confidence": 0.88,
                        "provider_hint": "powershell_gdi",
                        "supports_click": true
                    }
                ]
            }"#,
            1_000,
        )
        .expect("parse");

        let page = extraction.page_evidence.expect("page evidence");
        assert_eq!(page.capture_backend.as_deref(), Some("powershell_gdi"));
        assert_eq!(page.content_provider_hint, None);
        assert_eq!(extraction.candidates[0].provider_hint, None);
        assert_eq!(
            extraction.candidates[0].capture_backend.as_deref(),
            Some("powershell_gdi")
        );
    }

    #[test]
    fn rejects_low_confidence_candidate() {
        let extraction = parse_structured_vision_candidates(
            r#"{
                "ui_candidates": [
                    {
                        "candidate_id": "weak",
                        "role": "button",
                        "center_x": 20,
                        "center_y": 20,
                        "confidence": 0.2
                    }
                ]
            }"#,
            1_000,
        )
        .expect("parse");

        assert!(extraction.candidates.is_empty());
        assert_eq!(extraction.rejected.len(), 1);
    }

    #[test]
    fn rejects_non_screen_coordinate_space() {
        let extraction = parse_structured_vision_candidates(
            r#"{
                "ui_candidates": [
                    {
                        "candidate_id": "image_space",
                        "role": "ranked_result",
                        "region": {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.1, "coordinate_space": "normalized"},
                        "confidence": 0.9
                    }
                ]
            }"#,
            1_000,
        )
        .expect("parse");

        assert!(extraction.candidates.is_empty());
        assert_eq!(extraction.rejected.len(), 1);
    }
}
