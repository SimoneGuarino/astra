use crate::{
    desktop_agent_types::{
        FocusedPerceptionRequest, PageEvidenceSource, PageSemanticEvidence, PerceptionRequestMode,
        ScreenAnalysisResult, SemanticScreenFrame, VisionAvailability,
    },
    semantic_frame::semantic_frame_from_vision_value,
    structured_vision::{
        parse_structured_vision_candidates_from_value, StructuredVisionExtraction,
    },
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use image::{imageops::FilterType, GenericImageView, ImageFormat};
use reqwest::Client;
use serde_json::{json, Value};
use std::{env, fs, io::Cursor};
use uuid::Uuid;

const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
const MAX_VISION_IMAGE_WIDTH: u32 = 1280;
const MAX_VISION_IMAGE_HEIGHT: u32 = 1280;
const SHARED_REGION_OUTPUT_INSTRUCTIONS: &str = concat!(
    "When a visible result title, thumbnail, avatar, button, row, card, or control is reasonably clear, include executable coordinates measured from the current analyzed image. ",
    "Use coordinate_space=\"screen\" and nested region objects with this placeholder-only illustrative shape: ",
    "{\"visible_result_items\":[{\"item_id\":\"result_1\",\"kind\":\"video\",\"title\":\"example title\",\"rank_overall\":1,\"rank_within_kind\":1,\"click_regions\":{\"title\":{\"region\":{\"x\":YOUR_ACTUAL_X_IN_PIXELS,\"y\":YOUR_ACTUAL_Y_IN_PIXELS,\"width\":YOUR_ACTUAL_WIDTH,\"height\":YOUR_ACTUAL_HEIGHT,\"coordinate_space\":\"screen\"},\"confidence\":YOUR_CONFIDENCE},\"thumbnail\":{\"region\":{\"x\":YOUR_ACTUAL_X_IN_PIXELS,\"y\":YOUR_ACTUAL_Y_IN_PIXELS,\"width\":YOUR_ACTUAL_WIDTH,\"height\":YOUR_ACTUAL_HEIGHT,\"coordinate_space\":\"screen\"},\"confidence\":YOUR_CONFIDENCE}}}],",
    "\"actionable_controls\":[{\"control_id\":\"play_button\",\"kind\":\"button\",\"label\":\"click to play video\",\"region\":{\"x\":YOUR_ACTUAL_X_IN_PIXELS,\"y\":YOUR_ACTUAL_Y_IN_PIXELS,\"width\":YOUR_ACTUAL_WIDTH,\"height\":YOUR_ACTUAL_HEIGHT,\"coordinate_space\":\"screen\"},\"confidence\":YOUR_CONFIDENCE}]}. ",
    "The uppercase coordinate/confidence tokens are fake placeholders only. Copying placeholder values is wrong. ",
    "All coordinates must be actual pixel positions measured from the current analyzed image resolution stated in the user prompt. ",
    "Also include optional provider-agnostic primary_list when the main user-interactable surface is a ranked list. primary_list means the main list, not a sidebar, ad rail, recommendation rail, or secondary navigation. ",
    "Use structural container_kind values such as result_list, never provider names. For each primary_list item include rank, title when visible, item_kind as video|article|product|site|channel|generic, confidence, and the best primary click region measured from the actual image. ",
    "Also include optional page_state with structural kind list|detail|player|form|mixed|unknown and dominant_content result_list|detail_view|video_player|article|product_detail|generic|unknown. page_state describes what the page is doing, not what site it belongs to. ",
    "Never output provider names like youtube, google, amazon, or github as page_state.kind, page_state.dominant_content, or primary_list.container_kind. ",
    "If uncertain, omit primary_list or page_state rather than hallucinating structure. ",
    "Placeholder-only primary_list shape: {\"primary_list\":{\"cluster_id\":\"SCHEMA_ONLY_DO_NOT_COPY\",\"container_kind\":\"result_list\",\"item_count\":YOUR_ACTUAL_ITEM_COUNT,\"items\":[{\"item_id\":\"SCHEMA_ONLY_DO_NOT_COPY\",\"rank\":YOUR_ACTUAL_RANK_INTEGER,\"title\":\"YOUR_ACTUAL_TITLE_FROM_SCREEN\",\"item_kind\":\"video|article|product|site|channel|generic\",\"is_sponsored\":false,\"confidence\":YOUR_ACTUAL_CONFIDENCE_0_TO_1,\"click_regions\":{\"primary\":{\"region\":{\"x\":YOUR_ACTUAL_X_IN_PIXELS,\"y\":YOUR_ACTUAL_Y_IN_PIXELS,\"width\":YOUR_ACTUAL_WIDTH,\"height\":YOUR_ACTUAL_HEIGHT,\"coordinate_space\":\"screen\"},\"confidence\":YOUR_ACTUAL_CONFIDENCE_0_TO_1}}}]},\"page_state\":{\"kind\":\"list|detail|player|form|mixed|unknown\",\"dominant_content\":\"result_list|detail_view|video_player|article|product_detail|generic|unknown\",\"list_visible\":true,\"detail_visible\":false}}. ",
    "Do not leave click_regions empty when a clear visible title link, thumbnail, avatar, or primary control boundary is present. ",
    "If the region is not reasonably clear, omit it and add uncertainty instead of guessing."
);

const SHARED_SEMANTIC_FRAME_OUTPUT_SUFFIX: &str = concat!(
    "Return JSON shaped as {\"semantic_frame\":{\"page_evidence\":{...},\"scene_summary\":\"...\",\"primary_list\":null,\"page_state\":null,\"visible_entities\":[],\"visible_result_items\":[],\"actionable_controls\":[],\"uncertainty\":[]}}."
);

const SHARED_LEGACY_CANDIDATE_EXAMPLE: &str = concat!(
    "For legacy ui_candidates, use the same placeholder-only screen coordinate contract, for example ",
    "{\"candidate_id\":\"result_1\",\"role\":\"ranked_result\",\"region\":{\"x\":YOUR_ACTUAL_X_IN_PIXELS,\"y\":YOUR_ACTUAL_Y_IN_PIXELS,\"width\":YOUR_ACTUAL_WIDTH,\"height\":YOUR_ACTUAL_HEIGHT,\"coordinate_space\":\"screen\"},\"confidence\":YOUR_CONFIDENCE,\"supports_focus\":false,\"supports_click\":true,\"label\":\"example title\",\"content_provider_hint\":\"youtube\",\"page_kind_hint\":\"search_results\",\"result_kind\":\"video\",\"rank\":1,\"rationale\":\"visible title link or thumbnail\"}. "
);

#[derive(Debug, Clone, Copy, PartialEq)]
struct SuspiciousPromptTemplateRegion {
    label: &'static str,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

const SUSPICIOUS_PROMPT_TEMPLATE_REGIONS: [SuspiciousPromptTemplateRegion; 2] = [
    SuspiciousPromptTemplateRegion {
        label: "legacy_result_title_template",
        x: 120.0,
        y: 240.0,
        width: 580.0,
        height: 36.0,
    },
    SuspiciousPromptTemplateRegion {
        label: "legacy_thumbnail_template",
        x: 90.0,
        y: 180.0,
        width: 320.0,
        height: 180.0,
    },
];

#[derive(Debug, Clone, Default, PartialEq)]
struct VisionGeometrySanitizationReport {
    matched_patterns: Vec<&'static str>,
    removed_click_regions: usize,
    removed_controls: usize,
    removed_candidates: usize,
    stripped_entity_regions: usize,
    stripped_attribute_regions: usize,
}

impl VisionGeometrySanitizationReport {
    fn record_pattern(&mut self, label: &'static str) {
        if !self
            .matched_patterns
            .iter()
            .any(|existing| existing == &label)
        {
            self.matched_patterns.push(label);
        }
    }

    fn total_matches(&self) -> usize {
        self.removed_click_regions
            + self.removed_controls
            + self.removed_candidates
            + self.stripped_entity_regions
            + self.stripped_attribute_regions
    }
}

#[derive(Debug, Clone, PartialEq)]
struct VisionResizePlan {
    original_width: u32,
    original_height: u32,
    analyzed_width: u32,
    analyzed_height: u32,
}

impl VisionResizePlan {
    fn new(original_width: u32, original_height: u32) -> Self {
        let width_scale = MAX_VISION_IMAGE_WIDTH as f64 / original_width as f64;
        let height_scale = MAX_VISION_IMAGE_HEIGHT as f64 / original_height as f64;
        let scale = width_scale.min(height_scale).min(1.0);
        let analyzed_width = ((original_width as f64) * scale).round().max(1.0) as u32;
        let analyzed_height = ((original_height as f64) * scale).round().max(1.0) as u32;
        Self {
            original_width,
            original_height,
            analyzed_width,
            analyzed_height,
        }
    }

    fn requires_resize(&self) -> bool {
        self.original_width != self.analyzed_width || self.original_height != self.analyzed_height
    }

    fn scale_x(&self) -> f64 {
        self.original_width as f64 / self.analyzed_width as f64
    }

    fn scale_y(&self) -> f64 {
        self.original_height as f64 / self.analyzed_height as f64
    }

    fn resolution_prompt(&self) -> String {
        format!(
            "Analyzed image resolution: {}x{} pixels. Original capture resolution: {}x{} pixels. Return actual pixel coordinates measured only from the analyzed image resolution. Do not reuse placeholder/example coordinates. Astra will reproject them back to the original capture after parsing.",
            self.analyzed_width,
            self.analyzed_height,
            self.original_width,
            self.original_height
        )
    }
}

#[derive(Debug, Clone)]
struct PreparedVisionImage {
    image_b64: String,
    resize_plan: VisionResizePlan,
}

fn semantic_perception_system_prompt() -> String {
    [
        "You are Astra Semantic Perception. Return only strict JSON, no markdown. ",
        "Produce a goal-aware semantic_frame for a Rust-governed desktop assistant. ",
        "Do not choose actions or command the desktop. Describe only visible or strongly inferable screen state. ",
        "Separate browser_app_hint, content_provider_hint, page_kind_hint, query_hint, and capture_backend. ",
        "The observation backend is technical metadata, never the website/provider. ",
        "For result pages, distinguish result kinds such as video, mix, playlist, channel, hotel, product, repository, and generic. ",
        "For each visible_result_item include rank_overall, rank_within_kind, confidence, and click_regions for executable visible regions when clear. ",
        SHARED_REGION_OUTPUT_INSTRUCTIONS,
    ]
    .concat()
}

fn semantic_perception_user_prompt(
    goal: &str,
    app_hint: &str,
    provider_hint: &str,
    provider: &str,
    resize_plan: &VisionResizePlan,
) -> String {
    format!(
        "Goal: {goal}. Browser/app hint: {app_hint}. Expected content provider hint: {provider_hint}. Observation backend: {provider}. {} {SHARED_SEMANTIC_FRAME_OUTPUT_SUFFIX}",
        resize_plan.resolution_prompt()
    )
}

fn focused_perception_prompts(
    request: &FocusedPerceptionRequest,
    goal: &str,
    app_hint: &str,
    provider_hint: &str,
    provider: &str,
    region: &str,
    resize_plan: &VisionResizePlan,
) -> (String, String) {
    let surface_context = request
        .verified_surface
        .as_ref()
        .map(|surface| {
            let bounds = surface
                .bounds
                .as_ref()
                .map(|bounds| {
                    format!(
                        "x={}, y={}, width={}, height={}, coordinate_space={}",
                        bounds.x, bounds.y, bounds.width, bounds.height, bounds.coordinate_space
                    )
                })
                .unwrap_or_else(|| "unavailable".into());
            format!(
                "kind={:?}, provider_hint={:?}, app_hint={:?}, source_frame_id={}, bounds={bounds}",
                surface.kind, surface.provider_hint, surface.app_hint, surface.source_frame_id
            )
        })
        .unwrap_or_else(|| "none".into());
    match request.mode {
        PerceptionRequestMode::TargetFocus => (
            [
                "You are Astra Focused Semantic Perception. Return only strict JSON, no markdown. ",
                "Inspect the full screenshot but focus your structured observations on the requested screen region. ",
                "When a verified interaction surface is supplied, treat it as the browser ownership boundary and ignore Astra chat, terminal, and unrelated desktop content outside it. ",
                "Do not choose actions or command the desktop. Return a semantic_frame compatible with the full-frame schema. ",
                "When the focused region contains a result card, title link, thumbnail, avatar, or control, provide precise click_regions. ",
                SHARED_REGION_OUTPUT_INSTRUCTIONS,
                "Keep browser_app_hint, content_provider_hint, page_kind_hint, query_hint, and capture_backend separated. ",
                "The observation backend is technical metadata and never the visible website/provider.",
            ]
            .concat(),
            format!(
                "Goal: {goal}. Focus reason: {}. Routing decision: {:?}. Focus target item: {:?}. Focus target entity: {:?}. Focus region: {region}. Verified interaction surface: {surface_context}. Browser/app hint: {app_hint}. Expected content provider hint: {provider_hint}. Observation backend: {provider}. {} {SHARED_SEMANTIC_FRAME_OUTPUT_SUFFIX}",
                request.reason,
                request.routing_decision,
                request.target_item_id,
                request.target_entity_id,
                resize_plan.resolution_prompt(),
            ),
        ),
        PerceptionRequestMode::VisiblePageRefinement => (
            [
                "You are Astra Visible Actionability Refinement. Return only strict JSON, no markdown. ",
                "The page is already visually verified; refine the semantic understanding of what is currently visible before any off-screen inference. ",
                "Inspect the full screenshot and improve visible_result_items, visible_entities, and actionable_controls for currently visible cards, rows, tiles, links, thumbnails, avatars, and controls. ",
                "When a verified interaction surface is supplied, refine only inside that browser-owned surface and do not reinterpret Astra chat, terminal, or unrelated desktop UI as results. ",
                "If visible items are weakly typed or missing titles, ranks, or click_regions, improve them conservatively. ",
                SHARED_REGION_OUTPUT_INSTRUCTIONS,
                "Do not invent off-screen targets or hidden content. Prefer explicit uncertainty over guessing. ",
                "Keep browser_app_hint, content_provider_hint, page_kind_hint, query_hint, and capture_backend separated.",
            ]
            .concat(),
            format!(
                "Goal: {goal}. Refinement reason: {}. Routing decision: {:?}. Refinement mode: {:?}. Refinement strategy: {:?}. Target item context: {:?}. Target entity context: {:?}. Visible region cluster: {region}. Verified interaction surface: {surface_context}. Browser/app hint: {app_hint}. Expected content provider hint: {provider_hint}. Observation backend: {provider}. {} {SHARED_SEMANTIC_FRAME_OUTPUT_SUFFIX}",
                request.reason,
                request.routing_decision,
                request.mode,
                request.refinement_strategy,
                request.target_item_id,
                request.target_entity_id
                ,
                resize_plan.resolution_prompt()
            ),
        ),
    }
}

fn structured_candidate_system_prompt() -> String {
    [
        "You are Astra Vision Target Grounding. Return only strict JSON, no markdown. ",
        "Your task is to propose visible UI element candidates for safe desktop automation. ",
        "Only include an element if you can provide an explicit rectangular region in screen pixel coordinates. ",
        "Use coordinate_space=\"screen\". Do not infer hidden elements. Do not include vague prose-only candidates. ",
        "Allowed legacy ui_candidate roles: search_input, ranked_result, button, link, text_input. ",
        "Each candidate must include candidate_id, role, region {x,y,width,height,coordinate_space}, confidence 0..1, ",
        "supports_focus, supports_click, optional label, optional browser_app_hint/app_hint, optional content_provider_hint/provider_hint, optional page_kind_hint, optional rank, and rationale. ",
        SHARED_LEGACY_CANDIDATE_EXAMPLE,
        SHARED_REGION_OUTPUT_INSTRUCTIONS,
        "Also include optional semantic_frame with page_evidence, scene_summary, primary_list, page_state, visible_entities, visible_result_items, actionable_controls, and uncertainty. ",
        "Also include page_evidence with browser_app_hint, content_provider_hint, page_kind_hint, query_hint, result_list_visible, confidence, and uncertainty. ",
        "For visible_result_items, distinguish video, mix, playlist, channel, hotel, product, repository, and generic; include rank_overall and rank_within_kind. ",
        "content_provider_hint/provider_hint means the visible site or content provider such as youtube, google, github, or amazon. ",
        "Never copy observation backend or screenshot backend identifiers into provider_hint/content_provider_hint. ",
        "If uncertain, return {\"page_evidence\":{\"confidence\":0.0,\"uncertainty\":[\"inconclusive\"]},\"semantic_frame\":{\"scene_summary\":\"inconclusive\",\"visible_entities\":[],\"visible_result_items\":[],\"actionable_controls\":[],\"uncertainty\":[\"inconclusive\"]},\"ui_candidates\":[]}.",
    ]
    .concat()
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

fn parse_model_json(content: &str, context: &str) -> Result<Value, String> {
    let json_text = extract_json_object(content)
        .ok_or_else(|| format!("{context} did not contain a JSON object"))?;
    serde_json::from_str(json_text).map_err(|error| format!("{context} JSON parse failed: {error}"))
}

fn prepare_image_for_vision(image_path: &str) -> Result<PreparedVisionImage, String> {
    let image_bytes = fs::read(image_path)
        .map_err(|error| format!("vision preprocessing failed to read capture: {error}"))?;
    let image = image::load_from_memory(&image_bytes)
        .map_err(|error| format!("vision preprocessing failed to decode capture: {error}"))?;
    let (original_width, original_height) = image.dimensions();
    let resize_plan = VisionResizePlan::new(original_width, original_height);
    let analyzed_bytes = if resize_plan.requires_resize() {
        let resized = image.resize_exact(
            resize_plan.analyzed_width,
            resize_plan.analyzed_height,
            FilterType::Triangle,
        );
        let mut cursor = Cursor::new(Vec::new());
        resized
            .write_to(&mut cursor, ImageFormat::Png)
            .map_err(|error| {
                format!("vision preprocessing failed to encode resized capture: {error}")
            })?;
        cursor.into_inner()
    } else {
        image_bytes
    };
    Ok(PreparedVisionImage {
        image_b64: BASE64_STANDARD.encode(analyzed_bytes),
        resize_plan,
    })
}

fn sanitize_suspicious_prompt_template_geometry(
    parsed: &mut Value,
) -> Option<VisionGeometrySanitizationReport> {
    let mut scan_report = VisionGeometrySanitizationReport::default();
    process_suspicious_prompt_geometry(parsed, false, &mut scan_report);
    if scan_report.total_matches() < 2 {
        return None;
    }

    let mut sanitized = VisionGeometrySanitizationReport::default();
    process_suspicious_prompt_geometry(parsed, true, &mut sanitized);
    if sanitized.total_matches() == 0 {
        return None;
    }

    let message = format!(
        "suspicious_prompt_template_geometry_removed:{}",
        sanitized.matched_patterns.join(",")
    );
    append_uncertainty_to_value(parsed, &message);
    if let Some(frame) = frame_value_mut(parsed) {
        append_uncertainty_to_value(frame, &message);
    }
    if let Some(page_evidence) = page_evidence_value_mut(parsed) {
        append_uncertainty_to_value(page_evidence, &message);
    }

    Some(sanitized)
}

fn process_suspicious_prompt_geometry(
    parsed: &mut Value,
    enforce: bool,
    report: &mut VisionGeometrySanitizationReport,
) {
    if let Some(frame) = frame_value_mut(parsed) {
        process_frame_prompt_geometry(frame, enforce, report);
    } else {
        process_frame_prompt_geometry(parsed, enforce, report);
    }
    process_ui_candidate_prompt_geometry(parsed, enforce, report);
}

fn frame_value_mut(parsed: &mut Value) -> Option<&mut Value> {
    if parsed.pointer("/semantic_frame").is_some() {
        return parsed.pointer_mut("/semantic_frame");
    }
    if parsed.pointer("/screen_frame").is_some() {
        return parsed.pointer_mut("/screen_frame");
    }
    if parsed.pointer("/frame").is_some() {
        return parsed.pointer_mut("/frame");
    }
    None
}

fn page_evidence_value_mut(parsed: &mut Value) -> Option<&mut Value> {
    if parsed.pointer("/page_evidence").is_some() {
        return parsed.pointer_mut("/page_evidence");
    }
    if parsed.pointer("/semantic_frame/page_evidence").is_some() {
        return parsed.pointer_mut("/semantic_frame/page_evidence");
    }
    if parsed.pointer("/screen_frame/page_evidence").is_some() {
        return parsed.pointer_mut("/screen_frame/page_evidence");
    }
    if parsed.pointer("/frame/page_evidence").is_some() {
        return parsed.pointer_mut("/frame/page_evidence");
    }
    None
}

fn process_frame_prompt_geometry(
    frame: &mut Value,
    enforce: bool,
    report: &mut VisionGeometrySanitizationReport,
) {
    let Some(object) = frame.as_object_mut() else {
        return;
    };

    if let Some(items) = object
        .get_mut("visible_result_items")
        .and_then(Value::as_array_mut)
    {
        process_item_prompt_geometry(items, enforce, report);
    }

    if let Some(items) = object
        .get_mut("primary_list")
        .and_then(Value::as_object_mut)
        .and_then(|primary_list| primary_list.get_mut("items"))
        .and_then(Value::as_array_mut)
    {
        process_item_prompt_geometry(items, enforce, report);
    }

    if let Some(controls) = object
        .get_mut("actionable_controls")
        .and_then(Value::as_array_mut)
    {
        let mut sanitized_controls = Vec::new();
        for control in controls.drain(..) {
            if let Some(pattern) = suspicious_prompt_template_label_for_entry(&control) {
                report.record_pattern(pattern);
                report.removed_controls += 1;
                if !enforce {
                    sanitized_controls.push(control);
                }
                continue;
            }
            sanitized_controls.push(control);
        }
        *controls = sanitized_controls;
    }

    if let Some(entities) = object
        .get_mut("visible_entities")
        .and_then(Value::as_array_mut)
    {
        for entity in entities {
            let Some(entity_object) = entity.as_object_mut() else {
                continue;
            };
            let Some(region_value) = entity_object.get("region") else {
                continue;
            };
            if let Some(pattern) = suspicious_prompt_template_label_for_region(region_value) {
                report.record_pattern(pattern);
                report.stripped_entity_regions += 1;
                if enforce {
                    entity_object.remove("region");
                }
            }
        }
    }
}

fn process_item_prompt_geometry(
    items: &mut Vec<Value>,
    enforce: bool,
    report: &mut VisionGeometrySanitizationReport,
) {
    for item in items {
        let Some(item_object) = item.as_object_mut() else {
            continue;
        };
        if let Some(click_regions) = item_object
            .get_mut("click_regions")
            .and_then(Value::as_object_mut)
        {
            let mut keys_to_remove = Vec::new();
            for (key, region_value) in click_regions.iter_mut() {
                if let Some(pattern) = suspicious_prompt_template_label_for_entry(region_value) {
                    report.record_pattern(pattern);
                    report.removed_click_regions += 1;
                    if enforce {
                        keys_to_remove.push(key.clone());
                    }
                }
            }
            if enforce {
                for key in keys_to_remove {
                    click_regions.remove(&key);
                }
            }
        }
        if let Some(attributes) = item_object
            .get_mut("attributes")
            .and_then(Value::as_object_mut)
        {
            strip_suspicious_attribute_region(attributes, enforce, report);
        }
    }
}

fn process_ui_candidate_prompt_geometry(
    parsed: &mut Value,
    enforce: bool,
    report: &mut VisionGeometrySanitizationReport,
) {
    let Some(candidates) = parsed
        .get_mut("ui_candidates")
        .and_then(Value::as_array_mut)
    else {
        return;
    };

    let mut sanitized_candidates = Vec::new();
    for candidate in candidates.drain(..) {
        if let Some(pattern) = suspicious_prompt_template_label_for_entry(&candidate) {
            report.record_pattern(pattern);
            report.removed_candidates += 1;
            if !enforce {
                sanitized_candidates.push(candidate);
            }
            continue;
        }
        sanitized_candidates.push(candidate);
    }
    *candidates = sanitized_candidates;
}

fn strip_suspicious_attribute_region(
    attributes: &mut serde_json::Map<String, Value>,
    enforce: bool,
    report: &mut VisionGeometrySanitizationReport,
) {
    for key in ["region", "bounding_region", "bounds"] {
        let Some(region_value) = attributes.get(key) else {
            continue;
        };
        if let Some(pattern) = suspicious_prompt_template_label_for_region(region_value) {
            report.record_pattern(pattern);
            report.stripped_attribute_regions += 1;
            if enforce {
                attributes.remove(key);
            }
            break;
        }
    }
}

fn suspicious_prompt_template_label_for_entry(value: &Value) -> Option<&'static str> {
    value
        .get("region")
        .or_else(|| value.get("bounds"))
        .or_else(|| value.get("bounding_region"))
        .and_then(suspicious_prompt_template_label_for_region)
        .or_else(|| {
            value
                .get("attributes")
                .and_then(Value::as_object)
                .and_then(|attributes| {
                    ["region", "bounding_region", "bounds"]
                        .iter()
                        .find_map(|key| {
                            attributes
                                .get(*key)
                                .and_then(suspicious_prompt_template_label_for_region)
                        })
                })
        })
}

fn suspicious_prompt_template_label_for_region(region: &Value) -> Option<&'static str> {
    let x = region.get("x").and_then(Value::as_f64)?;
    let y = region.get("y").and_then(Value::as_f64)?;
    let width = region.get("width").and_then(Value::as_f64)?;
    let height = region.get("height").and_then(Value::as_f64)?;
    SUSPICIOUS_PROMPT_TEMPLATE_REGIONS
        .iter()
        .find(|pattern| {
            approx_eq(x, pattern.x)
                && approx_eq(y, pattern.y)
                && approx_eq(width, pattern.width)
                && approx_eq(height, pattern.height)
        })
        .map(|pattern| pattern.label)
}

fn append_uncertainty_to_value(value: &mut Value, message: &str) {
    let Some(object) = value.as_object_mut() else {
        return;
    };
    let entry = object
        .entry("uncertainty")
        .or_insert_with(|| Value::Array(Vec::new()));
    let Some(items) = entry.as_array_mut() else {
        return;
    };
    if !items
        .iter()
        .any(|existing| existing.as_str() == Some(message))
    {
        items.push(json!(message));
    }
}

fn approx_eq(left: f64, right: f64) -> bool {
    (left - right).abs() < 0.01
}

fn maybe_reproject_model_coordinates(parsed: &mut Value, resize_plan: &VisionResizePlan) {
    if !resize_plan.requires_resize() {
        return;
    }
    reproject_value_coordinates(parsed, resize_plan);
}

fn sanitize_invalid_primary_list_geometry(
    parsed: &mut Value,
    resize_plan: &VisionResizePlan,
) -> usize {
    let Some(frame) = frame_value_mut(parsed) else {
        return 0;
    };
    let Some(items) = frame
        .get_mut("primary_list")
        .and_then(Value::as_object_mut)
        .and_then(|primary_list| primary_list.get_mut("items"))
        .and_then(Value::as_array_mut)
    else {
        return 0;
    };

    let mut removed_items = 0usize;
    let mut kept_items = Vec::new();
    for mut item in items.drain(..) {
        let Some(item_object) = item.as_object_mut() else {
            removed_items += 1;
            continue;
        };
        let had_click_regions = item_object.get("click_regions").is_some();
        if let Some(click_regions) = item_object
            .get_mut("click_regions")
            .and_then(Value::as_object_mut)
        {
            let invalid_keys = click_regions
                .iter()
                .filter_map(|(key, value)| {
                    let region = value.get("region").unwrap_or(value);
                    (!region_within_capture_bounds(region, resize_plan)).then(|| key.clone())
                })
                .collect::<Vec<_>>();
            for key in invalid_keys {
                click_regions.remove(&key);
            }
            if had_click_regions && click_regions.is_empty() {
                removed_items += 1;
                continue;
            }
        }
        kept_items.push(item);
    }
    *items = kept_items;

    if removed_items > 0 {
        append_uncertainty_to_value(
            frame,
            &format!("primary_list_invalid_geometry_items_discarded:{removed_items}"),
        );
    }
    removed_items
}

fn region_within_capture_bounds(region: &Value, resize_plan: &VisionResizePlan) -> bool {
    let Some(x) = region.get("x").and_then(Value::as_f64) else {
        return false;
    };
    let Some(y) = region.get("y").and_then(Value::as_f64) else {
        return false;
    };
    let Some(width) = region.get("width").and_then(Value::as_f64) else {
        return false;
    };
    let Some(height) = region.get("height").and_then(Value::as_f64) else {
        return false;
    };
    x.is_finite()
        && y.is_finite()
        && width.is_finite()
        && height.is_finite()
        && x >= 0.0
        && y >= 0.0
        && width > 0.0
        && height > 0.0
        && x + width <= resize_plan.original_width as f64 + 1.0
        && y + height <= resize_plan.original_height as f64 + 1.0
}

fn reproject_value_coordinates(value: &mut Value, resize_plan: &VisionResizePlan) {
    match value {
        Value::Object(object) => {
            if let Some(region) = reproject_region_object(object, resize_plan) {
                *value = Value::Object(region);
                return;
            }

            if let Some(center_x) = object.get_mut("center_x") {
                if let Some(value) = center_x.as_f64() {
                    *center_x = json!(reproject_x(value, resize_plan));
                }
            }
            if let Some(center_y) = object.get_mut("center_y") {
                if let Some(value) = center_y.as_f64() {
                    *center_y = json!(reproject_y(value, resize_plan));
                }
            }
            for child in object.values_mut() {
                reproject_value_coordinates(child, resize_plan);
            }
        }
        Value::Array(items) => {
            for item in items {
                reproject_value_coordinates(item, resize_plan);
            }
        }
        _ => {}
    }
}

fn reproject_region_object(
    object: &mut serde_json::Map<String, Value>,
    resize_plan: &VisionResizePlan,
) -> Option<serde_json::Map<String, Value>> {
    let x = object.get("x").and_then(Value::as_f64)?;
    let y = object.get("y").and_then(Value::as_f64)?;
    let width = object.get("width").and_then(Value::as_f64)?;
    let height = object.get("height").and_then(Value::as_f64)?;
    let mut region = object.clone();
    region.insert("x".into(), json!(reproject_x(x, resize_plan)));
    region.insert("y".into(), json!(reproject_y(y, resize_plan)));
    region.insert("width".into(), json!(reproject_x(width, resize_plan)));
    region.insert("height".into(), json!(reproject_y(height, resize_plan)));
    Some(region)
}

fn reproject_x(value: f64, resize_plan: &VisionResizePlan) -> f64 {
    ((value * resize_plan.scale_x()) * 100.0).round() / 100.0
}

fn reproject_y(value: f64, resize_plan: &VisionResizePlan) -> f64 {
    ((value * resize_plan.scale_y()) * 100.0).round() / 100.0
}

#[derive(Clone)]
pub struct ScreenVisionRuntime {
    client: Client,
}

impl ScreenVisionRuntime {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub async fn availability(&self) -> VisionAvailability {
        let installed_models = self.fetch_installed_models().await.unwrap_or_default();
        let candidates = vision_candidates();
        let selected_model = select_vision_model(&installed_models);
        VisionAvailability {
            available: selected_model.is_some(),
            selected_model,
            candidates,
        }
    }

    pub async fn analyze(
        &self,
        image_path: &str,
        captured_at: u64,
        provider: &str,
        question: Option<String>,
    ) -> Result<ScreenAnalysisResult, String> {
        let installed_models = self.fetch_installed_models().await.unwrap_or_default();
        let model = select_vision_model(&installed_models)
            .ok_or_else(|| "No compatible Ollama vision model found. Install one of: qwen2.5vl, llava, llava-phi3, moondream".to_string())?;

        let prepared_image = prepare_image_for_vision(image_path)?;

        let question = question
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "Describe what is visible on this screen, identify the active context, the main UI regions, and any likely issue or next action for the user.".to_string());

        let system = "You are Astra Vision, the screen-awareness subsystem of a local desktop assistant. Analyze screenshots conservatively and precisely. Describe only what is visible or strongly inferable. When helpful, structure the answer in short sections: context, notable elements, possible issue, next step. If the screen looks like code, IDE, terminal, browser, settings, or an error state, say so clearly. Do not invent hidden content. Keep the response concise but useful.";
        let analysis_prompt = format!(
            "{} {}",
            prepared_image.resize_plan.resolution_prompt(),
            question
        );

        let payload = json!({
            "model": model,
            "stream": false,
            "options": {
                "temperature": 0.2,
                "num_ctx": 8192,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": analysis_prompt, "images": [prepared_image.image_b64]}
            ]
        });

        let response = self
            .client
            .post(format!("{OLLAMA_BASE_URL}/api/chat"))
            .json(&payload)
            .send()
            .await
            .map_err(|error| format!("Ollama vision request failed: {error}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Ollama vision HTTP error {status}: {body}"));
        }

        let body: Value = response
            .json()
            .await
            .map_err(|error| format!("Ollama vision parse failed: {error}"))?;

        let answer = body
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "Ollama vision returned an empty response".to_string())?
            .to_string();
        let semantic_frame = self
            .perceive_semantic_frame(
                image_path,
                captured_at,
                provider,
                Some(&question),
                None,
                None,
            )
            .await
            .ok();

        Ok(ScreenAnalysisResult {
            analysis_id: Uuid::new_v4().to_string(),
            request_id: Uuid::new_v4().to_string(),
            captured_at,
            image_path: image_path.to_string(),
            model,
            provider: provider.to_string(),
            question,
            answer,
            ui_candidates: Vec::new(),
            structured_candidates_error: None,
            semantic_frame,
        })
    }

    pub async fn perceive_semantic_frame(
        &self,
        image_path: &str,
        captured_at: u64,
        provider: &str,
        goal: Option<&str>,
        app_hint: Option<&str>,
        provider_hint: Option<&str>,
    ) -> Result<SemanticScreenFrame, String> {
        let installed_models = self.fetch_installed_models().await.unwrap_or_default();
        let model = select_vision_model(&installed_models)
            .ok_or_else(|| "No compatible Ollama vision model found. Install one of: qwen2.5vl, llava, llava-phi3, moondream".to_string())?;

        let prepared_image = prepare_image_for_vision(image_path)?;
        let goal = goal.unwrap_or("understand the visible screen for safe desktop assistance");
        let app_hint = app_hint.unwrap_or("unknown");
        let provider_hint = provider_hint.unwrap_or("unknown");
        let system = semantic_perception_system_prompt();
        let user = semantic_perception_user_prompt(
            goal,
            app_hint,
            provider_hint,
            provider,
            &prepared_image.resize_plan,
        );

        let payload = json!({
            "model": model,
            "stream": false,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [prepared_image.image_b64]}
            ]
        });

        let response = self
            .client
            .post(format!("{OLLAMA_BASE_URL}/api/chat"))
            .json(&payload)
            .send()
            .await
            .map_err(|error| format!("Ollama semantic perception request failed: {error}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "Ollama semantic perception HTTP error {status}: {body}"
            ));
        }

        let body: Value = response
            .json()
            .await
            .map_err(|error| format!("Ollama semantic perception parse failed: {error}"))?;
        let content = body
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "Ollama semantic perception returned an empty response".to_string())?;
        let mut parsed = parse_model_json(content, "semantic screen perception")?;
        sanitize_suspicious_prompt_template_geometry(&mut parsed);
        maybe_reproject_model_coordinates(&mut parsed, &prepared_image.resize_plan);
        sanitize_invalid_primary_list_geometry(&mut parsed, &prepared_image.resize_plan);
        let mut frame = semantic_frame_from_vision_value(
            &parsed,
            captured_at,
            Some(image_path.to_string()),
            None,
            Vec::new(),
        )
        .ok_or_else(|| {
            "semantic screen perception response did not contain useful structured perception"
                .to_string()
        })?;
        frame.page_evidence.capture_backend = frame
            .page_evidence
            .capture_backend
            .clone()
            .or_else(|| Some(provider.to_string()));
        frame.page_evidence.observation_source = frame
            .page_evidence
            .observation_source
            .clone()
            .or_else(|| Some("semantic_perception".into()));
        Ok(frame)
    }

    #[allow(dead_code)]
    pub async fn perceive_focused_region(
        &self,
        image_path: &str,
        captured_at: u64,
        provider: &str,
        request: &FocusedPerceptionRequest,
        goal: Option<&str>,
        app_hint: Option<&str>,
        provider_hint: Option<&str>,
    ) -> Result<SemanticScreenFrame, String> {
        let installed_models = self.fetch_installed_models().await.unwrap_or_default();
        let model = select_vision_model(&installed_models)
            .ok_or_else(|| "No compatible Ollama vision model found. Install one of: qwen2.5vl, llava, llava-phi3, moondream".to_string())?;

        let prepared_image = prepare_image_for_vision(image_path)?;
        let goal = goal.unwrap_or("resolve the ambiguous UI target safely");
        let app_hint = app_hint.unwrap_or("unknown");
        let provider_hint = provider_hint.unwrap_or("unknown");
        let region = request
            .region
            .as_ref()
            .map(|region| {
                format!(
                    "x={}, y={}, width={}, height={}, coordinate_space={}",
                    region.x, region.y, region.width, region.height, region.coordinate_space
                )
            })
            .unwrap_or_else(|| "none".into());
        let (system, user) = focused_perception_prompts(
            request,
            goal,
            app_hint,
            provider_hint,
            provider,
            &region,
            &prepared_image.resize_plan,
        );

        let payload = json!({
            "model": model,
            "stream": false,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [prepared_image.image_b64]}
            ]
        });

        let response = self
            .client
            .post(format!("{OLLAMA_BASE_URL}/api/chat"))
            .json(&payload)
            .send()
            .await
            .map_err(|error| {
                format!("Ollama focused semantic perception request failed: {error}")
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "Ollama focused semantic perception HTTP error {status}: {body}"
            ));
        }

        let body: Value = response
            .json()
            .await
            .map_err(|error| format!("Ollama focused semantic perception parse failed: {error}"))?;
        let content = body
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                "Ollama focused semantic perception returned an empty response".to_string()
            })?;
        let mut parsed = parse_model_json(content, "focused semantic perception")?;
        sanitize_suspicious_prompt_template_geometry(&mut parsed);
        maybe_reproject_model_coordinates(&mut parsed, &prepared_image.resize_plan);
        sanitize_invalid_primary_list_geometry(&mut parsed, &prepared_image.resize_plan);
        let mut frame = semantic_frame_from_vision_value(
            &parsed,
            captured_at,
            Some(image_path.to_string()),
            None,
            Vec::new(),
        )
        .ok_or_else(|| {
            "focused semantic perception response did not contain useful structured perception"
                .to_string()
        })?;
        frame.page_evidence.capture_backend = frame
            .page_evidence
            .capture_backend
            .clone()
            .or_else(|| Some(provider.to_string()));
        frame.page_evidence.observation_source = frame
            .page_evidence
            .observation_source
            .clone()
            .or_else(|| Some("focused_semantic_perception".into()));
        Ok(frame)
    }

    pub async fn extract_ui_candidates(
        &self,
        image_path: &str,
        captured_at: u64,
        provider: &str,
        requested_roles: &[String],
        app_hint: Option<&str>,
        provider_hint: Option<&str>,
    ) -> Result<StructuredVisionExtraction, String> {
        let installed_models = self.fetch_installed_models().await.unwrap_or_default();
        let model = select_vision_model(&installed_models)
            .ok_or_else(|| "No compatible Ollama vision model found. Install one of: qwen2.5vl, llava, llava-phi3, moondream".to_string())?;

        let prepared_image = prepare_image_for_vision(image_path)?;
        let requested_roles = if requested_roles.is_empty() {
            "search_input, ranked_result, button, link, text_input".to_string()
        } else {
            requested_roles.join(", ")
        };
        let app_hint = app_hint.unwrap_or("unknown");
        let provider_hint = provider_hint.unwrap_or("unknown");

        let system = structured_candidate_system_prompt();
        let user = format!(
            "Extract UI candidates for requested roles: {requested_roles}. Browser/app hint: {app_hint}. Expected content provider hint: {provider_hint}. Observation backend: {provider} (technical capture metadata only; it is not the website/provider). {} Return JSON with keys page_evidence and ui_candidates.",
            prepared_image.resize_plan.resolution_prompt()
        );

        let payload = json!({
            "model": model,
            "stream": false,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [prepared_image.image_b64]}
            ]
        });

        let response = self
            .client
            .post(format!("{OLLAMA_BASE_URL}/api/chat"))
            .json(&payload)
            .send()
            .await
            .map_err(|error| format!("Ollama structured vision request failed: {error}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "Ollama structured vision HTTP error {status}: {body}"
            ));
        }

        let body: Value = response
            .json()
            .await
            .map_err(|error| format!("Ollama structured vision parse failed: {error}"))?;
        let content = body
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "Ollama structured vision returned an empty response".to_string())?;

        let mut parsed = parse_model_json(content, "structured vision extraction")?;
        sanitize_suspicious_prompt_template_geometry(&mut parsed);
        maybe_reproject_model_coordinates(&mut parsed, &prepared_image.resize_plan);
        sanitize_invalid_primary_list_geometry(&mut parsed, &prepared_image.resize_plan);
        let mut extraction = parse_structured_vision_candidates_from_value(&parsed, captured_at)?;
        let page_evidence = extraction
            .page_evidence
            .get_or_insert_with(|| PageSemanticEvidence {
                browser_app_hint: None,
                content_provider_hint: None,
                page_kind_hint: None,
                query_hint: None,
                result_list_visible: None,
                raw_confidence: None,
                confidence: 0.0,
                evidence_sources: vec![PageEvidenceSource::CaptureMetadata],
                capture_backend: Some(provider.to_string()),
                observation_source: Some("structured_vision".into()),
                uncertainty: vec!["vision_model_returned_no_page_evidence".into()],
            });
        page_evidence.capture_backend = page_evidence
            .capture_backend
            .clone()
            .or_else(|| Some(provider.to_string()));
        page_evidence.observation_source = page_evidence
            .observation_source
            .clone()
            .or_else(|| Some("structured_vision".into()));
        if let Some(frame) = extraction.semantic_frame.as_mut() {
            frame.image_path = frame.image_path.clone().or_else(|| Some(image_path.into()));
            frame.page_evidence.capture_backend = frame
                .page_evidence
                .capture_backend
                .clone()
                .or_else(|| Some(provider.to_string()));
            frame.page_evidence.observation_source = frame
                .page_evidence
                .observation_source
                .clone()
                .or_else(|| Some("structured_vision".into()));
        }
        Ok(extraction)
    }

    async fn fetch_installed_models(&self) -> Result<Vec<String>, String> {
        let response = self
            .client
            .get(format!("{OLLAMA_BASE_URL}/api/tags"))
            .send()
            .await
            .map_err(|error| format!("Ollama tags request failed: {error}"))?;

        if !response.status().is_success() {
            return Err(format!("Ollama tags HTTP error: {}", response.status()));
        }

        let body: Value = response
            .json()
            .await
            .map_err(|error| format!("Ollama tags parse failed: {error}"))?;

        Ok(body
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.get("name").and_then(Value::as_str))
            .map(ToOwned::to_owned)
            .collect())
    }
}

fn vision_candidates() -> Vec<String> {
    env::var("ASTRA_MODEL_VISION_CANDIDATES")
        .unwrap_or_else(|_| {
            "qwen2.5vl:7b,qwen2.5vl:3b,llava:7b,llava:13b,llava-phi3:3.8b,moondream:latest"
                .to_string()
        })
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>()
}

fn select_vision_model(installed_models: &[String]) -> Option<String> {
    let candidates = vision_candidates();

    if installed_models.is_empty() {
        return candidates.first().cloned();
    }

    let installed_lower = installed_models
        .iter()
        .map(|value| value.to_ascii_lowercase())
        .collect::<Vec<_>>();

    candidates.iter().find_map(|candidate| {
        let exact = candidate.to_ascii_lowercase();
        if installed_lower.iter().any(|installed| installed == &exact) {
            return Some(candidate.clone());
        }

        let base = exact.split(':').next().unwrap_or(&exact).to_string();
        installed_models.iter().find_map(|installed| {
            let installed_lower = installed.to_ascii_lowercase();
            (installed_lower == base || installed_lower.starts_with(&(base.clone() + ":")))
                .then(|| installed.clone())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desktop_agent_types::{
        FocusedPerceptionRequest, PerceptionRequestMode, PerceptionRoutingDecision,
        VisibleRefinementStrategy,
    };

    #[test]
    fn semantic_perception_prompt_includes_region_schema_examples() {
        let prompt = semantic_perception_system_prompt();
        assert!(prompt.contains("\"click_regions\""));
        assert!(prompt.contains("\"actionable_controls\""));
        assert!(prompt.contains("\"primary_list\""));
        assert!(prompt.contains("\"page_state\""));
        assert!(prompt.contains("\"coordinate_space\":\"screen\""));
        assert!(prompt.contains("Do not leave click_regions empty"));
        assert!(prompt.contains("Never output provider names"));
        assert!(prompt.contains("YOUR_ACTUAL_X_IN_PIXELS"));
        assert!(prompt.contains("fake placeholders"));
        assert!(!prompt.contains("\"x\":120"));
        assert!(!prompt.contains("\"x\":90"));
    }

    #[test]
    fn focused_refinement_prompt_includes_routing_and_region_contract() {
        let request = FocusedPerceptionRequest {
            request_id: "request_1".into(),
            iteration: 1,
            reason: "regionless visible target".into(),
            mode: PerceptionRequestMode::VisiblePageRefinement,
            routing_decision: PerceptionRoutingDecision::RegionlessTargetVisible,
            refinement_strategy: VisibleRefinementStrategy::VisibleCluster,
            target_item_id: Some("video".into()),
            target_entity_id: None,
            region: None,
            target_region_anchor_present: false,
            verified_surface: None,
        };

        let (system, user) = focused_perception_prompts(
            &request,
            "aprimi il primo video",
            "chrome",
            "youtube",
            "powershell_gdi",
            "none",
            &VisionResizePlan::new(3440, 1440),
        );

        assert!(system.contains("\"click_regions\""));
        assert!(system.contains("\"actionable_controls\""));
        assert!(system.contains("\"primary_list\""));
        assert!(user.contains("Routing decision"));
        assert!(user.contains("RegionlessTargetVisible"));
        assert!(user.contains("Analyzed image resolution: 1280x536 pixels"));
        assert!(user.contains("Do not reuse placeholder/example coordinates"));
    }

    #[test]
    fn structured_candidate_prompt_requires_explicit_screen_regions() {
        let prompt = structured_candidate_system_prompt();
        assert!(prompt
            .contains("Only include an element if you can provide an explicit rectangular region"));
        assert!(prompt.contains("\"ui_candidates\""));
        assert!(prompt.contains("\"role\":\"ranked_result\""));
        assert!(prompt.contains("primary_list"));
        assert!(prompt.contains("\"coordinate_space\":\"screen\""));
        assert!(prompt.contains("YOUR_ACTUAL_X_IN_PIXELS"));
        assert!(!prompt.contains("\"x\":90"));
    }

    #[test]
    fn resize_plan_downscales_ultrawide_capture_with_aspect_ratio_preserved() {
        let plan = VisionResizePlan::new(3440, 1440);

        assert!(plan.requires_resize());
        assert_eq!(plan.analyzed_width, 1280);
        assert_eq!(plan.analyzed_height, 536);
        assert!((plan.scale_x() - 2.6875).abs() < 0.0001);
        assert!((plan.scale_y() - (1440.0 / 536.0)).abs() < 0.0001);
    }

    #[test]
    fn reprojection_scales_model_coordinates_back_to_original_capture_space() {
        let plan = VisionResizePlan::new(3440, 1440);
        let mut parsed = json!({
            "semantic_frame": {
                "primary_list": {
                    "cluster_id": "main",
                    "container_kind": "result_list",
                    "item_count": 1,
                    "items": [{
                        "item_id": "p1",
                        "rank": 1,
                        "item_kind": "generic",
                        "click_regions": {
                            "primary": {
                                "region": {
                                    "x": 40.0,
                                    "y": 80.0,
                                    "width": 100.0,
                                    "height": 40.0,
                                    "coordinate_space": "screen"
                                },
                                "confidence": 0.9
                            }
                        },
                        "confidence": 0.9
                    }]
                },
                "visible_result_items": [{
                    "item_id": "v1",
                    "kind": "video",
                    "click_regions": {
                        "title": {
                            "region": {
                                "x": 120.0,
                                "y": 240.0,
                                "width": 580.0,
                                "height": 36.0,
                                "coordinate_space": "screen"
                            },
                            "confidence": 0.88
                        }
                    }
                }],
                "actionable_controls": [{
                    "control_id": "play",
                    "kind": "button",
                    "region": {
                        "x": 90.0,
                        "y": 180.0,
                        "width": 320.0,
                        "height": 180.0,
                        "coordinate_space": "screen"
                    },
                    "confidence": 0.82
                }]
            },
            "ui_candidates": [{
                "candidate_id": "result_1",
                "center_x": 100.0,
                "center_y": 200.0,
                "region": {
                    "x": 80.0,
                    "y": 160.0,
                    "width": 300.0,
                    "height": 120.0,
                    "coordinate_space": "screen"
                }
            }]
        });

        maybe_reproject_model_coordinates(&mut parsed, &plan);

        assert_eq!(
            parsed["semantic_frame"]["visible_result_items"][0]["click_regions"]["title"]["region"]
                ["x"],
            json!(322.5)
        );
        assert_eq!(
            parsed["semantic_frame"]["visible_result_items"][0]["click_regions"]["title"]["region"]
                ["y"],
            json!(644.78)
        );
        assert_eq!(
            parsed["semantic_frame"]["primary_list"]["items"][0]["click_regions"]["primary"]
                ["region"]["x"],
            json!(107.5)
        );
        assert_eq!(
            parsed["semantic_frame"]["primary_list"]["items"][0]["click_regions"]["primary"]
                ["region"]["width"],
            json!(268.75)
        );
        assert_eq!(parsed["ui_candidates"][0]["center_x"], json!(268.75));
        assert_eq!(parsed["ui_candidates"][0]["center_y"], json!(537.31));
    }

    #[test]
    fn suspicious_prompt_template_geometry_is_removed_when_multiple_template_hits_exist() {
        let mut parsed = json!({
            "semantic_frame": {
                "page_evidence": {"confidence": 0.9},
                "visible_result_items": [{
                    "item_id": "v1",
                    "kind": "video",
                    "click_regions": {
                        "title": {
                            "region": {
                                "x": 120.0,
                                "y": 240.0,
                                "width": 580.0,
                                "height": 36.0,
                                "coordinate_space": "screen"
                            },
                            "confidence": 0.88
                        }
                    }
                }],
                "actionable_controls": [{
                    "control_id": "play",
                    "kind": "button",
                    "region": {
                        "x": 90.0,
                        "y": 180.0,
                        "width": 320.0,
                        "height": 180.0,
                        "coordinate_space": "screen"
                    },
                    "confidence": 0.82
                }]
            },
            "ui_candidates": [{
                "candidate_id": "result_1",
                "region": {
                    "x": 120.0,
                    "y": 240.0,
                    "width": 580.0,
                    "height": 36.0,
                    "coordinate_space": "screen"
                },
                "supports_click": true,
                "supports_focus": false
            }]
        });

        let report =
            sanitize_suspicious_prompt_template_geometry(&mut parsed).expect("sanitization report");

        assert_eq!(report.removed_click_regions, 1);
        assert_eq!(report.removed_controls, 1);
        assert_eq!(report.removed_candidates, 1);
        assert!(
            parsed["semantic_frame"]["visible_result_items"][0]["click_regions"]
                .as_object()
                .expect("click regions")
                .is_empty()
        );
        assert!(parsed["semantic_frame"]["actionable_controls"]
            .as_array()
            .expect("controls")
            .is_empty());
        assert!(parsed["ui_candidates"]
            .as_array()
            .expect("candidates")
            .is_empty());
        assert!(parsed["semantic_frame"]["uncertainty"]
            .as_array()
            .expect("frame uncertainty")
            .iter()
            .any(|value| value.as_str().is_some_and(
                |value| value.contains("suspicious_prompt_template_geometry_removed")
            )));
    }

    #[test]
    fn single_template_hit_is_not_removed_without_additional_suspicion() {
        let mut parsed = json!({
            "semantic_frame": {
                "visible_result_items": [{
                    "item_id": "v1",
                    "kind": "video",
                    "click_regions": {
                        "title": {
                            "region": {
                                "x": 120.0,
                                "y": 240.0,
                                "width": 580.0,
                                "height": 36.0,
                                "coordinate_space": "screen"
                            },
                            "confidence": 0.88
                        }
                    }
                }]
            }
        });

        let report = sanitize_suspicious_prompt_template_geometry(&mut parsed);

        assert!(report.is_none());
        assert!(
            parsed["semantic_frame"]["visible_result_items"][0]["click_regions"]["title"]
                .is_object()
        );
    }

    #[test]
    fn invalid_primary_list_geometry_is_discarded_without_crashing() {
        let plan = VisionResizePlan::new(1000, 800);
        let mut parsed = json!({
            "semantic_frame": {
                "primary_list": {
                    "cluster_id": "main",
                    "container_kind": "result_list",
                    "item_count": 2,
                    "items": [
                        {
                            "item_id": "bad",
                            "rank": 1,
                            "item_kind": "generic",
                            "click_regions": {
                                "primary": {
                                    "region": {
                                        "x": 980.0,
                                        "y": 100.0,
                                        "width": 80.0,
                                        "height": 40.0,
                                        "coordinate_space": "screen"
                                    },
                                    "confidence": 0.9
                                }
                            },
                            "confidence": 0.9
                        },
                        {
                            "item_id": "good",
                            "rank": 2,
                            "item_kind": "generic",
                            "click_regions": {
                                "primary": {
                                    "region": {
                                        "x": 100.0,
                                        "y": 100.0,
                                        "width": 80.0,
                                        "height": 40.0,
                                        "coordinate_space": "screen"
                                    },
                                    "confidence": 0.9
                                }
                            },
                            "confidence": 0.9
                        }
                    ]
                }
            }
        });

        let removed = sanitize_invalid_primary_list_geometry(&mut parsed, &plan);

        assert_eq!(removed, 1);
        let items = parsed["semantic_frame"]["primary_list"]["items"]
            .as_array()
            .expect("items");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["item_id"], json!("good"));
        assert!(parsed["semantic_frame"]["uncertainty"]
            .as_array()
            .expect("uncertainty")
            .iter()
            .any(|value| value
                .as_str()
                .is_some_and(|value| value.contains("primary_list_invalid_geometry"))));
    }
}
