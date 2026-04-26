use crate::{
    desktop_agent_types::{ExecutableCoordinateInterpretation, ExecutableGeometryDiagnostic},
    ui_target_grounding::TargetRegion,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UIPrimitiveKind {
    ActivateWindowOrApp,
    FocusCurrentInput,
    TypeText,
    PressEnter,
    NavigateBack,
    ScrollViewport,
    ClickTargetCandidate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIPrimitiveCapability {
    pub primitive: UIPrimitiveKind,
    pub available: bool,
    pub enabled: bool,
    pub requires_screen_context: bool,
    pub requires_high_confidence_target: bool,
    pub requires_approval: bool,
    pub platform_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIPrimitiveCapabilitySet {
    pub platform: String,
    pub desktop_control_enabled: bool,
    pub primitives: Vec<UIPrimitiveCapability>,
}

impl UIPrimitiveCapabilitySet {
    pub fn for_runtime(desktop_control_enabled: bool) -> Self {
        let platform = platform_name().to_string();
        let keyboard_available = cfg!(target_os = "windows");
        let keyboard_enabled = keyboard_available && desktop_control_enabled;
        let pointer_available = cfg!(target_os = "windows");
        let pointer_enabled = pointer_available && desktop_control_enabled;
        let activation_available = cfg!(target_os = "windows");
        let activation_enabled = activation_available && desktop_control_enabled;
        let keyboard_note = if keyboard_available {
            "Windows SendKeys backend is available for focused-control keyboard primitives."
        } else {
            "Keyboard UI primitives are not implemented for this platform yet."
        };
        let pointer_note = if pointer_available {
            "Windows pointer backend is available when a high-confidence target candidate supplies screen coordinates."
        } else {
            "Pointer UI primitives are not implemented for this platform yet."
        };

        Self {
            platform,
            desktop_control_enabled,
            primitives: vec![
                capability(
                    UIPrimitiveKind::ActivateWindowOrApp,
                    activation_available,
                    activation_enabled,
                    true,
                    false,
                    false,
                    "Bounded browser foreground activation is available for known browser windows on Windows.",
                ),
                capability(
                    UIPrimitiveKind::FocusCurrentInput,
                    pointer_available,
                    pointer_enabled,
                    true,
                    true,
                    false,
                    pointer_note,
                ),
                capability(
                    UIPrimitiveKind::TypeText,
                    keyboard_available,
                    keyboard_enabled,
                    false,
                    false,
                    false,
                    keyboard_note,
                ),
                capability(
                    UIPrimitiveKind::PressEnter,
                    keyboard_available,
                    keyboard_enabled,
                    false,
                    false,
                    false,
                    keyboard_note,
                ),
                capability(
                    UIPrimitiveKind::NavigateBack,
                    keyboard_available,
                    keyboard_enabled,
                    false,
                    false,
                    false,
                    keyboard_note,
                ),
                capability(
                    UIPrimitiveKind::ScrollViewport,
                    false,
                    false,
                    true,
                    false,
                    false,
                    "Scroll viewport primitive is not safely implemented in this runtime yet.",
                ),
                capability(
                    UIPrimitiveKind::ClickTargetCandidate,
                    pointer_available,
                    pointer_enabled,
                    true,
                    true,
                    false,
                    pointer_note,
                ),
            ],
        }
    }

    pub fn get(&self, kind: &UIPrimitiveKind) -> Option<&UIPrimitiveCapability> {
        self.primitives
            .iter()
            .find(|capability| capability.primitive == *kind)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIPrimitiveRequest {
    pub primitive: UIPrimitiveKind,
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default)]
    pub target: Value,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UIPrimitiveStatus {
    Executed,
    Unsupported,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIPrimitiveResult {
    pub primitive: UIPrimitiveKind,
    pub status: UIPrimitiveStatus,
    pub message: String,
    #[serde(default)]
    pub geometry: Option<ExecutableGeometryDiagnostic>,
}

#[derive(Clone)]
pub struct UIControlRuntime {
    dry_run: bool,
}

impl UIControlRuntime {
    pub fn new() -> Self {
        Self { dry_run: false }
    }

    #[cfg(test)]
    pub fn dry_run() -> Self {
        Self { dry_run: true }
    }

    pub fn capabilities(&self, desktop_control_enabled: bool) -> UIPrimitiveCapabilitySet {
        UIPrimitiveCapabilitySet::for_runtime(desktop_control_enabled)
    }

    pub fn execute(
        &self,
        request: &UIPrimitiveRequest,
        capabilities: &UIPrimitiveCapabilitySet,
    ) -> UIPrimitiveResult {
        let Some(capability) = capabilities.get(&request.primitive) else {
            return primitive_result(
                request.primitive.clone(),
                UIPrimitiveStatus::Unsupported,
                "Primitive is not registered in this runtime.",
            );
        };

        if !capability.available || !capability.enabled {
            return primitive_result(
                request.primitive.clone(),
                UIPrimitiveStatus::Unsupported,
                &capability.platform_note,
            );
        }

        let pointer_target = match request.primitive {
            UIPrimitiveKind::FocusCurrentInput | UIPrimitiveKind::ClickTargetCandidate => {
                match validate_pointer_target(request, !self.dry_run) {
                    Ok(target) => Some(target),
                    Err(error) => {
                        return primitive_result_with_geometry(
                            request.primitive.clone(),
                            UIPrimitiveStatus::Failed,
                            &error.message,
                            error.geometry,
                        )
                    }
                }
            }
            _ => None,
        };
        let activation_target = match request.primitive {
            UIPrimitiveKind::ActivateWindowOrApp => match validate_activation_target(request) {
                Ok(target) => Some(target),
                Err(error) => {
                    return primitive_result(
                        request.primitive.clone(),
                        UIPrimitiveStatus::Failed,
                        &error,
                    )
                }
            },
            _ => None,
        };
        let pointer_geometry = pointer_target
            .as_ref()
            .map(|target| target.geometry.clone());

        if self.dry_run {
            return primitive_result_with_geometry(
                request.primitive.clone(),
                UIPrimitiveStatus::Executed,
                "Dry-run primitive execution accepted.",
                pointer_geometry.clone(),
            );
        }

        let result = match request.primitive {
            UIPrimitiveKind::TypeText => {
                let Some(value) = request.value.as_deref().filter(|value| !value.is_empty()) else {
                    return primitive_result(
                        request.primitive.clone(),
                        UIPrimitiveStatus::Failed,
                        "type_text requires a non-empty value.",
                    );
                };
                type_text_windows(value)
            }
            UIPrimitiveKind::PressEnter => send_keys_windows("{ENTER}"),
            UIPrimitiveKind::NavigateBack => send_keys_windows("%{LEFT}"),
            UIPrimitiveKind::ScrollViewport => Err(capability.platform_note.clone()),
            UIPrimitiveKind::FocusCurrentInput | UIPrimitiveKind::ClickTargetCandidate => {
                let Some(target) = pointer_target.as_ref() else {
                    return primitive_result(
                        request.primitive.clone(),
                        UIPrimitiveStatus::Failed,
                        "Pointer primitive requires a validated target candidate.",
                    );
                };
                click_windows(target.x, target.y)
            }
            UIPrimitiveKind::ActivateWindowOrApp => {
                let Some(target) = activation_target.as_ref() else {
                    return primitive_result(
                        request.primitive.clone(),
                        UIPrimitiveStatus::Failed,
                        "ActivateWindowOrApp requires a validated browser activation target.",
                    );
                };
                activate_window_or_app(target)
            }
        };

        match result {
            Ok(()) => primitive_result_with_geometry(
                request.primitive.clone(),
                UIPrimitiveStatus::Executed,
                "Primitive executed by the platform backend.",
                pointer_geometry.clone(),
            ),
            Err(error) => primitive_result_with_geometry(
                request.primitive.clone(),
                UIPrimitiveStatus::Failed,
                &error,
                pointer_geometry,
            ),
        }
    }
}

fn capability(
    primitive: UIPrimitiveKind,
    available: bool,
    enabled: bool,
    requires_screen_context: bool,
    requires_high_confidence_target: bool,
    requires_approval: bool,
    platform_note: &str,
) -> UIPrimitiveCapability {
    UIPrimitiveCapability {
        primitive,
        available,
        enabled,
        requires_screen_context,
        requires_high_confidence_target,
        requires_approval,
        platform_note: platform_note.to_string(),
    }
}

fn primitive_result(
    primitive: UIPrimitiveKind,
    status: UIPrimitiveStatus,
    message: &str,
) -> UIPrimitiveResult {
    primitive_result_with_geometry(primitive, status, message, None)
}

fn primitive_result_with_geometry(
    primitive: UIPrimitiveKind,
    status: UIPrimitiveStatus,
    message: &str,
    geometry: Option<ExecutableGeometryDiagnostic>,
) -> UIPrimitiveResult {
    UIPrimitiveResult {
        primitive,
        status,
        message: message.to_string(),
        geometry,
    }
}

struct PointerTarget {
    x: i32,
    y: i32,
    geometry: ExecutableGeometryDiagnostic,
}

struct PointerTargetValidationFailure {
    message: String,
    geometry: Option<ExecutableGeometryDiagnostic>,
}

struct ActivationTarget {
    app_hint: String,
    process_names: Vec<&'static str>,
}

fn validate_activation_target(request: &UIPrimitiveRequest) -> Result<ActivationTarget, String> {
    let requested = request
        .target
        .get("app")
        .or_else(|| request.target.get("browser_app"))
        .or_else(|| request.target.get("browser_app_hint"))
        .or_else(|| request.target.get("application"))
        .and_then(Value::as_str)
        .unwrap_or("browser");
    let normalized = normalize_app_hint(requested);
    let process_names = match normalized.as_str() {
        "chrome" => vec!["chrome"],
        "edge" => vec!["msedge"],
        "firefox" => vec!["firefox"],
        "browser" => vec!["chrome", "msedge", "firefox", "brave", "opera", "vivaldi"],
        _ => {
            return Err(
                "ActivateWindowOrApp only supports bounded known-browser activation targets."
                    .into(),
            )
        }
    };
    Ok(ActivationTarget {
        app_hint: normalized,
        process_names,
    })
}

fn normalize_app_hint(value: &str) -> String {
    match value
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "_")
        .as_str()
    {
        "google_chrome" | "chrome_browser" => "chrome".into(),
        "microsoft_edge" | "ms_edge" | "msedge" => "edge".into(),
        "mozilla_firefox" | "firefox_browser" => "firefox".into(),
        "default_browser" | "web_browser" | "browser_app" => "browser".into(),
        other => other.to_string(),
    }
}

fn validate_pointer_target(
    request: &UIPrimitiveRequest,
    allow_live_environment_bounds: bool,
) -> Result<PointerTarget, PointerTargetValidationFailure> {
    let candidate = request.target.get("candidate").unwrap_or(&request.target);
    let accessibility_sourced = is_accessibility_sourced_candidate(candidate);
    let confidence = candidate
        .get("confidence")
        .and_then(Value::as_f64)
        .ok_or_else(|| PointerTargetValidationFailure {
            message: "Pointer primitive requires target confidence.".into(),
            geometry: None,
        })?;
    let required_confidence = match request.primitive {
        UIPrimitiveKind::FocusCurrentInput => 0.78,
        UIPrimitiveKind::ClickTargetCandidate => 0.86,
        _ => 1.0,
    };
    if confidence < required_confidence {
        return Err(PointerTargetValidationFailure {
            message: format!(
                "Pointer target confidence {:.2} is below required {:.2}.",
                confidence, required_confidence
            ),
            geometry: None,
        });
    }

    match request.primitive {
        UIPrimitiveKind::FocusCurrentInput => {
            if candidate
                .get("supports_focus")
                .and_then(Value::as_bool)
                .is_some_and(|supports| !supports)
            {
                return Err(PointerTargetValidationFailure {
                    message: "Target candidate is not marked focusable.".into(),
                    geometry: None,
                });
            }
        }
        UIPrimitiveKind::ClickTargetCandidate => {
            if candidate
                .get("supports_click")
                .and_then(Value::as_bool)
                .is_some_and(|supports| !supports)
            {
                return Err(PointerTargetValidationFailure {
                    message: "Target candidate is not marked clickable.".into(),
                    geometry: None,
                });
            }
        }
        _ => {}
    }

    let raw_region = extract_target_region(candidate);
    let point = extract_center_point(candidate)
        .or_else(|| {
            request
                .target
                .get("candidate")
                .and_then(extract_center_point)
        })
        .ok_or_else(|| PointerTargetValidationFailure {
            message: "Pointer primitive requires target center coordinates or region.".into(),
            geometry: Some(rejected_geometry(
                raw_region.clone(),
                None,
                None,
                None,
                accessibility_sourced,
                ExecutableCoordinateInterpretation::RejectedUntrustedGeometry,
                "pointer target did not contain usable center coordinates or region geometry",
            )),
        })?;
    let browser_window_bounds = if accessibility_sourced {
        None
    } else {
        extract_browser_window_bounds(request, candidate).or_else(|| {
            allow_live_environment_bounds
                .then(|| browser_window_bounds_for_candidate(candidate))
                .flatten()
        })
    };
    let screen_bounds = extract_screen_bounds(request, candidate).or_else(|| {
        allow_live_environment_bounds
            .then(query_virtual_screen_bounds_windows)
            .flatten()
    });
    let coordinate_space = raw_region
        .as_ref()
        .map(|region| normalize_coordinate_space(&region.coordinate_space))
        .or_else(|| {
            candidate
                .get("coordinate_space")
                .and_then(Value::as_str)
                .map(normalize_coordinate_space)
        })
        .unwrap_or_else(|| "screen".into());

    let mut interpreted_region = raw_region.clone();
    let mut final_point = point;
    let mut interpretation = ExecutableCoordinateInterpretation::ScreenValidated;
    let mut translation_applied = false;

    match coordinate_space.as_str() {
        "screen" => {
            if let Some(bounds) = browser_window_bounds.as_ref() {
                if !point_within_region(point.0, point.1, bounds) {
                    let (interpretation, reason) = if point_looks_window_relative(point, bounds) {
                        (
                            ExecutableCoordinateInterpretation::RejectedSuspiciousGeometry,
                            "screen-space pointer target lies outside the expected browser surface and resembles copied or viewport-relative geometry; automatic translation is disabled for mislabeled screen coordinates",
                        )
                    } else {
                        (
                            ExecutableCoordinateInterpretation::RejectedOutsideBrowserSurface,
                            "pointer target lies outside the expected browser interaction surface",
                        )
                    };
                    let geometry = rejected_geometry(
                        raw_region,
                        browser_window_bounds,
                        screen_bounds,
                        Some(coordinate_space),
                        accessibility_sourced,
                        interpretation,
                        reason,
                    );
                    return Err(PointerTargetValidationFailure {
                        message: geometry.reason.clone().unwrap_or_else(|| {
                            "pointer target failed browser-surface validation".into()
                        }),
                        geometry: Some(geometry),
                    });
                }
            }
        }
        "window" | "window_relative" | "browser_window" | "browser" | "viewport"
        | "viewport_relative" | "content" | "content_relative" => {
            let Some(bounds) = browser_window_bounds.as_ref() else {
                let geometry = rejected_geometry(
                    raw_region,
                    browser_window_bounds,
                    screen_bounds,
                    Some(coordinate_space),
                    accessibility_sourced,
                    ExecutableCoordinateInterpretation::RejectedUntrustedGeometry,
                    "pointer target uses window-relative coordinates but browser window bounds are unavailable",
                );
                return Err(PointerTargetValidationFailure {
                    message: geometry
                        .reason
                        .clone()
                        .unwrap_or_else(|| "pointer target requires browser window bounds".into()),
                    geometry: Some(geometry),
                });
            };
            if !point_looks_window_relative(point, bounds) {
                let geometry = rejected_geometry(
                    raw_region,
                    browser_window_bounds,
                    screen_bounds,
                    Some(coordinate_space),
                    accessibility_sourced,
                    ExecutableCoordinateInterpretation::RejectedOutsideBrowserSurface,
                    "window-relative pointer target lies outside the browser window extent",
                );
                return Err(PointerTargetValidationFailure {
                    message: geometry.reason.clone().unwrap_or_else(|| {
                        "pointer target lies outside the bounded browser window extent".into()
                    }),
                    geometry: Some(geometry),
                });
            }
            final_point = translate_point(point, bounds);
            interpreted_region = raw_region
                .as_ref()
                .map(|region| translate_region(region, bounds));
            translation_applied = true;
            interpretation = ExecutableCoordinateInterpretation::WindowRelativeTranslated;
        }
        "normalized" | "percentage" | "percent" => {
            let geometry = rejected_geometry(
                raw_region,
                browser_window_bounds,
                screen_bounds,
                Some(coordinate_space),
                accessibility_sourced,
                ExecutableCoordinateInterpretation::RejectedUnsupportedCoordinateSpace,
                "normalized pointer coordinates are not safely executable in this runtime",
            );
            return Err(PointerTargetValidationFailure {
                message: geometry.reason.clone().unwrap_or_else(|| {
                    "pointer target used an unsupported coordinate space".into()
                }),
                geometry: Some(geometry),
            });
        }
        _ => {
            let geometry = rejected_geometry(
                raw_region,
                browser_window_bounds,
                screen_bounds,
                Some(coordinate_space),
                accessibility_sourced,
                ExecutableCoordinateInterpretation::RejectedUnsupportedCoordinateSpace,
                "pointer target used an unsupported coordinate space",
            );
            return Err(PointerTargetValidationFailure {
                message: geometry.reason.clone().unwrap_or_else(|| {
                    "pointer target used an unsupported coordinate space".into()
                }),
                geometry: Some(geometry),
            });
        }
    }

    if let Some(bounds) = screen_bounds.as_ref() {
        if !point_within_region(final_point.0, final_point.1, bounds) {
            let geometry = rejected_geometry(
                interpreted_region.or(raw_region),
                browser_window_bounds,
                screen_bounds,
                Some(coordinate_space),
                accessibility_sourced,
                ExecutableCoordinateInterpretation::RejectedOutsideScreenBounds,
                "pointer target lies outside the current virtual screen bounds",
            );
            return Err(PointerTargetValidationFailure {
                message: geometry.reason.clone().unwrap_or_else(|| {
                    "pointer target lies outside the current screen bounds".into()
                }),
                geometry: Some(geometry),
            });
        }
    }

    let final_x = rounded_screen_coordinate(final_point.0).map_err(|message| {
        PointerTargetValidationFailure {
            message: message.clone(),
            geometry: Some(rejected_geometry(
                interpreted_region.clone().or(raw_region.clone()),
                browser_window_bounds.clone(),
                screen_bounds.clone(),
                Some(coordinate_space.clone()),
                accessibility_sourced,
                ExecutableCoordinateInterpretation::RejectedUntrustedGeometry,
                &message,
            )),
        }
    })?;
    let final_y = rounded_screen_coordinate(final_point.1).map_err(|message| {
        PointerTargetValidationFailure {
            message: message.clone(),
            geometry: Some(rejected_geometry(
                interpreted_region.clone().or(raw_region.clone()),
                browser_window_bounds.clone(),
                screen_bounds.clone(),
                Some(coordinate_space.clone()),
                accessibility_sourced,
                ExecutableCoordinateInterpretation::RejectedUntrustedGeometry,
                &message,
            )),
        }
    })?;

    Ok(PointerTarget {
        x: final_x,
        y: final_y,
        geometry: ExecutableGeometryDiagnostic {
            raw_region,
            interpreted_region,
            raw_coordinate_space: Some(coordinate_space),
            interpretation,
            validation_passed: true,
            translation_applied,
            accessibility_sourced,
            screen_bounds,
            browser_window_bounds,
            final_x: Some(final_x),
            final_y: Some(final_y),
            reason: Some(if translation_applied {
                "pointer target was translated through bounded browser window geometry".into()
            } else {
                "pointer target was validated against the expected interaction surface".into()
            }),
        },
    })
}

fn is_accessibility_sourced_candidate(candidate: &Value) -> bool {
    candidate
        .get("accessibility_sourced")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || candidate
            .get("source")
            .and_then(Value::as_str)
            .is_some_and(|source| source == "accessibility_layer")
        || candidate
            .get("observation_source")
            .and_then(Value::as_str)
            .is_some_and(|source| source == "uia_snapshot")
        || candidate
            .get("element_id")
            .and_then(Value::as_str)
            .map(|id| id.starts_with("a11y_"))
            .unwrap_or(false)
}

fn extract_center_point(value: &Value) -> Option<(f64, f64)> {
    let direct = value
        .get("center_x")
        .and_then(Value::as_f64)
        .zip(value.get("center_y").and_then(Value::as_f64));
    if direct.is_some() {
        return direct;
    }

    let center = value
        .pointer("/center/x")
        .and_then(Value::as_f64)
        .zip(value.pointer("/center/y").and_then(Value::as_f64));
    if center.is_some() {
        return center;
    }

    extract_target_region(value).map(|region| region.center())
}

fn extract_target_region(value: &Value) -> Option<TargetRegion> {
    let region = value
        .get("region")
        .or_else(|| value.get("bounds"))
        .or_else(|| value.get("bounding_region"))
        .unwrap_or(value);
    let x = region
        .get("x")
        .or_else(|| region.get("left"))
        .and_then(Value::as_f64)?;
    let y = region
        .get("y")
        .or_else(|| region.get("top"))
        .and_then(Value::as_f64)?;
    let width = region.get("width").and_then(Value::as_f64).or_else(|| {
        let right = region.get("right")?.as_f64()?;
        Some(right - x)
    })?;
    let height = region.get("height").and_then(Value::as_f64).or_else(|| {
        let bottom = region.get("bottom")?.as_f64()?;
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
        coordinate_space: region
            .get("coordinate_space")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("screen")
            .to_string(),
    })
}

fn normalize_coordinate_space(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "_")
}

fn extract_screen_bounds(request: &UIPrimitiveRequest, candidate: &Value) -> Option<TargetRegion> {
    request
        .target
        .get("screen_bounds")
        .and_then(extract_target_region)
        .or_else(|| {
            candidate
                .get("screen_bounds")
                .and_then(extract_target_region)
        })
}

fn extract_browser_window_bounds(
    request: &UIPrimitiveRequest,
    candidate: &Value,
) -> Option<TargetRegion> {
    request
        .target
        .get("browser_window_bounds")
        .or_else(|| request.target.get("browser_bounds"))
        .and_then(extract_target_region)
        .or_else(|| {
            candidate
                .get("browser_window_bounds")
                .or_else(|| candidate.get("browser_bounds"))
                .and_then(extract_target_region)
        })
}

fn browser_window_bounds_for_candidate(candidate: &Value) -> Option<TargetRegion> {
    let app_hint = candidate
        .get("browser_app_hint")
        .or_else(|| candidate.get("app"))
        .or_else(|| candidate.get("app_hint"))
        .and_then(Value::as_str)?;
    let activation_target = activation_target_for_app_hint(app_hint)?;
    query_browser_window_bounds_windows(&activation_target)
}

fn activation_target_for_app_hint(app_hint: &str) -> Option<ActivationTarget> {
    let normalized = normalize_app_hint(app_hint);
    let process_names = match normalized.as_str() {
        "chrome" => vec!["chrome"],
        "edge" => vec!["msedge"],
        "firefox" => vec!["firefox"],
        "browser" => vec!["chrome", "msedge", "firefox", "brave", "opera", "vivaldi"],
        _ => return None,
    };
    Some(ActivationTarget {
        app_hint: normalized,
        process_names,
    })
}

fn point_within_region(x: f64, y: f64, region: &TargetRegion) -> bool {
    x >= region.x && y >= region.y && x <= region.x + region.width && y <= region.y + region.height
}

fn point_looks_window_relative(point: (f64, f64), browser_bounds: &TargetRegion) -> bool {
    point.0 >= 0.0
        && point.1 >= 0.0
        && point.0 <= browser_bounds.width
        && point.1 <= browser_bounds.height
}

fn translate_point(point: (f64, f64), browser_bounds: &TargetRegion) -> (f64, f64) {
    (browser_bounds.x + point.0, browser_bounds.y + point.1)
}

fn translate_region(region: &TargetRegion, browser_bounds: &TargetRegion) -> TargetRegion {
    TargetRegion {
        x: browser_bounds.x + region.x,
        y: browser_bounds.y + region.y,
        width: region.width,
        height: region.height,
        coordinate_space: "screen".into(),
    }
}

fn rejected_geometry(
    raw_region: Option<TargetRegion>,
    browser_window_bounds: Option<TargetRegion>,
    screen_bounds: Option<TargetRegion>,
    raw_coordinate_space: Option<String>,
    accessibility_sourced: bool,
    interpretation: ExecutableCoordinateInterpretation,
    reason: &str,
) -> ExecutableGeometryDiagnostic {
    ExecutableGeometryDiagnostic {
        raw_region,
        interpreted_region: None,
        raw_coordinate_space,
        interpretation,
        validation_passed: false,
        translation_applied: false,
        accessibility_sourced,
        screen_bounds,
        browser_window_bounds,
        final_x: None,
        final_y: None,
        reason: Some(reason.into()),
    }
}

fn rounded_screen_coordinate(value: f64) -> Result<i32, String> {
    if !value.is_finite() {
        return Err("Pointer coordinate is not finite.".into());
    }
    if value < i32::MIN as f64 || value > i32::MAX as f64 {
        return Err("Pointer coordinate is outside supported range.".into());
    }
    Ok(value.round() as i32)
}

fn activate_window_or_app(target: &ActivationTarget) -> Result<(), String> {
    if !cfg!(target_os = "windows") {
        return Err("ActivateWindowOrApp is only implemented for Windows browser windows.".into());
    }
    activate_window_or_app_windows(target)
}

#[cfg(target_os = "windows")]
fn activate_window_or_app_windows(target: &ActivationTarget) -> Result<(), String> {
    let names = target
        .process_names
        .iter()
        .map(|name| format!("'{}'", powershell_single_quoted(name)))
        .collect::<Vec<_>>()
        .join(",");
    let script = format!(
        r#"
$names = @({names})
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class AstraWin32 {{
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
}}
"@
$foregroundHandle = [AstraWin32]::GetForegroundWindow()
[uint32]$fgPid = 0
[AstraWin32]::GetWindowThreadProcessId($foregroundHandle, [ref]$fgPid) | Out-Null
$fgProcess = if ($fgPid -gt 0) {{
  Get-Process -Id ([int]$fgPid) -ErrorAction SilentlyContinue
}} else {{
  $null
}}
$activationSource = "fallback_recent_browser"
$candidate = if (
  $fgProcess -ne $null `
  -and $names -contains $fgProcess.ProcessName `
  -and $fgProcess.MainWindowHandle -ne 0
) {{
  $activationSource = "foreground_browser"
  $fgProcess
}} else {{
  Get-Process | Where-Object {{
    $names -contains $_.ProcessName -and $_.MainWindowHandle -ne 0
  }} | Sort-Object StartTime -Descending | Select-Object -First 1
}}
if ($null -eq $candidate) {{
  Write-Error "No matching browser window found."
  exit 2
}}
[AstraWin32]::ShowWindowAsync($candidate.MainWindowHandle, 9) | Out-Null
Start-Sleep -Milliseconds 80
if ([AstraWin32]::SetForegroundWindow($candidate.MainWindowHandle)) {{
  Write-Output ("Activated " + $candidate.ProcessName + " via " + $activationSource)
  exit 0
}}
Write-Error "SetForegroundWindow returned false."
exit 3
"#
    );
    run_powershell_script(&script).map_err(|error| {
        format!(
            "Browser activation for {} failed through bounded ActivateWindowOrApp: {error}",
            target.app_hint
        )
    })
}

#[cfg(not(target_os = "windows"))]
fn activate_window_or_app_windows(_target: &ActivationTarget) -> Result<(), String> {
    Err("ActivateWindowOrApp is only implemented for Windows browser windows.".into())
}

fn query_virtual_screen_bounds_windows() -> Option<TargetRegion> {
    if !cfg!(target_os = "windows") {
        return None;
    }
    let script = r#"
Add-Type -AssemblyName System.Windows.Forms
$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen
[ordered]@{
  x = $bounds.X
  y = $bounds.Y
  width = $bounds.Width
  height = $bounds.Height
  coordinate_space = "screen"
} | ConvertTo-Json -Compress
"#;
    run_powershell_script_capture_stdout(script)
        .ok()
        .and_then(|stdout| serde_json::from_str::<Value>(&stdout).ok())
        .and_then(|value| extract_target_region(&value))
}

fn query_browser_window_bounds_windows(target: &ActivationTarget) -> Option<TargetRegion> {
    if !cfg!(target_os = "windows") {
        return None;
    }
    let names = target
        .process_names
        .iter()
        .map(|name| format!("'{}'", powershell_single_quoted(name)))
        .collect::<Vec<_>>()
        .join(",");
    let script = format!(
        r#"
$names = @({names})
Add-Type @"
using System;
using System.Runtime.InteropServices;
public struct RECT {{
  public int Left;
  public int Top;
  public int Right;
  public int Bottom;
}}
public static class AstraWin32 {{
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
}}
"@
$candidate = Get-Process | Where-Object {{
  $names -contains $_.ProcessName -and $_.MainWindowHandle -ne 0
}} | Sort-Object StartTime -Descending | Select-Object -First 1
if ($null -eq $candidate) {{
  exit 0
}}
$rect = New-Object RECT
if (-not [AstraWin32]::GetWindowRect($candidate.MainWindowHandle, [ref]$rect)) {{
  exit 0
}}
[ordered]@{{
  x = $rect.Left
  y = $rect.Top
  width = ($rect.Right - $rect.Left)
  height = ($rect.Bottom - $rect.Top)
  coordinate_space = "screen"
}} | ConvertTo-Json -Compress
"#
    );
    run_powershell_script_capture_stdout(&script)
        .ok()
        .and_then(|stdout| {
            let trimmed = stdout.trim();
            (!trimmed.is_empty()).then_some(trimmed.to_string())
        })
        .and_then(|stdout| serde_json::from_str::<Value>(&stdout).ok())
        .and_then(|value| extract_target_region(&value))
}

fn type_text_windows(value: &str) -> Result<(), String> {
    if !cfg!(target_os = "windows") {
        return Err("type_text is currently implemented only on Windows.".into());
    }

    let value = powershell_single_quoted(value);
    let script = format!(
        concat!(
            "Add-Type -AssemblyName System.Windows.Forms; ",
            "$oldText = $null; $hadText = $false; ",
            "try {{ $oldText = [System.Windows.Forms.Clipboard]::GetText(); $hadText = $true; }} catch {{}} ",
            "[System.Windows.Forms.Clipboard]::SetText('{}'); ",
            "$wshell = New-Object -ComObject WScript.Shell; ",
            "Start-Sleep -Milliseconds 80; ",
            "$wshell.SendKeys('^v'); ",
            "Start-Sleep -Milliseconds 80; ",
            "if ($hadText) {{ [System.Windows.Forms.Clipboard]::SetText($oldText); }}"
        ),
        value
    );
    run_powershell_script(&script)
}

fn click_windows(x: i32, y: i32) -> Result<(), String> {
    if !cfg!(target_os = "windows") {
        return Err("click target primitive is currently implemented only on Windows.".into());
    }

    let script = format!(
        concat!(
            "Add-Type -TypeDefinition '",
            "using System; ",
            "using System.Runtime.InteropServices; ",
            "public class AstraMouse {{ ",
            "[DllImport(\"user32.dll\")] public static extern bool SetCursorPos(int X, int Y); ",
            "[DllImport(\"user32.dll\")] public static extern void mouse_event(int dwFlags, int dx, int dy, int dwData, int dwExtraInfo); ",
            "}}'; ",
            "[AstraMouse]::SetCursorPos({x}, {y}) | Out-Null; ",
            "Start-Sleep -Milliseconds 80; ",
            "[AstraMouse]::mouse_event(0x0002, {x}, {y}, 0, 0); ",
            "Start-Sleep -Milliseconds 40; ",
            "[AstraMouse]::mouse_event(0x0004, {x}, {y}, 0, 0); ",
            "Start-Sleep -Milliseconds 120;"
        ),
        x = x,
        y = y
    );
    run_powershell_script(&script)
}

fn send_keys_windows(sequence: &str) -> Result<(), String> {
    if !cfg!(target_os = "windows") {
        return Err("send_keys primitives are currently implemented only on Windows.".into());
    }

    let sequence = powershell_single_quoted(sequence);
    let script = format!(
        concat!(
            "$wshell = New-Object -ComObject WScript.Shell; ",
            "Start-Sleep -Milliseconds 80; ",
            "$wshell.SendKeys('{}'); ",
            "Start-Sleep -Milliseconds 80;"
        ),
        sequence
    );
    run_powershell_script(&script)
}

fn run_powershell_script(script: &str) -> Result<(), String> {
    run_powershell_output(script).map(|_| ())
}

fn run_powershell_script_capture_stdout(script: &str) -> Result<String, String> {
    run_powershell_output(script)
}

fn run_powershell_output(script: &str) -> Result<String, String> {
    let encoded = encode_powershell_command(script);
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-EncodedCommand",
            &encoded,
        ])
        .output()
        .map_err(|error| format!("ui primitive PowerShell backend failed: {error}"))?;

    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Err(if stderr.is_empty() {
        format!(
            "ui primitive PowerShell backend exited with status {:?}",
            output.status.code()
        )
    } else {
        stderr
    })
}

fn encode_powershell_command(script: &str) -> String {
    let mut bytes = Vec::with_capacity(script.len() * 2);
    for unit in script.encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    BASE64_STANDARD.encode(bytes)
}

fn powershell_single_quoted(value: &str) -> String {
    value.replace('\'', "''")
}

fn platform_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_desktop_control_disables_keyboard_primitives() {
        let caps = UIPrimitiveCapabilitySet::for_runtime(false);
        let type_text = caps.get(&UIPrimitiveKind::TypeText).expect("type_text");

        assert!(!type_text.enabled);
    }

    #[test]
    fn dry_run_executes_supported_primitive() {
        let runtime = UIControlRuntime::dry_run();
        let caps = UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![capability(
                UIPrimitiveKind::TypeText,
                true,
                true,
                false,
                false,
                false,
                "test",
            )],
        };
        let result = runtime.execute(
            &UIPrimitiveRequest {
                primitive: UIPrimitiveKind::TypeText,
                value: Some("hello".into()),
                target: serde_json::json!({}),
                reason: None,
            },
            &caps,
        );

        assert_eq!(result.status, UIPrimitiveStatus::Executed);
    }

    #[test]
    fn dry_run_validates_high_confidence_pointer_target() {
        let runtime = UIControlRuntime::dry_run();
        let caps = UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![capability(
                UIPrimitiveKind::ClickTargetCandidate,
                true,
                true,
                true,
                true,
                false,
                "test",
            )],
        };
        let result = runtime.execute(
            &UIPrimitiveRequest {
                primitive: UIPrimitiveKind::ClickTargetCandidate,
                value: None,
                target: serde_json::json!({
                    "candidate": {
                        "center_x": 320,
                        "center_y": 240,
                        "confidence": 0.93,
                        "supports_click": true
                    }
                }),
                reason: None,
            },
            &caps,
        );

        assert_eq!(result.status, UIPrimitiveStatus::Executed);
    }

    #[test]
    fn dry_run_accepts_bounded_browser_activation_target() {
        let runtime = UIControlRuntime::dry_run();
        let caps = UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![capability(
                UIPrimitiveKind::ActivateWindowOrApp,
                true,
                true,
                true,
                false,
                false,
                "test",
            )],
        };
        let result = runtime.execute(
            &UIPrimitiveRequest {
                primitive: UIPrimitiveKind::ActivateWindowOrApp,
                value: None,
                target: serde_json::json!({"app": "google chrome", "provider": "youtube"}),
                reason: None,
            },
            &caps,
        );

        assert_eq!(result.status, UIPrimitiveStatus::Executed);
    }

    #[test]
    fn activation_rejects_unbounded_app_targets_even_in_dry_run() {
        let runtime = UIControlRuntime::dry_run();
        let caps = UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![capability(
                UIPrimitiveKind::ActivateWindowOrApp,
                true,
                true,
                true,
                false,
                false,
                "test",
            )],
        };
        let result = runtime.execute(
            &UIPrimitiveRequest {
                primitive: UIPrimitiveKind::ActivateWindowOrApp,
                value: None,
                target: serde_json::json!({"app": "notepad"}),
                reason: None,
            },
            &caps,
        );

        assert_eq!(result.status, UIPrimitiveStatus::Failed);
    }

    #[test]
    fn pointer_target_rejects_low_confidence_even_in_dry_run() {
        let runtime = UIControlRuntime::dry_run();
        let caps = UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![capability(
                UIPrimitiveKind::ClickTargetCandidate,
                true,
                true,
                true,
                true,
                false,
                "test",
            )],
        };
        let result = runtime.execute(
            &UIPrimitiveRequest {
                primitive: UIPrimitiveKind::ClickTargetCandidate,
                value: None,
                target: serde_json::json!({
                    "candidate": {
                        "center_x": 320,
                        "center_y": 240,
                        "confidence": 0.40,
                        "supports_click": true
                    }
                }),
                reason: None,
            },
            &caps,
        );

        assert_eq!(result.status, UIPrimitiveStatus::Failed);
    }

    #[test]
    fn pointer_target_translates_window_relative_geometry_when_browser_bounds_are_known() {
        let runtime = UIControlRuntime::dry_run();
        let caps = UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![capability(
                UIPrimitiveKind::ClickTargetCandidate,
                true,
                true,
                true,
                true,
                false,
                "test",
            )],
        };
        let result = runtime.execute(
            &UIPrimitiveRequest {
                primitive: UIPrimitiveKind::ClickTargetCandidate,
                value: None,
                target: serde_json::json!({
                    "browser_window_bounds": {
                        "x": 900,
                        "y": 120,
                        "width": 1000,
                        "height": 800,
                        "coordinate_space": "screen"
                    },
                    "screen_bounds": {
                        "x": 0,
                        "y": 0,
                        "width": 1920,
                        "height": 1080,
                        "coordinate_space": "screen"
                    },
                    "candidate": {
                        "region": {
                            "x": 120,
                            "y": 160,
                            "width": 400,
                            "height": 60,
                            "coordinate_space": "window"
                        },
                        "confidence": 0.93,
                        "supports_click": true
                    }
                }),
                reason: None,
            },
            &caps,
        );

        assert_eq!(result.status, UIPrimitiveStatus::Executed);
        let geometry = result.geometry.expect("geometry");
        assert!(geometry.translation_applied);
        assert_eq!(
            geometry.interpretation,
            ExecutableCoordinateInterpretation::WindowRelativeTranslated
        );
        assert_eq!(geometry.final_x, Some(1220));
        assert_eq!(geometry.final_y, Some(310));
    }

    #[test]
    fn pointer_target_rejects_screen_geometry_outside_known_browser_surface() {
        let runtime = UIControlRuntime::dry_run();
        let caps = UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![capability(
                UIPrimitiveKind::ClickTargetCandidate,
                true,
                true,
                true,
                true,
                false,
                "test",
            )],
        };
        let result = runtime.execute(
            &UIPrimitiveRequest {
                primitive: UIPrimitiveKind::ClickTargetCandidate,
                value: None,
                target: serde_json::json!({
                    "browser_window_bounds": {
                        "x": 900,
                        "y": 120,
                        "width": 1000,
                        "height": 800,
                        "coordinate_space": "screen"
                    },
                    "screen_bounds": {
                        "x": 0,
                        "y": 0,
                        "width": 1920,
                        "height": 1080,
                        "coordinate_space": "screen"
                    },
                    "candidate": {
                        "region": {
                            "x": 2000,
                            "y": 900,
                            "width": 400,
                            "height": 60,
                            "coordinate_space": "screen"
                        },
                        "confidence": 0.93,
                        "supports_click": true
                    }
                }),
                reason: None,
            },
            &caps,
        );

        assert_eq!(result.status, UIPrimitiveStatus::Failed);
        let geometry = result.geometry.expect("geometry");
        assert!(!geometry.validation_passed);
        assert_eq!(
            geometry.interpretation,
            ExecutableCoordinateInterpretation::RejectedOutsideBrowserSurface
        );
    }

    #[test]
    fn pointer_target_accepts_accessibility_screen_bounds_without_browser_translation() {
        let runtime = UIControlRuntime::dry_run();
        let caps = UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![capability(
                UIPrimitiveKind::ClickTargetCandidate,
                true,
                true,
                true,
                true,
                false,
                "test",
            )],
        };
        let result = runtime.execute(
            &UIPrimitiveRequest {
                primitive: UIPrimitiveKind::ClickTargetCandidate,
                value: None,
                target: serde_json::json!({
                    "browser_window_bounds": {
                        "x": 900,
                        "y": 120,
                        "width": 1000,
                        "height": 800,
                        "coordinate_space": "screen"
                    },
                    "screen_bounds": {
                        "x": 0,
                        "y": 0,
                        "width": 3440,
                        "height": 1440,
                        "coordinate_space": "screen"
                    },
                    "candidate": {
                        "region": {
                            "x": 2100,
                            "y": 900,
                            "width": 400,
                            "height": 60,
                            "coordinate_space": "screen"
                        },
                        "confidence": 0.93,
                        "supports_click": true,
                        "accessibility_sourced": true,
                        "observation_source": "uia_snapshot"
                    }
                }),
                reason: None,
            },
            &caps,
        );

        assert_eq!(result.status, UIPrimitiveStatus::Executed);
        let geometry = result.geometry.expect("geometry");
        assert!(geometry.validation_passed);
        assert!(geometry.accessibility_sourced);
        assert!(!geometry.translation_applied);
        assert_eq!(geometry.final_x, Some(2300));
        assert_eq!(geometry.final_y, Some(930));
    }

    #[test]
    fn is_accessibility_sourced_detects_via_element_id_field() {
        let candidate = serde_json::json!({
            "element_id": "a11y_3"
        });

        assert!(is_accessibility_sourced_candidate(&candidate));
    }

    #[test]
    fn pointer_target_rejects_unsupported_normalized_coordinate_space() {
        let runtime = UIControlRuntime::dry_run();
        let caps = UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![capability(
                UIPrimitiveKind::ClickTargetCandidate,
                true,
                true,
                true,
                true,
                false,
                "test",
            )],
        };
        let result = runtime.execute(
            &UIPrimitiveRequest {
                primitive: UIPrimitiveKind::ClickTargetCandidate,
                value: None,
                target: serde_json::json!({
                    "candidate": {
                        "region": {
                            "x": 0.2,
                            "y": 0.3,
                            "width": 0.1,
                            "height": 0.05,
                            "coordinate_space": "normalized"
                        },
                        "confidence": 0.93,
                        "supports_click": true
                    }
                }),
                reason: None,
            },
            &caps,
        );

        assert_eq!(result.status, UIPrimitiveStatus::Failed);
        let geometry = result.geometry.expect("geometry");
        assert_eq!(
            geometry.interpretation,
            ExecutableCoordinateInterpretation::RejectedUnsupportedCoordinateSpace
        );
    }
}
