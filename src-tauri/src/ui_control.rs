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
                    false,
                    false,
                    true,
                    true,
                    false,
                    "Window activation needs a reliable window/app target resolver first.",
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
                match validate_pointer_target(request) {
                    Ok(target) => Some(target),
                    Err(error) => {
                        return primitive_result(
                            request.primitive.clone(),
                            UIPrimitiveStatus::Failed,
                            &error,
                        )
                    }
                }
            }
            _ => None,
        };

        if self.dry_run {
            return primitive_result(
                request.primitive.clone(),
                UIPrimitiveStatus::Executed,
                "Dry-run primitive execution accepted.",
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
            UIPrimitiveKind::FocusCurrentInput | UIPrimitiveKind::ClickTargetCandidate => {
                let Some(target) = pointer_target else {
                    return primitive_result(
                        request.primitive.clone(),
                        UIPrimitiveStatus::Failed,
                        "Pointer primitive requires a validated target candidate.",
                    );
                };
                click_windows(target.x, target.y)
            }
            UIPrimitiveKind::ActivateWindowOrApp => Err(capability.platform_note.clone()),
        };

        match result {
            Ok(()) => primitive_result(
                request.primitive.clone(),
                UIPrimitiveStatus::Executed,
                "Primitive executed by the platform backend.",
            ),
            Err(error) => {
                primitive_result(request.primitive.clone(), UIPrimitiveStatus::Failed, &error)
            }
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
    UIPrimitiveResult {
        primitive,
        status,
        message: message.to_string(),
    }
}

struct PointerTarget {
    x: i32,
    y: i32,
}

fn validate_pointer_target(request: &UIPrimitiveRequest) -> Result<PointerTarget, String> {
    let candidate = request.target.get("candidate").unwrap_or(&request.target);
    let confidence = candidate
        .get("confidence")
        .and_then(Value::as_f64)
        .ok_or_else(|| "Pointer primitive requires target confidence.".to_string())?;
    let required_confidence = match request.primitive {
        UIPrimitiveKind::FocusCurrentInput => 0.78,
        UIPrimitiveKind::ClickTargetCandidate => 0.86,
        _ => 1.0,
    };
    if confidence < required_confidence {
        return Err(format!(
            "Pointer target confidence {:.2} is below required {:.2}.",
            confidence, required_confidence
        ));
    }

    match request.primitive {
        UIPrimitiveKind::FocusCurrentInput => {
            if candidate
                .get("supports_focus")
                .and_then(Value::as_bool)
                .is_some_and(|supports| !supports)
            {
                return Err("Target candidate is not marked focusable.".into());
            }
        }
        UIPrimitiveKind::ClickTargetCandidate => {
            if candidate
                .get("supports_click")
                .and_then(Value::as_bool)
                .is_some_and(|supports| !supports)
            {
                return Err("Target candidate is not marked clickable.".into());
            }
        }
        _ => {}
    }

    let point = extract_center_point(candidate)
        .or_else(|| {
            request
                .target
                .get("candidate")
                .and_then(extract_center_point)
        })
        .ok_or_else(|| {
            "Pointer primitive requires target center coordinates or region.".to_string()
        })?;

    Ok(PointerTarget {
        x: rounded_screen_coordinate(point.0)?,
        y: rounded_screen_coordinate(point.1)?,
    })
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

    let region = value.get("region").unwrap_or(value);
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

    Some((x + (width / 2.0), y + (height / 2.0)))
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
        return Ok(());
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
}
