use crate::ui_target_grounding::{
    TargetGroundingSource, TargetRegion, UITargetCandidate, UITargetRole,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub type ScreenRegion = TargetRegion;
static ACCESSIBILITY_SNAPSHOT_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibleElement {
    pub element_id: String,
    #[serde(default)]
    pub automation_id: Option<String>,
    #[serde(default)]
    pub runtime_id: Option<String>,
    pub role: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default)]
    pub bounding_rect: Option<ScreenRegion>,
    pub is_enabled: bool,
    pub is_offscreen: bool,
    pub depth: u32,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub children: Vec<AccessibleElement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilitySnapshot {
    #[serde(default = "next_accessibility_snapshot_id")]
    pub snapshot_id: String,
    #[serde(default)]
    pub elements: Vec<AccessibleElement>,
    #[serde(default)]
    pub browser_url: Option<String>,
    #[serde(default)]
    pub browser_window_bounds: Option<ScreenRegion>,
    pub captured_at_ms: u64,
    pub capture_backend: String,
    pub element_count: usize,
    #[serde(default)]
    pub window_is_foreground: bool,
    #[serde(default)]
    pub window_pid: Option<u32>,
    #[serde(default)]
    pub window_process_name: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

impl AccessibilitySnapshot {
    fn unavailable(error: impl Into<String>) -> Self {
        Self {
            snapshot_id: next_accessibility_snapshot_id(),
            elements: Vec::new(),
            browser_url: None,
            browser_window_bounds: None,
            captured_at_ms: now_ms(),
            capture_backend: "unavailable".into(),
            element_count: 0,
            window_is_foreground: false,
            window_pid: None,
            window_process_name: None,
            error: Some(error.into()),
        }
    }

    fn normalize_counts(mut self) -> Self {
        self.element_count = self.elements.len();
        self
    }
}

pub fn capture_accessibility_snapshot(process_names: &[&str]) -> AccessibilitySnapshot {
    capture_accessibility_snapshot_platform(process_names)
}

#[cfg(target_os = "windows")]
fn capture_accessibility_snapshot_platform(process_names: &[&str]) -> AccessibilitySnapshot {
    capture_accessibility_snapshot_windows(process_names)
}

#[cfg(not(target_os = "windows"))]
fn capture_accessibility_snapshot_platform(_process_names: &[&str]) -> AccessibilitySnapshot {
    AccessibilitySnapshot::unavailable("accessibility snapshot not implemented on this platform")
}

#[cfg(target_os = "windows")]
pub fn capture_accessibility_snapshot_windows(process_names: &[&str]) -> AccessibilitySnapshot {
    let names = process_names
        .iter()
        .map(|name| format!("'{}'", powershell_single_quoted(name)))
        .collect::<Vec<_>>()
        .join(",");
    let script = format!(
        r#"
$ErrorActionPreference = "SilentlyContinue"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$started = [Diagnostics.Stopwatch]::StartNew()
$names = @({names})
$maxDepth = 12
$maxElements = 200
$deadlineMs = 2500
$errorText = $null
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class AstraForeground {{
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(
        IntPtr hWnd,
        out uint processId
    );
}}
"@

function Convert-Rect($rect) {{
  if ($null -eq $rect -or $rect.IsEmpty) {{ return $null }}
  if ($rect.Width -le 0 -or $rect.Height -le 0) {{ return $null }}
  [ordered]@{{
    x = [double]$rect.X
    y = [double]$rect.Y
    width = [double]$rect.Width
    height = [double]$rect.Height
    coordinate_space = "screen"
  }}
}}

function Control-Role($element) {{
  try {{
    $programmatic = $element.Current.ControlType.ProgrammaticName
    if ([string]::IsNullOrWhiteSpace($programmatic)) {{ return "unknown" }}
    return ($programmatic -replace "^ControlType\.", "").ToLowerInvariant()
  }} catch {{
    return "unknown"
  }}
}}

function Runtime-Id($element) {{
  try {{
    $runtimeId = $element.GetRuntimeId()
    if ($null -eq $runtimeId) {{ return $null }}
    return ($runtimeId -join ",")
  }} catch {{
    return $null
  }}
}}

function Element-Value($element) {{
  try {{
    $pattern = $element.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
    if ($null -ne $pattern) {{ return $pattern.Current.Value }}
  }} catch {{}}
  return $null
}}

function Include-Element($element, $role, $name, $automationId) {{
  if ($element.Current.IsOffscreen -or -not $element.Current.IsEnabled) {{ return $false }}
  if ([string]::IsNullOrWhiteSpace($name) -and [string]::IsNullOrWhiteSpace($automationId)) {{ return $false }}
  switch ($role) {{
    "hyperlink" {{ return $true }}
    "button" {{ return $true }}
    "listitem" {{ return $true }}
    "heading" {{ return $true }}
    "text" {{ return -not [string]::IsNullOrWhiteSpace($name) }}
    "image" {{ return -not [string]::IsNullOrWhiteSpace($name) }}
    "combobox" {{ return $true }}
    "checkbox" {{ return $true }}
    "edit" {{ return $true }}
    "menuitem" {{ return $true }}
    "tabitem" {{ return $true }}
    default {{ return $false }}
  }}
}}

$root = [System.Windows.Automation.AutomationElement]::RootElement
$foregroundHandle = [AstraForeground]::GetForegroundWindow()
[uint32]$fgPid = 0
[AstraForeground]::GetWindowThreadProcessId($foregroundHandle, [ref]$fgPid) | Out-Null
$fgProcess = if ($fgPid -gt 0) {{
  Get-Process -Id ([int]$fgPid) -ErrorAction SilentlyContinue
}} else {{
  $null
}}
$windowIsForeground = $false
$process = if (
  $fgProcess -ne $null `
  -and $names -contains $fgProcess.ProcessName `
  -and $fgProcess.MainWindowHandle -ne 0
) {{
  $windowIsForeground = $true
  $fgProcess
}} else {{
  $windowIsForeground = $false
  Get-Process | Where-Object {{
    $names -contains $_.ProcessName -and $_.MainWindowHandle -ne 0
  }} | Sort-Object StartTime -Descending | Select-Object -First 1
}}

if ($null -eq $process) {{
  [ordered]@{{
    elements = @()
    browser_url = $null
    browser_window_bounds = $null
    window_is_foreground = $false
    window_pid = $null
    window_process_name = $null
    capture_backend = "powershell_uia"
    error = "no matching browser window found"
  }} | ConvertTo-Json -Depth 8 -Compress
  exit 0
}}

$windowCondition = [System.Windows.Automation.PropertyCondition]::new(
  [System.Windows.Automation.AutomationElement]::NativeWindowHandleProperty,
  [int]$process.MainWindowHandle
)
$window = $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $windowCondition)
if ($null -eq $window) {{
  [ordered]@{{
    elements = @()
    browser_url = $null
    browser_window_bounds = $null
    window_is_foreground = [bool]$windowIsForeground
    window_pid = [uint32]$process.Id
    window_process_name = $process.ProcessName
    capture_backend = "powershell_uia"
    error = "browser automation window not found"
  }} | ConvertTo-Json -Depth 8 -Compress
  exit 0
}}

$browserBounds = Convert-Rect $window.Current.BoundingRectangle
$browserUrl = $null
try {{
  $editCondition = [System.Windows.Automation.PropertyCondition]::new(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::Edit
  )
  $edits = $window.FindAll([System.Windows.Automation.TreeScope]::Descendants, $editCondition)
  foreach ($edit in $edits) {{
    $editName = $edit.Current.Name
    $automationId = $edit.Current.AutomationId
    if ($editName -match "Address|Search|omnibox|indirizzo|adresse|direccion" -or $automationId -match "address|omnibox|url") {{
      $candidateUrl = Element-Value $edit
      if (-not [string]::IsNullOrWhiteSpace($candidateUrl)) {{
        $browserUrl = $candidateUrl
        break
      }}
    }}
  }}
}} catch {{}}

$content = $null
foreach ($controlType in @(
  [System.Windows.Automation.ControlType]::Document,
  [System.Windows.Automation.ControlType]::Pane
)) {{
  try {{
    $condition = [System.Windows.Automation.PropertyCondition]::new(
      [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
      $controlType
    )
    $candidate = $window.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
    if ($null -ne $candidate) {{
      $content = $candidate
      break
    }}
  }} catch {{}}
}}
if ($null -eq $content) {{ $content = $window }}

$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
$elements = New-Object System.Collections.ArrayList
$stack = New-Object System.Collections.Stack
$stack.Push([pscustomobject]@{{ Element = $content; Depth = 0; ParentId = $null }})
$truncated = $false

while ($stack.Count -gt 0) {{
  if ($started.ElapsedMilliseconds -gt $deadlineMs) {{
    $truncated = $true
    break
  }}
  if ($elements.Count -ge $maxElements) {{
    $truncated = $true
    break
  }}
  $entry = $stack.Pop()
  $element = $entry.Element
  $depth = [int]$entry.Depth
  $parentId = $entry.ParentId
  if ($depth -gt $maxDepth -or $null -eq $element) {{ continue }}

  $role = Control-Role $element
  $name = $element.Current.Name
  $automationId = $element.Current.AutomationId
  $currentId = $parentId
  if (Include-Element $element $role $name $automationId) {{
    $elementId = "a11y_" + $elements.Count
    $currentId = $elementId
    $cleanAutomationId = $null
    if (-not [string]::IsNullOrWhiteSpace($automationId)) {{ $cleanAutomationId = $automationId }}
    $cleanName = $null
    if (-not [string]::IsNullOrWhiteSpace($name)) {{ $cleanName = $name }}
    [void]$elements.Add([ordered]@{{
      element_id = $elementId
      automation_id = $cleanAutomationId
      runtime_id = Runtime-Id $element
      role = $role
      name = $cleanName
      value = Element-Value $element
      bounding_rect = Convert-Rect $element.Current.BoundingRectangle
      is_enabled = [bool]$element.Current.IsEnabled
      is_offscreen = [bool]$element.Current.IsOffscreen
      depth = [uint32]$depth
      parent_id = $parentId
      children = @()
    }})
  }}

  $children = New-Object System.Collections.ArrayList
  try {{
    $child = $walker.GetFirstChild($element)
    while ($null -ne $child) {{
      [void]$children.Add($child)
      $child = $walker.GetNextSibling($child)
    }}
  }} catch {{}}
  for ($i = $children.Count - 1; $i -ge 0; $i--) {{
    $stack.Push([pscustomobject]@{{ Element = $children[$i]; Depth = ($depth + 1); ParentId = $currentId }})
  }}
}}

if ($truncated) {{ $errorText = "truncated" }}
[ordered]@{{
  elements = @($elements)
  browser_url = $browserUrl
  browser_window_bounds = $browserBounds
  window_is_foreground = [bool]$windowIsForeground
  window_pid = [uint32]$process.Id
  window_process_name = $process.ProcessName
  capture_backend = "powershell_uia"
  error = $errorText
}} | ConvertTo-Json -Depth 8 -Compress
"#
    );

    match run_powershell_script_capture_stdout(&script) {
        Ok(stdout) => parse_snapshot_stdout(&stdout).unwrap_or_else(|error| {
            AccessibilitySnapshot::unavailable(format!(
                "accessibility snapshot parse failed: {error}"
            ))
        }),
        Err(error) => AccessibilitySnapshot::unavailable(error),
    }
}

pub fn candidate_from_accessible_element(
    element: &AccessibleElement,
    _browser_bounds: Option<&ScreenRegion>,
    snapshot_id: &str,
) -> Option<UITargetCandidate> {
    if element.is_offscreen || !element.is_enabled {
        return None;
    }
    let region = element.bounding_rect.clone()?;
    let (role, result_kind) = match normalize_role(&element.role).as_str() {
        "hyperlink" => (UITargetRole::Link, Some("link")),
        "listitem" => (UITargetRole::RankedResult, Some("generic")),
        "button" => (UITargetRole::Button, Some("button")),
        "heading" => (UITargetRole::Unknown, Some("heading")),
        "edit" => (UITargetRole::TextInput, None),
        _ => (UITargetRole::Unknown, None),
    };

    Some(UITargetCandidate {
        candidate_id: element.element_id.clone(),
        element_id: Some(element.element_id.clone()),
        accessibility_snapshot_id: Some(snapshot_id.to_string()),
        role,
        region: Some(region),
        center_x: None,
        center_y: None,
        app_hint: Some("browser".into()),
        browser_app_hint: Some("browser".into()),
        provider_hint: None,
        content_provider_hint: None,
        page_kind_hint: None,
        capture_backend: Some("powershell_uia".into()),
        observation_source: Some("uia_snapshot".into()),
        result_kind: result_kind.map(ToOwned::to_owned),
        confidence: 0.0,
        source: TargetGroundingSource::AccessibilityLayer,
        label: element
            .name
            .clone()
            .or_else(|| element.automation_id.clone()),
        rank: None,
        observed_at_ms: Some(now_ms()),
        reuse_eligible: false,
        supports_focus: matches!(normalize_role(&element.role).as_str(), "edit" | "combobox"),
        supports_click: !matches!(normalize_role(&element.role).as_str(), "edit"),
        rationale: "OS accessibility snapshot element with screen-space bounds".into(),
    })
}

pub fn synthesize_ranked_uia_result_candidates(
    snapshot: &AccessibilitySnapshot,
) -> Vec<UITargetCandidate> {
    let mut elements = snapshot
        .elements
        .iter()
        .filter(|element| element_is_rankable_result_candidate(element, snapshot))
        .collect::<Vec<_>>();

    elements.sort_by(|left, right| {
        let left_region = left.bounding_rect.as_ref().expect("filtered region");
        let right_region = right.bounding_rect.as_ref().expect("filtered region");
        left_region
            .y
            .partial_cmp(&right_region.y)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left_region
                    .x
                    .partial_cmp(&right_region.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.depth.cmp(&right.depth))
    });

    elements
        .into_iter()
        .enumerate()
        .filter_map(|(index, element)| {
            let mut candidate = candidate_from_accessible_element(
                element,
                snapshot.browser_window_bounds.as_ref(),
                &snapshot.snapshot_id,
            )?;
            candidate.role = match normalize_role(&element.role).as_str() {
                "button" => UITargetRole::Button,
                _ => UITargetRole::RankedResult,
            };
            candidate.result_kind = Some("generic".into());
            candidate.confidence = 0.94;
            candidate.rank = Some((index + 1) as u32);
            candidate.observed_at_ms = Some(snapshot.captured_at_ms);
            candidate.reuse_eligible = false;
            candidate.supports_focus = false;
            candidate.supports_click = true;
            candidate.rationale = format!(
                "provider-agnostic UIA result candidate rank {} from role {}",
                index + 1,
                element.role
            );
            Some(candidate)
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AccessibilityTargetSelection {
    #[serde(default)]
    pub selected_element_id: Option<String>,
    #[serde(default)]
    pub accessibility_snapshot_id: Option<String>,
    #[serde(default)]
    pub selection_kind: Option<String>,
    #[serde(default)]
    pub rank: Option<u32>,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub rationale: Option<String>,
}

pub fn parse_accessibility_target_selection_response(
    content: &str,
) -> Result<AccessibilityTargetSelection, String> {
    let value = parse_json_object(content)?;
    if value
        .get("selected_element_id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .is_none()
        && (value.get("x").is_some()
            || value.get("y").is_some()
            || value.get("region").is_some()
            || value.get("bounds").is_some()
            || value.get("click_region").is_some())
    {
        return Err("accessibility target selector returned coordinates without element_id".into());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

pub fn validate_accessibility_target_selection(
    selection: &AccessibilityTargetSelection,
    snapshot: &AccessibilitySnapshot,
    candidates: &[UITargetCandidate],
    required_confidence: f32,
) -> Result<UITargetCandidate, String> {
    if selection.confidence < required_confidence {
        return Err(format!(
            "accessibility target selector confidence {:.2} is below required {:.2}",
            selection.confidence, required_confidence
        ));
    }
    if let Some(selection_snapshot_id) = selection.accessibility_snapshot_id.as_deref() {
        if selection_snapshot_id != snapshot.snapshot_id {
            return Err("accessibility target selector returned a stale snapshot id".into());
        }
    }
    let element_id = selection
        .selected_element_id
        .as_deref()
        .filter(|id| id.starts_with("a11y_"))
        .ok_or_else(|| {
            "accessibility target selector did not return a valid element_id".to_string()
        })?;
    let element = snapshot
        .elements
        .iter()
        .find(|element| element.element_id == element_id)
        .ok_or_else(|| "accessibility target selector selected unknown element_id".to_string())?;
    if element.is_offscreen || !element.is_enabled {
        return Err("accessibility target selector selected unavailable element".into());
    }
    let region = element.bounding_rect.as_ref().ok_or_else(|| {
        "accessibility target selector selected element without bounds".to_string()
    })?;
    if !valid_region(region) {
        return Err("accessibility target selector selected invalid bounds".into());
    }
    if snapshot
        .browser_window_bounds
        .as_ref()
        .is_some_and(|browser_bounds| !region_center_within(region, browser_bounds))
    {
        return Err("accessibility target selector selected element outside browser bounds".into());
    }
    let candidate = candidates
        .iter()
        .find(|candidate| {
            candidate.element_id.as_deref() == Some(element_id)
                && candidate.accessibility_snapshot_id.as_deref()
                    == Some(snapshot.snapshot_id.as_str())
        })
        .ok_or_else(|| {
            "accessibility target selector selected element not present in current candidate set"
                .to_string()
        })?;
    if !candidate.supports_click {
        return Err("accessibility target selector selected non-clickable candidate".into());
    }
    let mut candidate = candidate.clone();
    candidate.confidence = selection.confidence;
    candidate.reuse_eligible = false;
    Ok(candidate)
}

fn parse_snapshot_stdout(stdout: &str) -> Result<AccessibilitySnapshot, String> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Err("empty stdout".into());
    }
    let value = serde_json::from_str::<Value>(trimmed).map_err(|error| error.to_string())?;
    if value.is_array() {
        let elements = serde_json::from_value::<Vec<AccessibleElement>>(value)
            .map_err(|error| error.to_string())?;
        return Ok(AccessibilitySnapshot {
            snapshot_id: next_accessibility_snapshot_id(),
            element_count: elements.len(),
            elements,
            browser_url: None,
            browser_window_bounds: None,
            captured_at_ms: now_ms(),
            capture_backend: "powershell_uia".into(),
            window_is_foreground: false,
            window_pid: None,
            window_process_name: None,
            error: None,
        });
    }

    #[derive(Deserialize)]
    struct SnapshotPayload {
        #[serde(default)]
        snapshot_id: Option<String>,
        #[serde(default)]
        elements: Vec<AccessibleElement>,
        #[serde(default)]
        browser_url: Option<String>,
        #[serde(default)]
        browser_window_bounds: Option<ScreenRegion>,
        #[serde(default)]
        capture_backend: Option<String>,
        #[serde(default)]
        error: Option<String>,
        #[serde(default)]
        window_is_foreground: bool,
        #[serde(default)]
        window_pid: Option<u32>,
        #[serde(default)]
        window_process_name: Option<String>,
    }

    let payload =
        serde_json::from_value::<SnapshotPayload>(value).map_err(|error| error.to_string())?;
    let error = if !payload.window_is_foreground && payload.error.is_none() {
        Some("foreground browser ownership unavailable; using fallback browser selection".into())
    } else {
        payload.error
    };
    Ok(AccessibilitySnapshot {
        snapshot_id: payload
            .snapshot_id
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_else(next_accessibility_snapshot_id),
        element_count: payload.elements.len(),
        elements: payload.elements,
        browser_url: payload.browser_url,
        browser_window_bounds: payload.browser_window_bounds,
        captured_at_ms: now_ms(),
        capture_backend: payload
            .capture_backend
            .unwrap_or_else(|| "powershell_uia".into()),
        window_is_foreground: payload.window_is_foreground,
        window_pid: payload.window_pid,
        window_process_name: payload.window_process_name,
        error,
    }
    .normalize_counts())
}

fn normalize_role(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace("controltype.", "")
        .replace([' ', '_'], "")
}

fn element_is_rankable_result_candidate(
    element: &AccessibleElement,
    snapshot: &AccessibilitySnapshot,
) -> bool {
    if element.is_offscreen || !element.is_enabled {
        return false;
    }
    let Some(region) = element
        .bounding_rect
        .as_ref()
        .filter(|region| valid_region(region))
    else {
        return false;
    };
    if snapshot
        .browser_window_bounds
        .as_ref()
        .is_some_and(|bounds| !region_center_within(region, bounds))
    {
        return false;
    }
    let role = normalize_role(&element.role);
    if !matches!(role.as_str(), "hyperlink" | "button" | "listitem") {
        return false;
    }
    if looks_like_browser_chrome_control(element) {
        return false;
    }
    if role == "listitem"
        && snapshot.elements.iter().any(|child| {
            child.parent_id.as_deref() == Some(element.element_id.as_str())
                && matches!(normalize_role(&child.role).as_str(), "hyperlink" | "button")
                && !child.is_offscreen
                && child.is_enabled
                && child.bounding_rect.as_ref().is_some_and(valid_region)
        })
    {
        return false;
    }
    true
}

fn looks_like_browser_chrome_control(element: &AccessibleElement) -> bool {
    let role = normalize_role(&element.role);
    if matches!(
        role.as_str(),
        "edit" | "tabitem" | "menuitem" | "combobox" | "checkbox"
    ) {
        return true;
    }
    let label = element
        .name
        .as_deref()
        .or(element.automation_id.as_deref())
        .map(normalize_role)
        .unwrap_or_default();
    [
        "address",
        "omnibox",
        "url",
        "back",
        "forward",
        "reload",
        "refresh",
        "newtab",
        "closetab",
        "extensions",
    ]
    .iter()
    .any(|token| label.contains(token))
}

fn valid_region(region: &ScreenRegion) -> bool {
    region.x.is_finite()
        && region.y.is_finite()
        && region.width.is_finite()
        && region.height.is_finite()
        && region.width > 0.0
        && region.height > 0.0
}

fn region_center_within(region: &ScreenRegion, bounds: &ScreenRegion) -> bool {
    let (x, y) = region.center();
    x >= bounds.x && y >= bounds.y && x <= bounds.x + bounds.width && y <= bounds.y + bounds.height
}

fn parse_json_object(content: &str) -> Result<Value, String> {
    let trimmed = content.trim();
    if trimmed.starts_with('{') {
        return serde_json::from_str(trimmed).map_err(|error| error.to_string());
    }
    let start = trimmed
        .find('{')
        .ok_or_else(|| "accessibility target selector response did not contain JSON".to_string())?;
    let end = trimmed
        .rfind('}')
        .ok_or_else(|| "accessibility target selector response did not contain JSON".to_string())?;
    serde_json::from_str(&trimmed[start..=end]).map_err(|error| error.to_string())
}

fn next_accessibility_snapshot_id() -> String {
    let seq = ACCESSIBILITY_SNAPSHOT_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("uia_{}_{}", now_ms(), seq)
}

#[cfg(target_os = "windows")]
fn run_powershell_script_capture_stdout(script: &str) -> Result<String, String> {
    run_powershell_output(script, Duration::from_millis(3_200))
}

#[cfg(target_os = "windows")]
fn run_powershell_output(script: &str, timeout: Duration) -> Result<String, String> {
    use std::process::Command;
    use std::thread::sleep;
    use std::time::Instant;

    let encoded = encode_powershell_command(script);
    let mut child = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-EncodedCommand",
            &encoded,
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|error| format!("accessibility PowerShell backend failed: {error}"))?;

    let started = Instant::now();
    loop {
        if let Some(_status) = child
            .try_wait()
            .map_err(|error| format!("accessibility PowerShell wait failed: {error}"))?
        {
            let output = child
                .wait_with_output()
                .map_err(|error| format!("accessibility PowerShell output failed: {error}"))?;
            if output.status.success() {
                return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
            }
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(if stderr.is_empty() {
                format!(
                    "accessibility PowerShell backend exited with status {:?}",
                    output.status.code()
                )
            } else {
                stderr
            });
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err("accessibility PowerShell backend timed out".into());
        }
        sleep(Duration::from_millis(20));
    }
}

#[cfg(target_os = "windows")]
fn encode_powershell_command(script: &str) -> String {
    let mut bytes = Vec::with_capacity(script.len() * 2);
    for unit in script.encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    BASE64_STANDARD.encode(bytes)
}

#[cfg(target_os = "windows")]
fn powershell_single_quoted(value: &str) -> String {
    value.replace('\'', "''")
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn element_with_bounds() -> AccessibleElement {
        AccessibleElement {
            element_id: "a11y_42".into(),
            automation_id: Some("result".into()),
            runtime_id: Some("1,2,3".into()),
            role: "hyperlink".into(),
            name: Some("Example result".into()),
            value: None,
            bounding_rect: Some(TargetRegion {
                x: 100.0,
                y: 200.0,
                width: 300.0,
                height: 40.0,
                coordinate_space: "screen".into(),
            }),
            is_enabled: true,
            is_offscreen: false,
            depth: 3,
            parent_id: None,
            children: Vec::new(),
        }
    }

    fn element(element_id: &str, role: &str, name: &str, x: f64, y: f64) -> AccessibleElement {
        AccessibleElement {
            element_id: element_id.into(),
            automation_id: None,
            runtime_id: None,
            role: role.into(),
            name: Some(name.into()),
            value: None,
            bounding_rect: Some(TargetRegion {
                x,
                y,
                width: 200.0,
                height: 40.0,
                coordinate_space: "screen".into(),
            }),
            is_enabled: true,
            is_offscreen: false,
            depth: 3,
            parent_id: None,
            children: Vec::new(),
        }
    }

    fn snapshot(elements: Vec<AccessibleElement>) -> AccessibilitySnapshot {
        AccessibilitySnapshot {
            snapshot_id: "uia_synthesis_test".into(),
            element_count: elements.len(),
            elements,
            browser_url: Some("https://example.test".into()),
            browser_window_bounds: Some(TargetRegion {
                x: 0.0,
                y: 0.0,
                width: 1_200.0,
                height: 900.0,
                coordinate_space: "screen".into(),
            }),
            captured_at_ms: 2_000,
            capture_backend: "powershell_uia".into(),
            window_is_foreground: true,
            window_pid: Some(1234),
            window_process_name: Some("chrome".into()),
            error: None,
        }
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn accessibility_snapshot_stub_returns_empty_on_non_windows() {
        let snapshot = capture_accessibility_snapshot(&["chrome"]);

        assert!(snapshot.elements.is_empty());
        assert_eq!(snapshot.capture_backend, "unavailable");
        assert!(snapshot.error.is_some());
    }

    #[test]
    fn accessibility_snapshot_foreground_fields_default_on_non_windows() {
        #[cfg(not(target_os = "windows"))]
        let snapshot = capture_accessibility_snapshot(&["chrome"]);
        #[cfg(target_os = "windows")]
        let snapshot = AccessibilitySnapshot::unavailable("stub defaults");

        assert!(snapshot.elements.is_empty());
        assert!(!snapshot.snapshot_id.is_empty());
        assert!(!snapshot.window_is_foreground);
        assert_eq!(snapshot.window_pid, None);
        assert_eq!(snapshot.window_process_name, None);
    }

    #[test]
    fn accessibility_snapshot_fallback_browser_is_accepted_with_diagnostic() {
        let snapshot = parse_snapshot_stdout(
            r#"{
                "elements": [],
                "browser_url": "https://example.test",
                "browser_window_bounds": null,
                "window_is_foreground": false,
                "window_pid": 1234,
                "window_process_name": "chrome",
                "capture_backend": "powershell_uia",
                "error": null
            }"#,
        )
        .expect("fallback snapshot parses");

        assert!(!snapshot.window_is_foreground);
        assert_eq!(snapshot.window_pid, Some(1234));
        assert_eq!(snapshot.window_process_name.as_deref(), Some("chrome"));
        assert_eq!(
            snapshot.error.as_deref(),
            Some("foreground browser ownership unavailable; using fallback browser selection")
        );
    }

    #[test]
    fn candidate_from_accessible_element_uses_bounding_rect_directly() {
        let candidate = candidate_from_accessible_element(&element_with_bounds(), None, "uia_test")
            .expect("candidate");

        assert_eq!(candidate.candidate_id, "a11y_42");
        assert_eq!(candidate.element_id.as_deref(), Some("a11y_42"));
        assert_eq!(
            candidate.accessibility_snapshot_id.as_deref(),
            Some("uia_test")
        );
        assert_eq!(candidate.source, TargetGroundingSource::AccessibilityLayer);
        assert_eq!(
            candidate.observation_source.as_deref(),
            Some("uia_snapshot")
        );
        assert_eq!(candidate.region.as_ref().expect("region").x, 100.0);
        assert_eq!(candidate.rank, None);
        assert_eq!(candidate.confidence, 0.0);
    }

    #[test]
    fn candidate_from_accessible_element_returns_none_when_offscreen() {
        let mut element = element_with_bounds();
        element.is_offscreen = true;

        assert!(candidate_from_accessible_element(&element, None, "uia_test").is_none());
    }

    #[test]
    fn uia_candidate_is_not_reuse_eligible() {
        let candidate = candidate_from_accessible_element(&element_with_bounds(), None, "uia_test")
            .expect("candidate");

        assert!(!candidate.reuse_eligible);
    }

    #[test]
    fn uia_candidate_carries_snapshot_id() {
        let snapshot = AccessibilitySnapshot {
            snapshot_id: "uia_test_snapshot".into(),
            elements: vec![element_with_bounds()],
            browser_url: None,
            browser_window_bounds: None,
            captured_at_ms: 1_000,
            capture_backend: "powershell_uia".into(),
            element_count: 1,
            window_is_foreground: true,
            window_pid: Some(42),
            window_process_name: Some("chrome".into()),
            error: None,
        };
        let candidate = candidate_from_accessible_element(
            &snapshot.elements[0],
            snapshot.browser_window_bounds.as_ref(),
            &snapshot.snapshot_id,
        )
        .expect("candidate");

        assert_eq!(
            candidate.accessibility_snapshot_id.as_deref(),
            Some(snapshot.snapshot_id.as_str())
        );
    }

    #[test]
    fn uia_synthesizes_ranked_candidates_from_visible_enabled_links() {
        let snapshot = snapshot(vec![
            element("a11y_1", "hyperlink", "First result", 100.0, 120.0),
            element("a11y_2", "hyperlink", "Second result", 100.0, 220.0),
        ]);

        let candidates = synthesize_ranked_uia_result_candidates(&snapshot);

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].rank, Some(1));
        assert_eq!(candidates[0].element_id.as_deref(), Some("a11y_1"));
        assert_eq!(candidates[1].rank, Some(2));
    }

    #[test]
    fn uia_synthesis_ignores_offscreen_disabled_or_unbounded_elements() {
        let mut offscreen = element("a11y_1", "hyperlink", "Offscreen", 100.0, 120.0);
        offscreen.is_offscreen = true;
        let mut disabled = element("a11y_2", "hyperlink", "Disabled", 100.0, 220.0);
        disabled.is_enabled = false;
        let mut unbounded = element("a11y_3", "hyperlink", "Unbounded", 100.0, 320.0);
        unbounded.bounding_rect = None;
        let valid = element("a11y_4", "hyperlink", "Valid", 100.0, 420.0);
        let snapshot = snapshot(vec![offscreen, disabled, unbounded, valid]);

        let candidates = synthesize_ranked_uia_result_candidates(&snapshot);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].element_id.as_deref(), Some("a11y_4"));
    }

    #[test]
    fn uia_synthesis_preserves_element_id_and_snapshot_id() {
        let snapshot = snapshot(vec![element(
            "a11y_7",
            "hyperlink",
            "Preserved",
            100.0,
            120.0,
        )]);

        let candidate = synthesize_ranked_uia_result_candidates(&snapshot)
            .pop()
            .expect("candidate");

        assert_eq!(candidate.element_id.as_deref(), Some("a11y_7"));
        assert_eq!(
            candidate.accessibility_snapshot_id.as_deref(),
            Some("uia_synthesis_test")
        );
        assert!(!candidate.reuse_eligible);
    }

    #[test]
    fn uia_synthesis_orders_candidates_top_to_bottom_left_to_right() {
        let snapshot = snapshot(vec![
            element("a11y_3", "hyperlink", "Lower", 50.0, 300.0),
            element("a11y_2", "hyperlink", "Top right", 300.0, 100.0),
            element("a11y_1", "hyperlink", "Top left", 100.0, 100.0),
        ]);

        let candidates = synthesize_ranked_uia_result_candidates(&snapshot);

        let ids = candidates
            .iter()
            .map(|candidate| candidate.element_id.as_deref())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![Some("a11y_1"), Some("a11y_2"), Some("a11y_3")]);
    }

    #[test]
    fn accessibility_target_selector_accepts_valid_element_id_json() {
        let snapshot = snapshot(vec![element("a11y_7", "hyperlink", "Match", 100.0, 120.0)]);
        let candidates = synthesize_ranked_uia_result_candidates(&snapshot);
        let selection = parse_accessibility_target_selection_response(
            r#"{"selected_element_id":"a11y_7","accessibility_snapshot_id":"uia_synthesis_test","selection_kind":"ranked_result","rank":1,"confidence":0.91,"rationale":"match"}"#,
        )
        .expect("selection");

        let candidate =
            validate_accessibility_target_selection(&selection, &snapshot, &candidates, 0.86)
                .expect("valid target");

        assert_eq!(candidate.element_id.as_deref(), Some("a11y_7"));
        assert_eq!(candidate.confidence, 0.91);
    }

    #[test]
    fn accessibility_target_selector_rejects_coordinate_only_response() {
        let error =
            parse_accessibility_target_selection_response(r#"{"x":100,"y":200,"confidence":0.99}"#)
                .expect_err("coordinate-only response rejected");

        assert!(error.contains("coordinates without element_id"));
    }

    #[test]
    fn accessibility_target_selector_rejects_unknown_element_id() {
        let snapshot = snapshot(vec![element("a11y_7", "hyperlink", "Match", 100.0, 120.0)]);
        let candidates = synthesize_ranked_uia_result_candidates(&snapshot);
        let selection = parse_accessibility_target_selection_response(
            r#"{"selected_element_id":"a11y_99","confidence":0.91}"#,
        )
        .expect("selection");

        let error =
            validate_accessibility_target_selection(&selection, &snapshot, &candidates, 0.86)
                .expect_err("unknown element rejected");

        assert!(error.contains("unknown element_id"));
    }

    #[test]
    fn accessibility_target_selector_rejects_stale_snapshot_id() {
        let snapshot = snapshot(vec![element("a11y_7", "hyperlink", "Match", 100.0, 120.0)]);
        let candidates = synthesize_ranked_uia_result_candidates(&snapshot);
        let selection = parse_accessibility_target_selection_response(
            r#"{"selected_element_id":"a11y_7","accessibility_snapshot_id":"uia_old","confidence":0.91}"#,
        )
        .expect("selection");

        let error =
            validate_accessibility_target_selection(&selection, &snapshot, &candidates, 0.86)
                .expect_err("stale snapshot rejected");

        assert!(error.contains("stale snapshot"));
    }

    #[test]
    fn selected_uia_target_requires_current_snapshot_id() {
        let snapshot = snapshot(vec![element("a11y_7", "hyperlink", "Match", 100.0, 120.0)]);
        let candidates = synthesize_ranked_uia_result_candidates(&snapshot);
        let selection = AccessibilityTargetSelection {
            selected_element_id: Some("a11y_7".into()),
            accessibility_snapshot_id: Some("uia_previous_snapshot".into()),
            selection_kind: Some("ranked_result".into()),
            rank: Some(1),
            confidence: 0.91,
            rationale: Some("stale".into()),
        };

        let error =
            validate_accessibility_target_selection(&selection, &snapshot, &candidates, 0.86)
                .expect_err("stale snapshot rejected");

        assert!(error.contains("stale snapshot"));
    }

    #[test]
    fn llm_uia_selector_rejects_unknown_or_stale_element_id() {
        let snapshot = snapshot(vec![element("a11y_7", "hyperlink", "Match", 100.0, 120.0)]);
        let candidates = synthesize_ranked_uia_result_candidates(&snapshot);
        let unknown = AccessibilityTargetSelection {
            selected_element_id: Some("a11y_404".into()),
            accessibility_snapshot_id: Some(snapshot.snapshot_id.clone()),
            selection_kind: Some("ranked_result".into()),
            rank: Some(1),
            confidence: 0.91,
            rationale: None,
        };
        let stale = AccessibilityTargetSelection {
            selected_element_id: Some("a11y_7".into()),
            accessibility_snapshot_id: Some("uia_stale".into()),
            selection_kind: Some("ranked_result".into()),
            rank: Some(1),
            confidence: 0.91,
            rationale: None,
        };

        let unknown_error =
            validate_accessibility_target_selection(&unknown, &snapshot, &candidates, 0.86)
                .expect_err("unknown element rejected");
        let stale_error =
            validate_accessibility_target_selection(&stale, &snapshot, &candidates, 0.86)
                .expect_err("stale snapshot rejected");

        assert!(unknown_error.contains("unknown element_id"));
        assert!(stale_error.contains("stale snapshot"));
    }

    #[test]
    fn accessibility_target_selector_rejects_low_confidence_selection() {
        let snapshot = snapshot(vec![element("a11y_7", "hyperlink", "Match", 100.0, 120.0)]);
        let candidates = synthesize_ranked_uia_result_candidates(&snapshot);
        let selection = parse_accessibility_target_selection_response(
            r#"{"selected_element_id":"a11y_7","confidence":0.40}"#,
        )
        .expect("selection");

        let error =
            validate_accessibility_target_selection(&selection, &snapshot, &candidates, 0.86)
                .expect_err("low confidence rejected");

        assert!(error.contains("below required"));
    }

    #[test]
    fn accessibility_snapshot_error_does_not_propagate_to_goal_loop() {
        let snapshot = AccessibilitySnapshot::unavailable("uia unavailable");

        assert!(snapshot.elements.is_empty());
        assert_eq!(snapshot.element_count, 0);
        assert_eq!(snapshot.error.as_deref(), Some("uia unavailable"));
    }
}
