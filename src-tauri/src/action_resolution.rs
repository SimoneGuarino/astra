use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedActionIntent {
    ToolActionRequest,
    ClarificationRequired,
    Unsupported,
}

impl ResolvedActionIntent {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ToolActionRequest => "tool_action_request",
            Self::ClarificationRequired => "clarification_required",
            Self::Unsupported => "unsupported",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActionDomain {
    Filesystem,
    Browser,
    Desktop,
    Terminal,
    BrowserScreenInteraction,
    Screen,
    Unknown,
}

impl ActionDomain {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Filesystem => "filesystem",
            Self::Browser => "browser",
            Self::Desktop => "desktop",
            Self::Terminal => "terminal",
            Self::BrowserScreenInteraction => "browser_screen_interaction",
            Self::Screen => "screen",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActionOperation {
    ReadFile,
    ReadAndSummarizeFile,
    WriteFile,
    SearchFile,
    BrowserSearch,
    BrowserOpen,
    DesktopLaunchApp,
    ScreenGuidedBrowserWorkflow,
    ScreenGuidedFollowupAction,
    ScreenGuidedNavigationWorkflow,
    Unknown,
}

impl ActionOperation {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ReadFile => "read_file",
            Self::ReadAndSummarizeFile => "read_and_summarize_file",
            Self::WriteFile => "write_file",
            Self::SearchFile => "search_file",
            Self::BrowserSearch => "browser_search",
            Self::BrowserOpen => "browser_open",
            Self::DesktopLaunchApp => "desktop_launch_app",
            Self::ScreenGuidedBrowserWorkflow => "screen_guided_browser_workflow",
            Self::ScreenGuidedFollowupAction => "screen_guided_followup_action",
            Self::ScreenGuidedNavigationWorkflow => "screen_guided_navigation_workflow",
            Self::Unknown => "unknown",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match normalize_label(value).as_str() {
            "read_file" | "filesystem_read" | "read_text" | "filesystem_read_text" => {
                Some(Self::ReadFile)
            }
            "read_and_summarize_file"
            | "summarize_file"
            | "file_summary"
            | "filesystem_summarize" => Some(Self::ReadAndSummarizeFile),
            "write_file" | "filesystem_write" | "filesystem_write_text" => Some(Self::WriteFile),
            "search_file" | "filesystem_search" => Some(Self::SearchFile),
            "browser_search" | "search_browser" | "web_search" => Some(Self::BrowserSearch),
            "browser_open" | "browser_open_url" | "open_url" => Some(Self::BrowserOpen),
            "desktop_launch_app" | "launch_app" | "desktop_launch" => Some(Self::DesktopLaunchApp),
            "screen_guided_browser_workflow"
            | "browser_screen_workflow"
            | "screen_guided_workflow" => Some(Self::ScreenGuidedBrowserWorkflow),
            "screen_guided_followup_action"
            | "screen_followup_action"
            | "screen_interaction_followup" => Some(Self::ScreenGuidedFollowupAction),
            "screen_guided_navigation_workflow"
            | "screen_navigation"
            | "screen_guided_navigation" => Some(Self::ScreenGuidedNavigationWorkflow),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QueryMode {
    Precise,
    Semantic,
}

impl QueryMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Precise => "precise",
            Self::Semantic => "semantic",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match normalize_label(value).as_str() {
            "precise" | "literal" | "exact" => Some(Self::Precise),
            "semantic" | "inferred" | "goal" => Some(Self::Semantic),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionSource {
    ModelAssisted,
    RustNormalizer,
    HeuristicFallback,
}

impl ResolutionSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ModelAssisted => "model_assisted",
            Self::RustNormalizer => "rust_normalizer",
            Self::HeuristicFallback => "heuristic_fallback",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResolution {
    pub intent: ResolvedActionIntent,
    pub domain: ActionDomain,
    pub operation: ActionOperation,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub query_mode: Option<QueryMode>,
    #[serde(default)]
    pub entities: Value,
    #[serde(default)]
    pub post_processing: Value,
    #[serde(default)]
    pub workflow_steps: Vec<String>,
    #[serde(default)]
    pub requires_screen_context: bool,
    #[serde(default)]
    pub ambiguity: Option<String>,
    pub confidence: f32,
    pub source: ResolutionSource,
    #[serde(default)]
    pub rationale: Option<String>,
}

impl ActionResolution {
    pub fn new(
        operation: ActionOperation,
        domain: ActionDomain,
        confidence: f32,
        source: ResolutionSource,
    ) -> Self {
        Self {
            intent: ResolvedActionIntent::ToolActionRequest,
            domain,
            operation,
            provider: None,
            query_mode: None,
            entities: json!({}),
            post_processing: json!({}),
            workflow_steps: Vec::new(),
            requires_screen_context: false,
            ambiguity: None,
            confidence: confidence.clamp(0.0, 1.0),
            source,
            rationale: None,
        }
    }

    pub fn diagnostic_value(&self) -> Value {
        json!({
            "resolved_intent": self.intent.as_str(),
            "domain": self.domain.as_str(),
            "operation": self.operation.as_str(),
            "provider": self.provider,
            "query_mode": self.query_mode.as_ref().map(QueryMode::as_str),
            "entities": self.entities,
            "post_processing": self.post_processing,
            "workflow_steps": self.workflow_steps,
            "requires_screen_context": self.requires_screen_context,
            "ambiguity": self.ambiguity,
            "confidence": self.confidence,
            "resolution_source": self.source.as_str(),
            "rationale": self.rationale,
        })
    }
}

pub fn normalize_label(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "_")
}
