use crate::desktop_agent_types::{Permission, RiskLevel, ToolDescriptor};

#[derive(Debug, Clone)]
pub struct ToolsRegistry {
    tools: Vec<ToolDescriptor>,
}

impl ToolsRegistry {
    pub fn new() -> Self {
        Self {
            tools: vec![
                ToolDescriptor {
                    tool_name: "filesystem.read_text".into(),
                    category: "filesystem".into(),
                    description: "Read a UTF-8 text file from an allowed root".into(),
                    required_permissions: vec![Permission::FilesystemRead],
                    default_risk: RiskLevel::Low,
                    requires_confirmation: false,
                },
                ToolDescriptor {
                    tool_name: "filesystem.write_text".into(),
                    category: "filesystem".into(),
                    description: "Create, overwrite, or append UTF-8 text files inside allowed roots".into(),
                    required_permissions: vec![Permission::FilesystemWrite],
                    default_risk: RiskLevel::High,
                    requires_confirmation: true,
                },
                ToolDescriptor {
                    tool_name: "filesystem.search".into(),
                    category: "filesystem".into(),
                    description: "Search files inside an allowed root by filename pattern".into(),
                    required_permissions: vec![Permission::FilesystemSearch],
                    default_risk: RiskLevel::Low,
                    requires_confirmation: false,
                },
                ToolDescriptor {
                    tool_name: "terminal.run".into(),
                    category: "terminal".into(),
                    description: "Execute an allowlisted terminal command inside an allowed working directory".into(),
                    required_permissions: vec![Permission::TerminalSafe],
                    default_risk: RiskLevel::Medium,
                    requires_confirmation: false,
                },
                ToolDescriptor {
                    tool_name: "browser.open".into(),
                    category: "browser".into(),
                    description: "Open a URL in the system browser".into(),
                    required_permissions: vec![Permission::BrowserAction],
                    default_risk: RiskLevel::Medium,
                    requires_confirmation: false,
                },
                ToolDescriptor {
                    tool_name: "browser.search".into(),
                    category: "browser".into(),
                    description: "Run a web search in the default browser".into(),
                    required_permissions: vec![Permission::BrowserRead],
                    default_risk: RiskLevel::Low,
                    requires_confirmation: false,
                },
                ToolDescriptor {
                    tool_name: "screen.analyze".into(),
                    category: "screen".into(),
                    description: "Capture or inspect the current screen and ask Astra Vision what is visible".into(),
                    required_permissions: vec![Permission::DesktopObserve],
                    default_risk: RiskLevel::Low,
                    requires_confirmation: false,
                },
                ToolDescriptor {
                    tool_name: "desktop.launch_app".into(),
                    category: "desktop".into(),
                    description: "Launch a desktop application or file through the operating system".into(),
                    required_permissions: vec![Permission::DesktopControl],
                    default_risk: RiskLevel::Medium,
                    requires_confirmation: false,
                },
            ],
        }
    }

    pub fn list(&self) -> Vec<ToolDescriptor> {
        self.tools.clone()
    }

    pub fn get(&self, tool_name: &str) -> Option<ToolDescriptor> {
        self.tools
            .iter()
            .find(|tool| tool.tool_name == tool_name)
            .cloned()
    }
}
