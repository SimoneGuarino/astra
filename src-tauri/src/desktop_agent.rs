use crate::{
    action_policy::DesktopAgentPolicy,
    audit_log::AuditLogStore,
    browser_agent::BrowserAgent,
    capability_manifest::build_capability_manifest,
    desktop_agent_types::{
        CapabilityManifest, DesktopActionRequest, DesktopActionResponse, DesktopActionStatus,
        DesktopAuditEvent, PendingApproval, ScreenAnalysisRequest, ScreenAnalysisResult,
        ScreenCaptureResult, ScreenObservationStatus,
    },
    filesystem_service::FilesystemService,
    permissions::PermissionProfile,
    screen_workflow::{
        execute_screen_workflow, refresh_screen_workflow_plan, ScreenWorkflow, ScreenWorkflowRun,
    },
    terminal_runner::TerminalRunner,
    tools_registry::ToolsRegistry,
    ui_control::{UIControlRuntime, UIPrimitiveCapabilitySet},
    ui_target_grounding::UITargetCandidate,
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

#[derive(Clone)]
pub struct DesktopAgentRuntime {
    policy: DesktopAgentPolicy,
    permissions: PermissionProfile,
    registry: ToolsRegistry,
    audit: AuditLogStore,
    pending: Arc<Mutex<HashMap<String, PendingActionEnvelope>>>,
    filesystem: FilesystemService,
    terminal: TerminalRunner,
    browser: BrowserAgent,
    pending_store: crate::pending_approvals_store::PendingApprovalsStore,
    screen_capture: crate::screen_capture::ScreenCaptureRuntime,
    screen_vision: crate::screen_vision::ScreenVisionRuntime,
    ui_control: UIControlRuntime,
    recent_workflow_targets: Arc<Mutex<Vec<UITargetCandidate>>>,
}

#[derive(Clone)]
struct PendingActionEnvelope {
    approval: PendingApproval,
    request: DesktopActionRequest,
}

impl DesktopAgentRuntime {
    pub fn new(project_root: PathBuf) -> Self {
        let policy = DesktopAgentPolicy::load_or_default(&project_root);
        let permissions = PermissionProfile::default_local_agent();
        let audit = AuditLogStore::new(&project_root);
        let pending_store =
            crate::pending_approvals_store::PendingApprovalsStore::new(&project_root);
        let pending_entries = pending_store.load();
        let pending = pending_entries
            .into_iter()
            .map(|(action_id, approval)| {
                let request = DesktopActionRequest {
                    tool_name: approval.tool_name.clone(),
                    params: approval.params.clone(),
                    preview_only: false,
                    reason: Some(approval.reason.clone()),
                };
                (action_id, PendingActionEnvelope { approval, request })
            })
            .collect::<HashMap<_, _>>();
        let screen_capture = crate::screen_capture::ScreenCaptureRuntime::new(&project_root);
        let screen_vision = crate::screen_vision::ScreenVisionRuntime::new();
        let ui_control = UIControlRuntime::new();
        Self {
            filesystem: FilesystemService::new(&policy.allowed_roots),
            terminal: TerminalRunner::new(&policy.terminal_allowed_commands, &policy.allowed_roots),
            browser: BrowserAgent,
            registry: ToolsRegistry::new(),
            audit,
            policy,
            permissions,
            pending: Arc::new(Mutex::new(pending)),
            pending_store,
            screen_capture,
            screen_vision,
            ui_control,
            recent_workflow_targets: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn list_tools(&self) -> Vec<crate::desktop_agent_types::ToolDescriptor> {
        self.registry.list()
    }

    pub async fn capability_manifest(&self) -> CapabilityManifest {
        let tools = self.registry.list();
        let policy = self.policy.snapshot(&self.permissions);
        let screen_status = self.screen_capture.status();
        let pending_approvals = self.pending_approvals();
        let vision = self.screen_vision.availability().await;
        build_capability_manifest(&tools, &policy, &screen_status, &pending_approvals, &vision)
    }
    pub fn policy_snapshot(&self) -> crate::desktop_agent_types::DesktopPolicySnapshot {
        self.policy.snapshot(&self.permissions)
    }
    pub fn recent_audit_events(&self, limit: usize) -> Vec<DesktopAuditEvent> {
        self.audit.tail(limit)
    }
    pub fn screen_status(&self) -> ScreenObservationStatus {
        self.screen_capture.status()
    }
    pub fn set_screen_observation_enabled(&self, enabled: bool) -> ScreenObservationStatus {
        self.screen_capture.set_enabled(enabled)
    }
    pub fn capture_screen_snapshot(&self) -> Result<ScreenCaptureResult, String> {
        self.screen_capture.capture_snapshot()
    }

    pub async fn analyze_screen(
        &self,
        request: ScreenAnalysisRequest,
    ) -> Result<ScreenAnalysisResult, String> {
        if !self
            .permissions
            .allows(&crate::desktop_agent_types::Permission::DesktopObserve)
            || !self
                .policy
                .permission_enabled(&crate::desktop_agent_types::Permission::DesktopObserve)
        {
            return Err("Permission denied for desktop observation".into());
        }

        let capture = if request.capture_fresh
            || self.screen_capture.latest_capture_path().is_none()
        {
            if !self.screen_capture.status().enabled {
                return Err("Screen observation is disabled. Enable observation before capturing a fresh screen snapshot.".into());
            }
            self.screen_capture.capture_snapshot()?
        } else {
            let status = self.screen_capture.status();
            let path = status.last_capture_path.ok_or_else(|| {
                "No screen capture available yet. Capture a snapshot first.".to_string()
            })?;
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            ScreenCaptureResult {
                capture_id: Uuid::new_v4().to_string(),
                captured_at: status.last_frame_at.unwrap_or_else(now_ms),
                image_path: path,
                width: None,
                height: None,
                bytes,
                provider: status.provider,
            }
        };

        let question = request.question.clone();
        let action_id = Uuid::new_v4().to_string();
        let request_id = Uuid::new_v4().to_string();
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id: action_id.clone(),
            request_id: request_id.clone(),
            tool_name: "screen.analyze".into(),
            stage: "execution".into(),
            status: "started".into(),
            timestamp: now_ms(),
            risk_level: crate::desktop_agent_types::RiskLevel::Low,
            details: json!({"image_path": capture.image_path, "capture_fresh": request.capture_fresh, "question": question}),
        });

        let result = self
            .screen_vision
            .analyze(
                &capture.image_path,
                capture.captured_at,
                &capture.provider,
                question,
            )
            .await;

        match result {
            Ok(result) => {
                self.audit.append(&DesktopAuditEvent {
                    audit_id: Uuid::new_v4().to_string(),
                    action_id,
                    request_id,
                    tool_name: "screen.analyze".into(),
                    stage: "execution".into(),
                    status: "completed".into(),
                    timestamp: now_ms(),
                    risk_level: crate::desktop_agent_types::RiskLevel::Low,
                    details: json!({
                        "model": result.model,
                        "provider": result.provider,
                        "image_path": result.image_path,
                        "question": result.question,
                    }),
                });
                Ok(result)
            }
            Err(error) => {
                self.audit.append(&DesktopAuditEvent {
                    audit_id: Uuid::new_v4().to_string(),
                    action_id,
                    request_id,
                    tool_name: "screen.analyze".into(),
                    stage: "execution".into(),
                    status: "failed".into(),
                    timestamp: now_ms(),
                    risk_level: crate::desktop_agent_types::RiskLevel::Low,
                    details: json!({"error": error}),
                });
                Err(error)
            }
        }
    }

    pub fn ui_primitive_capabilities(&self) -> UIPrimitiveCapabilitySet {
        let enabled = self
            .permissions
            .allows(&crate::desktop_agent_types::Permission::DesktopControl)
            && self
                .policy
                .permission_enabled(&crate::desktop_agent_types::Permission::DesktopControl);
        self.ui_control.capabilities(enabled)
    }

    pub fn execute_screen_workflow(&self, mut workflow: ScreenWorkflow) -> ScreenWorkflowRun {
        let request_id = Uuid::new_v4().to_string();
        let action_id = Uuid::new_v4().to_string();
        let capabilities = self.ui_primitive_capabilities();
        workflow.grounding.recent_target_candidates = self.recent_workflow_targets();
        refresh_screen_workflow_plan(&mut workflow);
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id: action_id.clone(),
            request_id: request_id.clone(),
            tool_name: "screen.workflow".into(),
            stage: "workflow".into(),
            status: "started".into(),
            timestamp: now_ms(),
            risk_level: crate::desktop_agent_types::RiskLevel::Medium,
            details: json!({
                "workflow": workflow.diagnostic_value(),
                "primitive_capabilities": capabilities,
            }),
        });

        let run = execute_screen_workflow(workflow, &self.ui_control, capabilities);
        self.remember_workflow_targets(&run);
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id,
            request_id,
            tool_name: "screen.workflow".into(),
            stage: "workflow".into(),
            status: run.status.as_str().into(),
            timestamp: now_ms(),
            risk_level: crate::desktop_agent_types::RiskLevel::Medium,
            details: run.diagnostic_value(),
        });
        run
    }

    fn recent_workflow_targets(&self) -> Vec<UITargetCandidate> {
        self.recent_workflow_targets
            .lock()
            .expect("recent workflow target memory poisoned")
            .clone()
    }

    fn remember_workflow_targets(&self, run: &ScreenWorkflowRun) {
        let mut selected = run
            .step_runs
            .iter()
            .filter(|step| {
                matches!(
                    step.status,
                    crate::screen_workflow::WorkflowRunStatus::Completed
                )
            })
            .filter_map(|step| step.target_selection.as_ref())
            .filter_map(|selection| selection.selected_candidate.clone())
            .collect::<Vec<_>>();

        if selected.is_empty() {
            return;
        }

        let now = now_ms();
        for candidate in &mut selected {
            candidate.observed_at_ms = Some(now);
            candidate.reuse_eligible = true;
        }

        let mut memory = self
            .recent_workflow_targets
            .lock()
            .expect("recent workflow target memory poisoned");
        for candidate in selected {
            memory.retain(|existing| existing.candidate_id != candidate.candidate_id);
            memory.push(candidate);
        }
        memory.sort_by(|left, right| right.observed_at_ms.cmp(&left.observed_at_ms));
        memory.truncate(12);
    }

    pub fn pending_approvals(&self) -> Vec<PendingApproval> {
        self.pending
            .lock()
            .expect("pending approvals mutex poisoned")
            .values()
            .map(|e| e.approval.clone())
            .collect()
    }

    pub fn reject_pending(&self, action_id: &str, note: Option<String>) -> Result<(), String> {
        let mut pending = self
            .pending
            .lock()
            .expect("pending approvals mutex poisoned");
        let Some(envelope) = pending.remove(action_id) else {
            return Err(format!("Pending approval not found: {action_id}"));
        };
        self.persist_pending_locked(&pending)?;
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id: envelope.approval.action_id.clone(),
            request_id: envelope.approval.request_id.clone(),
            tool_name: envelope.approval.tool_name.clone(),
            stage: "approval".into(),
            status: "rejected".into(),
            timestamp: now_ms(),
            risk_level: envelope.approval.risk_level.clone(),
            details: json!({"note": note}),
        });
        Ok(())
    }

    pub fn approve_pending(
        &self,
        action_id: &str,
        note: Option<String>,
    ) -> Result<DesktopActionResponse, String> {
        let envelope = {
            let mut pending = self
                .pending
                .lock()
                .expect("pending approvals mutex poisoned");
            let envelope = pending
                .remove(action_id)
                .ok_or_else(|| format!("Pending approval not found: {action_id}"))?;
            self.persist_pending_locked(&pending)?;
            envelope
        };
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id: envelope.approval.action_id.clone(),
            request_id: envelope.approval.request_id.clone(),
            tool_name: envelope.approval.tool_name.clone(),
            stage: "approval".into(),
            status: "approved".into(),
            timestamp: now_ms(),
            risk_level: envelope.approval.risk_level.clone(),
            details: json!({"note": note}),
        });
        self.execute_internal(
            envelope.approval.request_id,
            envelope.approval.action_id,
            envelope.request,
            true,
        )
    }

    pub fn submit_action(
        &self,
        request_id: String,
        request: DesktopActionRequest,
    ) -> Result<DesktopActionResponse, String> {
        let descriptor = self
            .registry
            .get(&request.tool_name)
            .ok_or_else(|| format!("Unknown tool: {}", request.tool_name))?;
        for permission in &descriptor.required_permissions {
            if !self.permissions.allows(permission) || !self.policy.permission_enabled(permission) {
                return Err(format!("Permission denied for {:?}", permission));
            }
        }

        let action_id = Uuid::new_v4().to_string();
        if request.preview_only {
            return Ok(DesktopActionResponse {
                action_id,
                request_id,
                tool_name: descriptor.tool_name,
                status: DesktopActionStatus::Executed,
                message: Some("Preview ready".into()),
                result: Some(json!({"preview": true, "params": request.params})),
                risk_level: Some(descriptor.default_risk),
            });
        }

        if descriptor.requires_confirmation
            || self.policy.requires_approval(&descriptor.default_risk)
        {
            let approval = PendingApproval {
                action_id: action_id.clone(),
                request_id: request_id.clone(),
                tool_name: descriptor.tool_name.clone(),
                params: request.params.clone(),
                risk_level: descriptor.default_risk.clone(),
                reason: request
                    .reason
                    .clone()
                    .unwrap_or_else(|| "Confirmation required by policy".into()),
                requested_at: now_ms(),
            };
            {
                let mut pending = self
                    .pending
                    .lock()
                    .expect("pending approvals mutex poisoned");
                pending.insert(
                    action_id.clone(),
                    PendingActionEnvelope {
                        approval: approval.clone(),
                        request,
                    },
                );
                self.persist_pending_locked(&pending)?;
            }
            self.audit.append(&DesktopAuditEvent {
                audit_id: Uuid::new_v4().to_string(),
                action_id: action_id.clone(),
                request_id: request_id.clone(),
                tool_name: approval.tool_name.clone(),
                stage: "policy".into(),
                status: "approval_required".into(),
                timestamp: now_ms(),
                risk_level: approval.risk_level.clone(),
                details: json!({"params": approval.params, "reason": approval.reason}),
            });
            return Ok(DesktopActionResponse {
                action_id,
                request_id,
                tool_name: descriptor.tool_name,
                status: DesktopActionStatus::ApprovalRequired,
                message: Some("Approval required".into()),
                result: None,
                risk_level: Some(descriptor.default_risk),
            });
        }

        self.execute_internal(request_id, action_id, request, false)
    }

    fn execute_internal(
        &self,
        request_id: String,
        action_id: String,
        request: DesktopActionRequest,
        was_approved: bool,
    ) -> Result<DesktopActionResponse, String> {
        let descriptor = self
            .registry
            .get(&request.tool_name)
            .ok_or_else(|| format!("Unknown tool: {}", request.tool_name))?;
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id: action_id.clone(),
            request_id: request_id.clone(),
            tool_name: request.tool_name.clone(),
            stage: "execution".into(),
            status: "started".into(),
            timestamp: now_ms(),
            risk_level: descriptor.default_risk.clone(),
            details: json!({"params": request.params, "approved": was_approved}),
        });

        let result = self.dispatch(&request);
        match result {
            Ok(result) => {
                self.audit.append(&DesktopAuditEvent {
                    audit_id: Uuid::new_v4().to_string(),
                    action_id: action_id.clone(),
                    request_id: request_id.clone(),
                    tool_name: request.tool_name.clone(),
                    stage: "execution".into(),
                    status: "completed".into(),
                    timestamp: now_ms(),
                    risk_level: descriptor.default_risk.clone(),
                    details: json!({"result": result}),
                });
                Ok(DesktopActionResponse {
                    action_id,
                    request_id,
                    tool_name: request.tool_name,
                    status: DesktopActionStatus::Executed,
                    message: Some(if was_approved {
                        "Approved action executed".into()
                    } else {
                        "Action executed".into()
                    }),
                    result: Some(result),
                    risk_level: Some(descriptor.default_risk),
                })
            }
            Err(error) => {
                self.audit.append(&DesktopAuditEvent {
                    audit_id: Uuid::new_v4().to_string(),
                    action_id: action_id.clone(),
                    request_id: request_id.clone(),
                    tool_name: request.tool_name.clone(),
                    stage: "execution".into(),
                    status: "failed".into(),
                    timestamp: now_ms(),
                    risk_level: descriptor.default_risk.clone(),
                    details: json!({"error": error}),
                });
                Err(error)
            }
        }
    }

    fn persist_pending_locked(
        &self,
        pending: &HashMap<String, PendingActionEnvelope>,
    ) -> Result<(), String> {
        self.pending_store
            .save(pending.values().map(|entry| entry.approval.clone()))
    }

    fn dispatch(&self, request: &DesktopActionRequest) -> Result<Value, String> {
        match request.tool_name.as_str() {
            "filesystem.read_text" => {
                let path = request
                    .params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "filesystem.read_text requires params.path".to_string())?;
                let mut result = self.filesystem.read_text(path)?;
                if let Some(post_processing) = request.params.get("post_processing") {
                    result["post_processing"] = post_processing.clone();
                }
                if let Some(operation) = request.params.get("operation") {
                    result["operation"] = operation.clone();
                }
                Ok(result)
            }
            "filesystem.write_text" => {
                let path = request
                    .params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "filesystem.write_text requires params.path".to_string())?;
                let create_empty = request
                    .params
                    .get("create_empty")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let content = request
                    .params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .or_else(|| create_empty.then_some(""))
                    .ok_or_else(|| "filesystem.write_text requires params.content".to_string())?;
                let mode = request
                    .params
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("overwrite");
                self.filesystem.write_text(path, content, mode)
            }
            "filesystem.search" => {
                let root = request.params.get("root").and_then(|v| v.as_str());
                let pattern = request
                    .params
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "filesystem.search requires params.pattern".to_string())?;
                let max_results = request
                    .params
                    .get("max_results")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(25) as usize;
                self.filesystem.search(root, pattern, max_results)
            }
            "terminal.run" => {
                let command = request
                    .params
                    .get("command")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "terminal.run requires params.command".to_string())?;
                let args = request
                    .params
                    .get("args")
                    .and_then(|v| v.as_array())
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|item| item.as_str().map(ToString::to_string))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let cwd = request.params.get("cwd").and_then(|v| v.as_str());
                self.terminal.run(command, &args, cwd)
            }
            "browser.open" => {
                let url = request
                    .params
                    .get("url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "browser.open requires params.url".to_string())?;
                self.browser.open(url)
            }
            "browser.search" => {
                let query = request
                    .params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "browser.search requires params.query".to_string())?;
                self.browser.search(query)
            }
            "desktop.launch_app" => {
                let requested =
                    string_param(&request.params, &["path", "app", "app_name", "application"])
                        .ok_or_else(|| {
                            "desktop.launch_app requires params.path or params.app_name".to_string()
                        })?;
                let path = resolve_launch_target(&requested);
                let args = string_array_param(&request.params, &["args", "arguments"]);
                launch_app(&path, &args)
            }
            _ => Err(format!("Unsupported tool: {}", request.tool_name)),
        }
    }
}

fn string_param(params: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        params
            .get(key)
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn string_array_param(params: &Value, keys: &[&str]) -> Vec<String> {
    keys.iter()
        .find_map(|key| {
            params
                .get(key)
                .and_then(|value| value.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(ToOwned::to_owned))
                        .collect::<Vec<_>>()
                })
        })
        .unwrap_or_default()
}

fn resolve_launch_target(requested: &str) -> String {
    let normalized = requested
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_ascii_lowercase()
        .replace('-', " ");
    let is_chrome = matches!(
        normalized.as_str(),
        "browser" | "chrome" | "google chrome" | "googlechrome"
    );

    if !is_chrome {
        return requested.to_string();
    }

    if cfg!(target_os = "windows") {
        let candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ];
        if let Some(path) = candidates
            .iter()
            .find(|path| std::path::Path::new(path).exists())
        {
            return (*path).to_string();
        }
        return "chrome.exe".into();
    }

    "google-chrome".into()
}

fn launch_app(path: &str, args: &[String]) -> Result<Value, String> {
    std::process::Command::new(path)
        .args(args)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(json!({"path": path, "args": args, "launched": true}))
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::DesktopAgentRuntime;
    use crate::desktop_agent_types::{DesktopActionRequest, DesktopActionStatus, PendingApproval};
    use serde_json::json;
    use uuid::Uuid;

    #[test]
    fn filesystem_write_creates_persisted_pending_approval() {
        let root = std::env::temp_dir().join(format!("astra_desktop_agent_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("temp root");
        let target_path = root.join("test.txt");
        let runtime = DesktopAgentRuntime::new(root.clone());

        let response = runtime
            .submit_action(
                "test-request".into(),
                DesktopActionRequest {
                    tool_name: "filesystem.write_text".into(),
                    params: json!({
                        "path": target_path.display().to_string(),
                        "content": "",
                        "mode": "overwrite",
                        "create_empty": true,
                    }),
                    preview_only: false,
                    reason: Some("test create empty file".into()),
                },
            )
            .expect("submit action");

        assert!(matches!(
            response.status,
            DesktopActionStatus::ApprovalRequired
        ));
        let pending_path = root
            .join(".astra")
            .join("state")
            .join("pending_approvals.json");
        assert!(pending_path.exists());
        let approvals = serde_json::from_str::<Vec<PendingApproval>>(
            &std::fs::read_to_string(&pending_path).expect("pending approvals file"),
        )
        .expect("pending approval json");
        assert_eq!(approvals.len(), 1);
        assert_eq!(approvals[0].tool_name, "filesystem.write_text");

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn launch_target_resolves_chrome_alias_without_losing_args_contract() {
        let target = super::resolve_launch_target("google-chrome");
        assert!(target.to_ascii_lowercase().contains("chrome"));
    }
}
