//! Follow-up email sender — compose and send meeting follow-up emails
//!
//! Generates follow-up emails from the exported meeting data (summary, action items, decisions)
//! and sends them via configured email service (SMTP or IMAP).

use super::types::*;
use std::collections::HashMap;

pub struct FollowUpSender {
    pub email_config: EmailConfig,
    pub templates: HashMap<String, String>,
    pub send_failed: bool,
}

/// Email configuration.
pub struct EmailConfig {
    pub smtp_host: String,
    pub smtp_port: u16,
    pub username: String,
    pub password: String,
    pub from_address: String,
}

impl FollowUpSender {
    pub fn new(smtp_host: String, smtp_port: u16, username: String, from_address: String) -> Self {
        Self {
            email_config: EmailConfig {
                smtp_host,
                smtp_port,
                username,
                password: String::new(),
                from_address,
            },
            templates: HashMap::new(),
            send_failed: false,
        }
    }

    /// Compose a follow-up email from the meeting data.
    pub fn compose(&self, exported: &ExportedMeeting, recipients: Vec<String>) -> String {
        let mut email = String::new();

        email.push_str(&format!(
            "Subject: Meeting Follow-up: {} ({})\n",
            exported.platform, exported.session_id
        ));
        email.push_str(&format!(
            "From: Astra Meet <{}>\n",
            self.email_config.from_address
        ));
        email.push_str(&format!("To: {}\n", recipients.join(", ")));
        email.push('\n');

        email.push_str("Hi team,\n\n");
        email.push_str(&format!(
            "Here's a summary of the meeting we had on {} ({}):
",
            exported.started_at.format("%Y-%m-%d %H:%M UTC"),
            exported.platform
        ));

        // Summary section
        if !exported.summary.is_empty() {
            email.push_str("## Summary\n");
            for entry in &exported.summary {
                email.push_str(&format!("- {}\n", entry.summary));
            }
            email.push('\n');
        }

        // Decisions section
        if !exported.decisions.is_empty() {
            email.push_str("## Decisions Made\n");
            let mut decision_no = 0;
            for decision in &exported.decisions {
                decision_no += 1;
                email.push_str(&format!("D{}: {}\n", decision_no, decision.decision));
                if !decision.rationale.is_empty() {
                    email.push_str(&format!("Rationale: {}\n", decision.rationale));
                }
            }
            email.push('\n');
        }

        // Action Items section
        if !exported.action_items.is_empty() {
            email.push_str("## Action Items\n");
            let mut item_no = 0;
            for item in &exported.action_items {
                item_no += 1;
                let assignee = item
                    .assignee
                    .as_ref()
                    .map(|p| p.name.clone())
                    .unwrap_or("tbd".to_string());
                email.push_str(&format!(
                    "{}- {} (Assigned to: {})\n",
                    item_no, item.description, assignee
                ));
            }
            email.push('\n');
        }

        // Transcript (abbreviated)
        email.push_str("For the full transcript, please see the attached document.\n\n");
        email.push_str("Best regards,\nAstra");

        email
    }

    /// Send the composed email to specified recipients.
    pub fn send(
        &mut self,
        _exported: &ExportedMeeting,
        _recipients: Vec<String>,
    ) -> Result<(), String> {
        self.send_failed = true;
        Err("follow-up sending is not yet supported".to_string())
    }
}

/// Compose a markdown-formatted email body from meeting data.
pub fn compose_markdown_email(exported: &ExportedMeeting) -> String {
    let mut email = String::new();

    email.push_str("#\n");
    email.push_str(&format!(
        "# Meeting Follow-up: {} (Session: {})\n",
        exported.platform, exported.session_id
    ));
    email.push_str(&format!(
        "# Date: {}\n",
        exported.started_at.format("%Y-%m-%d %H:%M UTC")
    ));
    email.push('\n');

    // Summary section
    if !exported.summary.is_empty() {
        email.push_str("##\n");
        for entry in &exported.summary {
            email.push_str(&format!("- {}\n", entry.summary));
        }
        email.push('\n');
    }

    // Decisions section
    if !exported.decisions.is_empty() {
        email.push_str("## Decisions Made\n");
        let mut decision_no = 0;
        for decision in &exported.decisions {
            decision_no += 1;
            email.push_str(&format!("### D{}: {}\n", decision_no, decision.decision));
            if !decision.rationale.is_empty() {
                email.push_str(&format!("**Rationale:** {}\n", decision.rationale));
            }
        }
        email.push('\n');
    }

    // Action Items section
    if !exported.action_items.is_empty() {
        email.push_str("## Action Items\n");
        let mut item_no = 0;
        for item in &exported.action_items {
            item_no += 1;
            let assignee = item
                .assignee
                .as_ref()
                .map(|p| p.name.clone())
                .unwrap_or("tbd".to_string());
            email.push_str(&format!(
                "- **{}: {}** (Assigned to: {})\n",
                item_no, item.description, assignee
            ));
        }
        email.push('\n');
    }

    email
}
