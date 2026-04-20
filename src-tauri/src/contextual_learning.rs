use crate::workflow_continuation::{
    ContextualFollowupInterpretation, ContextualMergeDiagnostic,
    ContinuationRegroundingDiagnostics, SemanticPageValidationResult,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedContinuationOutcome {
    pub run_id: String,
    pub status: String,
    pub completed_steps: usize,
    #[serde(default)]
    pub verifier_status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedContinuationLearningEvent {
    pub phrase: String,
    pub interpretation: ContextualFollowupInterpretation,
    #[serde(default)]
    pub merge: Option<ContextualMergeDiagnostic>,
    #[serde(default)]
    pub page_validation: Option<SemanticPageValidationResult>,
    #[serde(default)]
    pub regrounding: Option<ContinuationRegroundingDiagnostics>,
    pub outcome: VerifiedContinuationOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedContinuationLearningReceipt {
    pub accepted: bool,
    pub reason: String,
    pub summary: Value,
}

pub fn store_verified_continuation(
    event: VerifiedContinuationLearningEvent,
) -> VerifiedContinuationLearningReceipt {
    VerifiedContinuationLearningReceipt {
        accepted: true,
        reason: "verified continuation accepted by typed learning hook; persistence and retrieval are intentionally deferred".into(),
        summary: json!({
            "phrase": event.phrase,
            "continuation_kind": event.interpretation.continuation_kind,
            "provider": event.merge.as_ref().and_then(|merge| merge.effective_provider.clone()),
            "query": event.merge.as_ref().and_then(|merge| merge.effective_query.clone()),
            "browser_app": event.merge.as_ref().and_then(|merge| merge.effective_browser_app.clone()),
            "page_validation": event.page_validation.as_ref().map(|validation| &validation.status),
            "regrounding": event.regrounding.as_ref().map(|regrounding| &regrounding.final_status),
            "run_id": event.outcome.run_id,
            "status": event.outcome.status,
            "completed_steps": event.outcome.completed_steps,
            "verifier_status": event.outcome.verifier_status,
        }),
    }
}
