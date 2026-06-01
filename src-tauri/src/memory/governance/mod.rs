use crate::memory::types::{MemoryGovernancePolicySnapshot, MemoryVerificationStatus};
use serde_json::json;

pub fn governance_policy_snapshot() -> MemoryGovernancePolicySnapshot {
    MemoryGovernancePolicySnapshot {
        version: "astra_memory_governance_v1".into(),
        user_control_enabled: true,
        inferred_memory_default_weight: 0.58,
        user_confirmed_weight: 1.0,
        deprecated_memory_retrieval_enabled: false,
        hard_delete_enabled: false,
        allowed_statuses: vec![
            MemoryVerificationStatus::Unverified,
            MemoryVerificationStatus::LlmInferred,
            MemoryVerificationStatus::UserConfirmed,
            MemoryVerificationStatus::SystemVerified,
            MemoryVerificationStatus::Contradicted,
            MemoryVerificationStatus::Deprecated,
        ],
        metadata: json!({
            "source_of_truth": "sqlite_memory_graph",
            "governance_model": "user_visible_soft_state_controls",
            "hard_delete_note": "disabled in this phase; use deprecated state to exclude memory from retrieval",
            "metadata_only": true
        }),
    }
}

pub fn memory_status_weight(status: &MemoryVerificationStatus) -> f32 {
    match status {
        MemoryVerificationStatus::UserConfirmed => 1.0,
        MemoryVerificationStatus::SystemVerified => 0.92,
        MemoryVerificationStatus::LlmInferred => 0.64,
        MemoryVerificationStatus::Unverified => 0.52,
        MemoryVerificationStatus::Contradicted => 0.08,
        MemoryVerificationStatus::Deprecated => 0.0,
    }
}

pub fn is_retrieval_enabled_status(status: &MemoryVerificationStatus) -> bool {
    !matches!(status, MemoryVerificationStatus::Contradicted | MemoryVerificationStatus::Deprecated)
}
