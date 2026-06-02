use crate::cognitive_thinking::{
    DeepSearchReason, ThinkingPlan, ThinkingRoute, ThinkingUncertaintyLevel,
};
use serde::{Deserialize, Serialize};

const MAX_FINDINGS: usize = 12;
const MAX_TEXT_CHARS: usize = 220;
const THINKING_QUALITY_SAFEGUARDS: [&str; 5] = [
    "metadata_only",
    "no_raw_chain_of_thought",
    "no_tool_router_bypass",
    "no_canonical_memory_auto_promotion",
    "rust_runtime_owns_final_decision",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingQualityReport {
    pub score: f32,
    pub grade: ThinkingQualityGrade,
    pub status: ThinkingQualityStatus,
    pub route_consistency: f32,
    pub evidence_alignment: f32,
    pub memory_alignment: f32,
    pub tool_safety_alignment: f32,
    pub deep_search_alignment: f32,
    pub uncertainty_alignment: f32,
    #[serde(default)]
    pub findings: Vec<ThinkingQualityFinding>,
    #[serde(default)]
    pub safeguards: Vec<String>,
    pub metadata_only: bool,
    pub raw_chain_of_thought_included: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingQualityGrade {
    Excellent,
    Good,
    NeedsReview,
    Risky,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingQualityStatus {
    Pass,
    Observe,
    Review,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingQualityFinding {
    pub severity: ThinkingQualitySeverity,
    pub code: String,
    pub message: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingQualitySeverity {
    Info,
    Warning,
    Critical,
}

pub fn evaluate_thinking_plan(plan: &ThinkingPlan) -> ThinkingQualityReport {
    let mut findings = Vec::new();

    let route_consistency = evaluate_route_consistency(plan, &mut findings);
    let evidence_alignment = evaluate_evidence_alignment(plan, &mut findings);
    let memory_alignment = evaluate_memory_alignment(plan, &mut findings);
    let tool_safety_alignment = evaluate_tool_safety_alignment(plan, &mut findings);
    let deep_search_alignment = evaluate_deep_search_alignment(plan, &mut findings);
    let uncertainty_alignment = evaluate_uncertainty_alignment(plan, &mut findings);

    let mut weighted_score = route_consistency * 0.23
        + evidence_alignment * 0.18
        + memory_alignment * 0.16
        + tool_safety_alignment * 0.18
        + deep_search_alignment * 0.15
        + uncertainty_alignment * 0.10;

    if plan.planner_source == "deterministic_heuristic" {
        weighted_score -= 0.03;
        push_finding(
            &mut findings,
            ThinkingQualitySeverity::Info,
            "heuristic_planner_used",
            "Il ThinkingPlan è stato prodotto dal fallback deterministico.",
            "Verificare che il modello planner sia disponibile se la richiesta richiede ragionamento più ricco.",
        );
    }

    if !plan.warnings.is_empty() {
        weighted_score -= (plan.warnings.len().min(4) as f32) * 0.025;
    }

    let score = weighted_score.clamp(0.0, 1.0);
    let has_critical = findings.iter().any(|finding| finding.severity == ThinkingQualitySeverity::Critical);
    let has_warning = findings.iter().any(|finding| finding.severity == ThinkingQualitySeverity::Warning);
    let grade = if has_critical || score < 0.52 {
        ThinkingQualityGrade::Risky
    } else if has_warning || score < 0.70 {
        ThinkingQualityGrade::NeedsReview
    } else if score < 0.86 {
        ThinkingQualityGrade::Good
    } else {
        ThinkingQualityGrade::Excellent
    };
    let status = match grade {
        ThinkingQualityGrade::Excellent | ThinkingQualityGrade::Good => ThinkingQualityStatus::Pass,
        ThinkingQualityGrade::NeedsReview => ThinkingQualityStatus::Observe,
        ThinkingQualityGrade::Risky => ThinkingQualityStatus::Review,
    };

    ThinkingQualityReport {
        score,
        grade,
        status,
        route_consistency,
        evidence_alignment,
        memory_alignment,
        tool_safety_alignment,
        deep_search_alignment,
        uncertainty_alignment,
        findings: findings.into_iter().take(MAX_FINDINGS).collect(),
        safeguards: THINKING_QUALITY_SAFEGUARDS
            .iter()
            .map(|safeguard| (*safeguard).to_string())
            .collect(),
        metadata_only: true,
        raw_chain_of_thought_included: false,
    }
}

fn evaluate_route_consistency(plan: &ThinkingPlan, findings: &mut Vec<ThinkingQualityFinding>) -> f32 {
    let mut score = 1.0f32;
    match &plan.route {
        ThinkingRoute::DeepSearchRequired if !plan.deep_search.is_needed() => {
            score -= 0.44;
            push_finding(
                findings,
                ThinkingQualitySeverity::Critical,
                "deep_search_route_without_deep_search_decision",
                "La route richiede Deep Search ma la decisione Deep Search non risulta necessaria.",
                "Allineare route e deep_search.needed prima di eseguire ricerca o sintesi finale.",
            );
        }
        ThinkingRoute::ToolArbitrationRequired if !plan.tool_decision.tool_required => {
            score -= 0.38;
            push_finding(
                findings,
                ThinkingQualitySeverity::Critical,
                "tool_route_without_tool_requirement",
                "La route richiede tool arbitration ma tool_decision.tool_required è false.",
                "Mantenere tool-bound requests dentro il router governato e rendere esplicita la ragione.",
            );
        }
        ThinkingRoute::DirectAnswer if plan.tool_decision.tool_required => {
            score -= 0.48;
            push_finding(
                findings,
                ThinkingQualitySeverity::Critical,
                "direct_answer_with_tool_requirement",
                "Il piano propone risposta diretta pur indicando che serve un tool.",
                "Non permettere bypass del tool router; forzare ToolArbitrationRequired o chiedere chiarimento.",
            );
        }
        ThinkingRoute::DirectAnswer if plan.deep_search.is_needed() => {
            score -= 0.34;
            push_finding(
                findings,
                ThinkingQualitySeverity::Warning,
                "direct_answer_with_deep_search_needed",
                "Il piano propone risposta diretta ma segnala che Deep Search è necessaria.",
                "Preferire DeepSearchRequired o disabilitare esplicitamente Deep Search con una ragione policy-safe.",
            );
        }
        ThinkingRoute::MemoryGroundedAnswer if plan.memory_assessment.coverage < 0.20 => {
            score -= 0.28;
            push_finding(
                findings,
                ThinkingQualitySeverity::Warning,
                "memory_grounded_with_low_coverage",
                "La route è memory-grounded ma la coverage della memoria è molto bassa.",
                "Usare transcript/session routing, chiarimento o Deep Search invece di sovrastimare la memoria.",
            );
        }
        _ => {}
    }
    score.clamp(0.0, 1.0)
}

fn evaluate_evidence_alignment(plan: &ThinkingPlan, findings: &mut Vec<ThinkingQualityFinding>) -> f32 {
    let mut score = 1.0f32;
    if plan.evidence_assessment.requires_external_sources && !plan.deep_search.is_needed() {
        score -= 0.34;
        push_finding(
            findings,
            ThinkingQualitySeverity::Warning,
            "external_sources_required_without_deep_search",
            "L'evidence assessment richiede fonti esterne ma Deep Search non è marcata come necessaria.",
            "Allineare evidence.requires_external_sources con deep_search.needed o documentare un blocco policy.",
        );
    }
    if plan.evidence_assessment.requires_current_information
        && !matches!(&plan.deep_search.reason, DeepSearchReason::CurrentInformation | DeepSearchReason::HighStakes)
        && plan.deep_search.is_needed()
    {
        score -= 0.18;
        push_finding(
            findings,
            ThinkingQualitySeverity::Info,
            "current_info_reason_not_explicit",
            "La richiesta richiede informazioni correnti ma la reason Deep Search non esplicita current_information/high_stakes.",
            "Rendere la reason più precisa per audit e replay.",
        );
    }
    if !plan.evidence_assessment.has_local_evidence
        && !plan.evidence_assessment.requires_external_sources
        && matches!(&plan.route, ThinkingRoute::MemoryGroundedAnswer)
    {
        score -= 0.25;
        push_finding(
            findings,
            ThinkingQualitySeverity::Warning,
            "memory_route_without_local_evidence",
            "La route usa memoria ma non risultano evidenze locali disponibili.",
            "Non trattare la memoria come fonte se il packet non contiene evidenza locale sufficiente.",
        );
    }
    score.clamp(0.0, 1.0)
}

fn evaluate_memory_alignment(plan: &ThinkingPlan, findings: &mut Vec<ThinkingQualityFinding>) -> f32 {
    let mut score = 1.0f32;
    if plan.memory_assessment.coverage > 0.0 && !plan.memory_assessment.relevant {
        score -= 0.20;
        push_finding(
            findings,
            ThinkingQualitySeverity::Info,
            "memory_coverage_without_relevance",
            "La memoria ha coverage non nulla ma relevant=false.",
            "Normalizzare la semantica di coverage/relevance per evitare UI e audit ambigui.",
        );
    }
    if plan.memory_assessment.coverage < 0.35
        && matches!(&plan.route, ThinkingRoute::MemoryGroundedAnswer)
        && plan.memory_assessment.missing_information.is_empty()
    {
        score -= 0.16;
        push_finding(
            findings,
            ThinkingQualitySeverity::Info,
            "low_memory_coverage_without_missing_info",
            "Coverage memoria bassa senza missing_information esplicite.",
            "Popolare missing_information aiuta il planner e il pannello governance a spiegare il gap.",
        );
    }
    score.clamp(0.0, 1.0)
}

fn evaluate_tool_safety_alignment(plan: &ThinkingPlan, findings: &mut Vec<ThinkingQualityFinding>) -> f32 {
    let mut score = 1.0f32;
    if plan.tool_decision.tool_required && !matches!(&plan.route, ThinkingRoute::ToolArbitrationRequired) {
        score -= 0.42;
        push_finding(
            findings,
            ThinkingQualitySeverity::Critical,
            "tool_required_but_route_not_tool_arbitration",
            "tool_decision richiede tool, ma la route non è ToolArbitrationRequired.",
            "Forzare il passaggio dal router governato prima di browser, file, terminale o desktop actions.",
        );
    }
    if matches!(&plan.route, ThinkingRoute::ToolArbitrationRequired)
        && plan.tool_decision.reason.as_deref().unwrap_or_default().trim().is_empty()
    {
        score -= 0.14;
        push_finding(
            findings,
            ThinkingQualitySeverity::Info,
            "tool_route_without_reason",
            "La route tool arbitration non contiene una reason esplicita.",
            "Aggiungere una reason sintetica migliora auditabilità e diagnosi runtime.",
        );
    }
    score.clamp(0.0, 1.0)
}

fn evaluate_deep_search_alignment(plan: &ThinkingPlan, findings: &mut Vec<ThinkingQualityFinding>) -> f32 {
    let mut score = 1.0f32;
    if plan.deep_search.needed && matches!(&plan.deep_search.reason, DeepSearchReason::NotNeeded | DeepSearchReason::BlockedByPolicy) {
        score -= 0.44;
        push_finding(
            findings,
            ThinkingQualitySeverity::Critical,
            "deep_search_needed_with_blocking_reason",
            "deep_search.needed=true è incoerente con reason not_needed/blocked_by_policy.",
            "Normalizzare la decisione Deep Search prima della selection automatica.",
        );
    }
    if matches!(&plan.route, ThinkingRoute::DeepSearchRequired)
        && plan.deep_search.query_hint.as_deref().unwrap_or_default().trim().is_empty()
    {
        score -= 0.12;
        push_finding(
            findings,
            ThinkingQualitySeverity::Info,
            "deep_search_route_without_query_hint",
            "La route Deep Search non contiene query_hint.",
            "Fornire un query_hint bounded migliora discovery e riduce passaggi inutili.",
        );
    }
    if plan.deep_search.is_needed()
        && matches!(&plan.route, ThinkingRoute::ToolArbitrationRequired)
    {
        score -= 0.12;
        push_finding(
            findings,
            ThinkingQualitySeverity::Info,
            "deep_search_and_tool_arbitration_overlap",
            "Il piano segnala sia tool arbitration sia Deep Search.",
            "Verificare l'ordine: prima tool/session context se locale, poi Deep Search solo se serve evidenza esterna.",
        );
    }
    score.clamp(0.0, 1.0)
}

fn evaluate_uncertainty_alignment(plan: &ThinkingPlan, findings: &mut Vec<ThinkingQualityFinding>) -> f32 {
    let mut score = 1.0f32;
    if matches!(&plan.uncertainty.level, ThinkingUncertaintyLevel::High) && plan.confidence > 0.82 {
        score -= 0.22;
        push_finding(
            findings,
            ThinkingQualitySeverity::Warning,
            "high_uncertainty_high_confidence",
            "Uncertainty alta ma confidence molto elevata.",
            "Ridurre confidence, richiedere chiarimento o attivare evidenza aggiuntiva.",
        );
    }
    if plan.confidence < 0.45 && !matches!(&plan.route, ThinkingRoute::ClarifyRequired | ThinkingRoute::Refuse) {
        score -= 0.30;
        push_finding(
            findings,
            ThinkingQualitySeverity::Warning,
            "low_confidence_without_clarify_or_refuse",
            "Confidence bassa ma route non richiede chiarimento né safe refusal.",
            "Preferire clarify_required quando il piano non ha sufficiente certezza operativa.",
        );
    }
    if matches!(&plan.uncertainty.level, ThinkingUncertaintyLevel::Medium | ThinkingUncertaintyLevel::High)
        && plan.uncertainty.reasons.is_empty()
    {
        score -= 0.12;
        push_finding(
            findings,
            ThinkingQualitySeverity::Info,
            "uncertainty_without_reasons",
            "Uncertainty non bassa senza reasons esplicite.",
            "Aggiungere reasons sintetiche migliora tracciabilità e UI governance.",
        );
    }
    score.clamp(0.0, 1.0)
}

fn push_finding(
    findings: &mut Vec<ThinkingQualityFinding>,
    severity: ThinkingQualitySeverity,
    code: &str,
    message: &str,
    recommendation: &str,
) {
    if findings.iter().any(|finding| finding.code == code) {
        return;
    }
    findings.push(ThinkingQualityFinding {
        severity,
        code: cap_text(code, 96),
        message: cap_text(message, MAX_TEXT_CHARS),
        recommendation: cap_text(recommendation, MAX_TEXT_CHARS),
    });
}

fn cap_text(value: &str, max_chars: usize) -> String {
    let trimmed = value.trim().replace('\0', " ");
    if trimmed.chars().count() <= max_chars {
        return trimmed;
    }
    let mut capped = trimmed.chars().take(max_chars.saturating_sub(1)).collect::<String>();
    capped.push('…');
    capped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_thinking::{
        DeepSearchDecision, EvidenceAssessment, MemoryAssessment, ThinkingUncertainty,
        ThinkingUncertaintyLevel, ToolDecision,
    };

    fn base_plan(route: ThinkingRoute) -> ThinkingPlan {
        ThinkingPlan {
            request_id: "req_quality".into(),
            intent_summary: "quality test".into(),
            self_questions: Vec::new(),
            memory_assessment: MemoryAssessment {
                relevant: false,
                coverage: 0.0,
                node_count: 0,
                missing_information: Vec::new(),
            },
            evidence_assessment: EvidenceAssessment {
                has_local_evidence: false,
                has_current_session_evidence: false,
                requires_current_information: false,
                requires_external_sources: false,
                evidence_summary: "none".into(),
            },
            route,
            deep_search: DeepSearchDecision {
                needed: false,
                reason: DeepSearchReason::NotNeeded,
                query_hint: None,
            },
            tool_decision: ToolDecision {
                tool_required: false,
                reason: None,
                candidate_tool: None,
            },
            uncertainty: ThinkingUncertainty {
                level: ThinkingUncertaintyLevel::Low,
                reasons: Vec::new(),
            },
            user_visible_trace: Vec::new(),
            confidence: 0.78,
            planner_source: "test".into(),
            duration_ms: 8,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn regression_direct_answer_with_tool_requirement_is_risky() {
        let mut plan = base_plan(ThinkingRoute::DirectAnswer);
        plan.tool_decision.tool_required = true;
        plan.tool_decision.candidate_tool = Some("browser.open".into());

        let report = evaluate_thinking_plan(&plan);

        assert_eq!(report.status, ThinkingQualityStatus::Review);
        assert!(report.score < 0.70);
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "direct_answer_with_tool_requirement"));
        assert!(report.metadata_only);
        assert!(!report.raw_chain_of_thought_included);
    }

    #[test]
    fn safeguards_contract_is_stable_and_metadata_only() {
        let plan = base_plan(ThinkingRoute::DirectAnswer);
        let report = evaluate_thinking_plan(&plan);

        assert!(report.metadata_only);
        assert!(!report.raw_chain_of_thought_included);
        assert!(report.safeguards.iter().any(|item| item == "metadata_only"));
        assert!(report
            .safeguards
            .iter()
            .any(|item| item == "rust_runtime_owns_final_decision"));
    }

    #[test]
    fn regression_coherent_deep_search_plan_passes_or_observes_without_raw_cot() {
        let mut plan = base_plan(ThinkingRoute::DeepSearchRequired);
        plan.deep_search = DeepSearchDecision {
            needed: true,
            reason: DeepSearchReason::CurrentInformation,
            query_hint: Some("current AI regulations".into()),
        };
        plan.evidence_assessment.requires_current_information = true;
        plan.evidence_assessment.requires_external_sources = true;
        plan.confidence = 0.82;

        let report = evaluate_thinking_plan(&plan);

        assert!(matches!(
            report.status,
            ThinkingQualityStatus::Pass | ThinkingQualityStatus::Observe
        ));
        assert!(report.score >= 0.70);
        assert!(!report.raw_chain_of_thought_included);
        assert!(report.metadata_only);
    }

    #[test]
    fn regression_external_evidence_without_deep_search_is_flagged() {
        let mut plan = base_plan(ThinkingRoute::DirectAnswer);
        plan.evidence_assessment.requires_current_information = true;
        plan.evidence_assessment.requires_external_sources = true;

        let report = evaluate_thinking_plan(&plan);

        assert!(report.score < 0.86);
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "external_sources_required_without_deep_search"));
        assert!(report.metadata_only);
        assert!(!report.raw_chain_of_thought_included);
    }

    #[test]
    fn regression_tool_bound_plan_requires_tool_route() {
        let mut plan = base_plan(ThinkingRoute::DirectAnswer);
        plan.tool_decision.tool_required = true;
        plan.tool_decision.candidate_tool = Some("desktop.click".into());

        let report = evaluate_thinking_plan(&plan);

        assert_eq!(report.status, ThinkingQualityStatus::Review);
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "direct_answer_with_tool_requirement"));
    }

    #[test]
    fn regression_memory_grounded_with_low_coverage_needs_observation() {
        let mut plan = base_plan(ThinkingRoute::MemoryGroundedAnswer);
        plan.memory_assessment.relevant = true;
        plan.memory_assessment.coverage = 0.12;
        plan.memory_assessment.node_count = 1;

        let report = evaluate_thinking_plan(&plan);

        assert!(matches!(
            report.status,
            ThinkingQualityStatus::Observe | ThinkingQualityStatus::Review
        ));
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "memory_grounded_with_low_coverage"));
    }

    #[test]
    fn regression_low_confidence_without_clarify_is_flagged() {
        let mut plan = base_plan(ThinkingRoute::DirectAnswer);
        plan.confidence = 0.33;
        plan.uncertainty = ThinkingUncertainty {
            level: ThinkingUncertaintyLevel::High,
            reasons: vec!["ambiguous target".into()],
        };

        let report = evaluate_thinking_plan(&plan);

        assert!(report.score < 0.80);
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "low_confidence_without_clarify_or_refuse"));
        assert!(!report.raw_chain_of_thought_included);
    }

    #[test]
    fn regression_clarify_low_confidence_is_safer_than_direct_answer() {
        let mut plan = base_plan(ThinkingRoute::ClarifyRequired);
        plan.confidence = 0.33;
        plan.uncertainty = ThinkingUncertainty {
            level: ThinkingUncertaintyLevel::High,
            reasons: vec!["ambiguous target".into()],
        };

        let report = evaluate_thinking_plan(&plan);

        assert!(!report
            .findings
            .iter()
            .any(|finding| finding.code == "low_confidence_without_clarify_or_refuse"));
        assert!(report.metadata_only);
    }
}
