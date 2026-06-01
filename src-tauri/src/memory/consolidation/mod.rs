//! Memory consolidation entrypoint.
//!
//! Consolidation remains LLM-first and Rust-governed: the model may propose
//! nodes, relations, claims, or procedural lessons, but Rust validates and
//! persists only typed memory records.

pub mod conversation;
pub mod research;

pub use conversation::{
    consolidate_conversation_bundle, ConversationDecision, ConversationEntity, ConversationImportantPoint,
    ConversationMemoryBundle, ConversationMemoryConsolidationReceipt, ConversationPreference,
    ConversationProcedure, ConversationSemanticAtom,
};
pub use research::{
    consolidate_research_bundle, ResearchMemoryBundle, ResearchMemoryConsolidationReceipt,
};
pub mod reflection;
pub mod reconsolidation;


pub use reconsolidation::{
    finalize_reconsolidation_receipt, list_reconsolidation_candidates,
    MemoryReconsolidationCandidate, MemoryReconsolidationItemReceipt,
    MemoryReconsolidationReceipt, MemoryReconsolidationRequest,
};
