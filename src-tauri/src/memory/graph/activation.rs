//! Activation support for the cognitive memory graph.
//!
//! The first implementation lives in `store::sqlite_store` so activation stays
//! close to persistence and can update edge activation counters atomically. This
//! module is intentionally kept as the stable location for future propagation
//! strategies, including vector-informed spreading activation and decay.
