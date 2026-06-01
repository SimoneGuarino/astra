use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum MemoryError {
    Storage(String),
    Serialization(String),
    Validation(String),
}

impl Display for MemoryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Storage(message) => write!(f, "memory storage error: {message}"),
            Self::Serialization(message) => write!(f, "memory serialization error: {message}"),
            Self::Validation(message) => write!(f, "memory validation error: {message}"),
        }
    }
}

impl std::error::Error for MemoryError {}

impl From<rusqlite::Error> for MemoryError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Storage(value.to_string())
    }
}

impl From<serde_json::Error> for MemoryError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

pub type MemoryResult<T> = Result<T, MemoryError>;
