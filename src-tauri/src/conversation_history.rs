use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

const DEFAULT_MAX_MESSAGES: usize = 16;
const DEFAULT_MAX_CHARS_PER_MESSAGE: usize = 1_600;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone)]
pub struct ConversationHistoryManager {
    inner: Arc<Mutex<ConversationHistoryState>>,
}

#[derive(Debug)]
struct ConversationHistoryState {
    committed: VecDeque<ConversationMessage>,
    pending: HashMap<String, PendingTurn>,
    max_messages: usize,
    max_chars_per_message: usize,
}

#[derive(Debug)]
struct PendingTurn {
    user_message: String,
}

impl ConversationHistoryManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ConversationHistoryState {
                committed: VecDeque::new(),
                pending: HashMap::new(),
                max_messages: DEFAULT_MAX_MESSAGES,
                max_chars_per_message: DEFAULT_MAX_CHARS_PER_MESSAGE,
            })),
        }
    }

    pub fn begin_turn(&self, request_id: String, user_message: &str) {
        let normalized = sanitize_message(user_message, self.max_chars_per_message());
        if normalized.is_empty() {
            return;
        }

        let mut state = self.inner.lock().expect("conversation history mutex poisoned");
        state.pending.insert(request_id, PendingTurn { user_message: normalized });
    }

    pub fn commit_turn(&self, request_id: &str, assistant_message: &str) {
        let assistant_message = sanitize_message(assistant_message, self.max_chars_per_message());
        let mut state = self.inner.lock().expect("conversation history mutex poisoned");
        let Some(pending) = state.pending.remove(request_id) else {
            return;
        };

        state.committed.push_back(ConversationMessage {
            role: "user".to_string(),
            content: pending.user_message,
        });

        if !assistant_message.is_empty() {
            state.committed.push_back(ConversationMessage {
                role: "assistant".to_string(),
                content: assistant_message,
            });
        }

        while state.committed.len() > state.max_messages {
            state.committed.pop_front();
        }
    }

    pub fn discard_turn(&self, request_id: &str) {
        let mut state = self.inner.lock().expect("conversation history mutex poisoned");
        state.pending.remove(request_id);
    }

    pub fn recent_messages(&self, limit: usize) -> Vec<ConversationMessage> {
        let state = self.inner.lock().expect("conversation history mutex poisoned");
        let start = state.committed.len().saturating_sub(limit);
        state.committed.iter().skip(start).cloned().collect()
    }

    fn max_chars_per_message(&self) -> usize {
        let state = self.inner.lock().expect("conversation history mutex poisoned");
        state.max_chars_per_message
    }
}

fn sanitize_message(message: &str, max_chars: usize) -> String {
    let collapsed = message.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }

    trimmed.chars().take(max_chars).collect::<String>().trim().to_string()
}
