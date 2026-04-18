#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub segment_id: String,
    pub sequence: u32,
    pub text: String,
}

#[derive(Debug, Default)]
pub struct SentenceSegmenter {
    buffer: String,
    next_sequence: u32,
}

impl SentenceSegmenter {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            next_sequence: 1,
        }
    }

    pub fn push(&mut self, chunk: &str) -> Vec<SpeechSegment> {
        self.buffer.push_str(chunk);
        self.extract_ready_segments(false)
    }

    pub fn flush(&mut self) -> Vec<SpeechSegment> {
        self.extract_ready_segments(true)
    }

    fn extract_ready_segments(&mut self, flush: bool) -> Vec<SpeechSegment> {
        let mut segments = Vec::new();

        loop {
            self.drop_leading_whitespace();

            if self.buffer.is_empty() {
                break;
            }

            let boundary = find_boundary(&self.buffer, flush);
            let Some(boundary) = boundary else {
                break;
            };

            let raw = self.buffer[..boundary].to_string();
            self.buffer.drain(..boundary);

            let text = normalize_segment_text(&raw);
            if text.is_empty() {
                continue;
            }

            let sequence = self.next_sequence;
            self.next_sequence += 1;

            segments.push(SpeechSegment {
                segment_id: sequence.to_string(),
                sequence,
                text,
            });
        }

        if flush {
            self.drop_leading_whitespace();
            let trailing = normalize_segment_text(&self.buffer);
            self.buffer.clear();

            if !trailing.is_empty() {
                let sequence = self.next_sequence;
                self.next_sequence += 1;
                segments.push(SpeechSegment {
                    segment_id: sequence.to_string(),
                    sequence,
                    text: trailing,
                });
            }
        }

        segments
    }

    fn drop_leading_whitespace(&mut self) {
        let trimmed = self.buffer.trim_start();
        let drain_len = self.buffer.len() - trimmed.len();
        if drain_len > 0 {
            self.buffer.drain(..drain_len);
        }
    }
}

fn find_boundary(buffer: &str, flush: bool) -> Option<usize> {
    const MIN_SENTENCE_CHARS: usize = 20;
    const IDEAL_MAX_CHARS: usize = 110;
    const HARD_MAX_CHARS: usize = 170;

    let mut last_soft_boundary = None;
    let mut char_count = 0usize;

    for (idx, ch) in buffer.char_indices() {
        char_count += 1;
        let end = idx + ch.len_utf8();

        if matches!(ch, ',' | ';' | ':' | '\n') && char_count >= 70 {
            last_soft_boundary = Some(end);
        }

        if is_sentence_terminal(ch)
            && (char_count >= MIN_SENTENCE_CHARS
                || looks_like_short_conversational_segment(&buffer[..end]))
            && !looks_like_abbreviation(&buffer[..end])
            && !looks_like_decimal(buffer, idx)
            && is_followed_by_boundary(buffer, end)
        {
            let boundary = include_closing_punctuation(buffer, end);
            if should_hold_for_more_context(&buffer[..boundary], &buffer[boundary..], flush) {
                continue;
            }
            return Some(boundary);
        }

        if char_count >= IDEAL_MAX_CHARS {
            if let Some(soft) = last_soft_boundary {
                return Some(soft);
            }
        }

        if char_count >= HARD_MAX_CHARS {
            return Some(find_last_whitespace_before(buffer, end).unwrap_or(end));
        }
    }

    if flush && !buffer.trim().is_empty() {
        return Some(buffer.len());
    }

    None
}

fn is_sentence_terminal(ch: char) -> bool {
    matches!(ch, '.' | '?' | '!')
}

fn is_followed_by_boundary(buffer: &str, end: usize) -> bool {
    let rest = &buffer[end..];
    let Some(next) = rest.chars().next() else {
        return true;
    };

    next.is_whitespace() || matches!(next, '"' | '\'' | ')' | ']' | '}')
}

fn include_closing_punctuation(buffer: &str, mut end: usize) -> usize {
    while end < buffer.len() {
        let Some(ch) = buffer[end..].chars().next() else {
            break;
        };

        if matches!(ch, '"' | '\'' | ')' | ']' | '}') {
            end += ch.len_utf8();
            continue;
        }

        break;
    }

    end
}

fn looks_like_decimal(buffer: &str, dot_idx: usize) -> bool {
    let before = buffer[..dot_idx].chars().next_back();
    let after = buffer[dot_idx + 1..].chars().next();

    matches!((before, after), (Some(a), Some(b)) if a.is_ascii_digit() && b.is_ascii_digit())
}

fn looks_like_abbreviation(text: &str) -> bool {
    let token = text
        .trim()
        .split_whitespace()
        .next_back()
        .unwrap_or_default()
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | ')' | ']' | '}'))
        .to_ascii_lowercase();

    matches!(
        token.as_str(),
        "es."
            | "ecc."
            | "sig."
            | "sig.ra"
            | "dott."
            | "prof."
            | "ing."
            | "avv."
            | "dr."
            | "mr."
            | "mrs."
            | "vs."
            | "etc."
    )
}

fn looks_like_short_conversational_prefix(text: &str) -> bool {
    let normalized = text.trim().trim_end_matches(',').to_lowercase();
    matches!(
        normalized.as_str(),
        "certo"
            | "ok"
            | "okay"
            | "va bene"
            | "perfetto"
            | "capito"
            | "chiaro"
            | "bene"
            | "sì"
            | "si"
            | "dimmi"
            | "eccomi"
            | "allora"
    )
}

fn looks_like_short_conversational_segment(text: &str) -> bool {
    let normalized = text
        .trim()
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | ')' | ']' | '}'))
        .trim_end_matches(|ch: char| matches!(ch, '.' | '?' | '!'))
        .to_lowercase();

    if normalized.chars().count() < 3 || normalized.chars().count() > 42 {
        return false;
    }

    if text.trim_end().ends_with('?') && normalized.chars().count() >= 7 {
        return true;
    }

    let acknowledgements = [
        "certo",
        "ok",
        "okay",
        "va bene",
        "perfetto",
        "capito",
        "chiaro",
        "bene",
        "sì",
        "si",
        "assolutamente",
        "dimmi",
        "eccomi",
        "ci sono",
    ];

    acknowledgements
        .iter()
        .any(|prefix| normalized == *prefix || normalized.starts_with(&format!("{prefix}, ")))
}

fn should_hold_for_more_context(current: &str, rest: &str, flush: bool) -> bool {
    if flush {
        return false;
    }

    let trimmed = current.trim();
    if trimmed.is_empty() {
        return false;
    }

    let word_count = trimmed.split_whitespace().count();
    let char_count = trimmed.chars().count();
    let is_short = char_count <= 26 || word_count <= 4;
    let is_question = trimmed.ends_with('?');
    let is_ack = looks_like_short_conversational_segment(trimmed);

    if is_question || is_ack {
        return false;
    }

    if !is_short {
        return false;
    }

    let next = rest.trim_start();
    if next.is_empty() {
        return false;
    }

    let next_lower = next.to_lowercase();
    next_lower.starts_with("e ")
        || next_lower.starts_with("ma ")
        || next_lower.starts_with("però ")
        || next_lower.starts_with("quindi ")
        || next_lower.starts_with("perche ")
        || next_lower.starts_with("perché ")
        || next_lower.starts_with("che ")
        || next_lower.starts_with("se ")
}

fn find_last_whitespace_before(buffer: &str, end: usize) -> Option<usize> {
    buffer[..end]
        .char_indices()
        .rev()
        .find_map(|(idx, ch)| ch.is_whitespace().then_some(idx + ch.len_utf8()))
}

#[cfg(test)]
mod tests {
    use super::SentenceSegmenter;

    #[test]
    fn emits_short_acknowledgement_without_waiting_for_flush() {
        let mut segmenter = SentenceSegmenter::new();
        let segments = segmenter.push("Certo. Procedo con il controllo.");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Certo.");
    }

    #[test]
    fn emits_short_question_without_waiting_for_flush() {
        let mut segmenter = SentenceSegmenter::new();
        let segments = segmenter.push("Come va? Dimmi pure.");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Come va?");
    }
}

fn normalize_segment_text(input: &str) -> String {
    let mut text = input.replace("\r\n", "\n").replace('\r', "\n");
    text = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    collapse_whitespace(&text)
}

fn collapse_whitespace(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut previous_was_space = false;

    for ch in input.chars() {
        if ch.is_whitespace() {
            if !previous_was_space {
                output.push(' ');
                previous_was_space = true;
            }
        } else {
            output.push(ch);
            previous_was_space = false;
        }
    }

    output.trim().to_string()
}
