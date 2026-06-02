//! Bounded PDF full-text extraction for AstraOS Deep Search.
//!
//! This is intentionally a conservative, dependency-light extractor. It does not
//! attempt to render PDFs, execute embedded actions, or persist raw binaries. It
//! extracts text from unencrypted PDFs by scanning content streams, supporting
//! plain streams and FlateDecode streams. If the extracted text is not good
//! enough, the caller receives an explicit error instead of contaminating the
//! Memory Graph with garbage.

use crate::memory::errors::{MemoryError, MemoryResult};
use flate2::read::ZlibDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;

const MAX_STREAM_BYTES: usize = 3_000_000;
const MAX_EXTRACTED_TEXT_CHARS: usize = 160_000;
const MIN_USEFUL_TEXT_CHARS: usize = 360;
const MAX_SECTIONS: usize = 48;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PdfExtractionOutcome {
    pub text: String,
    pub content_hash: String,
    pub page_count_hint: Option<usize>,
    pub section_count: usize,
    pub method: String,
    pub warnings: Vec<String>,
}

pub(crate) fn extract_pdf_text(bytes: &[u8], max_text_chars: usize) -> MemoryResult<PdfExtractionOutcome> {
    if !looks_like_pdf(bytes) {
        return Err(MemoryError::Validation("payload is not a PDF".into()));
    }
    if contains_marker_case_insensitive(bytes, b"/Encrypt") {
        return Err(MemoryError::Validation(
            "PDF is encrypted; governed deep-search will not attempt password-protected extraction".into(),
        ));
    }

    let page_count_hint = count_marker(bytes, b"/Type /Page").or_else(|| count_marker(bytes, b"/Type/Page"));
    let mut warnings = Vec::<String>::new();
    let streams = extract_candidate_streams(bytes, &mut warnings);
    if streams.is_empty() {
        return Err(MemoryError::Validation("PDF contains no readable content streams".into()));
    }

    let mut raw_text = String::new();
    for stream in streams.iter().take(256) {
        let text = extract_text_from_content_stream(stream);
        if !text.trim().is_empty() {
            raw_text.push_str(&text);
            raw_text.push('\n');
        }
        if raw_text.chars().count() >= max_text_chars.min(MAX_EXTRACTED_TEXT_CHARS) {
            break;
        }
    }

    let normalized = cap_chars(clean_pdf_text(&raw_text), max_text_chars.min(MAX_EXTRACTED_TEXT_CHARS));
    if normalized.chars().count() < MIN_USEFUL_TEXT_CHARS {
        return Err(MemoryError::Validation(
            "PDF text extraction produced too little readable text; source was not ingested".into(),
        ));
    }

    let section_count = estimate_pdf_sections(&normalized);
    let content_hash = sha256_hex(&normalized);
    Ok(PdfExtractionOutcome {
        text: normalized,
        content_hash,
        page_count_hint,
        section_count,
        method: "bounded_pdf_stream_text_extractor_v0_6_5".into(),
        warnings,
    })
}

fn looks_like_pdf(bytes: &[u8]) -> bool {
    bytes.starts_with(b"%PDF") || bytes.windows(4).take(256).any(|w| w == b"%PDF")
}

fn contains_marker_case_insensitive(bytes: &[u8], marker: &[u8]) -> bool {
    let hay = String::from_utf8_lossy(bytes).to_ascii_lowercase();
    let needle = String::from_utf8_lossy(marker).to_ascii_lowercase();
    hay.contains(&needle)
}

fn count_marker(bytes: &[u8], marker: &[u8]) -> Option<usize> {
    let count = bytes.windows(marker.len()).filter(|window| *window == marker).count();
    (count > 0).then_some(count)
}

fn extract_candidate_streams(bytes: &[u8], warnings: &mut Vec<String>) -> Vec<Vec<u8>> {
    let mut streams = Vec::<Vec<u8>>::new();
    let mut offset = 0usize;
    while let Some(stream_pos_rel) = find_bytes(&bytes[offset..], b"stream") {
        let stream_marker = offset + stream_pos_rel;
        let content_start = after_pdf_line_break(bytes, stream_marker + b"stream".len());
        let Some(end_rel) = find_bytes(&bytes[content_start..], b"endstream") else { break; };
        let content_end = content_start + end_rel;
        if content_end > content_start && content_end.saturating_sub(content_start) <= MAX_STREAM_BYTES {
            let dict_start = stream_marker.saturating_sub(768);
            let dict = &bytes[dict_start..stream_marker];
            let raw = trim_pdf_stream_bytes(&bytes[content_start..content_end]);
            if has_flate_filter(dict) {
                match inflate_stream(raw) {
                    Ok(decoded) => streams.push(decoded),
                    Err(error) => warnings.push(format!("flate stream decode failed: {error}")),
                }
            } else if is_probably_text_stream(raw) {
                streams.push(raw.to_vec());
            }
        }
        offset = content_end.saturating_add(b"endstream".len());
        if streams.len() >= 512 { break; }
    }
    streams
}

fn after_pdf_line_break(bytes: &[u8], mut pos: usize) -> usize {
    if pos < bytes.len() && bytes[pos] == b'\r' { pos += 1; }
    if pos < bytes.len() && bytes[pos] == b'\n' { pos += 1; }
    pos
}

fn trim_pdf_stream_bytes(value: &[u8]) -> &[u8] {
    let mut start = 0usize;
    let mut end = value.len();
    while start < end && matches!(value[start], b'\r' | b'\n') { start += 1; }
    while end > start && matches!(value[end - 1], b'\r' | b'\n') { end -= 1; }
    &value[start..end]
}

fn has_flate_filter(dict: &[u8]) -> bool {
    let text = String::from_utf8_lossy(dict).to_ascii_lowercase();
    text.contains("/flatedecode") || text.contains("/filter[/flatedecode") || text.contains("/filter /fl")
}

fn inflate_stream(raw: &[u8]) -> Result<Vec<u8>, String> {
    let mut decoder = ZlibDecoder::new(raw);
    let mut out = Vec::<u8>::new();
    decoder.read_to_end(&mut out).map_err(|error| error.to_string())?;
    Ok(out)
}

fn is_probably_text_stream(raw: &[u8]) -> bool {
    let printable = raw.iter().filter(|b| b.is_ascii_graphic() || b.is_ascii_whitespace()).count();
    !raw.is_empty() && printable * 100 / raw.len().max(1) > 72
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}

fn extract_text_from_content_stream(stream: &[u8]) -> String {
    let mut out = String::new();
    let mut i = 0usize;
    while i < stream.len() {
        match stream[i] {
            b'(' => {
                if let Some((value, next)) = parse_pdf_literal_string(stream, i + 1) {
                    let value = decode_pdf_literal_string(&value);
                    if is_useful_text_fragment(&value) {
                        out.push_str(&value);
                        out.push(' ');
                    }
                    i = next;
                    continue;
                }
            }
            b'<' if i + 1 < stream.len() && stream[i + 1] != b'<' => {
                if let Some((value, next)) = parse_pdf_hex_string(stream, i + 1) {
                    let value = decode_pdf_hex_string(&value);
                    if is_useful_text_fragment(&value) {
                        out.push_str(&value);
                        out.push(' ');
                    }
                    i = next;
                    continue;
                }
            }
            _ => {}
        }
        i += 1;
    }
    out
}

fn parse_pdf_literal_string(stream: &[u8], mut i: usize) -> Option<(Vec<u8>, usize)> {
    let mut depth = 1i32;
    let mut escaped = false;
    let mut value = Vec::<u8>::new();
    while i < stream.len() {
        let b = stream[i];
        if escaped {
            value.push(b'\\');
            value.push(b);
            escaped = false;
        } else if b == b'\\' {
            escaped = true;
        } else if b == b'(' {
            depth += 1;
            value.push(b);
        } else if b == b')' {
            depth -= 1;
            if depth == 0 { return Some((value, i + 1)); }
            value.push(b);
        } else {
            value.push(b);
        }
        i += 1;
        if value.len() > 16_384 { return None; }
    }
    None
}

fn parse_pdf_hex_string(stream: &[u8], mut i: usize) -> Option<(Vec<u8>, usize)> {
    let mut value = Vec::<u8>::new();
    while i < stream.len() {
        let b = stream[i];
        if b == b'>' { return Some((value, i + 1)); }
        if b.is_ascii_hexdigit() { value.push(b); }
        i += 1;
        if value.len() > 32_768 { return None; }
    }
    None
}

fn decode_pdf_literal_string(value: &[u8]) -> String {
    let mut out = Vec::<u8>::new();
    let mut i = 0usize;
    while i < value.len() {
        if value[i] != b'\\' {
            out.push(value[i]);
            i += 1;
            continue;
        }
        i += 1;
        if i >= value.len() { break; }
        match value[i] {
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'b' => out.push(8),
            b'f' => out.push(12),
            b'(' => out.push(b'('),
            b')' => out.push(b')'),
            b'\\' => out.push(b'\\'),
            b'\r' | b'\n' => {}
            b'0'..=b'7' => {
                let mut oct = vec![value[i]];
                for _ in 0..2 {
                    if i + 1 < value.len() && matches!(value[i + 1], b'0'..=b'7') {
                        i += 1;
                        oct.push(value[i]);
                    }
                }
                if let Ok(text) = std::str::from_utf8(&oct) {
                    if let Ok(byte) = u8::from_str_radix(text, 8) { out.push(byte); }
                }
            }
            other => out.push(other),
        }
        i += 1;
    }
    decode_pdf_bytes(&out)
}

fn decode_pdf_hex_string(hex: &[u8]) -> String {
    let mut bytes = Vec::<u8>::new();
    let mut current = Vec::<u8>::new();
    for b in hex.iter().copied().filter(|b| b.is_ascii_hexdigit()) {
        current.push(b);
        if current.len() == 2 {
            if let Ok(s) = std::str::from_utf8(&current) {
                if let Ok(byte) = u8::from_str_radix(s, 16) { bytes.push(byte); }
            }
            current.clear();
        }
    }
    decode_pdf_bytes(&bytes)
}

fn decode_pdf_bytes(bytes: &[u8]) -> String {
    if bytes.starts_with(&[0xFE, 0xFF]) {
        let units = bytes[2..]
            .chunks_exact(2)
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        return String::from_utf16_lossy(&units);
    }
    String::from_utf8_lossy(bytes).to_string()
}

fn is_useful_text_fragment(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.len() < 2 { return false; }
    let alpha = trimmed.chars().filter(|ch| ch.is_alphabetic()).count();
    let control = trimmed.chars().filter(|ch| ch.is_control() && !ch.is_whitespace()).count();
    alpha >= 2 && control == 0
}

fn clean_pdf_text(value: &str) -> String {
    let mut out = String::new();
    let mut previous = String::new();
    for fragment in value.split_whitespace() {
        let cleaned = fragment.trim_matches(|ch: char| ch.is_control()).to_string();
        if cleaned.is_empty() || cleaned == previous { continue; }
        if should_insert_sentence_break(&previous, &cleaned) { out.push('\n'); }
        out.push_str(&cleaned);
        out.push(' ');
        previous = cleaned;
    }
    out.split('\n')
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| line.chars().count() >= 3)
        .collect::<Vec<_>>()
        .join("\n")
}

fn should_insert_sentence_break(previous: &str, current: &str) -> bool {
    if previous.ends_with(|ch: char| matches!(ch, '.' | '?' | '!' | ':')) { return true; }
    previous.chars().count() > 24 && current.chars().next().is_some_and(|ch| ch.is_uppercase())
}

fn estimate_pdf_sections(text: &str) -> usize {
    let count = text.lines().filter(|line| {
        let trimmed = line.trim();
        let len = trimmed.chars().count();
        len >= 4 && len <= 96 && (
            trimmed.chars().all(|ch| !ch.is_alphabetic() || ch.is_uppercase())
            || trimmed.starts_with("Abstract")
            || trimmed.starts_with("Introduction")
            || trimmed.starts_with("Methods")
            || trimmed.starts_with("Results")
            || trimmed.starts_with("Discussion")
            || trimmed.starts_with("Conclusion")
            || trimmed.starts_with("References")
        )
    }).count();
    count.clamp(1, MAX_SECTIONS)
}

fn cap_chars(value: impl Into<String>, max_chars: usize) -> String {
    let value = value.into();
    if value.chars().count() <= max_chars { return value; }
    value.chars().take(max_chars).collect()
}

fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_pdf_literal_string_escapes() {
        let text = decode_pdf_literal_string(br"Astra\040Memory\nRAG");
        assert!(text.contains("Astra Memory"));
        assert!(text.contains("RAG"));
    }

    #[test]
    fn rejects_encrypted_pdf_payload() {
        let bytes = b"%PDF-1.7\n1 0 obj << /Encrypt 2 0 R >> endobj";
        let error = extract_pdf_text(bytes, 10_000).unwrap_err().to_string();
        assert!(error.contains("encrypted"));
    }
}
