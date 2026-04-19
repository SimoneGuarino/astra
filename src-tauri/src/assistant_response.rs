use crate::desktop_agent_types::{DesktopActionResponse, DesktopActionStatus};
use serde_json::{Deserializer, Value};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct RenderedAssistantResponse {
    pub display_text: String,
    pub speech_text: String,
}

impl RenderedAssistantResponse {
    pub fn from_display(display_text: impl Into<String>) -> Self {
        let display_text = present_display_text(&display_text.into());
        let speech_text = speech_safe_text(&display_text);
        Self {
            display_text,
            speech_text,
        }
    }
}

#[derive(Debug, Default)]
pub struct StreamPresentationState {
    prefix_buffer: String,
    at_response_start: bool,
}

impl StreamPresentationState {
    pub fn new() -> Self {
        Self {
            prefix_buffer: String::new(),
            at_response_start: true,
        }
    }

    pub fn display_chunk(&mut self, raw_chunk: &str) -> String {
        if raw_chunk.is_empty() {
            return String::new();
        }

        if !self.at_response_start {
            return sanitize_display_fragment(raw_chunk);
        }

        self.prefix_buffer.push_str(raw_chunk);
        let trimmed = self.prefix_buffer.trim_start();

        if starts_like_internal_payload(trimmed) {
            let sanitized = sanitize_display_text(&self.prefix_buffer);
            if sanitized != self.prefix_buffer {
                self.at_response_start = false;
                self.prefix_buffer.clear();
                return sanitized;
            }

            if self.prefix_buffer.len() < 8_192 {
                return String::new();
            }
        }

        self.at_response_start = false;
        let chunk = sanitize_display_fragment(&self.prefix_buffer);
        self.prefix_buffer.clear();
        chunk
    }

    pub fn finish(&mut self) -> String {
        if self.prefix_buffer.is_empty() {
            return String::new();
        }
        self.at_response_start = false;
        sanitize_display_text(std::mem::take(&mut self.prefix_buffer).as_str())
    }
}

pub fn render_action_response(
    response: &DesktopActionResponse,
    original_message: &str,
) -> RenderedAssistantResponse {
    let italian = looks_italian(original_message);
    let display_text = match response.status {
        DesktopActionStatus::Executed => render_executed_action(response, italian),
        DesktopActionStatus::ApprovalRequired => {
            if italian {
                format!(
                    "{} e' pronta, ma serve la tua approvazione prima dell'esecuzione. Ho creato una richiesta di approvazione reale nel runtime.",
                    friendly_tool_name(&response.tool_name)
                )
            } else {
                format!(
                    "{} is ready, but it needs your approval before execution. I created a real pending approval in the runtime.",
                    friendly_tool_name(&response.tool_name)
                )
            }
        }
        DesktopActionStatus::Rejected => {
            if italian {
                format!(
                    "L'azione {} e' stata rifiutata.",
                    friendly_tool_name(&response.tool_name)
                )
            } else {
                format!(
                    "The {} action was rejected.",
                    friendly_tool_name(&response.tool_name)
                )
            }
        }
        DesktopActionStatus::Failed => response.message.clone().unwrap_or_else(|| {
            if italian {
                format!(
                    "L'azione {} non e' riuscita.",
                    friendly_tool_name(&response.tool_name)
                )
            } else {
                format!(
                    "The {} action failed.",
                    friendly_tool_name(&response.tool_name)
                )
            }
        }),
    };

    let speech_text = match response.status {
        DesktopActionStatus::Executed if response.tool_name == "filesystem.read_text" => {
            if italian {
                let name = action_result_path(response)
                    .map(|path| compact_path_label(&path))
                    .unwrap_or_else(|| "il file".into());
                format!("Ho letto {name}. Ti mostro il contenuto rilevante in chat.")
            } else {
                let name = action_result_path(response)
                    .map(|path| compact_path_label(&path))
                    .unwrap_or_else(|| "the file".into());
                format!("I read {name}. I am showing the relevant content in chat.")
            }
        }
        _ => speech_safe_text(&display_text),
    };

    RenderedAssistantResponse {
        display_text: present_display_text(&display_text),
        speech_text: speech_safe_text(&speech_text),
    }
}

pub fn sanitize_display_text(text: &str) -> String {
    let mut current = text.trim().to_string();

    loop {
        let stripped = strip_leading_code_fenced_payload(&current)
            .or_else(|| strip_leading_json_payload(&current))
            .or_else(|| strip_leading_internal_planning(&current));
        match stripped {
            Some(next) if next != current => current = next.trim_start().to_string(),
            _ => break,
        }
    }

    current
}

pub fn present_display_text(text: &str) -> String {
    normalize_ascii_tables(&strip_embedded_diagnostic_payloads(&sanitize_display_text(
        text,
    )))
}

pub fn speech_safe_text(text: &str) -> String {
    let display = present_display_text(text);
    if let Some(summary) = structured_speech_summary(&display) {
        return summary;
    }

    let mut output = String::with_capacity(display.len());
    let mut in_code_fence = false;

    for line in display.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence {
            continue;
        }

        let mut cleaned = trimmed
            .trim_start_matches('#')
            .trim_start_matches(|ch: char| matches!(ch, '-' | '*' | '+' | '>'))
            .trim()
            .replace("**", "")
            .replace("__", "")
            .replace('`', "")
            .replace('*', "")
            .replace('|', " ");

        if looks_like_json_line(&cleaned) {
            continue;
        }

        cleaned = cleaned
            .chars()
            .map(|ch| match ch {
                '{' | '}' | '[' | ']' => ' ',
                '"' => ' ',
                _ => ch,
            })
            .collect::<String>();

        if !cleaned.trim().is_empty() {
            if !output.is_empty() {
                output.push(' ');
            }
            output.push_str(cleaned.trim());
        }
    }

    suppress_repeated_sentences(&collapse_whitespace(&output))
}

pub fn fallback_display_for_empty_response(original_message: &str) -> String {
    let lower = original_message.to_lowercase();
    if lower.contains("tabella") || lower.contains("table") {
        return if looks_italian(original_message) {
            concat!(
                "Certo, ecco un esempio di tabella semplice:\n",
                "| Prodotto | Quantita | Prezzo |\n",
                "| --- | --- | --- |\n",
                "| Latte | 2 litri | 1,20 euro |\n",
                "| Pane | 1 kg | 0,80 euro |\n",
                "| Pasta | 500 g | 1,10 euro |"
            )
            .to_string()
        } else {
            concat!(
                "Here is a simple table example:\n",
                "| Product | Quantity | Price |\n",
                "| --- | --- | --- |\n",
                "| Milk | 2 liters | 1.20 euro |\n",
                "| Bread | 1 kg | 0.80 euro |\n",
                "| Pasta | 500 g | 1.10 euro |"
            )
            .to_string()
        };
    }

    "Non ho ricevuto una risposta testuale dal modello. Riprova o cambia modello.".into()
}

fn render_executed_action(response: &DesktopActionResponse, italian: bool) -> String {
    match response.tool_name.as_str() {
        "filesystem.read_text" => render_file_read(response, italian),
        "filesystem.write_text" => {
            let path = action_result_path(response).unwrap_or_else(|| "the file".into());
            if italian {
                format!("Ho scritto il file: {}.", compact_path_label(&path))
            } else {
                format!("I wrote the file: {}.", compact_path_label(&path))
            }
        }
        "filesystem.search" => render_file_search(response, italian),
        "browser.open" => {
            let url = response
                .result
                .as_ref()
                .and_then(|result| result.get("url"))
                .and_then(Value::as_str)
                .unwrap_or("the requested URL");
            if italian {
                format!("Ho aperto l'URL richiesto: {url}.")
            } else {
                format!("I opened the requested URL: {url}.")
            }
        }
        "browser.search" => {
            let query = response
                .result
                .as_ref()
                .and_then(|result| result.get("query"))
                .and_then(Value::as_str)
                .unwrap_or("the requested search");
            if italian {
                format!("Ho aperto la ricerca web per: {query}.")
            } else {
                format!("I opened the web search for: {query}.")
            }
        }
        "desktop.launch_app" => render_desktop_launch(response, italian),
        _ => {
            if italian {
                format!(
                    "Ho eseguito {} correttamente.",
                    friendly_tool_name(&response.tool_name)
                )
            } else {
                format!(
                    "I executed {} successfully.",
                    friendly_tool_name(&response.tool_name)
                )
            }
        }
    }
}

fn render_file_read(response: &DesktopActionResponse, italian: bool) -> String {
    let path = action_result_path(response).unwrap_or_else(|| "the file".into());
    let label = compact_path_label(&path);
    let content = response
        .result
        .as_ref()
        .and_then(|result| result.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let wants_summary = response
        .result
        .as_ref()
        .and_then(|result| result.get("post_processing"))
        .and_then(|post| post.get("mode"))
        .and_then(Value::as_str)
        .map(|mode| mode.eq_ignore_ascii_case("summary"))
        .unwrap_or(false)
        || response
            .result
            .as_ref()
            .and_then(|result| result.get("operation"))
            .and_then(Value::as_str)
            .map(|operation| operation == "read_and_summarize_file")
            .unwrap_or(false);
    let preview = content_preview(content);

    if italian {
        if preview.is_empty() {
            format!("Ho letto il file **{label}**. Il file e' vuoto.")
        } else if wants_summary {
            format!(
                "Ho letto il file **{label}**.\n\nSintesi:\n{}",
                summarize_text_content(content, true)
            )
        } else {
            format!("Ho letto il file **{label}**.\n\nContenuto rilevante:\n{preview}")
        }
    } else if preview.is_empty() {
        format!("I read **{label}**. The file is empty.")
    } else if wants_summary {
        format!(
            "I read **{label}**.\n\nSummary:\n{}",
            summarize_text_content(content, false)
        )
    } else {
        format!("I read **{label}**.\n\nRelevant content:\n{preview}")
    }
}

fn render_file_search(response: &DesktopActionResponse, italian: bool) -> String {
    let matches = response
        .result
        .as_ref()
        .and_then(|result| result.get("matches"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if matches.is_empty() {
        return if italian {
            "Non ho trovato file corrispondenti.".into()
        } else {
            "I did not find matching files.".into()
        };
    }

    let lines = matches
        .iter()
        .take(10)
        .filter_map(Value::as_str)
        .map(|path| format!("- {}", compact_path_label(path)))
        .collect::<Vec<_>>()
        .join("\n");
    if italian {
        format!("Ho trovato questi file:\n{lines}")
    } else {
        format!("I found these files:\n{lines}")
    }
}

fn render_desktop_launch(response: &DesktopActionResponse, italian: bool) -> String {
    let path = response
        .result
        .as_ref()
        .and_then(|result| result.get("path"))
        .and_then(Value::as_str)
        .unwrap_or("the application");
    let args = response
        .result
        .as_ref()
        .and_then(|result| result.get("args"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let url_arg = args.iter().filter_map(Value::as_str).find(|arg| {
        let lower = arg.to_ascii_lowercase();
        lower.starts_with("http://") || lower.starts_with("https://")
    });

    if path.to_ascii_lowercase().contains("chrome") {
        if let Some(url) = url_arg {
            if url.contains("youtube.com/results") {
                return if italian {
                    "Ho aperto Chrome con la ricerca YouTube richiesta.".into()
                } else {
                    "I opened Chrome with the requested YouTube search.".into()
                };
            }
            return if italian {
                format!("Ho aperto Chrome su {url}.")
            } else {
                format!("I opened Chrome at {url}.")
            };
        }
        return if italian {
            "Ho aperto Google Chrome.".into()
        } else {
            "I opened Google Chrome.".into()
        };
    }

    if italian {
        format!("Ho aperto {}.", compact_path_label(path))
    } else {
        format!("I opened {}.", compact_path_label(path))
    }
}

fn strip_leading_code_fenced_payload(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    if !trimmed.starts_with("```") {
        return None;
    }
    let after_open = trimmed.strip_prefix("```")?;
    let newline = after_open.find('\n')?;
    let body_start = newline + 1;
    let after_lang = &after_open[body_start..];
    let fence_end = after_lang.find("```")?;
    let body = &after_lang[..fence_end];
    if !is_internal_payload_str(body) {
        return None;
    }
    Some(after_lang[fence_end + 3..].trim_start().to_string())
}

fn strip_leading_json_payload(text: &str) -> Option<String> {
    let leading = text.len().saturating_sub(text.trim_start().len());
    let trimmed = text.trim_start();
    if !trimmed.starts_with('{') {
        return None;
    }

    let mut stream = Deserializer::from_str(trimmed).into_iter::<Value>();
    let parsed = stream.next()?.ok()?;
    if !is_internal_payload_value(&parsed) {
        return None;
    }
    let offset = stream.byte_offset();
    Some(text[leading + offset..].trim_start().to_string())
}

fn strip_leading_internal_planning(text: &str) -> Option<String> {
    let mut consumed = 0usize;
    let mut stripped_any = false;

    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            consumed += line.len();
            continue;
        }

        if is_internal_planning_line(trimmed) {
            consumed += line.len();
            stripped_any = true;
            continue;
        }

        break;
    }

    stripped_any.then(|| text[consumed..].trim_start().to_string())
}

fn strip_embedded_diagnostic_payloads(text: &str) -> String {
    let mut output = String::new();
    let mut index = 0usize;

    while let Some(relative_start) = text[index..].find('{') {
        let start = index + relative_start;
        let Some(end) = find_balanced_json_end(text, start) else {
            break;
        };
        let candidate = &text[start..end];
        let is_diagnostic = serde_json::from_str::<Value>(candidate)
            .map(|value| is_diagnostic_payload_value(&value))
            .unwrap_or(false);

        if is_diagnostic {
            output.push_str(text[index..start].trim_end());
            if !output.ends_with('\n') && !output.trim().is_empty() {
                output.push('\n');
            }
            index = end;
            continue;
        }

        output.push_str(&text[index..end]);
        index = end;
    }

    output.push_str(&text[index..]);
    output.trim().to_string()
}

fn find_balanced_json_end(text: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;

    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(start + offset + ch.len_utf8());
                }
            }
            _ => {}
        }
    }

    None
}

fn is_diagnostic_payload_value(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.contains_key("classifier_source")
        && object.contains_key("routed_to")
        && object.contains_key("message_excerpt")
}

fn is_internal_planning_line(line: &str) -> bool {
    let lower = line.to_lowercase();
    let mentions_tool = [
        "browser.open",
        "browser.search",
        "filesystem.",
        "screen.analyze",
        "desktop.launch_app",
        "tool",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    if !mentions_tool {
        return false;
    }

    [
        "we need to",
        "we should",
        "we will call",
        "i will call",
        "i need to use",
        "need to use",
        "devo usare",
        "dobbiamo usare",
        "chiameremo",
        "user wants",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
}

#[derive(Debug, Clone)]
struct ParsedTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn normalize_ascii_tables(text: &str) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut index = 0usize;

    while index < lines.len() {
        let line = lines[index];
        if line.trim_start().starts_with("```") {
            let start = index;
            index += 1;
            let mut fenced = Vec::new();
            while index < lines.len() && !lines[index].trim_start().starts_with("```") {
                fenced.push(lines[index]);
                index += 1;
            }
            let has_close = index < lines.len();
            if has_close {
                index += 1;
            }
            if let Some(table) = parse_table_lines(&fenced) {
                output.push(markdown_table(&table));
            } else {
                output.extend(lines[start..index].iter().map(|value| (*value).to_string()));
            }
            continue;
        }

        if is_tableish_line(line) {
            let start = index;
            let mut block = Vec::new();
            while index < lines.len() && is_tableish_line(lines[index]) {
                block.push(lines[index]);
                index += 1;
            }
            if let Some(table) = parse_table_lines(&block) {
                output.push(markdown_table(&table));
            } else {
                output.extend(lines[start..index].iter().map(|value| (*value).to_string()));
            }
            continue;
        }

        output.push(line.to_string());
        index += 1;
    }

    output.join("\n").trim().to_string()
}

fn structured_speech_summary(text: &str) -> Option<String> {
    let (table, preface) = first_table_with_preface(text)?;
    let italian = looks_italian(text);
    let mut parts = Vec::new();

    let preface = collapse_whitespace(&preface);
    if !preface.is_empty() {
        parts.push(preface);
    }

    parts.push(if italian {
        "Ecco un riepilogo della tabella.".to_string()
    } else {
        "Here is a summary of the table.".to_string()
    });

    let row_labels_it = ["Prima riga", "Seconda riga", "Terza riga"];
    let row_labels_en = ["First row", "Second row", "Third row"];
    for (index, row) in table.rows.iter().take(3).enumerate() {
        let label = if italian {
            row_labels_it.get(index).copied().unwrap_or("Riga")
        } else {
            row_labels_en.get(index).copied().unwrap_or("Row")
        };
        let cells = row
            .iter()
            .enumerate()
            .filter_map(|(cell_index, value)| {
                let value = speech_cell(value);
                if value.is_empty() {
                    return None;
                }
                let header = table
                    .headers
                    .get(cell_index)
                    .map(|header| speech_cell(header))
                    .filter(|header| !header.is_empty());
                Some(match header {
                    Some(header) => format!("{header}: {value}"),
                    None => value,
                })
            })
            .collect::<Vec<_>>()
            .join(", ");
        if !cells.is_empty() {
            parts.push(format!("{label}: {cells}."));
        }
    }

    if table.rows.len() > 3 {
        parts.push(if italian {
            format!(
                "Ci sono altre {} righe visibili in chat.",
                table.rows.len() - 3
            )
        } else {
            format!(
                "There are {} more rows visible in chat.",
                table.rows.len() - 3
            )
        });
    }

    Some(suppress_repeated_sentences(&collapse_whitespace(
        &parts.join(" "),
    )))
}

fn first_table_with_preface(text: &str) -> Option<(ParsedTable, String)> {
    let lines = text.lines().collect::<Vec<_>>();
    for start in 0..lines.len() {
        if !is_tableish_line(lines[start]) {
            continue;
        }
        let mut end = start;
        while end < lines.len() && is_tableish_line(lines[end]) {
            end += 1;
        }
        if let Some(table) = parse_table_lines(&lines[start..end]) {
            let preface = lines[..start]
                .iter()
                .filter(|line| !line.trim_start().starts_with("```"))
                .copied()
                .collect::<Vec<_>>()
                .join(" ");
            return Some((table, preface));
        }
    }
    None
}

fn parse_table_lines(lines: &[&str]) -> Option<ParsedTable> {
    let rows = lines
        .iter()
        .filter_map(|line| parse_table_row(line))
        .collect::<Vec<_>>();
    if rows.len() < 2 {
        return None;
    }

    let headers = rows.first()?.clone();
    let rows = rows.into_iter().skip(1).collect::<Vec<_>>();
    if headers.len() < 2 || rows.is_empty() {
        return None;
    }

    Some(ParsedTable { headers, rows })
}

fn parse_table_row(line: &str) -> Option<Vec<String>> {
    let trimmed = line.trim();
    if trimmed.starts_with('+') {
        return None;
    }
    if !trimmed.contains('|') {
        return None;
    }

    let cells = trimmed
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .filter(|cell| !cell.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    if cells.len() < 2 || cells.iter().all(|cell| is_markdown_separator_cell(cell)) {
        return None;
    }

    Some(cells)
}

fn markdown_table(table: &ParsedTable) -> String {
    let headers = table
        .headers
        .iter()
        .map(|cell| escape_table_cell(cell))
        .collect::<Vec<_>>();
    let mut lines = Vec::new();
    lines.push(format!("| {} |", headers.join(" | ")));
    lines.push(format!(
        "| {} |",
        headers
            .iter()
            .map(|_| "---")
            .collect::<Vec<_>>()
            .join(" | ")
    ));
    for row in &table.rows {
        let cells = (0..headers.len())
            .map(|index| {
                row.get(index)
                    .map(|cell| escape_table_cell(cell))
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        lines.push(format!("| {} |", cells.join(" | ")));
    }
    lines.join("\n")
}

fn is_tableish_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.starts_with('+') && trimmed.contains('-') {
        return true;
    }
    trimmed.matches('|').count() >= 2
}

fn is_markdown_separator_cell(cell: &str) -> bool {
    let trimmed = cell.trim();
    !trimmed.is_empty() && trimmed.chars().all(|ch| matches!(ch, '-' | ':' | ' '))
}

fn escape_table_cell(cell: &str) -> String {
    cell.replace('|', "/").trim().to_string()
}

fn speech_cell(value: &str) -> String {
    value
        .replace("€", " euro")
        .replace('|', " ")
        .replace('*', "")
        .replace('`', "")
        .trim()
        .to_string()
}

fn is_internal_payload_str(text: &str) -> bool {
    serde_json::from_str::<Value>(text)
        .map(|value| is_internal_payload_value(&value))
        .unwrap_or(false)
}

fn is_internal_payload_value(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.contains_key("tool_name")
        || (object.contains_key("parameters") && object.contains_key("arguments"))
        || (object.contains_key("parameters") && object.len() <= 3)
        || (object.contains_key("arguments") && object.len() <= 3)
}

fn starts_like_internal_payload(text: &str) -> bool {
    text.starts_with('{')
        || text.starts_with("```json")
        || text.starts_with("```")
        || text.contains("\"tool_name\"")
        || text.contains("'tool_name'")
        || is_internal_planning_start(text)
}

fn is_internal_planning_start(text: &str) -> bool {
    text.lines()
        .find(|line| !line.trim().is_empty())
        .map(is_internal_planning_line)
        .unwrap_or(false)
}

fn sanitize_display_fragment(text: &str) -> String {
    text.replace("\r\n", "\n")
}

fn looks_like_json_line(text: &str) -> bool {
    let trimmed = text.trim();
    (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || trimmed.contains("\"tool_name\"")
        || trimmed.contains("'tool_name'")
}

fn content_preview(content: &str) -> String {
    let mut lines = content
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .take(16)
        .collect::<Vec<_>>();

    if lines.is_empty() {
        return String::new();
    }

    let truncated = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count()
        > lines.len();
    let mut preview = lines.drain(..).collect::<Vec<_>>().join("\n");
    if preview.chars().count() > 2_000 {
        preview = preview.chars().take(2_000).collect::<String>();
        preview.push_str("\n...");
    } else if truncated {
        preview.push_str("\n...");
    }
    preview
}

fn summarize_text_content(content: &str, italian: bool) -> String {
    let lines = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(6)
        .collect::<Vec<_>>();

    if lines.is_empty() {
        return if italian {
            "Il file non contiene testo leggibile da riassumere.".into()
        } else {
            "The file does not contain readable text to summarize.".into()
        };
    }

    let joined = lines.join(" ");
    let mut summary = joined.chars().take(700).collect::<String>();
    if joined.chars().count() > summary.chars().count() {
        summary.push_str("...");
    }

    if italian {
        format!(
            "Il contenuto principale riguarda: {}",
            collapse_whitespace(&summary)
        )
    } else {
        format!(
            "The main content is about: {}",
            collapse_whitespace(&summary)
        )
    }
}

fn action_result_path(response: &DesktopActionResponse) -> Option<String> {
    response
        .result
        .as_ref()
        .and_then(|result| result.get("path"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn compact_path_label(path: &str) -> String {
    file_name(path)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.to_string())
}

fn file_name(path: &str) -> Option<&str> {
    Path::new(path).file_name().and_then(|name| name.to_str())
}

fn friendly_tool_name(tool_name: &str) -> &'static str {
    match tool_name {
        "filesystem.read_text" => "lettura file",
        "filesystem.write_text" => "scrittura file",
        "filesystem.search" => "ricerca file",
        "browser.open" => "apertura browser",
        "browser.search" => "ricerca web",
        "desktop.launch_app" => "apertura applicazione",
        "screen.analyze" => "analisi schermo",
        _ => "azione desktop",
    }
}

fn looks_italian(value: &str) -> bool {
    let lower = value.to_lowercase();
    [
        "puoi",
        "mi ",
        "apri",
        "aprimi",
        "cerca",
        "crea",
        "creami",
        "leggi",
        "schermo",
        "file",
        "contenente",
        "finestra",
        "scheda",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn suppress_repeated_sentences(value: &str) -> String {
    let mut output = Vec::new();
    let mut previous = String::new();

    for sentence in value.split_terminator(['.', '!', '?']) {
        let normalized = collapse_whitespace(sentence).trim().to_string();
        if normalized.is_empty() {
            continue;
        }
        let fingerprint = normalized.to_lowercase();
        if fingerprint == previous {
            continue;
        }
        previous = fingerprint;
        output.push(format!("{normalized}."));
    }

    if output.is_empty() {
        collapse_whitespace(value)
    } else {
        output.join(" ")
    }
}

fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::{
        fallback_display_for_empty_response, present_display_text, sanitize_display_text,
        speech_safe_text,
    };

    #[test]
    fn display_sanitizer_strips_leading_tool_payload() {
        let raw = r#"{ "tool_name": "filesystem.read_text", "arguments": { "path": "~/Desktop/a.txt" } }
Ho letto il file."#;

        assert_eq!(sanitize_display_text(raw), "Ho letto il file.");
    }

    #[test]
    fn speech_text_removes_markdown_and_json_symbols() {
        let raw = "**Risultato:**\n- `filesystem.read_text`\n```json\n{\"tool_name\":\"x\"}\n```";
        let spoken = speech_safe_text(raw);

        assert!(!spoken.contains('{'));
        assert!(!spoken.contains('`'));
        assert!(!spoken.contains('*'));
        assert!(spoken.contains("Risultato"));
    }

    #[test]
    fn display_sanitizer_strips_leading_internal_planning() {
        let raw = "We need to use browser.search.\nSto cercando Coca-Cola su Google.";

        assert_eq!(
            sanitize_display_text(raw),
            "Sto cercando Coca-Cola su Google."
        );
    }

    #[test]
    fn ascii_table_is_presented_as_markdown_table_and_spoken_as_summary() {
        let raw = "Certo, ecco un esempio:\n```\n+----------+----------+--------+\n| Prodotto | Quantita | Prezzo |\n+----------+----------+--------+\n| Latte    | 2 litri  | 1,20 € |\n| Pane     | 1 kg     | 0,80 € |\n+----------+----------+--------+\n```";
        let display = present_display_text(raw);
        let spoken = speech_safe_text(raw);

        assert!(display.contains("| Prodotto | Quantita | Prezzo |"));
        assert!(!display.contains("```"));
        assert!(!spoken.contains('+'));
        assert!(!spoken.contains('|'));
        assert!(spoken.contains("Latte"));
        assert!(spoken.contains("Prezzo: 1,20"));
    }

    #[test]
    fn display_sanitizer_removes_embedded_route_diagnostic_json() {
        let raw = r#"Non ho ricevuto una risposta testuale dal modello:
{
  "message_excerpt": "mi fai un esempio di tabella",
  "classifier_source": "classifier_unavailable",
  "routed_to": "normal_llm"
}"#;

        let display = present_display_text(raw);
        assert!(!display.contains("classifier_source"));
        assert!(!display.contains("routed_to"));
        assert!(display.starts_with("Non ho ricevuto"));
    }

    #[test]
    fn empty_table_prompt_gets_clean_deterministic_fallback() {
        let fallback = fallback_display_for_empty_response("astra, mi fai un esempio di tabella?");
        assert!(fallback.contains("| Prodotto | Quantita | Prezzo |"));
        assert!(!fallback.contains("classifier_source"));
    }
}
