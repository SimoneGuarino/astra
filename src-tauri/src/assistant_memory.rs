use crate::desktop_agent_types::{
    DesktopActionResponse, DesktopActionStatus, ScreenAnalysisResult,
};
use serde_json::Value;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub struct RecentArtifactMemory {
    inner: Arc<Mutex<RecentArtifacts>>,
}

#[derive(Debug, Default, Clone)]
struct RecentArtifacts {
    last_file_read: Option<RecentFileArtifact>,
    last_screen_analysis: Option<RecentScreenArtifact>,
    last_browser_action: Option<RecentBrowserArtifact>,
}

#[derive(Debug, Clone)]
struct RecentFileArtifact {
    path: String,
    content: String,
}

#[derive(Debug, Clone)]
struct RecentScreenArtifact {
    image_path: String,
    answer: String,
    model: String,
}

#[derive(Debug, Clone)]
struct RecentBrowserArtifact {
    query: Option<String>,
    url: Option<String>,
}

impl RecentArtifactMemory {
    pub fn remember_action_response(&self, response: &DesktopActionResponse) {
        if !matches!(response.status, DesktopActionStatus::Executed) {
            return;
        }

        let Some(result) = response.result.as_ref() else {
            return;
        };

        let mut inner = self.inner.lock().expect("recent artifact memory poisoned");
        match response.tool_name.as_str() {
            "filesystem.read_text" => {
                let Some(path) = value_str(result, "path") else {
                    return;
                };
                let Some(content) = value_str(result, "content") else {
                    return;
                };
                inner.last_file_read = Some(RecentFileArtifact {
                    path: path.to_string(),
                    content: cap_string(content, 128 * 1024),
                });
            }
            "browser.open" => {
                inner.last_browser_action = Some(RecentBrowserArtifact {
                    query: None,
                    url: value_str(result, "url").map(ToOwned::to_owned),
                });
            }
            "browser.search" => {
                inner.last_browser_action = Some(RecentBrowserArtifact {
                    query: value_str(result, "query").map(ToOwned::to_owned),
                    url: value_str(result, "url").map(ToOwned::to_owned),
                });
            }
            "desktop.launch_app" => {
                let url = result
                    .get("args")
                    .and_then(Value::as_array)
                    .and_then(|args| {
                        args.iter()
                            .filter_map(Value::as_str)
                            .find(|arg| arg.starts_with("http://") || arg.starts_with("https://"))
                    })
                    .map(ToOwned::to_owned);
                if url.is_some() {
                    inner.last_browser_action = Some(RecentBrowserArtifact { query: None, url });
                }
            }
            _ => {}
        }
    }

    pub fn remember_screen_analysis(&self, result: &ScreenAnalysisResult) {
        let mut inner = self.inner.lock().expect("recent artifact memory poisoned");
        inner.last_screen_analysis = Some(RecentScreenArtifact {
            image_path: result.image_path.clone(),
            answer: cap_string(&result.answer, 32 * 1024),
            model: result.model.clone(),
        });
    }

    pub fn answer_followup(&self, message: &str) -> Option<String> {
        let lower = message.to_lowercase();
        let inner = self.inner.lock().expect("recent artifact memory poisoned");

        if references_file(&lower) {
            if let Some(file) = inner.last_file_read.as_ref() {
                return Some(answer_file_followup(file, message));
            }
        }

        if references_screen(&lower) {
            if let Some(screen) = inner.last_screen_analysis.as_ref() {
                return Some(format!(
                    "Nell'ultima analisi della schermata con {} ho rilevato: {}\n\nRiferimento immagine: {}",
                    screen.model, screen.answer, screen.image_path
                ));
            }
        }

        if references_browser_result(&lower) {
            if let Some(browser) = inner.last_browser_action.as_ref() {
                let target = browser
                    .query
                    .as_deref()
                    .or(browser.url.as_deref())
                    .unwrap_or("l'ultima azione browser");
                return Some(format!("L'ultimo riferimento browser e': {target}."));
            }
        }

        None
    }
}

fn answer_file_followup(file: &RecentFileArtifact, message: &str) -> String {
    let lower = message.to_lowercase();
    let file_name = file
        .path
        .rsplit(|ch| ch == '\\' || ch == '/')
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or(&file.path);

    if lower.contains("password") || lower.contains("pwd") || lower.contains("pass") {
        let keywords = extract_lookup_keywords(message);
        if let Some(line) = find_secret_line(&file.content, &keywords) {
            return format!(
                "Nel file **{file_name}** ho trovato questa voce collegata alla richiesta:\n{line}"
            );
        }
        return format!(
            "Ho ancora il riferimento al file **{file_name}**, ma non trovo una password collegata chiaramente a {}.",
            if keywords.is_empty() {
                "quella richiesta".into()
            } else {
                keywords.join(", ")
            }
        );
    }

    if lower.contains("riassum") || lower.contains("summary") || lower.contains("conten") {
        let preview = file
            .content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .take(8)
            .collect::<Vec<_>>()
            .join("\n");
        if preview.is_empty() {
            return format!("Il file **{file_name}** e' vuoto.");
        }
        return format!(
            "Nel file **{file_name}** ho trovato queste informazioni principali:\n{preview}"
        );
    }

    format!(
        "Ho ancora il riferimento al file **{file_name}**. Dimmi quale dato vuoi estrarre da quel file."
    )
}

fn find_secret_line(content: &str, keywords: &[String]) -> Option<String> {
    let lines = content.lines().collect::<Vec<_>>();
    let lower_lines = lines
        .iter()
        .map(|line| line.to_lowercase())
        .collect::<Vec<_>>();

    let mut best: Option<(usize, usize)> = None;
    for (index, lower_line) in lower_lines.iter().enumerate() {
        let mut score = 0;
        if lower_line.contains("password")
            || lower_line.contains("pwd")
            || lower_line.contains("pass")
        {
            score += 3;
        }
        for keyword in keywords {
            if lower_line.contains(keyword) {
                score += 2;
            }
        }
        if score > 0
            && best
                .map(|(_, best_score)| score > best_score)
                .unwrap_or(true)
        {
            best = Some((index, score));
        }
    }

    let (index, _) = best?;
    let start = index.saturating_sub(1);
    let end = (index + 2).min(lines.len());
    let block = lines[start..end]
        .iter()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    (!block.is_empty()).then_some(block)
}

fn extract_lookup_keywords(message: &str) -> Vec<String> {
    let stopwords = [
        "puoi",
        "dirmi",
        "dimmi",
        "password",
        "pwd",
        "pass",
        "contenente",
        "contenuta",
        "contenuto",
        "quel",
        "quella",
        "quello",
        "file",
        "documento",
        "nel",
        "nella",
        "del",
        "della",
        "di",
        "the",
        "that",
        "from",
        "for",
    ];
    message
        .split(|ch: char| !ch.is_alphanumeric())
        .map(str::trim)
        .filter(|token| token.chars().count() >= 2)
        .map(|token| token.to_lowercase())
        .filter(|token| !stopwords.contains(&token.as_str()))
        .take(4)
        .collect()
}

fn references_file(lower: &str) -> bool {
    [
        "quel file",
        "quello file",
        "quel documento",
        "quel testo",
        "in quel file",
        "contenente in quel file",
        "file di prima",
        "documento di prima",
        "that file",
        "that document",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn references_screen(lower: &str) -> bool {
    [
        "quella schermata",
        "quello schermo",
        "schermata di prima",
        "screen di prima",
        "that screen",
        "previous screen",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn references_browser_result(lower: &str) -> bool {
    [
        "quella ricerca",
        "quello di prima",
        "risultato di prima",
        "that search",
        "previous search",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn value_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn cap_string(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut output = String::new();
    for ch in value.chars() {
        if output.len() + ch.len_utf8() > max_bytes {
            break;
        }
        output.push(ch);
    }
    output
}
