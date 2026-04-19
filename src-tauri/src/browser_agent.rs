use serde_json::json;
use std::process::Command;

#[derive(Clone, Default)]
pub struct BrowserAgent;

impl BrowserAgent {
    pub fn open(&self, url: &str) -> Result<serde_json::Value, String> {
        open_with_system(url)?;
        Ok(json!({"url": url, "opened": true}))
    }

    pub fn search(&self, query: &str) -> Result<serde_json::Value, String> {
        let encoded = url_encode(query);
        let url = format!("https://www.google.com/search?q={encoded}");
        self.open(&url)?;
        Ok(json!({"query": query, "url": url, "opened": true}))
    }
}

fn open_with_system(target: &str) -> Result<(), String> {
    if cfg!(target_os = "windows") {
        Command::new("cmd").args(["/C", "start", "", target]).spawn().map_err(|e| e.to_string())?;
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(target).spawn().map_err(|e| e.to_string())?;
    } else {
        Command::new("xdg-open").arg(target).spawn().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn url_encode(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            b' ' => "+".to_string(),
            _ => format!("%{:02X}", b),
        })
        .collect::<String>()
}
