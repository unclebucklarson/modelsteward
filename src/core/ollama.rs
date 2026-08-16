//! Ollama as a peer provider: never managed, always observed.
//!
//! Two facts matter to this app: what Ollama *has* (its library, which the
//! model scanner already reads from disk) and what it currently *holds in
//! VRAM* (`/api/ps`) — because a 20GB Ollama resident plus a 20GB router
//! load on one 24GB card is the single most common way local setups fall
//! over. We surface the collision; the user decides who wins.

use anyhow::{Context, Result};
use serde::Serialize;

pub const DEFAULT_PORT: u16 = 11434;

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct OllamaStatus {
    /// Daemon answered /api/ps. False ≠ "models gone" (see CLAUDE.md).
    pub reachable: bool,
    /// Models currently resident, with how much VRAM each holds.
    pub loaded: Vec<LoadedModel>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LoadedModel {
    pub name: String,
    pub size_vram: u64,
}

/// Parse `/api/ps`: `{"models":[{"name":…,"size_vram":…},…]}`.
pub fn parse_ps(body: &serde_json::Value) -> Vec<LoadedModel> {
    body.get("models")
        .and_then(|m| m.as_array())
        .map(|models| {
            models
                .iter()
                .filter_map(|m| {
                    Some(LoadedModel {
                        name: m.get("name")?.as_str()?.to_string(),
                        size_vram: m.get("size_vram").and_then(|v| v.as_u64()).unwrap_or(0),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn probe(port: u16) -> OllamaStatus {
    match fetch_ps(port) {
        Ok(loaded) => OllamaStatus {
            reachable: true,
            loaded,
        },
        Err(_) => OllamaStatus::default(),
    }
}

fn fetch_ps(port: u16) -> Result<Vec<LoadedModel>> {
    let body: serde_json::Value = ureq::get(&format!("http://127.0.0.1:{port}/api/ps"))
        .timeout(std::time::Duration::from_secs(2))
        .call()
        .context("ollama /api/ps")?
        .into_json()?;
    Ok(parse_ps(&body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ps_with_vram_sizes() {
        let body = serde_json::json!({"models": [
            {"name": "ornith:35b", "size": 22_000_000_000u64, "size_vram": 21_000_000_000u64},
            {"name": "tiny:latest", "size": 100, "details": {}}
        ]});
        let loaded = parse_ps(&body);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "ornith:35b");
        assert_eq!(loaded[0].size_vram, 21_000_000_000);
        assert_eq!(loaded[1].size_vram, 0, "missing size_vram is 0, not skipped");
    }

    #[test]
    fn empty_or_odd_body_is_no_models() {
        assert!(parse_ps(&serde_json::json!({})).is_empty());
        assert!(parse_ps(&serde_json::json!({"models": null})).is_empty());
    }
}
