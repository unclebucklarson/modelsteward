//! pi coding-agent connector (Connections p2, first external request —
//! built 2026-08-30). pi reads `~/.pi/agent/models.json`, a
//! `{ providers: { <key>: { baseUrl, api, models: [...] } } }` registry
//! — schema verified against earendil-works/pi
//! `packages/coding-agent/src/core/model-config.ts`, sample = the live
//! install on this machine.
//!
//! Why this exists at all: pi has NATIVE llama.cpp router support
//! (`/login llama.cpp`), but its provider takes context from the
//! router's `/models` metadata (`n_ctx ?? n_ctx_train ?? 128000`), and
//! our router reports `n_ctx: None` for unloaded models — measured on
//! 2026-08-30 — so pi assumes 128k for everything. This provider block
//! is where the MEASURED context windows flow in.
//!
//! Ownership rules (same spirit as the OpenCode connector):
//! - We own ONLY `providers.modelsteward`. Every other byte of the
//!   document round-trips untouched.
//! - `~/.pi/agent/` absent = pi not installed = skip, never an error.
//! - Always back up before writing.
//! - pi strips `//` comments on read, but our JSON round-trip would
//!   DELETE them — a file that fails a strict parse is refused with an
//!   honest error instead of silently flattened. (pi itself writes
//!   plain JSON; comments here are rare and deliberate.)

use crate::core::opencode::{DesiredModel, safety_context};
use crate::core::settings;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// The provider key we own inside models.json.
pub const PROVIDER_KEY: &str = "modelsteward";

pub fn default_models_path() -> PathBuf {
    settings::real_home().join(".pi/agent/models.json")
}

/// pi is "installed" when its agent dir exists — models.json itself may
/// legitimately not exist yet (we create it then).
pub fn pi_present(models_path: &Path) -> bool {
    models_path.parent().is_some_and(|d| d.is_dir())
}

#[derive(Debug, Default, PartialEq)]
pub struct PiSyncReport {
    pub added: Vec<String>,
    pub updated: Vec<String>,
    /// Entries dropped from OUR provider because their measurement is
    /// gone. Measurements persist across router restarts, so a stopped
    /// router never causes removals (house rule).
    pub removed: Vec<String>,
    pub created_file: bool,
    /// ~/.pi/agent doesn't exist — pi isn't installed; nothing written.
    pub skipped_missing: bool,
}

/// Our provider block: measured context (with the same 5% safety
/// haircut opencode.json gets), measured vision, maxTokens mirroring
/// pi's own native-provider convention (= contextWindow).
pub fn provider_json(base_url: &str, desired: &[DesiredModel]) -> serde_json::Value {
    let mut sorted: Vec<&DesiredModel> = desired.iter().collect();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    let models: Vec<serde_json::Value> = sorted
        .iter()
        .map(|d| {
            let ctx = safety_context(d.context);
            let mut input = vec!["text"];
            if d.vision {
                input.push("image");
            }
            serde_json::json!({
                "id": d.id,
                "name": d.display_name,
                "contextWindow": ctx,
                "maxTokens": ctx,
                "input": input,
            })
        })
        .collect();
    serde_json::json!({
        "name": "modelsteward (measured llama.cpp)",
        "api": "openai-completions",
        // llama-server without --api-key ignores it; pi's schema wants
        // a non-empty string and the wild sample uses the same trick
        // ("ollama" for ollama).
        "apiKey": "modelsteward",
        "baseUrl": base_url,
        "models": models,
    })
}

/// Sync our provider into models.json. Reads, diffs, backs up, writes.
pub fn sync_file(path: &Path, base_url: &str, desired: &[DesiredModel]) -> Result<PiSyncReport> {
    let mut report = PiSyncReport::default();
    if !pi_present(path) {
        report.skipped_missing = true;
        return Ok(report);
    }
    let mut doc: serde_json::Value = match std::fs::read_to_string(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            report.created_file = true;
            serde_json::json!({ "providers": {} })
        }
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        Ok(text) => serde_json::from_str(&text).with_context(|| {
            format!(
                "{} isn't plain JSON (comments?) — refusing to rewrite it and lose \
                 your annotations; remove them or sync by hand",
                path.display()
            )
        })?,
    };
    let new_block = provider_json(base_url, desired);
    let old_models: std::collections::BTreeMap<String, serde_json::Value> = doc
        .get("providers")
        .and_then(|p| p.get(PROVIDER_KEY))
        .and_then(|b| b.get("models"))
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| Some((m.get("id")?.as_str()?.to_string(), m.clone())))
                .collect()
        })
        .unwrap_or_default();
    for m in new_block["models"].as_array().unwrap() {
        let id = m["id"].as_str().unwrap().to_string();
        match old_models.get(&id) {
            None => report.added.push(id),
            Some(old) if old != m => report.updated.push(id),
            Some(_) => {}
        }
    }
    for id in old_models.keys() {
        if !desired.iter().any(|d| &d.id == id) {
            report.removed.push(id.clone());
        }
    }
    let unchanged = report.added.is_empty()
        && report.updated.is_empty()
        && report.removed.is_empty()
        && !report.created_file
        && doc.get("providers").and_then(|p| p.get(PROVIDER_KEY)).is_some();
    if unchanged {
        return Ok(report);
    }
    if !report.created_file {
        std::fs::copy(path, backup_path(path))
            .with_context(|| format!("backing up {}", path.display()))?;
    }
    doc.as_object_mut()
        .context("models.json root is not an object")?
        .entry("providers")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context("models.json 'providers' is not an object")?
        .insert(PROVIDER_KEY.to_string(), new_block);
    let text = serde_json::to_string_pretty(&doc)? + "\n";
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(report)
}

pub fn backup_path(path: &Path) -> PathBuf {
    path.with_extension("json.modelsteward.bak")
}

/// What our provider block currently declares — for the Connections
/// mirror. (id, contextWindow) pairs; empty when pi absent or unsynced.
pub fn configured_models(path: &Path) -> Vec<(String, u64)> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|doc| {
            let arr = doc
                .get("providers")?
                .get(PROVIDER_KEY)?
                .get("models")?
                .as_array()?
                .iter()
                .filter_map(|m| {
                    Some((
                        m.get("id")?.as_str()?.to_string(),
                        m.get("contextWindow")?.as_u64()?,
                    ))
                })
                .collect();
            Some(arr)
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desired(id: &str, ctx: u64, vision: bool) -> DesiredModel {
        DesiredModel {
            id: id.into(),
            display_name: format!("{id} (llama.cpp)"),
            context: ctx,
            tool_call: Some(true),
            vision,
        }
    }

    /// The live install's models.json, verbatim (2026-08-30) — the
    /// user's hand-configured ollama provider must survive our sync
    /// byte-for-byte at the value level.
    const REAL_WORLD: &str = r#"{
  "providers": {
    "ollama": {
      "api": "openai-completions",
      "apiKey": "ollama",
      "baseUrl": "http://127.0.0.1:11434/v1",
      "models": [
        {
          "_launch": true,
          "contextWindow": 131072,
          "id": "gemma4",
          "input": [
            "text",
            "image"
          ],
          "reasoning": true
        }
      ]
    }
  }
}"#;

    #[test]
    fn syncs_beside_the_real_ollama_provider_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        std::fs::write(&path, REAL_WORLD).unwrap();
        let want = [
            desired("qwen3.8-27b-ud-q4_k_xl", 113920, false),
            desired("gpt-oss-20b-f16", 131072, false),
        ];
        let r = sync_file(&path, "http://127.0.0.1:8080/v1", &want).unwrap();
        assert_eq!(r.added.len(), 2, "{r:?}");
        assert!(backup_path(&path).exists(), "backup before every rewrite");
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        // The user's provider: value-identical.
        let orig: serde_json::Value = serde_json::from_str(REAL_WORLD).unwrap();
        assert_eq!(doc["providers"]["ollama"], orig["providers"]["ollama"]);
        // Ours: measured ctx with the safety haircut, pi's field names.
        let ours = &doc["providers"]["modelsteward"];
        assert_eq!(ours["api"], "openai-completions");
        assert_eq!(ours["baseUrl"], "http://127.0.0.1:8080/v1");
        let m = &ours["models"][1]; // sorted by id: gpt-oss first
        assert_eq!(m["id"], "qwen3.8-27b-ud-q4_k_xl");
        assert_eq!(m["contextWindow"], safety_context(113920));
        assert_eq!(m["maxTokens"], m["contextWindow"]);
        assert_eq!(m["input"], serde_json::json!(["text"]));
        // Idempotent: a second sync reports nothing and rewrites nothing.
        let before = std::fs::read_to_string(&path).unwrap();
        let r2 = sync_file(&path, "http://127.0.0.1:8080/v1", &want).unwrap();
        assert_eq!(r2, PiSyncReport::default(), "{r2:?}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    }

    #[test]
    fn context_changes_update_and_lost_measurements_remove() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        std::fs::write(&path, "{\"providers\":{}}").unwrap();
        let v1 = [desired("a", 100_000, false), desired("b", 50_000, true)];
        sync_file(&path, "http://x/v1", &v1).unwrap();
        // b's vision made it in.
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            doc["providers"]["modelsteward"]["models"][1]["input"],
            serde_json::json!(["text", "image"])
        );
        // a remeasured bigger, b gone entirely.
        let v2 = [desired("a", 120_000, false)];
        let r = sync_file(&path, "http://x/v1", &v2).unwrap();
        assert_eq!(r.updated, vec!["a".to_string()]);
        assert_eq!(r.removed, vec!["b".to_string()]);
        assert_eq!(configured_models(&path), vec![("a".into(), safety_context(120_000))]);
    }

    #[test]
    fn missing_dir_skips_missing_file_creates() {
        let dir = tempfile::tempdir().unwrap();
        let absent = dir.path().join("nope/models.json");
        let r = sync_file(&absent, "http://x/v1", &[desired("a", 1000, false)]).unwrap();
        assert!(r.skipped_missing && !absent.exists());
        let fresh = dir.path().join("models.json");
        let r = sync_file(&fresh, "http://x/v1", &[desired("a", 1000, false)]).unwrap();
        assert!(r.created_file && fresh.exists());
        assert!(!backup_path(&fresh).exists(), "nothing to back up on create");
    }

    #[test]
    fn commented_files_are_refused_not_flattened() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        std::fs::write(&path, "{\n  // my note\n  \"providers\": {}\n}").unwrap();
        let e = sync_file(&path, "http://x/v1", &[desired("a", 1000, false)]).unwrap_err();
        assert!(format!("{e:#}").contains("comments"), "{e:#}");
        assert!(std::fs::read_to_string(&path).unwrap().contains("my note"));
    }
}
