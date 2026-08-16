//! opencode.json sync: make the `llamacpp` provider block mirror what the
//! router actually serves, with measured (never guessed) context limits.
//!
//! All edits go through the comment-preserving jsonc module. The file is
//! backed up before every write, and orphans (configured models the router
//! no longer lists) are *reported*, not removed — commenting them out is a
//! decision the user takes in the UI (M4), not a side effect of a sync.

use crate::core::jsonc;
use anyhow::{Context, Result};
use serde_json::json;
use std::path::{Path, PathBuf};

/// The provider id this app owns inside opencode.json. Other providers
/// (ollama, cloud) are never touched.
pub const PROVIDER_ID: &str = "llamacpp";

pub fn default_config_path() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".config")
        })
        .join("opencode/opencode.json")
}

/// One model entry we want present under `provider.llamacpp.models`.
#[derive(Debug, Clone)]
pub struct DesiredModel {
    /// The router alias — what OpenCode sends as the "model" field, which
    /// is exactly what the router routes on.
    pub id: String,
    pub display_name: String,
    /// Measured settled context (spike 2: from `/props`, not the GGUF).
    pub context: u64,
}

impl DesiredModel {
    /// The JSON written for a fresh entry. `tool_call: true` because agent
    /// use is the point of this provider; a model that can't call tools is
    /// the user's to demote by hand (we never overwrite their edit: syncs
    /// of existing entries only patch `limit.context`).
    fn entry(&self) -> serde_json::Value {
        json!({
            "name": self.display_name,
            "tool_call": true,
            "limit": {
                "context": self.context,
                "output": self.context.div_euclid(2).min(32_768),
            }
        })
    }

    /// The minimal patch for an entry that already exists: only the measured
    /// context. Everything the user hand-tuned (temperature, name, even
    /// tool_call) stays byte-for-byte.
    fn patch(&self) -> serde_json::Value {
        json!({ "limit": { "context": self.context } })
    }
}

#[derive(Debug, Default)]
pub struct SyncReport {
    pub added: Vec<String>,
    pub updated: Vec<String>,
    /// Configured under our provider but not desired — candidates for
    /// comment-out, surfaced to the user rather than acted on.
    pub orphans: Vec<String>,
}

/// Compute the new source text. Pure with respect to the filesystem —
/// reading, backup, and writing are the caller's (see [`sync_file`]).
pub fn sync_source(
    source: &str,
    base_url: &str,
    desired: &[DesiredModel],
) -> Result<(String, SyncReport)> {
    let scaffold = json!({
        "npm": "@ai-sdk/openai-compatible",
        "name": "llama.cpp (llamacppcodeconf)",
        "options": { "baseURL": base_url },
        "models": {}
    });

    let mut source = jsonc::ensure_models_container(source, PROVIDER_ID, &scaffold)
        .context("ensuring provider.llamacpp.models exists")?;

    let existing = existing_model_ids(&source)?;
    let mut report = SyncReport::default();

    for d in desired {
        if existing.contains(&d.id) {
            source = jsonc::merge_model(&source, PROVIDER_ID, &d.id, &d.patch())
                .with_context(|| format!("updating {}", d.id))?;
            report.updated.push(d.id.clone());
        } else {
            source = jsonc::add_model(&source, PROVIDER_ID, &d.id, &d.entry())
                .with_context(|| format!("adding {}", d.id))?;
            report.added.push(d.id.clone());
        }
    }

    let desired_ids: std::collections::HashSet<_> = desired.iter().map(|d| d.id.as_str()).collect();
    report.orphans = existing
        .into_iter()
        .filter(|id| !desired_ids.contains(id.as_str()))
        .collect();

    Ok((source, report))
}

/// Model ids currently under `provider.llamacpp.models` (empty when the
/// container is missing).
fn existing_model_ids(source: &str) -> Result<Vec<String>> {
    let value = jsonc_parser::parse_to_serde_value(source, &Default::default())
        .map_err(|e| anyhow::anyhow!("parsing opencode.json: {e}"))?
        .unwrap_or(serde_json::Value::Null);
    Ok(value
        .pointer(&format!("/provider/{PROVIDER_ID}/models"))
        .and_then(|m| m.as_object())
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default())
}

/// Read → sync → backup → write. The backup is a single rotating
/// `<name>.lcc.bak` next to the file (this user's config dir already
/// carries nine ad-hoc backups; we won't add a growing pile of our own).
pub fn sync_file(path: &Path, base_url: &str, desired: &[DesiredModel]) -> Result<SyncReport> {
    let original = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let (updated, report) = sync_source(&original, base_url, desired)?;
    write_backed_up(path, &original, &updated)?;
    Ok(report)
}

fn write_backed_up(path: &Path, original: &str, updated: &str) -> Result<()> {
    if updated == original {
        return Ok(());
    }
    let backup = path.with_extension("json.lcc.bak");
    std::fs::write(&backup, original)
        .with_context(|| format!("writing backup {}", backup.display()))?;
    let tmp = path.with_extension("json.lcc.tmp");
    std::fs::write(&tmp, updated).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).context("moving new config into place")?;
    Ok(())
}

/// One entry as it currently stands in opencode.json — what the OpenCode
/// tab displays. Values are whatever the file says (which may be the
/// user's hand edits), not what we'd write.
#[derive(Debug, Clone, Serialize)]
pub struct ConfiguredModel {
    pub id: String,
    pub name: Option<String>,
    pub context: Option<u64>,
    pub output: Option<u64>,
    pub tool_call: Option<bool>,
}

use serde::Serialize;

/// Read the current `provider.llamacpp.models` entries with their values.
pub fn configured_models(path: &Path) -> Result<Vec<ConfiguredModel>> {
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let value = jsonc_parser::parse_to_serde_value(&source, &Default::default())
        .map_err(|e| anyhow::anyhow!("parsing opencode.json: {e}"))?
        .unwrap_or(serde_json::Value::Null);
    Ok(value
        .pointer(&format!("/provider/{PROVIDER_ID}/models"))
        .and_then(|m| m.as_object())
        .map(|m| {
            m.iter()
                .map(|(id, v)| ConfiguredModel {
                    id: id.clone(),
                    name: v.get("name").and_then(|x| x.as_str()).map(String::from),
                    context: v.pointer("/limit/context").and_then(|x| x.as_u64()),
                    output: v.pointer("/limit/output").and_then(|x| x.as_u64()),
                    tool_call: v.get("tool_call").and_then(|x| x.as_bool()),
                })
                .collect()
        })
        .unwrap_or_default())
}

/// Model ids under our provider that aren't in `keep` — comment-out
/// candidates for the UI to offer.
pub fn orphans_in_file(path: &Path, keep: &[String]) -> Result<Vec<String>> {
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let keep: std::collections::HashSet<_> = keep.iter().map(String::as_str).collect();
    Ok(existing_model_ids(&source)?
        .into_iter()
        .filter(|id| !keep.contains(id.as_str()))
        .collect())
}

/// Comment one orphan out in place (never deletes; the entry stays visible
/// in the file under an explanatory note). Backed up like every write.
pub fn comment_out_in_file(path: &Path, model_id: &str) -> Result<()> {
    let original = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let note = "Commented out by llamacppcodeconf: not in the current router \npreset / never measured. Uncomment to restore, or delete these \nlines to discard permanently.";
    let updated = jsonc::comment_out_model(&original, PROVIDER_ID, model_id, note)
        .with_context(|| format!("commenting out {model_id}"))?;
    write_backed_up(path, &original, &updated)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        // the user's own note about this model
        "qwen3.6-27b-ud-q5_k_xl": {
          "name": "hand-tuned name",
          "tool_call": true,
          "temperature": 0.6,
          "limit": {
            "context": 262144,
            "output": 32768
          }
        },
        "stale-model": {
          "name": "no longer served"
        }
      }
    },
    "ollama": {
      "options": { "baseURL": "http://127.0.0.1:11434/v1" },
      "models": { "ornith:35b": { "name": "untouchable" } }
    }
  }
}"#;

    fn desired(id: &str, ctx: u64) -> DesiredModel {
        DesiredModel {
            id: id.into(),
            display_name: format!("{id} (llama.cpp)"),
            context: ctx,
        }
    }

    #[test]
    fn updates_existing_entry_only_touching_measured_context() {
        let (out, report) =
            sync_source(SAMPLE, "http://127.0.0.1:8080/v1", &[desired("qwen3.6-27b-ud-q5_k_xl", 72_960)])
                .unwrap();
        assert_eq!(report.updated, vec!["qwen3.6-27b-ud-q5_k_xl"]);
        assert!(out.contains(r#""context": 72960"#));
        // Hand-set values and the user's comment survive.
        assert!(out.contains("hand-tuned name"));
        assert!(out.contains(r#""temperature": 0.6"#));
        assert!(out.contains("the user's own note"));
        // Untouched entry keeps its old output limit.
        assert!(out.contains(r#""output": 32768"#));
    }

    #[test]
    fn adds_new_entry_and_reports_orphans() {
        let (out, report) =
            sync_source(SAMPLE, "http://127.0.0.1:8080/v1", &[desired("gemma4-latest", 36_096)])
                .unwrap();
        assert_eq!(report.added, vec!["gemma4-latest"]);
        assert!(out.contains(r#""gemma4-latest""#));
        assert!(out.contains(r#""context": 36096"#));
        // output = min(ctx/2, 32768)
        assert!(out.contains(r#""output": 18048"#));
        // Both pre-existing models are now orphans — reported, not removed.
        let mut orphans = report.orphans.clone();
        orphans.sort();
        assert_eq!(orphans, vec!["qwen3.6-27b-ud-q5_k_xl", "stale-model"]);
        assert!(out.contains("stale-model"), "orphans stay in the file");
    }

    #[test]
    fn never_touches_other_providers() {
        let (out, _) =
            sync_source(SAMPLE, "http://127.0.0.1:8080/v1", &[desired("x", 4096)]).unwrap();
        assert!(out.contains(r#""ornith:35b": { "name": "untouchable" }"#));
    }

    #[test]
    fn scaffolds_provider_into_a_bare_config() {
        let bare = "{\n  \"$schema\": \"https://opencode.ai/config.json\"\n}";
        let (out, report) =
            sync_source(bare, "http://127.0.0.1:9090/v1", &[desired("m", 8192)]).unwrap();
        assert_eq!(report.added, vec!["m"]);
        assert!(out.contains("@ai-sdk/openai-compatible"));
        assert!(out.contains("http://127.0.0.1:9090/v1"));
        let parsed = jsonc_parser::parse_to_serde_value(&out, &Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            parsed.pointer("/provider/llamacpp/models/m/limit/context"),
            Some(&serde_json::json!(8192))
        );
    }

    #[test]
    fn idempotent_when_nothing_changed() {
        let (once, _) =
            sync_source(SAMPLE, "http://127.0.0.1:8080/v1", &[desired("qwen3.6-27b-ud-q5_k_xl", 262_144)])
                .unwrap();
        let (twice, report) =
            sync_source(&once, "http://127.0.0.1:8080/v1", &[desired("qwen3.6-27b-ud-q5_k_xl", 262_144)])
                .unwrap();
        assert_eq!(once, twice);
        assert_eq!(report.updated.len(), 1);
    }
}
