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
    /// Measured tool-calling verdict; `None` = probe inconclusive.
    pub tool_call: Option<bool>,
    /// Served with a vision projector (the preset carries `mmproj =` for
    /// it), so the entry may declare image input. A projector merely on
    /// disk doesn't count — OpenCode would send images to a text-only
    /// server.
    pub vision: bool,
}

/// The context we WRITE for a measurement: 5% safety haircut (user
/// decision, 2026-08-16), floored to a 256 multiple. Settled context
/// varies a few percent with desktop VRAM at load time; the haircut keeps
/// OpenCode's budget inside what the server will actually have on a bad
/// day. UI comparisons must use this, not the raw measurement.
pub fn safety_context(measured: u64) -> u64 {
    let cut = measured - measured / 20;
    cut - (cut % 256)
}

impl DesiredModel {
    /// The JSON written for a fresh entry. `tool_call` is the measured
    /// verdict; when the probe was inconclusive we default to `true` —
    /// agent use is the point of this provider, and `false` would hide the
    /// model from OpenCode's agent picker on no evidence.
    fn entry(&self) -> serde_json::Value {
        let ctx = safety_context(self.context);
        let mut e = json!({
            "name": self.display_name,
            "tool_call": self.tool_call.unwrap_or(true),
            "limit": {
                "context": ctx,
                "output": ctx.div_euclid(2).min(32_768),
            }
        });
        if self.vision {
            e["modalities"] = json!({ "input": ["text", "image"], "output": ["text"] });
        }
        e
    }

    /// The patch for an entry that already exists: always the measured
    /// context; the measured `tool_call` ONLY when the entry doesn't have
    /// the key yet (`fill_tool_call`). A hand-set value is never overwritten
    /// in either direction — a probe false-negative must not clobber a
    /// deliberate `true`, nor vice versa.
    fn patch(&self, fill_tool_call: bool, fill_modalities: bool) -> serde_json::Value {
        let mut p = json!({ "limit": { "context": safety_context(self.context) } });
        if fill_tool_call && let Some(tc) = self.tool_call {
            p["tool_call"] = json!(tc);
        }
        if fill_modalities && self.vision {
            p["modalities"] = json!({ "input": ["text", "image"], "output": ["text"] });
        }
        p
    }
}

#[derive(Debug, Default)]
pub struct SyncReport {
    pub added: Vec<String>,
    pub updated: Vec<String>,
    /// Configured under our provider but not desired — candidates for
    /// comment-out, surfaced to the user rather than acted on.
    pub orphans: Vec<String>,
    /// Ghosts the sync commented out itself (reachable router omitted the
    /// id AND nothing measured backed it — user-approved 2026-08-26).
    pub ghosts_commented: Vec<String>,
    /// opencode.json doesn't exist — OpenCode isn't installed; nothing
    /// was written and that's fine (Connections serves other clients).
    pub skipped_missing: bool,
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
        "name": "llama.cpp (modelsteward)",
        "options": { "baseURL": base_url },
        "models": {}
    });

    let mut source = jsonc::ensure_models_container(source, PROVIDER_ID, &scaffold)
        .context("ensuring provider.llamacpp.models exists")?;

    let existing = existing_model_ids(&source)?;
    // Which existing entries already carry a tool_call key (hand-set or
    // previously written) — those are never patched. Same rule for
    // modalities: only filled where absent.
    let has_tool_call: std::collections::HashSet<String> = configured_from_source(&source)?
        .into_iter()
        .filter(|c| c.tool_call.is_some())
        .map(|c| c.id)
        .collect();
    let has_modalities = model_ids_with_key(&source, "modalities")?;
    let mut report = SyncReport::default();

    for d in desired {
        if existing.contains(&d.id) {
            let fill = !has_tool_call.contains(&d.id);
            let fill_mod = !has_modalities.contains(&d.id);
            source = jsonc::merge_model(&source, PROVIDER_ID, &d.id, &d.patch(fill, fill_mod))
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

/// Model ids whose entry already carries `key` — used to fill-not-overwrite.
fn model_ids_with_key(source: &str, key: &str) -> Result<std::collections::HashSet<String>> {
    let value = jsonc_parser::parse_to_serde_value(source, &Default::default())
        .map_err(|e| anyhow::anyhow!("parsing opencode.json: {e}"))?
        .unwrap_or(serde_json::Value::Null);
    Ok(value
        .pointer(&format!("/provider/{PROVIDER_ID}/models"))
        .and_then(|m| m.as_object())
        .map(|m| {
            m.iter()
                .filter(|(_, v)| v.get(key).is_some())
                .map(|(k, _)| k.clone())
                .collect()
        })
        .unwrap_or_default())
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

/// Read -> sync -> backup -> write (numbered backups, see write_backed_up).
/// A MISSING file is a graceful skip, not an error — OpenCode is
/// optional (usability review D7: Set Up Everything hard-failed for
/// anyone without OpenCode installed, despite the app serving any
/// OpenAI-compatible client).
pub fn sync_file(path: &Path, base_url: &str, desired: &[DesiredModel]) -> Result<SyncReport> {
    if !path.exists() {
        return Ok(SyncReport {
            skipped_missing: true,
            ..Default::default()
        });
    }
    let original = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let (updated, report) = sync_source(&original, base_url, desired)?;
    write_backed_up(path, &original, &updated)?;
    Ok(report)
}

const BACKUP_DEPTH: u32 = 5;

fn backup_path(path: &Path, n: u32) -> std::path::PathBuf {
    path.with_extension(format!("json.lcc.bak.{n}"))
}

/// Numbered backups, newest = .1, capped at [`BACKUP_DEPTH`]. A sync
/// followed by a comment-out no longer eats the only recovery point.
fn write_backed_up(path: &Path, original: &str, updated: &str) -> Result<()> {
    if updated == original {
        return Ok(());
    }
    for n in (1..BACKUP_DEPTH).rev() {
        let _ = std::fs::rename(backup_path(path, n), backup_path(path, n + 1));
    }
    std::fs::write(backup_path(path, 1), original)
        .with_context(|| format!("writing backup {}", backup_path(path, 1).display()))?;
    let tmp = path.with_extension("json.lcc.tmp");
    std::fs::write(&tmp, updated).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).context("moving new config into place")?;
    Ok(())
}

/// Swap the config with its newest backup — pressing twice toggles back,
/// so this is undo AND redo in one action. Nothing is ever deleted.
pub fn restore_last_backup(path: &Path) -> Result<String> {
    let bak = backup_path(path, 1);
    let current = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let backup = std::fs::read_to_string(&bak)
        .with_context(|| format!("no backup to restore ({})", bak.display()))?;
    if current == backup {
        return Ok("config and newest backup are identical — nothing to restore".into());
    }
    let tmp = path.with_extension("json.lcc.tmp");
    std::fs::write(&tmp, &backup).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::write(&bak, &current).context("stashing current into backup slot")?;
    std::fs::rename(&tmp, path).context("moving restored config into place")?;
    Ok(format!(
        "restored {} from {} (run again to toggle back)",
        path.display(),
        bak.display()
    ))
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
    configured_from_source(&source)
}

fn configured_from_source(source: &str) -> Result<Vec<ConfiguredModel>> {
    let value = jsonc_parser::parse_to_serde_value(source, &Default::default())
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

/// Auto-comment GHOSTS after a sync (user decision 2026-08-26, after the
/// same leftover confused twice): an orphan qualifies only when the
/// REACHABLE router omits its id — CLAUDE.md's precondition for orphaning
/// — AND no measurement record backs it (a measured-but-unoffered entry
/// can be a temporarily unoffered variant, so those stay reported-only).
/// `offered` must come from a live router; callers with a down router
/// must not call this. Comment-outs, never deletions; backed up like
/// every write. Returns what was commented.
pub fn comment_out_ghosts(
    path: &Path,
    orphans: &[String],
    offered: &[String],
    measurements: &crate::core::router::Measurements,
) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for id in orphans {
        if offered.iter().any(|o| o == id) || measurements.contains_key(id) {
            continue;
        }
        comment_out_in_file(path, id)?;
        out.push(id.clone());
    }
    Ok(out)
}

/// Comment one orphan out in place (never deletes; the entry stays visible
/// in the file under an explanatory note). Backed up like every write.
pub fn comment_out_in_file(path: &Path, model_id: &str) -> Result<()> {
    let original = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let note = "Commented out by modelsteward: not in the current router \npreset / never measured. Uncomment to restore, or delete these \nlines to discard permanently.";
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
            tool_call: None,
            vision: false,
        }
    }

    #[test]
    fn ghost_rules_are_conservative() {
        use crate::core::router::{Measurement, Measurements};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("opencode.json");
        std::fs::write(&path, SAMPLE).unwrap();
        // SAMPLE contains configured models incl. "stale-model".
        let orphans = vec![
            "stale-model".to_string(),
            "offered-elsewhere".to_string(),
            "measured-one".to_string(),
        ];
        let offered = vec!["offered-elsewhere".to_string()];
        let mut m = Measurements::new();
        m.insert("measured-one".into(), Measurement::default());
        let ghosts = comment_out_ghosts(&path, &orphans, &offered, &m).unwrap();
        // Only the unoffered, unmeasured one goes; note it must actually
        // exist in the file for comment-out to succeed.
        assert_eq!(ghosts, vec!["stale-model"]);
        let out = std::fs::read_to_string(&path).unwrap();
        assert!(out.contains("// \"stale-model\""), "{out}");
    }

    #[test]
    fn vision_writes_image_modality_but_never_overwrites() {
        // Fresh entry for a vision-served model -> image input declared.
        let d = DesiredModel { vision: true, ..desired("looker", 90_000) };
        let (out, _) = sync_source(SAMPLE, "http://127.0.0.1:8080/v1", &[d]).unwrap();
        let parsed = jsonc_parser::parse_to_serde_value(&out, &Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            parsed.pointer("/provider/llamacpp/models/looker/modalities/input"),
            Some(&serde_json::json!(["text", "image"]))
        );

        // Existing entry with NO modalities key -> filled in on update.
        let d = DesiredModel { vision: true, ..desired("stale-model", 90_000) };
        let (out, _) = sync_source(SAMPLE, "http://127.0.0.1:8080/v1", &[d]).unwrap();
        let parsed = jsonc_parser::parse_to_serde_value(&out, &Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            parsed.pointer("/provider/llamacpp/models/stale-model/modalities/input"),
            Some(&serde_json::json!(["text", "image"])),
            "absent modalities gets the measured truth"
        );

        // Non-vision model -> no modalities key invented.
        let d = desired("plain", 90_000);
        let (out, _) = sync_source(SAMPLE, "http://127.0.0.1:8080/v1", &[d]).unwrap();
        let parsed = jsonc_parser::parse_to_serde_value(&out, &Default::default())
            .unwrap()
            .unwrap();
        assert!(parsed
            .pointer("/provider/llamacpp/models/plain/modalities")
            .is_none());
    }

    #[test]
    fn measured_tool_call_fills_gaps_but_never_overwrites() {
        // "stale-model" in SAMPLE has NO tool_call key -> measured verdict
        // fills it in.
        let d = DesiredModel {
            tool_call: Some(false),
            ..desired("stale-model", 90_000)
        };
        let (out, _) = sync_source(SAMPLE, "http://127.0.0.1:8080/v1", &[d]).unwrap();
        let parsed = jsonc_parser::parse_to_serde_value(&out, &Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            parsed.pointer("/provider/llamacpp/models/stale-model/tool_call"),
            Some(&serde_json::json!(false)),
            "absent key gets the measured verdict"
        );

        // "qwen3.6-27b-ud-q5_k_xl" HAS tool_call: true in SAMPLE — a
        // measured `false` must NOT overwrite the recorded value.
        let d2 = DesiredModel {
            tool_call: Some(false),
            ..desired("qwen3.6-27b-ud-q5_k_xl", 91_000)
        };
        let (out2, _) = sync_source(SAMPLE, "http://127.0.0.1:8080/v1", &[d2]).unwrap();
        let parsed2 = jsonc_parser::parse_to_serde_value(&out2, &Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            parsed2.pointer("/provider/llamacpp/models/qwen3.6-27b-ud-q5_k_xl/tool_call"),
            Some(&serde_json::json!(true)),
            "present key is sacred"
        );
        assert_eq!(
            parsed2.pointer("/provider/llamacpp/models/qwen3.6-27b-ud-q5_k_xl/limit/context"),
            Some(&serde_json::json!(safety_context(91_000))),
            "context still refreshes"
        );

        // New entries carry the measured verdict directly.
        let d3 = DesiredModel {
            tool_call: Some(false),
            ..desired("brand-new", 8_192)
        };
        let (out3, _) = sync_source(SAMPLE, "http://127.0.0.1:8080/v1", &[d3]).unwrap();
        let parsed3 = jsonc_parser::parse_to_serde_value(&out3, &Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            parsed3.pointer("/provider/llamacpp/models/brand-new/tool_call"),
            Some(&serde_json::json!(false))
        );
    }

    #[test]
    fn updates_existing_entry_only_touching_measured_context() {
        let (out, report) =
            sync_source(SAMPLE, "http://127.0.0.1:8080/v1", &[desired("qwen3.6-27b-ud-q5_k_xl", 72_960)])
                .unwrap();
        assert_eq!(report.updated, vec!["qwen3.6-27b-ud-q5_k_xl"]);
        assert!(out.contains(&format!(r#""context": {}"#, safety_context(72_960))));
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
        let ctx = safety_context(36_096);
        assert!(out.contains(&format!(r#""context": {ctx}"#)));
        // output = min(ctx/2, 32768)
        assert!(out.contains(&format!(r#""output": {}"#, ctx / 2)));
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
            Some(&serde_json::json!(safety_context(8192)))
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

    #[test]
    fn safety_haircut_is_five_percent_floored_to_256() {
        assert_eq!(safety_context(72_960), 69_120);
        assert_eq!(safety_context(262_144), 248_832);
        assert_eq!(safety_context(256), 0, "tiny values floor to zero — callers must treat 0 as unusable");
    }

    #[test]
    fn backups_are_numbered_and_restore_toggles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("opencode.json");
        std::fs::write(&path, "v0").unwrap();
        for v in ["v1", "v2", "v3"] {
            let cur = std::fs::read_to_string(&path).unwrap();
            write_backed_up(&path, &cur, v).unwrap();
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "v3");
        assert_eq!(
            std::fs::read_to_string(backup_path(&path, 1)).unwrap(),
            "v2",
            "newest backup is .1"
        );
        assert_eq!(std::fs::read_to_string(backup_path(&path, 3)).unwrap(), "v0");

        // Restore = swap with .1; twice = back where we started.
        restore_last_backup(&path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "v2");
        assert_eq!(std::fs::read_to_string(backup_path(&path, 1)).unwrap(), "v3");
        restore_last_backup(&path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "v3");
    }
}