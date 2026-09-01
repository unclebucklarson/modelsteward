//! Hermes agent connector (Connections p2, half two — 2026-08-30).
//!
//! Hermes is a bigger surface than pi or OpenCode, so this connector is
//! deliberately narrow. Everything here was read out of the install's
//! own bundled source (`~/.hermes/hermes-agent`), never guessed:
//!
//! - `context_length_cache.yaml` is a flat `context_lengths:
//!   {model@base_url: int}` map that Hermes rewrites WHOLESALE itself
//!   (`agent/model_metadata.py::save_context_length` → atomic dump), so
//!   it carries no comments and a full round-trip is safe. Key format
//!   is `_context_cache_key`: `f"{model}@{base_url.rstrip('/')}"`.
//! - `config.yaml` is hand-editable, carries comments, holds API keys
//!   (mode 0600), and is read by a possibly-running gateway. Hermes
//!   itself edits it with a comment-preserving round-trip writer. So we
//!   READ it to detect our provider, and only ever APPEND one entry by
//!   surgical text edit, on an explicit click (user decision
//!   2026-08-30) — never a reserialize, never automatically.
//! - `MINIMUM_CONTEXT_LENGTH = 64_000` (model_metadata.py:413): Hermes
//!   REJECTS a model whose context is below that at agent init, so
//!   syncing a smaller measurement would hand the user a model that
//!   cannot start. Those are skipped and named instead.
//!
//! Hermes reconciles cached values against a live probe for local
//! endpoints, preferring what the server reports when it is reachable.
//! That is correct and we don't fight it: our values fill the gap where
//! the router reports `n_ctx: null`, which is every UNLOADED model —
//! i.e. almost always in router mode.

use crate::core::opencode::{DesiredModel, safety_context};
use crate::core::settings;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Hermes refuses to start a model under this many tokens
/// (`MINIMUM_CONTEXT_LENGTH`, agent/model_metadata.py:413).
pub const MINIMUM_CONTEXT: u64 = 64_000;

/// The provider name we register in config.yaml. Its runtime slug is
/// `custom:modelsteward` (`_normalize_custom_provider_name`: strip,
/// lowercase, spaces to dashes — kept space-free so the slug is stable).
pub const PROVIDER_NAME: &str = "modelsteward";

pub fn default_home() -> PathBuf {
    settings::real_home().join(".hermes")
}

pub fn hermes_present(home: &Path) -> bool {
    home.is_dir()
}

pub fn context_cache_path(home: &Path) -> PathBuf {
    home.join("context_length_cache.yaml")
}

pub fn config_path(home: &Path) -> PathBuf {
    home.join("config.yaml")
}

/// `_context_cache_key`: model@base_url, trailing slashes stripped so
/// `/v1` and `/v1/` don't create two entries that go stale apart.
pub fn cache_key(model: &str, base_url: &str) -> String {
    format!("{model}@{}", base_url.trim_end_matches('/'))
}

/// `_normalize_custom_provider_name`.
pub fn provider_slug(name: &str) -> String {
    name.trim().to_lowercase().replace(' ', "-")
}

/// Split the desired models into what Hermes can actually run and what
/// it would reject. Pure — the whole point is that the caller can TELL
/// the user which models were skipped and why.
pub fn partition_by_minimum(desired: &[DesiredModel]) -> (Vec<&DesiredModel>, Vec<&DesiredModel>) {
    desired
        .iter()
        .partition(|d| safety_context(d.context) >= MINIMUM_CONTEXT)
}

#[derive(Debug, Default, PartialEq)]
pub struct HermesSyncReport {
    /// Cache entries written (added or changed).
    pub written: Vec<String>,
    /// Models skipped because Hermes would refuse them (< 64k).
    pub below_minimum: Vec<String>,
    /// ~/.hermes doesn't exist — Hermes isn't installed.
    pub skipped_missing: bool,
    /// No custom provider in config.yaml points at our router yet, so
    /// the cache entries have nothing to attach to until one is
    /// registered. Not an error — the GUI offers the button.
    pub provider_unregistered: bool,
}

/// Parse `custom_providers` and return the NAME of the first entry
/// whose base_url matches ours (trailing slashes ignored). Read-only.
pub fn registered_provider(config_text: &str, base_url: &str) -> Option<String> {
    let doc: serde_yaml::Value = serde_yaml::from_str(config_text).ok()?;
    let want = base_url.trim_end_matches('/').to_lowercase();
    doc.get("custom_providers")?
        .as_sequence()?
        .iter()
        .find(|p| {
            p.get("base_url")
                .and_then(|b| b.as_str())
                .is_some_and(|b| b.trim_end_matches('/').to_lowercase() == want)
        })
        .and_then(|p| p.get("name")?.as_str().map(str::to_string))
}

/// Write measured contexts into the cache. Only our own keys are
/// touched; every other entry (the user's Ollama models, cloud
/// providers) round-trips untouched.
pub fn sync_context_cache(
    path: &Path,
    base_url: &str,
    desired: &[DesiredModel],
) -> Result<Vec<String>> {
    let (usable, _) = partition_by_minimum(desired);
    // Read the WHOLE document and edit it in place. The old code
    // extracted only the (String -> u64) entries it understood and then
    // wrote THAT back as the entire file — so a float value, a quoted
    // number, a null, or any unrelated top-level key was deleted on the
    // next sync, and a YAML the parser rejected became an empty map that
    // wiped every cached context Hermes had (review findings C5/C10,
    // 2026-08-31).
    let mut doc: serde_yaml::Value = match std::fs::read_to_string(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            serde_yaml::Value::Mapping(Default::default())
        }
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        Ok(text) if text.trim().is_empty() => serde_yaml::Value::Mapping(Default::default()),
        Ok(text) => serde_yaml::from_str(&text).with_context(|| {
            format!(
                "{} is not valid YAML — refusing to rewrite it, because doing so \
                 would discard every context it holds. Fix or remove the file, \
                 then sync again",
                path.display()
            )
        })?,
    };
    let map = doc
        .as_mapping_mut()
        .ok_or_else(|| anyhow::anyhow!("{}: root is not a YAML mapping", path.display()))?;
    let key_ctx = serde_yaml::Value::String("context_lengths".into());
    let entry = map
        .entry(key_ctx)
        .or_insert_with(|| serde_yaml::Value::Mapping(Default::default()));
    let ctxs = entry.as_mapping_mut().ok_or_else(|| {
        anyhow::anyhow!("{}: context_lengths is not a mapping", path.display())
    })?;
    let mut written = Vec::new();
    for d in &usable {
        let key = serde_yaml::Value::String(cache_key(&d.id, base_url));
        let ctx = safety_context(d.context);
        if ctxs.get(&key).and_then(|v| v.as_u64()) != Some(ctx) {
            ctxs.insert(key, serde_yaml::Value::Number(ctx.into()));
            written.push(d.id.clone());
        }
    }
    if written.is_empty() {
        return Ok(written);
    }
    if path.exists() {
        std::fs::copy(path, path.with_extension("yaml.modelsteward.bak"))
            .with_context(|| format!("backing up {}", path.display()))?;
    }
    crate::core::safefs::write_atomic(path, &serde_yaml::to_string(&doc)?)?;
    Ok(written)
}

/// The full sync: cache always, provider detection reported.
pub fn sync(home: &Path, base_url: &str, desired: &[DesiredModel]) -> Result<HermesSyncReport> {
    let mut report = HermesSyncReport::default();
    if !hermes_present(home) {
        report.skipped_missing = true;
        return Ok(report);
    }
    let (_, small) = partition_by_minimum(desired);
    report.below_minimum = small.iter().map(|d| d.id.clone()).collect();
    let cfg = std::fs::read_to_string(config_path(home)).unwrap_or_default();
    report.provider_unregistered = registered_provider(&cfg, base_url).is_none();
    report.written = sync_context_cache(&context_cache_path(home), base_url, desired)?;
    Ok(report)
}

/// The YAML block registering our router as a Hermes custom provider.
/// Rendered as text, not serialized, because it is APPENDED into a
/// file whose comments and formatting must survive.
pub fn provider_block(base_url: &str, default_model: &str) -> String {
    format!(
        "  - name: {PROVIDER_NAME}\n    \
         base_url: {base_url}\n    \
         api_key: {PROVIDER_NAME}\n    \
         model: {default_model}\n"
    )
}

/// Append our provider entry to `custom_providers` by surgical text
/// edit — comments, ordering, and quoting elsewhere are untouched.
/// Returns the new file text. Pure; the caller writes and backs up.
///
/// Two shapes are handled: an existing `custom_providers:` sequence
/// (we insert as its last item) and no such key (we append the block).
pub fn register_provider_text(
    config_text: &str,
    base_url: &str,
    default_model: &str,
) -> Option<String> {
    let entry = provider_block(base_url, default_model);
    // Present but not as a block-style key we can extend? Refuse. The
    // old code appended a SECOND `custom_providers:` key, which makes
    // the API-key-bearing config unparseable ("duplicate entry") —
    // reproduced 2026-08-31, review finding C3.
    let has_key = serde_yaml::from_str::<serde_yaml::Value>(config_text)
        .ok()
        .is_some_and(|d| d.get("custom_providers").is_some());
    let block_style = config_text
        .lines()
        .any(|l| l.trim_end() == "custom_providers:");
    if has_key && !block_style {
        return None;
    }
    // A flow-style `custom_providers: [{...}]` is legal YAML that this
    // line-based editor cannot extend. Appending a second
    // `custom_providers:` key would make the API-key-bearing config
    // unparseable, so refuse instead (review finding C10's flow-style half, 2026-08-31).
    // The caller turns None into an honest error.
    let Some(idx) = config_text
        .lines()
        .position(|l| l.trim_end() == "custom_providers:")
    else {
        let mut out = config_text.to_string();
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("custom_providers:\n");
        out.push_str(&entry);
        return Some(out);
    };
    // Find the end of that block: the first later line that is neither
    // blank nor indented (i.e. the next top-level key or a comment at
    // column 0).
    let lines: Vec<&str> = config_text.lines().collect();
    let mut end = lines.len();
    for (i, l) in lines.iter().enumerate().skip(idx + 1) {
        if !l.trim().is_empty() && !l.starts_with([' ', '\t']) {
            end = i;
            break;
        }
    }
    // Back off any trailing blank lines so the entry lands inside the
    // block, not after a gap.
    let mut insert_at = end;
    while insert_at > idx + 1 && lines[insert_at - 1].trim().is_empty() {
        insert_at -= 1;
    }
    let mut out: Vec<String> = lines[..insert_at].iter().map(|s| s.to_string()).collect();
    out.extend(entry.trim_end_matches('\n').lines().map(str::to_string));
    out.extend(lines[insert_at..].iter().map(|s| s.to_string()));
    let mut text = out.join("\n");
    if config_text.ends_with('\n') {
        text.push('\n');
    }
    Some(text)
}

/// Register with a backup. Refuses when an entry already points at
/// this base URL — appending a second one would be ambiguous.
pub fn register_provider(home: &Path, base_url: &str, default_model: &str) -> Result<()> {
    let path = config_path(home);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    if let Some(name) = registered_provider(&text, base_url) {
        anyhow::bail!("Hermes already has a provider for this router: {name:?}");
    }
    let new = register_provider_text(&text, base_url, default_model).ok_or_else(|| {
        anyhow::anyhow!(
            "{} declares custom_providers in a form this editor can't extend \
             safely (flow style?) — add the provider by hand, or reformat that \
             key as a block list first. Nothing was changed.",
            path.display()
        )
    })?;
    std::fs::copy(&path, path.with_extension("yaml.modelsteward.bak"))
        .with_context(|| format!("backing up {}", path.display()))?;
    crate::core::safefs::write_atomic(&path, &new)?;
    Ok(())
}

/// What our provider currently declares in the cache, for the mirror:
/// (model id, context) for keys pointing at this base URL.
pub fn cached_for(path: &Path, base_url: &str) -> Vec<(String, u64)> {
    let suffix = format!("@{}", base_url.trim_end_matches('/'));
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_yaml::from_str::<serde_yaml::Value>(&t).ok())
        .and_then(|d| {
            let m = d.get("context_lengths")?.as_mapping()?.clone();
            Some(
                m.into_iter()
                    .filter_map(|(k, v)| {
                        let k = k.as_str()?;
                        let id = k.strip_suffix(&suffix)?;
                        Some((id.to_string(), v.as_u64()?))
                    })
                    .collect(),
            )
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_damaged_cache_is_refused_not_silently_emptied() {
        // Review finding C5 (2026-08-31): a parse failure became an
        // empty map, and the empty map was written back as the WHOLE
        // file — wiping every context Hermes had cached for Ollama and
        // cloud providers, while reporting success.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("context_length_cache.yaml");
        let broken = "context_lengths:\n  good@http://a: 131072\n\tbad_indent: 1\n";
        std::fs::write(&path, broken).unwrap();
        let e = sync_context_cache(&path, "http://127.0.0.1:8080/v1", &[d("mine", 131_072)])
            .unwrap_err()
            .to_string();
        assert!(e.contains("refusing to rewrite"), "{e}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            broken,
            "the user's file must be untouched"
        );
    }

    #[test]
    fn entries_we_do_not_model_survive_a_sync() {
        // Review finding C10: values that aren't plain unsigned ints,
        // and unrelated top-level keys, were filtered out on read and
        // therefore deleted on write. Executed by the reviewer against
        // the real code; pinned here.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("context_length_cache.yaml");
        std::fs::write(
            &path,
            concat!(
                "context_lengths:\n",
                "  good@http://a: 131072\n",
                "  floaty@http://a: 1.5\n",
                "  quoted@http://a: '4096'\n",
                "  nulled@http://a: null\n",
                "other_top_level_key:\n",
                "  keep: me\n",
            ),
        )
        .unwrap();
        sync_context_cache(&path, "http://127.0.0.1:8080/v1", &[d("mine", 131_072)]).unwrap();
        let after = std::fs::read_to_string(&path).unwrap();
        for must in ["good@http://a", "floaty", "quoted", "nulled", "other_top_level_key", "keep"] {
            assert!(after.contains(must), "{must} was destroyed:\n{after}");
        }
        assert!(after.contains("mine@http://127.0.0.1:8080/v1"), "{after}");
    }

    #[test]
    fn flow_style_custom_providers_is_refused_not_duplicated() {
        // Review finding C10's flow-style half (2026-08-31), executed: appending a second
        // `custom_providers:` key makes the API-key-bearing config
        // unparseable. Refusing is the only safe answer.
        let flow = "model:\n  default: x\ncustom_providers: [{name: mine, base_url: \"http://127.0.0.1:11434/v1\"}]\n";
        assert!(
            register_provider_text(flow, "http://127.0.0.1:8080/v1", "q").is_none(),
            "must refuse rather than duplicate the key"
        );
        // And the block-style path still works.
        assert!(register_provider_text(REAL_CONFIG, "http://127.0.0.1:8080/v1", "q").is_some());
    }

    fn d(id: &str, ctx: u64) -> DesiredModel {
        DesiredModel {
            id: id.into(),
            display_name: format!("{id} (llama.cpp)"),
            context: ctx,
            tool_call: Some(true),
            vision: false,
        }
    }

    /// The live install's cache file, verbatim (2026-08-30).
    const REAL_CACHE: &str = "context_lengths:\n  \
        gemma4:latest@http://127.0.0.1:11434/v1: 131072\n  \
        ornith:35b@http://127.0.0.1:11434/v1: 262144\n";

    #[test]
    fn cache_key_and_slug_match_hermes_source() {
        // _context_cache_key strips trailing slashes.
        assert_eq!(
            cache_key("qwen3.8", "http://127.0.0.1:8080/v1/"),
            "qwen3.8@http://127.0.0.1:8080/v1"
        );
        // _normalize_custom_provider_name — verified against the live
        // config's observed slug custom:local-(127.0.0.1:11434).
        assert_eq!(provider_slug("Local (127.0.0.1:11434)"), "local-(127.0.0.1:11434)");
        assert_eq!(provider_slug(PROVIDER_NAME), "modelsteward");
    }

    #[test]
    fn hermes_minimum_context_is_respected_and_reported() {
        // MINIMUM_CONTEXT_LENGTH = 64_000: Hermes refuses to start a
        // model below it, so writing one would hand the user a broken
        // choice. Live case: gemma-4-31B measured 62,251 here -> 59,136
        // after the safety haircut -> rejected.
        let models = [d("big", 131_072), d("gemma-4-31B", 62_251)];
        let (ok, small) = partition_by_minimum(&models);
        assert_eq!(ok.len(), 1);
        assert_eq!(small[0].id, "gemma-4-31B");
        assert!(safety_context(62_251) < MINIMUM_CONTEXT);
    }

    #[test]
    fn cache_sync_preserves_other_providers_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("context_length_cache.yaml");
        std::fs::write(&path, REAL_CACHE).unwrap();
        let written =
            sync_context_cache(&path, "http://127.0.0.1:8080/v1", &[d("qwen3.8", 113_920)]).unwrap();
        assert_eq!(written, vec!["qwen3.8".to_string()]);
        let after: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let m = after.get("context_lengths").unwrap().as_mapping().unwrap();
        // The user's Ollama entries survive untouched.
        assert_eq!(
            m.get(serde_yaml::Value::String(
                "gemma4:latest@http://127.0.0.1:11434/v1".into()
            ))
            .unwrap()
            .as_u64(),
            Some(131072)
        );
        assert_eq!(
            m.get(serde_yaml::Value::String(
                "ornith:35b@http://127.0.0.1:11434/v1".into()
            ))
            .unwrap()
            .as_u64(),
            Some(262144)
        );
        // Ours landed with the safety haircut.
        assert_eq!(
            m.get(serde_yaml::Value::String(
                "qwen3.8@http://127.0.0.1:8080/v1".into()
            ))
            .unwrap()
            .as_u64(),
            Some(safety_context(113_920))
        );
        // Idempotent: nothing written the second time.
        assert!(
            sync_context_cache(&path, "http://127.0.0.1:8080/v1", &[d("qwen3.8", 113_920)])
                .unwrap()
                .is_empty()
        );
    }

    /// A trimmed copy of the live config.yaml's shape: a populated
    /// custom_providers block followed by comment-only sections, which
    /// is exactly where a naive reserialize would destroy the file.
    const REAL_CONFIG: &str = "model:\n  \
        default: gemma4:latest\ncustom_providers:\n  \
        - name: Local (127.0.0.1:11434)\n    \
        base_url: http://127.0.0.1:11434/v1\n    \
        api_key: ollama\n    \
        model: ornith:35b\n\n\
        # ── Security ──\n\
        # security:\n\
        #   redact_secrets: true\n";

    #[test]
    fn provider_registration_appends_and_keeps_comments() {
        assert_eq!(
            registered_provider(REAL_CONFIG, "http://127.0.0.1:11434/v1/"),
            Some("Local (127.0.0.1:11434)".into()),
            "existing ollama provider is detected by base_url"
        );
        assert_eq!(registered_provider(REAL_CONFIG, "http://127.0.0.1:8080/v1"), None);
        let new = register_provider_text(REAL_CONFIG, "http://127.0.0.1:8080/v1", "qwen3.8").unwrap();
        // The comment block survives verbatim — the whole reason this
        // is a text edit and not a serde round-trip.
        assert!(new.contains("# ── Security ──"), "{new}");
        assert!(new.contains("#   redact_secrets: true"), "{new}");
        // The user's provider survives, ours joins the same block.
        assert!(new.contains("- name: Local (127.0.0.1:11434)"));
        assert!(new.contains("- name: modelsteward"));
        // And it parses, with our entry now discoverable.
        assert_eq!(
            registered_provider(&new, "http://127.0.0.1:8080/v1"),
            Some(PROVIDER_NAME.into())
        );
        let doc: serde_yaml::Value = serde_yaml::from_str(&new).unwrap();
        assert_eq!(
            doc.get("custom_providers").unwrap().as_sequence().unwrap().len(),
            2
        );
    }

    #[test]
    fn registration_without_an_existing_block_creates_one() {
        let text = "model:\n  default: x\n";
        let new = register_provider_text(text, "http://127.0.0.1:8080/v1", "qwen3.8").unwrap();
        assert_eq!(
            registered_provider(&new, "http://127.0.0.1:8080/v1"),
            Some(PROVIDER_NAME.into())
        );
        assert!(new.starts_with("model:\n  default: x\n"), "{new}");
    }

    #[test]
    fn absent_install_skips_and_unregistered_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let absent = dir.path().join("nope");
        assert!(sync(&absent, "http://x/v1", &[d("a", 131_072)]).unwrap().skipped_missing);
        // Present but no provider pointing at us: cache still written,
        // and the caller is told registration is missing.
        let home = dir.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(config_path(&home), "model:\n  default: x\n").unwrap();
        let r = sync(&home, "http://127.0.0.1:8080/v1", &[d("a", 131_072)]).unwrap();
        assert!(r.provider_unregistered);
        assert_eq!(r.written, vec!["a".to_string()]);
    }
}
