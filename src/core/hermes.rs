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
    /// Models Hermes is configured to RUN on our router that the cache
    /// cannot serve — the check that "N contexts written" never made.
    pub unservable: Vec<DefaultGap>,
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

/// A model Hermes is CONFIGURED TO RUN through our router, paired with
/// what the context cache actually holds for it.
#[derive(Debug, PartialEq, Eq)]
pub struct DefaultGap {
    /// The model id Hermes will try to start.
    pub model: String,
    /// What the cache holds for it at our base URL, if anything.
    pub cached: Option<u64>,
    /// Where the setting lives, so the message can name the fix.
    pub site: GapSite,
}

#[derive(Debug, PartialEq, Eq)]
pub enum GapSite {
    /// Top-level `model.default`.
    Default,
    /// The `model:` of a `custom_providers` entry, named.
    Provider(String),
}

impl DefaultGap {
    /// The sentence a user can act on.
    pub fn message(&self) -> String {
        let where_ = match &self.site {
            GapSite::Default => "Hermes's default model is".to_string(),
            GapSite::Provider(n) => format!("Hermes provider {n:?} is set to run"),
        };
        match self.cached {
            None => format!(
                "{where_} {:?}, and no context is cached for it on this \
                 router — Hermes will fall back to a ~4k default and refuse it. \
                 Measure that model, then sync again",
                self.model
            ),
            Some(c) => format!(
                "{where_} {:?}, cached at {c} tokens — under Hermes's \
                 {MINIMUM_CONTEXT} minimum, so it will refuse to start",
                self.model
            ),
        }
    }
}

/// Audit: which models does Hermes intend to run through OUR base URL,
/// and does the cache actually carry a usable context for each?
///
/// A sync writes what is DESIRED; it has no opinion about what is
/// missing. So a model that measured `n_ctx: null` for one run drops
/// out of `desired`, never gets written, and — because the cache is
/// append-only — the hole persists across every later sync while each
/// one reports success. That is exactly how Hermes came to refuse
/// `qwen3.8-27b-ud-q4_k_xl` with "context window of 4,096 tokens"
/// after a port change: the 8181 block was born without it and no
/// sync ever noticed (2026-09-20).
///
/// Pure over its inputs.
pub fn default_model_gaps(
    config_text: &str,
    cached: &[(String, u64)],
    base_url: &str,
) -> Vec<DefaultGap> {
    let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(config_text) else {
        // An unparseable config is the register path's problem, not
        // ours; staying quiet beats guessing at its contents.
        return Vec::new();
    };
    let ours = |v: Option<&serde_yaml::Value>| {
        v.and_then(|b| b.as_str())
            .is_some_and(|b| b.trim_end_matches('/').eq_ignore_ascii_case(base_url.trim_end_matches('/')))
    };

    // Every place Hermes records "run THIS model, THERE".
    let mut intents: Vec<(String, GapSite)> = Vec::new();
    if let Some(m) = doc.get("model")
        && ours(m.get("base_url"))
        && let Some(id) = m.get("default").and_then(|d| d.as_str())
    {
        intents.push((id.to_string(), GapSite::Default));
    }
    if let Some(ps) = doc.get("custom_providers").and_then(|p| p.as_sequence()) {
        for p in ps {
            if !ours(p.get("base_url")) {
                continue;
            }
            let Some(id) = p.get("model").and_then(|m| m.as_str()) else {
                continue;
            };
            let name = p
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or(PROVIDER_NAME);
            intents.push((id.to_string(), GapSite::Provider(name.to_string())));
        }
    }

    intents
        .into_iter()
        .filter_map(|(model, site)| {
            let hit = cached.iter().find(|(id, _)| *id == model).map(|(_, c)| *c);
            // Cached and adequate is the healthy case — say nothing.
            match hit {
                Some(c) if c >= MINIMUM_CONTEXT => None,
                cached => Some(DefaultGap { model, cached, site }),
            }
        })
        .collect()
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
        crate::core::safefs::backup_rotated(path, "modelsteward")?;
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
    // AFTER the write, deliberately: the audit must judge the cache we
    // just left behind, not the one we found — otherwise the very sync
    // that repairs a hole still reports it.
    report.unservable = default_model_gaps(
        &cfg,
        &cached_for(&context_cache_path(home), base_url),
        base_url,
    );
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
    crate::core::safefs::backup_rotated(&path, "modelsteward")?;
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

    /// Live incident 2026-09-20. Hermes refused the user's default
    /// model with "context window of 4,096 tokens, below the minimum
    /// 64,000". Cause: `~/.hermes/context_length_cache.yaml` held 16
    /// entries for port 8181 and NOT the one model Hermes was
    /// configured to run. The 8181 block was a copy of the 8080 block
    /// minus exactly that model — it had measured `n_ctx: null` on the
    /// one sync that followed the port change, so `desired_from`
    /// dropped it, and the append-only cache kept the hole while every
    /// later sync reported success.
    ///
    /// Config text below is the real file's shape, including the
    /// duplicate `modelsteward` provider left by the port change.
    #[test]
    fn a_default_model_with_no_cached_context_is_reported() {
        let cfg = "\
model:
  default: qwen3.8-27b-ud-q4_k_xl
  provider: custom
  base_url: http://127.0.0.1:8181/v1
  api_key: ollama
custom_providers:
  - name: Local (127.0.0.1:11434)
    base_url: http://127.0.0.1:11434/v1
    model: ornith:35b
  - name: modelsteward
    base_url: http://127.0.0.1:8080/v1
    model: glm-4.5-air-ud-q3_k_xl
  - name: modelsteward
    base_url: http://127.0.0.1:8181/v1
    model: qwen3.8-27b-ud-q4_k_xl
";
        // What the cache actually held at 8181 that day: everything
        // except the model Hermes was about to start.
        let cached = vec![
            ("glm-4.5-air-ud-q3_k_xl".to_string(), 124_416),
            ("gpt-oss-20b-f16".to_string(), 124_416),
            ("qwen3.8-27b-ud-q5_k_xl".to_string(), 67_840),
        ];
        let gaps = default_model_gaps(cfg, &cached, "http://127.0.0.1:8181/v1");

        assert_eq!(gaps.len(), 2, "default + our provider at 8181: {gaps:?}");
        assert!(
            gaps.iter().all(|g| g.model == "qwen3.8-27b-ud-q4_k_xl"),
            "{gaps:?}"
        );
        assert!(
            gaps.iter().all(|g| g.cached.is_none()),
            "nothing was cached for it: {gaps:?}"
        );
        assert!(
            gaps.iter().any(|g| g.site == GapSite::Default),
            "the top-level default is the one that bit: {gaps:?}"
        );
        assert!(
            gaps.iter()
                .any(|g| g.site == GapSite::Provider("modelsteward".into())),
            "{gaps:?}"
        );
        assert!(gaps[0].message().contains("4k default"), "{}", gaps[0].message());
    }

    /// The other half of the same audit: a model Hermes WILL start but
    /// whose cached value is under its own 64,000 floor. Same user
    /// symptom, different cause, so it must not be silent either.
    #[test]
    fn a_default_model_cached_below_the_minimum_is_reported() {
        let cfg = "\
model:
  default: qwen3.8-27b-ud-q5_k_xl
  base_url: http://127.0.0.1:8181/v1
";
        let cached = vec![("qwen3.8-27b-ud-q5_k_xl".to_string(), 40_000)];
        let gaps = default_model_gaps(cfg, &cached, "http://127.0.0.1:8181/v1");
        assert_eq!(gaps.len(), 1, "{gaps:?}");
        assert_eq!(gaps[0].cached, Some(40_000));
        assert!(gaps[0].message().contains("64000"), "{}", gaps[0].message());
    }

    /// Models pointed at SOMEONE ELSE's endpoint are not ours to
    /// audit — we know nothing about what Ollama has cached, and a
    /// false alarm about a provider we don't serve is noise.
    #[test]
    fn models_on_another_base_url_are_not_our_business() {
        let cfg = "\
model:
  default: ornith:35b
  base_url: http://127.0.0.1:11434/v1
custom_providers:
  - name: Local (127.0.0.1:11434)
    base_url: http://127.0.0.1:11434/v1
    model: ornith:35b
";
        let gaps = default_model_gaps(cfg, &[], "http://127.0.0.1:8181/v1");
        assert!(gaps.is_empty(), "{gaps:?}");
    }

    /// A cache that already carries the model is the healthy case and
    /// must stay quiet, or the warning trains the user to ignore it.
    #[test]
    fn a_cached_default_above_the_minimum_is_silent() {
        let cfg = "\
model:
  default: qwen3.8-27b-ud-q4_k_xl
  base_url: http://127.0.0.1:8181/v1
";
        // The value one --sync actually wrote to repair the incident.
        let cached = vec![("qwen3.8-27b-ud-q4_k_xl".to_string(), 109_824)];
        let gaps = default_model_gaps(cfg, &cached, "http://127.0.0.1:8181/v1");
        assert!(gaps.is_empty(), "{gaps:?}");
    }

    /// The audit is worthless unless the SYNC runs it. Without this,
    /// `default_model_gaps` is a pure function nobody calls and the
    /// 2026-09-20 incident repeats with the report still saying
    /// "1 context(s) written".
    #[test]
    fn sync_reports_a_default_model_it_could_not_serve() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        std::fs::write(
            home.join("config.yaml"),
            "model:\n  default: qwen3.8-27b-ud-q4_k_xl\n  \
             base_url: http://127.0.0.1:8181/v1\ncustom_providers:\n  \
             - name: modelsteward\n    base_url: http://127.0.0.1:8181/v1\n",
        )
        .unwrap();
        // The model Hermes wants is NOT in desired — exactly the state
        // an `n_ctx: null` measurement leaves behind.
        let r = sync(
            home,
            "http://127.0.0.1:8181/v1",
            &[d("glm-4.5-air-ud-q3_k_xl", 131_072)],
        )
        .unwrap();
        assert_eq!(r.written, vec!["glm-4.5-air-ud-q3_k_xl".to_string()]);
        assert_eq!(r.unservable.len(), 1, "{:?}", r.unservable);
        assert_eq!(r.unservable[0].model, "qwen3.8-27b-ud-q4_k_xl");
        assert_eq!(r.unservable[0].cached, None);
    }

    /// And it must read the cache AFTER writing it, or the sync that
    /// repairs the hole still shouts about it.
    #[test]
    fn a_sync_that_fills_the_hole_does_not_then_complain_about_it() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        std::fs::write(
            home.join("config.yaml"),
            "model:\n  default: qwen3.8-27b-ud-q4_k_xl\n  \
             base_url: http://127.0.0.1:8181/v1\n",
        )
        .unwrap();
        let r = sync(
            home,
            "http://127.0.0.1:8181/v1",
            &[d("qwen3.8-27b-ud-q4_k_xl", 115_712)],
        )
        .unwrap();
        assert!(!r.written.is_empty());
        assert!(r.unservable.is_empty(), "{:?}", r.unservable);
    }
}
