//! Persisted app configuration. Every value has a working default so a
//! missing or partial config file never blocks startup — the file only
//! records what the user changed.

use crate::core::router::ModelOverrides;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The user's REAL home. Snap-packaged parents (VS Code's snap, most
/// commonly) redirect HOME into a per-REVISION dir like
/// /home/u/snap/code/258 — anything stored there is orphaned on the
/// next snap update (live casualty 2026-08-28: the managed llama.cpp
/// checkout vanished with an OS update). Strip the redirect.
pub fn real_home() -> PathBuf {
    let home = std::env::home_dir().unwrap_or_else(|| PathBuf::from("."));
    strip_snap_redirect(&home).unwrap_or(home)
}

/// Some(prefix-before-snap) when `p` looks like a snap HOME redirect:
/// .../snap/<app>/<revision>[/...] — "snap" must be followed by at
/// least two components, so a user literally named snap
/// (/home/snap) or a path like /data/snap/scott is left alone
/// (review catch 2026-08-28: the first cut truncated at any "snap").
fn strip_snap_redirect(p: &Path) -> Option<PathBuf> {
    let comps: Vec<_> = p.components().collect();
    let idx = comps
        .iter()
        .position(|c| c.as_os_str() == "snap")
        .filter(|&i| comps.len() >= i + 3)?;
    Some(comps[..idx].iter().collect())
}

/// The one XDG-with-snap-guard resolver every persistent dir uses —
/// three hand-rolled copies had already grown (review catch): a
/// snap-redirected env var or HOME never decides where data lives.
pub fn xdg_dir(var: &str, fallback_rel: &str) -> PathBuf {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|p| strip_snap_redirect(p).is_none())
        .unwrap_or_else(|| real_home().join(fallback_rel))
        .join("modelsteward")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    /// Shelf directories to scan for GGUFs (the Ollama store is found
    /// automatically and isn't listed here).
    pub scan_dirs: Vec<PathBuf>,
    /// Router port — also what opencode.json's baseURL is written with.
    pub port: u16,
    /// Explicit llama-server binary; `None` = auto-pick the newest install
    /// that can see the hardware.
    pub server_bin: Option<PathBuf>,
    /// Where the Ollama peer answers.
    pub ollama_port: u16,
    /// Router `--models-max`: how many models may be loaded at once.
    /// 1 fits one big model on a 24GB-class card; raise it to keep a small
    /// sidecar model (notes, embeddings) resident next to the big coder.
    pub models_max: u32,
    /// Per-model preset overrides, keyed by router id (preset alias or
    /// cache id). Living HERE — not in router.ini — is what lets preset
    /// regeneration keep the user's tuning instead of flattening it.
    pub overrides: std::collections::BTreeMap<String, ModelOverrides>,
    /// Extra llama.cpp checkouts the Build Advisor can analyze and
    /// rebuild (user decision 2026-08-28: manual checkouts with custom
    /// options, selectable for building; the active binary's checkout
    /// and the managed clone are always offered).
    pub checkouts: Vec<PathBuf>,
    /// $/Mtok OUTPUT price used by the Meter's cloud-comparison counter
    /// (M9 p1). A ballpark mid-tier API price as of 2026-08 — edit to
    /// whatever YOU would actually pay; the report labels it as your
    /// number, never a quote.
    pub cloud_price_per_mtok: f64,
    /// Models the user told the app to stop offering (user request
    /// 2026-08-28: the Ollama-only conversions that can NEVER load in
    /// llama.cpp sat in the Library as permanent red rows and were
    /// re-tried by every calibrate). Keys are row identities
    /// (path, else router id); ignored models leave the preset, the
    /// Library's main view, and the Lab — reversibly.
    pub ignored: Vec<String>,
    /// Autonomy for the managed checkout (user decision 2026-08-28):
    /// when the daily freshness check finds a new release, BUILD and
    /// ARCHIVE it automatically — but never pin it: what the router
    /// serves changes only by explicit choice (the b10630 −9%-context
    /// lesson, institutionalized). Default off.
    pub managed_auto_build: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        let mut scan_dirs = Vec::new();
        if let Some(home) = std::env::home_dir() {
            scan_dirs.push(home.join("models"));
        }
        Self {
            scan_dirs,
            port: 8080,
            server_bin: None,
            ollama_port: 11434,
            models_max: 1,
            overrides: Default::default(),
            checkouts: Vec::new(),
            cloud_price_per_mtok: 3.0,
            ignored: Vec::new(),
            managed_auto_build: false,
        }
    }
}

impl AppConfig {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))
    }
}

#[cfg(test)]
mod tests_snap {
    use super::*;

    #[test]
    fn snap_redirects_strip_but_snap_users_survive() {
        let strip = |p: &str| strip_snap_redirect(Path::new(p)).map(|r| r.display().to_string());
        // The real redirect shape: home/snap/<app>/<revision>.
        assert_eq!(strip("/home/buck/snap/code/258"), Some("/home/buck".into()));
        assert_eq!(
            strip("/home/buck/snap/code/258/.local/share"),
            Some("/home/buck".into())
        );
        // A user literally named snap, or snap as a plain dir: untouched.
        assert_eq!(strip("/home/snap"), None);
        assert_eq!(strip("/data/snap/scott"), None);
        assert_eq!(strip("/home/buck"), None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_and_tolerates_partial_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");

        // Partial file: only the port — everything else defaults.
        std::fs::write(&path, r#"{"port": 9090}"#).unwrap();
        let cfg = AppConfig::load(&path);
        assert_eq!(cfg.port, 9090);
        assert_eq!(cfg.ollama_port, 11434);
        assert_eq!(cfg.models_max, 1, "partial files keep the models_max default");
        assert!(cfg.server_bin.is_none());

        // Full roundtrip.
        let mut cfg2 = cfg.clone();
        cfg2.server_bin = Some(PathBuf::from("/opt/llama-server"));
        cfg2.save(&path).unwrap();
        assert_eq!(AppConfig::load(&path), cfg2);

        // Garbage file → defaults, not a crash.
        std::fs::write(&path, "not json").unwrap();
        assert_eq!(AppConfig::load(&path), AppConfig::default());
    }
}
