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
        // XDG spec: a set-but-empty variable means unset. Without this
        // the config path came out RELATIVE ("modelsteward/config.json")
        // — caught by the v0.5.2 release-checklist smoke, 2026-08-30.
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
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
    /// $/kWh for the Meter's measured-cost line — YOUR electricity
    /// price (default is a rough US average; edit to your bill's rate).
    /// Pinned advisory answerer (None = auto: best measured quality,
    /// fastest within the tie — see aiadvisor::pick_advisor). The seat
    /// is EARNED: pin a model its quality probe vouches for.
    pub advisor_model: Option<String>,
    pub kwh_price_usd: f64,
    /// $/Mtok OUTPUT price used by the Meter's cloud-comparison counter
    /// (M9 p1). A ballpark mid-tier API price as of 2026-08 — edit to
    /// whatever YOU would actually pay; the report labels it as your
    /// number, never a quote.
    pub cloud_price_per_mtok: f64,
    /// Models the user DISABLED (refined 2026-08-28: they stay visible
    /// in the Library — dimmed — but measure/bench/Lab/preset all skip
    /// them; the Ollama-only conversions that can never load were being
    /// re-tried by every calibrate). Keys: file path, else router id.
    #[serde(alias = "ignored")]
    pub disabled: Vec<String>,
    /// Autonomy for the managed checkout (user decision 2026-08-28):
    /// when the daily freshness check finds a new release, BUILD and
    /// ARCHIVE it automatically — but never pin it: what the router
    /// serves changes only by explicit choice (the b10630 −9%-context
    /// lesson, institutionalized). Default off.
    pub managed_auto_build: bool,
    /// Retention for archived managed builds: keep the newest N
    /// release archives, pruned after each successful build (0 = keep
    /// everything). The serving archive and custom-labeled archives
    /// are never pruned. Design session 2026-08-30.
    #[serde(default = "default_archives_keep")]
    pub archives_keep: u32,
}

fn default_archives_keep() -> u32 {
    5
}

impl Default for AppConfig {
    fn default() -> Self {
        // real_home, not raw HOME: a snap-launched app used to default
        // to a vanishing ~/snap/…/models (usability review D8).
        let scan_dirs = vec![real_home().join("models")];
        Self {
            scan_dirs,
            port: 8080,
            server_bin: None,
            ollama_port: 11434,
            models_max: 1,
            overrides: Default::default(),
            checkouts: Vec::new(),
            advisor_model: None,
            kwh_price_usd: 0.15,
            cloud_price_per_mtok: 3.0,
            disabled: Vec::new(),
            managed_auto_build: false,
            archives_keep: default_archives_keep(),
        }
    }
}

impl AppConfig {
    pub fn load(path: &Path) -> Self {
        let (cfg, err) = Self::load_checked(path);
        if let Some(e) = err {
            eprintln!(
                "WARNING: {} is unreadable ({e}) — running with defaults; the file \
                 will be preserved as config.json.corrupt on the next save",
                path.display()
            );
        }
        cfg
    }

    /// Missing file = quiet defaults. UNPARSEABLE file = defaults plus
    /// the parse error, surfaced — a hand-edit's trailing comma used to
    /// silently reset everything including kept trial winners
    /// (usability review C8, 2026-08-29).
    pub fn load_checked(path: &Path) -> (Self, Option<String>) {
        match std::fs::read_to_string(path) {
            Err(_) => (Self::default(), None),
            Ok(text) => match serde_json::from_str(&text) {
                Ok(cfg) => (cfg, None),
                Err(e) => (Self::default(), Some(e.to_string())),
            },
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        // Never clobber an unparseable original — it may hold settings
        // the defaults just replaced; rescue it beside the new file.
        if let Ok(existing) = std::fs::read_to_string(path)
            && serde_json::from_str::<Self>(&existing).is_err()
        {
            let rescue = path.with_extension("json.corrupt");
            let _ = std::fs::write(&rescue, existing);
            eprintln!(
                "preserved unreadable config as {} before saving",
                rescue.display()
            );
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))
    }
}

#[cfg(test)]
mod tests_load {
    use super::*;

    #[test]
    fn corrupt_configs_are_loud_and_never_silently_clobbered() {
        // Usability review C8 (2026-08-29): a hand-edit's trailing comma
        // silently reset EVERYTHING including kept trial winners, and
        // the next save overwrote the evidence. New contract: missing =
        // quiet defaults; unparseable = defaults + the parse error
        // surfaced; save() preserves an unparseable original as
        // config.json.corrupt before writing.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        // Missing: quiet defaults.
        let (cfg, err) = AppConfig::load_checked(&path);
        assert_eq!(cfg, AppConfig::default());
        assert!(err.is_none());
        // Corrupt: defaults + loud error.
        std::fs::write(&path, "{ \"port\": 8080, }").unwrap();
        let (cfg, err) = AppConfig::load_checked(&path);
        assert_eq!(cfg, AppConfig::default());
        assert!(err.is_some(), "parse failure must be surfaced");
        // Save rescues the corrupt original before overwriting.
        cfg.save(&path).unwrap();
        let rescued = dir.path().join("config.json.corrupt");
        assert!(rescued.exists(), "corrupt original preserved");
        assert!(std::fs::read_to_string(&rescued).unwrap().contains("8080,"));
        let (roundtrip, err) = AppConfig::load_checked(&path);
        assert_eq!(roundtrip, AppConfig::default());
        assert!(err.is_none());
    }

    #[test]
    fn default_scan_dir_survives_snap_home() {
        // Usability review D8: the default used raw home_dir() while
        // real_home() existed for exactly this — a snap-launched app got
        // a vanishing ~/snap/…/models scan dir and a blank Library.
        let d = AppConfig::default();
        assert!(
            !d.scan_dirs.iter().any(|p| p
                .components()
                .any(|c| c.as_os_str() == "snap")),
            "{:?}",
            d.scan_dirs
        );
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
    fn xdg_empty_or_relative_means_unset() {
        // XDG Base Directory spec: empty or relative values must be
        // ignored. Live catch 2026-08-30: XDG_CONFIG_HOME="" produced
        // a relative config path.
        unsafe {
            std::env::set_var("MODELSTEWARD_TEST_XDG", "");
        }
        assert!(xdg_dir("MODELSTEWARD_TEST_XDG", ".config").is_absolute());
        unsafe {
            std::env::set_var("MODELSTEWARD_TEST_XDG", "relative/path");
        }
        assert!(xdg_dir("MODELSTEWARD_TEST_XDG", ".config").is_absolute());
        unsafe {
            std::env::remove_var("MODELSTEWARD_TEST_XDG");
        }
    }

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

        // Garbage file -> defaults, not a crash.
        std::fs::write(&path, "not json").unwrap();
        assert_eq!(AppConfig::load(&path), AppConfig::default());
    }
}
