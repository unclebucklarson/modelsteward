//! Persisted app configuration. Every value has a working default so a
//! missing or partial config file never blocks startup — the file only
//! records what the user changed.

use crate::core::router::ModelOverrides;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
