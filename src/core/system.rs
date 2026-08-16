//! Assembly of the core modules into the operations the CLI and GUI share:
//! scan the system, pick a server, generate the preset, build router config.

use crate::core::{discover, library, router};
use anyhow::Result;
use serde::Serialize;
use std::path::PathBuf;

pub const DEFAULT_PORT: u16 = 8080;

#[derive(Debug, Clone, Serialize)]
pub struct ScanReport {
    pub installs: Vec<discover::LlamaInstall>,
    /// Devices as seen by `devices_from`. Newest build is asked first, but a
    /// build that reports no devices (e.g. a CUDA build whose runtime can't
    /// initialize here) is skipped in favor of one that can actually see the
    /// hardware.
    pub devices: Vec<discover::Device>,
    pub devices_from: Option<PathBuf>,
    pub models: Vec<library::ModelFile>,
}

pub fn config_dir() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".config")
        })
        .join("llamacppcodeconf")
}

pub fn preset_path() -> PathBuf {
    config_dir().join("router.ini")
}

/// The llama-server to run: newest install that can actually see devices,
/// else the newest install at all.
pub fn pick_server() -> Result<PathBuf> {
    let installs = discover::find_installs(&[]);
    let mut by_build: Vec<_> = installs.iter().collect();
    by_build.sort_by_key(|i| std::cmp::Reverse(i.build.unwrap_or(0)));
    by_build
        .iter()
        .find(|i| !discover::list_devices(&i.server_path).is_empty())
        .or(by_build.first())
        .map(|i| i.server_path.clone())
        .ok_or_else(|| anyhow::anyhow!("no llama-server found; point me at one"))
}

pub fn router_config(port: u16) -> router::RouterConfig {
    router::RouterConfig {
        server_bin: pick_server().unwrap_or_else(|_| PathBuf::from("llama-server")),
        port,
        preset_path: preset_path(),
        models_max: 1,
    }
}

pub fn scan_models(extra_dirs: &[PathBuf]) -> Vec<library::ModelFile> {
    let mut scan_dirs = extra_dirs.to_vec();
    if let Some(home) = std::env::home_dir() {
        scan_dirs.push(home.join("models"));
    }
    library::scan(&scan_dirs, &library::default_ollama_stores())
}

pub fn scan_report(extra_dirs: &[PathBuf]) -> ScanReport {
    let installs = discover::find_installs(&[]);
    let mut by_build: Vec<_> = installs.iter().collect();
    by_build.sort_by_key(|i| std::cmp::Reverse(i.build.unwrap_or(0)));
    let (devices, devices_from) = by_build
        .iter()
        .find_map(|i| {
            let d = discover::list_devices(&i.server_path);
            (!d.is_empty()).then(|| (d, Some(i.server_path.clone())))
        })
        .unwrap_or_default();

    ScanReport {
        installs,
        devices,
        devices_from,
        models: scan_models(extra_dirs),
    }
}

/// Text of a systemd *user* unit that runs the router at login. The unit
/// runs llama-server directly (not this app), with the same preset — the
/// router process it starts passes `router::find_preset_process`, so the
/// app recognizes it as ours and Stop works on it.
pub fn systemd_unit_text(cfg: &router::RouterConfig) -> String {
    format!(
        "[Unit]\n\
         Description=llama.cpp router (llamacppcodeconf)\n\
         After=network.target\n\n\
         [Service]\n\
         ExecStart={server} --models-preset {preset} --host 127.0.0.1 --port {port} --models-max {max}\n\
         Restart=on-failure\n\n\
         [Install]\n\
         WantedBy=default.target\n",
        server = cfg.server_bin.display(),
        preset = cfg.preset_path.display(),
        port = cfg.port,
        max = cfg.models_max,
    )
}

/// Write the user unit file. Enabling/starting is left to the user (one
/// visible command) so systemd never surprises them:
///   systemctl --user daemon-reload && systemctl --user enable --now llamacpp-router
pub fn install_systemd_unit(port: u16) -> Result<PathBuf> {
    let cfg = router_config(port);
    if !cfg.preset_path.exists() {
        write_preset(&[])?;
    }
    let dir = std::env::home_dir()
        .ok_or_else(|| anyhow::anyhow!("no home directory"))?
        .join(".config/systemd/user");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("llamacpp-router.service");
    std::fs::write(&path, systemd_unit_text(&cfg))?;
    Ok(path)
}

/// (Re)generate the preset from a scan; returns (path, model count).
pub fn write_preset(extra_dirs: &[PathBuf]) -> Result<(PathBuf, usize)> {
    let models = scan_models(extra_dirs);
    let entries = router::default_entries(&models);
    let ini = router::render_preset(&entries);
    std::fs::create_dir_all(config_dir())?;
    std::fs::write(preset_path(), &ini)?;
    Ok((preset_path(), entries.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systemd_unit_runs_llama_server_with_our_preset() {
        let cfg = router::RouterConfig {
            server_bin: PathBuf::from("/opt/bin/llama-server"),
            port: 8080,
            preset_path: PathBuf::from("/home/u/.config/llamacppcodeconf/router.ini"),
            models_max: 1,
        };
        let unit = systemd_unit_text(&cfg);
        assert!(unit.contains(
            "ExecStart=/opt/bin/llama-server --models-preset /home/u/.config/llamacppcodeconf/router.ini --host 127.0.0.1 --port 8080 --models-max 1"
        ));
        // The preset path in ExecStart is exactly what find_preset_process
        // matches on — this is the ownership handshake with the router module.
        assert!(unit.contains("WantedBy=default.target"));
    }
}
