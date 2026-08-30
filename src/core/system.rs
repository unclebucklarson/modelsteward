//! Assembly of the core modules into the operations the CLI and GUI share:
//! scan the system, pick a server, generate the preset, build router config.
//! Everything flows from the persisted [`settings::AppConfig`].

use crate::core::{discover, library, router, settings};
use anyhow::Result;
use serde::Serialize;
use std::path::PathBuf;

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
    settings::xdg_dir("XDG_CONFIG_HOME", ".config")
}

pub fn config_file() -> PathBuf {
    config_dir().join("config.json")
}

/// One-time rename migration (llamacppcodeconf → modelsteward,
/// 2026-08-26): move the old config and state dirs to the new names so
/// measurements, history, trials, and overrides survive the rename.
/// Never merges — if both old and new exist, the new wins and the old is
/// left untouched for the user to reconcile. Returns human-readable notes
/// (empty = nothing to do).
pub fn migrate_rename() -> Vec<String> {
    let old_config = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".config")
        })
        .join("llamacppcodeconf");
    let old_state = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".local/state")
        })
        .join("llamacppcodeconf");
    let mut notes = Vec::new();
    for (old, new) in [
        (old_config, config_dir()),
        (old_state, router::state_dir()),
    ] {
        if old.is_dir() && !new.exists() {
            match std::fs::rename(&old, &new) {
                Ok(()) => notes.push(format!("migrated {} → {}", old.display(), new.display())),
                Err(e) => notes.push(format!("could NOT migrate {}: {e}", old.display())),
            }
        }
    }
    if !notes.is_empty() {
        notes.push(
            "rename migration: a router started before the rename still references the \
             old preset path — Stop/Start it once; an installed systemd unit needs \
             --install-service again"
                .into(),
        );
    }
    notes
}

/// Rescue data written under a snap HOME redirect (the app launched
/// from e.g. VS Code's snap terminal wrote into
/// ~/snap/<app>/<revision>/...; the next snap update orphans it —
/// live casualty 2026-08-28). For each persistent dir: if the real
/// location is empty and a snap revision holds one, move it home.
pub fn migrate_snap_strays() -> Vec<String> {
    let mut notes = Vec::new();
    let snap_root = settings::real_home().join("snap");
    let targets = [
        (".config/modelsteward", config_dir()),
        (".local/state/modelsteward", router::state_dir()),
        (
            ".local/share/modelsteward",
            crate::core::managed::data_dir(),
        ),
    ];
    let Ok(apps) = std::fs::read_dir(&snap_root) else {
        return notes;
    };
    for app in apps.flatten() {
        let Ok(revs) = std::fs::read_dir(app.path()) else {
            continue;
        };
        for rev in revs.flatten() {
            for (rel, new) in &targets {
                let old = rev.path().join(rel);
                if old.is_dir() && !new.exists() {
                    notes.push(match std::fs::rename(&old, new) {
                        Ok(()) => format!(
                            "rescued snap-stranded data: {} → {}",
                            old.display(),
                            new.display()
                        ),
                        Err(e) => format!("could NOT rescue {}: {e}", old.display()),
                    });
                }
            }
        }
    }
    notes
}

pub fn load_config() -> settings::AppConfig {
    settings::AppConfig::load(&config_file())
}

pub fn preset_path() -> PathBuf {
    config_dir().join("router.ini")
}

/// A binary living in the app's own data dir (managed build or archive).
/// These are OFFERED, never forced: serving one requires an explicit pin
/// (live lesson 2026-08-28 — the managed b10673 out-built the user's
/// checkout and auto-pick silently started preferring it, which also
/// pointed the guided rebuild at the tag-pinned managed tree).
pub fn is_managed_install(path: &std::path::Path) -> bool {
    // find_installs stores canonicalized paths; canonicalize the data
    // dir too or a symlinked home defeats the guard and the newest
    // managed archive silently wins auto-pick again (review catch).
    let data = crate::core::managed::data_dir();
    let data = data.canonicalize().unwrap_or(data);
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    path.starts_with(data)
}

/// The llama-server to run: the configured override if set, else the newest
/// NON-MANAGED install that can see devices (managed builds and archives
/// serve only when pinned), else the newest non-managed at all — managed
/// as the last resort for machines with no other install (the crates.io
/// bootstrap case).
pub fn pick_server(cfg: &settings::AppConfig) -> Result<PathBuf> {
    if let Some(explicit) = &cfg.server_bin {
        anyhow::ensure!(
            explicit.is_file(),
            "configured llama-server does not exist: {}",
            explicit.display()
        );
        return Ok(explicit.clone());
    }
    let installs = discover::find_installs(&[]);
    let mut by_build: Vec<_> = installs.iter().collect();
    by_build.sort_by_key(|i| std::cmp::Reverse(i.build.unwrap_or(0)));
    let (own, managed): (Vec<_>, Vec<_>) = by_build
        .into_iter()
        .partition(|i| !is_managed_install(&i.server_path));
    own.iter()
        .find(|i| !discover::list_devices(&i.server_path).is_empty())
        .or(own.first())
        .or(managed.first())
        .map(|i| i.server_path.clone())
        .ok_or_else(|| anyhow::anyhow!(
            "no llama-server found — pick one in the GUI (Settings tab), set \
             \"server_bin\" in {}, or build one (GUI: Server → Build Advisor; \
             CLI: --advise)",
            config_file().display()
        ))
}

pub fn router_config(cfg: &settings::AppConfig) -> router::RouterConfig {
    router::RouterConfig {
        server_bin: pick_server(cfg).unwrap_or_else(|_| PathBuf::from("llama-server")),
        port: cfg.port,
        preset_path: preset_path(),
        models_max: cfg.models_max.max(1),
    }
}

pub fn scan_models(cfg: &settings::AppConfig, extra_dirs: &[PathBuf]) -> Vec<library::ModelFile> {
    let mut scan_dirs = cfg.scan_dirs.clone();
    scan_dirs.extend(extra_dirs.iter().cloned());
    library::scan(
        &scan_dirs,
        &library::default_ollama_stores(),
        library::default_hf_hub().as_deref(),
    )
}

pub fn scan_report(cfg: &settings::AppConfig, extra_dirs: &[PathBuf]) -> ScanReport {
    let installs = discover::find_installs(&[]);
    let mut by_build: Vec<_> = installs.iter().collect();
    by_build.sort_by_key(|i| std::cmp::Reverse(i.build.unwrap_or(0)));
    // Non-managed first: the device probe's install stamps env_build /
    // env_fingerprint into every measurement, and a newer managed
    // ARCHIVE that never serves must not claim measurements made on the
    // user's real binary (review catch 2026-08-28).
    let (own, managed_installs): (Vec<_>, Vec<_>) = by_build
        .into_iter()
        .partition(|i| !is_managed_install(&i.server_path));
    let (devices, devices_from) = own
        .iter()
        .chain(managed_installs.iter())
        .find_map(|i| {
            let d = discover::list_devices(&i.server_path);
            (!d.is_empty()).then(|| (d, Some(i.server_path.clone())))
        })
        .unwrap_or_default();

    ScanReport {
        installs,
        devices,
        devices_from,
        models: scan_models(cfg, extra_dirs),
    }
}

/// The build number of the install the devices were probed with — the
/// same one env_fingerprint hashes, exposed for measurement provenance
/// (the history journal records it per event).
pub fn env_build(report: &ScanReport) -> Option<u64> {
    report.devices_from.as_ref().and_then(|from| {
        report
            .installs
            .iter()
            .find(|i| &i.server_path == from)
            .and_then(|i| i.build)
    })
}

/// Environment half of a measurement fingerprint: which build measured, on
/// which devices. Pure over an existing scan so the GUI can compare without
/// re-probing.
pub fn env_fingerprint(report: &ScanReport) -> String {
    let build = env_build(report).unwrap_or(0);
    let devices: Vec<String> = report
        .devices
        .iter()
        .map(|d| format!("{}:{}", d.id, d.total_mib))
        .collect();
    router::fnv(&format!("b{build}|{}", devices.join(",")))
}

/// Text of a systemd *user* unit that runs the router at login. The unit
/// runs llama-server directly (not this app), with the same preset — the
/// router process it starts passes `router::find_preset_process`, so the
/// app recognizes it as ours and Stop works on it.
pub fn systemd_unit_text(cfg: &router::RouterConfig) -> String {
    format!(
        "[Unit]\n\
         Description=llama.cpp router (modelsteward)\n\
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
pub fn install_systemd_unit(cfg: &settings::AppConfig) -> Result<PathBuf> {
    let rcfg = router_config(cfg);
    if !rcfg.preset_path.exists() {
        write_preset(cfg, &[])?;
    }
    let dir = std::env::home_dir()
        .ok_or_else(|| anyhow::anyhow!("no home directory"))?
        .join(".config/systemd/user");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("llamacpp-router.service");
    std::fs::write(&path, systemd_unit_text(&rcfg))?;
    Ok(path)
}

/// (Re)generate the preset from a scan; returns (path, model count).
pub fn write_preset(
    cfg: &settings::AppConfig,
    extra_dirs: &[PathBuf],
) -> Result<(PathBuf, usize)> {
    let models = scan_models(cfg, extra_dirs);
    let mut entries = router::default_entries(&models);
    // Disabled models leave the preset — the router stops offering
    // them, calibrate/bench stop acting on them; the Library still
    // SHOWS them (dimmed) so nothing silently vanishes.
    entries.retain(|(alias, m, _)| {
        let by_path = cfg.disabled.contains(&m.path.display().to_string());
        !by_path && !cfg.disabled.contains(alias)
    });
    // Apply the user's per-model overrides (stored in config.json so preset
    // regeneration keeps them). Overrides keyed to ids that aren't preset
    // aliases (cache models) become bare sections configuring those ids.
    for (alias, m, ov) in &mut entries {
        if let Some(user_ov) = cfg.overrides.get(alias) {
            *ov = user_ov.clone();
        }
        // Feature keys ride along AFTER user overrides so they survive them
        // (a user override shouldn't silently strip a model's vision half).
        if let Some(proj) = &m.mmproj
            && !ov.no_mmproj
            && !ov.extra.iter().any(|(k, _)| k == "mmproj")
        {
            ov.extra.push(("mmproj".into(), proj.display().to_string()));
        }
        if library::is_embedding(m.meta.as_ref())
            && !ov.extra.iter().any(|(k, _)| k == "embedding")
        {
            ov.extra.push(("embedding".into(), "true".into()));
        }
    }
    let aliases: std::collections::HashSet<&str> =
        entries.iter().map(|(a, _, _)| a.as_str()).collect();
    let extra_sections: Vec<(String, router::ModelOverrides)> = cfg
        .overrides
        .iter()
        .filter(|(id, _)| !aliases.contains(id.as_str()))
        .map(|(id, ov)| (id.clone(), ov.clone()))
        .collect();
    let ini = router::render_preset(&entries, &extra_sections);
    std::fs::create_dir_all(config_dir())?;
    // The preset's [*] names this dir for slot snapshots — llama-server
    // won't create it, and a save into a missing dir is a 500.
    std::fs::create_dir_all(router::slot_save_dir())?;
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
            preset_path: PathBuf::from("/home/u/.config/modelsteward/router.ini"),
            models_max: 1,
        };
        let unit = systemd_unit_text(&cfg);
        assert!(unit.contains(
            "ExecStart=/opt/bin/llama-server --models-preset /home/u/.config/modelsteward/router.ini --host 127.0.0.1 --port 8080 --models-max 1"
        ));
        // The preset path in ExecStart is exactly what find_preset_process
        // matches on — this is the ownership handshake with the router module.
        assert!(unit.contains("WantedBy=default.target"));
    }

    #[test]
    fn env_fingerprint_changes_with_build_and_devices() {
        use crate::core::discover::{Device, LlamaInstall};
        let mk = |build, mib| ScanReport {
            installs: vec![LlamaInstall {
                server_path: PathBuf::from("/s"),
                build: Some(build),
                commit: None,
                built_with: None,
                backends: vec![],
            }],
            devices: vec![Device {
                id: "CUDA0".into(),
                name: "GPU".into(),
                total_mib: mib,
                free_mib: 0,
            }],
            devices_from: Some(PathBuf::from("/s")),
            models: vec![],
        };
        let a = env_fingerprint(&mk(10216, 24111));
        assert_eq!(a, env_fingerprint(&mk(10216, 24111)), "stable");
        assert_ne!(a, env_fingerprint(&mk(10360, 24111)), "build matters");
        assert_ne!(a, env_fingerprint(&mk(10216, 48000)), "devices matter");
    }
}
