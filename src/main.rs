//! CLI entry point. The GUI (M4) will live here too; until then the binary
//! exposes the headless core for testing and scripting:
//!
//!   llamacppcodeconf --scan [dir ...]    system report as JSON
//!   llamacppcodeconf --preset [dir ...]  (re)generate the router preset INI
//!   llamacppcodeconf --start [port]      start the router (default port 8080)
//!   llamacppcodeconf --status [port]     router + per-model status as JSON
//!   llamacppcodeconf --reload [port]     ask the router to re-read the preset
//!   llamacppcodeconf --stop              stop the router we started
//!   llamacppcodeconf --calibrate [port]  measure each preset model's settled
//!                                        context (loads each once — slow)
//!   llamacppcodeconf --sync [port]       write measured models into
//!                                        opencode.json (backs up first)

use llamacppcodeconf::core::{discover, library, opencode, router};
use serde::Serialize;
use std::path::PathBuf;

const DEFAULT_PORT: u16 = 8080;

#[derive(Serialize)]
struct ScanReport {
    installs: Vec<discover::LlamaInstall>,
    /// Devices as seen by `devices_from`. Newest build is asked first, but a
    /// build that reports no devices (e.g. a CUDA build whose runtime can't
    /// initialize here) is skipped in favor of one that can actually see the
    /// hardware.
    devices: Vec<discover::Device>,
    devices_from: Option<PathBuf>,
    models: Vec<library::ModelFile>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("--scan") => {
            let report = scan_report(paths_from(&args[1..]));
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            Ok(())
        }
        Some("--preset") => write_preset(paths_from(&args[1..])),
        Some("--start") => start(port_from(&args[1..])),
        Some("--status") => {
            let state = router::status(&router::state_dir(), &router_config(port_from(&args[1..])));
            println!("{}", serde_json::to_string_pretty(&state).unwrap());
            Ok(())
        }
        Some("--reload") => router::reload(port_from(&args[1..])).map(|models| {
            println!("{}", serde_json::to_string_pretty(&models).unwrap());
        }),
        Some("--stop") => router::stop(&router::state_dir()),
        Some("--calibrate") => calibrate(port_from(&args[1..])),
        Some("--sync") => sync(port_from(&args[1..])),
        _ => {
            eprintln!(
                "usage: llamacppcodeconf --scan|--preset [dir ...] | --start|--status|--reload|--calibrate|--sync [port] | --stop"
            );
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn paths_from(rest: &[String]) -> Vec<PathBuf> {
    rest.iter().map(PathBuf::from).collect()
}

fn port_from(rest: &[String]) -> u16 {
    rest.first()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

fn config_dir() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".config")
        })
        .join("llamacppcodeconf")
}

fn preset_path() -> PathBuf {
    config_dir().join("router.ini")
}

/// The llama-server to run: newest install that can actually see devices,
/// else the newest install at all.
fn pick_server() -> anyhow::Result<PathBuf> {
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

fn router_config(port: u16) -> router::RouterConfig {
    router::RouterConfig {
        server_bin: pick_server().unwrap_or_else(|_| PathBuf::from("llama-server")),
        port,
        preset_path: preset_path(),
        models_max: 1,
    }
}

fn scan_models(extra_dirs: Vec<PathBuf>) -> Vec<library::ModelFile> {
    let mut scan_dirs = extra_dirs;
    if let Some(home) = std::env::home_dir() {
        scan_dirs.push(home.join("models"));
    }
    library::scan(&scan_dirs, &library::default_ollama_stores())
}

fn scan_report(extra_dirs: Vec<PathBuf>) -> ScanReport {
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

fn write_preset(extra_dirs: Vec<PathBuf>) -> anyhow::Result<()> {
    let models = scan_models(extra_dirs);
    let entries = router::default_entries(&models);
    let ini = router::render_preset(&entries);
    std::fs::create_dir_all(config_dir())?;
    std::fs::write(preset_path(), &ini)?;
    println!(
        "wrote {} with {} models",
        preset_path().display(),
        entries.len()
    );
    Ok(())
}

fn calibrate(port: u16) -> anyhow::Result<()> {
    let dir = router::state_dir();
    let cfg = router_config(port);
    match router::status(&dir, &cfg) {
        router::RouterState::Ours { .. } => {}
        other => anyhow::bail!(
            "calibration needs our router running on port {port}; state is {other:?}. \
             Try --start first."
        ),
    }
    let results = router::calibrate(&dir, port, &mut |id, i, n| {
        eprintln!("[{i}/{n}] measuring {id} (loads the model — be patient)…");
    })?;
    for (id, m) in &results {
        println!("{id}: settled context {}", m.n_ctx);
    }
    println!("stored in {}", dir.join("measurements.json").display());
    Ok(())
}

fn sync(port: u16) -> anyhow::Result<()> {
    let measurements = router::read_measurements(&router::state_dir());
    if measurements.is_empty() {
        anyhow::bail!("no measurements yet — run --calibrate first (measured, not guessed)");
    }
    let desired: Vec<opencode::DesiredModel> = measurements
        .iter()
        .map(|(id, m)| opencode::DesiredModel {
            id: id.clone(),
            display_name: format!("{id} (llama.cpp)"),
            context: m.n_ctx,
        })
        .collect();
    let path = opencode::default_config_path();
    let base_url = format!("http://127.0.0.1:{port}/v1");
    let report = opencode::sync_file(&path, &base_url, &desired)?;
    println!(
        "synced {}: {} added, {} updated",
        path.display(),
        report.added.len(),
        report.updated.len()
    );
    for id in &report.added {
        println!("  + {id}");
    }
    for id in &report.updated {
        println!("  ~ {id} (context refreshed)");
    }
    if !report.orphans.is_empty() {
        println!("orphans (in config, not measured — left untouched):");
        for id in &report.orphans {
            println!("  ? {id}");
        }
    }
    Ok(())
}

fn start(port: u16) -> anyhow::Result<()> {
    if !preset_path().exists() {
        write_preset(Vec::new())?;
    }
    let cfg = router_config(port);
    let pid = router::start(&router::state_dir(), &cfg)?;
    println!(
        "router started: pid {pid}, port {port}, preset {}, log {}",
        cfg.preset_path.display(),
        router::state_dir().join("router.log").display()
    );
    Ok(())
}
