//! CLI entry point. The GUI (M4) will live here too; until then the binary
//! exposes the headless core for testing and scripting:
//!
//!   llamacppcodeconf --scan [dir ...]   full system report as JSON
//!                                       (installs, devices, models)

use llamacppcodeconf::core::{discover, library};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Serialize)]
struct ScanReport {
    installs: Vec<discover::LlamaInstall>,
    /// Devices as seen by `devices_from`. Newest build is asked first, but a
    /// build that reports no devices (e.g. a CUDA build whose runtime can't
    /// initialize here) is skipped in favor of one that can actually see the
    /// hardware.
    devices: Vec<discover::Device>,
    devices_from: Option<std::path::PathBuf>,
    models: Vec<library::ModelFile>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("--scan") => {
            let extra_dirs: Vec<PathBuf> = args.map(PathBuf::from).collect();
            let report = scan_report(extra_dirs);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
        Some(other) => {
            eprintln!("unknown argument: {other}");
            eprintln!("usage: llamacppcodeconf --scan [extra-model-dir ...]");
            std::process::exit(2);
        }
        None => {
            eprintln!("GUI not built yet (M4). Try: llamacppcodeconf --scan");
            std::process::exit(2);
        }
    }
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

    let mut scan_dirs = extra_dirs;
    if let Some(home) = std::env::home_dir() {
        scan_dirs.push(home.join("models"));
    }
    let models = library::scan(&scan_dirs, &library::default_ollama_stores());

    ScanReport {
        installs,
        devices,
        devices_from,
        models,
    }
}
