//! Entry point. No arguments → the GUI. With arguments → the headless CLI
//! (same core, scriptable):
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

use llamacppcodeconf::core::{opencode, router, system};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None => {
            if let Err(e) = llamacppcodeconf::ui::run() {
                eprintln!("GUI failed: {e}");
                std::process::exit(1);
            }
            Ok(())
        }
        Some("--scan") => {
            let report = system::scan_report(&paths_from(&args[1..]));
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            Ok(())
        }
        Some("--preset") => system::write_preset(&paths_from(&args[1..])).map(|(path, n)| {
            println!("wrote {} with {n} models", path.display());
        }),
        Some("--start") => start(port_from(&args[1..])),
        Some("--status") => {
            let state = router::status(
                &router::state_dir(),
                &system::router_config(port_from(&args[1..])),
            );
            println!("{}", serde_json::to_string_pretty(&state).unwrap());
            Ok(())
        }
        Some("--reload") => router::reload(port_from(&args[1..])).map(|models| {
            println!("{}", serde_json::to_string_pretty(&models).unwrap());
        }),
        Some("--stop") => router::stop(&router::state_dir(), &system::preset_path()),
        Some("--install-service") => {
            system::install_systemd_unit(port_from(&args[1..])).map(|path| {
                println!("unit written: {}", path.display());
                println!("activate: systemctl --user daemon-reload && systemctl --user enable --now llamacpp-router");
            })
        }
        Some("--calibrate") => calibrate(port_from(&args[1..])),
        Some("--sync") => sync(port_from(&args[1..])),
        _ => {
            eprintln!(
                "usage: llamacppcodeconf [no args → GUI] | --scan|--preset [dir ...] | --start|--status|--reload|--calibrate|--sync|--install-service [port] | --stop"
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
        .unwrap_or(system::DEFAULT_PORT)
}

fn start(port: u16) -> anyhow::Result<()> {
    if !system::preset_path().exists() {
        let (path, n) = system::write_preset(&[])?;
        println!("wrote {} with {n} models", path.display());
    }
    let cfg = system::router_config(port);
    let pid = router::start(&router::state_dir(), &cfg)?;
    println!(
        "router started: pid {pid}, port {port}, preset {}, log {}",
        cfg.preset_path.display(),
        router::state_dir().join("router.log").display()
    );
    Ok(())
}

fn calibrate(port: u16) -> anyhow::Result<()> {
    let dir = router::state_dir();
    let cfg = system::router_config(port);
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
