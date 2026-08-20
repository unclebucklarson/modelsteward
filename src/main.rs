//! Entry point. No arguments → the GUI. With arguments → the headless CLI
//! (same core, scriptable):
//!
//!   llamacppcodeconf --setup             one shot: start router if needed,
//!                                        measure anything unmeasured, sync
//!   llamacppcodeconf --scan [dir ...]    system report as JSON
//!   llamacppcodeconf --preset [dir ...]  (re)generate the router preset INI
//!   llamacppcodeconf --start [port]      start the router
//!   llamacppcodeconf --status [port]     router + per-model status as JSON
//!   llamacppcodeconf --reload [port]     ask the router to re-read the preset
//!   llamacppcodeconf --stop              stop the router we started
//!   llamacppcodeconf --calibrate [port] [force]
//!                                        measure settled contexts (skips
//!                                        fresh measurements unless `force`)
//!   llamacppcodeconf --sync [port]       write measured models into
//!                                        opencode.json (backs up first)
//!   llamacppcodeconf --advise            build advisor: check llama.cpp
//!                                        against upstream + this machine
//!   llamacppcodeconf --bench [id] [force]
//!                                        llama-bench baseline (pp/tg t/s)
//!                                        for one model, or every measured
//!                                        model missing a current baseline
//!   llamacppcodeconf --install-service   write the systemd user unit
//!
//! Ports default to the configured value (~/.config/llamacppcodeconf/config.json).

use llamacppcodeconf::core::{
    advisor, bench, discover, library, opencode, router, rows, settings, system,
};
use std::path::PathBuf;

fn main() {
    let cfg = system::load_config();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None => {
            if let Err(e) = llamacppcodeconf::ui::run() {
                eprintln!("GUI failed: {e}");
                std::process::exit(1);
            }
            Ok(())
        }
        Some("--setup") => setup(&cfg),
        Some("--scan") => {
            let report = system::scan_report(&cfg, &paths_from(&args[1..]));
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            Ok(())
        }
        Some("--preset") => system::write_preset(&cfg, &paths_from(&args[1..])).map(|(path, n)| {
            println!("wrote {} with {n} models", path.display());
        }),
        Some("--start") => start(&with_port(&cfg, &args[1..])),
        Some("--status") => {
            let cfg = with_port(&cfg, &args[1..]);
            let state = router::status(&router::state_dir(), &system::router_config(&cfg));
            println!("{}", serde_json::to_string_pretty(&state).unwrap());
            Ok(())
        }
        Some("--reload") => router::reload(with_port(&cfg, &args[1..]).port).map(|models| {
            println!("{}", serde_json::to_string_pretty(&models).unwrap());
        }),
        Some("--stop") => router::stop(&router::state_dir(), &system::preset_path()),
        Some("--calibrate") => {
            let force = args[1..].iter().any(|a| a == "force");
            calibrate(&with_port(&cfg, &args[1..]), force)
        }
        Some("--sync") => sync(&with_port(&cfg, &args[1..])),
        Some("--advise") => {
            let server = system::pick_server(&cfg).ok();
            let build = server.as_deref().and_then(discover::build_of);
            let measurements = router::read_measurements(&router::state_dir());
            let log = std::fs::read_to_string(router::state_dir().join("router.log")).ok();
            let check = advisor::check(server, build, &measurements, log.as_deref());
            for (headline, detail) in advisor::verdicts(&check) {
                println!("• {headline}\n  {detail}");
            }
            let sel = advisor::default_backends(&check);
            println!("\nA rebuild (backends: {}) would run:", {
                let mut v = Vec::new();
                if sel.cuda { v.push("CUDA"); }
                if sel.vulkan { v.push("Vulkan"); }
                if sel.hip { v.push("ROCm"); }
                if v.is_empty() { v.push("CPU-only"); }
                v.join(" + ")
            });
            for (cmd, args) in advisor::rebuild_commands(&check, sel) {
                println!("  {cmd} {}", args.join(" "));
            }
            Ok(())
        }
        Some("--bench") => bench_baselines(&cfg, &args[1..]),
        Some("--install-service") => system::install_systemd_unit(&cfg).map(|path| {
            println!("unit written: {}", path.display());
            println!(
                "activate: systemctl --user daemon-reload && systemctl --user enable --now llamacpp-router"
            );
        }),
        _ => {
            eprintln!(
                "usage: llamacppcodeconf [no args → GUI] | --setup | --scan|--preset [dir ...] | --start|--status|--reload|--sync [port] | --calibrate [port] [force] | --bench [id] [force] | --advise | --install-service | --stop"
            );
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

/// M7 baselines: run llama-bench (pp512 + tg128) per model and store the
/// tokens/sec beside the context measurements. llama-bench loads the model
/// itself, so the GPU must be free first — our router gets its loaded models
/// unloaded; a server we didn't start is never touched, only reported.
fn bench_baselines(cfg: &settings::AppConfig, rest: &[String]) -> anyhow::Result<()> {
    let force = rest.iter().any(|a| a == "force");
    let target = rest.iter().find(|a| *a != "force").cloned();

    let server = system::pick_server(cfg)?;
    let bench_bin = bench::bench_bin(&server);
    let current_build = discover::build_of(&server);

    let dir = router::state_dir();
    match router::status(&dir, &system::router_config(cfg)) {
        router::RouterState::Down => {}
        router::RouterState::Ours { models } => {
            for m in models
                .iter()
                .filter(|m| matches!(m.status.as_str(), "loaded" | "loading" | "sleeping"))
            {
                println!("unloading {} to free the GPU for benching…", m.id);
                router::unload_model(cfg.port, &m.id)?;
                router::wait_until_not_loaded(
                    cfg.port,
                    &m.id,
                    std::time::Duration::from_secs(30),
                );
            }
        }
        other => anyhow::bail!(
            "port {} is running a server this app doesn't own ({other:?}); \
             benching needs the GPU free, and that server is not ours to unload",
            cfg.port
        ),
    }

    // The same id → file mapping the Library rows use.
    let models = system::scan_models(cfg, &[]);
    let ids_by_path = rows::router_ids_by_path(&models);
    let mut by_id: std::collections::BTreeMap<String, &library::ModelFile> =
        std::collections::BTreeMap::new();
    for m in &models {
        if let Some(id) = ids_by_path.get(&m.path) {
            by_id.insert(id.clone(), m);
        }
    }

    let mut measurements = router::read_measurements(&dir);
    let targets: Vec<String> = match target {
        Some(id) => {
            anyhow::ensure!(
                by_id.contains_key(&id),
                "unknown model id {id:?} — --bench needs an id that maps to a file on disk \
                 (preset alias or hub-cache id; see --status)"
            );
            vec![id]
        }
        // Default sweep: every model proven loadable (measured ctx) whose
        // baseline is missing or from another build. Embedding models are
        // skipped — llama-bench's text baseline doesn't describe them.
        None => by_id
            .iter()
            .filter(|(id, file)| {
                let m = measurements.get(*id);
                let loadable = m.is_some_and(|m| m.n_ctx.is_some());
                let fresh = m.is_some_and(|m| {
                    m.pp_tps.is_some() && m.tg_tps.is_some() && m.bench_build == current_build
                });
                loadable
                    && !library::is_embedding(file.meta.as_ref())
                    && (force || !fresh)
            })
            .map(|(id, _)| id.clone())
            .collect(),
    };
    if targets.is_empty() {
        println!(
            "nothing to bench — every measured model already has a baseline from this \
             build (re-run with `force`, or name a model id)"
        );
        return Ok(());
    }

    let total = targets.len();
    for (i, id) in targets.iter().enumerate() {
        let n = i + 1;
        let file = by_id[id];
        let kv = cfg
            .overrides
            .get(id)
            .and_then(|o| o.cache_type_kv.clone())
            .unwrap_or_else(|| router::DEFAULT_KV_TYPE.to_string());
        let extra = vec!["-ctk".to_string(), kv.clone(), "-ctv".to_string(), kv];
        println!(
            "[{n}/{total}] benching {id} (pp512 + tg128 ×3 — a 27B takes about a minute)…"
        );
        match bench::run(&bench_bin, &file.path, &extra) {
            Ok(b) => {
                let fmt = |v: Option<f64>| v.map(|t| format!("{t:.1}")).unwrap_or("?".into());
                println!(
                    "[{n}/{total}] {id}: pp {} t/s, tg {} t/s",
                    fmt(b.pp_tps),
                    fmt(b.tg_tps)
                );
                let mut entry = measurements.get(id).cloned().unwrap_or_default();
                entry.pp_tps = b.pp_tps;
                entry.tg_tps = b.tg_tps;
                entry.bench_build = b.build;
                measurements.insert(id.clone(), entry);
                router::write_measurements(&dir, &measurements)?; // persist per model
            }
            Err(e) => eprintln!("[{n}/{total}] {id}: bench failed: {e:#}"),
        }
    }
    Ok(())
}

fn paths_from(rest: &[String]) -> Vec<PathBuf> {
    rest.iter().map(PathBuf::from).collect()
}

/// A positional port argument overrides the configured one for this run.
fn with_port(cfg: &settings::AppConfig, rest: &[String]) -> settings::AppConfig {
    let mut cfg = cfg.clone();
    if let Some(p) = rest.first().and_then(|p| p.parse().ok()) {
        cfg.port = p;
    }
    cfg
}

fn start(cfg: &settings::AppConfig) -> anyhow::Result<()> {
    if !system::preset_path().exists() {
        let (path, n) = system::write_preset(cfg, &[])?;
        println!("wrote {} with {n} models", path.display());
    }
    let rcfg = system::router_config(cfg);
    let pid = router::start(&router::state_dir(), &rcfg)?;
    println!(
        "router started: pid {pid}, port {}, preset {}, log {}",
        cfg.port,
        rcfg.preset_path.display(),
        router::state_dir().join("router.log").display()
    );
    Ok(())
}

fn calibrate(cfg: &settings::AppConfig, force: bool) -> anyhow::Result<()> {
    let dir = router::state_dir();
    match router::status(&dir, &system::router_config(cfg)) {
        router::RouterState::Ours { .. } => {}
        other => anyhow::bail!(
            "calibration needs our router running on port {}; state is {other:?}. \
             Try --start first.",
            cfg.port
        ),
    }
    let env_fp = system::env_fingerprint(&system::scan_report(cfg, &[]));
    let embed = router::embedding_ids_in_preset(&system::preset_path());
    let results = router::calibrate(&dir, cfg.port, &env_fp, force, &embed, &mut |line| {
        eprintln!("{line}");
    })?;
    for (id, m) in &results {
        match (&m.n_ctx, &m.error) {
            (Some(ctx), _) => println!("{id}: settled context {ctx}"),
            (None, Some(e)) => println!("{id}: FAILED — {e}"),
            _ => println!("{id}: unmeasured"),
        }
    }
    println!("stored in {}", dir.join("measurements.json").display());
    Ok(())
}

fn desired_from_measurements(m: &router::Measurements) -> Vec<opencode::DesiredModel> {
    // Embedding models serve /v1/embeddings — they don't belong in the
    // chat/agent config.
    let embed = router::embedding_ids_in_preset(&system::preset_path());
    m.iter()
        .filter(|(id, _)| !embed.contains(id.as_str()))
        .filter_map(|(id, m)| {
            m.n_ctx.map(|ctx| opencode::DesiredModel {
                id: id.clone(),
                display_name: format!("{id} (llama.cpp)"),
                context: ctx,
                tool_call: m.tool_call,
            })
        })
        .collect()
}

fn sync(cfg: &settings::AppConfig) -> anyhow::Result<()> {
    let measurements = router::read_measurements(&router::state_dir());
    let desired = desired_from_measurements(&measurements);
    if desired.is_empty() {
        anyhow::bail!("no successful measurements yet — run --calibrate first (measured, not guessed)");
    }
    let path = opencode::default_config_path();
    let base_url = format!("http://127.0.0.1:{}/v1", cfg.port);
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

/// The one-shot: preset if missing → start if down → wait healthy →
/// incremental calibrate → sync. Everything narrated; nothing guessed.
fn setup(cfg: &settings::AppConfig) -> anyhow::Result<()> {
    let dir = router::state_dir();
    let rcfg = system::router_config(cfg);
    match router::status(&dir, &rcfg) {
        router::RouterState::Ours { .. } => println!("router already running — good"),
        router::RouterState::Down => {
            start(cfg)?;
            print!("waiting for router");
            let mut up = false;
            for _ in 0..30 {
                std::thread::sleep(std::time::Duration::from_secs(1));
                if matches!(
                    router::status(&dir, &rcfg),
                    router::RouterState::Ours { .. }
                ) {
                    up = true;
                    break;
                }
                print!(".");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            println!();
            anyhow::ensure!(up, "router did not come up within 30s — see router.log");
        }
        other => anyhow::bail!("port {} is not ours to set up: {other:?}", cfg.port),
    }
    calibrate(cfg, false)?;
    sync(cfg)
}
