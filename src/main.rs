//! Entry point. No arguments → the GUI. With arguments → the headless CLI
//! (same core, scriptable):
//!
//!   modelsteward --setup             one shot: start router if needed,
//!                                        measure anything unmeasured, sync
//!   modelsteward --scan [dir ...]    system report as JSON
//!   modelsteward --preset [dir ...]  (re)generate the router preset INI
//!   modelsteward --start [port]      start the router
//!   modelsteward --status [port]     router + per-model status as JSON
//!   modelsteward --reload [port]     ask the router to re-read the preset
//!   modelsteward --stop              stop the router we started
//!   modelsteward --calibrate [port] [force]
//!                                        measure settled contexts (skips
//!                                        fresh measurements unless `force`)
//!   modelsteward --sync [port]       write measured models into
//!                                        opencode.json (backs up first)
//!   modelsteward --advise            build advisor: check llama.cpp
//!                                        against upstream + this machine
//!   modelsteward --bench [id] [force]
//!                                        llama-bench baseline (pp/tg t/s)
//!                                        for one model, or every measured
//!                                        model missing a current baseline
//!   modelsteward --trial <id> [spec|ub|kv|load|dials|moe|vision|cache|slots] [keep <variant>|keep baseline]
//!                                        measured config trial: baseline +
//!                                        the menu's variants (spec = ngram
//!                                        speculation, ub = prefill batch),
//!                                        server-timed A/B with a verdict;
//!                                        `keep` persists a winner
//!   modelsteward --verify-rebuild    after a rebuild: restart router,
//!                                        re-measure stale, sync, and report
//!                                        what changed (unlocked / still
//!                                        locked / context shifts)
//!   modelsteward --quality <id> [shots]
//!                                        eval battery + N-shot tool probe
//!                                        (quality gate v2; feeds the
//!                                        quant-choice advisor)
//!   modelsteward --report            write the sanitized findings report
//!                                        (hardware + build + measurements +
//!                                        trial verdicts) for manual sharing
//!   modelsteward --install-service   write the systemd user unit
//!
//! Ports default to the configured value (~/.config/modelsteward/config.json).

use modelsteward::core::{advisor, bench, cancel, discover, opencode, router, settings, system, trial};
use std::path::PathBuf;

fn main() {
    // Runs before anything reads config/state: the 2026-08-26 rename
    // moved both directories, and the data must follow the name.
    for note in system::migrate_rename() {
        eprintln!("{note}");
    }
    let cfg = system::load_config();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None => {
            if let Err(e) = modelsteward::ui::run() {
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
        Some("--trial") => trial_cmd(&cfg, &args[1..]),
        Some("--verify-rebuild") => verify_rebuild(&cfg),
        Some("--quality") => quality_cmd(&cfg, &args[1..]),
        Some("--report") => match modelsteward::core::report::generate(&cfg) {
            Ok(path) => {
                println!("findings report written: {}", path.display());
                println!(
                    "review it before sharing — it is sanitized (no paths/usernames) but \
                     the judgment is yours; nothing is ever sent by the app"
                );
                Ok(())
            }
            Err(e) => Err(e),
        },
        Some("--install-service") => system::install_systemd_unit(&cfg).map(|path| {
            println!("unit written: {}", path.display());
            println!(
                "activate: systemctl --user daemon-reload && systemctl --user enable --now llamacpp-router"
            );
        }),
        _ => {
            eprintln!(
                "usage: modelsteward [no args → GUI] | --setup | --scan|--preset [dir ...] | --start|--status|--reload|--sync [port] | --calibrate [port] [force] | --bench [id] [force] | --trial <id> [keep <variant>] | --verify-rebuild | --quality <id> [shots] | --report | --advise | --install-service | --stop"
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
/// tokens/sec beside the context measurements. The logic lives in
/// core::bench::run_baselines, shared with the GUI's Server → Bench items.
fn bench_baselines(cfg: &settings::AppConfig, rest: &[String]) -> anyhow::Result<()> {
    let force = rest.iter().any(|a| a == "force");
    let target = rest.iter().find(|a| *a != "force").cloned();
    let n = bench::run_baselines(cfg, target, force, &cancel::CancelToken::default(), &mut |line| {
        println!("{line}")
    })?;
    if n > 0 {
        println!("{n} model(s) benched — Speed column and measurements.json updated");
    }
    Ok(())
}

/// M7 phase 2: measured speculative-decoding trial for one model.
/// `--trial <id>` runs baseline + the ngram variants and prints the
/// verdict; `--trial <id> keep <variant>` persists a winner into
/// config.json (or `keep baseline` to strip the knob).
fn trial_cmd(cfg: &settings::AppConfig, rest: &[String]) -> anyhow::Result<()> {
    let Some(model) = rest.first().filter(|a| *a != "keep") else {
        anyhow::bail!("--trial needs a model id (a router alias or cache id)");
    };
    // "slots" is a workflow measurement, not a config menu — no variants,
    // no keep; it prints its ceiling and returns.
    if rest.get(1).map(String::as_str) == Some("slots") {
        trial::run_slot_trial(
            cfg,
            model,
            &cancel::CancelToken::default(),
            &mut |line| println!("{line}"),
        )?;
        return Ok(());
    }
    let menu_name = rest
        .get(1)
        .filter(|a| trial::menu(a).is_some())
        .map(String::as_str)
        .unwrap_or("spec");
    let (variants, goal) = trial::menu(menu_name).expect("validated above");
    if let Some(pos) = rest.iter().position(|a| a == "keep") {
        let label = rest
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("keep what? name a variant or `baseline`"))?;
        trial::keep_variant(&system::config_file(), cfg, model, &variants, label)?;
        println!("{model}: kept {label} — config.json updated, preset regenerated, router reloaded");
        // The kept config changed the measured context; the agent config
        // follows it now, not at some future sync.
        sync(cfg)?;
        return Ok(());
    }
    let report = trial::run_trial(
        cfg,
        model,
        menu_name,
        &variants,
        goal,
        &cancel::CancelToken::default(),
        &mut |line| println!("{line}"),
    )?;
    match &report.verdict.winner {
        Some(w) => println!(
            "verdict: {w} wins — apply with: modelsteward --trial {model} {menu_name} keep {w}"
        ),
        None => println!("verdict: keep baseline"),
    }
    for nm in &report.near_misses {
        println!(
            "your call: {} gains {} but costs {} — apply with: modelsteward --trial {model} {menu_name} keep {}",
            nm.label, nm.gain, nm.cost, nm.label
        );
    }
    println!("\nwhy:");
    for para in trial::explain(&report) {
        println!("  {para}");
    }
    Ok(())
}

/// Quality gate v2: eval battery + N-shot tool probe for one model.
fn quality_cmd(cfg: &settings::AppConfig, rest: &[String]) -> anyhow::Result<()> {
    let Some(model) = rest.first() else {
        anyhow::bail!("--quality needs a model id");
    };
    let shots: u32 = rest.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    match router::status(&router::state_dir(), &system::router_config(cfg)) {
        router::RouterState::Ours { .. } => {}
        other => anyhow::bail!(
            "quality probe needs our router on port {}; state is {other:?}",
            cfg.port
        ),
    }
    let score = modelsteward::core::quality::run_and_record(
        cfg,
        model,
        shots,
        &cancel::CancelToken::default(),
        &mut |l| println!("{l}"),
    )?;
    println!(
        "{model}: eval battery {:.0}% ({}/{}), tool-call reliability {:.0}% over {} shots",
        score.eval_score * 100.0,
        score.evals_passed,
        score.evals_total,
        score.tool_reliability * 100.0,
        score.tool_shots
    );
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
    let report = system::scan_report(cfg, &[]);
    let env_fp = system::env_fingerprint(&report);
    let build = system::env_build(&report);
    // Refresh the preset first so newly downloaded files are servable and
    // therefore measurable — "new" means new-on-disk, not new-to-router.
    let (_, n) = system::write_preset(cfg, &[])?;
    eprintln!("preset refreshed ({n} models); reloading router");
    if let Err(e) = router::reload(cfg.port) {
        eprintln!("router reload failed ({e:#}) — measuring what it currently offers");
    }
    let embed = router::embedding_ids_in_preset(&system::preset_path());
    let results = router::calibrate(&dir, cfg.port, &env_fp, build, force, &embed, &mut |line| {
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
    let vision = router::vision_ids_in_preset(&system::preset_path());
    m.iter()
        .filter(|(id, _)| !embed.contains(id.as_str()))
        .filter_map(|(id, m)| {
            m.n_ctx.map(|ctx| opencode::DesiredModel {
                id: id.clone(),
                display_name: format!("{id} (llama.cpp)"),
                context: ctx,
                tool_call: m.tool_call,
                vision: vision.contains(id.as_str()),
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
    // Ghost cleanup (user decision 2026-08-26): only against a LIVE router.
    if let router::RouterState::Ours { models } =
        router::status(&router::state_dir(), &system::router_config(cfg))
    {
        let offered: Vec<String> = models.into_iter().map(|m| m.id).collect();
        for id in
            opencode::comment_out_ghosts(&path, &report.orphans, &offered, &measurements)?
        {
            println!("  ✂ {id}: commented out (router omits it, nothing measured — a ghost)");
        }
    }
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

/// M6 phase 2: the post-rebuild verification loop. Snapshot → restart the
/// router (a running router keeps serving the OLD binary; children exec
/// whatever is on disk, so the mix is subtle and worth ending) → measure
/// everything the new environment fingerprint marked stale → sync →
/// report what the rebuild actually changed, measured.
fn verify_rebuild(cfg: &settings::AppConfig) -> anyhow::Result<()> {
    let dir = router::state_dir();
    let before = router::read_measurements(&dir);
    match router::status(&dir, &system::router_config(cfg)) {
        router::RouterState::Ours { .. } => {
            println!("restarting router so it runs the new binary…");
            router::stop(&dir, &system::preset_path())?;
        }
        router::RouterState::Down => {}
        other => anyhow::bail!(
            "port {} is not ours to restart ({other:?}) — verification needs our router",
            cfg.port
        ),
    }
    setup(cfg)?;
    let after = router::read_measurements(&dir);
    println!("— rebuild verification —");
    for line in advisor::verify_summary(&advisor::verify_outcome(&before, &after)) {
        println!("{line}");
    }
    Ok(())
}
