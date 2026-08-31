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
//!   modelsteward --meter [today|24h|7d]  # token ledger report (all time without a range)
//!   modelsteward --sync [port]       write measured models into
//!                                        opencode.json (backs up first)
//!   modelsteward --advise            build advisor: check llama.cpp
//!                                        against upstream + this machine
//!   modelsteward --bench [id] [force]
//!                                        llama-bench baseline (pp/tg t/s)
//!                                        for one model, or every measured
//!                                        model missing a current baseline
//!   modelsteward --trial <id> [spec|ub|kv|load|dials|moe|vision|cache|ckpt|slots] [keep <variant>|keep baseline]
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

use modelsteward::core::{
    advisor, bench, cancel, diagnose, discover, hermes, opencode, piagent, router, settings,
    system, trial,
};
use std::path::PathBuf;

const USAGE: &str = "usage: modelsteward [no args → GUI] | --setup | --scan|--preset [dir ...] \
| --start|--status|--reload|--stop [port] | --calibrate [port] [force] | --bench [id] [force] \
| --trial <id> [spec|ub|kv|load|dials|moe|vision|cache|ckpt|slots] [keep <variant>|keep baseline] \
| --quality <id> [shots] | --meter [today|24h|7d] | --sync [port] | --verify-rebuild | --report \
| --advise | --config | --install-service | --help | --version";

const HELP: &str = "modelsteward — measured, not guessed: a llama.cpp router manager that tunes \
local models and keeps OpenAI-compatible apps configured from real measurements.

  (no args)             launch the GUI
  --setup               one-shot: start router + measure + sync
  --scan [dir ...]      what's on this machine (JSON)
  --preset [dir ...]    write ~/.config/modelsteward/router.ini
  --start / --stop / --status / --reload [port]
  --calibrate [port] [force]   measure new/stale models (force = all)
  --bench [id] [force]  llama-bench pp/tg speed baselines
  --trial <id> [menu] [keep <variant>|keep baseline]
        menus: spec  speculation (ngram modes)      ub    prefill batch size
               kv    KV-cache precision             load  load/hot-swap mode
               dials speculation fine-tuning        moe   MoE expert placement
               vision serve with/without projector  cache --cache-reuse sweep
               ckpt  context checkpoints            slots slot save/restore ceiling
  --quality <id> [shots]  eval battery + tool calls + multi-hop agent loops
  --meter [today|24h|7d]  the token ledger: usage, cache %, measured cost
  --sync [port]         write measured limits into opencode.json
  --verify-rebuild      after a rebuild: restart, re-measure, report
  --report              shareable findings report (sanitized md + JSON)
  --advise              build advisor report
  --config              print the config file path + effective settings
  --install-service     write a systemd user unit for the router

exit codes: 0 ok · 1 error · 2 bad usage · 3 partial (some models measured/benched, some failed)
streams: progress → stderr, results → stdout (safe to pipe/redirect results)

files: config ~/.config/modelsteward/config.json · preset router.ini (same dir)
       measurements/trials/meter ~/.local/state/modelsteward/
docs:  https://github.com/unclebucklarson/modelsteward";

fn main() {
    // Pure-info commands answer before ANY state work — no migrations,
    // no config load, and above all no trial-heal (live incident
    // 2026-08-30: `--version` beside a running GUI campaign triggered
    // the heal and yanked the trial's preset mid-round).
    match std::env::args().nth(1).as_deref() {
        Some("--help" | "-h" | "help") => {
            println!("{HELP}");
            return;
        }
        Some("--version" | "-V") => {
            println!("modelsteward {}", system::build_id());
            return;
        }
        _ => {}
    }
    // Runs before anything reads config/state: the 2026-08-26 rename
    // moved both directories, and the data must follow the name.
    for note in system::migrate_snap_strays() {
        eprintln!("{note}");
    }
    for note in system::migrate_rename() {
        eprintln!("{note}");
    }
    let cfg = system::load_config();
    if let Some(note) = trial::heal_interrupted_trial(&cfg) {
        eprintln!("{note}");
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None => {
            if let Err(e) = modelsteward::ui::run() {
                eprintln!("GUI failed: {e}");
                eprintln!(
                    "(no display? the CLI works everywhere — run `modelsteward --help`)"
                );
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
        Some("--start") => with_port(&cfg, &args[1..]).and_then(|c| start(&c)),
        Some("--status") => with_port(&cfg, &args[1..]).map(|c| {
            let state = router::status(&router::state_dir(), &system::router_config(&c));
            println!("{}", serde_json::to_string_pretty(&state).unwrap());
        }),
        Some("--reload") => with_port(&cfg, &args[1..])
            .and_then(|c| router::reload(c.port))
            .map(|models| {
                println!("{}", serde_json::to_string_pretty(&models).unwrap());
            }),
        Some("--stop") => router::stop(&router::state_dir(), &system::preset_path()),
        Some("--calibrate") => {
            let force = args[1..].iter().any(|a| a == "force");
            with_port(&cfg, &args[1..]).and_then(|c| calibrate(&c, force))
        }
        Some("--sync") => with_port(&cfg, &args[1..]).and_then(|c| {
            if c.port != cfg.port {
                // Usability review C14: a one-off port bakes into
                // opencode.json and outlives this run.
                eprintln!(
                    "note: writing baseURL for port {} into opencode.json, but your \
                     configured port is {} — agents point at {} until you --sync again",
                    c.port, cfg.port, c.port
                );
            }
            sync(&c)
        }),
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
        Some("--meter") => meter_cmd(&cfg, args.get(1).map(String::as_str)),
        Some("--report") => match modelsteward::core::report::generate(&cfg) {
            Ok(path) => {
                // Usability review C18: TWO shareable files are written;
                // the warning must cover both.
                println!("findings report written: {}", path.display());
                println!("machine-readable twin: {}", path.with_extension("json").display());
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
        Some("--config") => {
            // Usability review C21/C8: make the effective settings and
            // their file inspectable — a corrupt or surprising config
            // becomes diagnosable in one command.
            println!("config file:   {}", system::config_file().display());
            println!(
                "measurements:  {}",
                router::state_dir().join("measurements.json").display()
            );
            println!("effective settings:");
            println!("{}", serde_json::to_string_pretty(&cfg).unwrap());
            Ok(())
        }
        Some("--help" | "-h" | "help") => {
            // Usability review C1: --help used to look like a crash
            // (usage → stderr, exit 2). Full help, stdout, exit 0.
            println!("{HELP}");
            Ok(())
        }
        Some("--version" | "-V") => {
            // Full build identity — what a bug report should carry.
            println!("modelsteward {}", system::build_id());
            Ok(())
        }
        _ => {
            eprintln!("{USAGE}\nrun `modelsteward --help` for details");
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

/// The token-ledger report — the shared core::system implementation,
/// harvesting the live log first so the numbers are current.
fn meter_cmd(cfg: &settings::AppConfig, range: Option<&str>) -> anyhow::Result<()> {
    print!("{}", system::meter_report_text(cfg, range, true)?);
    Ok(())
}

/// M7 baselines: run llama-bench (pp512 + tg128) per model and store the
/// tokens/sec beside the context measurements. The logic lives in
/// core::bench::run_baselines, shared with the GUI's Server → Bench items.
fn bench_baselines(cfg: &settings::AppConfig, rest: &[String]) -> anyhow::Result<()> {
    let force = rest.iter().any(|a| a == "force");
    let target = rest.iter().find(|a| *a != "force").cloned();
    let (n, failed) =
        bench::run_baselines(cfg, target, force, &cancel::CancelToken::default(), &mut |line| {
            eprintln!("{line}")
        })?;
    if n > 0 {
        println!(
            "{n} model(s) benched, {failed} failed — baselines stored in \
             measurements.json (the Library's Speed column)"
        );
    } else if failed == 0 {
        // Usability review C15: silence looked like a hang.
        println!("all baselines current — nothing to do (add `force` to re-run, or name a model id)");
    }
    if n == 0 && failed > 0 {
        anyhow::bail!("every bench failed — the lines above say why");
    }
    if failed > 0 {
        // Partial failure: scripted callers must not read this as green.
        std::process::exit(3);
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
            &mut |line| eprintln!("{line}"),
        )?;
        return Ok(());
    }
    // A typo'd menu is an error, never a silent different experiment
    // (usability review C3).
    let menu_name = trial::resolve_menu_arg(
        rest.get(1).map(String::as_str).filter(|a| *a != "keep"),
    )?;
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
        &mut |line| eprintln!("{line}"),
    )?;
    // The numbers themselves, aligned (usability review C11: the glossary
    // used to describe a table that was never printed).
    println!("{}", trial::fmt_results_table(&report.raced));
    println!();
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
    let shots: u32 = match rest.get(1) {
        Some(s) => s.parse().map_err(|_| {
            anyhow::anyhow!("shots must be a number, got {s:?} — e.g. --quality {model} 10")
        })?,
        None => 5,
    };
    match router::status(&router::state_dir(), &system::router_config(cfg)) {
        router::RouterState::Ours { .. } => {}
        other => anyhow::bail!(
            "quality probe needs our router on port {}; {other}",
            cfg.port
        ),
    }
    let score = modelsteward::core::quality::run_and_record(
        cfg,
        model,
        shots,
        &cancel::CancelToken::default(),
        &mut |l| eprintln!("{l}"),
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
/// A positional that ISN'T a port is an error, not a silent no-op
/// (usability review C12: `--start 80800` used to quietly use the
/// configured port).
fn with_port(cfg: &settings::AppConfig, rest: &[String]) -> anyhow::Result<settings::AppConfig> {
    let mut cfg = cfg.clone();
    if let Some(a) = rest.iter().find(|a| *a != "force") {
        cfg.port = a.parse().map_err(|_| {
            anyhow::anyhow!("not a port: {a:?} (expected a number 1-65535)")
        })?;
    }
    Ok(cfg)
}

fn start(cfg: &settings::AppConfig) -> anyhow::Result<()> {
    if let Err(e) = router::router_mode_supported(discover::build_of(
        &system::router_config(cfg).server_bin,
    )) {
        anyhow::bail!(e);
    }
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
            "measuring needs our router running on port {}; {other}. \
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
    let (mut measured, mut failed) = (0usize, 0usize);
    for (id, m) in &results {
        match (&m.n_ctx, &m.error) {
            (Some(ctx), _) => {
                measured += 1;
                println!("{id}: settled context {ctx} tokens");
            }
            (None, Some(e)) => {
                failed += 1;
                println!("{id}: FAILED — {e}");
                // The same plain-language brain the GUI's Why? uses
                // (usability review C6).
                println!(
                    "  ↳ {}",
                    diagnose::diagnose(Some(e), false, false, None, id).explanation
                );
            }
            _ => println!("{id}: unmeasured"),
        }
    }
    println!(
        "{measured} measured, {failed} failed — stored in {}",
        dir.join("measurements.json").display()
    );
    if measured == 0 && failed > 0 {
        anyhow::bail!("every model failed to measure — the lines above say why");
    }
    if failed > 0 {
        // Usability review C5: `--calibrate && --sync` must not read
        // green on a partial run. 3 = partial failure.
        std::process::exit(3);
    }
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
    if report.skipped_missing {
        // NOT a return: pi and Hermes are independent of OpenCode, and
        // returning here made `--sync` silently skip them on a machine
        // without OpenCode while the GUI synced them — same command, two
        // answers (review finding H7, 2026-08-31).
        println!(
            "opencode.json not found ({}) — OpenCode isn't installed; skipped. \
             Other apps connect via the base URL (see --help / Connections tab).",
            path.display()
        );
    } else {
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
    }
    // pi coding agent (Connections p2, 2026-08-30): measured context
    // windows into ~/.pi/agent/models.json — pi's native router
    // integration assumes 128k when the router doesn't report n_ctx.
    let pi_path = piagent::default_models_path();
    let known = router::ids_in_preset(&system::preset_path());
    match piagent::sync_file_with_known(&pi_path, &base_url, &desired, &known) {
        Ok(r) if r.skipped_missing => {}
        Ok(r) => {
            println!(
                "pi agent synced ({}): {} added, {} updated, {} removed{}",
                pi_path.display(),
                r.added.len(),
                r.updated.len(),
                r.removed.len(),
                if r.created_file { " — models.json created" } else { "" },
            );
            if !r.kept_unmeasured.is_empty() {
                println!(
                    "  ~ {} entr(ies) kept although not measurable right now — a \
                     transient load failure never deletes a config entry",
                    r.kept_unmeasured.len()
                );
            }
        }
        Err(e) => eprintln!("pi agent sync FAILED: {e:#}"),
    }
    // Hermes: measured contexts into its context cache. Registering the
    // provider itself is an explicit GUI action (it edits a live,
    // hand-maintained config) — the CLI reports when it's missing.
    let home = hermes::default_home();
    match hermes::sync(&home, &base_url, &desired) {
        Ok(r) if r.skipped_missing => {}
        Ok(r) => {
            println!("Hermes synced: {} context(s) written", r.written.len());
            if !r.below_minimum.is_empty() {
                println!(
                    "  ! {} model(s) skipped — under Hermes's 64,000-token minimum: {}",
                    r.below_minimum.len(),
                    r.below_minimum.join(", ")
                );
            }
            if r.provider_unregistered {
                println!(
                    "  ? no Hermes custom provider points at this router yet — \
                     register one in the GUI (Connections tab) or Hermes's own /model"
                );
            }
        }
        Err(e) => eprintln!("Hermes sync FAILED: {e:#}"),
    }
    Ok(())
}

/// The one-shot: preset if missing → start if down → wait healthy →
/// incremental calibrate → sync. Everything narrated; nothing guessed.
fn setup(cfg: &settings::AppConfig) -> anyhow::Result<()> {
    // Fail early and name the actual problem (usability review C7: a
    // model-less machine used to run the whole pipeline and die with a
    // circular 'run --calibrate first').
    let models = system::scan_models(cfg, &[]);
    if models.is_empty() {
        anyhow::bail!(
            "no GGUF models found. Looked in: {} — plus any Ollama store and the \
             HuggingFace cache. Put a .gguf under one of those, add a directory to \
             scan_dirs in {}, or pull one with Ollama, then re-run --setup",
            cfg.scan_dirs
                .iter()
                .map(|d| d.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
            system::config_file().display()
        );
    }
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
        other => anyhow::bail!("port {} is not ours to set up: {other}", cfg.port),
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
            "port {} is not ours to restart ({other}) — verification needs our router",
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
