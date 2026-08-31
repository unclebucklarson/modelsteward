//! The desktop shell, organized the way a user thinks about it:
//!
//! * **Library** — every model from every source, one row each, with
//!   hardware-aware advice, Load/Unload, and an "In OpenCode" checkbox.
//!   Loading a model measures it and adds it to OpenCode automatically.
//! * **Server** — the router's controls plus detail on what's loaded now.
//! * **OpenCode** — what opencode.json actually contains, with sync state
//!   and failures explained.
//! * **Settings** — scan paths, ports, binary override.
//!
//! Every slow operation runs on a worker thread reporting over a channel;
//! the UI thread never blocks on the network or a model load.

use crate::core::{
    advisor, aiadvisor, bench, cancel, diagnose, discover, evidence, history, managed, meter,
    hermes, ollama, opencode, piagent, router, rows, settings, system, trial,
};
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

pub fn run() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1150.0, 720.0])
            .with_title("modelsteward"),
        ..Default::default()
    };
    eframe::run_native(
        "modelsteward",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

enum Msg {
    Scanned(system::ScanReport),
    RouterState(router::RouterState),
    Ollama(ollama::OllamaStatus),
    Measurements(router::Measurements),
    Configured(Vec<opencode::ConfiguredModel>),
    /// Live (free, total) MiB for the primary CUDA card.
    Vram(Option<(u64, u64)>),
    /// NVIDIA persistence mode at startup / after an enable attempt.
    Persistence(Option<bool>),
    BuildCheck(Box<advisor::BuildCheck>),
    PresetWritten(PathBuf, usize),
    SyncDone(opencode::SyncReport),
    Progress(String),
    /// Updates the busy label mid-operation (e.g. "campaign 3/7") —
    /// usability review G11: long runs had no progress indicator.
    Phase(String),
    /// Terminates the busy state with a final line.
    Finished(String),
    Error(String),
    /// trials.json changed on disk.
    Trials(trial::Trials),
    /// A measured trial finished; opens the verdict dialog.
    /// Managed-checkout status refreshed by a worker.
    Managed(managed::ManagedStatus),
    /// An AI advisory finished: (subject, answering model, text).
    Advisory {
        subject: String,
        model: String,
        text: String,
    },
    TrialDone {
        model: String,
        menu: String,
        report: trial::TrialReport,
    },
    /// config.json was rewritten by a background action (e.g. trial keep).
    CfgReloaded(settings::AppConfig),
    /// The daily upstream freshness probe finished.
    Upstream(advisor::UpstreamStatus),
    /// history.jsonl changed on disk.
    History(Vec<history::Entry>),
    /// Prompt-cache effectiveness mined from router.log.
    CacheStats(Vec<evidence::ModelCacheStats>),
    /// Today's meter summary line (None = nothing metered yet today).
    Meter(Option<String>),
    /// Live router activity for the status bar (None = quiet).
    Activity(Option<String>),
}

/// Sub-tabs inside Connections. Three connectors stacked vertically put
/// Hermes below the fold entirely (user report 2026-08-30) — one at a
/// time, each with the full pane height, and it scales to agent four.
#[derive(PartialEq, Clone, Copy)]
enum ConnTab {
    AnyApp,
    OpenCode,
    Pi,
    Hermes,
}

#[derive(PartialEq, Clone, Copy)]
enum Pane {
    Library,
    Server,
    /// The performance lab: pick a model, pick campaigns, run, read
    /// verdicts + history (user-directed split from Library, 2026-08-25).
    Lab,
    Connections,
    Settings,
}

/// A deferred row action, collected during rendering and executed after
/// (so table closures never need `&mut self`).
#[derive(Clone)]
enum RowAction {
    Load(String),
    Unload(String),
    AddToOpenCode(String),
    RemoveFromOpenCode(String),
    /// Pull a cache/Ollama-owned file into the user's shelf (by path).
    Archive(PathBuf),
    /// Open the per-model override editor.
    EditOverrides(String),
}

struct App {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    pane: Pane,

    cfg: settings::AppConfig,
    scan: Option<system::ScanReport>,
    router_state: Option<router::RouterState>,
    ollama: ollama::OllamaStatus,
    measurements: router::Measurements,
    configured: Vec<opencode::ConfiguredModel>,
    rows: Vec<rows::Row>,
    ram_mib: u64,
    live_vram: Option<(u64, u64)>,

    // Settings pane edit buffers (applied on Save).
    edit_scan_dirs: String,
    edit_port: String,
    edit_server_bin: String,
    edit_ollama_port: String,
    edit_models_max: String,

    activity: Vec<String>,
    busy: Option<String>,
    show_about: bool,
    last_sync: Option<String>,
    override_editor: Option<OverrideEditor>,
    /// AI advisory outputs, newest first: (subject, answering model, text).
    /// Session-only — opinions aren't measurements and aren't persisted.
    advisories: Vec<(String, String, String)>,
    advisor_open: bool,
    tuning_question: String,
    settings_needs_apply: bool,
    show_meter: bool,
    conn_tab: ConnTab,
    confirm_disrupt: Option<(String, Disrupt)>,
    show_persistence_prompt: bool,
    /// Mirrors self.busy for the poller thread (auto-build must never
    /// compile beside a measurement/trial/bench).
    busy_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    meter_report: String,
    meter_report_range: Option<&'static str>,
    library_filter: String,
    library_selected: Option<String>,
    show_guide: bool,
    /// Cached managed-checkout status (fs + git probes are too heavy to
    /// run per frame); refreshed on build-check and after managed workers.
    managed_status: Option<managed::ManagedStatus>,
    /// Which checkout the Build Advisor analyzes; None = the active
    /// binary's (rung 1 of the checkout ladder, user decision 2026-08-28).
    sel_checkout: Option<PathBuf>,
    archive_label: String,
    managed_auto_edit: bool,
    show_advisor: bool,
    build_check: Option<advisor::BuildCheck>,
    backend_sel: Option<advisor::BackendSelection>,
    diagnosis: Option<DiagnosisView>,
    trials: trial::Trials,
    upstream: Option<advisor::UpstreamStatus>,
    history: Vec<history::Entry>,
    cache_stats: Vec<evidence::ModelCacheStats>,
    meter_line: Option<String>,
    live_activity: Option<String>,
    last_error: Option<String>,
    settings_error: Option<String>,
    /// Token for the CURRENT long-running operation; Cancel flips it and
    /// workers stop at their next safe boundary.
    cancel_token: cancel::CancelToken,
    start_prompt: Option<AfterStart>,
    lab_selected: Option<String>,
    lab_measure: bool,
    lab_bench: bool,
    lab_spec: bool,
    lab_ub: bool,
    lab_kv: bool,
    lab_quality: bool,
    lab_load: bool,
    lab_dials: bool,
    lab_moe: bool,
    lab_vision: bool,
    lab_cache: bool,
    lab_ckpt: bool,
    lab_slots: bool,
    /// Which menu's Why? explanation is expanded in the Lab, if any.
    lab_why: Option<String>,
}

/// A router-needing action intercepted while the router was down: offered
/// as "Start Router & Continue" instead of a dead-end error (user request
/// 2026-08-25). Never offered for an external server — not ours to start.
#[derive(Clone)]
enum AfterStart {
    Calibrate { force: bool },
    /// A Lab campaign: the selected model + which runs were checked.
    Lab {
        id: String,
        measure: bool,
        bench: bool,
        spec: bool,
        ub: bool,
        kv: bool,
        quality: bool,
        load: bool,
        dials: bool,
        moe: bool,
        vision: bool,
        cache: bool,
        ckpt: bool,
        slots: bool,
    },
}

/// A deferred action waiting behind the interrupt-confirmation dialog
/// (user request 2026-08-30, after nearly disturbing a live overnight
/// coding session): anything that would reload/unload/evict while the
/// router is actively serving asks first.
#[derive(Clone)]
enum Disrupt {
    Row(RowAction),
    Dispatch(AfterStart, bool),
    Stop,
    ApplySettings,
}

impl AfterStart {
    fn describe(&self) -> String {
        match self {
            AfterStart::Calibrate { force: true } => "re-measure ALL models".into(),
            AfterStart::Calibrate { force: false } => "measure new/stale models".into(),
            AfterStart::Lab { id, .. } => format!("run the Lab campaigns for {id}"),
        }
    }
}

/// A diagnosis being shown for one model row.
struct DiagnosisView {
    display: String,
    router_id: Option<String>,
    path: Option<PathBuf>,
    d: diagnose::Diagnosis,
}

/// Edit buffers for one model's preset overrides. Fields show EFFECTIVE
/// values (override if set, else the optimized default); on save, anything
/// equal to optimized is stored as "no override" so it keeps auto-adapting.
/// A knob promoted out of the free-form "extra flags" box into its own
/// field (M8 #9): proven trial targets show their measured optimum;
/// sampling defaults are config surface only (never a trial target —
/// agents override them per request).
struct PromotedField {
    key: &'static str,
    label: &'static str,
    text: String,
    hint: String,
    /// What the server uses when nothing is pinned — prefilled so the
    /// dialog shows the value the model will ACTUALLY get (its stated
    /// promise); a value left equal to this is stored as "no override".
    /// Read from llama.cpp's common.h sampling defaults (b10630).
    default_text: &'static str,
}

struct OverrideEditor {
    id: String,
    /// Validation failure shown IN the dialog (usability review G6: it
    /// used to land only in the activity log behind the window).
    error: Option<String>,
    ctx_text: String,
    kv_text: String,
    extra_text: String,
    promoted: Vec<PromotedField>,
    /// The optimized context baseline: what --fit measured on this machine
    /// (None = not measured yet -> auto).
    optimized_ctx: Option<u64>,
    /// This model has a vision projector available on disk.
    has_mmproj: bool,
    /// User chose to serve WITHOUT it (restores cache-reuse).
    no_mmproj: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = channel();
        let mut app = Self {
            tx,
            rx,
            pane: Pane::Library,
            cfg: {
                let cfg = system::load_config();
                // Interrupted-trial self-heal (usability review C4).
                if let Some(note) = trial::heal_interrupted_trial(&cfg) {
                    eprintln!("{note}");
                }
                cfg
            },
            scan: None,
            router_state: None,
            ollama: Default::default(),
            measurements: router::read_measurements(&router::state_dir()),
            trials: trial::read_trials(&router::state_dir()),
            upstream: advisor::read_upstream_status(&router::state_dir()),
            history: history::read_all(&router::state_dir()),
            cache_stats: Vec::new(),
            meter_line: meter::summary_line(
                &router::state_dir(),
                advisor::now_epoch(),
                None,
            ),
            live_activity: None,
            last_error: None,
            settings_error: None,
            cancel_token: cancel::CancelToken::default(),
            start_prompt: None,
            lab_selected: None,
            lab_measure: true,
            lab_bench: true,
            lab_spec: true,
            lab_ub: false,
            lab_kv: false,
            lab_quality: false,
            lab_load: false,
            lab_dials: false,
            lab_moe: false,
            lab_vision: false,
            lab_cache: false,
            lab_ckpt: false,
            lab_slots: false,
            lab_why: None,
            configured: Vec::new(),
            rows: Vec::new(),
            ram_mib: rows::read_ram_mib(),
            live_vram: None,
            edit_scan_dirs: String::new(),
            edit_port: String::new(),
            edit_server_bin: String::new(),
            edit_ollama_port: String::new(),
            edit_models_max: String::new(),
            activity: Vec::new(),
            busy: None,
            show_about: false,
            last_sync: None,
            override_editor: None,
            advisories: Vec::new(),
            advisor_open: false,
            tuning_question: String::new(),
            settings_needs_apply: false,
            show_meter: false,
            conn_tab: ConnTab::OpenCode,
            confirm_disrupt: None,
            show_persistence_prompt: false,
            busy_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            meter_report: String::new(),
            meter_report_range: None,
            library_filter: String::new(),
            library_selected: None,
            show_guide: false,
            managed_status: None,
            sel_checkout: None,
            archive_label: String::new(),
            managed_auto_edit: false,
            show_advisor: false,
            build_check: None,
            backend_sel: None,
            diagnosis: None,
        };
        app.reset_edit_buffers();
        app.spawn_scan();
        app.spawn_persistence_probe();
        app.spawn_status_poller(cc.egui_ctx.clone());
        app.spawn_config_read();
        app
    }

    fn reset_edit_buffers(&mut self) {
        self.edit_scan_dirs = self
            .cfg
            .scan_dirs
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        self.edit_port = self.cfg.port.to_string();
        self.edit_server_bin = self
            .cfg
            .server_bin
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        self.edit_ollama_port = self.cfg.ollama_port.to_string();
        self.edit_models_max = self.cfg.models_max.to_string();
        self.managed_auto_edit = self.cfg.managed_auto_build;
    }

    fn log(&mut self, line: impl Into<String>) {
        // HH:MM prefix (usability review G15): after a long session,
        // undated lines can't be told stale from fresh.
        let t = advisor::now_epoch();
        self.activity
            .push(format!("{:02}:{:02} {}", (t / 3600) % 24, (t / 60) % 60, line.into()));
        if self.activity.len() > 200 {
            self.activity.drain(..100);
        }
    }

    /// The server the app would launch, derived from the CACHED scan —
    /// zero subprocesses. system::pick_server probes every install with
    /// --version and --list-devices (CUDA init!); calling it per frame
    /// made the whole desktop sluggish once archives multiplied the
    /// candidate list (live casualty 2026-08-28, user relaunched 3x).
    fn picked_server(&self) -> Option<PathBuf> {
        if let Some(explicit) = &self.cfg.server_bin {
            return Some(explicit.clone());
        }
        let scan = self.scan.as_ref()?;
        scan.devices_from
            .clone()
            .filter(|p| !system::is_managed_install(p))
            .or_else(|| {
                let mut by_build: Vec<_> = scan
                    .installs
                    .iter()
                    .filter(|i| !system::is_managed_install(&i.server_path))
                    .collect();
                by_build.sort_by_key(|i| std::cmp::Reverse(i.build.unwrap_or(0)));
                by_build.first().map(|i| i.server_path.clone())
            })
            .or_else(|| {
                // Managed-only machine: newest managed, same order the
                // prober uses — never discovery-order-first (review catch).
                let mut managed: Vec<_> = scan
                    .installs
                    .iter()
                    .filter(|i| system::is_managed_install(&i.server_path))
                    .collect();
                managed.sort_by_key(|i| std::cmp::Reverse(i.build.unwrap_or(0)));
                managed.first().map(|i| i.server_path.clone())
            })
    }

    fn hardware(&self) -> rows::Hardware {
        // Physical, deduped, dedicated-only: the old first-CUDA pick was
        // right on this machine but a Vulkan-only box would have taken the
        // iGPU's phantom shared-RAM heap as VRAM.
        let vram_mib = self
            .scan
            .as_ref()
            .map(|s| discover::advice_vram_mib(&s.devices))
            .unwrap_or(0);
        rows::Hardware {
            vram_mib,
            ram_mib: self.ram_mib,
        }
    }

    fn rebuild_rows(&mut self) {
        let router_models = match &self.router_state {
            Some(router::RouterState::Ours { models }) => models.clone(),
            _ => Vec::new(),
        };
        let opencode_ids: Vec<String> = self.configured.iter().map(|c| c.id.clone()).collect();
        let models = self.scan.as_ref().map(|s| s.models.clone()).unwrap_or_default();
        self.rows = rows::assemble(
            &models,
            &router_models,
            &self.measurements,
            &opencode_ids,
            self.hardware(),
        );
    }

    // ─── workers ─────────────────────────────────────────────────────────

    fn spawn_scan(&self) {
        let tx = self.tx.clone();
        let cfg = self.cfg.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Msg::Scanned(system::scan_report(&cfg, &[])));
        });
    }

    /// One cheap nvidia-smi query at startup (no CUDA init) so the
    /// one-time persistence prompt can fire without waiting for a
    /// Build Advisor check.
    fn spawn_persistence_probe(&self) {
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let state = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=persistence_mode", "--format=csv,noheader"])
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                .and_then(|s| advisor::parse_persistence_mode(&s));
            let _ = tx.send(Msg::Persistence(state));
        });
    }

    fn spawn_config_read(&self) {
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let configured =
                opencode::configured_models(&opencode::default_config_path()).unwrap_or_default();
            let _ = tx.send(Msg::Configured(configured));
        });
    }

    /// Status poll loop: every 2s for the life of the app. Reads the config
    /// file each round so a saved port change takes effect live.
    fn spawn_status_poller(&self, ctx: egui::Context) {
        let tx = self.tx.clone();
        let app_busy = self.busy_flag.clone();
        std::thread::spawn(move || {
            let meas_path = router::state_dir().join("measurements.json");
            let trials_path = router::state_dir().join("trials.json");
            let history_path = router::state_dir().join("history.jsonl");
            let mut last_meas_mtime = None;
            let mut last_trials_mtime = None;
            let mut last_history_mtime = None;
            let log_path = router::state_dir().join("router.log");
            let mut last_log_len: u64 = 0;
            let mut last_activity_len: u64 = 0;
            let mut quiet_ticks: u32 = 0;
            let mut last_mine = std::time::Instant::now() - std::time::Duration::from_secs(60);
            let cfg_path = system::config_file();
            let mut last_cfg_mtime = None;
            let mut rcfg: Option<router::RouterConfig> = None;
            // A release the auto-build wants but couldn't start because the
            // machine was busy — retried each tick until quiet.
            let mut pending_autobuild: Option<u64> = None;
            // When the router last showed generation activity. Starts at
            // \"now\": a fresh app assumes recent activity until proven quiet.
            let mut last_activity_epoch: u64 = advisor::now_epoch();
            loop {
                let cfg = system::load_config();
                // router_config -> pick_server probes every install with
                // --version/--list-devices; doing that every 2s scaled
                // with the archive count and dragged the whole machine.
                // The pick only shifts when config or installs change —
                // recompute on config mtime change (and daily, below).
                let cfg_mtime =
                    std::fs::metadata(&cfg_path).and_then(|m| m.modified()).ok();
                if rcfg.is_none() || cfg_mtime != last_cfg_mtime {
                    last_cfg_mtime = cfg_mtime;
                    rcfg = Some(system::router_config(&cfg));
                }
                let state = router::status(
                    &router::state_dir(),
                    rcfg.as_ref().expect("set above"),
                );
                if tx.send(Msg::RouterState(state.clone())).is_err() {
                    return;
                }
                let _ = tx.send(Msg::Ollama(ollama::probe(cfg.ollama_port)));
                let _ = tx.send(Msg::Vram(discover::nvidia_vram_mib()));
                // Reload measurements when the file changes on disk (e.g. a
                // CLI calibration ran) — the OpenCode/Library tabs must
                // never show stale numbers.
                let mtime = std::fs::metadata(&meas_path).and_then(|m| m.modified()).ok();
                if mtime != last_meas_mtime {
                    last_meas_mtime = mtime;
                    let _ = tx.send(Msg::Measurements(router::read_measurements(
                        &router::state_dir(),
                    )));
                }
                let mtime = std::fs::metadata(&trials_path).and_then(|m| m.modified()).ok();
                if mtime != last_trials_mtime {
                    last_trials_mtime = mtime;
                    let _ = tx.send(Msg::Trials(trial::read_trials(&router::state_dir())));
                }
                let mtime = std::fs::metadata(&history_path).and_then(|m| m.modified()).ok();
                if mtime != last_history_mtime {
                    last_history_mtime = mtime;
                    let _ = tx.send(Msg::History(history::read_all(&router::state_dir())));
                }
                // Prompt-cache effectiveness: re-mine when the log grew,
                // throttled to twice a minute (the log can be large).
                let log_len = std::fs::metadata(&log_path).map(|m| m.len()).unwrap_or(0);
                // Live activity: when the log grew this tick, classify
                // its tail (8KB read, pure parse); when it goes quiet for
                // two ticks, clear. Answers "is it even working?" at a
                // glance instead of a log dive (two live cases in two
                // days).
                if log_len != last_activity_len {
                    last_activity_len = log_len;
                    quiet_ticks = 0;
                    let hint = std::fs::File::open(&log_path).ok().and_then(|mut f| {
                        use std::io::{Read, Seek, SeekFrom};
                        let start = log_len.saturating_sub(8192);
                        f.seek(SeekFrom::Start(start)).ok()?;
                        let mut tail = String::new();
                        f.read_to_string(&mut tail).ok()?;
                        evidence::activity_hint(&tail)
                    });
                    if hint.is_some() {
                        // Any classified activity — including a turn
                        // FINISHING — counts as \"recently active\".
                        last_activity_epoch = advisor::now_epoch();
                    }
                    let text = hint.and_then(|h| match h {
                        evidence::Activity::TurnStarted => Some("working…".to_string()),
                        evidence::Activity::Prefilling(pct) => {
                            Some(format!("prefilling {pct:.0}%"))
                        }
                        // A completed turn means quiet again — no claim.
                        evidence::Activity::TurnFinished { .. } => None,
                    });
                    let _ = tx.send(Msg::Activity(text));
                } else {
                    quiet_ticks += 1;
                    if quiet_ticks == 3 {
                        // ~6s of silence: whatever we claimed is stale.
                        let _ = tx.send(Msg::Activity(None));
                    }
                }
                if log_len != last_log_len && last_mine.elapsed().as_secs() >= 30 {
                    last_log_len = log_len;
                    last_mine = std::time::Instant::now();
                    if let Ok(text) = std::fs::read_to_string(&log_path) {
                        // One parse serves both the monitor and the meter
                        // (the log was parsed twice per tick — review
                        // catch), and the ledger + summary only rewrite
                        // when something was actually credited.
                        let stats = evidence::cache_effectiveness(&text);
                        let now = advisor::now_epoch();
                        let credited = meter::harvest_stats(
                            &router::state_dir(),
                            &stats,
                            &text,
                            now,
                        )
                        .unwrap_or(0);
                        let _ = tx.send(Msg::CacheStats(stats));
                        if credited > 0 {
                            let trials = trial::read_trials(&router::state_dir());
                            let j: std::collections::BTreeMap<String, f64> = trials
                                .iter()
                                .filter_map(|(m, t)| {
                                    trial::served_j_per_token(&cfg, m, t)
                                        .map(|j| (m.clone(), j))
                                })
                                .collect();
                            let _ = tx.send(Msg::Meter(meter::summary_line(
                                &router::state_dir(),
                                now,
                                Some((&j, cfg.kwh_price_usd)),
                            )));
                        }
                    }
                }
                // Daily upstream freshness (user request 2026-08-25): one
                // quiet fetch per day, remote-tracking refs only. The
                // persisted stamp keeps restarts from refetching early.
                let due = advisor::read_upstream_status(&router::state_dir())
                    .map(|s| {
                        advisor::now_epoch().saturating_sub(s.checked_epoch)
                            >= advisor::UPSTREAM_CHECK_INTERVAL_SECS
                    })
                    .unwrap_or(true);
                if due {
                    rcfg = Some(system::router_config(&cfg)); // daily re-pick
                }
                if due && let Some(server) =
                    rcfg.as_ref().map(|r| r.server_bin.clone()).filter(|p| p.is_file())
                {
                    // Reuse the pick the re-pick just made — a second
                    // pick_server here re-ran the whole probe sweep
                    // back-to-back (review catch 2026-08-28).
                    let build = discover::build_of(&server);
                    let s = advisor::upstream_probe(&server, build);
                    advisor::write_upstream_status(&router::state_dir(), &s);
                    // Managed autonomy (opt-in): a NEW release the archive
                    // doesn't have yet gets built + archived here in the
                    // poller thread — CPU-only, never touches serving.
                    if cfg.managed_auto_build
                        && managed::checkout_present()
                        && let Some(up) = s.upstream_build
                        && !managed::list_archives().iter().any(|a| a.build == Some(up))
                    {
                        pending_autobuild = Some(up);
                        let _ = tx.send(Msg::Progress(format!(
                            "[auto-build] b{up} queued — will build when the \
                             machine is quiet"
                        )));
                    }
                    let _ = tx.send(Msg::Upstream(s));
                }
                // Auto-build waits for a QUIET machine: a full-core cmake
                // beside a measurement skews the numbers being recorded.
                // Review-hardened (2026-08-28): rechecks the opt-in every
                // tick (unchecking cancels the queue), clears the queue
                // whenever the release stops being wanted (built manually,
                // setting off) so no per-tick churn survives, treats
                // loading/downloading as busy, reuses THIS tick's status
                // (zero extra probes), and takes the binary from the
                // cached rcfg — never pick_server in a poll tick.
                if let Some(up) = pending_autobuild {
                    if !cfg.managed_auto_build
                        || managed::list_archives().iter().any(|a| a.build == Some(up))
                    {
                        pending_autobuild = None;
                    } else {
                        // Design session 2026-08-30 (Scott's pick):
                        // \"quiet for 10 min\". A LOADED model no longer
                        // blocks — the daily-driver machine is never
                        // model-free, and the old gate starved auto-build
                        // for days (b10697 sat queued behind a 24h
                        // Minecraft session). Still deferred: any app
                        // operation (measure/trial/bench via busy_flag),
                        // a trial from ANY process (marker), models
                        // mid-load/download, and recent generation.
                        let quiet_secs =
                            advisor::now_epoch().saturating_sub(last_activity_epoch);
                        let transitioning = |m: &router::RouterModel| {
                            matches!(m.status.as_str(), "loading" | "downloading" | "downloaded")
                        };
                        let idle = !app_busy.load(std::sync::atomic::Ordering::Relaxed)
                            && !trial::trial_marker_present(&router::state_dir())
                            && quiet_secs >= 600
                            && match &state {
                                router::RouterState::Down => true,
                                router::RouterState::Ours { models } => {
                                    !models.iter().any(transitioning)
                                }
                                // External/Trouble: someone else's server —
                                // be polite, and say why nothing happens.
                                _ => false,
                            };
                        if idle
                            && let Some(server) = rcfg
                                .as_ref()
                                .map(|r| r.server_bin.clone())
                                .filter(|p| p.is_file())
                        {
                            let keep = cfg.archives_keep;
                            let serving = server.clone();
                            pending_autobuild = None;
                            let tx2 = tx.clone();
                            std::thread::spawn(move || {
                                let tx3 = tx2.clone();
                                let mut progress = move |line: String| {
                                    let _ = tx3
                                        .send(Msg::Progress(format!("[auto-build] {line}")));
                                };
                                let build = discover::build_of(&server);
                                let measurements =
                                    router::read_measurements(&router::state_dir());
                                let check = advisor::check(
                                    Some(server),
                                    build,
                                    &measurements,
                                    None,
                                );
                                let sel = advisor::default_backends(&check);
                                let result =
                                    managed::build_release(&check, sel, &mut progress);
                                if result.is_ok() {
                                    for l in managed::prune_archives(
                                        keep as usize,
                                        Some(&serving),
                                    ) {
                                        let _ = tx2.send(Msg::Progress(format!(
                                            "[auto-build] pruned old archive {l} \
                                             (retention: keep {keep})"
                                        )));
                                    }
                                }
                                let _ = tx2.send(Msg::Managed(managed::status()));
                                let _ = tx2.send(Msg::Progress(match result {
                                    Ok(b) => format!(
                                        "[auto-build] b{b} built + archived — select it \
                                         in Settings -> llama-server binary when you \
                                         want to serve it"
                                    ),
                                    Err(e) => format!("[auto-build] failed: {e:#}"),
                                }));
                            });
                        }
                    }
                }
                ctx.request_repaint();
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        });
    }

    fn spawn(&mut self, label: &str, job: impl FnOnce(&Sender<Msg>) + Send + 'static) {
        if let Some(b) = &self.busy {
            let b = b.clone();
            self.log(format!("busy with {b}; try again when it finishes"));
            return;
        }
        self.busy = Some(label.to_string());
        self.busy_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.log(format!("{label}…"));
        let tx = self.tx.clone();
        std::thread::spawn(move || job(&tx));
    }

    fn action_bench(&mut self, force: bool) {
        let cfg = self.cfg.clone();
        let label = if force {
            "re-benching ALL models (speed)"
        } else {
            "benching new/stale models (speed)"
        };
        let cancel_token = {
            self.cancel_token = cancel::CancelToken::default();
            self.cancel_token.clone()
        };
        self.spawn(label, move |tx| {
            let tx2 = tx.clone();
            let mut progress = move |line: String| {
                let _ = tx2.send(Msg::Progress(line));
            };
            let _ = tx.send(match bench::run_baselines(
                &cfg,
                None,
                force,
                &cancel_token,
                &mut progress,
            ) {
                Ok((0, 0)) => {
                    Msg::Finished("bench: nothing to do — all baselines current".into())
                }
                Ok((n, 0)) => Msg::Finished(format!(
                    "benched {n} model(s) — Speed column updated (pp/tg tokens per second)"
                )),
                Ok((n, f)) => Msg::Finished(format!(
                    "benched {n} model(s), {f} failed — the log lines above say why"
                )),
                Err(e) => Msg::Error(format!("bench: {e:#}")),
            });
        });
    }

    fn action_regen_preset(&mut self) {
        let cfg = self.cfg.clone();
        self.spawn("regenerating preset", move |tx| {
            match system::write_preset(&cfg, &[]) {
                Ok((path, n)) => {
                    let _ = tx.send(Msg::PresetWritten(path, n));
                    if let Ok(models) = router::reload(cfg.port) {
                        let _ = tx.send(Msg::Progress(format!(
                            "router reloaded: {} models listed",
                            models.len()
                        )));
                    }
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("preset: {e:#}")));
                }
            }
        });
    }

    fn action_start(&mut self) {
        let cfg = self.cfg.clone();
        self.spawn("starting router", move |tx| {
            let _ = tx.send(match start_router(&cfg) {
                Ok(pid) => Msg::Finished(format!("router started (pid {pid})")),
                Err(e) => Msg::Error(format!("start: {e:#}")),
            });
        });
    }

    fn action_stop(&mut self) {
        if let Some(msg) = self.serving_disruption("Stopping the router") {
            self.confirm_disrupt = Some((msg, Disrupt::Stop));
            return;
        }
        self.action_stop_now();
    }

    fn action_stop_now(&mut self) {
        self.spawn("stopping router", |tx| {
            let _ = tx.send(
                match router::stop(&router::state_dir(), &system::preset_path()) {
                    Ok(()) => Msg::Finished("router stopped".into()),
                    Err(e) => Msg::Error(format!("stop: {e:#}")),
                },
            );
        });
    }

    fn action_calibrate(&mut self, force: bool) {
        self.run_or_offer_start(AfterStart::Calibrate { force });
    }

    /// Run a router-needing action now, or — when our router is down /
    /// troubled — open the "Start Router & Continue?" prompt instead of a
    /// dead-end error. External servers are untouched: those actions keep
    /// their plain refusal.
    /// Persist a trial choice (winner, near-miss, or "baseline" to revert)
    /// through every config layer: override -> preset -> router reload ->
    /// measurement -> OpenCode limit. Keeps need no router prompt —
    /// keep_variant tolerates a down router (reload is best-effort).
    fn spawn_keep(&mut self, model: &str, menu: &str, label: &str) {
        let cfg = self.cfg.clone();
        let m = model.to_string();
        let menu = menu.to_string();
        let w = label.to_string();
        self.spawn(&format!("applying {w} to {m}"), move |tx| {
            let variants = trial::menu(&menu)
                .map(|(v, _)| v)
                .unwrap_or_else(trial::spec_decode_variants);
            let _ = match trial::keep_variant(&system::config_file(), &cfg, &m, &variants, &w) {
                Ok(()) => {
                    let _ = tx.send(Msg::CfgReloaded(system::load_config()));
                    let _ = tx.send(Msg::Measurements(router::read_measurements(
                        &router::state_dir(),
                    )));
                    match sync_single(&cfg, &router::read_measurements(&router::state_dir()), &m)
                    {
                        Ok(()) => {
                            send_configured(tx);
                            tx.send(Msg::Finished(format!(
                                "{m}: {w} applied — override saved, router reloaded, \
                                 OpenCode limit updated from the trial's measurement"
                            )))
                        }
                        Err(e) => tx.send(Msg::Finished(format!(
                            "{m}: {w} applied (OpenCode sync deferred: {e:#})"
                        ))),
                    }
                }
                Err(e) => tx.send(Msg::Error(format!("apply: {e:#}"))),
            };
        });
    }

    fn run_or_offer_start(&mut self, action: AfterStart) {
        let startable = matches!(
            self.router_state,
            Some(router::RouterState::Down) | Some(router::RouterState::Trouble { .. })
        );
        if startable {
            self.start_prompt = Some(action);
        } else {
            self.dispatch(action, false);
        }
    }

    fn dispatch(&mut self, action: AfterStart, start_first: bool) {
        // start_first means the router was down — nothing to disturb.
        if !start_first
            && let Some(msg) = self.serving_disruption(&format!(
                "Starting \"{}\" (campaigns unload models and swap configs to \
                 measure honestly)",
                action.describe()
            ))
        {
            self.confirm_disrupt = Some((msg, Disrupt::Dispatch(action, start_first)));
            return;
        }
        self.dispatch_now(action, start_first);
    }

    fn dispatch_now(&mut self, action: AfterStart, start_first: bool) {
        let cfg = self.cfg.clone();
        self.cancel_token = cancel::CancelToken::default();
        let cancel_token = self.cancel_token.clone();
        let label = if start_first {
            format!("starting router, then: {}", action.describe())
        } else {
            action.describe()
        };
        self.spawn(&label, move |tx| {
            if start_first && !start_router_and_wait(&cfg, tx) {
                return;
            }
            match action {
                AfterStart::Calibrate { force } => calibrate_worker(&cfg, force, tx),
                AfterStart::Lab {
                    id, measure, bench, spec, ub, kv, quality, load, dials, moe,
                    vision, cache, ckpt, slots,
                } => {
                    // Campaigns run in sequence on one worker: measure first
                    // (it's what puts a model into OpenCode at all), bench
                    // frees the GPU itself, each trial restores the preset.
                    let cancelled = || cancel_token.is_cancelled();
                    let total = [
                        measure, bench, spec, ub, kv, load, dials, moe, vision,
                        cache, ckpt, slots, quality,
                    ]
                    .iter()
                    .filter(|b| **b)
                    .count();
                    let mut step = 0usize;
                    let txp = tx.clone();
                    let idp = id.clone();
                    let mut phase = move |name: &str| {
                        step += 1;
                        let _ = txp.send(Msg::Phase(format!(
                            "Lab {idp}: {name} ({step}/{total})"
                        )));
                    };
                    if measure {
                        phase("measure");
                        if !measure_and_sync(&cfg, &id, false, false, tx) {
                            return; // reported; don't pile campaigns on a broken load
                        }
                    }
                    if bench {
                        phase("bench");
                        let tx2 = tx.clone();
                        let mut progress = move |line: String| {
                            let _ = tx2.send(Msg::Progress(line));
                        };
                        match bench::run_baselines(
                            &cfg,
                            Some(id.clone()),
                            true,
                            &cancel_token,
                            &mut progress,
                        ) {
                            Ok(_) => {}
                            Err(e) => {
                                let _ = tx.send(Msg::Error(format!("bench: {e:#}")));
                                return;
                            }
                        }
                    }
                    for (on, menu) in [
                        (spec, "spec"), (ub, "ub"), (kv, "kv"), (load, "load"),
                        (dials, "dials"), (moe, "moe"), (vision, "vision"),
                        (cache, "cache"), (ckpt, "ckpt"),
                    ] {
                        if on {
                            phase(&format!("{menu} trial"));
                            trial_worker(&cfg, &id, menu, &cancel_token, tx);
                        }
                    }
                    if slots {
                        phase("slots trial");
                        let tx2 = tx.clone();
                        let mut progress = move |line: String| {
                            let _ = tx2.send(Msg::Progress(line));
                        };
                        if let Err(e) =
                            trial::run_slot_trial(&cfg, &id, &cancel_token, &mut progress)
                        {
                            let _ = tx.send(Msg::Error(format!("slot trial: {e:#}")));
                        }
                    }
                    if quality {
                        phase("quality probe");
                        let tx2 = tx.clone();
                        let mut progress = move |line: String| {
                            let _ = tx2.send(Msg::Progress(line));
                        };
                        if let Err(e) = crate::core::quality::run_and_record(
                            &cfg,
                            &id,
                            5,
                            &cancel_token,
                            &mut progress,
                        ) {
                            let _ = tx.send(Msg::Error(format!("quality: {e:#}")));
                            return;
                        }
                        let _ = tx.send(Msg::Measurements(router::read_measurements(
                            &router::state_dir(),
                        )));
                    }
                    let _ = tx.send(Msg::Finished(if cancelled() {
                        format!("Lab: campaigns for {id} stopped by Cancel — partial results kept")
                    } else {
                        format!("Lab: campaigns for {id} complete")
                    }));
                }
            }
        });
    }

    fn action_sync(&mut self) {
        let cfg = self.cfg.clone();
        let measurements = self.measurements.clone();
        self.spawn("syncing opencode.json", move |tx| {
            match run_sync(&cfg, &measurements) {
                Ok((report, lines)) => {
                    let _ = tx.send(Msg::SyncDone(report));
                    for l in lines {
                        let _ = tx.send(Msg::Progress(l));
                    }
                    send_configured(tx);
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("sync: {e:#}")));
                }
            }
        });
    }

    /// The one-click flow: preset if missing -> start if down -> wait ->
    /// incremental calibrate -> sync. Each step narrates.
    fn action_setup(&mut self) {
        let cfg = self.cfg.clone();
        self.spawn("setting up everything", move |tx| {
            setup_flow(&cfg, tx);
        });
    }

    /// Some(message) when interrupting live serving is a real risk:
    /// our router, with a model loaded — strengthened when the activity
    /// monitor sees a request in flight right now.
    fn serving_disruption(&self, what: &str) -> Option<String> {
        let Some(router::RouterState::Ours { models }) = &self.router_state else {
            return None;
        };
        let loaded: Vec<&str> = models
            .iter()
            .filter(|m| matches!(m.status.as_str(), "loaded" | "loading"))
            .map(|m| m.id.as_str())
            .collect();
        if loaded.is_empty() {
            return None;
        }
        let activity = match &self.live_activity {
            Some(a) => format!(", mid-request right now ({a})"),
            None => String::new(),
        };
        Some(format!(
            "{} is live on the router{activity}.\n\n{what} will interrupt it — \
             an in-flight coding turn from OpenCode (or anything else connected) \
             would fail or stall.",
            loaded.join(", ")
        ))
    }

    /// Which row actions actually disturb serving — and only when they
    /// do: a Load into a FREE slot evicts nothing and passes silently.
    fn disruption_for_row(&self, action: &RowAction) -> Option<String> {
        let loaded: Vec<String> = match &self.router_state {
            Some(router::RouterState::Ours { models }) => models
                .iter()
                .filter(|m| matches!(m.status.as_str(), "loaded" | "loading"))
                .map(|m| m.id.clone())
                .collect(),
            _ => return None,
        };
        let evicting_load = |id: &String| {
            !loaded.contains(id) && loaded.len() >= self.cfg.models_max as usize
        };
        match action {
            RowAction::Unload(id) => {
                self.serving_disruption(&format!("Unloading {id}"))
            }
            RowAction::Load(id) if evicting_load(id) => self.serving_disruption(&format!(
                "Loading {id} (the router will evict a loaded model to make room)"
            )),
            RowAction::AddToOpenCode(id)
                if evicting_load(id)
                    && self
                        .measurements
                        .get(id)
                        .and_then(|m| m.n_ctx)
                        .is_none() =>
            {
                self.serving_disruption(&format!(
                    "Adding {id} to OpenCode (it is unmeasured, so it loads once \
                     to measure — evicting a loaded model)"
                ))
            }
            _ => None,
        }
    }

    fn run_row_action(&mut self, action: RowAction) {
        if let Some(msg) = self.disruption_for_row(&action) {
            self.confirm_disrupt = Some((msg, Disrupt::Row(action)));
            return;
        }
        self.run_row_action_now(action);
    }

    fn run_row_action_now(&mut self, action: RowAction) {
        let cfg = self.cfg.clone();
        match action {
            RowAction::Load(id) => {
                // Load = measure = make available to OpenCode, and keep it
                // warm for immediate use.
                self.spawn(&format!("loading + measuring {id}"), move |tx| {
                    measure_and_sync(&cfg, &id, true, true, tx);
                });
            }
            RowAction::Unload(id) => {
                self.spawn(&format!("unloading {id}"), move |tx| {
                    let _ = tx.send(match router::unload_model(cfg.port, &id) {
                        Ok(()) => Msg::Finished(format!(
                            "{id}: unloaded (still in OpenCode — the router reloads it on demand)"
                        )),
                        Err(e) => Msg::Error(format!("{e:#}")),
                    });
                });
            }
            RowAction::AddToOpenCode(id) => {
                let measured = self
                    .measurements
                    .get(&id)
                    .and_then(|m| m.n_ctx)
                    .is_some();
                if measured {
                    let measurements = self.measurements.clone();
                    self.spawn(&format!("adding {id} to OpenCode"), move |tx| {
                        let _ = match sync_single(&cfg, &measurements, &id) {
                            Ok(()) => {
                                send_configured(tx);
                                tx.send(Msg::Finished(format!("{id}: added to OpenCode")))
                            }
                            Err(e) => tx.send(Msg::Error(format!("{e:#}"))),
                        };
                    });
                } else {
                    // Not measured yet: load briefly, measure, add, unload.
                    self.spawn(&format!("measuring {id} for OpenCode"), move |tx| {
                        measure_and_sync(&cfg, &id, false, true, tx);
                    });
                }
            }
            RowAction::RemoveFromOpenCode(id) => {
                self.spawn(&format!("removing {id} from OpenCode"), move |tx| {
                    let path = opencode::default_config_path();
                    let _ = match opencode::comment_out_in_file(&path, &id) {
                        Ok(()) => {
                            send_configured(tx);
                            tx.send(Msg::Finished(format!(
                                "{id}: commented out of opencode.json (backup kept)"
                            )))
                        }
                        Err(e) => tx.send(Msg::Error(format!("remove: {e:#}"))),
                    };
                });
            }
            RowAction::EditOverrides(id) => {
                let ov = self.cfg.overrides.get(&id).cloned().unwrap_or_default();
                let optimized_ctx = self.measurements.get(&id).and_then(|m| m.n_ctx);
                // Vision toggle only shows when a projector exists to skip;
                // r.vision reflects a paired mmproj (or an already-active
                // no_mmproj override, in which case the row shows no badge
                // but the projector is still on disk — check both).
                let has_mmproj = self
                    .rows
                    .iter()
                    .find(|r| r.router_id.as_deref() == Some(id.as_str()))
                    .map(|r| r.vision)
                    .unwrap_or(false)
                    || ov.no_mmproj;
                // Promoted knobs: what THIS model's trials measured, shown
                // beside the field (empty = not pinned, keeps the default).
                let measured = |menu: &str, untrialed: &str| -> String {
                    match self.trials.get(&id).and_then(|t| trial::stored_report(menu, t)) {
                        Some(r) => match &r.verdict.winner {
                            Some(w) => format!("measured best: {w}"),
                            None => "measured: baseline wins here".into(),
                        },
                        None => untrialed.into(),
                    }
                };
                let sampling_hint = "server default; used only when a client \
                     sends no sampling of its own (agents always do). Model cards \
                     sometimes recommend family-specific values — set them here.";
                let mut promoted = vec![
                    PromotedField {
                        key: "spec-type",
                        label: "Speculation",
                        text: String::new(),
                        hint: measured("spec", "not trialed yet — ⚡ Lab races the ngram modes"),
                        default_text: "",
                    },
                    PromotedField {
                        key: "ubatch-size",
                        label: "Prefill batch (-ub)",
                        text: String::new(),
                        hint: measured("ub", "default 512 — ⚡ Lab races 1024/2048"),
                        default_text: router::DEFAULT_UBATCH,
                    },
                    PromotedField {
                        key: "temp",
                        label: "temp",
                        text: String::new(),
                        hint: sampling_hint.into(),
                        default_text: router::DEFAULT_TEMP,
                    },
                    PromotedField {
                        key: "top-k",
                        label: "top-k",
                        text: String::new(),
                        hint: sampling_hint.into(),
                        default_text: router::DEFAULT_TOP_K,
                    },
                    PromotedField {
                        key: "top-p",
                        label: "top-p",
                        text: String::new(),
                        hint: sampling_hint.into(),
                        default_text: router::DEFAULT_TOP_P,
                    },
                ];
                for f in &mut promoted {
                    f.text = f.default_text.to_string();
                }
                // Show effective values: the override where set, else the
                // optimized default — the dialog always tells the truth
                // about what the model will get. Promoted keys leave the
                // free-form box and land in their own fields.
                let mut extra_rest: Vec<String> = Vec::new();
                for (k, v) in &ov.extra {
                    if let Some(f) = promoted.iter_mut().find(|f| f.key == k.as_str()) {
                        f.text = v.clone();
                    } else {
                        extra_rest.push(format!("{k} = {v}"));
                    }
                }
                self.override_editor = Some(OverrideEditor {
                    id,
                    error: None,
                    ctx_text: ov
                        .ctx
                        .or(optimized_ctx)
                        .map(|c| c.to_string())
                        .unwrap_or_default(),
                    kv_text: ov
                        .cache_type_kv
                        .clone()
                        .unwrap_or_else(|| router::DEFAULT_KV_TYPE.to_string()),
                    has_mmproj,
                    no_mmproj: ov.no_mmproj,
                    extra_text: extra_rest.join("\n"),
                    promoted,
                    optimized_ctx,
                });
            }
            RowAction::Archive(path) => {
                let Some(model) = self
                    .scan
                    .as_ref()
                    .and_then(|s| s.models.iter().find(|m| m.path == path).cloned())
                else {
                    self.log("archive: model not found in scan — rescan first");
                    return;
                };
                let shelf_root = cfg
                    .scan_dirs
                    .first()
                    .cloned()
                    .unwrap_or_else(|| {
                        std::env::home_dir()
                            .unwrap_or_else(|| PathBuf::from("."))
                            .join("models")
                    });
                // The pre-archive identity, for measurement migration.
                let old_id = self
                    .scan
                    .as_ref()
                    .map(|s| rows::router_ids_by_path(&s.models))
                    .and_then(|ids| ids.get(&path).cloned());
                self.spawn(
                    &format!("archiving {} to your shelf", model.display_name()),
                    move |tx| {
                        match crate::core::library::archive_to_shelf(&model, &shelf_root) {
                            Ok(dest) => {
                                let _ = tx.send(Msg::Progress(format!(
                                    "archived -> {} (hardlinked when possible; yours now)",
                                    dest.display()
                                )));
                                // Make it servable: regenerate preset, tell
                                // the router, rescan so the row updates.
                                match system::write_preset(&cfg, &[]) {
                                    Ok((_, n)) => {
                                        let _ = tx.send(Msg::Progress(format!(
                                            "preset regenerated ({n} models)"
                                        )));
                                        if let Ok(models) = router::reload(cfg.port) {
                                            let _ = tx.send(Msg::Progress(format!(
                                                "router reloaded: {} models offered",
                                                models.len()
                                            )));
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx
                                            .send(Msg::Error(format!("archive/preset: {e:#}")));
                                        return;
                                    }
                                }
                                let report = system::scan_report(&cfg, &[]);
                                // The measurement follows the file to its new
                                // alias (fingerprints cleared -> re-measures
                                // next calibrate); the old cache-id row stops
                                // claiming it.
                                if let Some(old) = &old_id
                                    && let Some(new) =
                                        rows::router_ids_by_path(&report.models).get(&dest)
                                {
                                    let dir = router::state_dir();
                                    let mut all = router::read_measurements(&dir);
                                    router::migrate_measurement(&mut all, old, new);
                                    if router::write_measurements(&dir, &all).is_ok() {
                                        let _ = tx.send(Msg::Progress(format!(
                                            "measurement carried over: {old} -> {new}"
                                        )));
                                        let _ = tx.send(Msg::Measurements(all));
                                    }
                                }
                                let _ = tx.send(Msg::Scanned(report));
                                let _ = tx.send(Msg::Finished(
                                    "archive complete — the model is now a shelf model; Load it to measure + add to OpenCode".into(),
                                ));
                            }
                            Err(e) => {
                                let _ = tx.send(Msg::Error(format!("archive: {e:#}")));
                            }
                        }
                    },
                );
            }
        }
    }

    fn action_build_check(&mut self) {
        self.show_advisor = true;
        self.build_check = None;
        let cfg = self.cfg.clone();
        let measurements = self.measurements.clone();
        let sel = self.sel_checkout.clone();
        self.spawn(
            "checking your llama.cpp build (contacts the git remote)",
            move |tx| {
                // A selected checkout is analyzed via its own built binary
                // (repo_of walks back to it); unbuilt checkouts still get
                // their repo pinned so the guided rebuild can run.
                let server = match &sel {
                    Some(dir) => {
                        let bin = dir.join("build/bin/llama-server");
                        bin.is_file().then_some(bin)
                    }
                    None => system::pick_server(&cfg).ok(),
                };
                let build = server.as_deref().and_then(discover::build_of);
                let log =
                    std::fs::read_to_string(router::state_dir().join("router.log")).ok();
                let mut check = advisor::check(server, build, &measurements, log.as_deref());
                if check.repo.is_none() {
                    check.repo = sel;
                }
                let _ = tx.send(Msg::BuildCheck(Box::new(check)));
                let _ = tx.send(Msg::Managed(managed::status()));
            },
        );
    }

    fn action_rebuild(&mut self) {
        let Some(check) = self.build_check.clone() else {
            return;
        };
        let sel = self
            .backend_sel
            .unwrap_or_else(|| advisor::default_backends(&check));
        let cfg = self.cfg.clone();
        self.spawn(
            "updating + rebuilding llama.cpp, then verifying (this takes many minutes)",
            move |tx| {
                let progress_tx = tx.clone();
                let result = advisor::run_rebuild(&check, sel, &mut |line| {
                    let _ = progress_tx.send(Msg::Progress(line));
                });
                match result {
                    Ok(()) => {
                        // M6 phase 2: the verification loop. A running
                        // router keeps serving the OLD binary (children
                        // exec whatever is on disk — a subtle mix worth
                        // ending), so restart, then measure + sync, then
                        // report what the rebuild actually changed.
                        let _ = tx.send(Msg::Progress(
                            "rebuild complete ✔ — verifying what it changed…".into(),
                        ));
                        let dir = router::state_dir();
                        let before = router::read_measurements(&dir);
                        if matches!(
                            router::status(&dir, &system::router_config(&cfg)),
                            router::RouterState::Ours { .. }
                        ) {
                            let _ = tx.send(Msg::Progress(
                                "restarting router onto the new binary".into(),
                            ));
                            let _ = router::stop(&dir, &system::preset_path());
                        }
                        if setup_flow(&cfg, tx) {
                            let after = router::read_measurements(&dir);
                            for line in advisor::verify_summary(&advisor::verify_outcome(
                                &before, &after,
                            )) {
                                let _ = tx.send(Msg::Progress(format!("verification: {line}")));
                            }
                            let _ = tx.send(Msg::Finished(
                                "rebuild verified — the verification lines above are the \
                                 measured outcome"
                                    .into(),
                            ));
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Msg::Error(format!(
                            "rebuild failed: {e:#} — your existing binaries are untouched"
                        )));
                    }
                }
            },
        );
    }

    fn vram_contention(&self) -> Option<String> {
        let router_loaded: Vec<&str> = match &self.router_state {
            Some(router::RouterState::Ours { models }) => models
                .iter()
                .filter(|m| m.status == "loaded")
                .map(|m| m.id.as_str())
                .collect(),
            _ => return None,
        };
        if router_loaded.is_empty() || self.ollama.loaded.is_empty() {
            return None;
        }
        let ollama_names: Vec<String> = self
            .ollama
            .loaded
            .iter()
            .map(|m| format!("{} ({:.1} GB VRAM)", m.name, m.size_vram as f64 / 1e9))
            .collect();
        Some(format!(
            "VRAM contention: router has {} loaded while Ollama holds {} — `ollama stop <model>` frees it immediately",
            router_loaded.join(", "),
            ollama_names.join(", ")
        ))
    }

    fn drain_messages(&mut self) {
        let mut rebuild = false;
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Scanned(report) => {
                    self.log(format!(
                        "scan: {} installs, {} devices, {} models",
                        report.installs.len(),
                        report.devices.len(),
                        report.models.len()
                    ));
                    self.scan = Some(report);
                    self.busy = None;
                    rebuild = true;
                }
                Msg::RouterState(s) => {
                    self.router_state = Some(s);
                    rebuild = true;
                }
                Msg::Ollama(o) => self.ollama = o,
                Msg::Vram(v) => self.live_vram = v,
                Msg::Persistence(state) => {
                    if state == Some(false) && !self.cfg.persistence_prompt_dismissed {
                        self.show_persistence_prompt = true;
                    }
                    if state == Some(true) && self.show_persistence_prompt {
                        self.show_persistence_prompt = false;
                        self.log("GPU persistence mode is ON — model loads skip the driver re-init from now on");
                    }
                }
                Msg::BuildCheck(c) => {
                    self.backend_sel = Some(advisor::default_backends(&c));
                    self.build_check = Some(*c);
                    self.busy = None;
                }
                Msg::Configured(c) => {
                    self.configured = c;
                    rebuild = true;
                }
                // NOTE: does not clear `busy` — the poller sends this on
                // every on-disk change, including mid-calibration persists.
                // Operations end with Finished/Error/SyncDone.
                Msg::Measurements(m) => {
                    self.measurements = m;
                    rebuild = true;
                }
                Msg::Trials(t) => self.trials = t,
                Msg::History(h) => self.history = h,
                Msg::CacheStats(c) => self.cache_stats = c,
                Msg::Meter(m) => self.meter_line = m,
                Msg::Activity(a) => self.live_activity = a,
                // NOTE: does not clear `busy` — a Lab campaign may still be
                // mid-sequence; the worker ends with Finished. No dialog:
                // verdicts live in the Lab's standing recommendations
                // (user decision 2026-08-26 — a popup mid-campaign offered
                // Apply while the worker still owned the GPU).
                Msg::Managed(m) => self.managed_status = Some(m),
                Msg::Advisory { subject, model, text } => {
                    self.log(format!("advisory ready for {subject} (by {model})"));
                    self.advisories.insert(0, (subject, model, text));
                    self.advisor_open = true;
                }
                Msg::TrialDone { model, menu, report } => {
                    let _ = menu;
                    self.log(format!("trial {model}: {}", report.verdict.reason));
                    for nm in &report.near_misses {
                        self.log(format!(
                            "trial {model}: rules rejected {} — {} for {} (decide in the Lab)",
                            nm.label, nm.gain, nm.cost
                        ));
                    }
                }
                Msg::CfgReloaded(c) => {
                    self.cfg = c;
                    rebuild = true;
                }
                Msg::Upstream(s) => {
                    if let (Some(cur), Some(up)) = (s.current_build, s.upstream_build)
                        && up > cur
                    {
                        self.log(format!(
                            "llama.cpp upstream has b{up} (you run b{cur}) — Server -> \
                             Check My llama.cpp for the guided update"
                        ));
                    }
                    self.upstream = Some(s);
                }
                Msg::PresetWritten(path, n) => {
                    self.log(format!("preset written: {} models -> {}", n, path.display()));
                    self.busy = None;
                }
                Msg::SyncDone(r) => {
                    if r.skipped_missing {
                        self.log(
                            "sync: opencode.json not found — OpenCode isn't installed; \
                             skipped (other apps connect via the Connections tab)"
                                .to_string(),
                        );
                    }
                    for id in &r.ghosts_commented {
                        self.log(format!(
                            "sync: ✂ {id} commented out (router omits it, nothing measured — \
                             a ghost; uncomment in opencode.json to restore)"
                        ));
                    }
                    let line = format!(
                        "sync: {} added, {} updated, {} not synced",
                        r.added.len(),
                        r.updated.len(),
                        r.orphans.len()
                    );
                    self.log(line.clone());
                    self.last_sync = Some(line);
                    self.busy = None;
                }
                Msg::Progress(p) => self.log(p),
                Msg::Phase(p) => {
                    self.log(p.clone());
                    if self.busy.is_some() {
                        self.busy = Some(p);
                    }
                }
                Msg::Finished(p) => {
                    self.log(p);
                    self.busy = None;
                }
                Msg::Error(e) => {
                    self.log(format!("ERROR: {e}"));
                    self.last_error = Some(e);
                    self.busy = None;
                }
            }
        }
        if rebuild {
            self.rebuild_rows();
        }
    }

    // ─── menu ────────────────────────────────────────────────────────────

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("File", |ui| {
            if ui
                .button("Set Up Everything")
                .on_hover_text("Start router if needed, measure anything unmeasured, sync OpenCode")
                .clicked()
            {
                self.action_setup();
                ui.close();
            }
            ui.separator();
            if ui.button("Rescan System").clicked() {
                self.spawn_scan();
                self.spawn_config_read();
                ui.close();
            }
            if ui.button("Regenerate Preset").clicked() {
                self.action_regen_preset();
                ui.close();
            }
            if ui.button("Sync opencode.json").clicked() {
                self.action_sync();
                ui.close();
            }
            ui.separator();
            if ui.button("Quit").clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
        ui.menu_button("Server", |ui| {
            if ui.button("Start Router").clicked() {
                self.action_start();
                ui.close();
            }
            if ui.button("Stop Router").clicked() {
                self.action_stop();
                ui.close();
            }
            if ui.button("Reload Models").clicked() {
                let port = self.cfg.port;
                self.spawn("reloading models", move |tx| {
                    let _ = tx.send(match router::reload(port) {
                        Ok(m) => Msg::Finished(format!("reloaded: {} models", m.len())),
                        Err(e) => Msg::Error(format!("reload: {e:#}")),
                    });
                });
                ui.close();
            }
            ui.separator();
            if ui
                .button("Measure New/Stale Models (context + tool calls)")
                .on_hover_text(
                    "The FLEET sweep: refreshes the preset (new files become servable), \
                     then for every model that's new or whose config/build changed — \
                     loads it, reads the context --fit actually settled on, probes for a \
                     well-formed tool call, records both, and syncs OpenCode. Fresh \
                     models are skipped. For ONE model, use the Lab's Measure campaign \
                     — same operation, chosen scope.",
                )
                .clicked()
            {
                self.action_calibrate(false);
                ui.close();
            }
            if ui
                .button("Re-measure ALL (force)")
                .on_hover_text(
                    "The same fleet sweep, ignoring freshness — every model re-measures \
                     even if nothing changed. Takes minutes; useful when you distrust \
                     the recorded numbers.",
                )
                .clicked()
            {
                self.action_calibrate(true);
                ui.close();
            }
            ui.separator();
            if ui
                .button("Bench New/Stale Models (speed)")
                .on_hover_text(
                    "The FLEET sweep for speed: llama-bench baselines (prompt-processing + \
                     generation tokens/sec) for every measured model missing a current one. \
                     For ONE model, use the Lab's Bench campaign — same operation, chosen \
                     scope. Unloads the router's models \
                     first; a 27B takes about a minute each. Fills the Speed column.",
                )
                .clicked()
            {
                self.action_bench(false);
                ui.close();
            }
            if ui
                .button("Re-bench ALL (force)")
                .on_hover_text(
                    "Re-measure every model's speed baseline even if current — e.g. after \
                     a driver update or to double-check a number.",
                )
                .clicked()
            {
                self.action_bench(true);
                ui.close();
            }
            ui.separator();
            if ui.button("Check My llama.cpp (Build Advisor)…").clicked() {
                self.action_build_check();
                ui.close();
            }
            ui.separator();
            if ui.button("Install systemd User Unit…").clicked() {
                let cfg = self.cfg.clone();
                self.spawn("writing systemd unit", move |tx| {
                    let _ = tx.send(match system::install_systemd_unit(&cfg) {
                        Ok(path) => Msg::Finished(format!(
                            "unit written: {} — activate with: systemctl --user daemon-reload && systemctl --user enable --now llamacpp-router",
                            path.display()
                        )),
                        Err(e) => Msg::Error(format!("systemd unit: {e:#}")),
                    });
                });
                ui.close();
            }
        });
        ui.menu_button("Tools", |ui| {
            let mut open_path: Option<PathBuf> = None;
            if ui.button("Open Router Preset (router.ini)").clicked() {
                open_path = Some(system::preset_path());
                ui.close();
            }
            if ui.button("Open App Config (config.json)").clicked() {
                open_path = Some(system::config_file());
                ui.close();
            }
            if ui.button("Open opencode.json").clicked() {
                open_path = Some(opencode::default_config_path());
                ui.close();
            }
            if ui.button("Advisor — opinions + Ask about tuning").clicked() {
                self.advisor_open = true;
                ui.close();
            }
            if ui.button("Token Meter — usage & measured cost").clicked() {
                self.refresh_meter_report();
                self.show_meter = true;
                ui.close();
            }
            if ui
                .add_enabled(
                    self.busy.is_none(),
                    egui::Button::new("Fleet Brief (AI opinion)"),
                )
                .on_disabled_hover_text("busy — wait for the running operation to finish")
                .on_hover_text(
                    "Feeds this machine's findings report (the sanitized JSON the \
                     app already generates) to one of YOUR served models and asks: \
                     best daily driver, the one change worth making, and what to \
                     measure next. Labeled opinion; nothing leaves this machine.",
                )
                .clicked()
            {
                self.action_fleet_brief();
                ui.close();
            }
            if ui.button("Open Router Log").clicked() {
                open_path = Some(router::state_dir().join("router.log"));
                ui.close();
            }
            ui.separator();
            if ui
                .button("Export Findings Report…")
                .on_hover_text(
                    "Writes a sanitized markdown summary of this machine's measured \
                     results (hardware, build, measurements, trial verdicts, \
                     build-over-build history — no paths or usernames) for YOU to \
                     review and share where maintainers look. Nothing is sent \
                     anywhere by the app.",
                )
                .clicked()
            {
                let cfg = self.cfg.clone();
                self.spawn("exporting findings report", move |tx| {
                    let _ = tx.send(match crate::core::report::generate(&cfg) {
                        Ok(path) => {
                            // Open only after the file exists — review is
                            // the whole point.
                            let _ = std::process::Command::new("xdg-open").arg(&path).spawn();
                            Msg::Finished(format!(
                                "findings report written and opened: {} — review it, then \
                                 share it wherever you choose",
                                path.display()
                            ))
                        }
                        Err(e) => Msg::Error(format!("report: {e:#}")),
                    });
                });
                ui.close();
            }
            if let Some(path) = open_path {
                match std::process::Command::new("xdg-open").arg(&path).spawn() {
                    Ok(_) => self.log(format!("opened {}", path.display())),
                    Err(e) => self.log(format!("ERROR opening {}: {e}", path.display())),
                }
            }
            ui.separator();
            if ui
                .button("Restore opencode.json From Last Backup")
                .on_hover_text(
                    "Swaps opencode.json with its newest backup — run again to toggle back. \
                     Undo for the last sync/remove.",
                )
                .clicked()
            {
                self.spawn("restoring opencode.json from backup", |tx| {
                    let _ = tx.send(
                        match opencode::restore_last_backup(&opencode::default_config_path()) {
                            Ok(msg) => {
                                send_configured(tx);
                                Msg::Finished(msg)
                            }
                            Err(e) => Msg::Error(format!("restore: {e:#}")),
                        },
                    );
                });
                ui.close();
            }
        });
        ui.menu_button("Help", |ui| {
            if ui.button("Tuning Guide — the sequence").clicked() {
                self.show_guide = true;
                ui.close();
            }
            if ui.button("About").clicked() {
                self.show_about = true;
                ui.close();
            }
        });
    }

    // ─── Library ─────────────────────────────────────────────────────────

    fn library_pane(&mut self, ui: &mut egui::Ui) {
        if self.scan.is_none() {
            ui.spinner();
            ui.label("scanning…");
            return;
        }
        // First-run empty states (usability review D5/G14): a bare
        // header strip over nothing explained nothing.
        if self.rows.is_empty() {
            ui.add_space(12.0);
            ui.strong("No models found yet.");
            ui.label(format!(
                "Looked in: {} — plus any Ollama blob store and the HuggingFace cache.",
                self.cfg
                    .scan_dirs
                    .iter()
                    .map(|d| d.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            ui.label(
                "Add a directory with GGUF files in Settings -> Model scan \
                 directories, or pull a model with Ollama — it will appear here \
                 automatically.",
            );
            return;
        }
        if !self.measurements.values().any(|m| m.n_ctx.is_some()) {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "Nothing measured yet — File -> Set Up Everything starts the router, \
                 measures every model, and wires up OpenCode in one go.",
            );
            ui.add_space(4.0);
        }
        let router_up = matches!(self.router_state, Some(router::RouterState::Ours { .. }));
        if !router_up {
            ui.horizontal(|ui| {
                ui.label("Router is not running — Load and checkbox actions need it.");
                if ui.button("⏵ Start Router").clicked() {
                    self.action_start();
                }
            });
            ui.separator();
        }
        let mut pending: Option<RowAction> = None;
        let mut why: Option<DiagnosisView> = None;
        let rows = self.rows.clone();
        let is_disabled = |r: &rows::Row| {
            rows::ignore_key(r.path.as_deref(), r.router_id.as_deref())
                .is_some_and(|k| self.cfg.disabled.contains(&k))
        };
        let mut toggle_ignore: Option<(String, bool)> = None;
        // History trails for the ctx and Speed hovers: the journal's recent
        // entries per model, one line each, newest first.
        let age = |when: u64| -> String {
            let s = advisor::now_epoch().saturating_sub(when);
            match s {
                0..=3599 => format!("{}m ago", (s / 60).max(1)),
                3600..=172_799 => format!("{}h ago", s / 3600),
                _ => format!("{}d ago", s / 86_400),
            }
        };
        let mut ctx_trails: std::collections::HashMap<String, String> = Default::default();
        let mut speed_trails: std::collections::HashMap<String, String> = Default::default();
        {
            let mut per_model: std::collections::HashMap<&str, (Vec<String>, Vec<String>)> =
                Default::default();
            for e in self.history.iter().rev() {
                let (ctx_lines, speed_lines) = per_model.entry(&e.model).or_default();
                let b = e.build.map(|b| format!("b{b}")).unwrap_or_else(|| "b?".into());
                if (e.n_ctx.is_some() || e.error.is_some()) && ctx_lines.len() < 6 {
                    ctx_lines.push(match (&e.n_ctx, &e.error) {
                        (Some(c), _) => format!("{b} · ctx {c} · {}", age(e.when)),
                        (None, Some(_)) => format!("{b} · failed · {}", age(e.when)),
                        _ => continue,
                    });
                }
                if e.pp_tps.is_some() && speed_lines.len() < 6 {
                    speed_lines.push(format!(
                        "{b} · pp {:.0} / tg {:.0} · {}",
                        e.pp_tps.unwrap_or(0.0),
                        e.tg_tps.unwrap_or(0.0),
                        age(e.when)
                    ));
                }
            }
            for (model, (c, s)) in per_model {
                if !c.is_empty() {
                    ctx_trails.insert(model.to_string(), format!("History:\n{}", c.join("\n")));
                }
                if !s.is_empty() {
                    speed_trails
                        .insert(model.to_string(), format!("History:\n{}", s.join("\n")));
                }
            }
        }
        // Master-detail (usability review G5/G3, user decision
        // 2026-08-29): a slim identity grid with search + sort, and a
        // detail panel carrying the advice, actions, quality, and
        // history for the selected model — the value proposition never
        // scrolls off-screen again.
        ui.horizontal(|ui| {
            ui.label("Filter:");
            ui.add(
                egui::TextEdit::singleline(&mut self.library_filter)
                    .desired_width(220.0)
                    .hint_text("name contains…"),
            );
            if !self.library_filter.is_empty() && ui.small_button("✖").clicked() {
                self.library_filter.clear();
            }
        });
        let needle = self.library_filter.to_lowercase();
        let mut view: Vec<&rows::Row> = rows
            .iter()
            .filter(|r| needle.is_empty() || r.display.to_lowercase().contains(&needle))
            .collect();
        view.sort_by_key(|r| {
            let rank = match r.advice_level {
                rows::AdviceLevel::Good => 0u8,
                rows::AdviceLevel::Warn => 1,
                rows::AdviceLevel::Unknown => 2,
                rows::AdviceLevel::Bad => 3,
            } + if is_disabled(r) { 10 } else { 0 };
            (rank, r.display.to_lowercase())
        });
        // Resizable split (user request 2026-08-29): drag the border
        // between grid and detail, same as the activity log panel.
        egui::Panel::top("library_grid")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("library")
                .striped(true)
                .min_col_width(48.0)
                .show(ui, |ui| {
                    for h in [
                        "Model", "Source", "Size", "Quant", "Measured ctx", "Speed",
                        "Server", "OC",
                    ] {
                        ui.strong(h).on_hover_text(match h {
                            "Measured ctx" => "The context window --fit actually settled on when this model was measured on THIS machine (hover a value for its history).",
                            "Speed" => "Measured baseline: prompt-processing / generation tokens per second (hover for history).",
                            "OC" => "✔ = in opencode.json. Toggle in the detail panel below.",
                            _ => "Click a model to see its advice, actions, quality, and history below.",
                        });
                    }
                    ui.end_row();
                    for r in &view {
                        let disabled = is_disabled(r);
                        let selected = self.library_selected.as_deref() == Some(r.display.as_str());
                        let label = if disabled {
                            egui::RichText::new(&r.display).weak()
                        } else {
                            match r.advice_level {
                                rows::AdviceLevel::Bad => egui::RichText::new(&r.display)
                                    .color(ui.visuals().error_fg_color),
                                _ => egui::RichText::new(&r.display),
                            }
                        };
                        if ui.selectable_label(selected, label).clicked() {
                            self.library_selected = Some(r.display.clone());
                        }
                        ui.label(&r.source);
                        ui.label(if r.size_bytes > 0 {
                            format!("{:.1} GB", r.size_bytes as f64 / 1e9)
                        } else {
                            "—".into()
                        });
                        ui.label(&r.quant);
                        {
                            let resp = ui.label(
                                r.measured_ctx
                                    .map(|c| c.to_string())
                                    .unwrap_or_else(|| "—".into()),
                            );
                            if let Some(t) =
                                r.router_id.as_ref().and_then(|id| ctx_trails.get(id))
                            {
                                resp.on_hover_text(t);
                            }
                        }
                        {
                            let trail = r.router_id.as_ref().and_then(|id| speed_trails.get(id));
                            let fmt = |v: Option<f64>| {
                                v.map(|t| format!("{t:.0}")).unwrap_or_else(|| "?".into())
                            };
                            let text = match (r.pp_tps, r.tg_tps) {
                                (None, None) => "—".to_string(),
                                (pp, tg) => format!("{}/{}", fmt(pp), fmt(tg)),
                            };
                            let resp = ui.label(text);
                            if let Some(t) = trail {
                                resp.on_hover_text(t);
                            }
                        }
                        ui.label(r.server_status.as_deref().unwrap_or("—"));
                        ui.label(if r.in_opencode { "✔" } else { "—" });
                        ui.end_row();
                    }
                });
        });
            });
        // ── the detail panel for the selected model ──
        let sel = self
            .library_selected
            .clone()
            .and_then(|d| rows.iter().find(|r| r.display == d).cloned());
        let Some(r) = sel else {
            ui.weak("Select a model above for its advice, actions, quality scores, and history.");
            if let Some(action) = pending {
                self.run_row_action(action);
            }
            return;
        };
        let disabled = is_disabled(&r);
        egui::ScrollArea::vertical().id_salt("library_detail").show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.strong(&r.display);
            let mut badges: Vec<&str> = Vec::new();
            if r.vision { badges.push("👁 vision"); }
            if r.mtp { badges.push("⚡ MTP"); }
            if r.embedding { badges.push("⚛ embedding"); }
            if r.moe { badges.push("MoE"); }
            if !badges.is_empty() {
                ui.weak(badges.join(" · "));
            }
            if let Some(h) = &r.quant_header_disagrees {
                ui.weak(format!("(header says {h}; filename wins for dynamic quants)"));
            }
        });
        if let Some(m) = r.router_id.as_ref().and_then(|id| self.measurements.get(id)) {
            let mut parts = Vec::new();
            if let Some(e) = m.eval_score { parts.push(format!("evals {:.0}%", e * 100.0)); }
            if let Some(t) = m.tool_reliability { parts.push(format!("tools {:.0}%", t * 100.0)); }
            if let Some(l) = m.loop_reliability { parts.push(format!("agent loops {:.0}%", l * 100.0)); }
            if !parts.is_empty() {
                ui.label(format!("Quality: {}", parts.join(" · ")));
            }
        }
        let color = match r.advice_level {
            rows::AdviceLevel::Good => egui::Color32::from_rgb(0, 170, 0),
            rows::AdviceLevel::Warn => ui.visuals().warn_fg_color,
            rows::AdviceLevel::Bad => ui.visuals().error_fg_color,
            rows::AdviceLevel::Unknown => ui.visuals().weak_text_color(),
        };
        if disabled {
            ui.weak("disabled — not measured, benched, offered, or raced (Enable below resumes)");
        } else {
            ui.colored_label(color, &r.advice);
        }
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            // In-OpenCode toggle — the whole make-it-usable flow.
            let mut checked = r.in_opencode;
            let can_act = if r.in_opencode {
                r.router_id.is_some()
            } else {
                router_up && r.router_id.is_some() && r.server_status.is_some()
            };
            let cb = ui
                .add_enabled(can_act, egui::Checkbox::new(&mut checked, "In OpenCode"))
                .on_disabled_hover_text(
                    "needs the router up and this model servable — start the router \
                     / check the Server column",
                )
                .on_hover_text(
                    "Checked = in opencode.json. Checking an unmeasured model loads \
                     it briefly to measure the real context first.",
                );
            if cb.changed()
                && let Some(id) = &r.router_id
            {
                pending = Some(if checked {
                    RowAction::AddToOpenCode(id.clone())
                } else {
                    RowAction::RemoveFromOpenCode(id.clone())
                });
            }
            match (&r.router_id, r.server_status.as_deref()) {
                (Some(id), Some("loaded")) => {
                    if ui.button("Unload").clicked() {
                        pending = Some(RowAction::Unload(id.clone()));
                    }
                }
                (Some(id), Some("unloaded")) => {
                    if ui
                        .add_enabled(router_up, egui::Button::new("Load"))
                        .on_disabled_hover_text("router is down — Start Router first (top of the tab)")
                        .on_hover_text(
                            "Loads now, measures the real context, and adds it to \
                             OpenCode automatically.",
                        )
                        .clicked()
                    {
                        pending = Some(RowAction::Load(id.clone()));
                    }
                }
                _ => {}
            }
            if let Some(id) = &r.router_id
                && ui
                    .button("⚙ Tune")
                    .on_hover_text(
                        "Per-model overrides: pin context, KV cache type, extra \
                         llama-server flags. Stored in the app config; survives \
                         preset regeneration.",
                    )
                    .clicked()
            {
                pending = Some(RowAction::EditOverrides(id.clone()));
            }
            if r.archivable
                && let Some(path) = &r.path
                && ui
                    .button("to shelf")
                    .on_hover_text(
                        "Copies (hardlinks when free) this file into your models \
                         directory — a normal shelf model, safe from cache eviction \
                         or `ollama rm`.",
                    )
                    .clicked()
            {
                pending = Some(RowAction::Archive(path.clone()));
            }
            if r.advice_level != rows::AdviceLevel::Good
                && ui
                    .button("Why?")
                    .on_hover_text("Plain-language explanation and what to do about it")
                    .clicked()
            {
                let not_offered = router_up
                    && r.router_id.is_some()
                    && r.server_status.is_none()
                    && r.failure.is_none();
                let mut failure = r.failure.clone();
                if let (Some(f), Some(id)) = (&failure, &r.router_id)
                    && diagnose::classify(f) == diagnose::Cause::Unknown
                    && let Ok(log) =
                        std::fs::read_to_string(router::state_dir().join("router.log"))
                    && let Some(mined) = advisor::mine_failures(&log).get(id)
                {
                    failure = Some(format!("{f} — {mined}"));
                }
                why = Some(DiagnosisView {
                    display: r.display.clone(),
                    router_id: r.router_id.clone(),
                    path: r.path.clone(),
                    d: diagnose::diagnose(
                        failure.as_deref(),
                        not_offered,
                        r.archivable && r.path.is_some(),
                        self.build_check.as_ref().map(|c| {
                            matches!(
                                (c.current_build, c.upstream_build),
                                (Some(cur), Some(up)) if cur >= up
                            )
                        }),
                        &format!(
                            "{} {}",
                            r.display,
                            r.path
                                .as_deref()
                                .map(|p| p.display().to_string())
                                .unwrap_or_default()
                        ),
                    ),
                });
            }
            if let Some(key) = rows::ignore_key(r.path.as_deref(), r.router_id.as_deref()) {
                let label = if disabled { "Enable" } else { "Disable" };
                if ui
                    .small_button(label)
                    .on_hover_text(if disabled {
                        "Resume measuring, benching, and offering this model."
                    } else {
                        "Keep the row visible but stop acting on it: no measure, no \
                         bench, no Lab, not offered by the router. Reversible."
                    })
                    .clicked()
                {
                    toggle_ignore = Some((key, !disabled));
                }
            }
        });
        // History, visible instead of hover-only (review G25).
        if let Some(id) = &r.router_id {
            let (c, sp) = (ctx_trails.get(id), speed_trails.get(id));
            if c.is_some() || sp.is_some() {
                ui.add_space(4.0);
                ui.collapsing("History", |ui| {
                    if let Some(t) = c {
                        ui.label(t);
                    }
                    if let Some(t) = sp {
                        ui.label(t);
                    }
                });
            }
        }
        });
        if let Some(v) = why {
            self.diagnosis = Some(v);
        }
        if let Some((key, disable)) = toggle_ignore {
            if disable {
                self.cfg.disabled.push(key);
            } else {
                self.cfg.disabled.retain(|k| k != &key);
            }
            match self.cfg.save(&system::config_file()) {
                Ok(()) => {
                    // The preset follows immediately — the router stops (or
                    // resumes) offering it on its next reload.
                    let cfg = self.cfg.clone();
                    self.spawn("updating preset (disabled list changed)", move |tx| {
                        let _ = tx.send(match system::write_preset(&cfg, &[]) {
                            Ok((_, n)) => {
                                let _ = router::reload(cfg.port);
                                Msg::Finished(format!("preset regenerated ({n} models)"))
                            }
                            Err(e) => Msg::Error(format!("preset: {e:#}")),
                        });
                    });
                }
                Err(e) => self.log(format!("ERROR saving disabled list: {e:#}")),
            }
        }
        if let Some(action) = pending {
            self.run_row_action(action);
        }
    }

    // ─── Server ──────────────────────────────────────────────────────────

    /// The performance lab: pick a model, pick campaigns, run; results,
    /// verdicts, and history live here (Library stays everything+advice).
    fn lab_pane(&mut self, ui: &mut egui::Ui) {
        ui.label(
            "Pick a model, pick campaigns, Run. Every number is measured on this \
             machine; winners are offered, never auto-applied.",
        );
        // User question 2026-08-29 ("where do the lab settings go?") —
        // answer it on the page, not just in a hover.
        ui.weak(
            "Where applied settings live: Apply stores a winner as this model's \
             override in the app's config.json, rewrites the router preset \
             (router.ini), and reloads the router — llama-server gets those flags \
             next time it loads this model. opencode.json never carries these \
             knobs; it only gets the measured results (context limit, tool support).",
        );
        ui.separator();
        let candidates: Vec<(String, String)> = self
            .rows
            .iter()
            .filter(|r| r.router_id.is_some() && !r.embedding && r.failure.is_none())
            .filter(|r| {
                !rows::ignore_key(r.path.as_deref(), r.router_id.as_deref())
                    .is_some_and(|k| self.cfg.disabled.contains(&k))
            })
            .map(|r| (r.router_id.clone().unwrap(), r.display.clone()))
            .collect();
        let mut run_clicked = false;
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(300.0);
                ui.strong("Model");
                egui::ScrollArea::vertical()
                    .id_salt("lab-models")
                    .show(ui, |ui| {
                        for (id, display) in &candidates {
                            let sel = self.lab_selected.as_deref() == Some(id.as_str());
                            if ui.selectable_label(sel, display).clicked() {
                                self.lab_selected = Some(id.clone());
                            }
                        }
                    });
            });
            ui.separator();
            ui.vertical(|ui| {
                let Some(id) = self.lab_selected.clone() else {
                    ui.weak("Select a model on the left.");
                    return;
                };
                // Everything below scrolls: campaigns grow, results tables
                // grow, and the user shouldn't have to resize the window
                // (user request 2026-08-27).
                egui::ScrollArea::vertical()
                    .id_salt("lab-detail")
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                ui.strong(&id);
                if let Some(m) = self.measurements.get(&id) {
                    let speed = match (m.pp_tps, m.tg_tps) {
                        (Some(pp), Some(tg)) => {
                            format!(", pp {pp:.0} / tg {tg:.0} t/s (b{})",
                                m.bench_build.map(|b| b.to_string()).unwrap_or_else(|| "?".into()))
                        }
                        _ => ", speed not benched yet".into(),
                    };
                    let qual = match (m.eval_score, m.tool_reliability) {
                        (Some(e), Some(t)) => format!(
                            ", evals {:.0}% / tools {:.0}%{}",
                            e * 100.0,
                            t * 100.0,
                            m.loop_reliability
                                .map(|l| format!(" / agent loops {:.0}%", l * 100.0))
                                .unwrap_or_default()
                        ),
                        _ => String::new(),
                    };
                    ui.label(format!(
                        "Currently: ctx {}{speed}{qual}",
                        m.n_ctx.map(|c| c.to_string()).unwrap_or_else(|| "unmeasured".into()),
                    ));
                }
                ui.add_space(6.0);
                // Which campaigns actually apply to THIS model — the same
                // detection the advice column uses, offered as a one-click
                // selection (user request: guidance, not guesswork).
                let sel_row = self.rows.iter().find(|r| {
                    r.router_id.as_deref() == Some(id.as_str())
                });
                // Header-derived (expert_count) — a name heuristic missed
                // GLM-4.5-Air live, whose filename carries no A-number.
                let is_moe_over_vram = sel_row.is_some_and(|r| {
                    r.moe && r.size_bytes / (1024 * 1024) > self.hardware().vram_mib
                });
                // The sequencing guard (two live lessons in one battery,
                // 2026-08-28): without an applied placement, this model
                // measures everything at a crushed default context.
                let placement_applied = self.cfg.overrides.get(&id).is_some_and(|o| {
                    o.extra
                        .iter()
                        .any(|(k, _)| k == "cpu-moe" || k == "n-cpu-moe")
                });
                if is_moe_over_vram && !placement_applied {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        "⚠ No MoE placement applied yet — bench, agent-turn trials \
                         (vision/cache/ckpt), and quality would all measure at the \
                         crushed default context. Run the MoE-offload trial FIRST and \
                         Apply its winner, then the rest (Help -> Tuning Guide).",
                    );
                }
                let has_spec_kept =
                    !trial::applied_keys(&self.cfg, &id, "spec").is_empty();
                let has_mmproj = sel_row.is_some_and(|r| r.vision);
                ui.horizontal(|ui| {
                    ui.strong("Campaigns");
                    if ui
                        .small_button("select relevant")
                        .on_hover_text(
                            "Checks the campaigns that apply to this model: MoE offload only for MoE models bigger than VRAM; speculation dials only once a speculation mode is kept; vision cost only for models with a projector; the core set for everything.",
                        )
                        .clicked()
                    {
                        self.lab_measure = true;
                        self.lab_bench = true;
                        self.lab_spec = true;
                        self.lab_ub = true;
                        self.lab_kv = true;
                        self.lab_quality = true;
                        self.lab_load = true;
                        self.lab_dials = has_spec_kept;
                        self.lab_moe = is_moe_over_vram;
                        self.lab_vision = has_mmproj;
                        self.lab_cache = true;
                        self.lab_ckpt = true;
                        self.lab_slots = false;
                    }
                    if ui.small_button("select all").clicked() {
                        self.lab_measure = true;
                        self.lab_bench = true;
                        self.lab_spec = true;
                        self.lab_ub = true;
                        self.lab_kv = true;
                        self.lab_quality = true;
                        self.lab_load = true;
                        self.lab_dials = true;
                        self.lab_moe = true;
                        self.lab_vision = true;
                        self.lab_cache = true;
                        self.lab_ckpt = true;
                        self.lab_slots = true;
                    }
                    if ui.small_button("select none").clicked() {
                        self.lab_measure = false;
                        self.lab_bench = false;
                        self.lab_spec = false;
                        self.lab_ub = false;
                        self.lab_kv = false;
                        self.lab_quality = false;
                        self.lab_load = false;
                        self.lab_dials = false;
                        self.lab_moe = false;
                        self.lab_vision = false;
                        self.lab_cache = false;
                        self.lab_ckpt = false;
                        self.lab_slots = false;
                    }
                });
                ui.checkbox(
                    &mut self.lab_measure,
                    "Measure (settled context + tool calls; adds to OpenCode, ~2 min)",
                );
                ui.checkbox(&mut self.lab_bench, "Bench baseline (pp/tg via llama-bench, ~1 min)");
                ui.checkbox(
                    &mut self.lab_spec,
                    "Speculation trial (ngram modes vs baseline, ~10 min)",
                );
                ui.checkbox(
                    &mut self.lab_ub,
                    "Prefill-batch trial (-ub 1024/2048 vs 512, ~8 min)",
                );
                ui.checkbox(
                    &mut self.lab_kv,
                    "KV-precision trial (ctv q4_0 — more context if quality holds, ~6 min)",
                );
                ui.checkbox(
                    &mut self.lab_quality,
                    "Quality probe (eval battery + tool calls + 3 multi-hop agent loops, ~5-12 min)",
                );
                ui.checkbox(
                    &mut self.lab_load,
                    "Load-mode trial (hot-swap speed: dio/mlock vs auto, ~5 min)",
                );
                ui.checkbox(
                    &mut self.lab_dials,
                    "Speculation-dial trial (ngram lookup/draft lengths, ~12 min)",
                )
                .on_hover_text(if has_spec_kept {
                    "Tunes the dials of the speculation mode this model has kept."
                } else {
                    "Applies AFTER a speculation mode is kept — run the Speculation trial first and Keep a winner, then tune its dials."
                });
                ui.checkbox(
                    &mut self.lab_moe,
                    "MoE-offload trial (--cpu-moe, partial --n-cpu-moe + thread \
                     counts — for MoE models bigger than VRAM, ~25 min)",
                )
                .on_hover_text(if is_moe_over_vram {
                    "THIS model's headline trial: bigger than your VRAM and MoE — experts in RAM can beat default placement dramatically (an 80B A3B ran at full 262k context on a 24GB card)."
                } else {
                    "Only applies to MoE models bigger than your VRAM — this model fits (or is dense), so --cpu-moe would only slow it down."
                });
                ui.checkbox(
                    &mut self.lab_vision,
                    "Vision-cost trial (serve text-only vs with projector — measures \
                     the agent-turn cache tax, ~8 min)",
                )
                .on_hover_text(if has_mmproj {
                    "This model serves a vision projector. Vision's measured costs: VRAM (a smaller fitted context) and, on models whose cache can shift, the loss of mid-edit cache-reuse — append-style turns stay cached either way. Whether text-only helps YOUR model depends on its attention; this trial answers with numbers instead of a guess."
                } else {
                    "Only applies to models serving a vision projector — this one has none, so there is nothing to turn off."
                });
                ui.checkbox(
                    &mut self.lab_cache,
                    "Cache-reuse trial (--cache-reuse 0/256/1024 — second-turn \
                     prefill after a mid-prompt edit, ~8 min)",
                )
                .on_hover_text(
                    "Coding agents edit the middle of the prompt every turn; \
                     cache-reuse decides how much of the KV cache survives that. \
                     Measured with a two-turn probe, not guessed.",
                );
                ui.checkbox(
                    &mut self.lab_ckpt,
                    "Checkpoint trial (--checkpoint-min-step — mid-edit resume \
                     points for models whose cache can't shift, ~10 min)",
                )
                .on_hover_text(
                    "SWA/hybrid-attention models silently DISABLE cache-reuse and \
                     ctx-shift; a mid-prompt edit resumes from the nearest context \
                     checkpoint instead, and the defaults space them 8192 tokens \
                     apart — almost nowhere for coding-agent prompts. Measured live: \
                     a 1954ms second turn dropped to 467ms with min-step 128.",
                );
                ui.checkbox(
                    &mut self.lab_slots,
                    "Slot-persistence ceiling (save/restore a session's KV cache \
                     across a swap — groundwork measurement, ~6 min)",
                )
                .on_hover_text(
                    "Swapping models mid-conversation currently costs a full \
                     reprocess when you swap back — which is why picking one \
                     middle-of-the-road model beats switching. This measures what \
                     a snapshot/restore workflow would buy on your hardware. \
                     Nothing to apply yet; it prices the future feature.",
                );
                // Model-aware time estimate (user request 2026-08-28: a
                // 3-hour GLM battery arrived unannounced — the static
                // labels assume 27B-class speed). Measured, not guessed:
                // scaled from this model's benched tg/pp.
                {
                    let m = self.measurements.get(&id);
                    let tg = m.and_then(|m| m.tg_tps);
                    let pp = m.and_then(|m| m.pp_tps);
                    let selected: Vec<&str> = [
                        (self.lab_spec, "spec"),
                        (self.lab_ub, "ub"),
                        (self.lab_kv, "kv"),
                        (self.lab_quality, "quality"),
                        (self.lab_load, "load"),
                        (self.lab_dials, "dials"),
                        (self.lab_moe, "moe"),
                        (self.lab_vision, "vision"),
                        (self.lab_cache, "cache"),
                        (self.lab_ckpt, "ckpt"),
                    ]
                    .iter()
                    .filter(|(on, _)| *on)
                    .map(|(_, n)| *n)
                    .collect();
                    let mins: Option<f64> = selected
                        .iter()
                        .map(|n| trial::campaign_eta_minutes(n, tg, pp))
                        .sum::<Option<f64>>();
                    match mins {
                        Some(total) if !selected.is_empty() => {
                            let extra = 3.0 * (self.lab_measure as u8
                                + self.lab_bench as u8
                                + self.lab_slots as u8)
                                as f64;
                            let total = total + extra;
                            let msg = if total >= 90.0 {
                                format!(
                                    "⚠ Estimated {:.0} min – {:.0} h at this model's \
                                     measured speed — slow models pay per token; \
                                     consider fewer campaigns per sitting",
                                    total,
                                    (total * 1.3 / 60.0).ceil()
                                )
                            } else {
                                format!(
                                    "Estimated ~{total:.0} min at this model's measured speed"
                                )
                            };
                            if total >= 90.0 {
                                ui.colored_label(ui.visuals().warn_fg_color, msg);
                            } else {
                                ui.small(msg);
                            }
                        }
                        None if !selected.is_empty() => {
                            ui.small(
                                "No speed baseline yet — run Measure + Bench first \
                                 for a time estimate (the static labels assume a \
                                 fast 27B-class model).",
                            );
                        }
                        _ => {}
                    }
                }
                let idle = self.busy.is_none();
                let any = self.lab_measure
                    || self.lab_bench
                    || self.lab_spec
                    || self.lab_ub
                    || self.lab_kv
                    || self.lab_quality
                    || self.lab_load
                    || self.lab_dials
                    || self.lab_moe
                    || self.lab_vision
                    || self.lab_cache
                    || self.lab_ckpt
                    || self.lab_slots;
                if ui
                    .add_enabled(any && idle, egui::Button::new("⏵ Run selected campaigns"))
                    .on_disabled_hover_text("select at least one campaign — and nothing can start while another operation runs")
                    .on_hover_text(
                        "Runs in sequence, narrated in the activity log. Unloads models to \
                         get honest numbers; offers to start the router if it's down.",
                    )
                    .clicked()
                {
                    run_clicked = true;
                }
                // A cancel affordance next to the thing that started the
                // run, not only the tiny status-bar button (review G11).
                if self.busy.is_some() {
                    if self.cancel_token.is_cancelled() {
                        ui.weak("cancelling…");
                    } else if ui
                        .button("✖ Cancel run")
                        .on_hover_text(
                            "Stops at the next safe boundary; partial results are \
                             kept and your config is restored.",
                        )
                        .clicked()
                    {
                        self.cancel_token.cancel();
                    }
                }
                ui.add_space(8.0);
                ui.separator();
                ui.strong("Trial results");
                let mut apply: Option<(String, String)> = None; // (menu, label)
                let mut toggle_why: Option<String> = None;
                match self.trials.get(&id) {
                    Some(table) if !table.is_empty() => {
                        const MENUS: [&str; 9] = [
                            "spec", "ub", "kv", "load", "dials", "moe", "vision",
                            "cache", "ckpt",
                        ];
                        let reports: Vec<(&str, trial::TrialReport)> = MENUS
                            .iter()
                            .filter_map(|m| {
                                trial::stored_report(m, table).map(|r| (*m, r))
                            })
                            .collect();
                        let mut pending_winners: std::collections::HashSet<String> =
                            Default::default();
                        let mut applied_winners: std::collections::HashSet<String> =
                            Default::default();
                        for (m, r) in &reports {
                            if let Some(w) = r.verdict.winner.as_deref() {
                                if trial::winner_applied(&self.cfg, &id, m, w) {
                                    applied_winners.insert(w.to_string());
                                } else {
                                    pending_winners.insert(w.to_string());
                                }
                            }
                        }
                        // The headline (user request 2026-08-28: nine
                        // verdict sentences made the reader do the
                        // summarizing): only ACTIONABLE winners — already-
                        // applied ones collapse into the ✔ state.
                        let todo: Vec<String> = reports
                            .iter()
                            .filter_map(|(m, r)| {
                                let w = r.verdict.winner.as_deref()?;
                                pending_winners
                                    .contains(w)
                                    .then(|| format!("{w} ({m})"))
                            })
                            .collect();
                        if todo.is_empty() && !applied_winners.is_empty() {
                            ui.colored_label(
                                egui::Color32::from_rgb(0, 170, 0),
                                "✔ current config matches every measured recommendation",
                            );
                        } else if !todo.is_empty() {
                            ui.colored_label(
                                egui::Color32::from_rgb(0, 170, 0),
                                format!("Recommended (not yet applied): {}", todo.join(" · ")),
                            );
                            ui.small(
                                "★ green = winner awaiting your Apply · ✔ blue = winner \
                                 already applied. Different menus' winners stack — you \
                                 can want all of them at once.",
                            );
                        }
                        ui.add_space(4.0);
                        trial_table_grid(
                            ui,
                            "lab-trials",
                            table,
                            &pending_winners,
                            &applied_winners,
                        );
                        // Standing recommendations, recomputed from the
                        // stored numbers — applicable any time, not only
                        // in the moment the run's dialog was open.
                        for (menu_name, report) in &reports {
                            let menu_name = *menu_name;
                            let report = report.clone();
                            ui.add_space(6.0);
                            let applied = trial::applied_keys(&self.cfg, &id, menu_name);
                            let applied_text = if applied.is_empty() {
                                "baseline".to_string()
                            } else {
                                applied
                                    .iter()
                                    .map(|(k, v)| format!("{k} = {v}"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            };
                            ui.label(format!(
                                "{}: {}  (currently applied: {applied_text})",
                                match menu_name {
                                    "spec" => "Speculation",
                                    "ub" => "Prefill batch",
                                    "kv" => "KV precision",
                                    "load" => "Load mode",
                                    "dials" => "Speculation dials",
                                    "moe" => "MoE offload",
                                    "vision" => "Vision cost",
                                    "cache" => "Cache reuse",
                                    _ => "Checkpoints",
                                },
                                report.verdict.reason
                            ));
                            ui.horizontal_wrapped(|ui| {
                                // Applying mid-campaign would race the
                                // worker for the preset and GPU — buttons
                                // sit out until the run finishes.
                                if let Some(w) = &report.verdict.winner {
                                    if ui
                                        .add_enabled(
                                            idle,
                                            egui::Button::new(format!("Apply {w}")),
                                        )
                                        .on_disabled_hover_text("a campaign is running — Apply when it finishes")
                                        .on_hover_text(
                                            "Writes the override, regenerates the preset, \
                                             reloads the router, updates the OpenCode limit.",
                                        )
                                        .clicked()
                                    {
                                        apply = Some((menu_name.to_string(), w.clone()));
                                    }
                                }
                                if !applied.is_empty()
                                    && ui
                                        .add_enabled(
                                            idle,
                                            egui::Button::new("Revert to baseline"),
                                        )
                                        .on_disabled_hover_text("a campaign is running — revert when it finishes")
                                        .on_hover_text(
                                            "Strips this knob from the override — same \
                                             cascade, back to stock.",
                                        )
                                        .clicked()
                                {
                                    apply = Some((menu_name.to_string(), "baseline".into()));
                                }
                                for nm in &report.near_misses {
                                    if ui
                                        .add_enabled(
                                            idle,
                                            egui::Button::new(format!(
                                                "Apply {} anyway",
                                                nm.label
                                            )),
                                        )
                                        .on_disabled_hover_text("a campaign is running — apply when it finishes")
                                        .on_hover_text(format!(
                                            "Rules said no — {} for {}. Your call; reverses \
                                             the same way.",
                                            nm.gain, nm.cost
                                        ))
                                        .clicked()
                                    {
                                        apply =
                                            Some((menu_name.to_string(), nm.label.clone()));
                                    }
                                }
                                // Every measured option stays selectable
                                // (user request 2026-08-27): the rules
                                // recommend, they don't gatekeep — any row
                                // you can see, you can choose.
                                let offered: std::collections::HashSet<&str> = report
                                    .verdict
                                    .winner
                                    .iter()
                                    .map(String::as_str)
                                    .chain(report.near_misses.iter().map(|n| n.label.as_str()))
                                    .collect();
                                for v in trial::menu(menu_name)
                                    .map(|(v, _)| v)
                                    .unwrap_or_default()
                                {
                                    if offered.contains(v.label.as_str())
                                        || !report.raced.contains_key(&v.label)
                                    {
                                        continue;
                                    }
                                    if ui
                                        .add_enabled(
                                            idle,
                                            egui::Button::new(format!("Use {}", v.label)),
                                        )
                                        .on_disabled_hover_text("a campaign is running — apply when it finishes")
                                        .on_hover_text(
                                            "Not the rules' pick — but it's measured, and \
                                             the table shows its numbers. Your call; \
                                             reverses the same way.",
                                        )
                                        .clicked()
                                    {
                                        apply = Some((menu_name.to_string(), v.label.clone()));
                                    }
                                }
                                if ui.button("Why?").clicked() {
                                    toggle_why = Some(menu_name.to_string());
                                }
                            });
                            if self.lab_why.as_deref() == Some(menu_name) {
                                ui.add_space(4.0);
                                ui.set_max_width(560.0);
                                for para in trial::explain(&report) {
                                    ui.label(para);
                                    ui.add_space(4.0);
                                }
                            }
                        }
                        if let Some(line) = trial::slot_summary(table) {
                            ui.add_space(6.0);
                            ui.label(format!("Slot persistence: {line}"));
                        }
                    }
                    _ => {
                        ui.weak("No trials recorded for this model yet.");
                    }
                }
                if let Some(menu_name) = toggle_why {
                    self.lab_why = if self.lab_why.as_deref() == Some(menu_name.as_str()) {
                        None
                    } else {
                        Some(menu_name)
                    };
                }
                if let Some((menu_name, label)) = apply {
                    self.spawn_keep(&id, &menu_name, &label);
                }
                ui.add_space(8.0);
                ui.strong("History");
                let entries = history::for_model(&self.history, &id);
                if entries.is_empty() {
                    ui.weak("No journal entries yet — measures and benches land here.");
                } else {
                    for e in entries.iter().take(8) {
                        let b = e.build.map(|b| format!("b{b}")).unwrap_or_else(|| "b?".into());
                        let what = match (&e.n_ctx, &e.pp_tps, &e.error) {
                            (Some(c), _, _) => format!("ctx {c}"),
                            (_, Some(pp), _) => format!(
                                "pp {pp:.0} / tg {:.0} t/s",
                                e.tg_tps.unwrap_or(0.0)
                            ),
                            (_, _, Some(_)) => "failed".into(),
                            _ => "—".into(),
                        };
                        ui.label(format!("{b} · {what}"));
                    }
                }
                }); // lab-detail scroll
            });
        });
        if run_clicked {
            let action = AfterStart::Lab {
                id: self.lab_selected.clone().unwrap_or_default(),
                measure: self.lab_measure,
                bench: self.lab_bench,
                spec: self.lab_spec,
                ub: self.lab_ub,
                kv: self.lab_kv,
                quality: self.lab_quality,
                load: self.lab_load,
                dials: self.lab_dials,
                moe: self.lab_moe,
                vision: self.lab_vision,
                cache: self.lab_cache,
                ckpt: self.lab_ckpt,
                slots: self.lab_slots,
            };
            self.run_or_offer_start(action);
        }
    }

    fn server_pane(&mut self, ui: &mut egui::Ui) {
        if let Some(warning) = self.vram_contention() {
            ui.colored_label(ui.visuals().warn_fg_color, format!("⚠ {warning}"));
            ui.separator();
        }
        if let Some(line) = &self.meter_line {
            ui.label(line);
            ui.add_space(4.0);
        }
        if !self.cache_stats.is_empty() {
            ui.strong("Prompt cache — measured from your real sessions");
            for s in &self.cache_stats {
                if s.reuse_disabled {
                    let msg = if s.reuse_unsupported_context {
                        format!(
                            "{}: cache-reuse UNSUPPORTED — this model's attention \
                             (SWA/hybrid) can't shift its KV cache, vision or not; \
                             mid-edit turns resume from context checkpoints instead. \
                             Run the Lab's Checkpoint trial — it's the real lever here.",
                            s.model
                        )
                    } else {
                        format!(
                            "{}: cache-reuse off (vision serving disables mid-edit \
                             chunk reuse; appended turns still cache). Whether \
                             text-only would help depends on this model's attention — \
                             the Lab's Vision-cost trial answers with numbers.",
                            s.model
                        )
                    };
                    ui.colored_label(ui.visuals().warn_fg_color, msg);
                } else {
                    ui.label(format!(
                        "{}: {} turns, {:.0}% of prompt tokens reused ({} of {})",
                        s.model,
                        s.turns,
                        s.reuse_fraction() * 100.0,
                        s.reused_tokens,
                        s.prompt_tokens
                    ));
                }
            }
            // Topology advice rides on the same usage evidence (M8 #4).
            let used: Vec<(String, u32, u64)> = self
                .cache_stats
                .iter()
                .filter_map(|s| {
                    let size = self
                        .rows
                        .iter()
                        .find(|r| r.router_id.as_deref() == Some(s.model.as_str()))
                        .map(|r| r.size_bytes)?;
                    Some((s.model.clone(), s.turns, size))
                })
                .collect();
            if let Some(line) = evidence::topology_advice(
                &used,
                self.hardware().vram_mib,
                self.cfg.models_max,
            ) {
                ui.label(line);
            }
            ui.separator();
        }
        if let Some(s) = &self.upstream {
            let age_h = advisor::now_epoch().saturating_sub(s.checked_epoch) / 3600;
            let age = if age_h < 1 {
                "under an hour ago".to_string()
            } else if age_h < 48 {
                format!("{age_h}h ago")
            } else {
                format!("{}d ago", age_h / 24)
            };
            match (s.reachable, s.current_build, s.upstream_build) {
                (true, Some(cur), Some(up)) if up > cur => {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!(
                            "llama.cpp: b{up} available upstream (you run b{cur}{}) — \
                             checked {age}. Server -> Check My llama.cpp to update.",
                            s.behind
                                .map(|b| format!(", checkout {b} commits behind"))
                                .unwrap_or_default()
                        ),
                    );
                    ui.separator();
                }
                (true, Some(cur), Some(up)) => {
                    ui.weak(format!(
                        "llama.cpp b{cur} is current (upstream b{up}, checked {age})"
                    ));
                }
                _ => {
                    ui.weak(format!(
                        "llama.cpp freshness unknown — upstream unreachable (checked {age})"
                    ));
                }
            }
        }
        // What the last rebuild actually did, measured — the journal's
        // build-over-build advisory (M8 #5). Confounded comparisons
        // (config changed between builds) are excluded by construction.
        if let Some(line) = history::build_advisory(&self.history) {
            let regressed = line.contains("worst:");
            if regressed {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!("Rebuild scorecard: {line}"),
                )
            } else {
                ui.weak(format!("Rebuild scorecard: {line}"))
            }
            .on_hover_text(
                "Measured from your history journal: each model's newest numbers \
                 on the current build vs the build before it. Context deltas only \
                 compare identical configs; generation deltas come from llama-bench \
                 baselines. Re-measure/re-bench after a rebuild to feed this.",
            );
        }
        let state = self.router_state.clone();
        match &state {
            None => {
                ui.spinner();
            }
            Some(router::RouterState::Down) => {
                ui.label("Router is down.");
                ui.horizontal(|ui| {
                    if ui.button("⏵ Start Router").clicked() {
                        self.action_start();
                    }
                    if ui
                        .button("⚡ Set Up Everything")
                        .on_hover_text("Start, measure, sync — the whole flow")
                        .clicked()
                    {
                        self.action_setup();
                    }
                });
            }
            Some(router::RouterState::External { detail }) => {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!("External server on port {}: {detail}", self.cfg.port),
                );
                ui.label("Not started by this app — observing only, by design.");
            }
            Some(router::RouterState::Trouble { detail }) => {
                ui.colored_label(ui.visuals().warn_fg_color, format!("Trouble: {detail}"));
                if ui.button("■ Stop Router").clicked() {
                    self.action_stop();
                }
            }
            Some(router::RouterState::Ours { models }) => {
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "Router up on port {} — {} models offered —",
                        self.cfg.port,
                        models.len()
                    ));
                    if ui.button("■ Stop").clicked() {
                        self.action_stop();
                    }
                });
                ui.separator();
                let loaded: Vec<_> = models
                    .iter()
                    .filter(|m| {
                        matches!(
                            m.status.as_str(),
                            "loaded" | "loading" | "sleeping" | "downloading" | "downloaded"
                        )
                    })
                    .cloned()
                    .collect();
                if loaded.is_empty() {
                    ui.label("Nothing loaded right now. The router loads a model automatically when OpenCode (or anything) first asks for it — or use Load on the Library tab.");
                } else {
                    let mut unload: Option<String> = None;
                    for m in &loaded {
                        ui.strong(format!("Currently {}: {}", m.status, m.id));
                        egui::Grid::new(format!("detail-{}", m.id)).show(ui, |ui| {
                            let meas = self.measurements.get(&m.id);
                            ui.label("Measured context");
                            ui.label(
                                meas.and_then(|x| x.n_ctx)
                                    .map(|c| c.to_string())
                                    .unwrap_or_else(|| "not yet measured".into()),
                            );
                            ui.end_row();
                            ui.label("Source");
                            ui.label(m.source.as_deref().unwrap_or("?"));
                            ui.end_row();
                            ui.label("In OpenCode");
                            ui.label(if self.configured.iter().any(|c| c.id == m.id) {
                                "yes"
                            } else {
                                "no"
                            });
                            ui.end_row();
                            if let Some(scan) = &self.scan {
                                for d in &scan.devices {
                                    if d.id.starts_with("CUDA") {
                                        ui.label(format!("{} total", d.id));
                                        ui.label(format!("{} MiB", d.total_mib));
                                        ui.end_row();
                                    }
                                }
                            }
                        });
                        if m.status == "loaded" && ui.button("Unload").clicked() {
                            unload = Some(m.id.clone());
                        }
                        ui.add_space(6.0);
                    }
                    if let Some(id) = unload {
                        self.run_row_action(RowAction::Unload(id));
                    }
                }
            }
        }
        ui.separator();
        // Always present, whatever the router state — the user couldn't
        // find the logs when this lived inside the Ours-only block
        // (2026-08-30). Full file: Tools -> Open Router Log.
        egui::CollapsingHeader::new("Router log (live tail)")
            .default_open(false)
            .show(ui, |ui| {
                ui.weak("Last 16 KB, newest at the bottom. Tools -> Open Router Log for the whole file.");
                egui::ScrollArea::vertical()
                    .max_height(280.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.monospace(tail_of_log());
                    });
            });
        ui.separator();
        ui.strong("Ollama (peer — observed, never managed)");
        if !self.ollama.reachable {
            ui.label(format!(
                "daemon not answering on :{} (not proof its models are gone)",
                self.cfg.ollama_port
            ));
        } else if self.ollama.loaded.is_empty() {
            ui.label("daemon up, nothing loaded");
        } else {
            for m in &self.ollama.loaded {
                ui.label(format!(
                    "  {} — {:.1} GB VRAM",
                    m.name,
                    m.size_vram as f64 / 1e9
                ));
            }
        }
    }

    // ─── Connections ─────────────────────────────────────────────────────
    //
    // The router speaks the standard OpenAI-compatible API — ANY app can
    // connect. OpenCode is the first-class connector (config synced for
    // it); the generic section serves every other app: base URL, the
    // measured model list, and copy-paste snippets.

    fn connections_pane(&mut self, ui: &mut egui::Ui) {
        // The base URL is the identity of this whole tab — one line,
        // always visible above the sub-tabs.
        let base_url = format!("http://127.0.0.1:{}/v1", self.cfg.port);
        ui.horizontal(|ui| {
            ui.label("Base URL:");
            ui.monospace(&base_url);
            if ui.small_button("copy").clicked() {
                ui.ctx().copy_text(base_url.clone());
                self.log("base URL copied");
            }
        });
        ui.horizontal(|ui| {
            let pi_here = piagent::pi_present(&piagent::default_models_path());
            let hermes_here = hermes::hermes_present(&hermes::default_home());
            let oc_here = opencode::default_config_path().exists();
            let tab = |ui: &mut egui::Ui, cur: &mut ConnTab, which: ConnTab, label: String| {
                if ui.selectable_label(*cur == which, label).clicked() {
                    *cur = which;
                }
            };
            tab(ui, &mut self.conn_tab, ConnTab::AnyApp, "Any app".into());
            // Installed connectors read plainly; absent ones say so on
            // the tab itself, so nobody hunts for a section that has
            // nothing to show.
            tab(
                ui,
                &mut self.conn_tab,
                ConnTab::OpenCode,
                if oc_here { "OpenCode".into() } else { "OpenCode (absent)".to_string() },
            );
            tab(
                ui,
                &mut self.conn_tab,
                ConnTab::Pi,
                if pi_here { "pi".into() } else { "pi (absent)".to_string() },
            );
            tab(
                ui,
                &mut self.conn_tab,
                ConnTab::Hermes,
                if hermes_here { "Hermes".into() } else { "Hermes (absent)".to_string() },
            );
        });
        ui.separator();
        // Each sub-pane gets the whole remaining height and scrolls
        // inside it — no more fixed 140px windows fighting each other.
        egui::ScrollArea::vertical()
            .id_salt("connections-body")
            .auto_shrink([false, false])
            .show(ui, |ui| match self.conn_tab {
                ConnTab::AnyApp => self.any_app_section(ui, &base_url),
                ConnTab::OpenCode => {
                    ui.strong("OpenCode (synced connector)");
                    self.opencode_section(ui);
                }
                ConnTab::Pi => self.pi_section(ui),
                ConnTab::Hermes => self.hermes_section(ui),
            });
    }

    fn any_app_section(&mut self, ui: &mut egui::Ui, base_url: &str) {
        let base_url = base_url.to_string();
        ui.strong("Any app (OpenAI-compatible API)");
        ui.small(
            "Point any OpenAI-compatible client here (API key: anything, it's ignored). Use a model id below; the router loads it on first request.",
        );
        let measured: Vec<(String, u64, Option<bool>)> = self
            .measurements
            .iter()
            .filter_map(|(id, m)| m.n_ctx.map(|c| (id.clone(), c, m.tool_call)))
            .collect();
        ui.horizontal(|ui| {
            if ui
                .button("Copy curl example")
                .on_hover_text("A working chat request against your router")
                .clicked()
            {
                let model = measured
                    .first()
                    .map(|(id, _, _)| id.as_str())
                    .unwrap_or("<model-id>");
                ui.ctx().copy_text(format!(
                    "curl {base_url}/chat/completions -H 'Content-Type: application/json' -d '{{\n  \"model\": \"{model}\",\n  \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]\n}}'"
                ));
                self.log("curl example copied");
            }
            if ui
                .button("Copy Python (openai sdk)")
                .on_hover_text("Client setup for any Python app")
                .clicked()
            {
                let model = measured
                    .first()
                    .map(|(id, _, _)| id.as_str())
                    .unwrap_or("<model-id>");
                ui.ctx().copy_text(format!(
                    "from openai import OpenAI\n\nclient = OpenAI(base_url=\"{base_url}\", api_key=\"local\")\nreply = client.chat.completions.create(\n    model=\"{model}\",\n    messages=[{{\"role\": \"user\", \"content\": \"Hello!\"}}],\n)\nprint(reply.choices[0].message.content)"
                ));
                self.log("python snippet copied");
            }
            if ui
                .button("Copy models JSON")
                .on_hover_text(
                    "Machine-readable list of measured models with safe context limits — feed this to your own app's config",
                )
                .clicked()
            {
                let list: Vec<serde_json::Value> = measured
                    .iter()
                    .map(|(id, ctx, tools)| {
                        let safe = opencode::safety_context(*ctx);
                        serde_json::json!({
                            "id": id,
                            "context": safe,
                            "output": safe.div_euclid(2).min(32_768),
                            "tool_call": tools,
                        })
                    })
                    .collect();
                ui.ctx().copy_text(
                    serde_json::to_string_pretty(&serde_json::json!({
                        "base_url": base_url,
                        "models": list,
                    }))
                    .unwrap_or_default(),
                );
                self.log("models JSON copied");
            }
        });
        if measured.is_empty() {
            ui.small("No measured models yet — Load one on the Library tab first.");
        }
    }

    /// Hermes agent (Connections p2, half two). Narrower than the other
    /// connectors by design: we own the machine-written context cache,
    /// and touch the hand-maintained config.yaml only on an explicit
    /// click (user decision 2026-08-30).
    fn hermes_section(&mut self, ui: &mut egui::Ui) {
        let home = hermes::default_home();
        ui.strong("Hermes agent (synced connector)");
        if !hermes::hermes_present(&home) {
            ui.small(format!(
                "Not installed — nothing to sync (looked for {}).",
                home.display()
            ));
            return;
        }
        let base_url = format!("http://127.0.0.1:{}/v1", self.cfg.port);
        let cfg_text = std::fs::read_to_string(hermes::config_path(&home)).unwrap_or_default();
        let registered = hermes::registered_provider(&cfg_text, &base_url);
        ui.label(format!("Config: {}", hermes::config_path(&home).display()));
        ui.small(
            "Hermes reads llama.cpp's /v1/models for context, and our router reports \
             none for an UNLOADED model — so Hermes would assume 128,000 tokens. We \
             write each model's measured window into its context cache instead. \
             Hermes refuses any model under 64,000 tokens, so smaller ones are \
             skipped rather than offered and broken.",
        );
        ui.add_space(4.0);
        match &registered {
            Some(name) => {
                ui.colored_label(
                    egui::Color32::from_rgb(0, 170, 0),
                    format!("✔ registered in Hermes as provider \"{name}\" (use: /model)"),
                );
            }
            None => {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "No Hermes provider points at this router yet — the measured \
                     contexts below have nothing to attach to until one exists.",
                );
                let default_model = self
                    .measurements
                    .iter()
                    .filter(|(_, m)| {
                        m.n_ctx
                            .map(opencode::safety_context)
                            .is_some_and(|c| c >= hermes::MINIMUM_CONTEXT)
                    })
                    .map(|(id, _)| id.clone())
                    .next();
                let can = self.busy.is_none() && default_model.is_some();
                if ui
                    .add_enabled(can, egui::Button::new("Register this router with Hermes"))
                    .on_disabled_hover_text(
                        "needs at least one measured model of 64,000+ tokens — \
                         measure one on the Library tab first",
                    )
                    .on_hover_text(
                        "Appends one custom_providers entry to Hermes's config.yaml. \
                         The file is backed up first and every comment, quote, and \
                         ordering elsewhere is left exactly as it is. Restart Hermes \
                         to pick it up.",
                    )
                    .clicked()
                    && let Some(model) = default_model
                {
                    match hermes::register_provider(&home, &base_url, &model) {
                        Ok(()) => self.log(format!(
                            "registered with Hermes as '{}' (default model {model}) — \
                             restart Hermes, then /model to pick one",
                            hermes::PROVIDER_NAME
                        )),
                        Err(e) => self.log(format!("ERROR registering with Hermes: {e:#}")),
                    }
                }
            }
        }
        ui.add_space(4.0);
        let cached = hermes::cached_for(&hermes::context_cache_path(&home), &base_url);
        if cached.is_empty() {
            ui.small("No measured contexts written yet — press Sync all measured above.");
        } else {
            egui::Grid::new("hermes-grid").striped(true).show(ui, |ui| {
                ui.strong("Model");
                ui.strong("Context written");
                ui.end_row();
                for (id, ctx) in &cached {
                    ui.label(id);
                    ui.label(ctx.to_string());
                    ui.end_row();
                }
            });
        }
        // Name what Hermes will refuse, so a missing model is never a
        // mystery (its own failure mode is a startup error).
        let small: Vec<String> = self
            .measurements
            .iter()
            .filter_map(|(id, m)| {
                let c = opencode::safety_context(m.n_ctx?);
                (c < hermes::MINIMUM_CONTEXT).then(|| format!("{id} ({c})"))
            })
            .collect();
        if !small.is_empty() {
            ui.small(format!(
                "Below Hermes's 64,000-token minimum, so not offered there: {}",
                small.join(", ")
            ));
        }
    }

    /// pi coding agent (Connections p2, first external request —
    /// 2026-08-30). Same shape as the OpenCode mirror: what we wrote,
    /// what it means, and the one action. Absent install says so
    /// plainly rather than hiding — a reader shouldn't wonder whether
    /// the app looked.
    fn pi_section(&mut self, ui: &mut egui::Ui) {
        let path = piagent::default_models_path();
        let installed = piagent::pi_present(&path);
        ui.strong("pi coding agent (synced connector)");
        if !installed {
            ui.small(format!(
                "Not installed — nothing to sync. Looked for {}. \
                 Install pi and the next sync writes a provider there.",
                path.parent().map(|p| p.display().to_string()).unwrap_or_default()
            ));
            return;
        }
        ui.label(format!("Config: {}", path.display()));
        ui.small(
            "We own only the 'modelsteward' provider in this file; everything else \
             (your own providers, models, settings) is left untouched and backed up \
             before every write. pi's own llama.cpp integration assumes a 128k \
             context when the router doesn't report one — these entries carry the \
             context each model MEASURED on this machine instead.",
        );
        ui.add_space(4.0);
        let entries = piagent::configured_models(&path);
        if entries.is_empty() {
            ui.small(
                "No modelsteward provider in pi's config yet — press Sync below (or \
                 File -> Set Up Everything).",
            );
        } else {
            egui::Grid::new("pi-grid").striped(true).show(ui, |ui| {
                        ui.strong("Model (pi id)");
                        ui.strong("Context written");
                        ui.strong("Status");
                        ui.end_row();
                        for (id, ctx) in &entries {
                            ui.label(id);
                            ui.label(ctx.to_string());
                            // Same verdict vocabulary as the OpenCode
                            // mirror: does what we wrote still match
                            // what we measure today?
                            let measured = self
                                .measurements
                                .get(id)
                                .and_then(|m| m.n_ctx)
                                .map(opencode::safety_context);
                            match measured {
                                Some(want) if want == *ctx => {
                                    ui.colored_label(egui::Color32::from_rgb(0, 170, 0), "✔ synced")
                                }
                                Some(want) => ui.colored_label(
                                    ui.visuals().warn_fg_color,
                                    format!("⟳ differs (measured {want})"),
                                ),
                                None => ui.colored_label(
                                    ui.visuals().weak_text_color(),
                                    "? no current measurement",
                                ),
                            };
                            ui.end_row();
                        }
            });
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.busy.is_none(), egui::Button::new("Sync pi + OpenCode"))
                .on_disabled_hover_text("another operation is running")
                .on_hover_text(
                    "Writes every measured model into both agents' configs (same \
                     action as Sync all measured above).",
                )
                .clicked()
            {
                self.action_sync();
            }
            if ui
                .small_button("open config")
                .on_hover_text("Open pi's models.json in your editor")
                .clicked()
            {
                match std::process::Command::new("xdg-open").arg(&path).spawn() {
                    Ok(_) => self.log(format!("opened {}", path.display())),
                    Err(e) => self.log(format!("ERROR opening {}: {e}", path.display())),
                }
            }
            ui.weak("In pi: /model to pick one of these.");
        });
    }

    fn opencode_section(&mut self, ui: &mut egui::Ui) {
        ui.label(format!(
            "Config: {}",
            opencode::default_config_path().display()
        ));
        ui.small(
            "These are the llama.cpp models OpenCode can currently use. Loading a model on the \
             Library tab adds it here automatically with its measured context.",
        );
        ui.add_space(4.0);
        if self.configured.is_empty() {
            ui.label("No llama.cpp models in opencode.json yet — use ⚡ Set Up Everything, or Load a model on the Library tab.");
        }
        let mut pending: Option<RowAction> = None;
        let configured = self.configured.clone();
        egui::ScrollArea::horizontal().show(ui, |ui| {
            egui::Grid::new("configured")
                .striped(true)
                .min_col_width(56.0)
                .show(ui, |ui| {
                    for h in ["Model id", "Context", "Output", "Tools", "Status", ""] {
                        ui.strong(h);
                    }
                    ui.end_row();
                    for c in &configured {
                        ui.label(&c.id);
                        ui.label(c.context.map(|v| v.to_string()).unwrap_or_default());
                        ui.label(c.output.map(|v| v.to_string()).unwrap_or_default());
                        ui.label(match c.tool_call {
                            Some(true) => "yes",
                            Some(false) => "no",
                            None => "?",
                        });
                        // Status vs our measurements.
                        match self.measurements.get(&c.id) {
                            Some(m) if m.n_ctx.is_some() => {
                                let measured = m.n_ctx.unwrap();
                                let write_value = opencode::safety_context(measured);
                                if c.context == Some(write_value) {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(0, 170, 0),
                                        "✔ synced",
                                    )
                                    .on_hover_text(format!(
                                        "measured {measured}, written {write_value} (5% safety margin)"
                                    ));
                                } else {
                                    ui.colored_label(
                                        ui.visuals().warn_fg_color,
                                        format!("⟳ config differs (sync would write {write_value})"),
                                    )
                                    .on_hover_text(format!(
                                        "measured {measured}; the next sync will write {write_value} \
                                         (5% safety margin) over this value. To pin a custom context, \
                                         use the model's ⚙ overrides instead — that pins it at the \
                                         server, so config and server stay in agreement."
                                    ));
                                }
                            }
                            Some(m) if m.error.is_some() => {
                                let err = m.error.clone().unwrap_or_default();
                                ui.colored_label(
                                    ui.visuals().error_fg_color,
                                    "✖ can't load",
                                )
                                .on_hover_text(format!(
                                    "{}\n\nDetail: {err}",
                                    rows::failure_hint(&err)
                                ));
                            }
                            _ => {
                                ui.colored_label(
                                    ui.visuals().weak_text_color(),
                                    "? never measured",
                                )
                                .on_hover_text(
                                    "This entry wasn't written by this tool, or the model is \
                                     gone. Remove it, or measure it from the Library tab.",
                                );
                            }
                        }
                        if ui
                            .button("Remove")
                            .on_hover_text("Comments the entry out with a note — never deletes.")
                            .clicked()
                        {
                            pending = Some(RowAction::RemoveFromOpenCode(c.id.clone()));
                        }
                        ui.end_row();
                    }
                });
        });
        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .button("⚡ Set Up Everything")
                .on_hover_text("Start router if needed, measure anything unmeasured, sync — one click")
                .clicked()
            {
                self.action_setup();
            }
            if ui
                .button("Sync all measured")
                .on_hover_text("Writes every measured model's context into opencode.json")
                .clicked()
            {
                self.action_sync();
            }
        });
        if let Some(s) = &self.last_sync {
            ui.label(s);
        }
        if let Some(action) = pending {
            self.run_row_action(action);
        }
    }

    // ─── Settings ────────────────────────────────────────────────────────

    fn settings_pane(&mut self, ui: &mut egui::Ui) {
        ui.label(format!("Stored in {}", system::config_file().display()));
        ui.add_space(6.0);

        ui.strong("Model scan directories (one per line)");
        ui.small("Ollama's store and the HuggingFace cache are found automatically.");
        ui.add(
            egui::TextEdit::multiline(&mut self.edit_scan_dirs)
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        if ui.button("📁 Add directory…").clicked()
            && let Some(dir) = rfd::FileDialog::new()
                .set_title("Add a model scan directory")
                .pick_folder()
        {
            if !self.edit_scan_dirs.trim().is_empty() {
                self.edit_scan_dirs.push('\n');
            }
            self.edit_scan_dirs.push_str(&dir.display().to_string());
        }
        ui.add_space(6.0);

        egui::Grid::new("settings").num_columns(2).show(ui, |ui| {
            ui.label("Router port");
            ui.text_edit_singleline(&mut self.edit_port);
            ui.end_row();
            ui.label("llama-server binary");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut self.edit_server_bin);
                if ui.button("Browse…").clicked()
                    && let Some(file) = rfd::FileDialog::new()
                        .set_title("Choose the llama-server binary")
                        .pick_file()
                {
                    self.edit_server_bin = file.display().to_string();
                }
                ui.small("empty = auto-pick");
            });
            ui.end_row();
            ui.label("Max loaded models");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut self.edit_models_max);
                ui.small("1 = one big model; raise to keep a small sidecar model resident too")
                    .on_hover_text(
                        "Tradeoff: residents split the VRAM that --fit hands out, so \
                         each gets a smaller context — but switching between resident \
                         models is instant instead of a full reload. The Server tab \
                         suggests 2 when your real usage shows you alternating between \
                         two models that fit together.",
                    );
            });
            ui.end_row();
            ui.label("Ollama port");
            ui.text_edit_singleline(&mut self.edit_ollama_port);
            ui.end_row();
        });

        // Detected installs: autodiscovery meets manual pointing — one
        // click adopts a found binary instead of typing its path.
        if let Some(scan) = &self.scan
            && scan.installs.is_empty()
        {
            // Usability review D6: with zero installs this section used
            // to hide entirely — exactly when its guidance was needed.
            ui.add_space(4.0);
            ui.strong("No llama.cpp found on this machine");
            ui.label(
                "modelsteward drives llama.cpp's llama-server (b10216+). Either \
                 point the field above at an existing binary with Browse…, or let \
                 the app build one for you: Server menu -> Check My llama.cpp -> \
                 Managed llama.cpp -> Set up (clone + build newest release).",
            );
        }
        if let Some(scan) = &self.scan
            && !scan.installs.is_empty()
        {
            ui.add_space(4.0);
            ui.strong("llama-server binary — what serves your models");
            ui.small(
                "The ONE place that selects the serving binary. Building new \
                 binaries happens in the Build Advisor (Server menu); everything \
                 it produces appears here. Changes take effect on next router start.",
            );
            let installs = scan.installs.clone();
            let picked = self.picked_server();
            // Canonicalized ONCE per frame (installs paths are already
            // canonical; per-row data_dir + canonicalize churn was a
            // review catch).
            let managed_root = {
                let d = managed::data_dir();
                d.canonicalize().unwrap_or(d)
            };
            // The default: no explicit choice, newest of the user's OWN
            // installs (managed builds serve only when chosen here).
            ui.horizontal(|ui| {
                if ui.button("Use").clicked() {
                    self.action_pin(None);
                }
                let auto_now = self
                    .cfg
                    .server_bin
                    .is_none()
                    .then(|| picked.clone())
                    .flatten()
                    .and_then(|p| {
                        installs
                            .iter()
                            .find(|i| i.server_path == p)
                            .map(discover::install_alias)
                    })
                    .map(|a| format!(" (currently -> {a})"))
                    .unwrap_or_default();
                let text = format!("Auto — newest of your own installs{auto_now}");
                if self.cfg.server_bin.is_none() {
                    ui.colored_label(
                        egui::Color32::from_rgb(0, 170, 0),
                        format!("{text}   (current)"),
                    );
                } else {
                    ui.label(text);
                }
            });
            // The list grows one row per archived build — scroll it
            // (design session 2026-08-30, #4) instead of letting it
            // push the rest of Settings off-screen.
            egui::ScrollArea::vertical().id_salt("server-binaries").max_height(220.0).show(ui, |ui| {
            for inst in &installs {
                ui.horizontal(|ui| {
                    if ui.button("Use").clicked() {
                        self.action_pin(Some(inst.server_path.clone()));
                    }
                    // Feature-alias naming (user idea 2026-08-28): the
                    // build number plus the backends it was compiled with
                    // IS the build's identity — b10672-cuda-vulkan.
                    let alias = discover::install_alias(inst);
                    let tag = if inst.server_path.starts_with(&managed_root) {
                        " (app-managed)"
                    } else {
                        ""
                    };
                    let text = format!("{alias}{tag} — {}", inst.server_path.display());
                    let is_current = self.cfg.server_bin.is_some()
                        && picked.as_deref() == Some(inst.server_path.as_path());
                    if is_current {
                        ui.colored_label(
                            egui::Color32::from_rgb(0, 170, 0),
                            format!("{text}   (current)"),
                        );
                    } else {
                        ui.label(text);
                    }
                });
            }
            });
        }
        ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Advisor model");
                let current = self
                    .cfg
                    .advisor_model
                    .clone()
                    .unwrap_or_else(|| "Auto (best quality, fastest tie)".into());
                let mut change: Option<Option<String>> = None;
                egui::ComboBox::from_id_salt("advisor-model")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(
                                self.cfg.advisor_model.is_none(),
                                "Auto (best quality, fastest tie)",
                            )
                            .clicked()
                        {
                            change = Some(None);
                        }
                        for (id, m) in &self.measurements {
                            if m.n_ctx.is_none() {
                                continue;
                            }
                            let label = format!(
                                "{id}{}",
                                m.loop_reliability
                                    .map(|l| format!(" (loops {:.0}%)", l * 100.0))
                                    .unwrap_or_default()
                            );
                            if ui
                                .selectable_label(
                                    self.cfg.advisor_model.as_deref() == Some(id),
                                    label,
                                )
                                .clicked()
                            {
                                change = Some(Some(id.clone()));
                            }
                        }
                    });
                if let Some(c) = change {
                    self.cfg.advisor_model = c;
                    let _ = self.cfg.save(&system::config_file());
                }
                ui.small("answers failure explanations, briefs, triage, and tuning questions");
            });
            ui.checkbox(
                &mut self.managed_auto_edit,
                "Keep managed llama.cpp current (build + archive new releases \
                 automatically; never changes what's served)",
            )
            .on_hover_text(
                "When the daily upstream check finds a new bNNNN release and the \
                 managed checkout exists, the app builds it in the background \
                 (CPU-only work) and archives the binaries into the list above. \
                 SERVING a build always stays your explicit click. Save to apply.",
            );
        if let Some(picked) = self.picked_server() {
            let build = self
                .scan
                .as_ref()
                .and_then(|s| s.installs.iter().find(|i| i.server_path == picked))
                .and_then(|i| i.build)
                .map(|b| format!(" (b{b})"))
                .unwrap_or_default();
            ui.small(format!("currently using: {}{build}", picked.display()));
        }
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if let Some(err) = &self.settings_error {
                ui.colored_label(egui::Color32::from_rgb(220, 60, 60), format!("⚠ {err}"));
            }
            if ui.button("Save & Rescan").clicked() {
                self.settings_error = None;
                match self.parse_edit_buffers() {
                    Ok(new_cfg) => {
                        let port_changed = new_cfg.port != self.cfg.port;
                        let router_changed = port_changed
                            || new_cfg.server_bin != self.cfg.server_bin
                            || new_cfg.models_max != self.cfg.models_max;
                        self.cfg = new_cfg;
                        match self.cfg.save(&system::config_file()) {
                            Ok(()) => {
                                self.log("settings saved");
                                // Review G12: offer the follow-up, don't
                                // instruct it (the hand-fix->feature rule,
                                // applied to ourselves).
                                if router_changed {
                                    self.settings_needs_apply = true;
                                }
                                self.spawn_scan();
                            }
                            Err(e) => self.log(format!("ERROR saving settings: {e:#}")),
                        }
                    }
                    Err(e) => self.settings_error = Some(e),
                }
            }
            if ui.button("Revert").clicked() {
                self.reset_edit_buffers();
            }
        });
        if self.settings_needs_apply {
            ui.horizontal(|ui| {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "saved — but the running router still uses the old settings",
                );
                if ui
                    .add_enabled(
                        self.busy.is_none(),
                        egui::Button::new("Apply now: restart router + regen preset + sync"),
                    )
                    .on_disabled_hover_text("another operation is running — apply when it finishes")
                    .on_hover_text(
                        "Stops the router, rewrites the preset with the new settings, \
                         starts it again, and re-syncs opencode.json's baseURL. \
                         Loaded models will reload on next use.",
                    )
                    .clicked()
                {
                    self.settings_needs_apply = false;
                    self.action_apply_settings();
                }
                if ui.small_button("later").on_hover_text("Dismiss — apply by hand with Stop + Start on the Server tab").clicked() {
                    self.settings_needs_apply = false;
                }
            });
        }
    }

    /// Review G12: the one-click follow-through after router settings
    /// change — stop, regen preset, start, sync. What the log used to
    /// tell the user to go do by hand.
    fn action_apply_settings(&mut self) {
        if let Some(msg) =
            self.serving_disruption("Applying settings (a full router restart)")
        {
            self.confirm_disrupt = Some((msg, Disrupt::ApplySettings));
            return;
        }
        self.action_apply_settings_now();
    }

    fn action_apply_settings_now(&mut self) {
        let cfg = self.cfg.clone();
        let measurements = self.measurements.clone();
        self.spawn("applying settings: restarting router", move |tx| {
            // Stop errors are non-fatal: a router that isn't running is
            // already stopped for our purposes.
            if let Err(e) = router::stop(&router::state_dir(), &system::preset_path()) {
                let _ = tx.send(Msg::Progress(format!("stop: {e:#} — continuing")));
            }
            match system::write_preset(&cfg, &[]) {
                Ok((path, n)) => {
                    let _ = tx.send(Msg::PresetWritten(path, n));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("preset: {e:#}")));
                    return;
                }
            }
            if !start_router_and_wait(&cfg, tx) {
                return;
            }
            match run_sync(&cfg, &measurements) {
                Ok((report, lines)) => {
                    let _ = tx.send(Msg::SyncDone(report));
                    for l in lines {
                        let _ = tx.send(Msg::Progress(l));
                    }
                    send_configured(tx);
                    let _ = tx.send(Msg::Finished(
                        "settings applied: router restarted, preset regenerated, opencode.json synced".into(),
                    ));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("sync: {e:#}")));
                }
            }
        });
    }

    fn parse_edit_buffers(&self) -> Result<settings::AppConfig, String> {
        let port: u16 = self
            .edit_port
            .trim()
            .parse()
            .map_err(|_| format!("invalid router port: {:?}", self.edit_port))?;
        let ollama_port: u16 = self
            .edit_ollama_port
            .trim()
            .parse()
            .map_err(|_| format!("invalid ollama port: {:?}", self.edit_ollama_port))?;
        let models_max: u32 = self
            .edit_models_max
            .trim()
            .parse()
            .ok()
            .filter(|n| *n >= 1)
            .ok_or_else(|| format!("max loaded models must be ≥ 1: {:?}", self.edit_models_max))?;
        let scan_dirs: Vec<PathBuf> = self
            .edit_scan_dirs
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(PathBuf::from)
            .collect();
        let server_bin = match self.edit_server_bin.trim() {
            "" => None,
            path => {
                let p = PathBuf::from(path);
                if !p.is_file() {
                    return Err(format!("llama-server binary not found: {path}"));
                }
                Some(p)
            }
        };
        Ok(settings::AppConfig {
            scan_dirs,
            port,
            server_bin,
            ollama_port,
            models_max,
            checkouts: self.cfg.checkouts.clone(),
            advisor_model: self.cfg.advisor_model.clone(),
            kwh_price_usd: self.cfg.kwh_price_usd,
            cloud_price_per_mtok: self.cfg.cloud_price_per_mtok,
            disabled: self.cfg.disabled.clone(),
            managed_auto_build: self.managed_auto_edit,
            archives_keep: self.cfg.archives_keep,
            persistence_prompt_dismissed: self.cfg.persistence_prompt_dismissed,
            overrides: self.cfg.overrides.clone(),
        })
    }

    /// The per-model override dialog: pin context, KV type, extra flags.
    /// Saved to config.json -> preset regenerated -> router reloaded, so it
    /// takes effect on the model's next load. Changing flags makes the old
    /// measurement stale by fingerprint — the next measure catches it.
    fn override_dialog(&mut self, ctx: &egui::Context) {
        let Some(ed) = &mut self.override_editor else {
            return;
        };
        let mut save = false;
        let mut cancel = false;
        let mut open = true;
        egui::Window::new(format!("Overrides — {}", ed.id))
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.small(
                    "Showing the values this model will actually use. Anything you leave \
                     equal to the optimized default isn't pinned — it keeps auto-adapting.",
                );
                ui.add_space(4.0);
                egui::Grid::new("override-fields").num_columns(2).show(ui, |ui| {
                    ui.label("Context");
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut ed.ctx_text);
                        ui.small(match ed.optimized_ctx {
                            Some(c) => format!("optimized: {c} (measured by --fit)"),
                            None => "optimized: auto (--fit; not measured yet)".into(),
                        });
                    });
                    ui.end_row();
                    ui.label("KV cache type");
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut ed.kv_text);
                        ui.small(format!(
                            "optimized: {} — f16 | q8_0 | q4_0",
                            router::DEFAULT_KV_TYPE
                        ));
                    });
                    ui.end_row();
                    for f in ed
                        .promoted
                        .iter_mut()
                        .filter(|f| matches!(f.key, "spec-type" | "ubatch-size"))
                    {
                        ui.label(f.label);
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(&mut f.text);
                            ui.small(&f.hint);
                        });
                        ui.end_row();
                    }
                    ui.label("Sampling defaults");
                    ui.horizontal(|ui| {
                        for f in ed
                            .promoted
                            .iter_mut()
                            .filter(|f| matches!(f.key, "temp" | "top-k" | "top-p"))
                        {
                            ui.small(f.label);
                            ui.add(
                                egui::TextEdit::singleline(&mut f.text).desired_width(48.0),
                            )
                            .on_hover_text(&f.hint);
                        }
                        ui.small("(config only — never a trial target)");
                    });
                    ui.end_row();
                });
                if ed.has_mmproj {
                    let mut with_vision = !ed.no_mmproj;
                    ui.checkbox(&mut with_vision, "Serve with vision (mmproj)")
                        .on_hover_text(
                            "Vision costs VRAM (a smaller fitted context) and disables \
                             mid-edit cache-reuse on models whose cache can shift — \
                             appended turns stay cached either way. The Lab's \
                             Vision-cost trial measures what text-only is worth on THIS \
                             model. The projector stays on disk; one check restores \
                             vision, and OpenCode's image modality follows either way.",
                        );
                    ed.no_mmproj = !with_vision;
                }
                ui.label("Extra llama-server flags (key = value, one per line):");
                ui.add(
                    egui::TextEdit::multiline(&mut ed.extra_text)
                        .desired_rows(3)
                        .hint_text("fit-target = 2048   (or paste: --n-cpu-moe 32)")
                        .desired_width(f32::INFINITY),
                );
                ui.small("Applied on the model's next load. Re-measure afterwards — changed flags make old numbers stale.");
                if let Some(err) = &ed.error {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 60, 60),
                        format!("⚠ {err}"),
                    );
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        save = true;
                    }
                    if ui
                        .button("Reset to optimized")
                        .on_hover_text(
                            "Puts every field back to the measured/default values — the \
                             escape hatch when tuning went sideways. Save afterwards to apply.",
                        )
                        .clicked()
                    {
                        ed.ctx_text = ed
                            .optimized_ctx
                            .map(|c| c.to_string())
                            .unwrap_or_default();
                        ed.kv_text = router::DEFAULT_KV_TYPE.to_string();
                        ed.extra_text.clear();
                        for f in &mut ed.promoted {
                            f.text = f.default_text.to_string();
                        }
                        ed.no_mmproj = false;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });
        if !open || cancel {
            self.override_editor = None;
            return;
        }
        if !save {
            return;
        }
        let ed = self.override_editor.take().expect("checked above");
        let result: Result<(), String> = (|| {
            let ctx_val = match ed.ctx_text.trim() {
                "" => None,
                t => Some(t.parse::<u64>().map_err(|_| format!("invalid context: {t:?}"))?),
            };
            // Delta semantics: values equal to the optimized baseline are
            // stored as "no override" so they keep following auto-fit and
            // the global default rather than freezing today's numbers.
            let ctx_val = ctx_val.filter(|v| Some(*v) != ed.optimized_ctx);
            let kv = match ed.kv_text.trim() {
                "" => None,
                t if t.eq_ignore_ascii_case(router::DEFAULT_KV_TYPE) => None,
                t => Some(t.to_string()),
            };
            let mut extra = Vec::new();
            for line in ed.extra_text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                // Accepts `--n-cpu-moe 32` / `-ub 2048` as every model
                // card writes them (usability review G7).
                let (k, v) = router::parse_extra_line(line).map_err(|e| e.to_string())?;
                if ed.promoted.iter().any(|f| f.key == k) {
                    return Err(format!("{k} has its own field above — set it there"));
                }
                extra.push((k, v));
            }
            for f in &ed.promoted {
                let t = f.text.trim();
                if !t.is_empty() && t != f.default_text {
                    extra.push((f.key.to_string(), t.to_string()));
                }
            }
            let ov = router::ModelOverrides {
                cache_type_kv: kv,
                ctx: ctx_val,
                extra,
                no_mmproj: ed.no_mmproj,
            };
            if ov == router::ModelOverrides::default() {
                self.cfg.overrides.remove(&ed.id);
            } else {
                self.cfg.overrides.insert(ed.id.clone(), ov);
            }
            self.cfg
                .save(&system::config_file())
                .map_err(|e| format!("{e:#}"))?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.log(format!("{}: overrides saved", ed.id));
                let cfg = self.cfg.clone();
                self.spawn("applying overrides (preset + reload)", move |tx| {
                    let _ = tx.send(match system::write_preset(&cfg, &[]) {
                        Ok((_, n)) => {
                            let reload_note = match router::reload(cfg.port) {
                                Ok(_) => "router reloaded",
                                Err(_) => "router not running (will apply on next start)",
                            };
                            Msg::Finished(format!(
                                "preset regenerated ({n} models), {reload_note} — re-measure to refresh numbers"
                            ))
                        }
                        Err(e) => Msg::Error(format!("apply overrides: {e:#}")),
                    });
                });
            }
            Err(e) => {
                let mut ed = ed;
                ed.error = Some(e);
                self.override_editor = Some(ed); // reopen with user's text intact
            }
        }
    }

    /// Ask one of the router's models to explain a failure the rules
    /// couldn't. Backend + guardrails in core/aiadvisor.rs; the answer
    /// arrives as Msg::Advisory and renders in the labeled advisor window.
    fn spawn_failure_advisory(
        &mut self,
        display: &str,
        router_id: &Option<String>,
        path: &Option<PathBuf>,
        d: &diagnose::Diagnosis,
    ) {
        let subject = router_id.clone().unwrap_or_else(|| display.to_string());
        let Some(answerer) = self.pick_answerer(Some(&subject)) else {
            self.log(
                "advisory needs another servable model to ask — measure one first"
                    .to_string(),
            );
            return;
        };
        let error = d
            .evidence
            .clone()
            .unwrap_or_else(|| d.explanation.clone());
        // The SERVING binary's build — not discovery-order-first, which
        // could feed the model a false fact (review catch 2026-08-28).
        let served = self.picked_server();
        let build = self.scan.as_ref().and_then(|s| {
            s.installs
                .iter()
                .find(|i| Some(&i.server_path) == served.as_ref())
                .and_then(|i| i.build)
        });
        let gpus = self
            .scan
            .as_ref()
            .map(|s| {
                let phys = discover::physical_gpus(&s.devices);
                if phys.is_empty() {
                    "none detected".to_string()
                } else {
                    phys.iter()
                        .map(|g| format!("{} ({} MiB)", g.name, g.vram_mib))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            })
            .unwrap_or_else(|| "unknown".into());
        let file_gib = path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as f64 / (1024.0 * 1024.0 * 1024.0));
        let port = self.cfg.port;
        self.spawn(&format!("asking {answerer} about {display}"), move |tx| {
            let log = std::fs::read_to_string(router::state_dir().join("router.log"))
                .unwrap_or_default();
            let tail = aiadvisor::log_tail_for(&log, &subject, 60);
            let prompt = aiadvisor::failure_prompt(
                &subject,
                &error,
                build,
                &gpus,
                rows::read_ram_mib(),
                file_gib,
                &tail,
            );
            let backend = aiadvisor::RouterAdvisor {
                port,
                model: answerer.clone(),
            };
            use aiadvisor::Advisor as _;
            let _ = tx.send(match backend.ask(aiadvisor::SYSTEM, &prompt) {
                Ok(text) => Msg::Advisory {
                    subject,
                    model: backend.describe(),
                    text,
                },
                Err(e) => Msg::Error(format!("advisory: {e:#}")),
            });
            let _ = tx.send(Msg::Finished("advisory finished".into()));
        });
    }

    /// Delegates to the tested core rules: pinned advisor, else best
    /// quality tier, fastest within it (see aiadvisor::pick_advisor).
    fn pick_answerer(&self, exclude: Option<&str>) -> Option<String> {
        aiadvisor::pick_advisor(
            &self.measurements,
            self.cfg.advisor_model.as_deref(),
            exclude,
        )
    }

    /// Managed-checkout worker: clone if needed, fetch tags, check out
    /// the newest release tag, build with the advisor's backend logic,
    /// archive the binaries. Deterministic end to end.
    fn action_managed_build(&mut self) {
        let Some(check) = self.build_check.clone() else {
            self.log("run the build check first (Server -> Check My llama.cpp)".to_string());
            return;
        };
        let sel = self
            .backend_sel
            .unwrap_or_else(|| advisor::default_backends(&check));
        let keep = self.cfg.archives_keep as usize;
        let serving = self.picked_server();
        self.spawn("managed llama.cpp: fetch + build newest release", move |tx| {
            let tx2 = tx.clone();
            let mut progress = move |line: String| {
                let _ = tx2.send(Msg::Progress(line));
            };
            let result = managed::build_release(&check, sel, &mut progress);
            if result.is_ok() {
                for l in managed::prune_archives(keep, serving.as_deref()) {
                    let _ = tx.send(Msg::Progress(format!(
                        "pruned old archive {l} (retention: keep {keep})"
                    )));
                }
            }
            let _ = tx.send(Msg::Managed(managed::status()));
            let _ = tx.send(match result {
                Ok(b) => Msg::Finished(format!(
                    "managed llama.cpp built + archived b{b} — select it in Settings \
                     -> llama-server binary to serve it"
                )),
                Err(e) => Msg::Error(format!("managed build: {e:#}")),
            });
        });
    }

    /// Pin (or unpin) the server binary the router uses. Takes effect on
    /// the next router start — never restarts a running server itself.
    fn action_pin(&mut self, path: Option<PathBuf>) {
        self.cfg.server_bin = path.clone();
        self.edit_server_bin = path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        match self.cfg.save(&system::config_file()) {
            Ok(()) => self.log(match path {
                Some(p) => format!(
                    "serving binary set to {} — takes effect on next router start",
                    p.display()
                ),
                None => {
                    "serving binary set to Auto (newest of your own installs) on next start"
                        .into()
                }
            }),
            Err(e) => self.log(format!("ERROR saving pin: {e:#}")),
        }
    }

    /// Rebuild triage (advisory): what's in the pending update for YOUR
    /// models? Commits come from git; the judgment is a labeled opinion.
    fn action_rebuild_triage(&mut self) {
        let Some(check) = &self.build_check else { return };
        let (Some(cur), Some(up)) = (check.current_build, check.upstream_build) else {
            return;
        };
        let repo = check
            .repo
            .clone()
            .or_else(|| managed::checkout_present().then(managed::checkout_dir));
        let Some(repo) = repo else {
            self.log("no git checkout to read commits from".to_string());
            return;
        };
        let Some(answerer) = self.pick_answerer(None) else {
            self.log("advisory needs a measured model to ask — measure one first".to_string());
            return;
        };
        let models: Vec<String> = self
            .rows
            .iter()
            .filter(|r| r.router_id.is_some())
            .map(|r| {
                format!(
                    "{}{}{}",
                    r.display,
                    if r.vision { " (vision)" } else { "" },
                    if rows::looks_moe(None, &r.display) { " (MoE)" } else { "" },
                )
            })
            .collect();
        let port = self.cfg.port;
        self.spawn(&format!("triaging b{cur}->b{up} against your models"), move |tx| {
            let repo_s = repo.display().to_string();
            // Tags can lag the daily fetch (found live: b10630 running,
            // tag absent locally) — refresh them, then fall back to HEAD
            // if the current build's tag still isn't known.
            let _ = std::process::Command::new("git")
                .args(["-C", &repo_s, "fetch", "--quiet", "--tags", "origin"])
                .status();
            let log_range = |range: &str| -> Option<String> {
                let o = std::process::Command::new("git")
                    .args(["-C", &repo_s, "log", "--oneline", "--no-decorate", range])
                    .output()
                    .ok()?;
                o.status.success().then(|| {
                    String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .take(200)
                        .collect::<Vec<_>>()
                        .join("\n")
                })
            };
            let commits = log_range(&format!("b{cur}..origin/master"))
                .filter(|c| !c.is_empty())
                .or_else(|| log_range("HEAD..origin/master"))
                .unwrap_or_default();
            if commits.is_empty() {
                let _ = tx.send(Msg::Error(
                    "no commits found between builds (fetch may be stale)".into(),
                ));
                return;
            }
            let prompt = aiadvisor::triage_prompt(&commits, &models, cur, up);
            let backend = aiadvisor::RouterAdvisor {
                port,
                model: answerer,
            };
            use aiadvisor::Advisor as _;
            let _ = tx.send(match backend.ask(aiadvisor::SYSTEM, &prompt) {
                Ok(text) => Msg::Advisory {
                    subject: format!("update b{cur} -> b{up}"),
                    model: backend.describe(),
                    text,
                },
                Err(e) => Msg::Error(format!("triage: {e:#}")),
            });
            let _ = tx.send(Msg::Finished("triage finished".into()));
        });
    }

    /// Fleet brief (advisory): regenerate the findings report, feed its
    /// machine-readable JSON to a served model, collect the answer in
    /// the Advisor window. The JSON is sanitized at the source — the
    /// same artifact a user would share.
    fn action_fleet_brief(&mut self) {
        let Some(answerer) = self.pick_answerer(None) else {
            self.log("the fleet brief needs a measured model to ask — measure one first".to_string());
            return;
        };
        let cfg = self.cfg.clone();
        let port = self.cfg.port;
        self.spawn(&format!("fleet brief: asking {answerer}"), move |tx| {
            let result = (|| -> anyhow::Result<String> {
                crate::core::report::generate(&cfg)?;
                let json = std::fs::read_to_string(
                    router::state_dir().join("findings-report.json"),
                )?;
                let backend = aiadvisor::RouterAdvisor {
                    port,
                    model: answerer.clone(),
                };
                use aiadvisor::Advisor as _;
                backend.ask(aiadvisor::BRIEF_SYSTEM, &aiadvisor::fleet_prompt(&json))
            })();
            let _ = tx.send(match result {
                Ok(text) => Msg::Advisory {
                    subject: "fleet brief".to_string(),
                    model: answerer,
                    text,
                },
                Err(e) => Msg::Error(format!("fleet brief: {e:#}")),
            });
            let _ = tx.send(Msg::Finished("fleet brief finished".into()));
        });
    }

    /// The advisor pane: every AI opinion from this session, newest first,
    /// each naming the model that wrote it. Opinions, not measurements —
    /// the label does the quarantining.
    /// Help -> the tuning sequence (user request 2026-08-28, after two
    /// live lessons in one battery: measurements taken before a
    /// placement is applied reflect a config you'll never serve). The
    /// REAL guards are contextual (the Lab warns at the moment of
    /// mistake); this is the readable map.
    fn refresh_meter_report(&mut self) {
        // The poller harvests continuously — the window just reads the
        // ledger (harvest_first=false keeps this cheap and lock-free).
        self.meter_report =
            system::meter_report_text(&self.cfg, self.meter_report_range, false)
                .unwrap_or_else(|e| format!("meter: {e:#}"));
    }

    fn persistence_window(&mut self, ctx: &egui::Context) {
        if !self.show_persistence_prompt {
            return;
        }
        let mut open = true;
        egui::Window::new("One-time setup: GPU persistence")
            .collapsible(false)
            .resizable(false)
            .default_width(460.0)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(
                    "Your NVIDIA driver tears down and re-initializes on every \
                     model load — a fixed tax of up to a few seconds on each load, \
                     hot-swap, and measurement. Persistence mode keeps the driver \
                     resident (cost: a few idle watts).",
                );
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let has_pkexec = std::path::Path::new("/usr/bin/pkexec").exists();
                    if ui
                        .add_enabled(has_pkexec, egui::Button::new("Enable now (until reboot)"))
                        .on_disabled_hover_text("pkexec not found — run: sudo nvidia-smi -pm 1")
                        .on_hover_text(
                            "Runs `pkexec nvidia-smi -pm 1` — your desktop asks for \
                             your password; this app never handles it.",
                        )
                        .clicked()
                    {
                        let tx = self.tx.clone();
                        std::thread::spawn(move || {
                            let ok = std::process::Command::new("pkexec")
                                .args(["nvidia-smi", "-pm", "1"])
                                .status()
                                .map(|s| s.success())
                                .unwrap_or(false);
                            let _ = tx.send(Msg::Progress(if ok {
                                "persistence: enabled until reboot".into()
                            } else {
                                "persistence: not enabled (cancelled or failed)".into()
                            }));
                            // Re-probe so the window reacts to the truth,
                            // not the button click.
                            let state = std::process::Command::new("nvidia-smi")
                                .args(["--query-gpu=persistence_mode", "--format=csv,noheader"])
                                .output()
                                .ok()
                                .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                                .and_then(|s| advisor::parse_persistence_mode(&s));
                            let _ = tx.send(Msg::Persistence(state));
                        });
                    }
                    if ui
                        .button("Don't ask again")
                        .on_hover_text("The Build Advisor still shows the state and the commands.")
                        .clicked()
                    {
                        self.cfg.persistence_prompt_dismissed = true;
                        let _ = self.cfg.save(&system::config_file());
                        self.show_persistence_prompt = false;
                    }
                });
                ui.add_space(4.0);
                ui.weak(
                    "Make it permanent (survives reboot): Server -> Check My \
                     llama.cpp — the Build Advisor derives the exact fix for \
                     THIS machine's daemon setup (live catch 2026-08-30: the \
                     generic systemctl-enable advice fails on static units).",
                );
            });
        if !open {
            // Closed with the X = \"later\": ask again next launch.
            self.show_persistence_prompt = false;
        }
    }

    fn disrupt_window(&mut self, ctx: &egui::Context) {
        let Some((msg, _)) = &self.confirm_disrupt else {
            return;
        };
        let msg = msg.clone();
        let mut verdict: Option<bool> = None;
        egui::Window::new("Interrupt live serving?")
            .collapsible(false)
            .resizable(false)
            .default_width(440.0)
            .show(ctx, |ui| {
                ui.label(msg);
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("⚠ Interrupt and continue").clicked() {
                        verdict = Some(true);
                    }
                    if ui.button("Cancel — leave it serving").clicked() {
                        verdict = Some(false);
                    }
                });
            });
        match verdict {
            Some(true) => {
                if let Some((_, action)) = self.confirm_disrupt.take() {
                    match action {
                        Disrupt::Row(a) => self.run_row_action_now(a),
                        Disrupt::Dispatch(a, s) => self.dispatch_now(a, s),
                        Disrupt::Stop => self.action_stop_now(),
                        Disrupt::ApplySettings => self.action_apply_settings_now(),
                    }
                }
            }
            Some(false) => self.confirm_disrupt = None,
            None => {}
        }
    }

    fn meter_window(&mut self, ctx: &egui::Context) {
        if !self.show_meter {
            return;
        }
        let mut open = true;
        egui::Window::new("Token Meter — usage & measured cost")
            .collapsible(false)
            .default_width(620.0)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let mut pick = None;
                    for (label, range) in [
                        ("Today", Some("today")),
                        ("24h", Some("24h")),
                        ("7 days", Some("7d")),
                        ("All time", None),
                    ] {
                        if ui
                            .selectable_label(self.meter_report_range == range, label)
                            .clicked()
                        {
                            pick = Some(range);
                        }
                    }
                    if let Some(range) = pick {
                        self.meter_report_range = range;
                        self.refresh_meter_report();
                    }
                    if ui.button("⟳ Refresh").on_hover_text("Re-read the ledger (it grows every ~2s while the router serves)").clicked() {
                        self.refresh_meter_report();
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical().max_height(440.0).show(ui, |ui| {
                    ui.monospace(&self.meter_report);
                });
            });
        if !open {
            self.show_meter = false;
        }
    }

    fn guide_window(&mut self, ctx: &egui::Context) {
        if !self.show_guide {
            return;
        }
        let mut open = true;
        egui::Window::new("Tuning Guide — the sequence")
            .collapsible(false)
            .default_width(560.0)
            .open(&mut open)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().max_height(460.0).show(ui, |ui| {
                    for (head, body) in [
                        (
                            "Why order matters",
                            "Trials race configs on top of what's currently applied. \
                             Measuring before the right foundation is applied records \
                             numbers for a configuration you'll never actually serve — \
                             honest, but useless.",
                        ),
                        (
                            "1 · Measure + Bench",
                            "Library -> Load (or Lab -> Measure/Bench). This gives every \
                             later step its baseline: settled context, tool-call check, \
                             pp/tg speed — and the Lab's time estimates.",
                        ),
                        (
                            "2 · Placement first, for big MoE models",
                            "A Mixture-of-Experts model bigger than your VRAM settles at \
                             a crushed ~4k context by default. Run the MoE-offload trial \
                             and APPLY its winner before anything else — bench, the \
                             agent-turn trials, and quality all measure through whatever \
                             placement is applied. (The Lab warns you about this one.)",
                        ),
                        (
                            "3 · Speed menus",
                            "Speculation (and its dials, after a keep), prefill batch, \
                             KV precision. Winners are offered, never auto-applied; \
                             every option stays selectable with its numbers.",
                        ),
                        (
                            "4 · Agent-turn menus",
                            "Vision cost, cache-reuse, checkpoints — these need an \
                             agent-usable context (≥ ~24k), which is why placement came \
                             first. They price what real coding turns cost.",
                        ),
                        (
                            "5 · Quality probe",
                            "Eval battery + tool calls + multi-hop agent loops. Run it \
                             on the CONFIG you intend to serve.",
                        ),
                        (
                            "6 · Apply, re-measure, sync",
                            "Apply winners in the Lab, re-run Measure so opencode.json \
                             gets the limits of what you actually serve. Re-run \
                             affected menus after a llama.cpp rebuild — the Rebuild \
                             scorecard tells you when numbers went stale.",
                        ),
                        (
                            "Same thing, different names",
                            "Measure (GUI) = --calibrate (CLI): finding a model's real \
                             context and tool support. A trial races config variants \
                             for one model; each trial type is a menu (spec, kv, moe…); \
                             the Lab calls one queued batch of work a campaign. The \
                             preset is the router.ini file the router reads.",
                        ),
                    ] {
                        ui.strong(head);
                        ui.label(body);
                        ui.add_space(6.0);
                    }
                });
            });
        if !open {
            self.show_guide = false;
        }
    }

    fn ai_advisor_window(&mut self, ctx: &egui::Context) {
        if !self.advisor_open {
            return;
        }
        let mut open = true;
        let mut ask: Option<String> = None;
        egui::Window::new("Advisor — AI opinions, not measurements")
            .collapsible(false)
            .default_width(520.0)
            .open(&mut open)
            .show(ctx, |ui| {
                // Ask about tuning (the RAG-that-isn't, 2026-08-29):
                // grounded on the curated knob notes + THIS machine's
                // findings, answered by the earned advisor model.
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.tuning_question)
                            .desired_width(360.0)
                            .hint_text("Ask about tuning — e.g. should I use q4_0 KV?"),
                    );
                    if ui
                        .add_enabled(
                            self.busy.is_none() && !self.tuning_question.trim().is_empty(),
                            egui::Button::new("Ask"),
                        )
                        .on_disabled_hover_text("type a question — and asking waits while an operation runs")
                        .clicked()
                    {
                        ask = Some(self.tuning_question.trim().to_string());
                    }
                });
                ui.add_space(6.0);
                if self.advisories.is_empty() {
                    ui.weak(
                        "Nothing yet. When a failure stumps the rule-based Why?, \
                         its dialog offers \"Ask a served model\" — answers collect \
                         here, clearly labeled.",
                    );
                }
                egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                    for (subject, model, text) in &self.advisories {
                        ui.strong(subject);
                        ui.small(format!(
                            "answered by {model} — advisory only; verify before acting"
                        ));
                        ui.label(text);
                        ui.add_space(8.0);
                    }
                });
            });
        if !open {
            self.advisor_open = false;
        }
        if let Some(q) = ask {
            self.action_ask_tuning(&q);
        }
    }

    /// Ask-about-tuning worker: regenerate findings, ground the curated
    /// corpus + this machine's JSON, ask the earned advisor.
    fn action_ask_tuning(&mut self, question: &str) {
        let Some(answerer) = self.pick_answerer(None) else {
            self.log("tuning questions need a measured model to ask — measure one first".to_string());
            return;
        };
        let cfg = self.cfg.clone();
        let port = self.cfg.port;
        let question = question.to_string();
        self.tuning_question.clear();
        self.spawn(&format!("asking {answerer} about tuning"), move |tx| {
            let result = (|| -> anyhow::Result<String> {
                crate::core::report::generate(&cfg)?;
                let json = std::fs::read_to_string(
                    router::state_dir().join("findings-report.json"),
                )?;
                let backend = aiadvisor::RouterAdvisor {
                    port,
                    model: answerer.clone(),
                };
                use aiadvisor::Advisor as _;
                backend.ask(
                    aiadvisor::BRIEF_SYSTEM,
                    &aiadvisor::tuning_prompt(&question, &json),
                )
            })();
            let _ = tx.send(match result {
                Ok(text) => Msg::Advisory {
                    subject: format!("tuning: {question}"),
                    model: answerer,
                    text,
                },
                Err(e) => Msg::Error(format!("tuning question: {e:#}")),
            });
            let _ = tx.send(Msg::Finished("tuning answer ready".into()));
        });
    }

    fn diagnosis_window(&mut self, ctx: &egui::Context) {
        let Some(v) = &self.diagnosis else { return };
        let (display, d, router_id, path) =
            (v.display.clone(), v.d.clone(), v.router_id.clone(), v.path.clone());
        let mut open = true;
        let mut action: Option<RowAction> = None;
        let mut open_advisor = false;
        let mut show_log = false;
        let mut unload_others = false;
        let mut ask_ai = false;
        egui::Window::new(format!("Why? — {display}"))
            .collapsible(false)
            .default_width(460.0)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(&d.explanation);
                if let Some(ev) = &d.evidence {
                    ui.add_space(4.0);
                    ui.small("The exact error:");
                    ui.monospace(ev);
                }
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    for remedy in &d.remedies {
                        match remedy {
                            diagnose::Remedy::OpenBuildAdvisor => {
                                if ui.button("Open Build Advisor").clicked() {
                                    open_advisor = true;
                                }
                            }
                            diagnose::Remedy::ArchiveToShelf => {
                                if let Some(p) = &path
                                    && ui.button("Archive to my models folder").clicked()
                                {
                                    action = Some(RowAction::Archive(p.clone()));
                                }
                            }
                            diagnose::Remedy::UnloadOthers => {
                                if ui.button("Unload other models").clicked() {
                                    unload_others = true;
                                }
                            }
                            diagnose::Remedy::LoadAndMeasure => {
                                if let Some(id) = &router_id
                                    && ui.button("Load & measure now").clicked()
                                {
                                    action = Some(RowAction::Load(id.clone()));
                                }
                            }
                            diagnose::Remedy::ShowLog => {
                                if ui.button("Open the full log").clicked() {
                                    show_log = true;
                                }
                            }
                        }
                    }
                });
                // The AI layer activates where the rules gave up (design
                // decision 2026-08-27): one grounded, labeled, one-shot
                // explanation from a model the router already serves.
                if matches!(d.cause, diagnose::Cause::Unknown) {
                    ui.separator();
                    if ui
                        .add_enabled(
                            self.busy.is_none(),
                            egui::Button::new("Ask a served model (advisory)"),
                        )
                        .on_disabled_hover_text("busy — ask when the running operation finishes")
                        .on_hover_text(
                            "Sends this failure's log evidence to one of YOUR local \
                             models and shows its opinion, clearly labeled. Nothing \
                             leaves this machine; nothing is applied automatically.",
                        )
                        .clicked()
                    {
                        ask_ai = true;
                    }
                }
            });
        let acted =
            action.is_some() || open_advisor || show_log || unload_others;
        if action.is_some() || open_advisor || !open {
            self.diagnosis = None;
        }
        if open_advisor {
            self.action_build_check();
        }
        if show_log {
            let path = router::state_dir().join("router.log");
            let _ = std::process::Command::new("xdg-open").arg(&path).spawn();
            self.log(format!("opened {}", path.display()));
        }
        if ask_ai {
            self.spawn_failure_advisory(&display, &router_id, &path, &d);
        }
        if unload_others {
            let loaded: Vec<String> = match &self.router_state {
                Some(router::RouterState::Ours { models }) => models
                    .iter()
                    .filter(|m| m.status == "loaded")
                    .map(|m| m.id.clone())
                    .collect(),
                _ => Vec::new(),
            };
            let port = self.cfg.port;
            if loaded.is_empty() {
                self.log("nothing is loaded — VRAM is as free as the router can make it");
            } else {
                self.spawn("unloading all models", move |tx| {
                    for id in &loaded {
                        let _ = router::unload_model(port, id);
                    }
                    let _ = tx.send(Msg::Finished(format!("unloaded {}", loaded.join(", "))));
                });
            }
        }
        if let Some(a) = action {
            self.run_row_action(a);
        }
        let _ = acted;
    }

    fn advisor_window(&mut self, ctx: &egui::Context) {
        if !self.show_advisor {
            return;
        }
        let mut open = true;
        let mut rebuild = false;
        let mut recheck = false;
        let mut triage = false;
        let mut managed_build = false;
        let mut archive_now = false;
        egui::Window::new("Build Advisor")
            .collapsible(false)
            .default_width(560.0)
            .open(&mut open)
            .show(ctx, |ui| {
                // Which checkout is under analysis (rung 1): the active
                // binary's by default; the managed clone and any manually
                // added checkouts are selectable.
                ui.horizontal(|ui| {
                    ui.label("Analyzing:");
                    let name = |p: &Option<PathBuf>| match p {
                        None => "active binary's checkout".to_string(),
                        Some(d) => d.display().to_string(),
                    };
                    let current = name(&self.sel_checkout);
                    let mut choices: Vec<Option<PathBuf>> = vec![None];
                    if self.managed_status.as_ref().is_some_and(|m| m.present) {
                        choices.push(Some(managed::checkout_dir()));
                    }
                    // Every DISCOVERED install's checkout too (design
                    // session 2026-08-30: rebuilding your own
                    // ~/src/llama.cpp shouldn't require adding it by
                    // hand when the scan already knows it).
                    if let Some(scan) = &self.scan {
                        for i in &scan.installs {
                            if let Some(repo) = advisor::repo_of(&i.server_path)
                                && !choices.iter().any(|c| c.as_deref() == Some(&*repo))
                            {
                                choices.push(Some(repo));
                            }
                        }
                    }
                    for c in &self.cfg.checkouts {
                        if !choices.iter().any(|x| x.as_ref() == Some(c)) {
                            choices.push(Some(c.clone()));
                        }
                    }
                    egui::ComboBox::from_id_salt("advisor-checkout")
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for choice in choices {
                                let label = name(&choice);
                                if ui
                                    .selectable_label(self.sel_checkout == choice, label)
                                    .clicked()
                                    && self.sel_checkout != choice
                                {
                                    self.sel_checkout = choice;
                                    recheck = true;
                                }
                            }
                        });
                    if ui
                        .small_button("+ add checkout…")
                        .on_hover_text(
                            "Register another llama.cpp checkout (e.g. one you build \
                             with custom options) — analyzable and rebuildable here, \
                             archivable below.",
                        )
                        .clicked()
                        && let Some(dir) = rfd::FileDialog::new().pick_folder()
                    {
                        if dir.join(".git").exists() {
                            self.cfg.checkouts.push(dir.clone());
                            let _ = self.cfg.save(&system::config_file());
                            self.sel_checkout = Some(dir);
                            recheck = true;
                        } else {
                            self.log(format!(
                                "not a git checkout: {} (no .git)",
                                dir.display()
                            ));
                        }
                    }
                });
                if let Some(repo) = self.build_check.as_ref().and_then(|c| c.repo.as_ref()) {
                    // The confusion this answers (2026-08-30): the active
                    // binary can be an ARCHIVE whose analysis resolves to
                    // the managed checkout — say where a rebuild would run.
                    ui.small(format!(
                        "checkout under analysis: {} — Update & Rebuild builds there",
                        repo.display()
                    ));
                }
                ui.separator();
                let Some(check) = &self.build_check else {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("checking your llama.cpp against upstream…");
                    });
                    return;
                };
                for (headline, detail) in advisor::verdicts(check) {
                    ui.strong(headline);
                    ui.label(detail);
                    ui.add_space(6.0);
                }
                let mut sel = self
                    .backend_sel
                    .unwrap_or_else(|| advisor::default_backends(check));
                ui.separator();
                ui.strong("Build with:");
                ui.horizontal(|ui| {
                    ui.add_enabled(
                        check.nvcc.is_some(),
                        egui::Checkbox::new(&mut sel.cuda, "CUDA (NVIDIA)"),
                    )
                    .on_disabled_hover_text("needs the CUDA toolkit (nvcc) installed")
                    .on_hover_text(match &check.nvcc {
                        Some(v) => format!("nvcc: {v}"),
                        None => "needs the CUDA toolkit (nvcc)".into(),
                    });
                    ui.add_enabled(
                        check.glslc,
                        egui::Checkbox::new(&mut sel.vulkan, "Vulkan (any GPU)"),
                    )
                    .on_disabled_hover_text("needs the Vulkan SDK / shaderc (glslc) installed")
                    .on_hover_text(if check.glslc {
                        "glslc present — works on NVIDIA, AMD, and Intel".to_string()
                    } else {
                        "needs the Vulkan SDK / shaderc (glslc)".to_string()
                    });
                    ui.add_enabled(
                        check.hipcc.is_some(),
                        egui::Checkbox::new(&mut sel.hip, "ROCm (AMD)"),
                    )
                    .on_disabled_hover_text("needs ROCm (hipcc) installed")
                    .on_hover_text(match (&check.hipcc, &check.rocm_gfx) {
                        (Some(v), Some(gfx)) => format!("{v} — target {gfx}"),
                        (Some(v), None) => format!("{v} — no gfx target detected"),
                        _ => "needs ROCm (hipcc)".into(),
                    });
                });
                self.backend_sel = Some(sel);
                egui::CollapsingHeader::new("Advanced: exactly what a rebuild runs")
                    .default_open(false)
                    .show(ui, |ui| {
                        for (cmd, args) in advisor::rebuild_commands(check, sel) {
                            ui.monospace(format!("{cmd} {}", args.join(" ")));
                        }
                        ui.small("Fast-forward pull only — your local changes are never overwritten. Backends not selected are set OFF explicitly so stale cmake caches can't resurrect them.");
                    });
                ui.separator();
                // Tag-pinned = detached HEAD, wherever the checkout lives
                // (a user-added pinned checkout hits the same pull wall).
                let tag_pinned = check.detached == Some(true)
                    || check.repo.as_deref() == Some(managed::checkout_dir().as_path());
                if tag_pinned {
                    ui.small(
                        "This checkout is pinned to an exact commit (detached HEAD) — \
                         branch-style Update & Rebuild doesn't apply. For the managed \
                         checkout, use \"Fetch + build newest release\" below.",
                    );
                }
                ui.horizontal(|ui| {
                    let can_rebuild = !tag_pinned
                        && check.repo.is_some()
                        && check.cmake
                        && check.dirty != Some(true);
                    if ui
                        .add_enabled(can_rebuild, egui::Button::new("⏶ Update & Rebuild Now"))
                        .on_disabled_hover_text("needs a git checkout + cmake, a clean tree, and a branch (tag-pinned checkouts use the managed flow below)")
                        .on_hover_text(
                            "git pull --ff-only, then a full cmake build. Takes many minutes; \
                             progress streams to the activity log. Your current binaries keep \
                             working until the build succeeds.",
                        )
                        .clicked()
                    {
                        rebuild = true;
                    }
                    if ui.button("Re-check").clicked() {
                        recheck = true;
                    }
                });
                if check.dirty == Some(true) {
                    ui.small("Rebuild disabled: the checkout has local changes — commit/stash them first.");
                }
                // Rung 2: archive the ANALYZED checkout's current
                // build under a label — custom builds become
                // pinnable next to release archives. Rung 3 caveat
                // lives on the hover.
                if check.repo.is_some() {
                    ui.horizontal(|ui| {
                        ui.label(format!(
                    "Archive the analyzed checkout's build{} as:",
                    check
                        .current_build
                        .map(|b| format!(" (b{b})"))
                        .unwrap_or_default()
                ));
                        ui.add(
                            egui::TextEdit::singleline(&mut self.archive_label)
                                .desired_width(160.0)
                                .hint_text(
                                    // From the cached scan — never
                                    // probe binaries in a render loop.
                                    check
                                        .server_bin
                                        .as_deref()
                                        .and_then(|p| {
                                            self.scan.as_ref()?.installs
                                                .iter()
                                                .find(|i| i.server_path == p)
                                                .map(discover::install_alias)
                                        })
                                        .or_else(|| {
                                            check
                                                .current_build
                                                .map(|b| format!("b{b}-variant"))
                                        })
                                        .unwrap_or_else(|| "label".into()),
                                ),
                        );
                        if ui
                            .add_enabled(
                                self.busy.is_none()
                                    && !self.archive_label.trim().is_empty(),
                                egui::Button::new("Archive"),
                            )
                            .on_disabled_hover_text("type a label first — and archiving waits while an operation runs")
                            .on_hover_text(
                                "Snapshots the analyzed checkout's build/bin into \
                                 the pinnable archive set. CAVEAT: measurements key \
                                 builds by NUMBER — two variants of the same build \
                                 look identical to bench/history/scorecard (schema \
                                 work parked on the roadmap).",
                            )
                            .clicked()
                        {
                            archive_now = true;
                        }
                    });
                }

                if let (Some(cur), Some(up)) = (check.current_build, check.upstream_build)
                    && up > cur
                    && ui
                        .add_enabled(
                            self.busy.is_none(),
                            egui::Button::new("What's in this update for me? (advisory)"),
                        )
                        .on_disabled_hover_text("busy — triage when the running operation finishes")
                        .on_hover_text(
                            "Asks one of YOUR local models to read the commits between \
                             your build and upstream and say whether anything matters \
                             for the models you serve. Labeled opinion; nothing leaves \
                             this machine.",
                        )
                        .clicked()
                {
                    triage = true;
                }
                ui.separator();
                ui.strong("Managed llama.cpp");
                ui.small(
                    "An app-owned checkout in its own data dir — your checkout is \
                     never touched; builds are archived per release so rolling back \
                     is a pin, never a rebuild. Offered, never forced.",
                );
                match &self.managed_status {
                    None => {
                        ui.weak("status loads with the build check…");
                    }
                    Some(ms) => {
                        if !ms.present {
                            ui.label("No managed checkout yet.");
                        } else if let Some(b) = ms.built {
                            ui.label(format!("Checkout present; built b{b}."));
                        } else {
                            ui.label("Checkout present; not built yet.");
                        }
                        if ui
                            .add_enabled(
                                self.busy.is_none() && check.cmake,
                                egui::Button::new(if ms.present {
                                    "⏷ Fetch + build newest release"
                                } else {
                                    "⏷ Set up (clone + build newest release)"
                                }),
                            )
                            .on_disabled_hover_text("needs cmake installed — and waits while another operation runs")
                            .on_hover_text(
                                "Clones llama.cpp if needed, checks out the newest \
                                 bNNNN release tag, builds with the backends selected \
                                 above, and archives the binaries. Takes many minutes; \
                                 narrated in the activity log.",
                            )
                            .clicked()
                        {
                            managed_build = true;
                        }
                        if !ms.archives.is_empty() {
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label("Archived builds — keep newest");
                                let mut keep = self.cfg.archives_keep;
                                if ui
                                    .add(egui::DragValue::new(&mut keep).range(0..=99))
                                    .on_hover_text(
                                        "Older release archives are pruned after each \
                                         successful build. 0 = keep everything. The \
                                         serving archive and custom-named archives are \
                                         never pruned.",
                                    )
                                    .changed()
                                {
                                    self.cfg.archives_keep = keep;
                                    let _ = self.cfg.save(&system::config_file());
                                }
                                ui.label("(0 = all)");
                            });
                            let serving = self.picked_server();
                            let mut doomed: Option<String> = None;
                            egui::ScrollArea::vertical()
                                .id_salt("archives")
                                .max_height(140.0)
                                .show(ui, |ui| {
                                    for a in &ms.archives {
                                        ui.horizontal(|ui| {
                                            let is_serving =
                                                serving.as_deref() == Some(a.server.as_path());
                                            ui.monospace(&a.label);
                                            if is_serving {
                                                ui.weak("(serving — protected)");
                                            } else if ui
                                                .small_button("✖")
                                                .on_hover_text(
                                                    "Delete this archived build permanently \
                                                     (frees its disk; rebuilding it later \
                                                     means a full compile)",
                                                )
                                                .clicked()
                                            {
                                                doomed = Some(a.label.clone());
                                            }
                                        });
                                    }
                                });
                            if let Some(label) = doomed {
                                match managed::delete_archive(&label) {
                                    Ok(()) => {
                                        self.log(format!("archive {label} deleted"));
                                        self.managed_status = Some(managed::status());
                                    }
                                    Err(e) => self.log(format!("ERROR deleting {label}: {e:#}")),
                                }
                            }
                            ui.small(
                                "This window MAKES builds; choosing what serves \
                                 happens in Settings -> llama-server binary.",
                            );
                        }
                    }
                }
            });
        if !open {
            self.show_advisor = false;
        }
        if rebuild {
            self.action_rebuild();
        }
        if recheck {
            self.action_build_check();
        }
        if archive_now
            && let Some(repo) = self.build_check.as_ref().and_then(|c| c.repo.clone())
        {
            let label = self.archive_label.trim().to_string();
            self.spawn(&format!("archiving build as {label}"), move |tx| {
                let tx2 = tx.clone();
                let mut progress = move |line: String| {
                    let _ = tx2.send(Msg::Progress(line));
                };
                let result =
                    managed::archive_from(&repo.join("build/bin"), &label, &mut progress);
                let _ = tx.send(Msg::Managed(managed::status()));
                let _ = tx.send(match result {
                    Ok(p) => Msg::Finished(format!("archived to {}", p.display())),
                    Err(e) => Msg::Error(format!("archive: {e:#}")),
                });
            });
        }
        if triage {
            self.action_rebuild_triage();
        }
        if managed_build {
            self.action_managed_build();
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let (dot, text) = match &self.router_state {
                Some(router::RouterState::Ours { models }) => {
                    // Loading/downloading models are the activity users
                    // most often mistake for a hang — say so.
                    let active: Vec<String> = models
                        .iter()
                        .filter_map(|m| match m.status.as_str() {
                            "loaded" => Some(m.id.clone()),
                            "loading" => Some(format!("{} (loading…)", m.id)),
                            "downloading" => Some(format!("{} (downloading…)", m.id)),
                            _ => None,
                        })
                        .collect();
                    let mut text = if active.is_empty() {
                        "router: up (nothing loaded)".to_string()
                    } else {
                        format!("router: up — {}", active.join(", "))
                    };
                    if let Some(a) = &self.live_activity {
                        text.push_str(&format!(" · {a}"));
                    }
                    (egui::Color32::from_rgb(0, 170, 0), text)
                }
                Some(router::RouterState::External { .. }) => {
                    (egui::Color32::from_rgb(220, 150, 0), "external server".into())
                }
                Some(router::RouterState::Trouble { .. }) => {
                    (egui::Color32::from_rgb(220, 150, 0), "trouble".into())
                }
                Some(router::RouterState::Down) => (egui::Color32::GRAY, "router: down".into()),
                None => (egui::Color32::GRAY, "checking…".into()),
            };
            ui.colored_label(dot, "⏺");
            ui.label(text);
            ui.separator();
            if let Some((free, total)) = self.live_vram {
                // used/total, matching nvidia-smi's orientation — the
                // old "{free} / {total} MiB free" read as a used/total
                // meter and looked wildly wrong (user report 2026-08-30;
                // the NUMBER was correct, the label lied).
                ui.label(format!(
                    "VRAM: {} / {total} MiB used · {free} free",
                    total.saturating_sub(free)
                ))
                .on_hover_text("Live from nvidia-smi, whole card (all processes), ~2s refresh.");
                ui.separator();
            }
            if let Some(scan) = &self.scan {
                if self.live_vram.is_none()
                    && let Some(d) = scan.devices.iter().find(|d| d.id.starts_with("CUDA"))
                {
                    ui.label(format!("{}: {} MiB free (at scan)", d.id, d.free_mib));
                    ui.separator();
                }
                ui.label(format!("{} models", scan.models.len()));
                ui.separator();
            }
            if self.vram_contention().is_some() {
                ui.colored_label(ui.visuals().warn_fg_color, "⚠ VRAM contention");
                ui.separator();
            }
            if let Some(b) = &self.busy {
                ui.spinner();
                ui.label(b);
                if self.cancel_token.is_cancelled() {
                    ui.weak("cancelling — stopping at the next safe point…");
                } else if ui
                    .small_button("✖ Cancel")
                    .on_hover_text(
                        "Stops the running measurement at its next safe boundary \
                         (between rounds/models/items) — cleanup still runs, partial \
                         results are kept, and your config is restored.",
                    )
                    .clicked()
                {
                    self.cancel_token.cancel();
                }
            }
        });
    }
}

// ─── worker bodies ───────────────────────────────────────────────────────────

fn start_router(cfg: &settings::AppConfig) -> anyhow::Result<u32> {
    let rcfg = system::router_config(cfg);
    // Plain words beat a 30s timeout (usability review D4).
    if let Err(e) = router::router_mode_supported(discover::build_of(&rcfg.server_bin)) {
        anyhow::bail!(e);
    }
    if !system::preset_path().exists() {
        system::write_preset(cfg, &[])?;
    }
    router::start(&router::state_dir(), &rcfg)
}

fn run_calibration(
    cfg: &settings::AppConfig,
    force: bool,
    tx: &Sender<Msg>,
) -> anyhow::Result<router::Measurements> {
    let report = system::scan_report(cfg, &[]);
    let env_fp = system::env_fingerprint(&report);
    let build = system::env_build(&report);
    // A newly downloaded model isn't servable until the preset knows it —
    // refresh + hot-reload first, so "measure new" includes disk-new files
    // (user-found: a fresh download measured as "nothing to do").
    let (_, n) = system::write_preset(cfg, &[])?;
    let _ = tx.send(Msg::Progress(format!(
        "preset refreshed ({n} models); reloading router"
    )));
    if let Err(e) = router::reload(cfg.port) {
        let _ = tx.send(Msg::Progress(format!(
            "router reload failed ({e:#}) — measuring what it currently offers"
        )));
    }
    let embed = router::embedding_ids_in_preset(&system::preset_path());
    let progress_tx = tx.clone();
    router::calibrate(
        &router::state_dir(),
        cfg.port,
        &env_fp,
        build,
        force,
        &embed,
        &mut |line| {
            let _ = progress_tx.send(Msg::Progress(line));
        },
    )
}

/// Sync every connected agent: opencode.json (the original) plus each
/// detected peer agent (Connections p2, 2026-08-30). The second return
/// is one human line per agent that was actually present.
fn run_sync(
    cfg: &settings::AppConfig,
    measurements: &router::Measurements,
) -> anyhow::Result<(opencode::SyncReport, Vec<String>)> {
    let embed = router::embedding_ids_in_preset(&system::preset_path());
    let vision = router::vision_ids_in_preset(&system::preset_path());
    let desired: Vec<_> = measurements
        .iter()
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
        .collect();
    anyhow::ensure!(
        !desired.is_empty(),
        "no successful measurements yet — measure a model first (measured, not guessed)"
    );
    let path = opencode::default_config_path();
    let mut report = opencode::sync_file(
        &path,
        &format!("http://127.0.0.1:{}/v1", cfg.port),
        &desired,
    )?;
    // Ghost cleanup (user decision 2026-08-26): only against a LIVE router.
    if let router::RouterState::Ours { models } =
        router::status(&router::state_dir(), &system::router_config(cfg))
    {
        let offered: Vec<String> = models.into_iter().map(|m| m.id).collect();
        let ghosts =
            opencode::comment_out_ghosts(&path, &report.orphans, &offered, measurements)?;
        report.orphans.retain(|id| !ghosts.contains(id));
        report.ghosts_commented = ghosts;
    }
    // pi coding agent (Connections p2): measured contexts into
    // ~/.pi/agent/models.json — pi's native router integration assumes
    // 128k when the router doesn't say (measured 2026-08-30).
    let base_url = format!("http://127.0.0.1:{}/v1", cfg.port);
    let mut lines = Vec::new();
    let pi_path = piagent::default_models_path();
    match piagent::sync_file(&pi_path, &base_url, &desired) {
        Ok(r) if r.skipped_missing => {}
        Ok(r) => lines.push(format!(
            "pi agent synced ({}): {} added, {} updated, {} removed{}",
            pi_path.display(),
            r.added.len(),
            r.updated.len(),
            r.removed.len(),
            if r.created_file { " — models.json created" } else { "" },
        )),
        Err(e) => lines.push(format!("pi agent sync FAILED: {e:#}")),
    }
    // Hermes: measured contexts into its context cache. Registration of
    // the provider itself stays an explicit click (Connections).
    let home = hermes::default_home();
    match hermes::sync(&home, &base_url, &desired) {
        Ok(r) if r.skipped_missing => {}
        Ok(r) => {
            let mut l = format!("Hermes synced: {} context(s) written", r.written.len());
            if !r.below_minimum.is_empty() {
                l.push_str(&format!(
                    " — {} skipped below Hermes's 64k minimum ({})",
                    r.below_minimum.len(),
                    r.below_minimum.join(", ")
                ));
            }
            if r.provider_unregistered {
                l.push_str(
                    " · no Hermes provider points at this router yet — \
                     register it on the Connections tab",
                );
            }
            lines.push(l);
        }
        Err(e) => lines.push(format!("Hermes sync FAILED: {e:#}")),
    }
    Ok((report, lines))
}

/// Sync exactly one measured model into opencode.json.
fn sync_single(
    cfg: &settings::AppConfig,
    measurements: &router::Measurements,
    id: &str,
) -> anyhow::Result<()> {
    let m = measurements
        .get(id)
        .filter(|m| m.n_ctx.is_some())
        .ok_or_else(|| anyhow::anyhow!("{id} has no successful measurement"))?;
    let desired = opencode::DesiredModel {
        id: id.to_string(),
        display_name: format!("{id} (llama.cpp)"),
        context: m.n_ctx.unwrap(),
        tool_call: m.tool_call,
        vision: router::vision_ids_in_preset(&system::preset_path()).contains(id),
    };
    opencode::sync_file(
        &opencode::default_config_path(),
        &format!("http://127.0.0.1:{}/v1", cfg.port),
        &[desired],
    )?;
    Ok(())
}

/// The measured trial table, shared by the verdict dialog and the Lab.
fn trial_table_grid(
    ui: &mut egui::Ui,
    salt: &str,
    table: &std::collections::BTreeMap<String, trial::TrialResult>,
    pending_winners: &std::collections::HashSet<String>,
    applied_winners: &std::collections::HashSet<String>,
) {
    egui::Grid::new(salt).striped(true).show(ui, |ui| {
        for h in [
            "", "novel t/s", "rewrite t/s", "prefill t/s", "context", "load s",
            "2nd-turn ms", "J/tok", "fidelity", "accepted",
        ] {
            // Review G10: the one grid where a novice most needs
            // definitions had none. Same vocabulary as the CLI glossary.
            ui.strong(h).on_hover_text(match h {
                "novel t/s" => "Generation speed writing brand-new code, tokens/second — speculation's worst case (the did-anything-get-hurt check).",
                "rewrite t/s" => "Generation speed regenerating code the model was given — edits, refactors, diffs: most of what a coding agent does.",
                "prefill t/s" => "How fast it reads your prompt before the first token, tokens/second.",
                "context" => "The context window memory-fitting settled on under this config, in tokens.",
                "load s" => "Wall seconds from load request to ready — the hot-swap cost when the router switches models.",
                "2nd-turn ms" => "The agent-turn probe: a big prompt sent, then re-sent with a middle edit; this is the second turn's prefill time.",
                "J/tok" => "Marginal joules per generated token (GPU via NVML, CPU via RAPL when readable, idle draw subtracted) — the energy half of a token's cost.",
                "fidelity" => "The quality gate: how much of a module the model was told to preserve came back verbatim. A drop means the config degrades output — no speed win survives that.",
                "accepted" => "How many speculated tokens the model confirmed — higher acceptance = speculation is paying off.",
                _ => "★ green = this menu's winner, waiting for your Apply. ✔ blue = a winner already applied and serving.",
            });
        }
        ui.end_row();
        let fmt = |v: Option<f64>| v.map(|x| format!("{x:.1}")).unwrap_or_else(|| "—".into());
        for (label, r) in table {
            // Color = ACTION STATE, not ranking (user feedback
            // 2026-08-28: two same-green winners of different menus read
            // as competing choices): green ★ needs your Apply; blue ✔ is
            // a winner already serving.
            if pending_winners.contains(label.as_str()) {
                ui.colored_label(egui::Color32::from_rgb(0, 170, 0), format!("★ {label}"));
            } else if applied_winners.contains(label.as_str()) {
                ui.colored_label(
                    egui::Color32::from_rgb(100, 155, 255),
                    format!("✔ {label}"),
                );
            } else {
                ui.label(label);
            }
            if let Some(e) = &r.error {
                ui.label(format!("failed: {e}"));
                ui.end_row();
                continue;
            }
            ui.label(fmt(r.tg_novel));
            ui.label(fmt(r.tg_rewrite));
            ui.label(fmt(r.pp_prefill));
            ui.label(
                r.settled_ctx
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "—".into()),
            );
            ui.label(
                r.load_secs
                    .map(|l| format!("{l:.1}"))
                    .unwrap_or_else(|| "—".into()),
            );
            ui.label(
                r.turn2_prompt_ms
                    .map(|m| {
                        let reuse = r
                            .turn2_reuse
                            .map(|f| format!(" ({:.0}% cached)", f * 100.0))
                            .unwrap_or_default();
                        format!("{m:.0}{reuse}")
                    })
                    .unwrap_or_else(|| "—".into()),
            );
            ui.label(
                r.j_per_token
                    .map(|j| format!("{j:.1}"))
                    .unwrap_or_else(|| "—".into()),
            );
            ui.label(
                r.fidelity
                    .map(|f| format!("{:.0}%", f * 100.0))
                    .unwrap_or_else(|| "—".into()),
            );
            ui.label(
                r.accept_rewrite
                    .map(|a| format!("{:.0}%", a * 100.0))
                    .unwrap_or_else(|| "—".into()),
            );
            ui.end_row();
        }
    });
}

/// The Measure New/Stale worker: calibrate + report, off-thread.
fn calibrate_worker(cfg: &settings::AppConfig, force: bool, tx: &Sender<Msg>) {
    match run_calibration(cfg, force, tx) {
        Ok(m) => {
            let measured = m.values().filter(|x| x.n_ctx.is_some()).count();
            let failed = m.values().filter(|x| x.error.is_some()).count();
            let _ = tx.send(Msg::Measurements(m));
            let _ = tx.send(Msg::Finished(format!(
                "measuring finished: {measured} measured, {failed} known failures"
            )));
        }
        Err(e) => {
            let _ = tx.send(Msg::Error(format!("calibrate: {e:#}")));
        }
    }
}

/// One trial-menu run for one model, off-thread.
fn trial_worker(
    cfg: &settings::AppConfig,
    id: &str,
    menu: &str,
    cancel_token: &cancel::CancelToken,
    tx: &Sender<Msg>,
) {
    let Some((variants, goal)) = trial::menu(menu) else {
        let _ = tx.send(Msg::Error(format!("unknown trial menu {menu:?}")));
        return;
    };
    let tx2 = tx.clone();
    let mut progress = move |line: String| {
        let _ = tx2.send(Msg::Progress(line));
    };
    let _ = match trial::run_trial(cfg, id, menu, &variants, goal, cancel_token, &mut progress) {
        Ok(report) => tx.send(Msg::TrialDone {
            model: id.to_string(),
            menu: menu.to_string(),
            report,
        }),
        Err(e) => tx.send(Msg::Error(format!("trial: {e:#}"))),
    };
}

/// Start the router and wait for it to answer. Narrates via tx; on
/// failure it reports (Msg::Error) and returns false.
fn start_router_and_wait(cfg: &settings::AppConfig, tx: &Sender<Msg>) -> bool {
    let _ = tx.send(Msg::Progress("starting router".into()));
    if let Err(e) = start_router(cfg) {
        let _ = tx.send(Msg::Error(format!("start: {e:#}")));
        return false;
    }
    let rcfg = system::router_config(cfg);
    let dir = router::state_dir();
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_secs(1));
        if matches!(router::status(&dir, &rcfg), router::RouterState::Ours { .. }) {
            let _ = tx.send(Msg::Progress("router is up".into()));
            return true;
        }
    }
    let _ = tx.send(Msg::Error(
        "router did not come up within 30s — see router.log".into(),
    ));
    false
}

/// The setup sequence shared by Set Up Everything and the post-rebuild
/// verification: start if down -> wait healthy -> incremental calibrate ->
/// sync. Narrates via tx; on failure it reports (Msg::Error) and returns
/// false so callers stop there.
fn setup_flow(cfg: &settings::AppConfig, tx: &Sender<Msg>) -> bool {
    let step = |m: &str| {
        let _ = tx.send(Msg::Progress(format!("setup: {m}")));
    };
    let rcfg = system::router_config(cfg);
    let dir = router::state_dir();
    match router::status(&dir, &rcfg) {
        router::RouterState::Ours { .. } => step("router already running"),
        router::RouterState::Down => {
            if !start_router_and_wait(cfg, tx) {
                return false;
            }
        }
        other => {
            let _ = tx.send(Msg::Error(format!(
                "setup: port {} is not ours: {other}",
                cfg.port
            )));
            return false;
        }
    }
    step("measuring anything new or stale");
    let measurements = match run_calibration(cfg, false, tx) {
        Ok(m) => m,
        Err(e) => {
            let _ = tx.send(Msg::Error(format!("setup/calibrate: {e:#}")));
            return false;
        }
    };
    let _ = tx.send(Msg::Measurements(measurements.clone()));
    step("syncing opencode.json");
    match run_sync(cfg, &measurements) {
        Ok((report, lines)) => {
            let _ = tx.send(Msg::SyncDone(report));
            for l in lines {
                let _ = tx.send(Msg::Progress(l));
            }
            send_configured(tx);
            let _ = tx.send(Msg::Progress(
                "setup complete — OpenCode is ready to use these models".into(),
            ));
            true
        }
        Err(e) => {
            let _ = tx.send(Msg::Error(format!("setup/sync: {e:#}")));
            false
        }
    }
}

fn send_configured(tx: &Sender<Msg>) {
    let configured =
        opencode::configured_models(&opencode::default_config_path()).unwrap_or_default();
    let _ = tx.send(Msg::Configured(configured));
}

/// Load (if needed) + measure + record + add to opencode.json; optionally
/// leave the model warm. The Library's Load button and checkbox both land
/// here — loading IS measuring, so nothing enters the config unguessed.
/// `finish` = end the busy-flow with Msg::Finished; pass false when this
/// runs as one step of a longer sequence (the Lab). Returns success.
fn measure_and_sync(
    cfg: &settings::AppConfig,
    id: &str,
    keep_loaded: bool,
    finish: bool,
    tx: &Sender<Msg>,
) -> bool {
    let dir = router::state_dir();
    let report = system::scan_report(cfg, &[]);
    let env_fp = system::env_fingerprint(&report);
    let build = system::env_build(&report);
    let args_fp = router::status(&dir, &system::router_config(cfg));
    let args_fp = match args_fp {
        router::RouterState::Ours { models } => models
            .iter()
            .find(|m| m.id == id)
            .and_then(|m| m.args_fp.clone()),
        _ => None,
    };
    match router::fetch_settled_ctx(cfg.port, id) {
        Ok(ctx) => {
            let _ = tx.send(Msg::Progress(format!(
                "{id}: ctx {ctx}; probing tool calls (thinking models take a minute)…"
            )));
            let tool_call = match router::probe_tool_call(cfg.port, id) {
                Ok(v) => Some(v),
                Err(e) => {
                    let _ = tx.send(Msg::Progress(format!(
                        "{id}: tool probe inconclusive: {e:#}"
                    )));
                    None
                }
            };
            let mut all = router::read_measurements(&dir);
            let _ = crate::core::history::record(
                &dir,
                &crate::core::history::Entry {
                    when: advisor::now_epoch(),
                    model: id.to_string(),
                    build,
                    args_fp: args_fp.clone(),
                    n_ctx: Some(ctx),
                    ..Default::default()
                },
            );
            router::upsert_measurement(
                &mut all,
                id,
                router::Measurement {
                    n_ctx: Some(ctx),
                    tool_call,
                    error: None,
                    args_fp,
                    env_fp: Some(env_fp),
                    ..Default::default()
                },
            );
            let _ = router::write_measurements(&dir, &all);
            let _ = tx.send(Msg::Measurements(all.clone()));
            let ok = match sync_single(cfg, &all, id) {
                Ok(()) => {
                    send_configured(tx);
                    let tools = match tool_call {
                        Some(true) => ", tool calls work",
                        Some(false) => ", tool calls NOT produced",
                        None => "",
                    };
                    let line = format!(
                        "{id}: measured {ctx} context{tools}, added to OpenCode{}",
                        if keep_loaded { ", still loaded" } else { "" }
                    );
                    let _ = tx.send(if finish {
                        Msg::Finished(line)
                    } else {
                        Msg::Progress(line)
                    });
                    true
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{id} measured but sync failed: {e:#}")));
                    false
                }
            };
            if !keep_loaded {
                let _ = router::unload_model(cfg.port, id);
                router::wait_until_not_loaded(cfg.port, id, std::time::Duration::from_secs(30));
            }
            ok
        }
        Err(e) => {
            let detail = match router::mine_load_error(&dir) {
                Some(cause) => format!("{e:#} — {cause}"),
                None => format!("{e:#}"),
            };
            let mut all = router::read_measurements(&dir);
            let _ = crate::core::history::record(
                &dir,
                &crate::core::history::Entry {
                    when: advisor::now_epoch(),
                    model: id.to_string(),
                    build,
                    args_fp: args_fp.clone(),
                    error: Some(detail.clone()),
                    ..Default::default()
                },
            );
            router::upsert_measurement(
                &mut all,
                id,
                router::Measurement {
                    n_ctx: None,
                    tool_call: None,
                    error: Some(detail.clone()),
                    args_fp,
                    env_fp: Some(env_fp),
                    ..Default::default()
                },
            );
            let _ = router::write_measurements(&dir, &all);
            let _ = tx.send(Msg::Measurements(all));
            let _ = tx.send(Msg::Error(format!("{id}: {detail}")));
            false
        }
    }
}

/// Last few KB of the router log, for the Server pane.
fn tail_of_log() -> String {
    let path = router::state_dir().join("router.log");
    let Ok(meta) = std::fs::metadata(&path) else {
        return "(no log yet)".into();
    };
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(&path) else {
        return "(log unreadable)".into();
    };
    let start = meta.len().saturating_sub(16 * 1024);
    let _ = f.seek(SeekFrom::Start(start));
    let mut s = String::new();
    let _ = f.read_to_string(&mut s);
    s.lines().skip(1).collect::<Vec<_>>().join("\n")
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_messages();

        egui::Panel::top("menu").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| self.menu_bar(ui));
        });
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
        // The error banner (usability review G1): failures used to land
        // ONLY in the 80px log — a failed sync looked like a stopped
        // spinner. Persists until dismissed.
        if self.last_error.is_some() {
            egui::Panel::bottom("error-banner").show(ui, |ui| {
                ui.horizontal(|ui| {
                    let msg = self.last_error.clone().unwrap_or_default();
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 60, 60),
                        format!("⚠ {msg}"),
                    );
                    if ui.small_button("✖").clicked() {
                        self.last_error = None;
                    }
                });
            });
        }
        egui::Panel::bottom("activity")
            .resizable(true)
            .default_size(80.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.activity {
                            // Errors must not render like progress
                            // chatter (usability review G1).
                            if line.contains("ERROR") {
                                ui.colored_label(
                                    egui::Color32::from_rgb(220, 60, 60),
                                    egui::RichText::new(line).monospace(),
                                );
                            } else {
                                ui.monospace(line);
                            }
                        }
                    });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.pane, Pane::Library, "📚 Library");
                ui.selectable_value(&mut self.pane, Pane::Server, "🖥 Server");
                ui.selectable_value(&mut self.pane, Pane::Lab, "⚡ Lab");
                ui.selectable_value(&mut self.pane, Pane::Connections, "🔌 Connections");
                ui.selectable_value(&mut self.pane, Pane::Settings, "🔧 Settings");
            });
            ui.separator();
            match self.pane {
                Pane::Library => self.library_pane(ui),
                Pane::Server => self.server_pane(ui),
                Pane::Lab => self.lab_pane(ui),
                Pane::Connections => self.connections_pane(ui),
                Pane::Settings => self.settings_pane(ui),
            }
        });

        self.override_dialog(ui.ctx());
        self.diagnosis_window(ui.ctx());
        // One sync point for the poller's view of app business.
        self.busy_flag.store(
            self.busy.is_some(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.ai_advisor_window(ui.ctx());
        self.persistence_window(ui.ctx());
        self.disrupt_window(ui.ctx());
        self.meter_window(ui.ctx());
        self.guide_window(ui.ctx());
        self.advisor_window(ui.ctx());

        if self.show_about {
            egui::Window::new("About")
                .collapsible(false)
                .resizable(false)
                .open(&mut self.show_about)
                .show(ui.ctx(), |ui| {
                    let id = system::build_id();
                    ui.horizontal(|ui| {
                        ui.strong(format!("modelsteward {id}"));
                        if ui
                            .small_button("copy")
                            .on_hover_text(
                                "Copies the build id — paste it into a bug report \
                                 so we can dial into this exact build.",
                            )
                            .clicked()
                        {
                            ui.ctx().copy_text(id.clone());
                        }
                    });
                    ui.label("Manages llama.cpp (router mode) + OpenCode config.");
                    ui.label("Measured, not guessed.");
                    ui.add_space(6.0);
                    ui.hyperlink_to(
                        "Project home (GitHub)",
                        "https://github.com/unclebucklarson/modelsteward",
                    );
                    ui.hyperlink_to(
                        "User's guide — the first hour",
                        "https://github.com/unclebucklarson/modelsteward/blob/main/docs/GUIDE.md",
                    );
                    ui.hyperlink_to(
                        "Releases & changelog",
                        "https://github.com/unclebucklarson/modelsteward/releases",
                    );
                    // Pre-fill the issue with the build id: the whole
                    // point of carrying one (user request 2026-08-29).
                    let enc: String = id
                        .bytes()
                        .map(|b| match b {
                            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'.' | b'-' => {
                                (b as char).to_string()
                            }
                            _ => format!("%{b:02X}"),
                        })
                        .collect();
                    ui.hyperlink_to(
                        "Report a bug or send feedback",
                        format!(
                            "https://github.com/unclebucklarson/modelsteward/issues/new?body=Build%3A%20{enc}"
                        ),
                    );
                    ui.add_space(6.0);
                    ui.weak("Dual-licensed MIT OR Apache-2.0.");
                });
        }

        if let Some(action) = &self.start_prompt {
            let desc = action.describe();
            let mut go = false;
            let mut cancel = false;
            egui::Window::new("Router isn't running")
                .collapsible(false)
                .resizable(false)
                .show(ui.ctx(), |ui| {
                    ui.label(
                        "This needs the router — models load and measure through it — \
                         but it isn't running.",
                    );
                    ui.label(format!("Start it, then {desc}?"));
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button("⏵ Start Router & Continue").clicked() {
                            go = true;
                        }
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }
                    });
                });
            if go {
                if let Some(a) = self.start_prompt.take() {
                    self.dispatch(a, true);
                }
            } else if cancel {
                self.start_prompt = None;
            }
        }
    }
}

#[cfg(test)]
mod tofu_tests {
    /// Every non-ASCII character in a GUI-reachable string literal must
    /// have a glyph in egui's DEFAULT fonts — anything else renders as
    /// a tofu box (live complaint 2026-08-30; 🧪 and -> hit this in
    /// 2026-08-25 already). Scans ui.rs + core (core strings surface in
    /// the GUI via logs, advice, and the meter window); main.rs is
    /// terminal-only and exempt.
    #[test]
    fn every_ui_glyph_renders_in_egui_default_fonts() {
        let mut fonts = eframe::egui::epaint::text::Fonts::new(
            Default::default(),
            eframe::egui::FontDefinitions::default(),
        );
        let mut files: Vec<std::path::PathBuf> =
            vec![concat!(env!("CARGO_MANIFEST_DIR"), "/src/ui.rs").into()];
        let core = concat!(env!("CARGO_MANIFEST_DIR"), "/src/core");
        for e in std::fs::read_dir(core).unwrap() {
            files.push(e.unwrap().path());
        }
        let mut tofu = std::collections::BTreeMap::new();
        for f in files {
            let src = std::fs::read_to_string(&f).unwrap();
            // Test modules are out of scope: their fixtures are verbatim
            // copies of real-world files (Hermes's own box-drawing
            // comment banners, for one) and never reach a renderer.
            let src = src.split("#[cfg(test)]").next().unwrap_or("").to_string();
            for c in string_literal_chars(&src) {
                if c.is_ascii() {
                    continue;
                }
                let ok = fonts.has_glyph(&eframe::egui::FontId::proportional(16.0), c)
                    || fonts.has_glyph(&eframe::egui::FontId::monospace(16.0), c);
                if !ok {
                    tofu.entry(c).or_insert_with(Vec::new).push(
                        f.file_name().unwrap().to_string_lossy().to_string(),
                    );
                }
            }
        }
        assert!(
            tofu.is_empty(),
            "tofu glyphs (no glyph in egui default fonts): {:?}",
            tofu
        );
    }


    /// Chars inside normal "…" string literals (escapes skipped; raw
    /// strings and char literals are out of scope — none carry glyphs
    /// in this repo).
    fn string_literal_chars(src: &str) -> Vec<char> {
        let mut out = Vec::new();
        let mut chars = src.chars().peekable();
        let mut in_str = false;
        while let Some(c) = chars.next() {
            match (in_str, c) {
                (false, '"') => in_str = true,
                (true, '"') => in_str = false,
                (true, '\\') => {
                    chars.next(); // the escaped char, never a glyph
                }
                (true, c) => out.push(c),
                _ => {}
            }
        }
        out
    }
}
