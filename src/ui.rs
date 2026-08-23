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
    advisor, bench, diagnose, discover, ollama, opencode, router, rows, settings, system,
};
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

pub fn run() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1150.0, 720.0])
            .with_title("llama.cpp Code Conf"),
        ..Default::default()
    };
    eframe::run_native(
        "llamacppcodeconf",
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
    BuildCheck(Box<advisor::BuildCheck>),
    PresetWritten(PathBuf, usize),
    SyncDone(opencode::SyncReport),
    Progress(String),
    /// Terminates the busy state with a final line.
    Finished(String),
    Error(String),
}

#[derive(PartialEq, Clone, Copy)]
enum Pane {
    Library,
    Server,
    Connections,
    Settings,
}

/// A deferred row action, collected during rendering and executed after
/// (so table closures never need `&mut self`).
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
    show_advisor: bool,
    build_check: Option<advisor::BuildCheck>,
    backend_sel: Option<advisor::BackendSelection>,
    diagnosis: Option<DiagnosisView>,
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
struct OverrideEditor {
    id: String,
    ctx_text: String,
    kv_text: String,
    extra_text: String,
    /// The optimized context baseline: what --fit measured on this machine
    /// (None = not measured yet → auto).
    optimized_ctx: Option<u64>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = channel();
        let mut app = Self {
            tx,
            rx,
            pane: Pane::Library,
            cfg: system::load_config(),
            scan: None,
            router_state: None,
            ollama: Default::default(),
            measurements: router::read_measurements(&router::state_dir()),
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
            show_advisor: false,
            build_check: None,
            backend_sel: None,
            diagnosis: None,
        };
        app.reset_edit_buffers();
        app.spawn_scan();
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
    }

    fn log(&mut self, line: impl Into<String>) {
        self.activity.push(line.into());
        if self.activity.len() > 200 {
            self.activity.drain(..100);
        }
    }

    fn hardware(&self) -> rows::Hardware {
        let vram_mib = self
            .scan
            .as_ref()
            .map(|s| {
                s.devices
                    .iter()
                    .filter(|d| d.id.starts_with("CUDA"))
                    .chain(s.devices.iter())
                    .map(|d| d.total_mib)
                    .next()
                    .unwrap_or(0)
            })
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
        std::thread::spawn(move || {
            let meas_path = router::state_dir().join("measurements.json");
            let mut last_meas_mtime = None;
            loop {
                let cfg = system::load_config();
                let state = router::status(&router::state_dir(), &system::router_config(&cfg));
                if tx.send(Msg::RouterState(state)).is_err() {
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
        self.spawn(label, move |tx| {
            let tx2 = tx.clone();
            let mut progress = move |line: String| {
                let _ = tx2.send(Msg::Progress(line));
            };
            let _ = tx.send(match bench::run_baselines(&cfg, None, force, &mut progress) {
                Ok(0) => Msg::Finished("bench: nothing to do — all baselines current".into()),
                Ok(n) => Msg::Finished(format!(
                    "benched {n} model(s) — Speed column updated (pp/tg tokens per second)"
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
        let cfg = self.cfg.clone();
        let label = if force {
            "re-measuring ALL models (forced — minutes)"
        } else {
            "measuring new/stale models (fresh ones are skipped)"
        };
        self.spawn(label, move |tx| {
            match run_calibration(&cfg, force, tx) {
                Ok(m) => {
                    let measured = m.values().filter(|x| x.n_ctx.is_some()).count();
                    let failed = m.values().filter(|x| x.error.is_some()).count();
                    let _ = tx.send(Msg::Measurements(m));
                    let _ = tx.send(Msg::Finished(format!(
                        "calibration finished: {measured} measured, {failed} known failures"
                    )));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("calibrate: {e:#}")));
                }
            }
        });
    }

    fn action_sync(&mut self) {
        let cfg = self.cfg.clone();
        let measurements = self.measurements.clone();
        self.spawn("syncing opencode.json", move |tx| {
            match run_sync(&cfg, &measurements) {
                Ok(report) => {
                    let _ = tx.send(Msg::SyncDone(report));
                    send_configured(tx);
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("sync: {e:#}")));
                }
            }
        });
    }

    /// The one-click flow: preset if missing → start if down → wait →
    /// incremental calibrate → sync. Each step narrates.
    fn action_setup(&mut self) {
        let cfg = self.cfg.clone();
        self.spawn("setting up everything", move |tx| {
            let step = |m: &str| {
                let _ = tx.send(Msg::Progress(format!("setup: {m}")));
            };
            let rcfg = system::router_config(&cfg);
            let dir = router::state_dir();
            match router::status(&dir, &rcfg) {
                router::RouterState::Ours { .. } => step("router already running"),
                router::RouterState::Down => {
                    step("starting router");
                    if let Err(e) = start_router(&cfg) {
                        let _ = tx.send(Msg::Error(format!("setup/start: {e:#}")));
                        return;
                    }
                    let mut up = false;
                    for _ in 0..30 {
                        std::thread::sleep(std::time::Duration::from_secs(1));
                        if matches!(router::status(&dir, &rcfg), router::RouterState::Ours { .. })
                        {
                            up = true;
                            break;
                        }
                    }
                    if !up {
                        let _ = tx.send(Msg::Error(
                            "setup: router did not come up within 30s — see router.log".into(),
                        ));
                        return;
                    }
                    step("router is up");
                }
                other => {
                    let _ = tx.send(Msg::Error(format!(
                        "setup: port {} is not ours: {other:?}",
                        cfg.port
                    )));
                    return;
                }
            }
            step("measuring anything new or stale");
            let measurements = match run_calibration(&cfg, false, tx) {
                Ok(m) => m,
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("setup/calibrate: {e:#}")));
                    return;
                }
            };
            let _ = tx.send(Msg::Measurements(measurements.clone()));
            step("syncing opencode.json");
            match run_sync(&cfg, &measurements) {
                Ok(report) => {
                    let _ = tx.send(Msg::SyncDone(report));
                    send_configured(tx);
                    let _ = tx.send(Msg::Progress(
                        "setup complete — OpenCode is ready to use these models".into(),
                    ));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("setup/sync: {e:#}")));
                }
            }
        });
    }

    fn run_row_action(&mut self, action: RowAction) {
        let cfg = self.cfg.clone();
        match action {
            RowAction::Load(id) => {
                // Load = measure = make available to OpenCode, and keep it
                // warm for immediate use.
                self.spawn(&format!("loading + measuring {id}"), move |tx| {
                    measure_and_sync(&cfg, &id, true, tx);
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
                        measure_and_sync(&cfg, &id, false, tx);
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
                // Show effective values: the override where set, else the
                // optimized default — the dialog always tells the truth
                // about what the model will get.
                self.override_editor = Some(OverrideEditor {
                    id,
                    ctx_text: ov
                        .ctx
                        .or(optimized_ctx)
                        .map(|c| c.to_string())
                        .unwrap_or_default(),
                    kv_text: ov
                        .cache_type_kv
                        .clone()
                        .unwrap_or_else(|| router::DEFAULT_KV_TYPE.to_string()),
                    extra_text: ov
                        .extra
                        .iter()
                        .map(|(k, v)| format!("{k} = {v}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
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
                self.spawn(
                    &format!("archiving {} to your shelf", model.display_name()),
                    move |tx| {
                        match crate::core::library::archive_to_shelf(&model, &shelf_root) {
                            Ok(dest) => {
                                let _ = tx.send(Msg::Progress(format!(
                                    "archived → {} (hardlinked when possible; yours now)",
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
                                let _ = tx.send(Msg::Scanned(system::scan_report(&cfg, &[])));
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
        self.spawn(
            "checking your llama.cpp build (contacts the git remote)",
            move |tx| {
                let server = system::pick_server(&cfg).ok();
                let build = server.as_deref().and_then(discover::build_of);
                let log =
                    std::fs::read_to_string(router::state_dir().join("router.log")).ok();
                let check = advisor::check(server, build, &measurements, log.as_deref());
                let _ = tx.send(Msg::BuildCheck(Box::new(check)));
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
        self.spawn(
            "updating + rebuilding llama.cpp (this takes many minutes)",
            move |tx| {
                let progress_tx = tx.clone();
                let result = advisor::run_rebuild(&check, sel, &mut |line| {
                    let _ = progress_tx.send(Msg::Progress(line));
                });
                let _ = tx.send(match result {
                    Ok(()) => Msg::Finished(
                        "rebuild complete ✓ — now: Stop Router, Start Router, then Set Up \
                         Everything (a new build makes every measurement stale, so it \
                         re-measures and re-checks the previously locked models)"
                            .into(),
                    ),
                    Err(e) => Msg::Error(format!(
                        "rebuild failed: {e:#} — your existing binaries are untouched"
                    )),
                });
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
                Msg::PresetWritten(path, n) => {
                    self.log(format!("preset written: {} models → {}", n, path.display()));
                    self.busy = None;
                }
                Msg::SyncDone(r) => {
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
                Msg::Finished(p) => {
                    self.log(p);
                    self.busy = None;
                }
                Msg::Error(e) => {
                    self.log(format!("ERROR: {e}"));
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
            if ui.button("Measure New/Stale Models").clicked() {
                self.action_calibrate(false);
                ui.close();
            }
            if ui.button("Re-measure ALL (force)").clicked() {
                self.action_calibrate(true);
                ui.close();
            }
            ui.separator();
            if ui
                .button("Bench New/Stale Models (speed)")
                .on_hover_text(
                    "llama-bench baseline (prompt-processing + generation tokens/sec) for \
                     every measured model missing a current one. Unloads the router's models \
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
            if ui.button("Open Router Log").clicked() {
                open_path = Some(router::state_dir().join("router.log"));
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
        let router_up = matches!(self.router_state, Some(router::RouterState::Ours { .. }));
        if !router_up {
            ui.horizontal(|ui| {
                ui.label("Router is not running — Load and checkbox actions need it.");
                if ui.button("▶ Start Router").clicked() {
                    self.action_start();
                }
            });
            ui.separator();
        }
        let mut pending: Option<RowAction> = None;
        let mut why: Option<DiagnosisView> = None;
        let rows = self.rows.clone();
        egui::ScrollArea::both().show(ui, |ui| {
            egui::Grid::new("library")
                .striped(true)
                .min_col_width(48.0)
                .show(ui, |ui| {
                    for h in [
                        "Model", "Source", "Size", "Quant", "Feat", "Measured ctx", "Speed",
                        "Server", "OpenCode", "", "", "", "Advice", "",
                    ] {
                        ui.strong(h);
                    }
                    ui.end_row();
                    for r in &rows {
                        ui.label(&r.display).on_hover_text(
                            r.router_id
                                .as_deref()
                                .map(|id| format!("served as: {id}"))
                                .unwrap_or_else(|| "no servable identity".into()),
                        );
                        ui.label(&r.source).on_hover_text(match r.source.as_str() {
                            "shelf" => {
                                "Your shelf: locally stored, manually managed models — the \
                                 directories from Settings → scan dirs. No other tool touches \
                                 or expires these; '→ shelf' archives a copy here."
                            }
                            "ollama" => {
                                "Inside Ollama's blob store — managed by Ollama; `ollama rm` \
                                 deletes it. llama.cpp serves it directly, no copy."
                            }
                            "hf cache" => {
                                "HuggingFace download cache — managed by whichever tool \
                                 downloaded it; revisions shift and caches get pruned. \
                                 Archive to shelf to own it."
                            }
                            _ => "Offered by the running router (no scanned file matched).",
                        });
                        ui.label(if r.size_bytes > 0 {
                            format!("{:.1} GB", r.size_bytes as f64 / 1e9)
                        } else {
                            "—".into()
                        });
                        {
                            let mut badges = String::new();
                            if r.vision { badges.push('👁'); }
                            if r.mtp { badges.push('⚡'); }
                            if r.embedding { badges.push('🧬'); }
                            if badges.is_empty() {
                                ui.label("");
                            } else {
                                let mut notes: Vec<&str> = Vec::new();
                                if r.vision { notes.push("👁 vision: mmproj paired — served with image support"); }
                                if r.mtp { notes.push("⚡ MTP: multi-token-prediction layers present in the file (llama.cpp support pending upstream)"); }
                                if r.embedding { notes.push("🧬 embedding model: serves /v1/embeddings, excluded from the chat config"); }
                                ui.label(badges).on_hover_text(notes.join("\n"));
                            }
                        }
                        {
                            let resp = ui.label(&r.quant);
                            if let Some(h) = &r.quant_header_disagrees {
                                resp.on_hover_text(format!(
                                    "File name says {} but the file's own header stamps {h}. \
                                     For dynamic quants (e.g. Unsloth UD) the filename is the \
                                     truthful one — the header field can't express mixed \
                                     per-layer types.",
                                    r.quant
                                ));
                            }
                        }
                        ui.label(
                            r.measured_ctx
                                .map(|c| c.to_string())
                                .unwrap_or_else(|| "—".into()),
                        );
                        match (r.pp_tps, r.tg_tps) {
                            (None, None) => {
                                ui.label("—").on_hover_text(
                                    "No throughput baseline yet — run `llamacppcodeconf \
                                     --bench` with the GPU idle to measure it.",
                                );
                            }
                            (pp, tg) => {
                                let fmt = |v: Option<f64>| {
                                    v.map(|t| format!("{t:.0}")).unwrap_or_else(|| "?".into())
                                };
                                ui.label(format!("{}/{}", fmt(pp), fmt(tg))).on_hover_text(
                                    "Measured baseline, tokens per second: prompt processing \
                                     (pp512) / generation (tg128), benched at the serving KV \
                                     cache types via llama-bench.",
                                );
                            }
                        }
                        ui.label(r.server_status.as_deref().unwrap_or("—"));

                        // "In OpenCode" checkbox — the whole make-it-usable flow.
                        // Adding needs the router to actually offer this id;
                        // removing only needs the config file.
                        let mut checked = r.in_opencode;
                        let can_act = if r.in_opencode {
                            r.router_id.is_some()
                        } else {
                            router_up && r.router_id.is_some() && r.server_status.is_some()
                        };
                        let cb = ui.add_enabled(can_act, egui::Checkbox::without_text(&mut checked));
                        let cb = cb.on_hover_text(
                            "Checked = in opencode.json. Checking an unmeasured model loads it \
                             briefly to measure the real context first.",
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

                        // Load / Unload.
                        match (&r.router_id, r.server_status.as_deref()) {
                            (Some(id), Some("loaded")) => {
                                if ui.button("Unload").clicked() {
                                    pending = Some(RowAction::Unload(id.clone()));
                                }
                            }
                            (Some(id), Some("unloaded")) => {
                                if ui
                                    .add_enabled(router_up, egui::Button::new("Load"))
                                    .on_hover_text(
                                        "Loads now, measures the real context, and adds it to \
                                         OpenCode automatically.",
                                    )
                                    .clicked()
                                {
                                    pending = Some(RowAction::Load(id.clone()));
                                }
                            }
                            (Some(_), Some(_)) => {
                                ui.label("…");
                            }
                            _ => {
                                ui.label("—");
                            }
                        }

                        if let Some(id) = &r.router_id {
                            if ui
                                .button("⚙")
                                .on_hover_text(
                                    "Per-model overrides: pin context, KV cache type, extra \
                                     llama-server flags. Stored in the app config; survives \
                                     preset regeneration.",
                                )
                                .clicked()
                            {
                                pending = Some(RowAction::EditOverrides(id.clone()));
                            }
                        } else {
                            ui.label("");
                        }

                        // Archive: pull a cache/Ollama-owned file into the
                        // user's shelf, out of reach of other tools' pruning.
                        if r.archivable
                            && let Some(path) = &r.path
                        {
                            if ui
                                .button("→ shelf")
                                .on_hover_text(
                                    "Copies (hardlinks when free) this file into your models \
                                     directory. It becomes a normal shelf model: served by the \
                                     preset, measurable, and safe from cache eviction or \
                                     `ollama rm`.",
                                )
                                .clicked()
                            {
                                pending = Some(RowAction::Archive(path.clone()));
                            }
                        } else {
                            ui.label("");
                        }

                        let color = match r.advice_level {
                            rows::AdviceLevel::Good => egui::Color32::from_rgb(0, 170, 0),
                            rows::AdviceLevel::Warn => ui.visuals().warn_fg_color,
                            rows::AdviceLevel::Bad => ui.visuals().error_fg_color,
                            rows::AdviceLevel::Unknown => ui.visuals().weak_text_color(),
                        };
                        let short: String = if r.advice.chars().count() > 60 {
                            let mut s: String = r.advice.chars().take(57).collect();
                            s.push('…');
                            s
                        } else {
                            r.advice.clone()
                        };
                        ui.colored_label(color, short).on_hover_text(&r.advice);
                        if r.advice_level != rows::AdviceLevel::Good {
                            if ui
                                .button("Why?")
                                .on_hover_text("Plain-language explanation and what to do about it")
                                .clicked()
                            {
                                let not_offered = router_up
                                    && r.router_id.is_some()
                                    && r.server_status.is_none()
                                    && r.failure.is_none();
                                let mut failure = r.failure.clone();
                                // "failed(1)" alone explains nothing — mine
                                // the router log for this model's real error.
                                if let (Some(f), Some(id)) = (&failure, &r.router_id)
                                    && diagnose::classify(f) == diagnose::Cause::Unknown
                                    && let Ok(log) = std::fs::read_to_string(
                                        router::state_dir().join("router.log"),
                                    )
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
                                    ),
                                });
                            }
                        } else {
                            ui.label("");
                        }
                        ui.end_row();
                    }
                });
        });
        if let Some(v) = why {
            self.diagnosis = Some(v);
        }
        if let Some(action) = pending {
            self.run_row_action(action);
        }
    }

    // ─── Server ──────────────────────────────────────────────────────────

    fn server_pane(&mut self, ui: &mut egui::Ui) {
        if let Some(warning) = self.vram_contention() {
            ui.colored_label(ui.visuals().warn_fg_color, format!("⚠ {warning}"));
            ui.separator();
        }
        let state = self.router_state.clone();
        match &state {
            None => {
                ui.spinner();
            }
            Some(router::RouterState::Down) => {
                ui.label("Router is down.");
                ui.horizontal(|ui| {
                    if ui.button("▶ Start Router").clicked() {
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
                ui.separator();
                egui::CollapsingHeader::new("Router log (tail)")
                    .default_open(false)
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .max_height(200.0)
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                ui.monospace(tail_of_log());
                            });
                    });
            }
        }
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
        ui.strong("Any app (OpenAI-compatible API)");
        let base_url = format!("http://127.0.0.1:{}/v1", self.cfg.port);
        ui.horizontal(|ui| {
            ui.label("Base URL:");
            ui.monospace(&base_url);
            if ui.small_button("copy").clicked() {
                ui.ctx().copy_text(base_url.clone());
                self.log("base URL copied");
            }
        });
        ui.small(
            "Point any OpenAI-compatible client here (API key: anything, it's ignored).              Use a model id below; the router loads it on first request.",
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
                    "Machine-readable list of measured models with safe context limits —                      feed this to your own app's config",
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
        ui.add_space(8.0);
        ui.separator();
        ui.strong("OpenCode (synced connector)");
        self.opencode_section(ui);
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
        egui::ScrollArea::both().show(ui, |ui| {
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
                ui.small("1 = one big model; raise to keep a small sidecar model resident too");
            });
            ui.end_row();
            ui.label("Ollama port");
            ui.text_edit_singleline(&mut self.edit_ollama_port);
            ui.end_row();
        });

        // Detected installs: autodiscovery meets manual pointing — one
        // click adopts a found binary instead of typing its path.
        if let Some(scan) = &self.scan
            && !scan.installs.is_empty()
        {
            ui.add_space(4.0);
            ui.strong("Detected llama-server installs");
            let installs = scan.installs.clone();
            for inst in &installs {
                ui.horizontal(|ui| {
                    if ui.button("Use").clicked() {
                        self.edit_server_bin = inst.server_path.display().to_string();
                    }
                    let build = inst
                        .build
                        .map(|b| format!("b{b}"))
                        .unwrap_or_else(|| "?".into());
                    let backends = if inst.backends.is_empty() {
                        String::new()
                    } else {
                        // cpu-<arch> variants are noise at a glance.
                        let mut names: Vec<&str> = inst
                            .backends
                            .iter()
                            .map(String::as_str)
                            .filter(|b| !b.starts_with("cpu-"))
                            .collect();
                        names.dedup();
                        format!(" [{}]", names.join(", "))
                    };
                    ui.label(format!("{build}{backends} — {}", inst.server_path.display()));
                });
            }
        }
        if let Ok(picked) = system::pick_server(&self.cfg) {
            let build = discover::build_of(&picked)
                .map(|b| format!(" (b{b})"))
                .unwrap_or_default();
            ui.small(format!("currently using: {}{build}", picked.display()));
        }
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if ui.button("Save & Rescan").clicked() {
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
                                if port_changed {
                                    self.log(
                                        "port changed — regenerate preset + sync so opencode.json's baseURL follows",
                                    );
                                }
                                if router_changed {
                                    self.log(
                                        "router settings changed — Stop + Start the router to apply",
                                    );
                                }
                                self.spawn_scan();
                            }
                            Err(e) => self.log(format!("ERROR saving settings: {e:#}")),
                        }
                    }
                    Err(e) => self.log(format!("ERROR: {e}")),
                }
            }
            if ui.button("Revert").clicked() {
                self.reset_edit_buffers();
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
            overrides: self.cfg.overrides.clone(),
        })
    }

    /// The per-model override dialog: pin context, KV type, extra flags.
    /// Saved to config.json → preset regenerated → router reloaded, so it
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
                });
                ui.label("Extra llama-server flags (key = value, one per line):");
                ui.add(
                    egui::TextEdit::multiline(&mut ed.extra_text)
                        .desired_rows(3)
                        .hint_text("fit-target = 2048\nmodel-draft = /path/draft.gguf")
                        .desired_width(f32::INFINITY),
                );
                ui.small("Applied on the model's next load. Re-measure afterwards — changed flags make old numbers stale.");
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
                let (k, v) = line
                    .split_once('=')
                    .ok_or_else(|| format!("not `key = value`: {line:?}"))?;
                extra.push((k.trim().to_string(), v.trim().to_string()));
            }
            let ov = router::ModelOverrides {
                cache_type_kv: kv,
                ctx: ctx_val,
                extra,
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
                self.log(format!("ERROR: {e}"));
                self.override_editor = Some(ed); // reopen with user's text intact
            }
        }
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
        egui::Window::new("Build Advisor")
            .collapsible(false)
            .default_width(560.0)
            .open(&mut open)
            .show(ctx, |ui| {
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
                    .on_hover_text(match &check.nvcc {
                        Some(v) => format!("nvcc: {v}"),
                        None => "needs the CUDA toolkit (nvcc)".into(),
                    });
                    ui.add_enabled(
                        check.glslc,
                        egui::Checkbox::new(&mut sel.vulkan, "Vulkan (any GPU)"),
                    )
                    .on_hover_text(if check.glslc {
                        "glslc present — works on NVIDIA, AMD, and Intel".to_string()
                    } else {
                        "needs the Vulkan SDK / shaderc (glslc)".to_string()
                    });
                    ui.add_enabled(
                        check.hipcc.is_some(),
                        egui::Checkbox::new(&mut sel.hip, "ROCm (AMD)"),
                    )
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
                ui.horizontal(|ui| {
                    let can_rebuild =
                        check.repo.is_some() && check.cmake && check.dirty != Some(true);
                    if ui
                        .add_enabled(can_rebuild, egui::Button::new("⬆ Update & Rebuild Now"))
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
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let (dot, text) = match &self.router_state {
                Some(router::RouterState::Ours { models }) => {
                    let loaded: Vec<_> = models
                        .iter()
                        .filter(|m| m.status == "loaded")
                        .map(|m| m.id.as_str())
                        .collect();
                    (
                        egui::Color32::from_rgb(0, 170, 0),
                        if loaded.is_empty() {
                            "router: up (nothing loaded)".to_string()
                        } else {
                            format!("router: up — {}", loaded.join(", "))
                        },
                    )
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
            ui.colored_label(dot, "●");
            ui.label(text);
            ui.separator();
            if let Some((free, total)) = self.live_vram {
                ui.label(format!("VRAM: {free} / {total} MiB free"));
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
            }
        });
    }
}

// ─── worker bodies ───────────────────────────────────────────────────────────

fn start_router(cfg: &settings::AppConfig) -> anyhow::Result<u32> {
    if !system::preset_path().exists() {
        system::write_preset(cfg, &[])?;
    }
    router::start(&router::state_dir(), &system::router_config(cfg))
}

fn run_calibration(
    cfg: &settings::AppConfig,
    force: bool,
    tx: &Sender<Msg>,
) -> anyhow::Result<router::Measurements> {
    let env_fp = system::env_fingerprint(&system::scan_report(cfg, &[]));
    let embed = router::embedding_ids_in_preset(&system::preset_path());
    let progress_tx = tx.clone();
    router::calibrate(
        &router::state_dir(),
        cfg.port,
        &env_fp,
        force,
        &embed,
        &mut |line| {
            let _ = progress_tx.send(Msg::Progress(line));
        },
    )
}

fn run_sync(
    cfg: &settings::AppConfig,
    measurements: &router::Measurements,
) -> anyhow::Result<opencode::SyncReport> {
    let embed = router::embedding_ids_in_preset(&system::preset_path());
    let desired: Vec<_> = measurements
        .iter()
        .filter(|(id, _)| !embed.contains(id.as_str()))
        .filter_map(|(id, m)| {
            m.n_ctx.map(|ctx| opencode::DesiredModel {
                id: id.clone(),
                display_name: format!("{id} (llama.cpp)"),
                context: ctx,
                tool_call: m.tool_call,
            })
        })
        .collect();
    anyhow::ensure!(
        !desired.is_empty(),
        "no successful measurements yet — measure a model first (measured, not guessed)"
    );
    opencode::sync_file(
        &opencode::default_config_path(),
        &format!("http://127.0.0.1:{}/v1", cfg.port),
        &desired,
    )
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
    };
    opencode::sync_file(
        &opencode::default_config_path(),
        &format!("http://127.0.0.1:{}/v1", cfg.port),
        &[desired],
    )?;
    Ok(())
}

fn send_configured(tx: &Sender<Msg>) {
    let configured =
        opencode::configured_models(&opencode::default_config_path()).unwrap_or_default();
    let _ = tx.send(Msg::Configured(configured));
}

/// Load (if needed) + measure + record + add to opencode.json; optionally
/// leave the model warm. The Library's Load button and checkbox both land
/// here — loading IS measuring, so nothing enters the config unguessed.
fn measure_and_sync(cfg: &settings::AppConfig, id: &str, keep_loaded: bool, tx: &Sender<Msg>) {
    let dir = router::state_dir();
    let env_fp = system::env_fingerprint(&system::scan_report(cfg, &[]));
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
            match sync_single(cfg, &all, id) {
                Ok(()) => {
                    send_configured(tx);
                    let tools = match tool_call {
                        Some(true) => ", tool calls work",
                        Some(false) => ", tool calls NOT produced",
                        None => "",
                    };
                    let _ = tx.send(Msg::Finished(format!(
                        "{id}: measured {ctx} context{tools}, added to OpenCode{}",
                        if keep_loaded { ", still loaded" } else { "" }
                    )));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{id} measured but sync failed: {e:#}")));
                }
            }
            if !keep_loaded {
                let _ = router::unload_model(cfg.port, id);
            }
        }
        Err(e) => {
            let detail = match router::mine_load_error(&dir) {
                Some(cause) => format!("{e:#} — {cause}"),
                None => format!("{e:#}"),
            };
            let mut all = router::read_measurements(&dir);
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
    let start = meta.len().saturating_sub(4096);
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
        egui::Panel::bottom("activity")
            .resizable(true)
            .default_size(80.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.activity {
                            ui.monospace(line);
                        }
                    });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.pane, Pane::Library, "📚 Library");
                ui.selectable_value(&mut self.pane, Pane::Server, "🖥 Server");
                ui.selectable_value(&mut self.pane, Pane::Connections, "🔌 Connections");
                ui.selectable_value(&mut self.pane, Pane::Settings, "🔧 Settings");
            });
            ui.separator();
            match self.pane {
                Pane::Library => self.library_pane(ui),
                Pane::Server => self.server_pane(ui),
                Pane::Connections => self.connections_pane(ui),
                Pane::Settings => self.settings_pane(ui),
            }
        });

        self.override_dialog(ui.ctx());
        self.diagnosis_window(ui.ctx());
        self.advisor_window(ui.ctx());

        if self.show_about {
            egui::Window::new("About")
                .collapsible(false)
                .resizable(false)
                .open(&mut self.show_about)
                .show(ui.ctx(), |ui| {
                    ui.label(format!("llamacppcodeconf {}", env!("CARGO_PKG_VERSION")));
                    ui.label("Manages llama.cpp (router mode) + OpenCode config.");
                    ui.label("Measured, not guessed.");
                });
        }
    }
}
