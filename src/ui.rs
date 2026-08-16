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

use crate::core::{ollama, opencode, router, rows, settings, system};
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
    OpenCode,
    Settings,
}

/// A deferred row action, collected during rendering and executed after
/// (so table closures never need `&mut self`).
enum RowAction {
    Load(String),
    Unload(String),
    AddToOpenCode(String),
    RemoveFromOpenCode(String),
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

    // Settings pane edit buffers (applied on Save).
    edit_scan_dirs: String,
    edit_port: String,
    edit_server_bin: String,
    edit_ollama_port: String,

    activity: Vec<String>,
    busy: Option<String>,
    show_about: bool,
    last_sync: Option<String>,
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
            edit_scan_dirs: String::new(),
            edit_port: String::new(),
            edit_server_bin: String::new(),
            edit_ollama_port: String::new(),
            activity: Vec::new(),
            busy: None,
            show_about: false,
            last_sync: None,
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
            loop {
                let cfg = system::load_config();
                let state = router::status(&router::state_dir(), &system::router_config(&cfg));
                if tx.send(Msg::RouterState(state)).is_err() {
                    return;
                }
                let _ = tx.send(Msg::Ollama(ollama::probe(cfg.ollama_port)));
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
            let _ = tx.send(match run_calibration(&cfg, force, tx) {
                Ok(m) => Msg::Measurements(m),
                Err(e) => Msg::Error(format!("calibrate: {e:#}")),
            });
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
        }
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
                Msg::Configured(c) => {
                    self.configured = c;
                    rebuild = true;
                }
                Msg::Measurements(m) => {
                    let measured = m.values().filter(|x| x.n_ctx.is_some()).count();
                    let failed = m.values().filter(|x| x.error.is_some()).count();
                    self.log(format!(
                        "measurements: {measured} good, {failed} known failures"
                    ));
                    self.measurements = m;
                    self.busy = None;
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
        let rows = self.rows.clone();
        egui::ScrollArea::both().show(ui, |ui| {
            egui::Grid::new("library")
                .striped(true)
                .min_col_width(48.0)
                .show(ui, |ui| {
                    for h in [
                        "Model", "Source", "Size", "Quant", "Measured ctx", "Server", "OpenCode",
                        "", "Advice",
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
                        ui.label(&r.source);
                        ui.label(if r.size_bytes > 0 {
                            format!("{:.1} GB", r.size_bytes as f64 / 1e9)
                        } else {
                            "—".into()
                        });
                        ui.label(&r.quant);
                        ui.label(
                            r.measured_ctx
                                .map(|c| c.to_string())
                                .unwrap_or_else(|| "—".into()),
                        );
                        ui.label(r.server_status.as_deref().unwrap_or("—"));

                        // "In OpenCode" checkbox — the whole make-it-usable flow.
                        let mut checked = r.in_opencode;
                        let can_act = router_up && r.router_id.is_some();
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
                        ui.end_row();
                    }
                });
        });
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
                    .filter(|m| matches!(m.status.as_str(), "loaded" | "loading" | "sleeping"))
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

    // ─── OpenCode ────────────────────────────────────────────────────────

    fn opencode_pane(&mut self, ui: &mut egui::Ui) {
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
                                if c.context == Some(measured) {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(0, 170, 0),
                                        "✔ synced",
                                    );
                                } else {
                                    ui.colored_label(
                                        ui.visuals().warn_fg_color,
                                        format!("⟳ config differs (measured {measured})"),
                                    )
                                    .on_hover_text(
                                        "Sync opencode.json (File menu) refreshes this to the \
                                         measured value; if you set it by hand on purpose, leave it.",
                                    );
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
        ui.add_space(6.0);

        egui::Grid::new("settings").num_columns(2).show(ui, |ui| {
            ui.label("Router port");
            ui.text_edit_singleline(&mut self.edit_port);
            ui.end_row();
            ui.label("llama-server binary");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut self.edit_server_bin);
                ui.small("empty = auto-pick");
            });
            ui.end_row();
            ui.label("Ollama port");
            ui.text_edit_singleline(&mut self.edit_ollama_port);
            ui.end_row();
        });
        if let Ok(picked) = system::pick_server(&self.cfg) {
            ui.small(format!("currently using: {}", picked.display()));
        }
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if ui.button("Save & Rescan").clicked() {
                match self.parse_edit_buffers() {
                    Ok(new_cfg) => {
                        let port_changed = new_cfg.port != self.cfg.port;
                        self.cfg = new_cfg;
                        match self.cfg.save(&system::config_file()) {
                            Ok(()) => {
                                self.log("settings saved");
                                if port_changed {
                                    self.log(
                                        "port changed — regenerate preset + sync so opencode.json's baseURL follows",
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
        })
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
            if let Some(scan) = &self.scan {
                if let Some(d) = scan.devices.iter().find(|d| d.id.starts_with("CUDA")) {
                    ui.label(format!("{}: {} MiB free", d.id, d.free_mib));
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
    let progress_tx = tx.clone();
    router::calibrate(
        &router::state_dir(),
        cfg.port,
        &env_fp,
        force,
        &mut |line| {
            let _ = progress_tx.send(Msg::Progress(line));
        },
    )
}

fn run_sync(
    cfg: &settings::AppConfig,
    measurements: &router::Measurements,
) -> anyhow::Result<opencode::SyncReport> {
    let desired: Vec<_> = measurements
        .iter()
        .filter_map(|(id, m)| {
            m.n_ctx.map(|ctx| opencode::DesiredModel {
                id: id.clone(),
                display_name: format!("{id} (llama.cpp)"),
                context: ctx,
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
    let ctx = measurements
        .get(id)
        .and_then(|m| m.n_ctx)
        .ok_or_else(|| anyhow::anyhow!("{id} has no successful measurement"))?;
    let desired = opencode::DesiredModel {
        id: id.to_string(),
        display_name: format!("{id} (llama.cpp)"),
        context: ctx,
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
            let mut all = router::read_measurements(&dir);
            all.insert(
                id.to_string(),
                router::Measurement {
                    n_ctx: Some(ctx),
                    error: None,
                    args_fp,
                    env_fp: Some(env_fp),
                },
            );
            let _ = router::write_measurements(&dir, &all);
            let _ = tx.send(Msg::Measurements(all.clone()));
            match sync_single(cfg, &all, id) {
                Ok(()) => {
                    send_configured(tx);
                    let _ = tx.send(Msg::Finished(format!(
                        "{id}: measured {ctx} context, added to OpenCode{}",
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
            all.insert(
                id.to_string(),
                router::Measurement {
                    n_ctx: None,
                    error: Some(detail.clone()),
                    args_fp,
                    env_fp: Some(env_fp),
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
                ui.selectable_value(&mut self.pane, Pane::OpenCode, "⚙ OpenCode");
                ui.selectable_value(&mut self.pane, Pane::Settings, "🔧 Settings");
            });
            ui.separator();
            match self.pane {
                Pane::Library => self.library_pane(ui),
                Pane::Server => self.server_pane(ui),
                Pane::OpenCode => self.opencode_pane(ui),
                Pane::Settings => self.settings_pane(ui),
            }
        });

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
