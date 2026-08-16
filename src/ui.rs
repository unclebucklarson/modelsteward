//! The desktop shell: a traditional menu-bar application over the headless
//! core. Menus fire actions, panes show state, the status bar keeps the
//! vitals visible. Every slow operation (scan, calibrate, HTTP) runs on a
//! worker thread and reports back over a channel — the UI thread never
//! blocks on the network or a model load.

use crate::core::{ollama, opencode, router, settings, system};
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

pub fn run() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
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
    PresetWritten(PathBuf, usize),
    SyncDone(opencode::SyncReport),
    Orphans(Vec<String>),
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

struct App {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    pane: Pane,

    cfg: settings::AppConfig,
    scan: Option<system::ScanReport>,
    router_state: Option<router::RouterState>,
    ollama: ollama::OllamaStatus,
    measurements: router::Measurements,
    orphans: Vec<String>,

    // Settings pane edit buffers (applied on Save).
    edit_scan_dirs: String,
    edit_port: String,
    edit_server_bin: String,
    edit_ollama_port: String,

    /// Rolling activity log shown at the bottom of every pane.
    activity: Vec<String>,
    busy: Option<String>,
    show_about: bool,
    last_sync: Option<String>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let cfg = system::load_config();
        let mut app = Self {
            tx: channel().0, // replaced below
            rx: channel().1,
            pane: Pane::Library,
            edit_scan_dirs: String::new(),
            edit_port: String::new(),
            edit_server_bin: String::new(),
            edit_ollama_port: String::new(),
            cfg,
            scan: None,
            router_state: None,
            ollama: Default::default(),
            measurements: router::read_measurements(&router::state_dir()),
            orphans: Vec::new(),
            activity: Vec::new(),
            busy: None,
            show_about: false,
            last_sync: None,
        };
        let (tx, rx) = channel();
        app.tx = tx;
        app.rx = rx;
        app.reset_edit_buffers();
        app.spawn_scan();
        app.spawn_status_poller(cc.egui_ctx.clone());
        app.spawn_orphan_check();
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

    // ─── workers ─────────────────────────────────────────────────────────

    fn spawn_scan(&self) {
        let tx = self.tx.clone();
        let cfg = self.cfg.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Msg::Scanned(system::scan_report(&cfg, &[])));
        });
    }

    /// Status poll loop: every 2s for the life of the app. Exits when the
    /// channel closes (app dropped). Reads the config file each round so a
    /// saved port change takes effect without restarting the poller.
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

    /// Recompute the orphan list off-thread (reads opencode.json).
    fn spawn_orphan_check(&self) {
        let tx = self.tx.clone();
        let keep: Vec<String> = self
            .measurements
            .iter()
            .filter(|(_, m)| m.n_ctx.is_some())
            .map(|(id, _)| id.clone())
            .collect();
        std::thread::spawn(move || {
            let orphans = opencode::orphans_in_file(&opencode::default_config_path(), &keep)
                .unwrap_or_default();
            let _ = tx.send(Msg::Orphans(orphans));
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
            let _ = tx.send(match run_sync(&cfg, &measurements) {
                Ok(report) => Msg::SyncDone(report),
                Err(e) => Msg::Error(format!("sync: {e:#}")),
            });
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

    /// Both engines holding VRAM at once is the classic local-stack failure;
    /// say so the moment it's true.
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
            "VRAM contention: router has {} loaded while Ollama holds {}",
            router_loaded.join(", "),
            ollama_names.join(", ")
        ))
    }

    fn drain_messages(&mut self) {
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
                }
                Msg::RouterState(s) => self.router_state = Some(s),
                Msg::Ollama(o) => self.ollama = o,
                Msg::Orphans(o) => self.orphans = o,
                Msg::Measurements(m) => {
                    let measured = m.values().filter(|x| x.n_ctx.is_some()).count();
                    let failed = m.values().filter(|x| x.error.is_some()).count();
                    self.log(format!(
                        "calibration: {measured} measured, {failed} known failures"
                    ));
                    self.measurements = m;
                    self.busy = None;
                    self.spawn_orphan_check();
                }
                Msg::PresetWritten(path, n) => {
                    self.log(format!("preset written: {} models → {}", n, path.display()));
                    self.busy = None;
                }
                Msg::SyncDone(r) => {
                    let line = format!(
                        "sync: {} added, {} updated, {} orphans",
                        r.added.len(),
                        r.updated.len(),
                        r.orphans.len()
                    );
                    self.log(line.clone());
                    self.orphans = r.orphans.clone();
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
    }

    // ─── panes ───────────────────────────────────────────────────────────

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

    /// Health of a library model, derived from measurements and (when the
    /// router is up) fingerprint freshness.
    fn health_of(&self, alias: &str) -> String {
        let Some(m) = self.measurements.get(alias) else {
            return "—".into();
        };
        let current_args_fp = match &self.router_state {
            Some(router::RouterState::Ours { models }) => models
                .iter()
                .find(|rm| rm.id == alias)
                .and_then(|rm| rm.args_fp.clone()),
            _ => None,
        };
        let env_fp = self.scan.as_ref().map(system::env_fingerprint);
        let fresh = match (&current_args_fp, &env_fp) {
            (Some(afp), Some(efp)) => m.is_fresh(Some(afp), efp),
            // Router down / no scan: can't judge staleness — show plain state.
            _ => true,
        };
        match (&m.n_ctx, &m.error, fresh) {
            (Some(_), _, true) => "✔ ok".into(),
            (Some(_), _, false) => "⟳ stale".into(),
            (None, Some(_), true) => "⚠ failed".into(),
            (None, Some(_), false) => "⚠ failed (stale)".into(),
            _ => "—".into(),
        }
    }

    fn library_pane(&mut self, ui: &mut egui::Ui) {
        let Some(scan) = self.scan.clone() else {
            ui.spinner();
            ui.label("scanning…");
            return;
        };
        egui::ScrollArea::both().show(ui, |ui| {
            egui::Grid::new("models")
                .striped(true)
                .min_col_width(60.0)
                .show(ui, |ui| {
                    for h in [
                        "Model", "Source", "Quant", "Size", "Train ctx", "Measured ctx", "Health",
                    ] {
                        ui.strong(h);
                    }
                    ui.end_row();
                    for m in &scan.models {
                        let alias = crate::core::library::alias_suggestion(m);
                        ui.label(m.display_name());
                        ui.label(match &m.source {
                            crate::core::library::Source::Shelf => "shelf".to_string(),
                            crate::core::library::Source::Ollama { .. } => "ollama".to_string(),
                        });
                        let meta = m.meta.as_ref();
                        ui.label(meta.and_then(|x| x.quantization.clone()).unwrap_or_default());
                        ui.label(format!("{:.1} GB", m.file_size as f64 / 1e9));
                        ui.label(
                            meta.and_then(|x| x.context_length)
                                .map(|c| c.to_string())
                                .unwrap_or_else(|| "?".into()),
                        );
                        ui.label(
                            self.measurements
                                .get(&alias)
                                .and_then(|mm| mm.n_ctx)
                                .map(|c| c.to_string())
                                .unwrap_or_else(|| "—".into()),
                        );
                        let health = self.health_of(&alias);
                        let resp = ui.label(&health);
                        if health.starts_with('⚠')
                            && let Some(err) = self
                                .measurements
                                .get(&alias)
                                .and_then(|mm| mm.error.clone())
                        {
                            resp.on_hover_text(format!(
                                "{err}\n\nLikely fix: a newer llama.cpp build (see ROADMAP: Build Advisor)."
                            ));
                        }
                        ui.end_row();
                    }
                });
        });
    }

    fn server_pane(&mut self, ui: &mut egui::Ui) {
        if let Some(warning) = self.vram_contention() {
            ui.colored_label(ui.visuals().warn_fg_color, format!("⚠ {warning}"));
            ui.small("Unload one side (Ollama models expire on their own after idle) or expect slow/failing loads.");
            ui.separator();
        }
        let mut load: Option<String> = None;
        let mut unload: Option<String> = None;
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
                    ui.label(format!("Router running on port {} —", self.cfg.port));
                    if ui.button("■ Stop").clicked() {
                        self.action_stop();
                    }
                });
                ui.separator();
                let models = models.clone();
                egui::ScrollArea::both().show(ui, |ui| {
                    egui::Grid::new("served").striped(true).show(ui, |ui| {
                        for h in ["Model", "Status", "Source", ""] {
                            ui.strong(h);
                        }
                        ui.end_row();
                        for m in &models {
                            ui.label(&m.id);
                            ui.label(&m.status);
                            ui.label(m.source.as_deref().unwrap_or("?"));
                            match m.status.as_str() {
                                "loaded" => {
                                    if ui.button("Unload").clicked() {
                                        unload = Some(m.id.clone());
                                    }
                                }
                                "unloaded" => {
                                    if ui.button("Load").clicked() {
                                        load = Some(m.id.clone());
                                    }
                                }
                                _ => {
                                    ui.label("…");
                                }
                            }
                            ui.end_row();
                        }
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

        let port = self.cfg.port;
        if let Some(id) = load {
            self.spawn(&format!("loading {id}"), move |tx| {
                let _ = tx.send(match router::load_model(port, &id) {
                    Ok(()) => Msg::Finished(format!("{id}: load requested")),
                    Err(e) => Msg::Error(format!("{e:#}")),
                });
            });
        }
        if let Some(id) = unload {
            self.spawn(&format!("unloading {id}"), move |tx| {
                let _ = tx.send(match router::unload_model(port, &id) {
                    Ok(()) => Msg::Finished(format!("{id}: unload requested")),
                    Err(e) => Msg::Error(format!("{e:#}")),
                });
            });
        }
    }

    fn opencode_pane(&mut self, ui: &mut egui::Ui) {
        ui.label(format!(
            "Config: {}",
            opencode::default_config_path().display()
        ));
        let measured = self
            .measurements
            .iter()
            .filter(|(_, m)| m.n_ctx.is_some())
            .count();
        let failed = self
            .measurements
            .iter()
            .filter(|(_, m)| m.error.is_some())
            .count();
        ui.label(format!(
            "Measured models ready to sync: {measured} (plus {failed} known failures, excluded)"
        ));
        egui::Grid::new("measured").striped(true).show(ui, |ui| {
            ui.strong("Model id");
            ui.strong("Measured context");
            ui.end_row();
            for (id, m) in &self.measurements {
                ui.label(id);
                match (&m.n_ctx, &m.error) {
                    (Some(ctx), _) => ui.label(ctx.to_string()),
                    (None, Some(_)) => ui.colored_label(ui.visuals().warn_fg_color, "load fails"),
                    _ => ui.label("—"),
                };
                ui.end_row();
            }
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
            if ui.button("Sync opencode.json").clicked() {
                self.action_sync();
            }
            if ui
                .button("Measure contexts")
                .on_hover_text(
                    "Loads each preset model once and records the context llama-server's \
                     --fit actually settled on for THIS machine — usually far less than the \
                     GGUF header claims. Sync only ever writes these measured values, so \
                     OpenCode never plans prompts against a context the server doesn't have. \
                     Fresh measurements are skipped; use Server → Re-measure ALL to force.",
                )
                .clicked()
            {
                self.action_calibrate(false);
            }
        });
        if let Some(s) = &self.last_sync {
            ui.label(s);
        }
        ui.small("Sync writes measured context limits only. Existing entries get a minimal patch; your hand-edits and comments survive. Orphans are reported, never deleted.");

        let mut to_comment: Option<String> = None;
        if !self.orphans.is_empty() {
            ui.separator();
            ui.strong("Orphans (in your config, but not measured/served)");
            for id in &self.orphans {
                ui.horizontal(|ui| {
                    ui.label(id);
                    if ui
                        .button("Comment out")
                        .on_hover_text(
                            "Comments the entry out in place with a note — never deletes. \
                             Uncomment in the file to restore.",
                        )
                        .clicked()
                    {
                        to_comment = Some(id.clone());
                    }
                });
            }
        }
        if let Some(id) = to_comment {
            let keep: Vec<String> = self
                .measurements
                .iter()
                .filter(|(_, m)| m.n_ctx.is_some())
                .map(|(id, _)| id.clone())
                .collect();
            self.spawn(&format!("commenting out {id}"), move |tx| {
                let path = opencode::default_config_path();
                match opencode::comment_out_in_file(&path, &id) {
                    Ok(()) => {
                        let _ =
                            tx.send(Msg::Finished(format!("{id}: commented out (backup kept)")));
                        let orphans = opencode::orphans_in_file(&path, &keep).unwrap_or_default();
                        let _ = tx.send(Msg::Orphans(orphans));
                    }
                    Err(e) => {
                        let _ = tx.send(Msg::Error(format!("comment out: {e:#}")));
                    }
                }
            });
        }
    }

    fn settings_pane(&mut self, ui: &mut egui::Ui) {
        ui.label(format!("Stored in {}", system::config_file().display()));
        ui.add_space(6.0);

        ui.strong("Model scan directories (one per line)");
        ui.small("The Ollama store is found automatically and doesn't belong here.");
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

// ─── worker bodies shared by actions and the one-click setup ────────────────

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
        "no successful measurements yet — measure contexts first (measured, not guessed)"
    );
    opencode::sync_file(
        &opencode::default_config_path(),
        &format!("http://127.0.0.1:{}/v1", cfg.port),
        &desired,
    )
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
