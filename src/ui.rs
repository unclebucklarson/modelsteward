//! The desktop shell: a traditional menu-bar application over the headless
//! core. Menus fire actions, panes show state, the status bar keeps the
//! vitals visible. Every slow operation (scan, calibrate, HTTP) runs on a
//! worker thread and reports back over a channel — the UI thread never
//! blocks on the network or a model load.

use crate::core::{opencode, router, system};
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
    Measurements(router::Measurements),
    PresetWritten(PathBuf, usize),
    SyncDone(opencode::SyncReport),
    Progress(String),
    Error(String),
}

#[derive(PartialEq, Clone, Copy)]
enum Pane {
    Library,
    Server,
    OpenCode,
}

struct App {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    pane: Pane,

    scan: Option<system::ScanReport>,
    router_state: Option<router::RouterState>,
    measurements: router::Measurements,
    port: u16,

    /// Rolling activity log shown at the bottom of every pane.
    activity: Vec<String>,
    busy: Option<String>,
    show_about: bool,
    last_sync: Option<String>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = channel();
        let app = Self {
            tx,
            rx,
            pane: Pane::Library,
            scan: None,
            router_state: None,
            measurements: router::read_measurements(&router::state_dir()),
            port: system::DEFAULT_PORT,
            activity: Vec::new(),
            busy: None,
            show_about: false,
            last_sync: None,
        };
        app.spawn_scan();
        app.spawn_status_poller(cc.egui_ctx.clone());
        app
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
        std::thread::spawn(move || {
            let _ = tx.send(Msg::Scanned(system::scan_report(&[])));
        });
    }

    /// Status poll loop: every 2s for the life of the app. Exits when the
    /// channel closes (app dropped).
    fn spawn_status_poller(&self, ctx: egui::Context) {
        let tx = self.tx.clone();
        let port = self.port;
        std::thread::spawn(move || {
            loop {
                let state = router::status(&router::state_dir(), &system::router_config(port));
                if tx.send(Msg::RouterState(state)).is_err() {
                    return;
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

    fn action_regen_preset(&mut self) {
        self.spawn("regenerating preset", |tx| {
            match system::write_preset(&[]) {
                Ok((path, n)) => {
                    let _ = tx.send(Msg::PresetWritten(path, n));
                    // A running router picks the change up via reload.
                    if let Ok(models) = router::reload(system::DEFAULT_PORT) {
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
        let port = self.port;
        self.spawn("starting router", move |tx| {
            if !system::preset_path().exists()
                && let Err(e) = system::write_preset(&[])
            {
                let _ = tx.send(Msg::Error(format!("preset: {e:#}")));
                return;
            }
            match router::start(&router::state_dir(), &system::router_config(port)) {
                Ok(pid) => {
                    let _ = tx.send(Msg::Progress(format!("router started (pid {pid})")));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("start: {e:#}")));
                }
            }
        });
    }

    fn action_stop(&mut self) {
        self.spawn("stopping router", |tx| {
            match router::stop(&router::state_dir()) {
                Ok(()) => {
                    let _ = tx.send(Msg::Progress("router stopped".into()));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("stop: {e:#}")));
                }
            }
        });
    }

    fn action_calibrate(&mut self) {
        let port = self.port;
        self.spawn("calibrating (loads each model once — minutes)", move |tx| {
            let progress_tx = tx.clone();
            let result = router::calibrate(&router::state_dir(), port, &mut |id, i, n| {
                let _ = progress_tx.send(Msg::Progress(format!("[{i}/{n}] measuring {id}…")));
            });
            match result {
                Ok(m) => {
                    let _ = tx.send(Msg::Measurements(m));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("calibrate: {e:#}")));
                }
            }
        });
    }

    fn action_sync(&mut self) {
        let port = self.port;
        let measurements = self.measurements.clone();
        self.spawn("syncing opencode.json", move |tx| {
            if measurements.is_empty() {
                let _ = tx.send(Msg::Error(
                    "no measurements yet — calibrate first (measured, not guessed)".into(),
                ));
                return;
            }
            let desired: Vec<_> = measurements
                .iter()
                .map(|(id, m)| opencode::DesiredModel {
                    id: id.clone(),
                    display_name: format!("{id} (llama.cpp)"),
                    context: m.n_ctx,
                })
                .collect();
            let path = opencode::default_config_path();
            match opencode::sync_file(&path, &format!("http://127.0.0.1:{port}/v1"), &desired) {
                Ok(report) => {
                    let _ = tx.send(Msg::SyncDone(report));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("sync: {e:#}")));
                }
            }
        });
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
                Msg::Measurements(m) => {
                    self.log(format!("calibration finished: {} models measured", m.len()));
                    self.measurements = m;
                    self.busy = None;
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
                    for o in &r.orphans {
                        self.log(format!("  orphan (left in config): {o}"));
                    }
                    self.last_sync = Some(line);
                    self.busy = None;
                }
                Msg::Progress(p) => self.log(p),
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
                let port = self.port;
                self.spawn("reloading models", move |tx| {
                    let _ = tx.send(match router::reload(port) {
                        Ok(m) => Msg::Progress(format!("reloaded: {} models", m.len())),
                        Err(e) => Msg::Error(format!("reload: {e:#}")),
                    });
                });
                ui.close();
            }
            ui.separator();
            if ui.button("Calibrate All Models").clicked() {
                self.action_calibrate();
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

    fn library_pane(&mut self, ui: &mut egui::Ui) {
        let Some(scan) = &self.scan else {
            ui.spinner();
            ui.label("scanning…");
            return;
        };
        egui::ScrollArea::both().show(ui, |ui| {
            egui::Grid::new("models")
                .striped(true)
                .min_col_width(60.0)
                .show(ui, |ui| {
                    for h in ["Model", "Source", "Quant", "Size", "Train ctx", "Measured ctx"] {
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
                                .map(|mm| mm.n_ctx.to_string())
                                .unwrap_or_else(|| "—".into()),
                        );
                        ui.end_row();
                    }
                });
        });
    }

    fn server_pane(&mut self, ui: &mut egui::Ui) {
        let mut load: Option<String> = None;
        let mut unload: Option<String> = None;
        let state = self.router_state.clone();
        match &state {
            None => {
                ui.spinner();
            }
            Some(router::RouterState::Down) => {
                ui.label("Router is down.");
                if ui.button("▶ Start Router").clicked() {
                    self.action_start();
                }
            }
            Some(router::RouterState::External { detail }) => {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!("External server on port {}: {detail}", self.port),
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
                    ui.label(format!("Router running on port {} —", self.port));
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
        let port = self.port;
        if let Some(id) = load {
            self.spawn(&format!("loading {id}"), move |tx| {
                let _ = tx.send(match router::load_model(port, &id) {
                    Ok(()) => Msg::Progress(format!("{id}: load requested")),
                    Err(e) => Msg::Error(format!("{e:#}")),
                });
            });
        }
        if let Some(id) = unload {
            self.spawn(&format!("unloading {id}"), move |tx| {
                let _ = tx.send(match router::unload_model(port, &id) {
                    Ok(()) => Msg::Progress(format!("{id}: unload requested")),
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
        ui.label(format!(
            "Measured models ready to sync: {}",
            self.measurements.len()
        ));
        egui::Grid::new("measured").striped(true).show(ui, |ui| {
            ui.strong("Model id");
            ui.strong("Measured context");
            ui.end_row();
            for (id, m) in &self.measurements {
                ui.label(id);
                ui.label(m.n_ctx.to_string());
                ui.end_row();
            }
        });
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Sync opencode.json").clicked() {
                self.action_sync();
            }
            if ui.button("Calibrate first").clicked() {
                self.action_calibrate();
            }
        });
        if let Some(s) = &self.last_sync {
            ui.label(s);
        }
        ui.small("Sync writes measured context limits only. Existing entries get a minimal patch; your hand-edits and comments survive. Orphans are reported, never deleted.");
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
            if let Some(b) = &self.busy {
                ui.spinner();
                ui.label(b);
            }
        });
    }
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
            });
            ui.separator();
            match self.pane {
                Pane::Library => self.library_pane(ui),
                Pane::Server => self.server_pane(ui),
                Pane::OpenCode => self.opencode_pane(ui),
            }
        });

        if self.show_about {
            egui::Window::new("About")
                .collapsible(false)
                .resizable(false)
                .open(&mut self.show_about)
                .show(ui.ctx(), |ui| {
                    ui.label(format!(
                        "llamacppcodeconf {}",
                        env!("CARGO_PKG_VERSION")
                    ));
                    ui.label("Manages llama.cpp (router mode) + OpenCode config.");
                    ui.label("Measured, not guessed.");
                });
        }
    }
}
