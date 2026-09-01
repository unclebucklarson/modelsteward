//! The router server: preset generation and lifecycle for one long-lived
//! llama-server in router mode (see docs/spikes.md for the verified
//! behavior this module is built on).
//!
//! Ownership rule, inherited from llm_forge's launcher: we only ever signal
//! a process whose pid we recorded at spawn time AND whose /proc cmdline
//! still names our preset file. Anything else answering the port is
//! *external* — observed, reported, never touched.

use crate::core::library::{self, ModelFile};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Everything needed to run (or find) our router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub server_bin: PathBuf,
    pub port: u16,
    pub preset_path: PathBuf,
    /// Max concurrently loaded models (`--models-max`). 1 is the right
    /// default on a single 24GB-class GPU; the router still *lists* every
    /// model and swaps on demand.
    pub models_max: u32,
}

// ─── preset generation ───────────────────────────────────────────────────────

/// Per-model knobs the app exposes; everything unset inherits `[*]` or
/// llama-server's own defaults (notably `--fit on` and `-ngl auto`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelOverrides {
    /// KV cache type; `None` inherits the global default (q8_0).
    pub cache_type_kv: Option<String>,
    /// Explicit context; `None` lets `--fit` decide (recommended).
    pub ctx: Option<u64>,
    /// Extra raw `key = value` INI lines for anything we don't model yet.
    pub extra: Vec<(String, String)>,
    /// Serve WITHOUT the vision projector even though one exists on disk
    /// (user toggle in ⚙): llama.cpp disables cache-reuse for multimodal
    /// serving, so text-only work pays a real prefill cost for unused
    /// vision. Off by default — vision serves when the file is there.
    pub no_mmproj: bool,
}

/// The oldest llama-server this app can drive: router mode
/// (`--models-preset`, hot-swap, `--fit`) landed around b10216 — the
/// build the founding spikes verified against. Older builds exit
/// instantly and used to surface as an unexplained 30s timeout
/// (usability review D4).
pub const MIN_ROUTER_BUILD: u64 = 10_216;

/// Plain-language pre-start check. None (unprobeable) passes — the
/// timeout path still catches truly broken binaries.
pub fn router_mode_supported(build: Option<u64>) -> Result<(), String> {
    match build {
        Some(b) if b < MIN_ROUTER_BUILD => Err(format!(
            "your llama-server is b{b}, which predates router mode — this app needs \
             b{MIN_ROUTER_BUILD} or newer. The Build Advisor (Server -> Check My \
             llama.cpp) can build a current one for you."
        )),
        _ => Ok(()),
    }
}

/// One extra-flags line -> (key, value). Accepts what model cards teach
/// (usability review G7): leading dashes stripped, space OR `=`
/// separates, a bare flag means true.
pub fn parse_extra_line(line: &str) -> anyhow::Result<(String, String)> {
    let line = line.trim().trim_start_matches('-');
    anyhow::ensure!(!line.is_empty(), "empty line");
    let (k, v) = match line.split_once('=') {
        Some((k, v)) => (k.trim(), v.trim()),
        None => match line.split_once(char::is_whitespace) {
            Some((k, v)) => (k.trim(), v.trim()),
            None => (line, "true"),
        },
    };
    anyhow::ensure!(
        !k.is_empty() && !k.contains(char::is_whitespace) && !v.contains(char::is_whitespace),
        "not a flag line (want `key = value`, `--key value`, or a bare `--flag`): {line:?}"
    );
    Ok((k.to_string(), v.to_string()))
}

/// The KV cache type the generated preset's `[*]` section applies to every
/// model — the measured-on-this-hardware good default (≈2x usable context
/// vs f16). The override dialog treats this as the "optimized" baseline.
pub const DEFAULT_KV_TYPE: &str = "q8_0";

/// KV block reuse for slot prompt caching: when an agent resends a prompt
/// edited in the MIDDLE, the server shifts and reuses cache beyond the
/// edit instead of reprocessing everything after it. The single biggest
/// prefill win for agentic workloads.
pub const DEFAULT_CACHE_REUSE: u32 = 256;

/// Host-RAM budget for keeping prompt caches of swapped-out slots (MiB).
/// llama-server's default is 8192; agent workflows with several parallel
/// sessions benefit from more on RAM-rich machines.
pub const DEFAULT_CACHE_RAM_MIB: u32 = 24_576;

/// llama-server's own effective defaults, shown prefilled in the ⚙
/// dialog (its promise: the values the model will actually use) with
/// leave-equal-means-not-pinned semantics. Read from llama.cpp
/// common.h at b10630 — bump alongside upstream default changes; they
/// live HERE, not in ui.rs, because they are config-truth policy
/// (review catch 2026-08-28).
pub const DEFAULT_UBATCH: &str = "512";
pub const DEFAULT_TEMP: &str = "0.8";
pub const DEFAULT_TOP_K: &str = "40";
pub const DEFAULT_TOP_P: &str = "0.95";

/// Where slot KV snapshots live. Setting `slot-save-path` in `[*]` turns
/// on llama-server's `/slots/{id}?action=save|restore` API for every
/// served model — pure enablement, zero cost until something calls it.
/// Groundwork for cheap model-swap-back (see roadmap: slot persistence).
pub fn slot_save_dir() -> PathBuf {
    state_dir().join("slots")
}

/// Render the router preset INI. Pure — writing it and telling the server to
/// reload are separate steps.
///
/// Defaults encode the north star for single-user agentic coding:
/// `np = 1` (one slot, so OpenCode gets the whole fitted context — instances
/// default to 4 slots sharing it) and `q8_0` KV cache (measured on this
/// machine: ~2x usable context vs f16 for the same VRAM).
pub fn render_preset(
    models: &[(String, &ModelFile, ModelOverrides)],
    extra_sections: &[(String, ModelOverrides)],
) -> String {
    let mut out = format!(
        "; Generated by modelsteward — edits are welcome but regeneration\n\
         ; overwrites this file; put keep-forever tweaks in the app instead.\n\
         version = 1\n\n\
         [*]\n\
         np = 1\n\
         cache-type-k = {kv}\n\
         cache-type-v = {kv}\n\
         cache-reuse = {reuse}\n\
         cache-ram = {ram}\n\
         slot-save-path = {slots}\n",
        kv = DEFAULT_KV_TYPE,
        reuse = DEFAULT_CACHE_REUSE,
        ram = DEFAULT_CACHE_RAM_MIB,
        slots = slot_save_dir().display(),
    );
    for (alias, model, ov) in models {
        out.push('\n');
        out.push_str(&format!("[{alias}]\n"));
        out.push_str(&format!("model = {}\n", model.path.display()));
        if let Some(kv) = &ov.cache_type_kv {
            out.push_str(&format!("cache-type-k = {kv}\ncache-type-v = {kv}\n"));
        }
        if let Some(c) = ov.ctx {
            out.push_str(&format!("c = {c}\n"));
        }
        for (k, v) in &ov.extra {
            out.push_str(&format!("{k} = {v}\n"));
        }
    }
    // Sections for models the router already knows by id (its cache
    // downloads): per llama-server docs, a section whose name matches an
    // existing model configures it — no `model =` line needed or wanted.
    for (id, ov) in extra_sections {
        out.push('\n');
        out.push_str(&format!("[{id}]\n"));
        if let Some(kv) = &ov.cache_type_kv {
            out.push_str(&format!("cache-type-k = {kv}\ncache-type-v = {kv}\n"));
        }
        if let Some(c) = ov.ctx {
            out.push_str(&format!("c = {c}\n"));
        }
        for (k, v) in &ov.extra {
            out.push_str(&format!("{k} = {v}\n"));
        }
    }
    out
}

/// Default (alias, model, overrides) list for a scanned library: every model
/// with readable metadata, aliased via [`library::alias_suggestion`].
/// Duplicate aliases get a numeric suffix — never two sections one name.
/// HF-hub cache files are excluded: the router serves those natively as
/// "cache" models, and a preset entry would create a second identity.
pub fn default_entries(models: &[ModelFile]) -> Vec<(String, &ModelFile, ModelOverrides)> {
    let mut used = std::collections::HashSet::new();
    models
        .iter()
        .filter(|m| m.meta.is_some())
        .filter(|m| !matches!(m.source, crate::core::library::Source::HfHub { .. }))
        .map(|m| {
            let base = library::alias_suggestion(m);
            let mut alias = base.clone();
            let mut n = 2;
            while !used.insert(alias.clone()) {
                alias = format!("{base}-{n}");
                n += 1;
            }
            (alias, m, ModelOverrides::default())
        })
        .collect()
}

// ─── state marker (ownership) ────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct Marker {
    pub pid: u32,
    pub port: u16,
    pub preset_path: PathBuf,
    pub server_bin: PathBuf,
}

pub fn state_dir() -> PathBuf {
    crate::core::settings::xdg_dir("XDG_STATE_HOME", ".local/state")
}

fn marker_path(dir: &Path) -> PathBuf {
    dir.join("router.json")
}

pub fn read_marker(dir: &Path) -> Option<Marker> {
    serde_json::from_str(&std::fs::read_to_string(marker_path(dir)).ok()?).ok()
}

fn write_marker(dir: &Path, m: &Marker) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(marker_path(dir), serde_json::to_string_pretty(m)?)
        .with_context(|| format!("writing {}", marker_path(dir).display()))
}

fn clear_marker(dir: &Path) {
    let _ = std::fs::remove_file(marker_path(dir));
}

/// Is the marker's process still alive AND still ours? "Ours" means its
/// /proc cmdline names both llama-server and our preset file — a recycled
/// pid fails this and is treated as gone.
pub fn marker_is_live(m: &Marker) -> bool {
    let Ok(cmdline) = std::fs::read(format!("/proc/{}/cmdline", m.pid)) else {
        return false;
    };
    let cmdline = String::from_utf8_lossy(&cmdline).replace('\0', " ");
    cmdline_matches(&cmdline, &m.preset_path)
}

fn cmdline_matches(cmdline: &str, preset: &Path) -> bool {
    cmdline.contains("llama-server") && cmdline.contains(&preset.display().to_string())
}

/// As [`cmdline_matches`], but the process must also be serving THIS
/// port. Ownership of a port needs both credentials: the C8 fix
/// compared the marker's port but left this arm port-blind, so our old
/// router still alive on 18080 vouched for a stranger answering on 8080
/// after a Settings port change — the most likely moment for the two to
/// coexist (review finding F2, 2026-09-01). Every spawn path we own
/// passes `--port` explicitly (start() and the systemd unit both).
fn cmdline_matches_port(cmdline: &str, preset: &Path, port: u16) -> bool {
    let want = port.to_string();
    cmdline_matches(cmdline, preset)
        && cmdline
            .split_whitespace()
            .collect::<Vec<_>>()
            .windows(2)
            .any(|w| w[0] == "--port" && w[1] == want)
}

/// Find any process running llama-server with OUR preset file — covers the
/// systemd-user-unit case, where the marker (written by whichever process
/// spawned the server) doesn't exist in this app's state. Running our
/// generated preset is the ownership credential; an arbitrary llama-server
/// never matches.
pub fn find_preset_process(preset: &Path) -> Option<u32> {
    find_preset_process_inner(preset, None)
}

/// Port-checked variant for OWNERSHIP claims (status). The port-less
/// variant stays for stop/identity paths, where "runs our preset" is
/// the right credential regardless of which port config points at now.
pub fn find_preset_process_on(preset: &Path, port: u16) -> Option<u32> {
    find_preset_process_inner(preset, Some(port))
}

fn find_preset_process_inner(preset: &Path, port: Option<u16>) -> Option<u32> {
    let proc = std::fs::read_dir("/proc").ok()?;
    for e in proc.flatten() {
        let name = e.file_name();
        let Some(pid) = name.to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        if let Ok(cmdline) = std::fs::read(e.path().join("cmdline")) {
            let cmdline = String::from_utf8_lossy(&cmdline).replace('\0', " ");
            let hit = match port {
                Some(p) => cmdline_matches_port(&cmdline, preset, p),
                None => cmdline_matches(&cmdline, preset),
            };
            if hit {
                return Some(pid);
            }
        }
    }
    None
}

// ─── lifecycle ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RouterState {
    /// Nothing answers the port and no live marker.
    Down,
    /// Our router is up; per-model statuses included.
    Ours { models: Vec<RouterModel> },
    /// Something we didn't start answers the port. Observed, never touched.
    External { detail: String },
    /// Marker says ours but the port isn't answering (yet / anymore).
    Trouble { detail: String },
}

/// Plain sentences for user-facing messages (usability review C9/G13:
/// `{:?}` was leaking Rust enum syntax at users).
impl std::fmt::Display for RouterState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouterState::Down => write!(f, "nothing is answering that port"),
            RouterState::Ours { models } => {
                write!(f, "our router is up ({} models)", models.len())
            }
            RouterState::External { detail } => write!(
                f,
                "another server (not started by this app) is using the port — {detail}. \
                 This app observes it but won't touch it; change the port or stop that server"
            ),
            RouterState::Trouble { detail } => write!(
                f,
                "our router should be there but the port isn't answering — {detail}"
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RouterModel {
    pub id: String,
    /// "loaded" | "loading" | "unloaded" | "sleeping" | "downloading" |
    /// "downloaded" (HF-repo presets fetch first); failed loads become
    /// "failed(<exit>)" so the UI never mistakes them for merely-unloaded.
    pub status: String,
    /// Where the router got this model: "preset" (ours) or "cache"
    /// (LLAMA_CACHE downloads the router discovered on its own).
    pub source: Option<String>,
    /// Fingerprint of the effective child args the router would launch this
    /// model with — the "did its config change?" half of measurement
    /// staleness (the other half is the environment fingerprint).
    pub args_fp: Option<String>,
}

/// Drop volatile flag pairs from a child-args list before fingerprinting:
/// the router stamps `--port 0` on unloaded entries but the *actual*
/// ephemeral port after a load, so including it would mark every model
/// stale the moment it had ever been loaded.
fn stable_args<'a>(args: &[&'a str]) -> Vec<&'a str> {
    let mut out = Vec::with_capacity(args.len());
    let mut skip_value = false;
    for a in args {
        if skip_value {
            skip_value = false;
            continue;
        }
        if *a == "--port" {
            skip_value = true;
            continue;
        }
        out.push(*a);
    }
    out
}

/// FNV-1a 64-bit, hex. Stable across runs and Rust versions (unlike
/// DefaultHasher), which is what a persisted fingerprint requires.
pub fn fnv(s: &str) -> String {
    fnv_bytes(s.as_bytes())
}

/// FNV over raw bytes — for callers slicing at a fixed offset, where a
/// `&str` slice would panic on a multibyte boundary.
pub fn fnv_bytes(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// Parse the router's `GET /models` response (verified shape: spike 1).
pub fn parse_models_response(body: &serde_json::Value) -> Vec<RouterModel> {
    body.get("data")
        .and_then(|d| d.as_array())
        .map(|models| {
            models
                .iter()
                .filter_map(|m| {
                    let id = m.get("id")?.as_str()?.to_string();
                    let st = m.get("status")?;
                    let mut status = st.get("value")?.as_str()?.to_string();
                    if st.get("failed").and_then(|f| f.as_bool()) == Some(true) {
                        let code = st.get("exit_code").and_then(|c| c.as_i64()).unwrap_or(-1);
                        status = format!("failed({code})");
                    }
                    let source = m.get("source").and_then(|s| s.as_str()).map(String::from);
                    let args_fp = st
                        .get("args")
                        .and_then(|a| a.as_array())
                        .map(|args| {
                            let strs: Vec<&str> =
                                args.iter().filter_map(|v| v.as_str()).collect();
                            fnv(&stable_args(&strs).join(" "))
                        });
                    Some(RouterModel {
                        id,
                        status,
                        source,
                        args_fp,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn fetch_models(port: u16) -> Result<Vec<RouterModel>> {
    let body: serde_json::Value = ureq::get(&format!("http://127.0.0.1:{port}/models"))
        .timeout(std::time::Duration::from_secs(5))
        .call()
        .context("router /models not answering")?
        .into_json()
        .context("router /models returned non-JSON")?;
    Ok(parse_models_response(&body))
}

/// What is true about our configured port right now. "Ours" = the marker we
/// wrote at spawn is live, OR some process (e.g. a systemd user unit) is
/// running llama-server with our preset file.
pub fn status(dir: &Path, cfg: &RouterConfig) -> RouterState {
    // Ownership includes the PORT. Marker.port was written at spawn and
    // never read, so pointing the app at a port where the user runs
    // their OWN llama-server made us claim it, drive it, and record its
    // numbers as measurements — the "never touch a server we didn't
    // start" rule leaking through a dimension nobody checked (review
    // finding C8, 2026-08-31).
    let ours = read_marker(dir).is_some_and(|m| marker_is_live(&m) && m.port == cfg.port)
        || find_preset_process_on(&cfg.preset_path, cfg.port).is_some();
    match fetch_models(cfg.port) {
        Ok(models) if ours => RouterState::Ours { models },
        Ok(models) => RouterState::External {
            detail: format!(
                "port {} serves {} model(s) but we have no record of starting it",
                cfg.port,
                models.len()
            ),
        },
        Err(e) if ours => RouterState::Trouble {
            detail: format!("our router process is alive but: {e:#}"),
        },
        Err(_) => RouterState::Down,
    }
}

/// Start the router. Refuses if the port already answers (ours or not) or a
/// live marker exists — "start" never doubles as "restart".
pub fn start(dir: &Path, cfg: &RouterConfig) -> Result<u32> {
    if let Some(m) = read_marker(dir)
        && marker_is_live(&m)
    {
        bail!("our router is already running (pid {})", m.pid);
    }
    if fetch_models(cfg.port).is_ok() {
        bail!(
            "port {} is already serving — a server we didn't start; refusing to interfere",
            cfg.port
        );
    }
    std::fs::create_dir_all(dir)?;
    let log = std::fs::File::create(dir.join("router.log")).context("creating router.log")?;
    let child = std::process::Command::new(&cfg.server_bin)
        .arg("--models-preset")
        .arg(&cfg.preset_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(cfg.port.to_string())
        .arg("--models-max")
        .arg(cfg.models_max.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone().context("cloning log handle")?)
        .stderr(log)
        .spawn()
        .with_context(|| format!("spawning {}", cfg.server_bin.display()))?;
    let pid = child.id();
    write_marker(
        dir,
        &Marker {
            pid,
            port: cfg.port,
            preset_path: cfg.preset_path.clone(),
            server_bin: cfg.server_bin.clone(),
        },
    )?;
    Ok(pid)
}

/// Stop our router: SIGTERM, but only to a pid whose cmdline names our
/// preset — the marker pid normally, or a preset-matched process (systemd
/// case) when no marker exists. External servers are untouchable by design.
pub fn stop(dir: &Path, preset: &Path) -> Result<()> {
    let pid = match read_marker(dir) {
        Some(m) if marker_is_live(&m) => m.pid,
        Some(m) => {
            clear_marker(dir);
            match find_preset_process(preset) {
                Some(pid) => pid,
                None => bail!(
                    "our router (pid {}) is already gone; cleared the stale record",
                    m.pid
                ),
            }
        }
        None => find_preset_process(preset)
            .ok_or_else(|| anyhow::anyhow!("no llama-server running our preset found"))?,
    };
    let ok = std::process::Command::new("kill")
        .arg(pid.to_string())
        .status()
        .context("sending SIGTERM")?
        .success();
    if !ok {
        bail!("kill {pid} failed");
    }
    // Router children get llama-server's own stop-timeout, and a router
    // with a loaded model can take well over 5s to wind down. Waiting too
    // short is worse than waiting long: the SIGTERM stays in flight, the
    // router dies AFTER the error, and the caller walks away believing it
    // is still up (live failure 2026-08-25 during --verify-rebuild).
    for _ in 0..300 {
        if find_preset_process(preset).is_none() {
            clear_marker(dir);
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    bail!(
        "router (pid {pid}) still running 30s after SIGTERM; not escalating \
         automatically — it may exit late, so check --status before restarting"
    )
}

// ─── calibration ─────────────────────────────────────────────────────────────
//
// The context a model *actually* gets is decided by `--fit` at load time and
// is only knowable by loading the model (spike 2: 27B-Q5 settled at 72,960
// with q8_0 KV where the GGUF header says 262,144). Calibration loads each
// preset model once, records the settled n_ctx, and unloads it. Results are
// keyed by alias and kept in the state dir; opencode sync refuses to guess
// and only writes measured numbers.

/// One calibration result — success (`n_ctx` measured) or remembered
/// failure (`error`), stamped with fingerprints so staleness is detectable.
/// Old measurement files (bare `{"n_ctx": N}`) still parse; their missing
/// fingerprints read as "stale", which re-measures them — the right thing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Measurement {
    /// The context `--fit` settled on. `None` = the load failed.
    pub n_ctx: Option<u64>,
    /// Whether the model produced a well-formed tool call when probed —
    /// measured, like everything else. `None` = probe not run / inconclusive.
    pub tool_call: Option<bool>,
    /// Why the load failed, when it did.
    pub error: Option<String>,
    /// Fingerprint of the model's effective launch args at measurement time.
    pub args_fp: Option<String>,
    /// Fingerprint of the environment (server build + devices).
    pub env_fp: Option<String>,
    /// Baseline prompt-processing tokens/sec (llama-bench pp512, M7).
    pub pp_tps: Option<f64>,
    /// Baseline generation tokens/sec (llama-bench tg128).
    pub tg_tps: Option<f64>,
    /// llama.cpp build the bench numbers came from — a rebuild shifts
    /// throughput without touching any model file, so this is the bench
    /// half's staleness signal.
    pub bench_build: Option<u64>,
    /// Quality gate v2 (M8): fraction of the fixed eval battery answered
    /// correctly, and N-shot tool-call reliability, measured on demand
    /// via the Lab's Quality campaign.
    pub eval_score: Option<f64>,
    pub tool_reliability: Option<f64>,
    /// Multi-hop agent-loop completion rate (quality probe): a model
    /// can ace single tool calls yet quit mid-loop — the MoE lesson.
    pub loop_reliability: Option<f64>,
}

impl Default for Measurement {
    fn default() -> Self {
        Self {
            n_ctx: None,
            tool_call: None,
            error: None,
            args_fp: None,
            env_fp: None,
            pp_tps: None,
            tg_tps: None,
            bench_build: None,
            eval_score: None,
            tool_reliability: None,
            loop_reliability: None,
        }
    }
}

impl Measurement {
    /// Fresh = measured (or failed) under exactly this config + environment.
    /// A success without a tool-call verdict counts as stale (pre-tool-probe
    /// data) so the next calibration upgrades it.
    pub fn is_fresh(&self, current_args_fp: Option<&str>, env_fp: &str) -> bool {
        // A failure diagnosed as TRANSIENT (the server was busy) is
        // never fresh: its own advice says "re-measure when idle", and
        // treating it as fresh made calibrate refuse to follow that
        // advice (live catch 2026-09-01). Permanent failures stay
        // fresh so a broken conversion isn't re-loaded every run.
        let transient = self.error.as_deref().is_some_and(|e| {
            crate::core::diagnose::classify(e) == crate::core::diagnose::Cause::ServerBusy
        });
        let complete =
            self.error.is_some() || (self.n_ctx.is_some() && self.tool_call.is_some());
        !transient
            && complete
            && self.env_fp.as_deref() == Some(env_fp)
            && self.args_fp.as_deref() == current_args_fp
            && current_args_fp.is_some()
    }
}

pub type Measurements = std::collections::BTreeMap<String, Measurement>;

fn measurements_path(dir: &Path) -> PathBuf {
    dir.join("measurements.json")
}

/// Read the measurement store. A DAMAGED file is rescued aside and
/// reported loudly rather than read as empty: the old
/// `.ok().unwrap_or_default()` turned a half-written file into an empty
/// map, and the next write persisted that emptiness over every model's
/// context, speed and quality numbers (review finding C1, 2026-08-31).
pub fn read_measurements(dir: &Path) -> Measurements {
    let (m, _) = read_measurements_checked(dir);
    m
}

/// As [`read_measurements`], but returns the damage reason so a caller
/// can surface it. The rescue happens here so the damaged file cannot
/// be overwritten by the next save.
pub fn read_measurements_checked(dir: &Path) -> (Measurements, Option<String>) {
    let path = measurements_path(dir);
    match crate::core::safefs::read_json::<Measurements>(&path) {
        crate::core::safefs::Loaded::Ok(m) => (m, None),
        crate::core::safefs::Loaded::Missing => (Measurements::default(), None),
        crate::core::safefs::Loaded::Damaged(why) => {
            let saved = crate::core::safefs::rescue(&path);
            let note = format!(
                "{} is damaged ({why}) — starting from empty{}. Nothing was \
                 silently discarded; re-measure to rebuild.",
                path.display(),
                saved
                    .map(|p| format!("; the damaged file is preserved as {}", p.display()))
                    .unwrap_or_default()
            );
            eprintln!("WARNING: {note}");
            (Measurements::default(), Some(note))
        }
    }
}

/// Carry a model's measurement to its new alias when a file changes
/// identity (archiving a cache model gives it a preset alias). n_ctx and
/// tool_call travel with fingerprints CLEARED — the serving args changed,
/// so the next calibrate re-measures — while bench numbers keep their
/// build stamp (same bytes, same build -> still meaningful). The old key is
/// removed so the leftover cache-id row stops claiming a measurement it no
/// longer describes; an existing entry under the new id is never clobbered.
pub fn migrate_measurement(all: &mut Measurements, old_id: &str, new_id: &str) {
    if old_id == new_id || all.contains_key(new_id) {
        return;
    }
    if let Some(old) = all.remove(old_id) {
        all.insert(
            new_id.to_string(),
            Measurement {
                args_fp: None,
                env_fp: None,
                ..old
            },
        );
    }
}

/// Insert a (re)measurement, carrying over bench baselines taken under the
/// same config + environment — re-measuring ctx must not wipe them. A
/// changed fingerprint means the old numbers describe a different setup,
/// so they drop with it.
pub fn upsert_measurement(all: &mut Measurements, id: &str, mut m: Measurement) {
    if let Some(old) = all.get(id)
        && old.args_fp == m.args_fp
        && old.env_fp == m.env_fp
    {
        m.pp_tps = old.pp_tps;
        m.tg_tps = old.tg_tps;
        m.bench_build = old.bench_build;
        m.eval_score = old.eval_score;
        m.tool_reliability = old.tool_reliability;
        m.loop_reliability = old.loop_reliability;
    }
    all.insert(id.to_string(), m);
}

pub fn write_measurements(dir: &Path, m: &Measurements) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    // Atomic: this file is rewritten after EVERY model during
    // calibration, so a Ctrl-C used to be able to truncate it.
    crate::core::safefs::write_atomic(&measurements_path(dir), &serde_json::to_string_pretty(m)?)
}

/// Settled context for one model, measured without any long-blocking
/// request: explicit load -> poll status until loaded/failed (cold-cache
/// loads of 20GB files can take minutes; a hanging GET would trip HTTP
/// timeouts and the router's request-scoped kill paths) -> read `/props`
/// with autoload off.
pub fn fetch_settled_ctx(port: u16, model: &str) -> Result<u64> {
    // A load request for a model that is already loaded/loading returns
    // 400 — "already underway", not a failure. Found live with the 80B
    // MoE: its multi-minute load made every re-request a 400 that got
    // recorded as a model fault. Only POST when the model is idle; in
    // every in-flight state, fall through to the poll loop.
    let in_flight = fetch_models(port)?
        .into_iter()
        .find(|m| m.id == model)
        .is_some_and(|m| {
            matches!(
                m.status.as_str(),
                "loaded" | "sleeping" | "loading" | "downloading" | "downloaded"
            )
        });
    if !in_flight {
        load_model(port, model)?;
    }
    const LOAD_BUDGET: std::time::Duration = std::time::Duration::from_secs(600);
    let mut deadline = std::time::Instant::now() + LOAD_BUDGET;
    loop {
        let status = fetch_models(port)?
            .into_iter()
            .find(|m| m.id == model)
            .map(|m| m.status)
            .ok_or_else(|| anyhow::anyhow!("{model} vanished from /models during load"))?;
        match status.as_str() {
            "loaded" | "sleeping" => break,
            s if s.starts_with("failed") => {
                bail!("{model} did not load: {s} (see router.log)")
            }
            // HF-repo presets download before they load; a 20GB pull can
            // outlast any sane load budget. The router owns download pacing
            // and flips status on failure, so the budget only starts once
            // the download is over.
            "downloading" | "downloaded" => {
                deadline = std::time::Instant::now() + LOAD_BUDGET;
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            _ if std::time::Instant::now() > deadline => {
                let _ = unload_model(port, model); // don't leave it wedged
                bail!("{model} still not loaded after 600s; gave up")
            }
            _ => std::thread::sleep(std::time::Duration::from_secs(1)),
        }
    }
    let url = format!(
        "http://127.0.0.1:{port}/props?model={}&autoload=false",
        urlencode(model)
    );
    let body: serde_json::Value = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(15))
        .call()
        .with_context(|| format!("/props for {model}"))?
        .into_json()?;
    body.get("default_generation_settings")
        .and_then(|g| g.get("n_ctx"))
        .and_then(|n| n.as_u64())
        .ok_or_else(|| anyhow::anyhow!("no n_ctx in /props for {model}"))
}

/// Some OTHER model currently loaded/loading on the router — the signature
/// of another session's traffic colliding with a measurement (models_max=1
/// makes their load our 500). Evidence for "busy", not "broken".
pub fn loaded_other(port: u16, model: &str) -> Option<String> {
    fetch_models(port).ok()?.into_iter().find_map(|m| {
        (m.id != model && matches!(m.status.as_str(), "loaded" | "loading"))
            .then_some(m.id)
    })
}

/// Request a load (async server-side; poll `status` to watch it land).
pub fn load_model(port: u16, model: &str) -> Result<()> {
    ureq::post(&format!("http://127.0.0.1:{port}/models/load"))
        .timeout(std::time::Duration::from_secs(30))
        .send_json(serde_json::json!({ "model": model }))
        .with_context(|| format!("loading {model}"))?;
    Ok(())
}

pub fn unload_model(port: u16, model: &str) -> Result<()> {
    ureq::post(&format!("http://127.0.0.1:{port}/models/unload"))
        .timeout(std::time::Duration::from_secs(30))
        .send_json(serde_json::json!({ "model": model }))
        .with_context(|| format!("unloading {model}"))?;
    Ok(())
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

/// Measure preset models one at a time (load -> read settled ctx -> unload),
/// updating stored measurements as it goes. **Incremental by default**:
/// models whose stored measurement is fresh (same effective args, same
/// build/devices) are skipped — including remembered *failures*, so a model
/// this build can't load doesn't re-fail on every run. `force` re-measures
/// everything. `progress` narrates each decision.
pub fn calibrate(
    dir: &Path,
    port: u16,
    env_fp: &str,
    build: Option<u64>,
    force: bool,
    no_tool_probe: &std::collections::HashSet<String>,
    disabled: &std::collections::HashSet<String>,
    progress: &mut dyn FnMut(String),
) -> Result<Measurements> {
    // Preset models AND the router's own HF cache downloads ("cache" source
    // — e.g. models pulled by `llama-server -hf` or vendor tools). Both are
    // servable through the router, so both deserve measurement; anything
    // else (a future source we don't know) is left alone.
    let all: Vec<_> = fetch_models(port)?
        .into_iter()
        .filter(|m| matches!(m.source.as_deref(), Some("preset") | Some("cache")))
        .collect();
    // Disabled models are never measured. Filtering only in the preset
    // was not enough: the router also discovers models by itself (the
    // `cache` source), so those reached calibration however the user had
    // marked them (live catch 2026-08-31).
    let skipped = all.iter().filter(|m| disabled.contains(&m.id)).count();
    let models: Vec<_> = all.into_iter().filter(|m| !disabled.contains(&m.id)).collect();
    if skipped > 0 {
        progress(format!("skipping {skipped} disabled model(s)"));
    }
    if models.is_empty() {
        bail!("router lists no preset or cache models to calibrate");
    }
    let mut out = read_measurements(dir);
    let total = models.len();
    for (i, m) in models.iter().enumerate() {
        let n = i + 1;
        if !force
            && let Some(stored) = out.get(&m.id)
            && stored.is_fresh(m.args_fp.as_deref(), env_fp)
        {
            let what = match (&stored.n_ctx, &stored.error) {
                (Some(ctx), _) => format!("ctx {ctx}"),
                (None, Some(_)) => "known load failure".to_string(),
                _ => unreachable!("is_fresh requires one of them"),
            };
            progress(format!("[{n}/{total}] {} — fresh ({what}), skipping", m.id));
            continue;
        }
        progress(format!("[{n}/{total}] measuring {} (loads the model)…", m.id));
        // First failure gets one retry after a settle: unloads are async,
        // so a load can race a predecessor still releasing VRAM and fail
        // (or fit against less memory than it will really have).
        let result = fetch_settled_ctx(port, &m.id).or_else(|first_err| {
            progress(format!(
                "[{n}/{total}] {} failed once ({first_err:#}); settling 5s and retrying…",
                m.id
            ));
            std::thread::sleep(std::time::Duration::from_secs(5));
            fetch_settled_ctx(port, &m.id).map_err(|e| e.context("retry also failed"))
        });
        let measurement = match result {
            Ok(n_ctx) => {
                // While it's loaded, measure tool-calling too — the other
                // half of "is this model actually useful to OpenCode".
                let tool_call = if no_tool_probe.contains(&m.id) {
                    progress(format!(
                        "[{n}/{total}] {}: ctx {n_ctx}; embedding model — tool probe skipped",
                        m.id
                    ));
                    Some(false)
                } else {
                progress(format!("[{n}/{total}] {}: ctx {n_ctx}; probing tool calls…", m.id));
                match probe_tool_call(port, &m.id) {
                    Ok(v) => {
                        progress(format!(
                            "[{n}/{total}] {}: tool calls {}",
                            m.id,
                            if v { "work" } else { "NOT produced" }
                        ));
                        Some(v)
                    }
                    Err(e) => {
                        progress(format!("[{n}/{total}] {}: tool probe inconclusive: {e:#}", m.id));
                        None
                    }
                }
                };
                Measurement {
                    n_ctx: Some(n_ctx),
                    tool_call,
                    error: None,
                    args_fp: m.args_fp.clone(),
                    env_fp: Some(env_fp.to_string()),
                    ..Default::default()
                }
            }
            Err(e) => {
                // Another session's model on the server means the failure
                // is contention, not a property of THIS model — skip
                // without recording so the row never goes red over it.
                if let Some(other) = loaded_other(port, &m.id) {
                    progress(format!(
                        "[{n}/{total}] {}: skipped — server busy with {other} \
                         (another session?); re-measure when idle",
                        m.id
                    ));
                    continue;
                }
                let detail = match mine_load_error(dir) {
                    Some(cause) => format!("{e:#} — {cause}"),
                    None => format!("{e:#}"),
                };
                progress(format!("[{n}/{total}] {} failed: {detail}", m.id));
                Measurement {
                    n_ctx: None,
                    tool_call: None,
                    error: Some(detail),
                    args_fp: m.args_fp.clone(),
                    env_fp: Some(env_fp.to_string()),
                    ..Default::default()
                }
            }
        };
        let _ = crate::core::history::record(
            dir,
            &crate::core::history::Entry {
                when: crate::core::advisor::now_epoch(),
                model: m.id.clone(),
                build,
                args_fp: measurement.args_fp.clone(),
                n_ctx: measurement.n_ctx,
                error: measurement.error.clone(),
                ..Default::default()
            },
        );
        upsert_measurement(&mut out, &m.id, measurement);
        write_measurements(dir, &out)?; // persist per model — a mid-run
        // failure keeps everything measured so far
        let _ = unload_model(port, &m.id);
        wait_until_not_loaded(port, &m.id, std::time::Duration::from_secs(30));
    }
    Ok(out)
}

/// Does the response contain a well-formed tool call? Pure over the JSON so
/// it's testable: requires a non-empty `tool_calls` whose first entry names
/// our probe function with arguments that parse as a JSON object.
pub fn parse_tool_call_probe(body: &serde_json::Value) -> bool {
    let Some(tc) = body
        .pointer("/choices/0/message/tool_calls/0/function")
    else {
        return false;
    };
    let named_ours = tc.get("name").and_then(|n| n.as_str()) == Some("get_weather");
    let args_ok = tc
        .get("arguments")
        .and_then(|a| a.as_str())
        .and_then(|a| serde_json::from_str::<serde_json::Value>(a).ok())
        .is_some_and(|v| v.is_object());
    named_ours && args_ok
}

/// Probe a LOADED model's tool-calling: one chat request with a single tool
/// and an instruction to use it. Thinking models reason before calling, so
/// the token budget is generous and the timeout long. `Ok(false)` is a real
/// measurement ("answered, but no usable tool call"); `Err` is inconclusive
/// (network/server trouble) and stored as `None`.
pub fn probe_tool_call(port: u16, model: &str) -> Result<bool> {
    let body: serde_json::Value = ureq::post(&format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .timeout(std::time::Duration::from_secs(300))
        .send_json(serde_json::json!({
            "model": model,
            "messages": [{"role": "user",
                "content": "What is the weather in Paris right now? Use the get_weather tool to find out."}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {"type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]}}}],
            "max_tokens": 2500,
            "temperature": 0
        }))
        .with_context(|| format!("tool probe for {model}"))?
        .into_json()
        .context("tool probe returned non-JSON")?;
    Ok(parse_tool_call_probe(&body))
}

/// The most recent child-load error in router.log — called right after a
/// failed load, when the last "error loading model:" line is that load's.
/// Heuristic by design; worst case it attributes a neighbor's error, which
/// still beats a bare "failed(1)".
pub fn mine_load_error(dir: &Path) -> Option<String> {
    let path = dir.join("router.log");
    let len = std::fs::metadata(&path).ok()?.len();
    let mut f = std::fs::File::open(&path).ok()?;
    use std::io::{Read, Seek, SeekFrom};
    let start = len.saturating_sub(64 * 1024);
    f.seek(SeekFrom::Start(start)).ok()?;
    let mut tail = String::new();
    f.read_to_string(&mut tail).ok()?;
    tail.lines().rev().find_map(|l| {
        let idx = l.find("error loading model:")?;
        Some(l[idx + "error loading model:".len()..].trim().to_string())
    })
}

/// Block until the router no longer reports `model` as loaded/loading —
/// i.e. its VRAM is actually back. Measurements taken without this race
/// the previous model's teardown and come out low (or fail outright).
pub fn wait_until_not_loaded(port: u16, model: &str, timeout: std::time::Duration) {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        match fetch_models(port) {
            Ok(models) => {
                let busy = models
                    .iter()
                    .any(|m| m.id == model && matches!(m.status.as_str(), "loaded" | "loading"));
                if !busy {
                    return;
                }
            }
            Err(_) => return, // router gone — nothing to wait for
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

/// Section names in the preset carrying an `mmproj =` line — models
/// actually SERVED with a vision projector (not merely having one on
/// disk), so their opencode.json entries may declare image input.
pub fn vision_ids_in_preset(preset: &Path) -> std::collections::HashSet<String> {
    ids_with_key_in_preset(preset, |line| line.starts_with("mmproj="))
}

/// Section names in the preset carrying `embedding = true` — models served
/// for embeddings, to be excluded from chat sync and tool probes.
pub fn embedding_ids_in_preset(preset: &Path) -> std::collections::HashSet<String> {
    ids_with_key_in_preset(preset, |line| line == "embedding=true")
}

/// Every model section in the preset — i.e. what the router is
/// currently able to serve. This is the honest "still in the fleet"
/// signal for connector removals: a model that merely failed to load
/// today is still here, while one whose file is gone (or that the user
/// disabled) has left the preset entirely (review finding C4).
pub fn ids_in_preset(preset: &Path) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    let Ok(text) = std::fs::read_to_string(preset) else {
        return out;
    };
    for line in text.lines() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix('[').and_then(|r| r.strip_suffix(']'))
            && name != "*"
        {
            out.insert(name.to_string());
        }
    }
    out
}

fn ids_with_key_in_preset(
    preset: &Path,
    matches: impl Fn(&str) -> bool,
) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let Ok(text) = std::fs::read_to_string(preset) else {
        return out;
    };
    let mut current: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            current = Some(name.to_string());
        } else if matches(&line.replace(' ', ""))
            && let Some(sec) = &current
            && sec != "*"
        {
            out.insert(sec.clone());
        }
    }
    out
}

/// Ask a running router to re-read its model sources (`/models?reload=1`,
/// verified in spike 1) — how preset edits land without a restart.
pub fn reload(port: u16) -> Result<Vec<RouterModel>> {
    let body: serde_json::Value = ureq::get(&format!("http://127.0.0.1:{port}/models?reload=1"))
        .timeout(std::time::Duration::from_secs(10))
        .call()
        .context("router reload failed")?
        .into_json()?;
    Ok(parse_models_response(&body))
}

#[cfg(test)]
mod tests_version_gate {
    use super::*;

    #[test]
    fn pre_router_builds_get_a_plain_language_refusal() {
        // Usability review D4: a b10088 used to die as "router did not
        // come up within 30s — see router.log".
        let e = router_mode_supported(Some(10_088)).unwrap_err();
        assert!(e.contains("b10088") && e.contains("10216") && e.contains("Build Advisor"));
        assert!(router_mode_supported(Some(10_216)).is_ok());
        assert!(router_mode_supported(Some(10_675)).is_ok());
        assert!(router_mode_supported(None).is_ok(), "unprobeable -> let the timeout judge");
    }
}

#[cfg(test)]
mod tests_flags {
    use super::*;

    #[test]
    fn extra_flag_lines_accept_what_model_cards_teach() {
        // Usability review G7: every llama.cpp README writes
        // `--n-cpu-moe 32` / `-ub 2048`; the dialog demanded
        // `key = value`. Contract: leading dashes stripped, space OR
        // `=` separates, bare flags mean true, garbage still errors.
        assert_eq!(parse_extra_line("fit-target = 2048").unwrap(), ("fit-target".into(), "2048".into()));
        assert_eq!(parse_extra_line("--n-cpu-moe 32").unwrap(), ("n-cpu-moe".into(), "32".into()));
        assert_eq!(parse_extra_line("-ub 2048").unwrap(), ("ub".into(), "2048".into()));
        assert_eq!(parse_extra_line("--cpu-moe").unwrap(), ("cpu-moe".into(), "true".into()));
        assert_eq!(parse_extra_line("cache-reuse=512").unwrap(), ("cache-reuse".into(), "512".into()));
        assert!(parse_extra_line("what even is this line here").is_err());
        assert!(parse_extra_line("").is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::library::Source;

    #[test]
    fn port_ownership_needs_both_credentials() {
        // Review finding F2 (2026-09-01): our old router alive on 18080
        // must not vouch for a stranger answering on 8080 after a
        // Settings port change.
        let preset = Path::new("/home/u/.config/modelsteward/router.ini");
        let ours_18080 = "/opt/bin/llama-server --models-preset \
             /home/u/.config/modelsteward/router.ini --host 127.0.0.1 --port 18080";
        assert!(cmdline_matches(ours_18080, preset), "preset credential holds");
        assert!(
            !cmdline_matches_port(ours_18080, preset, 8080),
            "but it must NOT claim ownership of port 8080"
        );
        assert!(cmdline_matches_port(ours_18080, preset, 18080));
        // A stranger's server never matches either way.
        assert!(!cmdline_matches_port(
            "/usr/bin/llama-server -m model.gguf --port 8080",
            preset,
            8080
        ));
    }

    #[test]
    fn transient_failures_are_never_fresh_so_remeasure_actually_remeasures() {
        // Live catch 2026-09-01 (Scott): a model failed with the
        // ServerBusy signature, its advice said "re-measure when idle",
        // and calibrate then skipped it as "fresh (known load failure)"
        // — the app refusing to follow its own instruction. A stored
        // failure DOES count as fresh when it is permanent (an
        // Ollama-only conversion must not re-load 20 GB every run), but
        // a failure diagnosed as transient must always be retried.
        let busy = Measurement {
            // A REAL stored shape: the router 500s a load while another
            // session holds the slot; classify() maps it to ServerBusy.
            error: Some("load failed: /models/load: status code 500".into()),
            args_fp: Some("aaaa".into()),
            env_fp: Some("eeee".into()),
            ..Default::default()
        };
        assert!(
            !busy.is_fresh(Some("aaaa"), "eeee"),
            "a ServerBusy failure must be stale so the promised re-measure happens"
        );
        // A permanent failure under the same fingerprints stays fresh.
        let broken = Measurement {
            error: Some("unknown model architecture 'qwen3.5-ollama'".into()),
            args_fp: Some("aaaa".into()),
            env_fp: Some("eeee".into()),
            ..Default::default()
        };
        assert!(
            broken.is_fresh(Some("aaaa"), "eeee"),
            "a permanent failure is not re-tried on every calibrate"
        );
    }

    #[test]
    fn a_truncated_measurements_file_is_rescued_not_read_as_empty() {
        // Review finding C1 (2026-08-31): --calibrate rewrites this file
        // after EVERY model, so a Ctrl-C could truncate it. The old read
        // turned that into an empty map, and the next write persisted
        // the emptiness over every model's context, speed and quality.
        let dir = tempfile::tempdir().unwrap();
        let good: Measurements = [(
            "qwen".to_string(),
            Measurement { n_ctx: Some(113920), ..Default::default() },
        )]
        .into_iter()
        .collect();
        write_measurements(dir.path(), &good).unwrap();
        assert_eq!(read_measurements(dir.path()).len(), 1);

        // Truncate it the way an interrupted write would.
        let path = measurements_path(dir.path());
        let whole = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, &whole[..whole.len() / 2]).unwrap();

        let (m, damage) = read_measurements_checked(dir.path());
        assert!(m.is_empty(), "a damaged file yields no measurements");
        let why = damage.expect("damage must be REPORTED, not swallowed");
        assert!(why.contains("damaged"), "{why}");
        // The damaged bytes must survive beside the file, so the next
        // write cannot destroy the user's only copy.
        let rescued = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| e.file_name().to_string_lossy().ends_with(".corrupt"));
        assert!(rescued.is_some(), "damaged file must be preserved");
    }

    #[test]
    fn router_state_display_is_plain_language() {
        // Usability review C9/G13: no Rust enum syntax at users.
        let ext = RouterState::External { detail: "llama-server pid 4242".into() };
        let text = format!("{ext}");
        assert!(!text.contains('{') && !text.contains("External"), "{text}");
        assert!(text.contains("won't touch"), "{text}");
        assert!(format!("{}", RouterState::Down).contains("nothing is answering"));
        let ours = RouterState::Ours { models: vec![] };
        assert!(format!("{ours}").contains("0 models"));
    }

    fn model(path: &str) -> ModelFile {
        ModelFile {
            path: PathBuf::from(path),
            file_size: 0,
            source: Source::Shelf,
            meta: Some(Default::default()),
            mmproj: None,
        }
    }

    #[test]
    fn preset_has_coding_agent_defaults_and_per_model_sections() {
        let m1 = model("/models/a.gguf");
        let m2 = model("/models/b.gguf");
        let entries = vec![
            ("alpha".to_string(), &m1, ModelOverrides::default()),
            (
                "beta".to_string(),
                &m2,
                ModelOverrides {
                    cache_type_kv: Some("f16".into()),
                    ctx: Some(16384),
                    extra: vec![("fit-target".into(), "2048".into())],
                    ..Default::default()
                },
            ),
        ];
        let extra = vec![(
            "unsloth/Some-GGUF:Q5_K_XL".to_string(),
            ModelOverrides {
                ctx: Some(32768),
                ..Default::default()
            },
        )];
        let ini = render_preset(&entries, &extra);
        assert!(ini.contains("version = 1"));
        // Cache-id section: configures an existing router model, no path.
        assert!(ini.contains("[unsloth/Some-GGUF:Q5_K_XL]\nc = 32768"));
        assert!(ini.contains("[*]\nnp = 1\ncache-type-k = q8_0"));
        // Agent-workload cache defaults (Tier A, 2026-08-17): mid-prompt
        // KV reuse + a bigger host-RAM budget for swapped-out slot caches.
        assert!(ini.contains("cache-reuse = 256"));
        assert!(ini.contains("cache-ram = 24576"));
        // Slot save/restore API enabled fleet-wide (groundwork for cheap
        // swap-back; the dir itself is created by write_preset).
        assert!(ini.contains("slot-save-path = "));
        assert!(ini.contains("[alpha]\nmodel = /models/a.gguf\n"));
        // beta overrides the global KV type and pins ctx.
        assert!(ini.contains("[beta]"));
        assert!(ini.contains("cache-type-k = f16"));
        assert!(ini.contains("c = 16384"));
        assert!(ini.contains("fit-target = 2048"));
    }

    #[test]
    fn duplicate_aliases_get_suffixes_not_collisions() {
        let m1 = model("/x/Same-Name.gguf");
        let m2 = model("/y/Same-Name.gguf");
        let models = vec![m1, m2];
        let entries = default_entries(&models);
        assert_eq!(entries[0].0, "same-name");
        assert_eq!(entries[1].0, "same-name-2");
    }

    #[test]
    fn metaless_models_are_excluded_from_the_preset() {
        let mut broken = model("/x/broken.gguf");
        broken.meta = None;
        assert!(default_entries(&[broken]).is_empty());
    }

    #[test]
    fn parses_models_response_including_failures() {
        let body = serde_json::json!({"data": [
            {"id": "good", "status": {"value": "loaded", "args": []}},
            {"id": "bad", "status": {"value": "unloaded", "failed": true, "exit_code": 1}},
            {"id": "pending", "status": {"value": "loading"}}
        ]});
        let models = parse_models_response(&body);
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].status, "loaded");
        assert_eq!(models[1].status, "failed(1)");
        assert_eq!(models[2].status, "loading");
    }

    #[test]
    fn fnv_is_stable_and_discriminating() {
        assert_eq!(fnv("abc"), fnv("abc"));
        assert_ne!(fnv("abc"), fnv("abd"));
        assert_eq!(fnv(""), "cbf29ce484222325");
    }

    #[test]
    fn args_fingerprint_comes_from_status_args() {
        let body = serde_json::json!({"data": [
            {"id": "a", "status": {"value": "unloaded", "args": ["llama-server", "-c", "4096"]}},
            {"id": "b", "status": {"value": "unloaded"}}
        ]});
        let models = parse_models_response(&body);
        assert_eq!(models[0].args_fp.as_deref(), Some(fnv("llama-server -c 4096").as_str()));
        assert!(models[1].args_fp.is_none());
    }

    #[test]
    fn ephemeral_port_does_not_change_the_fingerprint() {
        let fp = |port: &str| {
            let body = serde_json::json!({"data": [
                {"id": "a", "status": {"value": "unloaded",
                    "args": ["llama-server", "--port", port, "-c", "4096"]}}
            ]});
            parse_models_response(&body)[0].args_fp.clone().unwrap()
        };
        assert_eq!(fp("0"), fp("53001"), "load-assigned port must not mark models stale");
        let no_port = {
            let body = serde_json::json!({"data": [
                {"id": "a", "status": {"value": "unloaded", "args": ["llama-server", "-c", "4096"]}}
            ]});
            parse_models_response(&body)[0].args_fp.clone().unwrap()
        };
        assert_eq!(fp("0"), no_port);
    }

    #[test]
    fn tool_call_probe_parsing_is_strict() {
        // The real shape from spike 2's live tool-call test.
        let good = serde_json::json!({"choices":[{"message":{"tool_calls":[
            {"function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
            "finish_reason":"tool_calls"}]});
        assert!(parse_tool_call_probe(&good));

        // Truncated arguments (the max_tokens pitfall) -> not a pass.
        let truncated = serde_json::json!({"choices":[{"message":{"tool_calls":[
            {"function":{"name":"get_weather","arguments":"{"}}]},
            "finish_reason":"length"}]});
        assert!(!parse_tool_call_probe(&truncated));

        // Plain text answer, no tool call.
        let text = serde_json::json!({"choices":[{"message":{"content":"It is sunny."},
            "finish_reason":"stop"}]});
        assert!(!parse_tool_call_probe(&text));

        // Hallucinated different function name.
        let wrong = serde_json::json!({"choices":[{"message":{"tool_calls":[
            {"function":{"name":"weather_lookup","arguments":"{}"}}]}}]});
        assert!(!parse_tool_call_probe(&wrong));
    }

    #[test]
    fn migration_moves_ctx_stale_and_keeps_bench() {
        let mut all = Measurements::new();
        all.insert(
            "unsloth/Repo:Q5".into(),
            Measurement {
                n_ctx: Some(90_000),
                tool_call: Some(true),
                args_fp: Some("aaaa".into()),
                env_fp: Some("eeee".into()),
                pp_tps: Some(1500.0),
                tg_tps: Some(40.0),
                bench_build: Some(10454),
                ..Default::default()
            },
        );
        migrate_measurement(&mut all, "unsloth/Repo:Q5", "repo-q5");
        assert!(!all.contains_key("unsloth/Repo:Q5"), "old key removed");
        let m = &all["repo-q5"];
        assert_eq!(m.n_ctx, Some(90_000));
        assert_eq!(m.tool_call, Some(true));
        assert!(m.args_fp.is_none() && m.env_fp.is_none(), "stale -> re-measures");
        assert_eq!(m.tg_tps, Some(40.0), "bench travels (same bytes, same build)");
        assert_eq!(m.bench_build, Some(10454));

        // Never clobber an existing entry under the new id.
        let mut all2 = Measurements::new();
        all2.insert("old".into(), Measurement { n_ctx: Some(1), ..Default::default() });
        all2.insert("new".into(), Measurement { n_ctx: Some(2), ..Default::default() });
        migrate_measurement(&mut all2, "old", "new");
        assert_eq!(all2["new"].n_ctx, Some(2));
        assert!(all2.contains_key("old"), "kept when nothing moved");
    }

    #[test]
    fn freshness_requires_matching_fingerprints() {
        let m = Measurement {
            n_ctx: Some(1000),
            tool_call: Some(true),
            error: None,
            args_fp: Some("aaaa".into()),
            env_fp: Some("eeee".into()),
            ..Default::default()
        };
        assert!(m.is_fresh(Some("aaaa"), "eeee"));
        assert!(!m.is_fresh(Some("bbbb"), "eeee"), "args changed -> stale");
        assert!(!m.is_fresh(Some("aaaa"), "ffff"), "env changed -> stale");
        assert!(!m.is_fresh(None, "eeee"), "unknown current args -> stale");

        let failed = Measurement {
            n_ctx: None,
            tool_call: None,
            error: Some("boom".into()),
            args_fp: Some("aaaa".into()),
            env_fp: Some("eeee".into()),
            ..Default::default()
        };
        assert!(failed.is_fresh(Some("aaaa"), "eeee"), "failures are remembered");
    }

    #[test]
    fn old_measurement_files_parse_as_stale() {
        let old: Measurements =
            serde_json::from_str(r#"{"alias": {"n_ctx": 72960}}"#).unwrap();
        let m = &old["alias"];
        assert_eq!(m.n_ctx, Some(72960));
        assert!(
            !m.is_fresh(Some("anything"), "env"),
            "no fingerprints -> re-measure"
        );
    }

    #[test]
    fn marker_roundtrip_and_pid_recycling_safety() {
        let dir = tempfile::tempdir().unwrap();
        let m = Marker {
            pid: 1, // pid 1 exists but is init, not llama-server
            port: 18080,
            preset_path: PathBuf::from("/tmp/p.ini"),
            server_bin: PathBuf::from("/usr/bin/llama-server"),
        };
        write_marker(dir.path(), &m).unwrap();
        let back = read_marker(dir.path()).unwrap();
        assert_eq!(back.pid, 1);
        // init's cmdline doesn't mention llama-server or our preset ->
        // a recycled pid is "not ours", never a kill target.
        assert!(!marker_is_live(&back));
    }

    #[test]
    fn cmdline_matching_requires_both_binary_and_preset() {
        let preset = Path::new("/home/u/.config/app/router.ini");
        assert!(cmdline_matches(
            "/opt/llama-server --models-preset /home/u/.config/app/router.ini",
            preset
        ));
        assert!(!cmdline_matches("/opt/llama-server -m other.gguf", preset));
        assert!(!cmdline_matches(
            "vim /home/u/.config/app/router.ini",
            preset
        ));
    }
}
