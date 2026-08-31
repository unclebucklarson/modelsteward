//! Build Advisor: is the installed llama.cpp the one this machine and this
//! model collection deserve? Deterministic probe + rules — the AI layer
//! (later) explains and diagnoses but never invents flags.
//!
//! Verdicts speak outcomes ("rebuilding would unlock 3 models you own"),
//! not flags; the exact commands live in an advanced section.

use crate::core::diagnose::{self, Cause};
use crate::core::router::Measurements;
use anyhow::{Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Default, Serialize)]
pub struct BuildCheck {
    /// The llama-server binary this app runs.
    pub server_bin: Option<PathBuf>,
    /// Git checkout the binary was built from, when we can find it.
    pub repo: Option<PathBuf>,
    pub current_build: Option<u64>,
    /// Build tag of the checked-out SOURCE (git describe HEAD). When this
    /// is newer than `current_build`, the user pulled but never rebuilt —
    /// the fix is a rebuild alone, no network needed.
    pub source_build: Option<u64>,
    /// Latest upstream build tag (e.g. 10366 from "b10366"); None = fetch
    /// failed (offline) or no repo.
    pub upstream_build: Option<u64>,
    /// Commits behind upstream master; None when unknown.
    pub behind: Option<u64>,
    /// Uncommitted changes in the checkout (a pull would be risky).
    pub dirty: Option<bool>,
    /// Checkout is on a detached HEAD (tag-pinned — branch-style
    /// `pull --ff-only` cannot apply; review catch 2026-08-28: a
    /// user-registered tag-pinned checkout hit the same wall the
    /// managed tree did, and a path compare only covered the latter).
    pub detached: Option<bool>,
    /// e.g. "86" for compute capability 8.6.
    pub cuda_arch: Option<String>,
    pub nvcc: Option<String>,
    /// Vulkan shader compiler (glslc) — required to BUILD the Vulkan
    /// backend (a runtime alone isn't enough).
    pub glslc: bool,
    /// Vulkan runtime present (vulkaninfo answers) — devices could use a
    /// Vulkan build even without the build toolchain installed.
    pub vulkan_runtime: bool,
    /// ROCm compiler (hipcc) version line, when installed.
    pub hipcc: Option<String>,
    /// AMD GPU target from rocminfo, e.g. "gfx1100".
    pub rocm_gfx: Option<String>,
    pub cmake: bool,
    /// NVIDIA driver persistence mode: Some(false) = off, which adds
    /// driver-init latency to every model load. None = not an NVIDIA box
    /// or nvidia-smi absent.
    pub persistence_mode: Option<bool>,
    /// The tailored make-it-stick command when we can derive one from
    /// how nvidia-persistenced is actually launched on THIS machine
    /// (live incident 2026-08-30: `systemctl enable nvidia-persistenced`
    /// fails on static units, and Ubuntu launches the daemon with
    /// --no-persistence-mode so it deliberately does nothing).
    pub persistence_fix: Option<String>,
    /// Models whose stored failure looks like "needs a newer build".
    pub locked_models: Vec<String>,
}

/// The daily upstream freshness probe (user request 2026-08-25: manual
/// checks left the checkout 167 commits stale). One `git fetch` — updates
/// remote-tracking refs only, never the working tree — persisted so app
/// restarts don't refetch inside the interval.
#[derive(Debug, Clone, serde::Deserialize, Serialize)]
pub struct UpstreamStatus {
    /// Unix seconds when the probe ran.
    pub checked_epoch: u64,
    /// Binary build at probe time (staleness comparisons stay honest even
    /// if the binary changes later).
    pub current_build: Option<u64>,
    pub upstream_build: Option<u64>,
    /// Commits between the checkout's HEAD and origin/master.
    pub behind: Option<u64>,
    /// false = fetch failed (offline) — freshness is UNKNOWN, not "fine".
    pub reachable: bool,
}

pub const UPSTREAM_CHECK_INTERVAL_SECS: u64 = 24 * 3600;

fn upstream_path(dir: &Path) -> PathBuf {
    dir.join("upstream.json")
}

pub fn read_upstream_status(dir: &Path) -> Option<UpstreamStatus> {
    serde_json::from_str(&std::fs::read_to_string(upstream_path(dir)).ok()?).ok()
}

pub fn write_upstream_status(dir: &Path, s: &UpstreamStatus) {
    let _ = std::fs::create_dir_all(dir);
    if let Ok(json) = serde_json::to_string_pretty(s) {
        let _ = crate::core::safefs::write_atomic(&upstream_path(dir), &json);
    }
}

pub fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Run the probe (network: one fetch; can take seconds — worker thread).
pub fn upstream_probe(server_bin: &Path, current_build: Option<u64>) -> UpstreamStatus {
    let mut s = UpstreamStatus {
        checked_epoch: now_epoch(),
        current_build,
        upstream_build: None,
        behind: None,
        reachable: false,
    };
    let Some(repo) = repo_of(server_bin) else {
        return s;
    };
    let repo_s = repo.display().to_string();
    s.reachable = Command::new("git")
        // --tags: release tags (bNNNN) are how builds are addressed by
        // the rebuild triage and the managed checkout — without them the
        // local refs know the commits but not their names.
        .args(["-C", &repo_s, "fetch", "--quiet", "--tags", "origin", "master"])
        .status()
        .map(|st| st.success())
        .unwrap_or(false);
    if s.reachable {
        s.upstream_build = run(
            "git",
            &["-C", &repo_s, "describe", "--tags", "--abbrev=0", "origin/master"],
        )
        .as_deref()
        .and_then(parse_build_tag);
        s.behind = run(
            "git",
            &["-C", &repo_s, "rev-list", "--count", "HEAD..origin/master"],
        )
        .and_then(|c| c.parse().ok());
    }
    s
}

/// What a rebuild actually changed, measured — the honest answer to "did
/// that rebuild help?". Computed from before/after measurement snapshots
/// (live case 2026-08-25: b10454->b10630 unlocked nothing, confirmed four
/// Ollama-only conversions, and cost ~9% context across the board).
#[derive(Debug, Clone, Default, Serialize)]
pub struct VerifyReport {
    /// Errored before, measured now — the rebuild's wins.
    pub unlocked: Vec<String>,
    /// Errored before and after — confirmed not-a-build-problem.
    pub still_locked: Vec<String>,
    /// Measured before, errored now — a rebuild REGRESSION, lead with it.
    pub newly_locked: Vec<String>,
    /// (id, ctx_before, ctx_after) where both measured and differ — new
    /// builds shift VRAM use, and synced limits follow the measurement.
    pub ctx_shifts: Vec<(String, u64, u64)>,
}

pub fn verify_outcome(before: &Measurements, after: &Measurements) -> VerifyReport {
    let mut r = VerifyReport::default();
    for (id, b) in before {
        let Some(a) = after.get(id) else { continue };
        match (b.error.is_some(), a.error.is_some()) {
            (true, false) if a.n_ctx.is_some() => r.unlocked.push(id.clone()),
            (true, true) => r.still_locked.push(id.clone()),
            (false, true) => r.newly_locked.push(id.clone()),
            _ => {}
        }
        if let (Some(bc), Some(ac)) = (b.n_ctx, a.n_ctx)
            && bc != ac
        {
            r.ctx_shifts.push((id.clone(), bc, ac));
        }
    }
    r
}

#[cfg(test)]
mod verify_tests {
    use super::*;

    fn m(ctx: Option<u64>, err: Option<&str>) -> crate::core::router::Measurement {
        crate::core::router::Measurement {
            n_ctx: ctx,
            error: err.map(String::from),
            ..Default::default()
        }
    }

    #[test]
    fn verify_outcome_classifies_the_b10630_case() {
        // The live 2026-08-25 shape: nothing unlocked, locks persist,
        // context dropped everywhere.
        let mut before = Measurements::new();
        let mut after = Measurements::new();
        before.insert("locked".into(), m(None, Some("rope sections")));
        after.insert("locked".into(), m(None, Some("rope sections")));
        before.insert("fine".into(), m(Some(120_064), None));
        after.insert("fine".into(), m(Some(111_872), None));
        before.insert("wins".into(), m(None, Some("old format")));
        after.insert("wins".into(), m(Some(90_000), None));
        before.insert("regresses".into(), m(Some(50_000), None));
        after.insert("regresses".into(), m(None, Some("boom")));
        let r = verify_outcome(&before, &after);
        assert_eq!(r.unlocked, vec!["wins"]);
        assert_eq!(r.still_locked, vec!["locked"]);
        assert_eq!(r.newly_locked, vec!["regresses"]);
        assert_eq!(r.ctx_shifts, vec![("fine".to_string(), 120_064, 111_872)]);
        let text = verify_summary(&r).join("\n");
        assert!(text.contains("REGRESSION"), "{text}");
        assert!(text.contains("unlocked ✔"), "{text}");
        assert!(text.contains("-7%") || text.contains("−7%"), "{text}");
    }
}

/// The report in sentences (shared by CLI and GUI log lines).
pub fn verify_summary(r: &VerifyReport) -> Vec<String> {
    let mut out = Vec::new();
    if !r.newly_locked.is_empty() {
        out.push(format!(
            "⚠ REGRESSION: {} loaded before this rebuild but fail now — consider \
             rebuilding an older tag: {}",
            r.newly_locked.len(),
            r.newly_locked.join(", ")
        ));
    }
    if !r.unlocked.is_empty() {
        out.push(format!(
            "{} model(s) unlocked ✔: {}",
            r.unlocked.len(),
            r.unlocked.join(", ")
        ));
    }
    if !r.still_locked.is_empty() {
        out.push(format!(
            "still locked ({}): {} — see each row's Why? (a current build that \
             still rejects a file usually means an Ollama-only conversion)",
            r.still_locked.len(),
            r.still_locked.join(", ")
        ));
    }
    if !r.ctx_shifts.is_empty() {
        let mean: f64 = r
            .ctx_shifts
            .iter()
            .map(|(_, b, a)| *a as f64 / *b as f64 - 1.0)
            .sum::<f64>()
            / r.ctx_shifts.len() as f64;
        out.push(format!(
            "measured context shifted on {} model(s) (mean {:+.0}%) — synced \
             limits already follow the new measurements",
            r.ctx_shifts.len(),
            mean * 100.0
        ));
    }
    if out.is_empty() {
        out.push("no measured changes — same locks, same contexts".into());
    }
    // Findings a maintainer would want to hear about deserve a nudge —
    // the report is generated locally and shared only by the user.
    let big_ctx_regression = !r.ctx_shifts.is_empty()
        && r.ctx_shifts
            .iter()
            .map(|(_, b, a)| *a as f64 / *b as f64 - 1.0)
            .sum::<f64>()
            / r.ctx_shifts.len() as f64
            <= -0.05;
    if !r.newly_locked.is_empty() || big_ctx_regression {
        out.push(
            "this looks worth reporting upstream — Tools -> Export Findings Report \
             (or `--report`) writes a sanitized summary you can review and post"
                .to_string(),
        );
    }
    out
}

/// Locate the git checkout for a llama-server at `<repo>/build/bin/llama-server`.
/// A PINNED ARCHIVE binary (builds/bN/llama-server) has no checkout above
/// it — its source is the managed clone, so freshness checks, triage, and
/// the Build Advisor resolve there (live misdiagnosis 2026-08-28: pinning
/// b10675 made the header claim "upstream unreachable (offline?)").
pub fn repo_of(server_bin: &Path) -> Option<PathBuf> {
    let repo = server_bin.parent()?.parent()?.parent()?;
    if repo.join(".git").exists() {
        return Some(repo.to_path_buf());
    }
    let managed = crate::core::managed::data_dir();
    let canon = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    (canon(server_bin).starts_with(canon(&managed))
        && crate::core::managed::checkout_present())
    .then(crate::core::managed::checkout_dir)
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// "b10366" or "b10366-14-gabc" -> 10366. Pure for testing.
pub fn parse_build_tag(tag: &str) -> Option<u64> {
    tag.trim()
        .strip_prefix('b')?
        .split(['-', '.'])
        .next()?
        .parse()
        .ok()
}

/// nvidia-smi persistence_mode line -> Some(enabled). Pure for testing.
/// From `systemctl show nvidia-persistenced -p ExecStart` output,
/// derive the exact permanent fix — or None when the generic advice
/// applies. Pure for testing; pinned with the real Ubuntu line from
/// the 2026-08-30 incident.
pub fn persistence_fix_from_execstart(show: &str) -> Option<String> {
    // ExecStart={ path=... ; argv[]=/usr/bin/nvidia-persistenced --user X --no-persistence-mode --verbose ; ... }
    let argv = show.split("argv[]=").nth(1)?.split(" ; ").next()?.trim();
    if !argv.contains("--no-persistence-mode") {
        return None;
    }
    let fixed: Vec<&str> = argv
        .split_whitespace()
        .filter(|a| *a != "--no-persistence-mode")
        .collect();
    Some(format!(
        "your persistence daemon is RUNNING but launched with \
         --no-persistence-mode (your distro's default), and its unit is \
         static — `systemctl enable` refuses it. Permanent fix: \
         sudo systemctl edit nvidia-persistenced   and add:\n\
         [Service]\nExecStart=\nExecStart={}\n\
         then: sudo systemctl restart nvidia-persistenced",
        fixed.join(" ")
    ))
}

pub fn parse_persistence_mode(s: &str) -> Option<bool> {
    match s.lines().next()?.trim() {
        "Enabled" => Some(true),
        "Disabled" => Some(false),
        _ => None,
    }
}

/// nvidia-smi compute_cap "8.6" -> cmake arch "86". Pure for testing.
pub fn parse_compute_cap(s: &str) -> Option<String> {
    let first = s.lines().next()?.trim();
    let (major, minor) = first.split_once('.')?;
    let major: u32 = major.trim().parse().ok()?;
    let minor: u32 = minor.trim().parse().ok()?;
    Some(format!("{major}{minor}"))
}

/// First AMD "gfx…" target in rocminfo output. Pure for testing.
pub fn parse_gfx_target(out: &str) -> Option<String> {
    out.lines().find_map(|l| {
        let name = l.trim().strip_prefix("Name:")?.trim();
        name.starts_with("gfx").then(|| name.to_string())
    })
}

/// Which backends a rebuild should enable. Defaults come from detection;
/// the Build Advisor window exposes them as checkboxes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BackendSelection {
    pub cuda: bool,
    pub vulkan: bool,
    pub hip: bool,
}

/// Sane defaults: every backend whose build toolchain AND hardware signal
/// are present. Vulkan is the universal fallback — it only needs glslc.
pub fn default_backends(c: &BuildCheck) -> BackendSelection {
    BackendSelection {
        cuda: c.cuda_arch.is_some() && c.nvcc.is_some(),
        vulkan: c.glslc,
        hip: c.hipcc.is_some() && c.rocm_gfx.is_some(),
    }
}

/// Per-model load errors mined from a router log: the last
/// "error loading model:" line before each "name=<id> failed to load".
/// Rescues failures stored before log-mining existed ("failed(1)" says
/// nothing) — and feeds the "Why?" panel richer evidence.
pub fn mine_failures(log: &str) -> std::collections::BTreeMap<String, String> {
    let mut out = std::collections::BTreeMap::new();
    let mut last_error: Option<String> = None;
    for line in log.lines() {
        if let Some(idx) = line.find("error loading model:") {
            last_error = Some(line[idx + "error loading model:".len()..].trim().to_string());
        }
        if let Some(idx) = line.find("name=") {
            let rest = &line[idx + "name=".len()..];
            if let Some(name) = rest.strip_suffix_or_find(" failed to load")
                && let Some(err) = &last_error
            {
                out.insert(name.to_string(), err.clone());
            }
        }
    }
    out
}

trait StripHelper {
    fn strip_suffix_or_find<'a>(&'a self, marker: &str) -> Option<&'a str>;
}
impl StripHelper for str {
    /// The model name runs from here to ` failed to load`, wherever that
    /// lands in the line (router logs wrap it in JSON sometimes).
    fn strip_suffix_or_find<'a>(&'a self, marker: &str) -> Option<&'a str> {
        let end = self.find(marker)?;
        Some(&self[..end])
    }
}

/// Which failures a newer build would likely fix. Stored errors are
/// classified directly; unclassifiable ones ("failed(1)") get a second
/// chance via the mined router log.
pub fn locked_models(measurements: &Measurements, log: Option<&str>) -> Vec<String> {
    let mined = log.map(mine_failures).unwrap_or_default();
    measurements
        .iter()
        .filter(|(id, m)| {
            let Some(err) = m.error.as_deref() else {
                return false;
            };
            match diagnose::classify(err) {
                Cause::NeedsNewerBuild => true,
                Cause::Unknown => mined
                    .get(id.as_str())
                    .is_some_and(|e| diagnose::classify(e) == Cause::NeedsNewerBuild),
                _ => false,
            }
        })
        .map(|(id, _)| id.clone())
        .collect()
}

/// The full check. Network access: one `git fetch` against the checkout's
/// origin (tolerated failure -> upstream fields stay None). Run on a worker
/// thread; this can take seconds.
pub fn check(
    server_bin: Option<PathBuf>,
    current_build: Option<u64>,
    measurements: &Measurements,
    router_log: Option<&str>,
) -> BuildCheck {
    let mut c = BuildCheck {
        locked_models: locked_models(measurements, router_log),
        cuda_arch: run(
            "nvidia-smi",
            &["--query-gpu=compute_cap", "--format=csv,noheader"],
        )
        .as_deref()
        .and_then(parse_compute_cap),
        nvcc: run("nvcc", &["--version"]).map(|v| {
            v.lines()
                .find(|l| l.contains("release"))
                .unwrap_or("present")
                .trim()
                .to_string()
        }),
        glslc: run("glslc", &["--version"]).is_some(),
        vulkan_runtime: run("vulkaninfo", &["--summary"]).is_some(),
        hipcc: run("hipcc", &["--version"]).map(|v| {
            v.lines().next().unwrap_or("present").trim().to_string()
        }),
        rocm_gfx: run("rocminfo", &[]).as_deref().and_then(parse_gfx_target),
        cmake: run("cmake", &["--version"]).is_some(),
        persistence_mode: run(
            "nvidia-smi",
            &["--query-gpu=persistence_mode", "--format=csv,noheader"],
        )
        .as_deref()
        .and_then(parse_persistence_mode),
        persistence_fix: run(
            "systemctl",
            &["show", "nvidia-persistenced", "-p", "ExecStart", "--no-pager"],
        )
        .as_deref()
        .and_then(persistence_fix_from_execstart),
        current_build,
        server_bin: server_bin.clone(),
        ..Default::default()
    };
    let Some(repo) = server_bin.as_deref().and_then(repo_of) else {
        return c;
    };
    let repo_s = repo.display().to_string();
    c.repo = Some(repo.clone());
    c.source_build = run("git", &["-C", &repo_s, "describe", "--tags", "--abbrev=0", "HEAD"])
        .as_deref()
        .and_then(parse_build_tag);
    c.dirty = run("git", &["-C", &repo_s, "status", "--porcelain"]).map(|s| !s.is_empty());
    // symbolic-ref succeeds only on a branch; a tag-pinned checkout
    // (the managed tree, or any user-added one) is detached.
    c.detached =
        Some(run("git", &["-C", &repo_s, "symbolic-ref", "-q", "HEAD"]).is_none());
    // Fetch is the only network step; failure leaves upstream unknown.
    let fetched = Command::new("git")
        // --tags: release tags (bNNNN) are how builds are addressed by
        // the rebuild triage and the managed checkout — without them the
        // local refs know the commits but not their names.
        .args(["-C", &repo_s, "fetch", "--quiet", "--tags", "origin", "master"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if fetched {
        c.upstream_build = run(
            "git",
            &["-C", &repo_s, "describe", "--tags", "--abbrev=0", "origin/master"],
        )
        .as_deref()
        .and_then(parse_build_tag);
        c.behind = run(
            "git",
            &["-C", &repo_s, "rev-list", "--count", "HEAD..origin/master"],
        )
        .and_then(|s| s.parse().ok());
    }
    c
}

/// The verdict cards, novice-first. Each is (headline, detail).
pub fn verdicts(c: &BuildCheck) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if let (Some(cur), Some(src)) = (&c.current_build, &c.source_build)
        && src > cur
    {
        out.push((
            format!("Your source is b{src} but the built binary is b{cur}"),
            "Someone updated the checkout without rebuilding — a rebuild alone (no \
             download needed) brings the binary up to date."
                .into(),
        ));
    }
    match (&c.current_build, &c.upstream_build, &c.behind) {
        (Some(cur), Some(up), Some(behind)) if up > cur => out.push((
            format!("Your llama.cpp binary is {} builds behind upstream (b{cur} -> b{up})", up - cur),
            format!(
                "The checkout itself is {behind} commit(s) behind the remote. Newer builds \
                 add model-format support and performance work."
            ),
        )),
        (Some(cur), Some(_), _) => out.push((
            format!("Your llama.cpp (b{cur}) is up to date"),
            "No rebuild needed for freshness.".into(),
        )),
        (Some(cur), None, _) => out.push((
            format!("Your llama.cpp is b{cur}; upstream unreachable"),
            "Couldn't contact the git remote — offline, or this binary has no \
             git checkout to fetch from — freshness unknown."
                .into(),
        )),
        _ => out.push((
            "Couldn't determine your llama.cpp version".into(),
            "Set the llama-server binary in Settings, or point it at a git checkout's build."
                .into(),
        )),
    }
    if !c.locked_models.is_empty() {
        let up_to_date = matches!((c.current_build, c.upstream_build), (Some(cur), Some(up)) if cur >= up);
        if up_to_date {
            out.push((
                format!(
                    "{} model(s) fail with format errors even on the newest build",
                    c.locked_models.len()
                ),
                format!(
                    "{} — these conversions appear specific to another tool (usually \
                     Ollama). They still run through Ollama; for llama.cpp, download \
                     llama.cpp-native GGUFs of the same models.",
                    c.locked_models.join(", ")
                ),
            ));
        } else {
            out.push((
                format!(
                    "Rebuilding would likely unlock {} model(s) you own",
                    c.locked_models.len()
                ),
                c.locked_models.join(", "),
            ));
        }
    }
    if !c.cmake {
        out.push((
            "cmake not found".into(),
            "Install cmake to rebuild llama.cpp.".into(),
        ));
    } else {
        let sel = default_backends(c);
        let mut ready: Vec<String> = Vec::new();
        if sel.cuda {
            ready.push(format!(
                "CUDA (compute capability {})",
                c.cuda_arch.as_deref().unwrap_or("?")
            ));
        }
        if sel.vulkan {
            ready.push("Vulkan (NVIDIA/AMD/Intel)".into());
        }
        if sel.hip {
            ready.push(format!("ROCm ({})", c.rocm_gfx.as_deref().unwrap_or("?")));
        }
        if ready.is_empty() {
            out.push((
                "No GPU build toolchain found — CPU-only build possible".into(),
                "Install the CUDA toolkit (NVIDIA), ROCm (AMD), or the Vulkan SDK \
                 with glslc (any GPU) to build with acceleration."
                    .into(),
            ));
        } else {
            out.push((
                format!("Ready to build with: {}", ready.join(" + ")),
                "Backends are selectable below; defaults match your detected \
                 hardware and toolchains."
                    .into(),
            ));
        }
        // Near-misses a novice would never spot on their own.
        if c.cuda_arch.is_some() && c.nvcc.is_none() {
            out.push((
                "NVIDIA GPU found, but the CUDA compiler (nvcc) is missing".into(),
                "Install the CUDA toolkit for the fastest backend on this card — \
                 or build Vulkan, which works without it.".into(),
            ));
        }
        if c.vulkan_runtime && !c.glslc {
            out.push((
                "Vulkan runtime present, but the shader compiler (glslc) is missing".into(),
                "Install shaderc / the Vulkan SDK to include the Vulkan backend.".into(),
            ));
        }
        if c.rocm_gfx.is_some() && c.hipcc.is_none() {
            out.push((
                "AMD GPU found, but the ROCm compiler (hipcc) is missing".into(),
                "Install ROCm for the native AMD backend — or build Vulkan instead.".into(),
            ));
        }
    }
    if c.persistence_mode == Some(false) {
        let stick = c.persistence_fix.clone().unwrap_or_else(|| {
            "a udev rule or the nvidia-persistenced service makes it stick".into()
        });
        out.push((
            "GPU persistence mode is off".into(),
            format!(
                "The driver re-initializes on every model load, adding latency. \
                 Enable now with: sudo nvidia-smi -pm 1 (holds until reboot). \
                 To survive reboots: {stick}"
            ),
        ));
    }
    if !crate::core::energy::rapl_readable() {
        out.push((
            "CPU energy counters are root-locked (RAPL)".into(),
            "The Lab's J/token column measures GPU energy only until the kernel \
             lets us read the CPU package counter. Unlock once with: sudo chmod \
             a+r /sys/class/powercap/intel-rapl:0/energy_uj (resets on reboot; a \
             udev rule makes it stick). Nothing is estimated in its absence."
                .into(),
        ));
    }
    if c.dirty == Some(true) {
        out.push((
            "Your llama.cpp checkout has local changes".into(),
            "The updater uses a fast-forward pull only and will refuse rather than touch \
             your changes."
                .into(),
        ));
    }
    out
}

/// The exact commands the rebuild runs — also shown in the advanced
/// section so nothing is hidden. Pure over the check + selection; no
/// backend selected = a CPU-only build (valid, if slow).
pub fn rebuild_commands(c: &BuildCheck, sel: BackendSelection) -> Vec<(String, Vec<String>)> {
    let repo = c
        .repo
        .as_ref()
        .map(|r| r.display().to_string())
        .unwrap_or_default();
    let mut cmds = vec![(
        "git".to_string(),
        vec![
            "-C".to_string(),
            repo.clone(),
            "pull".to_string(),
            "--ff-only".to_string(),
        ],
    )];
    cmds.extend(build_commands(&repo, c, sel));
    cmds
}

/// The cmake configure+build steps alone (no git) — shared with the
/// managed checkout, whose git state is tag-pinned (a `pull --ff-only`
/// on a detached HEAD would fail).
pub fn build_commands(
    repo: &str,
    c: &BuildCheck,
    sel: BackendSelection,
) -> Vec<(String, Vec<String>)> {
    let repo = repo.to_string();
    let mut cmds = Vec::new();
    let mut configure: Vec<String> = Vec::new();
    configure.push("-S".into());
    configure.push(repo.clone());
    configure.push("-B".into());
    configure.push(format!("{repo}/build"));
    // Explicit ON/OFF for every backend: the cmake cache remembers old
    // options, so an unstated backend silently inherits the previous
    // configure — exactly how builds drift.
    configure.push(format!("-DGGML_CUDA={}", if sel.cuda { "ON" } else { "OFF" }));
    if sel.cuda && let Some(arch) = &c.cuda_arch {
        configure.push(format!("-DCMAKE_CUDA_ARCHITECTURES={arch}"));
    }
    configure.push(format!("-DGGML_VULKAN={}", if sel.vulkan { "ON" } else { "OFF" }));
    configure.push(format!("-DGGML_HIP={}", if sel.hip { "ON" } else { "OFF" }));
    if sel.hip && let Some(gfx) = &c.rocm_gfx {
        configure.push(format!("-DAMDGPU_TARGETS={gfx}"));
    }
    cmds.push(("cmake".to_string(), configure));
    cmds.push((
        "cmake".to_string(),
        vec![
            "--build".into(),
            format!("{repo}/build"),
            "--config".into(),
            "Release".into(),
            "-j".into(),
        ],
    ));
    cmds
}

/// Run the rebuild, streaming output lines to `progress`. Stops at the
/// first failing step. Refuses without a repo.
pub fn run_rebuild(
    c: &BuildCheck,
    sel: BackendSelection,
    progress: &mut dyn FnMut(String),
) -> Result<()> {
    anyhow::ensure!(c.repo.is_some(), "no git checkout to rebuild");
    if let Some(repo) = &c.repo {
        clear_stale_build_cache(repo, progress);
    }
    run_steps(&rebuild_commands(c, sel), progress)
}

/// A moved checkout leaves a CMakeCache.txt whose recorded source dir
/// no longer matches — cmake refuses it outright (live casualty
/// 2026-08-28: the managed checkout migrated out of a snap-redirected
/// path and every build failed until the cache was wiped by hand). The
/// build dir is pure machine-generated artifact: when its cache names
/// a different source, delete it and configure fresh.
pub fn clear_stale_build_cache(repo: &Path, progress: &mut dyn FnMut(String)) {
    let build = repo.join("build");
    let Ok(text) = std::fs::read_to_string(build.join("CMakeCache.txt")) else {
        return;
    };
    let Some(recorded) = text
        .lines()
        .find_map(|l| l.strip_prefix("CMAKE_HOME_DIRECTORY:INTERNAL="))
    else {
        return;
    };
    let recorded_p = Path::new(recorded.trim());
    let same = match (recorded_p.canonicalize(), repo.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        // Old path gone (the snap case) -> definitely stale.
        _ => recorded_p == repo,
    };
    if !same {
        progress(format!(
            "build cache was created at {} — checkout has moved; clearing the \
             cmake cache for a fresh configure (built binaries kept until the \
             new build succeeds)",
            recorded.trim()
        ));
        // Only the configure state — never build/bin: if the rebuild then
        // fails, the machine must still have its working llama-server
        // (review catch 2026-08-28; the old code wiped all of build/).
        let _ = std::fs::remove_file(build.join("CMakeCache.txt"));
        let _ = std::fs::remove_dir_all(build.join("CMakeFiles"));
    }
}

/// Run a command sequence with streamed, heartbeat-throttled output —
/// the engine under run_rebuild and the managed checkout's operations.
pub fn run_steps(
    steps: &[(String, Vec<String>)],
    progress: &mut dyn FnMut(String),
) -> Result<()> {
    for (cmd, args) in steps {
        progress(format!("$ {cmd} {}", args.join(" ")));
        let mut child = Command::new(cmd)
            .args(args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning {cmd}"))?;
        // Drain stderr on a thread so neither pipe can fill and deadlock.
        let stderr = child.stderr.take();
        let err_thread = std::thread::spawn(move || {
            use std::io::BufRead;
            let mut tail: Vec<String> = Vec::new();
            if let Some(e) = stderr {
                for line in std::io::BufReader::new(e).lines().map_while(|l| l.ok()) {
                    tail.push(line);
                    if tail.len() > 30 {
                        tail.remove(0);
                    }
                }
            }
            tail
        });
        if let Some(out) = child.stdout.take() {
            use std::io::BufRead;
            let mut n = 0u32;
            for line in std::io::BufReader::new(out).lines().map_while(|l| l.ok()) {
                n += 1;
                // Builds emit thousands of lines; narrate the interesting
                // ones and a heartbeat for the rest.
                if line.contains('%') || line.contains("Building") || n % 50 == 0 {
                    progress(line);
                }
            }
        }
        let status = child.wait().context("waiting for build step")?;
        let err_tail = err_thread.join().unwrap_or_default();
        if !status.success() {
            for l in &err_tail {
                progress(format!("stderr: {l}"));
            }
            anyhow::bail!("`{cmd}` failed with {status}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::router::Measurement;

    #[test]
    fn persistence_fix_derived_from_the_real_ubuntu_unit() {
        // Live incident 2026-08-30: Scott ran `systemctl enable --now
        // nvidia-persistenced` and hit "unit files have no installation
        // config" — the unit is static AND launches the daemon with
        // --no-persistence-mode. The exact `systemctl show` line from
        // that machine:
        let show = "ExecStart={ path=/usr/bin/nvidia-persistenced ; \
argv[]=/usr/bin/nvidia-persistenced --user nvidia-persistenced \
--no-persistence-mode --verbose ; ignore_errors=no ; start_time=[n/a] ; \
stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }";
        let fix = persistence_fix_from_execstart(show).unwrap();
        assert!(fix.contains("systemctl edit nvidia-persistenced"), "{fix}");
        assert!(
            fix.contains("/usr/bin/nvidia-persistenced --user nvidia-persistenced --verbose"),
            "the derived ExecStart must be the real one minus the flag: {fix}"
        );
        assert!(!fix.contains("--no-persistence-mode\n"), "{fix}");
        // A daemon launched WITHOUT the flag needs no override.
        assert!(persistence_fix_from_execstart(
            "ExecStart={ path=/usr/bin/nvidia-persistenced ; argv[]=/usr/bin/nvidia-persistenced --verbose ; }"
        ).is_none());
        // No daemon at all -> generic advice.
        assert!(persistence_fix_from_execstart("").is_none());
    }

    #[test]
    fn parses_build_tags_and_compute_caps() {
        assert_eq!(parse_build_tag("b10366"), Some(10366));
        assert_eq!(parse_build_tag("b10366-14-gdeadbeef"), Some(10366));
        assert_eq!(parse_build_tag("v1.2"), None);
        assert_eq!(parse_compute_cap("8.6\n"), Some("86".into()));
        assert_eq!(parse_compute_cap("12.0"), Some("120".into()));
        assert_eq!(parse_compute_cap("garbage"), None);
    }

    #[test]
    fn locked_models_are_the_needs_newer_build_failures() {
        let mut m = Measurements::new();
        m.insert(
            "old-format".into(),
            Measurement {
                error: Some("key x.rope.dimension_sections has wrong array length".into()),
                ..Default::default()
            },
        );
        m.insert(
            "bad-blob".into(),
            Measurement {
                error: Some("wrong number of tensors; expected 2131, got 720".into()),
                ..Default::default()
            },
        );
        m.insert(
            "fine".into(),
            Measurement {
                n_ctx: Some(50_000),
                ..Default::default()
            },
        );
        assert_eq!(locked_models(&m, None), vec!["old-format".to_string()]);

        // "failed(1)" alone is unclassifiable — the router log rescues it.
        let mut m2 = Measurements::new();
        m2.insert(
            "gemma4-latest".into(),
            Measurement {
                error: Some("retry also failed: gemma4-latest did not load: failed(1)".into()),
                ..Default::default()
            },
        );
        let log = "\
[123] 0.0 E llama_model_load: error loading model: error loading model hyperparameters: key x has wrong array length\n\
0.49 W srv operator(): got exception: {\"error\":{\"code\":500,\"message\":\"model name=gemma4-latest failed to load\",\"type\":\"server_error\"}}\n";
        assert_eq!(locked_models(&m2, Some(log)), vec!["gemma4-latest".to_string()]);
        assert!(locked_models(&m2, None).is_empty());
    }

    #[test]
    fn verdicts_speak_outcomes() {
        let c = BuildCheck {
            current_build: Some(10216),
            upstream_build: Some(10366),
            behind: Some(150),
            locked_models: vec!["a".into(), "b".into()],
            cuda_arch: Some("86".into()),
            nvcc: Some("release 12.4".into()),
            cmake: true,
            ..Default::default()
        };
        let v = verdicts(&c);
        assert!(v[0].0.contains("150 builds behind"));
        assert!(v.iter().any(|(h, _)| h.contains("unlock 2 model(s)")));
        assert!(
            v.iter()
                .any(|(h, _)| h.contains("Ready to build with: CUDA (compute capability 86)")),
            "toolchain card names the ready backends"
        );
        assert!(
            v.iter().all(|(h, _)| !h.contains("-D")),
            "headlines never contain cmake flags"
        );
    }

    #[test]
    fn rebuild_commands_are_ff_only_arch_pinned_and_backend_explicit() {
        let c = BuildCheck {
            repo: Some(PathBuf::from("/home/u/src/llama.cpp")),
            cuda_arch: Some("86".into()),
            rocm_gfx: Some("gfx1100".into()),
            ..Default::default()
        };
        let all = BackendSelection { cuda: true, vulkan: true, hip: true };
        let cmds = rebuild_commands(&c, all);
        assert_eq!(cmds[0].0, "git");
        assert!(cmds[0].1.contains(&"--ff-only".to_string()), "never clobbers local work");
        let cfg = &cmds[1].1;
        assert!(cfg.iter().any(|a| a == "-DGGML_CUDA=ON"));
        assert!(cfg.iter().any(|a| a == "-DCMAKE_CUDA_ARCHITECTURES=86"));
        assert!(cfg.iter().any(|a| a == "-DGGML_VULKAN=ON"));
        assert!(cfg.iter().any(|a| a == "-DGGML_HIP=ON"));
        assert!(cfg.iter().any(|a| a == "-DAMDGPU_TARGETS=gfx1100"));
        assert!(cmds[2].1.iter().any(|a| a == "Release"));

        // Deselected backends are explicitly OFF — the cmake cache would
        // otherwise resurrect whatever the last configure used.
        let cuda_only = BackendSelection { cuda: true, vulkan: false, hip: false };
        let cfg = &rebuild_commands(&c, cuda_only)[1].1;
        assert!(cfg.iter().any(|a| a == "-DGGML_VULKAN=OFF"));
        assert!(cfg.iter().any(|a| a == "-DGGML_HIP=OFF"));
        assert!(!cfg.iter().any(|a| a.contains("AMDGPU_TARGETS")));
    }

    #[test]
    fn backend_defaults_follow_toolchain_and_hardware() {
        let mut c = BuildCheck {
            cuda_arch: Some("86".into()),
            nvcc: Some("release 12.4".into()),
            glslc: true,
            ..Default::default()
        };
        assert_eq!(
            default_backends(&c),
            BackendSelection { cuda: true, vulkan: true, hip: false }
        );
        // AMD box: no nvcc, ROCm installed.
        c = BuildCheck {
            hipcc: Some("HIP version 6.1".into()),
            rocm_gfx: Some("gfx1100".into()),
            glslc: false,
            ..Default::default()
        };
        assert_eq!(
            default_backends(&c),
            BackendSelection { cuda: false, vulkan: false, hip: true }
        );
    }

    #[test]
    fn parses_persistence_mode() {
        assert_eq!(parse_persistence_mode("Enabled\n"), Some(true));
        assert_eq!(parse_persistence_mode("Disabled"), Some(false));
        assert_eq!(parse_persistence_mode("garbage"), None);
    }

    #[test]
    fn parses_rocminfo_gfx_target() {
        let out = "  Marketing Name:  AMD Radeon RX 7900\n  Name:                    gfx1100\n  Name: amdgcn-amd-amdhsa--gfx1100\n";
        assert_eq!(parse_gfx_target(out), Some("gfx1100".into()));
        assert_eq!(parse_gfx_target("Name: some-cpu"), None);
    }
    #[test]
    fn moved_checkouts_get_a_fresh_build_dir() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("llama.cpp");
        let build = repo.join("build");
        std::fs::create_dir_all(&build).unwrap();
        let log = std::cell::RefCell::new(Vec::new());
        let mut progress = |l: String| log.borrow_mut().push(l);

        // Cache from a DIFFERENT (now nonexistent) source dir -> wiped.
        std::fs::write(
            build.join("CMakeCache.txt"),
            "CMAKE_HOME_DIRECTORY:INTERNAL=/home/u/snap/code/258/.local/share/modelsteward/llama.cpp\n",
        )
        .unwrap();
        std::fs::create_dir_all(build.join("CMakeFiles")).unwrap();
        std::fs::create_dir_all(build.join("bin")).unwrap();
        std::fs::write(build.join("bin/llama-server"), b"precious").unwrap();
        clear_stale_build_cache(&repo, &mut progress);
        assert!(!build.join("CMakeCache.txt").exists(), "cache file cleared");
        assert!(!build.join("CMakeFiles").exists(), "cmake state cleared");
        assert!(
            build.join("bin/llama-server").exists(),
            "built binaries must survive a cache clear"
        );
        assert!(log.borrow()[0].contains("checkout has moved"), "{:?}", log.borrow());

        // Cache matching the real location -> untouched.
        std::fs::create_dir_all(&build).unwrap();
        std::fs::write(
            build.join("CMakeCache.txt"),
            format!("CMAKE_HOME_DIRECTORY:INTERNAL={}\n", repo.canonicalize().unwrap().display()),
        )
        .unwrap();
        clear_stale_build_cache(&repo, &mut progress);
        assert!(build.exists(), "matching cache must survive");
    }
}
