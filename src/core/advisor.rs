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
    /// e.g. "86" for compute capability 8.6.
    pub cuda_arch: Option<String>,
    pub nvcc: Option<String>,
    pub cmake: bool,
    /// Models whose stored failure looks like "needs a newer build".
    pub locked_models: Vec<String>,
}

/// Locate the git checkout for a llama-server at `<repo>/build/bin/llama-server`.
pub fn repo_of(server_bin: &Path) -> Option<PathBuf> {
    let repo = server_bin.parent()?.parent()?.parent()?;
    repo.join(".git").exists().then(|| repo.to_path_buf())
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// "b10366" or "b10366-14-gabc" → 10366. Pure for testing.
pub fn parse_build_tag(tag: &str) -> Option<u64> {
    tag.trim()
        .strip_prefix('b')?
        .split(['-', '.'])
        .next()?
        .parse()
        .ok()
}

/// nvidia-smi compute_cap "8.6" → cmake arch "86". Pure for testing.
pub fn parse_compute_cap(s: &str) -> Option<String> {
    let first = s.lines().next()?.trim();
    let (major, minor) = first.split_once('.')?;
    let major: u32 = major.trim().parse().ok()?;
    let minor: u32 = minor.trim().parse().ok()?;
    Some(format!("{major}{minor}"))
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
/// origin (tolerated failure → upstream fields stay None). Run on a worker
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
        cmake: run("cmake", &["--version"]).is_some(),
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
    // Fetch is the only network step; failure leaves upstream unknown.
    let fetched = Command::new("git")
        .args(["-C", &repo_s, "fetch", "--quiet", "origin", "master"])
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
            format!("Your llama.cpp binary is {} builds behind upstream (b{cur} → b{up})", up - cur),
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
            "Couldn't contact the git remote (offline?) — freshness unknown.".into(),
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
    match (&c.cuda_arch, &c.nvcc, c.cmake) {
        (Some(arch), Some(_), true) => out.push((
            "Toolchain ready to build".into(),
            format!("CUDA compiler present; GPU compute capability {arch}."),
        )),
        (Some(_), None, _) => out.push((
            "CUDA compiler (nvcc) not found".into(),
            "Install the CUDA toolkit to rebuild with GPU support.".into(),
        )),
        (_, _, false) => out.push((
            "cmake not found".into(),
            "Install cmake to rebuild llama.cpp.".into(),
        )),
        _ => {}
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
/// section so nothing is hidden. Pure over the check.
pub fn rebuild_commands(c: &BuildCheck) -> Vec<(String, Vec<String>)> {
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
    let mut configure: Vec<String> = Vec::new();
    configure.push("-S".into());
    configure.push(repo.clone());
    configure.push("-B".into());
    configure.push(format!("{repo}/build"));
    configure.push("-DGGML_CUDA=ON".into());
    if let Some(arch) = &c.cuda_arch {
        configure.push(format!("-DCMAKE_CUDA_ARCHITECTURES={arch}"));
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
pub fn run_rebuild(c: &BuildCheck, progress: &mut dyn FnMut(String)) -> Result<()> {
    anyhow::ensure!(c.repo.is_some(), "no git checkout to rebuild");
    for (cmd, args) in rebuild_commands(c) {
        progress(format!("$ {cmd} {}", args.join(" ")));
        let mut child = Command::new(&cmd)
            .args(&args)
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
        assert!(v.iter().any(|(h, _)| h.contains("Toolchain ready")));
        assert!(
            v.iter().all(|(h, _)| !h.contains("-D")),
            "headlines never contain cmake flags"
        );
    }

    #[test]
    fn rebuild_commands_are_ff_only_and_arch_pinned() {
        let c = BuildCheck {
            repo: Some(PathBuf::from("/home/u/src/llama.cpp")),
            cuda_arch: Some("86".into()),
            ..Default::default()
        };
        let cmds = rebuild_commands(&c);
        assert_eq!(cmds[0].0, "git");
        assert!(cmds[0].1.contains(&"--ff-only".to_string()), "never clobbers local work");
        assert!(cmds[1].1.iter().any(|a| a == "-DCMAKE_CUDA_ARCHITECTURES=86"));
        assert!(cmds[2].1.iter().any(|a| a == "Release"));
    }
}
