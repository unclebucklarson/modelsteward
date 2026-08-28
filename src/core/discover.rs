//! Discovery: which llama.cpp installations exist on this machine, and which
//! compute devices the chosen one can see.
//!
//! An "install" is a `llama-server` binary. We probe candidates from the
//! user's PATH, a set of conventional build locations, and any manually
//! configured paths, then interrogate each with `--version`. Devices come
//! from `--list-devices` — asked of the *binary*, not the OS, because what
//! matters is what that build can actually use (a CUDA card is invisible to
//! a CPU-only build). GPU state is always a `Vec`; see CLAUDE.md.

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize)]
pub struct LlamaInstall {
    /// Absolute path to the `llama-server` binary.
    pub server_path: PathBuf,
    /// Build number from `--version` (e.g. 10216). `None` if the binary
    /// didn't answer — still listed, so a broken install is visible.
    pub build: Option<u64>,
    /// Commit hash from `--version` (e.g. "876a43211").
    pub commit: Option<String>,
    /// The full "built with …" line, verbatim (compiler, target, vendor
    /// annotations like unsloth's).
    pub built_with: Option<String>,
    /// Backend libraries sitting next to the binary (cuda, vulkan, cpu, …).
    /// Empty for static builds — absence of evidence only.
    pub backends: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Device {
    /// Backend-qualified id as llama.cpp names it: "CUDA0", "Vulkan1", …
    pub id: String,
    pub name: String,
    pub total_mib: u64,
    pub free_mib: u64,
}

/// One physical GPU, deduped across backend views: the same card appears
/// as CUDA0 AND Vulkan0 (with different memory figures — CUDA reports
/// usable, Vulkan the raw heap), and an iGPU advertises borrowed system
/// RAM as if it were VRAM. Summing the raw device list triples reality
/// (user-caught 2026-08-26 reading the findings report).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PhysicalGpu {
    pub name: String,
    /// The smallest figure across backend views — the conservative
    /// usable number.
    pub vram_mib: u64,
    /// Backend ids that are views of this card ("CUDA0", "Vulkan0", …).
    pub ids: Vec<String>,
    /// Integrated / shared-memory device: its "VRAM" is system RAM in a
    /// costume and must not count toward serving capacity.
    pub shared_memory: bool,
}

/// Marker heuristic for shared-memory graphics; wrong in exotic cases,
/// but wrong in the safe direction (a discrete card misread as shared
/// only shrinks advice thresholds, never inflates them).
fn is_shared_memory_gpu(name: &str) -> bool {
    ["UHD Graphics", "Iris", "Radeon(TM) Graphics", "Radeon Graphics", "llvmpipe"]
        .iter()
        .any(|m| name.contains(m))
}

/// Group backend device views into physical GPUs by name, first-seen order.
pub fn physical_gpus(devices: &[Device]) -> Vec<PhysicalGpu> {
    let mut out: Vec<PhysicalGpu> = Vec::new();
    for d in devices {
        if let Some(g) = out.iter_mut().find(|g| g.name == d.name) {
            g.ids.push(d.id.clone());
            g.vram_mib = g.vram_mib.min(d.total_mib);
        } else {
            out.push(PhysicalGpu {
                name: d.name.clone(),
                vram_mib: d.total_mib,
                ids: vec![d.id.clone()],
                shared_memory: is_shared_memory_gpu(&d.name),
            });
        }
    }
    out
}

/// The VRAM figure serving advice should use: the largest DEDICATED
/// physical GPU. Falls back to the largest anything only when no
/// dedicated card exists at all.
pub fn advice_vram_mib(devices: &[Device]) -> u64 {
    let gpus = physical_gpus(devices);
    gpus.iter()
        .filter(|g| !g.shared_memory)
        .map(|g| g.vram_mib)
        .max()
        .or_else(|| gpus.iter().map(|g| g.vram_mib).max())
        .unwrap_or(0)
}

/// Conventional places a self-built or vendored llama-server lives, relative
/// to $HOME. Checked in addition to PATH and manual paths.
const HOME_CANDIDATES: &[&str] = &[
    "src/llama.cpp/build/bin/llama-server",
    "llama.cpp/build/bin/llama-server",
    ".unsloth/llama.cpp/llama-server",
];

/// Find llama-server binaries: manual paths first (they outrank guesses),
/// then PATH, then conventional build locations. Deduplicated by canonical
/// path; a path that isn't an executable file is skipped, but a binary that
/// fails `--version` is kept with `build: None`.
pub fn find_installs(manual: &[PathBuf]) -> Vec<LlamaInstall> {
    let mut candidates: Vec<PathBuf> = manual.to_vec();
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            candidates.push(dir.join("llama-server"));
        }
    }
    if let Some(home) = std::env::home_dir() {
        for rel in HOME_CANDIDATES {
            candidates.push(home.join(rel));
        }
    }
    // The app-managed checkout's build and its pinnable archives are
    // installs like any other — offered, never forced.
    candidates.push(crate::core::managed::managed_server_bin());
    for (_, server) in crate::core::managed::list_archives() {
        candidates.push(server);
    }

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for c in candidates {
        let Ok(real) = c.canonicalize() else { continue };
        if !real.is_file() || !seen.insert(real.clone()) {
            continue;
        }
        out.push(probe_install(&real));
    }
    out
}

fn probe_install(server: &Path) -> LlamaInstall {
    let version_output = Command::new(server)
        .arg("--version")
        .output()
        .ok()
        .map(|o| {
            let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
            s.push_str(&String::from_utf8_lossy(&o.stderr));
            s
        })
        .unwrap_or_default();
    let (build, commit, built_with) = parse_version_output(&version_output);
    LlamaInstall {
        backends: sibling_backends(server),
        server_path: server.to_path_buf(),
        build,
        commit,
        built_with,
    }
}

/// Parse `llama-server --version` output. Two dialects exist:
/// ```text
/// version: 10216 (876a43211)                        # up to ~b10216
/// version: 0.1.0-dev (build 10454, commit 4df29be4f) # newer builds
/// built with GNU 15.2.0 for Linux x86_64
/// ```
fn parse_version_output(out: &str) -> (Option<u64>, Option<String>, Option<String>) {
    let mut build = None;
    let mut commit = None;
    let mut built_with = None;
    for line in out.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("version:") {
            let rest = rest.trim();
            if let Some(bpos) = rest.find("build ") {
                // New dialect: numbers live in "(build N, commit H)".
                build = rest[bpos + "build ".len()..]
                    .split([',', ')', ' '])
                    .next()
                    .and_then(|n| n.parse().ok());
                if let Some(cpos) = rest.find("commit ") {
                    commit = rest[cpos + "commit ".len()..]
                        .split([',', ')', ' '])
                        .next()
                        .map(str::to_string);
                }
            } else {
                let mut parts = rest.splitn(2, ' ');
                build = parts.next().and_then(|n| n.parse().ok());
                commit = parts
                    .next()
                    .map(|h| h.trim().trim_start_matches('(').trim_end_matches(')').to_string());
            }
        } else if let Some(rest) = line.strip_prefix("built with") {
            built_with = Some(rest.trim().to_string());
        }
    }
    (build, commit, built_with)
}

/// Backend names from `libggml-<backend>.so*` files next to the binary.
/// "base" is plumbing, not a backend, and is excluded.
fn sibling_backends(server: &Path) -> Vec<String> {
    let Some(dir) = server.parent() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut backends: Vec<String> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let rest = name.strip_prefix("libggml-")?;
            let backend = rest.split(".so").next()?;
            (!backend.is_empty() && backend != "base").then(|| backend.to_string())
        })
        .collect();
    backends.sort();
    backends.dedup();
    backends
}

/// Build number of one llama-server binary (`--version` probe).
pub fn build_of(server: &Path) -> Option<u64> {
    let out = Command::new(server).arg("--version").output().ok()?;
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    parse_version_output(&s).0
}

/// Live VRAM (free, total) in MiB for the primary NVIDIA card, via
/// nvidia-smi — cheap enough for a 2s poll, and unlike `--list-devices`
/// it doesn't initialize CUDA. `None` when nvidia-smi is absent/fails.
pub fn nvidia_vram_mib() -> Option<(u64, u64)> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let line = String::from_utf8_lossy(&out.stdout);
    let line = line.lines().next()?;
    let (used, total) = line.split_once(',')?;
    let used: u64 = used.trim().parse().ok()?;
    let total: u64 = total.trim().parse().ok()?;
    Some((total.saturating_sub(used), total))
}

/// Ask an install what devices it can see. Errors (bad binary, no devices)
/// yield an empty list — callers render that as "CPU only / unknown".
pub fn list_devices(server: &Path) -> Vec<Device> {
    let Some(output) = Command::new(server)
        .arg("--list-devices")
        .output()
        .ok()
        .map(|o| {
            let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
            s.push_str(&String::from_utf8_lossy(&o.stderr));
            s
        })
    else {
        return Vec::new();
    };
    parse_device_list(&output)
}

/// Parse `--list-devices` output:
/// ```text
/// Available devices:
///   CUDA0: NVIDIA GeForce RTX 3090 Ti (24111 MiB, 23328 MiB free)
///   Vulkan1: Intel(R) UHD Graphics 770 (ADL-S GT1) (48012 MiB, 43210 MiB free)
/// ```
/// The name itself may contain parentheses, so the memory figures are taken
/// from the *last* parenthesised group.
fn parse_device_list(out: &str) -> Vec<Device> {
    out.lines()
        .filter_map(|line| {
            let line = line.trim();
            let (id, rest) = line.split_once(": ")?;
            if id.contains(' ') || id.is_empty() {
                return None; // prose line, not a device id
            }
            let open = rest.rfind('(')?;
            let mem = rest[open + 1..].trim_end_matches(')');
            let name = rest[..open].trim();
            let (total, free) = mem.split_once(',')?;
            let total_mib = total.trim().strip_suffix(" MiB")?.parse().ok()?;
            let free_mib = free
                .trim()
                .strip_suffix(" MiB free")?
                .parse()
                .ok()?;
            Some(Device {
                id: id.to_string(),
                name: name.to_string(),
                total_mib,
                free_mib,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(id: &str, name: &str, mib: u64) -> Device {
        Device {
            id: id.into(),
            name: name.into(),
            total_mib: mib,
            free_mib: mib,
        }
    }

    #[test]
    fn physical_gpus_dedupe_backend_views_and_flag_shared_memory() {
        // This machine's real shape: one 3090 Ti seen by CUDA and Vulkan
        // (different figures), plus an iGPU advertising ~48GB of borrowed
        // system RAM.
        let devices = vec![
            dev("CUDA0", "NVIDIA GeForce RTX 3090 Ti", 24_111),
            dev("Vulkan0", "NVIDIA GeForce RTX 3090 Ti", 24_564),
            dev("Vulkan1", "Intel(R) UHD Graphics 770 (ADL-S GT1)", 48_012),
        ];
        let gpus = physical_gpus(&devices);
        assert_eq!(gpus.len(), 2, "three views, two physical devices");
        assert_eq!(gpus[0].vram_mib, 24_111, "conservative (min) figure wins");
        assert_eq!(gpus[0].ids, vec!["CUDA0", "Vulkan0"]);
        assert!(!gpus[0].shared_memory);
        assert!(gpus[1].shared_memory, "iGPU flagged");
        assert_eq!(advice_vram_mib(&devices), 24_111, "advice never sees the phantom heap");

        // Vulkan-only box (the latent-bug case): iGPU listed FIRST must
        // still lose to the discrete card.
        let vulkan_only = vec![
            dev("Vulkan0", "Intel(R) UHD Graphics 770", 48_012),
            dev("Vulkan1", "AMD Radeon RX 7900 XTX", 24_560),
        ];
        assert_eq!(advice_vram_mib(&vulkan_only), 24_560);
        // All-shared fallback: better a shared figure than zero.
        let igpu_only = vec![dev("Vulkan0", "Intel(R) UHD Graphics 770", 48_012)];
        assert_eq!(advice_vram_mib(&igpu_only), 48_012);
    }

    #[test]
    fn parses_version_output() {
        let out = "version: 10216 (876a43211)\nbuilt with GNU 15.2.0 for Linux x86_64\n";
        let (build, commit, built_with) = parse_version_output(out);
        assert_eq!(build, Some(10216));
        assert_eq!(commit.as_deref(), Some("876a43211"));
        assert_eq!(built_with.as_deref(), Some("GNU 15.2.0 for Linux x86_64"));
    }

    #[test]
    fn parses_vendored_version_output() {
        let out = "version: 10360 (90e6a9131)\nbuilt with GNU 11.4.0 for Linux x86_64 (Compiled by the Unsloth team)\n";
        let (build, _, built_with) = parse_version_output(out);
        assert_eq!(build, Some(10360));
        assert!(built_with.unwrap().contains("Unsloth"));
    }

    #[test]
    fn parses_the_new_version_dialect() {
        // Banner format changed upstream around b104xx — found live when
        // the user's first advisor-driven rebuild "lost" its version.
        let out = "version: 0.1.0-dev (build 10454, commit 4df29be4f)\nbuilt with GNU 15.2.0 for Linux x86_64\n";
        let (build, commit, built_with) = parse_version_output(out);
        assert_eq!(build, Some(10454));
        assert_eq!(commit.as_deref(), Some("4df29be4f"));
        assert!(built_with.unwrap().contains("GNU 15.2.0"));
    }

    #[test]
    fn version_garbage_yields_nones_not_panic() {
        let (build, commit, built_with) = parse_version_output("segfault lol");
        assert_eq!((build, commit, built_with), (None, None, None));
    }

    #[test]
    fn parses_device_list_including_parenthesised_names() {
        let out = "\
Available devices:
  CUDA0: NVIDIA GeForce RTX 3090 Ti (24111 MiB, 23328 MiB free)
  Vulkan0: NVIDIA GeForce RTX 3090 Ti (24564 MiB, 23328 MiB free)
  Vulkan1: Intel(R) UHD Graphics 770 (ADL-S GT1) (48012 MiB, 43210 MiB free)
";
        let devices = parse_device_list(out);
        assert_eq!(devices.len(), 3);
        assert_eq!(devices[0].id, "CUDA0");
        assert_eq!(devices[0].total_mib, 24111);
        assert_eq!(devices[2].id, "Vulkan1");
        assert_eq!(devices[2].name, "Intel(R) UHD Graphics 770 (ADL-S GT1)");
        assert_eq!(devices[2].free_mib, 43210);
    }

    #[test]
    fn device_prose_lines_are_ignored() {
        assert!(parse_device_list("Available devices:\nno devices found\n").is_empty());
    }
}
