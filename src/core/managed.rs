//! The app-managed llama.cpp checkout (decision A, 2026-08-27, revised
//! from binary-archiving-only after user pushback — see ROADMAP M8 #5).
//!
//! Why it exists: a crates.io user has no `~/src/llama.cpp` — this
//! module is what bootstraps llama.cpp for non-experts at all, and it
//! closes the loop the scorecard opened: freshness → fetch → build a
//! TAGGED candidate → archive its binaries → pin or roll back, no git
//! knowledge required.
//!
//! Boundary rules (settled):
//! - The clone lives in the app's data dir. The USER'S checkout is
//!   never touched by anything here.
//! - The managed install is offered through the normal installs picker
//!   and pin buttons — never forced.
//! - Everything here is deterministic (git + cmake via advisor's
//!   engine). The AI advisor may triage WHEN a build is worth making;
//!   it never drives HOW.
//! - Binary archiving lives INSIDE this design: every completed build's
//!   binaries are copied to builds/bN/ before the next build can
//!   overwrite build/bin — rollback is a pin, never a rebuild.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub const REPO_URL: &str = "https://github.com/ggml-org/llama.cpp";

/// ~/.local/share/modelsteward (XDG_DATA_HOME respected) — code and
/// binaries, distinct from the state dir's measurements and logs.
pub fn data_dir() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        // A snap-redirected XDG var points into a per-revision dir that
        // the next snap update deletes — never persist a git checkout
        // or build archive there (see settings::real_home).
        .filter(|p| !p.components().any(|c| c.as_os_str() == "snap"))
        .unwrap_or_else(|| crate::core::settings::real_home().join(".local/share"))
        .join("modelsteward")
}

pub fn checkout_dir() -> PathBuf {
    data_dir().join("llama.cpp")
}

pub fn archive_dir() -> PathBuf {
    data_dir().join("builds")
}

/// The managed checkout's freshly-built server binary (may not exist).
pub fn managed_server_bin() -> PathBuf {
    checkout_dir().join("build/bin/llama-server")
}

pub fn checkout_present() -> bool {
    checkout_dir().join(".git").is_dir()
}

fn git(args: &[&str]) -> (String, Vec<String>) {
    let mut v = vec!["-C".to_string(), checkout_dir().display().to_string()];
    v.extend(args.iter().map(|s| s.to_string()));
    ("git".to_string(), v)
}

/// Clone if absent (a shallow-ish full clone; tags are how builds are
/// addressed, so they must come along). No-op when present.
pub fn ensure_clone(progress: &mut dyn FnMut(String)) -> Result<()> {
    if checkout_present() {
        return Ok(());
    }
    std::fs::create_dir_all(data_dir())?;
    progress(format!("cloning {REPO_URL} into {}", checkout_dir().display()));
    crate::core::advisor::run_steps(
        &[(
            "git".to_string(),
            vec![
                "clone".to_string(),
                REPO_URL.to_string(),
                checkout_dir().display().to_string(),
            ],
        )],
        progress,
    )
}

/// Fetch tags so newest_known_build() sees current upstream releases.
pub fn fetch_tags(progress: &mut dyn FnMut(String)) -> Result<()> {
    crate::core::advisor::run_steps(&[git(&["fetch", "--tags", "origin"])], progress)
}

/// Fetch + check out the tag for `build` (detached HEAD — the managed
/// tree is always at an exact tagged release, never tracking master).
pub fn checkout_build(build: u64, progress: &mut dyn FnMut(String)) -> Result<()> {
    crate::core::advisor::run_steps(
        &[
            git(&["fetch", "--tags", "origin"]),
            git(&["checkout", "--force", &format!("b{build}")]),
        ],
        progress,
    )
    .with_context(|| format!("checking out b{build}"))
}

/// The newest build tag the checkout knows (after a fetch): tags are
/// bNNNN, so the max numeric suffix is the newest release.
pub fn newest_known_build() -> Option<u64> {
    let out = std::process::Command::new("git")
        .args(["-C", &checkout_dir().display().to_string(), "tag", "--list", "b*"])
        .output()
        .ok()?;
    parse_newest_tag(&String::from_utf8_lossy(&out.stdout))
}

/// Pure over `git tag --list b*` output, for tests.
pub fn parse_newest_tag(tags: &str) -> Option<u64> {
    tags.lines()
        .filter_map(|l| l.trim().strip_prefix('b')?.parse::<u64>().ok())
        .max()
}

/// Build the managed checkout with the same backend selection logic the
/// guided rebuild uses (no git step — the tree is tag-pinned).
pub fn build(
    c: &crate::core::advisor::BuildCheck,
    sel: crate::core::advisor::BackendSelection,
    progress: &mut dyn FnMut(String),
) -> Result<()> {
    let repo = checkout_dir().display().to_string();
    crate::core::advisor::run_steps(
        &crate::core::advisor::build_commands(&repo, c, sel),
        progress,
    )
}

/// Copy the just-built binaries into builds/bN/ — the pinnable archive.
/// build/bin is overwritten by every rebuild; the archive is what makes
/// rollback a pin instead of a rebuild.
pub fn archive_build(build: u64, progress: &mut dyn FnMut(String)) -> Result<PathBuf> {
    archive_from(&checkout_dir().join("build/bin"), &format!("b{build}"), progress)
}

/// Archive ANY checkout's built binaries under a user-chosen label
/// (rung 2 of the checkout ladder, user decision 2026-08-28): custom
/// builds become pinnable side by side with release archives. CAVEAT
/// (rung 3 parked): measurements key builds by NUMBER — two variants
/// of the same build are indistinguishable to bench/history/scorecard.
pub fn archive_from(
    src: &Path,
    label: &str,
    progress: &mut dyn FnMut(String),
) -> Result<PathBuf> {
    anyhow::ensure!(
        !label.is_empty() && label.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)),
        "label must be filesystem-plain (letters, digits, -_.)"
    );
    let dst = archive_dir().join(label);
    anyhow::ensure!(
        src.join("llama-server").is_file(),
        "no built llama-server at {}",
        src.display()
    );
    std::fs::create_dir_all(&dst)?;
    let mut n = 0u32;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
            n += 1;
        }
    }
    progress(format!("archived {n} files to {}", dst.display()));
    Ok(dst)
}

/// One pinnable archived build.
#[derive(Debug, Clone, PartialEq)]
pub struct Archive {
    /// Directory name: bNNNN for release archives, free label otherwise.
    pub label: String,
    /// Parsed build number when the label carries one.
    pub build: Option<u64>,
    pub server: PathBuf,
}

/// Archived builds: release tags newest first, then labeled variants.
pub fn list_archives() -> Vec<Archive> {
    list_archives_in(&archive_dir())
}

/// Pure-ish core of list_archives, testable against a temp dir.
pub fn list_archives_in(dir: &Path) -> Vec<Archive> {
    let mut out: Vec<Archive> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| {
                    let e = e.ok()?;
                    let label = e.file_name().to_str()?.to_string();
                    let server = e.path().join("llama-server");
                    server.is_file().then(|| Archive {
                        build: label.strip_prefix('b').and_then(|b| b.parse().ok()),
                        label,
                        server,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| match (a.build, b.build) {
        (Some(x), Some(y)) => y.cmp(&x),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.label.cmp(&b.label),
    });
    out
}

/// What the Settings/Build-Advisor surface shows about the managed side.
#[derive(Debug, Clone, Default)]
pub struct ManagedStatus {
    pub present: bool,
    /// Build of the binary sitting in the managed build tree, if built.
    pub built: Option<u64>,
    pub archives: Vec<Archive>,
}

pub fn status() -> ManagedStatus {
    ManagedStatus {
        present: checkout_present(),
        built: managed_server_bin()
            .is_file()
            .then(|| crate::core::discover::build_of(&managed_server_bin()))
            .flatten(),
        archives: list_archives(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newest_tag_parses_build_tags_only() {
        assert_eq!(parse_newest_tag("b10454\nb10630\nb9999\n"), Some(10_630));
        assert_eq!(parse_newest_tag("v1.0\nmaster\n"), None);
        assert_eq!(parse_newest_tag(""), None);
    }

    #[test]
    fn archives_list_by_build_and_require_a_server_binary() {
        let dir = tempfile::tempdir().unwrap();
        for (name, with_bin) in [("b10454", true), ("b10630", true), ("b7", false), ("junk", true)]
        {
            let d = dir.path().join(name);
            std::fs::create_dir_all(&d).unwrap();
            if with_bin {
                std::fs::write(d.join("llama-server"), b"x").unwrap();
            }
        }
        let got = list_archives_in(dir.path());
        let labels: Vec<&str> = got.iter().map(|a| a.label.as_str()).collect();
        // b7 (no binary) is excluded; releases newest-first, then labels.
        assert_eq!(labels, vec!["b10630", "b10454", "junk"]);
        assert_eq!(got[0].build, Some(10_630));
        assert!(got[0].server.ends_with("b10630/llama-server"));
        assert_eq!(got[2].build, None, "labeled variant carries no number");
    }
}
