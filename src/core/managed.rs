//! The app-managed llama.cpp checkout (decision A, 2026-08-27, revised
//! from binary-archiving-only after user pushback — see ROADMAP M8 #5).
//!
//! Why it exists: a crates.io user has no `~/src/llama.cpp` — this
//! module is what bootstraps llama.cpp for non-experts at all, and it
//! closes the loop the scorecard opened: freshness -> fetch -> build a
//! TAGGED candidate -> archive its binaries -> pin or roll back, no git
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
    crate::core::settings::xdg_dir("XDG_DATA_HOME", ".local/share")
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

/// Check out the tag for `build` (detached HEAD — the managed tree is
/// always at an exact tagged release, never tracking master). Callers
/// fetch first (build_release does); the fetch used to live here too
/// and every build paid for it twice.
pub fn checkout_release(build: u64, progress: &mut dyn FnMut(String)) -> Result<()> {
    crate::core::advisor::run_steps(
        &[git(&["checkout", "--force", &format!("b{build}")])],
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
pub fn build_tree(
    c: &crate::core::advisor::BuildCheck,
    sel: crate::core::advisor::BackendSelection,
    progress: &mut dyn FnMut(String),
) -> Result<()> {
    crate::core::advisor::clear_stale_build_cache(&checkout_dir(), progress);
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
    // "." and ".." pass the character test but escape or alias the
    // archive dir (review catch 2026-08-28); a leading dot also hides
    // the entry from listings.
    anyhow::ensure!(
        !label.starts_with('.'),
        "label can't start with a dot"
    );
    let dst = archive_dir().join(label);
    anyhow::ensure!(
        !dst.join("llama-server").exists(),
        "an archive named {label:?} already exists — pick another label \
         (overwriting could swap the binary behind an existing pin)"
    );
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

/// Which archives auto-prune may delete (design session 2026-08-30,
/// Scott chose keep-5-configurable): among BUILD-NUMBERED archives,
/// newest-first, everything past `keep` — except the one currently
/// serving. Custom-labeled archives (no parseable build number) are a
/// deliberate act and never pruned. keep == 0 means unlimited.
/// Pure over the list; deletion is the thin part.
pub fn prune_candidates(
    archives: &[Archive],
    keep: usize,
    serving: Option<&Path>,
) -> Vec<String> {
    if keep == 0 {
        return Vec::new();
    }
    archives
        .iter()
        .filter(|a| a.build.is_some())
        .skip(keep)
        .filter(|a| serving != Some(a.server.as_path()))
        .map(|a| a.label.clone())
        .collect()
}

/// Permanently delete one archived build. Refuses anything that isn't
/// a real archive directory (path-traversal labels never reach the fs).
pub fn delete_archive(label: &str) -> Result<()> {
    anyhow::ensure!(
        !label.is_empty() && !label.contains('/') && !label.contains("..") && !label.starts_with('.'),
        "not a valid archive label: {label:?}"
    );
    let dir = archive_dir().join(label);
    anyhow::ensure!(
        dir.join("llama-server").is_file(),
        "no archived build named {label:?}"
    );
    std::fs::remove_dir_all(&dir).map_err(Into::into)
}

/// Apply the retention policy; returns the labels actually deleted.
pub fn prune_archives(keep: usize, serving: Option<&Path>) -> Vec<String> {
    prune_candidates(&list_archives(), keep, serving)
        .into_iter()
        .filter(|l| delete_archive(l).is_ok())
        .collect()
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

/// One managed build at a time: the daily auto-build thread and the
/// Build Advisor button share the same checkout, and two concurrent
/// `git checkout --force` + cmake runs would archive an interleaved
/// binary (review catch 2026-08-28). Try-lock; the loser narrates and
/// walks away.
static BUILD_LOCK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub struct BuildGuard;
impl Drop for BuildGuard {
    fn drop(&mut self) {
        BUILD_LOCK.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

pub fn try_lock_build() -> Option<BuildGuard> {
    BUILD_LOCK
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
        )
        .is_ok()
        .then_some(BuildGuard)
}

/// The whole release pipeline in one place — clone if needed, fetch
/// tags, check out the newest release, build, archive — so the button,
/// the auto-build poller, and any future CLI produce identical
/// artifacts (review catch 2026-08-28: two hand-rolled copies had
/// already drifted). Holds the build lock for the duration.
pub fn build_release(
    c: &crate::core::advisor::BuildCheck,
    sel: crate::core::advisor::BackendSelection,
    progress: &mut dyn FnMut(String),
) -> Result<u64> {
    let Some(_guard) = try_lock_build() else {
        anyhow::bail!("a managed build is already running — let it finish");
    };
    ensure_clone(progress)?;
    fetch_tags(progress)?;
    let build = newest_known_build()
        .ok_or_else(|| anyhow::anyhow!("no bNNNN release tags found"))?;
    // Already built + archived WITH THE SAME BACKENDS -> nothing to do.
    // (An earlier plain existence check silently discarded the caller's
    // backend selection — a CUDA rebuild of a CPU-only archive reported
    // success without ever building; review catch 2026-08-28.)
    let wanted = |backends: &[String]| {
        let has = |n: &str| backends.iter().any(|b| b == n);
        has("cuda") == sel.cuda && has("vulkan") == sel.vulkan && has("hip") == sel.hip
    };
    if let Some(a) = list_archives()
        .iter()
        .find(|a| a.build == Some(build))
    {
        if wanted(&crate::core::discover::sibling_backends(&a.server)) {
            progress(format!(
                "b{build} is the newest release and it's already built + archived \
                 with these backends — nothing new upstream"
            ));
            return Ok(build);
        }
        progress(format!(
            "b{build} exists as {} but with different backends — building the \
             requested variant",
            a.label
        ));
    }
    progress(format!("newest release: b{build} — checking out + building"));
    checkout_release(build, progress)?;
    build_tree(c, sel, progress)?;
    // Same-number variants archive under a backend-suffixed label so
    // both stay pinnable side by side.
    let base_label = format!("b{build}");
    let label = if archive_dir().join(&base_label).join("llama-server").exists() {
        let mut names = Vec::new();
        if sel.cuda { names.push("cuda"); }
        if sel.vulkan { names.push("vulkan"); }
        if sel.hip { names.push("hip"); }
        if names.is_empty() { names.push("cpu"); }
        format!("{base_label}-{}", names.join("-"))
    } else {
        base_label
    };
    archive_from(&checkout_dir().join("build/bin"), &label, progress)?;
    Ok(build)
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
    fn prune_keeps_newest_serving_and_custom_labels() {
        let a = |label: &str, build: Option<u64>| Archive {
            label: label.into(),
            build,
            server: std::path::PathBuf::from(format!("/x/{label}/llama-server")),
        };
        // Newest-first numbered, then a custom label — list_archives order.
        let archives = vec![
            a("b10697", Some(10697)),
            a("b10687", Some(10687)),
            a("b10680", Some(10680)),
            a("b10679", Some(10679)),
            a("b10675", Some(10675)),
            a("fast-mmq", None),
        ];
        let serving = std::path::PathBuf::from("/x/b10679/llama-server");
        // keep 2: b10680 and b10675 are past the line; b10679 survives
        // because it is serving; fast-mmq survives because it is custom.
        assert_eq!(
            prune_candidates(&archives, 2, Some(&serving)),
            vec!["b10680".to_string(), "b10675".to_string()]
        );
        // keep 0 = unlimited: nothing is ever deleted.
        assert!(prune_candidates(&archives, 0, Some(&serving)).is_empty());
        assert!(prune_candidates(&archives, 99, None).is_empty());
    }

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
