//! Durable file writing, and reads that tell the truth when they fail.
//!
//! Written 2026-08-31 in response to the code review, which found that
//! **nothing in this codebase was written atomically** except
//! `opencode.json`: eleven plain `fs::write` call sites, including the
//! measurement store the whole product is built on. A crash, an ENOSPC,
//! or an OOM-kill mid-write truncates the file, and every reader used
//! `.ok().unwrap_or_default()` — so a truncated file read as *empty*,
//! and the next write persisted that emptiness permanently.
//!
//! Two rules here, both learned the hard way:
//!
//! 1. **Write to a temp file in the same directory, fsync it, rename.**
//!    The rename is atomic within a filesystem, so a reader sees either
//!    the whole old file or the whole new one. The fsync matters: a
//!    durable rename over non-durable contents can leave an empty file
//!    after a power loss.
//! 2. **A read failure is not an empty file.** Missing is silent and
//!    default; anything else is loud, and the unreadable original is
//!    rescued beside itself rather than overwritten.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Write `contents` to `path` atomically and durably.
///
/// Preserves the destination's permissions when it already exists (the
/// review found writes widening a 0600 config to 0664), and follows a
/// symlink to its target instead of replacing the link (dotfiles
/// managers like stow/chezmoi symlink these configs; replacing the link
/// silently diverges the repo copy).
pub fn write_atomic(path: &Path, contents: &str) -> Result<()> {
    let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let dir = target
        .parent()
        .ok_or_else(|| anyhow::anyhow!("{} has no parent directory", target.display()))?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

    // Unique temp name: two processes writing the same file must not
    // share a staging path (the old fixed `.lcc.tmp` could interleave).
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = dir.join(format!(
        ".{}.{}.{}.tmp",
        target
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "file".into()),
        std::process::id(),
        stamp
    ));

    let write_and_sync = || -> Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(contents.as_bytes())
            .with_context(|| format!("writing {}", tmp.display()))?;
        // Durability: without this the rename can land while the bytes
        // have not, leaving an empty file after a power loss.
        f.sync_all()
            .with_context(|| format!("flushing {}", tmp.display()))?;
        Ok(())
    };
    if let Err(e) = write_and_sync() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }

    // Carry the destination's mode across, so a 0600 config holding API
    // keys does not come back world-readable.
    #[cfg(unix)]
    if let Ok(meta) = std::fs::metadata(&target) {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(meta.permissions().mode()));
    }

    if let Err(e) = std::fs::rename(&tmp, &target) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("replacing {}", target.display()));
    }
    // Best-effort directory fsync so the rename itself is durable.
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

/// What a read of a state file found.
#[derive(Debug, PartialEq)]
pub enum Loaded<T> {
    /// No file yet — start from defaults, silently.
    Missing,
    /// Parsed cleanly.
    Ok(T),
    /// The file exists but could not be read or parsed. Carries the
    /// reason, so callers can be loud instead of pretending it was
    /// empty.
    Damaged(String),
}

/// Read + parse JSON, distinguishing "missing" from "damaged".
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Loaded<T> {
    match std::fs::read_to_string(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Loaded::Missing,
        // A permission change, an EIO, an NFS hiccup: NOT an empty file.
        Err(e) => Loaded::Damaged(format!("unreadable: {e}")),
        Ok(text) => match serde_json::from_str(&text) {
            Ok(v) => Loaded::Ok(v),
            Err(e) => Loaded::Damaged(format!("unparseable: {e}")),
        },
    }
}

/// Move a damaged file aside so the next write cannot destroy it.
/// Returns the rescue path when one was made.
pub fn rescue(path: &Path) -> Option<PathBuf> {
    if !path.exists() {
        return None;
    }
    let rescue = path.with_extension(format!(
        "{}.corrupt",
        path.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default()
    ));
    std::fs::rename(path, &rescue).ok().map(|_| rescue)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_replaces_content_and_leaves_no_temp_files() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("state.json");
        write_atomic(&p, "{\"a\":1}").unwrap();
        write_atomic(&p, "{\"a\":2}").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "{\"a\":2}");
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_preserves_mode_and_does_not_replace_a_symlink() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        // Mode: a 0600 config must not come back world-readable
        // (review finding: writes widened 0600 to 0664).
        let p = dir.path().join("secret.json");
        std::fs::write(&p, "{}").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600)).unwrap();
        write_atomic(&p, "{\"k\":1}").unwrap();
        let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "mode must survive the write");

        // Symlink: dotfiles managers (stow/chezmoi/yadm) symlink these
        // configs. Replacing the link silently diverges the repo copy.
        let real = dir.path().join("repo-copy.json");
        std::fs::write(&real, "{}").unwrap();
        let link = dir.path().join("linked.json");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        write_atomic(&link, "{\"v\":9}").unwrap();
        assert!(
            std::fs::symlink_metadata(&link).unwrap().file_type().is_symlink(),
            "the symlink itself must survive"
        );
        assert_eq!(
            std::fs::read_to_string(&real).unwrap(),
            "{\"v\":9}",
            "the write must reach the link's target"
        );
    }

    #[test]
    fn reads_tell_missing_apart_from_damaged() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.json");
        // Missing: silent defaults are correct here.
        assert_eq!(
            read_json::<serde_json::Value>(&p),
            Loaded::Missing
        );
        // Truncated mid-write — the exact shape a Ctrl-C during
        // --calibrate leaves behind. Must NOT read as empty.
        std::fs::write(&p, "{\"qwen\": {\"n_ctx\": 1113").unwrap();
        match read_json::<serde_json::Value>(&p) {
            Loaded::Damaged(e) => assert!(e.contains("unparseable"), "{e}"),
            other => panic!("a truncated file must be Damaged, got {other:?}"),
        }
        // And rescuing it moves it aside so the next write can't eat it.
        let saved = rescue(&p).expect("rescue should have moved the file");
        assert!(saved.exists() && !p.exists());
        assert!(saved.to_string_lossy().ends_with(".corrupt"), "{saved:?}");
    }
}
