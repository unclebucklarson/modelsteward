//! Embed a build identity for the About dialog and `--version` (user
//! request 2026-08-29: "if a user has an issue it would allow us to
//! dial right in"). Never fails the build: crates.io and tarball
//! builds have no `.git`, so the id degrades honestly to "release".

use std::process::Command;

fn main() {
    let git = Command::new("git")
        .args(["describe", "--always", "--dirty=+"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "release".to_string());
    // Seconds-since-epoch keeps the build reproducible-enough while
    // still dating it; rendered as a date where shown.
    let date = Command::new("date")
        .args(["-u", "+%Y-%m-%d"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=STEWARD_BUILD_GIT={git}");
    println!("cargo:rustc-env=STEWARD_BUILD_DATE={date}");
    // Re-embed when HEAD moves (a plain `cargo build` after a commit).
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");
}
