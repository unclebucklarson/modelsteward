//! Append-only measurement journal (user decision 2026-08-25): every
//! context measurement and bench result also lands here, so build-over-
//! build comparisons are a query instead of archaeology. The current-truth
//! files (measurements.json, trials.json) are untouched — history is a
//! side effect of writing them, never a source the app reasons from
//! directly.
//!
//! Format: JSONL, one event per line, pruned to the newest N per model so
//! the file stays trivially small forever.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// One journal event. Context measurements carry `n_ctx`/`error`; bench
/// results carry `pp_tps`/`tg_tps`. `build` is the llama.cpp build that
/// produced the numbers — the axis most comparisons care about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Entry {
    pub when: u64,
    pub model: String,
    pub build: Option<u64>,
    pub args_fp: Option<String>,
    pub n_ctx: Option<u64>,
    pub pp_tps: Option<f64>,
    pub tg_tps: Option<f64>,
    pub eval_score: Option<f64>,
    pub tool_reliability: Option<f64>,
    pub error: Option<String>,
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            when: 0,
            model: String::new(),
            build: None,
            args_fp: None,
            n_ctx: None,
            pp_tps: None,
            tg_tps: None,
            eval_score: None,
            tool_reliability: None,
            error: None,
        }
    }
}

/// Newest entries kept per model when pruning.
pub const KEEP_PER_MODEL: usize = 50;
/// Total-line threshold that triggers a prune pass.
const PRUNE_AT: usize = 2000;

fn path(dir: &Path) -> PathBuf {
    dir.join("history.jsonl")
}

/// Every journal entry, file order (oldest first). Unparseable lines are
/// skipped — the journal is advisory, never load-bearing.
pub fn read_all(dir: &Path) -> Vec<Entry> {
    std::fs::read_to_string(path(dir))
        .map(|s| {
            s.lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Append one event; prunes (newest KEEP_PER_MODEL per model) when the
/// file grows past the threshold.
pub fn record(dir: &Path, entry: &Entry) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut line = serde_json::to_string(entry)?;
    line.push('\n');
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path(dir))?
        .write_all(line.as_bytes())?;
    let count = std::fs::read_to_string(path(dir))
        .map(|s| s.lines().count())
        .unwrap_or(0);
    if count > PRUNE_AT {
        prune(dir)?;
    }
    Ok(())
}

fn prune(dir: &Path) -> Result<()> {
    let all = read_all(dir);
    let mut per_model: std::collections::BTreeMap<&str, Vec<&Entry>> = Default::default();
    for e in &all {
        per_model.entry(&e.model).or_default().push(e);
    }
    let mut keep: Vec<&Entry> = per_model
        .values()
        .flat_map(|v| v.iter().rev().take(KEEP_PER_MODEL).copied().collect::<Vec<_>>())
        .collect();
    keep.sort_by_key(|e| e.when);
    let text: String = keep
        .iter()
        .filter_map(|e| serde_json::to_string(e).ok())
        .map(|l| l + "\n")
        .collect();
    std::fs::write(path(dir), text)?;
    Ok(())
}

/// A model's entries, newest first — what the hover trails render.
pub fn for_model<'a>(all: &'a [Entry], model: &str) -> Vec<&'a Entry> {
    let mut v: Vec<&Entry> = all.iter().filter(|e| e.model == model).collect();
    v.reverse();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(when: u64, model: &str, ctx: Option<u64>) -> Entry {
        Entry {
            when,
            model: model.into(),
            build: Some(10_630),
            n_ctx: ctx,
            ..Default::default()
        }
    }

    #[test]
    fn journal_appends_reads_and_orders() {
        let dir = tempfile::tempdir().unwrap();
        record(dir.path(), &e(1, "a", Some(100))).unwrap();
        record(dir.path(), &e(2, "b", Some(200))).unwrap();
        record(dir.path(), &e(3, "a", Some(90))).unwrap();
        let all = read_all(dir.path());
        assert_eq!(all.len(), 3);
        let a = for_model(&all, "a");
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].when, 3, "newest first");
        assert_eq!(a[0].n_ctx, Some(90));
    }

    #[test]
    fn pruning_keeps_newest_per_model_and_survives_junk() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..(KEEP_PER_MODEL as u64 + 20) {
            record(dir.path(), &e(i, "m", Some(i))).unwrap();
        }
        // Junk lines are skipped on read, dropped on prune.
        use std::io::Write;
        writeln!(
            std::fs::OpenOptions::new()
                .append(true)
                .open(dir.path().join("history.jsonl"))
                .unwrap(),
            "not json"
        )
        .unwrap();
        prune(dir.path()).unwrap();
        let all = read_all(dir.path());
        assert_eq!(all.len(), KEEP_PER_MODEL);
        assert_eq!(
            for_model(&all, "m")[0].when,
            KEEP_PER_MODEL as u64 + 19,
            "newest survives"
        );
    }
}
