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
    pub loop_reliability: Option<f64>,
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
            loop_reliability: None,
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

/// Read a JSONL file, skipping unparseable lines — shared by the
/// journal and the meter ledger (they were line-for-line copies;
/// review catch 2026-08-28). Advisory data, never load-bearing.
pub fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Vec<T> {
    std::fs::read_to_string(path)
        .map(|s| {
            s.lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Append records to a JSONL file, creating parents as needed.
pub fn append_jsonl<T: serde::Serialize>(path: &Path, items: &[T]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut lines = String::new();
    for item in items {
        lines.push_str(&serde_json::to_string(item)?);
        lines.push('\n');
    }
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(lines.as_bytes())?;
    Ok(())
}

/// Every journal entry, file order (oldest first).
pub fn read_all(dir: &Path) -> Vec<Entry> {
    read_jsonl(&path(dir))
}

/// Append one event; prunes (newest KEEP_PER_MODEL per model) when the
/// file grows past the threshold.
pub fn record(dir: &Path, entry: &Entry) -> Result<()> {
    append_jsonl(&path(dir), std::slice::from_ref(entry))?;
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

/// One model's numbers on the current build vs the build before it —
/// the journal answering "what did the last rebuild actually do here?".
#[derive(Debug, Clone, PartialEq)]
pub struct BuildDelta {
    pub model: String,
    pub prev_build: u64,
    pub cur_build: u64,
    /// (previous, current) measured context — only when both builds were
    /// measured under the SAME args fingerprint (a config change between
    /// builds would confound the comparison; those models are skipped).
    pub ctx: Option<(u64, u64)>,
    /// (previous, current) llama-bench tg t/s — config-independent
    /// baselines, so no fingerprint gate.
    pub tg: Option<(f64, f64)>,
}

/// Compare each model's newest numbers on the newest build in the journal
/// against its newest numbers on the build seen just before it. Models
/// measured on only one build produce nothing — advice never extrapolates.
pub fn build_deltas(all: &[Entry]) -> Vec<BuildDelta> {
    let cur_build = match all.iter().filter_map(|e| e.build).max() {
        Some(b) => b,
        None => return Vec::new(),
    };
    let mut models: Vec<&str> = all.iter().map(|e| e.model.as_str()).collect();
    models.sort();
    models.dedup();
    let mut out = Vec::new();
    for model in models {
        let mine: Vec<&Entry> = all
            .iter()
            .filter(|e| e.model == model && e.build.is_some() && e.error.is_none())
            .collect();
        let prev_build = match mine
            .iter()
            .filter_map(|e| e.build)
            .filter(|b| *b < cur_build)
            .max()
        {
            Some(b) => b,
            None => continue,
        };
        let latest = |build: u64, pick: &dyn Fn(&Entry) -> bool| -> Option<&Entry> {
            mine.iter()
                .filter(|e| e.build == Some(build) && pick(e))
                .max_by_key(|e| e.when)
                .copied()
        };
        let ctx = match (
            latest(prev_build, &|e| e.n_ctx.is_some()),
            latest(cur_build, &|e| e.n_ctx.is_some()),
        ) {
            (Some(p), Some(c)) if p.args_fp == c.args_fp && p.args_fp.is_some() => {
                Some((p.n_ctx.unwrap(), c.n_ctx.unwrap()))
            }
            _ => None,
        };
        let tg = match (
            latest(prev_build, &|e| e.tg_tps.is_some()),
            latest(cur_build, &|e| e.tg_tps.is_some()),
        ) {
            (Some(p), Some(c)) => Some((p.tg_tps.unwrap(), c.tg_tps.unwrap())),
            _ => None,
        };
        if ctx.is_some() || tg.is_some() {
            out.push(BuildDelta {
                model: model.to_string(),
                prev_build,
                cur_build,
                ctx,
                tg,
            });
        }
    }
    out
}

/// The fleet-level one-liner for the Server tab and findings report:
/// what the newest build cost or bought vs the one before, averaged over
/// the models measured on both. None until two builds share evidence.
pub fn build_advisory(all: &[Entry]) -> Option<String> {
    let mut deltas = build_deltas(all);
    // Only average comparisons that share ONE (prev, cur) pair — a
    // model whose previous build differs would fold multi-release
    // drift into a line labeled as a single rebuild (review catch
    // 2026-08-28). The most-observed pair wins the headline.
    let pair = {
        let mut counts: std::collections::BTreeMap<(u64, u64), usize> = Default::default();
        for d in &deltas {
            *counts.entry((d.prev_build, d.cur_build)).or_default() += 1;
        }
        counts.into_iter().max_by_key(|(_, n)| *n)?.0
    };
    deltas.retain(|d| (d.prev_build, d.cur_build) == pair);
    let (mut ctx_pct, mut tg_pct) = (Vec::new(), Vec::new());
    for d in &deltas {
        if let Some((p, c)) = d.ctx
            && p > 0
        {
            ctx_pct.push((c as f64 / p as f64 - 1.0) * 100.0);
        }
        if let Some((p, c)) = d.tg
            && p > 0.0
        {
            tg_pct.push((c / p - 1.0) * 100.0);
        }
    }
    if ctx_pct.is_empty() && tg_pct.is_empty() {
        return None;
    }
    let d = &deltas[0];
    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mut parts = Vec::new();
    if !ctx_pct.is_empty() {
        parts.push(format!(
            "context {:+.0}% avg ({} model{})",
            avg(&ctx_pct),
            ctx_pct.len(),
            if ctx_pct.len() == 1 { "" } else { "s" }
        ));
    }
    if !tg_pct.is_empty() {
        parts.push(format!(
            "generation {:+.0}% avg ({} model{})",
            avg(&tg_pct),
            tg_pct.len(),
            if tg_pct.len() == 1 { "" } else { "s" }
        ));
    }
    let worst = deltas
        .iter()
        .filter_map(|d| {
            d.ctx
                .filter(|(p, _)| *p > 0)
                .map(|(p, c)| (d.model.clone(), (c as f64 / p as f64 - 1.0) * 100.0))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .filter(|(_, pct)| *pct <= -5.0)
        .map(|(m, pct)| format!("; worst: {m} {pct:+.0}% context"))
        .unwrap_or_default();
    Some(format!(
        "b{} vs b{}: {}{}",
        d.cur_build,
        d.prev_build,
        parts.join(", "),
        worst
    ))
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
    fn build_deltas_compare_builds_not_configs() {
        let mk = |when, model: &str, build, ctx: Option<u64>, tg: Option<f64>, fp: &str| Entry {
            when,
            model: model.into(),
            build: Some(build),
            n_ctx: ctx,
            tg_tps: tg,
            args_fp: if fp.is_empty() { None } else { Some(fp.into()) },
            ..Default::default()
        };
        let all = vec![
            // model "a": measured on both builds, same fingerprint — the
            // 9% ctx regression b10630 really cost this machine.
            mk(1, "a", 10_454, Some(128_000), None, "fp1"),
            mk(2, "a", 10_630, Some(116_480), None, "fp1"),
            // bench baselines: no fingerprint needed.
            mk(3, "a", 10_454, None, Some(40.0), ""),
            mk(4, "a", 10_630, None, Some(41.0), ""),
            // model "b": config changed between builds — ctx delta is
            // confounded and must be skipped.
            mk(5, "b", 10_454, Some(100_000), None, "fp1"),
            mk(6, "b", 10_630, Some(50_000), None, "fp2"),
            // model "c": only ever on the new build — nothing to compare.
            mk(7, "c", 10_630, Some(9000), None, "fp1"),
        ];
        let d = build_deltas(&all);
        assert_eq!(d.len(), 1, "b confounded, c single-build: {d:?}");
        assert_eq!(d[0].model, "a");
        assert_eq!(d[0].ctx, Some((128_000, 116_480)));
        assert_eq!(d[0].tg, Some((40.0, 41.0)));
        let line = build_advisory(&all).unwrap();
        assert!(line.contains("b10630 vs b10454"), "{line}");
        assert!(line.contains("context -9% avg (1 model)"), "{line}");
        assert!(line.contains("worst: a -9% context"), "{line}");
        // One build only -> no advisory (never extrapolate).
        assert!(build_advisory(&all[6..]).is_none());
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
