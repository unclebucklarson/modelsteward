//! M9 phase 1 — the Meter's token ledger (design settled with user
//! 2026-08-27; "local AI is not free" made measurable).
//!
//! The router truncates router.log on every start, so token evidence
//! dies with each restart unless it is harvested continuously: the
//! poller (and every `--meter` run) parses the CURRENT log instance,
//! credits only what hasn't been credited yet, and appends the deltas
//! to meter.jsonl as hour buckets. The cursor (meter-cursor.json)
//! remembers the log instance fingerprint and what was already
//! credited from it.
//!
//! Extraction-clean by decision: inputs are log text + timestamps,
//! outputs are buckets and reports — no app types in the API, so this
//! can become a standalone crate if it proves out.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One credited slice of usage: `when` is the UTC hour it was credited
/// in (harvest time, not request time — the log carries no wall clock,
/// and hour-grain is honest about that).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bucket {
    pub when: u64,
    pub model: String,
    pub turns: u64,
    pub prompt: u64,
    pub generated: u64,
    pub reused: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tally {
    pub turns: u64,
    pub prompt: u64,
    pub generated: u64,
    pub reused: u64,
}

/// What was already credited from the current log instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Cursor {
    pub fingerprint: String,
    /// v2 (2026-08-28 review): the first LINE alone is near-constant
    /// across router starts — v2 hashes the first 4KB, which always
    /// carries run-specific ports/timestamps. The legacy field remains
    /// so existing cursors aren't treated as a fresh instance (which
    /// would double-credit everything once).
    pub fingerprint_v2: Option<String>,
    pub credited: BTreeMap<String, Tally>,
}

/// Cross-process harvest lock: the GUI poller and a `--meter` CLI run
/// are separate processes sharing one cursor — an unlocked
/// read-modify-write double-credits the append-only ledger (review
/// catch 2026-08-28). Advisory lockfile; a stale one (>120s: a crashed
/// holder) is stolen.
pub struct HarvestLock(PathBuf);
impl Drop for HarvestLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn try_lock(dir: &Path) -> Option<HarvestLock> {
    let path = dir.join("meter.lock");
    let _ = std::fs::create_dir_all(dir);
    match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(_) => Some(HarvestLock(path)),
        Err(_) => {
            let stale = std::fs::metadata(&path)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age.as_secs() > 120);
            if stale {
                let _ = std::fs::remove_file(&path);
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .ok()
                    .map(|_| HarvestLock(path))
            } else {
                None
            }
        }
    }
}

fn ledger_path(dir: &Path) -> PathBuf {
    dir.join("meter.jsonl")
}

fn cursor_path(dir: &Path) -> PathBuf {
    dir.join("meter-cursor.json")
}

/// A log instance's identity: the first 4KB (spawn lines and stamps in
/// there vary per run; the same instance only ever grows, so the prefix
/// is stable). The legacy first-line fingerprint lives on only to
/// recognize cursors written before v2.
pub fn log_fingerprint(log: &str) -> String {
    crate::core::router::fnv(&log[..log.len().min(4096)])
}

fn legacy_fingerprint(log: &str) -> String {
    crate::core::router::fnv(log.lines().next().unwrap_or(""))
}

/// Pure: what the current totals add beyond what was already credited.
/// `same_instance` false means a fresh log — everything is new. Totals
/// that went BACKWARD under the same instance (shouldn't happen) are
/// credited at zero, never negative.
pub fn deltas(
    totals: &BTreeMap<String, Tally>,
    cursor: &Cursor,
    same_instance: bool,
) -> Vec<(String, Tally)> {
    let fresh = !same_instance;
    totals
        .iter()
        .filter_map(|(model, t)| {
            let base = if fresh {
                Tally::default()
            } else {
                cursor.credited.get(model).cloned().unwrap_or_default()
            };
            let d = Tally {
                turns: t.turns.saturating_sub(base.turns),
                prompt: t.prompt.saturating_sub(base.prompt),
                generated: t.generated.saturating_sub(base.generated),
                reused: t.reused.saturating_sub(base.reused),
            };
            (d.turns > 0 || d.prompt > 0 || d.generated > 0)
                .then(|| (model.clone(), d))
        })
        .collect()
}

/// Parse the log, credit the new usage into the ledger, advance the
/// cursor. Returns how many models got new credit (0 also when another
/// process holds the harvest lock — the next tick gets it). Crediting
/// is idempotent per log content.
pub fn harvest(dir: &Path, log: &str, now: u64) -> Result<usize> {
    let stats = crate::core::evidence::cache_effectiveness(log);
    harvest_stats(dir, &stats, log, now)
}

/// The poller already computes cache stats for the monitor; this
/// variant credits from them without a second full-log parse (review
/// catch 2026-08-28 — the log was being parsed twice per mine tick).
pub fn harvest_stats(
    dir: &Path,
    stats: &[crate::core::evidence::ModelCacheStats],
    log: &str,
    now: u64,
) -> Result<usize> {
    let Some(_lock) = try_lock(dir) else {
        return Ok(0);
    };
    let totals: BTreeMap<String, Tally> = stats
        .iter()
        .filter(|s| s.turns > 0)
        .map(|s| {
            (
                s.model.clone(),
                Tally {
                    turns: s.turns as u64,
                    prompt: s.prompt_tokens,
                    generated: s.generated_tokens,
                    reused: s.reused_tokens,
                },
            )
        })
        .collect();
    let fp = log_fingerprint(log);
    let cursor: Cursor = std::fs::read_to_string(cursor_path(dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let same_instance = match &cursor.fingerprint_v2 {
        Some(v2) => *v2 == fp,
        // Pre-v2 cursor: recognize the instance by the legacy hash so
        // the upgrade itself can't double-credit.
        None => !cursor.fingerprint.is_empty() && cursor.fingerprint == legacy_fingerprint(log),
    };
    let new = deltas(&totals, &cursor, same_instance);
    std::fs::create_dir_all(dir)?;
    if !new.is_empty() {
        let hour = now - now % 3600;
        let buckets: Vec<Bucket> = new
            .iter()
            .map(|(model, d)| Bucket {
                when: hour,
                model: model.clone(),
                turns: d.turns,
                prompt: d.prompt,
                generated: d.generated,
                reused: d.reused,
            })
            .collect();
        crate::core::history::append_jsonl(&ledger_path(dir), &buckets)?;
    }
    let credited = new.len();
    // Advance the cursor when anything actually changed (a quiet tick
    // rewrote it every 30s forever — write amplification for nothing).
    let advanced = Cursor {
        fingerprint: legacy_fingerprint(log),
        fingerprint_v2: Some(fp),
        credited: totals,
    };
    let changed = credited > 0
        || cursor.fingerprint_v2.as_deref() != advanced.fingerprint_v2.as_deref()
        || cursor.credited != advanced.credited;
    if changed {
        std::fs::write(cursor_path(dir), serde_json::to_string_pretty(&advanced)?)?;
    }
    Ok(credited)
}

/// Every ledger line, file order.
pub fn read_all(dir: &Path) -> Vec<Bucket> {
    crate::core::history::read_jsonl(&ledger_path(dir))
}

/// The pure-token report over an optional [since, until) range.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    pub fleet: Tally,
    pub per_model: BTreeMap<String, Tally>,
    /// (hour epoch, prompt+generated tokens) with the most traffic.
    pub busiest_hour: Option<(u64, u64)>,
    /// UTC-day epoch → tokens (prompt+generated), for the series view.
    pub per_day: BTreeMap<u64, u64>,
}

pub fn report(buckets: &[Bucket], since: Option<u64>, until: Option<u64>) -> Report {
    let mut r = Report::default();
    let mut per_hour: BTreeMap<u64, u64> = BTreeMap::new();
    for b in buckets {
        if since.is_some_and(|s| b.when < s) || until.is_some_and(|u| b.when >= u) {
            continue;
        }
        let add = |t: &mut Tally| {
            t.turns += b.turns;
            t.prompt += b.prompt;
            t.generated += b.generated;
            t.reused += b.reused;
        };
        add(&mut r.fleet);
        add(r.per_model.entry(b.model.clone()).or_default());
        *per_hour.entry(b.when).or_default() += b.prompt + b.generated;
        *r.per_day.entry(b.when - b.when % 86_400).or_default() += b.prompt + b.generated;
    }
    r.busiest_hour = per_hour.into_iter().max_by_key(|(_, n)| *n);
    r
}

/// One line for the Server tab: today's usage at a glance.
pub fn summary_line(dir: &Path, now: u64) -> Option<String> {
    let day = now - now % 86_400;
    let r = report(&read_all(dir), Some(day), None);
    (r.fleet.turns > 0).then(|| {
        format!(
            "Meter today: {} turns, {} prompt + {} generated tokens{}",
            r.fleet.turns,
            group(r.fleet.prompt),
            group(r.fleet.generated),
            if r.fleet.prompt > 0 {
                format!(
                    " ({:.0}% of prompt served from cache)",
                    r.fleet.reused as f64 / r.fleet.prompt as f64 * 100.0
                )
            } else {
                String::new()
            }
        )
    })
}

fn group(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The CLI report. `cloud_price` is $/Mtok for the comparison counter —
/// user-editable config, deliberately dated in its doc.
pub fn fmt_report(r: &Report, label: &str, cloud_price: f64) -> String {
    let mut out = format!("Meter — {label} (UTC buckets, hour grain)\n");
    if r.fleet.turns == 0 {
        out.push_str("no usage recorded in this range\n");
        return out;
    }
    let f = &r.fleet;
    out.push_str(&format!(
        "fleet: {} turns · prompt {} · generated {} · {} of prompt from cache\n",
        f.turns,
        group(f.prompt),
        group(f.generated),
        if f.prompt > 0 {
            format!("{:.0}%", f.reused as f64 / f.prompt as f64 * 100.0)
        } else {
            "—".into()
        }
    ));
    out.push_str(&format!(
        "shape: {:.1} prompt tokens per generated token (agent work is prompt-heavy)\n",
        if f.generated > 0 {
            f.prompt as f64 / f.generated as f64
        } else {
            0.0
        }
    ));
    if let Some((hour, n)) = r.busiest_hour {
        out.push_str(&format!(
            "busiest hour: {} UTC — {} tokens\n",
            fmt_hour(hour),
            group(n)
        ));
    }
    out.push_str(&format!(
        "cloud comparison: {} generated tokens ≈ ${:.2} at ${cloud_price}/Mtok output \
         (price is YOUR editable config value, not a quote)\n",
        group(f.generated),
        f.generated as f64 / 1e6 * cloud_price,
    ));
    out.push_str("\nper model:\n");
    for (m, t) in &r.per_model {
        out.push_str(&format!(
            "  {m}: {} turns · prompt {} · generated {}\n",
            t.turns,
            group(t.prompt),
            group(t.generated)
        ));
    }
    if r.per_day.len() > 1 {
        out.push_str("\nper day (UTC):\n");
        for (day, n) in &r.per_day {
            out.push_str(&format!("  {}: {} tokens\n", fmt_day(*day), group(*n)));
        }
    }
    out
}

fn fmt_day(epoch: u64) -> String {
    crate::core::report::date_from_epoch(epoch)
}

fn fmt_hour(epoch: u64) -> String {
    format!("{} {:02}:00", fmt_day(epoch), (epoch % 86_400) / 3600)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tal(turns: u64, prompt: u64, generated: u64, reused: u64) -> Tally {
        Tally {
            turns,
            prompt,
            generated,
            reused,
        }
    }

    #[test]
    fn deltas_credit_only_the_new_and_survive_truncation() {
        let mut totals = BTreeMap::new();
        totals.insert("m".to_string(), tal(5, 1000, 200, 300));
        // Same instance, partially credited → only the growth.
        let cursor = Cursor {
            fingerprint: "fp1".into(),
            fingerprint_v2: None,
            credited: BTreeMap::from([("m".to_string(), tal(3, 600, 120, 200))]),
        };
        let d = deltas(&totals, &cursor, true);
        assert_eq!(d, vec![("m".to_string(), tal(2, 400, 80, 100))]);
        // New instance (router restarted) → everything is new credit.
        let d = deltas(&totals, &cursor, false);
        assert_eq!(d, vec![("m".to_string(), tal(5, 1000, 200, 300))]);
        // Fully credited, same instance → nothing.
        let cursor = Cursor {
            fingerprint: "fp1".into(),
            fingerprint_v2: None,
            credited: totals.clone(),
        };
        assert!(deltas(&totals, &cursor, true).is_empty());
    }

    #[test]
    fn harvest_is_idempotent_and_ledger_accumulates_across_instances() {
        let dir = tempfile::tempdir().unwrap();
        let log1 = "load: spawning server instance with name=m on port 40001\n\
             [40001] I slot update_slots: id  0 | task 7 | new prompt, n_ctx_slot = 8, prompt processing, n_tokens = 100\n\
             [40001] I slot print_timing: id  0 | task 7 | n_gen = 20\n\
             [40001] I slot release: id  0 | task 7 | stop processing: n_tokens = 140, truncated = 0\n";
        assert_eq!(harvest(dir.path(), log1, 3600).unwrap(), 1);
        assert_eq!(harvest(dir.path(), log1, 3700).unwrap(), 0, "idempotent");
        // Router restart: NEW log (different first line), same-shaped turn.
        let log2 = log1.replace("40001", "40002");
        assert_eq!(harvest(dir.path(), &log2, 7300).unwrap(), 1);
        let all = read_all(dir.path());
        let r = report(&all, None, None);
        assert_eq!(r.fleet.turns, 2);
        assert_eq!(r.fleet.prompt, 240, "release-n_gen per turn: 120 x2");
        assert_eq!(r.fleet.generated, 40);
        assert_eq!(r.fleet.reused, 40, "(120-100) x2 served from cache");
        // Range query: only the second instance's hour.
        let r = report(&all, Some(7200), None);
        assert_eq!(r.fleet.turns, 1);
        // Buckets landed on hour boundaries.
        assert!(all.iter().all(|b| b.when % 3600 == 0));
    }

    #[test]
    fn harvest_lock_prevents_concurrent_double_credit() {
        let dir = tempfile::tempdir().unwrap();
        let log = "load: spawning server instance with name=m on port 40001\n\
             [40001] I slot update_slots: id  0 | task 7 | new prompt, n_ctx_slot = 8, prompt processing, n_tokens = 100\n\
             [40001] I slot print_timing: id  0 | task 7 | n_gen = 20\n\
             [40001] I slot release: id  0 | task 7 | stop processing: n_tokens = 140, truncated = 0\n";
        // A held lock makes harvest skip (0 credited, ledger untouched).
        let held = try_lock(dir.path()).expect("first lock");
        assert_eq!(harvest(dir.path(), log, 3600).unwrap(), 0);
        assert!(read_all(dir.path()).is_empty());
        drop(held);
        assert_eq!(harvest(dir.path(), log, 3600).unwrap(), 1);
        assert_eq!(read_all(dir.path()).len(), 1, "credited exactly once");
    }

    #[test]
    fn pre_v2_cursors_are_recognized_not_double_credited() {
        let dir = tempfile::tempdir().unwrap();
        let log = "load: spawning server instance with name=m on port 40001\n\
             [40001] I slot update_slots: id  0 | task 7 | new prompt, n_ctx_slot = 8, prompt processing, n_tokens = 100\n\
             [40001] I slot print_timing: id  0 | task 7 | n_gen = 20\n\
             [40001] I slot release: id  0 | task 7 | stop processing: n_tokens = 140, truncated = 0\n";
        assert_eq!(harvest(dir.path(), log, 3600).unwrap(), 1);
        // Rewrite the cursor as a pre-upgrade one: legacy fingerprint
        // only. The same log must NOT be re-credited after upgrading.
        let cur: Cursor = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("meter-cursor.json")).unwrap(),
        )
        .unwrap();
        let legacy = Cursor {
            fingerprint: cur.fingerprint.clone(),
            fingerprint_v2: None,
            credited: cur.credited.clone(),
        };
        std::fs::write(
            dir.path().join("meter-cursor.json"),
            serde_json::to_string(&legacy).unwrap(),
        )
        .unwrap();
        assert_eq!(harvest(dir.path(), log, 3700).unwrap(), 0, "no double credit");
        assert_eq!(read_all(dir.path()).len(), 1);
    }

    #[test]
    fn report_ranges_shape_and_busiest_hour() {
        let b = |when, model: &str, prompt, generated| Bucket {
            when,
            model: model.into(),
            turns: 1,
            prompt,
            generated,
            reused: 0,
        };
        let buckets = vec![
            b(0, "a", 900, 100),
            b(3600, "a", 100, 50),
            b(90_000, "b", 10, 5),
        ];
        let r = report(&buckets, None, None);
        assert_eq!(r.busiest_hour, Some((0, 1000)));
        assert_eq!(r.per_day.len(), 2, "two UTC days");
        assert_eq!(r.per_model["a"].prompt, 1000);
        let text = fmt_report(&r, "all time", 3.0);
        assert!(text.contains("6.5 prompt tokens per generated"), "{text}");
        assert!(text.contains("≈ $0.00"), "{text}");
        // Range excludes day two.
        let r = report(&buckets, None, Some(86_400));
        assert!(!r.per_model.contains_key("b"));
    }
}
