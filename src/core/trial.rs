//! M7 phase 2: the measured-trial harness. A trial takes a model and a
//! set of candidate config deltas, measures each exactly the way spike 5
//! did — same prompts, server-timed generation, temperature 0 — and
//! produces a measured verdict: keep the best candidate or keep the
//! baseline. Candidates are applied through an in-memory config clone
//! (preset regenerated, router reloaded), so config.json is only written
//! when a winner is explicitly kept.
//!
//! The baseline is the model's current config MINUS every key the trial's
//! variants set — an already-adopted winner re-competes fairly instead of
//! being compared against itself.

use crate::core::{router, settings, system};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// One candidate configuration delta: extra preset keys for the model's
/// section. The label is the identity trials are stored under.
#[derive(Debug, Clone)]
pub struct Variant {
    pub label: String,
    pub extra: Vec<(String, String)>,
}

/// The speculative-decoding menu (spike 5 methodology). ngram modes cost
/// zero VRAM; classic draft models were measured off this hardware class
/// (see docs/spikes.md spike 5) and aren't retried by default.
pub fn spec_decode_variants() -> Vec<Variant> {
    ["ngram-simple", "ngram-map-k4v", "ngram-mod"]
        .into_iter()
        .map(|t| Variant {
            label: t.to_string(),
            extra: vec![("spec-type".into(), t.into())],
        })
        .collect()
}

/// The physical-batch menu: larger `-ub` speeds prefill at a VRAM (and
/// therefore context) cost — measured, the verdict decides if the trade
/// pays. Judged by the Prefill goal.
pub fn ubatch_variants() -> Vec<Variant> {
    [1024u32, 2048]
        .into_iter()
        .map(|n| Variant {
            label: format!("ub-{n}"),
            extra: vec![("ubatch-size".into(), n.to_string())],
        })
        .collect()
}

/// What a trial menu is optimizing for — picks the verdict rules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Goal {
    /// Faster generation on edit-heavy work (speculation menus).
    RewriteTg,
    /// Faster prompt processing (batch-size menus) — agent turns are
    /// prefill-dominated when the context is large.
    Prefill,
}

/// A named menu: which variants to race and how to judge them.
pub fn menu(name: &str) -> Option<(Vec<Variant>, Goal)> {
    match name {
        "spec" => Some((spec_decode_variants(), Goal::RewriteTg)),
        "ub" => Some((ubatch_variants(), Goal::Prefill)),
        _ => None,
    }
}

/// Measured outcome of one configuration on one model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TrialResult {
    /// Mean generation tokens/sec over the novel-code prompts.
    pub tg_novel: Option<f64>,
    /// Generation tokens/sec on the rewrite prompt (agent-workload proxy).
    pub tg_rewrite: Option<f64>,
    /// Draft-token acceptance on the rewrite prompt, when speculation ran.
    pub accept_rewrite: Option<f64>,
    /// Prompt-processing tokens/sec on the long prefill probe.
    pub pp_prefill: Option<f64>,
    /// Context `--fit` settled on under this config.
    pub settled_ctx: Option<u64>,
    /// Load or generation failure — the result is still recorded so a
    /// crashing variant is remembered, not retried forever.
    pub error: Option<String>,
    /// llama.cpp build that produced this (staleness signal, like bench).
    pub build: Option<u64>,
}

/// trials.json: model id → variant label → result. "baseline" is a label.
pub type Trials =
    std::collections::BTreeMap<String, std::collections::BTreeMap<String, TrialResult>>;

pub const BASELINE: &str = "baseline";

fn trials_path(dir: &Path) -> std::path::PathBuf {
    dir.join("trials.json")
}

pub fn read_trials(dir: &Path) -> Trials {
    std::fs::read_to_string(trials_path(dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn write_trials(dir: &Path, t: &Trials) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(trials_path(dir), serde_json::to_string_pretty(t)?)
        .with_context(|| format!("writing {}", trials_path(dir).display()))
}

/// The measured verdict over one model's trial table. Pure, so the rules
/// are testable: a winner must not cost novel-code speed (>3%) or context
/// (>2%), and must earn its keep on rewrite work (≥10% faster).
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub winner: Option<String>,
    pub reason: String,
}

pub fn verdict(
    goal: Goal,
    baseline: &TrialResult,
    candidates: &std::collections::BTreeMap<String, TrialResult>,
) -> Verdict {
    let (Some(b_novel), Some(b_rewrite), Some(b_ctx)) =
        (baseline.tg_novel, baseline.tg_rewrite, baseline.settled_ctx)
    else {
        return Verdict {
            winner: None,
            reason: "baseline itself failed to measure — nothing to compare against".into(),
        };
    };
    // The metric a candidate must improve ≥10%; everything else is a guard.
    let primary = |r: &TrialResult| match goal {
        Goal::RewriteTg => r.tg_rewrite,
        Goal::Prefill => r.pp_prefill,
    };
    let Some(b_primary) = primary(baseline) else {
        return Verdict {
            winner: None,
            reason: "baseline lacks the goal metric — re-run the trial".into(),
        };
    };
    let mut best: Option<(&String, f64)> = None;
    for (label, r) in candidates {
        if label == BASELINE {
            continue;
        }
        let (Some(novel), Some(rewrite), Some(ctx), Some(p)) =
            (r.tg_novel, r.tg_rewrite, r.settled_ctx, primary(r))
        else {
            continue; // failed variants can't win
        };
        if novel < b_novel * 0.97
            || rewrite < b_rewrite * 0.97
            || (ctx as f64) < b_ctx as f64 * 0.98
        {
            continue; // costs something baseline work can't spare
        }
        if p < b_primary * 1.10 {
            continue; // doesn't earn its keep
        }
        if best.is_none_or(|(_, p0)| p > p0) {
            best = Some((label, p));
        }
    }
    let metric = match goal {
        Goal::RewriteTg => "rewrite",
        Goal::Prefill => "prefill",
    };
    match best {
        Some((label, p)) => Verdict {
            winner: Some(label.clone()),
            reason: format!(
                "{label}: {metric} {p:.1} t/s vs baseline {b_primary:.1} ({:+.0}%), \
                 everything else preserved",
                (p / b_primary - 1.0) * 100.0
            ),
        },
        None => Verdict {
            winner: None,
            reason: format!(
                "no candidate beat baseline by ≥10% on {metric} without costing \
                 generation speed or context — keeping baseline"
            ),
        },
    }
}

// ---- measurement machinery -------------------------------------------------

/// Two novel-code prompts (nothing to copy — speculation's worst case) and
/// one rewrite prompt (heavy copying — the agent-workload sweet spot).
/// Fixed forever so results stay comparable across sessions and models.
const NOVEL_PROMPTS: [&str; 2] = [
    "Write a complete Python class for a doubly-linked list with insert, \
     delete, find, and iterator support. Include docstrings.",
    "Write a Rust function that parses a simple INI file into a \
     BTreeMap<String, BTreeMap<String, String>>, with unit tests.",
];

fn rewrite_prompt() -> String {
    let code: String = (0..10)
        .map(|i| {
            format!(
                "def process_batch_{i}(items, config):\n    results = []\n    for item in items:\n        value = item.get('field_{i}')\n        if value is not None and value > config.threshold_{i}:\n            results.append({{'id': item['id'], 'value': value * {m}}})\n    return results\n",
                m = i + 1
            )
        })
        .collect();
    format!(
        "Here is a Python module:\n```python\n{code}```\nRewrite it EXACTLY as-is but add \
         a docstring \"Process batch items against config thresholds.\" to every function. \
         Change nothing else — output the complete module."
    )
}

/// ~6k tokens of deterministic code for the prefill probe — big enough
/// that prompt processing dominates, fixed forever for comparability.
fn prefill_prompt() -> String {
    let code: String = (0..40)
        .map(|i| {
            format!(
                "def transform_stage_{i}(records, options):\n    output = []\n    for record in records:\n        key = record.get('key_{i}')\n        weight = options.weights.get('w_{i}', 1.0)\n        if key is not None and record['score_{i}'] * weight > options.cutoff_{i}:\n            output.append({{'key': key, 'rank': record['score_{i}'] * weight}})\n    return sorted(output, key=lambda r: r['rank'], reverse=True)\n",
            )
        })
        .collect();
    format!("Here is a Python module:\n```python\n{code}```\nReply with just: OK")
}

struct GenStats {
    tps: f64,
    prompt_tps: Option<f64>,
    draft_n: Option<u64>,
    draft_accepted: Option<u64>,
}

fn timed_generation(port: u16, model: &str, prompt: &str, max_tokens: u32) -> Result<GenStats> {
    let body: serde_json::Value =
        ureq::post(&format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .timeout(std::time::Duration::from_secs(600))
            .send_json(serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "cache_prompt": false,
            }))
            .context("timed generation request")?
            .into_json()?;
    let t = body
        .get("timings")
        .ok_or_else(|| anyhow::anyhow!("response has no timings block"))?;
    Ok(GenStats {
        tps: t
            .get("predicted_per_second")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow::anyhow!("timings has no predicted_per_second"))?,
        prompt_tps: t.get("prompt_per_second").and_then(|v| v.as_f64()),
        draft_n: t.get("draft_n").and_then(|v| v.as_u64()),
        draft_accepted: t.get("draft_n_accepted").and_then(|v| v.as_u64()),
    })
}

/// Measure the currently-served config of a loaded model: warmup, two
/// novel-code generations, one rewrite generation.
fn measure_loaded(port: u16, model: &str, settled_ctx: u64) -> Result<TrialResult> {
    let _ = timed_generation(port, model, NOVEL_PROMPTS[0], 128); // warmup: CUDA graphs
    let mut novel = Vec::new();
    for p in NOVEL_PROMPTS {
        novel.push(timed_generation(port, model, p, 512)?.tps);
    }
    let rw = timed_generation(port, model, &rewrite_prompt(), 1024)?;
    let pf = timed_generation(port, model, &prefill_prompt(), 8)?;
    Ok(TrialResult {
        tg_novel: Some(novel.iter().sum::<f64>() / novel.len() as f64),
        tg_rewrite: Some(rw.tps),
        accept_rewrite: match (rw.draft_accepted, rw.draft_n) {
            (Some(a), Some(n)) if n > 0 => Some(a as f64 / n as f64),
            _ => None,
        },
        pp_prefill: pf.prompt_tps,
        settled_ctx: Some(settled_ctx),
        error: None,
        build: None,
    })
}

/// The model's override with every key any variant sets stripped — the
/// fair baseline (an adopted winner re-competes instead of racing itself).
pub fn baseline_override(
    cfg: &settings::AppConfig,
    model: &str,
    variants: &[Variant],
) -> router::ModelOverrides {
    let knob_keys: std::collections::HashSet<&str> = variants
        .iter()
        .flat_map(|v| v.extra.iter().map(|(k, _)| k.as_str()))
        .collect();
    let mut ov = cfg.overrides.get(model).cloned().unwrap_or_default();
    ov.extra.retain(|(k, _)| !knob_keys.contains(k.as_str()));
    ov
}

/// Run one model's trial campaign: baseline + each variant, measured in
/// sequence, persisted per variant. Restores the on-disk config's preset
/// afterwards and leaves the model unloaded. Returns the verdict.
pub fn run_trial(
    cfg: &settings::AppConfig,
    model: &str,
    variants: &[Variant],
    goal: Goal,
    progress: &mut dyn FnMut(String),
) -> Result<Verdict> {
    let dir = router::state_dir();
    match router::status(&dir, &system::router_config(cfg)) {
        router::RouterState::Ours { .. } => {}
        other => anyhow::bail!(
            "trials need our router running on port {}; state is {other:?}",
            cfg.port
        ),
    }
    let build = system::pick_server(cfg)
        .ok()
        .as_deref()
        .and_then(crate::core::discover::build_of);

    let base_ov = baseline_override(cfg, model, variants);
    let mut all = read_trials(&dir);
    let total = variants.len() + 1;

    let mut round = |n: usize, label: &str, extra: &[(String, String)]| -> TrialResult {
        progress(format!("[{n}/{total}] {model} · {label}: applying config + loading…"));
        let mut trial_cfg = cfg.clone();
        let mut ov = base_ov.clone();
        ov.extra.extend(extra.iter().cloned());
        trial_cfg.overrides.insert(model.to_string(), ov);
        let attempt = (|| -> Result<TrialResult> {
            system::write_preset(&trial_cfg, &[])?;
            router::reload(cfg.port)?;
            let ctx = router::fetch_settled_ctx(cfg.port, model)?;
            progress(format!(
                "[{n}/{total}] {model} · {label}: loaded (ctx {ctx}), timing generations…"
            ));
            measure_loaded(cfg.port, model, ctx)
        })();
        let _ = router::unload_model(cfg.port, model);
        router::wait_until_not_loaded(cfg.port, model, std::time::Duration::from_secs(30));
        match attempt {
            Ok(mut r) => {
                r.build = build;
                progress(format!(
                    "[{n}/{total}] {model} · {label}: novel {:.1} t/s, rewrite {:.1} t/s{}",
                    r.tg_novel.unwrap_or(0.0),
                    r.tg_rewrite.unwrap_or(0.0),
                    r.accept_rewrite
                        .map(|a| format!(", acceptance {:.0}%", a * 100.0))
                        .unwrap_or_default()
                ));
                r
            }
            Err(e) => {
                progress(format!("[{n}/{total}] {model} · {label}: FAILED — {e:#}"));
                TrialResult {
                    error: Some(format!("{e:#}")),
                    build,
                    ..Default::default()
                }
            }
        }
    };

    let baseline = round(1, BASELINE, &[]);
    all.entry(model.to_string())
        .or_default()
        .insert(BASELINE.to_string(), baseline.clone());
    write_trials(&dir, &all)?;

    // Verdicts only compare what THIS run raced — stored results from
    // other menus stay in trials.json for display but can't win here.
    let mut raced = std::collections::BTreeMap::new();
    for (i, v) in variants.iter().enumerate() {
        let r = round(i + 2, &v.label, &v.extra);
        raced.insert(v.label.clone(), r.clone());
        all.entry(model.to_string())
            .or_default()
            .insert(v.label.clone(), r);
        write_trials(&dir, &all)?;
    }

    // Whatever happened, put the real config's preset back.
    system::write_preset(cfg, &[])?;
    router::reload(cfg.port)?;

    let v = verdict(goal, &baseline, &raced);
    progress(format!("{model}: {}", v.reason));
    Ok(v)
}

/// Persist a trial winner: merge its keys into the model's override in
/// config.json (stripping competing knob keys first), regenerate, reload.
pub fn keep_variant(
    cfg_path: &Path,
    cfg: &settings::AppConfig,
    model: &str,
    variants: &[Variant],
    label: &str,
) -> Result<()> {
    let mut new_cfg = cfg.clone();
    let mut ov = baseline_override(cfg, model, variants);
    if label != BASELINE {
        let v = variants
            .iter()
            .find(|v| v.label == label)
            .ok_or_else(|| anyhow::anyhow!("unknown variant {label:?}"))?;
        ov.extra.extend(v.extra.iter().cloned());
    }
    if ov == router::ModelOverrides::default() {
        new_cfg.overrides.remove(model);
    } else {
        new_cfg.overrides.insert(model.to_string(), ov);
    }
    new_cfg.save(cfg_path)?;
    system::write_preset(&new_cfg, &[])?;
    let _ = router::reload(new_cfg.port);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(novel: f64, rewrite: f64, ctx: u64) -> TrialResult {
        TrialResult {
            tg_novel: Some(novel),
            tg_rewrite: Some(rewrite),
            settled_ctx: Some(ctx),
            ..Default::default()
        }
    }

    #[test]
    fn verdict_picks_best_rewrite_that_costs_nothing() {
        let base = r(38.6, 39.2, 116_224);
        let mut c = std::collections::BTreeMap::new();
        c.insert("ngram-simple".to_string(), r(40.4, 86.8, 117_248));
        c.insert("ngram-mod".to_string(), r(39.9, 70.0, 117_000));
        let v = verdict(Goal::RewriteTg, &base, &c);
        assert_eq!(v.winner.as_deref(), Some("ngram-simple"));
    }

    #[test]
    fn prefill_goal_judges_by_prompt_speed() {
        let mut base = r(38.6, 39.2, 116_224);
        base.pp_prefill = Some(1400.0);
        let mut c = std::collections::BTreeMap::new();
        let mut ub = r(38.5, 39.0, 115_500);
        ub.pp_prefill = Some(1800.0);
        c.insert("ub-1024".to_string(), ub);
        // Faster prefill but pays with generation speed → rejected.
        let mut bad = r(35.0, 39.0, 116_000);
        bad.pp_prefill = Some(2000.0);
        c.insert("ub-2048".to_string(), bad);
        let v = verdict(Goal::Prefill, &base, &c);
        assert_eq!(v.winner.as_deref(), Some("ub-1024"), "{}", v.reason);
        assert!(v.reason.contains("prefill"));
    }

    #[test]
    fn verdict_rejects_context_or_speed_costs() {
        let base = r(38.6, 39.2, 116_224);
        let mut c = std::collections::BTreeMap::new();
        // Spike 5's classic draft: faster rewrite is irrelevant when the
        // context collapses.
        c.insert("draft-4b".to_string(), r(38.0, 60.0, 4_096));
        // Fast rewrite but slower novel code → rejected.
        c.insert("slower-novel".to_string(), r(30.0, 80.0, 117_000));
        // Barely faster rewrite → not worth a config change.
        c.insert("meh".to_string(), r(39.0, 41.0, 117_000));
        let v = verdict(Goal::RewriteTg, &base, &c);
        assert_eq!(v.winner, None, "{}", v.reason);
    }

    #[test]
    fn verdict_ignores_failed_variants_and_failed_baseline() {
        let mut c = std::collections::BTreeMap::new();
        c.insert(
            "crashed".to_string(),
            TrialResult { error: Some("boom".into()), ..Default::default() },
        );
        assert_eq!(verdict(Goal::RewriteTg, &r(38.0, 39.0, 100_000), &c).winner, None);
        assert!(verdict(Goal::RewriteTg, &TrialResult::default(), &c)
            .reason
            .contains("baseline itself failed"));
    }

    #[test]
    fn baseline_strips_only_knob_keys() {
        let mut cfg = settings::AppConfig::default();
        cfg.overrides.insert(
            "m".into(),
            router::ModelOverrides {
                cache_type_kv: Some("q8_0".into()),
                ctx: None,
                extra: vec![
                    ("spec-type".into(), "ngram-simple".into()),
                    ("ub".into(), "1024".into()),
                ],
            },
        );
        let ov = baseline_override(&cfg, "m", &spec_decode_variants());
        assert_eq!(ov.cache_type_kv.as_deref(), Some("q8_0"), "unrelated override kept");
        assert_eq!(
            ov.extra,
            vec![("ub".to_string(), "1024".to_string())],
            "the knob under trial is stripped; unrelated extras survive"
        );
    }

    #[test]
    fn trials_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = Trials::new();
        t.entry("m".into())
            .or_default()
            .insert(BASELINE.into(), r(38.0, 39.0, 100_000));
        write_trials(dir.path(), &t).unwrap();
        let back = read_trials(dir.path());
        assert_eq!(back["m"][BASELINE].tg_novel, Some(38.0));
    }
}
