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

/// A candidate the rules rejected but a human might still want: it beat
/// the goal metric by ≥10% and lost only on a guard. The rules stay
/// conservative; the tradeoff is presented instead of swallowed
/// (2026-08-25: north-mini's +50% prefill died silently on the ctx guard).
#[derive(Debug, Clone, PartialEq)]
pub struct NearMiss {
    pub label: String,
    /// The win, in plain language ("+50% prefill (3583 → 5377 t/s)").
    pub gain: String,
    /// What it costs, in plain language ("21% of context (255744 → 202752)").
    pub cost: String,
}

/// Everything a trial run produced: the rule-based verdict, the raced
/// table (baseline included), and the guard-rejected tradeoffs.
#[derive(Debug, Clone)]
pub struct TrialReport {
    pub goal: Goal,
    pub verdict: Verdict,
    pub near_misses: Vec<NearMiss>,
    pub raced: std::collections::BTreeMap<String, TrialResult>,
}

/// Guard-rejected-but-goal-beating candidates, with gains and costs
/// spelled out. Pure over the raced table.
pub fn near_misses(
    goal: Goal,
    baseline: &TrialResult,
    raced: &std::collections::BTreeMap<String, TrialResult>,
) -> Vec<NearMiss> {
    let (Some(b_novel), Some(b_rewrite), Some(b_ctx)) =
        (baseline.tg_novel, baseline.tg_rewrite, baseline.settled_ctx)
    else {
        return Vec::new();
    };
    let (primary, metric): (fn(&TrialResult) -> Option<f64>, &str) = match goal {
        Goal::RewriteTg => (|r| r.tg_rewrite, "rewrite generation"),
        Goal::Prefill => (|r| r.pp_prefill, "prefill"),
    };
    let Some(b_primary) = primary(baseline) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (label, r) in raced {
        if label == BASELINE {
            continue;
        }
        let (Some(novel), Some(rewrite), Some(ctx), Some(p)) =
            (r.tg_novel, r.tg_rewrite, r.settled_ctx, primary(r))
        else {
            continue;
        };
        if p < b_primary * 1.10 {
            continue; // never beat the goal — not interesting
        }
        let mut costs = Vec::new();
        if novel < b_novel * 0.97 {
            costs.push(format!(
                "{:.0}% of novel-code speed ({b_novel:.0} → {novel:.0} t/s)",
                (1.0 - novel / b_novel) * 100.0
            ));
        }
        if rewrite < b_rewrite * 0.97 {
            costs.push(format!(
                "{:.0}% of rewrite speed ({b_rewrite:.0} → {rewrite:.0} t/s)",
                (1.0 - rewrite / b_rewrite) * 100.0
            ));
        }
        if (ctx as f64) < b_ctx as f64 * 0.98 {
            costs.push(format!(
                "{:.0}% of context ({b_ctx} → {ctx})",
                (1.0 - ctx as f64 / b_ctx as f64) * 100.0
            ));
        }
        if costs.is_empty() {
            continue; // it won outright — that's the verdict's business
        }
        out.push(NearMiss {
            label: label.clone(),
            gain: format!(
                "+{:.0}% {metric} ({b_primary:.0} → {p:.0} t/s)",
                (p / b_primary - 1.0) * 100.0
            ),
            cost: costs.join(" and "),
        });
    }
    out
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

/// The verdict explained in plain language, derived entirely from the
/// measured table — the rules that pick the winner narrate their own
/// reasoning (user request 2026-08-25: the table isn't self-evident to
/// someone just learning model optimization). Deterministic and testable;
/// no model involved.
pub fn explain(report: &TrialReport) -> Vec<String> {
    let mut out = Vec::new();
    out.push(
        "What the columns mean — novel: generating brand-new code, speculation's worst \
         case (the did-anything-get-hurt check). rewrite: regenerating code the model was \
         given — edits, refactors, applying diffs — which is most of what a coding agent \
         does. prefill: how fast it reads your prompt before the first token. context: \
         the window memory-fitting could afford under this config. accepted: how many \
         speculated tokens the model confirmed."
            .to_string(),
    );
    let Some(base) = report.raced.get(BASELINE) else {
        out.push("The baseline itself failed to measure, so nothing can be compared.".into());
        return out;
    };
    let (primary, metric): (fn(&TrialResult) -> Option<f64>, &str) = match report.goal {
        Goal::RewriteTg => (|r| r.tg_rewrite, "rewrite"),
        Goal::Prefill => (|r| r.pp_prefill, "prefill"),
    };
    let (Some(b_novel), Some(b_rewrite), Some(b_ctx), Some(b_p)) =
        (base.tg_novel, base.tg_rewrite, base.settled_ctx, primary(base))
    else {
        out.push("The baseline lacks the goal metric — re-run the trial.".into());
        return out;
    };
    let pct = |new: f64, old: f64| (new / old - 1.0) * 100.0;

    // The winner, with its evidence.
    if let Some(w) = &report.verdict.winner
        && let Some(r) = report.raced.get(w)
        && let Some(p) = primary(r)
    {
        out.push(format!(
            "{w} is recommended because the metric that matters most here — {metric} speed \
             — improved {:+.0}% ({b_p:.1} → {p:.1} t/s) while nothing was paid for it: \
             novel-code speed stayed within noise ({b_novel:.1} → {:.1}) and context is \
             effectively unchanged ({b_ctx} → {}).",
            pct(p, b_p),
            r.tg_novel.unwrap_or(0.0),
            r.settled_ctx.unwrap_or(0),
        ));
    } else {
        out.push(
            "No candidate is recommended: earning a config change takes a ≥10% win on the \
             goal metric without giving up generation speed or context, and none managed it."
                .to_string(),
        );
    }

    // Why each losing candidate lost.
    for (label, r) in &report.raced {
        if label == BASELINE || Some(label) == report.verdict.winner.as_ref() {
            continue;
        }
        if let Some(e) = &r.error {
            out.push(format!("{label} failed to run ({e}) — recorded, not retried blindly."));
            continue;
        }
        let (Some(novel), Some(rewrite), Some(ctx), Some(p)) =
            (r.tg_novel, r.tg_rewrite, r.settled_ctx, primary(r))
        else {
            continue;
        };
        if p < b_p * 1.10 {
            out.push(format!(
                "{label} gained only {:+.0}% on {metric} — under the 10% bar a config \
                 change must earn to be worth carrying.",
                pct(p, b_p)
            ));
        } else if novel < b_novel * 0.97 || rewrite < b_rewrite * 0.97
            || (ctx as f64) < b_ctx as f64 * 0.98
        {
            out.push(format!(
                "{label} beat the goal but paid for it elsewhere — that tradeoff is offered \
                 as a separate choice rather than silently picked."
            ));
        } else {
            out.push(format!(
                "{label} qualified ({:+.0}%) but the winner gained more.",
                pct(p, b_p)
            ));
        }
    }

    // The recurring counterintuitive one: acceptance is not speed.
    if let Some(w) = &report.verdict.winner
        && let Some(wr) = report.raced.get(w)
        && let (Some(w_acc), Some(w_p)) = (wr.accept_rewrite, primary(wr))
    {
        for (label, r) in &report.raced {
            if label == BASELINE || label == w {
                continue;
            }
            if let (Some(acc), Some(p)) = (r.accept_rewrite, primary(r))
                && acc > w_acc + 0.05
                && p < w_p
            {
                out.push(format!(
                    "Counterintuitive but consistent across every model measured so far: \
                     {label} had the HIGHER acceptance rate ({:.0}% vs {:.0}%) yet gained \
                     less. Acceptance counts agreements, not payoff — a mode that drafts \
                     short, cheap spans wins often and earns little. Speed is the truth, \
                     which is why the app measures it instead of trusting the proxy.",
                    acc * 100.0,
                    w_acc * 100.0
                ));
                break;
            }
        }
    }

    // Expectation-setting: fast models gain less from speculation.
    if report.goal == Goal::RewriteTg && b_rewrite > 100.0 {
        out.push(
            "If the gain looks smaller than the +100% some models see: this model is \
             already fast, and a fast model spends proportionally more of its time \
             verifying drafts — slower models have the most to win from speculation."
                .to_string(),
        );
    }
    // Context framing for the window this config affords.
    if b_ctx < 2 * crate::core::rows::AGENT_MIN_CTX {
        out.push(format!(
            "Separate observation: this config settles at {b_ctx} tokens of context — \
             above the ~{} floor where agent work gets painful, but modest. Think \"fast \
             helper\", not \"whole repo in the window\".",
            crate::core::rows::AGENT_MIN_CTX
        ));
    }
    out
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
) -> Result<TrialReport> {
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

    // A measurement taken while another session holds the server is noise,
    // not data: abort the whole trial rather than record it (the preset is
    // restored below via the ? paths' caller — see the contention bails).
    let round = |n: usize,
                 label: &str,
                 extra: &[(String, String)],
                 progress: &mut dyn FnMut(String)|
     -> Result<TrialResult> {
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
                    "[{n}/{total}] {model} · {label}: novel {:.1} t/s, rewrite {:.1} t/s{}{}",
                    r.tg_novel.unwrap_or(0.0),
                    r.tg_rewrite.unwrap_or(0.0),
                    r.pp_prefill
                        .map(|p| format!(", prefill {p:.0} t/s"))
                        .unwrap_or_default(),
                    r.accept_rewrite
                        .map(|a| format!(", acceptance {:.0}%", a * 100.0))
                        .unwrap_or_default()
                ));
                Ok(r)
            }
            Err(e) => {
                if let Some(other) = router::loaded_other(cfg.port, model) {
                    anyhow::bail!(
                        "the server is busy with {other} — likely another session's \
                         request; nothing recorded. Re-run the trial when idle"
                    );
                }
                progress(format!("[{n}/{total}] {model} · {label}: FAILED — {e:#}"));
                Ok(TrialResult {
                    error: Some(format!("{e:#}")),
                    build,
                    ..Default::default()
                })
            }
        }
    };

    // On any bail (contention included), restore the real preset before
    // returning — a temp trial config must never outlive its run.
    let body = (|| -> Result<TrialReport> {
        let baseline = round(1, BASELINE, &[], progress)?;
        all.entry(model.to_string())
            .or_default()
            .insert(BASELINE.to_string(), baseline.clone());
        write_trials(&dir, &all)?;

        // Verdicts only compare what THIS run raced — stored results from
        // other menus stay in trials.json for display but can't win here.
        let mut raced = std::collections::BTreeMap::new();
        for (i, v) in variants.iter().enumerate() {
            let r = round(i + 2, &v.label, &v.extra, progress)?;
            raced.insert(v.label.clone(), r.clone());
            all.entry(model.to_string())
                .or_default()
                .insert(v.label.clone(), r);
            write_trials(&dir, &all)?;
        }

        let v = verdict(goal, &baseline, &raced);
        let near = near_misses(goal, &baseline, &raced);
        progress(format!("{model}: {}", v.reason));
        for nm in &near {
            progress(format!(
                "{model}: rules rejected {} — {} for {} (your call)",
                nm.label, nm.gain, nm.cost
            ));
        }
        raced.insert(BASELINE.to_string(), baseline);
        Ok(TrialReport {
            goal,
            verdict: v,
            near_misses: near,
            raced,
        })
    })();

    // Whatever happened, put the real config's preset back.
    system::write_preset(cfg, &[])?;
    let _ = router::reload(cfg.port);
    body
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
    // The kept config changes what --fit settles on, and the trial already
    // measured exactly that: carry its settled ctx into the measurement
    // (fingerprints cleared → the normal loop re-verifies next calibrate)
    // so synced limits stay honest without waiting for a re-measure.
    let dir = router::state_dir();
    if let Some(r) = read_trials(&dir)
        .get(model)
        .and_then(|t| t.get(label))
        .filter(|r| r.settled_ctx.is_some())
    {
        let mut all = router::read_measurements(&dir);
        let mut entry = all.get(model).cloned().unwrap_or_default();
        entry.n_ctx = r.settled_ctx;
        entry.error = None;
        entry.args_fp = None;
        entry.env_fp = None;
        all.insert(model.to_string(), entry);
        router::write_measurements(&dir, &all)?;
        let _ = crate::core::history::record(
            &dir,
            &crate::core::history::Entry {
                when: crate::core::advisor::now_epoch(),
                model: model.to_string(),
                build: r.build,
                n_ctx: r.settled_ctx,
                ..Default::default()
            },
        );
    }
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
    fn near_misses_surface_guard_rejected_tradeoffs() {
        // The real north-mini case: +30% prefill, rejected only on ctx.
        let mut base = r(187.2, 249.6, 255_744);
        base.pp_prefill = Some(3583.0);
        let mut c = std::collections::BTreeMap::new();
        let mut ub = r(186.4, 293.3, 237_568);
        ub.pp_prefill = Some(4662.0);
        c.insert("ub-1024".to_string(), ub);
        // A variant that never beat the goal is NOT interesting.
        let mut dud = r(187.0, 250.0, 255_000);
        dud.pp_prefill = Some(3600.0);
        c.insert("dud".to_string(), dud);
        assert_eq!(verdict(Goal::Prefill, &base, &c).winner, None);
        let nm = near_misses(Goal::Prefill, &base, &c);
        assert_eq!(nm.len(), 1);
        assert_eq!(nm[0].label, "ub-1024");
        assert!(nm[0].gain.contains("+30% prefill"), "{}", nm[0].gain);
        assert!(nm[0].cost.contains("7% of context"), "{}", nm[0].cost);

        // An outright winner is the verdict's business, not a near-miss.
        let mut clean = r(188.0, 250.0, 255_744);
        clean.pp_prefill = Some(4700.0);
        let mut c2 = std::collections::BTreeMap::new();
        c2.insert("clean".to_string(), clean);
        assert!(near_misses(Goal::Prefill, &base, &c2).is_empty());
    }

    #[test]
    fn explain_narrates_the_ornith_table() {
        // The real ornith-1.5 verdict dialog (2026-08-25).
        let mut raced = std::collections::BTreeMap::new();
        let mut mk = |novel: f64, rewrite: f64, ctx: u64, acc: Option<f64>| {
            let mut t = r(novel, rewrite, ctx);
            t.accept_rewrite = acc;
            t.pp_prefill = Some(3050.0);
            t
        };
        raced.insert(BASELINE.into(), mk(148.5, 149.0, 37_376, None));
        raced.insert("ngram-map-k4v".into(), mk(148.6, 159.0, 37_120, Some(0.69)));
        raced.insert("ngram-mod".into(), mk(146.5, 161.8, 36_864, Some(0.48)));
        raced.insert("ngram-simple".into(), mk(147.7, 176.0, 37_376, Some(0.48)));
        let baseline = raced[BASELINE].clone();
        let v = verdict(Goal::RewriteTg, &baseline, &raced);
        assert_eq!(v.winner.as_deref(), Some("ngram-simple"));
        let report = TrialReport {
            goal: Goal::RewriteTg,
            near_misses: near_misses(Goal::RewriteTg, &baseline, &raced),
            verdict: v,
            raced,
        };
        let text = explain(&report).join("\n");
        assert!(text.contains("ngram-simple is recommended"), "{text}");
        assert!(text.contains("+18%"), "{text}");
        assert!(text.contains("under the 10% bar"), "{text}");
        assert!(
            text.contains("HIGHER acceptance rate (69% vs 48%)"),
            "acceptance lesson must fire: {text}"
        );
        assert!(text.contains("already fast"), "{text}");
        assert!(text.contains("fast helper"), "ctx framing for 37k: {text}");

        // No-winner tables explain the bar instead.
        let mut raced2 = std::collections::BTreeMap::new();
        raced2.insert(BASELINE.into(), mk(148.5, 149.0, 37_376, None));
        raced2.insert("meh".into(), mk(148.0, 152.0, 37_376, Some(0.3)));
        let base2 = raced2[BASELINE].clone();
        let report2 = TrialReport {
            goal: Goal::RewriteTg,
            verdict: verdict(Goal::RewriteTg, &base2, &raced2),
            near_misses: near_misses(Goal::RewriteTg, &base2, &raced2),
            raced: raced2,
        };
        let text2 = explain(&report2).join("\n");
        assert!(text2.contains("No candidate is recommended"), "{text2}");
        assert!(text2.contains("meh gained only +2%"), "{text2}");
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
