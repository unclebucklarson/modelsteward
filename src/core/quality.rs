//! Quality gate v2 (M8): measured output quality, beyond speed and
//! context. Two instruments, both deterministic and machine-checked:
//!
//! - an **eval battery** — small code-flavored tasks with exactly one
//!   verifiable answer each, fixed forever for comparability (like the
//!   trial prompts). Scored by strict matchers, never by judgment.
//! - **N-shot tool-call reliability** — the calibrate probe run several
//!   times; agents live and die on this, and a single shot can't tell
//!   90% reliable from 100%.
//!
//! This is what makes the quant-choice advisor honest: "the Q4 is faster
//! AND holds quality" is only a sayable sentence once quality is a number.

use crate::core::router;
use anyhow::{Context, Result};

/// How a battery item's answer is checked. Strict on purpose: models are
/// told to put the bare answer on the last line, and only that line counts.
#[derive(Debug, Clone)]
pub enum Check {
    /// The last non-empty line, trimmed of whitespace/backticks/periods,
    /// must equal this exactly (case-insensitive).
    LastLineEquals(&'static str),
    /// The first JSON object found in the response must equal this value
    /// structurally (key order and whitespace don't matter).
    JsonEquals(&'static str),
}

pub struct EvalItem {
    pub prompt: &'static str,
    pub check: Check,
}

/// The fixed battery. Answers are chosen to be distinctive (no bare "7"
/// that a stray token could fake) and the tasks span evaluation, string
/// work, format discipline, and code reading.
pub fn eval_battery() -> Vec<EvalItem> {
    vec![
        EvalItem {
            prompt: "What does this Python expression evaluate to?\n\
                     sorted(set([5, 3, 9, 3, 7]))[-2]\n\
                     Reply with ONLY the answer on the last line.",
            check: Check::LastLineEquals("7"),
        },
        EvalItem {
            prompt: "Reverse the string \"steward\". Reply with ONLY the reversed \
                     string on the last line.",
            check: Check::LastLineEquals("drawets"),
        },
        EvalItem {
            prompt: "Convert hexadecimal 0x2A to decimal. Reply with ONLY the \
                     number on the last line.",
            check: Check::LastLineEquals("42"),
        },
        EvalItem {
            prompt: "What does this Python print?\nprint(f\"{2 ** 10}\")\n\
                     Reply with ONLY the output on the last line.",
            check: Check::LastLineEquals("1024"),
        },
        EvalItem {
            prompt: "Produce a JSON object with exactly two keys: \"name\" set to \
                     the string \"steward\" and \"count\" set to the number 3. \
                     Reply with ONLY the JSON.",
            check: Check::JsonEquals(r#"{"name":"steward","count":3}"#),
        },
        EvalItem {
            prompt: "This Python loop is meant to visit every index of xs but has \
                     an off-by-one bug:\nfor i in range(1, len(xs)): visit(xs[i])\n\
                     Which single index does it skip? Reply with ONLY the index on \
                     the last line.",
            check: Check::LastLineEquals("0"),
        },
    ]
}

/// Pure scorer for one item — testable without a server.
pub fn score_response(check: &Check, response: &str) -> bool {
    match check {
        Check::LastLineEquals(want) => response
            .lines()
            .rev()
            .map(|l| l.trim().trim_matches(['`', '.', '"', '\'']).trim())
            .find(|l| !l.is_empty())
            .is_some_and(|l| l.eq_ignore_ascii_case(want)),
        Check::JsonEquals(want) => {
            let want: serde_json::Value = serde_json::from_str(want).expect("battery JSON");
            // First balanced {...} block in the response.
            let Some(start) = response.find('{') else {
                return false;
            };
            let mut depth = 0usize;
            for (i, c) in response[start..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return serde_json::from_str::<serde_json::Value>(
                                &response[start..start + i + 1],
                            )
                            .map(|got| got == want)
                            .unwrap_or(false);
                        }
                    }
                    _ => {}
                }
            }
            false
        }
    }
}

/// One quality measurement of the currently-served model.
#[derive(Debug, Clone)]
pub struct QualityScore {
    /// Fraction of eval items answered correctly.
    pub eval_score: f64,
    pub evals_passed: u32,
    pub evals_total: u32,
    /// Fraction of N tool probes that produced a well-formed call.
    pub tool_reliability: f64,
    pub tool_shots: u32,
}

fn ask(port: u16, model: &str, prompt: &str) -> Result<String> {
    let body: serde_json::Value =
        ureq::post(&format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .timeout(std::time::Duration::from_secs(600))
            .send_json(serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0,
                "cache_prompt": false,
            }))
            .context("eval request")?
            .into_json()?;
    Ok(body
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
        .unwrap_or_default()
        .to_string())
}

/// Run the battery + N tool probes against a LOADED model. The caller
/// owns loading/unloading; per-item failures score zero rather than abort.
pub fn run_quality(
    port: u16,
    model: &str,
    tool_shots: u32,
    progress: &mut dyn FnMut(String),
) -> Result<QualityScore> {
    let battery = eval_battery();
    let total = battery.len() as u32;
    let mut passed = 0u32;
    for (i, item) in battery.iter().enumerate() {
        let ok = ask(port, model, item.prompt)
            .map(|r| score_response(&item.check, &r))
            .unwrap_or(false);
        if ok {
            passed += 1;
        }
        progress(format!(
            "quality {model}: eval {}/{} {}",
            i + 1,
            total,
            if ok { "✓" } else { "✗" }
        ));
    }
    let mut tool_ok = 0u32;
    for i in 0..tool_shots {
        let ok = router::probe_tool_call(port, model).unwrap_or(false);
        if ok {
            tool_ok += 1;
        }
        progress(format!(
            "quality {model}: tool probe {}/{tool_shots} {}",
            i + 1,
            if ok { "✓" } else { "✗" }
        ));
    }
    Ok(QualityScore {
        eval_score: passed as f64 / total.max(1) as f64,
        evals_passed: passed,
        evals_total: total,
        tool_reliability: tool_ok as f64 / tool_shots.max(1) as f64,
        tool_shots,
    })
}

/// Load the model, run the full quality probe, record scores into
/// measurements + the journal, unload. The Lab's Quality campaign and
/// `--quality` both land here.
pub fn run_and_record(
    cfg: &crate::core::settings::AppConfig,
    model: &str,
    tool_shots: u32,
    progress: &mut dyn FnMut(String),
) -> Result<QualityScore> {
    use crate::core::{discover, history, system};
    progress(format!("quality {model}: loading…"));
    router::fetch_settled_ctx(cfg.port, model)?;
    let score = run_quality(cfg.port, model, tool_shots, progress)?;
    let dir = router::state_dir();
    let mut all = router::read_measurements(&dir);
    let mut entry = all.get(model).cloned().unwrap_or_default();
    entry.eval_score = Some(score.eval_score);
    entry.tool_reliability = Some(score.tool_reliability);
    all.insert(model.to_string(), entry);
    router::write_measurements(&dir, &all)?;
    let build = system::pick_server(cfg).ok().and_then(|s| discover::build_of(&s));
    let _ = history::record(
        &dir,
        &history::Entry {
            when: crate::core::advisor::now_epoch(),
            model: model.to_string(),
            build,
            eval_score: Some(score.eval_score),
            tool_reliability: Some(score.tool_reliability),
            ..Default::default()
        },
    );
    let _ = router::unload_model(cfg.port, model);
    router::wait_until_not_loaded(cfg.port, model, std::time::Duration::from_secs(30));
    progress(format!(
        "quality {model}: evals {}/{} ({:.0}%), tool calls {}/{} ({:.0}%)",
        score.evals_passed,
        score.evals_total,
        score.eval_score * 100.0,
        (score.tool_reliability * score.tool_shots as f64).round() as u32,
        score.tool_shots,
        score.tool_reliability * 100.0
    ));
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matchers_are_strict_but_tolerant_of_chatter() {
        let c = Check::LastLineEquals("7");
        assert!(score_response(&c, "Let me think.\nThe answer is computed.\n7"));
        assert!(score_response(&c, "reasoning...\n\n`7`\n"));
        assert!(!score_response(&c, "17"), "no substring credit");
        assert!(!score_response(&c, "7 is the answer"), "answer must be alone on the line");
        assert!(!score_response(&c, ""));

        let j = Check::JsonEquals(r#"{"name":"steward","count":3}"#);
        assert!(score_response(&j, "Here you go:\n{\"count\": 3, \"name\": \"steward\"}"));
        assert!(!score_response(&j, "{\"name\": \"steward\", \"count\": 4}"));
        assert!(!score_response(&j, "no json here"));
    }

    #[test]
    fn battery_answers_are_self_consistent() {
        // Every item's own expected answer must pass its own check when
        // given as a bare last line — the battery can't drift broken.
        for item in eval_battery() {
            let fake = match &item.check {
                Check::LastLineEquals(w) => format!("reasoning\n{w}"),
                Check::JsonEquals(w) => (*w).to_string(),
            };
            assert!(score_response(&item.check, &fake), "{}", item.prompt);
        }
    }
}
