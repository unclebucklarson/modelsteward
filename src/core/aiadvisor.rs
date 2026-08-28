//! The AI advisory layer (M6p2 tail, designed with user 2026-08-27).
//!
//! Scope, settled in the design talk: AI is NEVER load-bearing. The
//! deterministic layers own every verdict, flag, and file write; this
//! module exists only where language work is genuinely left — first
//! feature: explaining load failures the rules-based diagnose module
//! couldn't classify. One-shot, grounded generations; no chat.
//!
//! Backend: the router itself (the models the app already serves) —
//! offline, private, zero setup. The `Advisor` trait keeps the door
//! open for Ollama / OpenAI-compatible backends later; a cloud backend
//! would be explicit opt-in (standing rule: nothing leaves the machine).
//!
//! Guardrails: the prompt confines the model to the supplied evidence,
//! the UI renders output in a labeled advisory block naming the model
//! that wrote it, and nothing here is ever applied automatically.

use anyhow::{Context, Result};

/// A one-shot advisory backend. `describe()` names the model so the UI
/// can label whose opinion the text is.
pub trait Advisor {
    fn ask(&self, system: &str, user: &str) -> Result<String>;
    fn describe(&self) -> String;
}

/// The default backend: whatever model the app's own router serves.
pub struct RouterAdvisor {
    pub port: u16,
    pub model: String,
}

impl Advisor for RouterAdvisor {
    fn ask(&self, system: &str, user: &str) -> Result<String> {
        let body: serde_json::Value =
            ureq::post(&format!("http://127.0.0.1:{}/v1/chat/completions", self.port))
                .timeout(std::time::Duration::from_secs(300))
                .send_json(serde_json::json!({
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    // Reasoning-channel models think before they answer —
                    // a 49k-token fleet brief burned 500, then 2500 tokens
                    // of pure thinking with zero content (found live
                    // 2026-08-28). Ask for the answer, not the deliberation;
                    // templates without the kwarg ignore it.
                    "chat_template_kwargs": { "enable_thinking": false },
                    "max_tokens": 2500,
                    "temperature": 0.2,
                }))
                .context("advisor request")?
                .into_json()?;
        body.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|s| s.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "the model produced no answer (it may have spent its whole \
                     token budget on its reasoning channel) — try again or ask a \
                     different model"
                )
            })
    }

    fn describe(&self) -> String {
        self.model.clone()
    }
}

/// The system prompt shared by all advisory features: confine the model
/// to the evidence, forbid invention, keep it short and honest.
pub const SYSTEM: &str = "You are the advisory layer of modelsteward, a tool that \
manages llama.cpp servers. Reason ONLY from the data provided in the user message. \
If the data does not determine an answer, say exactly that — an honest 'the log \
doesn't show the cause' is a good answer. Never invent command-line flags, file \
paths, or version numbers not present in the data. No preamble; at most 150 words.";

/// Build the failure-explanation prompt from what the app already knows.
/// Pure and testable — the worker feeds it to an `Advisor`.
pub fn failure_prompt(
    model: &str,
    error: &str,
    build: Option<u64>,
    vram_mib: u64,
    ram_mib: u64,
    file_gib: Option<f64>,
    log_tail: &str,
) -> String {
    let mut p = format!(
        "A local model failed to load and the rule-based diagnoser couldn't \
         classify the failure. Explain the likely cause in plain language for \
         someone who is not a llama.cpp expert, and say what kind of fix to \
         look toward (without inventing exact flags).\n\n\
         Model: {model}\n\
         Stored error: {error}\n\
         llama.cpp build: {}\n\
         GPU VRAM: {vram_mib} MiB; system RAM: {ram_mib} MiB\n",
        build.map(|b| format!("b{b}")).unwrap_or_else(|| "unknown".into()),
    );
    if let Some(g) = file_gib {
        p.push_str(&format!("Model file size: {g:.1} GiB\n"));
    }
    if !log_tail.is_empty() {
        p.push_str(&format!("\nLast server log lines for this model:\n{log_tail}\n"));
    }
    p
}

/// System prompt for the fleet brief — a longer-form synthesis than the
/// failure explainer, still confined to the supplied data.
pub const BRIEF_SYSTEM: &str = "You are the advisory layer of modelsteward, a tool \
that manages llama.cpp servers and measures local models. Reason ONLY from the \
findings report provided. Cite the numbers you use. If the data can't answer a \
question, say so plainly. Never invent flags, models, or figures. No preamble; \
at most 300 words.";

/// The fleet brief (advisor feature #2, user-ranked; built after the
/// first honest full-campaign sweep landed): feed the app's OWN
/// machine-readable findings artifact to a served model and ask the
/// three questions a user actually has. The findings JSON is already
/// sanitized at the source — the same file a user would share.
pub fn fleet_prompt(findings_json: &str) -> String {
    format!(
        "Below is this machine's full findings report (JSON) from modelsteward: \
         hardware, measured context windows, speed baselines, quality scores, \
         and config-trial results per model. Answer three questions:\n\
         1. Which measured model is the best daily driver for coding-agent work, \
         and why (cite its numbers)? Weigh OUTPUT QUALITY (eval_score, \
         tool_reliability, fidelity) above raw speed — for coding agents a \
         wrong fast answer costs whole extra turns; prefer speed only between \
         models of equal measured quality.\n\
         2. What single change would most improve this machine's setup?\n\
         3. What is unmeasured or stale and most worth measuring next?\n\n\
         {findings_json}\n"
    )
}

/// Rebuild triage (advisor feature #3, user-ranked): given the commit
/// subjects between the running build and upstream, and the user's
/// actual model set, ask whether updating matters HERE. Informs WHEN to
/// build; the deterministic layers own how. Pure and testable.
pub fn triage_prompt(commits: &str, models: &[String], current: u64, upstream: u64) -> String {
    format!(
        "A user runs llama.cpp build b{current}; upstream is at b{upstream}. \
         Below are the commit subjects in between, and the models this user \
         actually serves. Say which commits (if any) plausibly matter for \
         THESE models — their architectures, MoE offload, vision/mmproj, \
         speculative decoding, KV cache, CUDA — and end with one line: \
         'Update: likely worth it' or 'Update: nothing relevant to your \
         models'. Judge only from the subjects; do not speculate beyond them.\n\n\
         User's models:\n{}\n\nCommits:\n{commits}\n",
        models.join("\n"),
    )
}

/// The last `n` router-log lines belonging to `model`'s child server —
/// the evidence a failure explanation reasons from. Empty when the model
/// never spawned (the stored error is then the only evidence).
pub fn log_tail_for(log: &str, model: &str, n: usize) -> String {
    let Some(port) = crate::core::evidence::child_port(log, model) else {
        return String::new();
    };
    let prefix = format!("[{port}]");
    let lines: Vec<&str> = log
        .lines()
        .filter(|l| l.starts_with(&prefix))
        .collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_prompt_carries_the_evidence_and_only_the_evidence() {
        let p = failure_prompt(
            "ornith-1.5",
            "exit status 1",
            Some(10_630),
            24_564,
            64_000,
            Some(18.6),
            "[40001] E ggml_cuda_init failed",
        );
        for needle in [
            "ornith-1.5",
            "exit status 1",
            "b10630",
            "24564 MiB",
            "18.6 GiB",
            "ggml_cuda_init failed",
        ] {
            assert!(p.contains(needle), "missing {needle} in {p}");
        }
        // The guardrails live in the system prompt, and honesty is an
        // allowed answer.
        assert!(SYSTEM.contains("Never invent"));
        assert!(SYSTEM.contains("doesn't show the cause"));
    }

    #[test]
    fn fleet_prompt_asks_the_three_questions_over_the_findings() {
        let p = fleet_prompt("{\"measurements\":{\"m\":{\"n_ctx\":115456}}}");
        assert!(p.contains("115456"), "findings ride along verbatim");
        for needle in ["daily driver", "single change", "worth measuring next"] {
            assert!(p.contains(needle), "missing {needle}");
        }
        assert!(BRIEF_SYSTEM.contains("Cite the numbers"));
    }

    #[test]
    fn triage_prompt_grounds_on_commits_and_the_users_models() {
        let p = triage_prompt(
            "abc123 cuda: fix moe expert offload\ndef456 metal: shader cleanup",
            &["qwen3.8-27b (dense, vision)".into(), "qwen3-next-80b (MoE)".into()],
            10_630,
            10_701,
        );
        for needle in ["b10630", "b10701", "moe expert offload", "qwen3-next-80b"] {
            assert!(p.contains(needle), "missing {needle}");
        }
        assert!(p.contains("nothing relevant"), "honesty exit is offered");
    }

    #[test]
    fn log_tail_filters_to_the_models_child_and_caps_length() {
        let log = "load: spawning server instance with name=a on port 40001\n\
                   load: spawning server instance with name=b on port 40002\n\
                   [40001] line1\n[40002] other\n[40001] line2\n[40001] line3\n";
        assert_eq!(log_tail_for(log, "a", 2), "[40001] line2\n[40001] line3");
        assert_eq!(log_tail_for(log, "b", 10), "[40002] other");
        assert_eq!(log_tail_for(log, "never-spawned", 10), "");
    }
}
