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

/// The kwargs every advisory request sends. Two dialects, one truth
/// (BENCHED BUG fixed 2026-08-30, diagnosed 2026-08-29): Qwen-family
/// templates listen to `enable_thinking`; gpt-oss's Harmony template
/// ignores it and listens to `reasoning_effort` — both verified in
/// llama.cpp's common/chat.cpp. Send BOTH; templates ignore kwargs
/// they don't know.
pub fn advisory_template_kwargs() -> serde_json::Value {
    serde_json::json!({ "enable_thinking": false, "reasoning_effort": "low" })
}

/// Pull the answer out of a chat completion: `content` when present;
/// otherwise fall back to the reasoning channel — a model that
/// deliberated its whole budget still usually SAID the useful thing
/// there, and a labeled fallback beats "produced no answer".
pub fn extract_answer(body: &serde_json::Value) -> Result<String> {
    let msg = body
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"));
    let text = |key: &str| {
        msg.and_then(|m| m.get(key))
            .and_then(|s| s.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    if let Some(content) = text("content") {
        return Ok(content);
    }
    if let Some(reasoning) = text("reasoning_content") {
        return Ok(format!(
            "(the model never finalized an answer — this is its reasoning \
             channel, read with that in mind)\n\n{reasoning}"
        ));
    }
    anyhow::bail!(
        "the model produced no answer (it may have spent its whole \
         token budget on its reasoning channel) — try again or ask a \
         different model"
    )
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
                    // 2026-08-28; gpt-oss's dialect found live 2026-08-29).
                    "chat_template_kwargs": advisory_template_kwargs(),
                    // Advisory prompts must not seed the server's prompt
                    // cache — the app MEASURES that cache (review catch
                    // 2026-08-28: a 49k-token brief would have skewed the
                    // agent-turn and cache-effectiveness numbers).
                    "cache_prompt": false,
                    "max_tokens": 2500,
                    "temperature": 0.2,
                }))
                .context("advisor request")?
                .into_json()?;
        extract_answer(&body)
    }

    fn describe(&self) -> String {
        self.model.clone()
    }
}

/// Pick who answers an advisory (test-first 2026-08-29): a pinned
/// advisor always wins; otherwise quality first — but within the top
/// quality tier the FASTER model answers (the fleet's best scores at
/// 160 t/s shouldn't lose the chair to an equal at 18). Models with no
/// quality data rank below any measured-quality model; when nothing
/// has quality data, fastest measured wins. `exclude` is never chosen.
pub fn pick_advisor(
    measurements: &std::collections::BTreeMap<String, crate::core::router::Measurement>,
    pinned: Option<&str>,
    exclude: Option<&str>,
) -> Option<String> {
    if let Some(p) = pinned
        && Some(p) != exclude
        && measurements.contains_key(p)
    {
        return Some(p.to_string());
    }
    let quality = |m: &crate::core::router::Measurement| -> Option<f64> {
        let parts: Vec<f64> = [m.eval_score, m.tool_reliability, m.loop_reliability]
            .into_iter()
            .flatten()
            .collect();
        (!parts.is_empty()).then(|| parts.iter().sum::<f64>() / parts.len() as f64)
    };
    let candidates: Vec<(&String, &crate::core::router::Measurement)> = measurements
        .iter()
        .filter(|(id, m)| Some(id.as_str()) != exclude && m.n_ctx.is_some())
        .collect();
    let best_q = candidates.iter().filter_map(|(_, m)| quality(m)).fold(
        f64::NEG_INFINITY,
        f64::max,
    );
    let tier: Vec<_> = if best_q.is_finite() {
        candidates
            .iter()
            .filter(|(_, m)| quality(m).is_some_and(|q| q >= best_q - 0.02))
            .collect()
    } else {
        candidates.iter().collect()
    };
    tier.iter()
        .max_by(|(_, a), (_, b)| {
            a.tg_tps
                .unwrap_or(0.0)
                .total_cmp(&b.tg_tps.unwrap_or(0.0))
        })
        .map(|(id, _)| (*id).clone())
}

/// The curated tuning-knowledge corpus the "Ask about tuning" advisory
/// grounds on — deliberately SMALL and versioned with the code (the
/// RAG alternative was rejected 2026-08-28: stale retrieval is worse
/// than no retrieval). Everything here is something this app measured
/// or reasoned in the open.
pub const TUNING_CORPUS: &str = "\
KNOB NOTES (measured on real hardware by this app's trials):\n\
- Speculation (spec-type): model-free ngram modes cost zero VRAM. Wildly \
  per-model: +121% rewrite on one 27B, a 3x SLOWDOWN on another model's \
  rewrite. Always trial, never copy another model's setting. Acceptance \
  rate is a bad proxy for speed. Classic draft-model speculation collapsed \
  context and ran 6x slower on a 24GB card.\n\
- Prefill batch (ubatch-size): raises prompt-processing speed at some VRAM \
  cost; wins are real (+26-50%) when context headroom exists.\n\
- KV precision (cache-type-v q4_0): buys context, can cost output quality \
  and rewrite speed — the fidelity gate exists because a config that \
  degrades output is never a win. q8_0 KV measured ~2x usable context vs \
  f16 at equal quality on this class of hardware.\n\
- MoE placement (cpu-moe / n-cpu-moe): the headline for models bigger \
  than VRAM — experts in system RAM restored 4096 -> 131k-262k context. \
  Partial offload (n-cpu-moe) trades context for speed; the VRAM wall is \
  found by measurement, and PLACEMENT MUST BE APPLIED FIRST or every \
  other measurement reflects a crushed default context.\n\
- Threads: only matter once experts live on CPU; E-cores measured -45% \
  to -59% generation — P-cores-only can win.\n\
- cache-reuse: only works on models whose attention can shift its KV \
  cache; SWA/hybrid models silently disable it and resume from sparse \
  checkpoints instead (checkpoint-min-step is the lever, late edits \
  only). Vision serving disables mid-edit reuse but NOT prefix caching; \
  vision's real cost is VRAM -> context.\n\
- Sampling (temp/top-k/top-p): shapes output, not speed; agent clients \
  send their own — server defaults only matter for simple clients.\n\
- ngl on a single GPU: don't — -ngl auto + --fit already place layers.\n\
SEQUENCE for valid results: Measure+Bench -> MoE placement (if over \
VRAM) -> speed menus -> agent-turn menus -> quality probe -> apply \
winners, re-measure, sync.\n";

/// Ask-about-tuning prompt: the user's question, this machine's own
/// findings, and the curated corpus — grounded opinion, never doctrine.
pub fn tuning_prompt(question: &str, findings_json: &str) -> String {
    format!(
        "A user of modelsteward asks a tuning question. Answer from the \
         KNOB NOTES and from THIS MACHINE'S findings report below — cite \
         the user's own numbers where they exist, recommend running the \
         relevant Lab trial when the honest answer is 'measure it', and \
         say when the notes don't cover something.\n\n\
         Question: {question}\n\n{TUNING_CORPUS}\n\
         This machine's findings (JSON):\n{findings_json}\n"
    )
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
    gpus: &str,
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
         GPUs: {gpus}; system RAM: {ram_mib} MiB\n",
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
    fn advisory_kwargs_speak_both_reasoning_dialects() {
        // BENCHED BUG, fixed 2026-08-30: the fleet brief on gpt-oss-20b
        // failed with "produced no answer" because only enable_thinking
        // was sent and Harmony listens to reasoning_effort. Both kwargs
        // ship on every advisory; unknown kwargs are ignored by templates.
        let k = advisory_template_kwargs();
        assert_eq!(k["enable_thinking"], false);
        assert_eq!(k["reasoning_effort"], "low");
    }

    #[test]
    fn empty_content_falls_back_to_the_reasoning_channel_labeled() {
        let body = serde_json::json!({"choices": [{"message": {
            "content": "", "reasoning_content": "The fastest quality model is gpt-oss."
        }}]});
        let a = extract_answer(&body).unwrap();
        assert!(a.contains("reasoning channel"), "{a}");
        assert!(a.contains("gpt-oss"), "{a}");
        // Real content wins outright, no label.
        let body = serde_json::json!({"choices": [{"message": {
            "content": "Use cpu-moe.", "reasoning_content": "hmm"
        }}]});
        assert_eq!(extract_answer(&body).unwrap(), "Use cpu-moe.");
        // Nothing at all is still an honest error.
        let body = serde_json::json!({"choices": [{"message": {"content": ""}}]});
        assert!(extract_answer(&body).is_err());
    }

    fn meas(
        eval: Option<f64>,
        tools: Option<f64>,
        loops: Option<f64>,
        tg: Option<f64>,
    ) -> crate::core::router::Measurement {
        crate::core::router::Measurement {
            n_ctx: Some(50_000),
            eval_score: eval,
            tool_reliability: tools,
            loop_reliability: loops,
            tg_tps: tg,
            ..Default::default()
        }
    }

    #[test]
    fn advisor_seat_rules() {
        // Contracts (test-first 2026-08-29):
        // 1. A pinned advisor always wins (that's what pinning means).
        // 2. Otherwise: quality first — but within the top quality tier
        //    (ties), the FASTER model answers (an 18 t/s giant should
        //    not write fleet briefs a 160 t/s equal could).
        // 3. Higher quality still beats higher speed across tiers.
        // 4. `exclude` (the failing model) is never chosen.
        // 5. Models with no quality data lose to any measured-quality
        //    model, but can win when nothing has quality data (fastest).
        let mut m = std::collections::BTreeMap::new();
        m.insert("giant".to_string(), meas(Some(0.83), Some(1.0), Some(1.0), Some(18.0)));
        m.insert("fast".to_string(), meas(Some(0.83), Some(1.0), Some(1.0), Some(160.0)));
        m.insert("weak".to_string(), meas(Some(0.5), Some(0.8), None, Some(200.0)));
        m.insert("unknown".to_string(), meas(None, None, None, Some(300.0)));

        // Rule 2: equal quality -> faster answers.
        assert_eq!(pick_advisor(&m, None, None).as_deref(), Some("fast"));
        // Rule 3: speedy-but-weaker never outranks the quality tier.
        assert_ne!(pick_advisor(&m, None, None).as_deref(), Some("weak"));
        // Rule 1: pin wins.
        assert_eq!(
            pick_advisor(&m, Some("giant"), None).as_deref(),
            Some("giant")
        );
        // Rule 4: exclusion respected even for the best.
        assert_eq!(
            pick_advisor(&m, None, Some("fast")).as_deref(),
            Some("giant")
        );
        // Rule 5: nothing has quality data -> fastest measured wins.
        let mut bare = std::collections::BTreeMap::new();
        bare.insert("a".to_string(), meas(None, None, None, Some(50.0)));
        bare.insert("b".to_string(), meas(None, None, None, Some(150.0)));
        assert_eq!(pick_advisor(&bare, None, None).as_deref(), Some("b"));
    }

    #[test]
    fn tuning_prompt_grounds_on_corpus_and_the_machines_numbers() {
        let p = tuning_prompt(
            "should I use q4_0 KV cache?",
            "{\"measurements\":{\"m\":{\"n_ctx\":115456}}}",
        );
        assert!(p.contains("should I use q4_0 KV cache?"));
        assert!(p.contains("115456"), "the user's own numbers ride along");
        // The curated corpus travels with every question.
        for needle in ["cache-reuse", "fidelity", "MoE", "placement"] {
            assert!(p.contains(needle), "corpus missing {needle}");
        }
    }

    #[test]
    fn failure_prompt_carries_the_evidence_and_only_the_evidence() {
        let p = failure_prompt(
            "ornith-1.5",
            "exit status 1",
            Some(10_630),
            "RTX 4090 (24564 MiB)",
            64_000,
            Some(18.6),
            "[40001] E ggml_cuda_init failed",
        );
        for needle in [
            "ornith-1.5",
            "exit status 1",
            "b10630",
            "RTX 4090 (24564 MiB)",
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
