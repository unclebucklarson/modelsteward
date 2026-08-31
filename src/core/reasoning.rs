//! What reasoning levels a model's OWN chat template accepts.
//!
//! Promoted to a first-class ⚙ field on user request 2026-08-31, after
//! a live finding: Qwen3.8-27B's template does
//! `reasoning_effort|default('xhigh')` — the most thinking of any level
//! unless told otherwise — while gpt-oss's Harmony template ignores
//! `enable_thinking` entirely and honors only `reasoning_effort`.
//! Family folklore is therefore useless here: the levels are read out
//! of the template that will actually render the prompt, so a dropdown
//! never offers a value the model would raise on.
//!
//! Pure string work over a Jinja template — no network, no subprocess.

use serde::{Deserialize, Serialize};

/// llama.cpp's documented ladder (`--reasoning-effort` help text,
/// b10680). Used only as a LABELLED fallback when a template clearly
/// consumes `reasoning_effort` but states no explicit set.
pub const LLAMA_CPP_LEVELS: [&str; 5] = ["minimal", "low", "medium", "high", "xhigh"];

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningSupport {
    /// Accepted effort levels, in the order the template lists them.
    pub levels: Vec<String>,
    /// True when `levels` came from the template's own membership test;
    /// false when we fell back to llama.cpp's documented ladder.
    pub levels_confirmed: bool,
    /// What the template uses when no effort is supplied — the value
    /// worth knowing, since it is often the most expensive one.
    pub default_level: Option<String>,
    /// The template branches on `enable_thinking`, so reasoning can be
    /// switched off wholesale (`--reasoning off`).
    pub can_disable: bool,
}

impl ReasoningSupport {
    pub fn is_supported(&self) -> bool {
        !self.levels.is_empty() || self.can_disable
    }
}

/// Read a chat template's reasoning contract. `None` when the template
/// never mentions reasoning at all.
pub fn parse_support(template: &str) -> Option<ReasoningSupport> {
    let mentions_effort = template.contains("reasoning_effort");
    let can_disable = template.contains("enable_thinking");
    if !mentions_effort && !can_disable {
        return None;
    }
    let mut s = ReasoningSupport {
        can_disable,
        ..Default::default()
    };
    if mentions_effort {
        s.default_level = default_after(template, "reasoning_effort|default(")
            .or_else(|| default_after(template, "reasoning_effort | default("));
        s.levels = membership_levels(template).unwrap_or_default();
        s.levels_confirmed = !s.levels.is_empty();
        if s.levels.is_empty() {
            s.levels = LLAMA_CPP_LEVELS.iter().map(|l| l.to_string()).collect();
        }
        // A stated default that somehow isn't in the set still belongs
        // in the list — the model plainly accepts it.
        if let Some(d) = &s.default_level
            && !s.levels.iter().any(|l| l == d)
        {
            s.levels.insert(0, d.clone());
        }
    }
    Some(s)
}

/// The quoted argument of `…|default('X')`.
fn default_after(template: &str, needle: &str) -> Option<String> {
    let rest = &template[template.find(needle)? + needle.len()..];
    quoted(rest)
}

/// Names a template assigns from `reasoning_effort`, so the validity
/// check can be found even when it tests an alias — the real Qwen3.8
/// template uses `resolved_reasoning_effort`, and a shorter alias would
/// otherwise slip past (test catch 2026-08-31).
fn effort_aliases(template: &str) -> Vec<String> {
    let mut names = vec!["reasoning_effort".to_string()];
    for line in template.lines() {
        let Some(at) = line.find("set ") else { continue };
        let rest = &line[at + 4..];
        let Some(eq) = rest.find('=') else { continue };
        if !rest[eq..].contains("reasoning_effort") {
            continue;
        }
        let name = rest[..eq].trim();
        if !name.is_empty()
            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
            && !names.iter().any(|n| n == name)
        {
            names.push(name.to_string());
        }
    }
    names
}

/// Levels from the template's own validity check — the authoritative
/// list, e.g. `resolved_reasoning_effort not in ('xhigh','medium','low')`.
fn membership_levels(template: &str) -> Option<Vec<String>> {
    let names = effort_aliases(template);
    // Search every ` in (` / ` in [` that follows a reasoning mention on
    // the same line: templates guard other variables too.
    for line in template.lines() {
        if !names.iter().any(|n| line.contains(n.as_str())) {
            continue;
        }
        for opener in [" in (", " in ["] {
            let Some(at) = line.find(opener) else { continue };
            let rest = &line[at + opener.len()..];
            let end = rest.find([')', ']'])?;
            let items: Vec<String> = split_quoted(&rest[..end]);
            if items.len() >= 2 {
                return Some(items);
            }
        }
    }
    None
}

/// First single- or double-quoted string in `s`.
fn quoted(s: &str) -> Option<String> {
    let start = s.find(['\'', '"'])?;
    let q = s.as_bytes()[start] as char;
    let rest = &s[start + 1..];
    let end = rest.find(q)?;
    Some(rest[..end].to_string())
}

/// Every quoted string in a comma-separated fragment.
fn split_quoted(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(start) = rest.find(['\'', '"']) {
        let q = rest.as_bytes()[start] as char;
        let after = &rest[start + 1..];
        let Some(end) = after.find(q) else { break };
        out.push(after[..end].to_string());
        rest = &after[end + 1..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim from the LIVE Qwen3.8-27B template, read off its child
    /// port 2026-08-31 (the model Scott watched over-think all day).
    const QWEN38: &str = "{%- if enable_thinking is undefined or enable_thinking is true %}\n\
        {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n\
        {%- if resolved_reasoning_effort == 'high' %}\n\
        {%- set resolved_reasoning_effort = 'xhigh' %}\n\
        {%- endif %}\n\
        {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}\n\
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. \
        Supported types are xhigh (default), medium, and low.') }}\n\
        {%- endif %}\n";

    #[test]
    fn qwen38_levels_come_from_its_own_template() {
        let s = parse_support(QWEN38).unwrap();
        // The finding that started this: the default is the MOST
        // expensive level, not a middle one.
        assert_eq!(s.default_level.as_deref(), Some("xhigh"));
        assert_eq!(s.levels, vec!["xhigh", "medium", "low"]);
        assert!(s.levels_confirmed, "read from the membership test, not guessed");
        // 'high' is silently rewritten to 'xhigh' by this template, so
        // it is deliberately NOT offered as a distinct choice.
        assert!(!s.levels.iter().any(|l| l == "high"));
        assert!(s.can_disable, "template branches on enable_thinking");
        assert!(s.is_supported());
    }

    #[test]
    fn harmony_style_effort_without_a_stated_set_falls_back_labelled() {
        // gpt-oss's Harmony consumes reasoning_effort but states no
        // membership tuple, and never mentions enable_thinking (the
        // 2026-08-29 advisory bug).
        let t = "{%- set eff = reasoning_effort %}<|start|>system reasoning: {{ eff }}";
        let s = parse_support(t).unwrap();
        assert!(!s.levels_confirmed, "must be marked unconfirmed");
        assert_eq!(s.levels, LLAMA_CPP_LEVELS.to_vec());
        assert!(!s.can_disable);
    }

    #[test]
    fn thinking_toggle_without_effort_levels() {
        // Qwen3-era templates: on/off only, no effort ladder.
        let s = parse_support("{%- if enable_thinking %}<think>{% endif %}").unwrap();
        assert!(s.can_disable);
        assert!(s.levels.is_empty(), "no ladder to offer");
        assert!(s.is_supported());
    }

    #[test]
    fn templates_without_reasoning_say_so() {
        assert!(parse_support("{{ messages[0].content }}").is_none());
        assert!(parse_support("").is_none());
    }

    #[test]
    fn a_default_outside_the_stated_set_is_still_offered() {
        // Defensive: a template whose default isn't in its own tuple
        // still accepts that default, so hiding it would be wrong.
        let t = "{%- set e = reasoning_effort|default('turbo') %}\n\
                 {%- if e not in ('low', 'medium') %}fail{% endif %}";
        let s = parse_support(t).unwrap();
        assert_eq!(s.levels, vec!["turbo", "low", "medium"]);
    }
}
