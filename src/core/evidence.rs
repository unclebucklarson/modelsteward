//! Runtime evidence mined from router.log (M8): what serving ACTUALLY did
//! during real sessions, as opposed to what trials measured in isolation.
//! First instrument: prompt-cache effectiveness — how many prompt tokens
//! your agent turns reused instead of reprocessing. Built empirically on
//! the observed llama-server log grammar:
//!
//!   `load: spawning server instance with name=MODEL on port PORT`
//!   `[PORT] … | task N | prompt processing, n_tokens = X, progress = …`
//!   `[PORT] … | task N | n_gen =  X, tg = …`
//!   `[PORT] … | task N | stop processing: n_tokens = X, truncated = …`
//!   `[PORT] … cache_reuse is not supported by multimodal, it will be disabled`
//!
//! Per task: total_prompt = release_total − n_gen; reused = total_prompt −
//! prompt_processed. Only tasks where all three figures exist are counted
//! (short probe turns never print n_gen — fine, the monitor targets real
//! sessions).

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelCacheStats {
    pub model: String,
    /// Turns with complete evidence.
    pub turns: u32,
    pub prompt_tokens: u64,
    pub reused_tokens: u64,
    /// llama-server announced cache-reuse disabled (any reason).
    pub reuse_disabled: bool,
    /// The non-multimodal reason: "not supported by this context" — the
    /// model's attention (SWA/hybrid) has a non-shiftable KV cache, so
    /// cache-reuse AND context-shift are off REGARDLESS of vision
    /// (found live 2026-08-27 on the text-only daily driver: removing
    /// the projector changed nothing; checkpoints are the actual lever).
    pub reuse_unsupported_context: bool,
}

impl ModelCacheStats {
    pub fn reuse_fraction(&self) -> f64 {
        if self.prompt_tokens == 0 {
            0.0
        } else {
            self.reused_tokens as f64 / self.prompt_tokens as f64
        }
    }
}

fn field_u64(line: &str, key: &str) -> Option<u64> {
    let idx = line.find(key)? + key.len();
    let rest = line[idx..].trim_start_matches([' ', '=']);
    let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

fn task_id(line: &str) -> Option<u64> {
    field_u64(line, "task")
}

fn port_prefix(line: &str) -> Option<(u32, &str)> {
    let rest = line.strip_prefix('[')?;
    let end = rest.find(']')?;
    let port: u32 = rest[..end].parse().ok()?;
    Some((port, rest[end + 1..].trim_start()))
}

/// The port of the child server instance currently serving `model`,
/// mined from the router's spawn lines (the LAST one wins — the router
/// respawns on every load and the log keeps history). Slot save/restore
/// must talk to the child directly: the router proxies chat by the
/// body's model field, but `/slots` carries no model.
pub fn child_port(log: &str, model: &str) -> Option<u16> {
    let mut port = None;
    for line in log.lines() {
        if let Some(idx) = line.find("spawning server instance with name=") {
            let rest = &line[idx + "spawning server instance with name=".len()..];
            if let Some((name, port_part)) = rest.split_once(" on port ")
                && name.trim() == model
            {
                port = port_part.trim().parse::<u16>().ok().or(port);
            }
        }
    }
    port
}

/// M8 #4 tail — models_max topology advice, from usage evidence: when
/// the log shows the user really alternating between 2+ models whose
/// files fit TOGETHER in VRAM with room to breathe, suggest raising
/// models_max so both stay resident (no swap cost). Honest about the
/// price: residents split the VRAM --fit hands out, so each gets less
/// context. Pure and testable; caller supplies sizes and VRAM.
pub fn topology_advice(
    used: &[(String, u32, u64)], // (model, turns, file bytes)
    vram_mib: u64,
    models_max: u32,
) -> Option<String> {
    if models_max != 1 || vram_mib == 0 {
        return None;
    }
    let mut active: Vec<&(String, u32, u64)> =
        used.iter().filter(|(_, turns, _)| *turns >= 3).collect();
    if active.len() < 2 {
        return None;
    }
    active.sort_by_key(|(_, turns, _)| std::cmp::Reverse(*turns));
    let (a, b) = (active[0], active[1]);
    let sum_mib = (a.2 + b.2) / (1024 * 1024);
    // Files + KV caches + runtime must all fit: only advise when the
    // pair leaves ≥30% of VRAM free for context.
    if sum_mib * 10 > vram_mib * 7 {
        return None;
    }
    Some(format!(
        "You alternate between {} ({} turns) and {} ({} turns), and both fit \
         in VRAM together ({sum_mib} of {vram_mib} MiB). Raising Max loaded \
         models to 2 (Settings) keeps both resident — no swap cost — at the \
         price of each getting a smaller fitted context. Re-measure after.",
        a.0, a.1, b.0, b.1
    ))
}

/// Mine the whole log. Later instances of a port override earlier ones for
/// the port→model mapping (ports get reused across restarts within one log).
pub fn cache_effectiveness(log: &str) -> Vec<ModelCacheStats> {
    #[derive(Default)]
    struct Task {
        prompt: Option<u64>,
        generated: Option<u64>,
        release: Option<u64>,
    }
    let mut port_model: BTreeMap<u32, String> = BTreeMap::new();
    let mut disabled_ports: std::collections::BTreeSet<u32> = Default::default();
    let mut context_ports: std::collections::BTreeSet<u32> = Default::default();
    // (port, task) → figures; flushed into per-model tallies at the end
    // using the FINAL port mapping seen — close enough for a monitor, and
    // per-restart precision isn't worth a full state machine here.
    let mut tasks: BTreeMap<(u32, u64), Task> = BTreeMap::new();

    for line in log.lines() {
        if let Some(idx) = line.find("spawning server instance with name=") {
            let rest = &line[idx + "spawning server instance with name=".len()..];
            if let Some((name, port_part)) = rest.split_once(" on port ") {
                if let Ok(port) = port_part.trim().parse::<u32>() {
                    port_model.insert(port, name.trim().to_string());
                }
            }
            continue;
        }
        let Some((port, body)) = port_prefix(line) else {
            continue;
        };
        if body.contains("cache_reuse is not supported") {
            disabled_ports.insert(port);
            if body.contains("not supported by this context") {
                context_ports.insert(port);
            }
            continue;
        }
        let Some(task) = task_id(body) else { continue };
        let t = tasks.entry((port, task)).or_default();
        if body.contains("prompt processing, n_tokens =") {
            let n = field_u64(body, "n_tokens");
            t.prompt = t.prompt.max(n);
        } else if body.contains("| n_gen =") || body.contains("| task") && body.contains(" n_gen =")
        {
            let n = field_u64(body, "n_gen");
            t.generated = t.generated.max(n);
        } else if body.contains("stop processing: n_tokens =") {
            t.release = field_u64(body, "n_tokens");
        }
    }

    let mut per_model: BTreeMap<String, ModelCacheStats> = BTreeMap::new();
    for ((port, _), t) in &tasks {
        let Some(model) = port_model.get(port) else {
            continue;
        };
        let (Some(prompt), Some(generated), Some(release)) = (t.prompt, t.generated, t.release) else {
            continue;
        };
        let total_prompt = release.saturating_sub(generated);
        if total_prompt == 0 {
            continue;
        }
        let reused = total_prompt.saturating_sub(prompt);
        let s = per_model
            .entry(model.clone())
            .or_insert_with(|| ModelCacheStats {
                model: model.clone(),
                turns: 0,
                prompt_tokens: 0,
                reused_tokens: 0,
                reuse_disabled: false,
                reuse_unsupported_context: false,
            });
        s.turns += 1;
        s.prompt_tokens += total_prompt;
        s.reused_tokens += reused;
    }
    for port in &disabled_ports {
        if let Some(model) = port_model.get(port)
            && let Some(s) = per_model.get_mut(model)
        {
            s.reuse_disabled = true;
            s.reuse_unsupported_context |= context_ports.contains(port);
        }
    }
    // Also surface disabled models that had no complete turns yet.
    for port in &disabled_ports {
        if let Some(model) = port_model.get(port) {
            per_model
                .entry(model.clone())
                .or_insert_with(|| ModelCacheStats {
                    model: model.clone(),
                    turns: 0,
                    prompt_tokens: 0,
                    reused_tokens: 0,
                    reuse_disabled: true,
                    reuse_unsupported_context: context_ports.contains(port),
                });
        }
    }
    per_model.into_values().filter(|s| s.turns > 0 || s.reuse_disabled).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mines_reuse_from_real_grammar() {
        let log = "\
1.0 I srv load: spawning server instance with name=coder-q4 on port 40001\n\
1.1 I srv load: spawning server instance with name=vision-q4 on port 40002\n\
[40002] 0.03 W srv load_model: cache_reuse is not supported by multimodal, it will be disabled\n\
[40001] 0.05 I slot print_timing: id  0 | task 10 | prompt processing, n_tokens =   2000, progress = 0.40\n\
[40001] 0.06 I slot print_timing: id  0 | task 10 | prompt processing, n_tokens =   3000, progress = 1.00\n\
[40001] 0.09 I slot print_timing: id  0 | task 10 | n_gen =    500, tg =  40.00 t/s\n\
[40001] 0.10 I slot      release: id  0 | task 10 | stop processing: n_tokens = 10500, truncated = 0\n\
[40002] 0.05 I slot print_timing: id  0 | task 7 | prompt processing, n_tokens =   9000, progress = 1.00\n\
[40002] 0.09 I slot print_timing: id  0 | task 7 | n_gen =   1000, tg =  40.00 t/s\n\
[40002] 0.10 I slot      release: id  0 | task 7 | stop processing: n_tokens = 10000, truncated = 0\n";
        let stats = cache_effectiveness(log);
        assert_eq!(stats.len(), 2);
        let coder = stats.iter().find(|s| s.model == "coder-q4").unwrap();
        // total_prompt = 10500-500 = 10000; processed 3000 → reused 7000.
        assert_eq!(coder.turns, 1);
        assert_eq!(coder.prompt_tokens, 10_000);
        assert_eq!(coder.reused_tokens, 7_000);
        assert!((coder.reuse_fraction() - 0.7).abs() < 1e-9);
        assert!(!coder.reuse_disabled);
        let vision = stats.iter().find(|s| s.model == "vision-q4").unwrap();
        // 10000-1000 = 9000 prompt, processed 9000 → zero reuse, flagged.
        assert_eq!(vision.reused_tokens, 0);
        assert!(vision.reuse_disabled);
    }

    #[test]
    fn incomplete_tasks_and_unknown_ports_are_ignored() {
        let log = "\
1.0 I srv load: spawning server instance with name=m on port 5\n\
[5] x I slot print_timing: id 0 | task 1 | prompt processing, n_tokens = 100, progress = 1.00\n\
[9] x I slot release: id 0 | task 3 | stop processing: n_tokens = 50, truncated = 0\n";
        assert!(cache_effectiveness(log).is_empty());
    }
    #[test]
    fn context_unsupported_is_distinguished_from_multimodal() {
        let log = "load: spawning server instance with name=swa-model on port 41001\n\
                   load: spawning server instance with name=vis-model on port 41002\n\
                   [41001] 0.02 W srv load_model: cache_reuse is not supported by this context, it will be disabled\n\
                   [41002] 0.03 W srv load_model: cache_reuse is not supported by multimodal, it will be disabled\n";
        let stats = cache_effectiveness(log);
        let get = |m: &str| stats.iter().find(|s| s.model == m).unwrap();
        assert!(get("swa-model").reuse_disabled && get("swa-model").reuse_unsupported_context);
        assert!(get("vis-model").reuse_disabled && !get("vis-model").reuse_unsupported_context);
    }

    #[test]
    fn topology_advice_needs_real_alternation_and_room() {
        let gib = 1024u64 * 1024 * 1024;
        let used = vec![
            ("small-a".to_string(), 12u32, 5 * gib),
            ("small-b".to_string(), 8u32, 6 * gib),
            ("rarely".to_string(), 1u32, 4 * gib),
        ];
        // 11 GiB pair in 24 GiB VRAM (under 70%) → advised.
        let line = topology_advice(&used, 24_564, 1).unwrap();
        assert!(line.contains("small-a") && line.contains("small-b"), "{line}");
        assert!(line.contains("smaller fitted context"), "{line}");
        // Already models_max=2 → quiet.
        assert!(topology_advice(&used, 24_564, 2).is_none());
        // Pair too big for the 70% budget → quiet.
        let big = vec![
            ("a".to_string(), 9u32, 12 * gib),
            ("b".to_string(), 9u32, 11 * gib),
        ];
        assert!(topology_advice(&big, 24_564, 1).is_none());
        // Only one model actually used → quiet.
        assert!(topology_advice(&used[..1], 24_564, 1).is_none());
    }

    #[test]
    fn child_port_takes_the_last_spawn_for_the_model() {
        let log = "load: spawning server instance with name=alpha on port 40001\n\
                   load: spawning server instance with name=beta on port 40002\n\
                   load: spawning server instance with name=alpha on port 40007\n";
        assert_eq!(child_port(log, "alpha"), Some(40007));
        assert_eq!(child_port(log, "beta"), Some(40002));
        assert_eq!(child_port(log, "gamma"), None);
    }
}
