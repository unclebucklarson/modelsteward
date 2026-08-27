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
    /// llama-server announced cache-reuse disabled (multimodal).
    pub reuse_disabled: bool,
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
    fn child_port_takes_the_last_spawn_for_the_model() {
        let log = "load: spawning server instance with name=alpha on port 40001\n\
                   load: spawning server instance with name=beta on port 40002\n\
                   load: spawning server instance with name=alpha on port 40007\n";
        assert_eq!(child_port(log, "alpha"), Some(40007));
        assert_eq!(child_port(log, "beta"), Some(40002));
        assert_eq!(child_port(log, "gamma"), None);
    }
}
