//! M7 performance lab, phase 1: baseline throughput per model via
//! llama-bench — pp (prompt processing) and tg (generation) tokens/sec.
//!
//! llama-bench loads the model itself, outside the router, so callers must
//! make sure nothing else holds the GPU first (router idle, no Ollama model
//! resident) or the numbers measure the contention, not the model. Baselines
//! run with the same KV cache types the preset serves with, so the stored
//! numbers describe what OpenCode actually gets.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// One baseline: average tokens/sec for the standard pp512 and tg128 tests,
/// plus the llama.cpp build that produced them (the staleness signal — a
/// rebuild can shift throughput without touching any model file).
#[derive(Debug, Clone, PartialEq)]
pub struct Baseline {
    pub pp_tps: Option<f64>,
    pub tg_tps: Option<f64>,
    pub build: Option<u64>,
}

/// llama-bench sits next to llama-server in every install layout we know.
pub fn bench_bin(server_bin: &Path) -> PathBuf {
    server_bin.with_file_name("llama-bench")
}

/// Parse `llama-bench -o json` output: an array of test objects where the
/// pp test has `n_prompt > 0, n_gen == 0` and the tg test the reverse
/// (shape verified live against build 10454).
pub fn parse_output(body: &serde_json::Value) -> Baseline {
    let mut out = Baseline {
        pp_tps: None,
        tg_tps: None,
        build: None,
    };
    let Some(tests) = body.as_array() else {
        return out;
    };
    for t in tests {
        let n_prompt = t.get("n_prompt").and_then(|v| v.as_u64()).unwrap_or(0);
        let n_gen = t.get("n_gen").and_then(|v| v.as_u64()).unwrap_or(0);
        if out.build.is_none() {
            out.build = t.get("build_number").and_then(|v| v.as_u64());
        }
        let avg = t.get("avg_ts").and_then(|v| v.as_f64());
        match (n_prompt > 0, n_gen > 0) {
            (true, false) => out.pp_tps = avg,
            (false, true) => out.tg_tps = avg,
            _ => {} // mixed pp+tg tests aren't part of the baseline
        }
    }
    out
}

/// Run the standard baseline (pp512 + tg128, 3 repetitions) for one model
/// file. `extra_args` mirrors the model's serving config (KV cache types,
/// ubatch, …). Blocks for the duration — a 27B runs about a minute.
pub fn run(bench: &Path, model: &Path, extra_args: &[String]) -> Result<Baseline> {
    anyhow::ensure!(
        bench.is_file(),
        "llama-bench not found at {} — it builds alongside llama-server (Build Advisor → rebuild)",
        bench.display()
    );
    let output = std::process::Command::new(bench)
        .args(["-o", "json", "-r", "3", "-p", "512", "-n", "128"])
        .arg("-m")
        .arg(model)
        .args(extra_args)
        .output()
        .with_context(|| format!("running {}", bench.display()))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = err.lines().filter(|l| !l.trim().is_empty()).collect();
        let tail = tail.iter().rev().take(3).rev().cloned().collect::<Vec<_>>();
        anyhow::bail!(
            "llama-bench failed for {}: {}",
            model.display(),
            tail.join(" | ")
        );
    }
    let body: serde_json::Value =
        serde_json::from_slice(&output.stdout).context("llama-bench emitted non-JSON")?;
    let b = parse_output(&body);
    anyhow::ensure!(
        b.pp_tps.is_some() || b.tg_tps.is_some(),
        "llama-bench ran but produced no pp/tg results"
    );
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pp_and_tg_from_real_shape() {
        // Trimmed from a live run (build 10454).
        let body: serde_json::Value = serde_json::from_str(
            r#"[
            {"build_number": 10454, "n_prompt": 512, "n_gen": 0, "avg_ts": 1234.5, "stddev_ts": 10.0},
            {"build_number": 10454, "n_prompt": 0, "n_gen": 128, "avg_ts": 42.25, "stddev_ts": 0.5},
            {"build_number": 10454, "n_prompt": 512, "n_gen": 128, "avg_ts": 99.0}
        ]"#,
        )
        .unwrap();
        let b = parse_output(&body);
        assert_eq!(b.pp_tps, Some(1234.5));
        assert_eq!(b.tg_tps, Some(42.25));
        assert_eq!(b.build, Some(10454), "mixed pp+tg row ignored");
    }

    #[test]
    fn tolerates_junk() {
        assert_eq!(
            parse_output(&serde_json::json!({"not": "an array"})),
            Baseline { pp_tps: None, tg_tps: None, build: None }
        );
        assert_eq!(
            parse_output(&serde_json::json!([])),
            Baseline { pp_tps: None, tg_tps: None, build: None }
        );
    }

    #[test]
    fn bench_bin_is_a_sibling() {
        assert_eq!(
            bench_bin(Path::new("/opt/llama.cpp/bin/llama-server")),
            PathBuf::from("/opt/llama.cpp/bin/llama-bench")
        );
    }
}
