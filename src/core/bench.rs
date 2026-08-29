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

/// The full baseline sweep, shared by the CLI (`--bench`) and the GUI
/// (Server → Bench). Unloads OUR router's models to free the GPU (a server
/// we didn't start is never touched — it errors instead), then benches
/// `target` if given, else every measured, non-embedding model whose
/// baseline is missing or from another build. Returns how many were benched;
/// per-model failures are narrated and skipped, not fatal.
pub fn run_baselines(
    cfg: &crate::core::settings::AppConfig,
    target: Option<String>,
    force: bool,
    cancel: &crate::core::cancel::CancelToken,
    progress: &mut dyn FnMut(String),
) -> Result<usize> {
    use crate::core::{discover, library, router, rows, system};

    let server = system::pick_server(cfg)?;
    let bin = bench_bin(&server);
    let current_build = discover::build_of(&server);
    let dir = router::state_dir();

    // The same id → file mapping the Library rows use.
    let models = system::scan_models(cfg, &[]);
    let ids_by_path = rows::router_ids_by_path(&models);
    let mut by_id: std::collections::BTreeMap<String, &library::ModelFile> =
        std::collections::BTreeMap::new();
    for m in &models {
        if let Some(id) = ids_by_path.get(&m.path) {
            by_id.insert(id.clone(), m);
        }
    }

    let mut measurements = router::read_measurements(&dir);
    let targets: Vec<String> = match target {
        Some(id) => {
            anyhow::ensure!(
                by_id.contains_key(&id),
                "unknown model id {id:?} — benching needs an id that maps to a file on disk \
                 (preset alias or hub-cache id)"
            );
            vec![id]
        }
        None => by_id
            .iter()
            .filter(|(id, file)| {
                let m = measurements.get(*id);
                let loadable = m.is_some_and(|m| m.n_ctx.is_some());
                let fresh = m.is_some_and(|m| {
                    m.pp_tps.is_some() && m.tg_tps.is_some() && m.bench_build == current_build
                });
                loadable
                    && !library::is_embedding(file.meta.as_ref())
                    && (force || !fresh)
            })
            .map(|(id, _)| id.clone())
            .collect(),
    };
    if targets.is_empty() {
        progress(
            "nothing to bench — every measured model already has a baseline from this build"
                .into(),
        );
        return Ok(0);
    }

    // Only now, with real work ahead, free the GPU: our router's resident
    // models get unloaded; a server we didn't start is never touched.
    match router::status(&dir, &system::router_config(cfg)) {
        router::RouterState::Down => {}
        router::RouterState::Ours { models } => {
            for m in models
                .iter()
                .filter(|m| matches!(m.status.as_str(), "loaded" | "loading" | "sleeping"))
            {
                progress(format!("unloading {} to free the GPU for benching…", m.id));
                router::unload_model(cfg.port, &m.id)?;
                router::wait_until_not_loaded(
                    cfg.port,
                    &m.id,
                    std::time::Duration::from_secs(30),
                );
            }
        }
        other => anyhow::bail!(
            "port {} is running a server this app doesn't own ({other:?}); \
             benching needs the GPU free, and that server is not ours to unload",
            cfg.port
        ),
    }

    let total = targets.len();
    let mut benched = 0;
    for (i, id) in targets.iter().enumerate() {
        cancel.check()?;
        let n = i + 1;
        let file = by_id[id];
        let ov = cfg.overrides.get(id);
        let kv = ov
            .and_then(|o| o.cache_type_kv.clone())
            .unwrap_or_else(|| router::DEFAULT_KV_TYPE.to_string());
        let mut extra = vec!["-ctk".to_string(), kv.clone(), "-ctv".to_string(), kv];
        // An over-VRAM MoE can't load raw — llama-bench has no --fit —
        // but it DOES take --n-cpu-moe, so the bench mirrors the model's
        // applied placement (cpu-moe → all layers; n-cpu-moe → as set).
        // Without an applied placement the failure was guaranteed (found
        // live 2026-08-28: GLM + the 80B errored while gpt-oss benched).
        if let Some(o) = ov {
            let has = |k: &str| o.extra.iter().find(|(ek, _)| ek == k).map(|(_, v)| v.clone());
            if let Some(n) = has("n-cpu-moe") {
                extra.extend(["-ncmoe".to_string(), n]);
            } else if has("cpu-moe").is_some() {
                extra.extend(["-ncmoe".to_string(), "999".to_string()]);
            }
        }
        progress(format!(
            "[{n}/{total}] benching {id} (pp512 + tg128 ×3 — a 27B takes about a minute)…"
        ));
        match run(&bin, &file.path, &extra) {
            Ok(b) => {
                let fmt = |v: Option<f64>| v.map(|t| format!("{t:.1}")).unwrap_or("?".into());
                progress(format!(
                    "[{n}/{total}] {id}: pp {} t/s, tg {} t/s",
                    fmt(b.pp_tps),
                    fmt(b.tg_tps)
                ));
                let mut entry = measurements.get(id).cloned().unwrap_or_default();
                entry.pp_tps = b.pp_tps;
                entry.tg_tps = b.tg_tps;
                entry.bench_build = b.build;
                measurements.insert(id.clone(), entry);
                router::write_measurements(&dir, &measurements)?; // persist per model
                let _ = crate::core::history::record(
                    &dir,
                    &crate::core::history::Entry {
                        when: crate::core::advisor::now_epoch(),
                        model: id.clone(),
                        build: b.build,
                        pp_tps: b.pp_tps,
                        tg_tps: b.tg_tps,
                        ..Default::default()
                    },
                );
                benched += 1;
            }
            Err(e) => progress(format!("[{n}/{total}] {id}: bench failed: {e:#}")),
        }
    }
    Ok(benched)
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
