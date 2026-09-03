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
    /// Prompt processing at depth 0. Barely affected by occupancy
    /// (measured 1063 -> 944 t/s at 32k, ~11%), so one number is honest.
    pub pp_tps: Option<f64>,
    /// Generation from an EMPTY cache. Kept because every stored
    /// baseline and the rebuild scorecard are denominated in it — but
    /// it is NOT what a user gets mid-conversation; see `tg_deep_tps`.
    pub tg_tps: Option<f64>,
    /// Generation with `tg_depth` tokens already in the cache: the
    /// number that describes real agent work. Attention costs grow with
    /// occupancy, so this is materially lower (measured 38.25 -> 28.99
    /// t/s at 110k on a 27B; modellab handoff 2026-09-02).
    pub tg_deep_tps: Option<f64>,
    pub tg_depth: Option<u64>,
    pub build: Option<u64>,
}

/// KV depth to bench generation at, from the model's settled context.
///
/// A FIXED ladder, not a fraction of the settled context: settled
/// context itself moves between calibrations (105,472–118,016 across 28
/// runs of one model, because `--fit` sizes against whatever VRAM was
/// free), and a moving depth would silently compare two different tests
/// in the rebuild scorecard. Returns 0 when there is no room for the
/// probe (512 prompt + 128 generation) on top of the depth.
pub fn depth_rung(settled_ctx: Option<u64>) -> u64 {
    match settled_ctx {
        Some(c) if c >= 65_536 => 32_768,
        Some(c) if c >= 32_768 => 16_384,
        Some(c) if c >= 16_384 => 8_192,
        Some(c) if c >= 8_192 => 4_096,
        _ => 0,
    }
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
        tg_deep_tps: None,
        tg_depth: None,
        build: None,
    };
    let Some(tests) = body.as_array() else {
        return out;
    };
    for t in tests {
        let n_prompt = t.get("n_prompt").and_then(|v| v.as_u64()).unwrap_or(0);
        let n_gen = t.get("n_gen").and_then(|v| v.as_u64()).unwrap_or(0);
        // Absent on pre-`-d` builds and on legacy stored output: treated
        // as depth 0, which is what those runs actually measured.
        let depth = t.get("n_depth").and_then(|v| v.as_u64()).unwrap_or(0);
        if out.build.is_none() {
            out.build = t.get("build_number").and_then(|v| v.as_u64());
        }
        let avg = t.get("avg_ts").and_then(|v| v.as_f64());
        match (n_prompt > 0, n_gen > 0, depth) {
            (true, false, 0) => out.pp_tps = avg,
            (false, true, 0) => out.tg_tps = avg,
            // Deepest reported generation wins, so a multi-rung sweep
            // still yields one honest headline.
            (false, true, d) if out.tg_depth.is_none_or(|prev| d >= prev) => {
                out.tg_deep_tps = avg;
                out.tg_depth = Some(d);
            }
            _ => {} // pp-at-depth and mixed tests aren't part of the baseline
        }
    }
    out
}

/// Run the standard baseline (pp512 + tg128, 3 repetitions) for one model
/// file. `extra_args` mirrors the model's serving config (KV cache types,
/// ubatch, …). Blocks for the duration — a 27B runs about a minute.
pub fn run(
    bench: &Path,
    model: &Path,
    extra_args: &[String],
    depth: u64,
) -> Result<Baseline> {
    anyhow::ensure!(
        bench.is_file(),
        "llama-bench not found at {} — it builds alongside llama-server (Build Advisor -> rebuild)",
        bench.display()
    );
    // `-d 0,N` measures BOTH the empty cache (historical baseline, and
    // what the rebuild scorecard compares) and a realistic occupancy in
    // one model load. Depth 0 alone overstates generation by ~24% at a
    // 27B's own settled context (modellab handoff 2026-09-02).
    let depths = if depth > 0 {
        format!("0,{depth}")
    } else {
        "0".to_string()
    };
    let output = std::process::Command::new(bench)
        .args(["-o", "json", "-r", "3", "-p", "512", "-n", "128", "-d", &depths])
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
/// (Server -> Bench). Unloads OUR router's models to free the GPU (a server
/// we didn't start is never touched — it errors instead), then benches
/// `target` if given, else every measured, non-embedding model whose
/// baseline is missing or from another build. Returns (benched, failed);
/// per-model failures are narrated and skipped, not fatal — callers
/// decide whether a partial run counts as success (usability review C5).
pub fn run_baselines(
    cfg: &crate::core::settings::AppConfig,
    target: Option<String>,
    force: bool,
    cancel: &crate::core::cancel::CancelToken,
    progress: &mut dyn FnMut(String),
) -> Result<(usize, usize)> {
    use crate::core::{discover, library, router, rows, system};

    let server = system::pick_server(cfg)?;
    let bin = bench_bin(&server);
    let current_build = discover::build_of(&server);
    let dir = router::state_dir();

    // The same id -> file mapping the Library rows use.
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
    // Disabled models are never benched either — same rule as measuring
    // (live catch 2026-08-31: the predicate lived only in write_preset).
    let off = system::disabled_ids(cfg, &models);
    let targets: Vec<String> = match target {
        Some(id) => {
            anyhow::ensure!(
                by_id.contains_key(&id),
                "unknown model id {id:?} — benching needs an id that maps to a file on disk \
                 (preset alias or hub-cache id)"
            );
            anyhow::ensure!(
                !off.contains(&id),
                "{id} is disabled — enable it on the Library tab first"
            );
            vec![id]
        }
        None => by_id
            .iter()
            .filter(|(id, _)| !off.contains(id.as_str()))
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
        return Ok((0, 0));
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
            "port {} is running a server this app doesn't own ({other}); \
             benching needs the GPU free, and that server is not ours to unload",
            cfg.port
        ),
    }

    // This module's header has always said callers must make sure
    // nothing else holds the GPU — but nothing enforced it, and the
    // Ollama peer is the tenant we can see and never checked
    // (modellab handoff 2026-09-02, issue 4). A contended baseline is
    // not a slower number, it is a WRONG one, and it gets written into
    // measurements.json as if it were the model's speed. Refuse.
    let (free_vram_mib, tenant) = system::gpu_conditions(cfg);
    if let Some(t) = &tenant {
        anyhow::bail!(
            "{t} is holding the GPU — a baseline measured against that would \
             record the contention, not the model. Free the card (e.g. `ollama \
             stop <model>`, or wait for its keep-alive to expire) and bench again."
        );
    }

    let total = targets.len();
    let mut benched = 0;
    let mut failed = 0;
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
        // applied placement (cpu-moe -> all layers; n-cpu-moe -> as set).
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
        // Bench generation at a realistic KV depth as well as empty.
        let depth = depth_rung(measurements.get(id).and_then(|m| m.n_ctx));
        progress(format!(
            "[{n}/{total}] benching {id} (pp512 + tg128 ×3{})…",
            if depth > 0 {
                format!(
                    ", plus generation with {depth} tokens already in cache — \
                     the speed you actually get mid-conversation, at the cost \
                     of that extra prefill"
                )
            } else {
                " — a 27B takes about a minute".to_string()
            }
        ));
        match run(&bin, &file.path, &extra, depth) {
            Ok(b) => {
                let fmt = |v: Option<f64>| v.map(|t| format!("{t:.1}")).unwrap_or("?".into());
                progress(format!(
                    "[{n}/{total}] {id}: pp {} t/s, tg {} t/s (empty cache){}",
                    fmt(b.pp_tps),
                    fmt(b.tg_tps),
                    match (b.tg_deep_tps, b.tg_depth) {
                        (Some(t), Some(d)) => format!(", tg {t:.1} t/s at {d} depth"),
                        _ => String::new(),
                    }
                ));
                let mut entry = measurements.get(id).cloned().unwrap_or_default();
                entry.pp_tps = b.pp_tps;
                entry.tg_tps = b.tg_tps;
                entry.tg_deep_tps = b.tg_deep_tps;
                entry.tg_depth = b.tg_depth;
                entry.bench_build = b.build;
                // The condition the numbers were taken under.
                entry.free_vram_mib = free_vram_mib;
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
            Err(e) => {
                let msg = format!("{e:#}");
                // A recognized cause gets its plain-language hint right
                // in the narration (usability review C6).
                let hint = match crate::core::diagnose::classify(&msg) {
                    crate::core::diagnose::Cause::Unknown => String::new(),
                    _ => format!(" — {}", crate::core::diagnose::short_hint(&msg)),
                };
                failed += 1;
                progress(format!("[{n}/{total}] {id}: bench failed: {msg}{hint}"));
            }
        }
    }
    Ok((benched, failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_rung_is_a_fixed_ladder_that_leaves_room_for_the_probe() {
        // A generation benchmark from an EMPTY cache overstates what a
        // user gets: modellab measured this model at 38.25 t/s empty
        // and 28.99 t/s at its own settled context — 24% less
        // (handoff 2026-09-02). We now also bench at depth. The rung is
        // FIXED rather than "settled_ctx / 2" because settled context
        // itself varies run to run (105,472–118,016 across 28
        // calibrations on the same model), and a moving depth would
        // make the rebuild scorecard compare two different tests.
        assert_eq!(depth_rung(Some(131_072)), 32_768);
        assert_eq!(depth_rung(Some(65_536)), 32_768);
        assert_eq!(depth_rung(Some(65_535)), 16_384, "just under the rung");
        assert_eq!(depth_rung(Some(32_768)), 16_384);
        assert_eq!(depth_rung(Some(16_384)), 8_192);
        assert_eq!(depth_rung(Some(9_000)), 4_096);
        // Small or unknown contexts get no deep pass: the probe itself
        // needs 512 prompt + 128 gen of headroom on top of the depth.
        assert_eq!(depth_rung(Some(8_000)), 0);
        assert_eq!(depth_rung(None), 0);
        // Every rung must fit its own probe inside the context.
        for ctx in [8_192u64, 16_384, 32_768, 65_536, 131_072, 262_144] {
            let d = depth_rung(Some(ctx));
            assert!(d + 512 + 128 < ctx, "rung {d} does not fit in {ctx}");
        }
    }

    #[test]
    fn parse_output_separates_the_empty_cache_run_from_the_deep_one() {
        // llama-bench emits one row per (test, depth); the JSON carries
        // n_depth (verified in tools/llama-bench/llama-bench.cpp).
        let body = serde_json::json!([
            {"n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 1063.0, "build_number": 10760},
            {"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 38.25, "build_number": 10760},
            {"n_prompt": 512, "n_gen": 0, "n_depth": 32768, "avg_ts": 944.0, "build_number": 10760},
            {"n_prompt": 0, "n_gen": 128, "n_depth": 32768, "avg_ts": 30.10, "build_number": 10760},
        ]);
        let b = parse_output(&body);
        // Depth-0 keeps its historical meaning so old baselines and the
        // rebuild scorecard stay comparable.
        assert_eq!(b.pp_tps, Some(1063.0));
        assert_eq!(b.tg_tps, Some(38.25));
        // The honest headline: generation at a depth users actually run.
        assert_eq!(b.tg_deep_tps, Some(30.10));
        assert_eq!(b.tg_depth, Some(32_768));
        assert_eq!(b.build, Some(10760));

        // A legacy single-depth run (no n_depth field) still parses,
        // and claims no deep number rather than inventing one.
        let legacy = serde_json::json!([
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 1000.0, "build_number": 10454},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 40.0, "build_number": 10454},
        ]);
        let b = parse_output(&legacy);
        assert_eq!(b.tg_tps, Some(40.0));
        assert_eq!(b.tg_deep_tps, None);
        assert_eq!(b.tg_depth, None);
    }

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
            Baseline {
                pp_tps: None,
                tg_tps: None,
                tg_deep_tps: None,
                tg_depth: None,
                build: None
            }
        );
        assert_eq!(
            parse_output(&serde_json::json!([])),
            Baseline {
                pp_tps: None,
                tg_tps: None,
                tg_deep_tps: None,
                tg_depth: None,
                build: None
            }
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
