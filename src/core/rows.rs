//! One row per model, joined from every truth source: disk (library),
//! the running router (server status), measurements, and opencode.json.
//! This is what the Library tab renders — a model's whole story in one line.

use crate::core::gguf::GgufMeta;
use crate::core::library::{self, ModelFile, Source};
use crate::core::router::{Measurements, RouterModel};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum AdviceLevel {
    /// Fine as-is.
    Good,
    /// Usable with caveats worth knowing.
    Warn,
    /// Not usable on this machine / this build.
    Bad,
    /// We simply don't know yet.
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct Row {
    /// Human name (file/ollama/repo based).
    pub display: String,
    /// The id the router serves this model under (preset alias or cache
    /// id). `None` when we can't predict it (then it can't be loaded from
    /// the Library row either).
    pub router_id: Option<String>,
    pub source: String,
    pub size_bytes: u64,
    pub quant: String,
    pub train_ctx: Option<u64>,
    /// Measured settled context, when calibrated on this machine.
    pub measured_ctx: Option<u64>,
    pub failure: Option<String>,
    /// Live router status ("loaded"/"unloaded"/…), when the router is up
    /// and offers this model.
    pub server_status: Option<String>,
    pub in_opencode: bool,
    pub advice_level: AdviceLevel,
    pub advice: String,
}

/// Hardware picture for advice. `vram_mib` is the largest single device
/// (models load onto one card until told otherwise); `ram_mib` total system.
#[derive(Debug, Clone, Copy)]
pub struct Hardware {
    pub vram_mib: u64,
    pub ram_mib: u64,
}

pub fn read_ram_mib() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("MemTotal:")?
                    .trim()
                    .strip_suffix(" kB")?
                    .parse::<u64>()
                    .ok()
            })
        })
        .map(|kb| kb / 1024)
        .unwrap_or(0)
}

/// Context below which agentic coding gets painful: OpenCode's system
/// prompt + tool schemas + a modest file few files already approach this.
const AGENT_MIN_CTX: u64 = 24_576;

/// Turn a stored load-failure string into something a user can act on.
pub fn failure_hint(error: &str) -> String {
    let lower = error.to_lowercase();
    if lower.contains("rope.dimension_sections")
        || lower.contains("hyperparameters")
        || lower.contains("unknown model architecture")
        || lower.contains("unknown architecture")
    {
        "this model's format is newer than your llama.cpp build — updating/rebuilding llama.cpp will likely fix it".into()
    } else if lower.contains("wrong number of tensors") {
        "the file looks like a partial download or a multimodal blob llama.cpp can't load standalone".into()
    } else if lower.contains("out of memory") || lower.contains("failed to allocate") {
        "not enough memory to load with current settings".into()
    } else {
        "load failed — see the Server tab log for the exact error".into()
    }
}

fn advice_for(
    meta: Option<&GgufMeta>,
    size_bytes: u64,
    measured: Option<u64>,
    failure: Option<&str>,
    hw: Hardware,
) -> (AdviceLevel, String) {
    if let Some(err) = failure {
        return (AdviceLevel::Bad, failure_hint(err));
    }
    if let Some(ctx) = measured {
        return if ctx < AGENT_MIN_CTX {
            (
                AdviceLevel::Warn,
                format!(
                    "runs, but the measured context ({ctx}) is small for agentic coding — consider a smaller quant"
                ),
            )
        } else {
            (
                AdviceLevel::Good,
                format!("measured: {ctx} context on this machine"),
            )
        };
    }
    // Unmeasured: predict from sizes. Weights + KV need headroom; a file
    // near/above VRAM spills to CPU (slow), near RAM+VRAM won't run at all.
    let size_mib = size_bytes / (1024 * 1024);
    if meta.is_none() {
        return (
            AdviceLevel::Bad,
            "unreadable GGUF header — file may be corrupt or partial".into(),
        );
    }
    if hw.vram_mib > 0 && size_mib > hw.vram_mib + hw.ram_mib.saturating_sub(8 * 1024) {
        (
            AdviceLevel::Bad,
            format!(
                "too large for this machine ({:.0} GB file vs {:.0} GB VRAM + {:.0} GB RAM)",
                size_mib as f64 / 1024.0,
                hw.vram_mib as f64 / 1024.0,
                hw.ram_mib as f64 / 1024.0
            ),
        )
    } else if hw.vram_mib > 0 && size_mib > hw.vram_mib.saturating_sub(1024) {
        (
            AdviceLevel::Warn,
            format!(
                "bigger than your {:.0} GB VRAM — will run partly on CPU (slow); Load to measure what you actually get",
                hw.vram_mib as f64 / 1024.0
            ),
        )
    } else {
        (
            AdviceLevel::Unknown,
            "not measured yet — Load it (or Set Up Everything) for real numbers".into(),
        )
    }
}

/// Predict the router id for a library file: preset alias normally, cache
/// id for HF-hub files.
fn predicted_router_id(m: &ModelFile) -> Option<String> {
    match &m.source {
        Source::HfHub { .. } => m.router_cache_id(),
        _ => m.meta.is_some().then(|| library::alias_suggestion(m)),
    }
}

pub fn assemble(
    models: &[ModelFile],
    router_models: &[RouterModel],
    measurements: &Measurements,
    opencode_ids: &[String],
    hw: Hardware,
) -> Vec<Row> {
    // A router entry belongs to exactly ONE row — two quants of the same
    // repo must never share an identity (a Q4 file falling back onto the
    // repo's only listed variant, the Q5, made both rows load/measure as
    // one model). Exact id matches claim first; the unique-repo fallback
    // runs ONLY for files whose tag we couldn't predict at all, and only
    // onto still-unclaimed entries.
    let mut used_router_ids: std::collections::HashSet<String> = models
        .iter()
        .filter_map(|m| {
            let id = predicted_router_id(m)?;
            router_models.iter().any(|rm| rm.id == id).then_some(id)
        })
        .collect();
    let mut rows: Vec<Row> = Vec::new();

    for m in models {
        let mut router_id = predicted_router_id(m);
        let mut server_status = None;
        if let Some(id) = &router_id {
            if let Some(rm) = router_models.iter().find(|rm| &rm.id == id) {
                server_status = Some(rm.status.clone());
            }
            // Predicted-but-unoffered: keep the id for display, no status —
            // the router doesn't serve this exact variant right now.
        } else if let Source::HfHub { repo } = &m.source {
            let of_repo: Vec<&RouterModel> = router_models
                .iter()
                .filter(|rm| {
                    rm.id.starts_with(&format!("{repo}:")) && !used_router_ids.contains(&rm.id)
                })
                .collect();
            if of_repo.len() == 1 {
                router_id = Some(of_repo[0].id.clone());
                server_status = Some(of_repo[0].status.clone());
                used_router_ids.insert(of_repo[0].id.clone());
            }
        }

        let measurement = router_id.as_ref().and_then(|id| measurements.get(id));
        let measured_ctx = measurement.and_then(|mm| mm.n_ctx);
        let failure = measurement.and_then(|mm| mm.error.clone());
        let (advice_level, advice) = if !router_models.is_empty()
            && router_id.is_some()
            && server_status.is_none()
            && measured_ctx.is_none()
            && failure.is_none()
        {
            (
                AdviceLevel::Unknown,
                "on disk, but the router doesn't currently offer this variant".into(),
            )
        } else {
            advice_for(
                m.meta.as_ref(),
                m.file_size,
                measured_ctx,
                failure.as_deref(),
                hw,
            )
        };
        rows.push(Row {
            display: m.display_name(),
            in_opencode: router_id
                .as_ref()
                .is_some_and(|id| opencode_ids.contains(id)),
            router_id,
            source: match &m.source {
                Source::Shelf => "shelf".into(),
                Source::Ollama { .. } => "ollama".into(),
                Source::HfHub { .. } => "hf cache".into(),
            },
            size_bytes: m.file_size,
            quant: m
                .meta
                .as_ref()
                .and_then(|x| x.quantization.clone())
                .unwrap_or_default(),
            train_ctx: m.meta.as_ref().and_then(|x| x.context_length),
            measured_ctx,
            failure,
            server_status,
            advice_level,
            advice,
        });
    }

    // Router models we couldn't tie to a file (unpredictable cache tags,
    // deleted files still listed): shown so nothing servable is invisible.
    for rm in router_models {
        if used_router_ids.contains(&rm.id) {
            continue;
        }
        let measurement = measurements.get(&rm.id);
        let measured_ctx = measurement.and_then(|mm| mm.n_ctx);
        let failure = measurement.and_then(|mm| mm.error.clone());
        let (advice_level, advice) = match (&measured_ctx, &failure) {
            (Some(ctx), _) if *ctx >= AGENT_MIN_CTX => (
                AdviceLevel::Good,
                format!("measured: {ctx} context on this machine"),
            ),
            (Some(ctx), _) => (
                AdviceLevel::Warn,
                format!("runs, but measured context ({ctx}) is small for agentic coding"),
            ),
            (None, Some(err)) => (AdviceLevel::Bad, failure_hint(err)),
            (None, None) => (
                AdviceLevel::Unknown,
                "offered by the router (file location unknown) — Load to measure".into(),
            ),
        };
        rows.push(Row {
            display: rm.id.clone(),
            in_opencode: opencode_ids.contains(&rm.id),
            router_id: Some(rm.id.clone()),
            source: rm.source.clone().unwrap_or_else(|| "router".into()),
            size_bytes: 0,
            quant: String::new(),
            train_ctx: None,
            measured_ctx,
            failure,
            server_status: Some(rm.status.clone()),
            advice_level,
            advice,
        });
    }

    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::router::Measurement;
    use std::path::PathBuf;

    const HW: Hardware = Hardware {
        vram_mib: 24_000,
        ram_mib: 64_000,
    };

    fn file(name: &str, source: Source, gb: u64, meta: bool) -> ModelFile {
        ModelFile {
            path: PathBuf::from(format!("/m/{name}")),
            file_size: gb * 1024 * 1024 * 1024,
            source,
            meta: meta.then(GgufMeta::default),
        }
    }

    fn rm(id: &str, status: &str, source: &str) -> RouterModel {
        RouterModel {
            id: id.into(),
            status: status.into(),
            source: Some(source.into()),
            args_fp: None,
        }
    }

    #[test]
    fn advice_grades_by_size_measurement_and_failure() {
        // Too large outright.
        let (lvl, msg) = advice_for(Some(&GgufMeta::default()), 120 * 1024 * 1024 * 1024, None, None, HW);
        assert_eq!(lvl, AdviceLevel::Bad);
        assert!(msg.contains("too large"), "{msg}");
        // Bigger than VRAM → CPU spill warning.
        let (lvl, msg) = advice_for(Some(&GgufMeta::default()), 30 * 1024 * 1024 * 1024, None, None, HW);
        assert_eq!(lvl, AdviceLevel::Warn);
        assert!(msg.contains("partly on CPU"), "{msg}");
        // Measured small ctx.
        let (lvl, msg) = advice_for(Some(&GgufMeta::default()), 10 << 30, Some(8192), None, HW);
        assert_eq!(lvl, AdviceLevel::Warn);
        assert!(msg.contains("small for agentic"), "{msg}");
        // Measured healthy.
        let (lvl, _) = advice_for(Some(&GgufMeta::default()), 10 << 30, Some(80_000), None, HW);
        assert_eq!(lvl, AdviceLevel::Good);
        // Failure dominates.
        let (lvl, msg) = advice_for(
            Some(&GgufMeta::default()),
            10 << 30,
            None,
            Some("key qwen35moe.rope.dimension_sections has wrong array length"),
            HW,
        );
        assert_eq!(lvl, AdviceLevel::Bad);
        assert!(msg.contains("newer than your llama.cpp build"), "{msg}");
    }

    #[test]
    fn joins_disk_router_measurements_and_config() {
        let models = vec![file("A.gguf", Source::Shelf, 10, true)];
        let routers = vec![rm("a", "loaded", "preset")];
        let mut meas = Measurements::new();
        meas.insert(
            "a".into(),
            Measurement {
                n_ctx: Some(90_000),
                error: None,
                args_fp: None,
                env_fp: None,
            },
        );
        let rows = assemble(&models, &routers, &meas, &["a".into()], HW);
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.router_id.as_deref(), Some("a"));
        assert_eq!(r.server_status.as_deref(), Some("loaded"));
        assert_eq!(r.measured_ctx, Some(90_000));
        assert!(r.in_opencode);
        assert_eq!(r.advice_level, AdviceLevel::Good);
    }

    #[test]
    fn unique_repo_match_rescues_unpredictable_cache_tags() {
        // No quant token in the filename → tag not derivable → fallback OK.
        let models = vec![file(
            "gemma-4-31B-instruct.gguf",
            Source::HfHub {
                repo: "google/gemma-4-31B-it-qat-q4_0-gguf".into(),
            },
            17,
            true,
        )];
        let routers = vec![rm("google/gemma-4-31B-it-qat-q4_0-gguf:IT", "unloaded", "cache")];
        let rows = assemble(&models, &routers, &Measurements::new(), &[], HW);
        assert_eq!(rows.len(), 1, "must join, not duplicate");
        assert_eq!(
            rows[0].router_id.as_deref(),
            Some("google/gemma-4-31B-it-qat-q4_0-gguf:IT")
        );
    }

    #[test]
    fn two_quants_of_one_repo_never_share_an_identity() {
        // The user's real bug: Q4 and Q5 files of the same repo, router
        // offers only the Q5 → the Q4 must NOT glue itself onto it.
        let models = vec![
            file(
                "Qwen3.8-27B-UD-Q4_K_XL.gguf",
                Source::HfHub {
                    repo: "unsloth/Qwen3.8-27B-GGUF".into(),
                },
                18,
                true,
            ),
            file(
                "Qwen3.8-27B-UD-Q5_K_XL.gguf",
                Source::HfHub {
                    repo: "unsloth/Qwen3.8-27B-GGUF".into(),
                },
                20,
                true,
            ),
        ];
        let routers = vec![rm("unsloth/Qwen3.8-27B-GGUF:Q5_K_XL", "loaded", "cache")];
        let rows = assemble(&models, &routers, &Measurements::new(), &[], HW);
        assert_eq!(rows.len(), 2);
        let q4 = rows.iter().find(|r| r.display.contains("Q4_K_XL")).unwrap();
        let q5 = rows.iter().find(|r| r.display.contains("Q5_K_XL")).unwrap();
        assert_eq!(
            q5.router_id.as_deref(),
            Some("unsloth/Qwen3.8-27B-GGUF:Q5_K_XL")
        );
        assert_eq!(q5.server_status.as_deref(), Some("loaded"));
        assert_eq!(
            q4.router_id.as_deref(),
            Some("unsloth/Qwen3.8-27B-GGUF:Q4_K_XL"),
            "keeps its own (unoffered) identity"
        );
        assert!(q4.server_status.is_none(), "not offered ≠ offered");
        assert!(q4.advice.contains("doesn't currently offer"), "{}", q4.advice);
    }

    #[test]
    fn router_only_models_still_get_rows() {
        let routers = vec![rm("mystery/model:Q4", "unloaded", "cache")];
        let rows = assemble(&[], &routers, &Measurements::new(), &[], HW);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].source, "cache");
        assert!(rows[0].advice.contains("Load to measure"));
    }
}
