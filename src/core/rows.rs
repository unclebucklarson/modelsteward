//! One row per model, joined from every truth source: disk (library),
//! the running router (server status), measurements, and opencode.json.
//! This is what the Library tab renders — a model's whole story in one line.

use crate::core::gguf::GgufMeta;
use crate::core::library::{self as library, ModelFile, Source};
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
    /// Set when the filename and header disagree about the quant — the
    /// column shows the filename's (more truthful for dynamic quants);
    /// this holds the header's stamp for the tooltip.
    pub quant_header_disagrees: Option<String>,
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
    /// Disk path when this row came from a scanned file (router-only rows
    /// have none). Drives the Archive action.
    pub path: Option<std::path::PathBuf>,
    /// Archivable = lives in a store some other tool owns/prunes.
    pub archivable: bool,
    /// Feature badges: vision projector paired, MTP tensors present,
    /// embedding architecture.
    pub vision: bool,
    pub mtp: bool,
    pub embedding: bool,
    /// Baseline throughput (llama-bench, M7): prompt-processing and
    /// generation tokens/sec, when benched on this machine.
    pub pp_tps: Option<f64>,
    pub tg_tps: Option<f64>,
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
pub const AGENT_MIN_CTX: u64 = 24_576;

/// Turn a stored load-failure string into something a user can act on.
/// One-liner for table cells; the full story lives in core::diagnose.
pub fn failure_hint(error: &str) -> String {
    crate::core::diagnose::short_hint(error)
}

fn advice_for(
    meta: Option<&GgufMeta>,
    size_bytes: u64,
    measured: Option<u64>,
    tool_call: Option<bool>,
    failure: Option<&str>,
    hw: Hardware,
) -> (AdviceLevel, String) {
    if let Some(err) = failure {
        return (AdviceLevel::Bad, failure_hint(err));
    }
    if let Some(ctx) = measured {
        return if tool_call == Some(false) {
            (
                AdviceLevel::Warn,
                format!(
                    "measured {ctx} context, but produced NO tool call when probed — of limited use to OpenCode agents"
                ),
            )
        } else if ctx < AGENT_MIN_CTX {
            (
                AdviceLevel::Warn,
                format!(
                    "runs, but the measured context ({ctx}) is small for agentic coding — consider a smaller quant"
                ),
            )
        } else {
            (
                AdviceLevel::Good,
                format!(
                    "measured: {ctx} context{} on this machine",
                    if tool_call == Some(true) { ", tool calls work" } else { "" }
                ),
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

/// Router ids for library files, computed with the SAME logic the preset
/// generator uses — including duplicate-alias suffixing ("-2"). Predicting
/// aliases independently glued two same-named files onto one router entry
/// while the real "-2" entry showed as a phantom row (user-found). Public
/// because the bench CLI needs the same id → file mapping.
pub fn router_ids_by_path(
    models: &[ModelFile],
) -> std::collections::HashMap<std::path::PathBuf, String> {
    let mut out: std::collections::HashMap<std::path::PathBuf, String> =
        crate::core::router::default_entries(models)
            .into_iter()
            .map(|(alias, m, _)| (m.path.clone(), alias))
            .collect();
    for m in models {
        if let Source::HfHub { .. } = &m.source
            && let Some(id) = m.router_cache_id()
        {
            out.insert(m.path.clone(), id);
        }
    }
    out
}

/// For a router cache entry ("org/repo:TAG") no scanned file claimed, find a
/// scanned file that carries the same model: an HF-hub file of the same repo,
/// or any file whose name contains the repo's base name — same quant tag
/// either way. Loading the cache entry downloads from HuggingFace; a match
/// here means those bytes (or a prior revision of them) are already on disk.
fn local_equivalent<'a>(models: &'a [ModelFile], cache_id: &str) -> Option<&'a ModelFile> {
    let (repo, tag) = cache_id.rsplit_once(':')?;
    let tag = tag.to_uppercase();
    let mut base = repo.rsplit('/').next()?.to_uppercase();
    if let Some(stripped) = base.strip_suffix("-GGUF") {
        base = stripped.to_string();
    }
    models.iter().find(|m| {
        if library::quant_token_from_filename(&m.path) != Some(tag.clone()) {
            return false;
        }
        if let Source::HfHub { repo: r } = &m.source {
            return r == repo;
        }
        let stem = m
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().to_uppercase())
            .unwrap_or_default();
        !base.is_empty() && stem.contains(&base)
    })
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
    let ids_by_path = router_ids_by_path(models);
    let mut used_router_ids: std::collections::HashSet<String> = models
        .iter()
        .filter_map(|m| {
            let id = ids_by_path.get(&m.path)?;
            router_models
                .iter()
                .any(|rm| &rm.id == id)
                .then(|| id.clone())
        })
        .collect();
    // Same file name + same size in two places = almost certainly the same
    // bytes stored twice; say so instead of leaving the user to infer it.
    let mut name_size_counts: std::collections::HashMap<(String, u64), u32> =
        std::collections::HashMap::new();
    for m in models {
        if let Some(name) = m.path.file_name() {
            *name_size_counts
                .entry((name.to_string_lossy().into_owned(), m.file_size))
                .or_default() += 1;
        }
    }
    let mut rows: Vec<Row> = Vec::new();

    for m in models {
        let mut router_id = ids_by_path.get(&m.path).cloned();
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
        let tool_call = measurement.and_then(|mm| mm.tool_call);
        let failure = measurement.and_then(|mm| mm.error.clone());
        let (pp_tps, tg_tps) = (
            measurement.and_then(|mm| mm.pp_tps),
            measurement.and_then(|mm| mm.tg_tps),
        );
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
                tool_call,
                failure.as_deref(),
                hw,
            )
        };
        let vision = m.mmproj.is_some();
        let mtp = m.meta.as_ref().is_some_and(|x| x.has_mtp);
        let embedding = library::is_embedding(m.meta.as_ref());
        let (advice_level, advice) = if embedding {
            (
                AdviceLevel::Good,
                format!(
                    "embedding model — serves /v1/embeddings for RAG/search (e.g. a notes app){}; not a chat/agent model",
                    measured_ctx.map(|c| format!(", measured {c} ctx")).unwrap_or_default()
                ),
            )
        } else {
            (advice_level, advice)
        };
        let is_dup = m
            .path
            .file_name()
            .map(|n| {
                name_size_counts
                    .get(&(n.to_string_lossy().into_owned(), m.file_size))
                    .copied()
                    .unwrap_or(0)
                    > 1
            })
            .unwrap_or(false);
        let advice = if is_dup {
            format!("{advice}; note: another copy with the same name + size exists — likely the same model stored twice")
        } else {
            advice
        };
        rows.push(Row {
            display: m.display_name(),
            in_opencode: router_id
                .as_ref()
                .is_some_and(|id| opencode_ids.contains(id)),
            router_id,
            path: Some(m.path.clone()),
            archivable: !matches!(m.source, Source::Shelf),
            source: match &m.source {
                Source::Shelf => "shelf".into(),
                Source::Ollama { .. } => "ollama".into(),
                Source::HfHub { .. } => "hf cache".into(),
            },
            size_bytes: m.file_size,
            quant: {
                let header = m.meta.as_ref().and_then(|x| x.quantization.clone());
                library::quant_token_from_filename(&m.path)
                    .or(header)
                    .unwrap_or_default()
            },
            quant_header_disagrees: {
                let header = m.meta.as_ref().and_then(|x| x.quantization.clone());
                match (library::quant_token_from_filename(&m.path), header) {
                    (Some(f), Some(h)) if f != h => Some(h),
                    _ => None,
                }
            },
            train_ctx: m.meta.as_ref().and_then(|x| x.context_length),
            measured_ctx,
            failure,
            server_status,
            advice_level,
            advice,
            vision,
            mtp,
            embedding,
            pp_tps,
            tg_tps,
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
        let (pp_tps, tg_tps) = (
            measurement.and_then(|mm| mm.pp_tps),
            measurement.and_then(|mm| mm.tg_tps),
        );
        // A "cache" entry no scanned file claimed means Load fetches from
        // HuggingFace first — a 20GB-class pull, and a *re*-download whenever
        // the repo moved past what was cached. Say so before the click, and
        // point at a local file carrying the same model when one exists.
        let is_hf_fetch = rm.source.as_deref() == Some("cache");
        let twin = is_hf_fetch
            .then(|| local_equivalent(models, &rm.id))
            .flatten();
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
            (None, None) if is_hf_fetch => (
                AdviceLevel::Warn,
                "no file on disk matches this entry — Load downloads it from HuggingFace \
                 first (20GB-class models take many minutes)"
                    .into(),
            ),
            (None, None) => (
                AdviceLevel::Unknown,
                "offered by the router (file location unknown) — Load to measure".into(),
            ),
        };
        let advice = match twin {
            Some(t) => format!(
                "{advice}; '{}' looks like the same model already on disk — prefer that row to avoid the download",
                t.display_name()
            ),
            None => advice,
        };
        rows.push(Row {
            display: rm.id.clone(),
            in_opencode: opencode_ids.contains(&rm.id),
            router_id: Some(rm.id.clone()),
            path: None,
            archivable: false,
            quant_header_disagrees: None,
            source: rm.source.clone().unwrap_or_else(|| "router".into()),
            size_bytes: 0,
            quant: String::new(),
            train_ctx: None,
            measured_ctx,
            failure,
            server_status: Some(rm.status.clone()),
            advice_level,
            advice,
            vision: false,
            mtp: false,
            embedding: false,
            pp_tps,
            tg_tps,
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
            mmproj: None,
        }
    }

    #[test]
    fn feature_flags_flow_into_rows() {
        let mut vision = file("gemma-it.gguf", Source::Shelf, 10, true);
        vision.mmproj = Some(PathBuf::from("/m/mmproj-F16.gguf"));
        let mut mtp = file("qwen-mtp.gguf", Source::Shelf, 10, true);
        mtp.meta.as_mut().unwrap().has_mtp = true;
        let mut embed = file("bge-small.gguf", Source::Shelf, 1, true);
        embed.meta.as_mut().unwrap().architecture = Some("nomic-bert".into());

        let rows = assemble(&[vision, mtp, embed], &[], &Measurements::new(), &[], HW);
        assert!(rows[0].vision && !rows[0].mtp && !rows[0].embedding);
        assert!(rows[1].mtp && !rows[1].vision);
        assert!(rows[2].embedding);
        assert!(
            rows[2].advice.contains("embedding model"),
            "embed advice replaces chat-centric advice: {}",
            rows[2].advice
        );
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
        let (lvl, msg) = advice_for(Some(&GgufMeta::default()), 120 * 1024 * 1024 * 1024, None, None, None, HW);
        assert_eq!(lvl, AdviceLevel::Bad);
        assert!(msg.contains("too large"), "{msg}");
        // Bigger than VRAM → CPU spill warning.
        let (lvl, msg) = advice_for(Some(&GgufMeta::default()), 30 * 1024 * 1024 * 1024, None, None, None, HW);
        assert_eq!(lvl, AdviceLevel::Warn);
        assert!(msg.contains("partly on CPU"), "{msg}");
        // Measured small ctx.
        let (lvl, msg) = advice_for(Some(&GgufMeta::default()), 10 << 30, Some(8192), None, None, HW);
        assert_eq!(lvl, AdviceLevel::Warn);
        assert!(msg.contains("small for agentic"), "{msg}");
        // Measured healthy.
        let (lvl, _) = advice_for(Some(&GgufMeta::default()), 10 << 30, Some(80_000), None, None, HW);
        assert_eq!(lvl, AdviceLevel::Good);
        // Failure dominates.
        let (lvl, msg) = advice_for(
            Some(&GgufMeta::default()),
            10 << 30,
            None,
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
                tool_call: Some(true),
                error: None,
                args_fp: None,
                env_fp: None,
                ..Default::default()
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
    fn same_named_files_get_the_preset_suffix_not_a_shared_identity() {
        // Two files with identical stems (user's manual download + archived
        // copy). The preset generator aliases them base and base-2; rows
        // must map one file to each — never both onto base with a phantom
        // "-2" row left over.
        let models = vec![
            file("Qwen3.8-27B-UD-Q4_K_XL.gguf", Source::Shelf, 18, true),
            {
                let mut f2 = file("Qwen3.8-27B-UD-Q4_K_XL.gguf", Source::Shelf, 18, true);
                f2.path = std::path::PathBuf::from("/other/Qwen3.8-27B-UD-Q4_K_XL.gguf");
                f2
            },
        ];
        let routers = vec![
            rm("qwen3.8-27b-ud-q4_k_xl", "loaded", "preset"),
            rm("qwen3.8-27b-ud-q4_k_xl-2", "unloaded", "preset"),
        ];
        let rows = assemble(&models, &routers, &Measurements::new(), &[], HW);
        assert_eq!(rows.len(), 2, "no phantom router-only row");
        let ids: Vec<_> = rows.iter().filter_map(|r| r.router_id.clone()).collect();
        assert!(ids.contains(&"qwen3.8-27b-ud-q4_k_xl".to_string()));
        assert!(ids.contains(&"qwen3.8-27b-ud-q4_k_xl-2".to_string()));
        // Statuses map per file, not shared.
        let loaded: Vec<_> = rows
            .iter()
            .filter(|r| r.server_status.as_deref() == Some("loaded"))
            .collect();
        assert_eq!(loaded.len(), 1);
        // Both carry the duplicate note.
        assert!(rows.iter().all(|r| r.advice.contains("stored twice")), "{:?}",
            rows.iter().map(|r| &r.advice).collect::<Vec<_>>());
    }

    #[test]
    fn filename_quant_outranks_a_lying_header_stamp() {
        // Real case: unsloth UD-Q5_K_XL stamped general.file_type=Q4_K_S
        // while its tensors are predominantly q5_K/q6_K.
        let mut m = file("Qwen3.8-27B-UD-Q5_K_XL.gguf", Source::Shelf, 20, true);
        m.meta.as_mut().unwrap().quantization = Some("Q4_K_S".into());
        let rows = assemble(&[m], &[], &Measurements::new(), &[], HW);
        assert_eq!(rows[0].quant, "Q5_K_XL");
        assert_eq!(rows[0].quant_header_disagrees.as_deref(), Some("Q4_K_S"));

        // Agreement (or no filename token) → header value, no note.
        let mut m2 = file("model-Q4_K_M.gguf", Source::Shelf, 10, true);
        m2.meta.as_mut().unwrap().quantization = Some("Q4_K_M".into());
        let rows = assemble(&[m2], &[], &Measurements::new(), &[], HW);
        assert_eq!(rows[0].quant, "Q4_K_M");
        assert!(rows[0].quant_header_disagrees.is_none());
    }

    #[test]
    fn router_only_models_still_get_rows() {
        let routers = vec![rm("mystery/model:Q4", "unloaded", "cache")];
        let rows = assemble(&[], &routers, &Measurements::new(), &[], HW);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].source, "cache");
        // Cache entries with no file behind them download on Load — the row
        // must say so, and grade as a caveat rather than a blank unknown.
        assert_eq!(rows[0].advice_level, AdviceLevel::Warn);
        assert!(rows[0].advice.contains("downloads it from HuggingFace"), "{}", rows[0].advice);
    }

    #[test]
    fn cache_row_points_at_local_twin() {
        // The user's real incident: the shelf holds the exact quant the
        // router's HF cache entry names, but the entry (stale revision)
        // would re-download 20GB. The row must steer to the shelf copy.
        let models = vec![file("Qwen3.8-27B-UD-Q5_K_XL.gguf", Source::Shelf, 20, true)];
        let routers = vec![rm("unsloth/Qwen3.8-27B-GGUF:Q5_K_XL", "unloaded", "cache")];
        let rows = assemble(&models, &routers, &Measurements::new(), &[], HW);
        let cache_row = rows
            .iter()
            .find(|r| r.router_id.as_deref() == Some("unsloth/Qwen3.8-27B-GGUF:Q5_K_XL"))
            .expect("cache entry keeps its own row");
        assert_eq!(cache_row.advice_level, AdviceLevel::Warn);
        assert!(cache_row.advice.contains("downloads it from HuggingFace"), "{}", cache_row.advice);
        assert!(
            cache_row.advice.contains("'Qwen3.8-27B-UD-Q5_K_XL'"),
            "must name the on-disk twin: {}",
            cache_row.advice
        );
    }

    #[test]
    fn twin_match_requires_the_same_quant() {
        // A Q4 on the shelf is NOT a stand-in for the cache entry's Q5.
        let models = vec![file("Qwen3.8-27B-UD-Q4_K_XL.gguf", Source::Shelf, 18, true)];
        let routers = vec![rm("unsloth/Qwen3.8-27B-GGUF:Q5_K_XL", "unloaded", "cache")];
        let rows = assemble(&models, &routers, &Measurements::new(), &[], HW);
        let cache_row = rows
            .iter()
            .find(|r| r.router_id.as_deref() == Some("unsloth/Qwen3.8-27B-GGUF:Q5_K_XL"))
            .unwrap();
        assert!(cache_row.advice.contains("downloads it from HuggingFace"), "{}", cache_row.advice);
        assert!(
            !cache_row.advice.contains("same model already on disk"),
            "different quant must not be offered as a twin: {}",
            cache_row.advice
        );
    }
}
