//! The diagnosis brain: map a failure to a cause, a cause to plain
//! language and concrete remedies. Shared by the per-model "Why?" panel
//! and the Build Advisor — same rules, two faces. Never says "see logs"
//! without also saying what the log said.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Cause {
    /// Model format newer than the installed llama.cpp understands.
    NeedsNewerBuild,
    /// Blob is partial / a multimodal component / not standalone-loadable.
    BadBlob,
    /// Ran out of memory while loading.
    OutOfMemory,
    /// File exists but the router doesn't currently offer this variant.
    NotOffered,
    /// Nothing wrong — just never measured.
    Unmeasured,
    /// We genuinely don't recognize the failure.
    Unknown,
}

/// A concrete next step the UI can render as a button.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Remedy {
    OpenBuildAdvisor,
    ArchiveToShelf,
    UnloadOthers,
    LoadAndMeasure,
    ShowLog,
}

#[derive(Debug, Clone, Serialize)]
pub struct Diagnosis {
    pub cause: Cause,
    /// One or two sentences a novice can act on. No jargon, no flag names.
    pub explanation: String,
    /// The single most relevant log/error line, when we have one.
    pub evidence: Option<String>,
    pub remedies: Vec<Remedy>,
}

/// Classify a stored load-failure string.
pub fn classify(error: &str) -> Cause {
    let lower = error.to_lowercase();
    if lower.contains("rope.dimension_sections")
        || lower.contains("hyperparameters")
        || lower.contains("unknown model architecture")
        || lower.contains("unknown architecture")
        || lower.contains("unsupported model architecture")
    {
        Cause::NeedsNewerBuild
    } else if lower.contains("wrong number of tensors")
        || lower.contains("bad magic")
        || lower.contains("truncated")
    {
        Cause::BadBlob
    } else if lower.contains("out of memory")
        || lower.contains("failed to allocate")
        || lower.contains("cuda_out_of_memory")
    {
        Cause::OutOfMemory
    } else {
        Cause::Unknown
    }
}

/// Full diagnosis for a model row. `error` is the stored measurement
/// failure (with mined log line, when we got one); `not_offered` is the
/// on-disk-but-router-doesn't-list state; `archivable` gates the
/// archive remedy.
pub fn diagnose(
    error: Option<&str>,
    not_offered: bool,
    archivable: bool,
    build_is_current: Option<bool>,
) -> Diagnosis {
    if let Some(err) = error {
        let cause = classify(err);
        // The mined log line rides in after "— " in stored errors; show
        // the tail as evidence when present, else the whole string.
        let evidence = Some(
            err.rsplit_once("— ")
                .map(|(_, tail)| tail.trim().to_string())
                .unwrap_or_else(|| err.to_string()),
        );
        let (explanation, remedies) = match cause {
            Cause::NeedsNewerBuild => match build_is_current {
                // Newest build still rejects it → the file itself is the
                // outlier (typically an Ollama-specific conversion).
                Some(true) => (
                    "Your llama.cpp is already the newest build, and it still can't \
                     read this file's format — so this looks like a conversion made \
                     specifically for another tool (usually Ollama). It will keep \
                     working through Ollama itself; for llama.cpp, download a \
                     llama.cpp-native GGUF of the same model instead."
                        .to_string(),
                    vec![Remedy::ShowLog],
                ),
                _ => (
                    "This model uses a newer format than your installed llama.cpp \
                     understands. The file is fine — the program reading it is out of \
                     date. Updating and rebuilding llama.cpp will most likely unlock it."
                        .to_string(),
                    vec![Remedy::OpenBuildAdvisor, Remedy::ShowLog],
                ),
            },
            Cause::BadBlob => (
                "This file isn't a complete standalone model — it's either a \
                 partial download or a component (like a vision add-on) that can't \
                 be loaded by itself. Re-downloading the full model is the usual fix."
                    .to_string(),
                vec![Remedy::ShowLog],
            ),
            Cause::OutOfMemory => (
                "There wasn't enough memory to load this model. Free some VRAM \
                 (unload other models, close GPU-hungry apps) and try again, or use \
                 a smaller quantization."
                    .to_string(),
                vec![Remedy::UnloadOthers, Remedy::LoadAndMeasure, Remedy::ShowLog],
            ),
            _ => (
                "The load failed for a reason these rules don't recognize yet. The \
                 exact error is below; the router log has the full story."
                    .to_string(),
                vec![Remedy::ShowLog],
            ),
        };
        return Diagnosis {
            cause,
            explanation,
            evidence,
            remedies,
        };
    }
    if not_offered {
        let mut remedies = vec![];
        if archivable {
            remedies.push(Remedy::ArchiveToShelf);
        }
        return Diagnosis {
            cause: Cause::NotOffered,
            explanation: "The file is on disk, but the server doesn't currently \
                offer this exact variant (its download index points at a different \
                version of the same model). Archiving it to your models folder makes \
                it servable under your own control."
                .to_string(),
            evidence: None,
            remedies,
        };
    }
    Diagnosis {
        cause: Cause::Unmeasured,
        explanation: "Nothing is wrong — this model just hasn't been measured yet. \
            Load it once and the real numbers (context, tool support) get recorded \
            and written to OpenCode automatically."
            .to_string(),
        evidence: None,
        remedies: vec![Remedy::LoadAndMeasure],
    }
}

/// One-line version for table cells (kept for rows::failure_hint).
pub fn short_hint(error: &str) -> String {
    match classify(error) {
        Cause::NeedsNewerBuild => {
            "this model's format is newer than your llama.cpp build — updating/rebuilding llama.cpp will likely fix it".into()
        }
        Cause::BadBlob => {
            "the file looks like a partial download or a multimodal blob llama.cpp can't load standalone".into()
        }
        Cause::OutOfMemory => "not enough memory to load with current settings".into(),
        _ => "load failed — click Why? for the details".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_the_real_errors_from_this_machine() {
        assert_eq!(
            classify("key qwen35moe.rope.dimension_sections has wrong array length; expected 4, got 3"),
            Cause::NeedsNewerBuild
        );
        assert_eq!(
            classify("done_getting_tensors: wrong number of tensors; expected 2131, got 720"),
            Cause::BadBlob
        );
        assert_eq!(classify("ggml_backend_cuda: failed to allocate 2.3 GiB"), Cause::OutOfMemory);
        assert_eq!(classify("something exotic"), Cause::Unknown);
    }

    #[test]
    fn current_build_reframes_format_errors_as_ollama_specific() {
        let d = diagnose(
            Some("key qwen35.rope.dimension_sections has wrong array length; expected 4, got 3"),
            false,
            false,
            Some(true),
        );
        assert_eq!(d.cause, Cause::NeedsNewerBuild);
        assert!(d.explanation.contains("another tool"), "{}", d.explanation);
        assert!(
            !d.remedies.contains(&Remedy::OpenBuildAdvisor),
            "no point advising a rebuild that provably won't help"
        );
    }

    #[test]
    fn diagnosis_gives_novice_language_and_actions() {
        let d = diagnose(
            Some("did not load: failed(1) — error loading model hyperparameters: key x has wrong array length"),
            false,
            false,
            None,
        );
        assert_eq!(d.cause, Cause::NeedsNewerBuild);
        assert!(d.remedies.contains(&Remedy::OpenBuildAdvisor));
        assert!(d.evidence.unwrap().contains("wrong array length"), "evidence is the mined tail");
        assert!(!d.explanation.contains("cmake"), "no jargon in explanations");

        let d = diagnose(None, true, true, None);
        assert_eq!(d.cause, Cause::NotOffered);
        assert_eq!(d.remedies, vec![Remedy::ArchiveToShelf]);

        let d = diagnose(None, false, false, None);
        assert_eq!(d.cause, Cause::Unmeasured);
        assert_eq!(d.remedies, vec![Remedy::LoadAndMeasure]);
    }
}
