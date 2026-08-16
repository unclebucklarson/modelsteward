//! The model library: one deduplicated view of every GGUF this machine has,
//! wherever it lives — the user's shelf directories and Ollama's blob store.
//!
//! Servers answer what they're *serving*; the library answers what *could
//! be* served.
//!
//! Harvested from llm_forge (src/library.rs), extended for multiple Ollama
//! store locations (user-level `~/.ollama` vs the system service's
//! `/usr/share/ollama/.ollama` — the dev machine uses the latter).

use crate::core::gguf::{self, GgufMeta};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Source {
    /// Found under a configured scan directory.
    Shelf,
    /// A weights blob in Ollama's store; `name` is the `model:tag` Ollama
    /// knows it by. The blob is a raw GGUF and llama-server can load it
    /// directly — zero duplication.
    Ollama { name: String },
    /// A GGUF in the HuggingFace hub cache (`~/.cache/huggingface/hub`),
    /// downloaded by llama-server's `-hf`, unsloth studio, or the hf CLI.
    /// `repo` is "org/name"; llama-server's router serves these natively
    /// under ids like "org/name:QUANT", so they are NOT written into the
    /// preset — the router already offers them.
    HfHub { repo: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelFile {
    pub path: PathBuf,
    pub file_size: u64,
    pub source: Source,
    /// `None` when the header couldn't be read; the file is still listed so
    /// a broken download is visible rather than invisible.
    pub meta: Option<GgufMeta>,
}

impl ModelFile {
    /// The label a row leads with: Ollama's name, the HF repo + file, or
    /// the file stem, in that order of preference.
    pub fn display_name(&self) -> String {
        let stem = || {
            self.path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| self.path.display().to_string())
        };
        match &self.source {
            Source::Ollama { name } => name.clone(),
            Source::HfHub { repo } => format!("{repo} — {}", stem()),
            Source::Shelf => stem(),
        }
    }

    /// The id the running router knows this model by, when predictable:
    /// preset models go by their alias; HF-hub files by "repo:QUANT" (the
    /// router tags cache models with the quant embedded in the filename).
    pub fn router_cache_id(&self) -> Option<String> {
        let Source::HfHub { repo } = &self.source else {
            return None;
        };
        let stem = self.path.file_stem()?.to_string_lossy().to_uppercase();
        // Filename like Qwen3.8-27B-UD-Q5_K_XL → tag Q5_K_XL: take the
        // longest known quant token the stem ends with.
        for tag in [
            "Q8_0", "Q6_K", "Q5_K_XL", "Q5_K_M", "Q5_K_S", "Q4_K_XL", "Q4_K_M", "Q4_K_S",
            "Q4_0", "Q3_K_M", "Q2_K", "IQ4_XS", "IQ4_NL", "BF16", "F16", "IT",
        ] {
            if stem.ends_with(tag) {
                return Some(format!("{repo}:{tag}"));
            }
        }
        None
    }
}

/// Ollama store locations worth probing, most specific first: the
/// `OLLAMA_MODELS` env var, the per-user store, then the system service's
/// store. Only existing directories are returned.
pub fn default_ollama_stores() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(env_store) = std::env::var("OLLAMA_MODELS") {
        candidates.push(PathBuf::from(env_store));
    }
    if let Some(home) = std::env::home_dir() {
        candidates.push(home.join(".ollama/models"));
    }
    candidates.push(PathBuf::from("/usr/share/ollama/.ollama/models"));
    candidates.retain(|p| p.join("manifests").is_dir());
    candidates
}

/// The HuggingFace hub cache, when present.
pub fn default_hf_hub() -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HOME")
        .map(|h| PathBuf::from(h).join("hub"))
        .unwrap_or_else(|| {
            std::env::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".cache/huggingface/hub")
        });
    hub.is_dir().then_some(hub)
}

/// GGUFs in the HF hub cache: `hub/models--org--name/snapshots/<rev>/*.gguf`.
pub fn hf_hub_models(hub: &Path) -> Vec<ModelFile> {
    let mut out = Vec::new();
    let Ok(repos) = std::fs::read_dir(hub) else {
        return out;
    };
    for repo_dir in repos.flatten() {
        let dirname = repo_dir.file_name().to_string_lossy().into_owned();
        let Some(rest) = dirname.strip_prefix("models--") else {
            continue;
        };
        let repo = rest.replace("--", "/");
        let snapshots = repo_dir.path().join("snapshots");
        let Ok(revs) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        for rev in revs.flatten() {
            let Ok(files) = std::fs::read_dir(rev.path()) else {
                continue;
            };
            for f in files.flatten() {
                let path = f.path();
                // mmproj-*.gguf are vision projectors that ride along with a
                // main model — companions, not servable models themselves.
                let is_mmproj = path
                    .file_stem()
                    .is_some_and(|s| s.to_string_lossy().to_lowercase().starts_with("mmproj"));
                if !is_mmproj
                    && path
                        .extension()
                        .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
                {
                    // Snapshot entries are usually symlinks into blobs/;
                    // metadata() follows them for the real size.
                    let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    out.push(ModelFile {
                        meta: gguf::read_meta(&path).ok(),
                        path,
                        file_size,
                        source: Source::HfHub { repo: repo.clone() },
                    });
                }
            }
        }
    }
    out
}

/// Scan shelf directories, the Ollama stores, and (when given) the HF hub
/// cache. Never errors as a whole — an unreadable directory contributes
/// nothing rather than sinking the scan. Duplicate paths (a shelf dir
/// listed twice, two store candidates that resolve to the same blob)
/// collapse to one entry.
pub fn scan(
    scan_dirs: &[PathBuf],
    ollama_stores: &[PathBuf],
    hf_hub: Option<&Path>,
) -> Vec<ModelFile> {
    let mut out = Vec::new();
    for dir in scan_dirs {
        walk_gguf(dir, 0, &mut out);
    }
    for store in ollama_stores {
        out.extend(ollama_models(store));
    }
    if let Some(hub) = hf_hub {
        out.extend(hf_hub_models(hub));
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|m| {
        let key = m.path.canonicalize().unwrap_or_else(|_| m.path.clone());
        seen.insert(key)
    });
    // Stable order: shelf, then Ollama, then HF hub — alphabetical within.
    out.sort_by(|a, b| {
        let rank = |m: &ModelFile| match m.source {
            Source::Shelf => 0u8,
            Source::Ollama { .. } => 1,
            Source::HfHub { .. } => 2,
        };
        (rank(a), a.display_name().to_lowercase()).cmp(&(rank(b), b.display_name().to_lowercase()))
    });
    out
}

/// Depth-limited recursive walk collecting `*.gguf`. The shelf layout here
/// is `~/models/<ModelName>/<file>.gguf`, so a few levels is plenty.
fn walk_gguf(dir: &Path, depth: usize, out: &mut Vec<ModelFile>) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let path = e.path();
        if path.is_dir() {
            walk_gguf(&path, depth + 1, out);
        } else if path
            .extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
        {
            let file_size = e.metadata().map(|m| m.len()).unwrap_or(0);
            out.push(ModelFile {
                meta: gguf::read_meta(&path).ok(),
                path,
                file_size,
                source: Source::Shelf,
            });
        }
    }
}

/// Enumerate Ollama's store by walking its manifests: each manifest is JSON
/// whose layer with mediaType `…image.model` names the weights blob.
///
/// Manifest paths look like
/// `manifests/registry.ollama.ai/library/<name>/<tag>`; the name shown is
/// `<name>:<tag>` (with the namespace kept when it isn't `library`).
pub fn ollama_models(store: &Path) -> Vec<ModelFile> {
    let mut out = Vec::new();
    let manifests = store.join("manifests");
    let mut stack = vec![manifests.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Some(m) = ollama_model_from_manifest(store, &manifests, &path) {
                out.push(m);
            }
        }
    }
    out
}

fn ollama_model_from_manifest(
    store: &Path,
    manifests_root: &Path,
    manifest: &Path,
) -> Option<ModelFile> {
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(manifest).ok()?).ok()?;
    let digest = json.get("layers")?.as_array()?.iter().find_map(|l| {
        l.get("mediaType")?
            .as_str()?
            .ends_with("image.model")
            .then(|| l.get("digest")?.as_str().map(str::to_string))?
    })?;
    let blob = store.join("blobs").join(digest.replace(':', "-"));

    // manifests/<host>/<namespace>/<name>/<tag> → "name:tag" (or
    // "namespace/name:tag" for non-library namespaces).
    let rel = manifest.strip_prefix(manifests_root).ok()?;
    let parts: Vec<_> = rel.iter().map(|c| c.to_string_lossy()).collect();
    let name = match parts.as_slice() {
        [_host, ns, name, tag] if ns == "library" => format!("{name}:{tag}"),
        [_host, ns, name, tag] => format!("{ns}/{name}:{tag}"),
        _ => return None,
    };

    let file_size = std::fs::metadata(&blob).map(|m| m.len()).unwrap_or(0);
    Some(ModelFile {
        meta: gguf::read_meta(&blob).ok(),
        path: blob,
        file_size,
        source: Source::Ollama { name },
    })
}

/// Suggested serve alias for a model file: the file stem (or Ollama name),
/// lowercased, restricted to characters that survive shells, JSON keys, and
/// opencode model ids unquoted-ish. The user can always override.
pub fn alias_suggestion(m: &ModelFile) -> String {
    let raw = match &m.source {
        Source::Ollama { name } => name.clone(),
        _ => m.display_name(),
    };
    let mut s: String = raw
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    s.trim_matches('-').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gguf::tests::synthetic_gguf;

    fn shelf_with(files: &[(&str, &[u8])]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (rel, bytes) in files {
            let p = dir.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, bytes).unwrap();
        }
        dir
    }

    #[test]
    fn finds_ggufs_in_nested_shelf_layout() {
        let shelf = shelf_with(&[
            (
                "Qwen3.6-27B/Qwen3.6-27B-UD-Q5_K_XL.gguf",
                &synthetic_gguf("qwen3", 262_144, 17)[..],
            ),
            ("Tiny/tiny.GGUF", &synthetic_gguf("llama", 4096, 15)[..]),
            ("notes/readme.txt", b"not a model"),
        ]);
        let models = scan(&[shelf.path().to_path_buf()], &[], None);
        assert_eq!(models.len(), 2);
        assert!(models.iter().all(|m| matches!(m.source, Source::Shelf)));
        let qwen = models
            .iter()
            .find(|m| m.display_name().starts_with("Qwen3.6"))
            .unwrap();
        assert_eq!(qwen.meta.as_ref().unwrap().context_length, Some(262_144));
        assert_eq!(
            qwen.meta.as_ref().unwrap().quantization.as_deref(),
            Some("Q5_K_M")
        );
    }

    #[test]
    fn broken_gguf_is_listed_without_meta() {
        let shelf = shelf_with(&[("Broken/broken.gguf", b"corrupt")]);
        let models = scan(&[shelf.path().to_path_buf()], &[], None);
        assert_eq!(models.len(), 1);
        assert!(models[0].meta.is_none(), "visible, but honestly meta-less");
    }

    #[test]
    fn missing_scan_dir_contributes_nothing() {
        let models = scan(&[PathBuf::from("/nonexistent/nowhere")], &[], None);
        assert!(models.is_empty());
    }

    #[test]
    fn duplicate_scan_dirs_collapse_to_one_entry() {
        let shelf = shelf_with(&[(
            "Tiny/tiny.gguf",
            &synthetic_gguf("llama", 4096, 15)[..],
        )]);
        let dir = shelf.path().to_path_buf();
        let models = scan(&[dir.clone(), dir], &[], None);
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn reads_the_ollama_store_layout() {
        let store = tempfile::tempdir().unwrap();
        let blob_bytes = synthetic_gguf("gemma3", 131_072, 15);
        let digest = "sha256-abc123";
        std::fs::create_dir_all(store.path().join("blobs")).unwrap();
        std::fs::write(store.path().join("blobs").join(digest), &blob_bytes).unwrap();
        let mdir = store
            .path()
            .join("manifests/registry.ollama.ai/library/gemma4");
        std::fs::create_dir_all(&mdir).unwrap();
        std::fs::write(
            mdir.join("latest"),
            r#"{"layers":[
                {"mediaType":"application/vnd.ollama.image.template","digest":"sha256-zzz","size":10},
                {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:abc123","size":100}
            ]}"#,
        )
        .unwrap();

        let models = ollama_models(store.path());
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].display_name(), "gemma4:latest");
        assert_eq!(
            models[0].meta.as_ref().unwrap().context_length,
            Some(131_072)
        );
        assert!(models[0].path.ends_with("blobs/sha256-abc123"));
    }

    #[test]
    fn alias_suggestions_are_shell_and_config_safe() {
        let shelf = shelf_with(&[(
            "Qwen3.6-27B/Qwen3.6-27B-UD-Q5_K_XL.gguf",
            &synthetic_gguf("qwen3", 1, 1)[..],
        )]);
        let models = scan(&[shelf.path().to_path_buf()], &[], None);
        assert_eq!(alias_suggestion(&models[0]), "qwen3.6-27b-ud-q5_k_xl");

        let ollama = ModelFile {
            path: PathBuf::from("/b"),
            file_size: 0,
            source: Source::Ollama {
                name: "rafw007/qwen36 coder:q4_K_M".into(),
            },
            meta: None,
        };
        assert_eq!(alias_suggestion(&ollama), "rafw007-qwen36-coder-q4_k_m");
    }
}
