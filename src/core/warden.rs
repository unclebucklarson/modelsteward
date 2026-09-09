//! Reading **modelwarden's** published inventory — the family's storage
//! truth — so this app stops re-deriving what a sibling already knows.
//!
//! The three tools agree on a content-addressed identity
//! (`sha256:<hex>`), and warden and modellab already key their published
//! files by it. This app keyed models only by its own alias strings,
//! which is why the three saw 47, 25 and 22 models with no way to join
//! them (architecture review, 2026-09-08).
//!
//! **A file contract, not a connection.** We read warden's published
//! `inventory.json` and never write it, exactly as modellab does. Warden
//! not being installed is normal and silent; a *damaged* inventory is
//! reported rather than read as empty, the same rule as everywhere else
//! in this codebase.

use crate::core::safefs::{self, Loaded};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Schema we know how to read. Warden freezes its schema and bumps this
/// when the shape changes; reading a future version as if it were v1
/// would silently mis-join models, so a mismatch refuses instead.
pub const SUPPORTED_SCHEMA: u32 = 1;

#[derive(Debug, Deserialize, Default)]
pub struct Inventory {
    #[serde(default)]
    pub schema_version: u32,
    #[serde(default)]
    pub roots: Vec<Root>,
    #[serde(default)]
    pub models: BTreeMap<String, Entry>,
}

#[derive(Debug, Deserialize)]
pub struct Root {
    pub id: String,
    pub path: PathBuf,
    /// "shelf" | "ollama" | "hf_hub" | "removable" — warden's own
    /// classification of the store.
    #[serde(default)]
    pub kind: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Entry {
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub locations: Vec<Location>,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Deserialize)]
pub struct Location {
    pub root_id: String,
    pub rel_path: PathBuf,
    #[serde(default)]
    pub accessible: bool,
    #[serde(default)]
    pub dev: Option<u64>,
    #[serde(default)]
    pub ino: Option<u64>,
}

/// Where warden publishes, honouring `XDG_STATE_HOME` the same way our
/// own state dir does.
pub fn inventory_path() -> PathBuf {
    crate::core::settings::xdg_base("XDG_STATE_HOME", ".local/state")
        .join("modelwarden")
        .join("inventory.json")
}

/// Load and validate. `Missing` means warden isn't installed or hasn't
/// run — a normal state this app must work without.
pub fn load(path: &Path) -> Loaded<Inventory> {
    match safefs::read_json::<Inventory>(path) {
        Loaded::Ok(inv) if inv.schema_version != SUPPORTED_SCHEMA => Loaded::Damaged(format!(
            "inventory.json is schema v{} but this build reads v{SUPPORTED_SCHEMA} — \
             upgrade modelsteward, or it would mis-join models",
            inv.schema_version
        )),
        other => other,
    }
}

/// A content identity is only usable as a join key once warden has
/// actually hashed the bytes. Its other key forms — `pending:<dev>:<ino>:<size>`
/// while the hash worker is behind, `unknown:<root>:<rel>` when the bytes
/// are unreachable — are placeholders that change, so they are never
/// recorded as an identity.
pub fn is_content_id(key: &str) -> bool {
    key.starts_with("sha256:") && key.len() > "sha256:".len()
}

impl Inventory {
    /// The identity of the file at `path`, if warden knows it.
    ///
    /// Matched on `(dev, ino)` first and the resolved absolute path
    /// second. Inode is the stronger credential: this app's own library
    /// is already inode-deduped, and it survives a model reached through
    /// a different mount point, a symlink, or a hard link — all normal
    /// for the Ollama blob store and the HF cache.
    pub fn content_id_for(&self, path: &Path) -> Option<&str> {
        let ids = std::fs::metadata(path).ok().map(|m| {
            use std::os::unix::fs::MetadataExt;
            (m.dev(), m.ino())
        });
        let mut by_path: Option<&str> = None;
        for (key, entry) in &self.models {
            if !is_content_id(key) {
                continue;
            }
            for loc in &entry.locations {
                if let Some((dev, ino)) = ids
                    && loc.dev == Some(dev)
                    && loc.ino == Some(ino)
                {
                    return Some(key);
                }
                if by_path.is_none() && self.absolute(loc).as_deref() == Some(path) {
                    by_path = Some(key);
                }
            }
        }
        by_path
    }

    /// Resolve a location against its root. Returns `None` for a
    /// location whose root warden no longer lists.
    pub fn absolute(&self, loc: &Location) -> Option<PathBuf> {
        let root = self.roots.iter().find(|r| r.id == loc.root_id)?;
        Some(root.path.join(&loc.rel_path))
    }
}

/// Where warden says models live, filtered to what is usable right now.
///
/// Warden owns "what exists and where"; keeping our own parallel list of
/// stores is exactly the duplication the family realignment is removing.
/// Asking it for roots means a shelf the user added in warden, or a
/// backup drive they just plugged in, becomes servable here without
/// being configured twice.
///
/// Offline roots are skipped rather than reported: an unplugged drive is
/// a normal state, and handing a missing directory to the scanner would
/// just walk nothing. `hf_hub` is deliberately not returned — this app
/// already locates the hub cache itself, and the router serves those
/// natively.
#[derive(Debug, Default, PartialEq)]
pub struct Roots {
    /// Directories to walk for GGUFs: warden's shelves, plus removable
    /// roots that are currently mounted.
    pub shelves: Vec<PathBuf>,
    /// Ollama blob stores.
    pub ollama: Vec<PathBuf>,
}

pub fn servable_roots(inv: &Inventory) -> Roots {
    let mut out = Roots::default();
    for r in &inv.roots {
        // `exists` is the mount check: warden lists removable roots
        // whether or not the drive is plugged in.
        if !r.path.is_dir() {
            continue;
        }
        match r.kind.as_deref() {
            Some("ollama") => out.ollama.push(r.path.clone()),
            Some("shelf") | Some("removable") => out.shelves.push(r.path.clone()),
            // hf_hub: found by us already. Anything unknown is left
            // alone rather than guessed at — a future root kind must not
            // silently become a directory we walk.
            _ => {}
        }
    }
    out.shelves.sort();
    out.shelves.dedup();
    out.ollama.sort();
    out.ollama.dedup();
    out
}

/// Map this app's aliases to warden identities, for every model warden
/// knows about. Pure over its inputs so the caller decides where the
/// aliases and the inventory come from.
///
/// Aliases warden does not recognise are simply absent — a model can be
/// servable here without warden having catalogued it, and that is not an
/// error. The router's own HF-cache downloads land in that category
/// today: we learn them from its HTTP API, which reports no path.
pub fn identities<'a>(
    aliased: impl IntoIterator<Item = (&'a str, &'a Path)>,
    inv: &Inventory,
) -> BTreeMap<String, String> {
    aliased
        .into_iter()
        .filter_map(|(alias, path)| {
            inv.content_id_for(path).map(|id| (alias.to_string(), id.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped from the real file on this machine (2026-09-08): one model
    /// present in two places, one warden hasn't hashed yet, real root
    /// kinds.
    fn fixture(extra_dev_ino: Option<(u64, u64, &str)>) -> String {
        let (dev, ino, rel) = extra_dev_ino.unwrap_or((0, 0, "none.gguf"));
        format!(
            r#"{{
  "schema_version": 1,
  "generated_unix": 1788900000,
  "roots": [
    {{"id":"hf-hub-79a0b45a","kind":"hf_hub","label":null,"path":"/home/buck/.cache/huggingface/hub"}},
    {{"id":"shelf-0079bf06","kind":"shelf","label":null,"path":"/home/buck/models"}},
    {{"id":"ext-53b9be4e","kind":"removable","label":null,"path":"/run/media/buck/disk/model_backup"}}
  ],
  "models": {{
    "sha256:00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71": {{
      "display_name": "unsloth/gemma-4-31B-it-qat-GGUF",
      "size": 17287670048,
      "locations": [
        {{"root_id":"hf-hub-79a0b45a","rel_path":"models--unsloth--gemma-4/snap/g.gguf","accessible":true,"dev":66306,"ino":45351456}},
        {{"root_id":"ext-53b9be4e","rel_path":"gemma-4-31B.gguf","accessible":false,"dev":null,"ino":null}}
      ]
    }},
    "pending:66306:99999:123": {{
      "display_name": "not hashed yet",
      "size": 123,
      "locations": [{{"root_id":"shelf-0079bf06","rel_path":"unhashed.gguf","accessible":true,"dev":{dev},"ino":{ino}}}]
    }},
    "sha256:aa11bb22cc33dd44ee55ff6677889900aabbccddeeff00112233445566778899": {{
      "display_name": "local/on-disk",
      "size": 4096,
      "locations": [{{"root_id":"shelf-0079bf06","rel_path":"{rel}","accessible":true,"dev":{dev},"ino":{ino}}}]
    }}
  }}
}}"#
        )
    }

    fn parse(s: &str) -> Inventory {
        serde_json::from_str(s).expect("fixture parses")
    }

    #[test]
    fn resolves_identity_by_absolute_path() {
        let inv = parse(&fixture(None));
        let p = Path::new("/home/buck/.cache/huggingface/hub/models--unsloth--gemma-4/snap/g.gguf");
        assert_eq!(
            inv.content_id_for(p),
            Some("sha256:00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71")
        );
    }

    /// The stronger join: a real file reached by a path warden never
    /// recorded still resolves, because the inode matches. This is the
    /// Ollama-blob / HF-symlink case.
    #[test]
    fn resolves_identity_by_inode_when_the_path_is_unknown() {
        let dir = std::env::temp_dir().join(format!("ms-warden-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let real = dir.join("actually-here.gguf");
        std::fs::write(&real, b"x").unwrap();
        let md = std::fs::metadata(&real).unwrap();
        use std::os::unix::fs::MetadataExt;
        // Warden recorded this inode under a DIFFERENT path.
        let inv = parse(&fixture(Some((md.dev(), md.ino(), "a-different-name.gguf"))));
        assert_eq!(
            inv.content_id_for(&real),
            Some("sha256:aa11bb22cc33dd44ee55ff6677889900aabbccddeeff00112233445566778899"),
            "inode must win where the path is not the one warden saw"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `pending:` and `unknown:` keys are placeholders that change as the
    /// hash worker catches up. Recording one as an identity would write a
    /// join key that silently stops matching.
    #[test]
    fn placeholder_keys_are_never_offered_as_an_identity() {
        assert!(is_content_id(
            "sha256:00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71"
        ));
        assert!(!is_content_id("pending:66306:99999:123"));
        assert!(!is_content_id("unknown:shelf-0079bf06:foo.gguf"));
        assert!(!is_content_id("sha256:"), "a bare prefix is not an identity");

        let inv = parse(&fixture(None));
        let p = Path::new("/home/buck/models/unhashed.gguf");
        assert_eq!(inv.content_id_for(p), None, "the pending entry must not match");
    }

    #[test]
    fn a_model_warden_does_not_know_has_no_identity() {
        let inv = parse(&fixture(None));
        assert_eq!(inv.content_id_for(Path::new("/home/buck/models/stranger.gguf")), None);
    }

    /// One identity, many locations — the same bytes in the HF cache and
    /// on a backup drive. Both paths must resolve to the one identity.
    #[test]
    fn one_identity_spans_its_locations() {
        let inv = parse(&fixture(None));
        let want = Some("sha256:00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71");
        assert_eq!(
            inv.content_id_for(Path::new(
                "/home/buck/.cache/huggingface/hub/models--unsloth--gemma-4/snap/g.gguf"
            )),
            want
        );
        assert_eq!(
            inv.content_id_for(Path::new("/run/media/buck/disk/model_backup/gemma-4-31B.gguf")),
            want,
            "the offline backup copy is the same content"
        );
    }

    #[test]
    fn a_location_whose_root_is_gone_resolves_to_nothing() {
        let inv = parse(&fixture(None));
        let orphan = Location {
            root_id: "root-that-warden-dropped".into(),
            rel_path: "x.gguf".into(),
            accessible: true,
            dev: None,
            ino: None,
        };
        assert_eq!(inv.absolute(&orphan), None);
    }

    // ── the file contract itself ──────────────────────────────────────

    #[test]
    fn warden_not_installed_is_silent_not_an_error() {
        let p = std::env::temp_dir().join("ms-no-such-inventory-xyz.json");
        std::fs::remove_file(&p).ok();
        assert!(matches!(load(&p), Loaded::Missing));
    }

    /// The C1 rule, applied to a sibling's file: a half-written
    /// inventory must be reported, never read as "warden knows nothing"
    /// — which would silently drop every identity we had joined.
    #[test]
    fn a_damaged_inventory_is_reported_not_read_as_empty() {
        let p = std::env::temp_dir().join(format!("ms-damaged-inv-{}.json", std::process::id()));
        std::fs::write(&p, "{ \"schema_version\": 1, \"models\": {").unwrap();
        match load(&p) {
            Loaded::Damaged(why) => assert!(!why.is_empty()),
            other => panic!("expected Damaged, got {other:?}"),
        }
        std::fs::remove_file(&p).ok();
    }

    /// A file contract needs a version gate at BOTH ends. If warden
    /// moves to v2 and we keep reading it as v1, every join is quietly
    /// wrong — worse than not joining at all.
    #[test]
    fn a_future_schema_is_refused_rather_than_misread() {
        let p = std::env::temp_dir().join(format!("ms-future-inv-{}.json", std::process::id()));
        std::fs::write(&p, r#"{"schema_version":2,"roots":[],"models":{}}"#).unwrap();
        match load(&p) {
            Loaded::Damaged(why) => {
                assert!(why.contains("v2"), "name the version we found: {why}");
                assert!(why.contains("v1"), "and the one we read: {why}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn inventory_path_follows_xdg_state_home() {
        // Documented behaviour rather than a live env mutation: tests run
        // in one process and XDG is global. The shared helper is covered
        // by settings::tests; this pins the tail we add to it.
        let p = inventory_path();
        assert!(
            p.ends_with("modelwarden/inventory.json"),
            "must read warden's published path, got {}",
            p.display()
        );
        // The bug this test originally missed: xdg_dir() appends OUR app
        // name, so the path became .../modelsteward/modelwarden/... and
        // warden read as "not installed". ends_with() was true either
        // way; only the absence of our own name distinguishes them.
        assert!(
            !p.to_string_lossy().contains("modelsteward"),
            "a sibling's file does not live under our app dir: {}",
            p.display()
        );
    }
    #[test]
    fn maps_aliases_to_identities_and_omits_what_warden_does_not_know() {
        let inv = parse(&fixture(None));
        let known = Path::new(
            "/home/buck/.cache/huggingface/hub/models--unsloth--gemma-4/snap/g.gguf",
        );
        let stranger = Path::new("/home/buck/models/never-catalogued.gguf");
        let map = identities(
            vec![("gemma-4-31b", known), ("mystery", stranger)],
            &inv,
        );
        assert_eq!(
            map.get("gemma-4-31b").map(String::as_str),
            Some("sha256:00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71")
        );
        assert!(
            !map.contains_key("mystery"),
            "an uncatalogued model is absent, not an error: {map:?}"
        );
    }

    /// Two aliases can legitimately point at one file (a preset entry and
    /// a hand-added duplicate); both carry the same identity.
    #[test]
    fn two_aliases_on_one_file_share_the_identity() {
        let inv = parse(&fixture(None));
        let p = Path::new(
            "/home/buck/.cache/huggingface/hub/models--unsloth--gemma-4/snap/g.gguf",
        );
        let map = identities(vec![("a", p), ("b", p)], &inv);
        assert_eq!(map.get("a"), map.get("b"));
        assert_eq!(map.len(), 2);
    }

    /// Warden's real root kinds, from the live file on this machine.
    fn roots_fixture(paths: &[(&str, &str, &str)]) -> Inventory {
        let roots: Vec<String> = paths
            .iter()
            .map(|(id, kind, path)| {
                format!(r#"{{"id":"{id}","kind":"{kind}","label":null,"path":"{path}"}}"#)
            })
            .collect();
        parse(&format!(
            r#"{{"schema_version":1,"roots":[{}],"models":{{}}}}"#,
            roots.join(",")
        ))
    }

    #[test]
    fn classifies_warden_roots_for_our_scanner() {
        let tmp = std::env::temp_dir().join(format!("ms-roots-{}", std::process::id()));
        let shelf = tmp.join("shelf");
        let olla = tmp.join("ollama");
        let drive = tmp.join("drive");
        let hub = tmp.join("hub");
        for d in [&shelf, &olla, &drive, &hub] {
            std::fs::create_dir_all(d).unwrap();
        }
        let inv = roots_fixture(&[
            ("a", "shelf", shelf.to_str().unwrap()),
            ("b", "ollama", olla.to_str().unwrap()),
            ("c", "removable", drive.to_str().unwrap()),
            ("d", "hf_hub", hub.to_str().unwrap()),
        ]);
        let r = servable_roots(&inv);
        assert!(r.shelves.contains(&shelf), "a shelf is ours to walk");
        assert!(
            r.shelves.contains(&drive),
            "a MOUNTED backup drive holds servable models: {r:?}"
        );
        assert_eq!(r.ollama, vec![olla]);
        assert!(
            !r.shelves.contains(&hub),
            "the hub cache is found by us and served natively by the router: {r:?}"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// An unplugged backup drive is a normal state, not an error, and
    /// must never be handed to the scanner as a directory.
    #[test]
    fn an_offline_root_is_skipped() {
        let inv = roots_fixture(&[(
            "gone",
            "removable",
            "/run/media/buck/definitely-not-mounted-xyz",
        )]);
        assert_eq!(servable_roots(&inv), Roots::default());
    }

    /// A root kind warden invents later must not silently become a
    /// directory we walk.
    #[test]
    fn an_unknown_root_kind_is_left_alone() {
        let tmp = std::env::temp_dir().join(format!("ms-unk-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let inv = roots_fixture(&[("x", "s3-bucket-of-the-future", tmp.to_str().unwrap())]);
        assert_eq!(servable_roots(&inv), Roots::default());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn no_inventory_means_no_extra_roots() {
        assert_eq!(servable_roots(&Inventory::default()), Roots::default());
    }

}
