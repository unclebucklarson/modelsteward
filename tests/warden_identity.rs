//! Integration: the family's shared content identity, end to end.
//!
//! Step 1 of the 2026-09-08 family realignment. Warden and modellab
//! already key models by a sha256 content identity; this app keyed them
//! only by its own alias strings, so the three tools saw the same fleet
//! three different ways with no way to join them.
//!
//! Drives the real modules against tempdirs: no router, no network, no
//! touching `~`.

use modelsteward::core::{safefs::Loaded, warden};
use std::path::Path;

/// Build an inventory in warden's real published shape, pointing at
/// files that actually exist in `dir` so inode matching is exercised
/// rather than mocked.
fn publish_inventory(dir: &Path, files: &[(&str, &str)]) -> std::path::PathBuf {
    let mut models = String::new();
    for (i, (name, id)) in files.iter().enumerate() {
        let p = dir.join(name);
        std::fs::write(&p, format!("gguf-bytes-{i}")).unwrap();
        if i > 0 {
            models.push(',');
        }
        models.push_str(&format!(
            r#""{id}": {{
                 "display_name": "test/{name}",
                 "size": 13,
                 "locations": [{{"root_id":"shelf","rel_path":"{name}","accessible":true,
                                 "dev":null,"ino":null}}]
               }}"#
        ));
    }
    let inv = format!(
        r#"{{"schema_version":1,"generated_unix":1788900000,
            "roots":[{{"id":"shelf","kind":"shelf","label":null,"path":"{}"}}],
            "models":{{{models}}}}}"#,
        dir.display()
    );
    let path = dir.join("inventory.json");
    std::fs::write(&path, inv).unwrap();
    path
}

const GEMMA: &str = "sha256:00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71";
const QWEN: &str = "sha256:aa11bb22cc33dd44ee55ff6677889900aabbccddeeff00112233445566778899";

#[test]
fn a_published_inventory_becomes_alias_to_identity() {
    let tmp = tempfile::tempdir().unwrap();
    let path = publish_inventory(tmp.path(), &[("gemma.gguf", GEMMA), ("qwen.gguf", QWEN)]);

    let inv = match warden::load(&path) {
        Loaded::Ok(i) => i,
        other => panic!("expected a readable inventory, got {other:?}"),
    };
    assert_eq!(inv.models.len(), 2);

    let (g, q, m) = (
        tmp.path().join("gemma.gguf"),
        tmp.path().join("qwen.gguf"),
        tmp.path().join("mystery.gguf"),
    );
    let map = warden::identities(
        [
            ("gemma-4-31b", g.as_path()),
            ("qwen3.8-27b", q.as_path()),
            ("not-catalogued", m.as_path()),
        ],
        &inv,
    );

    assert_eq!(map.get("gemma-4-31b").map(String::as_str), Some(GEMMA));
    assert_eq!(map.get("qwen3.8-27b").map(String::as_str), Some(QWEN));
    assert!(!map.contains_key("not-catalogued"));
}

/// The identity must survive the store it is written into, because the
/// whole point is that a LATER reader (this app tomorrow, or a human
/// joining three files) can line the numbers up.
#[test]
fn identities_persist_through_the_measurement_store() {
    use modelsteward::core::router::{self, Measurement, Measurements};
    let tmp = tempfile::tempdir().unwrap();
    let path = publish_inventory(tmp.path(), &[("gemma.gguf", GEMMA)]);
    let inv = match warden::load(&path) {
        Loaded::Ok(i) => i,
        other => panic!("{other:?}"),
    };
    let g = tmp.path().join("gemma.gguf");
    let ids = warden::identities([("gemma-4-31b", g.as_path())], &inv);

    let state = tmp.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let mut m = Measurements::new();
    m.insert(
        "gemma-4-31b".into(),
        Measurement {
            n_ctx: Some(262_144),
            tool_call: Some(true),
            content_id: ids.get("gemma-4-31b").cloned(),
            ..Default::default()
        },
    );
    router::write_measurements(&state, &m).unwrap();

    let back = router::read_measurements(&state);
    assert_eq!(
        back["gemma-4-31b"].content_id.as_deref(),
        Some(GEMMA),
        "the join key must survive a write/read cycle"
    );
    assert_eq!(back["gemma-4-31b"].n_ctx, Some(262_144));
}

/// Warden absent is the state this app shipped in for its whole life.
/// It must stay completely normal: no identities, no warning, no
/// failure.
#[test]
fn warden_absent_changes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("inventory.json");
    assert!(matches!(warden::load(&missing), Loaded::Missing));
}

/// A half-written sibling file must be reported, never read as "warden
/// knows nothing" — that would silently drop identities we had already
/// recorded and look like the models had changed.
#[test]
fn a_damaged_inventory_is_loud() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("inventory.json");
    std::fs::write(&p, r#"{"schema_version":1,"roots":[],"models":{"a":"#).unwrap();
    match warden::load(&p) {
        Loaded::Damaged(why) => assert!(!why.is_empty(), "must carry a reason"),
        other => panic!("expected Damaged, got {other:?}"),
    }
}

/// Both ends of a file contract need a version gate. If warden moves to
/// v2 and we keep reading it as v1, every join is quietly wrong — which
/// is worse than not joining at all.
#[test]
fn a_newer_schema_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("inventory.json");
    std::fs::write(&p, r#"{"schema_version":99,"roots":[],"models":{}}"#).unwrap();
    match warden::load(&p) {
        Loaded::Damaged(why) => assert!(why.contains("v99"), "name what it found: {why}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}
