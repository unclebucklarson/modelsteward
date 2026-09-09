//! Integration tests for the multi-connector sync flow.
//!
//! CLAUDE.md has promised these since 2026-08-29 ("integration tests for
//! flows that cross modules … using tempdirs and synthetic fixtures —
//! never the user's real state dirs") and the 2026-08-31 review found
//! `tests/` held fixtures and no tests, while the flow itself was
//! implemented twice and had already diverged.
//!
//! These drive the real core modules against temp directories: no
//! router, no network, no touching `~`.

use modelsteward::core::{hermes, jsonc, opencode, piagent, router, safefs};
use std::collections::BTreeSet;

fn desired(id: &str, ctx: u64) -> opencode::DesiredModel {
    opencode::DesiredModel {
        id: id.into(),
        display_name: format!("{id} (llama.cpp)"),
        context: ctx,
        tool_call: Some(true),
        vision: false,
    }
}

/// The whole point of the app, end to end across three config formats:
/// measured contexts land in every connected agent, each in its own
/// schema, without disturbing anything the user put there.
#[test]
fn one_sync_reaches_every_connector_and_disturbs_nothing_else() {
    let tmp = tempfile::tempdir().unwrap();

    // A user who already has OpenCode configured by hand, with comments
    // and their own provider.
    let oc = tmp.path().join("opencode.json");
    std::fs::write(
        &oc,
        "{\n  \
           // my own notes, must survive\n  \
           \"provider\": {\n    \
             \"anthropic\": { \"models\": { \"claude\": {} } }\n  \
           }\n\
         }\n",
    )
    .unwrap();

    // pi, with its own ollama provider.
    let pi_dir = tmp.path().join(".pi/agent");
    std::fs::create_dir_all(&pi_dir).unwrap();
    let pi = pi_dir.join("models.json");
    std::fs::write(
        &pi,
        r#"{"providers":{"ollama":{"baseUrl":"http://127.0.0.1:11434/v1","models":[{"id":"gemma4"}]}}}"#,
    )
    .unwrap();

    // Hermes, with a cached context for an unrelated provider.
    let hm = tmp.path().join(".hermes");
    std::fs::create_dir_all(&hm).unwrap();
    std::fs::write(
        hermes::context_cache_path(&hm),
        "context_lengths:\n  gemma4:latest@http://127.0.0.1:11434/v1: 131072\n",
    )
    .unwrap();

    let want = [desired("qwen", 113_920), desired("small", 62_000)];
    let base = "http://127.0.0.1:8080/v1";

    // OpenCode
    let r = opencode::sync_file(&oc, base, &want).unwrap();
    assert_eq!(r.added.len(), 2, "{r:?}");
    let after = std::fs::read_to_string(&oc).unwrap();
    assert!(after.contains("my own notes"), "comments must survive:\n{after}");
    assert!(after.contains("anthropic"), "their provider must survive");
    jsonc::strictly_valid(&after).expect("a real JSON parser must accept our output");

    // pi
    let known: BTreeSet<String> = want.iter().map(|d| d.id.clone()).collect();
    let r = piagent::sync_file_with_known(&pi, base, &want, &known).unwrap();
    assert_eq!(r.added.len(), 2, "{r:?}");
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&pi).unwrap()).unwrap();
    assert!(doc["providers"]["ollama"]["models"][0]["id"] == "gemma4");
    let entry = |id: &str| -> u64 {
        doc["providers"]["modelsteward"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .unwrap_or_else(|| panic!("{id} missing from pi's config"))["contextWindow"]
            .as_u64()
            .unwrap()
    };
    assert_eq!(entry("qwen"), opencode::safety_context(113_920));
    assert_eq!(
        entry("small"),
        opencode::safety_context(62_000),
        "pi has no minimum-context floor, so the small model belongs here"
    );

    // Hermes — and its 64k floor means only the big one lands.
    let r = hermes::sync(&hm, base, &want).unwrap();
    assert_eq!(r.written, vec!["qwen".to_string()]);
    assert_eq!(
        r.below_minimum,
        vec!["small".to_string()],
        "a model Hermes would refuse must be skipped AND named"
    );
    let cache = std::fs::read_to_string(hermes::context_cache_path(&hm)).unwrap();
    assert!(cache.contains("gemma4:latest@"), "their entry survives:\n{cache}");
    assert!(cache.contains("qwen@http://127.0.0.1:8080/v1"), "{cache}");
}

/// The house rule, across every connector: a model that merely failed to
/// load today must never be deleted from a user's config.
#[test]
fn a_transient_load_failure_deletes_nothing_anywhere() {
    let tmp = tempfile::tempdir().unwrap();
    let base = "http://127.0.0.1:8080/v1";
    let both = [desired("keeper", 120_000), desired("flaky", 120_000)];

    let oc = tmp.path().join("opencode.json");
    std::fs::write(&oc, "{}\n").unwrap();
    opencode::sync_file(&oc, base, &both).unwrap();

    let pi_dir = tmp.path().join(".pi/agent");
    std::fs::create_dir_all(&pi_dir).unwrap();
    let pi = pi_dir.join("models.json");
    piagent::sync_file(&pi, base, &both).unwrap();

    // Next sync: 'flaky' failed to load, so it has no measured context
    // and drops out of `desired`. Both models are still in the fleet.
    let only_keeper = [desired("keeper", 120_000)];
    let still_here: BTreeSet<String> =
        ["keeper".to_string(), "flaky".to_string()].into_iter().collect();

    let r = opencode::sync_file(&oc, base, &only_keeper).unwrap();
    assert!(
        r.orphans.contains(&"flaky".to_string()),
        "OpenCode reports it as an orphan rather than removing it: {r:?}"
    );
    assert!(
        std::fs::read_to_string(&oc).unwrap().contains("flaky"),
        "and the entry is still in the file"
    );

    let r = piagent::sync_file_with_known(&pi, base, &only_keeper, &still_here).unwrap();
    assert!(r.removed.is_empty(), "pi must not delete it: {r:?}");
    assert_eq!(r.kept_unmeasured, vec!["flaky".to_string()]);
    assert!(
        piagent::configured_models(&pi).iter().any(|(id, _)| id == "flaky"),
        "and the entry is still in the file"
    );
}

/// Every connector must refuse a file it cannot safely rewrite, rather
/// than replacing the user's content with what little it understood.
#[test]
fn damaged_config_files_are_refused_not_flattened() {
    let tmp = tempfile::tempdir().unwrap();
    let base = "http://127.0.0.1:8080/v1";
    let want = [desired("qwen", 120_000)];

    // pi: a JSONC file (comments) — a round trip would delete them.
    let pi_dir = tmp.path().join(".pi/agent");
    std::fs::create_dir_all(&pi_dir).unwrap();
    let pi = pi_dir.join("models.json");
    let annotated = "{\n  // hand-tuned, do not clobber\n  \"providers\": {}\n}";
    std::fs::write(&pi, annotated).unwrap();
    assert!(piagent::sync_file(&pi, base, &want).is_err());
    assert_eq!(std::fs::read_to_string(&pi).unwrap(), annotated);

    // Hermes: malformed YAML — reading it as empty would wipe the cache.
    let hm = tmp.path().join(".hermes");
    std::fs::create_dir_all(&hm).unwrap();
    let broken = "context_lengths:\n  good@http://a: 1\n\tbad: 2\n";
    std::fs::write(hermes::context_cache_path(&hm), broken).unwrap();
    assert!(hermes::sync(&hm, base, &want).is_err());
    assert_eq!(
        std::fs::read_to_string(hermes::context_cache_path(&hm)).unwrap(),
        broken
    );
}

/// State survives an interrupted write: the store is either wholly old
/// or wholly new, and a damaged file is never silently read as empty.
#[test]
fn measurement_state_survives_interruption() {
    let dir = tempfile::tempdir().unwrap();
    let mut m = router::Measurements::new();
    m.insert(
        "qwen".into(),
        router::Measurement { n_ctx: Some(113_920), ..Default::default() },
    );
    router::write_measurements(dir.path(), &m).unwrap();

    // No stray temp files: a crash mid-write leaves the old file intact.
    let strays: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.contains(".tmp"))
        .collect();
    assert!(strays.is_empty(), "{strays:?}");

    // Truncate it the way an interrupted write would, and confirm the
    // damage is reported rather than read as "no measurements".
    let path = dir.path().join("measurements.json");
    let whole = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, &whole[..whole.len() / 2]).unwrap();
    let (loaded, damage) = router::read_measurements_checked(dir.path());
    assert!(loaded.is_empty());
    assert!(damage.is_some(), "a damaged store must be reported");
    assert!(
        matches!(
            safefs::read_json::<serde_json::Value>(&path),
            safefs::Loaded::Missing
        ),
        "the damaged file was moved aside, so the path is now free"
    );
}

// ── the Connector fan-out (family realignment, step 3) ───────────────

use modelsteward::core::connector::{self, Connector, SyncContext};

/// The fan-out drives real connectors against real file formats — the
/// layer that was previously written twice (once in the CLI, once in the
/// GUI) and had already drifted in wording, with no test over either.
#[test]
fn the_fan_out_writes_every_present_agent_and_says_so_once() {
    let tmp = tempfile::tempdir().unwrap();

    // pi installed: ~/.pi/agent/ exists.
    let pi_dir = tmp.path().join(".pi/agent");
    std::fs::create_dir_all(&pi_dir).unwrap();
    let pi_models = pi_dir.join("models.json");

    // Hermes installed: ~/.hermes/ exists.
    let hermes_home = tmp.path().join(".hermes");
    std::fs::create_dir_all(&hermes_home).unwrap();

    let want = [desired("qwen3.8-27b", 108_208), desired("big-moe", 262_144)];
    let known: BTreeSet<String> = want.iter().map(|d| d.id.clone()).collect();
    let cs: Vec<Box<dyn Connector>> = vec![
        Box::new(connector::PiConnector::at(pi_models.clone())),
        Box::new(connector::HermesConnector::at(hermes_home.clone())),
    ];
    let lines = connector::sync_all(
        &cs,
        &SyncContext { base_url: "http://127.0.0.1:8181/v1", desired: &want, known: &known },
    );

    assert!(
        lines.iter().any(|l| l.contains("pi agent synced")),
        "pi reported: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.contains("Hermes synced")),
        "hermes reported: {lines:?}"
    );
    assert!(pi_models.exists(), "pi's config was actually written");

    // The measured context reached pi's own schema, at the new port.
    let pi_text = std::fs::read_to_string(&pi_models).unwrap();
    assert!(pi_text.contains("8181"), "the base URL follows the router: {pi_text}");
    assert!(pi_text.contains("qwen3.8-27b"), "{pi_text}");
}

/// Neither agent installed: the whole fan-out is silent. A user who runs
/// only OpenCode must never read about pi or Hermes.
#[test]
fn the_fan_out_is_silent_when_no_agent_is_installed() {
    let tmp = tempfile::tempdir().unwrap();
    let want = [desired("m", 65_536)];
    let known: BTreeSet<String> = want.iter().map(|d| d.id.clone()).collect();
    let cs: Vec<Box<dyn Connector>> = vec![
        Box::new(connector::PiConnector::at(tmp.path().join("nope/models.json"))),
        Box::new(connector::HermesConnector::at(tmp.path().join("also-nope"))),
    ];
    let lines = connector::sync_all(
        &cs,
        &SyncContext { base_url: "http://127.0.0.1:8080/v1", desired: &want, known: &known },
    );
    assert!(lines.is_empty(), "absent agents say nothing: {lines:?}");
}

/// OpenCode through the trait produces the same file the typed path
/// does — the guarantee that lets agent #4 copy this pattern.
#[test]
fn opencode_through_the_trait_writes_a_real_config() {
    let tmp = tempfile::tempdir().unwrap();
    let oc = tmp.path().join("opencode.json");
    std::fs::write(&oc, "{\n  \"$schema\": \"https://opencode.ai/config.json\"\n}\n").unwrap();

    let want = [desired("qwen3.8-27b", 108_208)];
    let known: BTreeSet<String> = want.iter().map(|d| d.id.clone()).collect();
    let c = connector::OpenCodeConnector::at(oc.clone());
    assert!(c.present(), "a config that exists means OpenCode is installed");

    let out = c
        .sync(&SyncContext {
            base_url: "http://127.0.0.1:8080/v1",
            desired: &want,
            known: &known,
        })
        .expect("sync");
    assert!(!out.skipped_missing);
    assert!(out.summary.unwrap().contains("OpenCode synced"));

    let text = std::fs::read_to_string(&oc).unwrap();
    assert!(text.contains("qwen3.8-27b"), "{text}");
    assert!(jsonc::strictly_valid(&text).is_ok(), "still valid JSONC:\n{text}");
}
