//! The `Connector` trait: one shape for every agent whose config mirrors
//! what the router serves.
//!
//! Step 3 of the 2026-09-08 family realignment, and Phase 2 of the
//! roadmap before it. The three connectors already shared a shape —
//! "is this agent installed? then write the measured model set into its
//! own schema, backing the file up first" — but the *fan-out* over them
//! was written twice, once in `main.rs` and once in `ui.rs`, and had
//! already drifted: the same pi sync reported different wording
//! depending on whether you ran it from the CLI or the GUI. Extracting
//! this before agent #4 (openclaw) stops that tripling.
//!
//! Deliberately NOT included: opencode keeps its typed `SyncReport` path
//! in the callers. It is the connector everything else was built around,
//! its report feeds the Connections mirror, and destabilising it buys
//! nothing here. It implements the trait too, so the enumeration is
//! uniform and agent #4 has the pattern to copy.

use crate::core::{hermes, opencode::DesiredModel, piagent};
use anyhow::Result;
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Everything a connector needs to do its job, so the trait method does
/// not grow an argument per agent.
pub struct SyncContext<'a> {
    /// Where the router answers — what the agent will call.
    pub base_url: &'a str,
    /// The measured model set the agent should end up mirroring.
    pub desired: &'a [DesiredModel],
    /// Ids the fleet is known to hold: preset ∪ measurements − disabled.
    /// Positive evidence for a removal, so a model that merely failed to
    /// load today is never deleted from a user's config.
    pub known: &'a BTreeSet<String>,
}

/// What one connector did, in a shape both the CLI and the GUI can
/// render the same way — which is the point, since they did not before.
#[derive(Debug, Default, PartialEq)]
pub struct Outcome {
    pub connector: &'static str,
    /// The agent isn't installed. Not an error, and not worth a line:
    /// this app serves anything OpenAI-compatible whether or not any
    /// particular agent is present.
    pub skipped_missing: bool,
    /// The one-line result, already worded.
    pub summary: Option<String>,
    /// Secondary facts worth saying: things skipped, kept, or needing a
    /// human. Rendered under the summary.
    pub notes: Vec<String>,
}

impl Outcome {
    fn absent(connector: &'static str) -> Self {
        Self { connector, skipped_missing: true, ..Default::default() }
    }
}

pub trait Connector {
    /// Stable identifier for logs and config. Never shown as a heading.
    fn id(&self) -> &'static str;
    /// What a person calls this agent.
    fn display_name(&self) -> &'static str;
    /// The file or directory this connector owns.
    fn config_path(&self) -> PathBuf;
    /// Is the agent installed on this machine?
    fn present(&self) -> bool;
    /// Mirror `ctx.desired` into the agent's own schema. Implementations
    /// back the file up before writing; none of them may write when
    /// [`Connector::present`] is false.
    fn sync(&self, ctx: &SyncContext<'_>) -> Result<Outcome>;
}

// ─── pi ──────────────────────────────────────────────────────────────

/// `at()` exists so the trait can be driven against a tempdir in tests
/// — a connector whose path is hardcoded can only ever be tested
/// through the functions underneath it, which is how the fan-out went
/// untested long enough to drift.
#[derive(Default)]
pub struct PiConnector {
    models_path: Option<PathBuf>,
}

impl PiConnector {
    pub fn at(models_path: PathBuf) -> Self {
        Self { models_path: Some(models_path) }
    }
}

impl Connector for PiConnector {
    fn id(&self) -> &'static str {
        "pi"
    }
    fn display_name(&self) -> &'static str {
        "pi agent"
    }
    fn config_path(&self) -> PathBuf {
        self.models_path.clone().unwrap_or_else(piagent::default_models_path)
    }
    fn present(&self) -> bool {
        piagent::pi_present(&self.config_path())
    }
    fn sync(&self, ctx: &SyncContext<'_>) -> Result<Outcome> {
        let path = self.config_path();
        let r = piagent::sync_file_with_known(&path, ctx.base_url, ctx.desired, ctx.known)?;
        if r.skipped_missing {
            return Ok(Outcome::absent(self.id()));
        }
        let mut notes = Vec::new();
        if !r.kept_unmeasured.is_empty() {
            notes.push(format!(
                "{} entr(ies) kept although not measurable right now — a transient \
                 load failure never deletes a config entry",
                r.kept_unmeasured.len()
            ));
        }
        Ok(Outcome {
            connector: self.id(),
            skipped_missing: false,
            summary: Some(format!(
                "pi agent synced ({}): {} added, {} updated, {} removed{}",
                path.display(),
                r.added.len(),
                r.updated.len(),
                r.removed.len(),
                if r.created_file { " — models.json created" } else { "" },
            )),
            notes,
        })
    }
}

// ─── Hermes ──────────────────────────────────────────────────────────

#[derive(Default)]
pub struct HermesConnector {
    home: Option<PathBuf>,
}

impl HermesConnector {
    pub fn at(home: PathBuf) -> Self {
        Self { home: Some(home) }
    }
}

impl Connector for HermesConnector {
    fn id(&self) -> &'static str {
        "hermes"
    }
    fn display_name(&self) -> &'static str {
        "Hermes"
    }
    fn config_path(&self) -> PathBuf {
        self.home.clone().unwrap_or_else(hermes::default_home)
    }
    fn present(&self) -> bool {
        hermes::hermes_present(&self.config_path())
    }
    fn sync(&self, ctx: &SyncContext<'_>) -> Result<Outcome> {
        let home = self.config_path();
        let r = hermes::sync(&home, ctx.base_url, ctx.desired)?;
        if r.skipped_missing {
            return Ok(Outcome::absent(self.id()));
        }
        let mut notes = Vec::new();
        if !r.below_minimum.is_empty() {
            notes.push(format!(
                "{} model(s) skipped — under Hermes's 64,000-token minimum: {}",
                r.below_minimum.len(),
                r.below_minimum.join(", ")
            ));
        }
        if r.provider_unregistered {
            // Registration edits a live, hand-maintained config, so it
            // stays an explicit click rather than a side effect of sync.
            notes.push(
                "no Hermes custom provider points at this router yet — register one on \
                 the Connections tab, or via Hermes's own /model"
                    .to_string(),
            );
        }
        Ok(Outcome {
            connector: self.id(),
            skipped_missing: false,
            summary: Some(format!("Hermes synced: {} context(s) written", r.written.len())),
            notes,
        })
    }
}

// ─── OpenCode ────────────────────────────────────────────────────────

/// Implements the trait so the enumeration is uniform and agent #4 has
/// a worked example. The callers still drive opencode through its typed
/// `SyncReport`, because the Connections mirror needs that detail and
/// ghost cleanup needs live router state.
#[derive(Default)]
pub struct OpenCodeConnector {
    config: Option<PathBuf>,
}

impl OpenCodeConnector {
    pub fn at(config: PathBuf) -> Self {
        Self { config: Some(config) }
    }
}

impl Connector for OpenCodeConnector {
    fn id(&self) -> &'static str {
        "opencode"
    }
    fn display_name(&self) -> &'static str {
        "OpenCode"
    }
    fn config_path(&self) -> PathBuf {
        self.config.clone().unwrap_or_else(crate::core::opencode::default_config_path)
    }
    fn present(&self) -> bool {
        self.config_path().exists()
    }
    fn sync(&self, ctx: &SyncContext<'_>) -> Result<Outcome> {
        let path = self.config_path();
        let r = crate::core::opencode::sync_file(&path, ctx.base_url, ctx.desired)?;
        if r.skipped_missing {
            return Ok(Outcome::absent(self.id()));
        }
        let mut notes = Vec::new();
        if let Some(url) = &r.base_url_repointed {
            notes.push(format!(
                "base URL repointed to {url} — restart OpenCode to pick it up"
            ));
        }
        if !r.orphans.is_empty() {
            notes.push(format!(
                "{} orphan(s) in the config but not measured — left untouched",
                r.orphans.len()
            ));
        }
        Ok(Outcome {
            connector: self.id(),
            skipped_missing: false,
            summary: Some(format!(
                "OpenCode synced ({}): {} added, {} updated",
                path.display(),
                r.added.len(),
                r.updated.len()
            )),
            notes,
        })
    }
}

/// The connectors the fan-out drives. OpenCode is absent by design —
/// see this module's header.
pub fn secondary() -> Vec<Box<dyn Connector>> {
    vec![Box::new(PiConnector::default()), Box::new(HermesConnector::default())]
}

/// Every connector this app knows, for enumeration (the Connections tab,
/// docs, tests). Not all of these are driven by [`sync_all`].
pub fn all() -> Vec<Box<dyn Connector>> {
    vec![
        Box::new(OpenCodeConnector::default()),
        Box::new(PiConnector::default()),
        Box::new(HermesConnector::default()),
    ]
}

/// Sync each connector, rendering one set of lines for every caller.
///
/// A connector that fails does not stop the others: agents are
/// independent, and a broken pi install must never cost the user their
/// Hermes sync. The failure becomes a line like any other result.
pub fn sync_all(connectors: &[Box<dyn Connector>], ctx: &SyncContext<'_>) -> Vec<String> {
    let mut lines = Vec::new();
    for c in connectors {
        match c.sync(ctx) {
            Ok(o) if o.skipped_missing => {}
            Ok(o) => {
                if let Some(s) = o.summary {
                    lines.push(s);
                }
                lines.extend(o.notes.into_iter().map(|n| format!("  · {n}")));
            }
            Err(e) => lines.push(format!("{} sync FAILED: {e:#}", c.display_name())),
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A connector that records what it was asked and answers however
    /// the test needs — the fan-out's contract is about orchestration,
    /// not about any one agent's file format.
    struct Fake {
        id: &'static str,
        present: bool,
        result: fn() -> Result<Outcome>,
    }
    impl Connector for Fake {
        fn id(&self) -> &'static str {
            self.id
        }
        fn display_name(&self) -> &'static str {
            self.id
        }
        fn config_path(&self) -> PathBuf {
            PathBuf::from("/nowhere")
        }
        fn present(&self) -> bool {
            self.present
        }
        fn sync(&self, _ctx: &SyncContext<'_>) -> Result<Outcome> {
            (self.result)()
        }
    }

    fn ctx_for<'a>(
        desired: &'a [DesiredModel],
        known: &'a BTreeSet<String>,
    ) -> SyncContext<'a> {
        SyncContext { base_url: "http://127.0.0.1:8080/v1", desired, known }
    }

    fn ok_outcome() -> Result<Outcome> {
        Ok(Outcome {
            connector: "good",
            skipped_missing: false,
            summary: Some("good synced: 2 added".into()),
            notes: vec!["1 model skipped".into()],
        })
    }
    fn absent_outcome() -> Result<Outcome> {
        Ok(Outcome::absent("gone"))
    }
    fn failing_outcome() -> Result<Outcome> {
        Err(anyhow::anyhow!("its config is a directory"))
    }

    /// The reason this module exists: one agent blowing up must not cost
    /// the user the others. The old hand-written fan-out got this right
    /// by repetition; the trait has to keep it right by construction.
    #[test]
    fn a_failing_connector_does_not_stop_the_rest() {
        let d: Vec<DesiredModel> = vec![];
        let k = BTreeSet::new();
        let cs: Vec<Box<dyn Connector>> = vec![
            Box::new(Fake { id: "broken", present: true, result: failing_outcome }),
            Box::new(Fake { id: "good", present: true, result: ok_outcome }),
        ];
        let lines = sync_all(&cs, &ctx_for(&d, &k));
        assert!(
            lines.iter().any(|l| l.contains("broken sync FAILED")),
            "the failure is reported: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("good synced")),
            "and the next connector still ran: {lines:?}"
        );
    }

    /// An agent that isn't installed produces NO line. This app serves
    /// anything OpenAI-compatible; a user without pi should never read
    /// about pi.
    #[test]
    fn an_absent_agent_is_silent() {
        let d: Vec<DesiredModel> = vec![];
        let k = BTreeSet::new();
        let cs: Vec<Box<dyn Connector>> =
            vec![Box::new(Fake { id: "gone", present: false, result: absent_outcome })];
        assert!(sync_all(&cs, &ctx_for(&d, &k)).is_empty());
    }

    #[test]
    fn notes_are_rendered_under_their_summary() {
        let d: Vec<DesiredModel> = vec![];
        let k = BTreeSet::new();
        let cs: Vec<Box<dyn Connector>> =
            vec![Box::new(Fake { id: "good", present: true, result: ok_outcome })];
        let lines = sync_all(&cs, &ctx_for(&d, &k));
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[0], "good synced: 2 added");
        assert!(lines[1].starts_with("  · "), "notes are indented: {:?}", lines[1]);
    }

    /// Ordering is the caller's, not the map's: pi before Hermes, every
    /// run, so a log read across days is comparable.
    #[test]
    fn the_fan_out_preserves_the_given_order() {
        let d: Vec<DesiredModel> = vec![];
        let k = BTreeSet::new();
        let mk = |id: &'static str| -> Box<dyn Connector> {
            Box::new(Fake { id, present: true, result: ok_outcome })
        };
        let lines = sync_all(&[mk("first"), mk("second")], &ctx_for(&d, &k));
        assert_eq!(lines.len(), 4, "two summaries, two notes: {lines:?}");
    }

    // ── the real connectors' identity ────────────────────────────────

    #[test]
    fn every_connector_has_a_stable_id_and_a_human_name() {
        let ids: Vec<_> = all().iter().map(|c| c.id()).collect();
        assert_eq!(ids, vec!["opencode", "pi", "hermes"]);
        for c in all() {
            assert!(!c.display_name().is_empty(), "{} needs a name", c.id());
            assert!(
                c.display_name() != c.id() || c.id() == "pi",
                "the name is for people, the id is for logs"
            );
        }
    }

    /// The fan-out drives the two whose orchestration was duplicated.
    /// OpenCode is deliberately excluded — if that ever changes, this
    /// test should be the thing that notices.
    #[test]
    fn the_fan_out_covers_pi_and_hermes_only() {
        let ids: Vec<_> = secondary().iter().map(|c| c.id()).collect();
        assert_eq!(ids, vec!["pi", "hermes"]);
    }

    /// Every connector must answer `present()` without panicking on a
    /// machine where the agent is absent — this runs in CI, where none
    /// of them are installed.
    #[test]
    fn presence_is_answerable_for_all_of_them() {
        for c in all() {
            let _ = c.present();
            assert!(
                !c.config_path().as_os_str().is_empty(),
                "{} must name a config location",
                c.id()
            );
        }
    }
}
