# Code review — correctness, safety, and structure

2026-08-31 · v0.6.5 + post-tag commits · 22,086 lines Rust, 189 tests ·
three independent reviewers, every finding below re-verified by the dev
session against the actual source before it landed here.

Scott asked for brutal and honest. This is that. Companion document:
[EFFICIENCY-RELIABILITY.md](EFFICIENCY-RELIABILITY.md).

## The verdict in one paragraph

**The core engine is in good shape; the third of the codebase that
isn't core is where the danger lives.** `src/core/` has real module
boundaries, sane function lengths, and parsers and accounting tested to
a standard most projects never reach — the ledger, the JSONC editor,
the diagnosis rules, and the verdict guards all carry incident dates and
live numbers in their test bodies, exactly as CLAUDE.md prescribes. But
`src/ui.rs` (6,466 lines) plus `src/main.rs` (670) is **32% of the
codebase carrying one test, and that test checks font glyphs.** That
untested third holds the sync orchestration, the auto-build gate, the
install pick, and three copies of a verdict rule. Worse, the newest
work — the agent connectors shipped 2026-08-30 — landed *there*, and
introduced the two data-loss bugs at the top of this list. The debt is
narrow, specific, and already written down in ROADMAP; it is simply
being overtaken by feature work.

**Honest note on provenance:** the two most severe connector findings
(C4, C5) are in code the dev session wrote this week, and one of them
is *pinned by a test that asserts the wrong behavior*. The reviews
caught what the author did not.

---

## CRITICAL — data loss

### C1. `measurements.json` can be silently zeroed, then permanently overwritten

`core/router.rs:599` reads with `.ok().and_then(...).unwrap_or_default()`
— **any** parse failure yields an empty map, with no error. `router.rs:650`
writes the whole map with a plain truncating `fs::write`, no temp+rename.
Every mutation is read → modify → write-whole-file, from eleven call
sites.

**Failure chain (verified):** `--calibrate` persists after each model.
Ctrl-C inside that write leaves half a JSON document. Next launch: parse
fails, error swallowed, map empty, Library renders blank. The user
measures one model → a one-entry map is written, **permanently
destroying every other model's context, tool-call verdict, speed
baselines, and quality scores.** No backup, no warning.

`core/settings.rs:178` already implements exactly the right rescue for
`config.json` (`.corrupt` sidecar + loud error). `router.rs` has none.

**Fix:** temp+rename for the write; make the read distinguish missing
(silent default) from unparseable (rescue + refuse to overwrite).

### C2. `trials.json` — identical pattern, wider interrupt window

`core/trial.rs:425` (silent-empty read) and `:434` (truncating write).
`trial.rs:1692/1704` rewrite the whole file **after every variant
round** — dozens of truncating writes across a 20-minute campaign. One
corrupt read plus one more round wipes every stored variant for every
model, including applied winners.

### C3. `restore_last_backup` destroys the backup before the restore lands

`core/opencode.rs:245-247`, in this order:

```rust
std::fs::write(&tmp, &backup)?;      // stage restore
std::fs::write(&bak, &current)?;     // backup slot NOW holds the bad config
std::fs::rename(&tmp, path)?;        // restore lands only here
```

**Failure scenario:** the user hits "restore backup" after a bad sync.
The process is killed — or the rename fails on a read-only dir or full
disk — between those lines. `opencode.json` still holds the bad config,
and `.bak.1` now holds it too. **The good config is gone from both.**
This is the undo path: the one place a user reaches when something has
already gone wrong.

**Fix:** rename tmp→path first, *then* stash the previous content.

### C4. The pi connector deletes user entries on a transient measurement failure

`core/piagent.rs:132-136` removes any id absent from `desired` — with no
reachability check. And `desired` is built by filtering on
`m.n_ctx.is_some()` (`ui.rs:5761`, `main.rs:494`), while a failed
re-measure writes `n_ctx: None` (`router.rs:864`).

**So a model that measured fine yesterday and merely fails to load today
— VRAM held by another session, a transient OOM — is deleted from the
user's `~/.pi/agent/models.json`.** This directly violates the
CLAUDE.md invariant *"A stopped provider is not evidence its models are
gone"*, and `opencode.rs` does the opposite in the same situation
(reports an orphan, refuses to touch it).

**The test pins the bug**:
`piagent.rs:262` `context_changes_update_and_lost_measurements_remove`
asserts `r.removed == vec!["b"]`. It will need rewriting, not just the
code. The doc comment at `piagent.rs:47-50` claims the house rule is
honored; it is wrong.

### C5. A YAML hiccup makes the Hermes connector wipe other providers' cached contexts

`core/hermes.rs:119` — parse failure → `.ok()` → `None` →
`unwrap_or_default()` → **empty map**, which `hermes.rs:158` then
serializes as the *entire file*.

**Failure scenario:** the user hand-edits
`~/.hermes/context_length_cache.yaml` and leaves a tab indent. Next
sync: parse fails silently, our models are inserted into an empty map,
and the file is rewritten containing **only modelsteward entries** —
every Ollama and cloud context Hermes had cached is deleted. A second
leak in the same expression: entries whose value isn't a plain unsigned
integer are dropped even on a *successful* parse, then the survivors
written back.

`piagent.rs:104` handles this class correctly — it *refuses* to rewrite
a file it can't parse, with a test. Hermes needs the same.

---

## HIGH — wrong behavior users will hit

### H1. `Msg::Scanned` clears `busy` without ever having claimed it — the concurrency gate collapses

`spawn_scan` (`ui.rs:499`) is a raw thread that never sets `busy`, but
its message handler sets `self.busy = None` (`ui.rs:1595`). The buttons
that trigger it ("Rescan System", "Save & Rescan") are ungated.

**Verified scenario:** a 40-minute Lab campaign is running. The user
clicks File → Rescan System. The scan finishes in seconds and clears
`busy`. The gate is now open: a second worker can start, **two workers
drive the same router and both read-modify-write `measurements.json` and
`trials.json`** — last writer wins. Worse, "Regenerate Preset" is now
enabled and will overwrite the trial's *temporary* preset mid-round, so
the remaining rounds measure a config nobody chose and record it as
truth.

Same class: `Msg::PresetWritten` clears `busy` while
`action_apply_settings_now` keeps working for many more seconds.

**Fix:** only `Finished`/`Error`/`SyncDone` clear `busy`; give
`Scanned`/`PresetWritten` the `// does not clear busy` treatment
`Msg::Measurements` already has.

### H2. Cancel silently stops cancelling

`ui.rs:832` (and `:986`) replace `self.cancel_token` **before** calling
`spawn`, which may bail early because something else is busy.

**Verified scenario:** a campaign is running holding token T1. The user
clicks an ungated Server-menu item (`ui.rs:1784/1796/1811/1822` carry no
`add_enabled` guard) → the token is swapped to T2 → spawn logs "busy
with…" and returns. The user now clicks ✖ Cancel, which flips **T2**,
which nobody holds. **The campaign runs to completion — potentially
hours of GPU time — while the UI displays "cancelling — stopping at the
next safe point…"**

**Fix:** move the token reset inside `spawn`, after the busy check.

### H3. A non-ASCII character in the log permanently kills the status poller

`core/meter.rs:107`: `&log[..log.len().min(4096)]` — a fixed **byte**
offset into a `&str`. A multibyte character straddling byte 4096 panics
with "byte index 4096 is not a char boundary".

**Reachable:** the first ~1KB of `router.log` echoes preset aliases
derived from model filenames, HF repo ids, and `$HOME` — a non-ASCII
username or an accented model directory puts multibyte bytes in range.
**There is no `catch_unwind` and no panic hook anywhere in `src/`**, so
the poller thread simply dies: router status, VRAM, meter line and the
activity indicator freeze forever with no error shown, and the GUI looks
fine. Because the log only appends, the first 4KB never changes — it
re-panics on every launch until the log is deleted.

**Fix:** hash bytes, not `&str` (`fnv` immediately calls `as_bytes()`
anyway, so nothing is gained by the `&str` slice).

### H4. Commenting out ghosts burns all five `opencode.json` backups in one sync

`opencode.rs:325` loops, and each iteration does a full read + its own
`write_backed_up`, which rotates `.bak.1→.2→…→.5` and drops what falls
off the end. **A sync that finds 5 ghosts performs 6 rotations, so the
user's pre-sync config walks off the end of the backup stack and is
gone**, with all five recovery slots holding intermediate states of the
same operation.

**Fix:** apply all comment-outs to one in-memory source, write once.

### H5. Auto-build can compile beside a measurement

The idle gate (`ui.rs:741`) checks `busy_flag` before *starting* a
build, but the build thread never sets it. **So a build started at 03:00
does not stop a bench started at 03:05** — `llama-bench` then measures
tokens/sec on a machine saturated by cmake and writes that as a
baseline, which is precisely what the gate's own comment says must never
happen. Related: `advisor::run_rebuild` (`advisor.rs:745`) never takes
`BUILD_LOCK`, and that lock is a process-local `AtomicBool` anyway — no
protection against a CLI `--verify-rebuild` racing the GUI.

### H6. `heal_interrupted_trial` reports a restoration it may not have performed

`trial.rs:354`: `let _ = write_preset(...); let _ = reload(...);` then
clears the marker and returns "restored the real preset". If the write
failed (full disk, unwritable dir) the user is told it was restored, the
breadcrumb is destroyed, **and the router keeps serving the trial's
temporary config indefinitely** — every subsequent measurement and every
synced limit taken against a config nobody chose.

### H7. The sync flow is implemented twice, and the copies already disagree

`main.rs::sync` (91 lines) and `ui.rs::run_sync` (83 lines) are
independent implementations of the same pipeline. Neither is tested.
**Verified divergence:** `main.rs:531` returns early when `opencode.json`
is missing — *before* the pi and Hermes blocks. So on a machine without
OpenCode, `--sync` silently skips both agents while the GUI syncs them.
Same command, two answers.

### H8. Two divergent implementations of "which llama-server do we use"

`system::pick_server` (what the router actually launches) and
`App::picked_server` (what the GUI displays and what feeds the AI
advisory's build number) apply different rules and can name different
binaries. Neither has a test. ROADMAP already logs this as "aligned
today, still two copies" — the alignment does not survive reading the
two bodies.

### H9. `config.json` lost update: the GUI overwrites CLI writes

The GUI loads config once at startup and holds it for the session. A CLI
`--trial … keep <winner>` writes the winning override to disk; the next
Settings save in the GUI writes the **stale** struct back, erasing the
kept winner, plus any `disabled` entries or checkouts the CLI added.

### H10. Backup policy is triplicated and unequal

`opencode.rs` has the good one (5-deep rotation, temp+rename,
no-op-when-unchanged, tested). pi and Hermes each hand-roll **a single
fixed backup slot, overwritten every sync**, followed by a
non-atomic write — so **two consecutive syncs destroy the user's
pre-modelsteward original.** `hermes::register_provider` non-atomically
rewrites the 0600 `config.yaml` holding cloud API keys and has no test
at all.

---

## MEDIUM (abridged — full detail in the reviewer notes)

- **M1. The findings report leaks filesystem paths through trials.**
  `report.rs` carefully sanitizes measurement and history errors, then
  passes `trials` straight through to both the markdown and the JSON
  sidecar. `TrialResult.error` holds real load-failure strings — the
  exact place paths hide. The report is the artifact meant for public
  sharing. It also uses `env::home_dir()` rather than
  `settings::real_home()`, so under a snap-redirected HOME it sanitizes
  the wrong string and leaks the real one.
- **M2. The build-history table keeps the OLDEST measurement per build**
  (`report.rs:186`, `or_insert`) while its comment says "Latest" — and
  `history::build_deltas` answers the same question with `max_by_key`.
  A build that improved after one bad first measurement is reported as
  not having improved.
- **M3. `safety_context` can write `"context": 0`** into every agent
  config for a tiny measured window. The test states "callers must treat
  0 as unusable" — **no caller does.**
- **M4. `upsert_measurement` carries six fields forward** with no direct
  test. Add a seventh measured field and forget the list, and every
  re-measure silently wipes it.
- **M5. Meter cursor is a non-atomic two-phase commit.** A crash between
  ledger append and cursor write re-credits; a torn cursor write yields
  a default cursor and re-credits **the entire log**.
- **M6. `try_lock`'s stale-steal is a TOCTOU** — two processes can both
  believe they hold the harvest lock after a prior crash.
- **M7. The trial marker is a breadcrumb, not a mutex.** A CLI trial
  started beside a GUI campaign overwrites the marker, both write
  competing presets, both drive the same router, and every recorded
  number is garbage.
- **M8. `improvement()` scores a perfect result as the worst possible**
  — a zero candidate on the inverted goals (load time, agent turn)
  returns `0.0` and is silently discarded as "doesn't earn its keep".
- **M9. Log-tail readers render empty on a split UTF-8 boundary**
  (three sites), so the Server tab's log can go blank and a load failure
  can lose its actual cause.
- **M10. Nothing is written atomically except `opencode.json`** and
  shelf copies — eleven truncating `fs::write` call sites, including
  the Hermes config that holds API keys.
- **M11. pi/Hermes mirror readers swallow parse errors into an empty
  Vec**, so a broken file renders as "not synced yet" rather than
  "broken" — inviting a re-sync onto a file we can't read.
- **M12. Three untested predicates that fail in the safe-looking
  direction**: the bench freshness gate can report "nothing to bench"
  and exit 0 having done nothing; `--quality <id> 0` records a *measured*
  reliability of 0.0 that then demotes a good model in advisor
  selection; `gguf::as_u64` sign-extends `-1` into 1.8e19, which would
  make anything read as MoE.
- **M13. `opencode::default_config_path` is a fourth hand-rolled XDG
  resolver** without the snap guard the other three now share.

## Structural findings

- **S1. `spawn_status_poller` is 277 lines of untestable policy** —
  auto-build eligibility, queue lifecycle, activity debounce, upstream
  scheduling. ROADMAP already specifies extracting it to
  `managed::auto_build_tick`. It has been re-tuned twice from live
  incidents, which is exactly the code CLAUDE.md says must ship pinned.
- **S2. `system::write_preset` has 14 call sites and zero tests** — it
  owns disabled-model filtering, override application, the mmproj
  ride-along rule, and embedding flags. Every branch has a comment
  explaining a past bug; none has a test. If it regresses, the router
  serves a different config and *every measurement after is against a
  config nobody chose* — invisible, because the numbers still look
  plausible.
- **S3. The Connector trait was designed, agreed, ordered first — and
  two connectors shipped without it.** Honest scoping from the reviewer:
  sell it as **one backup path, one removal precondition, one presence
  contract** — not as deduplication. As line-count reduction it will
  disappoint; as the fix for C4, C5 and H10 by construction, it earns
  itself.
- **S4. The 13-item Lab menu list is repeated across 11 sites** in
  `ui.rs` with no test that the strings resolve; a typo is a runtime
  error, never a compile error.
- **S5. Ten UI functions exceed 250 lines** (`lab_pane` 625,
  `library_pane` 446, `advisor_window` 401). The problem isn't length —
  core's `rows::assemble` is 264 lines with 12 tests — it's length
  *plus* zero reachability. Named extractable candidates: history-trail
  assembly, row sort ranking, ETA composition, `parse_edit_buffers`,
  `serving_disruption`/`vram_contention`.
- **S6. `tests/` contains fixtures and zero integration tests**, while
  CLAUDE.md promises them for "scan → preset → measurements" and
  "harvest → ledger → report".
- **S7. CLAUDE.md's own architecture map is stale**: it says `src/ui/`
  (a directory; it's one file) and lists 20 modules where `core.rs`
  declares 26 — **3,224 lines absent from the document a new maintainer
  is told to read first.**
- **S8. `lib.rs` exposes `pub mod ui`**, which disables dead-code
  detection crate-wide *and* makes the entire egui shell the public API
  of a published crate — every internal refactor is a semver break.

---

## What's genuinely solid (probed and cleared, not courtesy)

- **JSONC comment preservation is correct by construction** — a
  span-splice editor, not a reserialize. User comments and hand-set keys
  survive round trips, test-verified.
- **The orphan/ghost policy matches the house rule exactly**: removal
  requires *both* a reachable router omitting the id *and* no backing
  measurement.
- **Router ownership is sound.** Marker + `/proc` cmdline matching
  require both `llama-server` and our preset path, so a recycled PID is
  never a kill target. No violation of "never touch a server we didn't
  start" was found.
- **`settings.rs`'s corrupt-config handling is the model** the
  measurements and trials stores should copy.
- **No `Mutex`/`RwLock` anywhere** — all cross-thread state moves
  through channels, so there is no lock-poisoning or deadlock surface;
  `run_steps` correctly drains stderr on its own thread so neither pipe
  can fill and deadlock.
- **Migrations are gated on `!new.exists()`** — a rename can never
  overwrite live data.
- **The best tests here are excellent and follow the stated practice**:
  the meter's idempotence-across-instances test, the two double-credit
  incident pins, `build_deltas_compare_builds_not_configs`, the snap
  redirect test covering `/home/snap` and `/data/snap/scott`, the prune
  test with both exemptions, and the diagnosis tests that assert the
  *absence* of a bad remedy.
- **`rows.rs` proves the extraction pattern works** — 942 lines of what
  would otherwise be UI logic, in core, with 12 tests. It is the model
  for S5.
- **ROADMAP is an honest debt ledger** — all four structural findings
  above were already written down in it. This codebase is not blind to
  its own debt; it is being outrun by feature work.

---

## Recommended order

1. **C1, C2** — atomic writes + rescue-don't-zero for measurements and
   trials. Copy the pattern `settings.rs` already has.
2. **C3, H4** — the two `opencode.json` backup bugs. Both are small and
   both destroy the user's last recovery point.
3. **C4, C5** — the connector data-loss pair. Fix the behavior *and* the
   test that pins C4.
4. **H1, H2** — the busy-gate and cancel-token bugs. Both are a few
   lines and both currently mislead the user about what is happening.
5. **H3** — the poller-killing panic. One-line fix.
6. **H5, H6, H10** — build-lock symmetry, honest heal reporting, one
   shared backup helper.
7. **H7, H8, H9** — extract the shared sync/pick/config paths into core
   so the two surfaces stop diverging. This is the S3 refactor's real
   payoff.
8. Structural work (S1, S2, S6) as capacity allows; the MEDIUM list
   opportunistically.

## Dev session responses

*(Append here when acting on these, as the usability review does.)*
