# Code review — correctness, safety, and structure

2026-08-31 · v0.6.5 + post-tag commits · 22,086 lines Rust, 189 tests ·
four independent reviewers, every finding below re-verified by the dev
session against the actual source before it landed here. Findings marked
**EXECUTED** were reproduced by compiling the real code against a
synthetic input in a scratch copy — not inferred by reading.

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
introduced three of the ten data-loss findings below. The debt is
narrow, specific, and already written down in ROADMAP; it is simply
being overtaken by feature work.

**The single worst finding is C6**, and it is not new code: the app's
own ghost-commenting can make `opencode.json` unparseable, and because
our reader is lenient about the exact damage our writer causes, **every
surface reports success while OpenCode cannot load the file.** Two
reviewers independently converged on the same shape — *a lenient reader
hiding a destructive writer* — which is also C7's shape. That pattern,
not any individual bug, is the thing to internalize.

**Honest note on provenance:** the connector findings (C4, C5, C10) are
in code the dev session wrote this week, and one is *pinned by a test
that asserts the wrong behavior*. The reviews caught what the author did
not.

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

### C6. Sync can corrupt `opencode.json` into invalid JSON, and the app cannot see that it did — **EXECUTED**

`core/jsonc.rs:607-613`. When inserting a new model, the cursor walks
back over **whitespace only** to decide where the separating comma goes.
It has no idea whether it landed inside a `//` comment — so when the
last content before the closing brace is a line comment, **the comma is
written inside that comment**, where a strict parser never sees it.

**The app creates its own trigger.** `comment_out_ghosts` runs at the
end of every sync and leaves a commented block as the last content; the
next sync's insert then splices after it. Reproduced end-to-end through
the real `sync_file`:

```
"keeper": { ... }
// Commented out by modelsteward: ...
// "ghost": { "name": "Ghost" },     <-- the comma landed in here
"newmodel": { ... }
```

`keeper` and `newmodel` now have no separator between them.

**The blindness is the dangerous half.** `jsonc_parser` is lenient about
missing commas, so our own re-parse, the model-id listing, and the
Connections mirror all still succeed. The sync reports success, the UI
shows a healthy mirror — and **OpenCode cannot load the file**
(`expected ',' or '}' at line 8`). The next sync writes again, rotating
the last good backup away while OpenCode stays broken.

Also fires from a user's own trailing comment, an inline `// note` after
the last entry, and inside a single entry during the tool-call/modality
fill-in. Block comments are safe.

**Fix:** validate before writing — a strict, comment-stripped
`serde_json` parse of the output before `write_backed_up`. A lenient
re-parse cannot catch this.

### C7. An *unreadable* `config.json` is silently replaced by defaults, and the rescue never fires — **EXECUTED**

`core/settings.rs:158`: `Err(_) => (Self::default(), None)`. The C8 fix
from the usability review guards a **parse** failure. A **read** failure
— a permission change, an EIO, an NFS hiccup — takes this arm and
reports "nothing wrong". Then `save()` gates its `.corrupt` rescue on a
successful read, so the rescue is skipped too and the write clobbers.

Executed: scan dirs, port, **per-model overrides (where kept trial
winners live)**, checkouts and the disabled list all silently replaced
by defaults, `err=None`, no rescue file. This is the C8 incident
re-entering through a different door.

**Fix:** match `NotFound` specifically; every other `Err` is loud.

### C8. `RouterState::Ours` never checks the port, so we can drive a server we didn't start — **VERIFIED**

`core/router.rs:409` proves ownership by preset path and PID only.
`Marker.port` is written at `router.rs:461` and — confirmed by grep —
**never read anywhere in the tree**.

**Scenario:** our router runs on 18080; the user changes the port to
8080 in Settings, where their own hand-run llama-server lives (a setup
CLAUDE.md explicitly names). `fetch_models(8080)` succeeds against plain
llama-server and returns an empty list rather than an error, so the
state is `Ours`. `Ours` gates every mutating path — `run_trial` then
fires reloads, loads, and chat probes at **the stranger's server** and
records the results as measurements. `stop()` takes no port at all.

This is the "never touch a server we didn't start" rule leaking through
the port dimension. `start()` already passes `--port`, so the fix is one
comparison.

### C9. A failed RAPL read is laundered into a phantom 262 kJ in the cost line — **VERIFIED (code); live reachability hypothetical**

`core/energy.rs:62` swallows a failed read into `0`, which
`counter_delta` then treats as a counter wraparound, returning
`max_range - before` — **262 kJ presented as measured joules for one
trial window** on this machine. It flows into `served_j_per_token` →
`cost_report` → the `--meter` measured-cost line with no plausibility
clamp. Dormant only because RAPL is permission-locked here; it arms the
moment the user follows the app's own `chmod a+r` advice. `None` is the
honest value, and the module already uses it correctly elsewhere.

### C10. Hermes deletes unmodeled entries and unrelated top-level keys — **EXECUTED**

Beyond C5's parse-failure wipe: even on a **successful** parse, entries
whose value isn't a plain unsigned integer (a float, a quoted number, a
null) are filtered out, and any unrelated top-level key in the file is
dropped — then the survivors are written back as the whole file.
Executed: `floaty`, `quoted`, `nulled` and an entire unrelated section
all vanished.

Related, also executed: `hermes::register_provider` detects the provider
block by exact line match, so a **flow-style** `custom_providers: [{…}]`
— legal YAML — isn't recognized and a **second** `custom_providers:` key
is appended, making the API-key-bearing config.yaml unparseable
("duplicate entry with key"). The fallback must refuse when the key
exists in a form it can't edit.

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

### H10. Writing `opencode.json` destroys a dotfiles symlink and widens its permissions — **EXECUTED**

`opencode.rs:230` renames over the path itself. Executed: a symlinked
`opencode.json` (stow/chezmoi/yadm — a large fraction of people who hand
-edit `~/.config`) **is replaced by a regular file**; the repo copy keeps
the old content and silently diverges, and the next `chezmoi apply`
clobbers the app's writes. Separately, a 0600 config (they carry cloud
API keys) comes back **0664**, and the backup is created world-readable
too. Note the inconsistency: pi, Hermes and settings use `fs::write`,
which follows symlinks and preserves mode — three connectors, three
behaviors.

### H11. The meter cannot tell "you used nothing" from "I can no longer read your log" — **VERIFIED against the live 820 KB log**

`evidence.rs:262` silently `continue`s every task it fails to parse, so
an empty result has two causes and nothing downstream separates them:
`fmt_report` prints *"no usage recorded in this range"* — a confident,
wrong statement. CLAUDE.md warns about exactly this ("when the meter
reads zero while tokens flow, suspect a new dialect"), and there is **no
drift detector anywhere**.

**The bias is already live**: the real log has 400 `stop processing`
lines but 399 `prompt eval time` lines, because llama.cpp omits that
line when nothing was processed — i.e. exactly the 100%-cache-hit turns
the meter exists to celebrate. The dropped turn was 66,418 tokens, ~7%
of the log's credited prompt tokens. **The headline cache-reuse number
is biased downward.** The fix is one invariant: count `stop processing:`
lines seen versus turns credited.

### H12. Quality scores an HTTP failure as "the model got it wrong" — **VERIFIED**

`quality.rs:270-286` collapses `Err` into `false`. `probe_tool_call`'s
own doc states the contract — *"`Ok(false)` is a real measurement; `Err`
is inconclusive and stored as `None`"* — and `calibrate` honors it. One
connection reset permanently lowers `eval_score` / `tool_reliability` /
`loop_reliability` in `measurements.json`, and those drive the advisor
seat and the fleet brief. Nothing marks them degraded.

### H13. A truncated generation yields a plausible number, not an error — **VERIFIED**

`chat_template_kwargs` appears in exactly one file. `trial::timed_generation`
and `quality::chat` send none, check no `finish_reason`, and floor on no
`predicted_n` (which *is* guarded — but only as the energy divisor). A
model that burns its budget on `reasoning_content` returns empty content
with a valid tokens/sec, so the trial records a real-looking
`tg_rewrite` plus `fidelity: 0.0` — and the fidelity gate then vetoes
that side. **A confident wrong verdict produced by a truncation.**

### H14. `skip_value` reads unbounded bytes from one corrupt GGUF length field — **VERIFIED**

`gguf.rs:225`. `read_string` caps declared lengths at 64 MB; `skip_value`'s
string arm does not, and the `MAX_HEADER_BYTES` guard only runs *after*
it returns. One corrupt byte, or a `.gguf` from a stranger's repo, makes
`read_meta` read the entire 20 GB file before erroring — and
`library::scan` calls it on **every** GGUF at GUI startup. Terminates
honestly, so this is a stall, not a wrong number.

### H15. Backup policy is triplicated and unequal

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
- **The JSONC editor's *removal* path is careful about comments** — it
  correctly skips both `//` and `/* */` when locating a separator. C6 is
  precisely that the *insertion* path never got the same treatment.
- **Fill-don't-overwrite is real and tested in both directions** — a
  probe false-negative can never clobber a hand-set value.
- **All 12 HTTP call sites set a timeout**, ureq's is a total deadline
  covering the body read, non-2xx becomes `Err`, and no TLS feature is
  compiled — so no call site can parse an error page as data or reach
  an `https://` host.
- **`bench::run` refuses to record an empty baseline** — the right
  instinct, and the model for what H11 and H12 should do.
- **Honest `None`s where it counts**: energy reports `cpu_j: None` when
  RAPL is locked, the meter reports `uncovered_generated` rather than a
  zero-dollar estimate, and discover keeps `build: None` distinct from a
  parsed number.

---

## Recommended order

1. **C6 + C7** — the lenient-reader/destructive-writer pair. Both fixes
   are roughly one line (a strict parse before writing; match `NotFound`
   specifically), and together they close the largest gap between this
   app's stated contract and its behavior.
2. **C1, C2** — atomic writes + rescue-don't-zero for measurements and
   trials. Copy the pattern `settings.rs` already has.
3. **C3, H4** — the two `opencode.json` backup bugs. Both are small and
   both destroy the user's last recovery point.
4. **C4, C5, C10** — the connector data-loss set. Fix the behavior *and*
   the test that pins C4.
5. **C8** — compare the port before claiming a router is ours. One
   comparison, and it closes a hole in a house rule.
6. **C9** — `None`, not `0`, for a failed energy read.
7. **H1, H2** — the busy-gate and cancel-token bugs. Both are a few
   lines and both currently mislead the user about what is happening.
8. **H3** — the poller-killing panic. One-line fix.
9. **H11** — the meter's drift detector (count `stop processing:` lines
   versus turns credited). One invariant, and it protects the number the
   whole product is sold on.
10. **H5, H6, H10, H12, H15** — build-lock symmetry, honest heal
    reporting, symlink/mode preservation, inconclusive-not-failed
    quality scoring, one shared backup helper.
11. **H7, H8, H9** — extract the shared sync/pick/config paths into core
    so the two surfaces stop diverging. This is the S3 refactor's real
    payoff.
12. Structural work (S1, S2, S6) as capacity allows; the MEDIUM list
    opportunistically.

## Dev session responses

*(Append here when acting on these, as the usability review does.)*

- 2026-08-31, dev session (v0.6.75). FIXED, each test-pinned:
  **C6** — the comma now anchors to the last property's VALUE via the
  AST, never the last byte, so it can't land inside a trailing comment;
  plus `jsonc::strictly_valid` runs before every opencode.json write and
  refuses BEFORE the backup rotates. Two tests reproduce the original
  corruption (ghost block, inline note) and assert a strict parse.
  **C7** — only `NotFound` is silent; every other read error is loud and
  the rescue fires for unreadable as well as unparseable files.
  **C1/C2** — new `core::safefs`: temp+fsync+rename writes, and reads
  that distinguish Missing from Damaged, with the damaged file moved
  aside so the next write cannot eat it. Applied to measurements,
  trials, config, preset, history, meter cursor, upstream, pi, hermes.
  **C3** — restore writes the good content first, then reuses the slot.
  **C4** — pi removes an entry only when the id has genuinely left the
  preset; a failed load is reported as `kept_unmeasured` and the entry
  is carried through verbatim. The test that pinned the WRONG behavior
  was rewritten. **C5/C10** — hermes edits its cache in place and
  refuses a YAML it cannot parse; unmodeled values and unrelated
  top-level keys survive. Flow-style `custom_providers` is refused
  rather than duplicated. **C8** — ownership now compares the port, so
  a stranger's server on our port is never claimed. **C9** — a failed
  RAPL read is `None`; one missing package makes the whole CPU figure
  None rather than inventing 262 kJ. **H1** — `Msg::Scanned` and
  `Msg::PresetWritten` no longer clear a busy they never claimed.
  **H2** — the cancel token is only replaced when a worker actually
  starts. **H3** — `fnv_bytes` ends the poller-killing panic. **H4** —
  ghost commenting is one read/one write, costing one backup slot.
  **H5** — the auto-build holds a build flag for its duration and the
  GUI refuses to measure beside it. **H6** — heal only claims a restore
  that happened, and keeps the marker when it fails. **H7** — the CLI no
  longer returns before pi/Hermes when opencode.json is absent. **H10**
  — writes preserve mode and follow symlinks. **H11** —
  `evidence::Coverage` compares finished turns against credited ones and
  warns once when the log stops parsing.
  DEFERRED with reasons: H8/H9 (pick_server + config lost-update) and
  the structural items S1-S8 need the core extraction, which is Phase 2
  of the roadmap; M-list items remain open. First integration tests
  landed: `tests/sync_flow.rs`.
