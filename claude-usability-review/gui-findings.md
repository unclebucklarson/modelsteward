# GUI usability findings — src/ui.rs

Reviewed line by line at commit 4742534. IDs are stable for cross-referencing.
Severity is about *user confusion*, not code quality.

First, credit where due (do not "fix" these away): hover-text coverage is
unusually good; the "Start Router & Continue?" interception, the Why? dialog
with remedy buttons, near-miss "your call" buttons, the model-aware time
estimate, and the MoE sequencing warning are all genuinely user-protective.
The problems below are mostly about *where feedback lands*, not what it says.

---

## HIGH

### G1. Errors are invisible: everything lands in the activity log
`Msg::Error` → `self.log("ERROR: …")` (ui.rs:1420 area, `drain_messages`).
The activity log is an 80px monospace strip at the bottom, uncolored — ERROR
lines render identically to progress chatter. A user whose sync/load/trial
fails sees the spinner stop and *nothing else*. Same for `spawn()`'s busy
rejection (ui.rs:680): clicking any action while busy appears to do nothing
unless you happen to be watching the log.

**Fix:** errors deserve a modal or toast (egui has no toast built in, but a
bottom-anchored colored banner that persists until dismissed is ~30 lines).
At minimum: color ERROR lines red in the activity log and auto-expand the
log panel when one arrives. Busy rejection should be prevented, not logged —
see G4.

### G2. Disabled controls never explain themselves
16 `add_enabled` sites, **zero** `on_disabled_hover_text`. In egui,
`.on_hover_text` does not fire on a disabled widget — so every tooltip
attached to a gated control (Load button ui.rs:1873, the In-OpenCode
checkbox ui.rs:1846, ▶ Run selected campaigns ui.rs:2437, every
Apply/Revert/Use button in the Lab, Fleet Brief, Update & Rebuild) shows
*nothing* exactly when the user is wondering why it's gray.

Worst case: during any long campaign, all Lab Apply buttons gray out
(deliberate — worker owns the GPU) with no visible reason.

**Fix:** mechanical sweep — every `add_enabled(cond, …)` gets an
`.on_disabled_hover_text("disabled because …")` naming the condition
("router is down — start it first", "a campaign is running — Apply after it
finishes"). This is probably the highest confusion-per-effort win in the app.

### G3. Library: no search, no sort, no filter
`rows::assemble` does no sorting (grep confirms: no `sort` call in
rows.rs assembly path) — rows appear in scan order. With Ollama blobs + HF
cache + shelf a real machine shows dozens of rows, and the user's only tool
is eyeball-scanning an unordered 15-column grid. There is also no way to
hide the HF-cache noise short of clicking Disable per row.

**Fix:** a filter text box above the grid (substring match on display name)
plus a default sort (e.g. advice-level, then name) would transform the tab.
Column-click sorting is nice-to-have; search is the must-have.

### G4. One global `busy` slot silently serializes the whole app
`spawn()` (ui.rs:680) rejects any second action with a log line (see G1).
During a 25-minute MoE trial the user cannot: save overrides, sync, unload,
archive, pin a binary — and gets no UI feedback about why. A visible-but-
inert UI reads as "the app is broken."

**Fix (layered):** (a) while busy, actually disable action buttons (with
disabled-hover per G2) instead of letting clicks no-op into the log;
(b) consider letting pure-config actions (override save, pin) proceed —
they already tolerate a down router. Full queueing is likely overkill.

### G5. The Library grid outgrows the window; the payoff columns fall off
15 columns ("Model" … "Advice", "Why", disable) in a 1150px default window
inside `ScrollArea::both()` (ui.rs:1739). **Advice — the app's whole value
proposition — is column 13** and lands off-screen; users must discover
horizontal scrolling in a table. egui Grid has no frozen header/columns, so
scrolling down also loses the header row, and 15 unlabeled-at-a-glance
columns is a lot.

**Fix options (judgment call for the user):** (a) master-detail — trim the
grid to ~7 identity/status columns and move Advice/Why/Tune/Archive into a
selection detail panel below; (b) move Advice to column 2 and let metadata
scroll instead; (c) wider default window + narrower columns. Flagging for
discussion rather than prescribing.

### G6. Dialog validation errors appear *behind* the dialog
Override editor: a bad context / bad `key = value` line logs `ERROR: …` to
the activity log and re-opens the dialog (ui.rs:3545-3550) — the user sees
the dialog blink and their Save silently not take. The Settings pane's parse
errors (bad port etc.) land in the same log the Settings pane doesn't show.

**Fix:** hold an `error: Option<String>` on the editor/settings state and
render it in red inside the dialog/pane, next to the Save button.

### G7. Extra-flags syntax rejects what every model card teaches
The override dialog demands `key = value` per line (`fit-target = 2048`),
but every llama.cpp README, model card, and forum post the target user will
copy from writes `--n-cpu-moe 32` / `-ub 2048`. Pasting those fails with
"not `key = value`" — which the user won't even see (G6).

**Fix:** normalize input — strip leading `-`/`--`, accept space *or* `=` as
separator. Ten lines in the parser, removes a guaranteed first-session
papercut. (Test-first: the contract is knowable.)

---

## MEDIUM

### G8. Six user-visible strings render with giant whitespace gaps
Broken string-literal continuations left runs of 10-30 spaces mid-sentence,
visible in hovers and labels:
- ui.rs:2188 ("select relevant" hover)
- ui.rs:2269 (dials hover), 2279 (moe hover), 2287 (vision hover), 2289
- ui.rs:2946 (Connections intro `ui.small`)

**Fix:** trivial — collapse the whitespace (a `\` continuation eats leading
whitespace only when the `\` is present).

### G9. Vocabulary drift across surfaces
The same operation is "Measure" (GUI), "--calibrate" (CLI), "calibration
finished" (worker message). The Lab says "campaigns", the code/CLI say
"trial", results speak of "menus" ("unknown trial menu"). "Preset" vs
"router.ini". None of these are individually fatal; together they force the
user to build a translation table. **Fix:** pick one term per concept
(suggest: Measure, Trial, preset) and sweep messages; keep CLI flag names as
aliases for compatibility.

### G10. Trial-results table headers are unexplained jargon
"novel t/s", "rewrite t/s", "2nd-turn ms", "J/tok", "fidelity", "accepted"
(ui.rs:4571) have **no hover text on the headers** — the one grid where a
novice most needs definitions. The Why? button explains verdicts, not
columns. **Fix:** `ui.strong(h).on_hover_text(…)` with one plain sentence
each ("novel t/s — generation speed writing new code, tokens/second").

### G11. Long operations: cancel is a tiny status-bar button, progress is
prose. A 25-min (or 3-hour, per the live incident) run's only controls are a
small "✖ Cancel" bottom-right and log-line narration. There's no "campaign
3 of 7" indicator and no cancel affordance near the thing the user clicked.
**Fix:** show `(k/n)` campaign progress in the busy label; duplicate a
Cancel button in the Lab next to Run while running.

### G12. Settings tells the user to do follow-up steps instead of offering
them. After Save: "port changed — regenerate preset + sync so opencode.json's
baseURL follows" and "router settings changed — Stop + Start the router to
apply" (ui.rs:3310-3320) are *instructions in a log*. This is the repo's own
hand-fix→feature rule violated in-app. **Fix:** offer buttons ("Apply now:
regen preset + sync + restart router?") or just do it after confirmation.

### G13. Debug formatting leaks into user-facing messages
`setup: port {} is not ours: {other:?}` (ui.rs:4780, setup_flow) prints a
Rust enum debug dump. **Fix:** a Display impl or match with plain words
("another server is already using port 8080 — it looks external; this app
won't touch it. Change the port in Settings or stop that server.").

### G14. First-run empty states are bare
- Zero models: the Library renders a header row over nothing — no "No
  models found. Add a directory in Settings, or install one with Ollama."
- Zero measured models: Connections says so (good), but Library gives no
  "start here" cue toward File → Set Up Everything — the README knows it's
  the first-run action; the UI doesn't say it anywhere.
**Fix:** empty-state labels with the next action, and consider surfacing
"⚡ Set Up Everything" as a visible button on the Library tab when nothing
is measured yet (it already exists on Server/Connections).

### G15. Activity log lacks timestamps and levels
Lines like "scan: 3 installs…" age invisibly; after an overnight session you
can't tell stale from fresh, and ERROR lines (G1) don't stand out. **Fix:**
prefix HH:MM, color by level. Cheap.

### G16. Emoji-dependent UI may render as tofu
Tabs (📚 🖥 ⚡ 🔌 🔧), Feat badges (👁 ⚡ 🧬), buttons (⚙, ✂ in log lines,
▶ ■ ✔). egui bundles only a monochrome emoji subset; 🧬 and 👁 in
particular are worth a visual check on a clean machine. Feat badges are
emoji-*only* columns — if they tofu, the feature disappears. **Fix:** verify
rendering; give badges text fallbacks ("vis/mtp/emb") if any glyph is
missing from egui's font.

### G17. Checkbox side effects: "In OpenCode" can trigger a multi-minute
model load. Hover does explain it, but a checkbox that starts a heavy
GPU operation (and can *fail*, reporting only per G1) defies the "checkbox =
instant state" convention. **Fix (mild):** on click for an unmeasured model,
show the row action inline ("measuring…" in the row's Server cell) or a
small confirm ("This loads the model to measure it first, ~2 min — go?").

### G18. Lab campaign selections and the selected model reset every launch
All `lab_*` booleans are hardcoded defaults in `App::new`. A user tuning one
model across sittings re-picks everything each time. **Fix:** persist the
last Lab selection set in config.json (they're already config-shaped).

---

## LOW

- **G19.** Status bar "VRAM: 5000 / 24000 MiB free" parses as either "5000
  of 24000 free" or "5000 used of 24000". Say "VRAM free: 5000 of 24000 MiB".
- **G20.** README calls the archive button "→ shelf"; the UI renders "to
  shelf" (ui.rs:1902). Trivial doc/UI mismatch.
- **G21.** "Restore opencode.json From Last Backup" (undo for sync) lives in
  Tools, far from the Connections tab where sync happens. A "Restore backup"
  button beside "Sync all measured" would put undo next to do.
- **G22.** The Tuning Guide is a static wall of text with step names that
  reference UI ("Library → Load") but no buttons that jump there. Fine for
  now; linkifying steps would be a polish item.
- **G23.** About dialog shows version + two lines; no link to README/repo,
  no config/state paths. Users hunting "where is my data" would benefit.
- **G24.** No confirm on "Re-measure ALL (force)" / "Re-bench ALL" — both
  multi-minute, single-click. Cancel exists, so LOW, but a "~N models,
  ~M min — proceed?" would match the Lab's estimate culture.
- **G25.** History trails exist only as hovers on ctx/Speed cells —
  excellent feature, near-zero discoverability. A 🕘 affordance or a "hover
  for history" hint in the column header would help.

---

## Dev session responses

- 2026-08-29, dev session: G1/G2/G6/G7/G8/G15 fixed; G4 via G2's
  disabled-reasons; G17 WONTFIX. G5/G3: design talk held — Scott chose
  MASTER-DETAIL; built same day: slim 8-column identity grid (Model/
  Source/Size/Quant/ctx/Speed/Server/OC) with filter box + advice-level
  sort, selection detail panel carrying full advice, quality scores,
  all actions (Load/Tune/shelf/Why?/Disable/In-OpenCode), and a visible
  History section (also resolves G25's discoverability). G10 partially
  addressed (grid header hovers); trial-table headers still pending in
  P3. Tab-semantics: DECIDED by Scott 2026-08-29 — the OpenCode
  mirror STAYS inside Connections; the generalization was deliberate.
  Closed.
- 2026-08-29, dev session (P3 GUI batch): G9 addressed — worker says
  "measuring finished" to match the GUI's Measure; CLI errors say
  "measuring needs…"; the Tuning Guide gains a "Same thing, different
  names" entry mapping Measure/--calibrate, trial/menu/campaign, and
  preset/router.ini (flag names kept as compatible aliases). G10 fixed:
  every trial-table header has a one-sentence hover, plus the ★/✓
  legend on the blank corner. G11 fixed: the Lab worker now reports
  "Lab <id>: <campaign> (k/n)" into the busy label AND the log, and a
  Cancel button sits next to Run while a run is live. G12 fixed:
  Settings no longer instructs — after a router-affecting save it
  offers "Apply now: restart router + regen preset + sync" (one worker
  does all three) with a "later" dismiss. G13 fixed via the new
  RouterState Display (CLI batch). 167 tests green.
