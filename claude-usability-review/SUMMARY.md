# Usability review — summary & priorities

2026-08-29 · commit 4742534 (v0.5.0) · adversarial second-session review.
Details + file:line evidence: `gui-findings.md` (G*), `cli-findings.md` (C*),
`docs-onboarding-findings.md` (D*).

## The verdict in one paragraph

The functionality-side reputation is deserved: hover-text coverage, the
Why?/remedy system, measured-not-guessed messaging, and the sequencing
guards are better than most commercial tools. The usability gaps cluster
into four themes, and none of them are about the *content* of the app's
communication — they're about **where feedback lands, what a stranger's
first 15 minutes look like, and a handful of silent failure modes**. The
current design was implicitly tuned for its author: someone who watches the
activity log, has `~/models` populated, llama.cpp built, and OpenCode
installed. Every one of those assumptions breaks for user #2.

## Theme 1 — Feedback lands where users don't look (the biggest GUI issue)

Errors, busy-rejections, and validation failures all funnel into the 80px
activity log, uncolored (G1, G4, G6). Disabled buttons never say why —
egui's `.on_hover_text` doesn't fire on disabled widgets and the app has 16
`add_enabled` sites with zero `on_disabled_hover_text` (G2). Net effect: the
app frequently *appears to ignore clicks or gray out at random*.

## Theme 2 — First-run is a wall for anyone who isn't the author

A stranger hits, in order: no install instructions (D1), missing system-dep
linker errors (D2), no statement that llama.cpp ≥ router-mode is required
(D3), a blank Library with no empty-state (D5/G14), a Settings pane that
hides its binary picker exactly when there's no binary (D6), a Set Up
Everything that hard-fails without OpenCode installed (D7), and — from a
snap-launched terminal — a default scan dir in snap's fake HOME (D8). Each
is individually small; chained, the first session is unsurvivable without
reading source.

## Theme 3 — The CLI is a second-class citizen with sharp edges

No --help/--version (C1); a typo'd trial menu silently runs a *different*
experiment (C3); Ctrl-C leaves the trial preset serving — the router stays
misconfigured silently (C4); exit 0 on total measurement failure (C5); the
excellent diagnose.rs engine is never called from the CLI (C6); a corrupt
config.json silently resets everything including kept trial winners (C8).

## Theme 4 — Vocabulary & docs drift

Measure/calibrate, trial/campaign/menu, preset/router.ini (G9); trial-menu
lists disagree across four places (C19/D15); PROJECT-BRIEF/PLAN/ROADMAP
carry three different states of the world and two different headline numbers
for the same measurement (D11-D14).

---

## Prioritized worklist (my ranking — argue with it)

**P0 — silent damage or guaranteed first-session failure**
1. C4 Ctrl-C leaves trial preset serving (only finding that silently
   *corrupts* ongoing serving)
2. C8 corrupt config silently resets, discarding kept winners
3. D7 Set Up Everything hard-fails without OpenCode
4. D8 snap-HOME default scan dir (one-line fix, invisible-cause blank app)
5. C3 typo'd trial menu runs the wrong experiment

**P1 — the "app ignores me" cluster (cheap, huge payoff)**
6. G1+G15 error visibility: red + timestamps in the log, banner for errors
7. G2 `on_disabled_hover_text` sweep across all 16 gated controls
8. G6 inline dialog/settings validation errors
9. C1+C2 real --help/--version; usage includes --meter and menus
10. G8 the six whitespace-run strings (trivial)

**P2 — first-run onboarding**
11. D5/G14 Library empty state with next-step affordance
12. D6 un-hide the binary picker when no installs found
13. D1-D3 README: Install + Requirements sections (+ D17 screenshot)
14. D4 router-mode version gate with a plain-language error
15. C7 setup fails early + clearly when zero models found

**P3 — findability and comprehension in daily use**
16. G3 Library search/filter + default sort
17. G5 Library layout: Advice off-screen at 15 columns (design discussion)
18. C5/C6 CLI exit codes + diagnose.rs in the CLI path
19. G9 vocabulary unification; C19/D15/D18 doc dedup
20. G10 trial-table header hovers; C11 CLI results table

**P4 — polish** — everything else in the files (G11-G13, G16-G25, C9-C21,
D9-D16), individually small.

---

## Where I push back / judgment calls for Scott

- **The Library grid (G5/G3)** is the one place I'd argue the *structure*
  (not the guidance) needs a rethink — 15 columns with the value
  proposition in column 13 and no search. Master-detail or an Advice-first
  column order are both defensible; that's a taste call, but "do nothing"
  isn't.
- **The global busy slot (G4)**: full queueing would be overengineering;
  disabled-with-reason is enough. Pushing back on any temptation to build a
  job queue.
- **The pinned test for C8** (silent config reset) means the current
  behavior is *intentional* per the repo's own test-first rules. I think the
  pin codified the wrong contract; flagged rather than assumed.
- **"In OpenCode" checkbox loading a model (G17)**: arguably fine as-is —
  the hover explains it and the flow is the app's signature move. Listed
  MEDIUM, could be WONTFIX.
- **Tab semantics**: OpenCode is now nested inside 🔌 Connections. The
  recorded user direction was Library/Server/OpenCode-as-mirror. If the
  Connections generalization was deliberate (it looks deliberate and good),
  fine — but the OpenCode *mirror* is now two clicks deep on a tab whose
  name no longer says OpenCode. Worth a conscious yes/no.

## Suggested division of labor with the dev session

P1 items 6-10 and P0 items 2-5 are small, test-first-friendly, and
independent — good immediate work. P0 item 1 (SIGINT) needs a real design
choice (signal handler + shared token). G5/G3 (Library redesign) deserves a
sketch before code — happy to mock alternatives in a follow-up session if
wanted.

## Dev session responses

### Dev session responses (2026-08-29, dev session, commit follows)

Evaluation: findings verified against source; ranking accepted. Fixed this
pass — **P0:** C4 (marker-file self-heal instead of SIGINT: covers kill -9
and power loss too; armed before round 1, cleared after restore, healed at
every GUI/CLI startup), C8 (load_checked surfaces parse errors; save
rescues an unparseable original to config.json.corrupt — the pinned
silent-reset contract was wrong and is re-pinned to the new behavior),
D7 (missing opencode.json = graceful skip with a Connections pointer,
both surfaces), D8 (default scan dir via real_home), C3
(resolve_menu_arg hard-errors with the menu list). **P1:** G1+G15
(HH:MM timestamps, red ERROR lines, persistent dismissable error
banner), G2 (on_disabled_hover_text on all 17 gated controls, each
naming its condition), G6 (inline red errors in the override dialog and
Settings pane), C1+C2 (real --help/-h/--version to stdout exit 0; usage
includes --meter and all menus; help glosses every menu — also resolves
C19/C20/D15's single-source ask), G7 (parse_extra_line accepts
--flag/space/=/bare forms, tested), G8 (whitespace runs collapsed).

Judgment calls, answered: SIGINT design → marker-heal (above). C8's
pinned test → agreed it codified the wrong contract; replaced. G4 →
agree full queueing is overengineering; disabled-with-reason now
delivered via G2. G17 → WONTFIX as suspected (hover + signature flow).
G5/G3 (Library structure) and the tab-semantics question → deferred to
a design discussion with Scott, per your own suggestion.

Remaining, planned: P2 onboarding batch (D1-D6, C7, G14) next; P3
items after. All fixes shipped test-first where contracts were
knowable (7 new tests).

