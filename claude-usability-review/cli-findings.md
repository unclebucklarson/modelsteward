# CLI usability findings — src/main.rs and friends

Adversarial pass over the CLI surface as a new, non-expert user. The
highest-severity items were independently re-verified against the source by
the coordinating session before publication.

---

## HIGH

### C1. No `--help`, `-h`, or `--version` at all
main.rs:59-180. `--help` falls into the `_` arm: one ~300-char usage line to
**stderr**, exit **2**. A new user's very first command looks like a crash.
The genuinely good documentation (the 40-line module doc at main.rs:1-43,
with menu names, `force`, config paths) is compiled away and unreachable at
runtime. **Fix:** match `--help|-h|help` → print full help to stdout, exit 0;
add `--version` from `CARGO_PKG_VERSION`; keep exit 2 for unknown args only.

### C2. The usage line omits `--meter` and all trial menu names
main.rs:177. Usage shows `--trial <id> [keep <variant>]` with no menu
positional, and `--meter` is absent entirely. Following the usage literally
(`--trial m keep 2048`) races the *spec* menu and fails with
`unknown variant "2048"` — an error that never says a menu name was needed.
**Fix:** generate usage/help from one source of truth; `keep` errors should
echo the valid variant labels for the resolved menu.

### C3. A mistyped trial menu silently runs a *different* experiment
main.rs:222-226: `rest.get(1).filter(|a| trial::menu(a).is_some())
.unwrap_or("spec")`. `--trial m ubatch` (or `kvcache`, or any typo) silently
runs the **spec** menu for 5-20 minutes. The user asked for prefill batch
and got speculative decoding, unwarned. **Fix:** if a second positional
exists, isn't `keep`, and isn't a valid menu → hard error listing the menus.

### C4. Ctrl-C during a CLI trial leaves the trial preset serving
Every CLI path constructs `CancelToken::default()` and there is no SIGINT
handler anywhere in src/. The preset-restore in trial.rs:1588-1592 runs only
on normal return/bail — Ctrl-C mid-trial kills the process and the router
keeps serving the temporary trial config (draft model, altered ub/kv)
indefinitely, silently. cancel.rs's promise ("config restored") is false for
CLI users. **Fix:** SIGINT handler flips a shared CancelToken passed into
bench/trial/quality; exit 130 after restore. (GUI Cancel already does the
right thing — this is CLI-only.)

### C5. `--calibrate` exits 0 when every model failed
main.rs:351-359: per-model `FAILED` lines print, function returns Ok. A
scripted `--calibrate && --sync` reads green on a run that measured nothing.
Same pattern in `--bench`. **Fix:** nonzero exit (e.g. 3 = partial failure)
plus a `N measured, M failed` summary line.

### C6. The diagnosis engine is GUI-only; the CLI prints raw internals
diagnose.rs produces exactly the actionable plain-language text the app
promises ("model format newer than your llama.cpp — rebuild"), but it is
referenced only from ui.rs/advisor.rs. The CLI prints the raw llama.cpp
error (`unknown model architecture 'glm4moe'`, tensor dumps). **Fix:** run
`diagnose::diagnose` on calibrate/bench failures and print explanation +
remedy, phrased for the CLI.

### C7. First-run `--setup` on a machine without `~/models` ends in a
circular error. Default scan_dirs is `~/models` (settings.rs:95-97). Setup
narrates `preset refreshed (0 models)` (stderr) then dies with
`no successful measurements yet — run --calibrate first` — after setup just
ran calibrate. The actual problem (no GGUF found; here's where I looked) is
never stated. **Fix:** empty model set → fail early with the scanned dirs
and how to add one.

### C8. A corrupt config.json silently resets every setting
settings.rs:116-121: `read_to_string(...).ok().and_then(parse.ok())
.unwrap_or_default()` — a trailing comma from a hand-edit silently discards
port, disabled list, **and all kept trial overrides**, with no message on
any stream; the next Save overwrites the file with defaults. (Behavior is
even pinned by a test — worth revisiting the pinned contract.) **Fix:**
distinguish missing (silent defaults) from unparseable (loud stderr warning
with the serde error; refuse to overwrite on next save).

---

## MEDIUM

### C9. `{:?}` Debug dumps in user-facing errors
main.rs:275-278, 331-335, 449, 469-472 (also trial.rs:1466, 1286): a foreign
server yields `state is External { detail: "..." }` — Rust struct syntax at
the user. **Fix:** `Display` impl on RouterState with plain sentences.
(Same defect exists in the GUI: see gui-findings G13.)

### C10. GUI language leaks into CLI output
"Speed column … updated" (main.rs:198), "set one in Settings"
(system.rs:166 — the most likely first-run failure, whose only stated remedy
is a GUI tab), "run a Lab trial" (meter.rs:409), "offered as a button"
(trial.rs:886). An SSH user is told to click things that don't exist.
**Fix:** dual-phrase remedies (config-file path *or* GUI location).

### C11. Trial CLI prints a column glossary for a table it never prints
trial.rs:794-809 via main.rs:260-263: "What the columns mean — novel: …"
with no table emitted; the per-variant numbers live only in scrollback.
**Fix:** print an aligned results table before the glossary (or trim the
glossary in CLI mode).

### C12. Unparsable positionals are silently ignored
`--start 80800` (port out of range) silently uses the configured port;
`--quality m 10x` silently runs 5 shots; `--meter 30d` silently prints
all-time. Three silent no-ops, same pattern (main.rs:303-308, 272, 128-133).
**Fix:** real parse errors naming valid values.

### C13. Positional args mean different things per command
First positional = dirs (`--scan`), port (`--start/--sync/--calibrate`), or
model id (`--bench/--trial/--quality`); `force` is a bare magic word (a
model literally named "force" is unbenchable). **Fix:** named flags
(`--port`, `--model`, `--force`) with positionals kept as aliases.

### C14. `--sync <port>` permanently bakes a one-off port into opencode.json
main.rs:91→388: the "for this run" override rewrites the config's baseURL
to a port that may not exist tomorrow. **Fix:** warn when the sync port
differs from cfg.port.

### C15. `--bench` with nothing stale prints nothing and exits 0
main.rs:197-199: the only summary is inside `if n > 0`. Indistinguishable
from a hang/no-op. **Fix:** an `else` line ("all baselines current — add
`force` to re-run").

### C16. Bare `modelsteward` over SSH dumps a raw winit error
main.rs:60-65: headless → `GUI failed: <EventLoop error>`, exit 1; the user
never learns a CLI exists. **Fix:** on GUI init failure, print "no display —
run `modelsteward --help` for the CLI" + usage.

---

## LOW / MED

### C17. Progress/results stream assignment is inconsistent
`--calibrate` narrates on stderr, results on stdout; `--bench/--trial/
--quality` put progress on stdout. Redirection keeps or loses narration
depending on which command you ran. **Fix:** one rule everywhere (progress →
stderr, results → stdout).

### C18. `--report` writes two files, mentions one
report.rs:285-292 writes findings-report.json AND .md; only the .md path is
printed, so the "review before sharing" warning covers one of two shareable
artifacts. **Fix:** print both paths.

### C19. Trial-menu documentation disagrees across three places
CLAUDE.md says `[spec|ub|kv]` (stale, and omits --advise/--report/--quality/
--verify-rebuild); README:143 and main.rs:24 list ten; usage line lists
none. **Fix:** single source (the new --help), others point at it.

### C20. Menu names and units unexplained at first contact
`kv|dials|ckpt|moe` never glossed anywhere reachable; "settled context
32768" without units; pp/tg unexpanded. The good glossary (trial.rs:794)
appears only *after* a 10-minute run. **Fix:** one-line gloss per menu in
--help; units on output lines.

### C21. No `--config` command; config surface partly unadvertised
`models_max`, `managed_auto_build`, `checkouts`, `cloud_price_per_mtok` have
no CLI accessor and aren't in the README settings paragraph; there's an
undocumented `ignored` serde alias. **Fix:** `--config` prints resolved path
+ effective values (also makes C8 diagnosable).

---

## Dev session responses

- 2026-08-29, dev session: C1-C8 fixed (earlier passes). P3 batch:
  C5 done properly (exit 3 partial, bail on total failure, N measured/M
  failed summary — calibrate AND bench), C6 (diagnose explanations on
  calibrate failures; classified hints on bench failures), C9
  (RouterState Display, all five bail sites), C10 (four GUI-language
  strings dual-phrased), C11 (aligned results table before the
  glossary, tested), C12 (port/shots/meter-range parse errors naming
  valid values), C14 (sync-port warning), C15 (all-current message),
  C16 (headless → points at --help), C17 (progress→stderr,
  results→stdout, documented in --help), C18 (both report paths), C20
  (units on settled context; menu glosses were already in --help), C21
  (--config prints path + effective settings). C13 DEFERRED
  deliberately: every positional is now error-checked, and a named-flag
  redesign is a compatibility break to design on its own, not batch in.
  C19 was closed by the earlier single-source --help.
