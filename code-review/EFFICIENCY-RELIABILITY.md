# Efficiency & reliability review — work per frame, per tick, per operation

2026-08-31 · v0.6.5 + post-tag commits · independent reviewer, findings
re-verified by the dev session before landing here.

Scott's stated priority: **reliability and stability first**, raw speed
second. This document is ordered that way — unbounded growth and
resource exhaustion outrank micro-optimizations, and things explicitly
not worth doing are named as such so nobody spends a day on them.

## The premise that calibrates everything (verified)

eframe is *reactive*: at true idle the app repaints about every 2
seconds. So "runs 60×/sec" is not the resting state — **but two things
make full frame rate the common state**:

1. `ui.spinner()` requests a repaint every frame by design, and the
   status bar renders one whenever `self.busy` is set
   (`src/ui.rs:5678`). **For the entire duration of every calibration,
   trial, bench, or quality run — minutes to hours — the app repaints
   at vsync rate.**
2. Mouse movement over the window repaints. Reading a pane repaints it.

Net: per-frame costs are real, and they peak **exactly when the machine
is already busy measuring**, competing with the thing being measured.
That inverts the usual "it's only 60fps when idle" excuse, and it is
why the per-frame findings below are ranked as high as they are.

---

## CRITICAL — unbounded growth (will break eventually, silently taxes until then)

### C1. The poller re-reads and re-parses **all of router.log** every 30 seconds

`src/ui.rs:640-648` — `std::fs::read_to_string(&log_path)` on the whole
file, gated only by a 30-second throttle, then a full parse building one
map entry per turn ever logged.

- **Growth term**: bytes since the last router *start*. The log is only
  truncated when the router starts (`core/router.rs:443`), and router
  mode is explicitly designed to be one long-lived server.
- **Measured here**: 829 KB after ~3 hours of one session ≈ 6.5 MB/day.
- **Failure scenario**: a router up for a month ≈ 195 MB. Every 30
  seconds: a 195 MB allocation, ~2M lines parsed, ~100k map entries
  built and thrown away. Long before OOM this shows as periodic
  multi-second stalls and page-cache thrash — the same signature as the
  probe-storm incident this project already ate once.
- **The tell**: the comment at `ui.rs:601` says "throttled to twice a
  minute (the log can be large)". A throttle *divides* an unbounded
  cost; it does not bound it.
- **Fix**: mine incrementally — keep a byte offset plus the accumulated
  tally across ticks, seek, parse only new bytes. The meter's cursor
  already proves this pattern works here, and the 8 KB tail read at
  `ui.rs:611` is the bounded-read model to copy. Cap the log too.

### C2. `meter.jsonl` grows forever, and is fully re-parsed on every crediting tick

`core/meter.rs:206` appends; **there is no prune path anywhere**. Then
`ui.rs:667` calls `summary_line` → `report(&read_all(dir))` — a full
read and JSON parse of the entire ledger — once per crediting tick.

- **Growth term**: days since install × active hours. Rows are appended
  per model per crediting tick (up to every 30s), not one per hour
  bucket, so several rows share an hour.
- **Measured here**: 96 KB / 898 lines in ~4 days ≈ 225 lines/day ≈ 9
  MB/year, forever. At two years the app JSON-parses 18 MB every 30
  seconds while serving.
- **The asymmetry is the tell**: `history.jsonl` got a prune
  (`PRUNE_AT = 2000`, `KEEP_PER_MODEL = 50`, `core/history.rs:53-55`).
  The ledger, which grows *faster*, got none.
- **Fix**: (a) compact on append — roll same-(hour, model) rows together
  instead of appending every 30s; (b) prune like history does, e.g.
  hour-grain for 30 days then fold to day-grain. Separately,
  `summary_line` only ever needs today: read the tail, not the file.

---

## HIGH — real waste, and one probe-storm recurrence

### H1. `/proc` is swept every 2 seconds whenever the router is down

`core/router.rs:410` — `find_preset_process` does `read_dir("/proc")`
plus a read and two String allocations **per PID on the machine**.

- **Free while our router runs** (the `||` short-circuits on a live
  marker — verified). It runs when the router is **down, crashed, or
  the marker is stale** — which is the app's resting state before you
  press Start Router.
- **Measured here**: 568 processes → ~568 file reads + ~1,100
  allocations every 2 seconds ≈ 43,000 full sweeps/day.
- **Why it matters beyond cost**: it degrades as the *rest of the
  machine* gets busier, which is backwards. Same shape as the
  documented `pick_server` incident: an unbounded sweep on a fixed tick.
- **Fix**: the systemd-unit case this covers doesn't change second to
  second. Cache it, re-scan every 30-60s, or only when the port answers
  but our marker doesn't — the one state where the distinction matters.

### H2. Library rebuilds every model's history trail from scratch, every frame

`src/ui.rs:2025-2058`. Full traversal of `self.history` per frame, with
a `format!` allocation per entry *before* the guard that discards it,
plus a `SystemTime::now()` syscall per rendered line, building two
HashMaps that are dropped at end of frame.

- **Verified redundant**: `self.history` is only replaced on
  `Msg::History`, which the poller sends only on mtime change. The
  input is provably stable between frames.
- At the history cap under a spinner: ~120,000 allocations/second to
  redraw text that changes approximately never.

### H3. Connector panes read and parse the user's config files every frame

`ui.rs:3489` (Hermes config, full YAML parse), `ui.rs:3551` (Hermes
cache, second YAML parse + mapping clone), `ui.rs:3610` (pi models.json,
full JSON parse).

- **Measured file sizes here**: 8.0 KB + 1.0 KB + 5.1 KB.
- On the Hermes sub-tab: 2 reads + 2 full YAML parses per frame ≈ 540
  KB/s of YAML parsing at 60 Hz, indefinitely, while you look at it.
- Also a correctness smell: two reads in one frame can straddle an
  external edit, so the pane can render a half-updated view.
- **Fix**: mtime-gate these in the poller exactly like
  `measurements.json` / `trials.json` / `history.jsonl` already are.
  (The sub-tab structure is *right* — it already limits this to the
  selected connector. The leaf functions are the problem.)

### H4. The Lab re-derives nine trial verdicts, with clones, every frame

`ui.rs:2777-2782` → `trial::stored_report` per menu: clones the baseline,
builds a map of cloned variant results, runs the verdict and near-miss
guard math. Then `ui.rs:2839` clones the report again per rendered menu.

- **The Lab is precisely the tab you leave open while a campaign runs** —
  which sets `busy`, which shows the spinner, which pins this at 60 Hz.
  The app re-derives every verdict in the fleet 60×/second while the GPU
  does the actual measurement. That is CPU contention against your own
  numbers.
- **Verified redundant**: `self.trials` only changes on `Msg::Trials`.

### H5. `server_pane` runs an O(models × history entries) comparison every frame

`ui.rs:3135` → `history::build_advisory` → `build_deltas`
(`core/history.rs:149-203`): a full filter-collect per unique model,
then four more passes. ~40,000 iterations + 20 Vec allocations per
frame at the caps.

**H2, H4 and H5 are one refactor**: derive on the mtime-gated message
that already exists, cache the result on `App`. Three findings, one fix.

---

## MEDIUM

- **M1. `rebuild_rows()` runs every 2s regardless of change.** The
  poller sends `Msg::RouterState` unconditionally (`ui.rs:571`), which
  triggers a full rebuild: cloning every `ModelFile`, two HashMaps, and
  O(models × router_models) work. `RouterState` derives `PartialEq` —
  the guard is one line: `if state != last_state`.
- **M2. `self.rows.clone()` per frame** in `library_pane` (`ui.rs:2009`),
  plus `sort_by_key` allocating a fresh lowercase String O(n log n)
  times. `sort_by_cached_key` is a one-word fix for the sort half; the
  clone (a borrow-checker dodge) needs a small refactor.
- **M3. `system::load_config()` every poll tick** (`ui.rs:559`) — reads
  and parses config.json 43,200×/day. Seven lines later the code stats
  that same file for its mtime to gate something else. The guard is
  already there; it just isn't applied to the read beside it.
- **M4. `managed::list_archives()` every 2s while an auto-build is
  queued** (`ui.rs:722`) — and the code's own comment documents that a
  queue can sit for *days*. A directory sweep on a fixed tick, again.
- **M5. `self.advisories` is never trimmed** (`ui.rs:1642`) and every
  entry is re-rendered each frame. Compare `self.activity`, which is
  correctly capped at 200. Cap it the same way.

---

## Explicitly NOT worth doing

Named so nobody spends time here: the three `is_dir()` stats for tab
labels (`ui.rs:3308`); `evidence.rs`'s per-number String allocations
(real, but fixing C1 makes them irrelevant — do not fix independently);
`library.rs:280`'s sort comparator (a worker that runs rarely);
`vram_contention()` being called twice per frame; the O(stats × rows)
cache-stat lookup. All micro; all bounded by model count.

One **reliability note that isn't a cost**: `fetch_models` has a 5s
timeout and `ollama::probe` 2s, while the loop sleeps 2s *after* the
work — so a hung router stretches the real poll period to ~9s. It
cannot freeze the UI (separate thread, both calls have timeouts), but
"2-second poller" is aspirational under trouble. Worth a comment, not a
fix.

---

## What's already well-guarded (verified, not assumed)

The fixes above are mostly *extending patterns this codebase already
has*, which is a good sign:

- **The documented probe rule holds.** `pick_server`, `find_installs`,
  `scan_report` and `list_devices` appear **only inside worker threads**
  — never in a render path. `picked_server()` derives from the cached
  scan with zero subprocesses. No recurrence of the literal bug.
- **mtime-gated reloads** for measurements, trials, and history — the
  exact pattern H3 and M3 should adopt.
- **The 8 KB tail read** for live activity is bounded by construction:
  the model C1 should have followed.
- **`history.jsonl` is pruned** and **the activity log is capped at
  200** — which is why H2/H5 are bounded rather than critical.
- **The meter's cross-process lock** with stale-lock stealing, and the
  cursor fingerprint that avoids double-crediting across upgrades:
  genuinely careful concurrent-state work.
- **Cursor write amplification was already caught and fixed**, with the
  reasoning in the comment.
- **The double-parse of the log was already caught and fixed** — one
  parse now serves both the monitor and the meter. Right instinct; C1
  is just the next step of it.
- **No blocking I/O or network on the UI thread.** Every network call
  and subprocess is on the poller or a worker, and both network calls
  carry explicit timeouts. The module's promise holds.

---

## Recommended order of attack

1. **C1** — incremental log mining. Biggest single reliability win.
2. **C2** — prune/compact the ledger; make `summary_line` read the tail.
3. **H1** — cache or throttle the `/proc` sweep.
4. **H2 + H4 + H5** — one refactor: derive on message, cache on `App`.
5. **H3** — move connector reads into the poller's mtime-gated set.
6. **M1** — the one-line `PartialEq` guard.
7. M2-M5 opportunistically. Skip everything under "not worth doing".

## Dev session responses

*(Append here when acting on these, as the usability review does.)*
