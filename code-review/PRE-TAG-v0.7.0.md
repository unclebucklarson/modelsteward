# Pre-tag review — v0.7.0 (2026-09-11)

A review of the full diff since `v0.6.75` (23 commits, ~3.5k lines of
Rust), run at Scott's request as the middle stage of a
smoke-test → review → tag sequence. Ten finder angles, deduped.

**Fixed before the tag** (commit `c51e4e7`), all but one a regression
introduced since the last release:

| # | Where | What |
|---|---|---|
| 1 | `router.rs` upsert | `content_id` missing from the carry-forward list — a calibrate with warden absent wiped every recorded identity |
| 2 | `router.rs` `failure_reason` | severity column compared `"E"` against a lowercased line, so the branch was dead; then fixed again to prefer the CAUSE (bind failure) over the consequence ("exiting due to HTTP server error") |
| 3 | `bench.rs` | `-d` passed unconditionally, breaking every bench on builds predating the flag |
| 4 | `warden.rs` | `removable` roots walked as shelves — a mounted backup drive would duplicate the whole fleet under `-2` aliases and serve it from USB |
| 5 | `safefs.rs` | `write_atomic` forced 0644 on new files, making first writes world-readable under `umask 077` |

## Open — carried into the backlog

Ordered by severity. None blocks the release; all are real.

- **`quality.rs:383` — a failed probe leaves the model resident.**
  `run_and_record` propagates the new `run_quality` errors with `?`,
  skipping `unload_model` + `wait_until_not_loaded`. A connection reset
  mid-battery leaves 20 GB loaded, and the next bench or calibrate sees
  a contended card — which this codebase treats as a *wrong*
  measurement, not a slow one.
- **`quality.rs:212` — H12 is still live in `agent_loop_shot`.** A
  transport failure during a loop shot is scored as "the model quit
  mid-loop", permanently lowering `loop_reliability` in
  `measurements.json` over a network blip. The evals and tool probes two
  loops above were fixed for exactly this; the loop probe was missed.
- **`bench.rs:317` — bench overwrites calibrate's free-VRAM sample.**
  `free_vram_mib` exists to explain the settled `n_ctx` it was measured
  beside. A later `--bench` stamps its own (post-unload, therefore
  larger) sample onto the same entry, so `--report` prints a context and
  a free-VRAM figure from different runs under a preamble promising they
  belong together. One sample is also reused across every model in a
  multi-model run.
- **`system.rs:434` — `fleet_known_ids` never forgets.** It unions every
  key of `measurements.json`, which is only ever added to. A deleted
  model stays "known" forever, so piagent's removal rule can never fire
  and the dead entry sits in `~/.pi/agent/models.json` indefinitely —
  disabling the model is the only way to evict it.
- **`ui.rs:727` — meter log-head read can silently produce `""`.**
  `read_to_string` over a fixed 8192-byte `Take` errors when a multibyte
  character straddles the cut; `.ok()?` then makes the harvest
  fingerprint the empty string, so a router restart stops being detected
  as a new instance and the CLI `--meter` (which passes the whole log)
  computes a different fingerprint and re-credits the entire log. The
  miner feed immediately above already does the right thing with
  `from_utf8_lossy` over raw bytes.
- **`evidence.rs:264` — `LogMiner` retains every task forever.** The old
  unbounded per-tick CPU was traded for unbounded memory: ~10k turns/day
  leaves ~300k permanent entries in a month, and `results()` re-walks all
  of them every tick. Completed tasks already folded into the cursor can
  never contribute again and could be dropped after crediting.
- **`evidence.rs:274` — `feed` copies the whole log.** The one-shot
  `cache_effectiveness` path runs `format!("{}{}", self.partial, chunk)`
  over a file the module's own comment puts at ~200 MB on a month-old
  router. The previous implementation iterated `lines()` with no copies;
  skipping the concatenation when `partial` is empty restores that.
- **`system.rs:61` — `meter_report_text` parses `router.log` twice**,
  once inside `harvest` and again for `coverage.note()`. `harvest_stats`
  exists precisely so one parse can serve both.
- **`ui.rs:6181` — sync now walks the whole model tree.** `known` used to
  be one small preset read; it is now
  `fleet_known_ids(cfg, &scan_models(..))`, so "Set Up Everything" walks
  every scan dir, blob store and hub cache an extra time — the waste
  review finding F11 removed from this very flow. The scan result is
  already in hand at both call sites.
- **`opencode.rs:246` — the strict-JSON write gate waives itself.** It
  now runs only when the ORIGINAL parses strictly, which disables the C6
  protection for exactly the missing-comma files C6 was about. The
  trailing-comma fix that motivated the change was right; the scope was
  too wide. It should compare damage — refuse when the edit introduces a
  NEW class of error — rather than skip the check wholesale.

Two lower-severity notes: `rows.rs:575` still ranks quant "speed" from
the empty-cache `tg_tps` although `tg_deep_tps` was added to `Row` for
exactly the reason CLAUDE.md gives; and `quality.rs:275`'s doc comment
now contradicts the function it documents.
