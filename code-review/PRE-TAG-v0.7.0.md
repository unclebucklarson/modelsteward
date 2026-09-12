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

## Also fixed before the tag — Group A (commit follows)

Scott reviewed the ten open findings and took the four that write wrong
data into durable state:

| # | Where | What |
|---|---|---|
| A1 | `quality.rs` | a failed battery returned past the unload, leaving 20 GB resident so the next measurement was contended |
| A2 | `quality.rs` | transport failure and model failure both came back as `Err(String)`, so a network blip permanently lowered `loop_reliability`. Now separated BY TYPE: outer `Result` = unreachable (abort, as the eval loop already did), inner = the model's behaviour (score) |
| A3 | `bench.rs` | `--bench` stamped its own post-unload VRAM sample over calibrate's, so `--report` printed a context and a condition from different runs |
| A5 | `ui.rs` → `system::read_head` | a multibyte character straddling the 8 KB window emptied the meter fingerprint, letting `--meter` re-credit a log it had already counted |

## Open — carried into the backlog

Ordered by severity. None blocks the release; all are real.

**Group A is resolved** (before the tag), **C7/C8/C9** after it as safe
filler, and **B4/B10** after that with Scott's design input. What remains
is **C6 alone** — see `docs/design/c6-logminer-retention.md`.

**B4** — `fleet_known_ids` now answers a present-tense question. It no
longer unions measurement keys (a historical record, only ever added to,
which is why no removal could ever fire); it takes the router's own
offered list, and returns `None` when the router is down so nothing is
removed at all. Scott's insight supplied the missing half: warden owns
existence, and its inventory distinguishes "on an unplugged drive" from
"gone". Step 1's content identity is what lets the two views be joined.

**B10** — the strict-write gate keeps its narrow "only if we made it
worse" rule, and OpenCode's own parser is now a second layer with the
same rule. Two live discoveries shaped it: `opencode debug config`
NORMALISES what it reads (it writes `$schema` into the file), so it can
never be pointed at the user's real config — it runs against throwaway
copies; and it reports SCHEMA violations as well as syntax errors, so the
verdict is decided by comparing outcomes for the original and the
candidate, never by reading the message.

- **`evidence.rs:264` — `LogMiner` retains every task forever.** The old
  unbounded per-tick CPU was traded for unbounded memory: ~10k turns/day
  leaves ~300k permanent entries in a month, and `results()` re-walks all
  of them every tick. Completed tasks already folded into the cursor can
  never contribute again and could be dropped after crediting.
Two lower-severity notes: `rows.rs:575` still ranks quant "speed" from
the empty-cache `tg_tps` although `tg_deep_tps` was added to `Row` for
exactly the reason CLAUDE.md gives; and `quality.rs:275`'s doc comment
now contradicts the function it documents.
