# C6 — LogMiner retention

Design note for the last open finding from the v0.7.0 pre-tag review.
Readable version, with the diagrams:
<https://claude.ai/code/artifact/47efaed0-a8b5-4ac4-bc56-5ffe73c90197>

## The finding

`LogMiner` folds `router.log` into tasks keyed by
`(port, spawn generation, task id)`, one per agent turn, each assembled
from three lines that arrive at different times (prompt eval, eval,
release). A task only counts once all three are present. **Nothing is
ever removed from that map.**

## The invariant that makes a fix delicate

`results()` must return the **cumulative** totals since the router
started. The meter credits `totals − cursor.credited` into an
append-only ledger. Totals that shrink stop crediting; a figure counted
twice over-reports tokens and cost *permanently*, in the number users
are most likely to trust.

## Measured cost (this machine, release build, synthetic logs)

| Turns retained | Log | Memory held | `results()` per 30 s tick |
|---|---|---|---|
| 2,000 | 727 KB | 288 KB | 29 µs |
| 20,000 | 7.3 MB | 2.7 MB | 245 µs |
| 60,000 | 22 MB | ~8 MB | 756 µs |

**The CPU half of the finding is a non-issue** — 756 µs every 30 s at a
month of heavy use is a 0.0025% duty cycle. The original write-up
framed it as "O(all tasks ever) per tick", which is true and misleading.

**The memory half is real but slow**: ~133 bytes per turn, ~8 MB/month
of heavy use. Worth fixing; not worth risking the ledger for.

## Why "credit and drop on release" is a trap

The miner deliberately revises fields after a task first looks complete
(`t.generated = t.generated.max(n)`; `t.release` overwrites). Credit and
delete on release, and a later line for that task id recreates it via
`or_default()`, completes it again, and credits it again — silently and
permanently.

**Measured against the live log** (4,004 lines, 148 turns): 0 keys saw
any event after their release, 0 were released twice. Encouraging, but
one model on one port over half an hour is a sample, not a proof —
speculative decoding, multimodal turns, parallel slots and
cancellations are all unrepresented.

## Options

- **A — credit and drop on release.** NOT recommended: rests entirely on
  the terminal-release assumption, and its failure is invisible and
  permanent.
- **B — retire whole generations** (recommended). When a new
  `spawning server instance` line appears for a port, the previous
  generation's tasks are finished by definition; fold and drop them.
  Safe by construction. Frees nothing while one model stays loaded for
  weeks, which is the common case on this machine.
- **C — B plus a cap** (~10,000) evicting the oldest *credited* tasks,
  so memory is bounded regardless of generation lifetime. Re-opens A's
  hazard only for tasks 10,000 turns old.

**Recommendation:** B now; C only if the bound must be guaranteed.
Doing nothing is also defensible — 8 MB/month with no CPU cost.

## Whatever lands, this test comes first

- Feed the same log **twice**; assert the totals do not move.
- Feed in chunks vs one shot (exists already; must keep passing).
- Feed, prune, then feed a **late line for a pruned task**; assert the
  total is unchanged rather than increased.
