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

**The CPU half of the finding is a non-issue** — `results()` costs
**756 µs once every 30 s** at 60,000 retained turns, a 0.0025% duty
cycle. The original write-up framed it as "O(all tasks ever) per tick",
which is true and misleading.

The memory half is real. **Corrected 2026-09-12**: the first figure
(~133 B/turn) was measured in a process whose RSS had already grown, so
it understated. Measuring each size in a fresh process, with only the
miner resident:

| | Bytes per turn | At 60,000 turns (≈1 month heavy) |
|---|---|---|
| Before | **217 B** | 12.7 MB |
| After the model dedup | **124 B** | 7.2 MB |

A 43% reduction, and the original problem was *larger* than first
documented, not smaller.

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

## Done: the model dedup (2026-09-12)

Not in the original option list, and the better first move — it takes
40%+ of the memory with **zero** risk to the ledger, because it changes
nothing about retention.

`Task.model` was redundant. On a spawn line the miner updates
`port_model[port]` and bumps `port_gen[port]` *in the same branch*, so
within one generation a port's model never changes — which makes
`(port, generation)` a unique determinant of the model, and it is
already the task's key. The per-task `Option<String>` plus its heap
allocation was therefore a copy, per turn, of a value shared by every
turn in that generation.

It is now `gen_model: BTreeMap<(port, generation), String>` — **one
entry per spawn**, not per turn. `results()` reads attribution from the
key, keeping the port's final tenant as the documented fallback for
tasks whose lines preceded any spawn line.

Pinned first by two tests that did not exist: one port reused by two
models with the same task id attributes each turn correctly, and a task
seen before any spawn line falls back to the port's tenant. Both were
written and passing against the OLD implementation before it changed,
which is what makes them a check on the refactor rather than a
description of it.

## Options for the remainder

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

**Recommendation:** B next; C only if the bound must be guaranteed.
With the dedup done, the remaining exposure is ~7 MB per month of heavy
use and no CPU cost, so stopping here is also defensible.

## Whatever lands, this test comes first

- Feed the same log **twice**; assert the totals do not move.
- Feed in chunks vs one shot (exists already; must keep passing).
- Feed, prune, then feed a **late line for a pruned task**; assert the
  total is unchanged rather than increased.
