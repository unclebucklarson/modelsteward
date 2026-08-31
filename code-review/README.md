# Code review — 2026-08-31

Three independent reviewers plus a dev-session verification pass, run at
Scott's request ("brutal and honest") after a stretch of opportunistic
feature work.

- **[CODE-REVIEW.md](CODE-REVIEW.md)** — correctness, safety, structure.
  Five CRITICAL data-loss findings, ten HIGH.
- **[EFFICIENCY-RELIABILITY.md](EFFICIENCY-RELIABILITY.md)** — work per
  frame, per tick, per operation. Two CRITICAL unbounded-growth
  findings, five HIGH.

## How these were produced

Four reviewers worked in parallel on separate dimensions (correctness &
concurrency; efficiency & resources; architecture & test coverage;
robustness & external input), each instructed to cite `file:line`, to
supply a concrete failure scenario for every claim, and to say
explicitly whether a finding was verified or remained a hypothesis.
Every CRITICAL and HIGH finding was then re-checked against the source
by the dev session before landing in these documents; measured numbers
(log growth, process counts, file sizes) come from Scott's live machine.

## Coordination protocol

Same as `claude-usability-review/`: when acting on a finding, append to
the **Dev session responses** section at the bottom of the relevant
document — what was fixed, what was deferred with reasoning, and what
turned out to be wrong. That record is why the usability review's
deferrals (C13, D16) are still defensible weeks later.

## Reading order

Both documents end with a recommended order of attack. The short
version: fix the five data-loss findings first (they are cheap and they
are the ones that lose a user's work), then the two unbounded-growth
findings, then the shared-code extraction that stops the two surfaces
from diverging.
