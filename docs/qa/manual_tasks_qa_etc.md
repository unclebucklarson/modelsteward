# Scott's manual task list — QA, validations, and what to record

Written 2026-09-02 at HEAD `70e8a80` (v0.6.75 + 11 commits of post-tag
fixes; 217 unit + 4 integration tests green). This is everything that
is waiting on *you*, in priority order, with exact steps, what you
should see, and what to write down. Pick it up whenever — nothing here
goes stale except the "current state" numbers.

## How to record and report back

- Keep notes in **`docs/qa/notes.md`** (create it; any format — bullet
  points are fine). Date each session. When something surprises you,
  paste the *exact* text you saw (error message, log line, button
  label) — exact text is what turns a report into a fix.
- Screenshots: **save them to disk** under `docs/qa/` (e.g.
  `docs/qa/g15-library.png`). Screenshots pasted into chat are visible
  to me once, but only files on disk can go into the repo/README.
- Terminal output: pipe it — e.g.
  `cargo install modelsteward 2>&1 | tee docs/qa/g15-install.log`.
- When you're back, just say "notes are in docs/qa" and I'll take it
  from there: rough edges become pre-flight rules, numbers get
  verified, bugs get regression tests.

---

## Task 1 — Desktop validations (Phase 0; ~15 minutes total)

These validate code that is already live. Do them on the 24 GB desktop.
**Restart the GUI first** (`cargo run`) so you're on the latest build —
your last running instance predates several fixes.

### 1a. Reasoning → low on the daily driver

Qwen3.8's own chat template defaults to `xhigh` — the most thinking of
any level — which is why it over-thinks.

1. Library → click **Qwen3.8-27B-UD-Q4_K_XL** → **⚙ Tune**.
2. **Reasoning** dropdown → you should see exactly:
   `template default (xhigh)`, `xhigh`, `medium`, `low`,
   `off — no thinking` (the list is read from the model's template;
   `high` is deliberately absent because the template silently rewrites
   it to xhigh).
3. Pick **low** → Save. Takes effect on the model's next load.
4. **Record**: after your next real OpenCode session — does it feel
   like it gets to the answer faster? Any quality drop you notice? One
   or two sentences of subjective read is genuinely useful here; the
   objective version (the `think` trial menu) is designed but not built.

### 1b. pi agent

1. In a terminal: `pi`, then `/model`.
2. **You should see** a `modelsteward (measured llama.cpp)` provider
   with ~18 models, each showing its measured context (e.g. the Q4
   daily driver near 105k, NOT 128k).
3. Pick one, send a trivial prompt, confirm it answers through the
   router (the ModelSteward status bar shows the model loading).
4. **Record**: did the provider appear? did the contexts look like the
   measured numbers from the Library? did a chat work? Any error text
   verbatim.

### 1c. Hermes

1. ModelSteward → Connections → **Hermes** sub-tab → click
   **Register this router with Hermes** (one-time; appends one entry to
   `~/.hermes/config.yaml` with a backup, comments preserved).
2. Restart Hermes, then `/model` inside it.
3. **You should see** a `modelsteward` provider. Note: three of your
   models are deliberately absent — Hermes refuses anything under
   64,000 tokens (gemma-4-31B, ornith-1.5-35b, and the Q5 Qwen3.8).
4. **Record**: did registration succeed (the activity log line)? did
   Hermes still start cleanly? does `/model` list the router models?
   did a chat work?

### 1d. AI advisor on gpt-oss

Needs the GPU free enough to load gpt-oss-20b (models_max=1: this
evicts whatever is loaded — the interrupt dialog will warn you if
something is serving).

1. Tools → **Advisor — opinions + Ask about tuning** → ask anything
   ("which model should I use for long refactors?") or run a fleet
   brief from the Tools menu.
2. **You should see** an actual answer attributed to gpt-oss-20b — not
   "produced no answer". This validates the dual reasoning-kwarg fix
   end to end (the last piece of it that's never been live-tested).
3. **Record**: answered or not; if it answered, was it grounded in your
   real measured numbers; response time roughly.

---

## Task 2 — The G15 laptop (Phase 1.5; one afternoon, highest value)

RTX 3070 Ti mobile, 8 GB. This is the first time the app ever runs on
hardware that isn't the dev box, and the first time on the VRAM class
most real users have. One session here compounds four roadmap items.

### 2a. Prep (before you start, on the laptop)

- Install Rust (`rustup`) and the build deps from the README's Install
  section (the apt line).
- You need at least one model file. Your fleet's big quants won't fit
  in 8 GB — **that's a feature of this test**, the advice column should
  SAY so — but you also want something that runs. Suggestions:
  `ollama pull qwen3.5:4b` (or any ~4-8B model), or copy
  `unsloth/Qwen3.5-4B` (3 GB) from the desktop into `~/models`.
- Optional but valuable: install OpenCode and/or pi there too, so the
  connector story gets tested on machine #2.

### 2b. Install — use the PUBLISHED path

`cargo install modelsteward 2>&1 | tee ~/install.log`

**Honest caveat, decided by you later**: crates.io serves v0.6.75,
which is 11+ commits behind main and contains one known regression
fixed since — if the laptop's `opencode.json` has a trailing comma
anywhere, sync will refuse with "the edit would produce invalid JSON".
If you hit that, it's expected; note it and either hand-remove the
comma or switch to the git build
(`cargo install --git https://github.com/unclebucklarson/modelsteward`).
Testing the published artifact warts-and-all is the point — but if
you'd rather I cut a **v0.7.0 tag first** so the laptop tests current
code, say the word before you start and I'll ship it (the tag decision
is yours per house rules).

**Record**: which install path, how long, any missing-dependency
errors verbatim, and whether `modelsteward --version` prints a sane
build id.

### 2c. Walk docs/GUIDE.md end to end

The guide's "You should see" checkpoints are the test script. Walk it
in order: Settings → (Build Advisor if no llama.cpp — this laptop
likely has none, so the **managed clone + build path gets its first
foreign-machine test**, including backend auto-detection on a mobile
GPU) → Set Up Everything → Library → a small Lab run → Connections →
meter.

**The one recording rule that matters**: every time you are unsure
what to do next, or the app's message doesn't match what you see,
write down (1) where you were, (2) what you expected, (3) what
happened — verbatim. **That list IS the pre-flight rule set**; the
pre-flight feature is designed and waiting on exactly this input.
Nothing is too small: "I didn't know which tab to go to" is a finding.

### 2d. 8 GB-specific things to capture

- The Library's **advice column** for models that don't fit: does it
  say something honest and actionable, or something confusing?
- Measured context for your small model on 8 GB (`--fit`'s answer),
  and pp/tg from a Bench — these become the second row of every
  "measured on this machine" claim, and showcase material.
- The **GPU persistence prompt** (fires on first launch if the mode is
  off): does the pkexec button work on this machine's driver setup?
- If Hermes/pi are installed there: the 64k floor will likely exclude
  *everything* on 8 GB — what does the app tell you? Is it clear or
  alarming?
- Whether laptop thermals/power profile make measurements visibly
  noisy (run the same Bench twice; note the two numbers).

---

## Task 2.5 — Re-bench the fleet once (NEW 2026-09-02, desktop, ~30-45 min unattended)

Why: benchmarks now also measure generation **at a realistic KV depth**,
not just from an empty cache. Every stored baseline predates that, so
every Speed number in your Library is currently the optimistic
empty-cache figure. One re-bench replaces them with the honest pair.

**Preconditions the app now enforces** (it will refuse and tell you):
nothing else may hold the GPU. So: stop OpenCode, and if Ollama has a
model resident run `ollama stop <model>` (or wait out its keep-alive).

1. `modelsteward --bench force 2>&1 | tee docs/qa/rebench.log`
   (or Server → Bench in the GUI). It takes longer than before —
   the deep pass has to prefill 32k tokens per model.
2. **Record**: the log. I specifically want, per model, the pair
   `tg X t/s (empty cache), tg Y t/s at N depth`. The gap between them
   is the thing modellab measured at 24% on the 27B; I want to know
   whether that holds across quant sizes and MoE models on your card.
3. **Also record**: did it refuse when something held the GPU? Was the
   message clear enough to act on?

If the deep pass makes benching unbearably slow on the G15's 8 GB
(likely — 32k of prefill on a mobile 3070 Ti is not fast), say so; the
rung ladder can be made configurable.

## Task 3 — Finish the Minecraft showcase

`docs/showcase/minecraft-clone.md` is live with day-one receipts and
marked in progress. To close it I need three things from you:

1. The clone's **repo link** (publish it wherever you like).
2. A **screenshot of the game running**, saved to
   `docs/showcase/minecraft.png`.
3. The word "done" — I'll then freeze the final meter numbers into the
   entry and link it from the README.

## Task 4 — The README screenshot (D17, two minutes)

Save one good capture of the **Library tab** (grid + a selected model's
detail panel) to `docs/screenshot-lab.png`. That's the last thing the
README needs before it stops being 200 lines of imageless prose. Any
resolution; I'll wire it in.

## Task 5 — Desktop Lab hygiene (optional, when the GPU is idle)

- Open the **Lab** and check each model's standing recommendation
  blocks: any ★ (green, "waiting for your Apply") winners you never
  applied — the 80B's `ncpu-moe-32` and GLM's `cpu-moe` were pending
  at last check. Apply what you want, then **re-run Measure** on those
  models so the synced limits match the applied config.
- GLM's quality probe was killed mid-run weeks ago and never re-run —
  if you apply its placement winner, run the Quality campaign on it
  once.

## Task 6 — Decisions parked for you (no action, just answers when ready)

1. **Tag v0.7.0 now or after the G15 walk?** Main has 11+ commits of
   real fixes past the published 0.6.75, including one shipped
   regression fix. My recommendation: tag BEFORE the laptop test so
   the published artifact is what gets validated — but it's your call.
2. **After the walk**: green-light the pre-flight build once your
   rough-edge list exists.
3. **Someday**: whether the `think` trial menu is worth its new
   instrument (your 1a subjective read feeds this).

---

## Current state, for orientation when you return

- HEAD `70e8a80`, main, all pushed; 217 unit + 4 integration tests,
  zero warnings. Published: v0.6.75 on crates.io + GH (behind main).
- The code-review ledger is **empty of open findings** except: H1
  (/proc sweep while router down), H2–H5 (per-frame caching; only
  bites while mousing over the app), H8/H9 (await the Phase-2 core
  extraction), M2–M5 efficiency smalls, and the cross-process build
  lock (documented accepted risk).
- Low-impact mode is live and validated: the GUI no longer perturbs
  inference (0 GPU-driver queries during a turn; ~94% of the
  GPU-exclusive ceiling with the app open).
- What I can do next without you: Phase-2 foundations (Connector
  trait, thinking/latency instrument), H1/H2–H5, Help → First Run.
  Say "keep going" if you want any of that done while you're busy.
