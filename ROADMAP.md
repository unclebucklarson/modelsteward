# Roadmap

The tracking document for modelsteward. PLAN.md holds the founding
design; this file holds what's done, what's next, and the idea backlog.
North star for every entry: **maximum llama.cpp + OpenCode performance
without requiring expertise — measured, not guessed.**

## Done

- **M0** — plan, spikes against llama-server b10216 (router mode, `--fit`,
  hot reload, Ollama blobs as direct preset paths). `docs/spikes.md`.
- **M1** — headless core: discover (installs + devices), gguf reader,
  model library (shelf + Ollama store). `--scan`.
- **M2** — router lifecycle: preset generation (np=1, q8_0 KV defaults),
  start/stop/status/reload with strict ownership (marker + cmdline).
- **M3** — calibration (per-model measured `--fit` context) and
  opencode.json sync through the comment-preserving JSONC editor.
- **M4** — egui GUI: menu bar, Library/Server/OpenCode panes, activity
  log, status bar; all slow work off-thread.
- **M5** — Ollama peer awareness + VRAM contention warning, orphan
  comment-out UI, systemd user unit with preset-based ownership, README.

## M5.5 — Usability pass (DONE)

1. ✔ **Settings pane + persisted config** (`config.json`): scan dirs, router
   port, llama-server binary override, Ollama port.
2. ✔ **Incremental + stale-aware calibration**: measurements fingerprinted
   by env (build + devices) and effective child args (volatile `--port`
   excluded); fresh ones skipped; `force` re-measures.
3. ✔ **Remember failed loads**: failures persisted with reason, shown in a
   Library "Health" column with a hover tooltip; excluded from sync.
4. ✔ **One-click setup**: File → Set Up Everything (also on the Server and
   OpenCode panes; CLI `--setup`): start if down → measure stale/missing →
   sync, fully narrated.

Found and fixed during this pass: calibration loads now go through explicit
`POST /models/load` + status polling instead of a long-blocking `/props`
autoload (cold-cache 20GB loads take minutes and tripped HTTP timeouts /
router kill paths), and each measurement waits for the previous model's
teardown before loading.

## M5.7 — Information-architecture redesign (DONE, per user direction)

The tabs now match how a user thinks:

- **Library** = every model from every source — user scan dirs, Ollama
  store, HuggingFace hub cache (`~/.cache/huggingface/hub`, mmproj
  companions excluded) — one row each with hardware-aware advice
  ("too large for this machine", "bigger than VRAM — will spill to CPU",
  "measured: N context", failure reason + likely fix), live server
  status, Load/Unload, and an **In OpenCode** checkbox.
- **Loading IS measuring**: Load button and checkbox both go through
  load → read settled ctx → record → auto-sync into opencode.json.
  The batch calibrate remains as optional pre-warming.
- **Server** = router controls + detail on the currently loaded model
  (measured ctx, source, in-OpenCode, device table) + router log tail +
  Ollama peer section.
- **OpenCode** = the actual opencode.json entries with their values and
  per-entry status (✔ synced / ⟳ differs / ✖ can't load with hint /
  ? never measured) and Remove (comment-out).
- Failure reasons now mined from router.log into measurements.

## M5.6 — Test, refine, then deepen

5. ✔ **Measured tool-calling** (DONE): calibration probes each loaded model
   with a one-shot tools request (strict validation: right function name,
   arguments parse as JSON; truncated calls don't count). New config
   entries get the measured verdict; existing entries only have the key
   filled when absent — hand-edits are never overwritten. Advice column
   reflects it. Live: all 11 loadable models pass. Bonus fix: GUI now
   auto-reloads measurements.json on change.
6. ✔ **Per-model override editor** (DONE): ⚙ per Library row — ctx pin,
   KV type, extra flags; stored in config.json (survives regeneration),
   works for preset aliases AND router cache ids (bare sections), preset
   regenerated + router hot-reloaded on save. Device pinning fields join
   when GPU #2 exists.
7. ~~Model downloads~~ → moved to the modelwarden sibling project
   (acquisition is storage-side; see boundary contract).
8. ✔ **Log viewer + Tools menu** (DONE): router.log tail on the Server tab;
   Tools menu opens preset/config/opencode.json/log and restores the last
   opencode.json backup (swap semantics).

## Smaller items (fold in opportunistically)

- **Router start options** (user request 2026-08-17): user prefers manual
  start for now — Start Router button already covers it. Down the road:
  a Settings toggle "start router when the app opens", and a first-run
  hint on the Server tab explaining the three ways to start (button,
  `--start`/`--setup` CLI, systemd unit).

- ✔ **Cache-source models are calibrated/synced AND listed** (DONE): the
  Library shows router-only cache entries as their own rows (nothing
  servable is invisible), warns that loading one downloads from
  HuggingFace, and points at an on-disk twin when one exists.
- ✔ GUI reloads measurements.json on disk change (DONE). Still open:
  display-name + alias shown consistently across panes.
- ✔ **Migrate measurements on archive** (DONE 2026-08-24 night):
  archive-to-shelf carries the measurement to the new alias (fingerprints
  cleared → re-measures next calibrate; bench numbers keep their build
  stamp) and removes the old cache-id entry so the leftover row stops
  claiming it. The mmproj companion travels too (resolved blob linked,
  never the relative symlink).

- ✔ Numbered backups (5) + Tools → Restore From Last Backup (DONE).
- ✔ Measured-ctx variance policy: DECIDED (user) — sync writes measured
  minus 5%, floored to a 256 multiple (`opencode::safety_context`).
- ✔ Status-bar VRAM live on the 2s poll (DONE).
- ✔ Contention warning names the remedy (DONE).
- `limit.output` exposed in the override editor (crude ctx/2 cap today).
- Library staleness badge driven by fingerprint mismatch.
- ✔ **Measurement history journal** (DONE 2026-08-25, user decision after
  the b10630 diff nearly required transcript archaeology): every ctx
  measurement and bench result also appends to history.jsonl (build,
  args_fp, values; newest 50 per model kept), surfaced as hover trails
  on the Measured ctx and Speed cells. Foundation for build-over-build
  advisories ("this build cost you 9% ctx"). Current-truth files
  untouched — history is a side effect of writing, never a source.

## M6 — Build Advisor + Diagnosis ("Why?" panel)

**Diagnosis** (user idea, agreed 2026-08-16): every unavailable/failed
model gets a "Why?" button opening a plain-language panel: what happened,
the evidence (exact log line, pre-extracted — never "see logs"), and what
to do next as actionable buttons (Archive to shelf / Rebuild llama.cpp /
Free VRAM / Re-measure). Shares its brain with the Build Advisor: the
same rules that map an error to a cause map a cause to the fix. Rules
first; the local-model Advisor explains the weird ones later.

## M6 — Build Advisor

**Phase 1 DONE (probe + rules + Diagnosis):** `core/diagnose.rs` (error →
cause → plain language + remedy buttons; "Why?" on every non-green Library
row, log-mining for old "failed(1)" records) and `core/advisor.rs`
(git checkout state incl. source-vs-binary split, upstream distance via
fetch, compute capability, toolchain, locked-model list; verdict cards in
outcome language; ff-only + arch-pinned rebuild commands; streaming
rebuild runner). GUI: Server → Check My llama.cpp; CLI: `--advise`.
First live run found: checkout already pulled to b10448 but binary still
b10216 — rebuild alone unlocks 3 models.

**Multi-backend (DONE):** probe detects CUDA (nvcc + compute cap), Vulkan
(glslc + runtime), ROCm (hipcc + gfx target); Build Advisor window has
per-backend checkboxes with detection-based defaults; all backends passed
to cmake explicitly ON/OFF so stale caches can't drift a build; verdicts
call out near-misses (GPU present, toolchain missing → suggest Vulkan).

**Phase 2 — verification loop DONE (2026-08-25):** the guided rebuild
now chains straight into stop → start (onto the new binary) → measure
stale → sync → a measured report: unlocked ✓ / still locked (with the
Ollama-only reframe) / ⚠ REGRESSION / context shifts (new builds move
VRAM use — b10630 cost ~9% fleet-wide, limits followed honestly). CLI
`--verify-rebuild` covers out-of-band rebuilds. Found live: stop's 5s
wait raced llama-server's own wind-down (SIGTERM landed after the
error) — now 30s with an honest late-exit message.

**Phase 2 remaining:**
- The `Advisor` AI layer (default backend: the local model this app
  serves) for build-log diagnosis and tradeoff explanations — picks from
  the rules engine's flag allowlist, never invents flags.

## Connections (server-manager pivot) — phase 1 DONE 2026-08-17

User direction: this is a llama.cpp server manager for ANY local-AI app
(note-taking app next), not OpenCode-only. OpenCode tab → **Connections**:
OpenCode stays the first-class synced connector; new generic panel gives
any OpenAI-compatible app the base URL, measured model list, and copy-
paste snippets (curl / Python SDK / models-JSON with safe limits).
`models_max` setting already landed for multi-app residency. Remaining:
per-app connectors when a second app has a real config file to sync;
multi-app VRAM residency guidance.

## M6.5 — Feature-aware serving (discussed 2026-08-17)

Load every model with the options its features deserve. Detection is
deterministic (GGUF metadata + file siblings + load-log mining), enablement
is rules in the preset generator, benefit is measured. NOT per-model-type
llama.cpp builds — architecture support is source version, not cmake flags;
the Build Advisor's "stay current" already owns that lever. (Small truth
kept: a few global perf cmake knobs, e.g. extended FA kernel coverage for
quantized KV, belong in the Advisor's advanced section someday.)

1. ✔ **Detect & show** (DONE 2026-08-17): gguf reader scans tensor names
   → `has_mtp`; scan pairs mmproj siblings per directory (all sources);
   embedding archs recognized. Library "Feat" column: 👁 vision, ⚡ MTP,
   🧬 embed with explanatory hovers. Live: 4 MTP models, 3 vision, one
   both (Qwen3.5-4B). Still open: Advisor card "features your build
   ignores".
2. ✔ **Safe auto-enablement** (DONE, first slice): preset gains `mmproj =`
   for shelf vision models (hub-cache vision models: the router pairs
   mmproj itself) and `embedding = true` for embed archs — feature keys
   applied after user overrides so overrides can't strip them. Embedding
   models are excluded from OpenCode chat sync and skip the tool probe
   (recorded tool_call=false). ✔ OpenCode image modality (DONE 2026-08-22):
   sync writes `modalities.input ["text","image"]` for models the preset
   actually serves with `mmproj =` (vision_ids_in_preset — served truth,
   not on-disk truth), fill-not-overwrite like tool_call. Open:
   archive-to-shelf should carry the mmproj companion along (user's
   Qwen3.8 shelf dir was missing its projector — fixed by hand via
   hardlink from the hub snapshot's blob; NOTE the snapshot entries are
   relative symlinks, so archive must link the resolved blob, not the
   symlink).
3. **Measured trials** (merges into M7): draft-model pairing for
   speculative decoding (same-family small model, e.g. Qwen3.5-4B drafting
   for the 27Bs), MTP self-speculation when upstream lands the flag —
   each as try → measured verdict → keep/discard.

## Optimization tiers (discussed with user, 2026-08-17 night session)

**Tier A — shipped as defaults:** `cache-reuse = 256` (mid-prompt KV reuse:
the biggest agent prefill win — edits in the middle of a resent prompt no
longer reprocess everything after them) and `cache-ram = 24576` in the
preset `[*]`; Build Advisor now flags GPU persistence mode off (driver
re-init latency on every load; `sudo nvidia-smi -pm 1`).

**Tier B — measured-trial menu (feeds M7):** ✔ speculative decoding
TRIED (spike 5, 2026-08-24): classic 4B draft REJECTED on 24GB (context
collapse + 6x slower), **ngram-simple adopted** on the daily driver (zero
VRAM, +121% on rewrite work; `spec-type = ngram-simple` override) — the
M7 harness should trial ngram variants per model and a classic draft
again when a second GPU exists. Still to trial: `-ub` physical batch
512→1024/2048 (prefill speed vs a small VRAM/context cost); asymmetric
KV quant (ctk q8_0 + ctv q4_0) for more context, quality-checked;
**MoE-aware offload** (`--cpu-moe` / `-ot 'exps=CPU'`) — advise
automatically when a detected MoE model exceeds VRAM.

**Tier C — niche/delightful:** slot persistence (`--slot-save-path` +
save/restore API) so a router restart doesn't cost a full agent-session
reprocess; LAN serving (`--host 0.0.0.0` + `--api-key`) as Connections
phase 2 with a proper security warning.

Deliberately rejected: `--context-shift` (silent truncation is
anti-honesty; OpenCode compacts properly), sampling tuning (client's job),
blanket `--mlock`.

## M7 — Performance lab

**Phase 1 DONE (2026-08-20) — baselines:** `core/bench.rs` runs llama-bench
(pp512 + tg128 ×3, at the model's serving KV types incl. overrides), parsed
from `-o json`; results stored in measurements.json (`pp_tps`/`tg_tps` +
`bench_build` as the staleness signal — re-calibration under unchanged
fingerprints preserves them via `upsert_measurement`). CLI `--bench [id]
[force]`: sweeps every measured, non-embedding model missing a current-build
baseline; unloads our router's models first (never touches a foreign
server). Library gains a Speed column (pp/tg t/s). First live numbers:
Qwen3.5-4B Q4_K_XL — pp 6034 t/s, tg 163 t/s (build 10454).

**GUI Bench action (DONE 2026-08-22):** Server → "Bench New/Stale Models
(speed)" / "Re-bench ALL (force)", sharing `bench::run_baselines` with the
CLI; narrated in the activity log, Speed column live-updates per model via
the measurements watcher. GPU is only freed once there's real work to do
(a bad id or nothing-to-bench never unloads anything).

**Phase 2 COMPLETE** — A/B-with-verdict became the trial harness; the
speculative-decoding trial ran (classic draft measured and REJECTED on
this hardware, ngram adopted instead — spike 5); results-next-to-
recommendations became the Lab's standing recommendation blocks.

## Where things stand (2026-08-27, 122 tests green — v0.2.0)

Since the 08-26 stamp: quality gate v2 + quant-choice advisor (live,
with measured quality parity on the qwen3.8 quants); cache-effectiveness
monitor (found vision silently disabling cache-reuse on both daily
drivers) + the ⚙ vision toggle; lower-is-better trial goals + load-mode
menu (verdict: keep baseline — warm loads are 4s and dio measured 2x
slower) + speculation-dials menu (built, first run pending); Lab detail
scrollbar; Cancel for long-running work; findings report physical-GPU
truth + JSON sidecar; ghost auto-comment; published as modelsteward on
GitHub + crates.io with tag-triggered releases.

### (2026-08-26 stamp)

Since the 08-25 stamp: M6p2 verification loop DONE (`--verify-rebuild` +
auto-chained after guided rebuilds; live verdict: b10630 unlocked
nothing, confirmed 4 Ollama-only conversions, cost ~9% ctx fleet-wide —
honestly synced); measurement history journal (history.jsonl + hover
trails); measure flows refresh the preset first (new-on-disk models are
measurable); Start-Router-&-Continue prompt replaces dead-end errors;
trial verdicts explain themselves (Why? — deterministic, no AI); the
⚡ Lab tab landed with five campaigns (Measure/Bench/spec/ub/kv),
standing recommendations with Apply/Revert (mid-campaign popup removed),
and the quality gate (rewrite fidelity, multiset-scored) unlocking the
KV-precision menu. Findings-report export (tier 1 sharing) in flight.

Older stamps below kept for history.

### (2026-08-25 stamp)

M7 phase 2 core landed and campaigned (ngram-simple kept on 4 models;
north-mini also kept ub-2048 via the tradeoff flow), the wild-readiness
pass closed the babysitter gaps, archive migrates measurements + carries
mmproj, action columns got headers, daily upstream freshness landed.

M7 is real: baselines swept (10 models, build 10454), GUI Bench action
landed, and the first Tier B trial ran end to end (spike 5) — ngram-simple
speculative decoding ADOPTED on the daily driver, classic 4B draft
REJECTED on single-24GB hardware. Also landed since the 08-20 marker:
HF-download timeout fix, ghost-alias cleanup (opencode.json `-2` entry),
vision serving for the Qwen3.8 shelf models (mmproj re-linked + measured
ctx correction) and OpenCode image-modality sync.

**M7 phase 2 core DONE (2026-08-24 evening):** `core/trial.rs` +
`--trial <id> [keep <variant>]` + per-row 🧪 with verdict dialog (Keep
persists via the override path; trials.json watched like measurements).
Fleet campaign ran: ngram-simple KEPT on qwen3.8-q4/q5, laguna,
north-mini (+29% to +118% rewrite); ornith + qwen3.6 keep baseline
(under the 10% bar). See spike 5 addendum for the numbers and the
acceptance-is-a-bad-proxy finding.

Next, in rough order:

**Wild-readiness pass (DONE 2026-08-25, user-directed):** every manual
Claude intervention became a feature — (1) verdict dialog v2 shows the
full measured table and surfaces guard-rejected tradeoffs as explicit
"Keep X anyway" choices with gains/costs in plain language (north-mini's
+50%-prefill case would have died silently); (2) contention-aware
measuring: calibrate skips-without-recording and trials abort cleanly
when another session's model holds the server (router::loaded_other
evidence), and diagnose classifies legacy 500/limit records as
ServerBusy, not model faults; (3) a trial keep carries its measured
settled-ctx into measurements (stale-marked) and re-syncs the OpenCode
limit immediately. Standing rule adopted: every hand-fix prompts
"should the app do this itself?".

1. **M7 phase 2 remainder** — CLOSED into M8: ctv q4_0 became the kv
   menu (quality-gated), the menu picker became the Lab's campaign
   checkboxes, results-next-to-recommendations became the Lab's standing
   blocks. Still open in M8: `--cpu-moe`, remaining ngram variants.
2. ✔ **Ghost comment-out by the app** (DONE 2026-08-26, user-approved
   after the same leftover confused twice): sync auto-comments an orphan
   only when the REACHABLE router omits its id AND no measurement backs
   it — measured-but-unoffered entries stay reported-only. Comment-outs
   with note + backup, announced in the sync summary (✂). First live
   catch: the unsloth Q5 cache entry whose blob/preset had evaporated.
3. **M6 phase 2**: post-rebuild verification loop; local-AI advisor
   layer. ✔ AI advisor MVP (DONE 2026-08-27, design talk held): core/
   aiadvisor.rs — `Advisor` trait, RouterAdvisor backend (the models the
   app already serves; offline, private). Settled scope: AI is NEVER
   load-bearing — one-shot grounded generations only, no chat, output
   labeled "opinions, not measurements" naming the answering model,
   nothing auto-applied, nothing leaves the machine. First feature:
   "Ask a served model" on Why? dialogs whose failure the rules
   couldn't classify (Cause::Unknown) — prompt carries the stored
   error, machine facts, and the child server's log tail (mined via
   evidence::child_port); answers collect in the Advisor window
   (Tools menu). NEXT advisor features (user-ranked): fleet brief
   (synthesize the findings JSON), rebuild triage (read upstream
   commits against YOUR model set); later backends: Ollama, OpenAI-
   compatible URL (cloud = explicit opt-in).
   ✔ Daily upstream freshness (DONE 2026-08-25, user request —
   manual checks left the checkout 167 commits stale): the status poller
   runs one quiet `git fetch` per day (remote-tracking refs only, never
   the working tree), persists the stamp to upstream.json, shows
   freshness at the top of the Server tab, and logs when a newer build
   appears.
4. Connections phase 2; Tier C (slot persistence, LAN serving).

## M8 — Peak performance program (mapped 2026-08-25 with user; "tweak
them to their absolute maximum")

**✔ The quality gate — DONE (2026-08-26):** every trial round scores
rewrite fidelity (multiset line-match against the rewrite prompt's known
answer — set-matching scored a half-dropped module 0.7, the unit test
caught it); the verdict disqualifies >5-point fidelity drops outright,
near-misses spell quality costs in capitals, table + Why? teach the
column. **✔ Quality gate v2 (DONE 2026-08-27):** six-item fixed eval
battery with strict machine-checked answers (last-line / structural-JSON
matchers, self-consistency tested) + N-shot tool reliability; Lab's
sixth campaign and `--quality <id> [shots]`; scores persist to
measurements + journal. Then:

1. **Asymmetric KV quant trial** (`ctv q4_0`): MENU BUILT (kv, Context
   goal — ≥10% more settled ctx with speed and fidelity held). First
   fleet run pending.
2. ✔ **Quant-choice advisor (DONE 2026-08-27)**: rows of the same model
   family (display minus quant token) with ≥2 measured quants get advice
   on the non-preferred rows — fastest tg crowned ("prefer the Q4_K_XL:
   +10% speed, +88% context"), quality respected: parity stated when
   measured, absence said out loud, and a QUALITY VETO (faster quant
   >1 eval item worse) turns the crown into a stated tradeoff.
3. **Speculation leftovers**: ngram-map-k / ngram-cache campaign (cheap);
   MTP self-speculation when upstream lands the flag (⚡ models draft for
   themselves — could win on NOVEL code, which ngram can't); investigate
   the DeepSeek-V4-Flash BF16 "won't load standalone" blob — it lives in
   a dspark/ dir and may be a draft-dspark aux artifact, not a broken
   model (diagnose would learn a new category).
4. **Latency-between-generations**: ✔ foundation DONE 2026-08-27 (the
   harness handles lower-is-better goals via improvement ratios; trial
   rounds time load-request→loaded) and ✔ the load-mode menu landed
   (`load`: dio/mlock vs auto, Lab campaign + CLI — first run pending).
   Slot persistence groundwork ✔ DONE 2026-08-27 (user decision:
   best-effort now, full workflow later): the preset's `[*]` sets
   `slot-save-path` (save/restore API live fleet-wide, zero cost unused),
   and the `slots` Lab campaign / `--trial <id> slots` measures the
   ceiling — save → swap → restore → edited turn vs a cold swap-back,
   reported as a standing Lab line (no Apply; it's a workflow, not a
   knob). BACKLOG — snapshot/resume workflow, revisit when the measured
   ceiling justifies it: the user today picks ONE middle-of-the-road
   model precisely because swap-back costs a full reprocess; cheap
   restore would unlock specialist-model-per-task. Design constraints
   recorded: the app is NOT in the router's eviction path (llama-server
   evicts on the incoming request), so automation means either a manual
   "snapshot session" affordance or upstream save-on-evict — watch
   llama.cpp for the latter. Snapshot files scale with conversation
   length (GBs); restore speed is disk-bound. Still open:
   `models_max = 2` topology advice.
5. **Build advisory**: ✔ DONE 2026-08-27 — `history::build_advisory`
   compares each model's newest numbers on the current build vs the
   build before it (context only under identical args fingerprints —
   config changes between builds would confound it; generation from
   llama-bench baselines, config-free). Surfaced as the "Rebuild
   scorecard" line on the Server tab (warn-colored when a model lost
   ≥5% context) and atop the findings report's history section. Still
   open: pinning the best-measured build (waits on the
   app-managed-checkout decision).
6. **Speculation-dial trials** (2026-08-26 knob review): the adopted
   ngram modes have untouched dials (`--spec-draft-p-min`, draft-length)
   — sweep them as a trial menu on top of the kept winners; the likeliest
   source of further free tokens/sec.
7. **FA-engaged check** (Advisor): PARKED 2026-08-27 with evidence —
   the premise fails: router-mode child servers filter llama internals
   below warning level out of router.log (verified on a 1.1MB live log:
   zero `flash_attn`/`llama_context` lines; only srv/slot/cmn I-lines
   survive). Mining can't see what isn't logged, and raising child
   verbosity fleet-wide would flood the log for a check whose expected
   answer is "fine" (`-fa auto` chooses correctly upstream). Revisit
   only if llama-server grows an FA field in `/props` or a slot
   endpoint.
8. **cpu-moe trial folds in thread count**: ✔ DONE (t8/t24 sub-variants;
   the 80B proved t8 — P-cores only — and t24 cost 45%). v2 landed
   2026-08-27: partial offload via `--n-cpu-moe 40/32/24` — each layer's
   experts pulled back to the GPU is generation speed reclaimed; a step
   that over-commits VRAM fails its round honestly and the table says so.
9. **⚙ field promotion**: ✔ DONE 2026-08-27 — spec-type and
   ubatch-size have their own fields in the override dialog with this
   model's measured optimum beside them (trial winner, "baseline wins
   here", or a pointer at the Lab); temp/top-k/top-p sit in a compact
   sampling-defaults row (config only, never a trial target — agents
   send their own). Promoted keys typed into the free-form box are
   rejected with a pointer at their field; storage is unchanged
   (still ov.extra), so trial keeps interoperate. Later: cache-type-v
   once the kv menu's winner deserves its own field.
10. **cache-reuse sweep**: ✔ DONE 2026-08-27 — the `cache` Lab campaign
    races `--cache-reuse 0/1024` against the shipped 256, judged by the
    new agent-turn probe (a big prompt re-sent with a MIDDLE edit;
    second-turn prefill ms + %-served-from-cache recorded as the
    `2nd-turn ms` column). Same probe powers the `vision` campaign,
    which PRICES the multimodal cache tax the evidence miner found
    (text-only vs with-projector, fair baseline forces vision back on).

**Evaluated and REJECTED (2026-08-26 knob review, with reasoning — so
they don't resurface):**
- *top-k / top-p as performance knobs*: sampling shapes output, not
  speed, and agent clients (OpenCode) override server defaults per
  request — a server-side setting would be silently ignored. Standing
  decision since 2026-08-17: sampling is the client's job. (Defaults for
  non-agent Connections clients: see item 9.)
- *manual n_gpu_layers on single GPU*: `-ngl auto` + `--fit` already
  optimize placement upstream — founding rule: don't reimplement
  llama.cpp's placement math. Becomes real with multi-GPU hardware;
  for over-VRAM models the right lever is cpu-moe, not ngl.
- *standalone n_threads*: zero effect while models run fully on GPU
  (see item 8 for when it matters).
- *standalone n_batch*: `-ub` dominates; `-b` combos may join the ub
  menu for completeness, expected marginal.
- *FlashAttention as a lever*: auto already picks it; only verification
  is useful (item 7).
- *a separate Tuning tab*: ⚙ is the manual-knob surface, the Lab is the
  measured-tuning surface; a third tab would duplicate ⚙ and split the
  apply path. Knobs get promoted into ⚙, evidence stays in the Lab.

**✔ UI home — "⚡ Lab" tab (DONE 2026-08-26):** model picker + five
campaign checkboxes (Measure/Bench/spec/ub/kv) queued on one narrated
worker; results table, standing per-menu recommendations with
Apply/Revert/"anyway" + Why?, currently-applied display, history trail.
Library slimmed back to everything+advice (Trial column removed);
mid-campaign verdict popup removed — the Lab is the one verdict surface,
and apply buttons disable while anything runs.

**Findings sharing (2026-08-26 discussion):**
- ✔ **Tier 1 — findings-report export (DONE 2026-08-26)**: sanitized
  markdown + JSON sidecar (same inputs, sanitized once at the source)
  written locally for user review; Tools menu + `--report`; verify loop
  flags report-worthy regressions. User-caught fix: the Machine section
  now reports PHYSICAL GPUs (backend views deduped by name, conservative
  min figure, iGPU shared-RAM heaps labeled and never counted) — the raw
  device list showed ~96GB on a 24GB machine. Same model now feeds the
  Library advice VRAM figure (discover::advice_vram_mib), fixing a
  latent bug where a Vulkan-only box would have taken the iGPU's phantom
  heap as its VRAM.
- **Tier 2 (future)**: automated submission IF a community dataset home
  ever exists — the tier-1 format is deliberately the ingestible
  primitive. Not ours to host (server/moderation/abuse = its own
  project).
- **Tier 3 — the standing line**: any outbound data is opt-in, reviewed
  before sending, manually triggered. This app sends nothing anywhere by
  default (today: only your router + one daily git fetch).
- ✔ **Cache-effectiveness monitor (DONE 2026-08-27)**: core/evidence.rs
  mines router.log (grammar learned empirically) for per-model prompt
  reuse across real sessions; Server tab shows turns + reuse % and warns
  when llama.cpp disabled cache-reuse for vision-served models — the
  miner's FIRST live report: both qwen3.8 daily drivers reprocess full
  prompts every turn because mmproj disables reuse. FA-engaged check
  deferred: no FA evidence exists at this log verbosity.

## Sibling project: modelwarden (`~/src2/modelwarden`)

Inventory / backup / archival for model files lives in its own project —
see its HANDOFF.md for mission, boundary contract, and harvest map.
Boundary: **warden owns storage truth; this app owns serving + OpenCode.**
Consequence here: roadmap item 7 (HF downloads) moves to modelwarden —
acquisition is storage-side.

## Parked / ideas

- **App-managed llama.cpp checkout** (user idea 2026-08-25): the app
  creates and owns a default checkout (e.g. `~/.local/share/
  modelsteward/llama.cpp`), telling the user hands-off — making
  ff-pulls, rebuilds, and freshness checks always safe (no dirty-state
  or diverged-branch surprises). Tradeoffs to settle before building:
  disk cost of a second checkout when `~/src/llama.cpp` already exists;
  the observe-don't-touch principle (an app-OWNED repo is exempt, like
  the router we start); how the Settings binary picker presents
  "managed" vs "mine"; and whether the Build Advisor's guided rebuild
  then defaults to the managed copy. (The verification loop is built;
  it would simply target the managed checkout if this lands.)

- **Serve unoffered HF-hub variants**: a hub file whose variant the router's
  cache index doesn't list (e.g. a second quant of the same repo) currently
  shows "not offered". It could be made servable by writing a preset entry
  pointing at its snapshot path — needs care to keep one identity per file.
- ✔ **Archive to shelf** (user idea, DONE): per-row "→ shelf" button pulls
  a cache/Ollama model into the user's models dir (hardlink when same
  filesystem — instant, zero extra disk; temp-named copy otherwise),
  regenerates the preset, hot-reloads the router, rescans. Scan dedupes
  by inode so the shelf copy replaces the cache row. Solves both the
  "unoffered variant" quirk and "at the mercy of Ollama/HF" pruning.

- Multi-GPU override UI (device pinning, per-device fit targets) — core
  is list-based already; UI lands when a second GPU exists to test with.
- OpenCode deeper integration (default model selection, agent presets).
- Prompt-cache tuning surfaced (`--cache-ram`, context checkpoints) with
  measured effect on agent-turn latency.
