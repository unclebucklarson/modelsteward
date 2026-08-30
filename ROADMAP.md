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

## Work queue — next up (ordered; 2026-08-29)

1. **BENCHED BUG (top priority when work resumes): advisories fail on
   gpt-oss** — "produced no answer (reasoning channel)". Diagnosed:
   RouterAdvisor sends only `enable_thinking:false` (Qwen-family
   kwarg); gpt-oss's Harmony template listens to `reasoning_effort`
   instead (both exist in llama.cpp's chat layer — verified in
   common/chat.cpp), so the earned advisor reasons at default effort
   and exhausts its 2,500-token budget before answering. Fix: send
   BOTH kwargs (`reasoning_effort:"low"` + `enable_thinking:false`),
   consider a reasoning-content fallback + a larger budget for the
   49k-token brief. The new error banner surfaced this exactly as
   designed.
2. ✔ Library master-detail BUILT 2026-08-29 (Scott chose it from three
   mocked options): slim 8-column grid + filter + advice sort;
   selection detail panel with full advice, quality scores, every
   action, and visible history (G25 resolved). Tab-semantics DECIDED
   2026-08-29: OpenCode mirror stays inside Connections (Scott's yes).
3. ✔ Usability P3 batch DONE 2026-08-29 (two commits): CLI exit codes
   + diagnosis + parse errors + --config; Lab (k/n) progress + inline
   Cancel; trial-header hovers; Settings "Apply now" button; vocabulary
   note in the Tuning Guide. C13 (named flags) deliberately deferred.
   Was: Usability P3 batch: CLI diagnose + exit codes, vocabulary
   unification, trial-header hovers, Settings offers actions not
   instructions, G11 progress/cancel placement.
4. ✔ QA/guides/work-left discussion HELD 2026-08-29 (The Steward's
   Map brief). Scott's decisions: tab semantics = Connections keeps the
   mirror; QA = release checklist + pre-tag review now, CLI harness
   later (docs/RELEASE-CHECKLIST.md); docs/GUIDE.md written (Scott
   working through it); next tag = v0.5.1 (tagged). Lab gained a
   where-settings-live note (user question → feature, per the rule).
5. gpt-oss advisor reasoning_effort fix — NOW the top dev item (see 1).
6. README screenshot (D17) once one is saved to disk.
7. Help → First Run (D10) — small, next polish pass.

## Where things stand (2026-08-29, 165 tests green — v0.5.0 shipped; M9 closed; advisor finished; usability P0-P2 fixed)

v0.4.0 shipped after an 8-angle pre-tag review (13 findings fixed).
Since the tag: the llama.cpp management intent-split (Settings selects,
Build Advisor builds — Pin/Unpin vocabulary retired), managed autonomy
(opt-in auto-build of new releases, idle-gated so it never compiles
beside a measurement; short-circuits when the newest release is already
archived), split-GGUF support (first shard = the model, sizes summed),
and M9 phase 1 — the token meter (continuous router.log harvest into
meter.jsonl, `--meter [today|24h|7d]`, Server-tab line, cloud-compare
against an editable price) which immediately caught the b10630→b10672
log-grammar drift that had silently broken the cache monitor.

The MoE reliability arc closed with numbers: the agent-loop probe
(3 multi-hop tool-executor drives per quality run) scored the 80B at
100% loops / 100% tools / 83% evals, and the moe-v2 rematch found the
sweet spot — ncpu-moe-32: 52 t/s generation (+30% over cpu-moe),
303 t/s prefill (+49%), full 262,144 context; ncpu-moe-24 finds the
VRAM wall honestly. The verdict machinery learned its final guard
lesson: an agent-unusable baseline (ctx < 24,576) no longer sets the
price floor for the Context goal — fidelity still never waives. The
live-session pain was quantified as cold-prefill UX (~30s per 10k
tokens), not unreliability; MoE documentation claims stand on measured
ground.

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
3. **Speculation leftovers**: ✔ ngram-map-k + ngram-cache joined the
   spec menu 2026-08-27 (names verified against common/speculative.cpp;
   the full model-free ngram family now races — re-run spec campaigns to
   rank them). ✔ DeepSeek dspark blob RESOLVED: it IS a draft-dspark
   speculator sidecar (Markov head, confirmed in llama.cpp source);
   diagnose learned Cause::DraftSidecar — name/path-matched (the display
   alone doesn't say dspark; the path does), "expected and harmless"
   wording. Still open: MTP self-speculation when upstream lands the
   flag (⚡ models draft for themselves — could win on NOVEL code, which
   ngram can't).
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
   knob). NIGHT-MEASURED CEILING (2026-08-27/28, b10630): ~1.0x — a
   solid negative. Snapshots round-trip perfectly (2637 tokens, 237MiB,
   47ms restore) but post-restore requests get cache_n = 0 on the OAI
   endpoint: tested on the SWA daily driver (checkpoint wall) AND on
   north-mini (id_slot-pinned too) — this build's restore rehydrates
   bytes the prefix matcher never consults. The probe now measures the
   CONTINUATION scenario (unedited turn; an edited turn tests the
   checkpoint story, not the slot story). Watch upstream: when restore
   feeds prefix matching (or preserves SWA checkpoints), the campaign
   will light up on its own — the instrument is in place. BACKLOG —
   snapshot/resume workflow, revisit when the measured ceiling
   justifies it: the user today picks ONE middle-of-the-road
   model precisely because swap-back costs a full reprocess; cheap
   restore would unlock specialist-model-per-task. Design constraints
   recorded: the app is NOT in the router's eviction path (llama-server
   evicts on the incoming request), so automation means either a manual
   "snapshot session" affordance or upstream save-on-evict — watch
   llama.cpp for the latter. Snapshot files scale with conversation
   length (GBs); restore speed is disk-bound. ✔ `models_max = 2`
   topology advice DONE 2026-08-27: evidence::topology_advice — fires
   only from real usage (two models with ≥3 logged turns each whose
   files fit together under 70% of VRAM), names the pair and the price
   (residents split the fitted context); Server tab + a tradeoff hover
   on the Settings control.
5. **Build advisory**: ✔ DONE 2026-08-27 — `history::build_advisory`
   compares each model's newest numbers on the current build vs the
   build before it (context only under identical args fingerprints —
   config changes between builds would confound it; generation from
   llama-bench baselines, config-free). Surfaced as the "Rebuild
   scorecard" line on the Server tab (warn-colored when a model lost
   ≥5% context) and atop the findings report's history section. Still
   open: pinning the best-measured build — DECIDED 2026-08-27, revised
   same day after user pushback: **option A, app-managed checkout, with
   binary archiving as its internal mechanism.** Rationale: (1) the
   installs picker already handles multi-checkout clarity — the managed
   clone is one more detected install, clearly labeled, never forced
   (user's point, conceded); (2) the published-app audience is the
   decider — a crates.io user with no ~/src/llama.cpp has nothing for
   option B to archive; the managed checkout is what bootstraps
   llama.cpp for non-experts at all (north star verbatim); (3) closed
   loop: freshness → fetch → build candidate → verify → scorecard →
   pin/rollback, no git knowledge required. Design rules: the clone
   lives in the app's data dir; the USER'S checkout is never touched;
   every VERIFIED build's binaries are archived so rollback never
   rebuilds; checkout management is 100% deterministic rules — the AI
   advisor may triage WHEN to build (rebuild triage), never HOW
   (settled: AI is not load-bearing). ✔ BUILT 2026-08-27:
   core/managed.rs (clone/fetch-tags/checkout-bNNNN/build via the
   advisor's engine — run_steps/build_commands factored out —
   archive_build to builds/bN/); managed bin + archives join installs
   discovery; Build Advisor gained the Managed llama.cpp section
   (set-up/update button, archive list with Pin/Unpin — pin writes
   server_bin, takes effect on next router start, never restarts a
   running server) and the rebuild-triage button ("What's in this
   update for me?" — commits between builds + the user's model set →
   labeled advisory; found live: the daily probe's fetch lacked --tags
   so release names lagged — fixed, with a HEAD..origin/master
   fallback). First live clone+build pending user GPU idle time.
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
  **REVISED by the night investigation (2026-08-27/28)**: vision was
  NOT the binding constraint on the daily driver — its SWA/hybrid
  attention makes the KV cache non-shiftable, so llama.cpp disables
  cache-reuse AND ctx-shift regardless ("not supported by this
  context"); mid-edit turns resume from context checkpoints whose
  defaults sit 8192 tokens apart. Measured: exact resend 2626/2630
  cached (123ms); mid-edit 42 cached (1954ms full reprocess);
  --checkpoint-min-step 128 → 2114 cached, 467ms. Consequences all
  shipped: probe instrument reads timings.cache_n (the server's own
  accounting — the old prompt_n ratio read 0% where truth was 1.6%);
  miner distinguishes the two disable reasons; warning text points at
  the real lever; NEW `ckpt` campaign races checkpoint-min-step
  1024/256/256-x64 under the agent-turn goal. The [*] cache-reuse=256
  default stays (a no-op on SWA models, real on shiftable ones).

## Code-review debt (two 8-angle reviews, 2026-08-28; all correctness
findings fixed in their respective passes — deliberately deferred
consolidations only):
- Auto-build gate policy lives in the GUI poller; extract to core
  (managed::auto_build_tick) if a headless trigger path ever exists.
- BuildCheck models "no repo backs this binary" only in message text;
  a repo-absence reason field would let every surface say the true
  cause (offline vs checkout-less).
- Pending auto-build queue is in-memory: app exit drops it until the
  next daily check (harmless, documented).
Older items:
- One shared advisory-ask helper (3 copies in ui.rs workers).
- One pure install-pick policy fn shared by system::pick_server and
  App::picked_server (aligned today, still two copies).
- Per-frame caching for build_advisory / topology_advice /
  stored_report in the Server and Lab panes (correct, just re-computed
  per repaint).
- newest_known_build on advisor::run + parse_build_tag.
- topology_advice does scalar-VRAM fit math in core (CLAUDE.md
  tension, accepted for advisory-only use; becomes real work when
  multi-GPU hardware exists to test against).

## Backlog — discussed 2026-08-28 (user ideas + pushback, logged)

- **✔ Build-corner design session (2026-08-30, all four BUILT same
  day):** Scott's nitpicks, decided via mocked options. (1) Analyzing
  selector now offers every DISCOVERED install's checkout (not just
  active+managed+added) and a line names the resolved checkout —
  "Update & Rebuild builds there" (the active-binary-is-an-archive →
  managed-checkout resolution was silently confusing). (2) THE STARVED
  AUTO-BUILD, root-caused: the idle gate counted a LOADED model as
  busy, and a daily-driver machine is never model-free — b10697 sat
  queued behind the 24h Minecraft session. New gate (Scott: "quiet for
  10 min"): no generation activity for 600s + no app operation
  (busy_flag) + no trial marker (any process) + no model mid-
  load/download; loaded-but-quiet builds. (3) Retention (Scott: keep
  5, configurable): archives_keep config (0=all), auto-prune after
  each successful build (serving + custom-labeled archives never
  pruned, tested), per-archive ✖ delete button. (4) Settings binary
  list and the archive list scroll.

- **Showcase program (user idea 2026-08-30, scaffolded same day):**
  docs/showcase/ holds measured case studies of real projects built on
  modelsteward-served models — aggregates only (meter/trials/quality),
  never transcripts; artifacts live in their own linked repos; case-
  study framing with caveats stated. First entry: the Minecraft clone
  (Qwen3.8-27B, trial-crowned ngram-simple n8/m64, day-one receipts
  captured: 257 turns, 856k prompt/116k generated, ~30% cache, $0.03
  measured vs ~$0.35 cloud). AWAITING from Scott: project completion,
  the clone's own repo link, a screenshot. Consider a README link to
  the showcase once entry #1 is finished.

- **Thinking as a first-class knob (user question 2026-08-30):** llama-
  server exposes reasoning at the server level — `--reasoning
  on|off|auto`, `--reasoning-effort minimal…xhigh`, `--chat-template-
  kwargs` (verified in the b10672 help) — so it is per-model preset
  territory, already settable TODAY via ⚙ Tune extra flags (e.g.
  `--reasoning off` or `--reasoning-effort low`). Promotion candidates,
  in the spirit of the spec-type/ubatch ⚙ promotions: (a) a first-class
  ⚙ "reasoning" field with the measured hint; (b) a `think` trial menu
  racing on/off/low for agent work — tg + 2nd-turn ms + the quality
  probe deciding whether thinking EARNS its latency on this model
  (measured, not guessed, on the most-debated knob in local LLMs);
  (c) folds into the gpt-oss advisor fix (top of work queue), which
  needs reasoning_effort plumbing anyway. Scope note: request-level
  chat_template_kwargs from the client (OpenCode, our advisor) override
  the server default — three scopes: template default < per-model
  preset (OURS) < per-request (the harness's).

- **Pre-flight check (user idea 2026-08-30, design settled, build AFTER
  Scott's GUIDE.md QA walk):** a deterministic checklist engine in core
  — `preflight::check(scan, measurements, trials, cfg) →
  Vec<(severity, finding, action)>` — composing the EXISTING tested
  signals (Build Advisor verdicts, router state + version gate,
  unmeasured/failed/oversized rows, stale bench baselines, unapplied ★
  winners, MoE placement-first guard, OpenCode drift states, missing
  quality/advisor seat) into one surface: each finding a plain sentence
  plus a Fix button wired to the existing action. Placement (Scott's
  pick): "Pre-flight Check…" in the File menu beside Set Up Everything
  opening a findings dialog, plus a clickable "N suggested next steps"
  status-bar line whenever findings exist. Distinction to preserve:
  Set Up Everything RUNS the pipeline; Pre-flight EXAMINES and
  proposes, touching nothing until a Fix is clicked. AI is an optional
  "ask the advisor about this list" opinion — never the source of
  findings. Scott's QA walk feeds the rule list: every rough edge from
  walking docs/GUIDE.md becomes a pre-flight rule.

- **Campaign ETAs — ✔ BUILT same day (deterministic, no AI needed):**
  the Lab sums a per-campaign estimate from the model's MEASURED tg/pp
  (probe token budgets are known), warns loudly ≥90 min, and asks for
  Measure+Bench first when no baseline exists. The 3-hour GLM surprise
  can't recur unannounced.
- **✔ BUILT 2026-08-29 (test-first; the advisor's finished form):**
  gpt-oss-20b PASSED its seat exam — 6/6 evals, 5/5 tools, 3/3 agent
  loops, the fleet's best quality card, at 160 t/s — and the tested
  seat rules (aiadvisor::pick_advisor: pin wins; else best quality
  tier, fastest within it) now select it automatically, no pin
  needed. Settings gained an Advisor-model selector (Auto default,
  loop scores shown). "Ask about tuning" shipped: a question box in
  the Advisor window, grounded on TUNING_CORPUS (the curated,
  code-versioned knob notes — the RAG-that-isn't) plus this machine's
  findings JSON, answered by the earned advisor; first live answer
  cited real rows and recommended the honest trial. Original plan: (a) answerer selection goes quality-first,
  speed-weighted (a 15-18 t/s giant should not write fleet briefs),
  plus a user-pinnable "advisor model" in config; (b) the DESIGNATED
  CANDIDATE for that seat is **gpt-oss-20b** — measured on this
  machine at 131k ctx, 5,861 pp / 160 tg, fully VRAM-resident
  (~13GB), downloaded 2026-08-28 — to be confirmed by its quality
  probe (evals + tools + agent loops) before it gets the chair: the
  advisor model must EARN the seat with measured scores, same as any
  other recommendation in this app; (c) build "Ask about tuning"
  (below) on the same selection.
- **"Hand testing off to an AI"** — REJECTED with reasoning: campaign
  orchestration, ETAs, and scheduling are deterministic and already
  encoded; AI is never load-bearing here (standing rule). The AI's
  role stays advisory (explain, brief, triage).
- **Tuning-knowledge assistant — YES; RAG — NO (for now):** a vector
  store over a curated knob corpus is heavy machinery whose failure
  mode is confidently-stale advice, and llama.cpp semantics drift
  weekly (this week alone: log grammar, fit-vs-override behavior).
  The app's real, current knowledge is SMALL — the explain()
  glossaries, the knob-review REJECTED list, the roadmap findings,
  the findings JSON — well under a context window. Build instead: an
  "Ask about tuning" advisory that prompt-grounds on that curated
  corpus + the user's own measurements (no retrieval layer, no
  embeddings to rot). Revisit RAG only if the corpus outgrows
  context; the boundary rule stands (the user's notes-app RAG is a
  Connections CLIENT, not this app's job).

## M9 — The Meter (queued 2026-08-27, design talk with user)

**Premise:** local AI is not free — you pay in electricity and hardware
instead of API bills. The Meter measures what a token actually costs on
THIS machine, in this project's spirit: measured, not guessed. (Founding
data point: during a live OpenCode session the GPU sat at 446W of its
450W limit — napkin math put the 27B around $0.5–0.6 per million output
tokens at $0.15/kWh, cheaper than cloud but decidedly not free.)

**Settled decisions (user):**
- **Home:** steward core now, standalone later — `core/meter.rs` written
  extraction-clean (inputs: log text + timestamps + power samples;
  outputs: counters/reports; no steward-specific types in its API) so it
  can become a shared crate if it proves out.
- **Accounting:** BOTH, clearly separated — measured marginal
  electricity is the headline; amortized hardware (user-entered cost +
  lifetime) is a labeled estimate below it, never silently blended.
- **Pure token reporting is a first-class requirement**, not a side
  effect (user request): totals, averages, and datetime-range queries
  over everything below.

**Token metrics (phase 1): ✔ DONE 2026-08-28** — core/meter.rs
(extraction-clean: log text + timestamps in, buckets/reports out).
Continuous harvest in the poller + on every `--meter` run; cursor
(meter-cursor.json) fingerprints the log instance so crediting is
idempotent and truncation-proof; ledger (meter.jsonl) accumulates
hour buckets forever. Surfaces: Server-tab "Meter today" line,
`--meter [today|24h|7d]` (totals, per-model, prompt:generated shape,
busiest hour, per-day series, cloud-comparison against the editable
config price). Live shakedown found b10630→b10672 LOG GRAMMAR DRIFT
(n_gen/progress lines nearly gone) — the evidence parser now speaks
both dialects, which also healed the silently-broken cache monitor.
Original spec:
- Totals: prompt vs generated vs cache-reused tokens, per model and
  fleet-wide (reused tokens are the "tokens you didn't pay for" —
  ties the cache monitor into the money story).
- Rates & shapes: tokens/turn, turns/day, prompt:generated ratio
  (agent workloads are prompt-heavy — expect prefill to dominate),
  realized tokens/sec vs benched ceiling (utilization).
- Time series: per-hour/per-day buckets, queryable by datetime range;
  busiest hour/day; duty cycle (fraction of uptime actually generating).
- Comparative counters: lifetime tokens ≈ $X at (dated, user-editable)
  cloud prices vs $Y measured local cost.
- **Design constraint (learned from the log grammar):** router.log is
  truncated on router start — token evidence must be harvested
  continuously (poller snapshots deltas into meter.jsonl, append-only
  like history.jsonl) or counts die with each restart.

**Energy metrics (phase 2): ✔ INSTRUMENT BUILT 2026-08-28** —
core/energy.rs: GPU joules via NVML sampling (all GPUs summed, ~2Hz,
unprivileged), CPU joules via RAPL package counters (wraparound
handled; root-locked on this kernel → honestly None, never estimated,
with a Build Advisor unlock hint mirroring the persistence-mode one).
Marginal accounting: idle baseline sampled just before the window,
subtracted, clamped at zero. First consumer: every trial round now
records **J/token** over the novel-generation window (TrialResult.
j_per_token, "J/tok" table column, glossary) — energy is a verdict-
visible axis from the next campaign run onward. Still open in p2:
J/token → verdict guards/tie-breaks once real numbers exist; meter
$-line using measured J/token × kwh price (phase 3). Original spec:
- NVML power sampling (GPU) + Intel RAPL energy counters (CPU package,
  /sys/class/powercap — real joules) around generations: idle baseline
  vs under-load → marginal J/token per model, measured not nameplate.
- Honesty band: NVML+RAPL miss PSU losses/RAM/fans — label estimates
  ±20%; a smart-plug/wall-meter calibration knob is a later extension.
- **J/token as a trial column**: verdicts gain an efficiency axis
  ("t24 was 45% slower AND burned more energy per token").

**Surfaces (phase 3): ✔ the dollar line landed 2026-08-29 (test-first,
contracts before code) — M9 CLOSED.** cost_report bills generated
tokens at each model's measured J/token for the config it actually
SERVES (trial::served_j_per_token: applied winner's row, else stock
baselines, else honestly None); uncovered tokens are excluded and
said so, never estimated. `--meter` prints measured local cost vs the
cloud counter (first live reading: 116,620 tokens = $0.0057 measured
vs $0.35 cloud — the milestone's thesis, answered); the Server-tab
line appends ~$ measured. kwh_price_usd joins config. Moved to
backlog: J/token as a verdict guard (needs accumulated numbers to
set honest thresholds); prefill-side energy attribution.

**Later / parked:** background-service mode beyond the GUI poller
(systemd timer harvesting the log); smart-plug calibration; Ollama-peer
metering; standalone-crate extraction.

## Sibling project: modelwarden (`~/src2/modelwarden`)

Inventory / backup / archival for model files lives in its own project —
see its HANDOFF.md for mission, boundary contract, and harvest map.
Boundary: **warden owns storage truth; this app owns serving + OpenCode.**
Consequence here: roadmap item 7 (HF downloads) moves to modelwarden —
acquisition is storage-side.

## Parked / ideas

- ✔ SHIPPED 2026-08-27/28 (see M8 #5 + Managed llama.cpp in the Build
  Advisor) — kept below as the original idea record.
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

- ✔ Live activity indicator (built 2026-08-29, test-first): the status
  bar now says what the router is DOING — "model (loading…)",
  "prefilling 84%", "working…" — from router /models statuses plus a
  pure log-tail classifier (evidence::activity_hint) gated by
  log-growth freshness in the poller. Born from two is-it-even-working
  sessions in two days.
- Multi-GPU override UI (device pinning, per-device fit targets) — core
  is list-based already; UI lands when a second GPU exists to test with.
- OpenCode deeper integration (default model selection, agent presets).
- Prompt-cache tuning surfaced (`--cache-ram`, context checkpoints) with
  measured effect on agent-turn latency.
