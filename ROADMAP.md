# Roadmap

The tracking document for llamacppCodeConf. PLAN.md holds the founding
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
8. **Log viewer + Tools menu**: tail router.log in-app; open preset/config.

## Smaller items (fold in opportunistically)

- **Router start options** (user request 2026-08-17): user prefers manual
  start for now — Start Router button already covers it. Down the road:
  a Settings toggle "start router when the app opens", and a first-run
  hint on the Server tab explaining the three ways to start (button,
  `--start`/`--setup` CLI, systemd unit).

- **Cache-source models are now calibrated/synced** (found by user: a
  `llama-server -hf` download appeared on the Server tab but could never
  reach OpenCode). Remaining gap: the Library tab doesn't list cache
  models (this llama-server build doesn't expose their file paths via
  /models) — consider merging the router's model list into the Library
  view when the router is up.
- GUI should reload measurements.json when it changes on disk (stale
  OpenCode tab after CLI calibration) and show display-name + alias
  consistently across panes (found by user).
- **Migrate measurements on archive**: archiving a cache model gives it a
  new preset alias, so its old measurement (keyed by the cache id) shows
  on a leftover router-only row while the shelf row reads "not measured".
  Copy n_ctx/tool_call to the new alias (fingerprints cleared → re-measures
  next calibrate) and consider suppressing the stale cache-id row.

- Numbered backups (cap ~5) or "Undo last config change" (.bak swap).
- **Measured-ctx variance policy**: settled ctx varies a few percent with
  desktop VRAM at load time (observed 83k–94k across runs for the same
  model). llama-server's own 1GiB fit margin absorbs moderate drift; decide
  whether sync should apply an additional haircut (e.g. round down 5%).
- Status-bar VRAM refreshed on the poll cycle, not scan-time.
- Contention warning names the remedy (`ollama stop <model>`).
- `limit.output` exposed in the override editor (crude ctx/2 cap today).
- Library staleness badge driven by fingerprint mismatch.

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

**Phase 2 remaining:**
- Post-rebuild verification loop: after a successful rebuild, offer/auto
  run stop → start → Set Up Everything and report "N models unlocked ✓"
  against the pre-rebuild locked list.
- The `Advisor` AI layer (default backend: the local model this app
  serves) for build-log diagnosis and tradeoff explanations — picks from
  the rules engine's flag allowlist, never invents flags.

## M6.5 — Feature-aware serving (discussed 2026-08-17)

Load every model with the options its features deserve. Detection is
deterministic (GGUF metadata + file siblings + load-log mining), enablement
is rules in the preset generator, benefit is measured. NOT per-model-type
llama.cpp builds — architecture support is source version, not cmake flags;
the Build Advisor's "stay current" already owns that lever. (Small truth
kept: a few global perf cmake knobs, e.g. extended FA kernel coverage for
quantized KV, belong in the Advisor's advanced section someday.)

1. **Detect & show**: `features.rs` — per-model feature set from gguf +
   siblings + logs: vision (mmproj sibling), MTP (`nextn` tensors — the
   Qwen3.8 load log already proves "unused tensor blk.64.nextn.*", i.e.
   capability present but unexploited by the build), embedding models
   (bge-small in the hub), thinking/reasoning, SWA family. Library badges;
   "your models carry features your build ignores" card in Build Advisor.
2. **Safe auto-enablement**: preset generator pairs mmproj automatically
   (and stops hiding mmproj files → pair instead of exclude), sets
   embedding mode for embed models, reasoning-format where applicable;
   OpenCode entries get image modality when vision is paired.
3. **Measured trials** (merges into M7): draft-model pairing for
   speculative decoding (same-family small model, e.g. Qwen3.5-4B drafting
   for the 27Bs), MTP self-speculation when upstream lands the flag —
   each as try → measured verdict → keep/discard.

## M7 — Performance lab

llama-bench integration: baseline pp/tg per model stored beside
measurements; A/B any preset change with measured verdict; one-click
speculative-decoding trial (`--spec-draft-model` + small draft model);
results shown next to every recommendation.

## Sibling project: modelwarden (`~/src2/modelwarden`)

Inventory / backup / archival for model files lives in its own project —
see its HANDOFF.md for mission, boundary contract, and harvest map.
Boundary: **warden owns storage truth; this app owns serving + OpenCode.**
Consequence here: roadmap item 7 (HF downloads) moves to modelwarden —
acquisition is storage-side.

## Parked / ideas

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
