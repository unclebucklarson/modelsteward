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

## M5.5 — Usability pass (in progress)

1. **Settings pane + persisted config** (`config.json`): scan dirs, router
   port, llama-server binary override, Ollama port. Manual pointing was in
   the original brief; give it UI.
2. **Incremental + stale-aware calibration**: fingerprint each measurement
   (server build + device set + effective model args); measure only new or
   stale models; `force` re-measures all.
3. **Remember failed loads**: persist the failure per model, badge it in
   the Library ("needs newer llama.cpp"), stop re-failing every calibration.
4. **One-click setup**: start router → measure missing → sync, narrated in
   the activity log. Menu item + button; CLI `--setup`.

## M5.6 — Test, refine, then deepen

5. **Measured tool-calling**: during calibration, fire a one-shot tools
   request at each loaded model and record whether well-formed `tool_calls`
   come back; sync writes measured `tool_call`, not assumed.
6. **Per-model override editor**: dialog for ctx, KV type, extra flags
   (later: device pinning) writing through preset + hot reload.
7. **Model downloads**: paste a HuggingFace repo (`user/model:quant`);
   llama-server `-hf` pulls into its cache, which the router already lists.
8. **Log viewer + Tools menu**: tail router.log in-app; open preset/config.

## Smaller items (fold in opportunistically)

- Numbered backups (cap ~5) or "Undo last config change" (.bak swap).
- Status-bar VRAM refreshed on the poll cycle, not scan-time.
- Contention warning names the remedy (`ollama stop <model>`).
- `limit.output` exposed in the override editor (crude ctx/2 cap today).
- Library staleness badge driven by fingerprint mismatch.

## M6 — Build Advisor

Deterministic hardware/toolchain probe + rules engine → recommended cmake
flags, staleness vs upstream ("b10216 can't load 4 of your models — a
newer build fixes the qwen3.5-MoE hyperparameter format"), optional
run-the-rebuild with log pane. Then the `Advisor` AI layer (default
backend: the local model this app serves) for build-log diagnosis and
tradeoff explanations — picks from the rules engine's flag allowlist,
never invents flags.

## M7 — Performance lab

llama-bench integration: baseline pp/tg per model stored beside
measurements; A/B any preset change with measured verdict; one-click
speculative-decoding trial (`--spec-draft-model` + small draft model);
results shown next to every recommendation.

## Parked / ideas

- Multi-GPU override UI (device pinning, per-device fit targets) — core
  is list-based already; UI lands when a second GPU exists to test with.
- OpenCode deeper integration (default model selection, agent presets).
- Prompt-cache tuning surfaced (`--cache-ram`, context checkpoints) with
  measured effect on agent-turn latency.
