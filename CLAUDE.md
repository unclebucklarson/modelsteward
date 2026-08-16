# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Rust + egui Linux desktop app that manages the local-LLM-for-OpenCode stack
end to end: discovers llama.cpp installs and GGUF models, runs llama-server in
**router mode** (one long-lived multi-model server with hot-swap), and keeps
`~/.config/opencode/opencode.json` in sync with what is actually servable.

**Read [PLAN.md](PLAN.md) before making design decisions** — it records the
north star (max coding-agent performance without requiring llama.cpp
expertise), the settled decisions (router-first architecture, Ollama as peer
provider, multi-GPU-as-a-list principle), milestones M0–M7, and the spike
findings the architecture depends on. Spike results live in `docs/spikes.md`.

## Commands

- Build: `cargo build`
- Test: `cargo test` (single test: `cargo test <name>`)
- Run GUI: `cargo run` (no args)
- CLI: `--setup` (one-shot start+measure+sync), `--scan`, `--preset`,
  `--start/--status/--reload/--stop`, `--calibrate [port] [force]`
  (incremental; skips fingerprint-fresh measurements), `--sync`,
  `--install-service`. App config lives at
  `~/.config/llamacppcodeconf/config.json` (see core/settings.rs);
  measurements at `~/.local/state/llamacppcodeconf/measurements.json`.
- Track work in [ROADMAP.md](ROADMAP.md); update it when milestones land.

## Architecture (big picture)

Single binary, strictly layered: `src/core/` is a headless, testable engine;
`src/ui/` is an egui shell over it. Core never depends on egui. Modules:
`discover` (llama.cpp installs + GPU inventory), `gguf` (header metadata
reader), `library` (unified model registry: scan dirs + Ollama blob store,
deduped by file sha), `router` (preset-INI generation and llama-server
lifecycle), `opencode` (opencode.json diff/apply with backups), `ollama`
(peer probe via `/api/tags`).

Non-obvious constraints that shape the code:

- **llama.cpp does the hard parts.** Model switching (router mode), memory
  auto-fit (`--fit`, `-ngl auto`), multi-GPU layer-split are upstream features.
  This app generates config for them and reports what they did; it must not
  reimplement placement/sizing math.
- **GPU state is always a `Vec`** (enumerated via `llama-server
  --list-devices` + NVML). No scalar "the GPU" anywhere in core; single-GPU is
  the `len == 1` case.
- **Never touch a server we didn't start.** A llama-server or Ollama daemon on
  the expected port that the app didn't launch is observed, never killed or
  reconfigured.
- **opencode.json is JSONC** (real configs contain `//` comments). Edits must
  preserve or consciously own the file, and always back up before writing.
- **A stopped provider is not evidence its models are gone** — never mark
  opencode.json entries orphaned unless the provider was reachable and
  actually omitted them.
- **opencode.json limits use server-reported values** — the context the server
  *settled on* after `--fit`, not the GGUF's `n_ctx_train`.

## Environment facts (dev machine)

- Primary llama.cpp: `~/src/llama.cpp/build/bin` (CUDA build; not on PATH).
- Test models: `~/models/*/**.gguf` (Qwen3.6-27B quants); Ollama blob store at
  `~/.ollama/models` (blobs are raw GGUFs, mapped by manifests).
- OpenCode config: `~/.config/opencode/opencode.json`.
- Ollama may be running on :11434 and llama-server on :8080 — use high ports
  (e.g. 18080) for spikes/tests and leave user services alone.
- Prior projects to harvest from (don't rewrite what they proved):
  `~/src2/llm_forge` (GGUF parser, Ollama blob reader, server guard),
  `~/src2/opencode_configuration_tool` (opencode.json probe/diff/write).
