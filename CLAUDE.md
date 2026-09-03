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
  (incremental; skips fingerprint-fresh measurements), `--bench [id] [force]`
  (llama-bench pp/tg baselines; skips current-build-fresh ones),
  `--trial <id> [menu] [keep <variant>]` (measured config A/B; menus and
  everything else: `modelsteward --help` is the single source of truth),
  `--quality <id>` (evals + tools + agent loops), `--meter [range]` (token
  ledger + measured cost), `--report`, `--advise`, `--verify-rebuild`,
  `--sync`, `--install-service`. App config lives at
  `~/.config/modelsteward/config.json` (see core/settings.rs);
  measurements at `~/.local/state/modelsteward/measurements.json`.
- Track work in [ROADMAP.md](ROADMAP.md); update it when milestones land.

## Development practice (user direction, 2026-08-29): test-first

- **Write the failing test before the code** for anything with a
  knowable contract — verdict rules, parsers, accounting math, path
  logic. The test states the intended behavior; the implementation
  earns its green. This codifies what already worked here: every
  verdict-logic change in this repo shipped pinned by a test carrying
  the live numbers that motivated it.
- **Integration tests** for flows that cross modules (scan → preset →
  measurements, harvest → ledger → report) using tempdirs and synthetic
  fixtures — never the user's real state dirs.
- **Live incidents become regression tests**, with the real numbers in
  the test body and a comment naming the date and what broke.
- **Honest limits**: egui rendering and subprocess/network edges get
  thin, logic-free wiring plus live validation — testability pressure
  belongs on core, which is headless precisely so it CAN be tested.
  If a UI closure grows logic, extract it into core and test it there.
- `cargo test` green (and zero warnings) before every commit.

## Architecture (big picture)

Single binary, strictly layered: `src/core/` is a headless, testable
engine; `src/ui.rs` (one large file) is an egui shell over it. Core
never depends on egui. Modules (`src/core.rs` is the authoritative
list): `discover` (llama.cpp installs + GPU inventory + aliases),
`gguf` (header metadata reader, incl. the chat-template reasoning
contract), `library` (unified model registry: scan dirs + Ollama blobs
+ HF cache, split-GGUF aware, inode-deduped), `router` (preset-INI
generation, llama-server lifecycle, measurements), `rows` (Library row
assembly + advice levels), `opencode` / `piagent` / `hermes` (agent
config connectors: diff/apply with backups; opencode.json via the
comment-preserving `jsonc` editor), `ollama` (peer probe), `trial`
(measured config A/B harness, verdicts with magnitude-scaled guards),
`quality` (eval battery + tool + agent-loop probes), `bench`
(llama-bench baselines), `evidence` (router.log miner: cache stats,
child ports, parse-coverage drift detection), `meter` (token ledger),
`energy` (NVML/RAPL joules per token), `history` (append-only journal
+ rebuild scorecard), `advisor` (build check/rebuild/verify engine),
`managed` (app-owned llama.cpp checkout + archived builds),
`aiadvisor` (grounded one-shot AI opinions — NEVER load-bearing),
`reasoning` (per-model template reasoning levels), `safefs` (atomic
durable writes; Missing-vs-Damaged reads with rescue), `diagnose`,
`report`, `jsonc`, `cancel`, `settings`, `system`.

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
- **AI is never load-bearing.** Verdicts, flags, syncs, and writes are
  deterministic; `aiadvisor` output is labeled opinion, grounded in
  supplied data only, never auto-applied.
- **Probing binaries is worker territory** — `--version`/`--list-devices`
  spawn subprocesses (the latter initializes CUDA); never call
  `pick_server`/`find_installs` in a render loop or poll tick (GUI uses
  the cached scan; the poller caches its router config).
- **Selection vs production**: Settings is the ONE surface that selects
  the serving binary; the Build Advisor makes builds and never selects.
  Managed builds/archives serve only when explicitly chosen.
- **Two speed numbers, and they mean different things.** `tg_tps` is
  generation from an EMPTY KV cache (llama-bench's default); `tg_deep_tps`
  is generation with `tg_depth` tokens resident, which is what a user
  gets mid-session and runs ~20-30% lower. Never present the empty-cache
  figure as "the speed". Likewise `n_ctx` is llama.cpp's `--fit`
  *projection* against free VRAM less a 1024 MiB margin — sound, but not
  the largest context the card can serve, and it moves with whatever
  else holds the GPU (hence `free_vram_mib` / `gpu_tenant`).
- **A contended measurement is a wrong measurement, not a slow one.**
  Anything that writes to `measurements.json` must establish its
  preconditions first (router idle, no Ollama residency) and refuse
  rather than record contention.
- **Log grammars drift** — evidence.rs speaks both the pre- and
  post-b10672 dialects; when the meter reads zero while tokens flow,
  suspect a new dialect before suspecting the code.

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
