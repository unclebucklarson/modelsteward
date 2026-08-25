# llamacppCodeConf — Plan (founding design, kept as history)

> **Status 2026-08-25:** M0–M5.7, M6 phase 1, M6.5 phases 1–2, the
> Connections pivot, M7 phases 1–2 (baselines + the measured-trial
> harness, campaigned: ngram speculation adopted fleet-wide where it
> earned its keep), and the wild-readiness pass are DONE — see
> [ROADMAP.md](ROADMAP.md) "Where things stand" for the live tracking.
> One scope evolution since this plan was written: the app is a
> **llama.cpp server manager for any OpenAI-compatible app**
> (Connections tab), with OpenCode as its first-class synced connector
> rather than its only purpose. Everything else here held up.

One Rust + egui desktop app for Linux that manages the whole local-LLM
stack: discovers llama.cpp installs and GGUF models, runs llama-server in
**router mode** (one long-lived, multi-model, hot-swapping server), keeps
`~/.config/opencode/opencode.json` correct and in sync with what is actually
servable, and hands any other app a measured, ready-to-paste connection.

## North star

**Squeeze maximum performance out of local models — for AI coding and any
other local-AI app — without the user needing to be a llama.cpp expert.**
The app encodes the expertise. Four consequences:

1. **Opinionated coding-agent defaults.** Agentic coding is prefill-heavy:
   OpenCode resends large, mostly-identical prompts every turn. So defaults
   optimize prompt-processing speed and cache reuse — generous `--cache-ram`,
   context checkpoints, KV cache quantization (`-ctk/-ctv q8_0`) to buy
   context headroom — not just raw generation speed.
2. **Measured, not guessed.** The bundled `llama-bench` binary turns config
   choices into numbers. Recommendations ship with measured pp/tg tok/s on
   *this* machine ("q8_0 KV: +6k ctx, −2% tg speed"), and any tweak can be
   A/B-benchmarked before it becomes the default.
3. **Expert levers, one click.** Advanced wins like speculative decoding
   (`--spec-draft-model` with a small draft model) become a "try this →
   measured verdict → keep or discard" flow instead of a research project.
4. **Teach while doing.** Every recommendation shows the exact flag it sets and
   a one-line why, so expertise transfers instead of staying locked in the tool.

## Decisions (2026-08-14)

| Decision | Choice |
|---|---|
| UI / stack | Rust + egui (eframe), native desktop GUI |
| Architecture | Router-first: one llama-server with `--models-preset` INI; no per-model process juggling |
| Prior code | Fresh repo; harvest proven modules from `llm_forge` (GGUF header parser, Ollama blob-store reader, server guard) and `opencode_configuration_tool` (opencode.json probe/diff/write) |
| Ollama | Peer provider: synced in opencode.json alongside llama.cpp, ports tracked, VRAM contention warned; its blobs also usable as llama.cpp models |

## Why router-first (what changed since the prior projects)

The installed llama.cpp build (`~/src/llama.cpp/build/bin`, b10216, CUDA) now has:

- **Router mode**: `--models-dir`, `--models-preset <ini>`, `--models-max N`,
  `--models-autoload` — one server exposes many models and loads/swaps them on
  demand. This removes the original pain ("llama.cpp doesn't switch models like
  Ollama").
- **Memory auto-fit**: `--fit on` and `-ngl auto` are defaults, with
  `--fit-target` / `--fit-ctx` tuning — llama-server sizes context and GPU
  offload to fit VRAM by itself. We surface/override this rather than
  reimplementing the math.
- `--jinja` on by default — tool calling works for OpenCode out of the box.

The app's job is therefore the *glue that is still missing*: discovery,
preset-INI management, server lifecycle, and opencode.json correctness.

## Multi-GPU readiness (design principle)

The dev machine has one GPU today, but nothing in the app may assume that.
llama.cpp already handles multi-GPU natively — `--split-mode layer` (the
default) splits models across GPUs automatically, `--fit-target` takes
per-device values (`MiB0,MiB1,...`), and `--device` / `--tensor-split` /
`--main-gpu` give manual control. So the rule is:

- **GPU state is always a list.** The engine enumerates devices with
  `llama-server --list-devices` (authoritative: it reflects what the chosen
  binary can actually see) plus NVML/nvidia-smi for live per-device VRAM use.
  Single-GPU is just the `len == 1` case; no scalar "the GPU" anywhere in core.
- **Defaults stay auto.** With multiple GPUs present, llama.cpp's own
  layer-split + fit behavior is the default; the app adds value by *showing*
  the placement, not by second-guessing it.
- **Overrides are per-model, optional.** The preset editor exposes `--device`,
  `--split-mode`, `--tensor-split`, `--main-gpu`, and per-device fit targets as
  advanced fields — e.g. pin a small model to GPU 1 while a big one spans both.
- **UI renders per-device.** The server pane shows one VRAM gauge per device;
  the Ollama contention warning is computed per device, not globally.

## Target system facts (dev machine)

- RTX 3090 Ti 24GB VRAM, 62GB RAM, Linux.
- llama.cpp builds: `~/src/llama.cpp/build/bin` (primary, CUDA b10216), second
  copy under `~/.unsloth/llama.cpp`. Nothing on PATH — discovery matters.
- GGUF shelf: `~/models` (~57GB, 3 quants of Qwen3.6-27B).
- Ollama: `/usr/local/bin/ollama`, 7 models (~130GB); blobs in
  `~/.ollama/models/blobs` are raw GGUFs, manifests under
  `~/.ollama/models/manifests` map names → blob digests.
- OpenCode: `~/.opencode/bin/opencode`; config at
  `~/.config/opencode/opencode.json` with existing `llamacpp` + `ollama`
  providers and many hand-managed backups (the problem statement in file form).

## Architecture

Single binary, layered so the engine is testable without the GUI:

```
src/
  core/
    discover.rs    llama.cpp install discovery: PATH, ~/src, ~/.unsloth, manual;
                   run `llama-server --version`, detect backends from libggml-*.so;
                   GPU inventory via `--list-devices` + NVML (always a Vec)
    gguf.rs        GGUF header reader: arch, params, quant, n_ctx_train, chat
                   template presence            [harvest: llm_forge]
    library.rs     model registry: scan_dirs + Ollama blob store, dedupe by file
                   sha, one unified "servable model" list
    router.rs      preset-INI generation (aliases, per-model overrides:
                   cache-type, fit-target, ctx), llama-server spawn/stop/health,
                   /v1/models + router endpoints, log capture; never touch a
                   server we didn't start     [harvest: llm_forge server guard]
    opencode.rs    read/diff/write opencode.json: real context limits from the
                   running server, tool_call flags, orphan detection, backup
                   before write               [harvest: opencode_configuration_tool]
    ollama.rs      peer probe (/api/tags), port + VRAM awareness
  ui/
    library_pane   models table (source, quant, size, ctx, servable-by)
    server_pane    router status, loaded models, per-device VRAM gauges,
                   start/stop, logs
    opencode_pane  three-list diff (configured / new / orphaned) + apply
    settings_pane  scan dirs, chosen llama.cpp install, ports, fit targets
  app config       ~/.config/llamacppCodeConf/config.json
```

## Spikes first (verify before building on them)

> **Status 2026-08-14: all four spikes run and confirmed** — see
> [docs/spikes.md](docs/spikes.md). Highlights: router mode + hot reload work;
> 27B swap in ~12s; `--fit` settles 27B-Q5 at 72,960 ctx with q8_0 KV (vs
> 36,096 with f16 KV) and `/props` reports it live; tool calls work through
> the router; Ollama store here is `/usr/share/ollama/.ollama`; JSONC editing
> harvested from prior tool. Architecture is validated — proceed to M1.

1. **Router mode reality check** (highest risk): with the real b10216 binary and
   a Qwen3.6 GGUF — exact preset-INI schema, what `/v1/models` reports per model,
   load/unload endpoints and swap latency, how `--models-max` interacts with a
   24GB card, whether OpenCode streaming + tool calls work through the router.
2. **Fit behavior**: are `--fit-*` settable per-model in the preset INI or only
   globally? What does the server report as the *actual* context it settled on
   (that number, not `n_ctx_train`, is what belongs in opencode.json limits)?
3. **Ollama blobs via router**: blobs are extensionless sha256 files — confirm a
   symlink farm (`<name>.gguf -> blob`) satisfies `--models-dir`/preset paths.
4. **opencode.json fidelity**: current file contains `//` comments (written by
   the prior tool) — confirm comment/format-preserving edit strategy, or decide
   the tool owns the file wholesale with backups.

## Build Advisor (post-MVP module)

Recommend — and optionally run — a llama.cpp rebuild optimized for this machine.
Three layers, so the feature degrades gracefully and never depends on AI to be
correct:

1. **Hardware/toolchain probe (deterministic).** GPU compute capability, CUDA/
   ROCm/Vulkan toolkit presence + versions, CPU ISA flags, compilers, RAM.
   Mostly shared with `discover.rs`.
2. **Rules engine (deterministic).** Probe → curated cmake flag set (e.g.
   NVIDIA → `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<cc>`); diffs against
   the installed build ("CUDA build but wrong arch", "N releases behind");
   can run the rebuild with a live log pane. Fully offline; this layer owns
   the actual flags.
3. **AI advisory (optional).** An `Advisor` trait with pluggable backends —
   default: the local llama-server the app already manages (no key, offline);
   alternates: Ollama or a cloud API. Used for what rules can't do: diagnosing
   failed build logs, explaining tradeoffs, "should I rebuild?" against release
   notes. Guardrail: the model selects/explains from layer 2's allowlist and
   annotates logs — it never invents flags, and the UI labels its output as
   advisory.

## Milestones

- **M0 — scaffold + spikes.** Repo, git init, CI-less cargo workspace; run the
  four spikes with throwaway scripts; write findings into `docs/spikes.md`.
- **M1 — headless core.** discover + gguf + library with unit tests; a `--scan`
  debug flag prints the unified model table as JSON.
- **M2 — router lifecycle.** Generate preset INI, start/stop/health llama-server,
  logs; debug flags to start/stop from CLI.
- **M3 — opencode sync.** Diff/apply against opencode.json with backup; entries
  carry server-reported context and tool_call.
- **M4 — GUI.** egui shell wiring the four panes to the core.
- **M5 — Ollama peer + polish.** /api/tags sync, VRAM contention warning when
  both servers hold models, blob symlink farm, optional systemd user unit for
  the router, README.
- **M6 — Build Advisor.** Probe + rules engine first (useful standalone:
  "rebuild recommended" banner in settings), then the `Advisor` AI layer with
  the local server as default backend.
- **M7 — Performance lab.** llama-bench integration: baseline pp/tg per model,
  A/B a config change, one-click speculative-decoding trial with measured
  verdict; benchmark results stored per model+config and shown next to
  recommendations.

Each milestone leaves something runnable; M1–M3 are usable from the CLI before
the GUI exists.
