# modelsteward — Project Brief

*A desktop manager for local AI model serving. Rust + egui, Linux.
Personal project, August 2026. This brief is written for a general
technical audience.*

## What it is

modelsteward manages the full lifecycle of running large language
models locally with llama.cpp: finding the models scattered across a
machine, serving them through one hot-swapping server, measuring what each
model can actually do on the available hardware, and keeping client
applications (AI coding agents, note-taking tools — anything speaking the
OpenAI-compatible API) configured with numbers that are true.

## The problem it addresses

Running LLMs locally is attractive — private, free per-token, offline —
but the tooling around it is fragmented in ways that punish non-experts:

- **Models scatter.** Ollama keeps them in a content-addressed blob store,
  HuggingFace tools keep a revision-pointer cache, manual downloads land
  wherever. A machine can hold 200+ GB of models with no single answer to
  "what do I have, and is it usable?" (During this project's development,
  its own scanner surfaced a 20 GB model the owner didn't know was there,
  and a downloaded model that was on disk but unreachable because a cache
  pointer had moved.)
- **Configuration runs on folklore.** Client apps need to know each model's
  context window. The number in the model file is the *training* figure;
  what actually fits after GPU memory allocation is routinely a third of
  that. Configs written from file headers cause silent truncation and
  mid-session failures that are miserable to debug.
- **The expertise barrier is real.** llama.cpp exposes dozens of flags with
  meaningful performance consequences (KV-cache quantization alone roughly
  doubled usable context in this project's measurements). Most users never
  find them; the knowledge lives in forum threads.
- **Things break with version skew.** Model formats evolve faster than
  installed binaries. A model that fails to load reports a cryptic error
  deep in a server log — actionable only if you already know the answer.

## The approach: measured, not guessed

The core design decision: **the app never writes a number it hasn't
measured.** Loading a model *is* measuring it — the app records the context
size the server actually settled on for this machine and probes whether the
model produces well-formed tool calls, then writes client configs from
those measurements (with a small safety margin). Measurements are
fingerprinted against the server build, GPU set, and per-model flags, so
they refresh automatically when anything relevant changes — and stale
numbers are labeled as stale rather than silently trusted.

The same philosophy extends to diagnostics. Every unavailable model has a
"Why?" button that gives a plain-language cause, the single relevant error
line as evidence, and remedy buttons — because "see the logs" is not an
answer a general user can act on. A built-in Build Advisor compares the
installed llama.cpp against its source checkout and upstream, reports in
outcomes ("rebuilding would unlock 3 models you own") rather than compiler
flags, and can run the guided rebuild itself. In live use it correctly
distinguished "your checkout was updated but never rebuilt" from "you need
to pull," and after a rebuild, re-measurement confirmed one previously
unloadable model working — and honestly reclassified three others as
incompatible-by-design rather than fixable, updating its own advice
accordingly.

## What it does today

- Discovers llama.cpp installations (version, GPU visibility per binary)
  and every GGUF model across user directories, the Ollama store, and the
  HuggingFace cache, deduplicated by file identity.
- Runs llama-server in router mode: one server, all models, on-demand
  hot-swap (~12 s for a 27 B-parameter model on the development machine).
- Generates server configuration with performance defaults measured on real
  hardware, per-model overrides that survive regeneration, and a
  reset-to-optimized escape hatch.
- Detects per-model capabilities from the files themselves — vision
  components (paired automatically), multi-token-prediction layers,
  embedding architectures — and configures serving accordingly.
- Syncs the OpenCode coding agent's config through a comment-preserving
  JSONC editor: hand edits survive, removals are reversible comment-outs,
  every write is backed up (numbered, with one-click restore).
- Provides a Connections panel for any other OpenAI-compatible app: base
  URL, measured model list, copy-paste client snippets.
- Coexists safely with other tools: a server it didn't start is observed,
  never signaled; model files are never deleted by the app, period.

## Engineering notes

- **Architecture:** a headless, fully-testable core under a thin GUI shell;
  the same engine drives both the desktop app and a scriptable CLI. 88 unit
  tests at the time of writing, most added alongside the bug or behavior
  they pin.
- **Verify before building:** the project started with a spike day that
  validated every load-bearing assumption against the real server binary
  before any application code depended on it; findings are documented in
  the repo.
- **Bugs found by use, fixed at the root:** live testing surfaced four
  identity/truth bugs (models sharing an alias, a quant label the file
  itself misstated — confirmed by reading the tensor table directly). Each
  fix removed the *class* of error (single source of truth for naming)
  rather than patching the instance, with a regression test citing the
  incident.
- **Safety invariants held throughout:** never touch a process the app
  didn't start (verified by process-table ownership handshakes, robust to
  PID recycling); never destroy user data (comment-out over delete,
  backup before write, refuse to overwrite).
- Built in Rust (egui for the GUI, minimal dependency footprint), developed
  in an AI-pair-programming workflow with the design decisions, testing
  discipline, and roadmap maintained as first-class artifacts in the repo.

## Current state and honest limitations

Working daily on its development machine (Linux, single NVIDIA GPU), where
it manages 15+ models across three stores and keeps two client apps
configured. Limitations worth stating: Linux-only today; multi-GPU support
is designed in (GPU state is a list everywhere) but untested on real
multi-GPU hardware; ROCm build support is implemented against
documentation and simulated tests, not yet validated on AMD hardware; and
it is a personal-scale project — one machine, one primary user — not yet
hardened by a wider install base.

The lab's most striking demonstration: models far larger than the GPU
are usable — an 80-billion-parameter mixture-of-experts coding model ran
at its full 262k context near 40 tokens/sec on a single consumer 24GB
card, with expert weights in system RAM and placement chosen by
measurement rather than folklore.

The performance lab is delivering: every servable model carries a
measured speed baseline, and the generalized trial harness A/Bs any
config change with a measured verdict — winners apply in one click,
rule-rejected tradeoffs are surfaced for human judgment rather than
swallowed. Its first campaigns adopted ngram speculative decoding
fleet-wide where it earned its keep (up to +118% generation on
edit-heavy agent work, zero VRAM) while measuring and rejecting the
classic draft-model pairing conventional wisdom recommended — on a
single 24GB card it costs 96% of usable context. Planned next: the
post-rebuild verification loop ("this rebuild unlocked N models ✓") and
a sibling project for model inventory/backup with content-hash identity.
