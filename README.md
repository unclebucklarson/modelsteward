# modelsteward

Squeeze maximum performance out of local models — llama.cpp serving any
OpenAI-compatible app, with first-class [OpenCode](https://opencode.ai)
integration — **without needing to be a llama.cpp expert**. A traditional
desktop app (Rust + egui) for Linux, with a scriptable CLI underneath.

The ethos in three words: **measured, not guessed.** The app loads your
models on your hardware, records what they actually do (real context after
memory fitting, real tool-calling behavior), and configures everything
downstream from those measurements.

## What it does

- **Runs models bigger than your GPU.** The headline nobody tells
  novices: a Mixture-of-Experts model several times your VRAM can run
  *well* with most experts in system RAM — measured here, an 80B A3B
  coder hit its full 262,144-token context at 52 tokens/sec generation
  and 303 t/s prefill on a single 24GB card (partial offload,
  `--n-cpu-moe 32`), 100% quality-gate fidelity and 100% multi-hop
  agent-loop reliability. The honest caveat is also measured: cold
  prefill costs ~30s per 10k prompt tokens, so the first turn of a big
  agent session takes a beat. The Library flags oversized MoE models
  automatically and the Lab's MoE-offload trial races full and partial
  placements to find the one your card affords.
- **Meters real usage** — a continuous token ledger (turns, prompt vs
  generated vs cache-served, busiest hours, per-model splits) that
  survives router restarts, with a cloud-price comparison you control.
  Local AI isn't free; now it's measured (`--meter`, Server tab).
- **Manages llama.cpp itself** (opt-in) — an app-owned checkout builds
  each new release in the background when the machine is idle, archives
  every build, and serving only ever changes by your explicit click.
  Rollback is a pin, never a rebuild.
- **Asks your own models for second opinions** — grounded, one-shot,
  clearly-labeled AI advisories (failure explanations, a fleet brief,
  rebuild triage) served by the models the app manages. Opinions, never
  load-bearing: every verdict and write stays deterministic.
- **Discovers** your llama.cpp installations (version, backends, which build
  can actually see your GPUs) and every GGUF on the system: your own
  directories ("shelf"), Ollama's blob store (blobs are raw GGUFs served
  directly, no duplication), and the HuggingFace hub cache.
- **Runs llama-server in router mode**: one long-lived server that lists all
  your models and hot-swaps them on demand (a 27B swap takes ~12s on a
  3090 Ti). Clients just name a model; the router does the rest.
- **Generates the router preset** with coding-agent-optimized defaults:
  `np = 1` (the whole context window goes to your agent) and `q8_0` KV cache
  (measured on real hardware: roughly double the usable context of f16 for
  the same VRAM). Every default is overridable per model (⚙ on each Library
  row — fields open pre-filled with the optimized values, with a "reset to
  optimized" escape hatch), and overrides live in the app config so preset
  regeneration never loses them.
- **Measures, never guesses**: loading a model *is* measuring it — the real
  context llama-server's `--fit` settled on (usually far below the GGUF
  header's training figure) and whether it produces well-formed tool calls.
  Measurements are fingerprinted (build + devices + effective flags) so only
  new or stale models get re-measured; failures are remembered with their
  reasons. Synced limits carry a 5% safety margin.
- **Feature-aware**: reads each model's actual capabilities from the file —
  vision (mmproj companions are paired automatically, and vision-served
  models get the image modality in OpenCode), MTP layers, embedding
  architectures (served via `/v1/embeddings`, kept out of chat configs) —
  and shows them as Library badges.
- **Benchmarks what it serves** (Server → Bench New/Stale Models, or
  `--bench`): llama-bench baselines — prompt-processing and generation
  tokens/sec at the real serving KV types — stored beside the measurements
  and shown in the Library's Speed column, re-measured only when the build
  changes.
- **Tunes with trials, not folklore** (⚡ Lab tab, or `--trial <id> <menu>
  <menu>`): pick a model, pick campaigns — Measure,
  Bench, speculation modes and their dials, prefill batch, KV precision,
  load mode, the quality probe — and Run (with a Cancel that stops at
  the next safe boundary, keeping partial results). Each trial
  races baseline vs candidates on fixed prompts, server-timed, including
  a **quality gate** (how much of a known-answer rewrite came back
  verbatim — no speed win survives degraded output). Strict rules pick
  winners that cost nothing; rejected tradeoffs (say, +50% prefill for
  context you'd never use) appear as explicit "your call" buttons, and a
  Why? explains the whole table in plain language. The Lab keeps standing
  recommendations with one-click **Apply/Revert** — applying cascades
  through the override, the preset, the live router, and the synced
  OpenCode limits. First campaigns: `ngram-simple` speculation adopted on
  four models (up to +118% generation on edit-heavy agent work, zero
  VRAM) while the folklore-favorite classic draft model was measured and
  rejected (on a single 24GB card it costs 96% of your context and is 6x
  slower — `docs/spikes.md`), and direct-IO loading measured 2x SLOWER
  than the page cache it bypasses on a warm rotation. Measurements that
  collide with your own coding session are skipped as "server busy",
  never recorded as model failures.
- **Judges quants and configs honestly**: a fixed eval battery and
  N-shot tool-reliability probe make quality a number, so the Library
  can say "prefer the Q4: +11% speed, +88% context (quality parity
  measured)" — with a veto when a faster config measurably answers
  worse. A cache-effectiveness monitor mines your real sessions'
  prompt-token reuse and flags models whose serving mode or attention
  disables cache-reuse (measured, with the Lab trial that prices it),
  which is one ⚙ checkbox to decide per model.
- **Remembers every measurement** (history.jsonl): each context measure
  and bench result is journaled with its llama.cpp build — hover the
  Measured ctx or Speed cells for the trail, so "what did that rebuild
  cost me?" is a glance, not archaeology.
- **Verifies rebuilds** (auto after a guided rebuild, or
  `--verify-rebuild`): restarts the router onto the new binary,
  re-measures everything the new build made stale, and reports the
  measured outcome — models unlocked ✓, still locked (with the
  explanation), regressions, and context shifts, with synced limits
  following automatically.
- **Connects your apps** (🔌 Connections tab): OpenCode gets full config
  sync through a comment-preserving JSONC editor (hand-edits survive,
  removals are comment-outs, every write has numbered backups + a restore
  action); any other OpenAI-compatible app gets the base URL, the measured
  model list, and copy-paste snippets (curl / Python SDK / models JSON).
- **Explains itself**: every unavailable model has a "Why?" button — plain
  language, the exact error line as evidence, and remedy buttons. No "see
  logs".
- **Advises on your build** (Server → Check My llama.cpp): compares your
  binary against your checkout and upstream (plus one quiet daily fetch so
  the Server tab always knows when a newer build exists — it observes your
  checkout, never modifies it), detects your toolchains
  (CUDA / Vulkan / ROCm with per-backend checkboxes), names the models a
  rebuild would unlock, and runs the guided rebuild itself — fast-forward
  pull only, every backend passed to cmake explicitly so stale caches can't
  drift a build.
- **Coexists with Ollama** as a peer: shows what the daemon holds in VRAM
  and warns on contention. It observes Ollama; it never manages it. The same
  discipline applies to any llama-server this app didn't start.

## Requirements

- **Linux.** (The GUI needs the usual desktop libs — see Install.)
- **llama.cpp's `llama-server`, build b10216 or newer** (router mode:
  `--models-preset`, hot-swap, `--fit`). Don't have one? The app builds it
  for you: Server menu → Check My llama.cpp → Managed llama.cpp → Set up.
- A GPU helps enormously but isn't required; NVIDIA/AMD/Intel all work via
  the backend the Build Advisor detects.
- OpenCode is **optional** — sync simply skips when it isn't installed, and
  any OpenAI-compatible app connects via the Connections tab.

## Install

```sh
# from crates.io
sudo apt-get install libgtk-3-dev libxkbcommon-dev libwayland-dev \
  libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev   # Debian/Ubuntu
# Fedora: gtk3-devel libxkbcommon-devel wayland-devel libxcb-devel
# Arch:   gtk3 libxkbcommon wayland libxcb
cargo install modelsteward

# or grab a prebuilt tarball from the GitHub releases page:
#   https://github.com/unclebucklarson/modelsteward/releases

# or from source
git clone https://github.com/unclebucklarson/modelsteward && cd modelsteward
cargo run --release            # GUI; CLI: cargo run --release -- --help
```

## Fifteen-second glossary

**GGUF** the model file format llama.cpp runs · **quant** (Q4_K_XL, q8_0…)
how compressed the weights are — smaller = less VRAM, some quality cost ·
**context** how many tokens a conversation can hold · **KV cache** the
memory holding that conversation (its precision is tunable) · **pp / tg**
prompt-processing / token-generation speed, tokens per second · **prefill**
reading your prompt before the first output token · **MoE / A3B** a
Mixture-of-Experts model; "A3B" = 3B parameters active per token — these
can run well even when far bigger than your VRAM · **mmproj** a model's
vision companion file · **router mode** one llama-server hot-swapping many
models on one port · **preset** the generated `router.ini` describing them
· **--fit** llama.cpp auto-sizing context to your VRAM · **speculation**
drafting tokens cheaply and letting the model confirm them.

## Quick start

```sh
cargo run --release            # the GUI
```

First run: **File → Set Up Everything** — starts the router, measures
anything unmeasured, syncs OpenCode. Or entirely from the CLI:

```sh
modelsteward --setup       # the one-shot: start + measure + sync
modelsteward --scan        # what's on this machine (JSON)
modelsteward --preset      # write ~/.config/modelsteward/router.ini
modelsteward --start       # router on :8080
modelsteward --calibrate   # measure new/stale models (add `force` for all)
modelsteward --bench       # speed baselines, new/stale (or: --bench <id>, add `force`)
modelsteward --trial <id>  # measured config trial ([spec|ub|kv|load|dials|moe|vision|cache|ckpt|slots]; `keep <variant>` applies)
modelsteward --quality <id>  # eval battery + tool + multi-hop agent-loop probes
modelsteward --meter       # token ledger: what your models actually served (today|24h|7d)
modelsteward --report      # shareable findings report (sanitized md + JSON)
modelsteward --sync        # write measured limits into opencode.json
modelsteward --verify-rebuild  # after a rebuild: restart, re-measure, report
modelsteward --advise      # build advisor report
modelsteward --status      # router + per-model state (JSON)
modelsteward --stop
```

Any OpenAI-compatible app connects with just the base URL (the Connections
tab has copy-paste snippets):

```
http://127.0.0.1:8080/v1       # api_key: anything, it's ignored
```

App settings (scan directories, ports, llama-server binary with browse +
detected-installs picker, max loaded models) live in the Settings tab,
persisted at `~/.config/modelsteward/config.json`. Measurements live at
`~/.local/state/modelsteward/measurements.json`.

## Surviving logout: systemd user unit

The router starts manually by design (Start Router button / `--start`).
To have it survive logout instead: `--install-service` (or *Server →
Install systemd User Unit* in the GUI) writes
`~/.config/systemd/user/llamacpp-router.service`, then:

```sh
systemctl --user daemon-reload && systemctl --user enable --now llamacpp-router
```

The unit runs llama-server with the app's preset file, which is also the
ownership handshake: the app recognizes that process as its own (Stop works
on it), while any other llama-server stays strictly observe-only.

## Glossary

- **shelf** — your own model directories (`~/models` plus whatever you add
  in Settings): locally stored, manually managed, touched by no other tool.
  The Library's "→ shelf" button archives a cache/Ollama model here
  (hardlink when free, copy otherwise; never overwrites).
- **measured ctx** — the context llama-server's `--fit` actually settled on
  for that model on this machine, not the training figure in the file.
- **Feat badges** — 👁 vision (mmproj paired), ⚡ MTP layers present,
  🧬 embedding model.

## Design rules (short version)

- llama.cpp does the hard parts (router mode, `--fit` memory auto-sizing,
  multi-GPU splits); this app configures them and reports what they did.
- GPU state is always a list — multi-GPU is not a special case.
- Never touch a server we didn't start.
- Config limits come from measurement, not from GGUF headers — and the
  filename's quant token outranks the header's stamp when they disagree
  (dynamic quants lie in `general.file_type`).
- Removals are comment-outs with a note, never deletions; the app never
  deletes model files at all.

See [PLAN.md](PLAN.md) for the founding design, [ROADMAP.md](ROADMAP.md)
for what's done and what's next, [docs/spikes.md](docs/spikes.md) for the
verified llama-server behavior this is built on, and
[docs/PROJECT-BRIEF.md](docs/PROJECT-BRIEF.md) for a general-audience
overview. Sibling project: `../modelwarden` (model inventory/backup —
storage truth; this app owns serving).

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT), at your option. Unless you explicitly state
otherwise, any contribution intentionally submitted for inclusion shall
be dual-licensed as above, without any additional terms or conditions.
