# llamacppCodeConf

Squeeze maximum performance out of local models — llama.cpp serving any
OpenAI-compatible app, with first-class [OpenCode](https://opencode.ai)
integration — **without needing to be a llama.cpp expert**. A traditional
desktop app (Rust + egui) for Linux, with a scriptable CLI underneath.

The ethos in three words: **measured, not guessed.** The app loads your
models on your hardware, records what they actually do (real context after
memory fitting, real tool-calling behavior), and configures everything
downstream from those measurements.

## What it does

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
- **Trials config changes instead of trusting folklore** (each row's
  Trial → Run, or `--trial <id> [spec|ub]`): baseline vs candidates,
  server-timed on fixed prompts, ending in a verdict dialog with the full
  measured table. Strict rules pick winners that cost nothing; tradeoffs
  the rules reject (say, +50% prefill for context you'd never use) are
  surfaced as explicit "your call" choices instead of dying silently.
  Keeping a winner writes the override, reloads the router, and syncs the
  honest new limits in one step. First campaigns: `ngram-simple`
  speculation adopted on four models (up to +118% generation on
  edit-heavy agent work, zero VRAM) while the folklore-favorite classic
  draft model was measured and rejected (on a single 24GB card it costs
  96% of your context and is 6x slower — `docs/spikes.md`). Measurements
  that collide with your own coding session are skipped as "server busy",
  never recorded as model failures.
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

## Quick start

```sh
cargo run --release            # the GUI
```

First run: **File → Set Up Everything** — starts the router, measures
anything unmeasured, syncs OpenCode. Or entirely from the CLI:

```sh
llamacppcodeconf --setup       # the one-shot: start + measure + sync
llamacppcodeconf --scan        # what's on this machine (JSON)
llamacppcodeconf --preset      # write ~/.config/llamacppcodeconf/router.ini
llamacppcodeconf --start       # router on :8080
llamacppcodeconf --calibrate   # measure new/stale models (add `force` for all)
llamacppcodeconf --bench       # speed baselines, new/stale (or: --bench <id>, add `force`)
llamacppcodeconf --trial <id>  # measured config trial ([spec|ub]; `keep <variant>` applies)
llamacppcodeconf --sync        # write measured limits into opencode.json
llamacppcodeconf --advise      # build advisor report
llamacppcodeconf --status      # router + per-model state (JSON)
llamacppcodeconf --stop
```

Any OpenAI-compatible app connects with just the base URL (the Connections
tab has copy-paste snippets):

```
http://127.0.0.1:8080/v1       # api_key: anything, it's ignored
```

App settings (scan directories, ports, llama-server binary with browse +
detected-installs picker, max loaded models) live in the Settings tab,
persisted at `~/.config/llamacppcodeconf/config.json`. Measurements live at
`~/.local/state/llamacppcodeconf/measurements.json`.

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
