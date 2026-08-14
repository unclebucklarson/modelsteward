# Spike findings (M0)

Run 2026-08-14 against llama-server b10216 (`~/src/llama.cpp/build/bin`,
CUDA build) on the RTX 3090 Ti dev machine. Spikes 1/3/4 used the tiny
`stories260K.gguf` for mechanics because an unrelated llama-server
(unsloth studio's) held 21.6GB of VRAM; spike 2 needs a free GPU and is
deferred (see bottom).

## Spike 1 — router mode: CONFIRMED, architecture-viable

Launched `llama-server --models-preset <ini> --port 18080` (no `--model` →
router mode).

- **Preset INI works as documented.** Sections = model IDs; keys = CLI args
  (long, short, or env-var names). `[*]` globals propagate to every instance;
  per-model keys override (verified `cache-type-k = q8_0` landing in one
  instance's args and not others). Preset-exclusive keys: `load-on-startup`,
  `stop-timeout`.
- **Architecture: router spawns one child llama-server per loaded model** on
  an ephemeral port (`--port 0`) and proxies. `/models` shows each entry's
  exact child args and generated preset — ideal for a UI "effective config"
  view.
- **Sources merge.** Preset entries (`"source":"preset"`) appear alongside
  `LLAMA_CACHE` HF downloads (`"source":"cache"`, e.g. previously-fetched
  unsloth/gemma models). The cache entries carry modality metadata
  (gemma-4 lists image input).
- **Lifecycle endpoints:** `POST /models/load {model}` (returns `success`
  immediately — loading is async, poll `/models` status:
  unloaded/loading/loaded/sleeping/failed+exit_code), `POST /models/unload`
  (also async — instance lingers up to `stop-timeout`, default 10s).
  Requesting an unloaded model via `/v1/chat/completions` **autoloads it**
  (default; `--no-models-autoload` or `?autoload=false` to disable).
  `--models-max` (default 4) caps concurrently-loaded instances.
- **Routing:** POST bodies route on `"model"`; GET endpoints take
  `?model=<urlencoded>` — `/props?model=X` returns that instance's **actual**
  `n_ctx` (the number opencode.json limits should use).
- **Hot reload: `GET /models?reload=1`** re-reads sources — appended a new INI
  section while the server ran and it appeared without a restart. The app can
  therefore manage the preset file live.

## Spike 3 — Ollama blobs via symlink farm: CONFIRMED (listing/metadata)

- **This machine's Ollama is the system service**: store is
  `/usr/share/ollama/.ollama/models` (owner `ollama:ollama`), *not*
  `~/.ollama`. Blobs are world-readable (644). Discovery must check both
  locations (and `OLLAMA_MODELS`).
- Manifests at `manifests/registry.ollama.ai/<ns>/<name>/<tag>` (JSON); the
  GGUF blob is the layer whose `mediaType` ends `image.model`; digest
  `sha256:X` → blob file `blobs/sha256-X`.
- Symlinked a blob as `<name>.gguf`, added a preset entry pointing at the
  symlink, hot-reloaded: listed with correct metadata (header read through the
  symlink works). Full load not exercised (VRAM busy) — no reason to expect
  different behavior; verify when running spike 2.

## Spike 4 — opencode.json JSONC editing: SOLVED by harvest

`~/src2/opencode_configuration_tool` contains a comment-preserving JSONC
editor: `src/config/jsonc.rs` on the `jsonc-parser` crate (span-based
splices), with `add_model` / `merge_model` / `comment_out_model` /
`ensure_models_container`, atomic writes, and the "comment out orphans with a
removal note" convention already visible in the live config. Harvest this
module wholesale. (Note: that tool's GUI is FLTK, not egui as PLAN.md first
assumed — irrelevant to the harvest, the jsonc module is GUI-independent.)

llm_forge harvest map: `src/gguf.rs` (header parser), `src/library.rs`
(scan + Ollama store), `src/serve.rs`/`launcher.rs` (server guard),
`src/atomic.rs`.

## Spike 2 — fit behavior: DEFERRED (needs free GPU)

Blocked on VRAM during this run. Questions to answer when the GPU is free:
per-model `--fit-target`/`--fit-ctx` in preset sections; what `n_ctx` a
27B Q5 settles at on 24GB with q8_0 KV; load/swap wall-clock for real models;
tool-call round-trip through the router from OpenCode.
