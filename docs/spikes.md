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

## Spike 2 — fit behavior: CONFIRMED (run after GPU freed, same day)

Real-model run: Qwen3.6-27B quants on the 24GB RTX 3090 Ti, via the router.

- **Load: 13s** for the 19GB UD-Q5_K_XL (cold-ish). **Full swap
  (unload + load other quant): 12s.** Model switching is fast enough to feel
  Ollama-like.
- **`--fit` settles context automatically and reports it.** UD-Q5_K_XL with
  `q8_0` KV cache: **n_ctx 72,960** (vs 262,144 train). Q5_K_M with default
  f16 KV: **n_ctx 36,096**. Same-size models, KV quantization ≈ doubled usable
  context — north-star exhibit A, and exactly the kind of default the app
  should set.
- VRAM after fit: ~22.5GB of 24.5GB both times — the default 1024 MiB
  `--fit-target` margin is honored.
- **The settled `n_ctx` is available live** at
  `/props?model=X` → `default_generation_settings.n_ctx`. This is the number
  to write into opencode.json `limit.context` (NOT `n_ctx_train`). Caveat:
  instances default to 4 parallel slots with unified KV, so n_ctx is shared
  across concurrent requests; OpenCode as sole client effectively gets all
  of it. Consider `-np 1` in generated presets for single-user coding.
- **Tool calling through the router works** (OpenAI-style `tools`,
  `finish_reason: "tool_calls"`, well-formed arguments). Note Qwen3.6
  reasons before calling tools — a small `max_tokens` truncates the call
  mid-arguments; don't cap output tightly in agent use.
- Load status polling behaved as documented (async load/unload, `loading` →
  `loaded`; unloaded instance freed VRAM before the next load completed).

## Spike 5 — speculative decoding on a single 24GB card (2026-08-24, b10454)

Target: qwen3.8-27b-ud-q4_k_xl (17.9GB weights + mmproj). Server-timed
generations (temperature 0, cache_prompt off), two novel-code prompts +
one rewrite prompt (add docstrings to a 10-function module).

| scenario            | baseline | 4B classic draft | ngram-simple |
|---------------------|----------|------------------|--------------|
| settled ctx (--fit) | 116,224  | 4,096            | 117,248      |
| novel code tg       | 38.6 t/s | 6.4 t/s          | 40.4 t/s     |
| rewrite tg          | 39.2 t/s | —                | 86.8 t/s     |

- **Classic draft (Qwen3.5-4B) REJECTED on this hardware:** the 2.8GB
  draft collapses `--fit` context to 4,096 and the leftover-VRAM draft
  placement makes it 6x SLOWER (43% acceptance can't save a slow
  drafter). Not viable until a second GPU exists.
- **Gotcha:** a draft file carrying MTP tensors makes llama-server
  auto-select `draft-mtp` mode, which asserts the draft's MTP width
  matches the TARGET's hidden width (4B=2560 vs 27B=5120 → crash).
  `spec-type = draft-simple` forces classic mode.
- **ngram-simple ADOPTED for the daily driver:** zero VRAM, zero context
  cost, +5% on novel code, **+121% on edit/rewrite work** (45%
  acceptance) — the agent-workload sweet spot. Enabled via override
  `spec-type = ngram-simple` on qwen3.8-27b-ud-q4_k_xl.
- Untried variants for the M7 harness: ngram-map-k/k4v/mod/cache,
  draft-eagle3/dflash/dspark (need matching aux models).

### Spike 5 addendum — fleet campaign via the trial harness (same day)

`--trial` run across the six daily models (baseline / ngram-simple /
ngram-map-k4v / ngram-mod each; server-timed, verdict rules in
core/trial.rs):

| model                | base rewrite | best              | verdict       |
|----------------------|--------------|-------------------|---------------|
| qwen3.8-q4           | 39.6         | simple 80.5 +103% | KEPT simple   |
| qwen3.8-q5           | 37.8         | simple 82.4 +118% | KEPT simple   |
| laguna-xs-2.1        | 174.0        | simple 224.2 +29% | KEPT simple   |
| north-mini-code      | 173.6        | simple 248.0 +43% | KEPT simple   |
| ornith-35b           | 168.5        | k4v 181.7 +8%     | baseline      |
| qwen3.6-ud-q5        | 38.5         | +9.9%             | baseline      |

Findings: ngram-simple beats map-k4v everywhere despite LOWER acceptance
(k4v accepts more but shorter/cheaper spans — acceptance rate is a bad
proxy for real speed, measure the speed). Speculation helps even
180 t/s MoE models on copy-heavy work. Qwen3.6 accepts drafts far less
than Qwen3.8 (15% vs 45% on identical prompts) — generation-level
behavioral difference, invisible in specs.
