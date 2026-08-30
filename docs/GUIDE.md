# The first hour with modelsteward

A walkthrough from empty machine to a tuned, metered local model serving
your coding agent. Work through it top to bottom; each step says what
you should see, so it doubles as the release-test script
([RELEASE-CHECKLIST.md](RELEASE-CHECKLIST.md) runs this document).

Words in *italics* are defined in the README's fifteen-second glossary.
The in-app **Help → Tuning Guide** covers step ordering and vocabulary.

## 0 · Install and prerequisites

Follow the README's Install section (`cargo install modelsteward`, a
release tarball, or from source — system deps listed there). You need:

- Linux, and a GPU helps but isn't required.
- A `llama-server` binary with router mode (build ≥ b10216). **Don't
  have one?** The app can build it for you — see step 2.
- At least one `.gguf` model file, or Ollama with models pulled, or a
  HuggingFace cache. No models yet? `ollama pull qwen3.5` (or any
  model) is the fastest start.
- OpenCode is optional. Without it, the Connections tab still serves
  any OpenAI-compatible app.

Launch from a normal terminal (`modelsteward`), not a snap-packaged
IDE's terminal — snap redirects HOME and the app will tell you if it
had to rescue files from that.

## 1 · Point it at your models

Open **🔧 Settings**.

- **Model directories**: the default is `~/models`. Add the
  directories that hold your `.gguf` files. The Ollama store and
  HuggingFace cache are found automatically — don't add those.
- **llama-server binary**: this is the ONE place that chooses what
  serves. "Auto" picks the best discovered install; the list shows
  every found install including app-managed builds. If the list is
  empty, the pane says so and points at the fix.

**You should see:** after Save & Rescan, the 📚 Library fills with one
row per model, from every source, deduplicated.

## 2 · No llama.cpp yet? Let the app build one

**🖥 Server → Check My llama.cpp** (the Build Advisor). It reports
what you have and what a rebuild would unlock. "Set up managed
checkout" clones llama.cpp into the app's own data directory, builds
it with auto-detected backends (CUDA/Vulkan/ROCm), and archives the
binary. Then select it in Settings — the Build Advisor makes builds,
Settings selects them; the two never mix.

**You should see:** the build narrated in the activity log, then the
new binary in Settings' list with a green "current" marker.

## 3 · Set Up Everything

**File → Set Up Everything** (also a button on empty panes). One shot:
start the router if it's down → *measure* every unmeasured model →
sync opencode.json. It's narrated line by line and can take a few
minutes per model (each one loads once).

**You should see:** the status bar spinner with a live label, then
"measuring finished: N measured, M known failures", then a sync
summary. If OpenCode isn't installed, the sync step says it skipped
and why — that's fine.

## 4 · Read the Library

**📚 Library** is a master-detail view:

- The **grid** is identity only: Model, Source, Size, Quant, Measured
  ctx, Speed, Server state, and whether it's in OpenCode. Type in the
  filter box to narrow it; rows sort problems-last by default. Red
  model names have a problem; dimmed ones are disabled.
- **Click a row** for the detail panel: the full advice line in color,
  quality scores (once the quality probe has run), the action buttons,
  and the measurement history.
- **Why?** on any non-green model explains the problem in plain
  language with a remedy button — out-of-date build, partial file,
  out of memory, or just "never measured".
- **Disable** keeps a row visible but stops everything acting on it —
  right for models your llama.cpp can't load (llama.cpp-incompatible
  conversions stay red until then).

Key columns: **Measured ctx** is the context window `--fit` actually
settled on for *your* VRAM — not the model card's number. **Speed** is
measured pp/tg (read/write tokens per second).

## 5 · Your first Lab run

**⚡ Lab.** Pick your daily-driver model. Check **Measure** and
**Bench** first if the row lacks ctx/Speed numbers — the time
estimates need them.

- The ETA line tells you how long the selected campaigns will take,
  loudly if it's 90+ minutes (big MoE models can take hours).
- **Follow the sequence** (Help → Tuning Guide): measure + bench →
  placement first for big MoE models (`moe` menu) → speed menus
  (`spec`, `ub`, `kv`) → agent-turn menus → quality probe. The Lab
  warns if you race ahead of an unapplied placement winner.
- **Run selected campaigns.** The busy label shows progress
  ("moe trial (3/7)"); Cancel sits next to Run and stops at the next
  safe boundary with your config restored.

**Reading results:** each menu's table has hover definitions on every
header. A green ★ row is a winner waiting for your Apply; a blue ✓ row
is already applied and serving. Winners are *offered, never
auto-applied* — guard-rejected candidates show as "your call" with the
gain and the cost spelled out.

**Where applied settings go** (they are NOT in opencode.json): Apply
writes the winner into this app's `config.json` as that model's
override, rewrites the router *preset* (`router.ini`), and reloads the
router. llama-server receives those flags the next time it loads the
model. opencode.json only ever gets measured *results* — the context
limit and capabilities — never the knobs.

After applying a winner, re-run **Measure** on that model so the
context limit in opencode.json matches what the new config actually
serves.

## 6 · Connect your agent

**🔌 Connections.** The top panel serves any OpenAI-compatible app:
base URL, the measured model list, and copy-paste snippets. Below it,
the OpenCode section mirrors `opencode.json` — every entry with its
sync state (✔ synced, ⟳ differs, ✖ can't load, ? never measured).
Every write is backed up first; **Tools → Restore opencode.json From
Last Backup** undoes the latest one.

**You should see:** your measured models in OpenCode's picker with
context limits that match the Library's Measured ctx column.

## 7 · Watch the meter

**🖥 Server** shows the loaded model, the token ledger ("Meter today"),
and the Ollama peer if one runs. After a coding session:

    modelsteward --meter today

Fleet totals, cache hit rates, and the measured cost line — your
electricity price vs the configured cloud price, priced from this
machine's own J/token measurements once trials have run.

## 8 · When something goes wrong

- The **error banner** and red log lines say what failed; **Why?** on
  the model explains it with a remedy.
- `modelsteward --config` prints the config path and effective
  settings; `--status` the router state as JSON.
- The router log: **Tools → Open Router Log**.
- CLI exit codes: 0 ok, 1 error, 2 bad usage, 3 partial failure.
  Progress goes to stderr, results to stdout.

## CLI equivalents

Everything above scripts. `modelsteward --help` is the single source;
the short map:

| GUI | CLI |
|---|---|
| Set Up Everything | `--setup` |
| Measure | `--calibrate [force]` |
| Bench | `--bench [id] [force]` |
| Lab campaign | `--trial <id> <menu>` then `--trial <id> <menu> keep <variant>` |
| Quality probe | `--quality <id>` |
| Sync | `--sync` |
| Meter | `--meter [today\|24h\|7d]` |
| Build Advisor report | `--advise` |
