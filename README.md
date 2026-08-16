# llamacppCodeConf

Squeeze maximum performance out of local models for AI coding — llama.cpp +
[OpenCode](https://opencode.ai) — **without needing to be a llama.cpp or
OpenCode expert**. A traditional desktop app (Rust + egui) for Linux, with a
scriptable CLI underneath.

## What it does

- **Discovers** your llama.cpp installations (version, backends, which build
  can actually see your GPUs) and every GGUF on the system — shelf
  directories *and* Ollama's blob store, whose blobs are raw GGUFs that
  llama.cpp serves directly, no duplication.
- **Runs llama-server in router mode**: one long-lived server that lists all
  your models and hot-swaps them on demand (a 27B swap takes ~12s on a
  3090 Ti). OpenCode just names a model; the router does the rest.
- **Generates the router preset** with coding-agent-optimized defaults:
  `np = 1` (the whole context window goes to your agent) and `q8_0` KV cache
  (measured on real hardware: roughly double the usable context of f16 for
  the same VRAM). Every default is overridable per model.
- **Measures, never guesses**: *calibration* loads each model once and
  records the context llama-server's `--fit` actually settled on for your
  machine — usually far below the GGUF header's training figure.
- **Syncs `opencode.json`** with those measured limits through a
  comment-preserving JSONC editor: your hand-edits and comments survive,
  existing entries get minimal patches, orphans are commented out (never
  deleted) and only when you say so, and every write is backed up.
- **Coexists with Ollama** as a peer: shows what the daemon holds in VRAM
  and warns on contention. It observes Ollama; it never manages it. The same
  discipline applies to any llama-server this app didn't start.

## Quick start

```sh
cargo run --release            # the GUI
```

Or entirely from the CLI:

```sh
llamacppcodeconf --scan        # what's on this machine (JSON)
llamacppcodeconf --preset      # write ~/.config/llamacppcodeconf/router.ini
llamacppcodeconf --start       # router on :8080
llamacppcodeconf --calibrate   # measure every model's real context (slow, once)
llamacppcodeconf --sync        # write measured limits into opencode.json
llamacppcodeconf --status      # router + per-model state (JSON)
llamacppcodeconf --stop
```

Then point OpenCode at it (the sync step writes this provider for you):

```jsonc
"llamacpp": {
  "npm": "@ai-sdk/openai-compatible",
  "options": { "baseURL": "http://127.0.0.1:8080/v1" }
}
```

## Surviving logout: systemd user unit

`--install-service` (or *Server → Install systemd User Unit* in the GUI)
writes `~/.config/systemd/user/llamacpp-router.service`, then:

```sh
systemctl --user daemon-reload && systemctl --user enable --now llamacpp-router
```

The unit runs llama-server with the app's preset file, which is also the
ownership handshake: the app recognizes that process as its own (Stop works
on it), while any other llama-server stays strictly observe-only.

## Design rules (short version)

- llama.cpp does the hard parts (router mode, `--fit` memory auto-sizing,
  multi-GPU splits); this app configures them and reports what they did.
- GPU state is always a list — multi-GPU is not a special case.
- Never touch a server we didn't start.
- opencode.json limits come from measurement, not from GGUF headers.
- Removals are comment-outs with a note, never deletions.

See [PLAN.md](PLAN.md) for the roadmap (build advisor, performance lab) and
[docs/spikes.md](docs/spikes.md) for the verified llama-server behavior this
is built on.
