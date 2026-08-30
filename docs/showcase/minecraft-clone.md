# Minecraft clone on a local 27B

**Status: in progress** — the coding session is live as this data is
captured; final numbers, the repo link, and a screenshot land when the
project does.

A Minecraft-style clone built with OpenCode driving
**Qwen3.8-27B-UD-Q4_K_XL**, served locally by modelsteward's router on
a single 24 GB consumer GPU.

## The serving config (measured into place, not guessed)

Every knob below was chosen by the app's trial harness — raced against
alternatives on this machine, winner applied by hand:

| Knob | Value | How it was chosen |
| --- | --- |---|
| Speculation | `ngram-simple` | spec trial: +121% rewrite speed at 45% draft acceptance, zero VRAM cost |
| Speculation dials | `n=8, m=64` | dials trial winner |
| Context | 111,360 tokens | what `--fit` measured on this GPU (not the model card's number) |
| Quality gate | evals 100% · tools 100% · agent loops 3/3 | the app's quality probe on this exact config |

## The session so far (token meter, day one — 2026-08-30 UTC)

| Metric | Value |
| --- | --- |
| Agent turns | 257 |
| Prompt tokens | 856,551 |
| Generated tokens | 115,976 |
| Prompt served from cache | ~30% |
| Shape | 7.4 prompt tokens per generated token |
| **Measured electricity cost** | **$0.03** (0.203 kWh at $0.15/kWh, marginal generation energy, NVML-sampled) |
| Same tokens at a $3/Mtok cloud output price | ~$0.35 |

The cost line is the app's own instrument: marginal joules per
generated token measured during trials, multiplied by this session's
metered tokens. Your electricity price is a config value; the cloud
price is yours to edit — the report labels both.

## Caveats, stated plainly

- One machine (a 24 GB consumer GPU), one project, one model. A case
  study, not a benchmark.
- Cache-heavy agent workloads flatter prompt throughput; cold prefills
  on a fresh conversation cost real seconds.
- Cloud comparison prices output tokens only, at a number you set.

## Reproduce it

```sh
cargo install modelsteward
modelsteward --setup                       # measure + sync
modelsteward --trial <your-model> spec     # race speculation modes
modelsteward --trial <your-model> dials    # tune the winner
modelsteward --quality <your-model>        # gate it
modelsteward --meter today                 # watch the receipts
```

Project repo: (link lands when published) · Screenshot: (pending)
