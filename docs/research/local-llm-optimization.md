# Optimizing local LLM serving — external research, graded

Compiled 2026-09-03 from a web survey of llama.cpp documentation,
maintainer statements, benchmarks, and practitioner writeups, then
checked against this repository's own code and measurements.

*(Scott asked for this as `Research_mid.md`; named for what it is.
This is the "we should have done this first" document — the outside
view we built the tool without. It is not a style guide: several
popular tips turn out to be unsupported, and those are called out as
loudly as the good ones.)*

## How to read this

Every claim carries a source grade. This matters more than usual here,
because this field's blog layer confidently repeats numbers nobody
measured:

| Grade | Meaning |
|---|---|
| **[DOC]** | Official llama.cpp documentation |
| **[MAINT]** | A maintainer's own statement (PR, issue, discussion) |
| **[BENCH]** | Someone published measurements and their method |
| **[VENDOR]** | Self-reported by the party publishing the artifact |
| **[FOLKLORE]** | Widely repeated, no traceable study |
| **[OURS]** | Measured on this machine by modelsteward or modellab |

**Standing warning:** several llama.cpp defaults moved within the last
12 months (context shift, `-sps`, `--spec-draft-n-max`,
`--ctx-checkpoints`). Anything we hardcode should be read back from the
binary's `--help`, not assumed. We already do this for sampling
defaults; the same discipline applies to everything below.

---

## 1. The two findings that should change behaviour

### 1.1 K and V are not equally quantizable — and we got this right by instinct

This is the best-evidenced finding in the whole survey, and it
contradicts the common practice of setting both KV types the same.

**[BENCH]** ([discussion #23470](https://github.com/ggml-org/llama.cpp/discussions/23470))
— Qwen2.5-7B, KL-divergence vs f16 baseline:

| K / V | mean KLD | top-p match |
|---|---|---|
| `q8_0` / `q4_0` | 0.004766 | 96.70% |
| `f16` / `q4_0` | 0.004047 | 96.89% |
| `q4_0` / `q4_0` | **5.508897** | **11.65%** ← collapse |

**[BENCH]** A second contributor, 500 ARC-Challenge questions,
deterministic decoding: `q4_0` **on K alone** dropped a model from 92%
to **24.2%**; `q4_0` **on V alone** changed **1 answer in 500**.

**[DOC]** ([function-calling.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md))
adds the agent-specific consequence in one sentence: *"Extreme KV
quantizations (e.g. `-ctk q4_0`) can substantially degrade the model's
tool calling performance."* For a coding-agent tuner this is arguably
the most important line in the llama.cpp docs — KV precision is a
**tool-reliability** decision, not only a memory one.

**[OURS] — verified in our source today:** our default is `q8_0` for
both (`router.rs:90`), which is on the safe side of K; and our `kv`
trial menu races **`cache-type-v q4_0` only** (`trial.rs:65-71`) — it
has never offered to quantize K. That is exactly the evidence-backed
direction. It was an unexamined choice until now; it is now a defended
one, and it should not be "generalized" to K by a future contributor.

**Contested, and we are positioned to settle it:** **[BENCH]**
([Blob Blog, Mar 2026](https://blog.teamblobfish.com/posts/benchmarking-llama-server/))
found on **Metal** that *mismatched* K/V types cost ~40% throughput,
while matched types showed no difference. **[MAINT]** confirms
non-whitelisted asymmetric combinations can **fall back to CPU**,
erasing the benefit. So asymmetric KV is quality-correct but possibly
performance-negative *depending on backend*. Our `kv` menu measures the
actual speed and gates on fidelity, so on any given machine we answer
this empirically rather than picking a side — which is the right
posture for a contested claim.

### 1.2 The n-gram benchmarking trap — which our harness happens to avoid

**[BENCH — negative result, the most instructive source found]**
([defilan, Apr 2026](https://dev.to/defilan/i-tested-speculative-decoding-on-my-home-gpu-cluster-heres-why-it-didnt-help-3ej6)):
testing speculative decoding with **repeated identical prompts** showed
a **4.75× speedup** that was entirely n-gram cache memorization. With
diverse prompts the same setup showed **0%**.

The same post is a useful humility check on speculation generally: on
2× RTX 5060 Ti, `ngram-mod` produced 0% on a 26B MoE and +1% on a 32B
dense, diagnosed as *"the bottleneck is memory bandwidth, not
compute"* — speculation trades spare compute for fewer sequential
passes, and there was none spare to trade.

**[OURS] — checked today:** our trial harness reuses fixed prompts
across variants, which is exactly the shape that manufactures this
illusion. We are safe for a structural reason worth writing down:
`trial.rs:1676-1705` gives every variant a **fresh model process** —
preset rewrite → `router::reload` → load → measure → `unload_model` —
and llama.cpp's router runs each model in its own process
(**[MAINT]**, [model-management blog](https://huggingface.co/blog/ggml-org/model-management-in-llamacpp)),
so no n-gram state survives between variants. `timed_generation` also
sends `cache_prompt: false`, so the prompt cache can't leak either.
**If anyone ever "optimizes" the harness by reusing a loaded model
across variants, this protection dies silently and every speculation
verdict becomes fiction.**

Our own measured `ngram-simple` win (+121% rewrite on a 27B) is
plausible on mechanism rather than artifact: **[DOC]**
([speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md))
names *"iterating over a block of text/code"* as the case n-gram modes
are for, and our rewrite probe regenerates given code — the model is
legitimately re-emitting tokens it can predict.

**Corollary we should adopt:** acceptance rate is a **diagnostic**, not
a success metric. **[BENCH]** ([#25198](https://github.com/ggml-org/llama.cpp/discussions/25198),
130+ real sessions) found *longer, more selective* drafts accept a
smaller fraction and win more (~20%). We learned the same thing
empirically in 2026-08 ("acceptance rate is a BAD proxy for speed" —
ROADMAP); it's good to see it independently confirmed.

---

## 2. Where the survey validated existing decisions

Recorded so nobody re-litigates them, and so the reasons are written
down rather than remembered:

- **`--models-max 1`.** **[DOC]** default is 4 with LRU eviction.
  **[ISSUE]** ([#11681](https://github.com/ggml-org/llama.cpp/issues/11681))
  `--ctx-size` is divided by `--parallel`, so a single-agent session
  wants `-np 1` to get the whole context. Our default of 1 model / 1
  slot is right for the workload, and now has a citation.
- **Not forcing `--flash-attn`.** It is `auto` and effectively on
  wherever supported. Benchmarks claiming "+X% from enabling `-fa`" on
  current builds are measuring noise. We correctly pass it never.
- **Context shift left off.** **[MAINT]** ggerganov disabled it by
  default ([PR #15416](https://github.com/ggml-org/llama.cpp/pull/15416))
  because it *"can destroy the structure of the chat template,
  degrading the quality"* and is confusing on `/chat/completions`. For
  agents a hard failure at context exhaustion beats silent corruption.
  **We should never "helpfully" re-enable it.**
- **Measuring `--cache-reuse` rather than trusting it.** The
  ubiquitous `--cache-reuse 256` is **[FOLKLORE]** — no study behind
  the number. We ship 256 as a default *and* race it in the `cache`
  trial menu, which is the correct treatment of an unsupported
  constant.
- **Per-model reasoning effort in the preset.** **[DOC]**
  ([preset.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/preset.md))
  shows `chat-template-kwargs = {"reasoning_effort": "high"}` in a
  router INI — the same place we put it. ⚠️ But see §4.1: the kwarg
  route is being deprecated in favour of `--reasoning`.

---

## 3. The measurement discipline this field mostly lacks

This section is the one most relevant to our product thesis, and it is
where the external material is thinnest — which is itself a finding.

**[DOC]** ([llama-bench README](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)):
*"The measurements with llama-bench do not include the times for
tokenization and for sampling."* Server-observed throughput is
therefore **legitimately lower** than llama-bench for the same config.
Mixing the two produces phantom regressions. This is precisely the
"two answers to two questions" boundary we drew with modellab, and it
has a documentary basis.

**`-d/--n-depth` is the flag that matters** — *"run tests at a
specified context depth, prefilling the KV cache with `<n>` tokens"*.
Default 0 benches an empty cache, a state no coding agent is ever in.
**[OURS]** modellab measured a 27B at 38.25 t/s empty and 28.99 t/s at
its settled context — 24% apart; we adopted `-d` on 2026-09-02
(commit `fc964b8`).

**Why generation slows with occupancy — and a myth to stop repeating.**
Each generated token attends over the whole resident cache, so decode
cost is **linear in KV depth**. Multiple blogs assert it is "quadratic";
that describes *total prefill* over a prompt, not per-token decode. If
we ever model this curve, model it linear. **[BENCH]** one published
sweep: 345 tok/s at 8K → 69 tok/s at 131K.

**Documented measurement mistakes** worth encoding in our checklist:
prompt-cache/n-gram memorization skew (§1.2); server ≠ llama-bench;
thermal drift; other GPU tenants; cold vs warm; build drift (results
are only comparable within a build — which is why our `bench_build`
staleness signal exists).

🚩 **Numbers to NOT encode.** Widely circulated thermal-drift figures
("12–18% swing", "3 reps → 6–8% stddev") trace to content farms, not
studies. The phenomenon is real and physical; the decimals are
invented. Same for the quantization speed/PPL deltas in §5.

---

## 4. Things we should probably act on

Ranked by value; none are urgent, all are cheap. Logged to ROADMAP.

### 4.1 The reasoning-kwarg deprecation (version-sensitive, affects us now)

**[ISSUE]** ([#23351](https://github.com/ggml-org/llama.cpp/discussions/23351))
builds **≥ b8322** emit: *"Setting 'enable_thinking' via
`--chat-template-kwargs` is deprecated. Use `--reasoning on` /
`--reasoning off` instead."*

We currently send **both** `enable_thinking` and `reasoning_effort` as
chat-template kwargs from `aiadvisor`, and write `reasoning-effort` into
presets. The preset route is fine (**[DOC]** still shows it), but the
advisor's kwarg route will start warning. **Action:** branch on build
version, or move to `--reasoning`. Note the structural limit while
we're here: **`--reasoning` is a server-startup flag with no
per-request toggle**, so serving thinking and non-thinking from one
model needs two preset entries — relevant if we ever offer that.

### 4.2 Prompt-prefix stability is the highest-leverage agent lever

**[PRACTITIONER, mechanism sound]**
([case study, Jun 2026](https://www.mykolaaleksandrov.dev/posts/2026/06/claude-code-llamacpp-prompt-cache-fix/)):
an attribution block at the **head** of an agent's system prompt
changed the prefix every turn, so llama-server logged *"forcing full
prompt re-processing"* on every request. Removing it flipped the log to
*"restored context checkpoint"* — ~511 ms for 212 tokens. This matches
[PR #16391](https://github.com/ggml-org/llama.cpp/pull/16391)'s design
intent exactly.

The generalizable rule: **anything varying at the head of the prompt —
timestamps, session IDs, non-deterministically ordered tool lists —
destroys prefix reuse and turns every turn into a full prefill.** At
100k context that dwarfs any batch-size win.

**[OURS] — checked today:** those two log markers appear **zero times**
in our 1 MB router.log; our child verbosity filters them (the same
reason the parked "FA-engaged check" was abandoned). **But we already
compute the equivalent signal by another route**: `evidence.rs` derives
reuse from per-turn token counts, and the meter reports it — Scott's
Minecraft session ran at **91% of prompt served from cache**. We
arrived at the field's top recommendation independently, via a metric
that doesn't depend on log verbosity. Worth keeping and worth
surfacing more prominently; a sustained drop in that percentage is the
symptom this case study describes.

### 4.3 `llama-fit-params` for advice without loading

**[DOC]** ([fit-params README](https://github.com/ggml-org/llama.cpp/blob/master/tools/fit-params/README.md))
prints the `-c`, `-ngl`, `-ts`, `-ot` a `--fit` run would choose, from
the model file and free VRAM, **without serving**. modellab flagged the
same tool. Our advice column could quote llama.cpp's own projection
instead of a file-size heuristic. Worker territory (initializes CUDA).

⚠️ **[ISSUE]** ([#18066](https://github.com/ggml-org/llama.cpp/issues/18066))
`--fit` is a projection, not a guarantee — a case exists of it
projecting a fit and then dying with `cudaMalloc failed`. Anything we
build on it must verify the server actually came up, which our
`fetch_settled_ctx` already does.

### 4.4 Settle the `-ncmoe` direction question

🚩 **Direct contradiction:** **[DOC]** says `--n-cpu-moe N` keeps the
MoE weights of the **first** N layers on CPU; a practitioner guide
(**[Jan 2026](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)**)
says it counts from the **highest**-numbered layers. One is wrong or
the behaviour changed. We ship `ncpu-moe-32` as a *measured* winner on
the 80B, so our verdict is safe either way — but if we ever explain the
flag in the UI, we would be repeating a claim we haven't verified.
Settleable by reading tensor placement in a server log.

---

## 5. Quantization: the consensus is weaker than it sounds

The best controlled public study found: **[BENCH]**
([arXiv 2601.14277](https://arxiv.org/html/2601.14277v1), Jan 2026) —
13 schemes on Llama-3.1-8B across GSM8K, HellaSwag, IFEval, MMLU,
TruthfulQA + WikiText-2. Findings: Q5_0 highest average; Q4_K_S the
balanced pick at ~71% size reduction.

**Limits worth stating plainly:** one model, one 8B checkpoint, **no
coding benchmark**, and **no imatrix/dynamic/UD quants evaluated at
all**. Its Q5_0 > Q5_K_M result runs contrary to K-quant consensus and
may be an artifact.

**The familiar ladder** (Q4_K_M default → Q5_K_M with headroom → Q6_K →
Q8_0 for coding) is **[FOLKLORE]** — near-universal, untraceable to a
controlled study. The *shape* (diminishing returns above ~5 bpw,
growing bandwidth cost) is sound; the specific numbers circulating
("Q8_0 is 29% slower", "PPL delta 0.0531") have no published method.
**Do not encode them.**

**Unsloth UD/dynamic quants** — which dominate Scott's fleet — are
**[VENDOR]**: the "99.9% KL divergence, SOTA Pareto" claims are
self-reported with no independent replication found. The *mechanism* is
checkable and plausible (promoting important tensors to higher bit
widths). Also practical: **UD file sizes are not comparable to stock
naming**, so size-matched comparisons must measure bytes. Reasonable
default choice; never quote the vendor's accuracy numbers as evidence.

**imatrix**: well-supported at 2–4 bpw, **no controlled evidence found**
at Q4_K_M and above.

---

## 6. System layer (hardware, thermals, VRAM)

*Pending: a second research pass covering GPU system tuning, VRAM
contention, thermals, and the 8 GB-class question for the G15. Will be
appended here when it completes; the sections above stand on their own.*

---

## 7. What this changes for us — summary

**Already right, now defended:** K/V asymmetry direction, `--models-max 1`,
never forcing `-fa`, context shift off, measuring `--cache-reuse`
instead of trusting 256, fresh process per trial variant, treating
acceptance rate as a diagnostic.

**Already fixed because of the parallel modellab work:** benching at
realistic KV depth (`-d`), recording free VRAM as a measurement
condition, and dropping the report's "measured … not estimated"
overclaim.

**Worth doing:** the reasoning-kwarg deprecation branch (§4.1), quoting
`llama-fit-params` in advice (§4.3), settling `-ncmoe`'s direction
before explaining it (§4.4), and surfacing prompt-cache health more
prominently (§4.2).

**The meta-lesson.** Most of this field's tuning advice is folklore
with invented decimals. The defensible parts are (a) llama.cpp's own
documentation and maintainer statements, and (b) measurements with
stated conditions. That is precisely the gap this tool exists to fill —
and the main risk to guard against is *us* becoming another source of
confident numbers without conditions. The corrections we shipped on
2026-09-02 were exactly that failure mode caught early.
