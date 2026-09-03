# Optimizing local LLM serving — external research, graded

Compiled 2026-09-03 from a web survey of llama.cpp documentation,
maintainer statements, benchmarks, and practitioner writeups, then
checked against this repository's own code and measurements.

*(Scott asked for this as `Research_mid.md`; named for what it is.
This is the "we should have done this first" document — the outside
view we built the tool without. It is not a style guide: several
popular tips turn out to be unsupported, and those are called out as
loudly as the good ones.)*

**Two sources, one document.** §§1–6 are the web survey. §7 reconciles
the practitioner guide Scott supplied himself
([`docs/human_research/`](../human_research/)) — better organised than
this one, and the source of an entire category we had missed: the
**host** layer (RAM speed, CPU governor, core placement, swap) as a
measurement precondition. §8 records where this desktop actually sits
against that checklist. Read §9 first if you want only the actions.

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
| **[FIELD]** | A practitioner's own measurements on one named machine — real numbers, one hardware sample (see §7) |

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

### 4.1 The reasoning-kwarg deprecation — checked, and it does NOT apply to us

**[ISSUE]** ([#23351](https://github.com/ggml-org/llama.cpp/discussions/23351))
builds **≥ b8322** emit: *"Setting 'enable_thinking' via
`--chat-template-kwargs` is deprecated. Use `--reasoning on` /
`--reasoning off` instead."* Our builds are well past that, so this
looked like an action item.

**[OURS] — it isn't, and the check took one command.** The deprecation
targets the **`--chat-template-kwargs` server flag**. We don't use it:
presets carry the dedicated `reasoning-effort` key (**[DOC]**, current),
and `aiadvisor` sends `chat_template_kwargs` as a **per-request JSON
field** in the chat-completions body — a different mechanism the
deprecation doesn't touch. Confirmed empirically: **zero** deprecation
lines in a 1 MB router.log from b10760.

*Left in this document deliberately, as a worked example of the
discipline it preaches: a plausible, well-sourced, version-matched
finding that still turned out not to apply. Grep before you refactor.*

Two real notes survive from it: **`--reasoning` is a server-startup
flag with no per-request toggle**, so serving thinking and non-thinking
from one model would need two preset entries; and the kwarg names
themselves are template-specific (we read ours from each model's own
chat template — see `core/reasoning.rs`), which is the right way to
avoid this whole class of problem.

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

## 6. System layer — GPU, memory, thermals

Researched 2026-09-03. Two sub-surveys completed and their content
recovered; the parent agent died to a 529 while summarizing, so the
**8 GB-class question is the one genuine gap** (§6.6).

### 6.1 ⚠️ Persistence mode — we shipped advice that is wrong for this desktop

**[DOC]** ([Driver Persistence](https://docs.nvidia.com/deploy/driver-persistence/overview.html)),
verbatim: *"Under Linux systems where X runs by default on the target
GPU, the kernel mode driver will generally be initalized and kept alive
from machine startup to shutdown, **courtesy of the X process**."*

**This inverts the blanket "always `nvidia-smi -pm 1`" tip — and we
built a first-run prompt around it on 2026-08-30.** Where the display
server runs on the NVIDIA GPU (Scott's desktop), persistence mode is
close to a no-op: the display stack already pins the driver. It earns
its keep when the dGPU is effectively headless — iGPU-drives-display,
a second card with no monitor, or `multi-user.target`.

NVIDIA's only quantified figure for the cost it avoids is *"order of
**1-3 second** ... due to **ECC scrubbing behavior**"* — and consumer
GeForce has no ECC, so even that component doesn't apply. The widely
quoted "500 ms–2 s per CUDA init" numbers are **[FOLKLORE]**; the one
source publishing them labels them "illustrative".

**[BENCH]** Throughput effect: none measured — persistence *"did not
produce a noticeable difference in steady-state kernel timing"*. It is
a startup-latency feature.

**Where it does matter in 2026 [DOC]:** the **open kernel modules**
README lists *"GPU initialization is slower. One possible mitigation is
to use **nvidia-persistenced**"* — and Blackwell+ is open-only. So the
advice is becoming *more* right on newer stacks, for a different
reason than the folklore gives.

**Action (logged):** our prompt should say what it actually buys on
*this* machine — and on a desktop with X on the NVIDIA card, that is
honestly "very little". It should not present a no-op as a fix. It also
correctly prefers `nvidia-persistenced` over `-pm`, which **[DOC]**
confirms: *"NVIDIA encourages customers to shift to this daemon
approach"* and calls `-pm` "near end-of-life".

### 6.2 Power limits: the number everyone quotes is a gaming benchmark

**pp is compute-bound, tg is memory-bandwidth-bound**, so a power cap
hits them completely differently. Any "% tokens/s lost" figure is
meaningless without saying which.

**[BENCH]** ([discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013))
RTX 5090, 400 W vs 600 W: **pp512 +16.0%**, **tg128 +1.4%**. Backwards:
dropping 33% of power costs 13.8% of prefill and 1.4% of decode.

**[BENCH]** ([RTX 3090 sweep, Apr 2026](https://jeanfbrito.github.io/posts/rtx-3090-power-limit-sweet-spot/))
— Qwen 3.6 27B, nearly Scott's rig: 350 W → 32.0 t/s; 300 W → 33.0;
250 W → 31.7; **200 W → 20.6 (−36%)**; 150 W → 8.3. Plateau, then
cliff. 350→250 W is inside measurement noise.

🚩 **The famous "cap to 70%, lose 3%" line is a Tom's Hardware
*ray-traced gaming* result, relabelled as tokens/s.** For llama.cpp
decode the honest number is ~1%; for prefill ~14%; for batched serving
~12% (**[BENCH]** 4×3090 vLLM).

**[BENCH — academic]** ([arXiv 2605.11999](https://arxiv.org/html/2605.11999),
Erlangen, May 2026) on an H200 measured decode drawing **137–300 W under
even the lowest 280 W cap** — the cap never engaged, throughput spread
0.3–2.8%, tensor cores idle >88%. **Clock locking Pareto-dominates
power capping: 24–32% decode energy saved for <1% throughput.**
⚠️ Do not over-generalize: on a 350 W 3090 the cap *does* bind below
250 W. Correct rule: **a cap costs throughput only once it engages, and
the engagement point is hardware-specific.**

### 6.3 ⭐ Throttle counters — a measurement-validity check we can actually build

**[DOC, verified on a consumer GeForce]**
`nvidia-smi --query-gpu=clocks_event_reasons_counters.sw_power_cap,...`
returns **cumulative microseconds per throttle reason**. Diff it before
and after a benchmark and you know exactly how long the card spent
throttled, with no sampling race. This works on GeForce, which the
per-sample `.active` bits make awkward.

Severity model that matters for not crying wolf:

| Reason | Meaning |
|---|---|
| `sw_power_cap` (0x4) | **Normal.** A card at its power limit is working correctly. |
| `sw_thermal_slowdown` (0x20) | Above max operating temp — **includes memory temperature** |
| `hw_thermal_slowdown` (0x40) | ≥2× clock cut. Emergency. |
| `hw_power_brake_slowdown` (0x80) | ≥2× cut, external power brake |

**[OURS]** Lifetime counters on Scott's 3090 Ti (driver 595.84):
`sw_power_cap` ≈ **1.9 hours accumulated**, and **every thermal counter
exactly zero across the card's entire history.** On a well-cooled
desktop the binding constraint is the power cap, never temperature —
which is why a "thermal" warning here would be noise, and why the
laptop's counters will be the interesting comparison.

⭐ **The one inference worth encoding:** `sw_thermal_slowdown` set
**while `temperature.gpu` reads fine** means the *memory junction*
tripped it — and that is **not readable on GeForce** (NVML exposes only
`NVML_TEMPERATURE_GPU`; no `nvidia` hwmon device exists). So the honest
message is *"VRAM thermal limit suspected, not readable on this GPU"*,
never "no thermal issue". Also read `GPU Slowdown Temp` / `GPU Max
Operating Temp` **per device** — Ada/Blackwell report *margins*
(`GPU Current T.Limit Temp`) instead of absolutes, a real telemetry
grammar drift of the kind we already know to expect from logs.

⚠️ GDDR6X's **EDR** (error-detection-and-replay) degrades *effective*
bandwidth while clocks, power, temperature and every throttle bit read
nominal. A tg regression with entirely clean telemetry on a hot GDDR6X
card is therefore a real and otherwise-invisible hypothesis, not noise.

### 6.4 Page cache dominates load time — and it confounded one of our menus

**[BENCH]** (PR #7420, Ryzen 9 7950X + PCIe-5 NVMe, 85 GB Mixtral):

| Mode | Load |
|---|---|
| `--no-mmap`, cold | 47.3 s |
| mmap, cold | 20.8 s |
| direct I/O | 7.3 s |
| **mmap, warm** | **4.5 s** |

**[OURS] — this found a bug in our `load` trial menu, fixed today.**
Our rounds run baseline first, variants after, so the baseline loaded
from whatever cache state the machine was in while every variant loaded
warm from the round before — a systematic bias toward whatever ran
last, of a magnitude (4.6×) that swamps any genuine load-mode
difference. `run_trial` now spends one **discarded** load warming the
cache before the first measured round, so every round is compared warm
— which is also the state our router genuinely runs in, being
long-lived and reloading models repeatedly.

Two related notes:
- **`--mlock`, `--mmap`/`--no-mmap` and `-dio` were deprecated in
  July 2026** in favour of `-lm/--load-mode`, and mixing old and new
  emits *"only the last flag on the command line will take effect"*.
  **[OURS]** We already emit only `load-mode` (`trial.rs:119`) — no
  action, and a live example of why reading `--help` beats copying
  blog examples.
- **[FOLKLORE, refuted]** *"mmap makes a 30B model need only 5.8 GB of
  RAM"* was debunked in the thread that produced it: *"Almost the
  entire model is needed for inference, so mmap doesn't reduce RAM usage
  at all. It's purely a measurement artifact."*
- For a **hot-swapping router, warm mmap beats direct I/O** — DIO
  bypasses the page cache, so repeated loads are ~10× slower. DIO wins
  cold loads only.

### 6.5 VRAM is not all yours, and `memory.free` lies in two directions

- **The compositor takes real memory.** Working figures: 1080p
  ≈ 150–400 MB; 4K/multi-monitor ≈ 700 MB–1.5 GB. ⚠️ And there is a
  **NVIDIA+Wayland retention bug** — a compositor grows until it holds
  roughly **10% of total VRAM** and does not release it (reproduced on
  KWin, Sway, Weston, GNOME Shell across several 5xx drivers; **not
  reproducible on AMD**). On a 24 GB card that is ~2.4 GB gone.
- **⭐ Your own CUDA context costs 200–600 MB** that `memory.free`
  reported before it existed. So free-memory readings are optimistic by
  at least a context.
- **[VENDOR — NVIDIA staff]** totals ≠ sum of per-process: *"the GPU has
  overheads ... not directly associated with a particular user
  process"*. **Never** compute free as `total − Σ(per-process)`.
- **Consequence for `--fit`'s 1024 MiB margin:** on a 24 GB desktop
  losing ~2.4 GB to a long-lived Wayland compositor, a 1 GB margin is
  not enough — llama.cpp issue #18390 asks for per-device margins for
  exactly this reason. This is the *other* half of why `measured_ctx`
  moves between runs, alongside the Ollama residency modellab found.
- **⭐ Neither tool can evict the other — a CUDA-level fact.** A process
  cannot free another process's device allocations. Both llama.cpp and
  Ollama can only measure free memory and leave a margin, and **that
  measurement is inherently racy**. Ollama's own default is
  `OLLAMA_MAX_LOADED_MODELS = 3 × GPU count` — **three models resident
  by default**, not one, which is worth knowing before blaming us for
  a failed fit.
- **⚠️ Linux has no silent system-memory fallback.** Windows spills to
  system RAM (driver 536.40+, with a Control Panel toggle); on Linux
  over-allocation gives you `cudaMalloc failed: out of memory`. The
  strongest evidence is that llama.cpp built the workaround *because* of
  it: `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` exists to *"allow swapping to
  system RAM instead of crashing"* — opt-in, off by default, and
  measured ~20× slower on prefill, with the maintainer concluding *"I
  don't think this would ever be worth using."* **Design for hard OOM.**

### 6.6 ⚠️ The 8 GB question — the genuine gap, and how to close it

The sub-survey covering 8 GB-class GPUs specifically did not complete
(529). Rather than fill it with plausible-sounding numbers, here is what
the *other* findings let us say with a source, and what must be
measured on the G15:

**Derivable now:**
- **The last-layer cliff [DOC]** — llama-bench's own README, llama 7B
  Q4_0 on CUDA: `ngl 34` gives 881 pp / 71.8 tg; `ngl 35` (all layers)
  gives **2400 pp / 131.7 tg**. **One layer short of full offload costs
  63% of prefill and 45% of generation.** On 8 GB this is the dominant
  effect: a model that *almost* fits is not "a bit slower", it falls off
  a cliff. Our advice column should say that.
- **RAM-bandwidth ceiling** — `max tok/s ≈ bandwidth ÷ bytes-read-per-token`,
  with 60–80% realized. DDR5-6000 dual channel ≈ 96 GB/s theoretical, so
  a 17 GB dense Q4 spilling to CPU tops out near **5.6 t/s** and lands
  at 3.4–4.5. That is why dense mid-size models are not viable once they
  spill, and why MoE offload changes the picture (it reads *active*
  params).
- **⚠️ But MoE is not automatically the answer on small/unified
  hardware [BENCH]** — arXiv 2606.21428 measured MoE **10% slower than
  same-active dense on an M2 Pro and 31% slower on a Jetson**, at 2.1×
  energy/token, concluding *"on bandwidth-bound edge hardware, inference
  cost tracks total parameters, not active ones."* And **[BENCH]** on a
  UMA Jetson, `--cpu-moe` gave ~6 t/s vs ~23 t/s GPU-only — a **4×
  slowdown**, because on unified memory there is no transfer to save.
  The G15 has discrete VRAM so this doesn't apply to it, but it means
  our MoE-offload advice must be **discrete-GPU-only**.
- **Laptop TGP is not implied by the GPU name [BENCH]** — an RTX 4090
  laptop at 80 W performs ~30% worse than the same silicon at 150 W. So
  `power.limit` / `enforced.power.limit` must be recorded alongside any
  G15 measurement, or it is not comparable to anything.

**Must be measured on the G15 (added to the QA task list):**
- Mobile clock-decay curve and time-to-throttle — **[GAP]** no sourced
  data found. The `clocks_event_reasons_counters` diff (§6.3) answers it
  directly, and Scott's desktop already provides the contrast case
  (zero thermal microseconds, ever).
- Whether `platform_profile` (`/sys/firmware/acpi/platform_profile` —
  **six** possible values, always read `platform_profile_choices`)
  measurably changes sustained pp/tg. **[GAP]** — no credible public
  benchmark of laptop CPU/GPU power-budget sharing versus sustained
  decode exists. The G15 harness would produce the first one I could
  find.
- Whether the deep-depth bench pass (added 2026-09-02) is tolerable at
  8 GB, or whether the rung ladder needs to be configurable.

---

## 7. The source Scott supplied, reconciled

Found in the repo on 2026-09-03 at
[`docs/human_research/`](../human_research/): a 1,193-line guide,
*Local LLM Inference Optimization — The Complete Guide*, beside a
`urls_to_look_at.txt` naming its origin
([carteakey.dev](https://carteakey.dev/blog/local-inference/local-llm-optimization/),
backed by the author's [l3ms](https://github.com/carteakey/l3ms) homelab
toolkit). It had been swept into a commit unread; this section is the
reconciliation it should have had. **It is the better-organised document
of the two** — symptom-first, and it grades its own claims *Tested here*
/ *Upstream behavior* / *Needs testing*, which is why it can be used at
all. Its numbers come from one node: RTX 4070 12 GB, i5-12600K,
DDR5-6000, CachyOS, CUDA.

### 7.1 Where it independently confirms this survey

Two documents compiled from different sources agreeing is worth more
than either alone. It reaches the same conclusions on:

- **KV precision** — `q8_0` as the text baseline, `f16` when validating
  a model, `q4_0` "only after an acceptance or quality check." Arrived
  at from VRAM budgeting rather than §1.1's KL-divergence route.
- **Bench at the context you serve.** *"At short bench contexts (512
  tokens), the KV cache is tiny and this effect is near-zero. Always
  test at your real serving context length."* — independent arrival at
  modellab's depth finding (§3), from the placement side.
- **`--poll`: do not tune it.** *"Confirmed across multiple sweeps —
  within noise at all poll levels."* We don't.
- **`--numa`: skip on single-socket.** We don't touch it.
- **`--flash-attn auto`, not forced** (§2). Same reading of upstream.
- **`--parallel 1` for a single user** — each slot carries its own KV
  cache; on gpt-oss-120b, 4→1 freed ~540 MiB, one more GPU layer,
  +1 t/s. Our `--models-max 1` reasoning, measured.
- **Fit is a projection, `llama-fit-params` makes it reproducible**
  (§4.3) — and it names the binary we hadn't got round to quoting.
- **`GGML_CUDA_FORCE_CUBLAS`: tested and closed** — ~45 t/s PP
  *regression* on mxfp4/Q4, no TG gain, because GGML's MMQ kernels are
  tuned for consumer decode batch sizes (1–16) and cuBLAS for
  datacenter batches. Do not add it as a build option.

### 7.2 What it adds that this survey missed — the host layer

This is the real contribution, and it is a whole category we had no
coverage of. We built `system::gpu_conditions` because modellab showed
an unrecorded *GPU* condition was moving results. The same argument
applies one layer down, and this document has the measurements:

| **[FIELD]** finding | Effect on that node | We record it? |
|---|---|---|
| RAM not at rated XMP/EXPO speed | MoE TG at **roughly one-third** | No |
| `power-profiles-daemon` degrading HWP | TG varies **20–30% between boots**, *while every sysfs value reads `performance`* | No |
| E-cores inside the inference thread set | TG **20–30%** lower | No |
| `vm.swappiness` + model near the RAM ceiling | TG **stalls mid-session** | No |
| CPU governor / EPP not `performance` | Sustained clocks below boost | No |

The second row is the dangerous one: a confound that **survives the
obvious check**. That is the same shape as the free-VRAM finding — a
condition that varies between runs, is invisible in the output, and
silently reorders a scorecard. For an MoE-heavy fleet whose crowned
config puts 32 layers of experts on the CPU, the first and third rows
are not marginal. §8 records where this desktop actually sits.

Its diagnostic checklist (§24 of that document: `dmidecode` memory
speed, governor, EPP, P-core frequency, free VRAM, thermals, CPU hogs,
`pswpin`/`pswpout`, PCIe link speed, `tuned-adm active`) is a
ready-made pre-flight gate, and every item is cheap.

### 7.3 What it found in our own code

**`--threads` is not affinity.** *"Most reliable way to keep inference
off E-cores … `taskset -c 0-11`."* Thread **count** and thread
**placement** are different knobs: the kernel schedules `--threads 8`
onto any core it likes, E-cores included. Our `moe` menu's comment
claimed its `cpu-moe-t8` / `cpu-moe-t24` pair raced "P-cores only vs
all threads". It does not — it races *fewer threads* against *one per
logical CPU*. **Comment corrected today**; real pinning needs
`taskset`/`sched_setaffinity` and is now a ROADMAP item. Note the
guide's `0-11` is *its* 12600K; this desktop's P-cores are `0-15`
(§8) — a range nobody should copy between machines.

**Our `kv` menu may measure the smaller half of the effect.**
**[FIELD]** *"switching f16 → q8_0 KV frees ~2 GB. That 2 GB lets
`llama-fit-params` keep one to two additional GPU layers — translating
directly to higher TG. Confirmed on Qwen3-Coder-Next: q8_0 KV at 64k
unlocked 2 extra GPU layers and added ~2 t/s TG."* Our `kv` menu's goal
is `Goal::Context` — *"a bigger settled context for the same VRAM."*
That is the right axis on a VRAM-starved model. But on a model already
at its full trained context — Scott's 80B settles at all 262,144 — the
freed VRAM **cannot** buy more context, so it buys layers instead, and
a Context goal scores that real TG win as **zero improvement**. The
menu isn't wrong; its goal is single-axis where the mechanism has two.

**`--mlock` is filed under the wrong goal.** We only ever score it in
the `load` menu against `Goal::LoadTime`. Its documented purpose is not
load speed — it is preventing a mid-session swap event from stalling
TG. Judged on load time it can lose while still being the right setting
for a machine that swaps (§8).

**Target and draft KV caches are one decision, not two.** **[FIELD]**
`-ctk/-ctv` set the target model's cache, `-ctkd/-ctvd` the draft's —
and quantizing the *target* to `q8_0` drove Gemma 4 MTP draft
acceptance *"close to zero"*, where `f16` kept it above 70% and was
much faster overall. Our `spec` and `kv` menus race independently and
apply independently. This is direct evidence those two knobs interact
violently on MTP models, so a `kv` winner applied after a `spec` winner
could silently destroy it. Worth a guard, or at least ordering advice.

### 7.4 Leads worth testing, not adopting

- **`--fit-target`.** Its tested values: **512 MiB** for text on 12 GB,
  **512–768** on 24 GB, **2048** for vision. We hardcode **1024**. An
  independent outside data point for the max-context question modellab
  raised (151,808 vs our 110,080) — *and* it supplies the mechanism our
  conservatism is really defending: **[FIELD]** *"CUDA's VMM pool grows
  as context fills. A 128 MiB target survived short benches and later
  failed."* The failure mode is **mid-session OOM, not load-time OOM**,
  which is exactly why a bench-validated small margin is a trap. So:
  1024 may cost Scott context, but the test has to be a long session,
  not a load.
- **QAT + MTP — the largest speed result in the document.** Gemma 4 26B
  38.5 → **100.6 t/s** (2.6×); 12B → 120.8 t/s (2.0×). Partly
  quantization-aware training (a 26B dropping 18 GB → 14.2 GB, so the
  layers land in VRAM at all) and partly multi-token-prediction
  speculation. This is **model-selection** advice, which is Library and
  advice territory rather than a flag: "a QAT build of this model
  exists" is a thing we could surface.
- **iGPU display routing frees 500–1000 MB VRAM** — and it resolves
  §6.1 into something constructive. Display-on-the-NVIDIA-card is the
  *same* root fact that makes persistence mode a near no-op there. One
  honest prompt replaces a misleading one: route the display to the
  motherboard output, and you free VRAM *and* make persistence mode
  mean something.
- **Build flags** we may not set: `GGML_LTO=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`,
  and `CMAKE_CUDA_ARCHITECTURES` matched to the card (89 for Ada, 86 for
  Scott's 3090 Ti) rather than a default fan-out. Worth auditing the
  Build Advisor against.
- **`llama-sweep-bench`** exists as an upstream binary for parameter
  sweeps. We hand-roll sweeps; worth a look.
- **THP**: its node runs `[always]`, ours `[madvise]`. Explicitly a
  "test both" in the source, so: untested lead.

### 7.5 Where not to follow it

One node, one CPU vendor, 12 GB of VRAM. Its `taskset -c 0-11`,
`--fit-target 512`, batch/ubatch profiles, MTP draft lengths (2 for
26B, 4 for 12B, against an upstream default of 3) and *"Linux was
15–20% faster than the Windows configuration I compared"* are all
single-sample results, and the author says so plainly. The
`tuned-ppd` fix is a diagnosis for a symptom, *"not a blanket desktop
recommendation"* — and this desktop does not have that symptom (§8).
Its "how throughput feels" table is subjective calibration, not data.

---

## 8. This desktop's host state, measured 2026-09-03

Ran §24's checklist against Scott's machine while it was otherwise
idle. **[OURS]**, and the point of recording it is that these are
conditions under which every number in `measurements.json` was taken.

| Check | Reading | Verdict |
|---|---|---|
| CPU | i9-12900K — **8 P-cores (cpu0–15, 5.1–5.2 GHz) + 8 E-cores (cpu16–23, 3.9 GHz)** | **Hybrid — §7.2 row 3 applies** |
| `scaling_governor` | `powersave` (all 24) | Not `performance` |
| `energy_performance_preference` | `balance_performance` | **Not `performance`** |
| `scaling_driver` | `intel_pstate`, `no_turbo=0` | HWP active, turbo enabled |
| `power-profiles-daemon` | **inactive** | The 20–30% ppd bug does not apply |
| `tuned` | inactive | Nothing is setting a profile either |
| RAM | 62 GiB total, 23 GiB free, 30 GiB cache | Ample |
| Swap | **3.6 GiB in use** of 8 GiB, `swappiness=60` | In use *with 23 GiB free* |
| `pswpout` / `pswpin` | 2,164,370 / 552,251 pages ≈ **8.3 GiB out, 2.1 GiB in** over 27 h uptime | Real swap traffic |
| THP | `[madvise]` | Source's node uses `always`; untested |
| RAM speed (XMP) | **UNKNOWN** — `dmidecode` needs sudo | ⚠️ **the #1 MoE culprit, unverified** |

**Reading this honestly.** Nothing here is proof that a stored
measurement is wrong. Three things are worth Scott's attention:

1. **RAM speed is unverified and it is the single highest-stakes item**
   in the guide — XMP off took *its* MoE generation to about a third.
   Scott's fleet is MoE-heavy and his crowned config
   (`ncpu-moe-32`) puts 32 layers of expert weights in system RAM,
   where RAM bandwidth *is* the generation bottleneck. One sudo command
   settles it; it is Task 2.7.
2. **Neither governor nor EPP is `performance`.** This is the stock
   desktop default, not a misconfiguration, and on `intel_pstate` with
   HWP `powersave` still boosts — so this is a lever to *measure*, not
   a bug to panic about. But it is the same knob in both directions:
   whatever it is worth, every stored measurement was taken at
   `balance_performance`.
3. **The machine is swapping while 23 GiB sits free** — that is
   `swappiness=60` being eager, not memory pressure. Harmless for most
   work; the guide's documented failure mode is precisely a swap event
   evicting RAM-resident expert weights mid-session, which presents as
   generation *stalling* rather than running slow. If Scott has ever
   seen an inexplicable mid-session pause on a big MoE model, this is
   the first suspect, and `--mlock` is the documented answer (§7.3).

The E-core question is **not** settled by this table: our `moe` menu
already races thread counts, so there may be data in `trials.json`
already — but per §7.3 it raced counts, not placement, so it cannot
answer it. That needs `taskset`.

---

## 9. What this changes for us — summary

**Already right, now defended:** K/V asymmetry direction, `--models-max 1`,
never forcing `-fa`, context shift off, measuring `--cache-reuse`
instead of trusting 256, fresh process per trial variant, treating
acceptance rate as a diagnostic. §7.1 adds independent confirmation of
six of these from a second, unrelated source — including "bench at the
context you serve", reached from the placement side rather than ours.

**Already fixed because of the parallel modellab work:** benching at
realistic KV depth (`-d`), recording free VRAM as a measurement
condition, and dropping the report's "measured … not estimated"
overclaim.

**Worth doing:** quoting `llama-fit-params` in advice (§4.3), settling
`-ncmoe`'s direction before explaining it (§4.4), and surfacing
prompt-cache health more prominently (§4.2). The reasoning-kwarg
deprecation (§4.1) was checked and does not apply — kept as a worked
example of verifying before acting.

**Fixed today because the research found it:** our `load` trial menu
compared load times across rounds without controlling page-cache state,
a 4.6× confound biased toward whatever ran last (§6.4).

**Advice we shipped that the research contradicts:** the GPU-persistence
prompt (§6.1). On a desktop with X on the NVIDIA card, persistence mode
buys close to nothing — NVIDIA's own documentation says the display
server already keeps the driver alive. The prompt needs to stop
presenting a no-op as a fix.

**The honest gap:** 8 GB-class guidance (§6.6). Partly derivable from
the last-layer cliff and bandwidth arithmetic; the rest needs the G15.

**The category we were missing entirely (§7.2).** Every measurement
condition we record is a *GPU* condition. The host — RAM at rated
speed, CPU governor and EPP, which cores the inference threads land on,
whether the machine is swapping — is unrecorded, and the supplied
source has measured effects there of 20–30% and worse, including one
confound that reads `performance` in every sysfs file while costing
20–30% between boots. This is the same argument that produced
`gpu_conditions`, one layer down.

**Found in our own code by reading it (§7.3):** the `moe` menu's
comment claimed to race "P-cores only vs all threads" when `--threads`
sets count, not affinity — corrected today. Two more, logged not fixed:
the `kv` menu's `Goal::Context` scores a real TG win as zero on a model
already at full context, because there the freed VRAM buys *layers*;
and `--mlock` is scored on load time when its documented purpose is
stopping mid-session swap stalls.

**This desktop, measured (§8):** hybrid CPU (so the E-core finding
applies), governor `powersave` / EPP `balance_performance` (stock, but
every stored number was taken there), swapping 3.6 GiB while 23 GiB
sits free — and **RAM speed unverified**, which is the guide's
single highest-stakes check and needs one sudo command from Scott.

**The meta-lesson.** Most of this field's tuning advice is folklore
with invented decimals. The defensible parts are (a) llama.cpp's own
documentation and maintainer statements, and (b) measurements with
stated conditions. That is precisely the gap this tool exists to fill —
and the main risk to guard against is *us* becoming another source of
confident numbers without conditions. The corrections we shipped on
2026-09-02 were exactly that failure mode caught early.
