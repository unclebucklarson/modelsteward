
> **Note:** This post was drafted with significant AI assistance, synthesizing notes, bench results, and scripts from the [l3ms homelab toolkit](https://github.com/carteakey/l3ms) and the series of model-running posts on this site. The experiments, numbers, and failure modes documented here are real - the synthesis and prose are AI-assisted.

## Preface

Over the past year I've written posts on running [gpt-oss-120b](/blog/local-inference/optimizing-gpt-oss-120b-local-inference/), [Qwen3-Coder-Next](/blog/local-inference/optimizing-qwen3-coder-next-local-inference/), [Gemma 4 26B](/blog/local-inference/running-gemma-4-26b-a4b-locally/), [Qwen3.6-35B-A3B](/blog/local-inference/running-qwen3-6-35b-a3b-locally/), and [Gemma 4 MTP](/blog/running-gemma-4-mtp-locally/) locally on consumer hardware. Each post has its own notes, failure modes, and tuning results - but the same lessons keep appearing: enable XMP, pin to P-cores, quantize your KV cache, don't trust the power profile.

This is my attempt at a master reference. Instead of re-discovering flags in every new model post, I want one doc to link back to. If you're hitting a performance wall, starting from scratch, or just want to understand what each knob actually does - start here.

The scope is intentionally wide. The numbers in this guide come from one machine: an RTX 4070 12 GB, i5-12600K, DDR5-6000, Linux, CUDA, and recent llama.cpp builds. Its RAM capacity changed during the project, so the dashboard records the hardware used for each run instead of pretending every result came from one fixed configuration. The Apple Silicon, AMD, multi-GPU, and server sections are useful starting points, but I haven't tested those setups myself.

Claims use three labels:

- **Tested here** means I measured it on the RTX 4070 node. The exact command and available run metadata live in the [L3MS dashboard](https://github.com/carteakey/l3ms).
- **Upstream behavior** means it comes from current llama.cpp documentation, checked on July 17, 2026.
- **Needs testing** means it is a useful lead, not a recommendation.

---

## 1. Start with the Symptom

| Symptom | Check first | Then change |
| --- | --- | --- |
| Model fails to load or dies late in a session | Free VRAM and real serving context | Reduce context or slots, quantize KV, increase fit headroom, then change [placement](#10-layer-placement-the-core-optimization-for-moe) |
| Prompt processing is slow | PP with a representative prompt | Sweep [`--ubatch-size`](#12-2-ubatch-size-ub-physical-micro-batch); check GPU offload and backend |
| Token generation is slow | TG, RAM speed, CPU frequency | Fix memory/power state, then compare [placement](#10-layer-placement-the-core-optimization-for-moe) and [P-core pinning](#14-3-taskset-p-core-pinning-linux) |
| Speculation adds no speed | Acceptance, TG, and end-to-end loop time | Shorten the draft or compare target/draft [KV precision](#19-2-target-and-draft-kv-cache-precision) |
| Two GPUs are slower than one | Split mode and interconnect | Start with [layer mode](#21-multi-gpu-primer); treat tensor mode as experimental |

For coding agents, save TTFT, PP, TG, cache behavior, and a complete tool loop. A fast decode number can still hide repeated prefill or a cold prompt cache.

### 1.2 Safe Starting Profiles

These are **tested-here starting profiles**, not llama.cpp defaults. They describe how I begin on this node before model-specific tuning.

| Workload | `--fit-target` | Context | KV cache | `--parallel` | Batch |
| --- | ---: | ---: | --- | ---: | ---: |
| Text, 12 GB VRAM | 512 MiB | 64k | `q8_0` / `q8_0` | 1 | 1024 |
| Text, 24 GB VRAM | 512–768 MiB | 128k | `q8_0` / `q8_0` | 1–2 | 1024 |
| Vision, 12 GB VRAM | 512+ MiB | 64k | `q8_0` / `q8_0` | 1 | 256 |
| MTP speculative decoding | 512+ MiB | 64k | Benchmark per model | 1 | 1024 |

> **Tested here:** 512 MiB of fit headroom, one slot, and q8 KV have been reliable for text on this 12 GB card. Vision needed more headroom. Treat those as local results, not capacity rules for another architecture.

## 2. What Has Actually Moved the Needle

| Evidence | Change | Result on this node |
| --- | --- | --- |
| **Tested here** | Enable the rated XMP profile | Restored MoE TG from roughly one-third speed |
| **Tested here** | Keep hybrid Intel E-cores out of the inference thread set | Improved TG by 20–30% in affected profiles |
| **Tested here** | Quantize text-server KV to q8 and use one slot | Freed enough VRAM for more GPU-resident weights |
| **Tested here** | Gemma 4 QAT plus MTP | 2.0–2.6× TG in the published runs; not a general MTP promise |
| **Upstream behavior** | Let fit select placement when no explicit placement is supplied | Current llama.cpp enables fit by default; use `llama-fit-params` to capture reproducible flags |
| **Needs testing** | n-gram speculation, CUDA graph optimization, tensor-parallel multi-GPU | Keep only when an end-to-end workload beats the baseline |

## 3. What to Measure Before Tuning

Optimization only makes sense if you know which phase is slow.

| Metric | What it tells you | Common bottleneck |
| --- | --- | --- |
| **TTFT** (time to first token) | How long before output starts | model load, prompt processing, cold cache |
| **PP** (prompt processing) | How fast the model reads input context | batch size, GPU kernels, long prompts |
| **TG** (token generation) | How fast output streams after prefill | VRAM/RAM bandwidth, layer placement, CPU pinning |
| **VRAM at load** | Whether weights, KV cache, and mmproj fit | context size, parallel slots, quant level |
| **VRAM after long sessions** | Whether memory grows into OOM territory | CUDA graphs, VMM pool growth, fit headroom |
| **RAM bandwidth / swap** | Whether hybrid MoE weights are bottlenecked | XMP/EXPO, channels, `--mlock`, swappiness |
| **Draft acceptance rate** | Whether speculative decoding is helping | draft quality, KV precision, spec length |
| **Prompt-cache hit rate** | Whether repeated agent context is being reused | cache key, prompt shape, server restarts |
| **Tool-call loop time** | User-visible agent latency | TTFT + tool runtime + repeated prefill |

Do not optimize from a single short prompt. Short prompts hide KV cache costs, long-context VMM growth, and parallel-slot allocation. Benchmark at the context length you actually serve.

---

## 4. Small Glossary

| Term | Definition |
| --- | --- |
| **PP / Prompt Processing** | Tokens per second while the model reads the prompt. |
| **TG / Token Generation** | Tokens per second while it generates output. |
| **KV Cache** | Buffer storing attention Key/Value tensors for all prior context tokens. Grows linearly with context length. Lives in VRAM. |
| **MoE** | A model that activates only some expert weights per token; offloaded experts can make system RAM bandwidth matter. |
| **MTP** | Speculation using a companion draft model trained for multi-token prediction. |
| **Fit** | llama.cpp's automatic VRAM-aware placement pass. |

---

## 5. Pick the Runtime Before the Flags

This guide is about tuning llama.cpp on a consumer CUDA workstation. If the requirement is a desktop model browser, use LM Studio; if it is simple model management, use Ollama; if it is high-throughput multi-user serving on datacenter GPUs, evaluate vLLM. The rest of this post assumes direct access to llama.cpp flags.

### 5.3 Local Inference Tools

| Tool | Best for | Notes |
| --- | --- | --- |
| **llama.cpp** | Performance tuning, full flag control, any hardware | Build from source; CLI-centric |
| **Ollama** | Zero-config, model management, Docker | Uses llama.cpp internally; limited tuning surface |
| **LM Studio** | Desktop GUI, Windows/macOS, model browsing, local OpenAI-compatible API | Good UX, supports server/headless workflows, JIT loading, TTL, and auto-evict |
| **vLLM** | Multi-user production serving, continuous batching | Designed for full-VRAM serving; not suited for consumer hybrid setups |
| **exllamav2** | Maximum speed for dense models | CUDA-only; excellent for models that fully fit in VRAM |
| **mlx** | Apple Silicon | macOS only; leverages unified memory; no CUDA |

**This guide focuses on llama.cpp.** Most concepts (KV cache, quantization, layer placement) generalize across tools.

### 5.4 Backends Within llama.cpp

| Backend | Build flag | Best for |
| --- | --- | --- |
| **CUDA** | `-DGGML_CUDA=ON` | NVIDIA GPUs; highest performance and most tested |
| **HIP / ROCm** | `-DGGML_HIP=ON` | AMD GPUs through ROCm/HIP |
| **Vulkan** | `-DGGML_VULKAN=ON` | AMD and Intel GPUs; cross-platform; works on NVIDIA too |
| **Metal** | (auto on macOS) | Apple Silicon; unified memory |
| **SYCL** | See llama.cpp SYCL docs | Intel Arc, Max, Flex, and some integrated GPUs |
| **CPU-only** | (no GPU flag) | Reference; small models or debugging |
| **RPC** | `-DGGML_RPC=ON` | Experimental distributed inference |

> **This document assumes CUDA.** Where behavior differs on Vulkan or CPU-only, it's marked `[Vulkan]` or `[CPU]`. If you're on AMD/Intel GPU, most flag logic is the same but CUDA-specific env vars don't apply.

You can build multiple backends into one tree, and current llama.cpp also supports dynamically loaded backends with `-DGGML_BACKEND_DL=ON`. That is useful if you move binaries between machines with different GPUs. For a single NVIDIA box, I still prefer a plain CUDA build because there are fewer moving parts when benchmarking.

---

## 6. Hardware

### 6.1 The Memory Hierarchy - The Most Important Mental Model

Token generation speed is limited by how fast the runtime can stream active weights through the memory hierarchy. Rough bandwidth numbers:

```
VRAM (GPU on-die)              ~600–1000 GB/s
Unified memory (Apple Silicon)  ~200–400 GB/s
System RAM                       ~50–200 GB/s   (varies hugely by speed and channel config)
NVMe SSD                          ~5–7 GB/s
SATA SSD / HDD                    ~0.5–3 GB/s
```

**Dense models**: every token reads all active weights. Model must fit in VRAM for full-speed inference. Any spill to system RAM causes a large TG drop.

**MoE models** often stream expert weights from system RAM, making memory bandwidth critical. On my machine, enabling XMP took generation from roughly one-third speed back to normal.

**Check that your RAM is running at its rated speed:**
```bash
sudo dmidecode -t memory | grep -E "Speed|Configured"
# "Configured Memory Speed" must match your XMP/EXPO profile speed.
# If it doesn't, enable XMP/EXPO in BIOS.
```

Check this before touching llama.cpp flags. It takes a minute and can save hours of tuning the wrong thing.

### 6.2 GPU / VRAM

VRAM is the primary inference resource. More VRAM = more layers on GPU = faster inference.

| VRAM | Dense examples | MoE hybrid examples |
| --- | --- | --- |
| 8 GB | 7B–13B Q4 | 30B–70B, heavy RAM offload |
| 12 GB | 13B–20B Q4; 7B Q8 | 70B–120B+ with CPU offload |
| 24 GB | 34B Q4; 13B Q8 | 120B+ moderate offload |
| 48 GB+ | 70B Q8; 34B FP16 | Most MoE partially/fully on GPU |
| 80 GB+ | 120B+ FP16 | No offload needed |

**iGPU display trick (desktop NVIDIA):** route display through the motherboard video output instead of the GPU. This frees 500–1000 MB VRAM the GPU was using for desktop composition.

### 6.3 CPU

For **dense models** (all in VRAM): CPU is nearly idle during inference. Core count has minimal impact.

For **MoE hybrid** runs, CPU-resident experts can make CPU and RAM the TG bottleneck. **Tested here:** including E-cores on the i5-12600K reduced TG by 20–30%, so these profiles pin inference to its P-cores:

```bash
taskset -c 0-11 llama-server ...   # i5-12600K: cores 0-11 are P-cores
```

Start with the P-core count for `--threads`, leaving capacity for the OS, then confirm with a short sweep.

### 6.4 How Throughput Feels on This Setup

| TG speed | Experience |
| --- | --- |
| < 5 t/s | Painful; barely usable |
| 5–10 t/s | Functional; near human reading speed (~7 t/s) |
| 10–20 t/s | Comfortable for interactive chat |
| 20–40 t/s | Fast; coding agents feel snappy |
| 40+ t/s | Near-instant for most tasks |

For coding agents, do not read this table alone. Cold TTFT and repeated prefill can dominate even when warm TG feels fast.

---

## 7. OS Choice

### 7.1 Linux

Linux exposes the CPU, memory, and service controls used in my profiles.

- **Tested here:** Linux was 15–20% faster than the Windows configuration I compared. That is one machine, not an OS constant.
- Full access to CPU governor, NUMA, huge pages, cgroups, headless mode.
- Choose a distribution with a current supported GPU stack. My node runs CachyOS; Ubuntu/Debian and Arch are common alternatives.

### 7.2 Windows

Reasonable; CUDA support is solid.

- Some overhead from Windows scheduler and CUDA runtime.
- **Power plan**: set to "High Performance" or "Ultimate Performance". Default "Balanced" throttles CPU under sustained inference load.
- NVIDIA Control Panel → "Power management mode" → "Prefer maximum performance".
- WSL2: close to native for CUDA; some VRAM overhead from virtualization. Generally acceptable if dual-boot isn't an option.

### 7.3 macOS

Different runtime stack - CUDA guidance does not apply.

- Metal backend activates automatically.
- **Apple Silicon unified memory**: CPU and GPU share the same physical pool. A machine with 128 GB RAM effectively has 128 GB "VRAM". Transforms the dense model size ceiling.
- Memory bandwidth is excellent (~400 GB/s on M3 Max); competitive with mid-range NVIDIA for the models it can run.
- `mlx` framework is worth evaluating alongside llama.cpp for Metal workloads.

### 7.4 Linux OS Tuning

These settings corrected measurable problems on this node. Apply them one at a time and keep a baseline.

#### CPU Governor and Power Profile
```bash
# Must be "performance" on all cores
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# EPP must also be "performance" - governor alone is not enough on intel_pstate
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference

# Verify actual P-core frequency near max boost
grep "cpu MHz" /proc/cpuinfo | sort -rn | head -6
```

**Tested here:** `power-profiles-daemon` sometimes left this Intel node 20–30% below its TG baseline even while common sysfs values reported `performance`. Replacing it with `tuned-ppd` made the runs consistent. This is a diagnosis for that symptom, not a blanket desktop recommendation.

Fix - replace with `tuned-ppd`:
```bash
# Arch / CachyOS
sudo pacman -S tuned-ppd       # removes power-profiles-daemon automatically
sudo systemctl enable --now tuned
sudo tuned-adm profile throughput-performance
# Reboot. TG will be stable and correct across all boots.

# Ubuntu / Debian
sudo apt install tuned
sudo systemctl enable --now tuned
sudo tuned-adm profile throughput-performance
```

#### Transparent Huge Pages
```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
# Test both values; this node uses [always]
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

#### Headless Mode
```bash
# Stop desktop compositor - frees 200-400 MB RAM + compositor VRAM
sudo systemctl isolate multi-user.target
sudo sync && sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"

# Restore when done
sudo systemctl isolate graphical.target
```

Without a display server, use `zellij` in a TTY for split panes: `zellij` in any TTY gives full terminal multiplexing without X or Wayland.

---

## 8. Why llama.cpp, and How to Build It

### 8.1 Why Not Ollama?

Ollama uses llama.cpp internally and deliberately exposes a smaller tuning surface. It is useful for quick setup and model management. Direct llama.cpp is the better fit when the work is comparing placement, KV precision, batch sizes, or backend flags.

### 8.2 Building from Source

For reproducible tuning, build from a recorded llama.cpp commit. A distribution package is fine when its version and backend match your needs; source builds make architecture flags and benchmark provenance explicit.

**CUDA build:**
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DGGML_NATIVE=ON \
  -DGGML_LTO=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
# 89 = Ada / RTX 40-series. Check NVIDIA's compute capability table for your card.

cmake --build . --config Release \
  --target llama-server llama-bench llama-fit-params llama-cli --parallel
```

`CMAKE_CUDA_ARCHITECTURES=89` is right for my RTX 4070. Don't cargo-cult the number; check NVIDIA's [compute capability table](https://developer.nvidia.com/cuda/gpus). If you have mixed cards, pass a semicolon list like `"86;89"`. If you're building a portable binary, turn `GGML_NATIVE` off and let the build cover more devices.

```bash
# Portable-ish CUDA build: larger binary, less tuned to one exact box
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF
```

**Vulkan build** (AMD, Intel GPU):
```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DLLAMA_CURL=ON \
  -DGGML_NATIVE=ON
# CUDA-specific build flags do not apply here
```

**ROCm / HIP build** (AMD GPU):
```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake .. \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --parallel
```

Set `GPU_TARGETS` if the build system guesses wrong or you are compiling for a different AMD GPU. For example, `gfx1100` is RDNA3-class desktop hardware.

**Dynamic backend build**:
```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BACKEND_DL=ON \
  -DGGML_CUDA=ON \
  -DGGML_VULKAN=ON
```

This lets backends load as shared libraries at runtime. Useful for packaged builds or mixed machines; less interesting for a locked-down homelab node.

Keep your build updated. Pull and rebuild regularly - especially before benchmarking a new model.

### 8.3 Key Binaries

| Binary | Purpose |
| --- | --- |
| `llama-server` | OpenAI-compatible HTTP inference server |
| `llama-bench` | Synthetic PP and TG benchmarking |
| `llama-fit-params` | Dry-run VRAM probe → outputs optimal `-ngl` + `-ot` flags |
| `llama-cli` | Interactive CLI prompt for quick tests |
| `llama-sweep-bench` | Sweep across parameters (batch size, ngl, etc.) |

---

## 9. Model Selection and Quantization

### 9.1 Dense vs MoE - Choose Your Tuning Strategy

| | Dense | MoE |
| --- | --- | --- |
| Active params/token | All | Small fraction (e.g. 3B of 80B) |
| Must fit in VRAM | Yes, for best performance | No - experts can live in RAM |
| Primary tuning lever | Quant level + context | Layer placement + RAM bandwidth |
| TG speed driver | VRAM bandwidth | RAM bandwidth + GPU layer count |
| Example models | Gemma 4, Llama 3, Mistral | Qwen3, DeepSeek, gpt-oss |

### 9.2 Quantization Reference

| Quant | Size vs FP16 | Quality | Notes |
| --- | --- | --- | --- |
| Q2_K | ~25% | Noticeable degradation | Use only under extreme size constraints |
| IQ3 / IQ4 | ~25–35% | Often better than older same-size quants | Calibration and runtime support matter; test speed too |
| Q4_K_M | ~35% | Good | Best size/quality balance for most situations |
| Q4_K_XL / UD-Q4_K_XL | ~35% | Often better than plain Q4 | Dynamic/tensor-sensitive allocation when available |
| Q5_K_M / Q5_K_XL | ~40% | Very close to FP16 | Strong default when VRAM allows |
| Q6_K | ~50% | Near-lossless | High-VRAM setups |
| Q8_0 | ~65% | Effectively lossless | If storage/RAM permits |
| FP16 | 100% | Reference | Maximum quality; maximum size |
| MXFP4 (native) | ~35% | Better than Q4_K_M | gpt-oss models: trained in MXFP4, not post-quantized |

**UD (Unsloth Dynamic) quants**: keep sensitive tensors at higher precision and push easier tensors lower. They can beat uniform quants at the same average size, but the calibration data and target workload still matter.

**Rule of thumb**: use the highest quant that fits your VRAM + RAM budget. Q5_K_XL or UD-Q5_K_XL is a strong default. Drop to Q4 when needed, then check PPL/KLD and your actual workload.

### 9.3 Quantization-Aware Training (QAT)

Standard Post-Training Quantization (PTQ) quantizes weights *after* the model is fully trained. When going down to 4-bit, this rounding process can throw away critical precision, leading to regressions in reasoning, logic, and acrostic constraints.

Quantization-Aware Training (QAT) bypasses this degradation by modeling low-precision rounding noise *during* the training or fine-tuning process. This enables the model weights to adapt to the low-bit limits.
* **Accuracy Recovery**: In the Gemma 4 QAT builds I tested, 4-bit QAT behaved much closer to Q8 than ordinary post-training 4-bit quantization.
* **VRAM Savings**: A 26B MoE model in standard Q8_0 or dynamic Q5 consumes ~18 GB, spilling heavily to system RAM on a 12GB card. The QAT Q4 model size drops to ~14.2 GB, allowing the vast majority of the model layers to load directly into VRAM for full GPU speed.

Still test it. QAT, Dynamic GGUF, and imatrix quants are different ways of spending a quality budget; none of them magically make low-bit files immune to workload-specific regressions.

### 9.4 iMatrix and IQ Quants

Importance-matrix quantization (`imatrix`) calibrates the quantizer against representative text instead of treating every tensor equally. In rough terms: if a weight matters more for the calibration activations, the quantizer tries harder not to damage it. This is why calibration data matters. A WikiText-ish imatrix and a code/chat imatrix are not protecting exactly the same behavior.

IQ quants are worth paying attention to when you are trying to push below the normal Q4/Q5 comfort zone without wrecking the model. The useful split:

- **mainline llama.cpp IQ quants**: easiest to run everywhere.
- **ik_llama.cpp / IK quants**: often where frontier IQ/K quant work appears first, especially for CPU or hybrid MoE runs.
- **Unsloth Dynamic GGUFs**: practical packaged quants that keep sensitive tensors at higher precision and push easier tensors lower.

Kawrakow's comparisons are useful because they explain the families, not just the filenames. Formats like `IQ4_XS`, `IQ4_K`, and `IQ3_K` use more careful/non-linear schemes than plain older low-bit quants, and can beat simple K-quant baselines at similar bits-per-weight. Unsloth's Qwen3.5 notes make the same point from the packaging side: expert tensors, attention tensors, and hybrid/state-space tensors do not all tolerate the same compression. "Q4" can mean very different files.

So my practical rule is:

| Situation | What I would try first |
| --- | --- |
| Trusted Unsloth Dynamic GGUF exists | Try `UD-Q4_K_XL` / `UD-Q5_K_XL` before a generic community Q4/Q5. |
| You need a dense 27B-ish model to squeeze into 12 GB | Try IQ/UD only if it still leaves room for q8 KV and real context. |
| CPU or hybrid CPU/GPU MoE | Check ik_llama.cpp results; this is one of its strongest lanes. |
| Maximum TG matters more than size | Compare against K-quants; IQ can be slower. |
| The model is already QAT or native low-bit | Test the native/QAT path first. Do not stack cleverness for sport. |

The trap: a quant that wins perplexity can still lose on real coding or long-context use. Wiki-style PPL/KLD, Aider/LiveCodeBench, and "does it stay coherent at my context length?" are not the same test. For this guide, a quant only "wins" if it survives the workload I actually run.

Things to record when comparing IQ / Unsloth Dynamic / IK / imatrix builds:

| Question | Why it matters |
| --- | --- |
| What calibration data was used? | Chat/code calibration and Wiki calibration preserve different behavior. |
| Is it dynamic per-layer/per-tensor, uniform, or fork-specific? | Same nominal "Q4" can mean very different files. |
| Is the quant mainline llama.cpp or ik_llama.cpp-only? | Runtime compatibility matters. |
| Does the model survive your real context length? | A 4k chat test says very little about a 64k coding session. |
| Did q8 KV still fit? | Weight savings that force q4 KV may be a wash. |
| Did TG drop vs K-quant? | IQ can trade speed for quality/size. |

### 9.5 Perplexity, KLD, and Actually Choosing a Quant

This is the part that gets muddy fast. File size and TG are easy to measure. Quality is not.

Perplexity is still useful as a smoke test. If a quant has much worse PPL than another quant of the same model and roughly the same size, something probably went wrong. But PPL can hide answer flips because better and worse token probabilities can average out.

KLD is a better compression metric because it compares the quantized model's output distribution against the original model. Unsloth leans on this in their Dynamic GGUF writeups, citing *Accuracy is Not All You Need*: two compressed models can have similar benchmark accuracy while changing different answers underneath. KLD tends to catch that drift better than raw accuracy or PPL.

Unsloth's GLM-5.2 results are a useful example because they separate "smaller" from "dumber." Their dynamic 4-bit and 5-bit GGUFs are mostly lossless by KLD, and even 1-bit / 2-bit keep high top-1 agreement with Q8.

But top-1 here is argmax-token match, not factual accuracy. 76% top-1 does not mean "wrong 24% of the time"; it means the quant picked the same next token as the reference 76% of the time. Many misses are wording, formatting, or high-entropy positions where the baseline was not that committed either.

{% image_cc "./src/static/img/local-inference/quant-eval-stack.svg", "Diagram showing file size, perplexity, KL divergence, task evals, and workload tests as a quantization evaluation stack", "w-full", "My read: PPL catches obviously damaged quants, KLD catches distribution drift, and the workload still gets the final vote." %}

My reading order:

| Metric | Good for | Failure mode |
| --- | --- | --- |
| File size / BPW | Fit planning | Says nothing about quality. |
| PPL | Quick regression check | Can miss answer flips and calibration overfit. |
| KLD | Distribution drift vs the original model | Depends heavily on eval/calibration data. |
| Task evals | Aider, LiveCodeBench, MMLU, tool calling | Slow and sometimes brittle to templates. |
| My workload | The real serving profile | Annoying to run, but this is the one that matters. |

Unsloth's warning about calibration overfit is the important bit for local users. If the imatrix and the evaluation both look like WikiText, a quant can look great on PPL/KLD and still be worse for chat, code, or tool use. Instruct models also have chat templates, so plain text calibration can under-test the actual path you use.

So I would not rank quants by one number. I would shortlist by size, PPL/KLD, and maintainer trust, then run the model on my actual context length with the sampling and KV settings I plan to serve.

---

## 10. Layer Placement - The Core Optimization for MoE

> For dense models fully in VRAM: use `-ngl 99` and skip to §11.

For MoE hybrid setups, layer placement is where most performance lives. The goal: keep as many blocks as possible on GPU (especially early layers and attention), while offloading expert weights to RAM.

### 10.1 `--n-gpu-layers` (`-ngl`)

How many transformer blocks to load onto GPU. Start at `99` (all layers). Drop if CUDA OOM.

```bash
llama-server -m model.gguf --n-gpu-layers 99    # all on GPU
llama-server -m model.gguf --n-gpu-layers 37    # 37 on GPU, rest on CPU
```

### 10.2 `--n-cpu-moe`

Integer count: keep the named number of MoE layers' expert weights on CPU. Quick coarse control.

```bash
--n-cpu-moe 31   # first 31 MoE layer experts on CPU
```

> ⚠️ **RAM ceiling**: for a ~60 GB model, putting all experts on CPU tries to load ~60 GB into RAM. On a 64 GB system this will hard-crash the machine. Always use `llama-fit-params` (§10.4) to find safe values before setting this high.

### 10.3 `--override-tensor` (`-ot`) - Fine-Grained Placement

Per-tensor, per-layer placement via regex. Most control, most complexity.

```bash
# All expert projections in all layers → CPU:
--override-tensor ".ffn_(up|down|gate)_(ch|)exps=CPU"

# Layers 5+ experts → CPU; keep layers 0-4 fully on GPU:
--override-tensor "blk\.(5|[6-9]|[0-9][0-9]+)\.ffn_(up|down|gate)_(ch|)exps=CPU"
```

**The shared expert gotcha**: some models (Qwen3.5-122B, certain gpt-oss variants) have two expert tensor families:
- Routed experts: `ffn_{up,down,gate}_exps`
- Shared expert (always active, 1 per layer): `ffn_{up,down,gate}_shexp`

A pattern matching only `_exps` leaves `_shexp` on GPU, silently consuming VRAM and causing CUDA OOM. The safe pattern captures both:

```bash
# (ch|) matches both _exps (routed) and _shexp (shared):
--override-tensor ".ffn_(up|down|gate)_(ch|)exps=CPU"
```

Safe to include `(ch|)` even for models without shared experts - it's harmless and future-proofs the pattern.

### 10.4 `--fit` - Automatic Placement

> **Upstream behavior:** current llama.cpp enables fit by default when placement is not supplied. Explicit `-ngl`, `--tensor-split`, or `--override-tensor` values take control instead.

Fit probes free VRAM at startup and computes `-ngl` and `-ot` placement. I still write `--fit on` in test commands because it makes the intent visible:

```bash
llama-server \
  -m model.gguf \
  --fit on \
  --fit-ctx 65536 \   # minimum context to guarantee fits; KV cache for this ctx is accounted for
  --fit-target 512 \  # VRAM headroom in MiB to leave free
  ...
```

> **Tested here:** CUDA's VMM pool grows as context fills. A 128 MiB target survived short benches and later failed, so my persistent text profiles leave at least 512 MiB. Vision uses 2048 MiB. Current llama.cpp builds also include mmproj memory in the fit calculation ([PR #21489](https://github.com/ggml-org/llama.cpp/pull/21489)).

**Dry run without starting a server:**
```bash
llama-fit-params \
  -m /path/to/model.gguf \
  -fitt 512 \     # fit-target MiB
  -fitc 65536     # fit-ctx tokens
# Outputs something like: -c 65536 -ngl 49 -ot "blk\.8\.ffn_...=CPU,..."
```

This output is what you hardcode for static placement.

### 10.5 Static vs Dynamic Placement

| | `--fit on` | Hardcoded `-ngl` + `-ot` |
| --- | --- | --- |
| Startup | +1–5s probe delay | Instant |
| Placement | Fresh each boot | Fixed |
| Tuning effort | None | One-time from `llama-fit-params` |
| Best for | Exploration | Repeated, comparable runs |
| Reproducibility | Good (if VRAM is free) | Deterministic |

For a repeatable profile, derive placement once with `llama-fit-params` and save the result with the benchmark. Re-run fit after changing the model, context, backend, or available VRAM.

---

## 11. Context and KV Cache

### 11.1 `--ctx-size` - Choosing Context Length

The KV cache grows linearly with context and lives in VRAM. Large context on small VRAM can push expert layers off GPU.

Approximate KV VRAM usage (varies by model, head count, and layer structure):

| Context | f16 KV | q8_0 KV |
| --- | --- | --- |
| 8k | ~0.5 GB | ~0.25 GB |
| 32k | ~2 GB | ~1 GB |
| 64k | ~4 GB | ~2 GB |
| 128k | ~8 GB | ~4 GB |

On a 12 GB card at 128k context with f16 KV, 8 GB goes to KV cache - leaving only ~4 GB for model weights and attention. Context choice directly affects how many GPU layers you can afford.

**Practical guidance:** coding sessions work well at 64k; long-context RAG may need 128k+; vision inference is typically safest at 64k on 12 GB VRAM.

### 11.2 KV Cache Quantization (`-ctk`, `-ctv`) [CUDA]

Quantizing the KV cache halves (q8_0) or further reduces (q4_0) its VRAM footprint.

```bash
-ctk q8_0 -ctv q8_0
```

| KV quant | VRAM vs f16 | Quality | Recommendation |
| --- | --- | --- | --- |
| `f16` | 1× | Baseline | Start here when validating a model |
| **`q8_0`** | **~0.5×** | **Measure against f16** | **Tested-here text baseline** |
| `q4_0` | ~0.33× | More likely to alter output | Use only after an acceptance or quality check |

**The compounding effect**: on a 12 GB card with 64k context, switching f16 → q8_0 KV frees ~2 GB. That 2 GB lets `llama-fit-params` keep one to two additional GPU layers - translating directly to higher TG. Confirmed on Qwen3-Coder-Next: q8_0 KV at 64k unlocked 2 extra GPU layers and added ~2 t/s TG vs f16 KV.

> At short bench contexts (512 tokens), the KV cache is tiny and this effect is near-zero. Always test at your real serving context length.

### 11.3 `--parallel` - Concurrent Inference Slots

Each slot maintains its own KV cache. `--parallel 4` multiplies KV VRAM by 4.

```bash
--parallel 1   # single user homelab: reclaims KV VRAM for model weights
```

On gpt-oss-120b, dropping `--parallel 4` → `--parallel 1` freed ~540 MiB VRAM - enough for one more GPU layer and +1 t/s TG.

### 11.4 Flash Attention (`-fa`)

Reduces attention memory traffic and avoids materializing the full attention matrix, which makes long-context inference more practical on constrained VRAM.

```bash
--flash-attn on
```

> **Upstream behavior:** current llama.cpp defaults to `--flash-attn auto`, not forced on. **Tested here:** forcing it on has worked for these CUDA profiles. Keep `auto` on unfamiliar backends, and force it only after checking model support and memory use.

> [Vulkan]: Flash attention support varies by driver version. Verify before relying on it.

---

## 12. Batch Sizes

### 12.1 `--batch-size` (`-b`) - Prompt Processing Throughput

Controls how many tokens are processed in one forward pass during prefill. Higher = better PP throughput; more VRAM required.

```bash
--batch-size 2048   # current upstream default
--batch-size 1024   # tested-here text profile
--batch-size 512    # lower peak memory
```

Reduce if you hit CUDA OOM during the prefill phase specifically.

### 12.2 `--ubatch-size` (`-ub`) - Physical Micro-Batch

Physical sub-batch within a logical batch. Must be ≤ `--batch-size`.

```bash
--ubatch-size 512   # typical default
```

Do not treat `512` as magic. It is a safe default, not a law. `--batch-size` controls the logical prefill batch; `--ubatch-size` controls how that work is physically split. Larger `ubatch` can improve PP, but it also raises peak VRAM during prefill.

Run a tiny sweep on the model and context shape you actually serve:

```bash
for ub in 128 256 512 1024; do
  ./build/bin/llama-bench \
    -m /models/model.gguf \
    -p 2048 -n 128 \
    -b 1024 -ub "$ub" \
    -ngl 99 \
    -ctk q8_0 -ctv q8_0 \
    -fa on
done
```

Then repeat the winning two values inside `llama-server`, because server memory layout, `--parallel`, and prompt cache can change the result.

**Vision/multimodal critical**: an image tokenizes to several hundred tokens. If `--ubatch-size` < image token count, llama.cpp throws an assertion during vision inference. Use `--ubatch-size 512` or higher and test with your actual image sizes. On 12 GB VRAM, `--batch-size 256 --ubatch-size 512` is a stable vision baseline.

---

## 13. Sampling Parameters

Sampling controls the probability distribution at each decode step. These affect output quality and - via vocabulary truncation (`top-k`) - slightly affect speed.

**Start with model card defaults.** Most GGUF releases specify tested values. Use those before experimenting.

### 13.1 `--temp` - Temperature

- `0.0`: greedy / deterministic. Best for coding agents where reproducibility matters.
- `0.7`: standard creative chat.
- `1.0`: no rescaling; follows raw model distribution. Most modern instruction-tuned models are calibrated for this.

### 13.2 `--top-k` - Vocabulary Truncation

Keeps only top K most probable tokens before sampling.

- `0`: full vocabulary (default; most diverse; slowest to compute)
- `100`: safe performance cap - confirmed no measurable quality loss on coding tasks (gpt-oss-120b)
- `20–64`: model-specific tighter caps

### 13.3 `--top-p` - Nucleus Sampling

Filters to tokens whose cumulative probability ≥ p. Applied after `top-k`.

- `1.0`: no filtering (gpt-oss, Sarvam defaults)
- `0.95`: standard for chat/code (Gemma 4, Qwen3 defaults)

### 13.4 `--min-p`

Filters tokens below `min-p × max_token_probability`.

- `0.0` off (most models)
- `0.01` light floor (Qwen3-Coder-Next)

### 13.5 `--repeat-penalty`

- `1.0`: no penalty. Recommended for code - code naturally repeats patterns (variable names, keywords) and penalizing them degrades output.
- `1.1–1.3`: mild penalty for prose.

### Recommended Defaults by Model Family

| Model | temp | top-k | top-p | min-p | repeat-penalty |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 | 1.0 | 64 | 0.95 | 0.0 | 1.0 |
| Qwen3 / Qwen3-Coder | 1.0 | 40 | 0.95 | 0.01 | 1.0 |
| gpt-oss-120b | 1.0 | 100 | 1.0 | 0.0 | 1.0 |
| Sarvam | 1.0 | 20 | 1.0 | 0.0 | 1.0 |

---

## 14. Threading and CPU Control

### 14.1 `--threads` (`-t`)

CPU threads for the token generation phase (expert compute in hybrid MoE setups).

```bash
--threads 10   # P-core count minus 1-2 for OS headroom
```

More threads than available P-cores is counterproductive - they contend for the same memory bus and typically reduce TG.

### 14.2 `--threads-batch`

CPU threads for the PP (prefill) phase. PP is a burst workload; you can set this to full thread count.

```bash
--threads-batch 12
```

### 14.3 `taskset` - P-Core Pinning [Linux]

Most reliable way to keep inference off E-cores on Intel 12th gen+ (and other hybrid architectures):

```bash
taskset -c 0-11 llama-server ...   # pin all process threads to P-cores
```

Verify your P-core range from CPU documentation or `lstopo`. On Intel 12600K, cores 0–11 (6 P-cores × 2 threads) are the P-cores; 12–15 are E-cores.

### 14.4 `--poll`

Controls CPU spin aggressiveness while waiting for GPU kernel completion.

- `0`: yield/sleep
- `100`: busy spin

**On hybrid CPU+GPU inference, this is flat.** GPU kernel execution and PCIe transfer dominate synchronization. Confirmed across multiple sweeps - within noise at all poll levels. Leave at default `50` or set to `0` to reduce idle CPU load. Do not tune this.

### 14.5 `--numa`

NUMA affinity modes: `distribute`, `isolate`, `numactl`.

**Single-socket systems: skip.** There is only one NUMA node; these modes provide no benefit and can hurt performance. Use `taskset -c` for affinity instead.

Relevant only on dual-socket server hardware (AMD EPYC, Intel Xeon) where NUMA topology is real.

---

## 15. Memory Control

### 15.1 `--no-mmap`

Without this, llama.cpp uses memory-mapped I/O. Expert weight accesses during decode are non-sequential - the OS page fault handler triggers repeatedly for cold pages, adding latency jitter to TG.

With `--no-mmap`, the entire model loads into RAM before inference begins. No page faults.

```bash
--no-mmap
```

> **Tested here:** this removed page-fault jitter from hybrid MoE runs. The tradeoff is a longer startup and full upfront RAM allocation, so compare it on the actual server workload.

### 15.2 `--mlock`

Pins model pages in RAM, preventing the OS from swapping them under memory pressure.

```bash
--mlock
```

Important when `vm.swappiness` is high (many Linux distributions default to 60–150 with ZRAM) or when running close to the RAM ceiling. Without it, a swap event mid-session can make TG appear to stall. Skip only if RAM is critically tight.

---

## 16. Priority and Process Settings

### 16.1 `--prio`

Scheduling priority for the inference process. Scale 0–3.

```bash
--prio 2   # high priority; reduces OS scheduling jitter on TG
```

### 16.2 `--no-warmup`

Skips initial kernel warmup pass at startup (compiles CUDA kernels on first real request instead).

```bash
--no-warmup   # reduces startup time; safe for persistent servers
```

---

## 17. CUDA-Specific Settings [CUDA]

### 17.1 Environment Variables

```bash
export GGML_CUDA_GRAPH_OPT=0    # Baseline; compare against 1 on your real workload
```

`LLAMA_SET_ROWS` was in my older scripts, but current llama.cpp does not read it. I removed it.

**`GGML_CUDA_GRAPH_OPT` caveat**: test `0` against `1` on a real session, not a short prompt. Graph re-capture can increase memory use as context grows, and some models get slower with it enabled. I leave it off unless it wins the long run.

### 17.2 Notable Build Flags

| Flag | Effect |
| --- | --- |
| `GGML_CUDA_GRAPHS=ON` | Enables CUDA graph capture at build time |
| `GGML_CUDA_FA=ON` | Compiles CUDA Flash Attention kernels |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | Flash attention for all quant types |
| `GGML_NATIVE=ON` | CPU-native optimization; don't use for distributed binaries |
| `GGML_LTO=ON` | Link-time optimization; slower build, faster runtime |

### 17.3 cuBLAS - Tested and Closed

`GGML_CUDA_FORCE_CUBLAS=ON` forces CUDA BLAS routines over the default GGML MMQ (mixed-precision matrix quantization) kernels.

Tested on mxfp4 and Q4 models: **slower** than default. GGML MMQ has native mxfp4/Q4 paths tuned for consumer decode batch sizes (1–16 tokens). cuBLAS is optimized for large datacenter batches. Result: ~45 t/s PP regression, no TG improvement. Default build wins on consumer hardware. May be worth re-evaluating on 24+ GB cards where larger batch sizes make cuBLAS more competitive.

---

## 18. Coding Workloads - What to Measure

For coding, TG is only part of the feel. Long prompts and repeated file context make PP and TTFT matter a lot.

Record these when testing a model:

| Metric | What I look for |
| --- | --- |
| TTFT | Does output start quickly after a tool call or file read? |
| PP | Can the server ingest 8k–64k of code/context without a huge pause? |
| TG | Does generation feel interactive after prefill? |
| Prompt-cache hits | Does repeated agent context actually reuse cache? |
| Draft acceptance | If using MTP or n-gram speculation, are accepted tokens high enough to matter? |
| Long-session stability | Does VRAM/RAM stay flat after 30–60 minutes? |

My preferred quick test:

1. Cold prompt after server start.
2. Same project, similar prompt, warm cache.
3. One small inspect/edit/explain loop.
4. One long-context prompt near the context size I actually use.

Minimum data to save with any published profile:

```text
model:
quant:
llama.cpp commit:
benchmark command:
context:
parallel:
batch / ubatch:
KV cache:
prompt-cache state:
PP:
TG:
TTFT:
draft acceptance:
VRAM at load:
VRAM after long session:
notes:
```

L3MS now publishes this evidence with each local profile. Old runs keep unknown fields as “not recorded” instead of inventing values. Community submissions use a [separate schema](https://github.com/carteakey/l3ms/blob/main/docs/community-runs.md) and appear in an unranked community view; they never enter the local RTX 4070 ranking.

---

## 19. Speculative Decoding & MTP (Multi-Token Prediction)

Autoregressive token generation (TG) is memory-bandwidth bound: the GPU must read all active model weights from memory for every single token it generates. Speculative decoding works around this by using a lightweight draft model to guess upcoming tokens, which the base model verifies in one forward pass.

On models trained with Multi-Token Prediction heads, a companion draft model can help. It only counts as a win when acceptance and end-to-end latency improve.

### 19.1 MTP Drafting Configuration Flags

Mainline llama.cpp supports companion MTP draft models through `--spec-draft-model`, `--spec-type draft-mtp`, and `--spec-draft-n-max`.

> **Tested here:** draft lengths of 2 for Gemma 4 26B and 4 for Gemma 4 12B won on this node. **Upstream behavior:** the current general default maximum is 3. Sweep around the default instead of copying my model-specific values.

### 19.2 Target and Draft KV Cache Precision

MTP has two distinct caches. `-ctk` and `-ctv` set the **target model** cache; `-ctkd` and `-ctvd` set the **draft model** cache. Treat them as separate tuning decisions.

In my Gemma 4 MTP tests, quantizing the target cache with `-ctk q8_0 -ctv q8_0` drove draft acceptance close to zero. Switching it to `f16` used more VRAM but kept acceptance above 70%, which was much faster overall. Qwen may behave differently, so I test this per model.

Start with a full-precision draft cache, then compare `q8_0` and `f16` for the target cache:

```bash
# Memory-saving target cache; full-precision draft cache
-ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16

# Gemma baseline that preserved acceptance in my tests
-ctk f16 -ctv f16 -ctkd f16 -ctvd f16
```

Record acceptance rate, TG, and VRAM. Saving memory is pointless if the draft model stops landing tokens.

### 19.3 Speculative Performance Gains

**Tested here** on a single RTX 4070 12 GB; exact run evidence belongs with the dashboard profile:
* Gemma 4 26B Baseline: 38.5 tok/s
* Gemma 4 26B QAT + MTP: **100.60 tok/s** (2.6x speedup)
* Gemma 4 12B QAT + MTP: **120.80 tok/s** (2.0x speedup)

### 19.4 n-gram Speculative Decoding

llama.cpp also has draftless speculative modes. They look for repeated token patterns in the current context or a small n-gram table.

> **Needs testing:** code repeats paths, imports, and edit patterns, but I have not measured an L3MS speedup yet.

Start with `ngram-mod`, which is the most interesting current option for persistent server use:

```bash
llama-server \
  -m model.gguf \
  --spec-type ngram-mod \
  --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 48 \
  --spec-ngram-mod-n-max 64 \
  --spec-draft-n-max 64 \
  ...
```

For simpler single-session repetition tests:

```bash
llama-server \
  -m model.gguf \
  --spec-type ngram-simple \
  --spec-draft-n-max 64 \
  ...
```

MTP and n-gram can be combined because llama.cpp accepts comma-separated speculative types:

```bash
llama-server \
  -m model.gguf \
  --spec-draft-model mtp-model.gguf \
  --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max 64 \
  --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 48 \
  --spec-ngram-mod-n-max 64 \
  ...
```

Do not assume the combined setup wins. Draftless decoding can have precedence over draft-model decoding in current llama.cpp, and long drafts can waste work if acceptance is poor. Measure acceptance, TG, and tool-loop time.

Suggested L3MS sweep:

| Variant | What to compare |
| --- | --- |
| baseline | no speculation |
| MTP only | current Gemma/Qwen MTP profile |
| `ngram-simple` | repetitive code prompt, no draft model |
| `ngram-mod` | persistent server / repeated file-edit session |
| `draft-mtp,ngram-mod` | combined mode; only keep if end-to-end loop wins |

---

## 20. Vision / Multimodal

### 20.1 `--mmproj`

Path to the multimodal projector file:

```bash
--mmproj /path/to/mmproj-BF16.gguf
```

Typically 1–3 GB. Allocates in VRAM at startup alongside the model.

### 20.2 OOM Failure Modes on Constrained VRAM

**Failure 1 - mmproj allocation**: current llama.cpp includes projector weights and compute buffers in `--fit`. If it still fails at load, check for an older build, other processes using VRAM, or a hardcoded placement created without the projector.

Fix: update llama.cpp and run `--fit` with the projector supplied. My 12 GB profile uses `--fit-target 2048` and has been stable.

**Failure 2 - batch assertion**: image token count exceeds `--ubatch-size`. An image can tokenize to several hundred tokens; if the batch is too small, llama.cpp asserts.

Fix: use `--ubatch-size 512` or higher.

### 20.3 Safe Vision Profile (12 GB VRAM)

This is the profile I actually use. You may be able to lower the 2048 MiB margin on a current build, but I prefer the headroom.

```bash
llama-server \
  -m model.gguf \
  --mmproj mmproj.gguf \
  --ctx-size 65536 \
  --fit on --fit-ctx 65536 --fit-target 2048 \
  -ctk q8_0 -ctv q8_0 \
  --flash-attn on \
  --batch-size 256 --ubatch-size 512 \
  --no-mmap --mlock \
  --parallel 1
```

Separate text and vision servers on different ports if running both workloads from the same GPU.

---

## 21. Multi-GPU Primer

> **Upstream behavior:** this section follows the current llama.cpp multi-GPU guide. **Needs testing:** I do not run a multi-GPU L3MS node, so none of these are local benchmark results.

llama.cpp has three split modes:

| Mode | What it does | Use when |
| --- | --- | --- |
| `layer` | Puts contiguous groups of layers on different GPUs | Default. Best first try, most compatible. |
| `row` | Older row-split path for dense weights | Deprecated upstream; avoid for new setups. |
| `tensor` | Experimental tensor parallelism across GPUs | Try for TG latency on supported dense models with fast interconnect. |

Default pipeline mode:

```bash
llama-server \
  -m model.gguf \
  --split-mode layer \
  --n-gpu-layers all
```

Custom split for mismatched GPUs:

```bash
# Example: GPU 0 gets 75%, GPU 1 gets 25%
llama-server -m model.gguf --split-mode layer --tensor-split 3,1
```

Tensor parallel mode:

```bash
llama-server \
  -m model.gguf \
  --split-mode tensor \
  -ctk f16 -ctv f16 \
  --flash-attn on
```

Important constraints:

- `layer` is the compatible default. Start there; automatic fit works in this mode unless explicit placement flags take over.
- `row` exists mostly for older setups; I would not build new guidance around it.
- `tensor` is experimental and does not support every architecture. Many MoE / hybrid architectures fail with an explicit "not implemented" error.
- `tensor` does not support quantized KV cache right now. Use `f16`, `bf16`, or `f32` KV.
- Automatic fit is disabled in `tensor` mode, so you need to manage context, parallel slots, and placement yourself.
- `GGML_CUDA_P2P=1` can help peer transfers when the driver and motherboard support it, but it can also crash or corrupt output on some systems. Test before keeping it.

If multi-GPU is slower than one GPU, the interconnect is probably the bottleneck. Try `layer` before `tensor`, check whether NCCL is available on CUDA builds, and don't assume a `--tensor-split` ratio from someone else's hardware means anything on yours.

---

## 22. ik_llama.cpp Fork [Advanced]

For highly specialized environments, the [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) fork exists. It focuses on MoE-specific kernel optimizations (such as fused MoE kernels).

However, it is not covered in detail in this guide because:
- **No Upstreaming**: Nothing developed in `ik_llama.cpp` is expected to make it upstream officially or directly.
- **Specialized Tuning**: It serves as a specialized option for custom, architecture-specific tuning once you have maximized standard configurations.

---

## 23. Security Notes

Local inference servers are still HTTP services. Do not expose `llama-server`, LM Studio, or any model gateway directly to the public internet without authentication, firewalling, and rate limits.

Minimum safe defaults:

- Bind to localhost for local tools unless you explicitly need LAN access.
- Put a reverse proxy with authentication in front of anything reachable outside the machine.
- Assume prompts, outputs, and tool calls may appear in app logs, shell history, reverse proxy logs, or frontend histories.
- Treat model files like software dependencies: check license terms, source, and expected file hashes when possible.
- Keep separate endpoints for trusted local agent workflows and anything exposed to other devices.

If you need remote access, prefer a private VPN, Tailscale, WireGuard, or a locked-down tunnel over opening the raw inference port.

---

## 24. Diagnostic Checklist

Run before benchmarking or when TG is unexpectedly low.

```bash
# 1. RAM speed - most common culprit for MoE TG underperformance
sudo dmidecode -t memory | grep -E "Speed|Configured"
# "Configured Memory Speed" must match your XMP/EXPO rated speed

# 2. CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Expected: performance

# 3. EPP - must be "performance" (governor alone is not enough on intel_pstate)
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
# Expected: performance (not balance_performance, not powersave)

# 4. Actual CPU frequency - P-cores should be near rated max boost
grep "cpu MHz" /proc/cpuinfo | sort -rn | head -6

# 5. Free VRAM - start inference from near-empty
nvidia-smi | grep MiB

# 6. Thermal - sustained load under throttle temperature
cat /sys/class/thermal/thermal_zone*/temp
# In millidegrees; 80000 = 80°C; throttle typically starts 85–105°C depending on chip

# 7. Background CPU hogs
ps aux --sort=-%cpu | head -10

# 8. Swap activity (high vm.swappiness systems)
cat /proc/vmstat | grep -E "pswpin|pswpout"
# Growing non-zero values = model weights being swapped mid-session

# 9. PCIe link speed [CUDA]
nvidia-smi -q | grep -A 3 "PCIe Generation"
# Expected: Current Gen = 3 or 4

# 10. Active tuned profile (if using tuned-ppd)
sudo tuned-adm active
# Expected: throughput-performance
```

### Known TG Variability Root Causes

| Root cause | Symptom | Fix |
| --- | --- | --- |
| `power-profiles-daemon` degrading HWP | TG varies 20–30% between boots; all sysfs checks look fine | Replace with `tuned-ppd` + `throughput-performance` |
| XMP/EXPO off | TG far below expected; stable but low | Enable the correct memory profile in BIOS |
| E-cores included in thread range | TG lower than P-core-only baseline | `taskset -c <p-cores-only>` |
| `vm.swappiness` + model too large for RAM | TG stalls mid-session | `--mlock`; reduce model or add RAM |
| `GGML_CUDA_GRAPH_OPT=1` with varying context | Intermittent OOM at long prompts | Set `GGML_CUDA_GRAPH_OPT=0` |
| `--fit-target` too small | OOM mid-session (not at startup) | Increase `--fit-target` to ≥512 MiB |
| Vision load fails with mmproj | Crash at model load | Update llama.cpp; run `--fit` with mmproj supplied; use more headroom |

---

## Changelog

| Date | Note |
| --- | --- |
| 2026-07-17 | Added evidence labels and a symptom-first path; removed generic cloud/local material; corrected current fit, batch, and flash-attention defaults; linked complete L3MS profile evidence and the separate community-run schema. |
| 2026-06-25 | Trimmed the coding-workload section, tightened quant/QAT/iMatrix/IQ guidance with Unsloth Dynamic and ikawrakow notes, and added a compressed GLM-5.2 PPL/KLD quant-eval section. |
| 2026-06-25 | Added Phase 2 material: coding-workload metrics, n-gram speculation recipes, ubatch sweep guidance, ROCm/HIP and dynamic backend notes, iMatrix/IQ quant guidance, and a multi-GPU primer. |
| 2026-06-22 | Tightened the tested scope, corrected mmproj fitting, separated target and draft KV caches, removed `LLAMA_SET_ROWS`, and turned CUDA graphs into an A/B test. |
| 2026-06-21 | Condensed ik_llama.cpp section to a brief advanced reference based on feedback. |
| 2026-06-21 | Added a problem-based reading map, centralized safe starting profiles, and consolidated the easy-to-miss vision, CUDA graph, hybrid CPU, and MTP guardrails. |
| 2026-06-17 | Reworked opening structure with a TL;DR, moved priority checklist, added measurement and security sections, updated llama.cpp/LM Studio guidance, tightened QAT/MTP wording, and fixed stale internal links. |
| 2026-06-12 | Updated optimization priority checklist, renumbered sections, and added dedicated guides for QAT quantization and MTP speculative decoding. |
| 2026-04-04 | Initial post - synthesized from l3ms scripts, bench-runbook, and model posts. |

## References

- [l3ms - homelab LLM toolkit (scripts, bench runbook, run scripts)](https://github.com/carteakey/l3ms)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [ik_llama.cpp fork](https://github.com/ikawrakow/ik_llama.cpp)
- [llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp speculative decoding guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
- [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141)
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- [FOSDEM 2025: History and advances of quantization in llama.cpp](https://archive.fosdem.org/2025/schedule/event/fosdem-2025-5991-history-and-advances-of-quantization-in-llama-cpp/)
- [ik_llama.cpp discussion: Will LQER improve k- and i-quants?](https://github.com/ikawrakow/ik_llama.cpp/discussions/15)
- [Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- [Unsloth Dynamic GGUFs on Aider Polyglot](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)
- [Unsloth GLM-5.2-GGUF KLD discussion](https://huggingface.co/unsloth/GLM-5.2-GGUF/discussions/3)
- [Unsloth Qwen3.5 GGUF benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [gpt-oss-120b optimization post](/blog/local-inference/optimizing-gpt-oss-120b-local-inference/)
- [Qwen3-Coder-Next 40 t/s post](/blog/local-inference/optimizing-qwen3-coder-next-local-inference/)
- [Gemma 4 26B local post](/blog/local-inference/running-gemma-4-26b-a4b-locally/)
- [Qwen3.6-35B-A3B local post](/blog/local-inference/running-qwen3-6-35b-a3b-locally/)
- [Gemma 4 MTP local post](/blog/running-gemma-4-mtp-locally/)

