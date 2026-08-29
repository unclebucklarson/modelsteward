# Docs & first-run findings — README, PLAN, PROJECT-BRIEF, onboarding paths

The adversarial persona here: a stranger who found modelsteward on GitHub or
crates.io, has a Linux box with a GPU, maybe has Ollama, has never built
llama.cpp.

---

## HIGH — the first 15 minutes

### D1. README has no Installation section; the quick start is unrunnable
README:127-152. The only instruction is `cargo run --release` (assumes a
clone that's never shown), then switches to `modelsteward --setup` — a
binary not on PATH after `cargo run`. Release CI publishes to crates.io and
builds tarballs, but the README mentions neither. **Fix:** an Install
section: `cargo install modelsteward`, release-tarball link, and the
from-source path with consistent invocations (`cargo run --release -- --scan`).

### D2. Build prerequisites live only in CI
ci.yml installs libgtk-3-dev, libxkbcommon-dev, libwayland-dev, libxcb-*
because eframe/rfd need them. `cargo install modelsteward` on a clean box →
linker error, no explanation. **Fix:** copy the apt line (+ Fedora/Arch
equivalents) into the README.

### D3. The llama.cpp prerequisite is never stated
The app requires a llama-server with router mode (`--models-preset`,
`--fit`), pinned by spikes to ≥ b10216 — the README never says so, never
says where to get one, and never mentions the app can *build* one for you
(Build Advisor → clone + build). **Fix:** a Requirements section: Linux,
llama-server ≥ bNNNN, GPU optional; plus "no llama.cpp yet? Server → Check
My llama.cpp → Set up (clone + build)".

### D4. No version gate: an older llama-server dies as an unexplained
30-second timeout. router.rs:382 passes `--models-preset` unconditionally; a
pre-router build exits instantly and the user gets "router did not come up
within 30s — see router.log" (ui.rs:4758) — directly contradicting the
README's own "No 'see logs'" promise (README:112-114). **Fix:** check the
discovered install's build number before start; say "your llama.cpp (b10088)
predates router mode — needs ≥ b10216; the Build Advisor can build a current
one". At minimum tail router.log into the error, as diagnose.rs already does
for model failures.

### D5. Zero-models first run: the Library is a bare header strip
ui.rs:1738-1752 renders headers over nothing; the default scan dir `~/models`
(settings.rs:95) usually doesn't exist. No message, no pointer to Settings.
(Cross-ref gui-findings G14.) **Fix:** empty state naming the three searched
places + current dirs + an "Add a model directory" affordance.

### D6. Zero-installs first run: Settings hides its only fix-it affordance
ui.rs:3197-3199 gates the entire "llama-server binary" block on
`!scan.installs.is_empty()`. With no llama.cpp, the user sees a bare text
field ("empty = auto-pick"); Start Router then logs `ERROR: … set one in
Settings` — pointing at the pane that just hid the guidance. **Fix:** invert
the gate: empty installs → "No llama.cpp found" + Browse + Build Advisor
link.

### D7. "Set Up Everything" hard-fails for anyone without OpenCode
opencode.rs:190-193 `read_to_string`s `~/.config/opencode/opencode.json`;
missing file → setup dies with `setup/sync: reading …: No such file or
directory`. The app markets itself to "any OpenAI-compatible app"
(README:3-5) but its documented first-run button requires OpenCode to be
installed. **Fix:** treat missing opencode.json as "skipped — OpenCode not
installed (Connections tab serves other apps)", not an error; README should
say OpenCode is optional.

### D8. Default scan dir bypasses the snap-HOME guard the module was built
around. settings.rs:95-97 uses `std::env::home_dir()` directly while
`real_home()` (settings.rs:15-18) exists precisely because snap parents
redirect HOME (documented live casualty in the same file). Launched from
snap-packaged VS Code, the default scan dir becomes
`~/snap/code/<rev>/models` — empty and vanishing on snap update — producing
D5's blank Library for an invisible reason. **Fix:** `real_home().join("models")`.

---

## MEDIUM — comprehension and staleness

### D9. The glossary defines 3 terms; the README uses ~15 undefined
Defined: shelf, measured ctx, Feat badges. Used undefined in the README's
own prose: router mode, preset, GGUF, quant, q8_0, KV cache, np, --fit,
MoE/A3B, --n-cpu-moe, pp/tg (partially), ngl, mmproj, MTP. For a project
whose stated north star is "without needing to be a llama.cpp expert", the
glossary is the headline gap. **Fix:** ~12 one-line entries, placed (or
linked) before the feature list.

### D10. No Help → Getting Started in-app
Help contains only the Tuning Guide (which starts at "1 · Measure + Bench" —
presuming models, an install, and a running router) and About. Step zero
lives nowhere in the app. **Fix:** "Help → First Run": (a) point Settings at
your models, (b) get/select llama-server, (c) Set Up Everything. Link the
repo/README from About.

### D11. PROJECT-BRIEF.md is stale on three counts
"88 unit tests" (actual: 157 `#[test]`s; ROADMAP says 147); "Planned next:
post-rebuild verification" (shipped 2026-08-25, documented in README);
MoE demo "near 40 t/s" vs the 52 t/s headline in README/ROADMAP. Two
numbers for the same measurement across docs is exactly the "guessed"
impression the project fights. **Fix:** single-source the demo figure;
refresh count and "planned next".

### D12. ROADMAP's status header trails the shipped version
"Where things stand (2026-08-28, 147 tests green — v0.4.0 + the meter)" vs
Cargo.toml 0.5.0 and M9 closed 2026-08-29. A reader can't tell what v0.5.0
contains; the repo has no CHANGELOG despite being a published crate.
**Fix:** bump the header; consider a CHANGELOG.md fed by the release notes.

### D13. ROADMAP still parks a feature the README sells as shipped
"App-managed llama.cpp checkout" sits in Parked/ideas with "tradeoffs to
settle", while README:30-33 advertises it working and managed.rs + the
Build Advisor UI ship it. **Fix:** move to Done with its ship date.

### D14. PLAN.md's status banner reads as current but froze at 2026-08-25
Lists progress through M7; M8/M9 absent. README calls it "the founding
design" without saying it's frozen. **Fix:** stamp the banner "frozen at
2026-08-25 — ROADMAP.md is current".

---

## LOW

### D15. README contradicts itself on the trial menus
README:71-72 `[spec|ub|kv|load|dials]` vs README:143's full ten. **Fix:**
one enumeration, one reference.

### D16. Legacy naming leaks
Backups are `opencode.json.lcc.bak.N` ("lcc" = llamacppCodeConf — opaque to
anyone finding them); the systemd unit is `llamacpp-router.service` rather
than modelsteward-namespaced. Crate/binary/window/config-dir all agree on
modelsteward (good). **Fix:** `.modelsteward.bak.N` with migration; a README
note on why the unit is named for what it runs (or rename it).

### D17. No screenshot in the README of a GUI app
216 lines of prose, five tabs, zero images. **Fix:** one Library-tab
screenshot near the top.

### D18. CLAUDE.md's command list is stale
`--trial <id> [spec|ub|kv]` (missing six menus + slots); `--advise`,
`--report`, `--quality`, `--verify-rebuild` absent. Mostly affects Claude
sessions, but it *is* the contributor-facing doc. **Fix:** refresh, or point
at the (to-be-created) `--help` as the single source.

---

## Dev session responses

(append here)
