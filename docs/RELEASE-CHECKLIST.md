# Release checklist

The manual QA pass before every tag (adopted 2026-08-29 from the
usability-review QA plan). ~15 minutes plus the automated gates.

## Automated gates (hard requirements)

- [ ] `cargo test` fully green.
- [ ] `cargo build` with **zero warnings**.
- [ ] `Cargo.toml` version bumped **before** tagging — release.yml
      refuses a tag that doesn't match it.

## Fresh-eyes pass (~15 min)

Run against a scratch config so first-run paths get exercised:

    HOME=$(mktemp -d) XDG_CONFIG_HOME= XDG_STATE_HOME= cargo run

- [ ] First run: Library shows the no-models empty state naming the
      searched directories; Settings shows fix-it guidance if no
      llama-server is found (never a bare hidden picker).
- [ ] Walk [GUIDE.md](GUIDE.md) top to bottom on the real config —
      every "You should see" holds.
- [ ] Each tab renders and its primary action works: Library
      (select a row → detail panel actions), Server (status + log),
      Lab (ETA line appears, Run enabled only with a selection),
      Connections (snippets + OpenCode mirror states), Settings
      (Save with a bad port shows the inline error, not a log line).
- [ ] Cancel a running campaign — config restored, partial results
      kept, no trial marker left behind (restart the app: no
      "healed interrupted trial" line).

## CLI smoke

- [ ] `--help` to stdout exit 0; `--version` prints the new version.
- [ ] `--config` prints paths + settings.
- [ ] Bad args error usefully: `--start 99999`, `--trial x nope`,
      `--meter 30d`, `--quality m abc` — each names valid values,
      exit nonzero.
- [ ] `--calibrate` with a known-bad model exits 3 and prints the
      diagnosis line.

## Measurement honesty (any release that touches measuring)

Added 2026-09-02 after the modellab handoff found a headline number
overstated by 24%. These are cheap to check and expensive to get wrong,
because a bad number is written into `measurements.json` and then into
the user's agent configs and the shareable report.

- [ ] **Benchmarks state their conditions.** Any tokens/sec the app
      shows or exports says whether it is empty-cache or at depth, and
      which. No bare "38 t/s".
- [ ] **Preconditions are enforced, not documented.** Before anything
      writes a measurement: our router's models unloaded, no Ollama
      residency, no managed build running. Verify by starting a bench
      with `ollama run <model>` resident — it must REFUSE and name the
      tenant, not produce a number.
- [ ] **Host conditions unchanged since the last baseline** (added
      2026-09-03 from the practitioner guide, `docs/research/…` §7.2).
      Not yet enforced in code, so check by hand before a release whose
      numbers matter: RAM at rated XMP speed (`sudo dmidecode -t memory
      | grep Configured` — below rated took that author's MoE
      generation to a third), CPU governor and EPP, and no swap growth
      during the run (`grep pswpout /proc/vmstat` before and after).
      A number taken under different host conditions is not comparable
      to a stored one, and none of these appear in the output.
- [ ] **Projections are labelled as projections.** `settled ctx` is
      llama.cpp's `--fit` output, not a measured ceiling; the report and
      any user-facing text must not call it "measured … not estimated".
- [ ] **`--report` renders without ragged rows** and its preamble
      matches what the columns actually contain (`cargo test report::`
      pins both, but read one real report before a tag).

## Pre-tag review (larger releases)

- [ ] A fresh adversarial session reviews the diff since the last tag
      (the v0.4.0 pattern: 8 angles, findings fixed before tagging).
      Skippable for pure-docs or single-fix point releases.

## Ship

- [ ] Bump `Cargo.toml`, commit, `git tag vX.Y.Z`, push with tags.
- [ ] Watch release.yml: tarball + sha256 assets on GitHub, crate on
      crates.io.
- [ ] Update ROADMAP's "Where things stand" header to the new tag.
