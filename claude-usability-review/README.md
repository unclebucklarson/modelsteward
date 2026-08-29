# Claude usability review (adversarial)

**Session role:** a second Claude Code session asked to adversarially evaluate
this repo for *usability* — anything that would confuse the user — and to
provide feedback only. Nothing outside this folder is modified by this session.

**Date:** 2026-08-29 · **Reviewed at:** commit 4742534 (v0.5.0)

**Lens:** the app's own north star — "max coding-agent performance *without
requiring llama.cpp expertise*." Every finding is judged against a user who
does NOT know what pp/tg, ub, KV, or mmproj mean and who does not read the
activity log unprompted.

## Contents

- `gui-findings.md` — the egui shell (src/ui.rs), the main event
- `cli-findings.md` — the CLI surface (src/main.rs + report/diagnose)
- `docs-onboarding-findings.md` — README, first-run, naming, jargon
- `SUMMARY.md` — prioritized synthesis + suggested next actions

## Note to the development session

This folder is the coordination channel (sessions can't talk directly).
Suggested protocol:

- Treat `SUMMARY.md` as the ranked worklist; findings files carry the
  file:line evidence.
- If you act on an item, append a line under "Dev session responses" at the
  bottom of the relevant file (finding ID + what you did / why you declined).
  This session may run again and will diff against those notes.
- Disagreement is expected and useful — several findings are judgment calls
  flagged as such.
