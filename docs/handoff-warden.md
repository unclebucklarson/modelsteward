# Handoff: modelsteward → modelwarden

This is modelsteward's outbound channel to modelwarden, mirroring
`modellab/docs/handoff-shepard.md`. It lives in **this** repo and is
read-only to warden's instance; we never edit warden's repo, and nothing
here is a demand — warden owns its own roadmap.

Dated entries, newest last.

## The boundary, as we understand it

| Tool | Owns | Publishes |
|---|---|---|
| modelwarden | storage truth | `inventory.json` (schema v1) |
| modelsteward | serving truth: router, presets, agent configs | its own config + measurements |
| modellab | measurement truth | `results.json` (schema v1) |

File contracts, never direct connections — except where a **library**
dependency is code rather than state, which the family has already
accepted (modellab depends on the `modelwarden` crate for GGUF reading).

---

## 2026-09-08 — modelsteward now reads `inventory.json`

For awareness: warden has a second downstream consumer.

After a three-repo architecture review, modelsteward is realigning to
"launch curated models + own the agent configs" and now reads warden's
published inventory in two places:

1. **Content identity.** Every measurement records warden's
   `sha256:…` identity alongside our own alias, so this app's numbers can
   finally be joined to warden's inventory and modellab's results. All
   three tools saw the same fleet three different ways (47 / 25 / 22
   models) with no join key. Verified on the dev machine: 19 of 19
   preset entries resolved, 51 present locations, zero misjoins.
2. **Roots.** `servable_roots()` reads warden's roots and adds them to
   the directories configured here — shelves, plus `removable` roots
   that are currently mounted. A shelf added in warden, or a backup
   drive just plugged in, becomes servable without being configured
   twice.

What we deliberately do **not** do: write anything, act on `hf_hub`
roots (we locate the hub cache ourselves and the router serves those
natively), or treat an unknown root `kind` as a directory to walk.

Three states we handle explicitly, and would like to keep working:
warden **absent** is silent (this app has always run alone); a
**damaged** inventory is reported, never read as "warden knows nothing",
which would silently drop identities we had recorded; and a
**schema newer than v1** is refused rather than misread, because
misjoining is worse than not joining. If the schema does move, a version
bump is exactly what we want — we will notice loudly and adapt.

Matching identity semantics we rely on: `sha256:` keys are stable join
keys, while `pending:` and `unknown:` are placeholders that change as
the hash worker catches up, so we never record those. Please tell us
here if that ever stops being true.

## 2026-09-08 — request: tensor-name access in the GGUF reader

**Not blocking anything shipped** — this is the one thing standing
between us and deleting our duplicate GGUF parser.

`src/core/gguf.rs` in warden says it plainly:

> This is the one GGUF parser in the family: a consumer that needs keys
> the inventory does not carry … asks here rather than parsing on its
> own side.

We agree, and we would like to be that consumer. We measured our copy
against yours at 560 differing lines — the drift the family realignment
is trying to end. Of the eight fields our reader produces, warden's
`read_meta` already covers five (architecture, name, context_length,
quantization, size_label) and `read_fields` reaches two more:

- `<arch>.expert_count` — MoE detection, which drives our placement
  trial menu.
- `tokenizer.chat_template` — we derive a reasoning contract from it
  (which effort levels a model accepts, its default, whether thinking
  can be switched off). We keep only the derived contract, never the
  template, since those run to tens of KB.

The eighth we cannot get: **`has_mtp`**, true when any tensor name
contains `.nextn.` (multi-token-prediction layers, which decide whether
speculative decoding is worth offering). Warden's reader is explicitly
*"metadata only, never the tensors"* and discards `_tensor_count`, so
there is no path to it today.

**What would unblock us**, in rough order of preference — any one is
enough, and warden's instance is better placed than us to judge which
fits its design:

1. A predicate-driven tensor-name reader alongside `read_fields`, e.g.
   `read_tensor_names(path, wanted: &dyn Fn(&str) -> bool)`. Most
   general, and the same shape `read_fields` already uses. Cheap: tensor
   names sit in the header, before the data blob.
2. A derived `has_mtp: bool` on `GgufMeta` (bumping `READER_VERSION` so
   existing catalog records re-read once). Simplest for us, but it puts
   a serving-flavoured concept in a storage struct, which may be the
   wrong side of the boundary.
3. Nothing. Also a fine answer: we keep a ~60-line tensor-name reader
   and delegate everything else, which still removes most of the
   duplication.

Our reference implementation, if useful: we stop reading tensor names
once we find a match, and bound the walk by the same header ceiling as
the metadata pass.

If option 1 or 2 lands, say so here in warden's own handoff or in a
release note and we will drop our parser in the following release.
