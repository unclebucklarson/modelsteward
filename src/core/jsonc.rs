//! Harvested verbatim from opencode_configuration_tool (src/config/jsonc.rs)
//! per PLAN.md — see that repo for its full change history.
//!
//! Comment-preserving JSONC editor.
//!
//! Uses `jsonc-parser` to build an AST with span info, then splices new /
//! updated model entries into the source string so user comments, indentation,
//! and trailing-comma style survive.
//!
//! Supports two operations against `provider.<id>.models`:
//!   * [`add_model`] — insert a new entry before the closing `}`.
//!   * [`merge_model`] — splice individual keys into an existing entry.
//!
//! There is deliberately no whole-entry replace. An earlier `update_model`
//! swapped an entry's entire value span, which silently discarded any key or
//! comment the user had written inside that entry by hand. Everything now goes
//! through [`merge_model`], which only touches the keys it is given.
//!
//! When the required containers (`provider`, `provider.<id>`, or their
//! `.models` map) are absent, an [`EditError::MissingContainer`] is returned so
//! the caller can fall back to a full pretty-print re-render.

use jsonc_parser::ast::{Object, Value};
use jsonc_parser::common::Range;
use jsonc_parser::{CollectOptions, ParseOptions, parse_to_ast};
use serde_json::Value as JsonValue;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EditError {
    #[error("failed to parse JSONC: {0}")]
    Parse(String),
    #[error("expected root value to be an object")]
    RootNotObject,
    #[error("{path} is not present in the source")]
    MissingContainer { path: String },
    #[error("{path} is present but is not an object")]
    NotObject { path: String },
    #[error("model {model_id:?} not found under provider {provider_id:?}")]
    ModelNotFound {
        provider_id: String,
        model_id: String,
    },
    #[error("internal: failed to serialize model entry: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// Insert a new `"model_id": entry` pair into `provider.<provider_id>.models`.
///
/// Preserves comments and existing formatting. Matches the trailing-comma
/// style already in use in the models object.
pub fn add_model(
    source: &str,
    provider_id: &str,
    model_id: &str,
    entry: &JsonValue,
) -> Result<String, EditError> {
    let ast = parse_source(source)?;
    let root = extract_root_object(&ast)?;
    let models = navigate_to_models(root, provider_id)?;
    let indent = detect_property_indent(source, models);
    let entry_text = serialize_entry(model_id, entry, &indent)?;

    Ok(splice_into_object(source, models, &entry_text))
}

/// Make sure `provider.<provider_id>.models` exists, creating whichever
/// containers are missing by splicing them in.
///
/// Without this, a config that names a provider but has no `models` key yet —
/// exactly what you get when someone sets up `npm` and `baseURL` first and
/// runs this tool second — sent the whole file down the pretty-print fallback,
/// destroying every comment in it. Creating the container in place keeps the
/// edit surgical.
///
/// `scaffold` is the full provider block to use if `provider.<provider_id>` is
/// absent entirely; it should already contain an empty `models` object.
pub fn ensure_models_container(
    source: &str,
    provider_id: &str,
    scaffold: &JsonValue,
) -> Result<String, EditError> {
    // `ProviderBlock` skips an empty `models` map when it serializes, so a
    // scaffold arrives here without one. Splicing it in as-is would create a
    // provider block that the very next `add_model` can't navigate into,
    // sending the file down the fallback this function exists to avoid.
    let mut scaffold = scaffold.clone();
    match scaffold.as_object_mut() {
        Some(obj) => {
            obj.entry("models").or_insert_with(|| serde_json::json!({}));
        }
        None => {
            return Err(EditError::NotObject {
                path: format!("scaffold for provider.{provider_id}"),
            });
        }
    }
    let scaffold = &scaffold;

    let ast = parse_source(source)?;
    let root = extract_root_object(&ast)?;

    // No `provider` container at all — add one holding just this provider.
    let Some(provider_prop) = root.get("provider") else {
        let indent = detect_property_indent(source, root);
        let block = serde_json::json!({ provider_id: scaffold });
        let text = serialize_entry("provider", &block, &indent)?;
        return Ok(splice_into_object(source, root, &text));
    };
    let providers = provider_prop
        .value
        .as_object()
        .ok_or_else(|| EditError::NotObject {
            path: "provider".into(),
        })?;

    // `provider` exists but not this one — add the scaffold beside its peers.
    let Some(block_prop) = providers.get(provider_id) else {
        let indent = detect_property_indent(source, providers);
        let text = serialize_entry(provider_id, scaffold, &indent)?;
        return Ok(splice_into_object(source, providers, &text));
    };
    let block = block_prop
        .value
        .as_object()
        .ok_or_else(|| EditError::NotObject {
            path: format!("provider.{provider_id}"),
        })?;

    // The block is there but has no `models` map yet.
    if block.get("models").is_none() {
        let indent = detect_property_indent(source, block);
        let text = serialize_entry("models", &serde_json::json!({}), &indent)?;
        return Ok(splice_into_object(source, block, &text));
    }

    Ok(source.to_string())
}

/// Merge `patch` into the existing `provider.<provider_id>.models.<model_id>`
/// entry, touching **only** the keys `patch` actually contains.
///
/// Semantics:
///   * a key in `patch` that already exists has its value replaced;
///   * a key in `patch` that is absent is inserted;
///   * a key *not* in `patch` is left byte-for-byte alone, including the
///     comments and whitespace around it;
///   * where both sides are objects (`limit`, `capabilities`) the merge
///     recurses key-by-key rather than replacing the object wholesale.
///
/// So refreshing a model's context length cannot disturb the `temperature` the
/// user set by hand two lines below it.
pub fn merge_model(
    source: &str,
    provider_id: &str,
    model_id: &str,
    patch: &JsonValue,
) -> Result<String, EditError> {
    let patch_obj = patch.as_object().ok_or_else(|| EditError::NotObject {
        path: format!("patch for model {model_id:?}"),
    })?;
    if patch_obj.is_empty() {
        return Ok(source.to_string());
    }

    let ast = parse_source(source)?;
    let root = extract_root_object(&ast)?;
    let models = navigate_to_models(root, provider_id)?;
    let prop = models
        .properties
        .iter()
        .find(|p| p.name.as_str() == model_id)
        .ok_or_else(|| EditError::ModelNotFound {
            provider_id: provider_id.into(),
            model_id: model_id.into(),
        })?;

    let mut splices = Vec::new();
    match prop.value.as_object() {
        Some(entry_obj) => collect_merge_splices(source, entry_obj, patch_obj, &mut splices)?,
        None => {
            // The entry isn't an object (null, a string, …) so there's nothing
            // to merge into — replace it outright.
            let indent = detect_property_indent(source, models);
            let r = value_range(&prop.value);
            splices.push(Splice {
                start: r.start,
                end: r.end,
                text: serialize_value_at_indent(patch, &indent)?,
            });
        }
    }
    Ok(apply_splices(source, splices))
}

/// Comment out an entire model entry, leaving it in the file.
///
/// Removal is the one destructive thing this tool does, so it isn't a
/// deletion: the entry is commented in place under `note`, which keeps the
/// original values visible and lets the user restore them by hand. Deleting
/// the commented block is then their decision, made with the values in front
/// of them.
///
/// The separating comma is removed so the result is still strict JSON: the
/// entry's own trailing comma normally, or the previous property's comma when
/// the entry being removed was the last one.
pub fn comment_out_model(
    source: &str,
    provider_id: &str,
    model_id: &str,
    note: &str,
) -> Result<String, EditError> {
    let ast = parse_source(source)?;
    let root = extract_root_object(&ast)?;
    let models = navigate_to_models(root, provider_id)?;
    let prop = models
        .properties
        .iter()
        .find(|p| p.name.as_str() == model_id)
        .ok_or_else(|| EditError::ModelNotFound {
            provider_id: provider_id.into(),
            model_id: model_id.into(),
        })?;

    let idx = models
        .properties
        .iter()
        .position(|p| p.name.as_str() == model_id)
        .expect("found above");
    let indent = detect_property_indent(source, models);
    let start = prop.range.start;
    let value_end = value_range(&prop.value).end;

    let mut splices = vec![Splice {
        start,
        end: value_end,
        text: comment_block(&source[start..value_end], &indent, note),
    }];

    // Commenting a property out leaves its separating comma behind as live
    // JSON. Locate that comma from AST positions rather than by scanning raw
    // bytes outward: the gap around a property can contain the user's own
    // comments, and a byte scan stops at the first one it meets.
    let comma = if let Some(next) = models.properties.get(idx + 1) {
        // Not last: drop our own trailing comma, in the gap before the next
        // property. Any comment in that gap belongs to the next property, so
        // it must stay put — which is why only the comma is removed.
        find_separator_comma(source, value_end, next.range.start)
    } else if idx > 0 {
        // Last property: our comma doesn't exist. The one after the *previous*
        // property is what would dangle before the closing brace.
        let prev_end = value_range(&models.properties[idx - 1].value).end;
        find_separator_comma(source, prev_end, start)
    } else {
        // Only property — there is no comma at all.
        None
    };
    if let Some(pos) = comma {
        splices.push(Splice {
            start: pos,
            end: pos + 1,
            text: String::new(),
        });
    }

    Ok(apply_splices(source, splices))
}

/// Find the property-separating comma in `source[from..to]`, ignoring any
/// commas that appear inside comments.
///
/// The span between two properties holds only whitespace, comments, and the
/// single separating comma, so the first comma found outside a comment is it.
fn find_separator_comma(source: &str, from: usize, to: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut i = from;
    while i < to.min(bytes.len()) {
        match bytes[i] {
            b',' => return Some(i),
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                // Line comment: skip to end of line.
                while i < to && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                // Block comment: skip to the closing delimiter.
                i += 2;
                while i + 1 < to && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Render `original` as a run of `//` comments at `indent`, under `note`.
///
/// The caller splices this in starting at the property's first character, so
/// the leading indent is already present in the source — the first line gets
/// no indent of its own. Later lines have the property indent stripped and
/// re-added so their relative nesting survives.
fn comment_block(original: &str, indent: &str, note: &str) -> String {
    let mut out = String::new();
    for (i, line) in note.lines().enumerate() {
        if i > 0 {
            out.push_str(indent);
        }
        out.push_str("// ");
        out.push_str(line);
        out.push('\n');
    }
    for (i, line) in original.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(indent);
        out.push_str("// ");
        out.push_str(line.strip_prefix(indent).unwrap_or(line));
    }
    out
}

// ─── internal helpers ────────────────────────────────────────────────────────

/// A pending text replacement: `source[start..end]` becomes `text`.
#[derive(Debug)]
struct Splice {
    start: usize,
    end: usize,
    text: String,
}

/// Walk `patch` against `target`, recording the minimal set of replacements.
///
/// Disjointness is structural: a key is either recursed into or replaced,
/// never both, and insertions land in the gap before the closing brace.
fn collect_merge_splices(
    source: &str,
    target: &Object<'_>,
    patch: &serde_json::Map<String, JsonValue>,
    out: &mut Vec<Splice>,
) -> Result<(), EditError> {
    let indent = detect_property_indent(source, target);
    let mut insertions: Vec<(&str, &JsonValue)> = Vec::new();

    for (key, new_value) in patch {
        match target.properties.iter().find(|p| p.name.as_str() == key) {
            Some(prop) => match (new_value.as_object(), prop.value.as_object()) {
                // Both objects — recurse, so untouched sibling keys and the
                // comments between them survive verbatim.
                (Some(sub_patch), Some(sub_target)) => {
                    collect_merge_splices(source, sub_target, sub_patch, out)?;
                }
                // Scalar, or a shape change (object <-> non-object): replace.
                _ => {
                    let r = value_range(&prop.value);
                    out.push(Splice {
                        start: r.start,
                        end: r.end,
                        text: serialize_value_at_indent(new_value, &indent)?,
                    });
                }
            },
            None => insertions.push((key.as_str(), new_value)),
        }
    }

    if !insertions.is_empty() {
        out.push(build_insertion_splice(
            source,
            target,
            &insertions,
            &indent,
        )?);
    }
    Ok(())
}

/// Build the single splice that appends `entries` before `object`'s closing
/// brace, matching the surrounding comma and indent style.
/// Where a separating comma must go when appending to `object`, and
/// where the new entry starts.
///
/// The comma belongs immediately after the last property's VALUE — NOT
/// after the last non-whitespace byte. Those differ whenever a comment
/// trails the last entry, and the app produces exactly that shape
/// itself (comment_out_ghosts leaves a commented block as the last
/// content). Writing the comma at the end of a `//` line buries it in
/// the comment, where a strict parser never sees it: the file becomes
/// invalid JSON while our own lenient reader still parses it, so every
/// surface reports success and OpenCode cannot load the file.
/// Live-reproduced 2026-08-31 (review finding C6).
///
/// Returns `(comma_pos, insert_pos)`: `comma_pos` is `None` when no
/// comma is needed (empty object, or the last property already has a
/// trailing comma).
fn append_positions(source: &str, object: &Object<'_>, close_pos: usize) -> (Option<usize>, usize) {
    // Insert point: back up over the whitespace run before `}` so the
    // entry lands after real content, not after a blank line.
    let before_close = &source[..close_pos];
    let bytes = before_close.as_bytes();
    let mut insert_pos = before_close.len();
    while insert_pos > 0 && bytes[insert_pos - 1].is_ascii_whitespace() {
        insert_pos -= 1;
    }
    let Some(last) = object.properties.last() else {
        return (None, insert_pos); // empty object: no comma needed
    };
    let value_end = value_range(&last.value).end;
    // Already terminated? Scan the gap between the last value and the
    // insert point for a comma that isn't inside a comment.
    let gap = &source[value_end..insert_pos];
    if gap_has_comma(gap) {
        return (None, insert_pos);
    }
    (Some(value_end), insert_pos)
}

/// True when `gap` contains a real `,` — one outside any `//` or `/* */`
/// comment. Mirrors find_separator_comma's comment-awareness, which the
/// removal path always had and the insertion path lacked.
fn gap_has_comma(gap: &str) -> bool {
    let b = gap.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b',' => return true,
            b'/' if i + 1 < b.len() && b[i + 1] == b'/' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < b.len() && b[i + 1] == b'*' => {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i += 2;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    false
}

fn build_insertion_splice(
    source: &str,
    object: &Object<'_>,
    entries: &[(&str, &JsonValue)],
    indent: &str,
) -> Result<Splice, EditError> {
    let close_pos = object.range.end - 1; // position of the `}` byte
    debug_assert_eq!(&source[close_pos..close_pos + 1], "}");

    let (comma_pos, cursor) = append_positions(source, object, close_pos);
    // The span starts at the comma position when one is needed, so the
    // comma lands right after the last VALUE and any trailing comment
    // is carried through verbatim after it.
    let start = comma_pos.unwrap_or(cursor);
    let mut text = String::new();
    if let Some(cp) = comma_pos {
        text.push(',');
        text.push_str(&source[cp..cursor]);
    }
    for (i, (key, value)) in entries.iter().enumerate() {
        if i > 0 {
            text.push(',');
        }
        text.push('\n');
        text.push_str(indent);
        text.push_str(&serialize_entry(key, value, indent)?);
    }
    text.push('\n');
    text.push_str(&outer_indent_of(source, object));

    Ok(Splice {
        start,
        end: close_pos,
        text,
    })
}

/// The indent of the line `object`'s opening brace sits on — i.e. where its
/// closing brace belongs.
fn outer_indent_of(source: &str, object: &Object<'_>) -> String {
    let open_pos = object.range.start;
    source[..open_pos]
        .rfind('\n')
        .map(|nl| {
            source[nl + 1..open_pos]
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .collect()
        })
        .unwrap_or_default()
}

/// Apply splices right-to-left so the offsets of the not-yet-applied ones stay
/// valid against the original source.
fn apply_splices(source: &str, mut splices: Vec<Splice>) -> String {
    splices.sort_by_key(|s| std::cmp::Reverse(s.start));
    debug_assert!(
        splices.windows(2).all(|w| w[1].end <= w[0].start),
        "merge splices must be disjoint: {splices:?}"
    );
    let mut out = source.to_string();
    for s in splices {
        out.replace_range(s.start..s.end, &s.text);
    }
    out
}

/// Ephemeral holder for the parsed AST + reference to its root value.
///
/// jsonc-parser's `Value` borrows from the source string, so we can't return
/// it directly out of a helper without lifetime gymnastics. Callers keep this
/// alive for the duration of the edit.
struct ParsedAst<'a> {
    value: Value<'a>,
}

/// Would a STRICT JSON parser accept this text once comments are
/// removed? Our own parser is lenient about missing commas — exactly
/// the damage a bad splice causes — so it cannot answer this, and
/// without it the app reported success on a file OpenCode could not
/// load (review finding C6, 2026-08-31). Comment ranges come from the
/// parser itself, so strings containing `//` are handled correctly.
pub fn strictly_valid(source: &str) -> Result<(), String> {
    let result = parse_to_ast(
        source,
        &CollectOptions {
            comments: jsonc_parser::CommentCollectionStrategy::Separate,
            tokens: false,
        },
        &ParseOptions::default(),
    )
    .map_err(|e| e.to_string())?;
    // Blank out every comment, preserving byte offsets and newlines so
    // any error message still points at the right line.
    let mut bytes = source.as_bytes().to_vec();
    if let Some(map) = result.comments {
        for comments in map.values() {
            for c in comments.iter() {
                let r = match c {
                    jsonc_parser::ast::Comment::Line(l) => l.range,
                    jsonc_parser::ast::Comment::Block(b) => b.range,
                };
                for b in &mut bytes[r.start..r.end.min(source.len())] {
                    if *b != b'\n' {
                        *b = b' ';
                    }
                }
            }
        }
    }
    let stripped = String::from_utf8_lossy(&bytes);
    serde_json::from_str::<serde_json::Value>(&stripped)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn parse_source(source: &str) -> Result<ParsedAst<'_>, EditError> {
    let result = parse_to_ast(source, &CollectOptions::default(), &ParseOptions::default())
        .map_err(|e| EditError::Parse(e.to_string()))?;
    let value = result.value.ok_or(EditError::RootNotObject)?;
    Ok(ParsedAst { value })
}

fn extract_root_object<'a, 'b>(ast: &'b ParsedAst<'a>) -> Result<&'b Object<'a>, EditError> {
    ast.value.as_object().ok_or(EditError::RootNotObject)
}

/// Navigate `root -> provider -> <provider_id> -> models`. Returns the models
/// object; returns `MissingContainer` if any hop is absent.
fn navigate_to_models<'a, 'b>(
    root: &'b Object<'a>,
    provider_id: &str,
) -> Result<&'b Object<'a>, EditError> {
    let provider = root
        .get("provider")
        .ok_or_else(|| EditError::MissingContainer {
            path: "provider".into(),
        })?;
    let providers = provider
        .value
        .as_object()
        .ok_or_else(|| EditError::NotObject {
            path: "provider".into(),
        })?;
    let block = providers
        .get(provider_id)
        .ok_or_else(|| EditError::MissingContainer {
            path: format!("provider.{provider_id}"),
        })?;
    let block_obj = block
        .value
        .as_object()
        .ok_or_else(|| EditError::NotObject {
            path: format!("provider.{provider_id}"),
        })?;
    let models = block_obj
        .get("models")
        .ok_or_else(|| EditError::MissingContainer {
            path: format!("provider.{provider_id}.models"),
        })?;
    models
        .value
        .as_object()
        .ok_or_else(|| EditError::NotObject {
            path: format!("provider.{provider_id}.models"),
        })
}

/// Detect the indent (whitespace prefix) used for the *contents* of `object`.
///
/// Strategy: peek at the source region between the open `{` and the first
/// existing property; take the whitespace between the last `\n` and that
/// property. If the object is empty, derive from the object's own line indent
/// + two extra spaces (a common default).
fn detect_property_indent(source: &str, object: &Object<'_>) -> String {
    if let Some(first) = object.properties.first() {
        let start = first.range.start;
        let before = &source[..start];
        if let Some(nl) = before.rfind('\n') {
            let indent: String = source[nl + 1..start]
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .collect();
            if !indent.is_empty() {
                return indent;
            }
        }
    }
    // Fallback: two spaces beyond the object's own indent.
    let obj_start = object.range.start;
    let before = &source[..obj_start];
    let outer_indent: String = before
        .rfind('\n')
        .map(|nl| {
            source[nl + 1..obj_start]
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .collect()
        })
        .unwrap_or_default();
    format!("{outer_indent}  ")
}

fn value_range(v: &Value<'_>) -> Range {
    match v {
        Value::StringLit(s) => s.range,
        Value::NumberLit(n) => n.range,
        Value::BooleanLit(b) => b.range,
        Value::NullKeyword(n) => n.range,
        Value::Object(o) => o.range,
        Value::Array(a) => a.range,
    }
}

/// Render `entry` as a JSONC object literal (multi-line, pretty), indented so
/// each line is prefixed by `property_indent` (matching the surrounding
/// properties).
fn serialize_entry(
    model_id: &str,
    entry: &JsonValue,
    property_indent: &str,
) -> Result<String, EditError> {
    let value_text = serialize_value_at_indent(entry, property_indent)?;
    let key = serde_json::to_string(model_id)?;
    Ok(format!("{key}: {value_text}"))
}

/// Serialize a JSON value with a fixed left-indent so its rendered form fits
/// as the value part of an object property at `property_indent`. Uses 2-space
/// nested indent (matches opencode's convention).
fn serialize_value_at_indent(v: &JsonValue, property_indent: &str) -> Result<String, EditError> {
    let raw = serde_json::to_string_pretty(v)?;
    // `to_string_pretty` uses two-space indent starting at column 0. Prepend
    // the property indent to every line *except the first* so the opening
    // token stays flush with the key.
    let mut out =
        String::with_capacity(raw.len() + raw.matches('\n').count() * property_indent.len());
    for (i, line) in raw.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
            out.push_str(property_indent);
        }
        out.push_str(line);
    }
    Ok(out)
}

/// Splice `entry_text` (e.g. `"foo": { ... }`) into `object` before its
/// closing `}`. Honors trailing-comma style: if existing last property has a
/// trailing comma, keep that; otherwise add one to the previous last property.
fn splice_into_object(source: &str, object: &Object<'_>, entry_text: &str) -> String {
    let close_pos = object.range.end - 1; // position of the `}` byte
    debug_assert_eq!(&source[close_pos..close_pos + 1], "}");
    let indent = detect_property_indent(source, object);

    // Find the last non-whitespace char before the `}` to see if we need a
    // preceding comma.
    // Same comment-safe anchoring as build_insertion_splice: the comma
    // goes after the last VALUE, never at the end of a trailing comment
    // line (review finding C6, 2026-08-31).
    let (comma_pos, cursor) = append_positions(source, object, close_pos);

    let mut out = String::with_capacity(source.len() + entry_text.len() + indent.len() + 4);
    match comma_pos {
        Some(cp) => {
            out.push_str(&source[..cp]);
            out.push(',');
            out.push_str(&source[cp..cursor]);
        }
        None => out.push_str(&source[..cursor]),
    }
    out.push('\n');
    out.push_str(&indent);
    out.push_str(entry_text);
    // Newline + the outer object's indent before the closing brace.
    out.push('\n');
    // Determine the indent of the closing brace's own line (the object's own
    // indent) by looking at what preceded the `{`.
    let open_pos = object.range.start;
    let outer_indent: String = source[..open_pos]
        .rfind('\n')
        .map(|nl| {
            source[nl + 1..open_pos]
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .collect()
        })
        .unwrap_or_default();
    out.push_str(&outer_indent);
    out.push_str(&source[close_pos..]);
    out
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Strip `//` line comments the way a strict JSON consumer must,
    /// then parse. This is the contract every write has to meet:
    /// OpenCode reads this file with a real JSON parser, and our own
    /// reader is LENIENT about missing commas — so only a strict parse
    /// can tell us we corrupted it. (Review 2026-08-31, finding C6.)
    fn strict_parse(src: &str) -> Result<serde_json::Value, String> {
        let stripped: String = src
            .lines()
            .map(|l| match l.find("//") {
                // Naive but sufficient for these fixtures: no `//`
                // appears inside a string value in them.
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n");
        serde_json::from_str(&stripped).map_err(|e| e.to_string())
    }

    #[test]
    fn inserting_after_a_trailing_comment_keeps_the_file_strictly_valid() {
        // The app CREATES this shape itself: comment_out_ghosts leaves a
        // commented block as the last content of the models object, and
        // the next sync inserts after it. The separating comma used to
        // land INSIDE the comment, where a strict parser never sees it —
        // and our lenient reader reported success while OpenCode could
        // not load the file.
        let src = r#"{
  "provider": {
    "llamacpp": {
      "models": {
        "keeper": { "name": "Keeper" }
        // Commented out by modelsteward: not in the current router preset.
        // "ghost": { "name": "Ghost" }
      }
    }
  }
}"#;
        let out = add_model(
            src,
            "llamacpp",
            "newmodel",
            &serde_json::json!({ "name": "New" }),
        )
        .unwrap();
        let parsed = strict_parse(&out).unwrap_or_else(|e| panic!("STRICT PARSE FAILED: {e}\n{out}"));
        let models = &parsed["provider"]["llamacpp"]["models"];
        assert!(models.get("keeper").is_some(), "{out}");
        assert!(models.get("newmodel").is_some(), "{out}");
        // The comment itself must survive — that is the whole point of
        // the JSONC editor.
        assert!(out.contains("Commented out by modelsteward"), "{out}");
        assert!(out.contains("// \"ghost\""), "{out}");
    }

    #[test]
    fn inserting_after_an_inline_comment_keeps_the_file_strictly_valid() {
        // Same bug, a user's own shape: a trailing note after the last
        // entry on the same line.
        let src = r#"{
  "provider": {
    "llamacpp": {
      "models": {
        "keeper": { "name": "Keeper" } // my daily driver
      }
    }
  }
}"#;
        let out = add_model(
            src,
            "llamacpp",
            "newmodel",
            &serde_json::json!({ "name": "New" }),
        )
        .unwrap();
        let parsed = strict_parse(&out).unwrap_or_else(|e| panic!("STRICT PARSE FAILED: {e}\n{out}"));
        assert!(parsed["provider"]["llamacpp"]["models"]["newmodel"].is_object(), "{out}");
        assert!(out.contains("my daily driver"), "comment must survive: {out}");
    }

    const REAL: &str = include_str!("../../tests/fixtures/opencode_real.json");
    const COMMENTED: &str = include_str!("../../tests/fixtures/opencode_commented.jsonc");

    #[test]
    fn parses_real_config_ok() {
        parse_source(REAL).expect("real config parses");
    }

    #[test]
    fn parses_commented_config_ok() {
        parse_source(COMMENTED).expect("commented config parses");
    }

    #[test]
    fn navigates_to_ollama_models() {
        let ast = parse_source(REAL).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let models = navigate_to_models(root, "ollama").unwrap();
        assert_eq!(models.properties.len(), 3);
    }

    #[test]
    fn navigate_missing_provider_errors() {
        let ast = parse_source(REAL).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let err = navigate_to_models(root, "vllm").unwrap_err();
        assert!(matches!(err, EditError::MissingContainer { .. }));
    }

    #[test]
    fn adds_model_and_keeps_valid_jsonc() {
        let entry = json!({
            "name": "qwen3.6-coder:7b",
            "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
            "limit": { "context": 65536, "output": 8192 }
        });
        let out = add_model(REAL, "ollama", "qwen3.6-coder:7b", &entry).unwrap();
        // Re-parsing must succeed and the model must appear.
        let ast = parse_source(&out).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let models = navigate_to_models(root, "ollama").unwrap();
        assert!(
            models
                .properties
                .iter()
                .any(|p| p.name.as_str() == "qwen3.6-coder:7b")
        );
        assert_eq!(models.properties.len(), 4, "3 existing + 1 new");
    }

    #[test]
    fn add_model_preserves_comments() {
        let entry = json!({"name": "new-model"});
        let out = add_model(COMMENTED, "ollama", "new-model", &entry).unwrap();
        // All the original comments must still be there.
        assert!(out.contains("// Opencode configuration"));
        assert!(out.contains("must remain untouched"));
        assert!(out.contains("hand-tuned display name"));
        assert!(out.contains("Pre-existing entry with minimal metadata"));
    }

    #[test]
    fn add_model_leaves_other_providers_untouched() {
        let entry = json!({"name": "x"});
        let out = add_model(REAL, "ollama", "x", &entry).unwrap();
        // The llamacpp block should be byte-identical to what it was.
        let llamacpp_orig = extract_provider_block(REAL, "llamacpp");
        let llamacpp_out = extract_provider_block(&out, "llamacpp");
        assert_eq!(
            llamacpp_orig, llamacpp_out,
            "llamacpp block must not change when editing ollama"
        );
    }

    /// The bug this whole merge path exists to fix: a refresh used to replace
    /// the entry's entire value span, taking the user's own keys and the
    /// comments between them with it.
    #[test]
    fn merge_preserves_hand_tuned_siblings_and_comments() {
        let src = r#"{
  "provider": {
    "ollama": {
      "models": {
        "m1": {
          // my own note
          "name": "m1",
          "temperature": 0.2,
          "hand_tuned": 42
        }
      }
    }
  }
}"#;
        let patch = json!({ "limit": { "context": 8192 } });
        let out = merge_model(src, "ollama", "m1", &patch).unwrap();

        assert!(out.contains("// my own note"), "comment lost:\n{out}");
        assert!(out.contains("\"temperature\""), "temperature lost:\n{out}");
        assert!(out.contains("\"hand_tuned\""), "hand_tuned lost:\n{out}");
        assert!(out.contains("8192"), "patch not applied:\n{out}");

        // And the result is still valid JSONC with the merged value in place.
        let ast = parse_source(&out).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let models = navigate_to_models(root, "ollama").unwrap();
        let entry = models.get("m1").unwrap().value.as_object().unwrap();
        let limit = entry.get("limit").unwrap().value.as_object().unwrap();
        assert_eq!(limit.get_number("context").unwrap().value, "8192");
    }

    /// Chosen semantics: a refresh overwrites keys the provider reports.
    #[test]
    fn merge_overwrites_an_existing_value() {
        let src = r#"{
  "provider": { "ollama": { "models": {
    "m1": { "name": "old", "limit": { "context": 32768, "output": 4096 } }
  } } }
}"#;
        let patch = json!({ "limit": { "context": 262144 } });
        let out = merge_model(src, "ollama", "m1", &patch).unwrap();

        let ast = parse_source(&out).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let models = navigate_to_models(root, "ollama").unwrap();
        let entry = models.get("m1").unwrap().value.as_object().unwrap();
        let limit = entry.get("limit").unwrap().value.as_object().unwrap();
        // context refreshed...
        assert_eq!(limit.get_number("context").unwrap().value, "262144");
        // ...while the sibling key inside the same nested object is untouched.
        assert_eq!(limit.get_number("output").unwrap().value, "4096");
        assert_eq!(entry.get_string("name").unwrap().value, "old");
    }

    #[test]
    fn merge_inserts_keys_that_are_absent() {
        let src = r#"{
  "provider": { "ollama": { "models": {
    "m1": { "name": "m1" }
  } } }
}"#;
        let patch = json!({
            "limit": { "context": 8192 },
            "capabilities": { "tools": true }
        });
        let out = merge_model(src, "ollama", "m1", &patch).unwrap();

        let ast = parse_source(&out).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let models = navigate_to_models(root, "ollama").unwrap();
        let entry = models.get("m1").unwrap().value.as_object().unwrap();
        assert_eq!(entry.get_string("name").unwrap().value, "m1");
        assert!(entry.get("limit").is_some());
        assert!(entry.get("capabilities").is_some());
    }

    #[test]
    fn merge_into_nested_object_that_does_not_exist_yet() {
        let src = r#"{
  "provider": { "ollama": { "models": { "m1": { "name": "m1" } } } }
}"#;
        let patch = json!({ "limit": { "context": 1024, "output": 512 } });
        let out = merge_model(src, "ollama", "m1", &patch).unwrap();
        let ast = parse_source(&out).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let models = navigate_to_models(root, "ollama").unwrap();
        let limit = models
            .get("m1")
            .unwrap()
            .value
            .as_object()
            .unwrap()
            .get("limit")
            .unwrap()
            .value
            .as_object()
            .unwrap();
        assert_eq!(limit.get_number("context").unwrap().value, "1024");
        assert_eq!(limit.get_number("output").unwrap().value, "512");
    }

    #[test]
    fn merge_with_empty_patch_is_a_no_op() {
        let out = merge_model(REAL, "ollama", "ornith:35b", &json!({})).unwrap();
        assert_eq!(out, REAL, "an empty patch must not rewrite the file");
    }

    #[test]
    fn merge_leaves_other_providers_untouched() {
        let patch = json!({ "limit": { "context": 4096 } });
        let out = merge_model(REAL, "ollama", "ornith:35b", &patch).unwrap();
        assert_eq!(
            extract_provider_block(REAL, "llamacpp"),
            extract_provider_block(&out, "llamacpp"),
            "llamacpp block must not change when merging into ollama"
        );
    }

    #[test]
    fn merge_on_the_commented_fixture_keeps_every_comment() {
        let patch = json!({ "limit": { "context": 4096 } });
        let out = merge_model(COMMENTED, "ollama", "ornith:35b", &patch).unwrap();
        assert!(out.contains("// Opencode configuration"));
        assert!(out.contains("must remain untouched"));
        assert!(out.contains("hand-tuned display name"));
        assert!(out.contains("Pre-existing entry with minimal metadata"));
    }

    #[test]
    fn merge_model_missing_errors() {
        let err = merge_model(REAL, "ollama", "does-not-exist", &json!({"name": "x"})).unwrap_err();
        assert!(matches!(err, EditError::ModelNotFound { .. }));
    }

    const NOTE: &str = "Removed: no longer reported by ollama.";

    /// Helper: comment out a model, then assert the file still parses and the
    /// model is gone from the parsed view.
    fn comment_out_and_reparse(src: &str, model: &str) -> String {
        let out = comment_out_model(src, "ollama", model, NOTE).unwrap();
        let ast = parse_source(&out).unwrap_or_else(|e| panic!("result must parse: {e}\n{out}"));
        let root = extract_root_object(&ast).unwrap();
        let models = navigate_to_models(root, "ollama").unwrap();
        assert!(
            !models.properties.iter().any(|p| p.name.as_str() == model),
            "model should be gone from the parsed config:\n{out}"
        );
        out
    }

    #[test]
    fn comment_out_a_middle_entry_keeps_the_file_parseable() {
        let src = r#"{
  "provider": {
    "ollama": {
      "models": {
        "a": { "name": "A" },
        "b": { "name": "B" },
        "c": { "name": "C" }
      }
    }
  }
}"#;
        let out = comment_out_and_reparse(src, "b");
        // Neighbours survive.
        let ast = parse_source(&out).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let models = navigate_to_models(root, "ollama").unwrap();
        assert_eq!(models.properties.len(), 2);
        // The original values are still readable in the file.
        assert!(out.contains(r#"// "b": { "name": "B" }"#), "{out}");
        assert!(out.contains(NOTE), "the note should explain why:\n{out}");
    }

    /// The comma case that breaks naive implementations: removing the last
    /// property leaves the previous one with a dangling comma.
    #[test]
    fn comment_out_the_last_entry_drops_the_dangling_comma() {
        let src = r#"{
  "provider": {
    "ollama": {
      "models": {
        "a": { "name": "A" },
        "b": { "name": "B" }
      }
    }
  }
}"#;
        let out = comment_out_and_reparse(src, "b");
        let ast = parse_source(&out).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let models = navigate_to_models(root, "ollama").unwrap();
        assert_eq!(models.properties.len(), 1);
        assert_eq!(models.properties[0].name.as_str(), "a");
    }

    #[test]
    fn comment_out_the_first_entry() {
        let src = r#"{
  "provider": {
    "ollama": {
      "models": {
        "a": { "name": "A" },
        "b": { "name": "B" }
      }
    }
  }
}"#;
        let out = comment_out_and_reparse(src, "a");
        let ast = parse_source(&out).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let models = navigate_to_models(root, "ollama").unwrap();
        assert_eq!(models.properties.len(), 1);
        assert_eq!(models.properties[0].name.as_str(), "b");
    }

    #[test]
    fn comment_out_the_only_entry_leaves_an_empty_object() {
        let src = r#"{
  "provider": {
    "ollama": {
      "models": {
        "solo": { "name": "Solo" }
      }
    }
  }
}"#;
        let out = comment_out_and_reparse(src, "solo");
        let ast = parse_source(&out).unwrap();
        let root = extract_root_object(&ast).unwrap();
        assert_eq!(
            navigate_to_models(root, "ollama").unwrap().properties.len(),
            0
        );
    }

    #[test]
    fn comment_out_preserves_a_multiline_entry_verbatim() {
        let src = r#"{
  "provider": {
    "ollama": {
      "models": {
        "big": {
          "name": "Big",
          "temperature": 0.2,
          "limit": { "context": 4096, "output": 1024 }
        },
        "other": { "name": "Other" }
      }
    }
  }
}"#;
        let out = comment_out_and_reparse(src, "big");
        // Every value the user might want back is still legible.
        for needle in [
            "\"name\": \"Big\"",
            "\"temperature\": 0.2",
            "\"context\": 4096",
        ] {
            assert!(out.contains(needle), "lost {needle} from:\n{out}");
        }
        // And nothing survives as live JSON.
        let ast = parse_source(&out).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let models = navigate_to_models(root, "ollama").unwrap();
        assert_eq!(models.properties.len(), 1);
    }

    /// Regression: the separating comma used to be located by scanning bytes
    /// outward from the property, which stops at the first comment it meets —
    /// leaving `},` followed by nothing but comments, i.e. a trailing comma.
    #[test]
    fn comment_out_last_entry_drops_the_comma_even_across_a_user_comment() {
        let src = r#"{
  "provider": {
    "ollama": {
      "models": {
        "a": { "name": "A" },
        // I pulled this one for a project last spring
        "b": { "name": "B" }
      }
    }
  }
}"#;
        let out = comment_out_and_reparse(src, "b");
        // The user's own note stays.
        assert!(out.contains("project last spring"), "{out}");
        // And no live trailing comma is left before the closing brace.
        let live: String = out
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !live.contains("},\n      }") && !live.contains("},\n"),
            "trailing comma left in live JSON:\n{live}"
        );
        // Strict JSON (no trailing commas allowed) must accept the result.
        let stripped: String = out
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        serde_json::from_str::<serde_json::Value>(&stripped)
            .unwrap_or_else(|e| panic!("result is not strict JSON: {e}\n{stripped}"));
    }

    #[test]
    fn comment_out_middle_entry_across_a_user_comment_stays_strict_json() {
        let src = r#"{
  "provider": {
    "ollama": {
      "models": {
        "a": { "name": "A" },
        "b": { "name": "B" },
        // belongs to c
        "c": { "name": "C" }
      }
    }
  }
}"#;
        let out = comment_out_and_reparse(src, "b");
        assert!(out.contains("belongs to c"), "{out}");
        let stripped: String = out
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        serde_json::from_str::<serde_json::Value>(&stripped)
            .unwrap_or_else(|e| panic!("result is not strict JSON: {e}\n{stripped}"));
    }

    #[test]
    fn comment_out_leaves_other_providers_untouched() {
        let out = comment_out_model(REAL, "ollama", "ornith:35b", NOTE).unwrap();
        assert_eq!(
            extract_provider_block(REAL, "llamacpp"),
            extract_provider_block(&out, "llamacpp"),
            "llamacpp block must not change"
        );
    }

    #[test]
    fn comment_out_keeps_surrounding_comments() {
        let out = comment_out_model(COMMENTED, "ollama", "ornith:35b", NOTE).unwrap();
        assert!(out.contains("// Opencode configuration"));
        assert!(out.contains("must remain untouched"));
        parse_source(&out).expect("still valid JSONC");
    }

    #[test]
    fn comment_out_missing_model_errors() {
        let err = comment_out_model(REAL, "ollama", "nope", NOTE).unwrap_err();
        assert!(matches!(err, EditError::ModelNotFound { .. }));
    }

    // Extract the raw source of a provider's block for comparison.
    fn extract_provider_block(source: &str, provider_id: &str) -> String {
        let ast = parse_source(source).unwrap();
        let root = extract_root_object(&ast).unwrap();
        let providers = root.get("provider").unwrap().value.as_object().unwrap();
        let block = providers.get(provider_id).unwrap();
        source[block.value.as_object().unwrap().range.start
            ..block.value.as_object().unwrap().range.end]
            .to_string()
    }
}
