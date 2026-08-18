//! Minimal GGUF header reader — metadata only, never the tensors.
//!
//! A GGUF file opens with a fixed header and a run of key/value metadata
//! pairs; the (multi-gigabyte) tensor data comes after. Everything the model
//! library needs — architecture, context length, quantization — lives in
//! those KVs, so this reads a few megabytes at most and works identically on
//! a 20 GB shelf file or an Ollama blob.
//!
//! Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md (v2/v3).
//!
//! Harvested from llm_forge (src/gguf.rs) per PLAN.md.

use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::io::{BufReader, Read};
use std::path::Path;

/// What the library shows per model file.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GgufMeta {
    /// `general.architecture` — "llama", "qwen3", …
    pub architecture: Option<String>,
    /// `general.name` — the model's self-declared name.
    pub name: Option<String>,
    /// `<arch>.context_length` — the training context window.
    pub context_length: Option<u64>,
    /// Human name for `general.file_type` (Q4_K_M, Q5_K_XL, …).
    pub quantization: Option<String>,
    /// `general.size_label` when present ("27B", "8x7B", …).
    pub size_label: Option<String>,
    /// Multi-token-prediction layers present (tensor names containing
    /// ".nextn."). Capability shipped in the file — whether the installed
    /// llama.cpp exploits it is a separate (build-version) question.
    #[serde(default)]
    pub has_mtp: bool,
}

/// Hard ceiling on how much header we're willing to read. Real metadata
/// (including 150k-entry tokenizer arrays) fits comfortably; a corrupt
/// length field must not make us slurp the tensor blob.
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

pub fn read_meta(path: &Path) -> Result<GgufMeta> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = CountingReader {
        inner: BufReader::new(file),
        read: 0,
    };

    let magic = read_u32(&mut r)?;
    if magic != 0x4655_4747 {
        bail!("not a GGUF file (bad magic)");
    }
    let version = read_u32(&mut r)?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version {version}");
    }
    let _tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;
    if kv_count > 100_000 {
        bail!("implausible metadata count {kv_count}");
    }

    let mut meta = GgufMeta::default();
    let mut arch_ctx: Option<u64> = None;
    let mut generic_ctx: Option<u64> = None;

    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let ty = read_u32(&mut r)?;
        match key.as_str() {
            "general.architecture" => meta.architecture = as_string(&mut r, ty)?,
            "general.name" => meta.name = as_string(&mut r, ty)?,
            "general.size_label" => meta.size_label = as_string(&mut r, ty)?,
            "general.file_type" => {
                if let Some(n) = as_u64(&mut r, ty)? {
                    meta.quantization = Some(file_type_name(n));
                }
            }
            k if k.ends_with(".context_length") => {
                // Prefer `<declared-arch>.context_length` but accept any
                // arch prefix — the declared arch KV may come later.
                let v = as_u64(&mut r, ty)?;
                if let (Some(arch), Some(v)) = (&meta.architecture, v)
                    && k == format!("{arch}.context_length")
                {
                    arch_ctx = Some(v);
                } else {
                    generic_ctx = generic_ctx.or(v);
                }
            }
            _ => skip_value(&mut r, ty)?,
        }
        if r.read > MAX_HEADER_BYTES {
            bail!("metadata exceeded {MAX_HEADER_BYTES} bytes; refusing to read further");
        }
    }
    meta.context_length = arch_ctx.or(generic_ctx);

    // Tensor infos follow the KVs: name, dims, type, offset. We only want
    // the names — feature tensors announce themselves there (MTP layers
    // are blk.N.nextn.*). Bounded by the same header ceiling.
    for _ in 0.._tensor_count {
        let name = read_string(&mut r)?;
        if name.contains(".nextn.") {
            meta.has_mtp = true;
        }
        let n_dims = read_u32(&mut r)?;
        if n_dims > 8 {
            bail!("implausible tensor rank {n_dims}");
        }
        for _ in 0..n_dims {
            let _ = read_u64(&mut r)?;
        }
        let _ty = read_u32(&mut r)?;
        let _offset = read_u64(&mut r)?;
        if r.read > MAX_HEADER_BYTES {
            bail!("tensor table exceeded {MAX_HEADER_BYTES} bytes; refusing to read further");
        }
    }
    Ok(meta)
}

struct CountingReader<R> {
    inner: R,
    read: u64,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.read += n as u64;
        Ok(n)
    }
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<()> {
    r.read_exact(buf).context("truncated GGUF header")
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    read_exact(r, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    read_exact(r, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)?;
    if len > 64 * 1024 * 1024 {
        bail!("implausible string length {len}");
    }
    let mut buf = vec![0u8; len as usize];
    read_exact(r, &mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// GGUF metadata value types.
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

fn fixed_size(ty: u32) -> Option<u64> {
    match ty {
        T_U8 | T_I8 | T_BOOL => Some(1),
        T_U16 | T_I16 => Some(2),
        T_U32 | T_I32 | T_F32 => Some(4),
        T_U64 | T_I64 | T_F64 => Some(8),
        _ => None,
    }
}

fn skip_bytes<R: Read>(r: &mut R, mut n: u64) -> Result<()> {
    let mut buf = [0u8; 8192];
    while n > 0 {
        let take = n.min(buf.len() as u64) as usize;
        read_exact(r, &mut buf[..take])?;
        n -= take as u64;
    }
    Ok(())
}

fn skip_value<R: Read>(r: &mut R, ty: u32) -> Result<()> {
    if let Some(sz) = fixed_size(ty) {
        return skip_bytes(r, sz);
    }
    match ty {
        T_STRING => {
            let len = read_u64(r)?;
            skip_bytes(r, len)
        }
        T_ARRAY => {
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)?;
            if let Some(sz) = fixed_size(elem_ty) {
                skip_bytes(r, count.saturating_mul(sz))
            } else if elem_ty == T_STRING {
                for _ in 0..count {
                    let len = read_u64(r)?;
                    skip_bytes(r, len)?;
                }
                Ok(())
            } else {
                // Nested arrays don't occur in real files; refuse rather
                // than mis-skip and misread everything after.
                bail!("unsupported array element type {elem_ty}")
            }
        }
        other => bail!("unknown GGUF value type {other}"),
    }
}

fn as_string<R: Read>(r: &mut R, ty: u32) -> Result<Option<String>> {
    if ty == T_STRING {
        Ok(Some(read_string(r)?))
    } else {
        skip_value(r, ty)?;
        Ok(None)
    }
}

fn as_u64<R: Read>(r: &mut R, ty: u32) -> Result<Option<u64>> {
    Ok(match ty {
        T_U32 => Some(read_u32(r)? as u64),
        T_U64 => Some(read_u64(r)?),
        T_I32 => Some(read_u32(r)? as i32 as u64),
        T_I64 => Some(read_u64(r)? as i64 as u64),
        _ => {
            skip_value(r, ty)?;
            None
        }
    })
}

/// `general.file_type` enum → the quant name users know. Partial by design:
/// unknown values render as `type N` rather than guessing.
fn file_type_name(n: u64) -> String {
    match n {
        0 => "F32".into(),
        1 => "F16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        7 => "Q8_0".into(),
        8 => "Q5_0".into(),
        9 => "Q5_1".into(),
        10 => "Q2_K".into(),
        11 => "Q3_K_S".into(),
        12 => "Q3_K_M".into(),
        13 => "Q3_K_L".into(),
        14 => "Q4_K_S".into(),
        15 => "Q4_K_M".into(),
        16 => "Q5_K_S".into(),
        17 => "Q5_K_M".into(),
        18 => "Q6_K".into(),
        19 => "IQ2_XXS".into(),
        20 => "IQ2_XS".into(),
        24 => "IQ4_NL".into(),
        25 => "IQ4_XS".into(),
        30 => "BF16".into(),
        other => format!("type {other}"),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// Build a tiny synthetic GGUF header for tests.
    pub(crate) fn synthetic_gguf(arch: &str, ctx: u64, file_type: u64) -> Vec<u8> {
        synthetic_gguf_with_tensors(arch, ctx, file_type, &[])
    }

    pub(crate) fn synthetic_gguf_with_tensors(
        arch: &str,
        ctx: u64,
        file_type: u64,
        tensor_names: &[&str],
    ) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend(0x4655_4747u32.to_le_bytes()); // "GGUF"
        b.extend(3u32.to_le_bytes()); // version
        b.extend((tensor_names.len() as u64).to_le_bytes()); // tensor count
        b.extend(5u64.to_le_bytes()); // kv count

        let kv_str = |b: &mut Vec<u8>, k: &str, v: &str| {
            b.extend((k.len() as u64).to_le_bytes());
            b.extend(k.as_bytes());
            b.extend(T_STRING.to_le_bytes());
            b.extend((v.len() as u64).to_le_bytes());
            b.extend(v.as_bytes());
        };
        let kv_u32 = |b: &mut Vec<u8>, k: &str, v: u32| {
            b.extend((k.len() as u64).to_le_bytes());
            b.extend(k.as_bytes());
            b.extend(T_U32.to_le_bytes());
            b.extend(v.to_le_bytes());
        };

        kv_str(&mut b, "general.architecture", arch);
        kv_str(&mut b, "general.name", "Synthetic Test Model");
        // A string array to prove the skipper walks composite values.
        let key = "tokenizer.ggml.tokens";
        b.extend((key.len() as u64).to_le_bytes());
        b.extend(key.as_bytes());
        b.extend(T_ARRAY.to_le_bytes());
        b.extend(T_STRING.to_le_bytes());
        b.extend(3u64.to_le_bytes());
        for tok in ["<s>", "hello", "world"] {
            b.extend((tok.len() as u64).to_le_bytes());
            b.extend(tok.as_bytes());
        }
        kv_u32(&mut b, &format!("{arch}.context_length"), ctx as u32);
        kv_u32(&mut b, "general.file_type", file_type as u32);
        // Tensor infos: name, n_dims=1, dim, type, offset.
        for name in tensor_names {
            b.extend((name.len() as u64).to_le_bytes());
            b.extend(name.as_bytes());
            b.extend(1u32.to_le_bytes());
            b.extend(64u64.to_le_bytes());
            b.extend(0u32.to_le_bytes());
            b.extend(0u64.to_le_bytes());
        }
        b
    }

    fn write_temp(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
        (dir, path)
    }

    #[test]
    fn reads_the_fields_the_library_needs() {
        let (_d, path) = write_temp(&synthetic_gguf("qwen3", 262_144, 17));
        let meta = read_meta(&path).unwrap();
        assert_eq!(meta.architecture.as_deref(), Some("qwen3"));
        assert_eq!(meta.name.as_deref(), Some("Synthetic Test Model"));
        assert_eq!(meta.context_length, Some(262_144));
        assert_eq!(meta.quantization.as_deref(), Some("Q5_K_M"));
    }

    #[test]
    fn detects_mtp_tensors_by_name() {
        let (_d, path) = write_temp(&synthetic_gguf_with_tensors(
            "qwen3",
            8192,
            17,
            &["blk.0.attn_q.weight", "blk.64.nextn.eh_proj.weight"],
        ));
        let meta = read_meta(&path).unwrap();
        assert!(meta.has_mtp);

        let (_d2, path2) = write_temp(&synthetic_gguf_with_tensors(
            "qwen3",
            8192,
            17,
            &["blk.0.attn_q.weight"],
        ));
        assert!(!read_meta(&path2).unwrap().has_mtp);
    }

    #[test]
    fn rejects_non_gguf() {
        let (_d, path) = write_temp(b"definitely not a gguf file");
        let err = read_meta(&path).unwrap_err();
        assert!(format!("{err}").contains("bad magic"));
    }

    #[test]
    fn truncated_header_errors_cleanly() {
        let full = synthetic_gguf("llama", 8192, 15);
        let (_d, path) = write_temp(&full[..full.len() / 2]);
        assert!(read_meta(&path).is_err());
    }

    #[test]
    fn unknown_file_type_is_reported_not_guessed() {
        let (_d, path) = write_temp(&synthetic_gguf("llama", 4096, 999));
        let meta = read_meta(&path).unwrap();
        assert_eq!(meta.quantization.as_deref(), Some("type 999"));
    }
}
