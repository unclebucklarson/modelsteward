//! The headless engine. Everything here is GUI-free and testable; `src/ui`
//! (M4) renders over it and must never be a dependency of it.

pub mod discover;
pub mod gguf;
pub mod library;
