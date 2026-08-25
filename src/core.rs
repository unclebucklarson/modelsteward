//! The headless engine. Everything here is GUI-free and testable; `src/ui`
//! (M4) renders over it and must never be a dependency of it.

pub mod advisor;
pub mod bench;
pub mod diagnose;
pub mod discover;
pub mod gguf;
pub mod jsonc;
pub mod library;
pub mod ollama;
pub mod opencode;
pub mod router;
pub mod rows;
pub mod settings;
pub mod system;
pub mod trial;
