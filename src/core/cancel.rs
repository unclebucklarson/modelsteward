//! Cooperative cancellation for long-running measurement work (user
//! request 2026-08-27: a 12-minute Lab campaign had no way out short of
//! killing the app). Workers check the token at SAFE boundaries — between
//! trial rounds, bench models, eval items — so cleanup (preset restore,
//! model unload) always runs; the current HTTP call is allowed to finish.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Default, Debug)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Bail with a recognizable message at a safe boundary.
    pub fn check(&self) -> anyhow::Result<()> {
        if self.is_cancelled() {
            anyhow::bail!("cancelled by user — partial results are kept, config restored");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_flips_once_and_shares() {
        let t = CancelToken::default();
        let t2 = t.clone();
        assert!(t.check().is_ok());
        t2.cancel();
        assert!(t.is_cancelled());
        assert!(t.check().unwrap_err().to_string().contains("cancelled by user"));
    }
}
