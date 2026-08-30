//! M9 phase 2 — the energy instrument: what a token costs in joules,
//! measured, not guessed (design settled 2026-08-27; built 2026-08-28).
//!
//! Sources:
//! - GPU: `nvidia-smi --query-gpu=power.draw` sampled on a thread —
//!   works unprivileged; every GPU is summed (GPU state is a Vec).
//! - CPU: Intel RAPL package energy counters — REAL joules, but
//!   root-locked on many kernels; when unreadable the CPU column is
//!   honestly None (never estimated) and the Build Advisor teaches
//!   the unlock (`sudo chmod a+r .../energy_uj`, or a udev rule).
//!
//! Marginal accounting: callers subtract an idle baseline taken
//! moments earlier — the watts the machine burns anyway are not the
//! model's bill.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// One RAPL package counter (µJ, wrapping at max_energy_range_uj).
struct Rapl {
    energy_path: PathBuf,
    max_range_uj: u64,
}

fn rapl_packages() -> Vec<Rapl> {
    let Ok(entries) = std::fs::read_dir("/sys/class/powercap") else {
        return Vec::new();
    };
    entries
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().to_str()?.to_string();
            // Top-level packages only (intel-rapl:N) — subzones
            // (intel-rapl:N:M) would double-count.
            if !name.starts_with("intel-rapl:") || name.matches(':').count() != 1 {
                return None;
            }
            let energy_path = e.path().join("energy_uj");
            // Readability check up front: root-locked counters are
            // reported as absent, not zero.
            std::fs::read_to_string(&energy_path).ok()?;
            let max_range_uj = std::fs::read_to_string(e.path().join("max_energy_range_uj"))
                .ok()?
                .trim()
                .parse()
                .ok()?;
            Some(Rapl {
                energy_path,
                max_range_uj,
            })
        })
        .collect()
}

/// Whether CPU energy is measurable without privileges — the Build
/// Advisor surfaces the unlock hint when it isn't.
pub fn rapl_readable() -> bool {
    !rapl_packages().is_empty()
}

fn rapl_read_uj(pkgs: &[Rapl]) -> Vec<u64> {
    pkgs.iter()
        .map(|p| {
            std::fs::read_to_string(&p.energy_path)
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0)
        })
        .collect()
}

/// Counter delta with wraparound (pure, tested).
pub fn counter_delta(before: u64, after: u64, max_range: u64) -> u64 {
    if after >= before {
        after - before
    } else {
        // Wrapped: distance to the top plus the new value.
        max_range.saturating_sub(before) + after
    }
}

fn gpu_power_w() -> Option<f64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=power.draw", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    out.status.success().then(|| {
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.trim().parse::<f64>().ok())
            .sum()
    })
}

/// What one measured window cost.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EnergySample {
    pub secs: f64,
    /// Integrated GPU draw over the window (all GPUs), joules.
    pub gpu_j: Option<f64>,
    /// RAPL package energy over the window, joules. None = root-locked.
    pub cpu_j: Option<f64>,
}

/// Average power over a quiet window — the baseline the marginal
/// accounting subtracts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Baseline {
    pub gpu_w: Option<f64>,
    pub cpu_w: Option<f64>,
}

pub fn idle_baseline(secs: f64) -> Baseline {
    let s = measure_window(|| std::thread::sleep(std::time::Duration::from_secs_f64(secs)))
        .sample;
    Baseline {
        gpu_w: s.gpu_j.map(|j| j / s.secs.max(0.001)),
        cpu_w: s.cpu_j.map(|j| j / s.secs.max(0.001)),
    }
}

/// Run `f`, integrating GPU power samples (~2Hz) and RAPL counters
/// around it.
pub fn measure_window<T>(f: impl FnOnce() -> T) -> EnergySampleOf<T> {
    let pkgs = rapl_packages();
    let cpu_before = rapl_read_uj(&pkgs);
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    let sampler = std::thread::spawn(move || {
        let mut joules = 0.0f64;
        let mut any = false;
        let mut last = std::time::Instant::now();
        loop {
            if let Some(w) = gpu_power_w() {
                let dt = last.elapsed().as_secs_f64();
                last = std::time::Instant::now();
                joules += w * dt;
                any = true;
            }
            if stop2.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        any.then_some(joules)
    });
    let t0 = std::time::Instant::now();
    let value = f();
    let secs = t0.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    let gpu_j = sampler.join().ok().flatten();
    let cpu_after = rapl_read_uj(&pkgs);
    let cpu_j = (!pkgs.is_empty()).then(|| {
        pkgs.iter()
            .zip(cpu_before.iter().zip(cpu_after.iter()))
            .map(|(p, (b, a))| counter_delta(*b, *a, p.max_range_uj) as f64 / 1e6)
            .sum()
    });
    EnergySampleOf {
        value,
        sample: EnergySample { secs, gpu_j, cpu_j },
    }
}

pub struct EnergySampleOf<T> {
    pub value: T,
    pub sample: EnergySample,
}

impl EnergySample {
    /// Joules attributable to the work itself: measured minus what the
    /// idle machine would have burned in the same time. Clamped at 0.
    pub fn marginal_j(&self, idle: &Baseline) -> Option<f64> {
        let gpu = match (self.gpu_j, idle.gpu_w) {
            (Some(j), Some(w)) => Some((j - w * self.secs).max(0.0)),
            _ => None,
        };
        let cpu = match (self.cpu_j, idle.cpu_w) {
            (Some(j), Some(w)) => Some((j - w * self.secs).max(0.0)),
            _ => None,
        };
        match (gpu, cpu) {
            (None, None) => None,
            (g, c) => Some(g.unwrap_or(0.0) + c.unwrap_or(0.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_delta_handles_wraparound() {
        assert_eq!(counter_delta(100, 250, 1000), 150);
        // Wrapped: 900 -> top (1000) is 100, plus 50 past zero.
        assert_eq!(counter_delta(900, 50, 1000), 150);
        assert_eq!(counter_delta(0, 0, 1000), 0);
    }

    #[test]
    fn marginal_subtracts_idle_and_never_goes_negative() {
        let s = EnergySample {
            secs: 10.0,
            gpu_j: Some(3000.0), // 300W avg
            cpu_j: Some(500.0),  // 50W avg
        };
        let idle = Baseline {
            gpu_w: Some(25.0),
            cpu_w: Some(10.0),
        };
        // (3000 - 250) + (500 - 100) = 3150
        assert_eq!(s.marginal_j(&idle), Some(3150.0));
        // Idle hotter than the window -> clamps to zero, not negative.
        let hot_idle = Baseline {
            gpu_w: Some(400.0),
            cpu_w: Some(100.0),
        };
        assert_eq!(s.marginal_j(&hot_idle), Some(0.0));
        // CPU untracked (RAPL root-locked) -> GPU-only, honestly partial.
        let gpu_only = EnergySample {
            secs: 10.0,
            gpu_j: Some(3000.0),
            cpu_j: None,
        };
        assert_eq!(gpu_only.marginal_j(&idle), Some(2750.0));
        // Nothing measurable -> None, never a guess.
        let none = EnergySample {
            secs: 10.0,
            gpu_j: None,
            cpu_j: None,
        };
        assert_eq!(none.marginal_j(&idle), None);
    }
}
