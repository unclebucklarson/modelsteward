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

fn rapl_read_uj(pkgs: &[Rapl]) -> Vec<Option<u64>> {
    pkgs.iter()
        .map(|p| {
            // A failed read is UNKNOWN, not zero. Zero was fed to
            // counter_delta as a "before", whose wraparound branch then
            // returned max_range - before — up to 262 kJ of phantom
            // energy in the measured-cost line (review finding C9).
            std::fs::read_to_string(&p.energy_path)
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
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

/// Sum the per-GPU `power.draw` column, or `None` when nothing usable
/// came back.
///
/// `filter_map(parse).sum()` over an all-rejected iterator is `0.0`, and
/// wrapping that in `Some` on a successful exit turned a card reporting
/// `[N/A]` — normal for some models and drivers — into a *measured* zero
/// watts. Extracted so it can be tested at all, which is why it went
/// unnoticed (adversarial review, 2026-09-12).
pub fn parse_power_w(stdout: &str) -> Option<f64> {
    let mut any = false;
    let mut total = 0.0;
    for l in stdout.lines() {
        let t = l.trim();
        if t.is_empty() {
            continue;
        }
        // `[N/A]`, `[Not Supported]`, anything non-numeric: the card is
        // not telling us, which is not the same as zero.
        let w: f64 = t.parse().ok()?;
        total += w;
        any = true;
    }
    any.then_some(total)
}

fn gpu_power_w() -> Option<f64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=power.draw", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_power_w(&String::from_utf8_lossy(&out.stdout))
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
    // Every package must have BOTH a before and an after reading, or
    // the CPU figure is unknown — reporting a partial sum (or worse, a
    // phantom wraparound from a zeroed "before") would inject invented
    // joules into the measured-cost line the product is sold on.
    let cpu_j: Option<f64> = if pkgs.is_empty() {
        None
    } else {
        pkgs.iter()
            .zip(cpu_before.iter().zip(cpu_after.iter()))
            .map(|(p, (b, a))| match (b, a) {
                (Some(b), Some(a)) => Some(counter_delta(*b, *a, p.max_range_uj) as f64 / 1e6),
                _ => None,
            })
            .sum::<Option<f64>>()
    };
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
        // BOTH halves or nothing. `g.unwrap_or(0.0) + c.unwrap_or(0.0)`
        // returned a partial sum dressed as a total — precisely what the
        // comment above the cpu_j computation says must not happen
        // ("reporting a partial sum … would inject invented joules into
        // the measured-cost line the product is sold on").
        //
        // On a stock machine RAPL is root-locked, so the CPU half is
        // always unknown, and the crowned `--cpu-moe` configs run their
        // experts there: the J/tok column was ranking CPU-offload
        // variants against GPU-resident ones with the offloaded half
        // priced at zero, and the meter's "measured local cost" line
        // inherited it (adversarial review, 2026-09-12).
        //
        // Absent is the honest answer, and the Build Advisor already
        // tells the user how to make RAPL readable.
        match (gpu, cpu) {
            (Some(g), Some(c)) => Some(g + c),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests_partial_energy {
    use super::{Baseline, EnergySample, parse_power_w};

    /// The defect, and it bites this machine: `/sys/class/powercap/
    /// intel-rapl:0/energy_uj` is root-locked here, so cpu_j is always
    /// None — and `g.unwrap_or(0.0) + c.unwrap_or(0.0)` returned the GPU
    /// figure alone as the TOTAL. The crowned ncpu-moe-32 config runs 32
    /// layers of experts on the CPU, so the Lab's J/tok column has been
    /// ranking CPU-offload variants against GPU-resident ones with the
    /// offloaded half priced at zero (review 2026-09-12).
    #[test]
    fn a_missing_cpu_half_is_unknown_not_a_total() {
        let s = EnergySample { secs: 10.0, gpu_j: Some(1000.0), cpu_j: None };
        let idle = Baseline { gpu_w: Some(20.0), cpu_w: Some(5.0) };
        assert_eq!(
            s.marginal_j(&idle),
            None,
            "GPU-only is not the total energy; the module's contract is \
             measured or absent, no estimates"
        );
    }

    #[test]
    fn a_missing_gpu_half_is_also_unknown() {
        let s = EnergySample { secs: 10.0, gpu_j: None, cpu_j: Some(200.0) };
        let idle = Baseline { gpu_w: Some(20.0), cpu_w: Some(5.0) };
        assert_eq!(s.marginal_j(&idle), None);
    }

    /// Both halves present: a real total, idle draw subtracted.
    #[test]
    fn both_halves_present_gives_the_marginal_total() {
        let s = EnergySample { secs: 10.0, gpu_j: Some(1000.0), cpu_j: Some(200.0) };
        let idle = Baseline { gpu_w: Some(20.0), cpu_w: Some(5.0) };
        // (1000 - 200) + (200 - 50) = 950
        assert_eq!(s.marginal_j(&idle), Some(950.0));
    }

    /// And the same rule on the GPU side of the input: nvidia-smi
    /// reporting no usable value is unknown, not zero watts.
    #[test]
    fn nvidia_smi_reporting_no_value_is_unknown() {
        assert_eq!(parse_power_w("[N/A]\n"), None);
        assert_eq!(parse_power_w("[Not Supported]"), None);
        assert_eq!(parse_power_w(""), None);
        assert_eq!(parse_power_w("   \n  \n"), None);
    }

    #[test]
    fn real_readings_still_sum_across_cards() {
        assert_eq!(parse_power_w("350.5\n"), Some(350.5));
        assert_eq!(parse_power_w("100.0\n200.0\n"), Some(300.0));
    }
}

mod tests {
    use super::*;

    #[test]
    fn a_failed_energy_read_is_unknown_not_a_phantom_wraparound() {
        // Review finding C9 (2026-08-31): a failed RAPL read became 0,
        // which counter_delta then read as a counter wrap and returned
        // max_range - 0 — on this machine 262 kJ of invented energy for
        // one trial window, flowing straight into the measured $/token
        // line with no plausibility clamp.
        let max = 262_143_328_850u64;
        // The old behavior, shown for what it was:
        assert_eq!(counter_delta(0, 0, max), 0);
        assert_eq!(
            counter_delta(500, 0, max),
            max - 500,
            "a zeroed 'after' really does look like a full wrap"
        );
        // The contract now: any missing reading makes the whole CPU
        // figure None. Summing Options is what enforces it.
        let partial: Option<f64> = [Some(1.0), None, Some(2.0)].into_iter().sum();
        assert_eq!(partial, None, "one missing package poisons the sum, by design");
        let complete: Option<f64> = [Some(1.0), Some(2.0)].into_iter().sum();
        assert_eq!(complete, Some(3.0));
    }

    #[test]
    fn an_unreadable_counter_file_reads_as_none_not_zero() {
        // The load-bearing half the audit found untested: rapl_read_uj
        // itself must yield None for a missing/unreadable path.
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("energy_uj");
        std::fs::write(&good, "123456\n").unwrap();
        let pkgs = vec![
            Rapl { energy_path: good, max_range_uj: 1_000_000 },
            Rapl { energy_path: dir.path().join("missing"), max_range_uj: 1_000_000 },
        ];
        assert_eq!(rapl_read_uj(&pkgs), vec![Some(123_456), None]);
    }

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
        // REVERSED 2026-09-12, with the reason recorded rather than the
        // assertion quietly edited. This previously expected
        // Some(2750.0) — the GPU figure alone — and called it "honestly
        // partial". Partial is only honest if the partiality TRAVELS,
        // and it does not: `TrialResult.j_per_token` is a bare
        // Option<f64> and `meter::cost_report` takes a plain f64, so the
        // number reaches the trial table (which RANKS configs by it) and
        // the "measured local cost" line with nothing saying the CPU was
        // excluded. Treating an unreadable CPU as zero joules is an
        // estimate, and a badly wrong one for the `--cpu-moe` configs
        // that put their experts there — which is exactly what this
        // machine's crowned config does, with RAPL root-locked.
        //
        // The module's own contract is "honestly None (never
        // estimated)". So: both halves, or no number.
        let gpu_only = EnergySample {
            secs: 10.0,
            gpu_j: Some(3000.0),
            cpu_j: None,
        };
        assert_eq!(gpu_only.marginal_j(&idle), None);
        // Nothing measurable -> None, never a guess.
        let none = EnergySample {
            secs: 10.0,
            gpu_j: None,
            cpu_j: None,
        };
        assert_eq!(none.marginal_j(&idle), None);
    }
}
