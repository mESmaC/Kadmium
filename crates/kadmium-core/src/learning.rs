//! Numeric learning primitives the planners and the worker share, after InferSSD's placement
//! policy (`engine/ggml/src/ggml-ssd/place/`): a bounded exponential average, a least-squares
//! line pulled toward a prior, the two-station tandem-queue time of a producer and a consumer
//! handing work through a ring, the expected expert coverage of a batch under top-k routing,
//! and a probe-then-decide chooser between two candidates.
//!
//! These are plain numbers. Nothing here holds a socket, a file or a worker handle, so the
//! same arithmetic prices a hop on the coordinator, a read on a worker and a plan in a test.

use serde::{Deserialize, Serialize};

/// Weight of a new sample in every exponential average here: the figure settles over about ten
/// samples, which is what InferSSD found keeps a learned cost line steady across a decode
/// token's layers while still following a drive or a link that warms up.
pub const EMA_ALPHA: f64 = 0.1;

/// A bounded exponential moving average seeded by its first sample.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Ewma {
    pub value: f64,
    pub samples: u64,
}

impl Ewma {
    /// Folds a sample in; the first sample becomes the value, later ones move it by
    /// [`EMA_ALPHA`]. Non-finite samples are ignored.
    pub fn observe(&mut self, sample: f64) {
        if !sample.is_finite() {
            return;
        }
        self.value = if self.samples == 0 {
            sample
        } else {
            (1.0 - EMA_ALPHA) * self.value + EMA_ALPHA * sample
        };
        self.samples = self.samples.saturating_add(1);
    }

    /// The average, once a sample has been seen.
    pub fn get(&self) -> Option<f64> {
        (self.samples > 0).then_some(self.value)
    }
}

/// Sums for a least-squares line `t = fixed + slope * x`: samples, `x`, `x²`, `t`, `x·t`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct LineFit {
    pub n: f64,
    pub sum_x: f64,
    pub sum_xx: f64,
    pub sum_t: f64,
    pub sum_xt: f64,
}

impl LineFit {
    pub fn add(&mut self, x: f64, t: f64) {
        if !x.is_finite() || !t.is_finite() {
            return;
        }
        self.n += 1.0;
        self.sum_x += x;
        self.sum_xx += x * x;
        self.sum_t += t;
        self.sum_xt += x * t;
    }

    pub fn merge(&mut self, other: &LineFit) {
        self.n += other.n;
        self.sum_x += other.sum_x;
        self.sum_xx += other.sum_xx;
        self.sum_t += other.sum_t;
        self.sum_xt += other.sum_xt;
    }

    pub fn is_empty(&self) -> bool {
        self.n < 1.0
    }

    pub fn mean_x(&self) -> f64 {
        if self.n > 0.0 {
            self.sum_x / self.n
        } else {
            0.0
        }
    }

    pub fn mean_t(&self) -> f64 {
        if self.n > 0.0 {
            self.sum_t / self.n
        } else {
            0.0
        }
    }

    /// The least-squares line through the samples, pulled by one pseudo-sample at their mean
    /// `x` toward the prior line scaled to the samples' mean (InferSSD's
    /// `kimi_prefill_fit_line`): a window of one `x` value stays solvable and keeps the prior's
    /// shape at this machine's level, and against hundreds of samples the pseudo-sample does not
    /// move the fit. Both terms stay `>= 0`. Without samples the prior is returned as it is.
    pub fn fit_with_prior(&self, prior_fixed: f64, prior_slope: f64) -> (f64, f64) {
        if self.is_empty() {
            return (prior_fixed.max(0.0), prior_slope.max(0.0));
        }
        let mean = self.mean_x();
        let (mut prior_fixed, mut prior_slope) = (prior_fixed.max(0.0), prior_slope.max(0.0));
        let prior_mean = prior_fixed + prior_slope * mean;
        if prior_mean > 0.0 && self.sum_t > 0.0 {
            let scale = self.sum_t / self.n / prior_mean;
            prior_fixed *= scale;
            prior_slope *= scale;
        }
        let a = self.n + 1.0;
        let b = self.sum_x;
        let d = self.sum_xx + mean * mean;
        let y0 = self.sum_t + prior_fixed;
        let y1 = self.sum_xt + mean * mean * prior_slope;
        let det = a * d - b * b;
        let (mut f, mut r) = if det > 0.0 {
            ((y0 * d - b * y1) / det, (a * y1 - b * y0) / det)
        } else {
            (-1.0, -1.0)
        };
        if f < 0.0 {
            f = 0.0;
            r = if d > 0.0 { y1 / d } else { 0.0 };
        } else if r < 0.0 {
            r = 0.0;
            f = y0 / a;
        }
        (f.max(0.0), r.max(0.0))
    }

    /// Ordinary nonnegative least squares with no prior, `None` until two distinct `x` values
    /// have been seen.
    pub fn fit(&self) -> Option<(f64, f64)> {
        if self.n < 2.0 {
            return None;
        }
        let det = self.n * self.sum_xx - self.sum_x * self.sum_x;
        if det <= 1e-12 * self.sum_xx.max(1.0) {
            return None;
        }
        let mut slope = (self.n * self.sum_xt - self.sum_x * self.sum_t) / det;
        let mut fixed = (self.sum_t - slope * self.sum_x) / self.n;
        if slope < 0.0 {
            slope = 0.0;
            fixed = self.mean_t();
        } else if fixed < 0.0 {
            fixed = 0.0;
            slope = if self.sum_xx > 0.0 {
                self.sum_xt / self.sum_xx
            } else {
                0.0
            };
        }
        Some((fixed.max(0.0), slope.max(0.0)))
    }
}

/// A learned line `t = fixed + slope * x` over a bounded window of recent samples, refitted
/// against a prior every time a sample lands: the shape InferSSD learns for its CPU, GPU and
/// drive lines, with the 64-sample window it keeps for the drive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowedLine {
    pub prior_fixed: f64,
    pub prior_slope: f64,
    pub window: usize,
    pub samples: Vec<(f64, f64)>,
    pub next: usize,
    pub fixed: f64,
    pub slope: f64,
    pub observed: u64,
}

/// InferSSD keeps the last 64 `(MB, µs)` drive reads of a decode.
pub const DEFAULT_LINE_WINDOW: usize = 64;

impl WindowedLine {
    pub fn new(prior_fixed: f64, prior_slope: f64, window: usize) -> Self {
        Self {
            prior_fixed: prior_fixed.max(0.0),
            prior_slope: prior_slope.max(0.0),
            window: window.max(1),
            samples: Vec::new(),
            next: 0,
            fixed: prior_fixed.max(0.0),
            slope: prior_slope.max(0.0),
            observed: 0,
        }
    }

    pub fn observe(&mut self, x: f64, t: f64) {
        if !x.is_finite() || !t.is_finite() || x < 0.0 || t < 0.0 {
            return;
        }
        // A line restored from a store may carry a zero window or a short sample ring; keep
        // the ring consistent rather than index past it.
        self.window = self.window.max(1);
        if self.samples.len() < self.window {
            self.samples.push((x, t));
            self.next = self.samples.len() % self.window;
        } else {
            self.samples.truncate(self.window);
            if self.next >= self.samples.len() {
                self.next = 0;
            }
            self.samples[self.next] = (x, t);
            self.next = (self.next + 1) % self.window;
        }
        self.observed = self.observed.saturating_add(1);
        let mut fit = LineFit::default();
        for (x, t) in &self.samples {
            fit.add(*x, *t);
        }
        let (fixed, slope) = fit.fit_with_prior(self.prior_fixed, self.prior_slope);
        self.fixed = fixed;
        self.slope = slope;
    }

    pub fn predict(&self, x: f64) -> f64 {
        self.fixed + self.slope * x.max(0.0)
    }

    pub fn is_learned(&self) -> bool {
        self.observed > 0
    }
}

/// The time a producer busy for `a` and a consumer busy for `b` take together when they hand
/// the work over through a ring of `n` buffers: the throughput of a two-station tandem queue
/// with `n` places between the stations, as a time (InferSSD's `kimi_prefill_coupled`). It is
/// `max(a, b)` when one is far shorter, rises smoothly as they approach, and is
/// `max * (n + 1) / n` when they are equal: each station then often waits for the other, a
/// full ring stopping the producer and an empty one the consumer.
pub fn coupled_pipeline_time(a: f64, b: f64, n: f64) -> f64 {
    let high = a.max(b);
    let low = a.min(b).max(0.0);
    if high.is_nan() || high <= 0.0 {
        return 0.0;
    }
    let n = n.max(1.0);
    let r = low / high;
    if r > 1.0 - 1e-9 {
        return high * (n + 1.0) / n;
    }
    high * (1.0 - r.powf(n + 1.0)) / (1.0 - r.powf(n))
}

/// The probability that an expert selected with probability `p_select` by one row is selected
/// by at least one of `rows` rows: `1 - (1 - p)^rows`. Under uniform top-k routing `p_select =
/// k / E`, which gives InferSSD's `touched = E * (1 - (1 - k/E)^n)` distinct experts per
/// layer. The exact expectation replaces a fitted growth exponent.
pub fn batch_coverage(p_select: f64, rows: f64) -> f64 {
    let p = p_select.clamp(0.0, 1.0);
    let rows = rows.max(0.0);
    if rows == 0.0 || p == 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    1.0 - (1.0 - p).powf(rows)
}

/// Distinct experts a layer of `experts` experts with `top_k` routing is expected to touch for
/// `rows` rows under uniform routing.
pub fn uniform_routing_touched(experts: u32, top_k: u32, rows: u32) -> f64 {
    if experts == 0 {
        return 0.0;
    }
    let p = (top_k as f64 / experts as f64).min(1.0);
    experts as f64 * batch_coverage(p, rows as f64)
}

/// InferSSD probes 400 experts per candidate before deciding.
pub const PROBE_SAMPLES: u64 = 400;
/// ... and re-probes after 40 000 experts, in case the machine's state has changed.
pub const PROBE_RECHECK_SAMPLES: u64 = 40_000;

/// A probe-then-decide chooser between two candidate settings (two thread counts, two
/// concurrency limits): it measures `probe_samples` costs under candidate 0, then as many under
/// candidate 1, keeps the cheaper, and re-opens the question after `recheck_samples` more. Equal
/// candidates or a lock settle it at once (InferSSD's `thread_learning`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateProbe {
    pub candidates: [u32; 2],
    pub current: usize,
    pub decided: Option<usize>,
    pub locked: bool,
    pub cost: [f64; 2],
    pub samples: [u64; 2],
    pub since_decision: u64,
    pub probe_samples: u64,
    pub recheck_samples: u64,
    pub decisions: u64,
}

/// What one observation changed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeDecision {
    pub changed: bool,
    pub mean0: f64,
    pub mean1: f64,
}

impl CandidateProbe {
    pub fn new(candidate0: u32, candidate1: u32) -> Self {
        Self {
            candidates: [candidate0, candidate1],
            current: 0,
            decided: None,
            locked: candidate0 == candidate1,
            cost: [0.0; 2],
            samples: [0; 2],
            since_decision: 0,
            probe_samples: PROBE_SAMPLES,
            recheck_samples: PROBE_RECHECK_SAMPLES,
            decisions: 0,
        }
    }

    pub fn with_probe(mut self, probe_samples: u64, recheck_samples: u64) -> Self {
        self.probe_samples = probe_samples.max(1);
        self.recheck_samples = recheck_samples.max(1);
        self
    }

    /// Pins the choice to a candidate, as an explicit setting does.
    pub fn lock(mut self, index: usize) -> Self {
        self.decided = Some(index.min(1));
        self.current = index.min(1);
        self.locked = true;
        self
    }

    /// The candidate in force: the decided one, or the one being probed.
    pub fn active(&self) -> u32 {
        self.candidates[self.decided.unwrap_or(self.current)]
    }

    /// Records the cost of one sample under the active candidate.
    pub fn observe(&mut self, cost: f64) -> ProbeDecision {
        let unchanged = ProbeDecision {
            changed: false,
            mean0: 0.0,
            mean1: 0.0,
        };
        if self.locked || !cost.is_finite() {
            return unchanged;
        }
        if self.decided.is_some() {
            self.since_decision += 1;
            if self.candidates[0] == self.candidates[1]
                || self.since_decision < self.recheck_samples
            {
                return unchanged;
            }
            self.decided = None;
            self.current = 0;
            self.cost = [0.0; 2];
            self.samples = [0; 2];
            return unchanged;
        }
        self.cost[self.current] += cost;
        self.samples[self.current] += 1;
        if self.samples[self.current] < self.probe_samples {
            return unchanged;
        }
        if self.current == 0 {
            self.current = 1;
            return unchanged;
        }
        let mean0 = self.cost[0] / self.samples[0].max(1) as f64;
        let mean1 = self.cost[1] / self.samples[1].max(1) as f64;
        self.decided = Some(if mean1 < mean0 { 1 } else { 0 });
        self.current = self.decided.unwrap_or(0);
        self.since_decision = 0;
        self.decisions += 1;
        ProbeDecision {
            changed: true,
            mean0,
            mean1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_seeds_on_first_sample_then_moves_a_tenth() {
        let mut e = Ewma::default();
        assert_eq!(e.get(), None);
        e.observe(100.0);
        assert_eq!(e.get(), Some(100.0));
        e.observe(200.0);
        assert!((e.value - 110.0).abs() < 1e-9);
        e.observe(f64::NAN);
        assert_eq!(e.samples, 2);
    }

    #[test]
    fn line_fit_with_prior_keeps_shape_on_one_x_and_follows_many_samples() {
        // One x value: the prior's shape, scaled to the sample level.
        let mut one = LineFit::default();
        one.add(10.0, 300.0);
        let (fixed, slope) = one.fit_with_prior(100.0, 10.0);
        // The prior at x=10 is 200, scaled by 1.5: fixed 150, slope 15, and the single
        // sample sits on that line.
        assert!(
            (fixed + slope * 10.0 - 300.0).abs() < 1e-6,
            "{fixed} {slope}"
        );
        assert!((fixed / slope - 10.0).abs() < 1e-6);

        // Many samples on t = 5 + 2x overrule a wrong prior: the pseudo-sample's pull is a
        // fraction of a percent of the fit at the samples' mean.
        let mut many = LineFit::default();
        for x in 0..200 {
            many.add(x as f64, 5.0 + 2.0 * x as f64);
        }
        let (fixed, slope) = many.fit_with_prior(1000.0, 0.0);
        assert!((slope - 2.0).abs() < 0.1, "{slope}");
        let at_mean = fixed + slope * many.mean_x();
        assert!((at_mean - 204.0).abs() < 2.0, "{fixed} {slope}");
        // With the right prior shape the fit is exact.
        let (fixed, slope) = many.fit_with_prior(5.0, 2.0);
        assert!((fixed - 5.0).abs() < 1e-6 && (slope - 2.0).abs() < 1e-9);

        // No samples: the prior as it is.
        assert_eq!(LineFit::default().fit_with_prior(7.0, 3.0), (7.0, 3.0));

        // Negative terms clamp.
        let mut falling = LineFit::default();
        falling.add(1.0, 10.0);
        falling.add(2.0, 1.0);
        let (fixed, slope) = falling.fit_with_prior(0.0, 1.0);
        assert!(fixed >= 0.0 && slope >= 0.0);
        let (fixed, slope) = falling.fit().unwrap();
        assert_eq!(slope, 0.0);
        assert!((fixed - 5.5).abs() < 1e-9);
        assert!(LineFit::default().fit().is_none());
    }

    #[test]
    fn windowed_line_refits_over_its_window() {
        let mut line = WindowedLine::new(1.0, 1.0, 4);
        assert!(!line.is_learned());
        for x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
            line.observe(x, 10.0 + 3.0 * x);
        }
        assert_eq!(line.samples.len(), 4);
        // In-sample the line is close; past the window the small window still feels the
        // prior's pull, which is the point of the pseudo-sample.
        assert!(
            (line.predict(6.0) - 28.0).abs() < 2.0,
            "{}",
            line.predict(6.0)
        );
        let ahead = line.predict(10.0);
        assert!((38.0..52.0).contains(&ahead), "{ahead}");
        assert!(line.slope > 2.5 && line.slope < 4.5, "{}", line.slope);
        assert!(line.is_learned());
    }

    #[test]
    fn a_restored_line_with_a_bad_window_still_observes() {
        let mut line = WindowedLine::new(1.0, 1.0, 4);
        line.window = 0;
        line.samples.clear();
        line.next = 9;
        line.observe(1.0, 2.0);
        assert_eq!(line.window, 1);
        assert_eq!(line.samples.len(), 1);
        let mut short = WindowedLine::new(1.0, 1.0, 4);
        for x in 0..4 {
            short.observe(x as f64, 1.0);
        }
        short.samples.truncate(2);
        short.next = 3;
        short.observe(5.0, 6.0);
        assert!(short.samples.len() <= 4);
    }

    #[test]
    fn coupled_time_is_the_longer_station_apart_and_a_ring_fraction_more_together() {
        assert_eq!(coupled_pipeline_time(0.0, 0.0, 8.0), 0.0);
        assert!((coupled_pipeline_time(100.0, 1.0, 8.0) - 100.0).abs() < 1e-6);
        assert!((coupled_pipeline_time(100.0, 100.0, 4.0) - 125.0).abs() < 1e-9);
        let between = coupled_pipeline_time(100.0, 80.0, 4.0);
        assert!(between > 100.0 && between < 125.0, "{between}");
        assert!(coupled_pipeline_time(100.0, 80.0, 16.0) < between);
    }

    #[test]
    fn coverage_is_exact_under_uniform_routing() {
        assert_eq!(batch_coverage(0.25, 1.0), 0.25);
        assert!((batch_coverage(0.25, 2.0) - 0.4375).abs() < 1e-12);
        assert_eq!(batch_coverage(1.5, 3.0), 1.0);
        assert_eq!(batch_coverage(0.3, 0.0), 0.0);
        // 8 experts, top-2, 1 row: 2 touched; 64 rows: nearly all.
        assert!((uniform_routing_touched(8, 2, 1) - 2.0).abs() < 1e-12);
        assert!(uniform_routing_touched(8, 2, 64) > 7.99);
        assert_eq!(uniform_routing_touched(0, 2, 4), 0.0);
    }

    #[test]
    fn probe_measures_both_candidates_then_keeps_the_cheaper_and_rechecks() {
        let mut probe = CandidateProbe::new(8, 4).with_probe(3, 6);
        assert_eq!(probe.active(), 8);
        for _ in 0..3 {
            assert!(!probe.observe(10.0).changed);
        }
        assert_eq!(probe.active(), 4);
        probe.observe(5.0);
        probe.observe(5.0);
        let decision = probe.observe(5.0);
        assert!(decision.changed);
        assert_eq!((decision.mean0, decision.mean1), (10.0, 5.0));
        assert_eq!(probe.active(), 4);
        assert_eq!(probe.decisions, 1);
        // Decided: samples count toward the recheck, then the probe reopens on candidate 0.
        for _ in 0..6 {
            assert!(!probe.observe(5.0).changed);
        }
        assert_eq!(probe.decided, None);
        assert_eq!(probe.active(), 8);

        let locked = CandidateProbe::new(8, 4).lock(1);
        assert_eq!(locked.active(), 4);
        let equal = CandidateProbe::new(6, 6);
        assert!(equal.locked);
    }
}
