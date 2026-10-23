//! Confidence-scheduled verification, after DSpark.
//!
//! Speculative decoding drafts `gamma` tokens and verifies them in one forward of the target.
//! Under batching every drafted token costs batch capacity, and a token that will be rejected
//! costs it for nothing; DSpark calls this verification waste and sizes each request's
//! verification length from its drafter's confidence. Its hardware-aware prefix scheduler
//! (section 3.2.2 of the paper, Algorithm 1) takes for each request `r` the conditional
//! survival probability `c_{r,k}` of every draft position, forms the prefix survival
//! `a_{r,j} = prod_{i <= j} c_{r,i}`, sorts every `(r, j)` pair by `a_{r,j}` descending, and
//! admits pairs while the objective `theta = tau x SPS(B)` improves, where `tau` is the
//! expected accepted tokens (`sum_r (1 + sum_{j <= l_r} a_{r,j})`), `B` the batch in tokens
//! (`sum_r (1 + l_r)`) and `SPS(B)` the engine's steps per second at that batch, profiled once
//! and kept as a cost table. The greedy order is exact because `a` is monotone in `j`; the
//! early stop keeps the admission decision causal (Appendix A: an unconstrained search would
//! let the sampled token leak into the decision). DeepInfer's production scheduler drops the
//! early stop (jagged `SPS` curves trap it) and sets the batch budget from confidences two
//! steps behind, because its zero-overhead scheduler must know the next batch before the
//! current step ends.
//!
//! Here the profile comes from calibration records across batch widths (measured on the
//! reference kernels) or, when fewer than two widths are measured, from the cost model
//! (modelled). There is no drafter: the confidences are modelled by [`ModelledDrafter`], a
//! per-session base and decay with a deterministic jitter, and say so. The scheduler itself,
//! the lag and the accounting are DSpark's and DeepInfer's. Left out: the semi-autoregressive
//! drafter and its confidence head, sequential temperature scaling, the variable-length
//! attention kernels.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{CalibrationRecord, ExecutionStrategy};

/// Draft length per step at most, DeepInfer's default (`DEEPINFER_USE_NEXTN_DECODE=5`).
pub const DEFAULT_MAX_DRAFT: u32 = 5;

/// Steps per second as a function of the batch width in rows (tokens verified per step),
/// DSpark's `SPS(B)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepsPerSecondProfile {
    /// `(batch width, steps per second)`, widths ascending and distinct.
    pub points: Vec<(u32, f64)>,
    /// `measured` from calibration records, or `modelled` from the cost model.
    pub source: String,
    /// The kernels the measured points ran on.
    pub kernels: String,
}

impl StepsPerSecondProfile {
    pub fn new(mut points: Vec<(u32, f64)>, source: &str, kernels: &str) -> Self {
        points.retain(|(w, sps)| *w > 0 && sps.is_finite() && *sps > 0.0);
        points.sort_by_key(|(w, _)| *w);
        points.dedup_by_key(|(w, _)| *w);
        Self {
            points,
            source: source.to_string(),
            kernels: kernels.to_string(),
        }
    }

    /// A profile from the trusted calibration records of one placement, one point per batch
    /// width: steps per second is the inverse of the step's wall, `rows x ms_per_token`.
    pub fn from_calibrations<'a>(records: impl IntoIterator<Item = &'a CalibrationRecord>) -> Self {
        let mut kernels = crate::CALIBRATION_KERNELS_REFERENCE.to_string();
        let points: Vec<(u32, f64)> = records
            .into_iter()
            .filter(|r| r.is_trusted())
            .map(|r| {
                kernels = r.kernels.clone();
                let step_ms = r.measured_ms_per_token * r.binding.rows.max(1) as f64;
                (
                    r.binding.rows.max(1),
                    if step_ms > 0.0 { 1000.0 / step_ms } else { 0.0 },
                )
            })
            .collect();
        Self::new(points, "measured", &kernels)
    }

    /// A profile from modelled step times, `(width, step ms)` per width.
    pub fn from_modelled_steps(step_ms: &[(u32, f64)]) -> Self {
        Self::new(
            step_ms
                .iter()
                .map(|(w, ms)| (*w, if *ms > 0.0 { 1000.0 / ms } else { 0.0 }))
                .collect(),
            "modelled",
            "cost model",
        )
    }

    pub fn is_measured(&self) -> bool {
        self.source == "measured"
    }

    pub fn widths(&self) -> Vec<u32> {
        self.points.iter().map(|(w, _)| *w).collect()
    }

    /// Steps per second at `width`: a measured point, a linear interpolation between the two
    /// around it, or beyond the last point the step time extended along the last two points'
    /// slope (never below a thousandth of the last rate). With one point the profile is flat.
    pub fn sps(&self, width: u32) -> f64 {
        let width = width.max(1);
        match self.points.len() {
            0 => 0.0,
            1 => self.points[0].1,
            _ => {
                if width <= self.points[0].0 {
                    return self.points[0].1;
                }
                for pair in self.points.windows(2) {
                    let (w0, s0) = pair[0];
                    let (w1, s1) = pair[1];
                    if width == w1 {
                        return s1;
                    }
                    if width > w0 && width < w1 {
                        let t = (width - w0) as f64 / (w1 - w0) as f64;
                        return s0 + (s1 - s0) * t;
                    }
                }
                // Beyond the last point: extend the step time linearly.
                let (w0, s0) = self.points[self.points.len() - 2];
                let (w1, s1) = self.points[self.points.len() - 1];
                let (t0, t1) = (1000.0 / s0, 1000.0 / s1);
                let slope = (t1 - t0) / (w1 - w0) as f64;
                let step_ms = t1 + slope * (width - w1) as f64;
                (1000.0 / step_ms.max(1e-9)).clamp(s1 * 0.001, s1.max(s0))
            }
        }
    }

    /// Step time at `width`, in ms.
    pub fn step_ms(&self, width: u32) -> f64 {
        let sps = self.sps(width);
        if sps > 0.0 { 1000.0 / sps } else { 0.0 }
    }

    pub fn summary(&self) -> String {
        format!(
            "{} over {} width{} ({}) on {}",
            self.source,
            self.points.len(),
            if self.points.len() == 1 { "" } else { "s" },
            self.points
                .iter()
                .map(|(w, s)| format!("{w}: {s:.1}/s"))
                .collect::<Vec<_>>()
                .join(", "),
            self.kernels
        )
    }
}

/// What the scheduler knows about one session's draft at one step: the conditional survival
/// probability of every position, `c_1 .. c_gamma`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionDraft {
    pub session_id: String,
    pub confidences: Vec<f64>,
}

impl SessionDraft {
    /// Prefix survival `a_j = prod_{i <= j} c_i`, non-increasing in `j`.
    pub fn survival(&self) -> Vec<f64> {
        let mut acc = 1.0;
        self.confidences
            .iter()
            .map(|c| {
                acc *= c.clamp(0.0, 1.0);
                acc
            })
            .collect()
    }

    /// Expected accepted draft tokens if every position were verified: `sum_j a_j`.
    pub fn expected_accepted(&self) -> f64 {
        self.survival().iter().sum()
    }
}

/// How the scheduler runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationPolicy {
    /// Draft positions per session at most (`gamma`).
    pub max_draft: u32,
    /// Stop at the first pair that does not improve the objective (DSpark's causal scheduler)
    /// or search every pair (DeepInfer's production variant).
    pub early_stop: bool,
    /// Which step's confidences size a step's verification: `1` plans step `t + 1` from the
    /// confidences of step `t - 1` (the latest settled when `t + 1` is planned here), `2` is
    /// DeepInfer's zero-overhead scheduler, `0` the paper's offline setting.
    pub confidence_lag: u32,
    /// Batch in tokens the mesh may verify at once, if bounded.
    pub max_batch_tokens: Option<u32>,
}

impl Default for VerificationPolicy {
    fn default() -> Self {
        Self {
            max_draft: DEFAULT_MAX_DRAFT,
            early_stop: true,
            confidence_lag: 1,
            max_batch_tokens: None,
        }
    }
}

/// What the scheduler decided for one step across the sessions it was given.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationSchedule {
    /// Verification length per session, in the drafts' order.
    pub lengths: Vec<u32>,
    /// `B`: rows verified this step across sessions, one golden row per session plus the drafts.
    pub batch_tokens: u32,
    /// `tau`: expected accepted tokens, one per session plus the admitted positions' survival.
    pub expected_accepted: f64,
    /// `theta = tau x SPS(B)`, expected accepted tokens per second.
    pub throughput: f64,
    /// `R x SPS(R)`: the objective with no draft verified at all.
    pub baseline_throughput: f64,
    /// Steps per second the profile gives at `batch_tokens`.
    pub steps_per_second: f64,
    pub pairs_considered: usize,
    pub pairs_admitted: usize,
    /// Whether the early stop ended the search before the last pair.
    pub early_stopped: bool,
    pub profile_source: String,
}

impl VerificationSchedule {
    pub fn length_for(&self, index: usize) -> u32 {
        self.lengths.get(index).copied().unwrap_or(0)
    }

    /// Expected gain over verifying nothing, as a fraction.
    pub fn gain(&self) -> f64 {
        if self.baseline_throughput <= 0.0 {
            0.0
        } else {
            self.throughput / self.baseline_throughput - 1.0
        }
    }
}

pub struct VerificationScheduler;

impl VerificationScheduler {
    /// DSpark's Algorithm 1 over `drafts`: every `(session, position)` pair in descending prefix
    /// survival (ties: the earlier session, then the earlier position), admitted while
    /// `tau x SPS(B)` improves; with `early_stop` the search ends at the first pair that does
    /// not, else the best point of the whole walk is kept.
    pub fn schedule(
        drafts: &[SessionDraft],
        profile: &StepsPerSecondProfile,
        policy: &VerificationPolicy,
    ) -> VerificationSchedule {
        let sessions = drafts.len();
        let mut pairs: Vec<(f64, usize, u32)> = Vec::new();
        for (r, draft) in drafts.iter().enumerate() {
            for (j, a) in draft
                .survival()
                .into_iter()
                .take(policy.max_draft as usize)
                .enumerate()
            {
                if a > 0.0 {
                    pairs.push((a, r, j as u32 + 1));
                }
            }
        }
        pairs.sort_by(|x, y| {
            y.0.partial_cmp(&x.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.1.cmp(&y.1))
                .then(x.2.cmp(&y.2))
        });

        let mut lengths = vec![0u32; sessions];
        let mut batch = sessions as u32;
        let mut tau = sessions as f64;
        let baseline = tau * profile.sps(batch.max(1));
        let mut best = (baseline, lengths.clone(), batch, tau);
        let mut admitted = 0usize;
        let mut early_stopped = false;
        for (index, (a, r, j)) in pairs.iter().enumerate() {
            if policy.max_batch_tokens.is_some_and(|cap| batch + 1 > cap) {
                break;
            }
            // Pairs arrive in prefix order per session because survival is monotone.
            lengths[*r] = *j;
            batch += 1;
            tau += a;
            let theta = tau * profile.sps(batch);
            if theta > best.0 {
                best = (theta, lengths.clone(), batch, tau);
                admitted = index + 1;
            } else if policy.early_stop {
                early_stopped = true;
                break;
            }
        }
        let (throughput, lengths, batch_tokens, expected_accepted) = best;
        VerificationSchedule {
            lengths,
            batch_tokens,
            expected_accepted,
            throughput,
            baseline_throughput: baseline,
            steps_per_second: profile.sps(batch_tokens.max(1)),
            pairs_considered: pairs.len(),
            pairs_admitted: admitted,
            early_stopped,
            profile_source: profile.source.clone(),
        }
    }
}

/// The confidences a drafter would report, modelled: a per-session base acceptance that
/// decays along the block (DSpark's suffix decay), with a small deterministic jitter per
/// step and position. There is no drafter behind these numbers; they stand in for one until
/// a real drafter reports its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DraftProfile {
    /// Conditional acceptance of the first draft position.
    pub base: f64,
    /// Drop in conditional acceptance per further position.
    pub decay: f64,
}

impl DraftProfile {
    /// Roughly DSpark's conditional acceptance on code (stable, high).
    pub fn code() -> Self {
        Self {
            base: 0.85,
            decay: 0.03,
        }
    }

    /// Roughly DSpark's conditional acceptance on open-ended chat (lower, decaying).
    pub fn chat() -> Self {
        Self {
            base: 0.62,
            decay: 0.06,
        }
    }

    /// A drafter that is never right: every confidence is zero, nothing is verified, and a
    /// speculative session decodes exactly as a plain one.
    pub fn none() -> Self {
        Self {
            base: 0.0,
            decay: 0.0,
        }
    }

    pub fn named(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "code" => Some(Self::code()),
            "chat" => Some(Self::chat()),
            "none" => Some(Self::none()),
            _ => None,
        }
    }
}

pub struct ModelledDrafter;

impl ModelledDrafter {
    /// A deterministic jitter in `[-1, 1)` from the session seed, the step and the position.
    fn jitter(seed: u64, step: u64, position: u32) -> f64 {
        let mut x = seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((position as u64) << 40);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        x ^= x >> 33;
        (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    /// The modelled conditional survival `c_1 .. c_gamma` of a session's draft at `step`.
    pub fn confidences(profile: &DraftProfile, seed: u64, step: u64, gamma: u32) -> Vec<f64> {
        if profile.base <= 0.0 {
            return vec![0.0; gamma as usize];
        }
        (0..gamma)
            .map(|k| {
                let c =
                    profile.base - profile.decay * k as f64 + 0.04 * Self::jitter(seed, step, k);
                (c * 10_000.0).round() / 10_000.0
            })
            .map(|c| c.clamp(0.02, 0.99))
            .collect()
    }

    /// Whether position `k` (1-based) survives verification, given its confidence and the
    /// verified row's digest: the coin that stands in for the target's acceptance test. The
    /// digest makes it a function of the bytes actually verified, so every node agrees.
    pub fn accepts(confidence: f64, row_digest: u64, seed: u64, k: u32) -> bool {
        let mut x = row_digest ^ seed.rotate_left(k % 63);
        x ^= x >> 29;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 32;
        let u = (x >> 11) as f64 / (1u64 << 53) as f64;
        u < confidence
    }
}

/// How the tokens of a speculative step are counted, side by side, the way DeepInfer's
/// benchmark README discloses it: its decode throughput counts `1 + nextn` tokens per request
/// per step regardless of the true acceptance rate (full acceptance), and its plotter scales
/// that by `tau = 1 + nextn x acceptance`. Here the realised acceptance is itself modelled.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceAccounting {
    pub steps: u64,
    /// Draft positions verified over the steps.
    pub proposed: u64,
    /// Draft positions the modelled acceptance test passed, in prefix.
    pub accepted: u64,
    /// `sum a_j` over the admitted positions: what the scheduler expected.
    pub expected_accepted: f64,
    /// Tokens under full-acceptance accounting: one golden plus every draft, per step.
    pub tokens_full_acceptance: u64,
    /// Tokens under the modelled acceptance: one golden plus the accepted prefix, per step.
    pub tokens_accepted: u64,
    /// `accepted / proposed`, the per-position acceptance rate the scaled figure uses.
    pub acceptance_rate: f64,
    /// Always true: the acceptance behind every figure is modelled, not a drafter's.
    pub modelled: bool,
}

impl AcceptanceAccounting {
    pub fn record_step(&mut self, proposed: u32, accepted: u32, expected: f64) {
        self.steps += 1;
        self.proposed += proposed as u64;
        self.accepted += accepted as u64;
        self.expected_accepted += expected;
        self.tokens_full_acceptance += 1 + proposed as u64;
        self.tokens_accepted += 1 + accepted as u64;
        self.acceptance_rate = if self.proposed > 0 {
            self.accepted as f64 / self.proposed as f64
        } else {
            0.0
        };
        self.modelled = true;
    }

    /// DeepInfer's plotter factor: `tau = 1 + nextn x acceptance`, with `nextn` the mean draft
    /// length verified per step.
    pub fn tau(&self) -> f64 {
        if self.steps == 0 {
            1.0
        } else {
            1.0 + (self.proposed as f64 / self.steps as f64) * self.acceptance_rate
        }
    }
}

/// The calibration records of one placement by batch width, for a profile.
pub fn records_for_placement<'a>(
    records: impl IntoIterator<Item = &'a CalibrationRecord>,
    model_id: &str,
    strategy: ExecutionStrategy,
    placement_digest: u64,
) -> BTreeMap<u32, &'a CalibrationRecord> {
    let mut by_width: BTreeMap<u32, &CalibrationRecord> = BTreeMap::new();
    for r in records {
        if r.binding.model_id == model_id
            && r.binding.strategy == strategy
            && r.binding.placement_digest == placement_digest
            && r.is_trusted()
        {
            let keep = by_width
                .get(&r.binding.rows)
                .is_none_or(|existing| r.sampled_unix_ms >= existing.sampled_unix_ms);
            if keep {
                by_width.insert(r.binding.rows, r);
            }
        }
    }
    by_width
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CalibrationBinding;

    fn profile(points: &[(u32, f64)]) -> StepsPerSecondProfile {
        StepsPerSecondProfile::new(points.to_vec(), "measured", "reference-synthetic")
    }

    #[test]
    fn the_profile_interpolates_and_extends_the_step_time() {
        let p = profile(&[(1, 100.0), (2, 80.0), (4, 50.0)]);
        assert_eq!(p.sps(1), 100.0);
        assert_eq!(p.sps(2), 80.0);
        assert_eq!(p.sps(4), 50.0);
        assert!((p.sps(3) - 65.0).abs() < 1e-9, "linear between 2 and 4");
        assert_eq!(
            p.sps(0),
            100.0,
            "below the first width the first point holds"
        );
        // Step time at 2 is 12.5 ms, at 4 is 20 ms: 3.75 ms per row beyond.
        assert!((p.step_ms(6) - 27.5).abs() < 1e-9, "{}", p.step_ms(6));
        assert!(p.sps(6) < p.sps(4));
        assert_eq!(profile(&[(3, 42.0)]).sps(9), 42.0, "one point is flat");
        assert_eq!(profile(&[]).sps(1), 0.0);
        assert!(p.summary().contains("measured over 3 widths"));

        let binding = |rows: u32| {
            CalibrationBinding::new(
                "toy",
                ExecutionStrategy::PipelineChain,
                7,
                vec!["a".into()],
                BTreeMap::new(),
                rows,
            )
        };
        let rec = |rows: u32, step_ms: f64| {
            CalibrationRecord::from_steps(
                binding(rows),
                &[step_ms],
                BTreeMap::new(),
                1,
                None,
                rows as u64,
            )
        };
        let records = [rec(4, 40.0), rec(1, 10.0), rec(2, 16.0)];
        let measured = StepsPerSecondProfile::from_calibrations(records.iter());
        assert!(measured.is_measured());
        assert_eq!(measured.widths(), vec![1, 2, 4]);
        assert!((measured.sps(1) - 100.0).abs() < 1e-6);
        assert!((measured.sps(4) - 25.0).abs() < 1e-6);
        let mut failed = rec(8, 80.0);
        failed.digest_matches_chain = Some(false);
        let by_width = records_for_placement(
            records.iter().chain(std::iter::once(&failed)),
            "toy",
            ExecutionStrategy::PipelineChain,
            7,
        );
        assert_eq!(
            by_width.len(),
            3,
            "a record that failed the digest gate is left out"
        );
        assert!(
            records_for_placement(records.iter(), "toy", ExecutionStrategy::ExpertParallel, 7)
                .is_empty()
        );
    }

    #[test]
    fn the_scheduler_admits_by_survival_while_the_objective_improves() {
        // Two sessions, a flat profile: every position with positive survival pays.
        let drafts = vec![
            SessionDraft {
                session_id: "code".into(),
                confidences: vec![0.9, 0.9, 0.9],
            },
            SessionDraft {
                session_id: "chat".into(),
                confidences: vec![0.6, 0.5, 0.4],
            },
        ];
        let flat = profile(&[(1, 10.0), (8, 10.0)]);
        let policy = VerificationPolicy {
            max_draft: 3,
            ..VerificationPolicy::default()
        };
        let s = VerificationScheduler::schedule(&drafts, &flat, &policy);
        assert_eq!(s.lengths, vec![3, 3]);
        assert_eq!(s.batch_tokens, 8);
        assert!(!s.early_stopped);
        assert_eq!(s.pairs_admitted, 6);
        let expected = 2.0 + (0.9 + 0.81 + 0.729) + (0.6 + 0.3 + 0.12);
        assert!((s.expected_accepted - expected).abs() < 1e-9);
        assert!((s.throughput - expected * 10.0).abs() < 1e-9);
        assert!(s.gain() > 1.0);

        // A steep profile: steps per second fall fast beyond two rows, so only the most
        // confident position pays (2.9 x 80 beats 2 x 100; 3.71 x 30 does not), and the code
        // session gets it.
        let steep = profile(&[
            (2, 100.0),
            (3, 80.0),
            (4, 30.0),
            (5, 12.0),
            (6, 5.0),
            (8, 1.0),
        ]);
        let s = VerificationScheduler::schedule(&drafts, &steep, &policy);
        assert_eq!(s.lengths, vec![1, 0], "{s:?}");
        assert_eq!(s.batch_tokens, 3);
        assert!(s.early_stopped);
        assert!((s.baseline_throughput - 200.0).abs() < 1e-9);
        assert!((s.throughput - (2.0 + 0.9) * 80.0).abs() < 1e-9);
        // Too steep for even the first position: nothing is verified.
        let nothing =
            VerificationScheduler::schedule(&drafts, &profile(&[(2, 100.0), (3, 60.0)]), &policy);
        assert_eq!(nothing.lengths, vec![0, 0]);
        assert_eq!(nothing.pairs_admitted, 0);
        assert_eq!(nothing.gain(), 0.0);

        // Without the early stop the walk continues and keeps the best point it saw.
        let global = VerificationScheduler::schedule(
            &drafts,
            &profile(&[(2, 100.0), (3, 10.0), (4, 90.0), (8, 1.0)]),
            &VerificationPolicy {
                max_draft: 3,
                early_stop: false,
                ..VerificationPolicy::default()
            },
        );
        assert_eq!(global.batch_tokens, 4, "{global:?}");
        assert!(!global.early_stopped);
        let causal = VerificationScheduler::schedule(
            &drafts,
            &profile(&[(2, 100.0), (3, 10.0), (4, 90.0), (8, 1.0)]),
            &policy,
        );
        assert_eq!(
            causal.batch_tokens, 2,
            "the dip at three rows stops the causal walk"
        );

        // A batch cap and no drafts at all.
        let capped = VerificationScheduler::schedule(
            &drafts,
            &flat,
            &VerificationPolicy {
                max_draft: 3,
                max_batch_tokens: Some(4),
                ..VerificationPolicy::default()
            },
        );
        assert_eq!(capped.batch_tokens, 4);
        assert_eq!(capped.lengths.iter().sum::<u32>(), 2);
        let none = VerificationScheduler::schedule(&[], &flat, &policy);
        assert_eq!(none.batch_tokens, 0);
        assert_eq!(none.lengths, Vec::<u32>::new());
        assert_eq!(
            SessionDraft {
                session_id: "z".into(),
                confidences: vec![0.0, 0.9]
            }
            .survival(),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn modelled_confidences_decay_and_the_coin_is_deterministic() {
        let code = ModelledDrafter::confidences(&DraftProfile::code(), 7, 3, 5);
        assert_eq!(code.len(), 5);
        assert!(code[0] > code[4], "{code:?}");
        assert!(code.iter().all(|c| (0.02..=0.99).contains(c)));
        let again = ModelledDrafter::confidences(&DraftProfile::code(), 7, 3, 5);
        assert_eq!(code, again);
        let other_step = ModelledDrafter::confidences(&DraftProfile::code(), 7, 4, 5);
        assert_ne!(code, other_step);
        let chat = ModelledDrafter::confidences(&DraftProfile::chat(), 7, 3, 5);
        assert!(chat[0] < code[0]);
        assert_eq!(
            ModelledDrafter::confidences(&DraftProfile::none(), 7, 3, 3),
            vec![0.0; 3]
        );
        assert!(DraftProfile::named("CODE").is_some());
        assert!(DraftProfile::named("poetry").is_none());

        assert!(
            ModelledDrafter::accepts(1.0, 42, 1, 1) || !ModelledDrafter::accepts(0.0, 42, 1, 1)
        );
        assert!(!ModelledDrafter::accepts(0.0, 42, 1, 1));
        assert_eq!(
            ModelledDrafter::accepts(0.5, 42, 1, 2),
            ModelledDrafter::accepts(0.5, 42, 1, 2)
        );
        // Over many digests the coin lands near its confidence.
        let hits = (0..10_000u64)
            .filter(|d| ModelledDrafter::accepts(0.7, d.wrapping_mul(0x9E37_79B9), 3, 1))
            .count();
        assert!((6_800..=7_200).contains(&hits), "{hits}");

        let mut acc = AcceptanceAccounting::default();
        acc.record_step(3, 2, 1.8);
        acc.record_step(3, 0, 1.2);
        assert_eq!(acc.tokens_full_acceptance, 8);
        assert_eq!(acc.tokens_accepted, 4);
        assert!((acc.acceptance_rate - 2.0 / 6.0).abs() < 1e-9);
        assert!((acc.tau() - 2.0).abs() < 1e-9, "1 + 3 x (1/3)");
        assert!(acc.modelled);
    }
}
