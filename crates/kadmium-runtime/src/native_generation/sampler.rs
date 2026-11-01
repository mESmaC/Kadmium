//! Token selection from the model's own generation settings.
//!
//! Implemented, in the order Transformers applies them: repetition penalty (also when decoding
//! greedily), then, when `do_sample` is set, temperature, top-k, top-p, min-p and typical-p,
//! then a draw from the remaining distribution. Without `do_sample` the token is the first index
//! of the largest logit, as `torch.argmax` picks it.
//!
//! Every other Transformers `GenerationConfig` setting that changes which tokens are produced is
//! listed in [`UNIMPLEMENTED`] and must hold its inactive value; an active one is refused, never
//! ignored. Settings outside that list do not select tokens (output flags, cache layout, beam-only
//! scoring when `num_beams` is 1) or are handled elsewhere (EOS ids and stop strings).
//!
//! The filters mask logits in `f32`, as Transformers does; probabilities, cumulative sums and the
//! draw are computed in `f64`. A draw is reproducible from the seed recorded in the report on the
//! coordinator that made it.

use super::GenerationError;
use crate::model_text::SamplingDefaults;
use serde_json::{Map, Value};
use std::collections::BTreeSet;

/// Transformers `GenerationConfig` settings this generator does not implement, each with the
/// number that leaves it inactive (`None`: only null, `false` or an empty list or map does).
pub const UNIMPLEMENTED: &[(&str, Option<f64>)] = &[
    ("num_beams", Some(1.0)),
    ("num_beam_groups", Some(1.0)),
    ("num_return_sequences", Some(1.0)),
    ("diversity_penalty", Some(0.0)),
    ("penalty_alpha", None),
    ("no_repeat_ngram_size", Some(0.0)),
    ("encoder_no_repeat_ngram_size", Some(0.0)),
    ("encoder_repetition_penalty", Some(1.0)),
    ("bad_words_ids", None),
    ("force_words_ids", None),
    ("constraints", None),
    ("sequence_bias", None),
    ("suppress_tokens", None),
    ("begin_suppress_tokens", None),
    ("forced_bos_token_id", None),
    ("forced_eos_token_id", None),
    ("forced_decoder_ids", None),
    ("min_length", Some(0.0)),
    ("min_new_tokens", Some(0.0)),
    ("epsilon_cutoff", Some(0.0)),
    ("eta_cutoff", Some(0.0)),
    ("exponential_decay_length_penalty", None),
    ("guidance_scale", Some(1.0)),
    ("renormalize_logits", None),
    ("remove_invalid_values", None),
    ("watermarking_config", None),
    ("max_time", None),
    ("dola_layers", None),
    ("prompt_lookup_num_tokens", None),
    ("assistant_early_exit", None),
    ("token_healing", None),
    ("top_h", None),
];

fn inactive(value: &Value, number: Option<f64>) -> bool {
    match value {
        Value::Null | Value::Bool(false) => true,
        Value::Array(items) => items.is_empty(),
        Value::Object(entries) => entries.is_empty(),
        Value::Number(n) => number.is_some_and(|inert| n.as_f64() == Some(inert)),
        _ => false,
    }
}

/// Refuses a declared setting this generator would otherwise ignore.
pub(crate) fn check_declared(declared: &Map<String, Value>) -> Result<(), GenerationError> {
    for &(key, number) in UNIMPLEMENTED {
        if let Some(value) = declared.get(key)
            && !inactive(value, number)
        {
            return Err(GenerationError::UnsupportedSetting {
                key: key.to_owned(),
                value: value.to_string(),
            });
        }
    }
    Ok(())
}

/// SplitMix64. Statistical quality is ample for drawing one token per step.
#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

pub(crate) struct Sampler {
    sampling: SamplingDefaults,
    rng: Option<SplitMix64>,
    /// Tokens of the context read so far, for the repetition penalty.
    seen: BTreeSet<u32>,
    seen_len: usize,
}

/// Every score is a number or `-inf`, and at least one is finite.
fn usable(scores: &[f32]) -> bool {
    scores.iter().all(|v| !v.is_nan() && *v != f32::INFINITY)
        && scores.iter().any(|v| v.is_finite())
}

/// Refuses a sampling value Transformers itself rejects.
pub(crate) fn check_sampling(sampling: &SamplingDefaults) -> Result<(), GenerationError> {
    if sampling.do_sample && sampling.typical_p <= 0.0 {
        return Err(GenerationError::InvalidSetting {
            key: "typical_p".into(),
            value: sampling.typical_p.to_string(),
            requirement: "0 < typical_p <= 1".into(),
        });
    }
    Ok(())
}

impl Sampler {
    /// `seed` is used only when the model samples.
    pub(crate) fn new(sampling: SamplingDefaults, seed: u64) -> Self {
        let rng = sampling.do_sample.then_some(SplitMix64(seed));
        Self {
            sampling,
            rng,
            seen: BTreeSet::new(),
            seen_len: 0,
        }
    }

    /// Selects the token at `position` from the logits of the position before it. `context` is
    /// every token so far, prompt included, as the repetition penalty reads it; each call's
    /// context extends the previous call's.
    ///
    /// The logits are an error if they hold NaN or `+inf`, or no finite value, before or after
    /// the penalty and the temperature, so an overflow never falls through to a token.
    pub(crate) fn select(
        &mut self,
        logits: &[f32],
        context: &[u32],
        position: usize,
    ) -> Result<u32, GenerationError> {
        let invalid = || GenerationError::InvalidLogits { position };
        if !usable(logits) {
            return Err(invalid());
        }
        let mut scores = logits.to_vec();
        let s = &self.sampling;
        if s.repetition_penalty != 1.0 {
            if let Some(new) = context.get(self.seen_len..) {
                self.seen.extend(new.iter().copied());
            }
            self.seen_len = context.len();
            let penalty = s.repetition_penalty as f32;
            for &token in &self.seen {
                if let Some(score) = scores.get_mut(token as usize) {
                    *score = if *score < 0.0 {
                        *score * penalty
                    } else {
                        *score / penalty
                    };
                }
            }
            if !usable(&scores) {
                return Err(invalid());
            }
        }
        let Some(rng) = self.rng.as_mut() else {
            return Ok(argmax(&scores));
        };
        if s.temperature != 1.0 {
            let temperature = s.temperature as f32;
            for score in &mut scores {
                *score /= temperature;
            }
            if !usable(&scores) {
                return Err(invalid());
            }
        }
        if s.top_k != 0 {
            top_k(&mut scores, s.top_k);
        }
        if s.top_p < 1.0 {
            top_p(&mut scores, s.top_p);
        }
        if let Some(min_p) = s.min_p {
            min_p_filter(&mut scores, min_p);
        }
        if s.typical_p < 1.0 {
            typical(&mut scores, s.typical_p);
        }
        let probabilities = softmax(&scores);
        let total: f64 = probabilities.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(invalid());
        }
        let target = rng.next_f64() * total;
        let mut cumulative = 0.0;
        let mut last_possible = None;
        for (index, &p) in probabilities.iter().enumerate() {
            if p <= 0.0 {
                continue;
            }
            cumulative += p;
            last_possible = Some(index);
            if cumulative > target {
                return Ok(index as u32);
            }
        }
        // Rounding left the target at the very top of the cumulative sum.
        last_possible.map(|index| index as u32).ok_or_else(invalid)
    }
}

/// First index of the largest value. `scores` holds no NaN.
fn argmax(scores: &[f32]) -> u32 {
    let mut best = 0;
    for (index, &score) in scores.iter().enumerate() {
        if score > scores[best] {
            best = index;
        }
    }
    best as u32
}

fn softmax(scores: &[f32]) -> Vec<f64> {
    let max: f64 = scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .into();
    scores
        .iter()
        .map(|&s| {
            if s == f32::NEG_INFINITY {
                0.0
            } else {
                (f64::from(s) - max).exp()
            }
        })
        .collect()
}

fn normalized(scores: &[f32]) -> Vec<f64> {
    let mut p = softmax(scores);
    let total: f64 = p.iter().sum();
    for value in &mut p {
        *value /= total;
    }
    p
}

/// Indices in ascending order of `key`; equal keys put the higher index first, so that among
/// equal logits the lower index counts as the more likely, as `argmax` treats it.
fn ascending_by<F: Fn(usize) -> f64>(len: usize, key: F) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| key(a).total_cmp(&key(b)).then(b.cmp(&a)));
    order
}

/// Keeps logits at least as large as the `k`-th largest.
fn top_k(scores: &mut [f32], k: u64) {
    let k = usize::try_from(k)
        .unwrap_or(usize::MAX)
        .clamp(1, scores.len());
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let threshold = sorted[k - 1];
    for score in scores.iter_mut() {
        if *score < threshold {
            *score = f32::NEG_INFINITY;
        }
    }
}

/// Removes the least likely tokens whose cumulative probability stays within `1 - top_p`,
/// always keeping the most likely.
fn top_p(scores: &mut [f32], top_p: f64) {
    let order = ascending_by(scores.len(), |i| f64::from(scores[i]));
    let probabilities = normalized(scores);
    let mut cumulative = 0.0;
    let mut remove = Vec::new();
    for &index in &order[..order.len() - 1] {
        cumulative += probabilities[index];
        if cumulative <= 1.0 - top_p {
            remove.push(index);
        }
    }
    for index in remove {
        scores[index] = f32::NEG_INFINITY;
    }
}

/// Removes tokens less likely than `min_p` times the most likely one.
fn min_p_filter(scores: &mut [f32], min_p: f64) {
    let probabilities = normalized(scores);
    let top = probabilities.iter().copied().fold(0.0, f64::max);
    let keep = argmax(scores) as usize;
    for (index, &p) in probabilities.iter().enumerate() {
        if index != keep && p < min_p * top {
            scores[index] = f32::NEG_INFINITY;
        }
    }
}

/// Locally typical sampling: keeps the tokens whose information content is closest to the
/// distribution's entropy, up to probability mass `mass`.
fn typical(scores: &mut [f32], mass: f64) {
    let probabilities = normalized(scores);
    let entropy: f64 = probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let shifted: Vec<f64> = probabilities
        .iter()
        .map(|&p| {
            if p > 0.0 {
                (-p.ln() - entropy).abs()
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let order = ascending_by(scores.len(), |i| shifted[i]);
    let mut cumulative = 0.0;
    let mut below = 0;
    for &index in &order {
        cumulative += probabilities[index];
        if cumulative < mass {
            below += 1;
        }
    }
    let cut = shifted[order[below.min(order.len() - 1)]];
    for (rank, &index) in order.iter().enumerate() {
        if rank > 0 && shifted[index] > cut {
            scores[index] = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy(repetition_penalty: f64) -> SamplingDefaults {
        SamplingDefaults {
            do_sample: false,
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            min_p: None,
            typical_p: 1.0,
            repetition_penalty,
        }
    }

    fn sampling() -> SamplingDefaults {
        SamplingDefaults {
            do_sample: true,
            top_k: 0,
            ..greedy(1.0)
        }
    }

    #[test]
    fn greedy_picks_the_first_largest_logit() {
        let mut sampler = Sampler::new(greedy(1.0), 0);
        assert_eq!(sampler.select(&[1.0, 3.0, 3.0, 2.0], &[], 0).unwrap(), 1);
    }

    #[test]
    fn repetition_penalty_divides_positive_and_multiplies_negative_logits() {
        let mut scores = vec![2.0f32, -1.0, 1.9];
        let mut sampler = Sampler::new(greedy(2.0), 0);
        // Token 0 seen: 2.0 / 2 = 1.0 < 1.9, so token 2 wins.
        assert_eq!(sampler.select(&scores, &[0, 0, 1], 0).unwrap(), 2);
        scores[2] = 0.5;
        assert_eq!(sampler.select(&scores, &[0], 0).unwrap(), 0);
    }

    #[test]
    fn nan_and_positive_infinity_are_errors_not_tokens() {
        let mut sampler = Sampler::new(greedy(1.0), 0);
        for bad in [f32::NAN, f32::INFINITY] {
            assert!(matches!(
                sampler.select(&[0.0, bad], &[], 7),
                Err(GenerationError::InvalidLogits { position: 7 })
            ));
        }
    }

    #[test]
    fn overflow_after_the_processors_is_an_error_not_a_token() {
        // A tiny positive temperature turns finite logits into +inf.
        let mut logits = vec![0.0f32; 4];
        logits[1] = 50.0;
        logits[2] = 49.0;
        let tiny = SamplingDefaults {
            temperature: 1e-38,
            ..sampling()
        };
        assert!(matches!(
            Sampler::new(tiny, 1).select(&logits, &[], 3),
            Err(GenerationError::InvalidLogits { position: 3 })
        ));
        // No finite logit is an error greedily and when sampling alike.
        let none = [f32::NEG_INFINITY; 3];
        for settings in [greedy(1.0), sampling()] {
            assert!(matches!(
                Sampler::new(settings, 1).select(&none, &[], 0),
                Err(GenerationError::InvalidLogits { position: 0 })
            ));
        }
        // A penalty below 1 can overflow a large positive logit.
        assert!(matches!(
            Sampler::new(greedy(1e-38), 0).select(&[f32::MAX, 0.0], &[0], 0),
            Err(GenerationError::InvalidLogits { .. })
        ));
    }

    #[test]
    fn the_penalty_reads_the_context_incrementally() {
        let mut sampler = Sampler::new(greedy(2.0), 0);
        let scores = [2.0f32, 1.5, 1.9];
        assert_eq!(sampler.select(&scores, &[1], 0).unwrap(), 0);
        // Token 0 joins the context: 2.0 / 2 = 1.0, so token 2 wins.
        assert_eq!(sampler.select(&scores, &[1, 0], 1).unwrap(), 2);
        // All three penalised: 1.0, 0.75 and 0.95, so token 0 wins again.
        assert_eq!(sampler.select(&scores, &[1, 0, 2], 2).unwrap(), 0);
    }

    #[test]
    fn typical_p_zero_is_refused_as_transformers_refuses_it() {
        let zero = SamplingDefaults {
            typical_p: 0.0,
            ..sampling()
        };
        assert!(matches!(
            check_sampling(&zero),
            Err(GenerationError::InvalidSetting { key, .. }) if key == "typical_p"
        ));
        check_sampling(&sampling()).unwrap();
        check_sampling(&SamplingDefaults {
            typical_p: 0.0,
            ..greedy(1.0)
        })
        .unwrap();
    }

    #[test]
    fn filters_keep_the_most_likely_token() {
        let scores = [0.0f32, 5.0, 1.0, 4.0];
        let mut s = scores.to_vec();
        top_k(&mut s, 2);
        assert_eq!(s, [f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY, 4.0].to_vec());
        let mut s = scores.to_vec();
        top_p(&mut s, 1e-9);
        assert_eq!(
            s.iter().filter(|v| v.is_finite()).count(),
            1,
            "top-p keeps the most likely"
        );
        assert_eq!(s[1], 5.0);
        let mut s = scores.to_vec();
        min_p_filter(&mut s, 1.0);
        assert_eq!(s.iter().filter(|v| v.is_finite()).count(), 1);
        assert_eq!(s[1], 5.0);
        let mut s = scores.to_vec();
        typical(&mut s, 1e-9);
        assert_eq!(s.iter().filter(|v| v.is_finite()).count(), 1);
    }

    #[test]
    fn top_p_keeps_the_smallest_set_reaching_the_mass() {
        // Probabilities 0.1, 0.2, 0.3, 0.4 in token order.
        let scores: Vec<f32> = [0.1f64, 0.2, 0.3, 0.4]
            .iter()
            .map(|p| p.ln() as f32)
            .collect();
        let mut s = scores.clone();
        top_p(&mut s, 0.65);
        // Ascending cumulative 0.1, 0.3, 0.6: the first two stay within 0.35 and are removed.
        assert!(s[0].is_infinite() && s[1].is_infinite());
        assert!(s[2].is_finite() && s[3].is_finite());
    }

    #[test]
    fn a_seed_reproduces_its_draws_and_follows_the_distribution() {
        let scores: Vec<f32> = [0.25f64, 0.75].iter().map(|p| p.ln() as f32).collect();
        let draw = |seed| {
            let mut sampler = Sampler::new(sampling(), seed);
            (0..4000)
                .map(|_| sampler.select(&scores, &[], 0).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(11), draw(11));
        let ones = draw(11).iter().filter(|&&t| t == 1).count();
        assert!((2800..3200).contains(&ones), "{ones} of 4000");
        let mut masked = Sampler::new(sampling(), 3);
        for _ in 0..200 {
            assert_eq!(
                masked
                    .select(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY], &[], 0)
                    .unwrap(),
                1
            );
        }
    }

    #[test]
    fn active_unimplemented_settings_are_refused() {
        let declared = |json: &str| -> Map<String, Value> { serde_json::from_str(json).unwrap() };
        check_declared(&declared(
            r#"{"num_beams":1,"no_repeat_ngram_size":0,"bad_words_ids":null,"suppress_tokens":[],"renormalize_logits":false,"guidance_scale":1.0,"max_new_tokens":17,"hidden_size":8}"#,
        ))
        .unwrap();
        for json in [
            r#"{"num_beams":4}"#,
            r#"{"no_repeat_ngram_size":3}"#,
            r#"{"suppress_tokens":[5]}"#,
            r#"{"forced_bos_token_id":0}"#,
            r#"{"renormalize_logits":true}"#,
            r#"{"max_time":30}"#,
            r#"{"min_new_tokens":2}"#,
        ] {
            let error = check_declared(&declared(json)).unwrap_err();
            assert!(
                matches!(error, GenerationError::UnsupportedSetting { .. }),
                "{json}: {error}"
            );
        }
    }
}
