//! Streaming the generated text without ever sending text that a later token changes.
//!
//! The tokenizer is not an incremental decoder: a token can complete a multi-byte character, and
//! a stop string can span tokens. So each step decodes the whole sequence, takes the text after
//! the decoded prompt, and sends only the part that cannot change or turn out to be the start of
//! a stop string:
//!
//! - a trailing U+FFFD, which the next byte-level or byte-fallback token may still complete, is
//!   held back;
//! - of the rest, a suffix that is a proper prefix of a stop string is held back until the next
//!   token decides it ([`held_back_len`]; **Proved (model)** that what is sent only grows and
//!   never reaches into a stop string, `Kadmium.Native.Generation.emitted_prefix` and
//!   `Kadmium.Native.Generation.emitLen_le_stop`);
//! - so is a suffix that could begin one of the tokenizer's clean-up replacements, which rewrite
//!   a space before punctuation.
//!
//! If the decoded text ever stops extending what was already sent, the run fails rather than
//! send text that contradicts it.

use super::GenerationError;
use crate::model_text::ModelText;

/// Hugging Face `clean_up_tokenization` patterns. Held back whether or not the tokenizer cleans
/// up: holding back only delays text by one token.
const CLEAN_UP: [&str; 10] = [
    " .", " ?", " !", " ,", " ' ", " n't", " 'm", " 's", " 've", " 're",
];

/// Length of the longest suffix of `text` that is a proper, nonempty prefix of a pattern. Byte
/// level: a pattern's prefix begins with a UTF-8 leading byte, so the cut falls on a character
/// boundary of a valid `text`.
pub fn held_back_len<P: AsRef<[u8]>>(text: &[u8], patterns: &[P]) -> usize {
    let mut longest = 0;
    for pattern in patterns {
        let pattern = pattern.as_ref();
        for k in (longest + 1)..pattern.len() {
            if text.ends_with(&pattern[..k]) {
                longest = k;
            }
        }
    }
    longest
}

/// Earliest occurrence of a stop string, and which; on a tie, the first listed.
fn earliest_stop<'s>(text: &str, stops: &'s [String]) -> Option<(usize, &'s str)> {
    let mut best: Option<(usize, &str)> = None;
    for stop in stops {
        if let Some(offset) = text.find(stop.as_str())
            && best.is_none_or(|(b, _)| offset < b)
        {
            best = Some((offset, stop));
        }
    }
    best
}

pub(crate) struct Step {
    pub delta: String,
    pub stop: Option<String>,
}

pub(crate) struct TextStream<'a> {
    model: &'a ModelText,
    skip_special_tokens: bool,
    stops: &'a [String],
    patterns: Vec<String>,
    ids: Vec<u32>,
    prompt_text: String,
    sent: String,
}

impl<'a> TextStream<'a> {
    pub(crate) fn new(
        model: &'a ModelText,
        prompt: &[u32],
        skip_special_tokens: bool,
        stops: &'a [String],
    ) -> Result<Self, GenerationError> {
        let prompt_text = model.decode(prompt, skip_special_tokens)?;
        let patterns = stops
            .iter()
            .cloned()
            .chain(CLEAN_UP.iter().map(|p| (*p).to_owned()))
            .collect();
        Ok(Self {
            model,
            skip_special_tokens,
            stops,
            patterns,
            ids: prompt.to_vec(),
            prompt_text,
            sent: String::new(),
        })
    }

    /// Text sent so far.
    pub(crate) fn sent(&self) -> &str {
        &self.sent
    }

    fn generated(&self) -> Result<String, GenerationError> {
        let full = self.model.decode(&self.ids, self.skip_special_tokens)?;
        full.strip_prefix(&self.prompt_text)
            .map(str::to_owned)
            .ok_or_else(|| GenerationError::TextUnstable {
                detail: "the decoded sequence no longer begins with the decoded prompt".into(),
            })
    }

    fn advance_to(&mut self, generated: &str, end: usize) -> Result<String, GenerationError> {
        let target = generated
            .get(..end)
            .ok_or_else(|| GenerationError::TextUnstable {
                detail: format!("byte {end} is not a character boundary of the generated text"),
            })?;
        let delta = target
            .strip_prefix(self.sent.as_str())
            .ok_or_else(|| GenerationError::TextUnstable {
                detail: "the generated text no longer begins with the text already sent".into(),
            })?
            .to_owned();
        self.sent = target.to_owned();
        Ok(delta)
    }

    /// Adds a generated token. If it completes a stop string, the text ends before the stop.
    pub(crate) fn push(&mut self, token: u32) -> Result<Step, GenerationError> {
        self.ids.push(token);
        let generated = self.generated()?;
        if let Some((offset, stop)) = earliest_stop(&generated, self.stops) {
            return Ok(Step {
                delta: self.advance_to(&generated, offset)?,
                stop: Some(stop.to_owned()),
            });
        }
        // First the text that can still change, then, of what is left, any start of a pattern:
        // a stop string can continue into a character the next token completes.
        let stable = generated.trim_end_matches('\u{FFFD}');
        let end = stable.len() - held_back_len(stable.as_bytes(), &self.patterns);
        if end <= self.sent.len() {
            if !generated.starts_with(self.sent.as_str()) {
                return Err(GenerationError::TextUnstable {
                    detail: "the generated text no longer begins with the text already sent".into(),
                });
            }
            return Ok(Step {
                delta: String::new(),
                stop: None,
            });
        }
        Ok(Step {
            delta: self.advance_to(&generated, end)?,
            stop: None,
        })
    }

    /// Sends everything held back: the run has ended without a stop string.
    pub(crate) fn finish(&mut self) -> Result<String, GenerationError> {
        let generated = self.generated()?;
        self.advance_to(&generated, generated.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_back_the_longest_partial_pattern() {
        let stops = ["END", "ENOUGH"];
        assert_eq!(held_back_len(b"the E", &stops), 1);
        assert_eq!(held_back_len(b"the EN", &stops), 2);
        assert_eq!(held_back_len(b"the ENOU", &stops), 4);
        assert_eq!(held_back_len(b"the ENX", &stops), 0);
        // A whole stop is not a proper prefix; the caller has already found it.
        assert_eq!(held_back_len(b"the END", &["END"]), 0);
        assert_eq!(held_back_len("é".as_bytes(), &["éa"]), 2);
    }
}
