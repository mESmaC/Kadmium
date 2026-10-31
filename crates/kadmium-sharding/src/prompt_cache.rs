//! A prompt cache across sessions, after InferSSD's server prompt cache
//! (`engine/tools/server/inferssd/prompt-cache.{h,cpp}`): the decoder state a session reached
//! after a prompt is kept, least recently used first, and a new request reuses the cached
//! state that lets it skip the most of its prompt. ARCHITECTURE 9.12 keeps a session's KV pages
//! as catalog content and replays a bounded suffix on resume; this module is the cross-session
//! hit policy it left out: which cached prompt to restore, how much of it, and what to evict.
//!
//! Decisions are InferSSD's:
//! - a cached prompt is reused only beyond what the slot can already reuse on its own;
//! - with a state that can be cut back to any prefix (paged KV), the candidate must keep at
//!   least a quarter of its own prompt (`f_keep >= 0.25`, "don't trash large prompts") and must
//!   beat the current best on both `f_keep` and its similarity to the new prompt;
//! - with a state that can only be restored whole or at a checkpoint (recurrent or compressed
//!   memory), the reuse is the whole prompt when it is a prefix, else the deepest checkpoint
//!   inside the common prefix, and one token of the new prompt is always left to produce logits;
//! - saving drops cached prompts that are prefixes of the new one (they are contained in it);
//! - eviction removes the least recently used entries while the cache is over its byte limit,
//!   always keeping one, then over its token limit, where the token limit is at least the byte
//!   limit divided by the measured bytes per token.
//!
//! Tokens are opaque `u32`s. [`PromptTokens::from_text`] gives a deterministic key for the
//! gateway's text prompts; it is a matching key, never an input to the model.

use serde::{Deserialize, Serialize};

/// A candidate under truncatable memory must keep this share of its own prompt.
pub const PROMPT_CACHE_KEEP_FRACTION: f64 = 0.25;

/// Whether a cached state can be cut back to any prefix, or only restored whole or at a
/// checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptMemoryKind {
    /// Paged KV: any prefix of the cached prompt can be kept.
    Truncatable,
    /// Recurrent or compressed state: the whole prompt or one of its checkpoints.
    Full,
}

/// A partial state kept after the first `n_tokens` of a prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCheckpoint {
    pub n_tokens: u64,
    pub bytes: u64,
}

/// One cached prompt: the tokens a session processed and the size of the state it kept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedPrompt {
    pub key: u64,
    /// The model whose state this is; a prompt is never reused across models.
    #[serde(default)]
    pub model_id: String,
    pub session_id: String,
    pub tokens: Vec<u32>,
    pub state_bytes: u64,
    pub checkpoints: Vec<PromptCheckpoint>,
    pub saved_unix_ms: u64,
    pub last_used_unix_ms: u64,
    pub hits: u64,
}

impl CachedPrompt {
    pub fn n_tokens(&self) -> u64 {
        self.tokens.len() as u64
    }

    pub fn size(&self) -> u64 {
        self.state_bytes + self.checkpoints.iter().map(|c| c.bytes).sum::<u64>()
    }

    /// The deepest checkpoint with at most `limit` tokens.
    fn deepest_checkpoint_within(&self, limit: u64) -> Option<&PromptCheckpoint> {
        self.checkpoints
            .iter()
            .filter(|c| c.bytes > 0 && c.n_tokens <= limit)
            .max_by_key(|c| c.n_tokens)
    }
}

/// What a lookup found: the cached prompt to restore and how much of the new prompt it skips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptReuse {
    #[serde(default)]
    pub model_id: String,
    pub session_id: String,
    pub key: u64,
    /// Tokens of the new prompt the restored state covers.
    pub reuse_tokens: u64,
    /// The checkpoint to apply on top of the whole state, if any (full memory only).
    pub checkpoint_tokens: Option<u64>,
    pub cached_tokens: u64,
    /// Share of the cached prompt kept.
    pub f_keep: f64,
    /// Share of the new prompt covered.
    pub similarity: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCacheStats {
    pub lookups: u64,
    pub hits: u64,
    pub tokens_reused: u64,
    pub saves: u64,
    pub contained: u64,
    pub obsolete_removed: u64,
    pub evictions: u64,
}

/// The cache: entries least recently used first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptCache {
    pub entries: Vec<CachedPrompt>,
    /// In bytes, 0 = no limit.
    pub limit_bytes: u64,
    /// In tokens, 0 = no limit.
    pub limit_tokens: u64,
    pub memory: PromptMemoryKind,
    /// Build, model and settings the states belong to; a different fingerprint empties it.
    pub fingerprint: String,
    pub stats: PromptCacheStats,
}

impl PromptCache {
    pub fn new(limit_bytes: u64, limit_tokens: u64, memory: PromptMemoryKind) -> Self {
        Self {
            entries: Vec::new(),
            limit_bytes,
            limit_tokens,
            memory,
            fingerprint: String::new(),
            stats: PromptCacheStats::default(),
        }
    }

    /// Binds the cache to a fingerprint; entries saved under another are dropped, as a server
    /// deletes prompt files of another build or model.
    pub fn with_fingerprint(mut self, fingerprint: &str) -> Self {
        if self.fingerprint != fingerprint {
            self.entries.clear();
            self.fingerprint = fingerprint.to_string();
        }
        self
    }

    pub fn size(&self) -> u64 {
        self.entries.iter().map(CachedPrompt::size).sum()
    }

    pub fn n_tokens(&self) -> u64 {
        self.entries.iter().map(CachedPrompt::n_tokens).sum()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// FNV-1a 64 over the token ids, byte by byte: InferSSD's prompt file name.
    pub fn prompt_key(tokens: &[u32]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for token in tokens {
            for byte in token.to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash
    }

    /// Length of the common prefix of two token sequences.
    pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    /// For a state that cannot be cut back: how many tokens of `tokens_new` a cached prompt
    /// lets a slot skip, restoring its whole state (when that is a prefix of the new prompt) or
    /// else its deepest checkpoint inside the common prefix, always leaving a token of the new
    /// prompt to process for the logits; `None` if neither fits.
    pub fn reuse_full(prompt: &CachedPrompt, tokens_new: &[u32]) -> Option<(u64, Option<u64>)> {
        let lcp = Self::common_prefix(&prompt.tokens, tokens_new) as u64;
        let limit = lcp.min((tokens_new.len() as u64).saturating_sub(1));
        if prompt.n_tokens() <= limit {
            return Some((prompt.n_tokens(), None));
        }
        prompt
            .deepest_checkpoint_within(limit)
            .map(|c| (c.n_tokens, Some(c.n_tokens)))
    }

    /// The cached state of `model_id` that lets a request skip the most of `tokens_new`, if
    /// that is more than `n_reuse_slot`, the tokens the slot can already reuse. A hit moves the
    /// entry to most recently used.
    pub fn lookup(
        &mut self,
        model_id: &str,
        tokens_new: &[u32],
        n_reuse_slot: u64,
        now: u64,
    ) -> Option<PromptReuse> {
        self.stats.lookups += 1;
        let mut best: Option<(usize, PromptReuse)> = None;
        for (index, prompt) in self.entries.iter().enumerate() {
            if prompt.model_id != model_id {
                continue;
            }
            let candidate = match self.memory {
                PromptMemoryKind::Full => {
                    let Some((reuse, checkpoint)) = Self::reuse_full(prompt, tokens_new) else {
                        continue;
                    };
                    if reuse <= n_reuse_slot {
                        continue;
                    }
                    let better = best.as_ref().is_none_or(|(_, b)| reuse > b.reuse_tokens);
                    if !better {
                        continue;
                    }
                    PromptReuse {
                        model_id: prompt.model_id.clone(),
                        session_id: prompt.session_id.clone(),
                        key: prompt.key,
                        reuse_tokens: reuse,
                        checkpoint_tokens: checkpoint,
                        cached_tokens: prompt.n_tokens(),
                        f_keep: reuse as f64 / prompt.n_tokens().max(1) as f64,
                        similarity: reuse as f64 / tokens_new.len().max(1) as f64,
                    }
                }
                PromptMemoryKind::Truncatable => {
                    let lcp = Self::common_prefix(&prompt.tokens, tokens_new) as u64;
                    // Leave a token for the logits.
                    let reuse = lcp.min((tokens_new.len() as u64).saturating_sub(1));
                    if reuse <= n_reuse_slot {
                        continue;
                    }
                    let f_keep = reuse as f64 / prompt.n_tokens().max(1) as f64;
                    let similarity = reuse as f64 / tokens_new.len().max(1) as f64;
                    if f_keep < PROMPT_CACHE_KEEP_FRACTION {
                        continue;
                    }
                    let better = best
                        .as_ref()
                        .is_none_or(|(_, b)| f_keep > b.f_keep && similarity > b.similarity);
                    if !better {
                        continue;
                    }
                    PromptReuse {
                        model_id: prompt.model_id.clone(),
                        session_id: prompt.session_id.clone(),
                        key: prompt.key,
                        reuse_tokens: reuse,
                        checkpoint_tokens: None,
                        cached_tokens: prompt.n_tokens(),
                        f_keep,
                        similarity,
                    }
                }
            };
            best = Some((index, candidate));
        }
        let (index, reuse) = best?;
        let mut entry = self.entries.remove(index);
        entry.last_used_unix_ms = entry.last_used_unix_ms.max(now);
        entry.hits += 1;
        self.entries.push(entry);
        self.stats.hits += 1;
        self.stats.tokens_reused += reuse.reuse_tokens;
        Some(reuse)
    }

    /// Whether a cached prompt already restores `tokens` exactly (whole or at a checkpoint
    /// under full memory, any prefix under truncatable memory). A match moves to most recently
    /// used.
    fn contains(&mut self, model_id: &str, tokens: &[u32], now: u64) -> bool {
        let n = tokens.len() as u64;
        let found = self.entries.iter().position(|p| {
            if p.model_id != model_id {
                return false;
            }
            let lcp = Self::common_prefix(&p.tokens, tokens) as u64;
            if lcp < n {
                return false;
            }
            match self.memory {
                PromptMemoryKind::Truncatable => true,
                PromptMemoryKind::Full => {
                    p.n_tokens() == n
                        || p.checkpoints.iter().any(|c| c.bytes > 0 && c.n_tokens == n)
                }
            }
        });
        match found {
            Some(index) => {
                let mut entry = self.entries.remove(index);
                entry.last_used_unix_ms = entry.last_used_unix_ms.max(now);
                self.entries.push(entry);
                true
            }
            None => false,
        }
    }

    /// Saves the state a session reached after `tokens`, unless the cache can already restore
    /// it. Under truncatable memory the cached prompts that are prefixes of this one are
    /// removed first: the new state contains them. Returns whether an entry was added.
    pub fn save(
        &mut self,
        model_id: &str,
        session_id: &str,
        tokens: &[u32],
        state_bytes: u64,
        checkpoints: Vec<PromptCheckpoint>,
        now: u64,
    ) -> bool {
        if tokens.is_empty() {
            return false;
        }
        if self.contains(model_id, tokens, now) {
            self.stats.contained += 1;
            return false;
        }
        if self.memory == PromptMemoryKind::Truncatable {
            let before = self.entries.len();
            self.entries.retain(|p| {
                let lcp = Self::common_prefix(&p.tokens, tokens);
                p.model_id != model_id || lcp < p.tokens.len() || p.tokens.len() >= tokens.len()
            });
            self.stats.obsolete_removed += (before - self.entries.len()) as u64;
        }
        self.entries.push(CachedPrompt {
            key: Self::prompt_key(tokens),
            model_id: model_id.to_string(),
            session_id: session_id.to_string(),
            tokens: tokens.to_vec(),
            state_bytes,
            checkpoints,
            saved_unix_ms: now,
            last_used_unix_ms: now,
            hits: 0,
        });
        self.stats.saves += 1;
        self.update();
        true
    }

    /// Evicts least recently used entries over the byte limit, always keeping one, then over
    /// the token limit, which is at least the byte limit's worth of tokens at the measured
    /// bytes per token.
    pub fn update(&mut self) {
        if self.limit_bytes > 0 {
            while self.entries.len() > 1 && self.size() > self.limit_bytes {
                self.entries.remove(0);
                self.stats.evictions += 1;
            }
        }
        if self.limit_tokens > 0 {
            let size_per_token = (self.size() / self.n_tokens().max(1)).max(1);
            let limit_tokens_cur = self.limit_tokens.max(self.limit_bytes / size_per_token);
            while self.entries.len() > 1 && self.n_tokens() > limit_tokens_cur {
                self.entries.remove(0);
                self.stats.evictions += 1;
            }
        }
    }

    /// Drops every cached prompt of a session (its pages are gone).
    pub fn forget_session(&mut self, session_id: &str) -> usize {
        let before = self.entries.len();
        self.entries.retain(|p| p.session_id != session_id);
        before - self.entries.len()
    }

    /// The most recently used entries first, for a status page.
    pub fn recent(&self, limit: usize) -> Vec<&CachedPrompt> {
        self.entries.iter().rev().take(limit).collect()
    }
}

/// Deterministic token ids for text prompts: one id per whitespace-separated word, FNV-1a over
/// its bytes. A matching key for the prompt cache, never an input to any model.
pub struct PromptTokens;

impl PromptTokens {
    pub fn from_text(text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| {
                let mut hash: u32 = 0x811c9dc5;
                for byte in word.as_bytes() {
                    hash ^= *byte as u32;
                    hash = hash.wrapping_mul(0x01000193);
                }
                hash
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: u32) -> Vec<u32> {
        (1..=n).collect()
    }

    #[test]
    fn truncatable_lookup_prefers_a_candidate_better_on_both_keep_and_similarity() {
        let mut cache = PromptCache::new(0, 0, PromptMemoryKind::Truncatable);
        assert!(cache.save("m", "short", &toks(10), 1000, Vec::new(), 1));
        assert!(cache.save("m", "long", &toks(100), 10_000, Vec::new(), 2));
        // "short" was a prefix of "long": removed as obsolete.
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats.obsolete_removed, 1);
        let mut other = toks(50);
        other.push(999);
        assert!(cache.save("m", "other", &other, 5_000, Vec::new(), 3));
        assert_eq!(cache.len(), 2);

        // A new prompt of 60 tokens sharing 50 with both: "other" keeps 50/51 of itself, "long"
        // only 50/100 (both over a quarter); "other" wins on f_keep with equal similarity only
        // if it also beats on similarity, so the first candidate seen ("long") stays unless
        // "other" is better on both. Similarity is equal, so "long" stays.
        let mut new = toks(50);
        new.extend([7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let hit = cache.lookup("m", &new, 0, 10).unwrap();
        assert_eq!(hit.reuse_tokens, 50);
        assert_eq!(hit.session_id, "long");
        // The hit is now most recently used.
        assert_eq!(cache.entries.last().unwrap().session_id, "long");
        assert_eq!(cache.stats.hits, 1);

        // A prompt that would keep under a quarter of a large cached prompt is not reused.
        let mut tiny = toks(20);
        tiny.push(555);
        let mut big_only = PromptCache::new(0, 0, PromptMemoryKind::Truncatable);
        big_only.save("m", "big", &toks(100), 10_000, Vec::new(), 1);
        assert!(big_only.lookup("m", &tiny, 0, 2).is_none());

        // Nothing beyond what the slot already reuses.
        assert!(cache.lookup("m", &new, 50, 11).is_none());
        // One token is always left for the logits.
        let exact = toks(100);
        let hit = cache.lookup("m", &exact, 0, 12).unwrap();
        assert_eq!(hit.reuse_tokens, 99);
    }

    #[test]
    fn full_memory_restores_whole_prompts_or_the_deepest_checkpoint() {
        let mut cache = PromptCache::new(0, 0, PromptMemoryKind::Full);
        let checkpoints = vec![
            PromptCheckpoint {
                n_tokens: 20,
                bytes: 200,
            },
            PromptCheckpoint {
                n_tokens: 60,
                bytes: 600,
            },
        ];
        cache.save("m", "a", &toks(100), 10_000, checkpoints, 1);
        assert_eq!(cache.size(), 10_800);
        // A prompt extending the cached one: the whole state.
        let hit = cache.lookup("m", &toks(120), 0, 2).unwrap();
        assert_eq!((hit.reuse_tokens, hit.checkpoint_tokens), (100, None));
        // A prompt diverging after 70 tokens: the checkpoint at 60.
        let mut diverged = toks(70);
        diverged.extend([900, 901]);
        let hit = cache.lookup("m", &diverged, 0, 3).unwrap();
        assert_eq!((hit.reuse_tokens, hit.checkpoint_tokens), (60, Some(60)));
        // Diverging before any checkpoint: nothing.
        let mut early = toks(10);
        early.push(1);
        assert!(cache.lookup("m", &early, 0, 4).is_none());
        // The same prompt again is contained, not saved twice; a checkpointed prefix too.
        assert!(!cache.save("m", "a", &toks(100), 10_000, Vec::new(), 5));
        assert!(!cache.save("m", "b", &toks(60), 600, Vec::new(), 6));
        assert_eq!(cache.stats.contained, 2);
        assert_eq!(cache.len(), 1);
        // Another model's identical prompt is neither contained nor reused.
        assert!(cache.save("other", "c", &toks(100), 10_000, Vec::new(), 7));
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup("third", &toks(120), 0, 8).is_none());
        assert_eq!(
            cache.lookup("other", &toks(120), 0, 9).unwrap().model_id,
            "other"
        );
    }

    #[test]
    fn eviction_follows_the_byte_limit_then_the_token_limit_and_keeps_one() {
        let mut cache = PromptCache::new(2_500, 0, PromptMemoryKind::Full);
        for (i, n) in [10u32, 20, 30].iter().enumerate() {
            let mut t = toks(*n);
            t.push(1000 + i as u32);
            cache.save("m", &format!("s{i}"), &t, 1_000, Vec::new(), i as u64);
        }
        // Three entries of 1000 bytes over a 2500 limit: the oldest goes.
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.entries[0].session_id, "s1");
        assert_eq!(cache.stats.evictions, 1);

        let mut tokens_limited = PromptCache::new(0, 25, PromptMemoryKind::Full);
        tokens_limited.save("m", "x", &toks(20), 100, Vec::new(), 1);
        let mut y = toks(10);
        y.push(77);
        tokens_limited.save("m", "y", &y, 100, Vec::new(), 2);
        assert_eq!(tokens_limited.len(), 1, "31 tokens over a limit of 25");
        assert_eq!(tokens_limited.entries[0].session_id, "y");

        // A tiny byte limit still keeps one entry.
        let mut one = PromptCache::new(1, 0, PromptMemoryKind::Full);
        one.save("m", "only", &toks(5), 10_000, Vec::new(), 1);
        assert_eq!(one.len(), 1);
        assert_eq!(one.forget_session("only"), 1);
        assert!(one.is_empty());
    }

    #[test]
    fn keys_fingerprints_and_text_tokens_are_deterministic() {
        assert_eq!(
            PromptCache::prompt_key(&[1, 2, 3]),
            PromptCache::prompt_key(&[1, 2, 3])
        );
        assert_ne!(
            PromptCache::prompt_key(&[1, 2, 3]),
            PromptCache::prompt_key(&[1, 2, 4])
        );
        let a = PromptTokens::from_text("the quick  brown fox");
        let b = PromptTokens::from_text("the quick brown fox jumps");
        assert_eq!(a.len(), 4);
        assert_eq!(PromptCache::common_prefix(&a, &b), 4);
        let mut cache =
            PromptCache::new(0, 0, PromptMemoryKind::Truncatable).with_fingerprint("v1");
        cache.save("m", "s", &a, 10, Vec::new(), 1);
        let same = cache.clone().with_fingerprint("v1");
        assert_eq!(same.len(), 1);
        let other = cache.with_fingerprint("v2");
        assert!(other.is_empty());
    }
}
