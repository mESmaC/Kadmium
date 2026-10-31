//! A worker's RAM expert cache, after InferSSD's expert cache
//! (`engine/ggml/src/ggml-ssd/host/expert-cache.h`, its production `count_decay2` profile)
//! and its next-layer prefetch (`cpu-sink.cpp` `prefetch_schedule` / `prefetch_drain`).
//!
//! Experts the router keeps asking for stay resident; the rest stream. Each `(model, revision,
//! layer, expert)` has a selection count that halves every two tokens (a half-life of two, the
//! figure InferSSD settled on for the RAM tier: long enough to keep this turn's experts, short
//! enough that the last turn's go). The clock is the cache's token clock, advanced by
//! [`ExpertCache::begin_token`] whatever layer runs, so a layer that stops running decays like
//! any other and a newcomer never waits behind stale counts. A step of a layer first pins every
//! resident expert it routes to, so a miss at the front of the batch never overwrites a later
//! hit, then admits the misses over the coldest unpinned residents, and only if the newcomer's
//! count is at least the victim's. Pins last until the next step of any layer.
//!
//! Counts are kept in log space, `ls = log2(count) + step / half_life`, which orders experts
//! without decaying anything: a resident's place in the victim index moves only when it is
//! routed to, and the coldest resident of a domain is the first entry of a sorted set. Every
//! lookup is a hash probe and every admission a few ordered-set operations; byte and entry
//! totals are kept incrementally.
//!
//! The cache is sized in bytes: an entry costs what its bytes measure once filled (an estimate
//! until then), and the number of entries is bounded. A prefetch reserves entries of the next
//! layer for the experts a speculative router predicts ([`crate::LayerReference::prefetch_hint`])
//! over residents whose count has decayed below one, and fills them before that layer runs; a
//! reservation the layer does not use is dropped, and one it does use counts as a prefetch hit.
//! The hint only fills entries. It never changes which experts the layer computes.
//!
//! The cache gives bytes back on request ([`ExpertCache::reclaim`]), coldest entries first, and
//! takes them back ([`ExpertCache::grow`]) when its owner finds room again, which is how it
//! registers with the host memory ledger. An online miss-ratio curve (LRU stack distances per
//! layer) says what a larger or smaller cache would have hit, for sizing.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// Locks a shared cache, recovering it if a panic poisoned the lock: a cache whose last holder
/// panicked still describes bytes that are all present or absent, so a heartbeat or the next
/// stage may keep using it rather than panic in turn.
pub fn lock_expert_cache(cache: &Mutex<ExpertCache>) -> MutexGuard<'_, ExpertCache> {
    cache.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A selection count halves every this many tokens.
pub const EXPERT_CACHE_HALF_LIFE_STEPS: f64 = 2.0;
/// Entries the cache holds at most, whatever its byte budget.
pub const DEFAULT_MAX_ENTRIES: usize = 4096;
/// Per-layer quotas the miss-ratio curve is reported at.
pub const MISS_RATIO_QUOTAS: [usize; 17] = [
    1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
];
/// A prefetch may evict a resident whose count has decayed below this.
pub const PREFETCH_VICTIM_MAX_SCORE: f64 = 1.0;
/// Counts below this are forgotten at the next prune.
const PRUNE_BELOW_COUNT: f64 = 1.0 / 64.0;
/// Counts are pruned every this many tokens.
const PRUNE_EVERY_STEPS: u64 = 256;
/// Layers the miss-ratio curve tracks at most, and the depth of each layer's stack.
const MAX_CURVES: usize = 4096;
const MAX_STACK_DEPTH: usize = 512;

/// How the budget is shared between layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertCacheAllocation {
    /// Every layer gets `budget / layers` bytes and `max_entries / layers` entries (InferSSD's
    /// `uniform`). Layers outside `0..layers` bypass the cache.
    Uniform { layers: u32 },
    /// Any layer may use any part of the budget; the victim is the coldest resident of any
    /// layer (InferSSD's `global`), for a worker that learns its layer range from its work.
    Global,
}

/// One cached expert: a model, the revision of its weights, a layer and an expert id. The
/// model term is the FNV-1a 64 of the model id; the revision is whatever fingerprint the owner
/// gives the weights (a checkpoint root, with the vector width folded in), so two models,
/// two checkpoints or two widths never alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExpertKey {
    pub model: u64,
    pub revision: u64,
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub fn new(model_id: &str, revision: u64, layer: u32, expert: u32) -> Self {
        Self {
            model: model_fingerprint(model_id),
            revision,
            layer,
            expert,
        }
    }
}

/// FNV-1a 64 of a model id, the model term of an [`ExpertKey`].
pub fn model_fingerprint(model_id: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in model_id.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// What a step found for one requested expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertLookup {
    /// Resident: read the entry.
    Hit,
    /// Admitted: load the expert and [`ExpertCache::fill`] the entry.
    Admit,
    /// Not admitted: load the expert for this step only.
    Bypass,
}

/// A reserved entry for a predicted expert of a coming layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefetchRequest {
    pub key: ExpertKey,
    pub generation: u64,
}

/// What a prefetch plan decided, per InferSSD's rank accounting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefetchPlan {
    /// Ranks of the hint the plan looked at.
    pub predicted: usize,
    /// Predicted experts not yet resident.
    pub novel: usize,
    /// Predicted experts already resident.
    pub skipped: usize,
    /// Reservations made.
    pub issued: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ExpertCacheStats {
    pub tokens: u64,
    pub steps: u64,
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub bypassed: u64,
    pub admissions: u64,
    pub rejections: u64,
    pub evictions: u64,
    /// Fills that found no room even after evicting, or whose entry was gone.
    pub fills_dropped: u64,
    pub prefetch_predicted: u64,
    pub prefetch_novel: u64,
    pub prefetch_skipped: u64,
    pub prefetch_issued: u64,
    pub prefetch_filled: u64,
    pub prefetch_hits: u64,
    pub prefetch_dropped: u64,
    pub reclaims: u64,
    pub reclaimed_bytes: u64,
    pub entries: usize,
    pub resident_entries: usize,
    pub resident_bytes: u64,
    pub budget_bytes: u64,
    pub configured_bytes: u64,
}

impl ExpertCacheStats {
    /// Hits over the requests the cache served (bypassed requests of unknown layers aside).
    pub fn hit_rate(&self) -> f64 {
        let served = self.hits + self.misses;
        if served == 0 {
            0.0
        } else {
            self.hits as f64 / served as f64
        }
    }

    /// Prefetch hits over prefetches filled: how often a prediction was used.
    pub fn prefetch_precision(&self) -> f64 {
        if self.prefetch_filled == 0 {
            0.0
        } else {
            self.prefetch_hits as f64 / self.prefetch_filled as f64
        }
    }
}

/// A log-space score with a total order, so it can key a sorted set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Ordered(i64);

impl Ordered {
    fn from(ls: f64) -> Self {
        let bits = ls.to_bits() as i64;
        Self(if bits < 0 { bits ^ i64::MAX } else { bits })
    }

    fn ls(self) -> f64 {
        let bits = if self.0 < 0 {
            self.0 ^ i64::MAX
        } else {
            self.0
        };
        f64::from_bits(bits as u64)
    }
}

/// An entry's place in its domain's victim index: coldest first, ties by key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Rank {
    ls: Ordered,
    key: ExpertKey,
}

#[derive(Debug, Clone)]
struct Entry {
    bytes: Option<Arc<Vec<u8>>>,
    /// The entry's cost against the budget: the estimate at admission, the measure once filled.
    len: u64,
    valid: bool,
    protected: bool,
    prefetched: bool,
    reserved: Option<u64>,
    /// The entry's rank in the victim index, `None` while reserved.
    rank: Option<Rank>,
    domain: u32,
}

/// A selection count in log space: `raw` selections as of `step`, each earlier one already
/// decayed to that step.
#[derive(Debug, Clone, Copy)]
struct Count {
    raw: f64,
    step: u64,
}

/// Per-layer LRU stack distances: what a per-layer quota of `q` entries under LRU would have hit.
#[derive(Debug, Clone, Default)]
pub struct MissRatioCurve {
    stacks: BTreeMap<(u64, u32), Vec<u32>>,
    hist: BTreeMap<(u64, u32), Vec<u64>>,
    cold: BTreeMap<(u64, u32), u64>,
}

/// The curve at the reporting quotas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissRatioReport {
    pub quotas: Vec<usize>,
    /// Hit rate over all layers' accesses with that many entries per layer.
    pub hit_rate: Vec<f64>,
    pub accesses: u64,
    pub layers: usize,
}

impl MissRatioCurve {
    /// Records one step of `(model, layer)` asking for `ids`. Tracks at most [`MAX_CURVES`]
    /// layers and [`MAX_STACK_DEPTH`] distinct experts per layer.
    pub fn observe(&mut self, model: u64, layer: u32, ids: &[u32]) {
        let curve = (model, layer);
        if !self.stacks.contains_key(&curve) && self.stacks.len() >= MAX_CURVES {
            return;
        }
        let stack = self.stacks.entry(curve).or_default();
        let hist = self.hist.entry(curve).or_default();
        let cold = self.cold.entry(curve).or_default();
        for id in ids {
            match stack.iter().position(|e| e == id) {
                Some(d) => {
                    if hist.len() <= d {
                        hist.resize(d + 1, 0);
                    }
                    hist[d] += 1;
                }
                None => *cold += 1,
            }
        }
        for id in ids.iter().rev() {
            if let Some(pos) = stack.iter().position(|e| e == id) {
                stack.remove(pos);
            }
            stack.insert(0, *id);
        }
        stack.truncate(MAX_STACK_DEPTH);
    }

    fn hits_at(&self, curve: (u64, u32), quota: usize) -> u64 {
        self.hist
            .get(&curve)
            .map(|h| h.iter().take(quota).sum())
            .unwrap_or(0)
    }

    pub fn accesses(&self) -> u64 {
        self.cold.values().sum::<u64>() + self.hist.values().flatten().sum::<u64>()
    }

    pub fn report(&self) -> MissRatioReport {
        let total = self.accesses();
        let hit_rate = MISS_RATIO_QUOTAS
            .iter()
            .map(|&q| {
                let hits: u64 = self.hist.keys().map(|c| self.hits_at(*c, q)).sum();
                if total == 0 {
                    0.0
                } else {
                    hits as f64 / total as f64
                }
            })
            .collect();
        MissRatioReport {
            quotas: MISS_RATIO_QUOTAS.to_vec(),
            hit_rate,
            accesses: total,
            layers: self.hist.len(),
        }
    }
}

/// The cache.
#[derive(Debug, Clone)]
pub struct ExpertCache {
    configured_bytes: u64,
    budget_bytes: u64,
    max_entries: usize,
    allocation: ExpertCacheAllocation,
    entries: HashMap<ExpertKey, Entry>,
    /// Per domain, every non-reserved entry ordered coldest first.
    index: BTreeMap<u32, BTreeSet<Rank>>,
    domain_bytes: BTreeMap<u32, u64>,
    domain_entries: BTreeMap<u32, usize>,
    resident_entries: usize,
    counts: HashMap<ExpertKey, Count>,
    step: u64,
    last_token_tag: Option<u64>,
    pinned: Vec<ExpertKey>,
    reservations: Vec<ExpertKey>,
    generation: u64,
    stats: ExpertCacheStats,
    mrc: MissRatioCurve,
}

impl ExpertCache {
    /// A cache of `budget_bytes` with at most [`DEFAULT_MAX_ENTRIES`] entries.
    pub fn new(budget_bytes: u64, allocation: ExpertCacheAllocation) -> Self {
        Self::with_max_entries(budget_bytes, allocation, DEFAULT_MAX_ENTRIES)
    }

    pub fn with_max_entries(
        budget_bytes: u64,
        allocation: ExpertCacheAllocation,
        max_entries: usize,
    ) -> Self {
        Self {
            configured_bytes: budget_bytes,
            budget_bytes,
            max_entries: max_entries.max(1),
            allocation,
            entries: HashMap::new(),
            index: BTreeMap::new(),
            domain_bytes: BTreeMap::new(),
            domain_entries: BTreeMap::new(),
            resident_entries: 0,
            counts: HashMap::new(),
            step: 0,
            last_token_tag: None,
            pinned: Vec::new(),
            reservations: Vec::new(),
            generation: 0,
            stats: ExpertCacheStats::default(),
            mrc: MissRatioCurve::default(),
        }
    }

    /// The budget the owner meant to give, which [`ExpertCache::grow`] works back toward when
    /// the cache started under pressure or gave bytes back.
    pub fn with_configured_bytes(mut self, configured_bytes: u64) -> Self {
        self.configured_bytes = configured_bytes.max(self.budget_bytes);
        self
    }

    pub fn allocation(&self) -> ExpertCacheAllocation {
        self.allocation
    }

    /// The budget in force: the configured budget less what was reclaimed and not regrown.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub fn configured_bytes(&self) -> u64 {
        self.configured_bytes
    }

    /// Bytes the budget could take back to reach the configured budget.
    pub fn grow_target_bytes(&self) -> u64 {
        self.configured_bytes.saturating_sub(self.budget_bytes)
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Bytes charged against the budget: filled entries at their measure, reservations and
    /// admissions in flight at their estimate.
    pub fn resident_bytes(&self) -> u64 {
        self.domain_bytes.values().sum()
    }

    /// The token clock.
    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn stats(&self) -> ExpertCacheStats {
        let mut stats = self.stats;
        stats.entries = self.entries.len();
        stats.resident_entries = self.resident_entries;
        stats.resident_bytes = self.resident_bytes();
        stats.budget_bytes = self.budget_bytes;
        stats.configured_bytes = self.configured_bytes;
        stats
    }

    pub fn miss_ratio_curve(&self) -> MissRatioReport {
        self.mrc.report()
    }

    /// Whether the cache has room for `layer` at all.
    pub fn serves_layer(&self, layer: u32) -> bool {
        match self.allocation {
            ExpertCacheAllocation::Uniform { layers } => layer < layers,
            ExpertCacheAllocation::Global => true,
        }
    }

    fn domain_of(&self, layer: u32) -> u32 {
        match self.allocation {
            ExpertCacheAllocation::Uniform { .. } => layer,
            ExpertCacheAllocation::Global => 0,
        }
    }

    /// The byte and entry quotas of a domain.
    fn domain_limits(&self) -> (u64, usize) {
        match self.allocation {
            ExpertCacheAllocation::Uniform { layers } if layers > 0 => (
                self.budget_bytes / layers as u64,
                (self.max_entries / layers as usize).max(1),
            ),
            _ => (self.budget_bytes, self.max_entries),
        }
    }

    /// Advances the token clock, unless `tag` names the token the clock is already on (every
    /// layer of a token calls this with the token's tag; a chain stage calls it with `None`
    /// once per row). Pins of the previous step are released. Every so often, counts that have
    /// decayed to nothing are forgotten.
    pub fn begin_token(&mut self, tag: Option<u64>) {
        if tag.is_some() && tag == self.last_token_tag {
            return;
        }
        self.last_token_tag = tag;
        self.step += 1;
        self.stats.tokens += 1;
        self.release_pins();
        if self.step.is_multiple_of(PRUNE_EVERY_STEPS) {
            let floor = PRUNE_BELOW_COUNT.log2() + self.step as f64 / EXPERT_CACHE_HALF_LIFE_STEPS;
            self.counts.retain(|_, c| Self::ls_of(*c) >= floor);
        }
    }

    fn release_pins(&mut self) {
        for key in self.pinned.drain(..) {
            if let Some(e) = self.entries.get_mut(&key) {
                e.protected = false;
            }
        }
    }

    fn ls_of(count: Count) -> f64 {
        count.raw.max(f64::MIN_POSITIVE).log2() + count.step as f64 / EXPERT_CACHE_HALF_LIFE_STEPS
    }

    fn count_now(&self, key: ExpertKey) -> f64 {
        match self.counts.get(&key) {
            Some(c) => {
                c.raw * (-((self.step - c.step) as f64) / EXPERT_CACHE_HALF_LIFE_STEPS).exp2()
            }
            None => 0.0,
        }
    }

    /// The decayed selection count of an expert: InferSSD's `count_decay2` score.
    pub fn score(&self, key: ExpertKey) -> f64 {
        self.count_now(key)
    }

    /// The log-space score of an expert now, comparable with every rank in the index.
    fn ls_now(&self, key: ExpertKey) -> f64 {
        match self.counts.get(&key) {
            Some(c) => Self::ls_of(*c),
            None => Self::ls_of(Count {
                raw: 0.0,
                step: self.step,
            }),
        }
    }

    /// The decayed count a rank's log-space score means at the current step.
    fn score_of_rank(&self, rank: Rank) -> f64 {
        (rank.ls.ls() - self.step as f64 / EXPERT_CACHE_HALF_LIFE_STEPS).exp2()
    }

    fn bump_count(&mut self, key: ExpertKey) -> f64 {
        let now = self.count_now(key) + 1.0;
        self.counts.insert(
            key,
            Count {
                raw: now,
                step: self.step,
            },
        );
        now
    }

    fn set_rank(&mut self, key: ExpertKey, ls: Option<f64>) {
        let Some(e) = self.entries.get_mut(&key) else {
            return;
        };
        let domain = e.domain;
        if let Some(old) = e.rank.take()
            && let Some(set) = self.index.get_mut(&domain)
        {
            set.remove(&old);
        }
        if let Some(ls) = ls {
            let rank = Rank {
                ls: Ordered::from(ls),
                key,
            };
            e.rank = Some(rank);
            self.index.entry(domain).or_default().insert(rank);
        }
    }

    fn insert_entry(&mut self, key: ExpertKey, entry: Entry) {
        let domain = entry.domain;
        *self.domain_bytes.entry(domain).or_default() += entry.len;
        *self.domain_entries.entry(domain).or_default() += 1;
        if entry.valid {
            self.resident_entries += 1;
        }
        self.entries.insert(key, entry);
    }

    fn remove_entry(&mut self, key: &ExpertKey) -> Option<Entry> {
        let e = self.entries.remove(key)?;
        if let Some(rank) = e.rank
            && let Some(set) = self.index.get_mut(&e.domain)
        {
            set.remove(&rank);
        }
        if let Some(b) = self.domain_bytes.get_mut(&e.domain) {
            *b = b.saturating_sub(e.len);
        }
        if let Some(n) = self.domain_entries.get_mut(&e.domain) {
            *n = n.saturating_sub(1);
        }
        if e.valid {
            self.resident_entries -= 1;
        }
        Some(e)
    }

    fn evict(&mut self, key: &ExpertKey) {
        if let Some(e) = self.remove_entry(key)
            && e.valid
        {
            self.stats.evictions += 1;
        }
    }

    /// The coldest entry of a domain that may be evicted: the first unprotected rank, provided
    /// (when `max_score` is given) a valid one still scores under it. Reserved entries are not
    /// in the index.
    fn victim(&self, domain: u32, max_score: Option<f64>) -> Option<ExpertKey> {
        let set = self.index.get(&domain)?;
        for rank in set {
            let e = &self.entries[&rank.key];
            if e.protected {
                continue;
            }
            if let Some(max) = max_score
                && e.valid
                && self.score_of_rank(*rank) >= max
            {
                return None;
            }
            return Some(rank.key);
        }
        None
    }

    /// Makes room in a domain for `bytes` more (and one more entry when `new_entry`), evicting
    /// victims no hotter than `candidate_ls` and, when `max_score` is given, scoring under it.
    fn make_room(
        &mut self,
        domain: u32,
        bytes: u64,
        new_entry: bool,
        candidate_ls: f64,
        max_score: Option<f64>,
    ) -> bool {
        let (byte_limit, entry_limit) = self.domain_limits();
        if bytes > byte_limit || entry_limit == 0 {
            return false;
        }
        loop {
            let used = self.domain_bytes.get(&domain).copied().unwrap_or(0);
            let count = self.domain_entries.get(&domain).copied().unwrap_or(0);
            if used + bytes <= byte_limit && count + usize::from(new_entry) <= entry_limit {
                return true;
            }
            let Some(victim) = self.victim(domain, max_score) else {
                return false;
            };
            let e = &self.entries[&victim];
            if e.valid
                && let Some(rank) = e.rank
                && candidate_ls < rank.ls.ls()
            {
                return false;
            }
            self.evict(&victim);
        }
    }

    /// One step of `layer` of `model_id` at weights `revision` asking for `requested` experts
    /// (duplicates count once), each expected to measure `expert_bytes`: releases the previous
    /// step's pins, drops the layer's unused reservations, bumps the counts, pins the hits and
    /// admits the misses. Returns one lookup per distinct requested expert, in request order.
    /// Call [`ExpertCache::begin_token`] first on a new token.
    pub fn prepare(
        &mut self,
        model_id: &str,
        revision: u64,
        layer: u32,
        requested: &[u32],
        expert_bytes: u64,
    ) -> Vec<(u32, ExpertLookup)> {
        let model = model_fingerprint(model_id);
        self.release_pins();
        let mut ids: Vec<u32> = Vec::with_capacity(requested.len());
        for id in requested {
            if !ids.contains(id) {
                ids.push(*id);
            }
        }
        self.stats.steps += 1;
        self.stats.requests += ids.len() as u64;
        self.mrc.observe(model, layer, &ids);
        if !self.serves_layer(layer) {
            self.stats.bypassed += ids.len() as u64;
            return ids
                .into_iter()
                .map(|id| (id, ExpertLookup::Bypass))
                .collect();
        }
        let domain = self.domain_of(layer);
        let key_of = |expert: u32| ExpertKey {
            model,
            revision,
            layer,
            expert,
        };

        // Drop the layer's unfilled reservations: a prefetch the step did not land.
        let mut kept = Vec::with_capacity(self.reservations.len());
        for key in std::mem::take(&mut self.reservations) {
            if key.model == model && key.revision == revision && key.layer == layer {
                self.remove_entry(&key);
                self.stats.prefetch_dropped += 1;
            } else {
                kept.push(key);
            }
        }
        self.reservations = kept;

        // Hits first, pinned, their ranks refreshed; misses noted.
        let mut out: Vec<(u32, ExpertLookup)> = Vec::with_capacity(ids.len());
        let mut misses: Vec<usize> = Vec::new();
        for (n, id) in ids.iter().enumerate() {
            let key = key_of(*id);
            self.bump_count(key);
            let ls = self.ls_now(key);
            match self.entries.get_mut(&key).filter(|e| e.valid) {
                Some(e) => {
                    e.protected = true;
                    if e.prefetched {
                        e.prefetched = false;
                        self.stats.prefetch_hits += 1;
                    }
                    self.pinned.push(key);
                    self.set_rank(key, Some(ls));
                    self.stats.hits += 1;
                    out.push((*id, ExpertLookup::Hit));
                }
                None => {
                    misses.push(n);
                    out.push((*id, ExpertLookup::Bypass));
                }
            }
        }

        // Then the misses: room, else the coldest victims if the newcomer is at least as hot.
        for n in misses {
            let key = key_of(ids[n]);
            self.stats.misses += 1;
            let candidate_ls = self.ls_now(key);
            if !self.make_room(domain, expert_bytes, true, candidate_ls, None) {
                self.stats.rejections += 1;
                continue;
            }
            self.insert_entry(
                key,
                Entry {
                    bytes: None,
                    len: expert_bytes,
                    valid: false,
                    protected: true,
                    prefetched: false,
                    reserved: None,
                    rank: None,
                    domain,
                },
            );
            self.set_rank(key, Some(f64::NEG_INFINITY));
            self.pinned.push(key);
            self.stats.admissions += 1;
            out[n].1 = ExpertLookup::Admit;
        }
        out
    }

    /// Lands an admitted or reserved expert's bytes. The entry is charged what the bytes
    /// measure; if that no longer fits beside the other residents, the coldest make way, and if
    /// nothing can, the entry is dropped (the caller keeps its bytes for this step).
    pub fn fill(&mut self, key: ExpertKey, bytes: Vec<u8>) -> bool {
        let Some(e) = self.entries.get(&key) else {
            self.stats.fills_dropped += 1;
            return false;
        };
        if e.valid {
            return true;
        }
        let domain = e.domain;
        let estimate = e.len;
        let measured = bytes.len() as u64;
        if measured > estimate {
            let candidate_ls = self.ls_now(key);
            if !self.make_room(domain, measured - estimate, false, candidate_ls, None) {
                self.remove_entry(&key);
                self.stats.fills_dropped += 1;
                return false;
            }
        }
        let was_reserved = {
            let e = self.entries.get_mut(&key).expect("entry kept");
            let charged = self.domain_bytes.entry(domain).or_default();
            *charged = *charged - estimate + measured;
            e.len = measured;
            e.bytes = Some(Arc::new(bytes));
            e.valid = true;
            let was_reserved = e.reserved.take().is_some();
            self.resident_entries += 1;
            was_reserved
        };
        if was_reserved {
            self.reservations.retain(|k| k != &key);
        }
        let ls = self.ls_now(key);
        self.set_rank(key, Some(ls));
        true
    }

    /// The bytes of a resident entry.
    pub fn bytes(&self, key: ExpertKey) -> Option<Arc<Vec<u8>>> {
        self.entries
            .get(&key)
            .filter(|e| e.valid)
            .and_then(|e| e.bytes.clone())
    }

    /// [`ExpertCache::prepare`] and the loads: every requested expert's bytes, from the cache
    /// when resident, through `load` otherwise (and kept when admitted).
    pub fn fetch(
        &mut self,
        model_id: &str,
        revision: u64,
        layer: u32,
        requested: &[u32],
        expert_bytes: u64,
        mut load: impl FnMut(u32) -> Vec<u8>,
    ) -> Vec<(u32, Arc<Vec<u8>>)> {
        let model = model_fingerprint(model_id);
        let lookups = self.prepare(model_id, revision, layer, requested, expert_bytes);
        lookups
            .into_iter()
            .map(|(expert, lookup)| {
                let key = ExpertKey {
                    model,
                    revision,
                    layer,
                    expert,
                };
                let bytes = match lookup {
                    ExpertLookup::Hit => self.bytes(key).unwrap_or_else(|| Arc::new(load(expert))),
                    ExpertLookup::Admit => {
                        let loaded = load(expert);
                        if self.fill(key, loaded.clone()) {
                            self.bytes(key).unwrap_or_else(|| Arc::new(loaded))
                        } else {
                            Arc::new(loaded)
                        }
                    }
                    ExpertLookup::Bypass => Arc::new(load(expert)),
                };
                (expert, bytes)
            })
            .collect()
    }

    /// Reserves entries of `layer` for the first `depth` experts of `hint` that are not already
    /// resident, over free room or residents whose count has decayed below
    /// [`PREFETCH_VICTIM_MAX_SCORE`]. The caller loads each reservation and hands it back with
    /// [`ExpertCache::prefetch_done`].
    pub fn plan_prefetch(
        &mut self,
        model_id: &str,
        revision: u64,
        layer: u32,
        hint: &[u32],
        depth: usize,
        expert_bytes: u64,
    ) -> (PrefetchPlan, Vec<PrefetchRequest>) {
        let model = model_fingerprint(model_id);
        let mut plan = PrefetchPlan::default();
        let mut requests = Vec::new();
        if !self.serves_layer(layer) {
            return (plan, requests);
        }
        let domain = self.domain_of(layer);
        self.generation += 1;
        let generation = self.generation;
        for &expert in hint.iter().take(depth) {
            plan.predicted += 1;
            let key = ExpertKey {
                model,
                revision,
                layer,
                expert,
            };
            if self.entries.contains_key(&key) {
                plan.skipped += 1;
                continue;
            }
            plan.novel += 1;
            // A prediction is admitted over any resident colder than the threshold, whatever
            // the predicted expert's own count: the point is to land what the next layer is
            // likely to ask for.
            if !self.make_room(
                domain,
                expert_bytes,
                true,
                f64::INFINITY,
                Some(PREFETCH_VICTIM_MAX_SCORE),
            ) {
                continue;
            }
            self.insert_entry(
                key,
                Entry {
                    bytes: None,
                    len: expert_bytes,
                    valid: false,
                    protected: false,
                    prefetched: true,
                    reserved: Some(generation),
                    rank: None,
                    domain,
                },
            );
            self.reservations.push(key);
            plan.issued += 1;
            requests.push(PrefetchRequest { key, generation });
        }
        self.stats.prefetch_predicted += plan.predicted as u64;
        self.stats.prefetch_novel += plan.novel as u64;
        self.stats.prefetch_skipped += plan.skipped as u64;
        self.stats.prefetch_issued += plan.issued as u64;
        (plan, requests)
    }

    /// Lands a prefetched expert; a stale reservation (reclaimed or dropped meanwhile) is
    /// ignored. Returns whether the entry was filled.
    pub fn prefetch_done(&mut self, request: PrefetchRequest, bytes: Vec<u8>) -> bool {
        let current = self
            .entries
            .get(&request.key)
            .is_some_and(|e| e.reserved == Some(request.generation));
        if !current {
            self.stats.prefetch_dropped += 1;
            return false;
        }
        if self.fill(request.key, bytes) {
            if let Some(e) = self.entries.get_mut(&request.key) {
                e.prefetched = true;
            }
            self.stats.prefetch_filled += 1;
            true
        } else {
            self.reservations.retain(|k| k != &request.key);
            self.stats.prefetch_dropped += 1;
            false
        }
    }

    /// Gives at least `bytes` back: the budget drops by that much (never below zero), and the
    /// coldest entries go until the residents fit the smaller budget. Pins do not protect an
    /// entry here. Returns the bytes given back, which the owner takes off the host ledger.
    pub fn reclaim(&mut self, bytes: u64) -> u64 {
        let freed = bytes.min(self.budget_bytes);
        self.budget_bytes -= freed;
        self.shrink_to_budget();
        self.stats.reclaims += 1;
        self.stats.reclaimed_bytes += freed;
        freed
    }

    fn shrink_to_budget(&mut self) {
        let (byte_limit, _) = self.domain_limits();
        let domains: Vec<u32> = self.domain_bytes.keys().copied().collect();
        for domain in domains {
            while self.domain_bytes.get(&domain).copied().unwrap_or(0) > byte_limit {
                let coldest = self
                    .index
                    .get(&domain)
                    .and_then(|set| set.iter().next())
                    .map(|r| r.key);
                match coldest {
                    Some(key) => self.evict(&key),
                    None => break,
                }
            }
        }
    }

    /// Takes up to `bytes` back toward the configured budget (InferSSD's `regrow`). Returns
    /// the bytes the budget grew by, which the owner books on the host ledger.
    pub fn grow(&mut self, bytes: u64) -> u64 {
        let grown = bytes.min(self.grow_target_bytes());
        self.budget_bytes += grown;
        grown
    }

    /// Resident experts of a layer, for a status page.
    pub fn resident_experts(&self, model_id: &str, revision: u64, layer: u32) -> Vec<u32> {
        let model = model_fingerprint(model_id);
        let mut out: Vec<u32> = self
            .entries
            .iter()
            .filter(|(k, e)| {
                e.valid && k.model == model && k.revision == revision && k.layer == layer
            })
            .map(|(k, _)| k.expert)
            .collect();
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: &str = "toy";
    const R: u64 = 7;
    const E: u64 = 64;

    fn bytes_for(layer: u32, expert: u32) -> Vec<u8> {
        vec![(layer * 16 + expert) as u8; E as usize]
    }

    fn key(layer: u32, expert: u32) -> ExpertKey {
        ExpertKey::new(M, R, layer, expert)
    }

    /// One token's step of `layer`.
    fn step(cache: &mut ExpertCache, layer: u32, ids: &[u32]) -> Vec<(u32, ExpertLookup)> {
        cache.begin_token(None);
        cache.prepare(M, R, layer, ids, E)
    }

    fn fetch(cache: &mut ExpertCache, layer: u32, ids: &[u32]) -> Vec<(u32, Arc<Vec<u8>>)> {
        cache.begin_token(None);
        cache.fetch(M, R, layer, ids, E, |e| bytes_for(layer, e))
    }

    #[test]
    fn budgets_are_bytes_and_entries_shared_uniformly_or_globally() {
        let cache =
            ExpertCache::with_max_entries(10 * E, ExpertCacheAllocation::Uniform { layers: 4 }, 8);
        assert_eq!(cache.budget_bytes(), 10 * E);
        assert_eq!(cache.domain_limits(), (2 * E + E / 2, 2));
        assert!(cache.serves_layer(3) && !cache.serves_layer(4));
        let global = ExpertCache::new(10 * E, ExpertCacheAllocation::Global);
        assert!(global.serves_layer(99));
        assert_eq!(global.max_entries(), DEFAULT_MAX_ENTRIES);
        assert_eq!(global.domain_limits(), (10 * E, DEFAULT_MAX_ENTRIES));
        assert_ne!(key(0, 1), ExpertKey::new("other", R, 0, 1));
        assert_ne!(key(0, 1), ExpertKey::new(M, R + 1, 0, 1));
        let under = ExpertCache::new(E, ExpertCacheAllocation::Global).with_configured_bytes(4 * E);
        assert_eq!(under.grow_target_bytes(), 3 * E);
    }

    #[test]
    fn hits_are_pinned_misses_admit_over_the_coldest_and_cold_newcomers_are_rejected() {
        let mut cache = ExpertCache::new(2 * E, ExpertCacheAllocation::Uniform { layers: 1 });
        // Token 1: experts 1 and 2 admitted into the two slots' worth of bytes.
        let got = fetch(&mut cache, 0, &[1, 2, 2]);
        assert_eq!(got.len(), 2, "duplicates count once");
        assert_eq!(got[0].1.as_slice(), bytes_for(0, 1).as_slice());
        let s = cache.stats();
        assert_eq!((s.hits, s.misses, s.admissions), (0, 2, 2));
        assert_eq!(cache.resident_experts(M, R, 0), vec![1, 2]);
        assert_eq!(cache.resident_bytes(), 2 * E);

        // Token 2: 1 hits and is pinned for the step; 3 is a newcomer with count 1 against
        // expert 2's decayed count of 0.707: admitted over 2, never over the pinned 1.
        let lookups = step(&mut cache, 0, &[1, 3]);
        assert_eq!(lookups[0].1, ExpertLookup::Hit);
        assert_eq!(lookups[1].1, ExpertLookup::Admit);
        assert!(cache.fill(key(0, 3), bytes_for(0, 3)));
        assert_eq!(cache.resident_experts(M, R, 0), vec![1, 3]);
        assert_eq!(cache.stats().evictions, 1);
        // The pins go at the next step.
        assert!(cache.entries.values().any(|e| e.protected));
        step(&mut cache, 0, &[1, 3]);
        step(&mut cache, 0, &[1, 3]);
        // Token 5: expert 9 (count 1) asks against residents whose scores are above 1: rejected.
        let lookups = step(&mut cache, 0, &[1, 9]);
        assert_eq!(lookups[1].1, ExpertLookup::Bypass);
        assert_eq!(cache.stats().rejections, 1);
        assert!(cache.score(key(0, 1)) > 1.0);
        // The bypassed expert is still served through `fetch`.
        let got = fetch(&mut cache, 0, &[9]);
        assert_eq!(got[0].1.as_slice(), bytes_for(0, 9).as_slice());

        // A layer outside the uniform range bypasses, and is counted so.
        let other = step(&mut cache, 7, &[1]);
        assert_eq!(other[0].1, ExpertLookup::Bypass);
        assert_eq!(cache.stats().bypassed, 1);
        assert!(cache.stats().hit_rate() > 0.0);
        // Another model's or another revision's expert 1 is a different entry.
        cache.begin_token(None);
        assert_ne!(
            cache.prepare("other", R, 0, &[1], E)[0].1,
            ExpertLookup::Hit
        );
        cache.begin_token(None);
        assert_ne!(cache.prepare(M, R + 1, 0, &[1], E)[0].1, ExpertLookup::Hit);
    }

    #[test]
    fn global_mode_keeps_caching_across_layers_and_stale_layers_decay() {
        let mut cache = ExpertCache::new(4 * E, ExpertCacheAllocation::Global);
        // Layers 0 and 1 warm for 20 tokens.
        for _ in 0..20 {
            cache.begin_token(None);
            cache.fetch(M, R, 0, &[1, 2], E, |e| bytes_for(0, e));
            cache.fetch(M, R, 1, &[1, 2], E, |e| bytes_for(1, e));
        }
        assert_eq!(cache.entry_count(), 4);
        // Layer 2 arrives with the budget full. Its experts start at count 1 against residents
        // at about 3.4; after a few tokens the old layers' counts have halved enough (the clock
        // runs for every layer) and the newcomers get in. Within 8 tokens, not 100.
        let mut first_admitted = None;
        for token in 0..16 {
            cache.begin_token(None);
            let lookups = cache.prepare(M, R, 2, &[5, 6], E);
            if lookups.iter().all(|(_, l)| *l != ExpertLookup::Bypass) && first_admitted.is_none() {
                first_admitted = Some(token);
            }
            for (expert, l) in lookups {
                if l == ExpertLookup::Admit {
                    cache.fill(key(2, expert), bytes_for(2, expert));
                }
            }
        }
        assert!(
            first_admitted.is_some_and(|t| t <= 8),
            "{first_admitted:?} {:?}",
            cache.stats()
        );
        assert_eq!(cache.resident_experts(M, R, 2), vec![5, 6]);
        assert!(cache.entry_count() <= 4);
        // Nothing stays pinned past the next step.
        cache.begin_token(None);
        assert!(cache.entries.values().all(|e| !e.protected));
    }

    #[test]
    fn fills_are_charged_what_they_measure_and_never_exceed_the_budget() {
        let mut cache = ExpertCache::new(8 * 1024, ExpertCacheAllocation::Global);
        // Admitted at a 64-byte estimate, filled with 6 KiB: still fits.
        let lookups = step(&mut cache, 0, &[1]);
        assert_eq!(lookups[0].1, ExpertLookup::Admit);
        assert!(cache.fill(key(0, 1), vec![1u8; 6 * 1024]));
        assert_eq!(cache.resident_bytes(), 6 * 1024);
        // A second 6 KiB fill cannot fit beside it without evicting the first (equal scores:
        // the newcomer, as hot, may replace it; the pin on 1 has gone with the step).
        let lookups = step(&mut cache, 1, &[2]);
        assert_eq!(lookups[0].1, ExpertLookup::Admit);
        assert!(cache.fill(key(1, 2), vec![2u8; 6 * 1024]));
        assert!(cache.resident_bytes() <= 8 * 1024);
        assert_eq!(cache.resident_experts(M, R, 0), Vec::<u32>::new());
        // A fill larger than the whole budget is dropped and the caller keeps the bytes.
        cache.begin_token(None);
        let got = cache.fetch(M, R, 2, &[3], E, |_| vec![3u8; 9 * 1024]);
        assert_eq!(got[0].1.len(), 9 * 1024);
        assert!(cache.stats().fills_dropped >= 1);
        assert!(cache.resident_bytes() <= 8 * 1024);
        assert_eq!(cache.stats().resident_entries, 1);
    }

    #[test]
    fn prefetch_reserves_over_cold_residents_fills_and_counts_hits_and_drops_unused() {
        let mut cache = ExpertCache::new(6 * E, ExpertCacheAllocation::Uniform { layers: 2 });
        fetch(&mut cache, 0, &[1]);
        let (plan, requests) = cache.plan_prefetch(M, R, 1, &[5, 6, 7], 2, E);
        assert_eq!((plan.predicted, plan.novel, plan.issued), (2, 2, 2));
        for r in &requests {
            assert!(cache.prefetch_done(*r, bytes_for(1, r.key.expert)));
        }
        assert_eq!(cache.stats().prefetch_filled, 2);
        let stale = PrefetchRequest {
            generation: 999,
            ..requests[0]
        };
        assert!(!cache.prefetch_done(stale, Vec::new()));
        // Layer 1 routes to 5 (a prefetch hit) and 8 (a miss into the third slot's bytes); 6
        // stays resident, prefetched and unused.
        let got = fetch(&mut cache, 1, &[5, 8]);
        assert_eq!(got.len(), 2);
        assert_eq!(cache.stats().prefetch_hits, 1);
        assert_eq!(cache.stats().prefetch_precision(), 0.5);
        // The layer is full. A new prediction takes a resident whose count has decayed below
        // one (6, never routed), never 5 or 8 at count 1, whatever its own count.
        let (plan, requests) = cache.plan_prefetch(M, R, 1, &[6, 9], 2, E);
        assert_eq!((plan.skipped, plan.novel, plan.issued), (1, 1, 1));
        assert_eq!(cache.resident_experts(M, R, 1), vec![5, 8]);
        assert!(cache.prefetch_done(requests[0], bytes_for(1, 9)));
        assert_eq!(cache.resident_experts(M, R, 1), vec![5, 8, 9]);
        // A full layer whose residents are all hot takes no prediction at all.
        step(&mut cache, 1, &[5, 8, 9]);
        let (plan, requests) = cache.plan_prefetch(M, R, 1, &[10], 1, E);
        assert_eq!((plan.novel, plan.issued), (1, 0));
        assert!(requests.is_empty());
        // A reservation left unfilled is dropped by the next step of its layer.
        let (_, requests) = cache.plan_prefetch(M, R, 0, &[2], 1, E);
        assert_eq!(requests.len(), 1);
        step(&mut cache, 0, &[1]);
        assert!(!cache.prefetch_done(requests[0], Vec::new()));
        assert!(cache.stats().prefetch_dropped >= 2);
    }

    #[test]
    fn reclaim_lowers_the_budget_evicting_the_coldest_and_grow_takes_it_back() {
        let mut cache = ExpertCache::new(6 * E, ExpertCacheAllocation::Uniform { layers: 2 });
        fetch(&mut cache, 0, &[1, 2, 3]);
        fetch(&mut cache, 1, &[4, 5, 6]);
        assert_eq!(cache.resident_bytes(), 6 * E);
        let freed = cache.reclaim(2 * E - 1);
        assert_eq!(freed, 2 * E - 1);
        assert_eq!(cache.budget_bytes(), 4 * E + 1);
        assert_eq!(cache.resident_experts(M, R, 0).len(), 2);
        assert_eq!(cache.resident_experts(M, R, 1).len(), 2);
        assert_eq!(cache.stats().reclaimed_bytes, 2 * E - 1);
        assert_eq!(cache.reclaim(u64::MAX / 2), 4 * E + 1);
        assert_eq!(cache.budget_bytes(), 0);
        assert_eq!(cache.entry_count(), 0);
        assert_eq!(step(&mut cache, 0, &[1])[0].1, ExpertLookup::Bypass);
        assert_eq!(cache.grow_target_bytes(), 6 * E);
        assert_eq!(cache.grow(2 * E), 2 * E);
        assert_eq!(cache.grow(100 * E), 4 * E);
        assert_eq!(cache.grow_target_bytes(), 0);
        assert_eq!(step(&mut cache, 0, &[1])[0].1, ExpertLookup::Admit);
    }

    #[test]
    fn entry_cap_bounds_the_cache_and_counts_are_pruned() {
        let mut cache = ExpertCache::with_max_entries(1 << 30, ExpertCacheAllocation::Global, 3);
        for e in 0..10 {
            fetch(&mut cache, 0, &[e]);
        }
        assert!(cache.entry_count() <= 3);
        assert!(cache.stats().evictions >= 7);
        assert_eq!(cache.counts.len(), 10);
        // After 256 idle tokens every count has decayed to nothing and is forgotten.
        for _ in 0..PRUNE_EVERY_STEPS {
            cache.begin_token(None);
        }
        assert!(cache.counts.is_empty(), "{}", cache.counts.len());
        // A token tag advances the clock once per token however many layers call in.
        let before = cache.step();
        cache.begin_token(Some(42));
        cache.begin_token(Some(42));
        cache.begin_token(Some(43));
        assert_eq!(cache.step(), before + 2);
    }

    #[test]
    fn miss_ratio_curve_reports_what_a_quota_would_have_hit() {
        let mut mrc = MissRatioCurve::default();
        for _ in 0..4 {
            mrc.observe(1, 0, &[1]);
            mrc.observe(1, 0, &[2]);
            mrc.observe(1, 0, &[3]);
        }
        let report = mrc.report();
        assert_eq!(report.accesses, 12);
        assert_eq!(report.quotas[1], 2);
        assert_eq!(report.hit_rate[1], 0.0);
        assert!(
            (report.hit_rate[3] - 9.0 / 12.0).abs() < 1e-9,
            "{:?}",
            report.hit_rate
        );
        // Models do not mix, and a stack never grows past its depth.
        mrc.observe(2, 0, &[1]);
        assert_eq!(mrc.report().layers, 2);
        let wide: Vec<u32> = (0..2000).collect();
        mrc.observe(3, 0, &wide);
        assert_eq!(mrc.stacks[&(3, 0)].len(), MAX_STACK_DEPTH);
        let cache = ExpertCache::new(4096, ExpertCacheAllocation::Global);
        assert_eq!(cache.miss_ratio_curve().accesses, 0);
    }

    #[test]
    fn ordered_scores_sort_like_floats() {
        let xs = [f64::NEG_INFINITY, -3.5, -0.0, 0.0, 1e-9, 2.0, 1e9];
        for w in xs.windows(2) {
            assert!(Ordered::from(w[0]) <= Ordered::from(w[1]), "{:?}", w);
        }
        assert_eq!(Ordered::from(-3.5).ls(), -3.5);
        assert_eq!(Ordered::from(2.0).ls(), 2.0);
    }
}
