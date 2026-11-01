/-!
# Erasure-coded bucket store bounds, shard sizing, and crash-safe overwrite durability

Models the shard-count validation, storage amplification bound, shard-size covering property,
and crash-recovery directory state machine of `crates/kadmium-data/src/storage_pool.rs`:

- `validateShardCounts` and `validateStoredShardCounts` model
  `DecentralizedStoragePoolEngine::validate_shard_counts` and
  `DecentralizedStoragePoolEngine::validate_stored_shard_counts`.
- `storedShardOverhead_le_three` proves that for every accepted `(k, m)` configuration,
  total stored shards `k + m` satisfy `k + m ≤ 3 * k` (at most `3x` physical storage overhead).
- `cauchyNodes_disjoint` proves that for every accepted `(k, m)` configuration, the Cauchy
  row evaluation points `x_p = k + p` (`p < m`) and column evaluation points `y_j = j` (`j < k`)
  lie in `0 .. 256` and are pairwise distinct (`x_p ≠ y_j`).
- `shardSize` models the per-shard byte size `payload.len().div_ceil(k).max(1)` in
  `encode_erasure_coded_shards`, and `payloadLen_le_k_mul_shardSize` proves that `k` shards of
  size `shardSize payloadLen k` cover every byte of `payload` without truncation.
- `SwapState`, `SwapEvent`, `recoverAndCleanup`, and `swapStep` model the directory rotation and
  startup/open recovery protocol in `BucketObjectStore::put_bytes` and
  `BucketObjectStore::recover_interrupted_writes`.
  `overwrite_durable` and `recover_restores_final` prove that across every sequence of
  staging, rotation, commit, rollback, crash, and recovery steps, a committed object is never
  lost, and `recoverAndCleanup` always leaves `objects/<dir>` populated (`final ≠ none`).

Modelling choices:
- Counts and payload lengths are unbounded `Nat`; `validateStoredShardCounts` bounds `k ≤ 16`
  and `m ≤ 8`.
- Directory states for a single object key `<dir>` are represented by `Option Nat` version tags
  for `objects/<dir>` (`final`), `objects/.old-<dir>-*` (`old`), and
  `objects/.staging-<dir>-*` (`staging`). Cross-process serialization via `<root>/.store.lock`
  means steps execute sequentially on `SwapState`.
-/

namespace Kadmium.StoragePool

/-- Error variants returned by `validateShardCounts` and `validateStoredShardCounts`. -/
inductive ShardCountError
  | invalidShardCounts
  | invalidStoredBounds
  | parityExceedsCap
  deriving DecidableEq, Repr

/-- Models `DecentralizedStoragePoolEngine::validate_shard_counts(k, m)`. -/
def validateShardCounts (k m : Nat) : Except ShardCountError Nat :=
  if k = 0 ∨ m = 0 ∨ k + m > 256 then
    .error .invalidShardCounts
  else
    .ok (k + m)

/-- Models `DecentralizedStoragePoolEngine::validate_stored_shard_counts(k, m)`. -/
def validateStoredShardCounts (k m : Nat) : Except ShardCountError Nat :=
  if k = 0 ∨ m = 0 ∨ k + m > 256 then
    .error .invalidShardCounts
  else if k > 16 ∨ m > 8 ∨ k + m > 20 then
    .error .invalidStoredBounds
  else if m > Nat.max k 2 then
    .error .parityExceedsCap
  else
    .ok (k + m)

/-- `validateShardCounts` succeeds if and only if `1 ≤ k`, `1 ≤ m`, and `k + m ≤ 256`. -/
theorem validateShardCounts_spec (k m : Nat) :
    validateShardCounts k m = .ok (k + m) ↔ (1 ≤ k ∧ 1 ≤ m ∧ k + m ≤ 256) := by
  unfold validateShardCounts
  split <;> simp <;> omega

/-- `validateStoredShardCounts` succeeds if and only if `1 ≤ k ≤ 16`, `1 ≤ m ≤ 8`,
`k + m ≤ 20`, and `m ≤ max k 2`. -/
theorem validateStoredShardCounts_spec (k m : Nat) :
    validateStoredShardCounts k m = .ok (k + m) ↔
      (1 ≤ k ∧ k ≤ 16 ∧ 1 ≤ m ∧ m ≤ 8 ∧ k + m ≤ 20 ∧ m ≤ Nat.max k 2) := by
  unfold validateStoredShardCounts
  split
  · simp; omega
  · split
    · simp; omega
    · split <;> simp <;> omega

/-- Every `(k, m)` accepted by `validateStoredShardCounts` has total shards `k + m ≤ 3 * k`,
establishing the `3x` physical storage amplification bound `(k + m) / k ≤ 3`. -/
theorem storedShardOverhead_le_three {k m total : Nat}
    (h : validateStoredShardCounts k m = .ok total) :
    total = k + m ∧ 1 ≤ k ∧ total ≤ 3 * k := by
  unfold validateStoredShardCounts at h
  split at h
  · contradiction
  · split at h
    · contradiction
    · split at h
      · contradiction
      · injection h with htot
        simp only [Nat.max_def] at *
        split at * <;> omega


/-- For every `(k, m)` accepted by `validateStoredShardCounts`, every Cauchy parity row point
`x_p = k + p` (`p < m`) and data column point `y_j = j` (`j < k`) are distinct and fit in `UInt8`. -/
theorem cauchyNodes_disjoint {k m total p j : Nat}
    (h : validateStoredShardCounts k m = .ok total) (hp : p < m) (hj : j < k) :
    k + p ≠ j ∧ k + p < 256 ∧ j < 256 := by
  unfold validateStoredShardCounts at h
  split at h
  · contradiction
  · split at h
    · contradiction
    · split at h
      · contradiction
      · omega

/-- Models `shard_size` in `encode_erasure_coded_shards`:
`if payload.is_empty() { 1 } else { payload.len().div_ceil(k).max(1) }`. -/
def shardSize (payloadLen k : Nat) : Nat :=
  if payloadLen = 0 then 1 else Nat.max ((payloadLen + k - 1) / k) 1

/-- Every data shard has positive byte length (`1 ≤ shardSize payloadLen k`). -/
theorem shardSize_pos (payloadLen k : Nat) : 1 ≤ shardSize payloadLen k := by
  unfold shardSize
  split
  · omega
  · exact Nat.le_max_right _ _

/-- `k` equal-sized data shards of size `shardSize payloadLen k` cover all `payloadLen` bytes
without truncation (`payloadLen ≤ k * shardSize payloadLen k`). -/
theorem payloadLen_le_k_mul_shardSize (payloadLen k : Nat) (hk : 1 ≤ k) :
    payloadLen ≤ k * shardSize payloadLen k := by
  unfold shardSize
  split
  · omega
  · have hge : (payloadLen + k - 1) / k ≤ Nat.max ((payloadLen + k - 1) / k) 1 :=
      Nat.le_max_left _ _
    have hdiv := Nat.div_add_mod (payloadLen + k - 1) k
    have hmod : (payloadLen + k - 1) % k < k := Nat.mod_lt _ (by omega)
    have hmul : k * ((payloadLen + k - 1) / k) ≤ k * Nat.max ((payloadLen + k - 1) / k) 1 :=
      Nat.mul_le_mul_left k hge
    omega

/-- Models the store-wide object count and physical byte quota check in
`BucketObjectStore::put_bytes` (`other_objects_count < max_objects` and
`other_physical_bytes + new_physical_bytes ≤ max_total`). -/
def checkStoreQuota (otherObjects maxObjects otherBytes newBytes maxTotalBytes : Nat) : Bool :=
  decide (otherObjects < maxObjects ∧ otherBytes + newBytes ≤ maxTotalBytes)

/-- `checkStoreQuota` accepts a write if and only if the resulting object count is at most
`maxObjects` (`otherObjects + 1 ≤ maxObjects`) and the resulting physical byte footprint is at
most `maxTotalBytes` (`otherBytes + newBytes ≤ maxTotalBytes`). -/
theorem checkStoreQuota_spec (otherObjects maxObjects otherBytes newBytes maxTotalBytes : Nat) :
    checkStoreQuota otherObjects maxObjects otherBytes newBytes maxTotalBytes = true ↔
      (otherObjects + 1 ≤ maxObjects ∧ otherBytes + newBytes ≤ maxTotalBytes) := by
  unfold checkStoreQuota
  simp
  omega



/-- On-disk directory state for a single object key `<dir>` in `BucketObjectStore`:
- `final`: committed object version at `objects/<dir>`
- `old`: backup version at `objects/.old-<dir>-*`
- `staging`: uncommitted version at `objects/.staging-<dir>-*` -/
structure SwapState where
  final : Option Nat
  old : Option Nat
  staging : Option Nat
  deriving DecidableEq, Repr

/-- Events in the `BucketObjectStore` overwrite and crash-recovery protocol. -/
inductive SwapEvent
  | stage (v : Nat)
  | rotate
  | commit
  | rollback
  | crash
  | recoverAndCleanup
  deriving DecidableEq, Repr

/-- Models `BucketObjectStore::recover_interrupted_writes`:
if `final` (`objects/<dir>`) is absent and `old` (`objects/.old-<dir>-*`) is present,
renames `old` back to `final`; otherwise removes superseded `old` and stale `staging`. -/
def recoverAndCleanup (s : SwapState) : SwapState :=
  match s.final, s.old with
  | none, some v => { final := some v, old := none, staging := none }
  | final, _ => { final := final, old := none, staging := none }

/-- Single transition of the directory swap state machine under the cross-process store lock. -/
def swapStep (s : SwapState) : SwapEvent → SwapState
  | .stage v => { s with staging := some v }
  | .rotate =>
    match s.final, s.staging with
    | some v, some stg => { final := none, old := some v, staging := some stg }
    | _, _ => s
  | .commit =>
    match s.final, s.staging with
    | none, some stg => { final := some stg, old := none, staging := none }
    | _, _ => s
  | .rollback =>
    match s.final, s.old with
    | none, some v => { final := some v, old := none, staging := none }
    | _, _ => { s with staging := none }
  | .crash => s
  | .recoverAndCleanup => recoverAndCleanup s

/-- Executes a trace of `SwapEvent` transitions from initial state `s`. -/
def runSwap (s : SwapState) (es : List SwapEvent) : SwapState :=
  es.foldl swapStep s

/-- Durability invariant: a committed object version is preserved in `final` or `old`. -/
def Durable (s : SwapState) : Prop :=
  s.final ≠ none ∨ s.old ≠ none

/-- Every single step (`stage`, `rotate`, `commit`, `rollback`, `crash`, `recoverAndCleanup`)
preserves `Durable`. -/
theorem swapStep_preserves_durable {s : SwapState} (h : Durable s) (e : SwapEvent) :
    Durable (swapStep s e) := by
  rcases s with ⟨fin, old, stg⟩
  cases e with
  | stage v =>
    exact h
  | rotate =>
    cases fin <;> cases stg <;> simp [swapStep, Durable] at h ⊢ <;> assumption
  | commit =>
    cases fin <;> cases stg <;> simp [swapStep, Durable] at h ⊢ <;> assumption
  | rollback =>
    cases fin <;> cases old <;> simp [swapStep, Durable] at h ⊢
  | crash =>
    exact h
  | recoverAndCleanup =>
    cases fin <;> cases old <;> simp [swapStep, recoverAndCleanup, Durable] at h ⊢

/-- Any finite sequence of overwrite, crash, and recovery events preserves `Durable`. -/
theorem runSwap_preserves_durable {s : SwapState} (h : Durable s) (es : List SwapEvent) :
    Durable (runSwap s es) := by
  induction es generalizing s with
  | nil => exact h
  | cons e rest ih =>
    exact ih (swapStep_preserves_durable h e)

/-- Starting from any committed object version `v` in `objects/<dir>`, the object remains durable
across every sequence of overwrite stages, rotations, commits, rollbacks, crashes, and cleanups. -/
theorem overwrite_durable (v : Nat) (es : List SwapEvent) :
    Durable (runSwap ⟨some v, none, none⟩ es) :=
  runSwap_preserves_durable (Or.inl (Option.some_ne_none v)) es

/-- Running `recoverAndCleanup` on any durable state restores `final` (`objects/<dir>`) so the
committed object is directly readable on disk (`final ≠ none`). -/
theorem recover_restores_final {s : SwapState} (h : Durable s) :
    (recoverAndCleanup s).final ≠ none := by
  rcases s with ⟨fin, old, stg⟩
  cases fin <;> cases old <;> simp [recoverAndCleanup, Durable] at h ⊢

/-- If an overwrite of committed version `v` by staged version `vNew` is interrupted right after
`rotate`, the next store operation's `recoverAndCleanup` restores `some v` to `final`. -/
theorem rotate_then_recover_restores_original (v vNew : Nat) :
    runSwap ⟨some v, none, some vNew⟩ [.rotate, .recoverAndCleanup] = ⟨some v, none, none⟩ := by
  rfl

/-- Multiplication in `GF(2^8)` with irreducible polynomial `x^8 + x^4 + x^3 + x^2 + 1` (`0x11d`). -/
def gf256MulStep (a b p : UInt8) : UInt8 × UInt8 × UInt8 :=
  let p' := if (b &&& 1) != 0 then p ^^^ a else p
  let hi := (a &&& 0x80) != 0
  let aShift := a <<< 1
  let a' := if hi then aShift ^^^ 0x1d else aShift
  let b' := b >>> 1
  (a', b', p')

def gf256MulLoop : Nat → UInt8 → UInt8 → UInt8 → UInt8
  | 0, _, _, p => p
  | n + 1, a, b, p =>
    let (a', b', p') := gf256MulStep a b p
    gf256MulLoop n a' b' p'

def gf256Mul (a b : UInt8) : UInt8 :=
  gf256MulLoop 8 a b 0

def gf256PowLoop : Nat → UInt8 → UInt8 → UInt8 → UInt8
  | 0, _, _, res => res
  | n + 1, base, exp, res =>
    if exp == 0 then res
    else
      let res' := if (exp &&& 1) != 0 then gf256Mul res base else res
      let base' := gf256Mul base base
      gf256PowLoop n base' (exp >>> 1) res'

/-- Multiplicative inverse in `GF(2^8)` via `a^254`. -/
def gf256Inv (a : UInt8) : UInt8 :=
  if a == 0 then 0 else gf256PowLoop 8 a 254 1

/-- Systematic Cauchy generator row for `shardIdx` and `k` data shards over `GF(2^8)`. -/
def systematicCauchyRow (shardIdx k : Nat) : List UInt8 :=
  if shardIdx < k then
    (List.range k).map (fun j => if j == shardIdx then 1 else 0)
  else
    let p := shardIdx - k
    let xp : UInt8 := (k + p).toUInt8
    (List.range k).map (fun j => gf256Inv (xp ^^^ j.toUInt8))

/-- **Conjecture**: Every `k × k` submatrix of the `(k + m) × k` systematic Cauchy matrix over
`GF(2^8)` (for `(k, m)` accepted by `validateStoredShardCounts`) has a non-zero pivot at every
Gaussian elimination column, so any `k` surviving shards out of `k + m` reconstruct the original
`k` data shards bit for bit. -/
def CauchyMdsConjecture : Prop :=
  ∀ (k m total : Nat),
    validateStoredShardCounts k m = .ok total →
    ∀ (chosen : List Nat),
      chosen.length = k →
      chosen.Nodup →
      (∀ idx ∈ chosen, idx < total) →
      ∃ (inv : List (List UInt8)),
        inv.length = k ∧
        ∀ (i j : Nat), i < k → j < k →
          ((List.range k).foldl
            (fun acc r =>
              let invEntry := (inv.getD i []).getD r 0
              let matEntry := (systematicCauchyRow (chosen.getD r 0) k).getD j 0
              acc ^^^ gf256Mul invEntry matEntry)
            0) = (if i == j then 1 else 0)

end Kadmium.StoragePool
