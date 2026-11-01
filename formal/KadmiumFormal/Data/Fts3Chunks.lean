/-!
# FTS3 chunk manifests

Models `Fts3ChunkSlicer::compute_chunk_manifest` in `crates/kadmium-data/src/transcoder.rs`.
The Rust walks `ktb_bytes.chunks(cs)` with `cs = chunk_size_bytes.max(65_536)` and records, for
every chunk in order, its index, byte offset and byte length together with the FNV-1a checksum of
its bytes. The model keeps the geometry; the checksum is the function modelled in
`KadmiumFormal.RepOps.Digest`.

Modelling choices:

* `slice::chunks` yields `⌈len / cs⌉` chunks and nothing for an empty slice; the model writes
  that count as `(len + cs - 1) / cs`.
* Lengths are unbounded `Nat`; the Rust casts to `u32` and `u64` cannot truncate below `2^32`
  chunks and `2^64` bytes, which no `.ktb` bank reaches.

Results: the chunks tile `[0, len)` contiguously in index order (`chunkOffset_succ`,
`chunks_sum`, `chunks_cover`), every chunk is non-empty and at most `cs` bytes
(`chunkLen_pos`, `chunkLen_le`), and every chunk but the last is exactly `cs` bytes
(`chunkLen_full`). A destination that holds every chunk of the manifest therefore holds every
byte of the bank exactly once, which is what `diff_missing_chunks` relies on.
-/

namespace Kadmium.Data

/-- The smallest chunk the slicer will cut: one `.ktb` page. -/
def MIN_CHUNK : Nat := 65536

/-- `chunk_size_bytes.max(65_536)`. -/
def chunkSize (requested : Nat) : Nat := max requested MIN_CHUNK

/-- How many chunks `len` bytes make at `cs` bytes each: `⌈len / cs⌉`. -/
def chunkCount (len cs : Nat) : Nat := (len + cs - 1) / cs

def chunkOffset (cs i : Nat) : Nat := i * cs

def chunkLen (len cs i : Nat) : Nat := min cs (len - i * cs)

/-- The manifest geometry: `(byte_offset, byte_len)` for every chunk index in order. -/
def manifest (len cs : Nat) : List (Nat × Nat) :=
  (List.range (chunkCount len cs)).map fun i => (chunkOffset cs i, chunkLen len cs i)

theorem chunkSize_pos (requested : Nat) : 0 < chunkSize requested :=
  Nat.lt_of_lt_of_le (by decide : 0 < MIN_CHUNK) (Nat.le_max_right _ _)

/-- A chunk index below the count starts inside the bank. -/
theorem lt_of_lt_chunkCount {len cs i : Nat} (hcs : 0 < cs) (h : i < chunkCount len cs) :
    i * cs < len := by
  unfold chunkCount at h
  have := (Nat.le_div_iff_mul_le hcs).mp h
  rw [Nat.succ_mul] at this
  omega

/-- The count is the least number of chunks that covers the bank. -/
theorem len_le_chunkCount_mul {len cs : Nat} (hcs : 0 < cs) : len ≤ chunkCount len cs * cs := by
  unfold chunkCount
  have h1 := Nat.div_add_mod (len + cs - 1) cs
  have h2 := Nat.mod_lt (len + cs - 1) hcs
  rw [Nat.mul_comm]
  omega

theorem chunkCount_eq_zero {len cs : Nat} (hcs : 0 < cs) : chunkCount len cs = 0 ↔ len = 0 := by
  constructor
  · intro h
    have := len_le_chunkCount_mul (len := len) hcs
    rw [h] at this
    omega
  · intro h
    subst h
    unfold chunkCount
    exact Nat.div_eq_of_lt (by omega)

theorem chunkLen_le (len cs i : Nat) : chunkLen len cs i ≤ cs := Nat.min_le_left _ _

theorem chunkLen_pos {len cs i : Nat} (hcs : 0 < cs) (h : i < chunkCount len cs) :
    0 < chunkLen len cs i := by
  have := lt_of_lt_chunkCount hcs h
  unfold chunkLen
  rw [Nat.min_def]
  split <;> omega

/-- Every chunk but the last is a full `cs` bytes. -/
theorem chunkLen_full {len cs i : Nat} (hcs : 0 < cs) (h : i + 1 < chunkCount len cs) :
    chunkLen len cs i = cs := by
  have := lt_of_lt_chunkCount hcs h
  unfold chunkLen
  rw [Nat.succ_mul] at this
  exact Nat.min_eq_left (by omega)

/-- Chunks are contiguous: each starts where the one before ends. -/
theorem chunkOffset_succ {len cs i : Nat} (hcs : 0 < cs) (h : i + 1 < chunkCount len cs) :
    chunkOffset cs (i + 1) = chunkOffset cs i + chunkLen len cs i := by
  rw [chunkLen_full hcs h]
  unfold chunkOffset
  rw [Nat.succ_mul]

theorem sum_chunkLen (len cs : Nat) :
    ∀ n, ((List.range n).map (chunkLen len cs)).sum = min len (n * cs) := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.sum_append, ih]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero, chunkLen,
      Nat.succ_mul]
    simp only [Nat.min_def]
    split <;> split <;> split <;> omega

/-- The chunk lengths add up to the bank. -/
theorem chunks_sum {len cs : Nat} (hcs : 0 < cs) :
    ((manifest len cs).map Prod.snd).sum = len := by
  unfold manifest
  rw [List.map_map]
  have : (Prod.snd ∘ fun i => (chunkOffset cs i, chunkLen len cs i)) = chunkLen len cs := rfl
  rw [this, sum_chunkLen]
  exact Nat.min_eq_left (len_le_chunkCount_mul hcs)

/-- The last chunk ends exactly at the end of the bank. -/
theorem chunks_cover {len cs : Nat} (hcs : 0 < cs) (hlen : 0 < len) :
    chunkOffset cs (chunkCount len cs - 1) + chunkLen len cs (chunkCount len cs - 1) = len := by
  have hc : 0 < chunkCount len cs := by
    rcases Nat.eq_zero_or_pos (chunkCount len cs) with h | h
    · rw [chunkCount_eq_zero hcs] at h; omega
    · exact h
  have h1 := lt_of_lt_chunkCount hcs (Nat.sub_lt hc Nat.one_pos)
  have h2 := len_le_chunkCount_mul (len := len) hcs
  have h3 : chunkCount len cs * cs = (chunkCount len cs - 1) * cs + cs := by
    rw [← Nat.succ_mul, Nat.succ_eq_add_one, Nat.sub_add_cancel hc]
  unfold chunkOffset chunkLen
  rw [Nat.min_def]
  split <;> omega

@[simp] theorem manifest_length (len cs : Nat) :
    (manifest len cs).length = chunkCount len cs := by
  simp [manifest]

theorem manifest_getElem (len cs i : Nat) (h : i < (manifest len cs).length) :
    (manifest len cs)[i] = (chunkOffset cs i, chunkLen len cs i) := by
  simp [manifest]

end Kadmium.Data
