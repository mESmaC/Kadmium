/-!
# Transport chunk maps

A prepared model moves between Kadmium coordinators as files split into verified chunks. The
sender builds each file's chunk map by reading the file (`hash_stream` in
`crates/kadmium-data/src/checkpoint/transport.rs`); the receiver accepts a map only if
`TransportManifest::validate` finds that chunk `i` starts at `i * chunk_bytes` and holds
`min(bytes - offset, chunk_bytes)` bytes, with `div_ceil(bytes, chunk_bytes)` chunks in all
(`chunk_plan` in the same file).

This module proves that the two agree and that the accepted map is a tiling:

* `readLoop_eq_chunkPlan`: the sender's read loop produces exactly `chunkPlan`.
* `validChunks_iff`: the receiver's check accepts exactly one map per file, `chunkPlan`.
* `chunkPlan_tiles`: that map starts at offset 0, each chunk starts where the previous one ends,
  every chunk holds between 1 and `chunk_bytes` bytes, and together they cover the file's bytes
  exactly once. An empty file has no chunks.
* `chunkPlan_aligned`: when the alignment divides the chunk size, every chunk offset is aligned.
* `layout_divides`: 4 KiB divides the 64 KiB bank page, which divides the 16 MiB chunk.

Modelling choices:

* Byte counts and offsets are unbounded `Nat`. Rust uses `u64`; a file of `2^64` bytes does not
  exist, so overflow is not modelled.
* A chunk is its `(offset, bytes)` pair. Digests are not modelled: they are checked separately on
  every chunk and file, and they do not affect the layout.
* `hash_stream` fills its buffer until it is full or a read returns 0, and a read returns 0 only
  at end of file. The model therefore reads `min(remaining, chunk_bytes)` bytes per step and stops
  when nothing remains.
* `validate` checks the count first and then every index, so the model states both conditions.
-/

namespace Kadmium.Transport

/-- `u64::div_ceil`. -/
def divCeil (n d : Nat) : Nat := (n + d - 1) / d

/-- The chunk `validate` expects at index `i` of a file of `bytes` bytes. -/
def chunkAt (bytes c i : Nat) : Nat × Nat := (i * c, min (bytes - i * c) c)

/-- `chunk_plan`: the only chunk map `validate` accepts for a file of `bytes` bytes. -/
def chunkPlan (bytes c : Nat) : List (Nat × Nat) :=
  (List.range (divCeil bytes c)).map (chunkAt bytes c)

/-- The chunk check in `TransportManifest::validate`. -/
def ValidChunks (bytes c : Nat) (cs : List (Nat × Nat)) : Prop :=
  cs.length = divCeil bytes c ∧ ∀ (i : Nat) (h : i < cs.length), cs[i] = chunkAt bytes c i

/-- `hash_stream`'s loop: read up to `c` bytes at `off` until no bytes remain. -/
def readLoop (c : Nat) (off rem : Nat) : List (Nat × Nat) :=
  if _h : rem = 0 ∨ c = 0 then []
  else (off, min rem c) :: readLoop c (off + min rem c) (rem - min rem c)
termination_by rem
decreasing_by omega

/-- A list of chunks tiles `rem` bytes starting at `off`: each chunk starts where the previous one
ended, holds between 1 and `c` bytes, and the chunks cover exactly `rem` bytes. -/
def Tiles (c : Nat) : Nat → Nat → List (Nat × Nat) → Prop
  | _, rem, [] => rem = 0
  | off, rem, (o, len) :: cs => o = off ∧ 0 < len ∧ len ≤ c ∧ len ≤ rem ∧ Tiles c (off + len) (rem - len) cs

theorem divCeil_zero (c : Nat) : divCeil 0 c = 0 := by
  unfold divCeil
  cases c <;> simp

theorem divCeil_of_le {n c : Nat} (hn : 0 < n) (h : n ≤ c) : divCeil n c = 1 := by
  unfold divCeil
  apply Nat.div_eq_of_lt_le <;> omega

theorem divCeil_sub {n c : Nat} (hc : 0 < c) (h : c < n) : divCeil n c = divCeil (n - c) c + 1 := by
  unfold divCeil
  have : n + c - 1 = (n - c + c - 1) + c := by omega
  rw [this, Nat.add_div_right _ hc]

/-- The read loop starting at `off` is `chunkPlan` shifted by `off`. -/
theorem readLoop_eq_shifted (c : Nat) (hc : 0 < c) (rem : Nat) : ∀ off : Nat,
    readLoop c off rem =
      (List.range (divCeil rem c)).map (fun i => (off + i * c, min (rem - i * c) c)) := by
  induction rem using Nat.strongRecOn with
  | ind rem ih =>
    intro off
    unfold readLoop
    by_cases h0 : rem = 0
    · subst h0; simp [divCeil_zero]
    · have hne : ¬(rem = 0 ∨ c = 0) := by omega
      simp only [hne, ↓reduceDIte]
      by_cases hle : rem ≤ c
      · have hm : min rem c = rem := Nat.min_eq_left hle
        rw [hm, divCeil_of_le (by omega) hle]
        unfold readLoop
        simp
        omega
      · have hm : min rem c = c := Nat.min_eq_right (by omega)
        rw [hm, ih (rem - c) (by omega) (off + c)]
        conv => rhs; rw [divCeil_sub (n := rem) hc (by omega), List.range_succ_eq_map]
        simp only [List.map_cons, List.map_map, Nat.zero_mul, Nat.add_zero, Nat.sub_zero, hm]
        congr 1
        apply List.map_congr_left
        intro i _
        simp only [Function.comp, Nat.succ_mul]
        refine Prod.ext ?_ ?_
        · simp only; omega
        · simp only; congr 1; omega

/-- The sender's read loop produces exactly the map the receiver accepts. -/
theorem readLoop_eq_chunkPlan (c : Nat) (hc : 0 < c) (bytes : Nat) :
    readLoop c 0 bytes = chunkPlan bytes c := by
  rw [readLoop_eq_shifted c hc bytes 0]
  unfold chunkPlan chunkAt
  simp

/-- The receiver's check accepts exactly one map per file. -/
theorem validChunks_iff (bytes c : Nat) (cs : List (Nat × Nat)) :
    ValidChunks bytes c cs ↔ cs = chunkPlan bytes c := by
  constructor
  · rintro ⟨hlen, hat⟩
    apply List.ext_getElem
    · simp [chunkPlan, hlen]
    · intro i h₁ h₂
      rw [hat i h₁]
      simp [chunkPlan]
  · rintro rfl
    refine ⟨by simp [chunkPlan], ?_⟩
    intro i h
    simp [chunkPlan]

theorem readLoop_tiles (c : Nat) (hc : 0 < c) (rem : Nat) : ∀ off : Nat,
    Tiles c off rem (readLoop c off rem) := by
  induction rem using Nat.strongRecOn with
  | ind rem ih =>
    intro off
    unfold readLoop
    by_cases h0 : rem = 0
    · subst h0; simp [Tiles]
    · have hne : ¬(rem = 0 ∨ c = 0) := by omega
      simp only [hne, ↓reduceDIte]
      refine ⟨rfl, by omega, Nat.min_le_right _ _, Nat.min_le_left _ _, ?_⟩
      exact ih _ (by omega) _

/-- The accepted map tiles the file: it starts at 0, has no gaps or overlaps, every chunk holds
between 1 and `c` bytes, and the chunks cover exactly the file's bytes. -/
theorem chunkPlan_tiles (c : Nat) (hc : 0 < c) (bytes : Nat) :
    Tiles c 0 bytes (chunkPlan bytes c) := by
  rw [← readLoop_eq_chunkPlan c hc]
  exact readLoop_tiles c hc bytes 0

/-- Every chunk offset is a multiple of any alignment that divides the chunk size. -/
theorem chunkPlan_aligned {a c : Nat} (h : a ∣ c) (bytes : Nat) :
    ∀ x ∈ chunkPlan bytes c, a ∣ x.1 := by
  intro x hx
  simp only [chunkPlan, List.mem_map, List.mem_range] at hx
  obtain ⟨i, _, rfl⟩ := hx
  exact Nat.dvd_trans h (Nat.dvd_mul_left c i)

/-- `TRANSPORT_ALIGNMENT_BYTES` (4 KiB) divides the KTB page (64 KiB), which divides
`TRANSPORT_CHUNK_BYTES` (16 MiB). -/
theorem layout_divides : 4096 ∣ 65536 ∧ 65536 ∣ 16 * 1024 * 1024 :=
  ⟨⟨16, rfl⟩, ⟨256, rfl⟩⟩

end Kadmium.Transport
