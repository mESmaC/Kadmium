import KadmiumFormal.Bytes

/-!
# MoE batch codecs

Models the two binary batch codecs a grid worker decodes from a `WorkDispatch` payload, in
`crates/kadmium-runtime/src/moe_layer.rs`:

* `TokenBatchF32`: `"F32B" | rows u32 | dim u32 | rows * dim f32`.
* `ExpertPartialsBatch`: `"PART" | layer u32 | rows u32 | dim u32 | count u32 |
  (row u32 | expert u32 | dim f32) * count`.

Both decoders check that the body length is exactly what the header implies before reading any
element. The model writes that check twice: `expected*` is the arithmetic the Rust means, and
`expected*Wrapped` is the same arithmetic in `usize` as a release build evaluates it, where the
multiplication wraps modulo `2^64`.

Results:

* `decodeF32_encodeF32`, `decodePart_encodePart`: the encoders' output decodes back to the input
  for every batch whose counts fit `u32`.
* `decodeF32_exact`, `decodePart_exact`: whatever decodes has exactly the length the header
  implies, so an accepted body is read in full and nothing is read past it.
* `wrapped_accepts_empty_f32`, `wrapped_accepts_header_only_part` (**finding**): with the wrapping
  arithmetic, a `2^31 × 2^31` `F32B` header with no body and a `PART` header declaring `2^31`
  partials of width `2^31 - 2` with no body both pass the length check. The checked arithmetic
  (`checked_rejects_empty_f32`, `checked_rejects_header_only_part`) rejects them. The Rust
  decoders now use `checked_mul`; the fixture test in
  `crates/kadmium-runtime/tests/batch_codec_lean_vectors.rs` feeds both headers and requires an
  error rather than a wrapped acceptance or a debug-build overflow panic.

Modelling choices: an `f32` is its four little-endian bytes; the model moves bytes and never
interprets them, which is also what the Rust does (`f32::from_le_bytes`). Bytes are `Nat` below
256; the Rust `usize` is 64 bits on every supported target.
-/

namespace Kadmium.Runtime

open Kadmium.Bytes

/-- `b"F32B"`. -/
def F32B : List Nat := [0x46, 0x33, 0x32, 0x42]
/-- `b"PART"`. -/
def PART : List Nat := [0x50, 0x41, 0x52, 0x54]

inductive CodecError where
  /-- `"...: missing F32B header"`, `"...: missing PART header"` -/
  | header
  /-- `"...: {} data bytes for a {rows} x {dim} batch"`, `"...: {} bytes do not hold ..."` -/
  | length
  /-- The checked arithmetic overflowed `usize`. -/
  | overflow
  deriving DecidableEq, Repr

/-! ## `TokenBatchF32` -/

/-- `TokenBatchF32::to_le_bytes`; `body` is the `rows * dim * 4` payload bytes. -/
def encodeF32 (rows dim : Nat) (body : List Nat) : List Nat :=
  F32B ++ (encodeLE 4 rows ++ (encodeLE 4 dim ++ body))

/-- The body length the header implies. -/
def expectedF32 (rows dim : Nat) : Nat := rows * dim * 4

/-- The same expression as a release build evaluates `(rows as usize) * (dim as usize) * 4`. -/
def expectedF32Wrapped (rows dim : Nat) : Nat := rows * dim % 2 ^ 64 * 4 % 2 ^ 64

/-- `TokenBatchF32::from_le_bytes` with checked arithmetic. -/
def decodeF32 (bytes : List Nat) : Except CodecError (Nat × Nat × List Nat) :=
  if bytes.length < 12 ∨ bytes.take 4 ≠ F32B then .error .header
  else
    let rows := decodeLE ((bytes.drop 4).take 4)
    let dim := decodeLE ((bytes.drop 8).take 4)
    let body := bytes.drop 12
    if body.length = expectedF32 rows dim then .ok (rows, dim, body) else .error .length

theorem decodeF32_encodeF32 (rows dim : Nat) (body : List Nat) (hr : rows < 2 ^ 32)
    (hd : dim < 2 ^ 32) (hb : body.length = rows * dim * 4) :
    decodeF32 (encodeF32 rows dim body) = .ok (rows, dim, body) := by
  have hlen : (encodeF32 rows dim body).length = 12 + body.length := by
    simp [encodeF32, F32B]; omega
  have h12 : ¬ (encodeF32 rows dim body).length < 12 := by rw [hlen]; omega
  have htake : (encodeF32 rows dim body).take 4 = F32B := by simp [encodeF32, F32B]
  have hdrop4 : (encodeF32 rows dim body).drop 4 = encodeLE 4 rows ++ (encodeLE 4 dim ++ body) := by
    simp [encodeF32, F32B]
  have hdrop8 : (encodeF32 rows dim body).drop 8 = encodeLE 4 dim ++ body := by
    simp [encodeF32, F32B, encodeLE]
  have hdrop12 : (encodeF32 rows dim body).drop 12 = body := by
    simp [encodeF32, F32B, encodeLE]
  unfold decodeF32
  simp only [h12, htake, hdrop4, hdrop8, hdrop12, ne_eq, not_true_eq_false, or_self,
    ↓reduceIte, List.take_left', length_encodeLE, decodeLE_encodeLE (lt_pow_four hr),
    decodeLE_encodeLE (lt_pow_four hd), expectedF32, hb]

/-- An accepted `F32B` body has exactly the implied length. -/
theorem decodeF32_exact {bytes : List Nat} {rows dim : Nat} {body : List Nat}
    (h : decodeF32 bytes = .ok (rows, dim, body)) :
    body.length = rows * dim * 4 ∧ bytes.length = 12 + rows * dim * 4 := by
  unfold decodeF32 at h
  split at h
  · exact absurd h (by simp)
  · rename_i hhead
    simp only at h
    split at h
    · rename_i hlen
      simp only [Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      refine ⟨hlen, ?_⟩
      rw [List.length_drop] at hlen
      unfold expectedF32 at hlen
      omega
    · exact absurd h (by simp)

/-- **Finding.** In `usize` arithmetic a `2^31 × 2^31` header implies an empty body. -/
theorem wrapped_accepts_empty_f32 : expectedF32Wrapped (2 ^ 31) (2 ^ 31) = 0 := by decide

theorem checked_rejects_empty_f32 : expectedF32 (2 ^ 31) (2 ^ 31) ≠ 0 := by decide

/-! ## `ExpertPartialsBatch` -/

/-- One partial: `row u32 | expert u32 | dim f32`, with `data` the raw `dim * 4` bytes. -/
structure Partial where
  row : Nat
  expert : Nat
  data : List Nat
  deriving DecidableEq, Repr

def encodeEntry (p : Partial) : List Nat := encodeLE 4 p.row ++ (encodeLE 4 p.expert ++ p.data)

/-- `ExpertPartialsBatch::to_le_bytes`. -/
def encodePart (layer rows dim : Nat) (partials : List Partial) : List Nat :=
  PART ++ (encodeLE 4 layer ++ (encodeLE 4 rows ++ (encodeLE 4 dim ++
    (encodeLE 4 partials.length ++ (partials.map encodeEntry).flatten))))

/-- `8 + dim * 4`. -/
def entryLen (dim : Nat) : Nat := 8 + dim * 4

/-- The total length the header implies. -/
def expectedPart (count dim : Nat) : Nat := 20 + count * entryLen dim

/-- The same expression as a release build evaluates it. -/
def expectedPartWrapped (count dim : Nat) : Nat :=
  (20 + count * ((8 + dim * 4) % 2 ^ 64) % 2 ^ 64) % 2 ^ 64

/-- Parse `count` entries of `entryLen dim` bytes each. -/
def parseEntries (dim : Nat) : Nat → List Nat → List Partial
  | 0, _ => []
  | count + 1, bytes =>
    ⟨decodeLE (bytes.take 4), decodeLE ((bytes.drop 4).take 4), (bytes.drop 8).take (dim * 4)⟩ ::
      parseEntries dim count (bytes.drop (entryLen dim))

/-- `ExpertPartialsBatch::from_le_bytes` with checked arithmetic. -/
def decodePart (bytes : List Nat) : Except CodecError (Nat × Nat × Nat × List Partial) :=
  if bytes.length < 20 ∨ bytes.take 4 ≠ PART then .error .header
  else
    let layer := decodeLE ((bytes.drop 4).take 4)
    let rows := decodeLE ((bytes.drop 8).take 4)
    let dim := decodeLE ((bytes.drop 12).take 4)
    let count := decodeLE ((bytes.drop 16).take 4)
    if bytes.length = expectedPart count dim then
      .ok (layer, rows, dim, parseEntries dim count (bytes.drop 20))
    else .error .length

/-- Every entry has the width the header declares. -/
def WellFormed (dim : Nat) (partials : List Partial) : Prop :=
  ∀ p ∈ partials, p.data.length = dim * 4

theorem length_encodeEntry {dim : Nat} {p : Partial} (h : p.data.length = dim * 4) :
    (encodeEntry p).length = entryLen dim := by
  simp [encodeEntry, entryLen, h]
  omega

theorem length_flatten_entries {dim : Nat} :
    ∀ {partials : List Partial}, WellFormed dim partials →
      ((partials.map encodeEntry).flatten).length = partials.length * entryLen dim := by
  intro partials
  induction partials with
  | nil => intro _; simp
  | cons p rest ih =>
    intro hw
    simp only [List.map_cons, List.flatten_cons, List.length_append, List.length_cons]
    rw [length_encodeEntry (hw p (by simp)), ih (fun q hq => hw q (by simp [hq])), Nat.succ_mul]
    omega

theorem parseEntries_flatten {dim : Nat} :
    ∀ {partials : List Partial} (rest : List Nat), WellFormed dim partials →
      (∀ p ∈ partials, p.row < 2 ^ 32 ∧ p.expert < 2 ^ 32) →
      parseEntries dim partials.length ((partials.map encodeEntry).flatten ++ rest) = partials := by
  intro partials
  induction partials with
  | nil => intro _ _ _; rfl
  | cons p rest ih =>
    intro tail hw hb
    have hd := hw p (by simp)
    obtain ⟨hrow, hexp⟩ := hb p (by simp)
    simp only [List.map_cons, List.flatten_cons, List.length_cons, parseEntries, List.append_assoc]
    have hdrop8 : (encodeEntry p ++ ((rest.map encodeEntry).flatten ++ tail)).drop 8 =
        p.data ++ ((rest.map encodeEntry).flatten ++ tail) := by
      simp [encodeEntry, encodeLE]
    have hdropE : (encodeEntry p ++ ((rest.map encodeEntry).flatten ++ tail)).drop (entryLen dim) =
        (rest.map encodeEntry).flatten ++ tail := by
      rw [List.drop_left' (length_encodeEntry hd)]
    rw [hdrop8, hdropE, ih tail (fun q hq => hw q (by simp [hq])) (fun q hq => hb q (by simp [hq]))]
    simp only [encodeEntry, List.append_assoc, List.take_left', length_encodeLE, List.drop_left',
      decodeLE_encodeLE (lt_pow_four hrow), decodeLE_encodeLE (lt_pow_four hexp)]
    rw [List.take_left' hd]

theorem decodePart_encodePart (layer rows dim : Nat) (partials : List Partial)
    (hl : layer < 2 ^ 32) (hr : rows < 2 ^ 32) (hd : dim < 2 ^ 32) (hc : partials.length < 2 ^ 32)
    (hw : WellFormed dim partials) (hb : ∀ p ∈ partials, p.row < 2 ^ 32 ∧ p.expert < 2 ^ 32) :
    decodePart (encodePart layer rows dim partials) = .ok (layer, rows, dim, partials) := by
  have hflat := length_flatten_entries (dim := dim) hw
  have hlen : (encodePart layer rows dim partials).length = expectedPart partials.length dim := by
    simp only [encodePart, PART, List.length_append, List.length_cons, List.length_nil,
      length_encodeLE, hflat, expectedPart]
    omega
  have h20 : ¬ expectedPart partials.length dim < 20 := by unfold expectedPart; omega
  have htake : (encodePart layer rows dim partials).take 4 = PART := by simp [encodePart, PART]
  have hdrop4 : (encodePart layer rows dim partials).drop 4 = encodeLE 4 layer ++ (encodeLE 4 rows ++
      (encodeLE 4 dim ++ (encodeLE 4 partials.length ++ (partials.map encodeEntry).flatten))) := by
    simp [encodePart, PART]
  have hdrop8 : (encodePart layer rows dim partials).drop 8 = encodeLE 4 rows ++
      (encodeLE 4 dim ++ (encodeLE 4 partials.length ++ (partials.map encodeEntry).flatten)) := by
    simp [encodePart, PART, encodeLE]
  have hdrop12 : (encodePart layer rows dim partials).drop 12 =
      encodeLE 4 dim ++ (encodeLE 4 partials.length ++ (partials.map encodeEntry).flatten) := by
    simp [encodePart, PART, encodeLE]
  have hdrop16 : (encodePart layer rows dim partials).drop 16 =
      encodeLE 4 partials.length ++ (partials.map encodeEntry).flatten := by
    simp [encodePart, PART, encodeLE]
  have hdrop20 : (encodePart layer rows dim partials).drop 20 = (partials.map encodeEntry).flatten := by
    simp [encodePart, PART, encodeLE]
  have hentries := parseEntries_flatten (dim := dim) [] hw hb
  rw [List.append_nil] at hentries
  unfold decodePart
  simp only [h20, htake, hdrop4, hdrop8, hdrop12, hdrop16, hdrop20, ne_eq, not_true_eq_false,
    or_false, ↓reduceIte, List.take_left', length_encodeLE, decodeLE_encodeLE (lt_pow_four hl),
    decodeLE_encodeLE (lt_pow_four hr), decodeLE_encodeLE (lt_pow_four hd),
    decodeLE_encodeLE (lt_pow_four hc), hlen, hentries]

/-- An accepted `PART` payload has exactly the implied length. -/
theorem decodePart_exact {bytes : List Nat} {layer rows dim : Nat} {partials : List Partial}
    (h : decodePart bytes = .ok (layer, rows, dim, partials)) :
    ∃ count, bytes.length = 20 + count * (8 + dim * 4) := by
  unfold decodePart at h
  split at h
  · exact absurd h (by simp)
  · simp only at h
    split at h
    · rename_i hlen
      simp only [Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨_, _, rfl, _⟩ := h
      exact ⟨_, hlen⟩
    · exact absurd h (by simp)

/-- **Finding.** In `usize` arithmetic, `2^31` partials of width `2^31 - 2` take `2^64` bytes,
which wraps to none: a bare 20-byte header passes the length check. -/
theorem wrapped_accepts_header_only_part : expectedPartWrapped (2 ^ 31) (2 ^ 31 - 2) = 20 := by
  decide

theorem checked_rejects_header_only_part : expectedPart (2 ^ 31) (2 ^ 31 - 2) ≠ 20 := by decide

end Kadmium.Runtime
