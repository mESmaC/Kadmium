import KadmiumFormal.Bytes

/-!
# FNV-1a digests

Models the digest primitive in `crates/kadmium-runtime/src/repops.rs` (`fnv1a`, `Digest::new`,
`Digest::update`, `digest_bytes`, `fold_chain`), which is also `compute_fnv1a64_checksum` in
`crates/kadmium-data/src/lib.rs` and `fnv1a_64` in `crates/kadmium-grid/src/work.rs`: every
content address, `.ktb` checksum and run digest in Kadmium is this one function.

FNV-1a 64 is a left fold over bytes: `h ← (h ⊕ b) · 0x100000001b3` from `0xcbf29ce484222325`,
in `u64` wrapping arithmetic. The model is that fold over `UInt64`.

Results:

* `fnv1a_nil`, `fnv1a_a`, `fnv1a_foobar`: the constants are the published FNV-1a 64 constants,
  checked against the reference vectors for `""`, `"a"` and `"foobar"`.
* `digestUpdate_append`: feeding a byte string in any chunking gives the same state, so
  `update_f32s`, `update_u64` and friends are chunkings of one byte stream and a digest never
  depends on how its input was split.
* `digestBytes_eq_fnv1a`: `digest_bytes(tag, bytes)` is FNV-1a of
  `manifest_id (8 bytes LE) ++ tag ++ [0] ++ bytes`.
* `preimage_inj`, `foldChain_preimage_inj`: that byte string determines the manifest id, tag and
  payload, and the chain encoding determines the digests, provided the tag has no NUL byte. Two
  digests computed under different manifests, tags or payloads are therefore digests of different
  byte strings. (FNV-1a is not collision resistant, so equal digests do not imply equal inputs;
  the injectivity here is of the encoding, not of the hash.)

Modelling choices: bytes are `Nat` below 256 as in `KadmiumFormal.Bytes`; `Nat.toUInt64` widens
them exactly. `fold_chain` writes `digests.len() as u32`, so `foldChain_preimage_inj` assumes
fewer than `2^32` digests, which the Rust also needs for the length to survive the cast.
-/

namespace Kadmium.RepOps

open Kadmium.Bytes

def FNV_OFFSET : UInt64 := 0xcbf29ce484222325
def FNV_PRIME : UInt64 := 0x100000001b3

def fnvStep (h : UInt64) (b : Nat) : UInt64 := (h ^^^ b.toUInt64) * FNV_PRIME

/-- Continue a digest from state `h` over `bytes`. -/
def fnvFold (h : UInt64) (bytes : List Nat) : UInt64 := bytes.foldl fnvStep h

/-- `fnv1a(bytes)`, `compute_fnv1a64_checksum(bytes)`, `fnv1a_64(bytes)`. -/
def fnv1a (bytes : List Nat) : UInt64 := fnvFold FNV_OFFSET bytes

theorem fnvFold_append (h : UInt64) (a b : List Nat) :
    fnvFold h (a ++ b) = fnvFold (fnvFold h a) b :=
  List.foldl_append

/-- Reference vectors for FNV-1a 64. -/
theorem fnv1a_nil : fnv1a [] = 0xcbf29ce484222325 := rfl

theorem fnv1a_a : fnv1a [0x61] = 0xaf63dc4c8601ec8c := by decide

theorem fnv1a_foobar : fnv1a [0x66, 0x6f, 0x6f, 0x62, 0x61, 0x72] = 0x85944171f73967e8 := by
  decide

/-! ## `Digest` -/

/-- `Digest::new(tag)`: the manifest id little-endian, the tag bytes, then one zero byte. -/
def digestNew (manifestId : Nat) (tag : List Nat) : UInt64 :=
  fnvFold FNV_OFFSET (encodeLE 8 manifestId ++ tag ++ [0])

/-- `Digest::update`. -/
def digestUpdate (state : UInt64) (bytes : List Nat) : UInt64 := fnvFold state bytes

/-- `digest_bytes(tag, bytes)`. -/
def digestBytes (manifestId : Nat) (tag payload : List Nat) : UInt64 :=
  digestUpdate (digestNew manifestId tag) payload

/-- The byte string `fold_chain` feeds after the tag: the count as `u32` LE, then every digest as
`u64` LE. -/
def chainBytes (digests : List Nat) : List Nat :=
  encodeLE 4 digests.length ++ (digests.map (encodeLE 8)).flatten

/-- `fold_chain(tag, digests)`. -/
def foldChain (manifestId : Nat) (tag : List Nat) (digests : List Nat) : UInt64 :=
  digestUpdate (digestNew manifestId tag) (chainBytes digests)

/-- Two updates are one update over the concatenation: a digest does not depend on how its input
was chunked. -/
theorem digestUpdate_append (s : UInt64) (a b : List Nat) :
    digestUpdate (digestUpdate s a) b = digestUpdate s (a ++ b) :=
  (fnvFold_append s a b).symm

theorem digestBytes_eq_fnv1a (manifestId : Nat) (tag payload : List Nat) :
    digestBytes manifestId tag payload = fnv1a (encodeLE 8 manifestId ++ tag ++ [0] ++ payload) :=
  (fnvFold_append _ _ _).symm

/-! ## The encoding is injective -/

/-- A NUL-free tag followed by a zero byte is a self-delimiting prefix. -/
theorem frame_inj {tag tag' p p' : List Nat} (ht : 0 ∉ tag) (ht' : 0 ∉ tag')
    (h : tag ++ 0 :: p = tag' ++ 0 :: p') : tag = tag' ∧ p = p' := by
  induction tag generalizing tag' with
  | nil =>
    cases tag' with
    | nil => simpa using h
    | cons y rest' =>
      simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
      exact absurd (h.1 ▸ List.mem_cons_self) ht'
  | cons x rest ih =>
    cases tag' with
    | nil =>
      simp only [List.cons_append, List.nil_append, List.cons.injEq] at h
      exact absurd (h.1 ▸ List.mem_cons_self) ht
    | cons y rest' =>
      simp only [List.cons_append, List.cons.injEq] at h
      obtain ⟨rfl, h2⟩ := h
      have := ih (fun hm => ht (List.mem_cons_of_mem _ hm)) (fun hm => ht' (List.mem_cons_of_mem _ hm)) h2
      exact ⟨by rw [this.1], this.2⟩

theorem two_pow_64 : (2 : Nat) ^ 64 = 256 ^ 8 := by decide

theorem two_pow_32 : (2 : Nat) ^ 32 = 256 ^ 4 := by decide

/-- The bytes a digest hashes determine its manifest, tag and payload. -/
theorem preimage_inj {id id' : Nat} {tag tag' p p' : List Nat} (hid : id < 2 ^ 64)
    (hid' : id' < 2 ^ 64) (ht : 0 ∉ tag) (ht' : 0 ∉ tag')
    (h : encodeLE 8 id ++ tag ++ 0 :: p = encodeLE 8 id' ++ tag' ++ 0 :: p') :
    id = id' ∧ tag = tag' ∧ p = p' := by
  rw [List.append_assoc, List.append_assoc] at h
  obtain ⟨h1, h2⟩ := List.append_inj h (by simp)
  rw [two_pow_64] at hid hid'
  exact ⟨encodeLE_inj hid hid' h1, frame_inj ht ht' h2⟩

theorem flatten_encodeLE8_inj :
    ∀ {ds ds' : List Nat}, ds.length = ds'.length → (∀ d ∈ ds, d < 2 ^ 64) →
      (∀ d ∈ ds', d < 2 ^ 64) →
      (ds.map (encodeLE 8)).flatten = (ds'.map (encodeLE 8)).flatten → ds = ds' := by
  intro ds
  induction ds with
  | nil => intro ds' hl _ _ _; cases ds' with | nil => rfl | cons => simp at hl
  | cons d rest ih =>
    intro ds' hl hb hb' h
    cases ds' with
    | nil => simp at hl
    | cons d' rest' =>
      simp only [List.map_cons, List.flatten_cons] at h
      obtain ⟨h1, h2⟩ := List.append_inj h (by simp)
      have hd := hb d (by simp)
      have hd' := hb' d' (by simp)
      rw [two_pow_64] at hd hd'
      have := ih (by simpa using hl) (fun x hx => hb x (by simp [hx]))
        (fun x hx => hb' x (by simp [hx])) h2
      rw [encodeLE_inj hd hd' h1, this]

/-- The chain encoding determines the digests it folds. -/
theorem foldChain_preimage_inj {ds ds' : List Nat} (hn : ds.length < 2 ^ 32)
    (hn' : ds'.length < 2 ^ 32) (hb : ∀ d ∈ ds, d < 2 ^ 64) (hb' : ∀ d ∈ ds', d < 2 ^ 64)
    (h : chainBytes ds = chainBytes ds') : ds = ds' := by
  unfold chainBytes at h
  obtain ⟨h1, h2⟩ := List.append_inj h (by simp)
  rw [two_pow_32] at hn hn'
  have hl := encodeLE_inj hn hn' h1
  exact flatten_encodeLE8_inj hl hb hb' h2

end Kadmium.RepOps
