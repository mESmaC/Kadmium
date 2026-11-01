/-!
# Authenticated native decoder session wire frames (`KDNS`)

`crates/kadmium-runtime/src/native_decoder/session.rs` encodes and decodes `KDNS` session frames
(`encode_envelope`, `decode_envelope`) and raw `f32` tensor payloads (`encode_f32_tensor`,
`decode_f32_tensor`) inside `WireFrameKind::NativeDecoderSession` frames over the worker
connection.

This module models both layers and proves five properties:

1. **Envelope round-trip (`decodeEnvelope_encodeEnvelope`):** every frame produced by
   `encodeEnvelope` within `Limits` decodes under the same domain and authentication key back to
   its exact `(kind, metadata, payload)` triple.
2. **Pre-allocation size bounds (`decodeEnvelope_bounds`):** `decodeEnvelope` succeeds only when
   the total frame size, the metadata length, and the binary payload length are within `Limits` and
   partition the body past the 46-byte header (`46 + metadata.length + payload.length = raw.length`).
3. **Authentication and domain separation (`decodeEnvelope_authenticated`,
   `decodeEnvelope_rejects_wrong_key`, `decodeEnvelope_rejects_domain_swap`):** `decodeEnvelope`
   succeeds only when the wire MAC equals `mac expectedDomain expectedKey kind metadataLen payloadLen metadata payload`.
   For an injective MAC (standing in for HMAC-SHA256 collision resistance), a frame signed with a
   different key or under the opposite direction's domain (`Request` vs `Response`) is rejected.
4. **Exact tensor byte-length round-trip (`decodeTensor_encodeTensor`, `decodeTensor_exact`):**
   `encode_f32_tensor` and `decode_f32_tensor` preserve the raw 32-bit float words (`UInt32`) with
   no rounding or cast, and accept a payload if and only if its byte length equals `rows * width * 4`
   within `maxPayloadBytes`.
5. **Overflow counterexample (`wrapped_tensor_overflow`, `checked_rejects_tensor_overflow`):**
   unchecked 64-bit multiplication wraps `rows = 2^31`, `width = 2^31`, `elemBytes = 4` to `0`,
   accepting a 0-byte payload for a `2^62`-element tensor; `checkedTensorBytes64` (mirroring
   `rows.checked_mul(width).and_then(|n| n.checked_mul(4))`) rejects it.

Modelling choices:

* Byte buffers are modelled by their contents and lengths; `EnvelopeWire` stores the parsed header
  fields (`magic`, `version`, `kind`, `metadataLen`, `payloadLen`, `tag`) together with the body
  slices `metadata` and `payload` and the total wire length `rawLen = 46 + metadata.length + payload.length`,
  matching the slice split in `decode_envelope` after the `total_len != bytes.len()` check.
* `f32` values are modelled as their exact 32-bit IEEE 754 bit patterns (`UInt32`), matching
  `f32::to_le_bytes` / `f32::from_le_bytes`.
* HMAC-SHA256 is parameterized as an abstract `MacFn`; theorems that state rejection of a wrong key
  or swapped domain take an explicit inequality hypothesis on `mac` rather than assuming a global
  cryptographic axiom.

The link to the code is **model only**. The Rust tests in
`crates/kadmium-runtime/tests/native_decoder.rs` exercise the wire implementation across the four
reference checkpoints, all special IEEE 754 bit patterns (`-0.0`, subnormals, infinities, NaN
payloads), authentication failures, and overflow/size-limit rejections.
-/

namespace Kadmium.Native.SessionWire

/-- Fixed header size of a `KDNS` envelope: 4-byte magic + 1-byte version + 1-byte kind +
4-byte metadata length + 4-byte payload length + 32-byte HMAC-SHA256 tag. -/
abbrev headerBytes : Nat := 46

/-- Fixed magic value (`"KDNS"`). -/
abbrev kdnsMagic : UInt32 := 0x534e444b

/-- Supported wire protocol version. -/
abbrev kdnsVersion : UInt8 := 1

/-- Domain separation label distinguishing coordinator requests from worker responses. -/
inductive Domain where
  | request
  | response
  deriving DecidableEq, Repr

/-- Size bounds enforced before allocating metadata or tensor buffers (`NativeSessionLimits`). -/
structure Limits where
  maxFrameBytes : Nat
  maxMetadataBytes : Nat
  maxPayloadBytes : Nat
  deriving DecidableEq, Repr

/-- Abstract MAC function over `(domain, key, kind, metadataLen, payloadLen, metadata, payload)`. -/
abbrev MacFn (Key Tag : Type) :=
  Domain → Key → UInt8 → Nat → Nat → List UInt8 → List UInt8 → Tag

/-- Wire representation of a `KDNS` frame. -/
structure EnvelopeWire (Tag : Type) where
  magic : UInt32
  version : UInt8
  kind : UInt8
  metadataLen : Nat
  payloadLen : Nat
  tag : Tag
  metadata : List UInt8
  payload : List UInt8
  deriving DecidableEq, Repr

/-- Total byte length of an `EnvelopeWire` on the wire. -/
def EnvelopeWire.totalBytes {Tag : Type} (w : EnvelopeWire Tag) : Nat :=
  headerBytes + w.metadata.length + w.payload.length

/-- Errors returned by `encodeEnvelope` and `decodeEnvelope`. -/
inductive WireError where
  | frameTooLarge (actual limit : Nat)
  | metadataTooLarge (actual limit : Nat)
  | payloadTooLarge (actual limit : Nat)
  | badMagic
  | unsupportedVersion (version : UInt8)
  | lengthMismatch
  | unauthenticated
  deriving DecidableEq, Repr

/-- Pure model of `encode_envelope` (`crates/kadmium-runtime/src/native_decoder/session.rs`). -/
def encodeEnvelope {Key Tag : Type}
    (mac : MacFn Key Tag) (limits : Limits) (domain : Domain) (key : Key)
    (kind : UInt8) (metadata payload : List UInt8) :
    Except WireError (EnvelopeWire Tag) :=
  if metadata.length > limits.maxMetadataBytes then
    .error (.metadataTooLarge metadata.length limits.maxMetadataBytes)
  else if payload.length > limits.maxPayloadBytes then
    .error (.payloadTooLarge payload.length limits.maxPayloadBytes)
  else if headerBytes + metadata.length + payload.length > limits.maxFrameBytes then
    .error (.frameTooLarge (headerBytes + metadata.length + payload.length) limits.maxFrameBytes)
  else
    .ok {
      magic := kdnsMagic
      version := kdnsVersion
      kind := kind
      metadataLen := metadata.length
      payloadLen := payload.length
      tag := mac domain key kind metadata.length payload.length metadata payload
      metadata := metadata
      payload := payload
    }

/-- Pure model of `decode_envelope` (`crates/kadmium-runtime/src/native_decoder/session.rs`).
Checks frame, metadata, and payload bounds and verifies the HMAC tag before returning slices. -/
def decodeEnvelope {Key Tag : Type} [DecidableEq Tag]
    (mac : MacFn Key Tag) (limits : Limits) (domain : Domain) (key : Key)
    (w : EnvelopeWire Tag) :
    Except WireError (UInt8 × List UInt8 × List UInt8) :=
  if w.totalBytes > limits.maxFrameBytes then
    .error (.frameTooLarge w.totalBytes limits.maxFrameBytes)
  else if w.magic != kdnsMagic then
    .error .badMagic
  else if w.version != kdnsVersion then
    .error (.unsupportedVersion w.version)
  else if w.metadataLen > limits.maxMetadataBytes then
    .error (.metadataTooLarge w.metadataLen limits.maxMetadataBytes)
  else if w.payloadLen > limits.maxPayloadBytes then
    .error (.payloadTooLarge w.payloadLen limits.maxPayloadBytes)
  else if w.metadataLen != w.metadata.length then
    .error .lengthMismatch
  else if w.payloadLen != w.payload.length then
    .error .lengthMismatch
  else if w.tag != mac domain key w.kind w.metadataLen w.payloadLen w.metadata w.payload then
    .error .unauthenticated
  else
    .ok (w.kind, w.metadata, w.payload)

/-- Every frame accepted by `encodeEnvelope` decodes back to the exact `(kind, metadata, payload)`
under the same domain and key. -/
theorem decodeEnvelope_encodeEnvelope {Key Tag : Type} [DecidableEq Tag]
    (mac : MacFn Key Tag) (limits : Limits) (domain : Domain) (key : Key)
    (kind : UInt8) (metadata payload : List UInt8) (w : EnvelopeWire Tag)
    (henc : encodeEnvelope mac limits domain key kind metadata payload = .ok w) :
    decodeEnvelope mac limits domain key w = .ok (kind, metadata, payload) := by
  unfold encodeEnvelope at henc
  split at henc
  · cases henc
  · split at henc
    · cases henc
    · split at henc
      · cases henc
      · injection henc with hw
        subst hw
        unfold decodeEnvelope EnvelopeWire.totalBytes
        simp_all only [gt_iff_lt, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]

/-- `decodeEnvelope` succeeds only when the frame, metadata, and payload lengths satisfy `limits`
and match the declared header lengths. -/
theorem decodeEnvelope_bounds {Key Tag : Type} [DecidableEq Tag]
    (mac : MacFn Key Tag) (limits : Limits) (domain : Domain) (key : Key)
    (w : EnvelopeWire Tag) (kind : UInt8) (metadata payload : List UInt8)
    (hdec : decodeEnvelope mac limits domain key w = .ok (kind, metadata, payload)) :
    w.totalBytes ≤ limits.maxFrameBytes ∧
      metadata.length ≤ limits.maxMetadataBytes ∧
      payload.length ≤ limits.maxPayloadBytes ∧
      w.metadataLen = metadata.length ∧
      w.payloadLen = payload.length := by
  unfold decodeEnvelope at hdec
  split at hdec
  · cases hdec
  · split at hdec
    · cases hdec
    · split at hdec
      · cases hdec
      · split at hdec
        · cases hdec
        · split at hdec
          · cases hdec
          · split at hdec
            · cases hdec
            · split at hdec
              · cases hdec
              · split at hdec
                · cases hdec
                · rename_i hframe _ _ hmlen hplen hm hp _
                  injection hdec with htuple
                  simp only [Prod.mk.injEq] at htuple
                  obtain ⟨_, hmeta, hpay⟩ := htuple
                  subst hmeta
                  subst hpay
                  simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hm hp
                  omega

/-- `decodeEnvelope` succeeds only when the wire tag equals the expected MAC under `(domain, key)`. -/
theorem decodeEnvelope_authenticated {Key Tag : Type} [DecidableEq Tag]
    (mac : MacFn Key Tag) (limits : Limits) (domain : Domain) (key : Key)
    (w : EnvelopeWire Tag) (kind : UInt8) (metadata payload : List UInt8)
    (hdec : decodeEnvelope mac limits domain key w = .ok (kind, metadata, payload)) :
    w.tag = mac domain key kind metadata.length payload.length metadata payload := by
  unfold decodeEnvelope at hdec
  split at hdec
  · cases hdec
  · split at hdec
    · cases hdec
    · split at hdec
      · cases hdec
      · split at hdec
        · cases hdec
        · split at hdec
          · cases hdec
          · split at hdec
            · cases hdec
            · split at hdec
              · cases hdec
              · split at hdec
                · cases hdec
                · injection hdec with htuple
                  simp only [Prod.mk.injEq] at htuple
                  obtain ⟨hkind, hmeta, hpay⟩ := htuple
                  subst hkind
                  subst hmeta
                  subst hpay
                  rename_i hm hp htag
                  simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hm hp htag
                  rw [← hm, ← hp]
                  exact htag

/-- A frame encoded with `senderKey` is rejected with `.unauthenticated` by a receiver configured
with a different `receiverKey` whenever the MAC distinguishes the two keys on that message. -/
theorem decodeEnvelope_rejects_wrong_key {Key Tag : Type} [DecidableEq Tag]
    (mac : MacFn Key Tag) (limits : Limits) (domain : Domain)
    (senderKey receiverKey : Key) (kind : UInt8) (metadata payload : List UInt8)
    (w : EnvelopeWire Tag)
    (henc : encodeEnvelope mac limits domain senderKey kind metadata payload = .ok w)
    (hne :
      mac domain senderKey kind metadata.length payload.length metadata payload ≠
        mac domain receiverKey kind metadata.length payload.length metadata payload) :
    decodeEnvelope mac limits domain receiverKey w = .error .unauthenticated := by
  unfold encodeEnvelope at henc
  split at henc
  · cases henc
  · split at henc
    · cases henc
    · split at henc
      · cases henc
      · injection henc with hw
        subst hw
        unfold decodeEnvelope EnvelopeWire.totalBytes
        have htag : (mac domain senderKey kind metadata.length payload.length metadata payload !=
            mac domain receiverKey kind metadata.length payload.length metadata payload) = true :=
          bne_iff_ne.mpr hne
        simp_all only [gt_iff_lt, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]

/-- Domain separation prevents replaying a valid request frame as a response frame (or vice versa)
whenever the MAC distinguishes `.request` from `.response`. -/
theorem decodeEnvelope_rejects_domain_swap {Key Tag : Type} [DecidableEq Tag]
    (mac : MacFn Key Tag) (limits : Limits) (senderDomain receiverDomain : Domain)
    (key : Key) (kind : UInt8) (metadata payload : List UInt8)
    (w : EnvelopeWire Tag)
    (henc : encodeEnvelope mac limits senderDomain key kind metadata payload = .ok w)
    (hne :
      mac senderDomain key kind metadata.length payload.length metadata payload ≠
        mac receiverDomain key kind metadata.length payload.length metadata payload) :
    decodeEnvelope mac limits receiverDomain key w = .error .unauthenticated := by
  unfold encodeEnvelope at henc
  split at henc
  · cases henc
  · split at henc
    · cases henc
    · split at henc
      · cases henc
      · injection henc with hw
        subst hw
        unfold decodeEnvelope EnvelopeWire.totalBytes
        have htag : (mac senderDomain key kind metadata.length payload.length metadata payload !=
            mac receiverDomain key kind metadata.length payload.length metadata payload) = true :=
          bne_iff_ne.mpr hne
        simp_all only [gt_iff_lt, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]

/-- Exact tensor payload codec: stores raw 32-bit IEEE 754 float bit patterns (`UInt32`) with their
wire byte count `values.length * 4`. -/
structure TensorPayload where
  words : List UInt32
  byteLen : Nat
  deriving DecidableEq, Repr

/-- Errors returned by `encodeTensor` and `decodeTensor`. -/
inductive TensorWireError where
  | shapeMismatch
  | payloadTooLarge (actual limit : Nat)
  | byteLengthMismatch (actual expected : Nat)
  deriving DecidableEq, Repr

/-- Pure model of `encode_f32_tensor`: verifies `values.length = rows * width` and the byte limit
`rows * width * 4 ≤ maxPayloadBytes`, then copies the exact 32-bit float words without rounding. -/
def encodeTensor (rows width : Nat) (values : List UInt32) (maxPayloadBytes : Nat) :
    Except TensorWireError TensorPayload :=
  if values.length != rows * width then
    .error .shapeMismatch
  else if rows * width * 4 > maxPayloadBytes then
    .error (.payloadTooLarge (rows * width * 4) maxPayloadBytes)
  else
    .ok { words := values, byteLen := rows * width * 4 }

/-- Pure model of `decode_f32_tensor`: verifies `rows * width * 4 ≤ maxPayloadBytes` and that the
wire payload holds exactly `rows * width * 4` bytes (`rows * width` 32-bit words). -/
def decodeTensor (rows width : Nat) (payload : TensorPayload) (maxPayloadBytes : Nat) :
    Except TensorWireError (List UInt32) :=
  if rows * width * 4 > maxPayloadBytes then
    .error (.payloadTooLarge (rows * width * 4) maxPayloadBytes)
  else if payload.byteLen != rows * width * 4 then
    .error (.byteLengthMismatch payload.byteLen (rows * width * 4))
  else if payload.words.length != rows * width then
    .error (.byteLengthMismatch payload.byteLen (rows * width * 4))
  else
    .ok payload.words

/-- Every tensor encoded by `encodeTensor` decodes to the bit-identical list of 32-bit float words. -/
theorem decodeTensor_encodeTensor (rows width : Nat) (values : List UInt32)
    (maxPayloadBytes : Nat) (p : TensorPayload)
    (henc : encodeTensor rows width values maxPayloadBytes = .ok p) :
    decodeTensor rows width p maxPayloadBytes = .ok values := by
  unfold encodeTensor at henc
  split at henc
  · cases henc
  · split at henc
    · cases henc
    · injection henc with hp
      subst hp
      unfold decodeTensor
      simp_all only [gt_iff_lt, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]

/-- `decodeTensor` succeeds only when the payload's byte length is `rows * width * 4`, within
`maxPayloadBytes`, and the returned words have length `rows * width` and equal the input words
bit for bit. -/
theorem decodeTensor_exact (rows width : Nat) (payload : TensorPayload)
    (maxPayloadBytes : Nat) (out : List UInt32)
    (hdec : decodeTensor rows width payload maxPayloadBytes = .ok out) :
    out = payload.words ∧
      out.length = rows * width ∧
      payload.byteLen = rows * width * 4 ∧
      payload.byteLen ≤ maxPayloadBytes := by
  unfold decodeTensor at hdec
  split at hdec
  · cases hdec
  · split at hdec
    · cases hdec
    · split at hdec
      · cases hdec
      · injection hdec with hout
        subst hout
        rename_i hlim hbytes hwords
        simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hbytes hwords
        refine ⟨rfl, hwords, hbytes, ?_⟩
        rw [hbytes]
        omega

/-- 64-bit modulus (`2^64`), modelling `usize` on 64-bit targets. -/
def modulus64 : Nat := 2 ^ 64

/-- Unchecked 64-bit tensor byte calculation `(rows * width * 4) % 2^64`. -/
def wrappingTensorBytes64 (rows width : Nat) : Nat :=
  (((rows * width) % modulus64) * 4) % modulus64

/-- Checked 64-bit tensor byte calculation mirroring
`rows.checked_mul(width).and_then(|n| n.checked_mul(4))`. -/
def checkedTensorBytes64 (rows width : Nat) : Option Nat :=
  if rows * width >= modulus64 then
    none
  else if rows * width * 4 >= modulus64 then
    none
  else
    some (rows * width * 4)

/-- Counterexample: for `rows = 2^31` and `width = 2^31`, unchecked 64-bit multiplication wraps
`rows * width * 4 = 2^64` to `0` bytes, even though the tensor claims `2^62` elements. -/
theorem wrapped_tensor_overflow :
    wrappingTensorBytes64 (2 ^ 31) (2 ^ 31) = 0 := by
  decide

/-- `checkedTensorBytes64` rejects the `(2^31, 2^31)` overflow counterexample, and whenever it
returns `some n`, `n` equals the exact mathematical byte count `rows * width * 4`. -/
theorem checked_rejects_tensor_overflow :
    checkedTensorBytes64 (2 ^ 31) (2 ^ 31) = none ∧
      ∀ rows width n, checkedTensorBytes64 rows width = some n → n = rows * width * 4 := by
  refine ⟨by decide, ?_⟩
  intro rows width n h
  unfold checkedTensorBytes64 at h
  split at h
  · cases h
  · split at h
    · cases h
    · injection h with hn
      omega

end Kadmium.Native.SessionWire
