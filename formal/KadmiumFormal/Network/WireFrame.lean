import KadmiumFormal.Bytes

/-!
# KDM2 wire framing

Models `WireFrame::encode_with_request_id`, `WireFrame::parse_header` and `WireFrame::decode` in
`crates/kadmium-network/src/lib.rs`, the framing every Kadmium TCP peer speaks:

```text
offset  size  field
     0     4  magic "KDM2"
     4     1  kind (WireFrameKind as u8, 1 ..= 15)
     5     4  epoch, u32 LE
     9     8  request_id, u64 LE
    17     4  payload length, u32 LE, at most 2^30
    21     n  payload
```

and the legacy 13-byte `KDMF` header without a request id. `read_from_stream` parses the same
header and allocates the payload buffer only after `parse_header` has accepted the length, so a
frame the decoder rejects is never allocated for.

Results:

* `decode_encode`: every frame with a legal kind, an epoch below `2^32`, a request id below `2^64`
  and a payload of at most `2^30` bytes decodes back to itself from any byte string that starts
  with its encoding. `decode_encodeLegacy` is the same for `KDMF` frames, which read as request
  id `0`.
* `decode_shortMagic`, `decode_badMagic`, `decode_shortHeader`, `decode_badKind`,
  `decode_oversize`, `decode_truncated`: the decoder rejects a short input, an unknown magic, a
  short header, a kind byte outside `1 ..= 15`, a declared length above `2^30` and a payload
  shorter than declared. `decode_oversize` holds for every payload, so an oversize length is
  refused before any payload byte is looked at.
* `decode_ok_payload_le`: whatever decodes carries at most `2^30` payload bytes.
* `encode_not_total` (**finding**): `encode` writes `payload.len() as u32` and never checks the
  `2^30` cap, so a frame with more than `2^30` payload bytes encodes to something that does not
  decode back to it: either the decoder refuses the length, or the length wrapped and the decoded
  payload is shorter. The cap is enforced on the receiving side only.

Modelling choices: bytes are `Nat` below 256 (`KadmiumFormal.Bytes`); the header parser reads
fields by peeling the prefix, which is `header[offset ..]` of the Rust; error strings are the
constructors of `DecodeError`.
-/

namespace Kadmium.Network

open Kadmium.Bytes

/-- `b"KDM2"`. -/
def MAGIC_V2 : List Nat := [0x4B, 0x44, 0x4D, 0x32]
/-- `b"KDMF"`. -/
def MAGIC_V1 : List Nat := [0x4B, 0x44, 0x4D, 0x46]
def HEADER_V2 : Nat := 21
def HEADER_V1 : Nat := 13
/-- `KDM_FRAME_MAX_PAYLOAD_BYTES`. -/
def MAX_PAYLOAD : Nat := 2 ^ 30

structure Frame where
  kind : Nat
  epoch : Nat
  requestId : Nat
  payload : List Nat
  deriving DecidableEq, Repr

inductive DecodeError where
  /-- `"WireFrame shorter than 4-byte magic"` -/
  | shortMagic
  /-- `"Invalid WireFrame magic (expected 'KDM2' or legacy 'KDMF')"` -/
  | badMagic
  /-- `"WireFrame shorter than {header_len}-byte header"` -/
  | shortHeader
  /-- `"Unknown WireFrameKind byte: {}"` -/
  | badKind
  /-- `"WireFrame payload length {len} exceeds the {max}-byte limit"` -/
  | oversize
  /-- `"WireFrame payload truncated"` -/
  | truncated
  deriving DecidableEq, Repr

/-- `WireFrameKind::from_u8` accepts `1 ..= 15`. -/
def kindOk (k : Nat) : Bool := 1 ≤ k && k ≤ 15

/-- `WireFrame::encode_with_request_id`. The length field is `payload.len() as u32`. -/
def encode (f : Frame) : List Nat :=
  MAGIC_V2 ++ f.kind :: (encodeLE 4 f.epoch ++ (encodeLE 8 f.requestId ++
    (encodeLE 4 f.payload.length ++ f.payload)))

/-- The legacy `KDMF` frame the decoder still accepts. -/
def encodeLegacy (kind epoch : Nat) (payload : List Nat) : List Nat :=
  MAGIC_V1 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 4 payload.length ++ payload))

/-- `WireFrame::header_len_for_magic`. -/
def headerLen (magic : List Nat) : Except DecodeError Nat :=
  if magic = MAGIC_V2 then .ok HEADER_V2
  else if magic = MAGIC_V1 then .ok HEADER_V1
  else .error .badMagic

/-- The header: `(header_len, kind, epoch, request_id, payload_len)`. This is the magic and
header-length checks of `decode` followed by `parse_header`, minus its length cap, which
`decode` below applies next. -/
def parseHeader (bytes : List Nat) : Except DecodeError (Nat × Nat × Nat × Nat × Nat) :=
  if bytes.length < 4 then .error .shortMagic
  else
    match headerLen (bytes.take 4) with
    | .error e => .error e
    | .ok hl =>
      if bytes.length < hl then .error .shortHeader
      else
        let afterMagic := bytes.drop 4
        let kind := afterMagic.headD 0
        if !kindOk kind then .error .badKind
        else
          let afterKind := afterMagic.drop 1
          let epoch := decodeLE (afterKind.take 4)
          let afterEpoch := afterKind.drop 4
          if hl = HEADER_V2 then
            .ok (hl, kind, epoch, decodeLE (afterEpoch.take 8), decodeLE ((afterEpoch.drop 8).take 4))
          else .ok (hl, kind, epoch, 0, decodeLE (afterEpoch.take 4))

/-- `WireFrame::decode`. -/
def decode (bytes : List Nat) : Except DecodeError Frame :=
  match parseHeader bytes with
  | .error e => .error e
  | .ok (hl, kind, epoch, requestId, len) =>
    if MAX_PAYLOAD < len then .error .oversize
    else if bytes.length < hl + len then .error .truncated
    else .ok ⟨kind, epoch, requestId, (bytes.drop hl).take len⟩

theorem length_encode (f : Frame) : (encode f).length = HEADER_V2 + f.payload.length := by
  simp [encode, MAGIC_V2, HEADER_V2]
  omega

theorem length_encodeLegacy (kind epoch : Nat) (payload : List Nat) :
    (encodeLegacy kind epoch payload).length = HEADER_V1 + payload.length := by
  simp [encodeLegacy, MAGIC_V1, HEADER_V1]
  omega

theorem headerLen_v2 : headerLen MAGIC_V2 = .ok HEADER_V2 := by simp [headerLen]

theorem headerLen_v1 : headerLen MAGIC_V1 = .ok HEADER_V1 := by
  simp [headerLen, MAGIC_V1, MAGIC_V2]

/-- The header of a `KDM2` frame followed by anything. -/
theorem parseHeader_v2 (kind epoch requestId len : Nat) (tail : List Nat) (hk : kindOk kind = true)
    (he : epoch < 2 ^ 32) (hr : requestId < 2 ^ 64) (hl : len < 2 ^ 32) :
    parseHeader (MAGIC_V2 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 8 requestId ++
      (encodeLE 4 len ++ tail)))) = .ok (HEADER_V2, kind, epoch, requestId, len) := by
  have hlen : (MAGIC_V2 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 8 requestId ++
      (encodeLE 4 len ++ tail)))).length = HEADER_V2 + tail.length := by
    simp [MAGIC_V2, HEADER_V2]; omega
  have h4 : ¬ (MAGIC_V2 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 8 requestId ++
      (encodeLE 4 len ++ tail)))).length < 4 := by rw [hlen]; unfold HEADER_V2; omega
  have h21 : ¬ (MAGIC_V2 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 8 requestId ++
      (encodeLE 4 len ++ tail)))).length < HEADER_V2 := by rw [hlen]; omega
  have htake : (MAGIC_V2 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 8 requestId ++
      (encodeLE 4 len ++ tail)))).take 4 = MAGIC_V2 := by simp [MAGIC_V2]
  have hdrop : (MAGIC_V2 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 8 requestId ++
      (encodeLE 4 len ++ tail)))).drop 4 = kind :: (encodeLE 4 epoch ++ (encodeLE 8 requestId ++
      (encodeLE 4 len ++ tail))) := by simp [MAGIC_V2]
  unfold parseHeader
  rw [htake, headerLen_v2]
  simp only [h4, h21, hdrop, hk, ↓reduceIte, Bool.not_true, Bool.false_eq_true, List.headD_cons,
    List.drop_succ_cons, List.drop_zero, List.take_left', length_encodeLE, List.drop_left',
    decodeLE_encodeLE (lt_pow_four he), decodeLE_encodeLE (lt_pow_eight hr),
    decodeLE_encodeLE (lt_pow_four hl)]

/-- The header of a legacy `KDMF` frame followed by anything. -/
theorem parseHeader_v1 (kind epoch len : Nat) (tail : List Nat) (hk : kindOk kind = true)
    (he : epoch < 2 ^ 32) (hl : len < 2 ^ 32) :
    parseHeader (MAGIC_V1 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 4 len ++ tail))) =
      .ok (HEADER_V1, kind, epoch, 0, len) := by
  have hlen : (MAGIC_V1 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 4 len ++ tail))).length =
      HEADER_V1 + tail.length := by
    simp [MAGIC_V1, HEADER_V1]; omega
  have h4 : ¬ (MAGIC_V1 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 4 len ++ tail))).length < 4 := by
    rw [hlen]; unfold HEADER_V1; omega
  have h13 : ¬ (MAGIC_V1 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 4 len ++ tail))).length <
      HEADER_V1 := by rw [hlen]; omega
  have htake : (MAGIC_V1 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 4 len ++ tail))).take 4 =
      MAGIC_V1 := by simp [MAGIC_V1]
  have hdrop : (MAGIC_V1 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 4 len ++ tail))).drop 4 =
      kind :: (encodeLE 4 epoch ++ (encodeLE 4 len ++ tail)) := by simp [MAGIC_V1]
  have hne : ¬ HEADER_V1 = HEADER_V2 := by decide
  unfold parseHeader
  rw [htake, headerLen_v1]
  simp only [h4, h13, hdrop, hk, hne, ↓reduceIte, Bool.not_true, Bool.false_eq_true,
    List.headD_cons, List.drop_succ_cons, List.drop_zero, List.take_left', length_encodeLE,
    List.drop_left', decodeLE_encodeLE (lt_pow_four he), decodeLE_encodeLE (lt_pow_four hl)]

/-- **Round trip.** A legal frame decodes from any byte string that begins with its encoding. -/
theorem decode_encode (f : Frame) (rest : List Nat) (hk : kindOk f.kind = true)
    (he : f.epoch < 2 ^ 32) (hr : f.requestId < 2 ^ 64) (hp : f.payload.length ≤ MAX_PAYLOAD) :
    decode (encode f ++ rest) = .ok f := by
  have hl32 : f.payload.length < 2 ^ 32 := by unfold MAX_PAYLOAD at hp; omega
  have hbytes : encode f ++ rest = MAGIC_V2 ++ f.kind :: (encodeLE 4 f.epoch ++
      (encodeLE 8 f.requestId ++ (encodeLE 4 f.payload.length ++ (f.payload ++ rest)))) := by
    simp [encode]
  have hdrop : (encode f ++ rest).drop HEADER_V2 = f.payload ++ rest := by
    simp [encode, MAGIC_V2, HEADER_V2, encodeLE]
  have hno : ¬ MAX_PAYLOAD < f.payload.length := Nat.not_lt.mpr hp
  have htr : ¬ (encode f ++ rest).length < HEADER_V2 + f.payload.length := by
    rw [List.length_append, length_encode]; omega
  unfold decode
  rw [hbytes, parseHeader_v2 _ _ _ _ _ hk he hr hl32, ← hbytes]
  simp only [hno, htr, ↓reduceIte, hdrop, List.take_left']

/-- Legacy `KDMF` frames decode with request id `0`. -/
theorem decode_encodeLegacy (kind epoch : Nat) (payload rest : List Nat) (hk : kindOk kind = true)
    (he : epoch < 2 ^ 32) (hp : payload.length ≤ MAX_PAYLOAD) :
    decode (encodeLegacy kind epoch payload ++ rest) = .ok ⟨kind, epoch, 0, payload⟩ := by
  have hl32 : payload.length < 2 ^ 32 := by unfold MAX_PAYLOAD at hp; omega
  have hbytes : encodeLegacy kind epoch payload ++ rest = MAGIC_V1 ++ kind :: (encodeLE 4 epoch ++
      (encodeLE 4 payload.length ++ (payload ++ rest))) := by
    simp [encodeLegacy]
  have hdrop : (encodeLegacy kind epoch payload ++ rest).drop HEADER_V1 = payload ++ rest := by
    simp [encodeLegacy, MAGIC_V1, HEADER_V1, encodeLE]
  have hno : ¬ MAX_PAYLOAD < payload.length := Nat.not_lt.mpr hp
  have htr : ¬ (encodeLegacy kind epoch payload ++ rest).length < HEADER_V1 + payload.length := by
    rw [List.length_append, length_encodeLegacy]; omega
  unfold decode
  rw [hbytes, parseHeader_v1 _ _ _ _ hk he hl32, ← hbytes]
  simp only [hno, htr, ↓reduceIte, hdrop, List.take_left']

/-! ## Rejections -/

theorem decode_shortMagic {bytes : List Nat} (h : bytes.length < 4) :
    decode bytes = .error .shortMagic := by
  simp [decode, parseHeader, h]

theorem decode_badMagic {bytes : List Nat} (h4 : 4 ≤ bytes.length) (h2 : bytes.take 4 ≠ MAGIC_V2)
    (h1 : bytes.take 4 ≠ MAGIC_V1) : decode bytes = .error .badMagic := by
  have : ¬ bytes.length < 4 := Nat.not_lt.mpr h4
  simp [decode, parseHeader, headerLen, this, h2, h1]

theorem decode_shortHeader {bytes : List Nat} (hm : bytes.take 4 = MAGIC_V2)
    (h : bytes.length < HEADER_V2) : decode bytes = .error .shortHeader := by
  have h4 : ¬ bytes.length < 4 := by
    intro hlt
    have := congrArg List.length hm
    simp [MAGIC_V2] at this
    omega
  unfold decode parseHeader
  rw [hm, headerLen_v2]
  simp [h4, h]

theorem decode_badKind {bytes : List Nat} (hm : bytes.take 4 = MAGIC_V2)
    (h : ¬ bytes.length < HEADER_V2) (hk : kindOk ((bytes.drop 4).headD 0) = false) :
    decode bytes = .error .badKind := by
  have h4 : ¬ bytes.length < 4 := by unfold HEADER_V2 at h; omega
  unfold decode parseHeader
  rw [hm, headerLen_v2]
  simp only [h4, h, hk, ↓reduceIte, Bool.not_false]

/-- A declared length above the cap is refused whatever follows the header, including nothing. -/
theorem decode_oversize (kind epoch requestId len : Nat) (payload : List Nat)
    (hk : kindOk kind = true) (he : epoch < 2 ^ 32) (hr : requestId < 2 ^ 64)
    (hlen : MAX_PAYLOAD < len) (h32 : len < 2 ^ 32) :
    decode (MAGIC_V2 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 8 requestId ++
      (encodeLE 4 len ++ payload)))) = .error .oversize := by
  unfold decode
  rw [parseHeader_v2 _ _ _ _ _ hk he hr h32]
  simp only [hlen, ↓reduceIte]

/-- A payload shorter than its declared length is refused. -/
theorem decode_truncated (kind epoch requestId len : Nat) (payload : List Nat)
    (hk : kindOk kind = true) (he : epoch < 2 ^ 32) (hr : requestId < 2 ^ 64)
    (hlen : len ≤ MAX_PAYLOAD) (hshort : payload.length < len) :
    decode (MAGIC_V2 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 8 requestId ++
      (encodeLE 4 len ++ payload)))) = .error .truncated := by
  have h32 : len < 2 ^ 32 := by unfold MAX_PAYLOAD at hlen; omega
  have hl : (MAGIC_V2 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 8 requestId ++
      (encodeLE 4 len ++ payload)))).length = HEADER_V2 + payload.length := by
    simp [MAGIC_V2, HEADER_V2]; omega
  have htr : (MAGIC_V2 ++ kind :: (encodeLE 4 epoch ++ (encodeLE 8 requestId ++
      (encodeLE 4 len ++ payload)))).length < HEADER_V2 + len := by rw [hl]; omega
  have hno : ¬ MAX_PAYLOAD < len := Nat.not_lt.mpr hlen
  unfold decode
  rw [parseHeader_v2 _ _ _ _ _ hk he hr h32]
  simp only [hno, htr, ↓reduceIte]

/-- Whatever decodes carries at most `MAX_PAYLOAD` bytes. -/
theorem decode_ok_payload_le {bytes : List Nat} {g : Frame} (h : decode bytes = .ok g) :
    g.payload.length ≤ MAX_PAYLOAD := by
  unfold decode at h
  split at h
  · exact absurd h (by simp)
  · split at h
    · exact absurd h (by simp)
    · split at h
      · exact absurd h (by simp)
      · rename_i hlen _
        simp only [Except.ok.injEq] at h
        subst h
        simp only [List.length_take]
        omega

/-- **Finding.** `encode` does not enforce the cap: a frame with more than `2^30` payload bytes
never round-trips. -/
theorem encode_not_total (f : Frame) (h : MAX_PAYLOAD < f.payload.length) :
    decode (encode f) ≠ .ok f := by
  intro hd
  have := decode_ok_payload_le hd
  omega

end Kadmium.Network
