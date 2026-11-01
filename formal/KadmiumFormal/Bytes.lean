/-!
# Little-endian byte strings

The byte model shared by the wire and storage formats under `formal/`. A byte string is a
`List Nat` whose entries are below `256` (`IsBytes`), and a multi-byte integer is little-endian,
as every Kadmium codec writes it (`u32::to_le_bytes`, `u64::to_le_bytes`).

`encodeLE w n` is the `w`-byte little-endian encoding of `n`. It keeps only the low `8 * w` bits,
which is exactly what the Rust casts `n as u32` and `n as u64` do before `to_le_bytes`;
`decodeLE_encodeLE_mod` states that, and `decodeLE_encodeLE` states that the pair is lossless
below `256 ^ w`. `encodeLE_decodeLE` is the other direction: reading `w` bytes and writing them
back reproduces the bytes.
-/

namespace Kadmium.Bytes

/-- `n.to_le_bytes()` truncated to `w` bytes. -/
def encodeLE : Nat → Nat → List Nat
  | 0, _ => []
  | w + 1, n => n % 256 :: encodeLE w (n / 256)

/-- `u*::from_le_bytes` over any number of bytes. -/
def decodeLE : List Nat → Nat
  | [] => 0
  | b :: rest => b + 256 * decodeLE rest

/-- Every entry is a byte. -/
def IsBytes (l : List Nat) : Prop := ∀ b ∈ l, b < 256

theorem isBytes_append {a b : List Nat} (ha : IsBytes a) (hb : IsBytes b) : IsBytes (a ++ b) := by
  intro x hx
  rw [List.mem_append] at hx
  exact hx.elim (ha x) (hb x)

theorem isBytes_cons {b : Nat} {l : List Nat} (hb : b < 256) (hl : IsBytes l) :
    IsBytes (b :: l) := by
  intro x hx
  simp only [List.mem_cons] at hx
  rcases hx with rfl | hx
  · exact hb
  · exact hl x hx

theorem isBytes_of_append_left {a b : List Nat} (h : IsBytes (a ++ b)) : IsBytes a :=
  fun x hx => h x (List.mem_append_left _ hx)

theorem isBytes_of_append_right {a b : List Nat} (h : IsBytes (a ++ b)) : IsBytes b :=
  fun x hx => h x (List.mem_append_right _ hx)

theorem isBytes_take {l : List Nat} (h : IsBytes l) (n : Nat) : IsBytes (l.take n) :=
  fun x hx => h x (List.mem_of_mem_take hx)

theorem isBytes_drop {l : List Nat} (h : IsBytes l) (n : Nat) : IsBytes (l.drop n) :=
  fun x hx => h x (List.mem_of_mem_drop hx)

@[simp] theorem length_encodeLE (w n : Nat) : (encodeLE w n).length = w := by
  induction w generalizing n with
  | zero => rfl
  | succ w ih => simp [encodeLE, ih]

theorem encodeLE_isBytes (w n : Nat) : IsBytes (encodeLE w n) := by
  induction w generalizing n with
  | zero => intro b h; simp [encodeLE] at h
  | succ w ih =>
    intro b h
    simp only [encodeLE, List.mem_cons] at h
    rcases h with rfl | h
    · exact Nat.mod_lt _ (by decide)
    · exact ih _ b h

/-- What the truncating cast returns: the low `8 * w` bits. -/
theorem decodeLE_encodeLE_mod (w n : Nat) : decodeLE (encodeLE w n) = n % 256 ^ w := by
  induction w generalizing n with
  | zero => simp [encodeLE, decodeLE, Nat.mod_one]
  | succ w ih =>
    simp only [encodeLE, decodeLE, ih]
    rw [Nat.pow_succ, Nat.mul_comm (256 ^ w) 256, Nat.mod_mul]

/-- Below `256 ^ w` the encoding is lossless. -/
theorem decodeLE_encodeLE {w n : Nat} (h : n < 256 ^ w) : decodeLE (encodeLE w n) = n := by
  rw [decodeLE_encodeLE_mod, Nat.mod_eq_of_lt h]

theorem decodeLE_lt {l : List Nat} (h : IsBytes l) : decodeLE l < 256 ^ l.length := by
  induction l with
  | nil => simp [decodeLE]
  | cons b rest ih =>
    have hb := h b (by simp)
    have hr := ih (fun x hx => h x (by simp [hx]))
    simp only [decodeLE, List.length_cons, Nat.pow_succ]
    have : 256 * (decodeLE rest + 1) ≤ 256 * 256 ^ rest.length := Nat.mul_le_mul_left 256 hr
    omega

/-- Reading `w` bytes and writing them back reproduces them. -/
theorem encodeLE_decodeLE {l : List Nat} (h : IsBytes l) :
    encodeLE l.length (decodeLE l) = l := by
  induction l with
  | nil => rfl
  | cons b rest ih =>
    have hb := h b (by simp)
    have hr := ih (fun x hx => h x (by simp [hx]))
    simp only [List.length_cons, encodeLE, decodeLE]
    rw [Nat.add_mul_mod_self_left, Nat.add_mul_div_left _ _ (by decide : 0 < 256),
      Nat.mod_eq_of_lt hb, Nat.div_eq_of_lt hb, Nat.zero_add, hr]

theorem decodeLE_append (a b : List Nat) :
    decodeLE (a ++ b) = decodeLE a + 256 ^ a.length * decodeLE b := by
  induction a with
  | nil => simp [decodeLE]
  | cons x rest ih =>
    simp only [List.cons_append, decodeLE, ih, List.length_cons, Nat.pow_succ]
    rw [Nat.mul_add, Nat.add_assoc, Nat.mul_comm (256 ^ rest.length) 256, Nat.mul_assoc]

/-- `u32` and `u64` ranges in the base the encoding uses. -/
theorem lt_pow_four {n : Nat} (h : n < 2 ^ 32) : n < 256 ^ 4 := by
  have e : (256 : Nat) ^ 4 = 2 ^ 32 := by decide
  rw [e]; exact h

theorem lt_pow_eight {n : Nat} (h : n < 2 ^ 64) : n < 256 ^ 8 := by
  have e : (256 : Nat) ^ 8 = 2 ^ 64 := by decide
  rw [e]; exact h

/-- The encoding is injective below `256 ^ w`. -/
theorem encodeLE_inj {w a b : Nat} (ha : a < 256 ^ w) (hb : b < 256 ^ w)
    (h : encodeLE w a = encodeLE w b) : a = b := by
  have := congrArg decodeLE h
  rwa [decodeLE_encodeLE ha, decodeLE_encodeLE hb] at this

end Kadmium.Bytes
