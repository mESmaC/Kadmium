/-!
# XOR page parity

Models `SwarmRaidStriper::encode_xor_parity_page` and `reconstruct_missing_page_xor` in
`crates/kadmium-network/src/raid_striper.rs`: the parity of a stripe of equal-length pages is the
byte-wise XOR of all of them, and a missing page is rebuilt by XOR-ing the parity with every other
page.

Modelling choices:

* A page is a function from byte position to byte value. The Rust asserts that every page in a
  stripe has the same length and zips byte by byte; positions past the common length are never
  read, so a model over total functions loses nothing.
* `reconstruct` skips the page at `missing_index` and panics if any other page is absent; the model
  takes the present pages (`List.eraseIdx`) and so covers every call that does not panic.

Result: `reconstruct_parity` states that for every stripe and every missing index, the rebuilt
page is bit for bit the page that was lost. `parity_isByte` states that the parity of bytes is a
byte. Nothing about the hash or the transfer is assumed; this is the GF(2) identity the striper
relies on when one of `k` seeders drops out.
-/

namespace Kadmium.Network

/-- Byte-wise XOR of two pages. -/
def pxor (p q : Nat → Nat) : Nat → Nat := fun j => p j ^^^ q j

def zeroPage : Nat → Nat := fun _ => 0

/-- `encode_xor_parity_page`. -/
def parity (pages : List (Nat → Nat)) : Nat → Nat := pages.foldl pxor zeroPage

/-- `reconstruct_missing_page_xor`: the parity XOR every page other than `missing`. -/
def reconstruct (pages : List (Nat → Nat)) (missing : Nat) (par : Nat → Nat) : Nat → Nat :=
  (pages.eraseIdx missing).foldl pxor par

/-- XOR of a list of bytes. -/
def xorAll (l : List Nat) : Nat := l.foldl (· ^^^ ·) 0

theorem foldl_xor_shift (a : Nat) (l : List Nat) : l.foldl (· ^^^ ·) a = a ^^^ xorAll l := by
  induction l generalizing a with
  | nil => simp [xorAll]
  | cons x rest ih =>
    simp only [List.foldl_cons, xorAll]
    rw [ih (a ^^^ x), ih (0 ^^^ x), Nat.zero_xor, Nat.xor_assoc]

theorem xorAll_append (a b : List Nat) : xorAll (a ++ b) = xorAll a ^^^ xorAll b := by
  show (a ++ b).foldl (· ^^^ ·) 0 = _
  rw [List.foldl_append]
  exact foldl_xor_shift _ _

theorem xorAll_cons (x : Nat) (l : List Nat) : xorAll (x :: l) = x ^^^ xorAll l := by
  show (x :: l).foldl (· ^^^ ·) 0 = _
  rw [List.foldl_cons, foldl_xor_shift, Nat.zero_xor]

theorem xor_rotate (t x d : Nat) : t ^^^ (x ^^^ d) = (t ^^^ d) ^^^ x := by
  rw [Nat.xor_comm x d, Nat.xor_assoc]

theorem xor_cancel_left (e x : Nat) : (e ^^^ x) ^^^ e = x := by
  rw [Nat.xor_comm e x, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- A fold of page XORs, read at one position, is the XOR of that position's bytes. -/
theorem foldl_pxor_apply (acc : Nat → Nat) (pages : List (Nat → Nat)) (j : Nat) :
    (pages.foldl pxor acc) j = acc j ^^^ xorAll (pages.map (· j)) := by
  induction pages generalizing acc with
  | nil => simp [xorAll]
  | cons p rest ih =>
    simp only [List.foldl_cons, List.map_cons]
    rw [ih, xorAll_cons, pxor, Nat.xor_assoc]

theorem map_eraseIdx_apply (pages : List (Nat → Nat)) (j : Nat) :
    ∀ m, (pages.eraseIdx m).map (· j) = (pages.map (· j)).eraseIdx m := by
  induction pages with
  | nil => intro m; simp
  | cons p rest ih =>
    intro m
    cases m with
    | zero => simp
    | succ m => simp [List.eraseIdx_cons_succ, ih]

/-- Removing one byte from a XOR and XOR-ing it back restores the whole. -/
theorem xorAll_eraseIdx (l : List Nat) (m : Nat) (hm : m < l.length) :
    xorAll l = xorAll (l.eraseIdx m) ^^^ l[m] := by
  have hsplit := congrArg xorAll (List.take_append_drop m l)
  rw [xorAll_append, List.drop_eq_getElem_cons hm, xorAll_cons] at hsplit
  rw [← hsplit, List.eraseIdx_eq_take_drop_succ, xorAll_append]
  exact xor_rotate _ _ _

/-- **Recovery.** For every stripe and every index below the stripe, the parity XOR the other
pages is the missing page, byte for byte. -/
theorem reconstruct_parity (pages : List (Nat → Nat)) {missing : Nat}
    (hm : missing < pages.length) :
    reconstruct pages missing (parity pages) = pages[missing] := by
  funext j
  unfold reconstruct parity
  rw [foldl_pxor_apply, foldl_pxor_apply]
  simp only [zeroPage, Nat.zero_xor]
  have hm' : missing < (pages.map (· j)).length := by simpa using hm
  rw [map_eraseIdx_apply, xorAll_eraseIdx _ missing hm', xor_cancel_left]
  simp

/-- The parity of bytes is a byte. -/
theorem parity_isByte (pages : List (Nat → Nat)) (j : Nat) (h : ∀ p ∈ pages, p j < 256) :
    parity pages j < 256 := by
  unfold parity
  rw [foldl_pxor_apply]
  simp only [zeroPage, Nat.zero_xor]
  have : ∀ l : List Nat, (∀ b ∈ l, b < 256) → xorAll l < 256 := by
    intro l
    induction l with
    | nil => intro _; simp [xorAll]
    | cons b rest ih =>
      intro hl
      rw [xorAll_cons]
      have h1 := hl b (by simp)
      have h2 := ih (fun x hx => hl x (by simp [hx]))
      exact Nat.xor_lt_two_pow (n := 8) h1 h2
  apply this
  intro b hb
  rw [List.mem_map] at hb
  obtain ⟨p, hp, rfl⟩ := hb
  exact h p hp

end Kadmium.Network
