/-!
# Powers of two

`usize::next_power_of_two(n)` is the smallest power of two that is at least `n` (and `1` for
`n = 0`). Kadmium's reduction tree and its streaming reducer size themselves with it. The model
writes it as `2 ^ ceilLog2 n` and proves that this is exactly the documented value: a power of
two, at least `n`, and no larger than any other power of two that is at least `n`.

The model uses unbounded `Nat`. Rust's `usize` overflows past `2 ^ 63` leaves; no allocation that
large exists, so the proofs assume it does not happen.
-/

namespace Kadmium

/-- The exponent of `n.next_power_of_two()`: the least `k` with `n ≤ 2 ^ k`. -/
def ceilLog2 (n : Nat) : Nat :=
  if n ≤ 1 then 0 else ceilLog2 ((n + 1) / 2) + 1
termination_by n
decreasing_by omega

/-- `usize::next_power_of_two`, as the model writes it. -/
def nextPow2 (n : Nat) : Nat := 2 ^ ceilLog2 n

theorem le_two_pow_ceilLog2 (n : Nat) : n ≤ 2 ^ ceilLog2 n := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    unfold ceilLog2
    split
    · simp only [Nat.pow_zero]; omega
    · have h := ih ((n + 1) / 2) (by omega)
      rw [Nat.pow_succ]
      omega

theorem ceilLog2_le_of_le_two_pow {n k : Nat} (h : n ≤ 2 ^ k) : ceilLog2 n ≤ k := by
  induction n using Nat.strongRecOn generalizing k with
  | ind n ih =>
    unfold ceilLog2
    split
    · omega
    · cases k with
      | zero => simp at h; omega
      | succ k =>
        rw [Nat.pow_succ] at h
        have := ih ((n + 1) / 2) (by omega) (k := k) (by omega)
        omega

/-- `ceilLog2 n` is the least exponent whose power of two covers `n`. -/
theorem ceilLog2_le_iff (n k : Nat) : ceilLog2 n ≤ k ↔ n ≤ 2 ^ k := by
  constructor
  · intro h
    exact Nat.le_trans (le_two_pow_ceilLog2 n) (Nat.pow_le_pow_right (by decide) h)
  · exact ceilLog2_le_of_le_two_pow

/-- The documented contract of `usize::next_power_of_two`: a power of two, at least `n`, and the
smallest such. -/
theorem nextPow2_spec (n : Nat) :
    (∃ k, nextPow2 n = 2 ^ k) ∧ n ≤ nextPow2 n ∧ ∀ k, n ≤ 2 ^ k → nextPow2 n ≤ 2 ^ k :=
  ⟨⟨ceilLog2 n, rfl⟩, le_two_pow_ceilLog2 n,
    fun _ hk => Nat.pow_le_pow_right (by decide) (ceilLog2_le_of_le_two_pow hk)⟩

theorem ceilLog2_mono {a b : Nat} (h : a ≤ b) : ceilLog2 a ≤ ceilLog2 b :=
  ceilLog2_le_of_le_two_pow (Nat.le_trans h (le_two_pow_ceilLog2 b))

theorem ceilLog2_two_pow (k : Nat) : ceilLog2 (2 ^ k) = k := by
  apply Nat.le_antisymm (ceilLog2_le_of_le_two_pow (Nat.le_refl _))
  exact (Nat.pow_le_pow_iff_right (by decide)).mp (le_two_pow_ceilLog2 (2 ^ k))

theorem ceilLog2_pos {n : Nat} (h : 2 ≤ n) : 0 < ceilLog2 n := by
  unfold ceilLog2
  split <;> omega

/-- For `n ≥ 2`, half of `n.next_power_of_two()` is strictly below `n`. -/
theorem half_nextPow2_lt {n : Nat} (h : 2 ≤ n) : 2 ^ (ceilLog2 n - 1) < n := by
  have hpos := ceilLog2_pos h
  apply Nat.lt_of_not_le
  intro hle
  have := ceilLog2_le_of_le_two_pow hle
  omega

end Kadmium
