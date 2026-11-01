/-!
# Integer-exact INT8 accumulation

Models `int8_sum_squares` and `int8_dot` in `crates/kadmium-runtime/src/repops.rs`:

```rust
codes.iter().map(|&q| (q as i64) * (q as i64)).sum()
a.iter().zip(b).map(|(&x, &y)| (x as i64) * (y as i64)).sum()
```

The RepOps conventions say these are "accumulated in i64, exact". Exactness has two parts, and
both are proved here for every input:

* `dot_perm`: the sum is the same for every summation order, because integer addition is
  associative and commutative. A backend may chunk or reassociate the accumulation freely.
* `dot_natAbs_le`, `dot_fits_i64`: the magnitude of the sum is at most `n * 2^14` for `n`
  products of INT8 codes, so it fits an `i64` whenever `n < 2^49`. No tensor has that many
  elements, so the Rust `i64` arithmetic never wraps. The source comment's `2^47` is a weaker
  statement of the same bound.

`zip` truncates to the shorter operand, and so does the model (`List.zipWith`).
-/

namespace Kadmium.RepOps

/-- An INT8 code, `i8` widened to `Int`. -/
def IsInt8 (x : Int) : Prop := -128 ≤ x ∧ x ≤ 127

def int8Dot (a b : List Int) : Int := (List.zipWith (· * ·) a b).sum

def int8SumSquares (a : List Int) : Int := int8Dot a a

theorem natAbs_le_of_isInt8 {x : Int} (h : IsInt8 x) : x.natAbs ≤ 128 := by
  obtain ⟨h1, h2⟩ := h
  omega

theorem natAbs_mul_le {x y : Int} (hx : IsInt8 x) (hy : IsInt8 y) :
    (x * y).natAbs ≤ 2 ^ 14 := by
  rw [Int.natAbs_mul]
  have := Nat.mul_le_mul (natAbs_le_of_isInt8 hx) (natAbs_le_of_isInt8 hy)
  omega

theorem sum_natAbs_le (l : List Int) (bound : Nat) (h : ∀ x ∈ l, x.natAbs ≤ bound) :
    l.sum.natAbs ≤ l.length * bound := by
  induction l with
  | nil => simp
  | cons x rest ih =>
    simp only [List.sum_cons, List.length_cons]
    have hx := h x (by simp)
    have hr := ih (fun y hy => h y (by simp [hy]))
    have := Int.natAbs_add_le x rest.sum
    rw [Nat.succ_mul]
    omega

/-- `|Σ xᵢ yᵢ| ≤ n * 2^14` for `n = min(a.len(), b.len())` INT8 codes. -/
theorem dot_natAbs_le {a b : List Int} (ha : ∀ x ∈ a, IsInt8 x) (hb : ∀ y ∈ b, IsInt8 y) :
    (int8Dot a b).natAbs ≤ min a.length b.length * 2 ^ 14 := by
  unfold int8Dot
  rw [← List.length_zipWith (f := (· * ·)) (l₁ := a) (l₂ := b)]
  apply sum_natAbs_le
  intro z hz
  rw [List.mem_iff_getElem] at hz
  obtain ⟨i, hi, rfl⟩ := hz
  rw [List.getElem_zipWith]
  have hia : i < a.length := by rw [List.length_zipWith] at hi; omega
  have hib : i < b.length := by rw [List.length_zipWith] at hi; omega
  exact natAbs_mul_le (ha _ (List.getElem_mem hia)) (hb _ (List.getElem_mem hib))

/-- The accumulation never leaves `i64` for fewer than `2^49` products. -/
theorem dot_fits_i64 {a b : List Int} (ha : ∀ x ∈ a, IsInt8 x) (hb : ∀ y ∈ b, IsInt8 y)
    (hn : min a.length b.length < 2 ^ 49) : (int8Dot a b).natAbs < 2 ^ 63 := by
  have h1 := dot_natAbs_le ha hb
  have h2 : min a.length b.length * 2 ^ 14 < 2 ^ 49 * 2 ^ 14 :=
    Nat.mul_lt_mul_of_pos_right hn (by decide)
  have h3 : 2 ^ 49 * 2 ^ 14 = 2 ^ 63 := by decide
  omega

theorem sumSquares_fits_i64 {a : List Int} (ha : ∀ x ∈ a, IsInt8 x) (hn : a.length < 2 ^ 49) :
    (int8SumSquares a).natAbs < 2 ^ 63 :=
  dot_fits_i64 ha ha (by simpa using hn)

/-- Integer sums do not depend on the order of the summands. -/
theorem sum_perm {l l' : List Int} (h : l.Perm l') : l.sum = l'.sum := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp [ih]
  | swap x y l =>
    simp only [List.sum_cons]
    rw [← Int.add_assoc, ← Int.add_assoc, Int.add_comm y x]
  | trans _ _ ih1 ih2 => exact ih1.trans ih2

/-- Any chunking, threading or reassociation of the INT8 accumulation gives the same integer:
summing the products in any order is `int8Dot`. -/
theorem dot_perm {a b : List Int} {order : List Int}
    (h : order.Perm (List.zipWith (· * ·) a b)) : order.sum = int8Dot a b :=
  sum_perm h

end Kadmium.RepOps
