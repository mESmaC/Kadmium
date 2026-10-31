import KadmiumFormal.Pow2

/-!
# The RepOps reduction tree

Models `tree_node` and `tree_sum` in `crates/kadmium-runtime/src/repops.rs`, and the legacy
`VerdeVerificationEngine::repops_pairwise_tree_sum` in `crates/kadmium-runtime/src/lib.rs`.

The addition is an arbitrary function `add : α → α → α`. No theorem here assumes it is
associative, commutative or has an identity. That is the point: IEEE `f32` and `f64` addition has
none of those laws, so a statement proved for every `add` holds bit for bit for floats, without a
floating-point model. Two reductions that evaluate the same `Shape` perform the same additions on
the same operands in the same positions, and return the same bits.

Results:

* `treeSum_eq_eval`: `tree_sum` evaluates one fixed `Shape`, the canonical tree.
* `canonical_leaves`: that tree reads every operand exactly once, in index order, and never reads
  past the end of the slice.
* `canonical_optimal`: among all binary reduction trees over the same `n` operands, the canonical
  tree performs the minimum number of additions (`n - 1`, which every tree performs) and has the
  minimum depth (`ceilLog2 n`, the critical path of a parallel evaluation). The fixed order costs
  nothing in work or span.
* `legacySum_eq_treeSum`: the legacy Verde engine's recursion builds the same tree, so the two
  implementations agree bit for bit on every input; its `n / 2` fallback is unreachable.
-/

namespace Kadmium.RepOps

variable {α : Type}

/-- `tree_node(values, lo, size)` with `size = 2 ^ k`. Rust halves `size`; the model counts the
halvings in `k`, which makes the recursion structural. The leaf read `values[lo]` becomes `f lo`;
`canonical_leaves` proves every such read is in bounds. -/
def treeNode (add : α → α → α) (f : Nat → α) (len lo : Nat) : Nat → α
  | 0 => f lo
  | k + 1 =>
    let left := treeNode add f len lo k
    let mid := lo + 2 ^ k
    if len ≤ mid then left else add left (treeNode add f len mid k)

/-- `tree_sum(values)`: `ZERO` for an empty slice, otherwise the tree over
`values.len().next_power_of_two()` leaves. -/
def treeSum (add : α → α → α) (zero : α) (f : Nat → α) (len : Nat) : α :=
  if len = 0 then zero else treeNode add f len 0 (ceilLog2 len)

/-! ## Reduction trees -/

/-- A binary reduction tree over operand indices. Every fixed-order reduction of a list with a
binary operation is one of these. -/
inductive Shape where
  | leaf (i : Nat)
  | node (l r : Shape)
  deriving Repr, DecidableEq

namespace Shape

def eval (add : α → α → α) (f : Nat → α) : Shape → α
  | leaf i => f i
  | node l r => add (eval add f l) (eval add f r)

/-- The operands the tree reads, left to right. -/
def leaves : Shape → List Nat
  | leaf i => [i]
  | node l r => leaves l ++ leaves r

/-- Additions performed: the internal nodes. -/
def adds : Shape → Nat
  | leaf _ => 0
  | node l r => adds l + adds r + 1

/-- Critical path in additions: the time of a parallel evaluation with unbounded workers. -/
def depth : Shape → Nat
  | leaf _ => 0
  | node l r => max (depth l) (depth r) + 1

/-- Every binary tree over `n` operands performs exactly `n - 1` additions. -/
theorem adds_add_one (t : Shape) : t.adds + 1 = t.leaves.length := by
  induction t with
  | leaf => simp [adds, leaves]
  | node l r ihl ihr => simp only [adds, leaves, List.length_append]; omega

theorem length_leaves_le (t : Shape) : t.leaves.length ≤ 2 ^ t.depth := by
  induction t with
  | leaf => simp [leaves, depth]
  | node l r ihl ihr =>
    simp only [leaves, depth, List.length_append, Nat.pow_succ]
    have hl : 2 ^ l.depth ≤ 2 ^ max l.depth r.depth :=
      Nat.pow_le_pow_right (by decide) (Nat.le_max_left _ _)
    have hr : 2 ^ r.depth ≤ 2 ^ max l.depth r.depth :=
      Nat.pow_le_pow_right (by decide) (Nat.le_max_right _ _)
    omega

/-- No binary tree over `n` operands is shallower than `ceilLog2 n`. -/
theorem ceilLog2_le_depth (t : Shape) : ceilLog2 t.leaves.length ≤ t.depth :=
  ceilLog2_le_of_le_two_pow (length_leaves_le t)

end Shape

/-- The tree `treeNode` evaluates. -/
def shape (len lo : Nat) : Nat → Shape
  | 0 => .leaf lo
  | k + 1 =>
    if len ≤ lo + 2 ^ k then shape len lo k
    else .node (shape len lo k) (shape len (lo + 2 ^ k) k)

theorem treeNode_eq_eval (add : α → α → α) (f : Nat → α) (len lo k : Nat) :
    treeNode add f len lo k = (shape len lo k).eval add f := by
  induction k generalizing lo with
  | zero => rfl
  | succ k ih =>
    by_cases h : len ≤ lo + 2 ^ k <;> simp [treeNode, shape, h, Shape.eval, ih]

/-- The canonical tree over `len` operands. -/
def canonical (len : Nat) : Shape := shape len 0 (ceilLog2 len)

theorem treeSum_eq_eval (add : α → α → α) (zero : α) (f : Nat → α) {len : Nat}
    (h : 0 < len) : treeSum add zero f len = (canonical len).eval add f := by
  simp [treeSum, canonical, treeNode_eq_eval, Nat.ne_of_gt h]

theorem shape_leaves (len lo k : Nat) (h : lo < len) :
    (shape len lo k).leaves = List.range' lo (min (2 ^ k) (len - lo)) := by
  induction k generalizing lo with
  | zero =>
    have : min (2 ^ 0) (len - lo) = 1 := by simp; omega
    simp [shape, Shape.leaves, this]
  | succ k ih =>
    have hp := Nat.two_pow_pos k
    by_cases hmid : len ≤ lo + 2 ^ k
    · have : min (2 ^ (k + 1)) (len - lo) = min (2 ^ k) (len - lo) := by
        rw [Nat.pow_succ]; omega
      simp [shape, hmid, ih lo h, this]
    · have hl : min (2 ^ k) (len - lo) = 2 ^ k := by omega
      have hr := ih (lo + 2 ^ k) (by omega)
      simp only [shape, hmid, ↓reduceIte, Shape.leaves, ih lo h, hr, hl]
      have happ := @List.range'_append lo (2 ^ k) (min (2 ^ k) (len - (lo + 2 ^ k))) 1
      rw [Nat.one_mul] at happ
      rw [happ]
      congr 1
      rw [Nat.pow_succ]; omega

/-- The canonical tree reads operands `0, 1, …, len - 1`, each once, in order. In particular
`tree_node` never indexes past the end of the slice. -/
theorem canonical_leaves {len : Nat} (h : 0 < len) : (canonical len).leaves = List.range len := by
  rw [canonical, shape_leaves _ _ _ h, List.range_eq_range']
  have := le_two_pow_ceilLog2 len
  congr 1
  omega

theorem shape_depth_le (len lo k : Nat) : (shape len lo k).depth ≤ k := by
  induction k generalizing lo with
  | zero => simp [shape, Shape.depth]
  | succ k ih =>
    by_cases h : len ≤ lo + 2 ^ k
    · simp only [shape, h, ↓reduceIte]; exact Nat.le_succ_of_le (ih lo)
    · simp only [shape, h, ↓reduceIte, Shape.depth]
      have := ih lo
      have := ih (lo + 2 ^ k)
      omega

/-- **Optimality of the RepOps order.** For `len ≥ 1` operands:

1. the canonical tree reads each operand exactly once, in index order;
2. it performs `len - 1` additions and has depth `ceilLog2 len`;
3. every binary reduction tree over the same operands, in any order, also performs `len - 1`
   additions and has depth at least `ceilLog2 len`.

So no fixed-order binary reduction does less work or has a shorter critical path. Fixing the
order for reproducibility costs nothing in either measure. -/
theorem canonical_optimal {len : Nat} (h : 0 < len) :
    (canonical len).leaves = List.range len ∧
    (canonical len).adds = len - 1 ∧
    (canonical len).depth = ceilLog2 len ∧
    ∀ t : Shape, t.leaves.Perm (List.range len) → t.adds = len - 1 ∧ ceilLog2 len ≤ t.depth := by
  have hleaves := canonical_leaves h
  refine ⟨hleaves, ?_, ?_, ?_⟩
  · have := Shape.adds_add_one (canonical len)
    rw [hleaves, List.length_range] at this
    omega
  · apply Nat.le_antisymm (shape_depth_le _ _ _)
    have := Shape.ceilLog2_le_depth (canonical len)
    rwa [hleaves, List.length_range] at this
  · intro t ht
    have hlen : t.leaves.length = len := by rw [ht.length_eq, List.length_range]
    have := Shape.adds_add_one t
    have := Shape.ceilLog2_le_depth t
    rw [hlen] at this
    exact ⟨by omega, this⟩

/-! ## The legacy Verde reduction -/

/-- The split point of `repops_pairwise_tree_sum` for `n ≥ 3`:
`let mid = n.next_power_of_two() / 2; if mid >= n { n / 2 } else { mid }`. -/
def legacyMid (n : Nat) : Nat :=
  if n ≤ nextPow2 n / 2 then n / 2 else nextPow2 n / 2

theorem nextPow2_half {n : Nat} (h : 2 ≤ n) : nextPow2 n / 2 = 2 ^ (ceilLog2 n - 1) := by
  have hpos := ceilLog2_pos h
  unfold nextPow2
  obtain ⟨k, hk⟩ : ∃ k, ceilLog2 n = k + 1 := ⟨ceilLog2 n - 1, by omega⟩
  rw [hk, Nat.pow_succ, Nat.mul_div_cancel _ (by decide)]
  simp

/-- The `n / 2` fallback in `repops_pairwise_tree_sum` is dead code: for every `n ≥ 2`, half of
`n.next_power_of_two()` is already below `n`. -/
theorem legacy_fallback_unreachable {n : Nat} (h : 2 ≤ n) : ¬ n ≤ nextPow2 n / 2 := by
  rw [nextPow2_half h]
  have := half_nextPow2_lt h
  omega

theorem legacyMid_eq {n : Nat} (h : 2 ≤ n) : legacyMid n = 2 ^ (ceilLog2 n - 1) := by
  simp only [legacyMid, legacy_fallback_unreachable h, ↓reduceIte]
  rw [nextPow2_half h]

theorem legacyMid_bounds {n : Nat} (h : 3 ≤ n) : 1 ≤ legacyMid n ∧ legacyMid n < n := by
  rw [legacyMid_eq (by omega)]
  exact ⟨Nat.one_le_two_pow, half_nextPow2_lt (by omega)⟩

/-- `repops_pairwise_tree_sum(&values[lo .. lo + n])`. -/
def legacySum (add : α → α → α) (zero : α) (f : Nat → α) (lo n : Nat) : α :=
  if _h0 : n = 0 then zero
  else if _h1 : n = 1 then f lo
  else if _h2 : n = 2 then add (f lo) (f (lo + 1))
  else
    add (legacySum add zero f lo (legacyMid n))
      (legacySum add zero f (lo + legacyMid n) (n - legacyMid n))
termination_by n
decreasing_by
  all_goals
    have := legacyMid_bounds (n := n) (by omega)
    omega

/-- `treeNode` reads `len` only to decide whether a right subtree is empty, so two bounds that
agree within the subtree give the same value. -/
theorem treeNode_len_congr (add : α → α → α) (f : Nat → α) (len len' lo k : Nat)
    (h : min len (lo + 2 ^ k) = min len' (lo + 2 ^ k)) :
    treeNode add f len lo k = treeNode add f len' lo k := by
  induction k generalizing lo with
  | zero => rfl
  | succ k ih =>
    rw [Nat.pow_succ] at h
    have hp := Nat.two_pow_pos k
    have hl := ih lo (by omega)
    have hr := ih (lo + 2 ^ k) (by omega)
    have hc : (len ≤ lo + 2 ^ k) ↔ (len' ≤ lo + 2 ^ k) := by omega
    simp only [treeNode, hl, hr, hc]

/-- A tree sized for more leaves than are present equals the tree sized by
`next_power_of_two` of the present leaves: absent subtrees pass their sibling up unchanged. -/
theorem treeNode_oversized (add : α → α → α) (f : Nat → α) (lo n k : Nat)
    (hk : ceilLog2 n ≤ k) :
    treeNode add f (lo + n) lo k = treeNode add f (lo + n) lo (ceilLog2 n) := by
  obtain ⟨d, rfl⟩ : ∃ d, k = ceilLog2 n + d := ⟨k - ceilLog2 n, by omega⟩
  induction d with
  | zero => rfl
  | succ d ih =>
    have := (ceilLog2_le_iff n (ceilLog2 n + d)).mp (by omega)
    rw [← ih (by omega), show ceilLog2 n + (d + 1) = (ceilLog2 n + d) + 1 by omega]
    simp only [treeNode]
    simp [show lo + n ≤ lo + 2 ^ (ceilLog2 n + d) by omega]

theorem legacySum_eq_treeNode (add : α → α → α) (zero : α) (f : Nat → α) :
    ∀ n lo, 0 < n → legacySum add zero f lo n = treeNode add f (lo + n) lo (ceilLog2 n) := by
  intro n
  induction n using Nat.strongRecOn with
  | ind n ih =>
    intro lo hn
    rw [legacySum]
    by_cases h1 : n = 1
    · subst h1; simp [treeNode, show ceilLog2 1 = 0 by simp [ceilLog2]]
    by_cases h2 : n = 2
    · subst h2
      have : ceilLog2 2 = 1 := by simp [ceilLog2]
      simp [treeNode, this]
    have h3 : 3 ≤ n := by omega
    simp only [show n ≠ 0 by omega, h1, h2, dite_false]
    obtain ⟨K, hK⟩ : ∃ K, ceilLog2 n = K + 1 := ⟨ceilLog2 n - 1, by have := ceilLog2_pos (n := n) (by omega); omega⟩
    have hmid : legacyMid n = 2 ^ K := by rw [legacyMid_eq (by omega), hK]; rfl
    have hbound := legacyMid_bounds h3
    have hcover : n ≤ 2 ^ K * 2 := by
      have := le_two_pow_ceilLog2 n; rwa [hK, Nat.pow_succ] at this
    rw [hmid] at hbound ⊢
    rw [ih (2 ^ K) (by omega) lo (by omega), ih (n - 2 ^ K) (by omega) (lo + 2 ^ K) (by omega)]
    rw [hK]
    simp only [treeNode, show ¬ (lo + n ≤ lo + 2 ^ K) by omega, ↓reduceIte]
    rw [ceilLog2_two_pow]
    congr 1
    · exact treeNode_len_congr add f _ _ lo K (by omega)
    · rw [show lo + n = lo + 2 ^ K + (n - 2 ^ K) by omega]
      exact (treeNode_oversized add f (lo + 2 ^ K) (n - 2 ^ K) K
        (ceilLog2_le_of_le_two_pow (by omega))).symm

/-- `repops_pairwise_tree_sum` and `tree_sum` evaluate the same tree, so they agree bit for bit
for every input and every `add`, including IEEE `f64` addition. -/
theorem legacySum_eq_treeSum (add : α → α → α) (zero : α) (f : Nat → α) (n : Nat) :
    legacySum add zero f 0 n = treeSum add zero f n := by
  by_cases h : n = 0
  · subst h; simp [legacySum, treeSum]
  · rw [legacySum_eq_treeNode add zero f n 0 (by omega)]
    simp [treeSum, h]

end Kadmium.RepOps
