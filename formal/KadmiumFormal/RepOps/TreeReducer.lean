import KadmiumFormal.RepOps.TreeSum

/-!
# The streaming reducer

Models `TreeReducer` in `crates/kadmium-runtime/src/repops.rs`: a heap of `2 * capacity`
optional nodes, root at `1`, leaves at `capacity .. 2 * capacity`. `push(index, value)` places a
leaf and forms every ancestor whose children are complete; `push_chunk` is a sequence of `push`
calls; `finish` returns the root once every leaf has arrived.

Main result, `reducer_eq_treeSum`: pushing the leaves in **any** order returns exactly
`tree_sum` of the same values. It holds for every `add`, with no algebraic law assumed, so for IEEE
`f32` and `f64` it is bit-for-bit equality. This is the claim in `docs/ARCHITECTURE.md` section
9.6 that a chunked, threaded or reassociated reduction "produces the same bits", proved for every
arrival order rather than sampled by tests.

Modelling choices, each checked against the Rust:

* The node vector is a function `Nat → Option α`; every index the algorithm touches is below
  `2 * capacity`, which the proofs track (`c < 2 * cap`), so the Rust indexing never panics on
  these paths.
* Error strings become `ReduceError` constructors.
* `usize` arithmetic is unbounded `Nat`; see `KadmiumFormal.Pow2`.
-/

namespace Kadmium.RepOps

variable {α : Type}

/-- The loop in `TreeReducer::absent`: double `node` until it reaches the leaf row. Rust never
calls it with `node = 0`; the model returns `0` there rather than looping. -/
def leftmost (cap node : Nat) : Nat :=
  if _h0 : node = 0 then 0
  else if _hc : node < cap then leftmost cap (2 * node) else node
termination_by cap - node
decreasing_by omega

/-- `TreeReducer::absent`: every leaf under `node` lies past the end of the input. -/
def absent (cap len node : Nat) : Bool := decide (len ≤ leftmost cap node - cap)

def setAt (nodes : Nat → Option α) (j : Nat) (v : Option α) : Nat → Option α :=
  fun x => if x = j then v else nodes x

/-- The `while node > 1` loop of `TreeReducer::push`. -/
def climb (add : α → α → α) (cap len : Nat) (nodes : Nat → Option α) (node : Nat) :
    Nat → Option α :=
  if _h : 1 < node then
    let parent := node / 2
    let left := 2 * parent
    let right := left + 1
    match nodes left with
    | none => nodes
    | some l =>
      if absent cap len right then climb add cap len (setAt nodes parent (some l)) parent
      else
        match nodes right with
        | some r => climb add cap len (setAt nodes parent (some (add l r))) parent
        | none => nodes
  else nodes
termination_by node
decreasing_by all_goals omega

inductive ReduceError where
  /-- `"TreeReducer: leaf {index} is past the end of {len} leaves"` -/
  | pastEnd
  /-- `"TreeReducer: leaf {index} pushed twice"` -/
  | pushedTwice
  /-- `"TreeReducer: {pushed} of {len} leaves arrived"` -/
  | incomplete
  /-- `"TreeReducer: root not formed"` -/
  | rootNotFormed
  deriving Repr, DecidableEq

structure Reducer (α : Type) where
  len : Nat
  capacity : Nat
  nodes : Nat → Option α
  pushed : Nat

namespace Reducer

/-- `TreeReducer::new(len)`: `capacity = len.next_power_of_two().max(1)`. -/
def new (len : Nat) : Reducer α :=
  { len, capacity := max (nextPow2 len) 1, nodes := fun _ => none, pushed := 0 }

/-- `TreeReducer::push`. -/
def push (add : α → α → α) (r : Reducer α) (index : Nat) (value : α) :
    Except ReduceError (Reducer α) :=
  if r.len ≤ index then .error .pastEnd
  else
    let node := r.capacity + index
    match r.nodes node with
    | some _ => .error .pushedTwice
    | none =>
      .ok { r with
        nodes := climb add r.capacity r.len (setAt r.nodes node (some value)) node
        pushed := r.pushed + 1 }

/-- A sequence of `push` calls, stopping at the first error. `push_chunk(start, values)` is
`pushAll` over `(start + offset, value)`. -/
def pushAll (add : α → α → α) : Reducer α → List (Nat × α) → Except ReduceError (Reducer α)
  | r, [] => .ok r
  | r, (i, v) :: rest =>
    match r.push add i v with
    | .error e => .error e
    | .ok r' => pushAll add r' rest

/-- `TreeReducer::finish`. -/
def finish (zero : α) (r : Reducer α) : Except ReduceError α :=
  if r.len = 0 then .ok zero
  else if r.pushed ≠ r.len then .error .incomplete
  else
    match r.nodes 1 with
    | some v => .ok v
    | none => .error .rootNotFormed

end Reducer

/-! ## The value each heap node must hold -/

/-- The value of heap node `j` over leaf values `f`: a leaf holds its value, an internal node the
sum of its children, or its left child alone when the right subtree is absent. -/
def hspec (add : α → α → α) (f : Nat → α) (cap len : Nat) (j : Nat) : α :=
  if _h0 : j = 0 then f 0
  else if _hc : j < cap then
    if absent cap len (2 * j + 1) then hspec add f cap len (2 * j)
    else add (hspec add f cap len (2 * j)) (hspec add f cap len (2 * j + 1))
  else f (j - cap)
termination_by cap - j
decreasing_by all_goals omega

theorem hspec_internal (add : α → α → α) (f : Nat → α) {cap len p : Nat} (h1 : 1 ≤ p)
    (h2 : p < cap) :
    hspec add f cap len p =
      if absent cap len (2 * p + 1) then hspec add f cap len (2 * p)
      else add (hspec add f cap len (2 * p)) (hspec add f cap len (2 * p + 1)) := by
  rw [hspec]
  simp [show p ≠ 0 by omega, h2]

theorem hspec_leaf (add : α → α → α) (f : Nat → α) {cap len : Nat} (hcap : 1 ≤ cap) (i : Nat) :
    hspec add f cap len (cap + i) = f i := by
  have h0 : ¬ cap + i = 0 := by omega
  have hc : ¬ cap + i < cap := by omega
  rw [hspec]
  simp only [h0, hc, ↓reduceDIte, Nat.add_sub_cancel_left]

/-! ## Heap arithmetic -/

theorem leftmost_of_le {cap j : Nat} (hc : 1 ≤ cap) (h : cap ≤ j) : leftmost cap j = j := by
  rw [leftmost]
  simp [show j ≠ 0 by omega, show ¬ j < cap by omega]

theorem leftmost_double {cap j : Nat} (h1 : 1 ≤ j) (h2 : j < cap) :
    leftmost cap j = leftmost cap (2 * j) := by
  rw [leftmost]
  simp [show j ≠ 0 by omega, h2]

theorem absent_double {cap len j : Nat} (h1 : 1 ≤ j) (h2 : j < cap) :
    absent cap len j = absent cap len (2 * j) := by
  simp [absent, leftmost_double h1 h2]

/-- A node `x` sits `k` levels above the leaf row when `x * 2 ^ k` is a leaf position. -/
theorem lvl_parent {K k x : Nat} (h1 : 2 ^ K ≤ x * 2 ^ (k + 1))
    (h2 : x * 2 ^ (k + 1) < 2 * 2 ^ K) : 1 ≤ x ∧ x < 2 ^ K := by
  have hp := Nat.two_pow_pos K
  have hk : 2 ≤ 2 ^ (k + 1) := by
    rw [Nat.pow_succ]; have := Nat.two_pow_pos k; omega
  refine ⟨?_, ?_⟩
  · rcases Nat.eq_zero_or_pos x with rfl | hx
    · simp at h1 <;> omega
    · exact hx
  · have : x * 2 ≤ x * 2 ^ (k + 1) := Nat.mul_le_mul_left x hk
    omega

/-- The right child of a node on the level `k + 1` is still inside the heap. -/
theorem lvl_right {K k x : Nat} (h1 : 2 ^ K ≤ x * 2 ^ (k + 1))
    (h2 : x * 2 ^ (k + 1) < 2 * 2 ^ K) : x * 2 ^ (k + 1) + 2 ^ k < 2 * 2 ^ K := by
  obtain ⟨hx, _⟩ := lvl_parent h1 h2
  have hkK : k ≤ K := by
    have : 2 ^ (k + 1) < 2 ^ (K + 1) := by
      rw [Nat.pow_succ 2 K]
      have : 2 ^ (k + 1) ≤ x * 2 ^ (k + 1) := Nat.le_mul_of_pos_left _ hx
      omega
    have := (Nat.pow_lt_pow_iff_right (by decide)).mp this
    omega
  have hsplit : 2 * 2 ^ K = 2 ^ (K - k) * 2 ^ (k + 1) := by
    rw [← Nat.pow_add, show K - k + (k + 1) = K + 1 by omega, Nat.pow_succ, Nat.mul_comm]
  rw [hsplit] at h2 ⊢
  have hlt : x < 2 ^ (K - k) := Nat.lt_of_mul_lt_mul_right h2
  have hle : (x + 1) * 2 ^ (k + 1) ≤ 2 ^ (K - k) * 2 ^ (k + 1) := Nat.mul_le_mul_right _ hlt
  rw [Nat.add_mul, Nat.one_mul] at hle
  have : 2 ^ (k + 1) = 2 ^ k * 2 := Nat.pow_succ 2 k
  have := Nat.two_pow_pos k
  omega

theorem leftmost_lvl (K : Nat) :
    ∀ k x, 2 ^ K ≤ x * 2 ^ k → x * 2 ^ k < 2 * 2 ^ K → leftmost (2 ^ K) x = x * 2 ^ k := by
  intro k
  induction k with
  | zero =>
    intro x h1 _
    simp only [Nat.pow_zero, Nat.mul_one] at h1 ⊢
    exact leftmost_of_le Nat.one_le_two_pow h1
  | succ k ih =>
    intro x h1 h2
    obtain ⟨hx1, hxK⟩ := lvl_parent h1 h2
    have e : 2 * x * 2 ^ k = x * 2 ^ (k + 1) := by grind
    rw [leftmost_double hx1 hxK, ih (2 * x) (by rw [e]; exact h1) (by rw [e]; exact h2), e]

/-- The heap value of node `x` on level `k` is the canonical tree over its leaves. -/
theorem hspec_eq_treeNode (add : α → α → α) (f : Nat → α) (K len : Nat) :
    ∀ k x, 2 ^ K ≤ x * 2 ^ k → x * 2 ^ k < 2 * 2 ^ K →
      hspec add f (2 ^ K) len x = treeNode add f len (x * 2 ^ k - 2 ^ K) k := by
  intro k
  induction k with
  | zero =>
    intro x h1 _
    simp only [Nat.pow_zero, Nat.mul_one] at h1 ⊢
    have := hspec_leaf add f (len := len) (Nat.one_le_two_pow (n := K)) (x - 2 ^ K)
    rw [show 2 ^ K + (x - 2 ^ K) = x by omega] at this
    rw [this, treeNode]
  | succ k ih =>
    intro x h1 h2
    obtain ⟨hx1, hxK⟩ := lvl_parent h1 h2
    have hr := lvl_right h1 h2
    have hpk := Nat.two_pow_pos k
    have e1 : 2 * x * 2 ^ k = x * 2 ^ (k + 1) := by grind
    have e2 : (2 * x + 1) * 2 ^ k = x * 2 ^ (k + 1) + 2 ^ k := by grind
    have hl' := ih (2 * x) (by rw [e1]; exact h1) (by rw [e1]; exact h2)
    have hr' := ih (2 * x + 1) (by rw [e2]; omega) (by rw [e2]; exact hr)
    have habs : absent (2 ^ K) len (2 * x + 1) =
        decide (len ≤ x * 2 ^ (k + 1) - 2 ^ K + 2 ^ k) := by
      rw [absent, leftmost_lvl K k (2 * x + 1) (by rw [e2]; omega) (by rw [e2]; exact hr), e2]
      simp only [decide_eq_decide]
      omega
    rw [hspec_internal add f hx1 hxK, hl', hr', habs, e1, e2,
      show x * 2 ^ (k + 1) + 2 ^ k - 2 ^ K = x * 2 ^ (k + 1) - 2 ^ K + 2 ^ k by omega]
    simp only [treeNode]
    by_cases hc : len ≤ x * 2 ^ (k + 1) - 2 ^ K + 2 ^ k <;> simp [hc]

/-- The root holds `tree_sum`. -/
theorem hspec_root (add : α → α → α) (f : Nat → α) (K len : Nat) :
    hspec add f (2 ^ K) len 1 = treeNode add f len 0 K := by
  have := hspec_eq_treeNode add f K len K 1 (by simp) (by have := Nat.two_pow_pos K; simp; omega)
  simpa using this

/-! ## Invariants -/

/-- Every formed node holds exactly its tree value. -/
def Sound (add : α → α → α) (f : Nat → α) (cap len : Nat) (nodes : Nat → Option α) : Prop :=
  ∀ j v, nodes j = some v → v = hspec add f cap len j

/-- Internal node `p` is formed whenever its children are complete. -/
def ClosedAt (cap len : Nat) (nodes : Nat → Option α) (p : Nat) : Prop :=
  (nodes (2 * p)).isSome = true →
    (absent cap len (2 * p + 1) = true ∨ (nodes (2 * p + 1)).isSome = true) →
    (nodes p).isSome = true

/-- `ClosedAt` holds at every internal node except possibly `e`. With `e = 0` it holds at all. -/
def ClosedExcept (cap len : Nat) (nodes : Nat → Option α) (e : Nat) : Prop :=
  ∀ p, 1 ≤ p → p < cap → p ≠ e → ClosedAt cap len nodes p

theorem set_parent {add : α → α → α} {f : Nat → α} {cap len p : Nat}
    {nodes : Nat → Option α} {v : α} (hS : Sound add f cap len nodes)
    (hC : ClosedExcept cap len nodes p) (hv : v = hspec add f cap len p) :
    Sound add f cap len (setAt nodes p (some v)) ∧
      ClosedExcept cap len (setAt nodes p (some v)) (p / 2) := by
  constructor
  · intro j w hj
    by_cases hjp : j = p
    · subst hjp
      simp only [setAt, ↓reduceIte, Option.some.injEq] at hj
      rw [← hj, hv]
    · simp only [setAt, hjp, ↓reduceIte] at hj
      exact hS j w hj
  · intro q hq hqc hqe
    by_cases hqp : q = p
    · subst hqp
      intro _ _
      simp [setAt]
    · have h2q : 2 * q ≠ p := by omega
      have h2q1 : 2 * q + 1 ≠ p := by omega
      have := hC q hq hqc hqp
      simpa [ClosedAt, setAt, h2q, h2q1, hqp] using this

theorem climb_inv (add : α → α → α) (f : Nat → α) (cap len : Nat) :
    ∀ c (nodes : Nat → Option α), c < 2 * cap →
      Sound add f cap len nodes → ClosedExcept cap len nodes (c / 2) →
      Sound add f cap len (climb add cap len nodes c) ∧
      ClosedExcept cap len (climb add cap len nodes c) 0 ∧
      ∀ j, cap ≤ j → climb add cap len nodes c j = nodes j := by
  intro c
  induction c using Nat.strongRecOn with
  | ind c ih =>
  intro nodes hc hS hC
  rw [climb]
  by_cases h1 : 1 < c
  · simp only [h1, dite_true]
    have hp1 : 1 ≤ c / 2 := by omega
    have hpc : c / 2 < cap := by omega
    -- Every step writes only `c / 2`, an internal node, and leaves the rest to the recursion.
    have step : ∀ v, v = hspec add f cap len (c / 2) →
        Sound add f cap len (climb add cap len (setAt nodes (c / 2) (some v)) (c / 2)) ∧
        ClosedExcept cap len (climb add cap len (setAt nodes (c / 2) (some v)) (c / 2)) 0 ∧
        ∀ j, cap ≤ j → climb add cap len (setAt nodes (c / 2) (some v)) (c / 2) j = nodes j := by
      intro v hv
      obtain ⟨hS', hC'⟩ := set_parent hS hC hv
      obtain ⟨r1, r2, r3⟩ := ih (c / 2) (by omega) _ (by omega) hS' hC'
      refine ⟨r1, r2, fun j hj => ?_⟩
      rw [r3 j hj]
      simp [setAt, show j ≠ c / 2 by omega]
    -- Stopping leaves `ClosedAt (c / 2)` true because its premise fails.
    have stop : ¬ ((nodes (2 * (c / 2))).isSome = true ∧
        (absent cap len (2 * (c / 2) + 1) = true ∨ (nodes (2 * (c / 2) + 1)).isSome = true)) →
        ClosedExcept cap len nodes 0 := by
      intro hn q hq hqc _
      by_cases hqp : q = c / 2
      · subst hqp; intro a b; exact absurd ⟨a, b⟩ hn
      · exact hC q hq hqc hqp
    cases hl : nodes (2 * (c / 2)) with
    | none =>
      simp only
      exact ⟨hS, stop (by simp [hl]), fun _ _ => by simp⟩
    | some lv =>
      simp only
      have hlv := hS _ _ hl
      cases ha : absent cap len (2 * (c / 2) + 1) with
      | true =>
        simp only [↓reduceIte]
        apply step
        simp [hspec_internal add f hp1 hpc, ha, hlv]
      | false =>
        simp only [Bool.false_eq_true, ↓reduceIte]
        cases hr : nodes (2 * (c / 2) + 1) with
        | none =>
          simp only
          exact ⟨hS, stop (by simp [hr, ha]), fun _ _ => by simp⟩
        | some rv =>
          simp only
          apply step
          simp [hspec_internal add f hp1 hpc, ha, hlv, hS _ _ hr]
  · simp only [h1, dite_false]
    exact ⟨hS, fun p hp hpc _ => hC p hp hpc (by omega), fun _ _ => by simp⟩

/-- When every present leaf has arrived, every node that is not absent is formed. -/
theorem complete (cap len : Nat) (nodes : Nat → Option α) (hcap : 1 ≤ cap)
    (hleaf : ∀ i, i < len → (nodes (cap + i)).isSome = true)
    (hC : ClosedExcept cap len nodes 0) (j : Nat) (hj1 : 1 ≤ j)
    (ha : absent cap len j = false) : (nodes j).isSome = true := by
  by_cases hjc : j < cap
  · have ha2 : absent cap len (2 * j) = false := by rw [← absent_double hj1 hjc]; exact ha
    have hl := complete cap len nodes hcap hleaf hC (2 * j) (by omega) ha2
    have hr : absent cap len (2 * j + 1) = true ∨ (nodes (2 * j + 1)).isSome = true := by
      cases hb : absent cap len (2 * j + 1)
      · exact Or.inr (complete cap len nodes hcap hleaf hC (2 * j + 1) (by omega) hb)
      · exact Or.inl rfl
    exact hC j hj1 hjc (by omega) hl hr
  · have hlt : j - cap < len := by
      simp only [absent, leftmost_of_le hcap (by omega : cap ≤ j), decide_eq_false_iff_not] at ha
      omega
    have := hleaf (j - cap) hlt
    rwa [show cap + (j - cap) = j by omega] at this
termination_by 2 * cap - j
decreasing_by all_goals omega

/-- The reducer state after the leaves in `P` have been pushed. -/
structure Inv (add : α → α → α) (f : Nat → α) (r : Reducer α) (P : List Nat) : Prop where
  cap_pos : 1 ≤ r.capacity
  len_le : r.len ≤ r.capacity
  pushed : r.pushed = P.length
  lt : ∀ i ∈ P, i < r.len
  leaf : ∀ i, (r.nodes (r.capacity + i)).isSome = true ↔ i ∈ P
  sound : Sound add f r.capacity r.len r.nodes
  closed : ClosedExcept r.capacity r.len r.nodes 0

theorem inv_new (add : α → α → α) (f : Nat → α) (len : Nat) :
    Inv add f (Reducer.new len : Reducer α) [] where
  cap_pos := Nat.le_max_right _ _
  len_le := Nat.le_trans (le_two_pow_ceilLog2 len) (Nat.le_max_left _ _)
  pushed := rfl
  lt := by simp
  leaf := by simp [Reducer.new]
  sound := by intro j v h; simp [Reducer.new] at h
  closed := by intro p _ _ _ h; simp [Reducer.new] at h

theorem push_inv (add : α → α → α) (f : Nat → α) (r : Reducer α) (P : List Nat) (i : Nat)
    (hI : Inv add f r P) (hi : i < r.len) (hiP : i ∉ P) :
    ∃ r', r.push add i (f i) = .ok r' ∧ Inv add f r' (i :: P) ∧
      r'.len = r.len ∧ r'.capacity = r.capacity := by
  have hnone : r.nodes (r.capacity + i) = none := by
    cases h : r.nodes (r.capacity + i) with
    | none => rfl
    | some _ => exact absurd ((hI.leaf i).mp (by simp [h])) hiP
  let nodes0 := setAt r.nodes (r.capacity + i) (some (f i))
  have hS0 : Sound add f r.capacity r.len nodes0 := by
    intro j v hj
    by_cases hji : j = r.capacity + i
    · subst hji
      simp only [nodes0, setAt, ↓reduceIte, Option.some.injEq] at hj
      rw [← hj, hspec_leaf add f hI.cap_pos]
    · simp only [nodes0, setAt, hji, ↓reduceIte] at hj
      exact hI.sound j v hj
  have hC0 : ClosedExcept r.capacity r.len nodes0 ((r.capacity + i) / 2) := by
    intro q hq hqc hqe
    have h2q : 2 * q ≠ r.capacity + i := by omega
    have h2q1 : 2 * q + 1 ≠ r.capacity + i := by omega
    have hqi : q ≠ r.capacity + i := by omega
    have := hI.closed q hq hqc (by omega)
    simpa [ClosedAt, nodes0, setAt, h2q, h2q1, hqi] using this
  obtain ⟨hS1, hC1, hL1⟩ :=
    climb_inv add f r.capacity r.len (r.capacity + i) nodes0 (by have := hI.len_le; omega) hS0 hC0
  refine ⟨{ r with
      nodes := climb add r.capacity r.len nodes0 (r.capacity + i)
      pushed := r.pushed + 1 }, ?_, ?_, rfl, rfl⟩
  · simp only [Reducer.push, show ¬ r.len ≤ i by omega, ↓reduceIte, hnone]
    rfl
  · refine ⟨hI.cap_pos, hI.len_le, by simp [hI.pushed], ?_, ?_, hS1, hC1⟩
    · intro j hj
      simp only [List.mem_cons] at hj
      rcases hj with rfl | hj
      · exact hi
      · exact hI.lt j hj
    · intro j
      simp only
      rw [hL1 _ (by omega)]
      by_cases hji : j = i
      · subst hji; simp [nodes0, setAt]
      · simp only [nodes0, setAt, show r.capacity + j ≠ r.capacity + i by omega, ↓reduceIte,
          List.mem_cons, hji, false_or]
        exact hI.leaf j

theorem pushAll_inv (add : α → α → α) (f : Nat → α) :
    ∀ (order : List Nat) (r : Reducer α) (P : List Nat), Inv add f r P → order.Nodup →
      (∀ i ∈ order, i < r.len ∧ i ∉ P) →
      ∃ r', r.pushAll add (order.map fun i => (i, f i)) = .ok r' ∧
        Inv add f r' (order.reverse ++ P) ∧ r'.len = r.len ∧ r'.capacity = r.capacity := by
  intro order
  induction order with
  | nil => intro r P hI _ _; exact ⟨r, rfl, by simpa using hI, rfl, rfl⟩
  | cons i rest ih =>
    intro r P hI hnd hmem
    rw [List.nodup_cons] at hnd
    obtain ⟨hi, hiP⟩ := hmem i (by simp)
    obtain ⟨r1, hpush, hI1, hlen1, hcap1⟩ := push_inv add f r P i hI hi hiP
    obtain ⟨r2, hall, hI2, hlen2, hcap2⟩ := ih r1 (i :: P) hI1 hnd.2 (by
      intro j hj
      obtain ⟨hjl, hjP⟩ := hmem j (by simp [hj])
      refine ⟨by omega, ?_⟩
      simp only [List.mem_cons, not_or]
      exact ⟨fun h => hnd.1 (h ▸ hj), hjP⟩)
    refine ⟨r2, ?_, by simpa using hI2, by omega, by omega⟩
    simp only [List.map_cons, Reducer.pushAll, hpush]
    exact hall

/-- **Order independence.** For any permutation `order` of `0 .. len`, pushing leaf `i` with value
`f i` in that order and then calling `finish` returns `tree_sum` of the values, for every `add`.
In particular no push fails and `finish` never reports an incomplete tree. -/
theorem reducer_eq_treeSum (add : α → α → α) (zero : α) (f : Nat → α) (len : Nat)
    (order : List Nat) (hperm : order.Perm (List.range len)) :
    ((Reducer.new len : Reducer α).pushAll add (order.map fun i => (i, f i)) >>=
      Reducer.finish zero) = .ok (treeSum add zero f len) := by
  have hnd : order.Nodup := hperm.nodup_iff.mpr List.nodup_range
  have hmem : ∀ i, i ∈ order ↔ i < len := fun i => by rw [hperm.mem_iff, List.mem_range]
  obtain ⟨r, hall, hI, hlen, hcap⟩ := pushAll_inv add f order (Reducer.new len) [] (inv_new add f len)
    hnd (fun i hi => ⟨(hmem i).mp hi, by simp⟩)
  rw [hall]
  simp only [Reducer.new] at hlen hcap
  have hcapK : r.capacity = 2 ^ ceilLog2 len := by
    rw [hcap, nextPow2]; exact Nat.max_eq_left Nat.one_le_two_pow
  show r.finish zero = _
  by_cases h0 : len = 0
  · simp [Reducer.finish, hlen, h0, treeSum]
  · have hpushed : r.pushed = r.len := by
      rw [hI.pushed, List.append_nil, List.length_reverse, hperm.length_eq, List.length_range, hlen]
    have hroot : (r.nodes 1).isSome = true := by
      apply complete r.capacity r.len r.nodes hI.cap_pos _ hI.closed 1 (by omega)
      · rw [absent, hcapK, leftmost_lvl (ceilLog2 len) (ceilLog2 len) 1 (by simp)
          (by have := Nat.two_pow_pos (ceilLog2 len); simp; omega)]
        simp [hlen]; omega
      · intro i hi
        apply (hI.leaf i).mpr
        simp only [List.append_nil, List.mem_reverse]
        exact (hmem i).mpr (by omega)
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp hroot
    have := hI.sound 1 v hv
    rw [hcapK, hlen, hspec_root] at this
    simp [Reducer.finish, hlen, h0, hpushed, hv, treeSum, this]

/-! ## Rejections -/

theorem push_pastEnd (add : α → α → α) (r : Reducer α) {i : Nat} (v : α) (h : r.len ≤ i) :
    r.push add i v = .error .pastEnd := by
  simp [Reducer.push, h]

theorem push_twice (add : α → α → α) (f : Nat → α) (r : Reducer α) (P : List Nat) {i : Nat}
    (hI : Inv add f r P) (hi : i < r.len) (hiP : i ∈ P) (v : α) :
    r.push add i v = .error .pushedTwice := by
  obtain ⟨w, hw⟩ := Option.isSome_iff_exists.mp ((hI.leaf i).mpr hiP)
  simp [Reducer.push, show ¬ r.len ≤ i by omega, hw]

theorem finish_incomplete (zero : α) (r : Reducer α) (h0 : r.len ≠ 0)
    (h : r.pushed ≠ r.len) : r.finish zero = .error .incomplete := by
  simp [Reducer.finish, h0, h]

end Kadmium.RepOps
