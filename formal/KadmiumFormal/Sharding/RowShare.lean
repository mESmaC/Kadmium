/-!
# Round-robin row shares

Models `ExpertPlacementPlan::row_share` and `ExpertRowShare::serves` in
`crates/kadmium-sharding/src/grid_plan.rs`. When an expert has `c` copies and a step has `rows`
rows, the rows are dealt round robin: copy `i` serves rows `i, i + c, i + 2c, …`, so

```rust
let mine = (rows - index).div_ceil(copies);      // rows this copy serves
self.count <= 1 || row % self.count == self.index // whether this copy serves `row`
```

An expert-parallel combine is correct only if every row is served by exactly one copy and the
copies' shares add up to the whole batch. Both are proved here for every `rows` and every `c ≥ 1`:

* `count_mine`: copy `i < c` serves exactly `mine rows c i` rows, the closed form the planner
  uses for its share.
* `sum_mine`: the shares of the `c` copies add up to `rows`.
* `exactly_one_serves`: every row is served by exactly one copy.

Modelling choices: `row_share` divides `mine` by `rows` to report an `f64` fraction; the model
stops at the integer `mine`, which is what the fixture compares. The Rust clamps `rows` and
`copies` to at least `1` before dividing; the theorems assume `1 ≤ c`, and `mine 0 c i = 0`.
-/

namespace Kadmium.Sharding

/-- `(rows - index).div_ceil(copies)`; `0` when the index is past the rows. -/
def mine (rows c i : Nat) : Nat := (rows - i + c - 1) / c

/-- `ExpertRowShare::serves`. -/
def serves (count index row : Nat) : Bool := count ≤ 1 || row % count == index

theorem serves_iff {count index row : Nat} (hi : index < count) :
    serves count index row = true ↔ row % count = index := by
  unfold serves
  simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq]
  constructor
  · rintro (h | h)
    · have hc : count = 1 := by omega
      have hi0 : index = 0 := by omega
      subst hc hi0
      exact Nat.mod_one row
    · exact h
  · exact Or.inr

/-- `(m + i) % c = i` exactly when `c` divides `m`, for `i < c`. -/
theorem add_mod_eq_iff {m i c : Nat} (hc : 1 ≤ c) (hi : i < c) : (m + i) % c = i ↔ m % c = 0 := by
  have hlt := Nat.mod_lt m hc
  constructor
  · intro h
    rw [Nat.add_mod, Nat.mod_eq_of_lt hi] at h
    by_cases hs : m % c + i < c
    · rw [Nat.mod_eq_of_lt hs] at h
      omega
    · rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)] at h
      omega
  · intro h
    rw [Nat.add_mod, h, Nat.zero_add, Nat.mod_mod, Nat.mod_eq_of_lt hi]

/-- One more row goes to copy `i` exactly when the new row index is `i` modulo `c`. -/
theorem mine_succ {rows c i : Nat} (hc : 1 ≤ c) (hi : i < c) :
    mine (rows + 1) c i = mine rows c i + (if rows % c = i then 1 else 0) := by
  unfold mine
  by_cases hri : rows < i
  · have h1 : rows + 1 - i + c - 1 = c - 1 := by omega
    have h2 : rows - i + c - 1 = c - 1 := by omega
    have h3 : rows % c = rows := Nat.mod_eq_of_lt (by omega)
    have h4 : ¬ rows = i := by omega
    rw [h1, h2, h3, Nat.div_eq_of_lt (show c - 1 < c by omega)]
    simp only [h4, ↓reduceIte]
  · obtain ⟨m, rfl⟩ : ∃ m, rows = m + i := ⟨rows - i, by omega⟩
    have e1 : m + i + 1 - i + c - 1 = m + c := by omega
    have e2 : m + i - i + c - 1 = m + (c - 1) := by omega
    rw [e1, e2, Nat.add_div_right _ hc]
    have hdiv := Nat.div_add_mod m c
    have hlt := Nat.mod_lt m hc
    by_cases h0 : m % c = 0
    · have hcond : (m + i) % c = i := (add_mod_eq_iff hc hi).mpr h0
      have : m + (c - 1) = (c - 1) + c * (m / c) := by omega
      rw [this, Nat.add_mul_div_left _ _ hc, Nat.div_eq_of_lt (show c - 1 < c by omega)]
      simp only [hcond, ↓reduceIte]
      omega
    · have hcond : ¬ (m + i) % c = i := fun h => h0 ((add_mod_eq_iff hc hi).mp h)
      have hmul : c * (m / c + 1) = c * (m / c) + c := Nat.mul_succ c (m / c)
      have : m + (c - 1) = (m % c - 1) + c * (m / c + 1) := by rw [hmul]; omega
      rw [this, Nat.add_mul_div_left _ _ hc,
        Nat.div_eq_of_lt (show m % c - 1 < c by omega)]
      simp only [hcond, ↓reduceIte]
      omega

/-- Copy `i < c` serves exactly `mine rows c i` of the rows `0 .. rows`. -/
theorem count_mine {c i : Nat} (hc : 1 ≤ c) (hi : i < c) :
    ∀ rows, ((List.range rows).filter (fun r => r % c == i)).length = mine rows c i := by
  intro rows
  induction rows with
  | zero =>
    show _ = (0 - i + c - 1) / c
    rw [Nat.div_eq_of_lt (show 0 - i + c - 1 < c by omega)]
    simp
  | succ n ih =>
    rw [List.range_succ, List.filter_append, List.length_append, ih, mine_succ hc hi]
    simp only [List.filter_cons, List.filter_nil, beq_iff_eq]
    split <;> simp_all

theorem length_filter_eq_sum_ite {α : Type} (p : α → Bool) (l : List α) :
    (l.filter p).length = (l.map fun x => if p x then 1 else 0).sum := by
  induction l with
  | nil => rfl
  | cons x rest ih =>
    simp only [List.filter_cons, List.map_cons, List.sum_cons]
    split <;> simp_all <;> omega

theorem sum_map_add {α : Type} (f g : α → Nat) (l : List α) :
    (l.map fun x => f x + g x).sum = (l.map f).sum + (l.map g).sum := by
  induction l with
  | nil => rfl
  | cons x rest ih => simp only [List.map_cons, List.sum_cons, ih]; omega

/-- Exactly one `i` below `c` equals a residue `v < c`. -/
theorem sum_ite_eq {v : Nat} : ∀ c, v < c → ((List.range c).map fun i => if v = i then 1 else 0).sum = 1 := by
  have zero_of_le : ∀ c, c ≤ v → ((List.range c).map fun i => if v = i then 1 else 0).sum = 0 := by
    intro c
    induction c with
    | zero => intro _; rfl
    | succ c ih =>
      intro h
      rw [List.range_succ, List.map_append, List.sum_append, ih (by omega)]
      have hne : ¬ v = c := by omega
      simp [hne]
  intro c
  induction c with
  | zero => intro h; omega
  | succ c ih =>
    intro h
    rw [List.range_succ, List.map_append, List.sum_append]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
    by_cases hv : v = c
    · subst hv
      rw [zero_of_le v (Nat.le_refl v)]
      simp
    · rw [ih (by omega)]
      simp only [hv, ↓reduceIte]

theorem sum_map_zero {α : Type} (l : List α) : (l.map fun _ => 0).sum = 0 := by
  induction l with
  | nil => rfl
  | cons _ _ ih => simp [ih]

/-- Every row has exactly one residue, so dealing rows round robin partitions them. -/
theorem partition_sum {c : Nat} (hc : 1 ≤ c) :
    ∀ l : List Nat, ((List.range c).map fun i => (l.filter (fun r => r % c == i)).length).sum =
      l.length := by
  intro l
  induction l with
  | nil => simp [sum_map_zero]
  | cons x rest ih =>
    have step : ∀ i ∈ List.range c, ((x :: rest).filter (fun r => r % c == i)).length =
        (rest.filter (fun r => r % c == i)).length + (if x % c = i then 1 else 0) := by
      intro i _
      simp only [List.filter_cons, beq_iff_eq]
      split <;> simp_all
    rw [List.map_congr_left step, sum_map_add, ih, sum_ite_eq c (Nat.mod_lt x hc)]
    simp

/-- The shares of the `c` copies add up to every row. -/
theorem sum_mine {c : Nat} (hc : 1 ≤ c) (rows : Nat) :
    ((List.range c).map (mine rows c)).sum = rows := by
  have := partition_sum hc (List.range rows)
  rw [List.length_range] at this
  refine Eq.trans ?_ this
  apply congrArg List.sum
  apply List.map_congr_left
  intro i hi
  rw [List.mem_range] at hi
  exact (count_mine hc hi rows).symm

/-- Every row is served by exactly one of the `c` copies. -/
theorem exactly_one_serves {c : Nat} (hc : 1 ≤ c) (row : Nat) :
    ((List.range c).filter fun i => serves c i row).length = 1 := by
  have hcongr : ∀ i ∈ List.range c, serves c i row = decide (row % c = i) := by
    intro i hi
    rw [List.mem_range] at hi
    rw [Bool.eq_iff_iff, decide_eq_true_iff]
    exact serves_iff hi
  rw [List.filter_congr hcongr, length_filter_eq_sum_ite]
  have : ((List.range c).map fun i => if decide (row % c = i) = true then 1 else 0) =
      (List.range c).map fun i => if row % c = i then 1 else 0 := by
    apply List.map_congr_left
    intro i _
    simp
  rw [this, sum_ite_eq c (Nat.mod_lt row hc)]

end Kadmium.Sharding
