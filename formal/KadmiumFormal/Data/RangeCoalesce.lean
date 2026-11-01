/-!
# HTTP range coalescing

Models `HfByteRangeSlicer::coalesce_range_requests` in `crates/kadmium-data/src/range_pull.rs`,
which merges the planned byte slices of one shard file into as few HTTP range requests as possible:

```rust
// slices sorted by (url, byte_start); for each slice s in that order:
if last.url == s.url && s.byte_start <= last.byte_end_inclusive + 1 + max_gap_bytes {
    last.gap_bytes += gap between last.byte_end_inclusive and s.byte_start, if any;
    last.byte_end_inclusive = last.byte_end_inclusive.max(s.byte_end_inclusive);
    last.members.push(index);
} else {
    requests.push(new request for s);
}
```

The model covers one URL: slices of different URLs never merge and the Rust handles each URL as an
independent run of this loop. Slices are a function `sl` from index to `(start, stop)` with `stop`
inclusive; the loop runs over indices `0 .. n` in order, so the model takes the sorted order as a
hypothesis (`Sorted`), which the Rust establishes with a stable `sort_by` first.

## Cost model

A request is an HTTP byte range `[S, E]` that serves a set of slices. Fetching bytes nobody asked
for is the cost the planner bounds: a request is **valid for `g`** when every window of `g + 1`
consecutive bytes inside `[S, E]` touches one of the slices it serves, so no run of more than `g`
unrequested bytes is ever fetched. The number of requests is the quantity minimised.

## Results

* `coalesce_members`: the members of the requests, in request order, are exactly the indices
  `0 .. n`, each once. Every slice is fetched by exactly one request.
* `coalesce_contains`: every member slice lies inside its request's byte range.
* `coalesce_start_attained`, `coalesce_stop_attained`: every request begins at a member's first
  byte and ends at a member's last byte, so `Valid` below, which only speaks about windows inside
  the request, leaves no unrequested run at either end.
* `coalesce_separated`: consecutive requests are more than `g` bytes apart, so no two could have
  been merged.
* `coalesce_valid`: every request the loop produces is valid for `g`.
* `coalesce_optimal`: **no valid plan uses fewer requests.** Any list of requests, each valid for
  `g` and containing the slices it serves, whose members together cover `0 .. n`, has at least as
  many requests as the loop produces. The proof shows that a valid request cannot serve slices from
  two different loop requests, because the bytes between those two loop requests form an uncovered
  window longer than `g`, and then counts.

Modelling choices: byte positions are unbounded `Nat`; the Rust `saturating_add(1 + max_gap)` only
differs at `u64::MAX`. The `gap_bytes` field is carried as the Rust computes it but no theorem is
stated about it; it is diagnostic.
-/

namespace Kadmium.Data

structure Slice where
  start : Nat
  /-- Inclusive end, `byte_end_inclusive`. -/
  stop : Nat
  deriving DecidableEq, Repr

structure Req where
  start : Nat
  stop : Nat
  /-- Positions in the planned slice list, in byte order. -/
  members : List Nat
  gap : Nat
  deriving DecidableEq, Repr

/-- One iteration of the merge loop; `acc` holds the requests newest first, as `last_mut` sees
them. -/
def step (g : Nat) (acc : List Req) (i : Nat) (s : Slice) : List Req :=
  match acc with
  | last :: rest =>
    if s.start ≤ last.stop + 1 + g then
      { start := last.start
        stop := max last.stop s.stop
        members := last.members ++ [i]
        gap := last.gap + (if last.stop + 1 < s.start then s.start - last.stop - 1 else 0) } :: rest
    else ⟨s.start, s.stop, [i], 0⟩ :: last :: rest
  | [] => [⟨s.start, s.stop, [i], 0⟩]

/-- The loop over indices `0 .. n`, requests newest first. -/
def coalesceRev (g : Nat) (sl : Nat → Slice) (n : Nat) : List Req :=
  (List.range n).foldl (fun acc i => step g acc i (sl i)) []

/-- `coalesce_range_requests` for one URL: requests in byte order. -/
def coalesce (g : Nat) (sl : Nat → Slice) (n : Nat) : List Req := (coalesceRev g sl n).reverse

/-- The slices arrive sorted by start, as `sort_by` leaves them. -/
def Sorted (sl : Nat → Slice) (n : Nat) : Prop :=
  ∀ i j, i ≤ j → j < n → (sl i).start ≤ (sl j).start

/-- Every slice is non-empty. -/
def WellFormed (sl : Nat → Slice) (n : Nat) : Prop := ∀ i, i < n → (sl i).start ≤ (sl i).stop

/-- A request `[S, E]` serving the slices `M` fetches no run of more than `g` unrequested bytes:
every window `[p, p + g]` inside it touches a member. -/
def Valid (g : Nat) (sl : Nat → Slice) (S E : Nat) (M : List Nat) : Prop :=
  ∀ p, S ≤ p → p + g < E → ∃ m ∈ M, (sl m).start ≤ p + g ∧ p ≤ (sl m).stop

theorem coalesceRev_succ (g : Nat) (sl : Nat → Slice) (k : Nat) :
    coalesceRev g sl (k + 1) = step g (coalesceRev g sl k) k (sl k) := by
  unfold coalesceRev
  rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

/-! ## The loop invariant -/

/-- What holds of the newest-first request list after the first `k` slices. -/
structure Inv (g : Nat) (sl : Nat → Slice) (n k : Nat) (acc : List Req) : Prop where
  /-- Members in request order are `0 .. k`. -/
  flat : (acc.reverse.map Req.members).flatten = List.range k
  members_lt : ∀ r ∈ acc, ∀ m ∈ r.members, m < k
  nonempty : ∀ r ∈ acc, r.members ≠ []
  contains : ∀ r ∈ acc, ∀ m ∈ r.members, r.start ≤ (sl m).start ∧ (sl m).stop ≤ r.stop
  ordered : ∀ r ∈ acc, r.start ≤ r.stop
  /-- Some member reaches the request's end. -/
  attained : ∀ r ∈ acc, ∃ m ∈ r.members, (sl m).stop = r.stop
  /-- Some member starts the request. -/
  start_attained : ∀ r ∈ acc, ∃ m ∈ r.members, (sl m).start = r.start
  /-- Every request starts at or before every slice still to come. -/
  start_le : ∀ r ∈ acc, ∀ j, k ≤ j → j < n → r.start ≤ (sl j).start
  /-- Newer requests start more than `g` bytes after older ones end. -/
  separated : acc.Pairwise fun newer older => older.stop + 1 + g < newer.start
  valid : ∀ r ∈ acc, Valid g sl r.start r.stop r.members

theorem inv_zero (g : Nat) (sl : Nat → Slice) (n : Nat) : Inv g sl n 0 [] where
  flat := by simp
  members_lt := by simp
  nonempty := by simp
  contains := by simp
  ordered := by simp
  attained := by simp
  start_attained := by simp
  start_le := by simp
  separated := List.Pairwise.nil
  valid := by simp

theorem inv_step (g : Nat) (sl : Nat → Slice) (n k : Nat) (acc : List Req) (hs : Sorted sl n)
    (hw : WellFormed sl n) (hk : k < n) (h : Inv g sl n k acc) :
    Inv g sl n (k + 1) (step g acc k (sl k)) := by
  have hwk := hw k hk
  cases acc with
  | nil =>
    -- The first request.
    have hflat := h.flat
    simp only [List.reverse_nil, List.map_nil, List.flatten_nil] at hflat
    have hk0 : k = 0 := by
      have := congrArg List.length hflat
      simpa using this.symm
    subst hk0
    simp only [step]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp
    · intro r hr m hm
      simp only [List.mem_singleton] at hr
      subst hr
      simp only [List.mem_singleton] at hm
      omega
    · simp
    · intro r hr m hm
      simp only [List.mem_singleton] at hr
      subst hr
      simp only [List.mem_singleton] at hm
      subst hm
      exact ⟨Nat.le_refl _, Nat.le_refl _⟩
    · intro r hr
      simp only [List.mem_singleton] at hr
      subst hr
      exact hwk
    · intro r hr
      simp only [List.mem_singleton] at hr
      subst hr
      exact ⟨0, by simp, rfl⟩
    · intro r hr
      simp only [List.mem_singleton] at hr
      subst hr
      exact ⟨0, by simp, rfl⟩
    · intro r hr j _ hjn
      simp only [List.mem_singleton] at hr
      subst hr
      exact hs 0 j (Nat.zero_le _) hjn
    · simp
    · intro r hr p hp hpg
      simp only [List.mem_singleton] at hr
      subst hr
      simp only at hp hpg
      exact ⟨0, by simp, by omega, by omega⟩
  | cons last rest =>
    simp only [step]
    split
    · -- Merge into `last`.
      rename_i hmerge
      have hlast : last ∈ last :: rest := by simp
      have hord := h.ordered last hlast
      obtain ⟨ma, hma, hmas⟩ := h.attained last hlast
      have hwma := hw ma (by have := h.members_lt last hlast ma hma; omega)
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · have := h.flat
        simp only [List.reverse_cons, List.map_append, List.map_cons, List.map_nil,
          List.flatten_append, List.flatten_cons, List.flatten_nil, List.append_nil] at this ⊢
        rw [List.range_succ, ← this, List.append_assoc]
      · intro r hr m hm
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · simp only [List.mem_append, List.mem_singleton] at hm
          rcases hm with hm | rfl
          · have := h.members_lt last hlast m hm; omega
          · omega
        · have := h.members_lt r (by simp [hr]) m hm; omega
      · intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · simp
        · exact h.nonempty r (by simp [hr])
      · intro r hr m hm
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · simp only [List.mem_append, List.mem_singleton] at hm
          rcases hm with hm | rfl
          · have := h.contains last hlast m hm
            exact ⟨this.1, by simp only; omega⟩
          · have := h.start_le last hlast m (Nat.le_refl _) hk
            exact ⟨this, by simp only; omega⟩
        · exact h.contains r (by simp [hr]) m hm
      · intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · simp only; omega
        · exact h.ordered r (by simp [hr])
      · intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · by_cases hc : (sl k).stop ≤ last.stop
          · exact ⟨ma, by simp [hma], by simp only; omega⟩
          · exact ⟨k, by simp, by simp only; omega⟩
        · exact h.attained r (by simp [hr])
      · intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · obtain ⟨m, hm, hms⟩ := h.start_attained last hlast
          exact ⟨m, by simp [hm], hms⟩
        · exact h.start_attained r (by simp [hr])
      · intro r hr j hj hjn
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · exact h.start_le last hlast j (by omega) hjn
        · exact h.start_le r (by simp [hr]) j (by omega) hjn
      · have := h.separated
        rw [List.pairwise_cons] at this ⊢
        exact ⟨fun older ho => this.1 older ho, this.2⟩
      · intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · intro p hp hpg
          simp only at hp hpg
          by_cases hpl : p + g < last.stop
          · obtain ⟨m, hm, hm1, hm2⟩ := h.valid last hlast p hp hpl
            exact ⟨m, by simp [hm], hm1, hm2⟩
          · by_cases hpl2 : p ≤ last.stop
            · -- The member that reaches `last.stop` meets the window there.
              exact ⟨ma, by simp [hma], by omega, by omega⟩
            · -- Past `last.stop`, the new slice starts within reach and ends past the window.
              have : last.stop < (sl k).stop := by
                rw [Nat.max_def] at hpg
                split at hpg <;> omega
              exact ⟨k, by simp, by omega, by omega⟩
        · exact h.valid r (by simp [hr])
    · -- Start a new request.
      rename_i hfar
      have hlast : last ∈ last :: rest := by simp
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · have := h.flat
        simp only [List.reverse_cons, List.map_append, List.map_cons, List.map_nil,
          List.flatten_append, List.flatten_cons, List.flatten_nil, List.append_nil] at this ⊢
        rw [List.range_succ, ← this, List.append_assoc]
      · intro r hr m hm
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · simp only [List.mem_singleton] at hm; omega
        · have := h.members_lt r (List.mem_cons.mpr hr) m hm; omega
      · intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · simp
        · exact h.nonempty r (List.mem_cons.mpr hr)
      · intro r hr m hm
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · simp only [List.mem_singleton] at hm
          subst hm
          exact ⟨Nat.le_refl _, Nat.le_refl _⟩
        · exact h.contains r (List.mem_cons.mpr hr) m hm
      · intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · exact hwk
        · exact h.ordered r (List.mem_cons.mpr hr)
      · intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · exact ⟨k, by simp, rfl⟩
        · exact h.attained r (List.mem_cons.mpr hr)
      · intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · exact ⟨k, by simp, rfl⟩
        · exact h.start_attained r (List.mem_cons.mpr hr)
      · intro r hr j hj hjn
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · exact hs k j (by omega) hjn
        · exact h.start_le r (List.mem_cons.mpr hr) j (by omega) hjn
      · rw [List.pairwise_cons]
        refine ⟨?_, h.separated⟩
        intro older ho
        simp only [List.mem_cons] at ho
        rcases ho with rfl | ho
        · simp only; omega
        · have hsep := h.separated
          rw [List.pairwise_cons] at hsep
          have h1 := hsep.1 older ho
          have h2 := h.ordered last hlast
          simp only; omega
      · intro r hr
        simp only [List.mem_cons] at hr
        rcases hr with rfl | hr
        · intro p hp hpg
          simp only at hp hpg
          exact ⟨k, by simp, by omega, by omega⟩
        · exact h.valid r (List.mem_cons.mpr hr)

theorem inv_coalesceRev (g : Nat) (sl : Nat → Slice) (n : Nat) (hs : Sorted sl n)
    (hw : WellFormed sl n) : ∀ k, k ≤ n → Inv g sl n k (coalesceRev g sl k) := by
  intro k
  induction k with
  | zero => intro _; exact inv_zero g sl n
  | succ k ih =>
    intro hk
    rw [coalesceRev_succ]
    exact inv_step g sl n k _ hs hw (by omega) (ih (by omega))

/-! ## Properties of the plan -/

section Plan

variable {g : Nat} {sl : Nat → Slice} {n : Nat}

/-- **Partition.** The members of the requests, in request order, are the slice indices `0 .. n`,
each exactly once. -/
theorem coalesce_members (hs : Sorted sl n) (hw : WellFormed sl n) :
    ((coalesce g sl n).map Req.members).flatten = List.range n :=
  (inv_coalesceRev g sl n hs hw n (Nat.le_refl n)).flat

/-- **Containment.** Every member slice lies inside its request's byte range. -/
theorem coalesce_contains (hs : Sorted sl n) (hw : WellFormed sl n) :
    ∀ r ∈ coalesce g sl n, ∀ m ∈ r.members,
      r.start ≤ (sl m).start ∧ (sl m).stop ≤ r.stop := by
  intro r hr
  exact (inv_coalesceRev g sl n hs hw n (Nat.le_refl n)).contains r (List.mem_reverse.mp hr)

theorem coalesce_ordered (hs : Sorted sl n) (hw : WellFormed sl n) :
    ∀ r ∈ coalesce g sl n, r.start ≤ r.stop := by
  intro r hr
  exact (inv_coalesceRev g sl n hs hw n (Nat.le_refl n)).ordered r (List.mem_reverse.mp hr)

theorem coalesce_nonempty (hs : Sorted sl n) (hw : WellFormed sl n) :
    ∀ r ∈ coalesce g sl n, r.members ≠ [] := by
  intro r hr
  exact (inv_coalesceRev g sl n hs hw n (Nat.le_refl n)).nonempty r (List.mem_reverse.mp hr)

theorem coalesce_members_lt (hs : Sorted sl n) (hw : WellFormed sl n) :
    ∀ r ∈ coalesce g sl n, ∀ m ∈ r.members, m < n := by
  intro r hr
  exact (inv_coalesceRev g sl n hs hw n (Nat.le_refl n)).members_lt r (List.mem_reverse.mp hr)

/-- **Tight ends.** Every request begins at a member's first byte and ends at a member's last
byte, so it fetches no unrequested byte before its first member or after its last. -/
theorem coalesce_start_attained (hs : Sorted sl n) (hw : WellFormed sl n) :
    ∀ r ∈ coalesce g sl n, ∃ m ∈ r.members, (sl m).start = r.start := by
  intro r hr
  exact (inv_coalesceRev g sl n hs hw n (Nat.le_refl n)).start_attained r (List.mem_reverse.mp hr)

theorem coalesce_stop_attained (hs : Sorted sl n) (hw : WellFormed sl n) :
    ∀ r ∈ coalesce g sl n, ∃ m ∈ r.members, (sl m).stop = r.stop := by
  intro r hr
  exact (inv_coalesceRev g sl n hs hw n (Nat.le_refl n)).attained r (List.mem_reverse.mp hr)

/-- **Separation.** An earlier request ends more than `g` bytes before a later one starts. -/
theorem coalesce_separated (hs : Sorted sl n) (hw : WellFormed sl n) :
    (coalesce g sl n).Pairwise fun older newer => older.stop + 1 + g < newer.start := by
  have := (inv_coalesceRev g sl n hs hw n (Nat.le_refl n)).separated
  unfold coalesce
  rw [List.pairwise_reverse]
  exact this

/-- **Soundness.** Every request the loop produces is valid for `g`. -/
theorem coalesce_valid (hs : Sorted sl n) (hw : WellFormed sl n) :
    ∀ r ∈ coalesce g sl n, Valid g sl r.start r.stop r.members := by
  intro r hr
  exact (inv_coalesceRev g sl n hs hw n (Nat.le_refl n)).valid r (List.mem_reverse.mp hr)

end Plan

/-! ## Optimality -/

/-- Lists whose concatenation has no duplicates are pairwise disjoint. -/
theorem disjoint_of_nodup_flatten :
    ∀ (L : List (List Nat)), L.flatten.Nodup → L.Pairwise fun b b' => ∀ x ∈ b, x ∉ b' := by
  intro L
  induction L with
  | nil => intro _; exact List.Pairwise.nil
  | cons b rest ih =>
    intro h
    rw [List.flatten_cons, List.nodup_append] at h
    obtain ⟨_, h2, h3⟩ := h
    rw [List.pairwise_cons]
    exact ⟨fun b' hb' x hx hx' => h3 x hx x (List.mem_flatten.mpr ⟨b', hb', hx'⟩) rfl, ih h2⟩

/-- Blocks that are non-empty and pairwise disjoint, requests that each lie inside one block and
together cover every block: there are at least as many requests as blocks. -/
theorem blocks_le_requests :
    ∀ (blocks reqs : List (List Nat)),
      (∀ b ∈ blocks, b ≠ []) →
      blocks.Pairwise (fun b b' => ∀ x ∈ b, x ∉ b') →
      (∀ r ∈ reqs, ∀ b ∈ blocks, ∀ b' ∈ blocks, (∃ x ∈ r, x ∈ b) → (∃ y ∈ r, y ∈ b') → b = b') →
      (∀ b ∈ blocks, ∀ x ∈ b, ∃ r ∈ reqs, x ∈ r) →
      blocks.length ≤ reqs.length := by
  intro blocks
  induction blocks with
  | nil => intro _ _ _ _ _; simp
  | cons b rest ih =>
    intro reqs hne hdisj hsub hcover
    have hb : b ≠ [] := hne b (by simp)
    obtain ⟨x, hx⟩ : ∃ x, x ∈ b := by
      cases b with
      | nil => exact absurd rfl hb
      | cons y _ => exact ⟨y, by simp⟩
    obtain ⟨r, hr, hxr⟩ := hcover b (by simp) x hx
    rw [List.pairwise_cons] at hdisj
    have hrest := ih (reqs.erase r) (fun b' hb' => hne b' (by simp [hb'])) hdisj.2
      (fun r' hr' b1 hb1 b2 hb2 => hsub r' (List.mem_of_mem_erase hr') b1 (by simp [hb1]) b2
        (by simp [hb2]))
      (by
        intro b' hb' y hy
        obtain ⟨r', hr', hyr'⟩ := hcover b' (by simp [hb']) y hy
        refine ⟨r', ?_, hyr'⟩
        apply (List.mem_erase_of_ne ?_).mpr hr'
        intro heq
        subst heq
        have hbb := hsub r' hr' b (by simp) b' (by simp [hb']) ⟨x, hxr, hx⟩ ⟨y, hyr', hy⟩
        exact hdisj.1 b' hb' x hx (hbb ▸ hx))
    rw [List.length_erase_of_mem hr] at hrest
    have : 0 < reqs.length := List.length_pos_of_mem hr
    simp only [List.length_cons]
    omega

section Optimal

variable {g : Nat} {sl : Nat → Slice} {n : Nat}

/-- A slice index belongs to the request of the plan that lists it. -/
theorem mem_members_of_flat (hs : Sorted sl n) (hw : WellFormed sl n) {i : Nat} (hi : i < n) :
    ∃ r ∈ coalesce g sl n, i ∈ r.members := by
  have hf := coalesce_members (g := g) hs hw
  have : i ∈ ((coalesce g sl n).map Req.members).flatten := by
    rw [hf]; exact List.mem_range.mpr hi
  rw [List.mem_flatten] at this
  obtain ⟨l, hl, hil⟩ := this
  rw [List.mem_map] at hl
  obtain ⟨r, hr, rfl⟩ := hl
  exact ⟨r, hr, hil⟩

/-- Distinct requests of the plan have no slice in common. -/
theorem coalesce_disjoint (hs : Sorted sl n) (hw : WellFormed sl n) :
    ((coalesce g sl n).map Req.members).Pairwise fun b b' => ∀ x ∈ b, x ∉ b' := by
  have hf := coalesce_members (g := g) hs hw
  apply disjoint_of_nodup_flatten
  rw [hf]
  exact List.nodup_range

/-- A request valid for `g` cannot serve slices from two different plan requests: the bytes
between the two plan requests are an uncovered window longer than `g`. -/
theorem valid_within_one (hs : Sorted sl n) (hw : WellFormed sl n) {S E : Nat} {M : List Nat}
    (hM : ∀ m ∈ M, m < n) (hcont : ∀ m ∈ M, S ≤ (sl m).start ∧ (sl m).stop ≤ E)
    (hv : Valid g sl S E M) {a b : Nat} (ha : a < (coalesce g sl n).length)
    (hb : b < (coalesce g sl n).length) {x y : Nat} (hx : x ∈ M) (hy : y ∈ M)
    (hxa : x ∈ (coalesce g sl n)[a].members) (hyb : y ∈ (coalesce g sl n)[b].members) : a = b := by
  have hsep := coalesce_separated (g := g) hs hw
  rw [List.pairwise_iff_getElem] at hsep
  -- With `ra` before `rb`, the window just past `ra` is inside `[S, E]` but touches no member.
  have key : ∀ a b (ha : a < (coalesce g sl n).length) (hb : b < (coalesce g sl n).length),
      ∀ x y, x ∈ M → y ∈ M → x ∈ ((coalesce g sl n)[a]'ha).members →
      y ∈ ((coalesce g sl n)[b]'hb).members → a < b → False := by
    intro a b ha hb x y hx hy hxa hyb hab
    have hlt := hsep a b ha hb hab
    have hra : (coalesce g sl n)[a] ∈ coalesce g sl n := List.getElem_mem ha
    have hrb : (coalesce g sl n)[b] ∈ coalesce g sl n := List.getElem_mem hb
    have hxin := coalesce_contains (g := g) hs hw _ hra x hxa
    have hyin := coalesce_contains (g := g) hs hw _ hrb y hyb
    have hxS := hcont x hx
    have hyE := hcont y hy
    have hwx := hw x (hM x hx)
    have hwy := hw y (hM y hy)
    have horda := coalesce_ordered (g := g) hs hw _ hra
    have hordb := coalesce_ordered (g := g) hs hw _ hrb
    obtain ⟨m, hm, hm1, hm2⟩ := hv ((coalesce g sl n)[a].stop + 1) (by omega) (by omega)
    obtain ⟨rc, hrc, hmc⟩ := mem_members_of_flat (g := g) hs hw (hM m hm)
    obtain ⟨c, hc, hrceq⟩ := List.mem_iff_getElem.mp hrc
    subst hrceq
    have hmin := coalesce_contains (g := g) hs hw _ hrc m hmc
    have hordc := coalesce_ordered (g := g) hs hw _ hrc
    rcases Nat.lt_trichotomy c a with hca | hca | hca
    · have := hsep c a hc ha hca
      omega
    · subst hca
      omega
    · rcases Nat.lt_trichotomy c b with hcb | hcb | hcb
      · have := hsep a c ha hc hca
        omega
      · subst hcb
        omega
      · have := hsep b c hb hc hcb
        omega
  rcases Nat.lt_trichotomy a b with hab | hab | hab
  · exact (key a b ha hb x y hx hy hxa hyb hab).elim
  · exact hab
  · exact (key b a hb ha y x hy hx hyb hxa hab).elim

/-- **Optimality.** Every list of requests, each valid for `g` and containing the slices it serves,
whose members together cover the slices `0 .. n`, has at least as many requests as the loop
produces. -/
theorem coalesce_optimal (hs : Sorted sl n) (hw : WellFormed sl n)
    (reqs : List (Nat × Nat × List Nat))
    (hvalid : ∀ q ∈ reqs, (∀ m ∈ q.2.2, m < n ∧ q.1 ≤ (sl m).start ∧ (sl m).stop ≤ q.2.1) ∧
      Valid g sl q.1 q.2.1 q.2.2)
    (hcover : ∀ i, i < n → ∃ q ∈ reqs, i ∈ q.2.2) :
    (coalesce g sl n).length ≤ reqs.length := by
  have := blocks_le_requests ((coalesce g sl n).map Req.members)
    (reqs.map fun q : Nat × Nat × List Nat => q.2.2) ?_ ?_ ?_ ?_
  · simpa using this
  · intro b hb
    rw [List.mem_map] at hb
    obtain ⟨r, hr, rfl⟩ := hb
    exact coalesce_nonempty (g := g) hs hw r hr
  · exact coalesce_disjoint (g := g) hs hw
  · intro r hr b hb b' hb' ⟨x, hxr, hxb⟩ ⟨y, hyr, hyb'⟩
    rw [List.mem_map] at hr hb hb'
    obtain ⟨q, hq, rfl⟩ := hr
    obtain ⟨ra, hra, rfl⟩ := hb
    obtain ⟨rb, hrb, rfl⟩ := hb'
    obtain ⟨a, ha, rfl⟩ := List.mem_iff_getElem.mp hra
    obtain ⟨b, hb, rfl⟩ := List.mem_iff_getElem.mp hrb
    have hq' := hvalid q hq
    have := valid_within_one (g := g) hs hw (fun m hm => (hq'.1 m hm).1)
      (fun m hm => (hq'.1 m hm).2) hq'.2 ha hb hxr hyr hxb hyb'
    subst this
    rfl
  · intro b hb x hx
    rw [List.mem_map] at hb
    obtain ⟨r, hr, rfl⟩ := hb
    have hxn := coalesce_members_lt (g := g) hs hw r hr x hx
    obtain ⟨q, hq, hxq⟩ := hcover x hxn
    exact ⟨q.2.2, List.mem_map.mpr ⟨q, hq, rfl⟩, hxq⟩

end Optimal

end Kadmium.Data
