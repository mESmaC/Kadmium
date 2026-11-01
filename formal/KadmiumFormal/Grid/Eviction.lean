/-!
# Eviction candidates

Models the selection loop of `ReplicaCatalog::evict_candidates` in
`crates/kadmium-grid/src/catalog.rs`. After the unpinned replicas of a node are sorted by
`(hits, last_access_unix_ms)`, the loop is

```rust
for r in candidates {
    if freed >= bytes_needed { break; }
    freed += r.content.bytes;
    out.push(r.content.clone());
}
```

The model takes the sorted byte sizes and returns how many leading candidates are chosen.
Results: the chosen replicas are a prefix of the sorted order (`pickCount_le`), they free at least
`bytes_needed` unless every unpinned replica is offered (`pick_reaches`), and no shorter prefix
would do (`pick_minimal`). The claim is about prefixes of the sorted order: the loop offers the
shortest sufficient prefix of the least valuable replicas, not the smallest sufficient subset over
all replicas, which it does not search for. Pinned replicas are filtered before the loop and are
never offered.

Modelling choices: byte counts are unbounded `Nat`; the Rust accumulates `freed` in a `u64` with
unchecked `+=`, which wraps only past `2^64` bytes of replicas on one node.
-/

namespace Kadmium.Grid

/-- Number of leading candidates the loop pushes, starting with `freed` bytes already counted. -/
def pickCount (need : Nat) : Nat → List Nat → Nat
  | _, [] => 0
  | freed, b :: rest => if need ≤ freed then 0 else pickCount need (freed + b) rest + 1

theorem pickCount_le (need freed : Nat) (l : List Nat) : pickCount need freed l ≤ l.length := by
  induction l generalizing freed with
  | nil => simp [pickCount]
  | cons b rest ih =>
    simp only [pickCount, List.length_cons]
    split
    · omega
    · have := ih (freed + b); omega

/-- The chosen prefix frees `need` bytes, or it is every candidate. -/
theorem pick_reaches (need : Nat) :
    ∀ (freed : Nat) (l : List Nat),
      need ≤ freed + (l.take (pickCount need freed l)).sum ∨ pickCount need freed l = l.length := by
  intro freed l
  induction l generalizing freed with
  | nil => simp [pickCount]
  | cons b rest ih =>
    simp only [pickCount]
    split
    · left; simp only [List.take_zero, List.sum_nil]; omega
    · rcases ih (freed + b) with h | h
      · left
        simp only [List.take_succ_cons, List.sum_cons]
        omega
      · right
        simp only [h, List.length_cons]

/-- Every strictly shorter prefix frees fewer than `need` bytes. -/
theorem pick_minimal (need : Nat) :
    ∀ (freed : Nat) (l : List Nat) (j : Nat), j < pickCount need freed l →
      freed + (l.take j).sum < need := by
  intro freed l
  induction l generalizing freed with
  | nil => intro j h; simp [pickCount] at h
  | cons b rest ih =>
    intro j h
    simp only [pickCount] at h
    split at h
    · omega
    · cases j with
      | zero => simp; omega
      | succ j =>
        have := ih (freed + b) j (by omega)
        simp only [List.take_succ_cons, List.sum_cons]
        omega

/-- The call site starts with nothing freed. -/
theorem evict_spec (need : Nat) (sizes : List Nat) :
    let k := pickCount need 0 sizes
    k ≤ sizes.length ∧ (need ≤ (sizes.take k).sum ∨ k = sizes.length) ∧
      ∀ j, j < k → (sizes.take j).sum < need := by
  refine ⟨pickCount_le need 0 sizes, ?_, fun j hj => ?_⟩
  · have := pick_reaches need 0 sizes
    simpa using this
  · have := pick_minimal need 0 sizes j hj
    simpa using this

end Kadmium.Grid
