/-!
# Redundancy policy normalisation

Models `RedundancyPolicy::normalized` in `crates/kadmium-grid/src/work.rs`:

```rust
self.min_quorum = self.min_quorum.max(1);
self.target_replicas = self.target_replicas.max(self.min_quorum);
self.max_replicas = self.max_replicas.max(self.target_replicas);
```

The scheduler validates a unit once `min_quorum` results agree and never creates more than
`max_replicas` instances, so the three counts must satisfy `1 ≤ min_quorum ≤ target_replicas ≤
max_replicas` or a unit could wait for more agreeing results than it is allowed to produce.

Results: `normalized_valid` (the invariant holds after normalisation), `normalized_idem`
(normalising twice is normalising once), `normalized_eq_of_valid` (a valid policy is unchanged),
`le_normalized` (normalisation never lowers a count, so it never weakens a quorum the operator
asked for) and `normalized_mono`.
-/

namespace Kadmium.Grid

/-- The three counts of a `RedundancyPolicy`; `validation` is not involved. -/
structure Redundancy where
  minQuorum : Nat
  targetReplicas : Nat
  maxReplicas : Nat
  deriving DecidableEq, Repr

namespace Redundancy

def Valid (p : Redundancy) : Prop :=
  1 ≤ p.minQuorum ∧ p.minQuorum ≤ p.targetReplicas ∧ p.targetReplicas ≤ p.maxReplicas

instance (p : Redundancy) : Decidable p.Valid := by unfold Valid; infer_instance

/-- `RedundancyPolicy::normalized`. -/
def normalized (p : Redundancy) : Redundancy :=
  let q := max p.minQuorum 1
  let t := max p.targetReplicas q
  let m := max p.maxReplicas t
  ⟨q, t, m⟩

theorem normalized_valid (p : Redundancy) : p.normalized.Valid := by
  simp only [normalized, Valid]
  omega

theorem normalized_idem (p : Redundancy) : p.normalized.normalized = p.normalized := by
  simp only [normalized, Redundancy.mk.injEq]
  omega

theorem normalized_eq_of_valid {p : Redundancy} (h : p.Valid) : p.normalized = p := by
  obtain ⟨h1, h2, h3⟩ := h
  cases p
  simp only [normalized, Redundancy.mk.injEq]
  simp only at h1 h2 h3
  omega

/-- Normalisation only raises counts: a quorum the operator asked for is never weakened. -/
theorem le_normalized (p : Redundancy) :
    p.minQuorum ≤ p.normalized.minQuorum ∧ p.targetReplicas ≤ p.normalized.targetReplicas ∧
      p.maxReplicas ≤ p.normalized.maxReplicas := by
  simp only [normalized]
  omega

theorem normalized_mono {p q : Redundancy} (h1 : p.minQuorum ≤ q.minQuorum)
    (h2 : p.targetReplicas ≤ q.targetReplicas) (h3 : p.maxReplicas ≤ q.maxReplicas) :
    p.normalized.minQuorum ≤ q.normalized.minQuorum ∧
      p.normalized.targetReplicas ≤ q.normalized.targetReplicas ∧
      p.normalized.maxReplicas ≤ q.normalized.maxReplicas := by
  simp only [normalized]
  omega

/-- The only fixed points are the valid policies. -/
theorem valid_iff_normalized_eq (p : Redundancy) : p.Valid ↔ p.normalized = p := by
  constructor
  · exact normalized_eq_of_valid
  · intro h
    have := normalized_valid p
    rwa [h] at this

end Redundancy

end Kadmium.Grid
