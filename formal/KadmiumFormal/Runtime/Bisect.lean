import KadmiumFormal.Pow2

/-!
# Verde layer bisection

Models the search in `VerdeVerificationEngine::bisect_faulty_layer` in
`crates/kadmium-runtime/src/lib.rs`. Given `L + 1` worker states and `L + 1` reference states,
`diverge i` says whether state `i` differs from the reference by more than the tolerance. When the
final state diverges, the Rust runs

```rust
let mut low = 1; let mut high = total_layers;
while low < high {
    let mid = low + (high - low) / 2;
    if states_diverge(mid) { high = mid } else { low = mid + 1 }
}
let faulty_layer = low - 1;
```

and then spends one more referee evaluation on `faulty_layer`.

The doc comment promises the exact layer whose output diverged while its input still matched,
in at most `⌈log₂ L⌉ + 1` referee evaluations. The search does not assume that divergence is
monotone along the trajectory, and neither do the theorems:

* `bisect_spec`: the returned layer `f` has `diverge (f + 1)`, and either `f = 0` or
  `¬ diverge f`; `f < L`; and the evaluation count is at most `ceilLog2 L + 1`. The `f = 0` case
  is honest: state `0` is the worker's input and is never compared, so a diverging input is
  reported as layer `0`.
* `bisect_none`: without a diverging final state, or with no layers, there is nothing to find.

The model returns the search result before the final referee confirmation; the Rust returns
`None` if that confirmation does not reproduce the divergence.
-/

namespace Kadmium.Runtime

open Kadmium

/-- The `while low < high` loop, counting evaluations in `steps`. -/
def loop (diverge : Nat → Bool) (low high steps : Nat) : Nat × Nat :=
  if _h : low < high then
    let mid := low + (high - low) / 2
    if diverge mid then loop diverge low mid (steps + 1)
    else loop diverge (mid + 1) high (steps + 1)
  else (low, steps)
termination_by high - low
decreasing_by all_goals omega

/-- `bisect_faulty_layer` up to the final confirmation: `(faulty_layer, referee_evaluations_used)`.
`L` is `worker_states.len() - 1`; the Rust returns `None` for fewer than two states. -/
def bisect (diverge : Nat → Bool) (L : Nat) : Option (Nat × Nat) :=
  if L < 1 then none
  else if !diverge L then none
  else
    let r := loop diverge 1 L 0
    some (r.1 - 1, r.2 + 1)

theorem ceilLog2_succ_half {n : Nat} (h : 1 < n) : ceilLog2 n = ceilLog2 ((n + 1) / 2) + 1 := by
  rw [ceilLog2]
  simp [show ¬ n ≤ 1 by omega]

theorem loop_spec (diverge : Nat → Bool) :
    ∀ n low high steps, high - low = n → 1 ≤ low → low ≤ high → diverge high = true →
      (low = 1 ∨ diverge (low - 1) = false) →
      let r := loop diverge low high steps
      low ≤ r.1 ∧ r.1 ≤ high ∧ diverge r.1 = true ∧ (r.1 = 1 ∨ diverge (r.1 - 1) = false) ∧
        r.2 ≤ steps + ceilLog2 (high - low + 1) := by
  intro n
  induction n using Nat.strongRecOn with
  | ind n ih =>
    intro low high steps hn h1 hlh hdiv hinv
    rw [loop]
    by_cases hlt : low < high
    · simp only [hlt, ↓reduceDIte]
      have hsize : 1 < high - low + 1 := by omega
      rw [ceilLog2_succ_half hsize]
      cases hd : diverge (low + (high - low) / 2)
      · simp only [Bool.false_eq_true, ↓reduceIte]
        have := ih (high - (low + (high - low) / 2 + 1)) (by omega) (low + (high - low) / 2 + 1)
          high (steps + 1) rfl (by omega) (by omega) hdiv (Or.inr (by simpa using hd))
        refine ⟨by omega, this.2.1, this.2.2.1, this.2.2.2.1, ?_⟩
        have hmono := ceilLog2_mono (show high - (low + (high - low) / 2 + 1) + 1 ≤
          (high - low + 1 + 1) / 2 by omega)
        omega
      · simp only [↓reduceIte]
        have := ih (low + (high - low) / 2 - low) (by omega) low (low + (high - low) / 2)
          (steps + 1) rfl h1 (by omega) hd hinv
        refine ⟨this.1, by omega, this.2.2.1, this.2.2.2.1, ?_⟩
        have heq : low + (high - low) / 2 - low + 1 = (high - low + 1 + 1) / 2 := by omega
        rw [heq] at this
        omega
    · simp only [hlt, ↓reduceDIte]
      have : low = high := by omega
      subst this
      exact ⟨Nat.le_refl _, Nat.le_refl _, hdiv, hinv, by omega⟩

/-- **Bisection.** With `L ≥ 1` layers and a diverging final state, the search returns a layer `f`
whose output state diverges while its input state does not (or is the uncompared input `0`), in
at most `⌈log₂ L⌉ + 1` referee evaluations. -/
theorem bisect_spec {diverge : Nat → Bool} {L : Nat} (hL : 1 ≤ L) (hd : diverge L = true) :
    ∃ f s, bisect diverge L = some (f, s) ∧ diverge (f + 1) = true ∧
      (f = 0 ∨ diverge f = false) ∧ f < L ∧ s ≤ ceilLog2 L + 1 := by
  have := loop_spec diverge (L - 1) 1 L 0 (by omega) (Nat.le_refl 1) hL hd (Or.inl rfl)
  obtain ⟨h1, h2, h3, h4, h5⟩ := this
  refine ⟨(loop diverge 1 L 0).1 - 1, (loop diverge 1 L 0).2 + 1, ?_, ?_, ?_, by omega, ?_⟩
  · simp [bisect, show ¬ L < 1 by omega, hd]
  · rwa [Nat.sub_add_cancel h1]
  · rcases h4 with h | h
    · left; omega
    · right; exact h
  · have : L - 1 + 1 = L := by omega
    rw [this] at h5
    omega

theorem bisect_none {diverge : Nat → Bool} {L : Nat} (h : L = 0 ∨ diverge L = false) :
    bisect diverge L = none := by
  rcases h with h | h
  · simp [bisect, h]
  · simp [bisect, h]

end Kadmium.Runtime
