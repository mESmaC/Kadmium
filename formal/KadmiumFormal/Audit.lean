import Lean
import KadmiumFormal.Pow2
import KadmiumFormal.RepOps.TreeSum
import KadmiumFormal.RepOps.TreeReducer

/-!
# Axiom audit

Every theorem cited from `formal/README.md` or the architecture docs is listed here. The build
fails if any of them depends on an axiom other than Lean's three standard ones (`propext`,
`Classical.choice`, `Quot.sound`). That rules out `sorry` (`sorryAx`), `native_decide`
(`Lean.ofReduceBool`) and any project-local `axiom`.

Add a line here when you add a theorem that a document cites.
-/

open Lean Elab Command

/-- Fails unless every axiom `id` depends on is one of Lean's standard axioms. -/
elab "#assert_standard_axioms " id:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo id
  let axioms ← liftCoreM <| collectAxioms name
  let allowed := [``propext, ``Classical.choice, ``Quot.sound]
  for ax in axioms do
    unless allowed.contains ax do
      throwError "{name} depends on the non-standard axiom {ax}"

#assert_standard_axioms Kadmium.nextPow2_spec
#assert_standard_axioms Kadmium.RepOps.treeSum_eq_eval
#assert_standard_axioms Kadmium.RepOps.canonical_leaves
#assert_standard_axioms Kadmium.RepOps.canonical_optimal
#assert_standard_axioms Kadmium.RepOps.legacy_fallback_unreachable
#assert_standard_axioms Kadmium.RepOps.legacySum_eq_treeSum
#assert_standard_axioms Kadmium.RepOps.reducer_eq_treeSum
#assert_standard_axioms Kadmium.RepOps.push_pastEnd
#assert_standard_axioms Kadmium.RepOps.push_twice
#assert_standard_axioms Kadmium.RepOps.finish_incomplete
