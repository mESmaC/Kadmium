/-!
# Native checkpoint layer placement across machines

Models `CheckpointMemoryProfile`, `order_candidate_machines`, and
`LayerPlacementPlanner::plan_with_budgets` in `crates/kadmium-sharding/src/layer_placement.rs`
(and the checkpoint wrapper in `crates/kadmium-runtime/src/native_decoder/placement.rs`).

Given a checkpoint's real in-memory weight sizes (`embeddingBytes`, per-layer `layerWeightBytes`,
`finalNormBytes`, `lmHeadBytes`, and `tieWordEmbeddings`) plus the per-layer `f32` KV cache bytes
for the model's full `maxPositionEmbeddings` context (`kvBytesPerLayer`), the planner assigns
contiguous non-empty layer ranges `[startLayer, endLayer)` to a subsequence of candidate machines.

Proved results:

* `planFrom_valid`: every accepted placement satisfies `ValidChain`.
* `stages_contiguous_cover`: an accepted placement partitions `[start, numLayers)` into contiguous
  non-empty ranges in order without gaps or overlaps, and is non-empty whenever `start < numLayers`.
* `stages_within_budget`: every stage's `weightBytes` and full-context `kvBytes` add up to
  `totalBytes`, and `totalBytes ≤ capacityBytes` with `0 < capacityBytes`.
* `stages_boundary_trusted`: stage 0 (`startLayer = 0`, `embedding = true`) and the final stage
  (`endLayer = numLayers`, `head = true`) are placed only on `TrustedCore` machines (`trusted = true`).
* `single_stage_shares_tied_embedding`: a single-machine placement (`0 .. numLayers`) of a model
  with `tieWordEmbeddings = true` counts the shared embedding matrix once, whereas splitting across
  two stages counts `embeddingBytes` on stage 0 and `lmHeadBytes` on the final stage.
* `planFrom_complete` and `planFrom_none_iff_infeasible`: `planFrom` returns `none` if and only if
  no valid full-context placement exists on the candidate machines.
* `planFrom_optimal`: among all valid placements on the candidate machines, `planFrom` minimizes
  the bottleneck per-machine relative memory load `totalBytes / capacityBytes` (compared exactly
  by cross-multiplication `loadLe`).
* `infeasible_when_total_exceeds_cluster_capacity`: the sum of stage `totalBytes` across any valid
  chain is bounded by the sum of `capacityBytes` across the candidate machines.

Modelling choices:

* Byte counts, layer indices and machine identifiers are unbounded `Nat`. In Rust, byte totals use
  `checked_add` / `checked_mul` in `u64` and cross-multiplication uses `u128` (`u64::MAX ^ 2 < u128::MAX`).
* `Machine.trusted` models `NodeTrustTier::can_host_boundary_ingress_egress()`.
* `Machine.capacityBytes` models the live polite-state resident memory offer `fast_bytes + warm_bytes`
  (never cold NVMe or NAS storage).
-/

namespace Kadmium.Sharding.LayerPlacement

/-- Exact byte profile of a checkpoint and its full-context KV cache. -/
structure Profile where
  numLayers : Nat
  maxPositionEmbeddings : Nat
  embeddingBytes : Nat
  layerWeightBytes : List Nat
  finalNormBytes : Nat
  lmHeadBytes : Nat
  tieWordEmbeddings : Bool
  kvBytesPerLayer : Nat
  deriving Repr, DecidableEq

/-- A candidate machine's resident memory capacity and boundary trust flag. -/
structure Machine where
  nodeId : Nat
  capacityBytes : Nat
  trusted : Bool
  deriving Repr, DecidableEq, Inhabited

/-- One planned pipeline stage running layers `[startLayer, endLayer)`. -/
structure Stage where
  nodeId : Nat
  startLayer : Nat
  endLayer : Nat
  embedding : Bool
  head : Bool
  weightBytes : Nat
  kvBytes : Nat
  totalBytes : Nat
  capacityBytes : Nat
  trusted : Bool
  deriving Repr, DecidableEq, Inhabited

/-- Sum of `weights[i]` for `start ≤ i < stop`. -/
def rangeWeightSum (weights : List Nat) (start stop : Nat) : Nat :=
  ((weights.drop start).take (stop - start)).sum

/-- In-memory weight bytes for `start .. stop` with `embedding` and `head`. -/
def sliceWeightBytes (p : Profile) (start stop : Nat) (embedding head : Bool) : Nat :=
  let emb := if embedding then p.embeddingBytes else 0
  let layers := rangeWeightSum p.layerWeightBytes start stop
  let hd := if head then
    p.finalNormBytes + (if embedding && p.tieWordEmbeddings then 0 else p.lmHeadBytes)
  else 0
  emb + layers + hd

/-- Full-context `f32` KV cache bytes for `start .. stop`. -/
def sliceKvBytes (p : Profile) (start stop : Nat) : Nat :=
  (stop - start) * p.kvBytesPerLayer

/-- Total resident bytes (`weightBytes + kvBytes`) for `start .. stop`. -/
def sliceTotalBytes (p : Profile) (start stop : Nat) (embedding head : Bool) : Nat :=
  sliceWeightBytes p start stop embedding head + sliceKvBytes p start stop

/-- Builds the stage record for `m` running `start .. stop`. -/
def buildStage (p : Profile) (m : Machine) (start stop : Nat) : Stage :=
  let embedding := start == 0
  let head := stop == p.numLayers
  let w := sliceWeightBytes p start stop embedding head
  let kv := sliceKvBytes p start stop
  { nodeId := m.nodeId
    startLayer := start
    endLayer := stop
    embedding := embedding
    head := head
    weightBytes := w
    kvBytes := kv
    totalBytes := w + kv
    capacityBytes := m.capacityBytes
    trusted := m.trusted }

/-- Whether machine `m` can validly run layers `start .. stop`. -/
def stageValid (p : Profile) (m : Machine) (start stop : Nat) : Bool :=
  let st := buildStage p m start stop
  (start < stop) && (stop ≤ p.numLayers) &&
    (!(st.embedding || st.head) || m.trusted) &&
    (0 < m.capacityBytes) &&
    (st.totalBytes ≤ m.capacityBytes)

/-- Orders eligible machines so `TrustedCore` machines bookend middle-only machines when at least
    two trusted machines and at least one middle-only machine exist. -/
def orderMachines (ms : List Machine) : List Machine :=
  let active := ms.filter (fun m => 0 < m.capacityBytes)
  let trusted := active.filter (fun m => m.trusted)
  let middle := active.filter (fun m => !m.trusted)
  if 2 ≤ trusted.length && !middle.isEmpty then
    trusted.dropLast ++ middle ++ [trusted.getLast!]
  else
    trusted ++ middle

/-- Exact rational comparison `a.1 / a.2 ≤ b.1 / b.2` via cross-multiplication. -/
def loadLe (a b : Nat × Nat) : Bool :=
  decide (a.1 * b.2 ≤ b.1 * a.2)

/-- Maximum of two `(bytes, capacity)` load pairs, keeping `a` on ties when `loadLe b a`. -/
def maxLoad (a b : Nat × Nat) : Nat × Nat :=
  if loadLe b a then a else b

/-- Peak `(totalBytes, capacityBytes)` load across a chain of stages. -/
def planLoad : List Stage → Nat × Nat
  | [] => (0, 1)
  | st :: rest => maxLoad (st.totalBytes, st.capacityBytes) (planLoad rest)

/-- Keeps `best` unless `cand` has strictly smaller peak load. -/
def pickBetter (best : Option (List Stage)) (cand : List Stage) : Option (List Stage) :=
  match best with
  | none => some cand
  | some b => if loadLe (planLoad b) (planLoad cand) then some b else some cand

/-- Single cut-point step for machine `m` over `start .. stop`. -/
def tryStopStep (p : Profile) (m : Machine) (next : Nat → Option (List Stage))
    (start stop : Nat) (best : Option (List Stage)) : Option (List Stage) :=
  if stageValid p m start stop then
    match next stop with
    | some tail => pickBetter best (buildStage p m start stop :: tail)
    | none => best
  else best

/-- Evaluates all cut points `stop ∈ stops` for machine `m` starting at `start`. -/
def tryStops (p : Profile) (m : Machine) (next : Nat → Option (List Stage))
    (start : Nat) : List Nat → Option (List Stage) → Option (List Stage)
  | [], best => best
  | stop :: stops, best =>
    tryStops p m next start stops (tryStopStep p m next start stop best)

/-- Dynamic programming over the ordered candidate machines for suffix `[start, p.numLayers)`. -/
def planFrom (p : Profile) : List Machine → Nat → Option (List Stage)
  | [], start => if start == p.numLayers then some [] else none
  | m :: rest, start =>
    if start == p.numLayers then
      some []
    else if p.numLayers < start then
      none
    else
      let stops := (List.range (p.numLayers - start)).map (fun k => start + 1 + k)
      let withM := tryStops p m (planFrom p rest) start stops none
      match planFrom p rest start with
      | some skipPlan => pickBetter withM skipPlan
      | none => withM

/-- Top-level placement entry point matching `LayerPlacementPlanner::plan_with_budgets`. -/
def plan (p : Profile) (ms : List Machine) : Option (List Stage) :=
  if p.numLayers == 0 || p.maxPositionEmbeddings == 0 || p.layerWeightBytes.length != p.numLayers then
    none
  else
    planFrom p (orderMachines ms) 0

/-! ## Specification predicates -/

/-- `stages` is a valid contiguous chain of stages covering `[start, p.numLayers)` on a
    subsequence of `ms`. -/
inductive ValidChain (p : Profile) : List Machine → Nat → List Stage → Prop where
  | nil (ms : List Machine) : ValidChain p ms p.numLayers []
  | skip {m : Machine} {rest : List Machine} {start : Nat} {stages : List Stage} :
      start < p.numLayers →
      ValidChain p rest start stages →
      ValidChain p (m :: rest) start stages
  | take {m : Machine} {rest : List Machine} {start stop : Nat} {tail : List Stage} :
      stageValid p m start stop = true →
      ValidChain p rest stop tail →
      ValidChain p (m :: rest) start (buildStage p m start stop :: tail)

/-- Contiguous non-empty layer ranges tiling `[start, target)`. -/
def ContiguousCover : Nat → Nat → List Stage → Prop
  | start, target, [] => start = target
  | start, target, st :: rest =>
      st.startLayer = start ∧ st.startLayer < st.endLayer ∧ st.endLayer ≤ target ∧
        ContiguousCover st.endLayer target rest

/-- Every stage in `stages` satisfies its exact byte accounting and capacity bound. -/
def WithinBudget (p : Profile) (st : Stage) : Prop :=
  st.weightBytes = sliceWeightBytes p st.startLayer st.endLayer st.embedding st.head ∧
    st.kvBytes = sliceKvBytes p st.startLayer st.endLayer ∧
    st.totalBytes = st.weightBytes + st.kvBytes ∧
    0 < st.capacityBytes ∧
    st.totalBytes ≤ st.capacityBytes

/-- Every stage's `embedding` and `head` flags match its layer boundaries, and boundary stages run
    only on trusted machines. -/
def BoundaryTrusted (p : Profile) (st : Stage) : Prop :=
  st.embedding = (st.startLayer == 0) ∧
    st.head = (st.endLayer == p.numLayers) ∧
    ((st.embedding = true ∨ st.head = true) → st.trusted = true)

/-! ## Helper lemmas for `stageValid` and `buildStage` -/

theorem stageValid_iff (p : Profile) (m : Machine) (start stop : Nat) :
    stageValid p m start stop = true ↔
      start < stop ∧ stop ≤ p.numLayers ∧
        (((start == 0) = true ∨ (stop == p.numLayers) = true) → m.trusted = true) ∧
        0 < m.capacityBytes ∧
        (buildStage p m start stop).totalBytes ≤ m.capacityBytes := by
  unfold stageValid buildStage
  simp only [Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true', Bool.or_eq_false_iff,
    decide_eq_true_eq]
  constructor
  · rintro ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩
    refine ⟨h1, h2, ?_, h4, h5⟩
    intro hbd
    cases h3 with
    | inl hnot =>
      cases hbd with
      | inl hs => rw [hnot.1] at hs; contradiction
      | inr he => rw [hnot.2] at he; contradiction
    | inr htr => exact htr
  · rintro ⟨h1, h2, h3, h4, h5⟩
    refine ⟨⟨⟨⟨h1, h2⟩, ?_⟩, h4⟩, h5⟩
    by_cases hbd : (start == 0) = true ∨ (stop == p.numLayers) = true
    · exact Or.inr (h3 hbd)
    · left
      constructor
      · cases h : start == 0 <;> simp_all
      · cases h : stop == p.numLayers <;> simp_all

theorem buildStage_withinBudget {p : Profile} {m : Machine} {start stop : Nat}
    (hv : stageValid p m start stop = true) :
    WithinBudget p (buildStage p m start stop) := by
  have ⟨_, _, _, hcap, htot⟩ := (stageValid_iff p m start stop).mp hv
  exact ⟨rfl, rfl, rfl, hcap, htot⟩

theorem buildStage_boundaryTrusted {p : Profile} {m : Machine} {start stop : Nat}
    (hv : stageValid p m start stop = true) :
    BoundaryTrusted p (buildStage p m start stop) := by
  have ⟨_, _, htr, _, _⟩ := (stageValid_iff p m start stop).mp hv
  exact ⟨rfl, rfl, htr⟩

/-! ## Consequences of `ValidChain` -/

theorem validChain_start_le {p : Profile} {ms : List Machine} {start : Nat} {stages : List Stage}
    (h : ValidChain p ms start stages) : start ≤ p.numLayers := by
  induction h with
  | nil => exact Nat.le_refl _
  | skip hlt _ => exact Nat.le_of_lt hlt
  | take hv _ =>
    have ⟨h1, h2, _, _, _⟩ := (stageValid_iff p _ _ _).mp hv
    omega

theorem stages_contiguous_cover {p : Profile} {ms : List Machine} {start : Nat}
    {stages : List Stage} (h : ValidChain p ms start stages) :
    ContiguousCover start p.numLayers stages ∧ (start < p.numLayers → stages ≠ []) := by
  induction h with
  | nil =>
    refine ⟨rfl, ?_⟩
    intro hlt
    omega
  | skip _ _ ih => exact ih
  | take hv _ ih =>
    have ⟨h1, h2, _, _, _⟩ := (stageValid_iff p _ _ _).mp hv
    exact ⟨⟨rfl, h1, h2, ih.1⟩, fun _ => List.cons_ne_nil _ _⟩

theorem stages_within_budget {p : Profile} {ms : List Machine} {start : Nat}
    {stages : List Stage} (h : ValidChain p ms start stages) :
    ∀ st ∈ stages, WithinBudget p st := by
  induction h with
  | nil => intro _ hm; cases hm
  | skip _ _ ih => exact ih
  | take hv _ ih =>
    intro st hm
    cases List.mem_cons.mp hm with
    | inl heq =>
      subst heq
      exact buildStage_withinBudget hv
    | inr hmem => exact ih st hmem

theorem stages_boundary_trusted {p : Profile} {ms : List Machine} {start : Nat}
    {stages : List Stage} (h : ValidChain p ms start stages) :
    ∀ st ∈ stages, BoundaryTrusted p st := by
  induction h with
  | nil => intro _ hm; cases hm
  | skip _ _ ih => exact ih
  | take hv _ ih =>
    intro st hm
    cases List.mem_cons.mp hm with
    | inl heq =>
      subst heq
      exact buildStage_boundaryTrusted hv
    | inr hmem => exact ih st hmem

/-- A single-machine whole-model slice shares a tied embedding matrix (`lmHeadBytes` is not added
    a second time), whereas splitting across two stages counts `embeddingBytes` on stage 0 and
    `lmHeadBytes` on the final stage. -/
theorem single_stage_shares_tied_embedding (p : Profile) (cut : Nat)
    (htie : p.tieWordEmbeddings = true) :
    sliceWeightBytes p 0 p.numLayers true true =
      p.embeddingBytes + rangeWeightSum p.layerWeightBytes 0 p.numLayers + p.finalNormBytes ∧
    sliceWeightBytes p 0 cut true false + sliceWeightBytes p cut p.numLayers false true =
      (p.embeddingBytes + rangeWeightSum p.layerWeightBytes 0 cut) +
        (rangeWeightSum p.layerWeightBytes cut p.numLayers + (p.finalNormBytes + p.lmHeadBytes)) := by
  simp [sliceWeightBytes, htie]

/-- Total bytes across any valid chain never exceed the total capacity of the candidate machines. -/
theorem infeasible_when_total_exceeds_cluster_capacity {p : Profile} {ms : List Machine}
    {start : Nat} {stages : List Stage} (h : ValidChain p ms start stages) :
    (stages.map Stage.totalBytes).sum ≤ (ms.map Machine.capacityBytes).sum := by
  induction h with
  | nil => simp
  | skip _ _ ih =>
    simp only [List.map_cons, List.sum_cons]
    omega
  | take hv _ ih =>
    have ⟨_, _, _, _, htot⟩ := (stageValid_iff p _ _ _).mp hv
    simp only [List.map_cons, List.sum_cons]
    have hst : (buildStage p _ _ _).totalBytes ≤ _ := htot
    omega

/-! ## Soundness of `planFrom` -/

theorem pickBetter_prop (P : List Stage → Prop) (best : Option (List Stage)) (cand : List Stage)
    (hbest : ∀ s, best = some s → P s) (hcand : P cand) :
    ∀ s, pickBetter best cand = some s → P s := by
  intro s hs
  unfold pickBetter at hs
  cases best with
  | none =>
    cases hs
    exact hcand
  | some b =>
    by_cases h : loadLe (planLoad b) (planLoad cand) = true
    · simp only [h, ↓reduceIte] at hs
      cases hs
      exact hbest s rfl
    · simp only [h] at hs
      cases hs
      exact hcand

theorem tryStopStep_prop (P : List Stage → Prop) (p : Profile) (m : Machine)
    (next : Nat → Option (List Stage)) (start stop : Nat) (best : Option (List Stage))
    (hbest : ∀ s, best = some s → P s)
    (hstep : ∀ tail, stageValid p m start stop = true → next stop = some tail →
      P (buildStage p m start stop :: tail)) :
    ∀ s, tryStopStep p m next start stop best = some s → P s := by
  intro s hs
  unfold tryStopStep at hs
  split at hs
  next hv =>
    cases hnext : next stop with
    | none =>
      rw [hnext] at hs
      exact hbest s hs
    | some tail =>
      rw [hnext] at hs
      exact pickBetter_prop P best _ hbest (hstep tail hv hnext) s hs
  next => exact hbest s hs

theorem tryStops_prop (P : List Stage → Prop) (p : Profile) (m : Machine)
    (next : Nat → Option (List Stage)) (start : Nat) (stops : List Nat)
    (best : Option (List Stage))
    (hbest : ∀ s, best = some s → P s)
    (hstep : ∀ stop ∈ stops, ∀ tail,
      stageValid p m start stop = true → next stop = some tail →
      P (buildStage p m start stop :: tail)) :
    ∀ s, tryStops p m next start stops best = some s → P s := by
  induction stops generalizing best with
  | nil => exact hbest
  | cons stop rest ih =>
    unfold tryStops
    apply ih
    · exact tryStopStep_prop P p m next start stop best hbest
        (hstep stop List.mem_cons_self)
    · intro st hmem
      exact hstep st (List.mem_cons_of_mem _ hmem)

/-- Every plan returned by `planFrom` is a valid contiguous chain. -/
theorem planFrom_valid (p : Profile) (ms : List Machine) :
    ∀ start stages, planFrom p ms start = some stages → ValidChain p ms start stages := by
  induction ms with
  | nil =>
    intro start stages h
    unfold planFrom at h
    split at h
    next heq =>
      cases h
      have hs : start = p.numLayers := LawfulBEq.eq_of_beq heq
      subst hs
      exact ValidChain.nil []
    next => contradiction
  | cons m rest ih =>
    intro start stages h
    unfold planFrom at h
    split at h
    next heq =>
      cases h
      have hs : start = p.numLayers := LawfulBEq.eq_of_beq heq
      subst hs
      exact ValidChain.nil (m :: rest)
    next hne =>
      split at h
      next => contradiction
      next hnotgt =>
        have hlt : start < p.numLayers := by
          have : start ≠ p.numLayers := fun he => hne (beq_iff_eq.mpr he)
          omega
        have hwithM : ∀ s,
            tryStops p m (planFrom p rest) start
              ((List.range (p.numLayers - start)).map (fun k => start + 1 + k)) none = some s →
            ValidChain p (m :: rest) start s := by
          apply tryStops_prop (ValidChain p (m :: rest) start)
          · intro _ hnone; contradiction
          · intro stop _ tail hv hnext
            exact ValidChain.take hv (ih stop tail hnext)
        cases hskip : planFrom p rest start with
        | none =>
          rw [hskip] at h
          exact hwithM stages h
        | some skipPlan =>
          rw [hskip] at h
          exact pickBetter_prop (ValidChain p (m :: rest) start) _ skipPlan hwithM
            (ValidChain.skip hlt (ih start skipPlan hskip)) stages h

/-! ## Completeness and optimality of `planFrom` -/

theorem loadLe_refl (a : Nat × Nat) : loadLe a a = true := by
  simp [loadLe]

theorem loadLe_total (a b : Nat × Nat) : loadLe a b = true ∨ loadLe b a = true := by
  unfold loadLe
  simp only [decide_eq_true_eq]
  omega

theorem loadLe_trans {a b c : Nat × Nat} (hb : 0 < b.2)
    (hab : loadLe a b = true) (hbc : loadLe b c = true) : loadLe a c = true := by
  unfold loadLe at *
  simp only [decide_eq_true_eq] at *
  have h1 : a.1 * b.2 * c.2 ≤ b.1 * a.2 * c.2 := Nat.mul_le_mul_right c.2 hab
  have h2 : b.1 * c.2 * a.2 ≤ c.1 * b.2 * a.2 := Nat.mul_le_mul_right a.2 hbc
  have h3 : b.1 * a.2 * c.2 = b.1 * c.2 * a.2 := by
    rw [Nat.mul_assoc, Nat.mul_comm a.2 c.2, ← Nat.mul_assoc]
  have h4 : c.1 * b.2 * a.2 = c.1 * a.2 * b.2 := by
    rw [Nat.mul_assoc, Nat.mul_comm b.2 a.2, ← Nat.mul_assoc]
  have h5 : a.1 * c.2 * b.2 = a.1 * b.2 * c.2 := by
    rw [Nat.mul_assoc, Nat.mul_comm c.2 b.2, ← Nat.mul_assoc]
  have h6 : a.1 * c.2 * b.2 ≤ c.1 * a.2 * b.2 := by omega
  exact Nat.le_of_mul_le_mul_right h6 hb

theorem maxLoad_pos {a b : Nat × Nat} (ha : 0 < a.2) (hb : 0 < b.2) :
    0 < (maxLoad a b).2 := by
  unfold maxLoad
  split <;> assumption

theorem planLoad_pos {p : Profile} {ms : List Machine} {start : Nat} {stages : List Stage}
    (h : ValidChain p ms start stages) : 0 < (planLoad stages).2 := by
  induction h with
  | nil => decide
  | skip _ _ ih => exact ih
  | take hv _ ih =>
    have ⟨_, _, _, hcap, _⟩ := (stageValid_iff p _ _ _).mp hv
    exact maxLoad_pos hcap ih

theorem loadLe_maxLoad_left (a b : Nat × Nat) : loadLe a (maxLoad a b) = true := by
  unfold maxLoad
  split
  · exact loadLe_refl a
  · next hnot =>
    cases loadLe_total a b with
    | inl hab => exact hab
    | inr hba => exact absurd hba hnot

theorem loadLe_maxLoad_right (a b : Nat × Nat) : loadLe b (maxLoad a b) = true := by
  unfold maxLoad
  split
  · next hba => exact hba
  · exact loadLe_refl b

theorem maxLoad_le {a b c : Nat × Nat} (hac : loadLe a c = true) (hbc : loadLe b c = true) :
    loadLe (maxLoad a b) c = true := by
  unfold maxLoad
  split <;> assumption

theorem maxLoad_mono_right (x : Nat × Nat) {a b : Nat × Nat} (hb : 0 < b.2)
    (hab : loadLe a b = true) : loadLe (maxLoad x a) (maxLoad x b) = true := by
  apply maxLoad_le
  · exact loadLe_maxLoad_left x b
  · exact loadLe_trans hb hab (loadLe_maxLoad_right x b)

theorem pickBetter_spec (best : Option (List Stage)) (cand : List Stage) :
    ∃ out, pickBetter best cand = some out ∧
      loadLe (planLoad out) (planLoad cand) = true ∧
      (∀ b, best = some b → loadLe (planLoad out) (planLoad b) = true) := by
  unfold pickBetter
  cases best with
  | none =>
    refine ⟨cand, rfl, loadLe_refl _, ?_⟩
    intro _ h; contradiction
  | some b =>
    by_cases hbc : loadLe (planLoad b) (planLoad cand) = true
    · simp only [hbc, ↓reduceIte]
      refine ⟨b, rfl, hbc, ?_⟩
      intro b' hb'
      cases hb'
      exact loadLe_refl _
    · simp only [hbc]
      have hcb : loadLe (planLoad cand) (planLoad b) = true := by
        cases loadLe_total (planLoad b) (planLoad cand) with
        | inl h => exact absurd h hbc
        | inr h => exact h
      refine ⟨cand, rfl, loadLe_refl _, ?_⟩
      intro b' hb'
      cases hb'
      exact hcb

theorem tryStopStep_pos (p : Profile) (m : Machine) (next : Nat → Option (List Stage))
    (start stop : Nat) (best : Option (List Stage))
    (hbest_pos : ∀ b, best = some b → 0 < (planLoad b).2)
    (hnext_pos : ∀ stp tail, next stp = some tail → 0 < (planLoad tail).2) :
    ∀ s, tryStopStep p m next start stop best = some s → 0 < (planLoad s).2 := by
  apply tryStopStep_prop (fun s => 0 < (planLoad s).2) p m next start stop best hbest_pos
  intro tail hv hnext
  have ⟨_, _, _, hcap, _⟩ := (stageValid_iff p m start stop).mp hv
  exact maxLoad_pos hcap (hnext_pos stop tail hnext)

theorem tryStopStep_preserve (p : Profile) (m : Machine) (next : Nat → Option (List Stage))
    (start stop : Nat) (best : Option (List Stage))
    (hbest_pos : ∀ b, best = some b → 0 < (planLoad b).2)
    (hnext_pos : ∀ stp tail, next stp = some tail → 0 < (planLoad tail).2)
    {b : List Stage} (hb : best = some b) :
    ∃ mid, tryStopStep p m next start stop best = some mid ∧
      0 < (planLoad mid).2 ∧ loadLe (planLoad mid) (planLoad b) = true := by
  have hpos := tryStopStep_pos p m next start stop best hbest_pos hnext_pos
  by_cases hv : stageValid p m start stop = true
  · cases hnext : next stop with
    | none =>
      have hstep : tryStopStep p m next start stop best = some b := by
        unfold tryStopStep
        simp [hv, hnext, hb]
      exact ⟨b, hstep, hbest_pos b hb, loadLe_refl _⟩
    | some tail =>
      obtain ⟨mid, hmid, _, hmb⟩ := pickBetter_spec best (buildStage p m start stop :: tail)
      have hstep : tryStopStep p m next start stop best = some mid := by
        unfold tryStopStep
        simp [hv, hnext, hmid]
      exact ⟨mid, hstep, hpos mid hstep, hmb b hb⟩
  · have hstep : tryStopStep p m next start stop best = some b := by
      unfold tryStopStep
      simp [hv, hb]
    exact ⟨b, hstep, hbest_pos b hb, loadLe_refl _⟩

theorem tryStops_mono (p : Profile) (m : Machine) (next : Nat → Option (List Stage))
    (start : Nat) (stops : List Nat) (best : Option (List Stage))
    (hbest_pos : ∀ b, best = some b → 0 < (planLoad b).2)
    (hnext_pos : ∀ stop tail, next stop = some tail → 0 < (planLoad tail).2) :
    ∀ b, best = some b →
      ∃ out, tryStops p m next start stops best = some out ∧
        0 < (planLoad out).2 ∧ loadLe (planLoad out) (planLoad b) = true := by
  induction stops generalizing best with
  | nil =>
    intro b hb
    exact ⟨b, hb, hbest_pos b hb, loadLe_refl _⟩
  | cons stop rest ih =>
    intro b hb
    unfold tryStops
    obtain ⟨mid, hmid, hmid_pos, hmb⟩ :=
      tryStopStep_preserve p m next start stop best hbest_pos hnext_pos hb
    have hstep_pos := tryStopStep_pos p m next start stop best hbest_pos hnext_pos
    obtain ⟨out, hout, hout_pos, hom⟩ := ih (tryStopStep p m next start stop best) hstep_pos mid hmid
    exact ⟨out, hout, hout_pos, loadLe_trans hmid_pos hom hmb⟩

theorem tryStops_hits (p : Profile) (m : Machine) (next : Nat → Option (List Stage))
    (start : Nat) (stops : List Nat) (best : Option (List Stage))
    (hbest_pos : ∀ b, best = some b → 0 < (planLoad b).2)
    (hnext_pos : ∀ stop tail, next stop = some tail → 0 < (planLoad tail).2)
    {stop : Nat} {tail : List Stage}
    (hmem : stop ∈ stops) (hv : stageValid p m start stop = true)
    (hnext : next stop = some tail) :
    ∃ out, tryStops p m next start stops best = some out ∧
      0 < (planLoad out).2 ∧
      loadLe (planLoad out) (planLoad (buildStage p m start stop :: tail)) = true := by
  induction stops generalizing best with
  | nil => cases hmem
  | cons hd rest ih =>
    unfold tryStops
    have hstep_pos := tryStopStep_pos p m next start hd best hbest_pos hnext_pos
    cases List.mem_cons.mp hmem with
    | inl heq =>
      subst heq
      obtain ⟨mid, hmid, hmc, _⟩ := pickBetter_spec best (buildStage p m start stop :: tail)
      have hstep_eq : tryStopStep p m next start stop best = some mid := by
        unfold tryStopStep
        simp [hv, hnext, hmid]
      have hmid_pos : 0 < (planLoad mid).2 := hstep_pos mid hstep_eq
      obtain ⟨out, hout, hout_pos, hom⟩ :=
        tryStops_mono p m next start rest (tryStopStep p m next start stop best)
          hstep_pos hnext_pos mid hstep_eq
      exact ⟨out, hout, hout_pos, loadLe_trans hmid_pos hom hmc⟩
    | inr hrest =>
      exact ih (tryStopStep p m next start hd best) hstep_pos hrest

theorem mem_stops {numLayers start stop : Nat} (h1 : start < stop) (h2 : stop ≤ numLayers) :
    stop ∈ (List.range (numLayers - start)).map (fun k => start + 1 + k) := by
  rw [List.mem_map]
  refine ⟨stop - (start + 1), ?_, by omega⟩
  rw [List.mem_range]
  omega

/-- Combined completeness and min-max optimality of `planFrom`: for every valid chain `candidate`,
    `planFrom` returns `some stages` with `loadLe (planLoad stages) (planLoad candidate) = true`. -/
theorem planFrom_complete_and_optimal {p : Profile} {ms : List Machine} {start : Nat}
    {candidate : List Stage} (h : ValidChain p ms start candidate) :
    ∃ stages, planFrom p ms start = some stages ∧
      loadLe (planLoad stages) (planLoad candidate) = true := by
  induction h with
  | nil ms =>
    cases ms with
    | nil =>
      refine ⟨[], ?_, loadLe_refl _⟩
      simp [planFrom]
    | cons m rest =>
      refine ⟨[], ?_, loadLe_refl _⟩
      simp [planFrom]
  | @skip m rest start candidate hlt hvalid ih =>
    obtain ⟨skipPlan, hskip, hopt⟩ := ih
    have hskip_pos : 0 < (planLoad skipPlan).2 :=
      planLoad_pos (planFrom_valid p rest start skipPlan hskip)
    unfold planFrom
    have hne : (start == p.numLayers) = false := beq_eq_false_iff_ne.mpr (by omega)
    have hnotgt : ¬ p.numLayers < start := by omega
    rw [hne]
    simp only [Bool.false_eq_true, ↓reduceIte, hnotgt, hskip]
    obtain ⟨out, hout, h_out_skip, _⟩ :=
      pickBetter_spec
        (tryStops p m (planFrom p rest) start
          ((List.range (p.numLayers - start)).map (fun k => start + 1 + k)) none)
        skipPlan
    exact ⟨out, hout, loadLe_trans hskip_pos h_out_skip hopt⟩
  | @take m rest start stop tail hv htail ih =>
    obtain ⟨optTail, htail_eq, htail_opt⟩ := ih
    have ⟨h1, h2, _, hcap, _⟩ := (stageValid_iff p m start stop).mp hv
    have hnext_pos : ∀ stp tl, planFrom p rest stp = some tl → 0 < (planLoad tl).2 :=
      fun stp tl heq => planLoad_pos (planFrom_valid p rest stp tl heq)
    obtain ⟨withMOut, hwithM, hwithM_pos, hwithM_le⟩ :=
      tryStops_hits p m (planFrom p rest) start
        ((List.range (p.numLayers - start)).map (fun k => start + 1 + k)) none
        (fun _ hnone => by contradiction) hnext_pos (mem_stops h1 h2) hv htail_eq
    have htail_pos : 0 < (planLoad tail).2 := planLoad_pos htail
    have hstep_le :
        loadLe (planLoad (buildStage p m start stop :: optTail))
          (planLoad (buildStage p m start stop :: tail)) = true :=
      maxLoad_mono_right ( (buildStage p m start stop).totalBytes,
        (buildStage p m start stop).capacityBytes ) htail_pos htail_opt
    have hopt_pos : 0 < (planLoad (buildStage p m start stop :: optTail)).2 :=
      maxLoad_pos hcap (hnext_pos stop optTail htail_eq)
    have hwithM_cand :
        loadLe (planLoad withMOut) (planLoad (buildStage p m start stop :: tail)) = true :=
      loadLe_trans hopt_pos hwithM_le hstep_le
    unfold planFrom
    have hne : (start == p.numLayers) = false := beq_eq_false_iff_ne.mpr (by omega)
    have hnotgt : ¬ p.numLayers < start := by omega
    rw [hne]
    simp only [Bool.false_eq_true, ↓reduceIte, hnotgt, hwithM]
    cases hskip : planFrom p rest start with
    | none => exact ⟨withMOut, rfl, hwithM_cand⟩
    | some skipPlan =>
      obtain ⟨out, hout, _, h_out_withM⟩ := pickBetter_spec (some withMOut) skipPlan
      exact ⟨out, hout, loadLe_trans hwithM_pos (h_out_withM withMOut rfl) hwithM_cand⟩

/-- Completeness: if any valid full-context placement exists on `ms`, `planFrom` succeeds. -/
theorem planFrom_complete {p : Profile} {ms : List Machine} {start : Nat}
    {candidate : List Stage} (h : ValidChain p ms start candidate) :
    (planFrom p ms start).isSome = true := by
  obtain ⟨stages, hstages, _⟩ := planFrom_complete_and_optimal h
  simp [hstages]

/-- `planFrom` returns `none` if and only if no valid placement exists on `ms`. -/
theorem planFrom_none_iff_infeasible (p : Profile) (ms : List Machine) (start : Nat) :
    planFrom p ms start = none ↔ ∀ candidate, ¬ ValidChain p ms start candidate := by
  constructor
  · intro hnone candidate hvalid
    obtain ⟨stages, hstages, _⟩ := planFrom_complete_and_optimal hvalid
    rw [hnone] at hstages
    contradiction
  · intro hall
    cases h : planFrom p ms start with
    | none => rfl
    | some stages =>
      exact absurd (planFrom_valid p ms start stages h) (hall stages)

/-- Optimality: the placement chosen by `planFrom` has bottleneck memory load no greater than any
    valid alternative `candidate` on `ms`. -/
theorem planFrom_optimal {p : Profile} {ms : List Machine} {start : Nat}
    {stages candidate : List Stage}
    (hplan : planFrom p ms start = some stages)
    (hvalid : ValidChain p ms start candidate) :
    loadLe (planLoad stages) (planLoad candidate) = true := by
  obtain ⟨stages', hstages', hopt⟩ := planFrom_complete_and_optimal hvalid
  rw [hplan] at hstages'
  cases hstages'
  exact hopt

end Kadmium.Sharding.LayerPlacement
