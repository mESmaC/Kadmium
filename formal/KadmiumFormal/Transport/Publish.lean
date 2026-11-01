/-!
# Publishing a received package

A receiving coordinator stages a transfer in `.receiving/<id>`, then `receive_finish` verifies it
and renames it into the library (`crates/kadmium-data/src/checkpoint/transport.rs`). A concurrent
`receive_begin` for the same checkpoint with a different transport manifest wipes and recreates
staging.

The original `receive_finish` verified staging in place and then renamed it. `legacy_unsafe`
checks the counterexample from the review of PR #62: a `begin` between the verification and the
rename publishes a directory without `manifest.json`.

`receive_finish` now first renames staging into a private claim, `.verifying/<claim>/<id>`
(`finish_claimed`), verifies the claim, and publishes the claim itself. `publish_safe` proves that,
for every interleaving of any number of begins, chunk arrivals and concurrent finishes, every
published package is one that verified as complete.

Modelling choices:

* A directory is abstracted to one flag: `true` when it holds every file of its own transport
  manifest, intact. `receive_begin` with another manifest leaves staging incomplete (`some false`);
  the last chunk arriving makes it complete.
* Renames are atomic, and a rename of a missing source fails without effect. This is filesystem
  behaviour, assumed, not proved.
* `begin` and chunk writes address only `.receiving/<id>`. Each finish addresses only its own
  claim, so a claim is changed only by events naming its index.
* Verification succeeds exactly when the claimed directory is complete and intact
  (`CheckpointLibrary::verify` checks identity, every byte, metadata and bank layout).
* A failed claim may go back to staging (missing chunks, for resume) or be discarded (failed
  verification); the model allows either.

The link to the code is **model only**. The unit test
`checkpoint::transport::tests::begin_during_finish_cannot_publish_a_package_without_its_manifest`
runs the legacy counterexample's interleaving against the Rust code by injecting the `begin` after
the claim.
-/

namespace Kadmium.Transport.Publish

/-- `(complete, verified)` for one claimed package. -/
abbrev Claim := Bool × Bool

structure St where
  staging : Option Bool
  claims : List Claim
  published : Option Bool

inductive Ev
  /-- `receive_begin` with another manifest: staging is wiped and recreated. -/
  | beginOther
  /-- The last missing chunk lands in staging. -/
  | chunksDone
  /-- `finish_claimed` renames staging into a new private claim. -/
  | take
  /-- Finish `i` verifies its claim. -/
  | verify (i : Nat)
  /-- Finish `i` renames its verified claim into the library. -/
  | publish (i : Nat)
  /-- Finish `i` gives up its unverified claim, putting it back if staging is free. -/
  | release (i : Nat) (putBack : Bool)

def mapAt (f : Claim → Claim) : Nat → List Claim → List Claim
  | _, [] => []
  | 0, x :: xs => f x :: xs
  | n + 1, x :: xs => x :: mapAt f n xs

def eraseAt : Nat → List Claim → List Claim
  | _, [] => []
  | 0, _ :: xs => xs
  | n + 1, x :: xs => x :: eraseAt n xs

def step (s : St) : Ev → St
  | .beginOther => { s with staging := some false }
  | .chunksDone => { s with staging := s.staging.map fun _ => true }
  | .take =>
    match s.staging with
    | some b => { s with staging := none, claims := s.claims ++ [(b, false)] }
    | none => s
  | .verify i => { s with claims := mapAt (fun c => (c.1, c.1)) i s.claims }
  | .publish i =>
    match s.claims[i]? with
    | some (b, true) => { s with published := some b, claims := eraseAt i s.claims }
    | _ => s
  | .release i putBack =>
    match s.claims[i]? with
    | some (b, false) =>
      { s with
        claims := eraseAt i s.claims
        staging := if putBack && s.staging.isNone then some b else s.staging }
    | _ => s

def run (s : St) (es : List Ev) : St := es.foldl step s

def init : St := ⟨none, [], none⟩

/-- Nothing incomplete is ever published. -/
def Safe (s : St) : Prop := ∀ m, s.published = some m → m = true

/-- A claim marked verified is complete. -/
def Good (c : Claim) : Prop := c.2 = true → c.1 = true

def Inv (s : St) : Prop := Safe s ∧ ∀ c ∈ s.claims, Good c

theorem mem_mapAt {f : Claim → Claim} (hf : ∀ c, Good (f c)) :
    ∀ (n : Nat) (l : List Claim), (∀ c ∈ l, Good c) → ∀ c ∈ mapAt f n l, Good c
  | _, [], _ => by simp [mapAt]
  | 0, x :: xs, h => by
    intro c hc
    simp only [mapAt, List.mem_cons] at hc
    rcases hc with rfl | hc
    · exact hf x
    · exact h c (List.mem_cons_of_mem _ hc)
  | n + 1, x :: xs, h => by
    intro c hc
    simp only [mapAt, List.mem_cons] at hc
    rcases hc with rfl | hc
    · exact h c List.mem_cons_self
    · exact mem_mapAt hf n xs (fun c hc => h c (List.mem_cons_of_mem _ hc)) c hc

theorem mem_eraseAt : ∀ (n : Nat) (l : List Claim) (c : Claim), c ∈ eraseAt n l → c ∈ l
  | _, [], _, h => by simp [eraseAt] at h
  | 0, _ :: xs, c, h => List.mem_cons_of_mem _ h
  | n + 1, x :: xs, c, h => by
    simp only [eraseAt, List.mem_cons] at h
    rcases h with rfl | h
    · exact List.mem_cons_self
    · exact List.mem_cons_of_mem _ (mem_eraseAt n xs c h)

theorem step_inv (s : St) (e : Ev) (h : Inv s) : Inv (step s e) := by
  obtain ⟨hsafe, hgood⟩ := h
  cases e with
  | beginOther => exact ⟨hsafe, hgood⟩
  | chunksDone => exact ⟨hsafe, hgood⟩
  | take =>
    rcases hst : s.staging with _ | b
    · simp only [step, hst]; exact ⟨hsafe, hgood⟩
    · simp only [step, hst]
      refine ⟨hsafe, ?_⟩
      intro c hc
      rcases List.mem_append.mp hc with hc | hc
      · exact hgood c hc
      · rw [List.mem_singleton] at hc
        subst hc
        intro hv; cases hv
  | verify i =>
    refine ⟨hsafe, ?_⟩
    exact mem_mapAt (fun _ hv => hv) i s.claims hgood
  | publish i =>
    rcases hget : s.claims[i]? with _ | ⟨b, v⟩
    · simp only [step, hget]; exact ⟨hsafe, hgood⟩
    · cases v
      · simp only [step, hget]; exact ⟨hsafe, hgood⟩
      · simp only [step, hget]
        refine ⟨?_, fun c hc => hgood c (mem_eraseAt i s.claims c hc)⟩
        intro m hm
        simp only [Option.some.injEq] at hm
        subst hm
        exact hgood (b, true) (List.mem_of_getElem? hget) rfl
  | release i putBack =>
    rcases hget : s.claims[i]? with _ | ⟨b, v⟩
    · simp only [step, hget]; exact ⟨hsafe, hgood⟩
    · cases v
      · simp only [step, hget]
        exact ⟨hsafe, fun c hc => hgood c (mem_eraseAt i s.claims c hc)⟩
      · simp only [step, hget]; exact ⟨hsafe, hgood⟩

theorem run_inv (es : List Ev) : ∀ s, Inv s → Inv (run s es) := by
  induction es with
  | nil => intro s h; exact h
  | cons e es ih => intro s h; exact ih (step s e) (step_inv s e h)

/-- Every package `receive_finish` publishes verified as complete, under every interleaving of
begins, chunk arrivals and concurrent finishes. -/
theorem publish_safe (es : List Ev) : Safe (run init es) :=
  (run_inv es init ⟨fun _ h => by simp [init] at h, fun _ h => by simp [init] at h⟩).1

/-! ## The legacy protocol -/

/-- The original `receive_finish`: verify staging in place, then rename it. -/
structure LegacySt where
  staging : Option Bool
  verified : Bool
  published : Option Bool

inductive LegacyEv | beginOther | chunksDone | finishVerify | finishRename

def legacyStep (s : LegacySt) : LegacyEv → LegacySt
  | .beginOther => { s with staging := some false }
  | .chunksDone => { s with staging := s.staging.map fun _ => true }
  | .finishVerify => { s with verified := s.staging == some true }
  | .finishRename =>
    if s.verified then { s with published := s.staging, staging := none } else s

/-- The counterexample from the review of PR #62: the legacy protocol publishes an incomplete
package when a `begin` lands between verification and rename. -/
theorem legacy_unsafe :
    (([.beginOther, .chunksDone, .finishVerify, .beginOther, .finishRename] : List LegacyEv).foldl
      legacyStep ⟨none, false, none⟩).published = some false := by
  decide

end Kadmium.Transport.Publish
