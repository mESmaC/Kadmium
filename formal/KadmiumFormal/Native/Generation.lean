import KadmiumFormal.Native.LayerSplit

/-!
# Generating across a chain of machines

`native_generation::generate` (`crates/kadmium-runtime/src/native_generation.rs`) runs the
stages of a chain on the step's input, selects a token from the last stage's output, stops or
feeds the token back. Two properties are proved here.

**Where the layers are cut cannot change a generated token.** `generate_chain_eq_whole`: running
the loop over a chain of slices produces exactly the tokens that running it over one slice holding
all their layers does, for any selection function and any stopping rule, starting from caches
with one entry per layer. It follows from `runChain_eq_runLayers` and from the fact, proved here
as `runChain_aligned`, that a pass keeps the caches aligned with the slices, so the next step's
chain is again aligned.

Modelling choices: the token embedding and the LM head are folded into `embed` and `pick`, and the
sampler's state, such as its random generator and the tokens so far, is threaded through `pick`.
The loop is bounded by `n` steps. The Rust loop's stopping rules, EOS, a stop string, an operator
limit and the full context, are functions of the tokens produced so far and are covered by `stop`
reading that state, or by `n`. The link to the
code is **model only**: the Rust test `chain_generation_is_identical_at_every_cut`
(`crates/kadmium-runtime/tests/native_generation.rs`) checks the property on four real checkpoint
layouts, every pair of cuts, greedily and with sampling from one seed.

**Streamed text never has to be taken back and never reaches into a stop string.** Before
sending, the coordinator holds back the longest suffix of the generated text that is a proper,
nonempty prefix of a pattern (`heldBack`), and sends the first `emitLen` elements.
`emitLen_mono` and `emitted_prefix`: when the text grows, what is sent only grows, and what was
sent before is a prefix of what is sent after. `emitLen_le_stop`: if the text so far does not
contain a whole stop string and a continuation completes one at offset `p`, nothing at or after
`p` was sent. Modelling choices: the text is a list of bytes of any type with decidable equality,
and the text grows only by appending. The Rust code first removes a trailing U+FFFD, which a later
token may still turn into a character, and checks at run time that the decoded text keeps
extending what was sent; the tokenizer itself is not modelled. The link to the code is a
differential fixture: `lake exe kadmium_vectors` writes
`crates/kadmium-runtime/tests/fixtures/generation_lean_vectors.txt` and the Rust test
`held_back_len_matches_the_lean_model` requires `native_generation::held_back_len` to reproduce
`heldBack` on every line.
-/

namespace Kadmium.Native.Generation

open Kadmium.Native

section Chain

variable {α κ σ : Type}

/-- The coordinator loop: run the network on the input, select a token from the output, then stop
or feed the token back. `stop` reads the selection state after the token as well as the token,
so a state that carries the tokens so far expresses stop strings, which depend on the decoded
history, as well as EOS tokens and an operator limit. -/
def generate {κs : Type} (run : α → κs → α × κs) (pick : σ → α → σ × Nat)
    (stop : σ → Nat → Bool) (embed : Nat → α) : Nat → σ → α → κs → List Nat
  | 0, _, _, _ => []
  | n + 1, st, x, cs =>
    let (y, cs') := run x cs
    let (st', tok) := pick st y
    if stop st' tok then [tok] else tok :: generate run pick stop embed n st' (embed tok) cs'

/-- A pass keeps one cache per cache it was given. -/
theorem runLayers_length (fs : List (α → κ → α × κ)) (x : α) (cs : List κ) :
    (runLayers fs x cs).2.length = cs.length := by
  induction fs generalizing x cs with
  | nil => cases cs <;> simp [runLayers]
  | cons f fs ih =>
    cases cs with
    | nil => simp [runLayers]
    | cons c cs => simp [runLayers, ih]

/-- A pass through a chain leaves its caches aligned with its slices. -/
theorem runChain_aligned (slices : List (List (α → κ → α × κ))) (x : α)
    (caches : List (List κ)) (h : Aligned slices caches) :
    Aligned slices (runChain slices x caches).2 := by
  induction slices generalizing x caches with
  | nil =>
    cases caches with
    | nil => simp [runChain, Aligned]
    | cons _ _ => simp [Aligned] at h
  | cons s ss ih =>
    cases caches with
    | nil => simp [Aligned] at h
    | cons c cs =>
      obtain ⟨hsc, hrest⟩ := h
      simp only [runChain, Aligned]
      exact ⟨by rw [runLayers_length, hsc], ih _ cs hrest⟩

/-- Generating over a chain of slices gives exactly the tokens of generating over one slice that
holds all their layers. -/
theorem generate_chain_eq_whole (slices : List (List (α → κ → α × κ)))
    (pick : σ → α → σ × Nat) (stop : σ → Nat → Bool) (embed : Nat → α) (n : Nat) (st : σ) (x : α)
    (caches : List (List κ)) (h : Aligned slices caches) :
    generate (runChain slices) pick stop embed n st x caches =
      generate (runLayers slices.flatten) pick stop embed n st x caches.flatten := by
  induction n generalizing st x caches with
  | zero => rfl
  | succ n ih =>
    obtain ⟨h1, h2⟩ := runChain_eq_runLayers slices x caches h
    have aligned := runChain_aligned slices x caches h
    have whole : runLayers slices.flatten x caches.flatten =
        ((runChain slices x caches).1, (runChain slices x caches).2.flatten) :=
      Prod.ext h1.symm h2.symm
    simp only [generate, whole]
    rcases hrun : runChain slices x caches with ⟨y, cs'⟩
    rw [hrun] at aligned
    simp only
    rcases pick st y with ⟨st', tok⟩
    simp only
    split
    · rfl
    · rw [ih st' (embed tok) cs' aligned]

end Chain

section Holdback

variable {β : Type} [DecidableEq β]

/-- Some pattern's first `k` elements, `0 < k < length`, end `t`. -/
def Partial (stops : List (List β)) (t : List β) (k : Nat) : Prop :=
  ∃ s ∈ stops, 0 < k ∧ k < s.length ∧ s.take k <:+ t

instance (stops : List (List β)) (t : List β) (k : Nat) : Decidable (Partial stops t k) := by
  unfold Partial; infer_instance

/-- The largest `k ≤ n` with `Partial stops t k`, or `0`. -/
def search (stops : List (List β)) (t : List β) : Nat → Nat
  | 0 => 0
  | k + 1 => if Partial stops t (k + 1) then k + 1 else search stops t k

/-- The length of the longest suffix of `t` that is a proper, nonempty prefix of a pattern. -/
def heldBack (stops : List (List β)) (t : List β) : Nat := search stops t t.length

/-- The number of elements sent. -/
def emitLen (stops : List (List β)) (t : List β) : Nat := t.length - heldBack stops t

theorem le_search (stops : List (List β)) (t : List β) (n k : Nat) (hk : k ≤ n)
    (hp : Partial stops t k) : k ≤ search stops t n := by
  induction n with
  | zero => exact hk
  | succ n ih =>
    unfold search
    split
    · exact hk
    · rename_i hn
      have : k ≠ n + 1 := fun e => hn (e ▸ hp)
      exact ih (by omega)

theorem search_partial (stops : List (List β)) (t : List β) (n : Nat) :
    search stops t n = 0 ∨ Partial stops t (search stops t n) := by
  induction n with
  | zero => exact Or.inl rfl
  | succ n ih =>
    unfold search
    split
    · rename_i h; exact Or.inr h
    · exact ih

omit [DecidableEq β] in
theorem partial_le_length {stops : List (List β)} {t : List β} {k : Nat}
    (hp : Partial stops t k) : k ≤ t.length := by
  obtain ⟨s, _, _, hk, hsuf⟩ := hp
  have := hsuf.length_le
  rw [List.length_take] at this
  omega

theorem le_heldBack {stops : List (List β)} {t : List β} {k : Nat} (hp : Partial stops t k) :
    k ≤ heldBack stops t :=
  le_search stops t t.length k (partial_le_length hp) hp

theorem heldBack_partial (stops : List (List β)) (t : List β) :
    heldBack stops t = 0 ∨ Partial stops t (heldBack stops t) :=
  search_partial stops t t.length

omit [DecidableEq β] in
/-- A partial pattern at the end of `t ++ u` that reaches into `t` is a partial pattern at the
end of `t`. -/
theorem partial_of_append {stops : List (List β)} {t u : List β} {k : Nat}
    (hp : Partial stops (t ++ u) k) (hk : u.length < k) : Partial stops t (k - u.length) := by
  obtain ⟨s, hs, _, hks, w, hw⟩ := hp
  have hlen : (s.take k).length = k := by rw [List.length_take]; omega
  rcases List.append_eq_append_iff.mp hw with ⟨a, ht, hsa⟩ | ⟨c, hwc, hu⟩
  · -- `s.take k = a ++ u`, `t = w ++ a`.
    have ha : a.length = k - u.length := by
      have := congrArg List.length hsa
      rw [List.length_append, hlen] at this
      omega
    refine ⟨s, hs, by omega, by omega, ?_⟩
    have : s.take (k - u.length) = a := by
      have e := congrArg (List.take (k - u.length)) hsa
      rw [List.take_take, Nat.min_eq_left (by omega), List.take_append_of_le_length (by omega),
        ← ha, List.take_length] at e
      rw [ha] at e
      exact e
    rw [this, ht]
    exact List.suffix_append w a
  · -- `u = c ++ s.take k` makes `u` at least `k` long.
    have := congrArg List.length hu
    rw [List.length_append, hlen] at this
    omega

theorem emitLen_le_length (stops : List (List β)) (t : List β) : emitLen stops t ≤ t.length := by
  unfold emitLen; omega

/-- What is sent only grows as the text grows. -/
theorem emitLen_mono (stops : List (List β)) (t u : List β) :
    emitLen stops t ≤ emitLen stops (t ++ u) := by
  unfold emitLen
  rcases heldBack_partial stops (t ++ u) with h0 | hp
  · rw [h0, List.length_append]; omega
  · have hle := partial_le_length hp
    rw [List.length_append] at hle ⊢
    by_cases hk : (heldBack stops (t ++ u)) ≤ u.length
    · omega
    · have := le_heldBack (partial_of_append hp (by omega))
      omega

/-- What was sent before is a prefix of what is sent after. -/
theorem emitted_prefix (stops : List (List β)) (t u : List β) :
    t.take (emitLen stops t) <+: (t ++ u).take (emitLen stops (t ++ u)) := by
  have hmono := emitLen_mono stops t u
  have hlen := emitLen_le_length stops t
  have e : t.take (emitLen stops t) =
      ((t ++ u).take (emitLen stops (t ++ u))).take (emitLen stops t) := by
    rw [List.take_take, Nat.min_eq_left hmono, List.take_append_of_le_length hlen]
  rw [e]
  exact List.take_prefix _ _

/-- If `t` holds no whole stop string and its continuation completes one at offset `p`, nothing
from `p` on was sent. -/
theorem emitLen_le_stop (stops : List (List β)) (t u s : List β) (p : Nat) (hs : s ∈ stops)
    (hp : p ≤ t.length) (hocc : s <+: (t ++ u).drop p) (hcross : t.length < p + s.length) :
    emitLen stops t ≤ p := by
  by_cases heq : p = t.length
  · have := emitLen_le_length stops t; omega
  · have hk : Partial stops t (t.length - p) := by
      refine ⟨s, hs, by omega, by omega, ?_⟩
      rw [List.drop_append_of_le_length hp] at hocc
      obtain ⟨r, hr⟩ := hocc
      have hd : (t.drop p).length = t.length - p := List.length_drop
      have e := congrArg (List.take (t.length - p)) hr
      rw [List.take_append_of_le_length (by omega), List.take_append_of_le_length (by omega),
        ← hd, List.take_length] at e
      rw [hd] at e
      rw [e]
      exact List.drop_suffix p t
    have := le_heldBack hk
    unfold emitLen
    omega

end Holdback

end Kadmium.Native.Generation
