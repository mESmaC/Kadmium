/-!
# Splitting a decoder's layers across slices

`DecoderSlice::forward` (`crates/kadmium-runtime/src/native_decoder/model.rs`) runs a slice's
layers in order. Each layer reads the activations and its own KV cache and returns new
activations and its updated cache. A machine that holds layers `[a, b)` runs one slice and passes
its activations to the machine that holds `[b, c)`.

`runLayers_append` proves that running a list of layers equals running a prefix and then the
rest, with the caches split at the same point. `runChain_eq_runLayers` extends this to any chain
of slices: running the slices one after another gives the same activations and the same caches as
running all their layers as one slice. So where the layers are cut, and how many machines hold
them, cannot change any output.

Modelling choices:

* A layer is any function `α → κ → α × κ`. Nothing about floating-point arithmetic is assumed,
  so the statements hold bit for bit for `f32`: two executions that apply the same functions to
  the same values in the same order produce the same values.
* Each layer touches only its own cache. In the Rust code block `index` of a slice reads and
  writes only `cache.keys[index]` and `cache.values[index]`.
* A slice's cache is the list of its layers' caches. The position counter `KvCache::positions`
  and the checks that reject a mismatched cache are not modelled; a misaligned cache list makes
  `runLayers` stop, and the theorems assume aligned lists.
* Activations crossing between slices are passed unchanged. The Rust test rebuilds them from a
  copy of their `f32` values.

The link to the code is **model only**. The Rust test
`splitting_the_layers_at_every_cut_is_bit_identical` (`crates/kadmium-runtime/tests/native_decoder.rs`)
checks the property on four real checkpoint layouts, every pair of cuts, through prefill and
token-by-token decode.
-/

namespace Kadmium.Native

variable {α κ : Type}

/-- Runs layers in order, each with its own cache. -/
def runLayers : List (α → κ → α × κ) → α → List κ → α × List κ
  | f :: fs, x, c :: cs =>
    let (y, c') := f x c
    let (z, cs') := runLayers fs y cs
    (z, c' :: cs')
  | _, x, cs => (x, cs)

/-- Runs slices in order, each with its own list of layer caches. -/
def runChain : List (List (α → κ → α × κ)) → α → List (List κ) → α × List (List κ)
  | s :: ss, x, c :: cs =>
    let (y, c') := runLayers s x c
    let (z, cs') := runChain ss y cs
    (z, c' :: cs')
  | _, x, cs => (x, cs)

/-- Cutting the layers at any point, with the caches cut at the same point, changes nothing. -/
theorem runLayers_append (fs gs : List (α → κ → α × κ)) (x : α) (cs ds : List κ)
    (h : cs.length = fs.length) :
    runLayers (fs ++ gs) x (cs ++ ds) =
      ((runLayers gs (runLayers fs x cs).1 ds).1,
        (runLayers fs x cs).2 ++ (runLayers gs (runLayers fs x cs).1 ds).2) := by
  induction fs generalizing x cs with
  | nil =>
    cases cs with
    | nil => simp [runLayers]
    | cons _ _ => simp at h
  | cons f fs ih =>
    cases cs with
    | nil => simp at h
    | cons c cs =>
      simp only [List.length_cons, Nat.add_right_cancel_iff] at h
      simp only [List.cons_append, runLayers, ih _ _ h]

/-- Each slice comes with exactly one cache per layer. -/
def Aligned : List (List (α → κ → α × κ)) → List (List κ) → Prop
  | [], [] => True
  | s :: ss, c :: cs => c.length = s.length ∧ Aligned ss cs
  | _, _ => False

/-- A chain of slices computes exactly what one slice holding all their layers computes. -/
theorem runChain_eq_runLayers (slices : List (List (α → κ → α × κ))) (x : α)
    (caches : List (List κ)) (h : Aligned slices caches) :
    (runChain slices x caches).1 = (runLayers slices.flatten x caches.flatten).1 ∧
      (runChain slices x caches).2.flatten = (runLayers slices.flatten x caches.flatten).2 := by
  induction slices generalizing x caches with
  | nil =>
    cases caches with
    | nil => simp [runChain, runLayers]
    | cons _ _ => simp [Aligned] at h
  | cons s ss ih =>
    cases caches with
    | nil => simp [Aligned] at h
    | cons c cs =>
      obtain ⟨hsc, hrest⟩ := h
      obtain ⟨ih1, ih2⟩ := ih (runLayers s x c).1 cs hrest
      simp only [runChain, List.flatten_cons, runLayers_append s _ x c _ hsc]
      exact ⟨ih1, by rw [ih2]⟩

end Kadmium.Native
