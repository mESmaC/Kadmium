/-!
Model of `model_text::ModelText::encode`'s context gate and `GenerationDefaults::is_eos`.
Tokenization and Jinja rendering are opaque inputs here, not proved implementations.
`TextVectors.lean` writes context/EOS cases consumed by the Rust integration test.
-/
namespace Kadmium.Native.Text

def accept (maximum : Nat) (tokens : List Nat) : Option (List Nat) :=
  if tokens.length ≤ maximum then some tokens else none

theorem accepted_unchanged (maximum : Nat) (tokens output : List Nat)
    (h : accept maximum tokens = some output) : output = tokens ∧ tokens.length ≤ maximum := by
  unfold accept at h
  split at h
  next bound => exact ⟨(Option.some.inj h).symm, bound⟩
  next => contradiction

theorem full_context_accepted (maximum : Nat) (tokens : List Nat)
    (h : tokens.length = maximum) : accept maximum tokens = some tokens := by
  simp [accept, h]

theorem over_context_rejected (maximum : Nat) (tokens : List Nat)
    (h : maximum < tokens.length) : accept maximum tokens = none := by
  simp [accept, Nat.not_le_of_gt h]

def isEos (stops : List Nat) (token : Nat) : Bool := stops.contains token

theorem eos_iff_member (stops : List Nat) (token : Nat) :
    isEos stops token = true ↔ token ∈ stops := by
  simp [isEos]

/-- `Some none` models an explicit JSON null; `none` models a missing key. -/
def resolveSpecial (declared : Option (Option Nat)) (fallback : Option Nat) : Option Nat :=
  match declared with
  | some value => value
  | none => fallback

theorem explicit_null_disables (fallback : Option Nat) :
    resolveSpecial (some none) fallback = none := rfl

theorem declared_special_wins (value fallback : Option Nat) :
    resolveSpecial (some value) fallback = value := rfl

end Kadmium.Native.Text

namespace Kadmium.Native.Text

/-- Per-attribute legacy overlay in model_text::specials::resolve. A missing map key is none;
    an explicit JSON null is some none. Token contents/ids are opaque Nat labels here. -/
def templateSpecial (modern : Bool) (config : Option Nat) (mapped : Option (Option Nat)) : Option Nat :=
  if modern then config else mapped.getD config

theorem modern_template_special (config : Option Nat) (mapped : Option (Option Nat)) :
    templateSpecial true config mapped = config := rfl

theorem legacy_template_override (config value : Option Nat) :
    templateSpecial false config (some value) = value := rfl

theorem legacy_template_missing (config : Option Nat) :
    templateSpecial false config none = config := rfl

/-- Counterexample to the previous config-only template context (ids 2 and 4 in the BPE fixture). -/
theorem legacy_template_diverged :
    some 2 ≠ templateSpecial false (some 2) (some (some 4)) := by decide

def templateAttributes : List String := ["bos_token", "eos_token", "unk_token", "sep_token",
  "pad_token", "cls_token", "mask_token", "additional_special_tokens"]

/-- A parsed attribute is populated iff its string or list is nonempty; null/missing is false.
    Descriptor parsing and tokenizer registration validation are outside this model. -/
def templateVisible (key : String) (populated : Bool) : Bool :=
  templateAttributes.contains key && populated

theorem template_visible_iff (key : String) (populated : Bool) :
    templateVisible key populated = true ↔ key ∈ templateAttributes ∧ populated = true := by
  simp [templateVisible]

theorem unset_template_hidden (key : String) : templateVisible key false = false := by
  simp [templateVisible]

theorem tokenizer_flags_hidden (populated : Bool) :
    templateVisible "add_bos_token" populated = false ∧
    templateVisible "add_eos_token" populated = false := by
  simp [templateVisible, templateAttributes]

/-- The supported string-only legacy additional-token merge. Descriptor merges are refused. -/
def mergeAdditional (config : List String) : List String → List String
  | [] => config
  | token :: rest => mergeAdditional
      (if token ∈ config then config else config ++ [token]) rest

theorem mem_mergeAdditional (config mapped : List String) (token : String) :
    token ∈ mergeAdditional config mapped ↔ token ∈ config ∨ token ∈ mapped := by
  induction mapped generalizing config with
  | nil => simp [mergeAdditional]
  | cons value rest ih =>
    by_cases h : value ∈ config
    · rw [mergeAdditional, ite_eq_left h, ih]
      by_cases ht : token = value
      · subst token; simp [h]
      · simp [ht]
    · rw [mergeAdditional, ite_eq_right h, ih]
      simp [List.mem_append, or_assoc]

end Kadmium.Native.Text
