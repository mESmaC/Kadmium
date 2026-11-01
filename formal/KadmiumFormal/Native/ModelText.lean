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
