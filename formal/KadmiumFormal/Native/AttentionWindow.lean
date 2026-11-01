import Std

/-!
# Mistral attention visibility

`DecoderConfig::attention_range` and `DecoderSlice::block_forward` use an exclusive absolute
query end and natural (saturating) subtraction. A window counts the current token. The model
describes only the attention mask, not floating-point scores or logits. Full KV storage and
the separate native context check remain in the Rust implementation.

The generated `attention_window_lean_vectors.txt` fixture compares `bounds` with the Rust
method. Windows are positive after config validation; `none` represents full attention.
-/

namespace Kadmium.Native.Attention

def bounds (window : Option Nat) (stop : Nat) : Nat × Nat :=
  (window.elim 0 (fun width => stop - width), stop)

def Visible (window : Option Nat) (stop key : Nat) : Prop :=
  (bounds window stop).1 ≤ key ∧ key < (bounds window stop).2

theorem visible_iff (width stop key : Nat) :
    Visible (some width) stop key ↔ key < stop ∧ stop ≤ key + width := by
  simp only [Visible, bounds, Option.elim_some]
  omega

theorem full_iff (stop key : Nat) : Visible none stop key ↔ key < stop := by
  simp [Visible, bounds]

theorem current_visible (width query : Nat) (h : 0 < width) :
    Visible (some width) (query + 1) query := by
  rw [visible_iff]
  omega

theorem width_exact (width stop : Nat) :
    (bounds (some width) stop).2 - (bounds (some width) stop).1 = min width stop := by
  simp only [bounds, Option.elim_some]
  omega

theorem wide_window_is_full (width stop : Nat) (h : stop ≤ width) :
    bounds (some width) stop = bounds none stop := by
  simp [bounds, Nat.sub_eq_zero_of_le h]

theorem batch_independent (window : Option Nat) (start row query : Nat)
    (h : start + row = query) :
    bounds window (start + row + 1) = bounds window (query + 1) := by
  rw [h]

end Kadmium.Native.Attention
