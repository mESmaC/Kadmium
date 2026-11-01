/-!
# Selecting the weights of one decoder slice

Models SlicePlan::tensor_names in native_decoder/model.rs and an abstract list filter, not the
raw tensor copies in CheckpointLibrary::select_tensors. Parts abstract validated tensor names: every tensor required
by block_specs belongs to its numbered layer. The embedding is also needed by a tied head.
Configuration/header validation, I/O, checksums and byte offsets are not modelled here. Payloads
are arbitrary values, not floating-point numbers. Selection retains an entire payload unchanged.

native_staging_lean_vectors.rs compares needed with the actual tensor names for four checkpoint
layouts. selected_packages_contain_exact_tensors_and_original_bits checks the copy against the
original bytes, dtype and shape as a Measured integration check. The list theorems are
Proved (model), not Aeneas extraction or proofs of copy offsets, padding or contiguity.
-/

namespace Kadmium.Native.Shard

inductive Part where
  | embedding
  | layer (index : Nat)
  | norm
  | head
  deriving DecidableEq

def needed (start stop : Nat) (embedding head tied : Bool) : Part → Bool
  | .embedding => embedding || (head && tied)
  | .layer index => decide (start ≤ index ∧ index < stop)
  | .norm => head
  | .head => head && !tied

theorem layer_needed_iff (start stop index : Nat) (embedding head tied : Bool) :
    needed start stop embedding head tied (.layer index) = true ↔
      start ≤ index ∧ index < stop := by
  simp [needed]

theorem tied_head_needs_embedding (start stop : Nat) (embedding : Bool) :
    needed start stop embedding true true .embedding = true := by
  simp [needed]

def select {α : Type} (start stop : Nat) (embedding head tied : Bool)
    (tensors : List (Part × α)) : List (Part × α) :=
  tensors.filter fun tensor => needed start stop embedding head tied tensor.1

theorem selected_iff {α : Type} (start stop : Nat) (embedding head tied : Bool)
    (tensors : List (Part × α)) (part : Part) (payload : α) :
    (part, payload) ∈ select start stop embedding head tied tensors ↔
      (part, payload) ∈ tensors ∧ needed start stop embedding head tied part = true := by
  simp [select]

theorem payload_preserved {α : Type} (start stop : Nat) (embedding head tied : Bool)
    (tensors : List (Part × α)) (part : Part) (payload : α)
    (present : (part, payload) ∈ select start stop embedding head tied tensors) :
    (part, payload) ∈ tensors :=
  ((selected_iff start stop embedding head tied tensors part payload).mp present).1

end Kadmium.Native.Shard
