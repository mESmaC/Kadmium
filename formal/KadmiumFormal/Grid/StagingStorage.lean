/-!
# Logical storage for resumable staging

Models TransportManifest::storage_bytes, PreparedSlice::staging_plan and the resident/pending
charges in ReplicaCatalog::staging_bytes_on, reserve_staging and apply_staging. The unit test
staging_storage_lean_vectors compares the catalogue arithmetic and capacity gate against generated
vectors. receiver_metadata_is_reserved_at_the_exact_offer_boundary measures the logical bytes
of a real received package against storage_bytes. Source failures and cancellation are exercised
by Rust regression tests, not modelled filesystem operations.

Payload includes every transported file; metadata is the serialized transport manifest and
receipts is the sum of marker payload lengths. Filesystem allocation blocks, directory entries,
concurrency and crash recovery are outside this model. Nat arithmetic models successful checked
u64 arithmetic only. Retained bytes being at most the reservation is an explicit hypothesis.
-/

namespace Kadmium.Grid.StagingStorage

def storageBytes (payload metadata receipts : Nat) : Nat := payload + metadata + receipts

def gateAccepts (resident reserved capacity : Nat) : Bool :=
  decide (resident + reserved ≤ capacity)

theorem legacy_gate_admits_offer_breach :
    gateAccepts 0 890343 890343 = true ∧ ¬ (0 + storageBytes 890343 4961 0 ≤ 890343) := by
  decide

theorem gate_respects_logical_offer (resident payload metadata receipts capacity : Nat) :
    gateAccepts resident (storageBytes payload metadata receipts) capacity = true ↔
      resident + payload + metadata + receipts ≤ capacity := by
  simp [gateAccepts, storageBytes, Nat.add_assoc]

theorem retained_within_offer (resident reserved retained capacity : Nat)
    (accepted : gateAccepts resident reserved capacity = true)
    (bounded : retained ≤ reserved) : resident + retained ≤ capacity := by
  have allowed : resident + reserved ≤ capacity := by simpa [gateAccepts] using accepted
  exact Nat.le_trans (Nat.add_le_add_left bounded resident) allowed

def charged (resident other pending : Nat) (resuming : Bool) : Nat :=
  resident + other + if resuming then 0 else pending

theorem resume_counts_once (resident other pending : Nat) :
    charged resident other pending true + pending = charged resident other pending false := by
  simp [charged]

theorem publish_counts_once (resident other pending : Nat) :
    charged (resident + pending) other 0 false = charged resident other pending false := by
  simp [charged, Nat.add_comm, Nat.add_left_comm]

end Kadmium.Grid.StagingStorage
