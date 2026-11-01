import Lean
import KadmiumFormal.Pow2
import KadmiumFormal.Bytes
import KadmiumFormal.RepOps.TreeSum
import KadmiumFormal.RepOps.TreeReducer
import KadmiumFormal.RepOps.Int8
import KadmiumFormal.RepOps.Digest
import KadmiumFormal.Network.WireFrame
import KadmiumFormal.Network.XorParity
import KadmiumFormal.Data.KtbLayout
import KadmiumFormal.Data.Fts3Chunks
import KadmiumFormal.Data.RangeCoalesce
import KadmiumFormal.Grid.Redundancy
import KadmiumFormal.Grid.Eviction
import KadmiumFormal.Sharding.RowShare
import KadmiumFormal.Sharding.LayerPlacement
import KadmiumFormal.Runtime.Bisect
import KadmiumFormal.Runtime.BatchCodec
import KadmiumFormal.StoragePool
import KadmiumFormal.Transport.ChunkMap
import KadmiumFormal.Transport.Publish
import KadmiumFormal.Native.LayerSplit
import KadmiumFormal.Native.AttentionWindow
import KadmiumFormal.Native.ModelText

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

-- Powers of two
#assert_standard_axioms Kadmium.nextPow2_spec

-- RepOps reduction tree and streaming reducer
#assert_standard_axioms Kadmium.RepOps.treeSum_eq_eval
#assert_standard_axioms Kadmium.RepOps.canonical_leaves
#assert_standard_axioms Kadmium.RepOps.canonical_optimal
#assert_standard_axioms Kadmium.RepOps.legacy_fallback_unreachable
#assert_standard_axioms Kadmium.RepOps.legacySum_eq_treeSum
#assert_standard_axioms Kadmium.RepOps.reducer_eq_treeSum
#assert_standard_axioms Kadmium.RepOps.push_pastEnd
#assert_standard_axioms Kadmium.RepOps.push_twice
#assert_standard_axioms Kadmium.RepOps.finish_incomplete

-- RepOps integer accumulation and digests
#assert_standard_axioms Kadmium.RepOps.dot_perm
#assert_standard_axioms Kadmium.RepOps.dot_fits_i64
#assert_standard_axioms Kadmium.RepOps.sumSquares_fits_i64
#assert_standard_axioms Kadmium.RepOps.fnv1a_a
#assert_standard_axioms Kadmium.RepOps.fnv1a_foobar
#assert_standard_axioms Kadmium.RepOps.digestUpdate_append
#assert_standard_axioms Kadmium.RepOps.digestBytes_eq_fnv1a
#assert_standard_axioms Kadmium.RepOps.preimage_inj
#assert_standard_axioms Kadmium.RepOps.foldChain_preimage_inj

-- Little-endian bytes
#assert_standard_axioms Kadmium.Bytes.decodeLE_encodeLE
#assert_standard_axioms Kadmium.Bytes.decodeLE_encodeLE_mod
#assert_standard_axioms Kadmium.Bytes.encodeLE_decodeLE

-- KDM2 wire framing
#assert_standard_axioms Kadmium.Network.decode_encode
#assert_standard_axioms Kadmium.Network.decode_encodeLegacy
#assert_standard_axioms Kadmium.Network.decode_shortMagic
#assert_standard_axioms Kadmium.Network.decode_badMagic
#assert_standard_axioms Kadmium.Network.decode_shortHeader
#assert_standard_axioms Kadmium.Network.decode_badKind
#assert_standard_axioms Kadmium.Network.decode_oversize
#assert_standard_axioms Kadmium.Network.decode_truncated
#assert_standard_axioms Kadmium.Network.decode_ok_payload_le
#assert_standard_axioms Kadmium.Network.encode_not_total

-- XOR page parity
#assert_standard_axioms Kadmium.Network.reconstruct_parity
#assert_standard_axioms Kadmium.Network.parity_isByte

-- `.ktb` bank layout
#assert_standard_axioms Kadmium.Data.offsets_aligned
#assert_standard_axioms Kadmium.Data.PAGE_le_offsets
#assert_standard_axioms Kadmium.Data.offsets_disjoint
#assert_standard_axioms Kadmium.Data.offsets_in_bounds
#assert_standard_axioms Kadmium.Data.bank_getElem?
#assert_standard_axioms Kadmium.Data.reader_accepts

-- FTS3 chunk manifests
#assert_standard_axioms Kadmium.Data.chunkOffset_succ
#assert_standard_axioms Kadmium.Data.chunks_sum
#assert_standard_axioms Kadmium.Data.chunks_cover
#assert_standard_axioms Kadmium.Data.chunkLen_pos
#assert_standard_axioms Kadmium.Data.chunkLen_le
#assert_standard_axioms Kadmium.Data.chunkLen_full

-- HTTP range coalescing
#assert_standard_axioms Kadmium.Data.coalesce_members
#assert_standard_axioms Kadmium.Data.coalesce_contains
#assert_standard_axioms Kadmium.Data.coalesce_start_attained
#assert_standard_axioms Kadmium.Data.coalesce_stop_attained
#assert_standard_axioms Kadmium.Data.coalesce_separated
#assert_standard_axioms Kadmium.Data.coalesce_valid
#assert_standard_axioms Kadmium.Data.coalesce_optimal

-- Grid redundancy and eviction
#assert_standard_axioms Kadmium.Grid.Redundancy.normalized_valid
#assert_standard_axioms Kadmium.Grid.Redundancy.normalized_idem
#assert_standard_axioms Kadmium.Grid.Redundancy.le_normalized
#assert_standard_axioms Kadmium.Grid.Redundancy.valid_iff_normalized_eq
#assert_standard_axioms Kadmium.Grid.evict_spec

-- Round-robin row shares
#assert_standard_axioms Kadmium.Sharding.count_mine
#assert_standard_axioms Kadmium.Sharding.sum_mine
#assert_standard_axioms Kadmium.Sharding.exactly_one_serves

-- Native checkpoint layer placement
#assert_standard_axioms Kadmium.Sharding.LayerPlacement.planFrom_valid
#assert_standard_axioms Kadmium.Sharding.LayerPlacement.stages_contiguous_cover
#assert_standard_axioms Kadmium.Sharding.LayerPlacement.stages_within_budget
#assert_standard_axioms Kadmium.Sharding.LayerPlacement.stages_boundary_trusted
#assert_standard_axioms Kadmium.Sharding.LayerPlacement.single_stage_shares_tied_embedding
#assert_standard_axioms Kadmium.Sharding.LayerPlacement.infeasible_when_total_exceeds_cluster_capacity
#assert_standard_axioms Kadmium.Sharding.LayerPlacement.planFrom_complete
#assert_standard_axioms Kadmium.Sharding.LayerPlacement.planFrom_none_iff_infeasible
#assert_standard_axioms Kadmium.Sharding.LayerPlacement.planFrom_optimal

-- Verde bisection
#assert_standard_axioms Kadmium.Runtime.bisect_spec
#assert_standard_axioms Kadmium.Runtime.bisect_none

-- MoE batch codecs
#assert_standard_axioms Kadmium.Runtime.decodeF32_encodeF32
#assert_standard_axioms Kadmium.Runtime.decodeF32_exact
#assert_standard_axioms Kadmium.Runtime.wrapped_accepts_empty_f32
#assert_standard_axioms Kadmium.Runtime.checked_rejects_empty_f32
#assert_standard_axioms Kadmium.Runtime.decodePart_encodePart
#assert_standard_axioms Kadmium.Runtime.decodePart_exact
#assert_standard_axioms Kadmium.Runtime.wrapped_accepts_header_only_part
#assert_standard_axioms Kadmium.Runtime.checked_rejects_header_only_part

-- Storage pool
#assert_standard_axioms Kadmium.StoragePool.validateShardCounts_spec
#assert_standard_axioms Kadmium.StoragePool.validateStoredShardCounts_spec
#assert_standard_axioms Kadmium.StoragePool.storedShardOverhead_le_three
#assert_standard_axioms Kadmium.StoragePool.cauchyNodes_disjoint
#assert_standard_axioms Kadmium.StoragePool.shardSize_pos
#assert_standard_axioms Kadmium.StoragePool.payloadLen_le_k_mul_shardSize
#assert_standard_axioms Kadmium.StoragePool.checkStoreQuota_spec
#assert_standard_axioms Kadmium.StoragePool.swapStep_preserves_durable
#assert_standard_axioms Kadmium.StoragePool.runSwap_preserves_durable
#assert_standard_axioms Kadmium.StoragePool.overwrite_durable
#assert_standard_axioms Kadmium.StoragePool.recover_restores_final
#assert_standard_axioms Kadmium.StoragePool.rotate_then_recover_restores_original
#assert_standard_axioms Kadmium.StoragePool.medium_respects_offer
#assert_standard_axioms Kadmium.StoragePool.qualityLimitsCtx_unknown

-- Checkpoint transport
#assert_standard_axioms Kadmium.Transport.readLoop_eq_chunkPlan
#assert_standard_axioms Kadmium.Transport.validChunks_iff
#assert_standard_axioms Kadmium.Transport.chunkPlan_tiles
#assert_standard_axioms Kadmium.Transport.chunkPlan_aligned
#assert_standard_axioms Kadmium.Transport.layout_divides
#assert_standard_axioms Kadmium.Transport.Publish.publish_safe
#assert_standard_axioms Kadmium.Transport.Publish.legacy_unsafe
#assert_standard_axioms Kadmium.Native.runLayers_append
#assert_standard_axioms Kadmium.Native.runChain_eq_runLayers
#assert_standard_axioms Kadmium.Native.Attention.visible_iff
#assert_standard_axioms Kadmium.Native.Attention.full_iff
#assert_standard_axioms Kadmium.Native.Attention.current_visible
#assert_standard_axioms Kadmium.Native.Attention.width_exact
#assert_standard_axioms Kadmium.Native.Attention.wide_window_is_full
#assert_standard_axioms Kadmium.Native.Attention.batch_independent
#assert_standard_axioms Kadmium.Native.Text.accepted_unchanged
#assert_standard_axioms Kadmium.Native.Text.full_context_accepted
#assert_standard_axioms Kadmium.Native.Text.over_context_rejected
#assert_standard_axioms Kadmium.Native.Text.eos_iff_member
#assert_standard_axioms Kadmium.Native.Text.explicit_null_disables
#assert_standard_axioms Kadmium.Native.Text.declared_special_wins
#assert_standard_axioms Kadmium.Native.Text.modern_template_special
#assert_standard_axioms Kadmium.Native.Text.legacy_template_override
#assert_standard_axioms Kadmium.Native.Text.legacy_template_missing
#assert_standard_axioms Kadmium.Native.Text.legacy_template_diverged
#assert_standard_axioms Kadmium.Native.Text.template_visible_iff
#assert_standard_axioms Kadmium.Native.Text.unset_template_hidden
#assert_standard_axioms Kadmium.Native.Text.tokenizer_flags_hidden
#assert_standard_axioms Kadmium.Native.Text.mem_mergeAdditional
