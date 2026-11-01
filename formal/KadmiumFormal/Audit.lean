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
import KadmiumFormal.Runtime.Bisect
import KadmiumFormal.Runtime.BatchCodec

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
