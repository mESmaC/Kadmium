# Formal design verification

`formal/` holds Lean 4 proofs about Kadmium's algorithms, data movement and storage formats. A proof here establishes that what the Rust code does is correct, or optimal under a stated cost model, and it is how a proposed improvement is shown correct before it is built. Lean is not part of any shipped binary. `cargo build` and `cargo test` do not need it.

## What a theorem here covers

Each theorem is about a Lean model of named Rust items. How the model is tied to the code is part of the claim:

| Link | Meaning | Label |
| :--- | :--- | :--- |
| Model with differential fixture | The model follows the Rust function's control flow, and every modelling choice is listed in its module docstring. An executable in this project evaluates the model and writes a fixture. A Rust test requires the implementation to reproduce every value bit for bit. The proof covers the model for every input; the fixture checks that model and code agree on the fixture's inputs. | **Proved (model)** |
| Aeneas translation | Charon and Aeneas translate the Rust source into Lean, so the theorem is about the code itself. Not used yet. | **Proved (Aeneas)** |

An optimality theorem names its cost model, such as the number of additions, the depth of a reduction tree or the number of HTTP requests. It is not a performance figure: a tree of minimal depth says nothing about wall-clock time on a given machine, which still needs a **Measured** entry in `docs/BENCHMARKS_2026-10.md`.

## Results

All results in the tables below are **Proved (model)**. Four executables write the differential fixtures: `lake exe repops_vectors` writes `crates/kadmium-runtime/tests/fixtures/repops_tree_vectors.txt`, checked by `crates/kadmium-runtime/tests/repops_lean_vectors.rs`; `lake exe transport_vectors` writes `crates/kadmium-data/tests/fixtures/transport_chunk_vectors.txt`, checked by `crates/kadmium-data/tests/transport_lean_vectors.rs`; `lake exe storage_pool_vectors` writes `crates/kadmium-data/tests/fixtures/storage_pool_vectors.txt`, checked by `crates/kadmium-data/tests/storage_pool_lean_vectors.rs`; `lake exe kadmium_vectors <repository root>` writes every `crates/*/tests/fixtures/*_lean_vectors.txt`, each checked by a `*_lean_vectors.rs` test in that crate's `tests/` directory. Where a row's Rust item has no fixture, the row says so.

### RepOps reductions (`KadmiumFormal/Pow2.lean`, `RepOps/TreeSum.lean`, `RepOps/TreeReducer.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Kadmium.nextPow2_spec` | `2 ^ ceilLog2 n` is a power of two, at least `n`, and no larger than any other power of two that is at least `n`. | `usize::next_power_of_two`, as used by the reduction |
| `RepOps.treeSum_eq_eval`, `RepOps.canonical_leaves` | `tree_sum` evaluates one fixed tree that reads operands `0 .. n` once each, in index order, never past the end of the slice. | `repops::tree_sum`, `tree_node` |
| `RepOps.canonical_optimal` | That tree performs `n - 1` additions and has depth `ceilLog2 n`. Every binary reduction tree over the same `n` operands, in any order, performs `n - 1` additions and has depth at least `ceilLog2 n`. A fixed reduction order costs nothing in work or critical path. | `repops::tree_sum` |
| `RepOps.reducer_eq_treeSum` | Pushing the leaves in any order, then calling `finish`, returns exactly `tree_sum` of the same values. No push fails and the root is always formed. | `repops::TreeReducer::{new, push, push_chunk, finish}` |
| `RepOps.push_pastEnd`, `RepOps.push_twice`, `RepOps.finish_incomplete` | A leaf past the end, a leaf pushed twice and an early `finish` are rejected. | `repops::TreeReducer` |
| `RepOps.legacySum_eq_treeSum` | The legacy Verde reduction builds the same tree as `tree_sum`, so the two agree bit for bit. | `VerdeVerificationEngine::repops_pairwise_tree_sum` |
| `RepOps.legacy_fallback_unreachable` | The legacy reduction's `if mid >= n { n / 2 }` branch is never taken for any `n ≥ 2`. | `VerdeVerificationEngine::repops_pairwise_tree_sum` |

Every RepOps tree theorem holds for an arbitrary addition `add : α → α → α` with no associativity, commutativity or identity assumed. IEEE `f32` and `f64` addition has none of those laws, so the statements hold bit for bit for floats without a floating-point model: two reductions that evaluate the same tree perform the same additions on the same operands.

### RepOps integer accumulation and digests (`RepOps/Int8.lean`, `RepOps/Digest.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `RepOps.dot_perm` | The INT8 dot product and sum of squares are the same integer for every summation order, so any chunking or threading of the accumulation is exact. | `repops::int8_dot`, `repops::int8_sum_squares` |
| `RepOps.dot_fits_i64`, `RepOps.sumSquares_fits_i64` | The accumulation of fewer than `2^49` products of INT8 codes has magnitude below `2^63`, so the `i64` never wraps. | `repops::int8_dot`, `repops::int8_sum_squares` |
| `RepOps.fnv1a_a`, `RepOps.fnv1a_foobar` | The FNV-1a 64 fold reproduces the published reference vectors for `"a"` and `"foobar"` (and `""` by definition). | `repops::fnv1a`, `compute_fnv1a64_checksum`, `kadmium_grid::fnv1a_64` |
| `RepOps.digestUpdate_append` | Two `update` calls equal one `update` over the concatenation: a digest never depends on how its input was chunked. | `repops::Digest::update` and the typed `update_*` helpers |
| `RepOps.digestBytes_eq_fnv1a` | `digest_bytes(tag, bytes)` is FNV-1a of `manifest id (8 bytes LE) ++ tag ++ [0] ++ bytes`. | `repops::Digest::new`, `repops::digest_bytes` |
| `RepOps.preimage_inj`, `RepOps.foldChain_preimage_inj` | That byte string determines the manifest id, the NUL-free tag and the payload, and the `fold_chain` encoding determines its digests (fewer than `2^32` of them). Digests computed under different manifests, tags or payloads hash different byte strings. Equal digests do not imply equal inputs: FNV-1a is not collision resistant, and no theorem says otherwise. | `repops::Digest::new`, `repops::fold_chain` |

Fixtures: the FNV-1a vectors are checked by `crates/kadmium-data/tests/fnv1a_lean_vectors.rs` (`compute_fnv1a64_checksum`), `crates/kadmium-grid/tests/fnv1a_lean_vectors.rs` (`fnv1a_64`) and `crates/kadmium-runtime/tests/digest_lean_vectors.rs`, which also checks `digest_bytes`, `Digest::{new, update, update_u32}` and `fold_chain` against the modelled preimage and chain bytes through the public API, with the manifest id the running binary reports. The private `repops::fnv1a` is reached only through those.

### Little-endian bytes (`KadmiumFormal/Bytes.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Bytes.decodeLE_encodeLE`, `Bytes.encodeLE_decodeLE` | A `w`-byte little-endian encoding is lossless below `256 ^ w`, in both directions. | `u32::to_le_bytes`, `u64::to_le_bytes` and `from_le_bytes` in every codec |
| `Bytes.decodeLE_encodeLE_mod` | Encoding a value that does not fit keeps its low `8 * w` bits, which is what `n as u32` does before `to_le_bytes`. | the `as u32` casts in `WireFrame::encode` and `fold_chain` |

### KDM2 wire framing (`Network/WireFrame.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Network.decode_encode`, `Network.decode_encodeLegacy` | A frame with a legal kind, an epoch below `2^32`, a request id below `2^64` and at most `2^30` payload bytes decodes back to itself from any byte string that starts with its encoding; a legacy `KDMF` frame decodes with request id `0`. | `WireFrame::encode_with_request_id`, `WireFrame::decode` |
| `Network.decode_shortMagic`, `decode_badMagic`, `decode_shortHeader`, `decode_badKind`, `decode_oversize`, `decode_truncated` | A short input, an unknown magic, a short header, a kind byte outside `1 ..= 15`, a declared length above `2^30` and a payload shorter than declared are rejected. The oversize rejection holds for every payload, so the length is refused before any payload byte is read or allocated. | `WireFrame::decode`, `WireFrame::parse_header`, `WireFrame::read_from_stream` |
| `Network.decode_ok_payload_le` | Whatever decodes carries at most `2^30` payload bytes. | `WireFrame::decode` |
| `Network.encode_not_total` | A frame with more than `2^30` payload bytes never round-trips: `encode` writes `payload.len() as u32` and does not check the cap. See Findings. | `WireFrame::encode_with_request_id` |

Fixture: `crates/kadmium-network/tests/wire_frame_lean_vectors.rs` runs every frame and every rejected input through both `WireFrame::decode` and `WireFrame::read_from_stream`.

### XOR page parity (`Network/XorParity.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Network.reconstruct_parity` | For every stripe of pages and every index, the parity XOR the other pages is the missing page, byte for byte. | `SwarmRaidStriper::encode_xor_parity_page`, `reconstruct_missing_page_xor` |
| `Network.parity_isByte` | The parity of bytes is a byte. | `SwarmRaidStriper::encode_xor_parity_page` |

### `.ktb` bank layout (`Data/KtbLayout.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Data.offsets_aligned`, `Data.PAGE_le_offsets` | Every tensor starts on a 64 KiB page boundary at or after page 1. | `KtbBankPacker::pack_to_bytes`, `checkpoint::write_bank` |
| `Data.offsets_disjoint`, `Data.offsets_in_bounds` | Tensor `i` ends at or before tensor `j > i` begins and before the end of the bank: the reservations tile `[PAGE, total)`. | `KtbBankPacker::pack_to_bytes`, `checkpoint::write_bank`, `Checkpoint::verify` |
| `Data.bank_getElem?` | Byte `j` of tensor `i` is at `offsets[i] + j` of the bank image, unchanged, for every tensor including empty ones. The copy into a bank is byte-exact. | `KtbBankPacker::pack_to_bytes`, `checkpoint::write_bank` |
| `Data.reader_accepts` | Every entry the packer writes passes the reader's range check, so `from_slice(pack(..))` never rejects a packed bank. | `KtbBankReader::from_slice`, `KtbBankReader::tensor_range` |

Fixture: `crates/kadmium-data/tests/ktb_layout_lean_vectors.rs` exercises `KtbBankPacker::pack_to_bytes` and `KtbBankReader`. `checkpoint::write_bank` and `Checkpoint::verify` compute the same reservation and offset formula but write and read files; they are covered as a model only, read against the model rather than run against the fixture.

### FTS3 chunk manifests (`Data/Fts3Chunks.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Data.chunkOffset_succ`, `Data.chunks_sum`, `Data.chunks_cover` | The chunks tile `[0, len)` contiguously in index order and their lengths add up to the bank. | `Fts3ChunkSlicer::compute_chunk_manifest` |
| `Data.chunkLen_pos`, `Data.chunkLen_le`, `Data.chunkLen_full` | Every chunk is non-empty and at most `chunk_size` bytes, and every chunk but the last is exactly `chunk_size` bytes. | `Fts3ChunkSlicer::compute_chunk_manifest` |

### HTTP range coalescing (`Data/RangeCoalesce.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Data.coalesce_members` | The members of the requests, in request order, are the slice indices `0 .. n`, each once: every slice is fetched by exactly one request. | `HfByteRangeSlicer::coalesce_range_requests` |
| `Data.coalesce_contains`, `Data.coalesce_start_attained`, `Data.coalesce_stop_attained`, `Data.coalesce_separated` | Every member lies inside its request's byte range, every request begins at a member's first byte and ends at a member's last byte, and consecutive requests are more than `max_gap_bytes` apart. | `HfByteRangeSlicer::coalesce_range_requests` |
| `Data.coalesce_valid` | Every window of `max_gap_bytes + 1` consecutive bytes inside a request touches a member, so, with the ends attained, a request fetches no run of more than `max_gap_bytes` unrequested bytes anywhere. | `HfByteRangeSlicer::coalesce_range_requests` |
| `Data.coalesce_optimal` | **Fewest requests.** Any list of requests, each containing the slices it serves and fetching no run of more than `max_gap_bytes` unrequested bytes, whose members together cover every slice, has at least as many requests as the loop produces. Cost model: the number of HTTP requests, subject to the gap bound. The model covers one URL; slices of different URLs never merge. | `HfByteRangeSlicer::coalesce_range_requests` |

### Storage pool (`KadmiumFormal/StoragePool.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `StoragePool.validateShardCounts_spec`, `StoragePool.validateStoredShardCounts_spec` | `validate_shard_counts` accepts `(k, m)` iff `1 ≤ k`, `1 ≤ m`, and `k + m ≤ 256`; `validate_stored_shard_counts` accepts `(k, m)` iff `1 ≤ k ≤ 16`, `1 ≤ m ≤ 8`, `k + m ≤ 20`, and `m ≤ max(k, 2)`. | `DecentralizedStoragePoolEngine::{validate_shard_counts, validate_stored_shard_counts}` |
| `StoragePool.storedShardOverhead_le_three`, `StoragePool.cauchyNodes_disjoint` | For every `(k, m)` accepted by `validate_stored_shard_counts`, total shards satisfy `k + m ≤ 3 * k` (`(k + m) / k ≤ 3` storage overhead bound), and Cauchy evaluation points `x_p = k + p` (`p < m`) and `y_j = j` (`j < k`) are pairwise distinct in `0 .. 256`. | `DecentralizedStoragePoolEngine::validate_stored_shard_counts`, `systematic_cauchy_row` |
| `StoragePool.shardSize_pos`, `StoragePool.payloadLen_le_k_mul_shardSize` | For `k ≥ 1`, `shardSize payloadLen k ≥ 1` and `payloadLen ≤ k * shardSize payloadLen k`, so `k` equal-sized data shards cover every payload byte without truncation. | `encode_erasure_coded_shards` |
| `StoragePool.checkStoreQuota_spec` | The store quota check accepts a write iff the post-write object count is at most `maxObjects` and total physical bytes are at most `maxTotalBytes`. | `BucketObjectStore::put_bytes` |
| `StoragePool.swapStep_preserves_durable`, `StoragePool.runSwap_preserves_durable`, `StoragePool.overwrite_durable`, `StoragePool.recover_restores_final`, `StoragePool.rotate_then_recover_restores_original` | Across any sequence of staging, rotation, commit, rollback, crash, and recovery events under the store lock, a committed object is never lost (`Durable`), and `recover_interrupted_writes` always restores `objects/<dir>` (`final ≠ none`). | `BucketObjectStore::{put_bytes, recover_interrupted_writes}` |
| `StoragePool.medium_respects_offer` | Available medium capacity `mediumAvailable offer hw` is at most `min(offer, hw)`, pledged bucket capacity `mediumPledged` is at most available capacity (and therefore at most `offer`), and a zero offer (`offer = 0`, e.g. `PoliteState::Draining`) pledges zero capacity (`RespectsOffer`). | `DecentralizedStoragePoolEngine::{compute_medium_availability_and_pledge_tenths, compute_medium_availability_and_pledge, inspect_node_storage_mediums_with_offer}` |
| `StoragePool.qualityLimitsCtx_unknown` | When a model is unknown (`known = none`) and no manual user override is set (`override = none`), no context window is fabricated (`qualityLimitsCtx none none = none`). | `ModelQualityLimits::{for_model, effective_context_window_tokens}` |

Conjectures stated in `formal/KadmiumFormal/StoragePool.lean` (**Conjecture**):
- `Kadmium.StoragePool.CauchyMdsConjecture` (**Conjecture**): Every `k × k` submatrix of the `(k + m) × k` systematic Cauchy matrix over `GF(2^8)` (for `(k, m)` accepted by `validate_stored_shard_counts`) is invertible under Gaussian elimination, so any `k` surviving shards out of `k + m` reconstruct the original `k` data shards bit for bit.

### Transport chunk maps (`Transport/ChunkMap.lean`)

The differential fixture is `crates/kadmium-data/tests/fixtures/transport_chunk_vectors.txt`, checked by `crates/kadmium-data/tests/transport_lean_vectors.rs` against both Rust functions, including short reads. Modelling choices are listed in `KadmiumFormal/Transport/ChunkMap.lean`: byte counts are unbounded `Nat`, digests are not modelled, and a read returns 0 only at end of file.

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Transport.readLoop_eq_chunkPlan` | The sender's read loop produces exactly `chunkPlan`, so every chunk map the sender builds passes the receiver's chunk check. | `checkpoint::transport::hash_stream`, `chunk_plan` |
| `Transport.validChunks_iff` | The receiver's chunk check accepts exactly one map per file: `chunkPlan`. | `TransportManifest::validate`, `chunk_plan` |
| `Transport.chunkPlan_tiles` | That map starts at offset 0, each chunk starts where the previous one ends, every chunk holds 1 to `chunk_bytes` bytes, and the chunks cover the file exactly once. An empty file has no chunks. | `chunk_plan` |
| `Transport.chunkPlan_aligned` | If an alignment divides the chunk size, every chunk offset is aligned to it. | `chunk_plan` |
| `Transport.layout_divides` | 4 KiB divides the 64 KiB KTB page, which divides the 16 MiB chunk. | `TRANSPORT_ALIGNMENT_BYTES`, `TRANSPORT_CHUNK_BYTES` |

### Publishing a received package (`Transport/Publish.lean`)

**Proved (model)**, about a model only: there is no differential fixture. Renames are assumed atomic, a directory is abstracted to whether it holds every file of its own manifest intact, and verification succeeds exactly for such a directory. The unit test `checkpoint::transport::tests::begin_during_finish_cannot_publish_a_package_without_its_manifest` runs the legacy counterexample's interleaving against the Rust code. Modelling choices are listed in `KadmiumFormal/Transport/Publish.lean`.

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Transport.Publish.publish_safe` | Claiming staging into a private directory before verifying it means every published package verified as complete, under every interleaving of begins, chunk arrivals and concurrent finishes. | `CheckpointLibrary::{receive_begin, receive_finish}`, `finish_claimed` |
| `Transport.Publish.legacy_unsafe` | The previous verify-in-place-then-rename order publishes an incomplete package when a begin lands between the two. | `receive_finish` before PR #62's fix |

### Grid redundancy and eviction (`Grid/Redundancy.lean`, `Grid/Eviction.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Grid.Redundancy.normalized_valid`, `normalized_idem`, `valid_iff_normalized_eq` | Normalisation establishes `1 ≤ min_quorum ≤ target_replicas ≤ max_replicas`, is idempotent, and leaves exactly the valid policies unchanged. | `RedundancyPolicy::normalized` |
| `Grid.Redundancy.le_normalized` | Normalisation never lowers a count, so a quorum the operator asked for is never weakened. | `RedundancyPolicy::normalized` |
| `Grid.evict_spec` | The replicas offered for eviction are the shortest prefix of the sorted unpinned replicas that frees the requested bytes, or every unpinned replica. This is about prefixes of the `(hits, last_access)` order; the loop does not search for the smallest sufficient subset over all replicas and the theorem does not claim one. | `ReplicaCatalog::evict_candidates` |

### Round-robin row shares (`Sharding/RowShare.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Sharding.count_mine` | Copy `i` of `c` serves exactly `(rows - i).div_ceil(c)` of the rows, the closed form the planner reports. | `ExpertPlacementPlan::row_share` |
| `Sharding.sum_mine`, `Sharding.exactly_one_serves` | The shares of the `c` copies add up to the batch, and every row is served by exactly one copy. | `ExpertPlacementPlan::row_share`, `ExpertRowShare::serves` |

### Native checkpoint layer placement (`Sharding/LayerPlacement.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Sharding.LayerPlacement.planFrom_valid`, `stages_contiguous_cover` | Every accepted placement partitions `0 .. numLayers` into contiguous non-empty stage ranges in order without gaps or overlaps, with `embedding = true` on stage 0 and `head = true` on the final stage. | `LayerPlacementPlanner::plan_with_budgets`, `plan_checkpoint_placement` |
| `Sharding.LayerPlacement.stages_within_budget`, `stages_boundary_trusted` | Every stage's `weightBytes` and full-context `kvBytes` add up to `totalBytes`, satisfy `totalBytes ≤ capacityBytes` with `0 < capacityBytes`, and place stage 0 and the final stage only on `TrustedCore` machines (`trusted = true`). | `LayerPlacementPlanner::plan_with_budgets`, `CheckpointMemoryProfile::slice_total_bytes` |
| `Sharding.LayerPlacement.single_stage_shares_tied_embedding` | A single-machine placement (`0 .. numLayers`) of a model with `tieWordEmbeddings = true` counts the shared embedding matrix once, whereas splitting across two stages counts `embeddingBytes` on stage 0 and `lmHeadBytes` on the final stage. | `CheckpointMemoryProfile::slice_weight_bytes`, `DecoderSlice::load_from` |
| `Sharding.LayerPlacement.planFrom_complete`, `planFrom_none_iff_infeasible`, `infeasible_when_total_exceeds_cluster_capacity` | `planFrom` returns `none` if and only if no valid full-context placement exists on the candidate machines, and any valid placement requires total stage bytes at most the sum of candidate machine capacities. | `LayerPlacementPlanner::plan_with_budgets` |
| `Sharding.LayerPlacement.planFrom_optimal` | **Minimax relative memory load.** Among all valid placements on the candidate machines, `planFrom` minimizes the bottleneck per-machine relative memory load `totalBytes / capacityBytes` (compared exactly by cross-multiplication `loadLe`). Cost model: peak stage `totalBytes / capacityBytes`. | `LayerPlacementPlanner::plan_with_budgets` |

Fixture: `crates/kadmium-sharding/tests/layer_placement_lean_vectors.rs` runs every fixture line from `crates/kadmium-sharding/tests/fixtures/layer_placement_lean_vectors.txt` through `CheckpointMemoryProfile` and `LayerPlacementPlanner::plan_with_budgets`. `crates/kadmium-runtime/tests/native_decoder.rs` checks that `checkpoint_memory_profile` matches `DecoderSlice::stored_weight_bytes()` and full-context `KvCache::bytes()` across all four model fixtures and every cut point.

### Verde bisection (`Runtime/Bisect.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Runtime.bisect_spec` | With a diverging final state, the search returns a layer whose output state diverges while its input state does not (or is the uncompared input state `0`), in at most `ceilLog2 L + 1` referee evaluations. Divergence is not assumed monotone along the trajectory. | `VerdeVerificationEngine::bisect_faulty_layer` |
| `Runtime.bisect_none` | Without layers or without a diverging final state there is nothing to find. | `VerdeVerificationEngine::bisect_faulty_layer` |

### MoE batch codecs (`Runtime/BatchCodec.lean`)

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Runtime.decodeF32_encodeF32`, `Runtime.decodePart_encodePart` | The encoders' output decodes back to the input for every batch whose counts fit `u32`. | `TokenBatchF32::{to_le_bytes, from_le_bytes}`, `ExpertPartialsBatch::{to_le_bytes, from_le_bytes}` |
| `Runtime.decodeF32_exact`, `Runtime.decodePart_exact` | An accepted `F32B` body has exactly `rows * dim * 4` bytes. An accepted `PART` input has exactly `20 + count * (8 + dim * 4)` bytes, where `count` is both the header's count field and the number of decoded partials, and every decoded partial carries `dim * 4` data bytes. Nothing is read past the input. | `TokenBatchF32::from_le_bytes`, `ExpertPartialsBatch::from_le_bytes` |
| `Runtime.wrapped_accepts_empty_f32`, `Runtime.wrapped_accepts_header_only_part`, and their `checked_rejects_*` counterparts | In wrapping `usize` arithmetic a `2^31 × 2^31` `F32B` header and a `PART` header for `2^31` partials of width `2^31 - 2` both imply an empty body; the checked arithmetic rejects both. See Findings. | `TokenBatchF32::from_le_bytes`, `ExpertPartialsBatch::from_le_bytes` |

### Native decoder layer splits (`Native/LayerSplit.lean`)

**Proved (model)**, about a model only: there is no differential fixture. A layer is any function of the activations and its own KV cache, so the statements hold bit for bit for `f32` without a floating-point model. The Rust test `splitting_the_layers_at_every_cut_is_bit_identical` in `crates/kadmium-runtime/tests/native_decoder.rs` checks the property on four checkpoint layouts, every pair of cuts, through prefill and token-by-token decode. Modelling choices are listed in the module docstring.

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Native.runLayers_append` | Running layers `fs ++ gs` with caches `cs ++ ds`, where `cs` has one cache per layer of `fs`, equals running `fs`, then `gs` on its activations, with the caches concatenated. | `native_decoder::DecoderSlice::forward` |
| `Native.runChain_eq_runLayers` | A chain of slices, each with one cache per layer, gives the same activations and caches as one slice holding all their layers. Where the layers are cut and how many machines hold them cannot change an output. | `DecoderSlice::forward` over a chain of slices |

### Native text adapter (`Native/ModelText.lean`)

**Proved (model)** for the context gate and EOS membership, with differential cases generated by
`TextVectors.lean` into `crates/kadmium-runtime/tests/fixtures/model_text_contract.txt` and checked
by `context_gate_and_eos_match_lean_vectors` in `tests/model_text.rs`. Tokenization and rendering
are opaque inputs to this model. These theorems do not prove Hugging Face or Jinja equivalence.

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Native.Text.accepted_unchanged` | Acceptance returns the complete input token list and its length is within context. | `model_text::ModelText::encode` context gate |
| `Native.Text.full_context_accepted` | A token list exactly filling context passes the gate. | `ModelText::encode` context gate |
| `Native.Text.over_context_rejected` | A token list longer than context fails the gate. | `ModelText::encode` context gate |
| `Native.Text.eos_iff_member` | EOS recognition is exactly membership in the configured stop-token list. | `model_text::GenerationDefaults::is_eos` |
| `Native.Text.explicit_null_disables`, `Native.Text.declared_special_wins` | An explicit special-token setting, including null, wins over the legacy fallback. Model only; the Rust test `empty_generation_config_is_not_legacy_config_and_explicit_null_disables_eos` checks examples. | `GenerationDefaults::load` special-token lookup |
| `Native.Text.modern_template_special`, `Native.Text.legacy_template_override`, `Native.Text.legacy_template_missing` | Modern tokenizer metadata ignores the legacy map. A legacy map entry, including null, overrides the config; a missing entry preserves it. Differential cases in `TextVectors.lean` exercise the resolved EOS template value. | `model_text::specials::resolve` |
| `Native.Text.legacy_template_diverged` | Counterexample to the previous config-only lookup: EOS id 2 in the config disagrees with legacy map id 4. The Transformers fixture `legacy_override` in `special_tokens_reference.json` ties this to Rust. | `specials::resolve`; previous `template::render` |
| `Native.Text.template_visible_iff`, `Native.Text.unset_template_hidden`, `Native.Text.tokenizer_flags_hidden` | Only populated special-token attributes enter the template context; unset values and BOS/EOS configuration flags do not. Parsing and descriptor registration are outside the model. The Transformers differential fixture checks these branches. | `specials::resolve`, `template::render` |
| `Native.Text.mem_mergeAdditional` | The string-only additional-token merge contains exactly tokens from either input. Model only, with the `legacy_null_and_union` Transformers differential example. Descriptor merges are refused rather than modelled. | `specials::resolve` additional-token merge |


Modelling assumptions shared by these results:

- Indices, byte counts and offsets are unbounded natural numbers. Where the Rust uses `checked_*` arithmetic the model covers every call that does not fail; where it casts with `as`, the model writes the truncation explicitly (`Bytes.decodeLE_encodeLE_mod`).
- Byte strings are lists of naturals below 256. An `f32` or `f64` in a codec is its four or eight little-endian bytes; the models move bytes and never interpret them, as the Rust codecs do.
- The reducer's `Vec<Option<T>>` is a function from index to `Option`. The proofs track that every index the algorithm touches is below `2 * capacity`.
- Error strings are modelled as the constructors of an error type per module, such as `ReduceError` and `ShardCountError`.
- `Summand::add` is a function of its two operands, which holds for `f32`, `f64` and `i64`.
- Sorting is a modelling boundary: `coalesce_range_requests` sorts its input with a stable `sort_by` before the loop, and the model takes the sorted order as a hypothesis.
- Directory states in `BucketObjectStore` are modelled per object slot as `(final, old, staging)` version tags serialized by `<root>/.store.lock`.


## Findings

- Kadmium has two implementations of the RepOps reduction: `repops::tree_sum` and the legacy `VerdeVerificationEngine::repops_pairwise_tree_sum`, which `kernels.rs`, `live_worker.rs` and the Verde engine still call. `legacySum_eq_treeSum` shows they agree on every input, so they can be consolidated without changing any digest. Its `n / 2` fallback is dead code.
- `WireFrame::encode_with_request_id` writes `payload.len() as u32` and does not check `KDM_FRAME_MAX_PAYLOAD_BYTES`; only the decoder does. `Network.encode_not_total` proves that a frame with more than `2^30` payload bytes never round-trips: the peer refuses the length, or, past `2^32` bytes, the length wraps and the peer reads a shorter payload. No caller builds such a frame today; the sender-side check is a follow-up.
- `TokenBatchF32::from_le_bytes` and `ExpertPartialsBatch::from_le_bytes` computed the expected body length with unchecked `usize` multiplication. `Runtime.wrapped_accepts_empty_f32` and `Runtime.wrapped_accepts_header_only_part` are the headers for which the wrapped product is `0`: a release build accepted them with no body (and `ExpertPartialsBatch` then reserved a `Vec` for `2^31` partials), a debug build panicked on the overflow. Both decoders now use `checked_mul` and return an error; `crates/kadmium-runtime/tests/batch_codec_lean_vectors.rs` feeds both headers.

## Building

The toolchain is pinned in `lean-toolchain`; `elan` installs it on first use. There are no Lake dependencies.

```bash
cd formal
lake build --wfail
lake exe repops_vectors ../crates/kadmium-runtime/tests/fixtures/repops_tree_vectors.txt
lake exe transport_vectors ../crates/kadmium-data/tests/fixtures/transport_chunk_vectors.txt
lake exe text_vectors ../crates/kadmium-runtime/tests/fixtures/model_text_contract.txt
lake exe storage_pool_vectors ../crates/kadmium-data/tests/fixtures/storage_pool_vectors.txt
lake exe kadmium_vectors ..
cd ..
cargo test --locked -p kadmium-runtime --test repops_lean_vectors
cargo test --locked -p kadmium-data --test transport_lean_vectors --test storage_pool_lean_vectors
for crate in kadmium-runtime kadmium-network kadmium-data kadmium-grid kadmium-sharding; do
  cargo test --locked -p "$crate" --test '*_lean_vectors'
done
cargo test --locked -p kadmium-runtime --test model_text
```

`--test` selects integration-test binaries by name and accepts a glob. A bare trailing filter such as `-- lean_vectors` would match test function names instead, and no test function is named that way, so it runs nothing and still exits 0.

`lake build --wfail` checks every proof and turns warnings, including `declaration uses 'sorry'`, into errors. `KadmiumFormal/Audit.lean` fails the build if a listed theorem depends on any axiom other than `propext`, `Classical.choice` and `Quot.sound`, which rules out `sorry`, `native_decide` and project-local axioms. CI runs all five fixture executables and fails if any regenerated fixture differs from the committed one.

## Adding a result

1. Model the Rust item in `KadmiumFormal/<Area>/`. Name the Rust file and function in the module docstring and list every modelling choice.
2. State the theorem for the general case, such as every `add` or every input length, not for sample inputs.
3. For an optimality claim, state the cost model and prove a lower bound over all alternatives, not only a bound for the chosen algorithm.
4. Add the theorem to `KadmiumFormal/Audit.lean`.
5. Add a fixture to `Fixtures.lean` and a Rust test named `<stem>_lean_vectors.rs` in the owning crate that reproduces it. A row whose Rust item is not run against a fixture says "model only".
6. Add a row to the results table, and cite the result elsewhere only with its label.
7. If a later Rust change alters the behaviour of a covered item, update the model and the proof in the same PR.
