# Formal design verification

`formal/` holds Lean 4 proofs about Kadmium's algorithms, data movement and storage formats. A proof here establishes that what the Rust code does is correct, or optimal under a stated cost model, and it is how a proposed improvement is shown correct before it is built. Lean is not part of any shipped binary. `cargo build` and `cargo test` do not need it.

## What a theorem here covers

Each theorem is about a Lean model of named Rust items. How the model is tied to the code is part of the claim:

| Link | Meaning | Label |
| :--- | :--- | :--- |
| Model with differential fixture | The model follows the Rust function's control flow, and every modelling choice is listed in its module docstring. An executable in this project evaluates the model and writes a fixture. A Rust test requires the implementation to reproduce every value bit for bit. The proof covers the model for every input; the fixture checks that model and code agree on the fixture's inputs. | **Proved (model)** |
| Aeneas translation | Charon and Aeneas translate the Rust source into Lean, so the theorem is about the code itself. Not used yet. | **Proved (Aeneas)** |

An optimality theorem names its cost model, such as the number of additions or the depth of a reduction tree. It is not a performance figure: a tree of minimal depth says nothing about wall-clock time on a given machine, which still needs a **Measured** entry in `docs/BENCHMARKS_2026-10.md`.

## Results

All results are **Proved (model)**. The RepOps differential fixture is `crates/kadmium-runtime/tests/fixtures/repops_tree_vectors.txt`, checked by `crates/kadmium-runtime/tests/repops_lean_vectors.rs`.

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Kadmium.nextPow2_spec` | `2 ^ ceilLog2 n` is a power of two, at least `n`, and no larger than any other power of two that is at least `n`. | `usize::next_power_of_two`, as used by the reduction |
| `RepOps.treeSum_eq_eval`, `RepOps.canonical_leaves` | `tree_sum` evaluates one fixed tree that reads operands `0 .. n` once each, in index order, never past the end of the slice. | `repops::tree_sum`, `tree_node` |
| `RepOps.canonical_optimal` | That tree performs `n - 1` additions and has depth `ceilLog2 n`. Every binary reduction tree over the same `n` operands, in any order, performs `n - 1` additions and has depth at least `ceilLog2 n`. A fixed reduction order costs nothing in work or critical path. | `repops::tree_sum` |
| `RepOps.reducer_eq_treeSum` | Pushing the leaves in any order, then calling `finish`, returns exactly `tree_sum` of the same values. No push fails and the root is always formed. | `repops::TreeReducer::{new, push, push_chunk, finish}` |
| `RepOps.push_pastEnd`, `RepOps.push_twice`, `RepOps.finish_incomplete` | A leaf past the end, a leaf pushed twice and an early `finish` are rejected. | `repops::TreeReducer` |
| `RepOps.legacySum_eq_treeSum` | The legacy Verde reduction builds the same tree as `tree_sum`, so the two agree bit for bit. | `VerdeVerificationEngine::repops_pairwise_tree_sum` |
| `RepOps.legacy_fallback_unreachable` | The legacy reduction's `if mid >= n { n / 2 }` branch is never taken for any `n ≥ 2`. | `VerdeVerificationEngine::repops_pairwise_tree_sum` |

### Transport chunk maps

**Proved (model).** The differential fixture is `crates/kadmium-data/tests/fixtures/transport_chunk_vectors.txt`, checked by `crates/kadmium-data/tests/transport_lean_vectors.rs` against both Rust functions, including short reads. Modelling choices are listed in `KadmiumFormal/Transport/ChunkMap.lean`: byte counts are unbounded `Nat`, digests are not modelled, and a read returns 0 only at end of file.

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Transport.readLoop_eq_chunkPlan` | The sender's read loop produces exactly `chunkPlan`, so every chunk map the sender builds passes the receiver's chunk check. | `checkpoint::transport::hash_stream`, `chunk_plan` |
| `Transport.validChunks_iff` | The receiver's chunk check accepts exactly one map per file: `chunkPlan`. | `TransportManifest::validate`, `chunk_plan` |
| `Transport.chunkPlan_tiles` | That map starts at offset 0, each chunk starts where the previous one ends, every chunk holds 1 to `chunk_bytes` bytes, and the chunks cover the file exactly once. An empty file has no chunks. | `chunk_plan` |
| `Transport.chunkPlan_aligned` | If an alignment divides the chunk size, every chunk offset is aligned to it. | `chunk_plan` |
| `Transport.layout_divides` | 4 KiB divides the 64 KiB KTB page, which divides the 16 MiB chunk. | `TRANSPORT_ALIGNMENT_BYTES`, `TRANSPORT_CHUNK_BYTES` |

### Publishing a received package

**Proved (model)**, about a model only: there is no differential fixture. Renames are assumed atomic, a directory is abstracted to whether it holds every file of its own manifest intact, and verification succeeds exactly for such a directory. The unit test `checkpoint::transport::tests::begin_during_finish_cannot_publish_a_package_without_its_manifest` runs the legacy counterexample's interleaving against the Rust code. Modelling choices are listed in `KadmiumFormal/Transport/Publish.lean`.

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Transport.Publish.publish_safe` | Claiming staging into a private directory before verifying it means every published package verified as complete, under every interleaving of begins, chunk arrivals and concurrent finishes. | `CheckpointLibrary::{receive_begin, receive_finish}`, `finish_claimed` |
| `Transport.Publish.legacy_unsafe` | The previous verify-in-place-then-rename order publishes an incomplete package when a begin lands between the two. | `receive_finish` before PR #62's fix |

### Native decoder layer splits

**Proved (model)**, about a model only: there is no differential fixture. A layer is any function of the activations and its own KV cache, so the statements hold bit for bit for `f32` without a floating-point model. The Rust test `splitting_the_layers_at_every_cut_is_bit_identical` in `crates/kadmium-runtime/tests/native_decoder.rs` checks the property on four checkpoint layouts, every pair of cuts, through prefill and token-by-token decode. Modelling choices are listed in `KadmiumFormal/Native/LayerSplit.lean`.

| Theorem | Statement | Rust item |
| :--- | :--- | :--- |
| `Native.runLayers_append` | Running layers `fs ++ gs` with caches `cs ++ ds`, where `cs` has one cache per layer of `fs`, equals running `fs`, then `gs` on its activations, with the caches concatenated. | `native_decoder::DecoderSlice::forward` |
| `Native.runChain_eq_runLayers` | A chain of slices, each with one cache per layer, gives the same activations and caches as one slice holding all their layers. Where the layers are cut and how many machines hold them cannot change an output. | `DecoderSlice::forward` over a chain of slices |

Every RepOps theorem holds for an arbitrary addition `add : α → α → α` with no associativity, commutativity or identity assumed. IEEE `f32` and `f64` addition has none of those laws, so the statements hold bit for bit for floats without a floating-point model: two reductions that evaluate the same tree perform the same additions on the same operands.

Modelling assumptions shared by these results:

- Indices are unbounded natural numbers. Rust `usize` arithmetic in `next_power_of_two` and `2 * capacity` overflows only past `2^63` leaves.
- The reducer's `Vec<Option<T>>` is a function from index to `Option`. The proofs track that every index the algorithm touches is below `2 * capacity`.
- Error strings are modelled as the constructors of `ReduceError`.
- `Summand::add` is a function of its two operands, which holds for `f32`, `f64` and `i64`.

## Findings

- Kadmium has two implementations of the RepOps reduction: `repops::tree_sum` and the legacy `VerdeVerificationEngine::repops_pairwise_tree_sum`, which `kernels.rs`, `live_worker.rs` and the Verde engine still call. `legacySum_eq_treeSum` shows they agree on every input, so they can be consolidated without changing any digest. Its `n / 2` fallback is dead code.

## Building

The toolchain is pinned in `lean-toolchain`; `elan` installs it on first use. There are no Lake dependencies yet.

```bash
cd formal
lake build --wfail
lake exe repops_vectors ../crates/kadmium-runtime/tests/fixtures/repops_tree_vectors.txt
lake exe transport_vectors ../crates/kadmium-data/tests/fixtures/transport_chunk_vectors.txt
cd ..
cargo test --locked -p kadmium-runtime --test repops_lean_vectors
cargo test --locked -p kadmium-data --test transport_lean_vectors
```

`lake build --wfail` checks every proof and turns warnings, including `declaration uses 'sorry'`, into errors. `KadmiumFormal/Audit.lean` fails the build if a listed theorem depends on any axiom other than `propext`, `Classical.choice` and `Quot.sound`, which rules out `sorry`, `native_decide` and project-local axioms. CI runs both steps and fails if the regenerated fixture differs from the committed one.

## Adding a result

1. Model the Rust item in `KadmiumFormal/<Area>/`. Name the Rust file and function in the module docstring and list every modelling choice.
2. State the theorem for the general case, such as every `add` or every input length, not for sample inputs.
3. For an optimality claim, state the cost model and prove a lower bound over all alternatives, not only a bound for the chosen algorithm.
4. Add the theorem to `KadmiumFormal/Audit.lean`.
5. Add a fixture generator and a Rust test that reproduces it.
6. Add a row to the results table, and cite the result elsewhere only with its label.
7. If a later Rust change alters the behaviour of a covered item, update the model and the proof in the same PR.
