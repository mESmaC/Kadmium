# Use distributed compute

For scheduler admission details and design gaps, see
[Local aggregate compute](local-compute-fabric.md). The canonical
[lease/data-plane/adapter target design](../ARCHITECTURE.md#921-local-aggregate-compute-target)
does not change the supported live workloads described here.

Condenser's compute dock (chip icon beside the dock tabs) connects to a Kadmium coordinator and runs bounded jobs on live TCP workers. Model inference through Ollama and Kadmium grid work are separate routes: connecting workers does not distribute a loaded Ollama model or train an imported checkpoint.

## Connect your machines

On the coordinator:

```sh
kadmium serve --live
```

On each additional machine, use the coordinator's reachable LAN address and a unique hostname:

```sh
kadmium join --coordinator http://192.168.1.48:7848 --hostname worker-2 --live-only
```

`192.168.1.48` is an example. Workers must reach the coordinator's HTTP port 7848, and the coordinator must reach each worker's TCP port 7850. The coordinator also starts its own worker. Keep these listeners on a trusted network; this is not authenticated public compute hosting. See [installation and network setup](../getting-started.md) for discovery and firewall details.

In Condenser, enter the coordinator URL and choose **Inspect compute**. Registered online workers are distinguished from planned demonstration hardware. Placement remains a modelled estimate, not loaded weights. Resource offers (`--offer-fast-gb`, `--offer-ram-gb`, `--offer-nvme-gb`, `--slots`) and polite availability constrain the existing scheduler.

The new job endpoint uses only online `TrustedCore` workers that do not require separate plan approval, and requires at least two. Workers started with `--require-approval` are excluded; this endpoint does not grant or bypass that approval. Update workers to the same build to enable the matrix executor. A disconnected or incompatible worker fails the job instead of executing its work on the coordinator as a simulated node.

## Run a job

Choose a workload, edit its JSON inputs, then choose **Run on live workers**. The report includes the graph ID, actual worker IDs, TCP dispatch traces, validated units, measured wall time, result and errors. `distributed` says whether successful work actually ran on more than one worker. No request is automatically retried.

| Workload | What executes |
| --- | --- |
| FP32 matrix multiplication | Row partitions of your supplied matrices execute on distinct workers' CPU kernels. Every dot product uses the RepOps product/reduction order. Inputs and outputs retain FP32; no Int8 transport, FP16 NPU substitution or quantization is used. |
| Training stage preview | Existing synthetic per-stage forward/backward work plus gradient reduction. Stages reset on each request, use reference Int8 boundaries, and do not update a persistent model. |
| Dream-RSI replay | Existing bundled in-memory discovery-tree workload. It does not generate model rollouts or update weights. |

For matrix multiplication, dimensions are at most 256, and distributed jobs require at least two rows. Values must be finite FP32 numbers; overflow is an error. This bounded CPU workload establishes usable distributed numerical computation, not GPU performance or a general arbitrary-code service. Results use the submitted row order even when workers finish out of order.

Send the same job directly to `POST /api/v1/compute/jobs` on the coordinator:

```json
{
  "kind": "matrix_multiply",
  "matrix": {
    "rows": 2,
    "inner": 3,
    "columns": 2,
    "left": [1, 2, 3, 4, 5, 6],
    "right": [7, 8, 9, 10, 11, 12]
  }
}
```

The row-major result is `[58, 64, 139, 154]`. Other request shapes are:

```json
{"kind":"training_preview","step_index":1,"learning_rate":0.01}
```

```json
{"kind":"replay","rollouts":8,"simulations":32}
```

Learning rate must be in `(0, 1]`; replay accepts 2–64 rollouts and 8–256 simulations. Unknown job kinds and fields fail validation. Jobs have a 60-second coordinator deadline and 15-second per-call deadlines. Timeout does not undo already-dispatched work. Inspect workers before retrying. The job route produces no monetary credits or paid-compute claims. Condenser proxies it at `POST /api/v1/compute/jobs` with `{base_url, api_key?, job}`; coordinator credentials remain request-scoped and redirects are not followed.

## Native model shard staging: library integration

This is the first library-level increment for staging a native decoder's weights. It has no
worker-socket adapter, CLI command, HTTP route or interface control yet. It does not turn the
matrix-job API into distributed model inference.

The entry points are `native_decoder::staging::PreparedSlice` and `SliceDestination` in
`kadmium-runtime`. Use a dedicated coordinator slice-package library and a separate destination
library for each worker. Keep the original full checkpoint. Supply a `SlicePlan` explicitly;
automatic layer assignment and execution-memory reservations are not part of this increment.

The library workflow is:

1. Call `PreparedSlice::prepare` with the original checkpoint ID and requested layer range.
2. Register the prepared source with `register_source` in the existing `ReplicaCatalog`.
3. Inspect `staging_plan` with the destination's hardware record and `ResourceOffer`. When it
   awaits approval, obtain a `PlanApproval` for the original model ID, destination node and
   `placement_digest`. A slice is not an inference allocation.
4. Call `stage_to` with exclusive access to that catalog and destination package. This is a
   synchronous library-to-library adapter, not an authenticated remote worker call. Serialize
   all receives into that destination package. Use the worker's disk offer, not its VRAM offer.
   If the catalog selects another source replica, this local adapter returns an error rather
   than substituting its own library for that source.

Cancellation is checked between raw-tensor and transport-chunk copies and around package
import. Existing integrity scans and normalization finish their current call before the
adapter observes cancellation.

`PreparedSlice::storage_bytes()` is the disk charge: transported payload files plus the serialized
`transport.json` and chunk-receipt payloads. `transport().total_bytes` is only the transported
payload, not the storage reservation. These are logical file lengths; filesystem allocation
blocks and directory overhead are not included.

Cancelled transfers, source-read failures and rejected chunks retain the verified chunks and
the full pending `StagingJob` reservation in `ReplicaCatalog`. Reuse that catalogue when resuming;
pending storage is not a verified replica or a source. To abandon a transfer, first delete or
reconcile its partial files, then call `release_staging`. Losing the catalogue requires reconciling
the worker library before new placements; this adapter does not recover reservations from disk.
The logical-byte gate, resume charge and pending-to-resident accounting are **Proved (model)**
by `Kadmium.Grid.StagingStorage.gate_respects_logical_offer`,
`Kadmium.Grid.StagingStorage.resume_counts_once` and
`Kadmium.Grid.StagingStorage.publish_counts_once` at `3ddc52ab01a5244b9117f22e9ef36ea2ce934b53`,
checked with `cd formal && lake build --wfail`. See the
[model assumptions and differential tests](../../formal/README.md#resumable-staging-storage-gridstagingstoragelean).

The selection rule is **Proved (model)** as `Kadmium.Native.Shard.selected_iff` and
`Kadmium.Native.Shard.payload_preserved` at `8c99026edeb99a53ed90427327985b61246d54e5`,
checked with `cd formal && lake build --wfail`; see the
[proof scope and Rust linkage](../../formal/README.md#native-decoder-shard-selection-nativeshardselectionlean).
Those theorems are about lists of tensor parts and unchanged payloads, not Rust extraction,
I/O, approval gates or hash security.

**Measured** reference checks for this increment run with:

```sh
cargo test --locked -p kadmium-runtime --test native_staging --test native_staging_lean_vectors
```

The checks exercise F32, F16 and BF16 copies, tied output heads, metadata preservation,
bitwise logits after loading a staged chain, exact-plan approvals, disk-offer boundaries,
draining and restricted-trust workers, corrupt chunks, resume, cancellation and stale catalog
entries. They use temporary libraries on one machine, not a multi-machine benchmark. Chunk
verification and publication use the existing checkpoint transport. Context, sampling,
generation settings and original source weights are not changed by shard selection.

Selected packages still carry the full source configuration and a temporary import name.
They are not standalone models; use them only through the explicit slice-loading API. A clear
slice-library name and marker remain follow-up work before exposing these packages in an interface.

Remaining integration work: connect these packages to authenticated worker sessions; reserve
execution memory for actual weights and the full-context KV cache; assign layers; and run the
cancellable token-generation chain. Full distributed inference is not available through these
entry points alone.

## Synchronize real training replicas with DiLoCo

For persistent training, use the existing stateful DiLoCo session API with your local learner. The coordinator synchronizes complete replicas; each island owns its optimizer, dataset and RNG state. This is a bounded FP32 parameter service for small models/adapters, not an integrated large-language-model trainer.

Create one session by posting this body to `/api/v1/training/diloco/sessions`:

```json
{
  "session_id": "lab-regression",
  "config": {"inner_steps":20,"outer_learning_rate":0.7,"outer_momentum":0.9},
  "island_ids": ["workstation", "laptop"],
  "shards": [{"shard_id":"linear","values":[0.0,0.0]}]
}
```

Each island then performs this loop:

1. Read `/api/v1/training/diloco/sessions/lab-regression/snapshot` and load **all** returned shards into its local replica. Retain `round` and `baseline_id`.
2. Run exactly 20 local optimizer steps on that island's data. Keep local optimizer and RNG state locally.
3. Post the final local parameter values to `/api/v1/training/diloco/sessions/lab-regression/updates`, using `{island_id, round, baseline_id, local_steps:20, shards}`. Supply final values, not gradients or deltas.
4. Wait for both islands to submit. The last accepted submission commits the shared outer update. Fetch the new snapshot before starting the next round. A missing island holds the barrier; stale or duplicate updates are rejected.

For a runnable, single-process learner using this contract:

```sh
kadmium diloco --inner-steps 20 --rounds 25 --islands 3 --json
```

That command runs persistent local AdamW learners on separate data ranges and a shared DiLoCo outer optimizer. It is an example to adapt into your island process, not a launcher for remote learners. Download the session's `/checkpoint` JSON to retain master parameters, outer momentum and pending submissions; restore with `POST /api/v1/training/diloco/restore`. Save local optimizer, data position and RNG state separately. Sessions are in memory until exported and do not replace the active inference checkpoint.

The resident budget is `parameters_per_island × (islands + 2) <= 262144` scalars. Each island has the same complete named parameter layout; assigning different layers to different islands is not supported by this API. See [the session API contract](../api-reference.md#stateful-diloco-sessions) and [training mechanisms](training-and-dream-rsi.md) for exact limits and the distinction from pipeline-sharded target designs.
