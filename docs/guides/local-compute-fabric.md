# Local aggregate compute: review and implementation plan

Review date: 2026-10-31. This document distinguishes implemented admission/transport
changes from proposed protocol and driver work. It does not claim that LAN machines
share coherent memory, that aggregate FLOPS predict application speed, or that the
current reference executors implement arbitrary distributed model training.

## What to borrow from scientific grids

BOINC's locality scheduling selects feasible work and prefers already-resident
inputs. Its host memory management accounts for competing tasks, not just one task
in isolation. CERN's WLCG separates data management from bulk transfer execution;
FTS provides reliable transfers with site-controlled network usage. HTCondor can
move inputs directly from a URL to the execution point and offers pool-wide
concurrency limits. These are useful patterns, not interchangeable implementations
of latency-sensitive model parallelism. See the primary references below.

For a personal LAN, use three execution classes:

- Independent replay, evaluation and preprocessing: bounded pull batches, locality,
  checkpoint/restart and modest control-plane overhead.
- Decode and pipeline stages: persistent push channels, measured critical-path
  latency, explicit activation ownership and small bounded queues.
- Training replicas: coarse local optimizer steps and explicit synchronization
  rounds, rather than treating every heterogeneous LAN as a GPU collective fabric.

## Evidence and gaps in Kadmium

| Area | Observed implementation | Gap and next action |
| --- | --- | --- |
| Admission | `kadmium-grid/src/scheduler.rs` checks slots and a per-unit minimum against a worker's free fast memory. | Several individually fitting units can exceed the same budget. This change adds opt-in exclusive scratch reservations, described below. |
| RPC cancellation | `kadmium-network/src/peer_channel.rs` demultiplexes replies and times out calls. Previously, dropping a call could retain its pending entry; only delta writes had a cancellation guard. | This change removes pending entries on every exit and poisons both plain and delta channels when a write is abandoned. |
| Retry semantics | `PeerChannelPool::call_with_connect_timeout` reconnects and resends once after an unhealthy-channel error. | Delivery may already have occurred. A transport request ID is not a durable execution deduplication key. Introduce worker attempt leases and replayable receipts before using this route for non-idempotent mutations. |
| Data path | `GridDispatcher::dispatch_ready` clones prerequisite activation bytes into each outbound dispatch. Catalog and staging abstractions already exist. | Fan-out can multiply coordinator memory and outgoing bytes. Add immutable output handles, pinned lifetimes and bounded direct worker-to-worker transfers, retaining inline payloads for small messages. |
| Local memory | `HostMemoryLedger`, `ExpertCache` and delta-store admission already manage parts of the host envelope. | Coordinator scratch admission is not an allocator or device-memory reservation. Extend the existing ledger with execution-scoped leases and distinguish host RAM, pinned staging, resident weights and device scratch. Do not create a competing memory manager. |
| Lifecycle | `WorkQueue` has retries, deadlines, cancellation, redundancy and serialized state. | Queue state does not prove a remote kernel stopped. Reconcile cancellation/expiry against worker acknowledgements before reclaiming physical resources or replaying side effects. Persist accepted attempts and fence old coordinator epochs. |
| Capacity modelling | `WorkerProfile` already learns per-application compute cost and hop cost; residency-first routing is configurable. | An application label and in-flight count do not capture tensor shape, kernel concurrency or transfer contention. Bucket measured costs by backend/shape/precision and measure queue wait separately. Retain conservative cold-start fallback. |
| Isolation | Live compute documentation explicitly restricts listeners to trusted networks. | Before accepting untrusted peers: authenticated identities, negotiated capabilities, bounded messages, per-peer quotas, artifact integrity and executor allowlists. These changes do not make public compute safe. |

Paths in this table are relative to `crates/`. These are static code observations,
not measured throughput results or a comprehensive security assessment.

## Implemented: exclusive scratch admission

`PlacementConstraints.scratch_fast_memory_gb` is additional exclusive memory for
each assigned/running instance. It defaults to zero on older serialized work units.
`min_fast_memory_gb` remains a feasibility floor, not a second allocation. Shared
resident weights must not be copied into each instance's scratch requirement.

For each worker, push and pull admission use:

```text
available = worker.free_fast_memory_gb - sum(live instance scratch on this worker)
eligible = available >= max(min_fast_memory_gb, scratch_fast_memory_gb)
```

The scheduler derives reservations from the queue's instance states, including
replicas and assignments made by earlier push or pull calls. It decrements an
admission snapshot within each batch without rewriting the worker's reported
budget. A serialized queue reconstructs reservations without a second mutable
reservation table. Negative/non-finite requirements and invalid memory telemetry
are ineligible. A failed/cancelled/returned attempt no longer reserves queue memory.

Example work-unit placement fragment:

```json
{"min_fast_memory_gb": 4.0, "scratch_fast_memory_gb": 2.0}
```

This requires at least 4 GiB of remaining admission headroom and reserves 2 GiB
while the instance is live. It does not reserve 6 GiB. `GiB` is binary memory even
though field names retain the repository's `_gb` convention.

Limits: producers must opt in with credible scratch estimates; existing executors
are not automatically profiled by this patch. All coordinating callers must share
the same queue for this accounting to be meaningful. The worker budget must exclude
resident allocations but not already subtract these same queue reservations. This
is coordinator admission only: remote cancellation/timeouts do not stop kernels,
and local state transitions do not certify that physical memory has been released.
Worker-enforced leases are the next prerequisite, not an implemented guarantee.

## Implemented: cancellation-safe persistent calls

A pending-reply guard removes a request's sender when its call future is dropped,
including cancellation while waiting for the writer or after sending. A second
guard makes an abandoned write fail the channel for plain frames as well as delta
frames. Reusing a partially written byte stream can corrupt subsequent messages;
delta encoding additionally advances per-peer chunk state.

Cancellation before acquiring the writer leaves the channel healthy. Cancellation
after a full write also leaves it healthy; a late KDM2 reply is ignored by request
ID without resolving another caller. Dropping a call is not a remote cancellation
protocol and does not make pool retries exactly-once. Legacy KDMF peers have no
request IDs; their existing serial fallback does not provide the same late-reply
isolation. Require negotiated request-ID support for the proposed lease protocol.

## Proposed internal communication contract

Extend the current framed transport rather than replace it with a second bespoke
socket stack. Proposed messages below are not new supported wire kinds yet:

1. **Hello/capabilities:** protocol version range, authenticated worker ID, boot ID,
   supported executor/precision versions, frame and outstanding-byte ceilings.
2. **Reserve:** durable `(coordinator_epoch, graph_id, unit_id, attempt_id)`, absolute
   resource vector, input digests and lease duration. Return either a lease token
   with admitted resources or `Busy` with bounded retry advice. Keep clocks local
   for expiry; exchange durations rather than trusting synchronized wall clocks.
3. **Execute:** lease token, immutable input/output handles and execution deadline.
   A repeated attempt key returns its existing state or receipt; mismatched payload
   under the same key is rejected. Persist the acceptance before side effects.
4. **Renew/status/cancel:** monotonic state sequence and boot/epoch fencing.
   Cancellation is acknowledged only after safe quiescence; an uninterruptible GPU
   kernel can report draining. Lost connections leave bounded leases, not permission
   to assume the operation never ran.
5. **Result/commit/release:** output checksums, execution version, measured queue,
   transfer and execution time, plus a durable receipt. Retain output handles until
   consumers acknowledge them or their leases expire. Fence stale results before
   advancing graph dependencies.

Use separate bounded budgets for control messages and bulk payloads so staging
cannot starve heartbeats, cancellation or result acknowledgements. Bound both call
count and outstanding bytes; advertise credits before accepting bulk work. Hash
immutable artifacts with a cryptographic digest; existing FNV labels are not a
malicious-peer integrity guarantee. Do not silently change existing content IDs.

## Driver plan: user space first

The first custom driver should be a Rust resource/transport adapter, not a kernel
module that promises remote VRAM. Reuse vendor-supported device APIs and the
existing CPU/CUDA/NPU executor boundaries. A LAN cannot turn independent memory
controllers into coherent local memory.

- **Host I/O adapter:** bounded asynchronous reads, explicit cancellation, aligned
  buffer pools and fallbacks to ordinary files. Prototype Windows IOCP/IoRing and
  Linux io_uring only after capability detection. Borrow InferSSD's separation of
  placement policy, host admission and I/O ownership rather than copying its code.
- **Device adapter:** execution-scoped device and pinned-host leases, event-backed
  completion, finite stream/queue counts and telemetry. Never free DMA-visible
  memory on RPC timeout; release it on completion or driver-confirmed teardown.
- **Transport adapter:** baseline TCP first; evaluate shared memory on one host and
  vendor-supported RDMA only on verified hardware. Keep the same lease/receipt
  contract and portable fallback. Do not infer RDMA capability from USB-C or USB4
  connector shape or add modelled link bandwidths as measured throughput.
- **Optional kernel component:** consider only after profiling identifies a missing
  user-space capability. Require a narrow ABI, signed installation, least-privilege
  device access, bounded mappings, suspend/unplug recovery and a separate threat
  review. No kernel module is implemented by this PR.

## Delivery order and acceptance gates

1. **Admission and channel hygiene (this change):** regression tests for mixed
   push/pull budgets, exact fit, independent workers, redundancy, returned replicas,
   expiry, state restore, invalid numbers and cancellation at three transport boundaries.
2. **Worker leases and durable attempts:** integrate with the existing host ledger;
   fault-test disconnect after acceptance, coordinator restart, duplicate execution,
   lease expiry during a kernel and cancellation before resource reclamation.
3. **Bounded data plane:** direct immutable output transfers, consumer reference
   tracking, per-link credits, resume/checksum verification and transfer concurrency
   limits. Test fan-out under a slow reader and memory pressure.
4. **Measured tuning and adapters:** compare serial, local-threaded and distributed
   execution on the same inputs. Record end-to-end p50/p95 latency, useful work/s,
   queue wait, bytes on every hop, peak RAM/VRAM, retry work and owner responsiveness.
   Sweep task size and concurrency; only enable an optimization where its measured
   benefit exceeds dispatch, staging and validation overhead. Record hardware,
   topology and provenance in the benchmark ledger before claiming a speedup.

## References and local review scope

Primary public references, consulted 2026-10-31:

- [BOINC locality scheduling paper](https://boinc.berkeley.edu/boinc_papers/locality/text.php).
- [BOINC memory management](https://github.com/BOINC/boinc/wiki/Memory-management).
- [CERN WLCG FTS service](https://wlcg-ops.web.cern.ch/fts).
- [HTCondor file transfer](https://htcondor.readthedocs.io/en/main/users-manual/file-transfer.html).
- [HTCondor concurrency limits](https://github.com/htcondor/htcondor/blob/main/docs/admin-manual/cm-configuration.rst).

Read-only local references supplied by the user:

- `C:/Users/engin/Desktop/harnesses/boinc/sched/sched_send.cpp`: host disk/resource feasibility.
- `C:/Users/engin/Desktop/harnesses/fts3/README.md`: independent transfer service and site controls.
- `C:/Users/engin/Desktop/harnesses/DeepEP/README.md`: specialized device collective and buffer lifecycle, not a portable heterogeneous-LAN replacement.
- `C:/Users/engin/Desktop/inferssd/docs/architecture.md`: shared host admission, placement feedback and bounded cache/I/O ownership.
- `C:/Users/engin/Desktop/t3/t3code/packages/contracts/src/orchestrationV2.ts`: typed command identity and sequenced lifecycle patterns, not a tensor transport.

These local paths record review provenance, not build dependencies. No external
project's code, hardware performance figures or licensing assumptions were imported.
