# Local aggregate compute: design proposal and admission guide

Review date: 2026-10-31. **Design proposal**, except for the coordinator scratch
admission described below. The canonical target design and acceptance gates live
in [Architecture section 9.21](../ARCHITECTURE.md#921-local-aggregate-compute-target).
This guide does not define another protocol, allocator or roadmap.

## Patterns from scientific grids

BOINC's locality scheduling favors resident inputs after resource feasibility;
its memory management considers competing tasks. CERN WLCG separates data
management from transfer execution, with site-controlled FTS transfers. HTCondor
provides direct URL input transfers and pool-wide concurrency limits. These are
patterns to adapt, not substitutes for latency-sensitive model parallelism.

For a personal LAN, distinguish independent replay/evaluation/preprocessing
(bounded pull batches and restart), decode/pipeline stages (persistent push and
measured critical-path latency), and training replicas (coarse local optimizer
steps and explicit synchronization). Aggregate FLOPS alone do not predict useful
application throughput; no speedup measurement is claimed here.

## Implemented: coordinator scratch admission

`PlacementConstraints.scratch_fast_memory_gb` defaults to zero for old producers
and serialized units. **No shipped workload currently supplies a nonzero estimate.**
It reserves exclusive scratch per `Assigned` or `Running` `WorkInstance`, including
replicas and earlier push/pull batches. Returned attempts release scratch even
while quorum validation waits. Reservations rebuild from queue state rather than
a second mutable reservation table; heartbeat budgets are not rewritten.

```text
admission headroom = reported free fast memory - outstanding queue scratch
eligible when headroom >= max(min_fast_memory_gb, scratch_fast_memory_gb)
```

For example, a minimum of 4 GiB and scratch of 2 GiB requires 4 GiB headroom and
reserves 2 GiB, not 6 GiB. Field names retain the repository's `_gb` convention
for binary GiB. Comparison tolerates four floating-point epsilon units scaled
by the larger magnitude or 1 GiB, to avoid rejecting decimal exact fits. This is
roundoff tolerance, not an oversubscription policy.

`submit` rejects negative, NaN and infinite minimum/scratch values with
`GridError::InvalidMemoryRequirement`; graph submission checks all units before
mutating queue state or emitting events. Matchmaking also rejects malformed
requirements and nonfinite telemetry. Corrupt live scratch in restored state
blocks admission on that worker instead of increasing its budget.

`WorkQueue::rank_admission` exposes ranked `MatchScore` diagnostics, including
outstanding reservations and an ineligibility reason, without assigning work.
It reports resource eligibility, not graph readiness or distinct-replica rules.
Operators must call it; there is no automatic UI warning or starvation prevention.

### Limits before workload integration

- The caller must report headroom **after resident `PlacementFit` allocations**,
  but before these same queue reservations. This precondition is not enforced:
  today's offer-based worker budget does not automatically subtract residency.
  Do not enable scratch producers until the shared authority in
  [issue #60](https://github.com/mESmaC/Kadmium/issues/60) is integrated.
- All coordinating callers must use one queue. This is an admission estimate,
  not a worker allocator, device-memory ticket or physical reclamation guarantee.
- Cancellation, expiry and failure release queue reservations although the remote
  kernel may still run. Only worker completion/teardown acknowledgement can make
  reuse safe. Do not use scheduler state alone to free DMA-visible memory.
- Conversely, abandoned `Assigned`/`Running` attempts can reserve indefinitely
  while a worker stays online, including originals whose duplicate validated.
  There is no safe coordinator-only grace release; worker lease reconciliation
  must fence execution before reclaiming its ticket.
- Small units can starve a large high-priority unit by repeatedly using available
  headroom. Diagnostics expose the shortfall, but aging/headroom protection is
  still target work. Each dispatch scans historical instances and clones worker
  profiles, including zero-scratch queues; no dispatch-throughput gain is claimed.

## Code evidence and design ownership

Static observations, not measured throughput or a comprehensive security audit:

| Area | Current path and remaining gap | Canonical design |
| --- | --- | --- |
| Retry | `PeerChannelPool::call_with_connect_timeout` can reconnect and resend after unhealthy-channel errors; request IDs are not durable deduplication keys. Transport cancellation fixes are upstream, not part of this change. | [Transport section 9.3](../ARCHITECTURE.md#93-transport), [local target](../ARCHITECTURE.md#921-local-aggregate-compute-target) |
| Fan-out | `GridDispatcher::dispatch_ready` clones prerequisite bytes into dispatches; coordinator copies can multiply. | Existing `ReplicaCatalog` / `StagingJob`, [FTS3 section 6.2](../ARCHITECTURE.md#62-third-party-transfer-scheduler-cern-fts3-pattern) |
| Resource spending | `HostMemoryLedger`, offers and placement exist, but queue scratch does not enforce physical memory ownership. | [Allocation section 9.5](../ARCHITECTURE.md#95-allocation-measured-learned-and-approved), issue #60 |
| Device I/O | KIP/1 has a host backend, bounded windows and adapter lifetime rules; it is not wired into execution paths and GPU adapters are target only. | [KIP/1 section 9.20](../ARCHITECTURE.md#920-kip1-interdevice-transfer-protocol-kadmium-datainterdevice), [adapter contract](interdevice-protocol.md) |
| Cost/isolation | Application-level cost samples do not model shape/concurrency contention; live listeners remain trusted-network only. | [Local target](../ARCHITECTURE.md#921-local-aggregate-compute-target) |

## Reproducible references

Primary design references:

- [BOINC locality scheduling](https://boinc.berkeley.edu/boinc_papers/locality/text.php)
  and [memory management](https://github.com/BOINC/boinc/wiki/Memory-management).
- [CERN WLCG FTS service](https://wlcg-ops.web.cern.ch/fts).
- [HTCondor file transfer](https://htcondor.readthedocs.io/en/main/users-manual/file-transfer.html)
  and [concurrency limits](https://github.com/htcondor/htcondor/blob/main/docs/admin-manual/cm-configuration.rst).

Pinned source snapshots inspected read-only (not build dependencies):

- [BOINC scheduler, bc8144cb](https://github.com/BOINC/boinc/blob/bc8144cbcb0a03e5b44354984ae0423f9efe2540/sched/sched_send.cpp): host resource feasibility.
- [FTS3, f19df43c](https://github.com/cern-fts/fts3/blob/f19df43c7db0764e2b6be30916ec406be15a1574/README.md): independent transfer service and site controls.
- [DeepEP, f6aa38fe](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/README.md): specialized device collectives, not a heterogeneous-LAN replacement.
- [InferSSD, b9cc72d6](https://github.com/mESmaC/inferssd/blob/b9cc72d6ec26944e628dceb0e8bbb4e610900052/docs/architecture.md): host admission and cache/I/O ownership.
- [T3 contracts, c6ef3b46](https://github.com/pingdotgg/t3code/blob/c6ef3b46ebed34fbefa1706674fe5748254cee18/packages/contracts/src/orchestrationV2.ts): command identity and sequenced lifecycle, not tensor transport.

No external project code, performance figures or licensing assumptions were imported.
