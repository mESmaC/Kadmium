# Kadmium Interdevice Protocol (KIP/1)

KIP is a node-local byte-transfer protocol in `kadmium_data::interdevice`. It selects a
supported route using that system's directional link costs and runs the transfer with
bounded queues, staging memory, and completion fences. It is separate from Kadmium's
TCP worker protocol: network peers cannot supply process pointers, file paths, or DMA
addresses through KIP.

**Implemented:** a versioned request and topology model, cost-based route planning,
demand/prefetch admission, conflict detection, a bounded executor, and a portable backend
that actually copies RAM and regular-file bytes. The example below runs that backend.
Files can reside on NVMe, SATA, or another filesystem; the adapter does not claim to
detect the physical medium. Filesystem caching remains active.

**Adapter contract, not implemented native acceleration:** CUDA/ROCm/Metal DMA,
GPU peer copies, pinned allocations, GPUDirect Storage, and OS asynchronous direct I/O.
These paths can be advertised by an adapter only when it implements and validates them.
Tests use delayed fake devices to verify protocol ordering; they are not GPU execution
or performance evidence. Existing inference, expert caches and checkpoint readers keep
their current paths until explicitly wired to an adapter. KIP does not change tensor
formats, quantization, model placement, cache admission, or kernel arithmetic.

## One system, explicit capabilities

Each `Backend` supplies an immutable `Topology` snapshot with protocol version `1`, a
system/adapter identity, directional `Link` records, and allocatable RAM endpoints.
Use `Ram(n)` for NUMA nodes, `Vram(n)` for devices and `Storage(n)` for volumes. Unified
memory can be exposed as RAM when compute and storage really share the same allocation;
a shared-memory marketing label does not imply that arbitrary regions alias.

A link declares its method (`Buffered`, `Dma`, `Peer`, or `DirectStorage`), bandwidth,
fixed latency, maximum operation size, and power-of-two offset/length alignment.
The adapter must validate resource eligibility too: e.g. DMA address alignment,
registration, pinning, peer access, and the file/device pairing for direct storage.
Unsupported capabilities are absent, not guessed from silicon names or OS versions.
For a GPU without a peer link, advertise device-to-RAM and RAM-to-device links to permit
bouncing through RAM. For NUMA machines, give the local and remote RAM links their
measured costs. Link directions may have different rates and alignment requirements.

The planner enumerates direct links and two-leg routes through allocatable RAM. For each
compatible route, it rounds the chunk size down to the strongest alignment and estimates:

`sum_over_legs(chunks * latency_ns + bytes / bytes_per_second * 1e9)`

The lowest estimated cost wins, with stable topology order breaking ties. Small
transfers can favor low setup latency; large transfers can favor bandwidth. Offsets and
the final length must satisfy every leg. An unaligned direct-storage tail selects a
compatible buffered route for the request; it is never padded into adjacent tensor
data. No compatible route returns `Unsupported` before allocation or copying. There is
no silent switch to a different method after a copy has partially executed.

Costs are caller-supplied measurements or explicitly labelled priors. Bind persisted
measurements to the system, driver/adapter, devices and registration mode that produced
them, and construct a new session when those change. The protocol does not benchmark
hardware automatically or claim its additive estimate captures concurrent contention.

## Request and resource lifetime

`Request` is serde-serializable and contains version, source/destination region IDs and
byte offsets, byte count, and `Demand` or `Prefetch` priority. It contains no address or
path. An adapter registers resources locally and returns `Region` metadata. Region IDs
are session-local, never recycled, and meaningless in another session. This is an
in-process contract, not a remote authentication or authorization boundary.

The executor checks the protocol version, registration, writable destination, overflow,
and both bounds before admission. Empty transfers complete without I/O. Overlapping
in-place transfers are rejected; disjoint regions of the same allocation are allowed.
In-flight or queued read/write and write/write overlap returns `WouldBlock`; multiple
readers can share a source. Submit dependent work after receiving completion. Separately
registered aliases of the same writable backing storage are prohibited; adapters must
canonicalize them or their caller must provide one shared registration.

The lifecycle is:

1. Register resources and retain their leases through terminal completion. Keep external
   writers away from leased ranges. Enqueue the validated request and receive a session
   transfer ID. Queue limits return `WouldBlock` for caller backpressure, and prefetch
   cannot consume the final queue slot reserved for demand.
2. Admit demand before prefetch. Each active staged transfer receives at most
   `staging_bytes / max_active` bytes, rounded down for alignment. Prefetch cannot use
   all active slots or the demand share of staging memory. A configuration too small
   for even one aligned staging chunk rejects that route.
3. Start a direct chunk, or start source-to-staging and then staging-to-destination.
   The second leg starts only after the first fence signals readiness. Staging is reused
   only after the final leg's fence completes. Requests can overlap on asynchronous
   backends; chunks within one request are ordered. There is no single-request double
   buffering or coalescing in this version.
4. Poll with `progress()` from the worker/event loop. One call polls each active fence
   once and submits at most one operation per active request. Native backends can run
   operations concurrently. The portable backend completes buffered operations inside
   `start`, so that call can block on file I/O.
5. Consume `Completion`. A failed or cancelled destination can contain completed chunks
   and a partial final chunk; never publish it as a valid expert, tensor or checkpoint.
   Retry into a private destination and publish only after complete success and any
   required content checksum. The protocol copies bytes; it does not add a checksum.

`Fence::poll` readiness means destination visibility to subsequent operations/compute.
Every fence owns the source, destination, device/stream and pinning leases. Its destructor
must synchronously drain or cancel before releasing them, even after a poll error.
`Backend::start` must leave no unowned work on error. Staging allocations must meet the
advertised native alignment/registration requirements; a plain `Vec` is not pinned RAM.
Adapters must retain resources if the driver cannot prove DMA has stopped; premature
release is never an acceptable failure fallback.

`cancel(id)` and dropping the engine drain fences before freeing staging. Cancellation
can therefore block on a device, does not roll back bytes, and is not a hard deadline.
Already terminal IDs cannot affect newer requests. Completion events are returned to
the caller rather than accumulated in an unbounded history.

The staging budget includes every active intermediate allocation until its last fence
is drained. It excludes caller-owned source/destination buffers, filesystem caches and
adapter metadata. Obtain that budget from the worker's existing host memory ledger;
KIP does not reserve additional host RAM behind the ledger's back. Kernel/compute leases
are the adapter/cache owner's responsibility, not inferred from copy completion.

## Using the portable backend

```sh
cargo run --locked -p kadmium-data --example interdevice_copy -- source.ktb new-copy.ktb
```

The destination must not exist. This copies a real file through at most 4 MiB of RAM for
the single request under the default 16 MiB/four-slot budget, handles the final short
chunk, and calls `sync_all` before reporting success. The printed elapsed time measures
that particular buffered copy including the final sync. It is not an NVMe benchmark.
An error leaves the newly created destination incomplete; it is never automatically
published or renamed over another checkpoint.

Library callers construct `HostBackend::new(system, HostCosts { ... })`, register RAM
and already-open regular files, then create `TransferEngine::new(backend, limits)`.
Files are never opened by an untrusted path in a request or resized by the adapter.
`HostRegion` retains a registration, exposes a RAM snapshot with `read_bytes`, and offers
explicit `sync_all` for file durability. Storage-copy completion alone only promises
visibility. Dropping a registration invalidates it; later use returns `NotFound`.

For expert loading, use the existing `ExpertCache` lookups to exclude resident hits and
Kadmium's `batch_read::plan_reads` to coalesce source ranges before creating transfers.
Retain coalesced buffers until all scatter/upload fences finish. Admit/fill the target
cache only on complete success; cancel obsolete speculative requests by their transfer
IDs. KIP performs no speculative eviction and cannot change which experts compute.

`TransferStats` reports complete/failed/cancelled requests, successful logical bytes,
successful physical bytes across every hop, operation count and current/peak staging
bytes. Two-hop copies account for twice the logical traffic. Partial failed physical
writes are not measured. These counters expose work and memory use without inventing a
throughput improvement.

## Reference and validation

The design was informed by the InferSSD reference architecture, especially:

- `engine/ggml/src/ggml-ssd/io/submit-window.h`: bounded submission and the rule that
  read completion does not release an H2D source slot.
- `engine/ggml/src/ggml-ssd/stream/copy-plan.h`: explicit source spans and checked sizes.
- `engine/ggml/src/ggml-cpu/ssd/cpu-sink-vram.inc`: directional placement costs and
  retained RAM leases for deferred GPU uploads.
- `engine/ggml/src/ggml-cuda/ssd/README.md` and
  `engine/ggml/src/ggml-ssd/gpu/kimi-vram-tier-api.h`: readiness, cancellation/drain and
  device ownership contracts.

KIP is an independent Rust implementation; no InferSSD C++ code or native driver is
vendored. Existing Kadmium coalescing and cache policy remain reusable layers above it.

Run `cargo test --locked -p kadmium-data interdevice` for directional route selection,
alignment fallback, bounds/version validation, demand priority, budget accounting,
delayed DMA ordering, cancellation/drop/start/poll/allocation failure cleanup, RAM
copies, real file round trips and truncated-file errors. Native adapters additionally
need hardware tests for visibility, pinning, numerical integration and failure drain.
