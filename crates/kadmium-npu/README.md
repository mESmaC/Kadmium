# kadmium-npu

Intel-first NPU kernels with a vendor-neutral Rust API. This first kernel computes
row-major `C[M,N] = A[M,K] * B[K,N]` with immutable weights compiled once and
reusable input/output allocations. `M = 1` also provides matrix-vector inference.
The runtime exposes the same API as `kadmium_runtime::npu`.

All new implementation code is Rust: tensor validation, graph generation, dynamic
driver bindings, allocation, submission and synchronization. The Intel backend
generates an OpenVINO IR graph in memory and sends it directly to the installed
Level Zero NPU driver. Intel's driver compiler generates device instructions.
There is no OpenVINO runtime dependency, Python, C/C++, bindgen, external model
file, build script, or downloaded kernel binary in this crate. The system driver
and firmware remain native vendor dependencies, as they are for CUDA.

## Use

```rust,no_run
use kadmium_npu::{MatMulKernel, MatMulShape, NpuBackend, intel::IntelNpu};

let npu = IntelNpu::open()?;
let shape = MatMulShape::new(2, 3, 2)?;
// B is K x N, not N x K.
let mut kernel = npu.prepare_matmul(shape, &[7., 8., 9., 10., 11., 12.])?;
let result = kernel.execute(&[1., 2., 3., 4., 5., 6.])?;
assert_eq!(result, [58., 64., 139., 154.]);
# Ok::<(), kadmium_npu::Error>(())
```

Intel execution needs a 64-bit little-endian Windows or Linux host, an Intel NPU,
the Level Zero loader (`ze_loader.dll` in Windows System32 or `libze_loader.so.1`
on Linux), and an NPU driver with a graph extension 1.x compiler supporting
`NGRAPH_LITE`. On Linux this also requires access to the accelerator device.
No SDK or hardware is required to build or run the ordinary workspace tests.
Other operating systems build successfully and return `Unavailable` from `open`.

```sh
cargo run --locked -p kadmium-npu --example intel_matmul
cargo test --locked -p kadmium-npu
cargo test --locked -p kadmium-npu --test intel_hardware -- --ignored
```

The example and opt-in tests require actual hardware and fail if it is absent;
they never silently run on the CPU. The CPU `reference_matmul` function is an
explicit validation oracle with an f64 accumulator.

## Contracts and limitations

- FP32 tensor storage, with `Arithmetic::ApproximateFp16`: the driver may use
  FP16 internal math, fusion and different reduction orders. There is no universal
  error bound or bit parity across devices. These results are **not RepOps digest
  evidence**. The existing exact dispatcher and grid verifier do not select NPU
  execution automatically.
- Shapes are nonzero, static and checked for overflow. Each tensor is capped at
  64 MiB. Values must be finite and within `[-65504, 65504]`; small values can
  underflow and large sums can overflow internally. Non-finite outputs are errors.
  A driver may reject otherwise valid shapes; compilation errors include its
  build log when graph extension 1.4 or later provides one.
- The compiler's reported I/O direction, precision, layout and dimensions must
  match contiguous FP32 `NC` buffers before the backend binds any memory. Argument
  indices are discovered, never assumed. Old metadata may pad unused dimensions
  with zeros or ones.
- Each prepared kernel owns its graph, command list, queue and host allocations.
  It retains the context and loader even after `IntelNpu` is dropped. Kernels are
  thread-confined and require mutable access to execute. Multiple independent
  kernels may coexist. Weights cannot change without preparing another kernel.
- Execution waits at most 30 seconds. A submission/wait failure permanently
  poisons the kernel. Destruction attempts one more bounded wait; if completion
  still cannot be established, it retains the native resources, loader and graph
  bytes until process exit to prevent device access to freed memory. This rare
  failure path intentionally leaks resources; drop may take another 30 seconds.
- This is a MatMul foundation, not a complete transformer backend. Quantized
  formats, attention, activation fusion, asynchronous scheduling, disk caching,
  and other vendors remain future kernels/backends. Implement `NpuBackend` and
  `MatMulKernel` for another vendor while retaining the validated shape and
  explicit arithmetic contract.

## Validation

Validated on 2026-10-30 on Windows, Intel AI Boost PCI device `0xad1d`, driver
`32.0.100.3717`, graph extension `1.9`, compiler `6.3`. Hardware tests cover scalar,
rectangular and non-tile-aligned shapes up through `16 x 256 x 128`, deterministic
random inputs, repeated changed inputs, zeros, invalid input recovery, independent
weights, and kernels outliving their backend. The bounded random fixtures are
checked against the f64 oracle using `abs(error) <= 0.01 + 0.01 * abs(reference)`;
this is a test acceptance criterion, not a general precision guarantee.

Host tests cover shape overflow, tensor validation, binary IR framing, ABI layout,
device metadata rejection, and injected submission/synchronization failures
(including resource quarantine). Linux driver execution has not been verified on
physical hardware. Normal CI exercises host tests on Windows, Linux and macOS.

ABI references: [Level Zero](https://github.com/oneapi-src/level-zero/blob/master/include/ze_api.h),
[Intel graph extension](https://github.com/intel/level-zero-npu-extensions/blob/main/ze_graph_ext.h),
[Intel IR framing](https://github.com/intel/linux-npu-driver/blob/main/validation/umd-test/graph_utilities.hpp).
