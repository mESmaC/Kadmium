# kadmium-cuda

CUDA kernels for Kadmium written in pure Rust and compiled to PTX by NVIDIA's
[cuda-oxide](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide). Each kernel produces
the same bits as Kadmium's RepOps reference math, so a CUDA worker's output can be checked
against a CPU, Metal or RISC-V worker by digest. The design is in
[ARCHITECTURE.md §9.17](../../docs/ARCHITECTURE.md).

| Path | What it is | Builds with |
| --- | --- | --- |
| `src/` | Lane math generic over `fp::Fp`, the warp-shaped RepOps tree, block formats, host references, launch geometry, catalog | `cargo` (workspace member, any OS, no GPU) |
| `device/` | The `#[kernel]` functions and the GPU parity harness | `cargo oxide` (Linux or WSL2, CUDA 13, pinned toolchain) |

## Kernels

| Kernel | Work split | Guarantee |
| --- | --- | --- |
| `q8k_quantize` | one warp per 32-element block | bit-identical to `Q8KBlockVector::quantize` |
| `q4k_q8k_matvec` | one warp per row, lane per block, `dp4a` + `prmt` | bit-identical to `Q4KBlockTensor::matvec_q8k` |
| `mxfp4_matvec` | one warp per row, lane per element | bit-identical to the RepOps dot of the dequantized row |
| `swiglu` | one thread per element | bit-identical to `MulMatIdExpertBank::swiglu_split` |
| `moe_top_k` | one warp per token, `k <= 32` | exact: NaN as `-f32::MAX`, ties to the lower id |
| `row_mask` | one thread per entry | exact |

"Bit-identical" counts every NaN as one value: IEEE 754 leaves a NaN's payload to the hardware
(`inf - inf` is `0xffc00000` on x86 and `0x7fffffff` on an NVIDIA GPU).

## Running the kernels

cuda-oxide supports Linux only; WSL2 with a current Windows NVIDIA driver works. You need:

- Rust 1.99.1 with `rust-src`, `rustc-dev` and `llvm-tools` (`device/rust-toolchain.toml` pins it)
- CUDA Toolkit 13.x (`nvcc`, libNVVM, nvJitLink, libdevice) and clang/libclang 21 for `bindgen`
- `cargo install --git https://github.com/NVIDIA/cuda-rust.git --rev 18a0407cc2 cargo-oxide`, then `cargo oxide doctor` (the revision matches the pinned device dependencies)

No root is needed if you install the toolkit into a micromamba environment
(`micromamba create -p ~/cuda-env -c nvidia -c conda-forge cuda-toolkit=13 clang=21 libclang=21`)
and export `CUDA_HOME=~/cuda-env`, `LIBCLANG_PATH=$CUDA_HOME/lib` and
`LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/nvvm/lib64:/usr/lib/wsl/lib`.

```sh
cd crates/kadmium-cuda/device
cargo oxide run              # parity: exits 1 if any kernel differs from the reference by a bit
cargo oxide run -- --bench   # also times both mat-vecs at 4096 x 4096
grep -c 'fma\.' kadmium_cuda_device.ptx   # 0: every f32 op is an explicit add/sub/mul/div.rn
```

The workspace tests that tie the references to the runtime run anywhere:

```sh
cargo test -p kadmium-cuda
cargo test -p kadmium-runtime cuda_parity
```

## Results

The figures below are reported hardware-harness results; exact checkout, run date, repetition count, and raw logs are not recorded here. They are also entered with Measured/Derived labels and provenance caveats in the [benchmark ledger](../../docs/BENCHMARKS_2026-10.md).

RTX 5070 (sm_120), CUDA 13.3, driver 616.92, WSL2 Ubuntu: every case bit-identical, including
partial and zero blocks, NaN and infinite inputs, every `E8M0` exponent (subnormal and
overflowing scales), tied router logits, and rows longer than 1024 reduction chunks.

| 4096 x 4096 (range over runs) | per launch | weight bandwidth |
| --- | ---: | ---: |
| `q4k_q8k_matvec` | 33-38 us | 330-380 GB/s |
| `mxfp4_matvec` | 100-110 us | 81-88 GB/s |

`mxfp4_matvec` makes every product a leaf of the RepOps tree, so it is bound by the reduction,
not by memory. An `MXFP4` x `Q8_K` integer block dot, as in InferSSD's expert path, would give
it the `Q4_K` kernel's shape.
