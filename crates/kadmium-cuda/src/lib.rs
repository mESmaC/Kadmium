//! Pure-Rust CUDA kernels for Kadmium, compiled with NVIDIA's cuda-oxide (`NVIDIA/cuda-rust`).
//!
//! Kadmium's runtime fixes its numerics with RepOps so a CPU worker, a Metal worker and a CUDA
//! worker can be checked against each other bit for bit. This crate carries that contract onto
//! NVIDIA GPUs. It is split in two:
//!
//! - **This crate** (a workspace member, no dependencies, builds anywhere) holds the lane math
//!   every kernel runs, generic over the four rounded operations in [`fp::Fp`]; the RepOps
//!   reduction tree in the shape a warp computes it ([`tree`]); the block formats ([`formats`]);
//!   a host [`reference`] for every kernel; launch geometry ([`launch`]); and the [`catalog`].
//! - **`device/`** (a standalone crate built with `cargo oxide`, Linux and CUDA 13 only) holds
//!   the `#[kernel]` functions. They call the same lane functions with device arithmetic, which
//!   lowers each operation to an explicit `.rn` instruction, then a parity harness runs every
//!   kernel on the GPU and compares the result with [`reference`] bit for bit.
//!
//! The kernels follow the InferSSD CUDA work that serves MoE experts from SSD: activation
//! codes quantized once per token and reused by every routed expert, a fixed-order `Q4_K` x
//! `Q8_K` mat-vec, `MXFP4` expert weights, a deterministic router top-k and per-row attention
//! masks. Where InferSSD pins its order to its own AVX2 CPU loop, these kernels pin theirs to
//! RepOps, so their output is also the output of Kadmium's verifier.

pub mod catalog;
pub mod formats;
pub mod fp;
pub mod launch;
pub mod math;
pub mod reference;
pub mod routing;
pub mod testing;
pub mod tree;

pub use catalog::{CudaKernel, Numerics};
pub use launch::{LaunchError, LaunchShape};
