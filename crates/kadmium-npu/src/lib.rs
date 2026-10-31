//! Vendor-neutral NPU kernels, starting with Intel AI Boost.
//!
//! All Kadmium code, including graph generation and driver bindings, is Rust.
//! Intel's installed Level Zero driver supplies the native compiler and firmware.
//! NPU results are approximate; they must not be used as RepOps digest evidence.
#![doc = include_str!("../README.md")]

pub mod intel;
mod ir;

use thiserror::Error;

/// Errors never silently select a CPU or GPU fallback.
#[derive(Debug, Error)]
pub enum Error {
    #[error("NPU unavailable: {0}")]
    Unavailable(String),
    #[error("unsupported NPU configuration: {0}")]
    Unsupported(String),
    #[error("invalid matrix: {0}")]
    InvalidMatrix(&'static str),
    #[error("{operation} failed with Level Zero result 0x{code:08x}")]
    Driver { operation: &'static str, code: u32 },
    #[error("Intel NPU graph compilation failed (0x{code:08x}): {log}")]
    Compilation { code: u32, log: String },
    #[error("NPU execution is poisoned after an earlier submission or synchronization failure")]
    Poisoned,
    #[error("NPU produced a non-finite result")]
    NonFiniteOutput,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Row-major C[M,N] = A[M,K] * B[K,N]. No broadcasting or implicit transpose.
/// Dimensions are immutable after validation; each tensor is limited to 64 MiB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatMulShape {
    m: usize,
    k: usize,
    n: usize,
}

impl MatMulShape {
    pub fn new(m: usize, k: usize, n: usize) -> Result<Self> {
        if [m, k, n].contains(&0) {
            return Err(Error::InvalidMatrix("dimensions must be nonzero"));
        }
        for (a, b) in [(m, k), (k, n), (m, n)] {
            let bytes = a.checked_mul(b).and_then(|v| v.checked_mul(4));
            if bytes.is_none_or(|v| v > 64 * 1024 * 1024) {
                return Err(Error::InvalidMatrix("tensor exceeds the 64 MiB limit"));
            }
        }
        Ok(Self { m, k, n })
    }

    pub fn m(self) -> usize {
        self.m
    }
    pub fn k(self) -> usize {
        self.k
    }
    pub fn n(self) -> usize {
        self.n
    }
    pub fn input_len(self) -> usize {
        self.m * self.k
    }
    pub fn weights_len(self) -> usize {
        self.k * self.n
    }
    pub fn output_len(self) -> usize {
        self.m * self.n
    }
}

/// Numerical contract, independent of vendor and host-facing tensor storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arithmetic {
    /// FP32 I/O; the device compiler may use FP16 internally, fuse and reassociate.
    /// No universal error bound or RepOps bit parity is promised.
    ApproximateFp16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub graph_extension_version: u32,
    pub compiler_version: (u16, u16),
    pub arithmetic: Arithmetic,
}

/// Implement this trait to add another vendor without changing callers.
/// Preparing compiles the shape and immutable row-major weights once.
pub trait NpuBackend {
    type Kernel: MatMulKernel;
    fn device(&self) -> &DeviceInfo;
    fn prepare_matmul(&self, shape: MatMulShape, weights: &[f32]) -> Result<Self::Kernel>;
}

/// A prepared kernel owns its driver resources and may be reused for new inputs.
/// Mutable access serializes submissions; implementations must not return before
/// output is host-readable or reuse buffers after a failed synchronization.
pub trait MatMulKernel {
    fn shape(&self) -> MatMulShape;
    fn arithmetic(&self) -> Arithmetic;
    fn execute(&mut self, input: &[f32]) -> Result<Vec<f32>>;
}

pub(crate) fn validate_tensor(values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        return Err(Error::InvalidMatrix(
            "tensor length does not match its shape",
        ));
    }
    if values.iter().any(|v| !v.is_finite() || v.abs() > 65504.0) {
        return Err(Error::InvalidMatrix(
            "values must be finite and within FP16 range",
        ));
    }
    Ok(())
}

/// Explicit CPU oracle for validation, using an f64 accumulator. Never dispatched
/// implicitly, and not a replacement for Kadmium's RepOps reference arithmetic.
pub fn reference_matmul(shape: MatMulShape, input: &[f32], weights: &[f32]) -> Result<Vec<f32>> {
    validate_tensor(input, shape.input_len())?;
    validate_tensor(weights, shape.weights_len())?;
    let mut output = vec![0.0; shape.output_len()];
    for row in 0..shape.m {
        for col in 0..shape.n {
            let sum: f64 = (0..shape.k)
                .map(|k| {
                    f64::from(input[row * shape.k + k]) * f64::from(weights[k * shape.n + col])
                })
                .sum();
            output[row * shape.n + col] = sum as f32;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_overflow_and_excessive_shapes() {
        for dims in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (usize::MAX, 2, 1),
            (1, 1, 16_777_217),
        ] {
            assert!(MatMulShape::new(dims.0, dims.1, dims.2).is_err());
        }
    }

    #[test]
    fn rectangular_matmul_uses_k_by_n_weights() {
        let shape = MatMulShape::new(2, 3, 2).unwrap();
        assert_eq!(
            reference_matmul(
                shape,
                &[1., 2., 3., 4., 5., 6.],
                &[7., 8., 9., 10., 11., 12.]
            )
            .unwrap(),
            [58., 64., 139., 154.]
        );
    }

    #[test]
    fn validates_lengths_and_numerical_domain() {
        let shape = MatMulShape::new(1, 1, 1).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 65505.] {
            assert!(reference_matmul(shape, &[value], &[1.]).is_err());
            assert!(reference_matmul(shape, &[1.], &[value]).is_err());
        }
        assert!(reference_matmul(shape, &[], &[1.]).is_err());
        assert!(reference_matmul(shape, &[1.], &[]).is_err());
        assert_eq!(
            reference_matmul(shape, &[-65504.], &[1.]).unwrap(),
            [-65504.]
        );
    }
}
