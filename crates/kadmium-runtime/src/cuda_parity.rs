//! Holds the `kadmium-cuda` host references to this runtime's own numerics, bit for bit.
//!
//! The cuda-oxide kernels in `kadmium-cuda/device` are checked on a GPU against
//! `kadmium_cuda::reference`; these tests close the chain by checking that reference against
//! the runtime paths a CPU worker actually runs, so a CUDA worker and a CPU worker produce the
//! same bits for the same operation.

use crate::kernels::{HardwareKernelTarget, MulMatIdExpertBank};
use crate::{Q4KBlockTensor, Q8KBlockVector, repops};
use kadmium_cuda::fp::{HostFp, canonical_bits};
use kadmium_cuda::reference::{self, Q4KView};
use kadmium_cuda::testing::SplitMix64;
use kadmium_cuda::{CudaKernel, math};

fn assert_same_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            canonical_bits(*g),
            canonical_bits(*w),
            "{what}[{i}]: kadmium-cuda {g:e}, runtime {w:e}"
        );
    }
}

#[test]
fn q8k_quantize_matches_the_runtime_quantizer() {
    let mut rng = SplitMix64::new(21);
    for len in [1usize, 31, 32, 33, 100, 4096, 4096 + 17] {
        let mut x = rng.activations(len);
        if len > 40 {
            x[5] = f32::NAN;
            x[len - 1] = f32::NEG_INFINITY;
        }
        let ours = reference::q8k_quantize(&x);
        let runtime = Q8KBlockVector::quantize(&x);
        assert_same_bits("scales", &ours.scales, &runtime.scales);
        assert_eq!(ours.qs, runtime.qs, "codes, len {len}");
        assert_eq!(ours.bsums, runtime.bsums, "code sums, len {len}");
    }
}

#[test]
fn q4k_q8k_matvec_matches_the_runtime_kernel() {
    let mut rng = SplitMix64::new(22);
    for (rows, cols) in [(1usize, 32usize), (6, 32 * 7), (16, 4096), (3, 32 * 1030)] {
        let weights: Vec<f32> = (0..rows * cols).map(|_| rng.unit_f32() * 0.1).collect();
        let w = Q4KBlockTensor::quantize_matrix(&weights, rows, cols);
        let x = Q8KBlockVector::quantize(&rng.activations(cols));
        let ours = reference::q4k_q8k_matvec(
            Q4KView {
                rows,
                cols,
                packed_nibbles: &w.packed_nibbles,
                block_scales: &w.block_scales,
                block_mins: &w.block_mins,
            },
            reference::Q8KView {
                scales: &x.scales,
                qs: &x.qs,
                bsums: &x.bsums,
            },
        );
        assert_same_bits(&format!("{rows} x {cols}"), &ours, &w.matvec_q8k(&x));
    }
}

#[test]
fn exp_and_silu_match_repops() {
    let mut rng = SplitMix64::new(23);
    let mut inputs: Vec<f32> = (0..20_000).map(|_| rng.unit_f32() * 110.0).collect();
    inputs.extend((0..2_000).map(|_| rng.wide_f32()));
    inputs.extend([
        0.0,
        -0.0,
        88.72283,
        88.72285,
        -103.97207,
        -103.97209,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ]);
    for &x in &inputs {
        assert_eq!(
            canonical_bits(math::exp::<HostFp>(x)),
            canonical_bits(repops::exp(x)),
            "exp({x:e})"
        );
        assert_eq!(
            canonical_bits(math::silu::<HostFp>(x)),
            canonical_bits(repops::silu(x)),
            "silu({x:e})"
        );
    }
}

#[test]
fn swiglu_matches_the_expert_bank_split() {
    let mut rng = SplitMix64::new(24);
    let n = 777;
    let gate_up: Vec<f32> = (0..2 * n).map(|_| rng.unit_f32() * 30.0).collect();
    let (gate, up) = gate_up.split_at(n);
    assert_same_bits(
        "swiglu",
        &reference::swiglu(gate, up),
        &MulMatIdExpertBank::swiglu_split(&gate_up, n),
    );
}

#[test]
fn only_the_cuda_target_runs_native_cuda_kernels() {
    assert_eq!(
        HardwareKernelTarget::CudaTensorCorePtx.native_cuda_kernels(),
        &CudaKernel::ALL
    );
    assert!(
        HardwareKernelTarget::MetalSimdgroupMsl
            .native_cuda_kernels()
            .is_empty()
    );
}
