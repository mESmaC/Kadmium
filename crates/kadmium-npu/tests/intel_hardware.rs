//! Explicit hardware tests: unavailable hardware is a failure, never a skip.
//! Run with `cargo test -p kadmium-npu --test intel_hardware -- --ignored`.
use kadmium_npu::{
    Error, MatMulKernel, MatMulShape, NpuBackend, intel::IntelNpu, reference_matmul,
};

fn samples(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

#[test]
#[ignore = "requires Intel AI Boost and its Level Zero driver/compiler"]
fn real_npu_rectangular_random_and_repeated_inputs() {
    let npu = IntelNpu::open().expect("Intel NPU must be available");
    eprintln!("{:?}", npu.device());
    for (m, k, n) in [
        (1, 1, 1),
        (2, 3, 4),
        (1, 33, 17),
        (7, 65, 31),
        (16, 256, 128),
    ] {
        let shape = MatMulShape::new(m, k, n).unwrap();
        let weights = samples(shape.weights_len(), 7);
        let mut kernel = npu.prepare_matmul(shape, &weights).unwrap();
        for seed in [1, 2, 3] {
            let input = samples(shape.input_len(), seed);
            let expected = reference_matmul(shape, &input, &weights).unwrap();
            let actual = kernel.execute(&input).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                // Acceptance criterion for these bounded test inputs only.
                assert!(
                    a.is_finite() && (a - b).abs() <= 0.01 + 0.01 * b.abs(),
                    "{m}x{k}x{n}, seed {seed}, element {index}: actual {a}, expected {b}"
                );
            }
        }
        assert!(
            kernel
                .execute(&vec![0.; shape.input_len()])
                .unwrap()
                .iter()
                .all(|v| *v == 0.)
        );
    }
}

#[test]
#[ignore = "requires Intel AI Boost and its Level Zero driver/compiler"]
fn kernels_outlive_backend_and_reject_invalid_inputs_without_poisoning() {
    let npu = IntelNpu::open().unwrap();
    let shape = MatMulShape::new(2, 3, 2).unwrap();
    assert!(npu.prepare_matmul(shape, &[]).is_err());
    assert!(npu.prepare_matmul(shape, &[f32::NAN; 6]).is_err());
    let mut one = npu.prepare_matmul(shape, &[1.; 6]).unwrap();
    let mut two = npu.prepare_matmul(shape, &[2.; 6]).unwrap();
    drop(npu);
    assert!(matches!(one.execute(&[]), Err(Error::InvalidMatrix(_))));
    assert!(matches!(
        one.execute(&[f32::INFINITY; 6]),
        Err(Error::InvalidMatrix(_))
    ));
    assert_eq!(one.execute(&[1.; 6]).unwrap(), vec![3.; 4]);
    assert_eq!(two.execute(&[1.; 6]).unwrap(), vec![6.; 4]);
    drop(one);
    assert_eq!(two.execute(&[2.; 6]).unwrap(), vec![12.; 4]);
}
