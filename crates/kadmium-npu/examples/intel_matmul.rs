use kadmium_npu::{MatMulKernel, MatMulShape, NpuBackend, intel::IntelNpu, reference_matmul};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let npu = IntelNpu::open()?;
    println!("Intel NPU: {:?}", npu.device());
    // Rectangular and non-tile-aligned cases exercise padding and row order.
    for (m, k, n) in [(1, 1, 1), (2, 3, 4), (1, 33, 17), (4, 64, 32), (8, 128, 64)] {
        let shape = MatMulShape::new(m, k, n)?;
        let weights: Vec<f32> = (0..shape.weights_len())
            .map(|i| ((i % 13) as f32 - 6.) / 16.)
            .collect();
        let start = std::time::Instant::now();
        let mut kernel = npu.prepare_matmul(shape, &weights)?;
        let compile = start.elapsed();
        for iteration in 0..3 {
            let input: Vec<f32> = (0..shape.input_len())
                .map(|i| (((i + iteration) % 11) as f32 - 5.) / 8.)
                .collect();
            let expected = reference_matmul(shape, &input, &weights)?;
            let start = std::time::Instant::now();
            let actual = kernel.execute(&input)?;
            let elapsed = start.elapsed();
            let mut max_error = 0f32;
            for (a, b) in actual.iter().zip(&expected) {
                let error = (a - b).abs();
                max_error = max_error.max(error);
                if error > 0.01 + 0.01 * b.abs() {
                    return Err(format!("{m}x{k}x{n}: got {a}, expected {b}").into());
                }
            }
            println!(
                "{m}x{k}x{n} run {iteration}: compile {compile:?}, execute {elapsed:?}, max abs error {max_error}"
            );
        }
    }
    Ok(())
}
