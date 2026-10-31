//! Runs every kernel on the GPU over seeded inputs, including the edge cases each one has to
//! get right (partial blocks, zero blocks, NaN and infinity, subnormal and overflowing MXFP4
//! scales, tied router logits, more than 1024 reduction chunks), and compares the result with
//! `kadmium_cuda::reference` bit for bit, every NaN counted as one value.

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig, Stream};
use kadmium_cuda::formats::{MXFP4_BLOCK_BYTES, NIBBLE_BYTES};
use kadmium_cuda::fp::canonical_bits;
use kadmium_cuda::launch::{self, LaunchShape};
use kadmium_cuda::reference::{self, Mxfp4View, Q4KView, Q8KActivations};
use kadmium_cuda::testing::SplitMix64;

use crate::kernels;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn config(shape: LaunchShape) -> LaunchConfig {
    LaunchConfig {
        grid_dim: shape.grid,
        block_dim: shape.block,
        shared_mem_bytes: shape.shared_mem_bytes,
    }
}

/// A value compared by its exact bits. Every `f32` NaN compares as one value: the hardware,
/// not the operation order, picks a NaN's payload (see `kadmium_cuda::fp::canonical_bits`).
trait Bits: Copy {
    fn bits(self) -> u64;
}

impl Bits for f32 {
    fn bits(self) -> u64 {
        canonical_bits(self) as u64
    }
}

impl Bits for i32 {
    fn bits(self) -> u64 {
        self as u32 as u64
    }
}

impl Bits for i8 {
    fn bits(self) -> u64 {
        self as u8 as u64
    }
}

/// FNV-1a 64 over the little-endian bits, for a one-line fingerprint of each output.
fn digest<T: Bits>(values: &[T]) -> u64 {
    let width = std::mem::size_of::<T>();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for byte in &v.bits().to_le_bytes()[..width] {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

struct Checker {
    passed: bool,
}

impl Checker {
    fn check<T: Bits + std::fmt::Debug>(&mut self, case: &str, got: &[T], want: &[T]) {
        let mismatches: Vec<usize> = (0..want.len().max(got.len()))
            .filter(|&i| got.get(i).map(|v| v.bits()) != want.get(i).map(|v| v.bits()))
            .collect();
        if mismatches.is_empty() {
            println!(
                "  ok    {case:<44} {:>9} values  digest {:016x}",
                want.len(),
                digest(want)
            );
        } else {
            self.passed = false;
            println!(
                "  FAIL  {case:<44} {} of {} values differ",
                mismatches.len(),
                want.len()
            );
            for &i in mismatches.iter().take(5) {
                println!(
                    "        [{i}] gpu {:?} ({:#x})  reference {:?} ({:#x})",
                    got.get(i),
                    got.get(i).map_or(0, |v| v.bits()),
                    want.get(i),
                    want.get(i).map_or(0, |v| v.bits())
                );
            }
        }
    }
}

struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<Stream>,
    module: kernels::LoadedModule,
}

pub fn run(bench: bool) -> Result<bool> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let gpu = Gpu {
        ctx,
        stream,
        module,
    };
    let mut checker = Checker { passed: true };
    let mut rng = SplitMix64::new(0x6b61_646d_6975_6d); // "kadmium"

    println!("q8k_quantize");
    for len in [1usize, 31, 32, 33, 4096, 4096 + 17, 1 << 16] {
        let mut x = rng.activations(len);
        if len > 40 {
            x[3] = f32::NAN; // ignored by the absmax, codes as zero
            x[len - 2] = f32::INFINITY; // zeroes its whole block
        }
        let got = gpu.q8k_quantize(&x)?;
        let want = reference::q8k_quantize(&x);
        checker.check(&format!("len {len} codes"), &got.qs, &want.qs);
        checker.check(&format!("len {len} scales"), &got.scales, &want.scales);
        checker.check(&format!("len {len} code sums"), &got.bsums, &want.bsums);
    }

    println!("q4k_q8k_matvec");
    for (rows, cols) in [(1usize, 32usize), (7, 32 * 33), (64, 4096), (40, 32 * 1030)] {
        let w = Q4KWeights::random(&mut rng, rows, cols);
        let x = reference::q8k_quantize(&rng.activations(cols));
        let got = gpu.q4k_q8k_matvec(&w, &x)?;
        let want = reference::q4k_q8k_matvec(w.view(), x.view());
        checker.check(&format!("{rows} x {cols}"), &got, &want);
    }

    println!("mxfp4_matvec");
    for (rows, cols, edge) in [
        (1usize, 32usize, false),
        (9, 32 * 45, false),
        (256, 4096, false),
        (16, 32 * 1100, false),
        (32, 1024, true),
    ] {
        let blocks = mxfp4_blocks(&mut rng, rows, cols, edge);
        let x: Vec<f32> = (0..cols).map(|_| rng.unit_f32()).collect();
        let got = gpu.mxfp4_matvec(&blocks, &x, rows)?;
        let want = reference::mxfp4_matvec(
            Mxfp4View {
                rows,
                cols,
                blocks: &blocks,
            },
            &x,
        );
        let label = if edge { " (all exponents)" } else { "" };
        checker.check(&format!("{rows} x {cols}{label}"), &got, &want);
    }

    println!("swiglu");
    for len in [1usize, 1000, 1 << 20] {
        let mut gate: Vec<f32> = (0..len).map(|_| rng.unit_f32() * 120.0).collect();
        let up: Vec<f32> = (0..len).map(|_| rng.wide_f32()).collect();
        if len > 4 {
            gate[0] = f32::NAN;
            gate[1] = f32::INFINITY;
            gate[2] = f32::NEG_INFINITY;
            gate[3] = -0.0;
        }
        let got = gpu.swiglu(&gate, &up)?;
        let want = reference::swiglu(&gate, &up);
        checker.check(&format!("len {len}"), &got, &want);
    }

    println!("moe_top_k");
    for (rows, experts, k) in [
        (1usize, 8usize, 8usize),
        (5, 64, 6),
        (33, 256, 8),
        (4, 1000, 32),
    ] {
        // Few distinct values, so ties are common; a sprinkle of NaN and -inf.
        let logits: Vec<f32> = (0..rows * experts)
            .map(|_| match rng.next_u32() % 40 {
                0 => f32::NAN,
                1 => f32::NEG_INFINITY,
                v => (v % 7) as f32 * 0.25 - 0.5,
            })
            .collect();
        let (got_ids, got_scores) = gpu.moe_top_k(&logits, experts, k)?;
        let (want_ids, want_scores) = reference::moe_top_k(&logits, experts, k);
        checker.check(
            &format!("{rows} rows x {experts} experts, k {k} ids"),
            &got_ids,
            &want_ids,
        );
        checker.check(
            &format!("{rows} rows x {experts} experts, k {k} scores"),
            &got_scores,
            &want_scores,
        );
    }

    println!("row_mask");
    for (rows, width) in [(3usize, 300usize), (100, 4096)] {
        let visible: Vec<i32> = (0..rows)
            .map(|_| (rng.next_u32() % (width as u32 + 20)) as i32 - 10)
            .collect();
        let got = gpu.row_mask(&visible, width)?;
        let want = reference::row_mask(&visible, width);
        checker.check(&format!("{rows} x {width}"), &got, &want);
    }

    if bench {
        gpu.bench(&mut rng)?;
    }
    Ok(checker.passed)
}

/// Random `Q4_K` weights laid out as `Q4KBlockTensor`.
struct Q4KWeights {
    rows: usize,
    cols: usize,
    packed: Vec<u8>,
    scales: Vec<f32>,
    mins: Vec<f32>,
}

impl Q4KWeights {
    fn random(rng: &mut SplitMix64, rows: usize, cols: usize) -> Self {
        let blocks = rows * cols / 32;
        Self {
            rows,
            cols,
            packed: rng.bytes(blocks * NIBBLE_BYTES),
            scales: (0..blocks).map(|_| rng.unit_f32().abs() * 0.02).collect(),
            mins: (0..blocks).map(|_| rng.unit_f32() * 0.15).collect(),
        }
    }

    fn view(&self) -> Q4KView<'_> {
        Q4KView {
            rows: self.rows,
            cols: self.cols,
            packed_nibbles: &self.packed,
            block_scales: &self.scales,
            block_mins: &self.mins,
        }
    }
}

/// Random `MXFP4` blocks: exponents near 1.0, or anywhere in `0..=255` for the edge case.
fn mxfp4_blocks(rng: &mut SplitMix64, rows: usize, cols: usize, edge: bool) -> Vec<u8> {
    let mut blocks = rng.bytes(rows * cols / 32 * MXFP4_BLOCK_BYTES);
    if !edge {
        for block in blocks.chunks_mut(MXFP4_BLOCK_BYTES) {
            block[0] = 118 + block[0] % 18;
        }
    }
    blocks
}

impl Gpu {
    fn upload<T: cuda_core::DeviceCopy>(&self, data: &[T]) -> Result<DeviceBuffer<T>> {
        Ok(DeviceBuffer::from_host(&self.stream, data)?)
    }

    fn zeroed<T: cuda_core::DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        Ok(DeviceBuffer::zeroed(&self.stream, len)?)
    }

    fn q8k_quantize(&self, x: &[f32]) -> Result<Q8KActivations> {
        let blocks = x.len().div_ceil(32);
        let x_dev = self.upload(x)?;
        let mut qs = self.zeroed::<i8>(x.len())?;
        let mut scales = self.zeroed::<f32>(blocks)?;
        let mut bsums = self.zeroed::<i32>(blocks)?;
        let cfg = config(launch::q8k_quantize(x.len())?);
        // SAFETY: the shape comes from `launch::q8k_quantize`: a 1-D grid with a warp per block.
        unsafe {
            self.module
                .q8k_quantize(&self.stream, cfg, &x_dev, &mut qs, &mut scales, &mut bsums)
        }?;
        Ok(Q8KActivations {
            qs: qs.to_host_vec(&self.stream)?,
            scales: scales.to_host_vec(&self.stream)?,
            bsums: bsums.to_host_vec(&self.stream)?,
        })
    }

    fn q4k_q8k_matvec(&self, w: &Q4KWeights, x: &Q8KActivations) -> Result<Vec<f32>> {
        let packed = self.upload(&w.packed)?;
        let scales = self.upload(&w.scales)?;
        let mins = self.upload(&w.mins)?;
        let x_scales = self.upload(&x.scales)?;
        let x_qs = self.upload(&x.qs)?;
        let x_bsums = self.upload(&x.bsums)?;
        let mut y = self.zeroed::<f32>(w.rows)?;
        let cfg = config(launch::matvec(w.rows, w.cols)?);
        // SAFETY: a 1-D grid with a warp per row, from `launch::matvec`.
        unsafe {
            self.module.q4k_q8k_matvec(
                &self.stream,
                cfg,
                &packed,
                &scales,
                &mins,
                &x_scales,
                &x_qs,
                &x_bsums,
                &mut y,
            )
        }?;
        Ok(y.to_host_vec(&self.stream)?)
    }

    fn mxfp4_matvec(&self, blocks: &[u8], x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let blocks_dev = self.upload(blocks)?;
        let x_dev = self.upload(x)?;
        let mut y = self.zeroed::<f32>(rows)?;
        let cfg = config(launch::matvec(rows, x.len())?);
        // SAFETY: a 1-D grid with a warp per row, from `launch::matvec`.
        unsafe {
            self.module
                .mxfp4_matvec(&self.stream, cfg, &blocks_dev, &x_dev, &mut y)
        }?;
        Ok(y.to_host_vec(&self.stream)?)
    }

    fn swiglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        let gate_dev = self.upload(gate)?;
        let up_dev = self.upload(up)?;
        let mut out = self.zeroed::<f32>(gate.len())?;
        let cfg = config(launch::thread_per_element(gate.len())?);
        // SAFETY: a 1-D grid with a thread per element, matching `thread::index_1d`.
        unsafe {
            self.module
                .swiglu(&self.stream, cfg, &gate_dev, &up_dev, &mut out)
        }?;
        Ok(out.to_host_vec(&self.stream)?)
    }

    fn moe_top_k(&self, logits: &[f32], experts: usize, k: usize) -> Result<(Vec<i32>, Vec<f32>)> {
        let rows = logits.len() / experts;
        let logits_dev = self.upload(logits)?;
        let mut ids = self.zeroed::<i32>(rows * k)?;
        let mut scores = self.zeroed::<f32>(rows * k)?;
        let cfg = config(launch::moe_top_k(rows, experts, k)?);
        // SAFETY: a 1-D grid with a warp per row and `k <= 32`, from `launch::moe_top_k`.
        unsafe {
            self.module.moe_top_k(
                &self.stream,
                cfg,
                &logits_dev,
                experts as u32,
                k as u32,
                &mut ids,
                &mut scores,
            )
        }?;
        Ok((
            ids.to_host_vec(&self.stream)?,
            scores.to_host_vec(&self.stream)?,
        ))
    }

    fn row_mask(&self, visible: &[i32], width: usize) -> Result<Vec<f32>> {
        let visible_dev = self.upload(visible)?;
        let mut out = self.zeroed::<f32>(visible.len() * width)?;
        let cfg = config(launch::row_mask(visible.len(), width)?);
        // SAFETY: grid x covers the columns and grid y the rows, from `launch::row_mask`.
        unsafe {
            self.module
                .row_mask(&self.stream, cfg, &visible_dev, width as u32, &mut out)
        }?;
        Ok(out.to_host_vec(&self.stream)?)
    }

    /// Times the two mat-vecs at 4096 x 4096 and reports weight bandwidth.
    fn bench(&self, rng: &mut SplitMix64) -> Result<()> {
        const ROWS: usize = 4096;
        const COLS: usize = 4096;
        const ITERS: u32 = 200;
        println!("\nbench ({ROWS} x {COLS}, {ITERS} launches each)");

        let w = Q4KWeights::random(rng, ROWS, COLS);
        let x = reference::q8k_quantize(&rng.activations(COLS));
        let (packed, scales, mins) = (
            self.upload(&w.packed)?,
            self.upload(&w.scales)?,
            self.upload(&w.mins)?,
        );
        let (x_scales, x_qs, x_bsums) = (
            self.upload(&x.scales)?,
            self.upload(&x.qs)?,
            self.upload(&x.bsums)?,
        );
        let mut y = self.zeroed::<f32>(ROWS)?;
        let cfg = config(launch::matvec(ROWS, COLS)?);
        let q4k_bytes = w.packed.len() + 8 * w.scales.len();
        let elapsed = self.time(ITERS, || {
            // SAFETY: as in `q4k_q8k_matvec`.
            unsafe {
                self.module.q4k_q8k_matvec(
                    &self.stream,
                    cfg,
                    &packed,
                    &scales,
                    &mins,
                    &x_scales,
                    &x_qs,
                    &x_bsums,
                    &mut y,
                )
            }
            .map_err(Into::into)
        })?;
        report("q4k_q8k_matvec", elapsed, q4k_bytes);

        let blocks = mxfp4_blocks(rng, ROWS, COLS, false);
        let xf: Vec<f32> = (0..COLS).map(|_| rng.unit_f32()).collect();
        let (blocks_dev, x_dev) = (self.upload(&blocks)?, self.upload(&xf)?);
        let elapsed = self.time(ITERS, || {
            // SAFETY: as in `mxfp4_matvec`.
            unsafe {
                self.module
                    .mxfp4_matvec(&self.stream, cfg, &blocks_dev, &x_dev, &mut y)
            }
            .map_err(Into::into)
        })?;
        report("mxfp4_matvec", elapsed, blocks.len());
        Ok(())
    }

    fn time(&self, iters: u32, mut launch: impl FnMut() -> Result<()>) -> Result<f64> {
        launch()?; // warm-up, module load and first-touch
        self.ctx.synchronize()?;
        let start = Instant::now();
        for _ in 0..iters {
            launch()?;
        }
        self.ctx.synchronize()?;
        Ok(start.elapsed().as_secs_f64() / iters as f64)
    }
}

fn report(name: &str, seconds: f64, weight_bytes: usize) {
    println!(
        "  {name:<16} {:>8.1} us/launch  {:>7.1} GB/s of weights",
        seconds * 1e6,
        weight_bytes as f64 / seconds / 1e9
    );
}
