//! Launch geometry for each kernel, computed once on the host so the device crate, the
//! runtime's dispatcher and the tests agree on how work maps to warps.
//!
//! Every kernel uses 256-thread blocks. The warp-per-item kernels (`q8k_quantize`, the two
//! mat-vecs, `moe_top_k`) put eight items in a block: InferSSD's dense `Q8_K` path settled on
//! eight warps per block for the same 32-lane-per-row shape.

use core::fmt;

/// Threads per block for every kernel.
pub const BLOCK_THREADS: u32 = 256;
/// Warps, and so warp-per-item work items, per block.
pub const WARPS_PER_BLOCK: u32 = BLOCK_THREADS / 32;
/// Largest grid `x` extent the driver accepts.
pub const MAX_GRID_X: u64 = (1 << 31) - 1;
/// Largest grid `y` extent the driver accepts.
pub const MAX_GRID_Y: u64 = 65_535;
/// Experts a `moe_top_k` row may select: one lane writes each winner.
pub const MAX_TOP_K: usize = 32;

/// Grid, block and dynamic shared memory of one launch, as `cuda_core::LaunchConfig` takes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchShape {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

/// A problem that does not fit one launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchError {
    Empty,
    GridTooLarge { axis: char, blocks: u64 },
    Unaligned { what: &'static str, value: usize },
    TopKTooLarge { k: usize, experts: usize },
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "nothing to launch"),
            Self::GridTooLarge { axis, blocks } => {
                write!(
                    f,
                    "grid {axis} needs {blocks} blocks, past the driver limit"
                )
            }
            Self::Unaligned { what, value } => {
                write!(f, "{what} = {value} is not a multiple of 32")
            }
            Self::TopKTooLarge { k, experts } => write!(
                f,
                "top-k {k} must be at most min({MAX_TOP_K}, experts = {experts})"
            ),
        }
    }
}

impl std::error::Error for LaunchError {}

fn grid_x(blocks: u64) -> Result<u32, LaunchError> {
    match blocks {
        0 => Err(LaunchError::Empty),
        b if b > MAX_GRID_X => Err(LaunchError::GridTooLarge { axis: 'x', blocks }),
        b => Ok(b as u32),
    }
}

fn aligned(what: &'static str, value: usize) -> Result<(), LaunchError> {
    if value == 0 {
        Err(LaunchError::Empty)
    } else if !value.is_multiple_of(32) {
        Err(LaunchError::Unaligned { what, value })
    } else {
        Ok(())
    }
}

/// One warp per item, eight items per block.
pub fn warp_per_item(items: usize) -> Result<LaunchShape, LaunchError> {
    Ok(LaunchShape {
        grid: (
            grid_x((items as u64).div_ceil(WARPS_PER_BLOCK as u64))?,
            1,
            1,
        ),
        block: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// One thread per element.
pub fn thread_per_element(elements: usize) -> Result<LaunchShape, LaunchError> {
    Ok(LaunchShape {
        grid: (
            grid_x((elements as u64).div_ceil(BLOCK_THREADS as u64))?,
            1,
            1,
        ),
        block: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// `q8k_quantize` over `len` activations: a warp per 32-element block.
pub fn q8k_quantize(len: usize) -> Result<LaunchShape, LaunchError> {
    warp_per_item(len.div_ceil(32))
}

/// `q4k_q8k_matvec` / `mxfp4_matvec`: a warp per output row.
pub fn matvec(rows: usize, cols: usize) -> Result<LaunchShape, LaunchError> {
    aligned("cols", cols)?;
    warp_per_item(rows)
}

/// `moe_top_k`: a warp per token row.
pub fn moe_top_k(rows: usize, experts: usize, k: usize) -> Result<LaunchShape, LaunchError> {
    if k == 0 || k > MAX_TOP_K || k > experts {
        return Err(LaunchError::TopKTooLarge { k, experts });
    }
    warp_per_item(rows)
}

/// `row_mask`: columns across grid `x`, one grid row per mask row.
pub fn row_mask(rows: usize, width: usize) -> Result<LaunchShape, LaunchError> {
    if rows as u64 > MAX_GRID_Y {
        return Err(LaunchError::GridTooLarge {
            axis: 'y',
            blocks: rows as u64,
        });
    }
    if rows == 0 {
        return Err(LaunchError::Empty);
    }
    Ok(LaunchShape {
        grid: (
            grid_x((width as u64).div_ceil(BLOCK_THREADS as u64))?,
            rows as u32,
            1,
        ),
        block: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_kernels_cover_every_item() {
        let shape = matvec(17, 4096).unwrap();
        assert_eq!(shape.grid, (3, 1, 1));
        assert!(shape.grid.0 * WARPS_PER_BLOCK >= 17);
        assert_eq!(q8k_quantize(4097).unwrap().grid.0, 17); // 129 blocks of 32
    }

    #[test]
    fn rejects_unlaunchable_problems() {
        assert_eq!(
            matvec(4, 100),
            Err(LaunchError::Unaligned {
                what: "cols",
                value: 100
            })
        );
        assert_eq!(matvec(0, 64), Err(LaunchError::Empty));
        assert!(moe_top_k(1, 8, 9).is_err());
        assert!(moe_top_k(1, 64, 33).is_err());
        assert!(row_mask(70_000, 8).is_err());
        assert_eq!(row_mask(3, 300).unwrap().grid, (2, 3, 1));
    }
}
