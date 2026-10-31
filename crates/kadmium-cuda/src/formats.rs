//! Block formats the kernels read, and the per-lane arithmetic on them.
//!
//! - **`Q8_K` activations**: Kadmium's `Q8KBlockVector`: 32-element blocks, an `f32` scale
//!   `d = absmax / 127`, `i8` codes in `[-127, 127]` and the `i32` code sum per block.
//! - **`Q4_K` weights**: Kadmium's `Q4KBlockTensor`: row-major, 32-element blocks with an
//!   `f32` scale and minimum each, two 4-bit codes per byte (even element in the low nibble).
//! - **`MXFP4` weights**: the GGUF / OCP microscaling block InferSSD serves DeepSeek's routed
//!   experts in: one `E8M0` exponent byte, then 16 bytes holding 32 `E2M1` codes, element
//!   `j < 16` in the low nibble of byte `j` and element `j + 16` in its high nibble.
//!
//! Every function a kernel calls takes scalars and returns scalars, so it compiles unchanged
//! for the device.

use crate::fp::Fp;
use crate::math::round_half_away;

/// Elements per `Q8_K`, `Q4_K` and `MXFP4` block.
pub const BLOCK: usize = 32;
/// Bytes of packed 4-bit codes per 32-element block.
pub const NIBBLE_BYTES: usize = BLOCK / 2;
/// Bytes per `MXFP4` block: the exponent, then the codes.
pub const MXFP4_BLOCK_BYTES: usize = 1 + NIBBLE_BYTES;

/// `E2M1` code points, doubled so they are integers: `0, 0.5, 1, 1.5, 2, 3, 4, 6` and negatives.
/// The block exponent is halved to match ([`e8m0_to_f32_half`]), as in `ggml`.
pub const E2M1_DOUBLED: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// `Q8_K` block scale and its reciprocal for a block whose largest magnitude is `absmax`, or
/// `None` for a block that quantizes to zeros (all zero, or a non-finite magnitude).
#[inline(always)]
pub fn q8k_scales<F: Fp>(absmax: f32) -> Option<(f32, f32)> {
    if absmax == 0.0 || !absmax.is_finite() {
        None
    } else {
        Some((F::div(absmax, 127.0), F::div(127.0, absmax)))
    }
}

/// The `Q8_K` code of `value` under reciprocal scale `inv_d`: round half away from zero, then
/// clamp to `[-127, 127]`; a NaN codes as zero, as Rust's `as i8` does.
#[inline(always)]
pub fn q8k_code<F: Fp>(value: f32, inv_d: f32) -> i8 {
    let q = round_half_away::<F>(F::mul(value, inv_d)) as i32;
    q.clamp(-127, 127) as i8
}

/// One block's term of a `Q4_K` x `Q8_K` row dot product:
/// `d_x * (min_w * bsum_x + scale_w * sum(q_w * q_x))`, each operation rounded once.
#[inline(always)]
pub fn q4k_q8k_block_term<F: Fp>(
    d_x: f32,
    scale_w: f32,
    min_w: f32,
    bsum_x: i32,
    code_dot: i32,
) -> f32 {
    if d_x == 0.0 {
        return 0.0;
    }
    F::mul(
        d_x,
        F::add(
            F::mul(min_w, bsum_x as f32),
            F::mul(scale_w, code_dot as f32),
        ),
    )
}

/// `sum(q_w * q_x)` over one block: 16 packed weight bytes against 32 activation codes.
/// Integer arithmetic, so any evaluation order gives the same value.
#[inline(always)]
pub fn q4k_code_dot(weights: &[u8], codes: &[i8]) -> i32 {
    let mut dot = 0i32;
    let mut j = 0;
    while j < NIBBLE_BYTES {
        let byte = weights[j];
        dot += (byte & 0x0f) as i32 * codes[2 * j] as i32;
        dot += (byte >> 4) as i32 * codes[2 * j + 1] as i32;
        j += 1;
    }
    dot
}

/// Half the `E8M0` block scale, `2^(e - 128)`, as `ggml_e8m0_to_fp32_half`: exponents 0 and 1
/// land in the subnormal range, and 255 is the finite `2^127` rather than a NaN.
#[inline(always)]
pub fn e8m0_to_f32_half(e: u8) -> f32 {
    let bits = if e < 2 {
        0x0020_0000u32 << e
    } else {
        ((e as u32) - 1) << 23
    };
    f32::from_bits(bits)
}

/// Element `j` of an `MXFP4` block. Exact: a small integer times a power of two.
#[inline(always)]
pub fn mxfp4_weight<F: Fp>(block: &[u8], j: usize) -> f32 {
    let byte = block[1 + (j % NIBBLE_BYTES)];
    let code = if j < NIBBLE_BYTES {
        byte & 0x0f
    } else {
        byte >> 4
    };
    F::mul(
        E2M1_DOUBLED[code as usize] as f32,
        e8m0_to_f32_half(block[0]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::HostFp;

    #[test]
    fn e8m0_half_matches_powers_of_two() {
        assert_eq!(e8m0_to_f32_half(127), 0.5);
        assert_eq!(e8m0_to_f32_half(128), 1.0);
        assert_eq!(e8m0_to_f32_half(130), 4.0);
        assert_eq!(e8m0_to_f32_half(1) as f64, 2f64.powi(-127));
        assert_eq!(e8m0_to_f32_half(0) as f64, 2f64.powi(-128));
        assert_eq!(e8m0_to_f32_half(255) as f64, 2f64.powi(127));
    }

    #[test]
    fn mxfp4_decodes_low_then_high_nibbles() {
        let mut block = [0u8; MXFP4_BLOCK_BYTES];
        block[0] = 127; // 2^0, stored halved as 0.5 against the doubled code points
        block[1] = 0x97; // element 0: code 7 (6.0); element 16: code 9 (-0.5)
        assert_eq!(mxfp4_weight::<HostFp>(&block, 0), 6.0);
        assert_eq!(mxfp4_weight::<HostFp>(&block, 16), -0.5);
        assert_eq!(mxfp4_weight::<HostFp>(&block, 1), 0.0);
    }

    #[test]
    fn q8k_codes_clamp_and_zero_nan() {
        let (d, inv_d) = q8k_scales::<HostFp>(2.0).unwrap();
        assert_eq!(d, 2.0 / 127.0);
        assert_eq!(q8k_code::<HostFp>(2.0, inv_d), 127);
        assert_eq!(q8k_code::<HostFp>(-2.0, inv_d), -127);
        assert_eq!(q8k_code::<HostFp>(f32::NAN, inv_d), 0);
        assert!(q8k_scales::<HostFp>(0.0).is_none());
        assert!(q8k_scales::<HostFp>(f32::INFINITY).is_none());
    }
}
