//! Scalar RepOps transcendentals on the four [`Fp`] operations.
//!
//! These mirror `kadmium_runtime::repops::{exp, silu}` operation for operation, with every
//! constant written the same way, so a kernel that calls them produces the runtime's bits. The
//! runtime's tests hold the two implementations to bit equality.

use crate::fp::Fp;

/// `f32::round` (half away from zero) for `|x| < 2^31`, from a truncating cast and exact
/// additions, so it needs no `libm` or `libdevice` call. The additions go through [`Fp`] too:
/// PTX may contract an addition that carries no rounding modifier. A NaN rounds to `0.0`; the callers
/// here only reach it with finite values, or cast the result to an integer, where Rust maps a
/// NaN to zero as well. The sign of a zero result follows `x`, as with `f32::round`.
#[inline(always)]
pub fn round_half_away<F: Fp>(x: f32) -> f32 {
    // Truncate through `i32`, then carry the sign over so `-0.3` rounds to `-0.0` like `round`.
    let truncated = f32::from_bits(((x as i32) as f32).to_bits() | (x.to_bits() & 0x8000_0000));
    // `x - trunc(x)` is exact: both share an exponent range and the result has fewer bits.
    let fraction = F::sub(x, truncated);
    if fraction >= 0.5 {
        F::add(truncated, 1.0)
    } else if fraction <= -0.5 {
        F::sub(truncated, 1.0)
    } else {
        truncated
    }
}

#[inline(always)]
fn pow2(k: i32) -> f32 {
    f32::from_bits(((k + 127) as u32) << 23)
}

/// `x * 2^k` through the exponent field, in at most two steps so every factor is a normal number.
#[inline(always)]
pub fn ldexp<F: Fp>(x: f32, k: i32) -> f32 {
    if k > 127 {
        F::mul(F::mul(x, pow2(127)), pow2((k - 127).min(127)))
    } else if k < -126 {
        F::mul(F::mul(x, pow2(-126)), pow2((k + 126).max(-126)))
    } else {
        F::mul(x, pow2(k))
    }
}

/// `e^x`: Cody-Waite `ln 2` reduction, a degree-7 Taylor polynomial in Horner order, then
/// scaling by the exponent field. Same thresholds and constants as `repops::exp`.
#[inline(always)]
pub fn exp<F: Fp>(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > 88.722_84 {
        return f32::INFINITY;
    }
    if x < -103.972_08 {
        return 0.0;
    }
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200);
    const LN2_LO: f32 = 1.428_606_8e-6_f32;
    let k = round_half_away::<F>(F::mul(x, core::f32::consts::LOG2_E));
    let r = F::sub(F::sub(x, F::mul(k, LN2_HI)), F::mul(k, LN2_LO));
    let mut p = 1.984_127e-4;
    p = F::add(0.001_388_889, F::mul(r, p));
    p = F::add(0.008_333_334, F::mul(r, p));
    p = F::add(0.041_666_668, F::mul(r, p));
    p = F::add(0.166_666_67, F::mul(r, p));
    p = F::add(0.5, F::mul(r, p));
    p = F::add(1.0, F::mul(r, p));
    p = F::add(1.0, F::mul(r, p));
    ldexp::<F>(p, k as i32)
}

/// `SiLU(x) = x / (1 + e^-x)`.
#[inline(always)]
pub fn silu<F: Fp>(x: f32) -> f32 {
    F::div(x, F::add(1.0, exp::<F>(-x)))
}

/// `max` that ignores a NaN operand, like `f32::max`. Over non-negative values (absolute values)
/// the result does not depend on the order of the operands, so a warp may reduce in any order.
#[inline(always)]
pub fn max_ignore_nan(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() || a >= b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::HostFp;

    #[test]
    fn round_half_away_matches_std_round() {
        let mut x = -300.0f32;
        while x <= 300.0 {
            for nudge in [0.0f32, 1e-3, -1e-3] {
                let v = x + nudge;
                assert_eq!(
                    round_half_away::<HostFp>(v).to_bits(),
                    v.round().to_bits(),
                    "{v}"
                );
            }
            x += 0.25;
        }
        for v in [
            0.49999997f32,
            -0.49999997,
            0.5,
            -0.5,
            2.5,
            -2.5,
            126.5,
            -127.5,
        ] {
            assert_eq!(round_half_away::<HostFp>(v), v.round(), "{v}");
        }
    }

    #[test]
    fn exp_matches_reference_accuracy() {
        for i in -2000..=2000 {
            let x = i as f32 * 0.044;
            let got = exp::<HostFp>(x);
            let want = (x as f64).exp();
            let rel = ((got as f64 - want) / want).abs();
            assert!(rel < 2e-6, "exp({x}) = {got}, want {want}");
        }
        assert_eq!(exp::<HostFp>(100.0), f32::INFINITY);
        assert_eq!(exp::<HostFp>(-200.0), 0.0);
        assert!(exp::<HostFp>(f32::NAN).is_nan());
    }

    #[test]
    fn max_ignore_nan_is_order_free() {
        let values = [0.0f32, 3.5, f32::NAN, 2.0, 7.25, f32::NAN, 1.0];
        let forward = values.iter().copied().fold(0.0, max_ignore_nan);
        let backward = values.iter().rev().copied().fold(0.0, max_ignore_nan);
        assert_eq!(forward, 7.25);
        assert_eq!(backward, 7.25);
    }
}
