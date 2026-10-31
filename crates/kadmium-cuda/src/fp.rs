//! The four rounded `f32` operations every kernel is written against.
//!
//! RepOps fixes the order of operations; it also forbids fused multiply-add, which Rust never
//! contracts on the host but a GPU backend may (`a * b + c` becomes one `fma.rn` under NVPTX's
//! default contraction level). The lane math in this crate is generic over [`Fp`] so the same
//! source compiles to plain operators on the host ([`HostFp`]) and to explicit
//! `add.rn.f32` / `mul.rn.f32` / `div.rn.f32` instructions on the device, which the PTX
//! assembler may not fuse. Subtraction is addition of the negation, which is exact.

/// The bits two backends must agree on: the value's own bits, except that every NaN is the
/// quiet NaN `0x7fc0_0000`. IEEE 754 leaves the payload and sign of a produced NaN to the
/// hardware (x86 returns `0xffc0_0000` for `inf - inf`, NVIDIA GPUs `0x7fff_ffff`), so no
/// fixed operation order can pin them; whether a result is NaN is still exact.
#[inline(always)]
pub fn canonical_bits(x: f32) -> u32 {
    if x.is_nan() { 0x7fc0_0000 } else { x.to_bits() }
}

/// Correctly rounded, round-to-nearest-even `f32` arithmetic with no contraction.
pub trait Fp {
    fn add(a: f32, b: f32) -> f32;
    fn mul(a: f32, b: f32) -> f32;
    fn div(a: f32, b: f32) -> f32;

    #[inline(always)]
    fn sub(a: f32, b: f32) -> f32 {
        Self::add(a, -b)
    }
}

/// Host arithmetic: Rust's operators, which are IEEE 754 round-to-nearest and never fused.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostFp;

impl Fp for HostFp {
    #[inline(always)]
    fn add(a: f32, b: f32) -> f32 {
        a + b
    }

    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        a * b
    }

    #[inline(always)]
    fn div(a: f32, b: f32) -> f32 {
        a / b
    }
}
