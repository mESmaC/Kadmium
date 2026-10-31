//! The kernels the device crate builds, what each one guarantees, and where it came from.

/// What a kernel's output is guaranteed to equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Numerics {
    /// Bit-identical to the RepOps host reference: fixed reduction tree, every floating-point
    /// operation an explicit round-to-nearest instruction, no fused multiply-add.
    RepOpsBitExact,
    /// Integer or comparison-only work, exact under any schedule.
    Exact,
}

/// A kernel compiled from `device/src/main.rs` by NVIDIA cuda-oxide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaKernel {
    Q8kQuantize,
    Q4kQ8kMatvec,
    Mxfp4Matvec,
    SwiGlu,
    MoeTopK,
    RowMask,
}

impl CudaKernel {
    pub const ALL: [CudaKernel; 6] = [
        Self::Q8kQuantize,
        Self::Q4kQ8kMatvec,
        Self::Mxfp4Matvec,
        Self::SwiGlu,
        Self::MoeTopK,
        Self::RowMask,
    ];

    /// The `#[kernel]` function name, which cuda-oxide also uses for the PTX entry.
    pub const fn entry(self) -> &'static str {
        match self {
            Self::Q8kQuantize => "q8k_quantize",
            Self::Q4kQ8kMatvec => "q4k_q8k_matvec",
            Self::Mxfp4Matvec => "mxfp4_matvec",
            Self::SwiGlu => "swiglu",
            Self::MoeTopK => "moe_top_k",
            Self::RowMask => "row_mask",
        }
    }

    pub const fn numerics(self) -> Numerics {
        match self {
            Self::MoeTopK | Self::RowMask => Numerics::Exact,
            _ => Numerics::RepOpsBitExact,
        }
    }

    pub const fn summary(self) -> &'static str {
        match self {
            Self::Q8kQuantize => {
                "Quantize activations to Q8_K once per token, so every expert reuses the codes"
            }
            Self::Q4kQ8kMatvec => {
                "Q4_K weights x Q8_K activations, integer block dots, RepOps tree"
            }
            Self::Mxfp4Matvec => "MXFP4 (E2M1 + E8M0) expert weights x f32 activations",
            Self::SwiGlu => "SiLU(gate) * up with the RepOps polynomial exp",
            Self::MoeTopK => "Deterministic router top-k: NaN as -f32::MAX, ties to the lower id",
            Self::RowMask => "Additive attention mask from per-row visible counts",
        }
    }

    /// The InferSSD CUDA source the kernel was adapted from, where there is one.
    pub const fn adapted_from(self) -> Option<&'static str> {
        match self {
            Self::Q8kQuantize => Some("engine/ggml/src/ggml-cuda/activation-quant-reuse.cu"),
            Self::Q4kQ8kMatvec => Some("engine/ggml/src/ggml-cuda/mmvq-q8k.cu"),
            Self::Mxfp4Matvec => Some("engine/ggml/src/ggml-cuda/ssd/mxfp4-kernels.cuh"),
            Self::SwiGlu => None,
            Self::MoeTopK => Some("engine/ggml/src/ggml-cuda/top-k.cu"),
            Self::RowMask => Some("engine/ggml/src/ggml-cuda/row-mask.cu"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_unique() {
        for (i, a) in CudaKernel::ALL.iter().enumerate() {
            for b in &CudaKernel::ALL[i + 1..] {
                assert_ne!(a.entry(), b.entry());
            }
        }
    }
}
