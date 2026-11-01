//! # `kadmium_runtime::native_decoder`
//!
//! Kadmium's own execution of a real checkpoint: dense decoder-only transformers (Llama, Mistral,
//! Qwen2, Qwen3) read from their Hugging Face SafeTensors files and run in pure Rust, one
//! contiguous range of layers at a time. This is the unit distributed inference is built from:
//! a machine holds a [`DecoderSlice`] for its layers and a [`KvCache`] for them, takes the
//! previous machine's [`Activations`], and hands its own to the next.
//!
//! Quality contract:
//!
//! - Weights stay in their stored F32, F16 or BF16 bits. Arithmetic is `f32`, every narrower
//!   weight widened exactly. Nothing is quantized, pruned or skipped; another dtype, or a
//!   checkpoint quantized before import, is refused.
//! - Keys and values are cached in `f32`.
//! - The context is the model's `max_position_embeddings`. A request beyond it is an error
//!   naming both numbers. Nothing is truncated or shifted.
//! - A configuration feature that cannot be executed exactly, such as hybrid attention or
//!   an unimplemented RoPE scaling type, is an error, never an approximation.
//!
//! Determinism: the arithmetic follows the RepOps conventions plus
//! [`math::NATIVE_DECODER_CONVENTIONS`]. Where the layers are cut cannot change an output:
//! **Proved (model)** as `Kadmium.Native.runChain_eq_runLayers` for any layer that reads only its
//! input and its own cache, with that property of this code checked by the test
//! `splitting_the_layers_at_every_cut_is_bit_identical`. The thread count cannot change an output
//! by construction, since each output feature is computed whole by one thread, and the test
//! `linear_is_identical_at_every_thread_count` pins 1 to 1,000 threads.
//!
//! Status: reference execution on the CPU, with byte-exact multi-machine layer placement
//! ([`plan_checkpoint_placement`]) and authenticated `KDNS` session frames over the worker
//! connection (`WireFrameKind::NativeDecoderSession`). `native_generation` generates across a
//! chain of slices in one process, and speed has not been measured.

mod config;
pub mod math;
mod model;
mod placement;
pub mod session;
pub mod staging;

pub use config::{Architecture, DecoderConfig, RopeScaling};
pub use kadmium_sharding::{
    CheckpointMemoryProfile, LayerPlacementError, LayerPlacementPlan, LayerPlacementPlanner,
    MachineMemoryBudget, PlannedLayerStage,
};
pub use model::{Activations, DecoderSlice, KvCache, SlicePlan, inverse_frequencies};
pub use placement::{
    DecoderPlacementPlan, PlannedDecoderSlice, checkpoint_memory_profile,
    plan_checkpoint_placement, plan_checkpoint_placement_with_budgets,
    plan_placement_from_manifest, plan_placement_with_budgets,
};
pub use session::{
    DEFAULT_MAX_ACTIVE_NATIVE_SESSIONS, DEFAULT_MAX_NATIVE_SESSION_FRAME_BYTES,
    DEFAULT_MAX_NATIVE_SESSION_METADATA_BYTES, DEFAULT_MAX_NATIVE_SESSION_PAYLOAD_BYTES,
    ForwardInput, ForwardOutput, MAX_IN_FLIGHT_NATIVE_SESSION_FRAMES, NATIVE_SESSION_HEADER_BYTES,
    NATIVE_SESSION_MAC_BYTES, NATIVE_SESSION_MAGIC, NATIVE_SESSION_MAX_MODEL_ID_BYTES,
    NATIVE_SESSION_MAX_SESSION_ID_BYTES, NATIVE_SESSION_MIN_KEY_BYTES, NATIVE_SESSION_VERSION,
    NativeSessionHost, NativeSessionLimits, SessionAuthKey, SessionCancelRequest,
    SessionCancelResponse, SessionCloseRequest, SessionCloseResponse, SessionForwardRequest,
    SessionForwardResponse, SessionOpenRequest, SessionOpenResponse, SessionRequest,
    SessionResponse, decode_envelope, decode_f32_tensor, decode_u32_slice, encode_envelope,
    encode_f32_tensor, encode_u32_slice, fnv1a64_f32_le, hmac_sha256, is_priority_control_frame,
};

use kadmium_data::SafetensorsManifestError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NativeDecoderError {
    #[error("model config: {0}")]
    Config(String),
    #[error("the native decoder does not support {0}")]
    Unsupported(String),
    #[error("tensor `{0}` is missing from the checkpoint")]
    MissingTensor(String),
    #[error("tensor `{tensor}` has shape {actual:?}; the config requires {expected:?}")]
    TensorShape {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<u64>,
    },
    #[error(
        "tensor `{tensor}` is stored as {dtype}; the native decoder runs F32, F16 and BF16 weights as stored and does not dequantize or cast"
    )]
    UnsupportedDtype { tensor: String, dtype: &'static str },
    #[error("layers {start}..{end} are not within the model's {layers} layers")]
    LayerRange {
        start: usize,
        end: usize,
        layers: usize,
    },
    #[error("this slice does not hold the {0}")]
    MissingPart(&'static str),
    #[error("token {token} is outside the vocabulary of {vocab_size}")]
    TokenOutOfRange { token: u32, vocab_size: usize },
    #[error(
        "the request reaches position {requested}; the model's native context is {maximum} positions"
    )]
    ContextExceeded { requested: usize, maximum: usize },
    #[error("the cache holds {expected} positions; the activations start at position {actual}")]
    CachePosition { expected: usize, actual: usize },
    #[error("could not allocate {bytes} bytes for the KV cache")]
    Memory { bytes: usize },
    #[error("native decoder session authentication failed: {0}")]
    Authentication(String),
    #[error("native decoder session frame is {actual} bytes; the limit is {limit} bytes")]
    FrameTooLarge { actual: usize, limit: usize },
    #[error("native decoder session frame: {0}")]
    Wire(String),
    #[error("native decoder session `{0}` was cancelled")]
    Cancelled(String),
    #[error("native decoder session: {0}")]
    Session(String),
    #[error("{0}")]
    Shape(String),
    #[error(transparent)]
    Placement(#[from] LayerPlacementError),
    #[error(transparent)]
    Checkpoint(#[from] SafetensorsManifestError),
}

pub type Result<T> = std::result::Result<T, NativeDecoderError>;
