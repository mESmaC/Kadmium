//! One hop of a generation chain: the layers one machine holds, behind a transport-neutral
//! interface. [`LocalStage`] runs a [`DecoderSlice`] in this process. A stage on another machine
//! implements the same trait over its worker connection.

use crate::native_decoder::{Activations, DecoderSlice, KvCache, NativeDecoderError, SlicePlan};
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;

/// A stage call in flight.
pub type StageFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, StageError>> + Send + 'a>>;

#[derive(Debug, Error)]
pub enum StageError {
    #[error(transparent)]
    Decoder(#[from] NativeDecoderError),
    /// The worker or its connection failed.
    #[error("{0}")]
    Worker(String),
    /// The stage answered, but not with what the request requires.
    #[error("{0}")]
    Contract(String),
}

/// What a stage holds and which model it belongs to. The coordinator checks that a chain's
/// descriptors describe one model, cut into contiguous ranges, before it sends anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageDescriptor {
    /// The machine or worker that runs the stage, as shown to the operator.
    pub worker: String,
    /// Identity of the checkpoint. Every stage of a chain must name the same one.
    pub model_id: String,
    pub plan: SlicePlan,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

impl StageDescriptor {
    pub fn for_slice(
        worker: impl Into<String>,
        model_id: impl Into<String>,
        slice: &DecoderSlice,
    ) -> Self {
        let config = slice.config();
        Self {
            worker: worker.into(),
            model_id: model_id.into(),
            plan: slice.plan().clone(),
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            max_position_embeddings: config.max_position_embeddings,
        }
    }

    pub fn layers(&self) -> Range<usize> {
        self.plan.layers.clone()
    }
}

/// What the coordinator sends a stage: token ids to the stage that holds the embedding, and the
/// previous stage's activations to every other.
#[derive(Debug, Clone, PartialEq)]
pub enum StageInput {
    Tokens {
        start_position: usize,
        tokens: Vec<u32>,
    },
    Hidden(Activations),
}

impl StageInput {
    pub fn start_position(&self) -> usize {
        match self {
            Self::Tokens { start_position, .. } => *start_position,
            Self::Hidden(activations) => activations.start_position(),
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Tokens { tokens, .. } => tokens.len(),
            Self::Hidden(activations) => activations.rows(),
        }
    }
}

/// What a stage returns: its activations, or, from the stage that holds the LM head, the
/// logits of the input's last row.
#[derive(Debug, Clone, PartialEq)]
pub enum StageOutput {
    Hidden(Activations),
    Logits { position: usize, values: Vec<f32> },
}

/// One stage of a generation chain, holding its own KV cache for one generation. A stage is a
/// session: the coordinator calls `forward` once per step, then exactly one of `close` (the run
/// completed) or `cancel` (the run was cancelled or another stage failed).
pub trait ChainStage: Send {
    fn descriptor(&self) -> &StageDescriptor;

    /// Runs the stage's layers over `input`, which starts at the position its cache has reached,
    /// and appends to the cache. A stage that holds the LM head returns the logits of the last
    /// input row; every other stage returns its activations. On error the cache must be left as
    /// it was, or the session ended.
    fn forward(&mut self, input: StageInput) -> StageFuture<'_, StageOutput>;

    /// Ends the session after a completed run and releases its cache.
    fn close(&mut self) -> StageFuture<'_, ()>;

    /// Abandons the session. It can arrive while the worker is still running a forward pass whose
    /// future the coordinator has dropped.
    fn cancel(&mut self) -> StageFuture<'_, ()>;
}

/// A stage that runs a [`DecoderSlice`] in this process, on a blocking thread.
///
/// Cancellation takes effect at the end of the forward pass in flight: the pass runs to the end
/// of the slice's layers on its blocking thread and its result is discarded.
pub struct LocalStage {
    descriptor: StageDescriptor,
    slice: Arc<DecoderSlice>,
    cache: Option<KvCache>,
}

impl LocalStage {
    pub fn new(
        worker: impl Into<String>,
        model_id: impl Into<String>,
        slice: Arc<DecoderSlice>,
    ) -> Self {
        let descriptor = StageDescriptor::for_slice(worker, model_id, &slice);
        let cache = Some(slice.new_cache());
        Self {
            descriptor,
            slice,
            cache,
        }
    }

    /// Positions in the stage's cache, or `None` once the session has ended.
    pub fn cache_positions(&self) -> Option<usize> {
        self.cache.as_ref().map(KvCache::positions)
    }
}

impl ChainStage for LocalStage {
    fn descriptor(&self) -> &StageDescriptor {
        &self.descriptor
    }

    fn forward(&mut self, input: StageInput) -> StageFuture<'_, StageOutput> {
        Box::pin(async move {
            let mut cache = self
                .cache
                .take()
                .ok_or_else(|| StageError::Contract("the stage's session has ended".into()))?;
            let slice = Arc::clone(&self.slice);
            let (cache, result) = tokio::task::spawn_blocking(move || {
                let result = run_slice(&slice, input, &mut cache);
                (cache, result)
            })
            .await
            .map_err(|e| StageError::Worker(format!("the stage's compute thread failed: {e}")))?;
            self.cache = Some(cache);
            result
        })
    }

    fn close(&mut self) -> StageFuture<'_, ()> {
        self.cache = None;
        Box::pin(async { Ok(()) })
    }

    fn cancel(&mut self) -> StageFuture<'_, ()> {
        self.cache = None;
        Box::pin(async { Ok(()) })
    }
}

/// Every error is raised before `forward` appends to the cache, so a failed call leaves it as it
/// was. After it, the head is present whenever the plan holds it and the rows have the model's
/// width, so the logits cannot fail.
fn run_slice(
    slice: &DecoderSlice,
    input: StageInput,
    cache: &mut KvCache,
) -> Result<StageOutput, StageError> {
    let plan = slice.plan();
    let hidden = match input {
        StageInput::Tokens {
            start_position,
            tokens,
        } => slice.embed(&tokens, start_position)?,
        StageInput::Hidden(_) if plan.embedding => {
            return Err(StageError::Contract(
                "this stage holds the token embedding and takes token ids, not activations".into(),
            ));
        }
        StageInput::Hidden(activations) => activations,
    };
    let output = slice.forward(&hidden, cache)?;
    if !plan.head {
        return Ok(StageOutput::Hidden(output));
    }
    // Rows are independent through the final norm and the LM head, so the last row alone gives
    // the same logits it would among all of them. Not proved: checked bit for bit by the test
    // `last_row_logits_equal_the_full_pass_bit_for_bit`.
    let (rows, width) = (output.rows(), output.hidden_size());
    let position = output.start_position() + rows - 1;
    let last = output.into_data().split_off((rows - 1) * width);
    let values = slice.logits(&Activations::new(position, width, last)?)?;
    Ok(StageOutput::Logits { position, values })
}
