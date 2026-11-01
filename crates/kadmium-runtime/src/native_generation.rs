//! # `kadmium_runtime::native_generation`
//!
//! The coordinator loop that generates tokens across a chain of machines. Each machine holds a
//! contiguous range of a checkpoint's layers as a [`ChainStage`]; the coordinator sends the
//! prompt's token ids to the first stage, passes each stage's activations to the next, selects a
//! token from the last stage's logits with the model's own generation settings, streams it, and
//! feeds it back until the model stops.
//!
//! - **Streaming.** Every token is sent as a [`GenerationEvent`] as soon as it is selected, with
//!   the text it adds when a tokenizer is supplied.
//! - **Cancellation.** [`CancelToken::cancel`] stops the run at once: the stage call in flight is
//!   abandoned, every stage is told to cancel its session, and the run ends with
//!   [`FinishReason::Cancelled`]. Dropping the event receiver cancels the same way.
//! - **Failure.** A stage that fails, or answers with the wrong shape or position, stops the run
//!   with [`GenerationError::Stage`] naming the stage, its worker and its layers. Every stage's
//!   session is cancelled. Nothing is retried on another machine or a smaller model.
//!
//! Quality contract:
//!
//! - The context is the model's native context. A prompt that leaves no position to generate is
//!   an error naming both numbers; the prompt is never truncated.
//! - There is no output cap unless the operator sets [`GenerationRequest::max_new_tokens`]. The
//!   checkpoint's declared `max_new_tokens` and `max_length` are metadata, not limits. Without an
//!   operator limit a run ends at an EOS token, at a stop string, or when the sequence fills the
//!   native context ([`FinishReason::ContextFull`]). An operator limit above what the context
//!   leaves is an error naming both numbers, and a run that ends at it says so
//!   ([`FinishReason::OperatorLimit`]).
//! - Tokens are selected with the checkpoint's own settings. A declared setting the sampler does
//!   not implement is refused ([`GenerationError::UnsupportedSetting`]), never ignored; see
//!   [`sampler::UNIMPLEMENTED`].
//! - Where the layers are cut cannot change a token: the stages compute the same logits as one
//!   slice holding every layer, so the same seed selects the same tokens. **Proved (model)** as
//!   `Kadmium.Native.Generation.generate_chain_eq_whole` from `Kadmium.Native.runChain_eq_runLayers`;
//!   checked on the Rust code by the test `chain_generation_is_identical_at_every_cut`.
//!
//! The loop does not interleave several requests, overlap stages, or retry. A stage on another
//! machine implements [`ChainStage`] over its worker connection; this module defines the
//! interface and runs [`LocalStage`]s in this process.

pub mod sampler;
mod stage;
mod text;

pub use stage::{
    ChainStage, LocalStage, StageDescriptor, StageError, StageFuture, StageInput, StageOutput,
};
pub use text::held_back_len;

use crate::model_text::{GenerationDefaults, ModelText, ModelTextError, SamplingDefaults};
use sampler::Sampler;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use text::TextStream;
use thiserror::Error;
use tokio::sync::{mpsc, watch};

#[derive(Debug, Error)]
pub enum GenerationError {
    #[error("generation chain: {0}")]
    Chain(String),
    #[error("generation request: {0}")]
    Request(String),
    #[error(
        "the prompt holds {prompt} tokens; the model's native context of {maximum} positions leaves none to generate"
    )]
    NoRoomToGenerate { prompt: usize, maximum: usize },
    #[error(
        "max_new_tokens is {requested}; the native context leaves {maximum} positions after the prompt"
    )]
    LimitAboveMaximum { requested: usize, maximum: usize },
    #[error(
        "generation setting `{key}` = {value} is not implemented by the native generator; it is refused rather than ignored"
    )]
    UnsupportedSetting { key: String, value: String },
    #[error("generation setting `{key}` = {value} is invalid; it must satisfy {requirement}")]
    InvalidSetting {
        key: String,
        value: String,
        requirement: String,
    },
    #[error(
        "the logits for position {position} hold NaN or positive infinity, or no finite value, before or after the sampling processors"
    )]
    InvalidLogits { position: usize },
    #[error(transparent)]
    Stage(Box<StageFailure>),
    #[error("streamed text: {detail}")]
    TextUnstable { detail: String },
    #[error(transparent)]
    Text(#[from] ModelTextError),
    #[error("the generation task stopped: {0}")]
    Task(String),
}

/// A stage that failed or broke the stage contract, which stopped the run.
#[derive(Debug, Error)]
#[error(
    "stage {stage} on {worker} (layers {}..{}) failed at position {position}: {source}{}",
    .layers.start, .layers.end, cleanup_note(.cleanup)
)]
pub struct StageFailure {
    pub stage: usize,
    pub worker: String,
    pub layers: Range<usize>,
    pub position: usize,
    pub tokens_generated: usize,
    pub source: StageError,
    /// Stages whose cancellation also failed.
    pub cleanup: Vec<String>,
}

fn cleanup_note(cleanup: &[String]) -> String {
    if cleanup.is_empty() {
        String::new()
    } else {
        format!(
            "; cancelling the session also failed on {}",
            cleanup.join("; ")
        )
    }
}

/// A cancellation signal shared by the caller and the run.
#[derive(Debug, Clone)]
pub struct CancelToken(Arc<watch::Sender<bool>>);

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(watch::channel(false).0))
    }

    pub fn cancel(&self) {
        self.0.send_replace(true);
    }

    pub fn is_cancelled(&self) -> bool {
        *self.0.borrow()
    }

    async fn cancelled(&self) {
        let mut receiver = self.0.subscribe();
        // The sender lives in `self`, so the wait cannot end with the channel closed.
        let _ = receiver.wait_for(|cancelled| *cancelled).await;
    }
}

/// The checked chain of stages for one generation.
pub struct Chain {
    stages: Vec<Box<dyn ChainStage>>,
    hidden_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
}

impl Chain {
    /// Checks that the stages hold one model's layers in order, the first with the token embedding
    /// and the last with the LM head, with no layer missing or held twice.
    pub fn new(stages: Vec<Box<dyn ChainStage>>) -> Result<Self, GenerationError> {
        let invalid = |message: String| Err(GenerationError::Chain(message));
        let Some(first) = stages.first().map(|s| s.descriptor().clone()) else {
            return invalid("a chain needs at least one stage".into());
        };
        let count = stages.len();
        let mut next_layer = 0;
        for (index, stage) in stages.iter().enumerate() {
            let d = stage.descriptor();
            let name = format!("stage {index} on {}", d.worker);
            if d.model_id != first.model_id
                || d.num_layers != first.num_layers
                || d.hidden_size != first.hidden_size
                || d.vocab_size != first.vocab_size
                || d.max_position_embeddings != first.max_position_embeddings
            {
                return invalid(format!(
                    "{name} holds `{}` ({} layers, hidden {}, vocabulary {}, context {}); stage 0 holds `{}` ({} layers, hidden {}, vocabulary {}, context {})",
                    d.model_id,
                    d.num_layers,
                    d.hidden_size,
                    d.vocab_size,
                    d.max_position_embeddings,
                    first.model_id,
                    first.num_layers,
                    first.hidden_size,
                    first.vocab_size,
                    first.max_position_embeddings
                ));
            }
            if d.plan.embedding != (index == 0) {
                return invalid(if index == 0 {
                    format!("{name} must hold the token embedding")
                } else {
                    format!("{name} holds the token embedding but is not the first stage")
                });
            }
            if d.plan.head != (index + 1 == count) {
                return invalid(if index + 1 == count {
                    format!("{name} must hold the final norm and LM head")
                } else {
                    format!("{name} holds the LM head but is not the last stage")
                });
            }
            if d.plan.layers.start != next_layer || d.plan.layers.end < d.plan.layers.start {
                return invalid(format!(
                    "{name} holds layers {}..{}; the chain needs layer {next_layer} next",
                    d.plan.layers.start, d.plan.layers.end
                ));
            }
            next_layer = d.plan.layers.end;
        }
        if next_layer != first.num_layers {
            return invalid(format!(
                "the stages hold layers 0..{next_layer} of the model's {}",
                first.num_layers
            ));
        }
        Ok(Self {
            stages,
            hidden_size: first.hidden_size,
            vocab_size: first.vocab_size,
            max_position_embeddings: first.max_position_embeddings,
        })
    }

    pub fn descriptors(&self) -> Vec<StageDescriptor> {
        self.stages.iter().map(|s| s.descriptor().clone()).collect()
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    /// One pass through every stage. Returns the logits for the position after the input, or
    /// the failing stage.
    async fn step(&mut self, input: StageInput) -> Result<Vec<f32>, (usize, StageError)> {
        let start = input.start_position();
        let rows = input.rows();
        let last_position = start + rows - 1;
        let count = self.stages.len();
        let mut current = input;
        for (index, stage) in self.stages.iter_mut().enumerate() {
            let output = stage.forward(current).await.map_err(|e| (index, e))?;
            let contract = |message: String| Err((index, StageError::Contract(message)));
            match (output, index + 1 == count) {
                (StageOutput::Hidden(activations), false) => {
                    if activations.start_position() != start
                        || activations.rows() != rows
                        || activations.hidden_size() != self.hidden_size
                    {
                        return contract(format!(
                            "returned {} rows of width {} from position {}; sent {rows} rows of width {} from position {start}",
                            activations.rows(),
                            activations.hidden_size(),
                            activations.start_position(),
                            self.hidden_size
                        ));
                    }
                    current = StageInput::Hidden(activations);
                }
                (StageOutput::Logits { position, values }, true) => {
                    if position != last_position || values.len() != self.vocab_size {
                        return contract(format!(
                            "returned {} logits for position {position}; expected {} for position {last_position}",
                            values.len(),
                            self.vocab_size
                        ));
                    }
                    return Ok(values);
                }
                (StageOutput::Hidden(_), true) => {
                    return contract("the last stage returned activations, not logits".into());
                }
                (StageOutput::Logits { .. }, false) => {
                    return contract("returned logits but is not the last stage".into());
                }
            }
        }
        // `Chain::new` refuses an empty chain.
        Err((0, StageError::Contract("the chain has no stages".into())))
    }

    /// Cancels every stage's session; returns the stages whose cancellation failed.
    async fn cancel_all(&mut self) -> Vec<String> {
        let mut failures = Vec::new();
        for (index, stage) in self.stages.iter_mut().enumerate() {
            if let Err(error) = stage.cancel().await {
                failures.push(format!(
                    "stage {index} on {}: {error}",
                    stage.descriptor().worker
                ));
            }
        }
        failures
    }

    async fn close_all(&mut self) -> Vec<String> {
        let mut failures = Vec::new();
        for (index, stage) in self.stages.iter_mut().enumerate() {
            if let Err(error) = stage.close().await {
                failures.push(format!(
                    "stage {index} on {}: {error}",
                    stage.descriptor().worker
                ));
            }
        }
        failures
    }
}

/// The model's generation settings, and its tokenizer when text is wanted.
#[derive(Clone)]
pub struct GenerationSettings {
    pub defaults: GenerationDefaults,
    pub text: Option<TextSettings>,
}

#[derive(Clone)]
pub struct TextSettings {
    pub model: Arc<ModelText>,
    /// Hide special tokens such as EOS and template markers in the streamed text.
    pub skip_special_tokens: bool,
}

impl GenerationSettings {
    /// Token ids only. A model that declares stop strings needs its tokenizer and is refused.
    pub fn tokens_only(defaults: GenerationDefaults) -> Self {
        Self {
            defaults,
            text: None,
        }
    }

    /// The checkpoint's generation settings and tokenizer.
    pub fn with_text(model: Arc<ModelText>, skip_special_tokens: bool) -> Self {
        Self {
            defaults: model.generation_defaults().clone(),
            text: Some(TextSettings {
                model,
                skip_special_tokens,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationRequest {
    pub prompt: Vec<u32>,
    /// Unset: no output limit; the run ends at an EOS token, a stop string or the native context.
    /// Set: responses stop at this many tokens, the operator's choice.
    pub max_new_tokens: Option<NonZeroUsize>,
    /// Seed for sampling. Unset: drawn from the operating system and recorded in the report.
    pub seed: Option<u64>,
}

impl GenerationRequest {
    pub fn new(prompt: Vec<u32>) -> Self {
        Self {
            prompt,
            max_new_tokens: None,
            seed: None,
        }
    }
}

/// One generated token.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationEvent {
    /// 0 for the first generated token.
    pub index: usize,
    /// The token's position in the sequence.
    pub position: usize,
    pub token: u32,
    /// Text this token adds. Empty while text is held back, and when no tokenizer is supplied.
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// The model produced one of its EOS tokens, which is the last token.
    EndOfSequence {
        token: u32,
    },
    /// The text reached a stop string; the text ends before it.
    StopString {
        stop: String,
    },
    /// The operator's `max_new_tokens`.
    OperatorLimit {
        max_new_tokens: usize,
    },
    /// The sequence fills the model's native context.
    ContextFull {
        maximum: usize,
    },
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationReport {
    pub prompt_tokens: usize,
    pub tokens: Vec<u32>,
    /// The generated text, when a tokenizer was supplied. After a cancellation, the text sent.
    pub text: Option<String>,
    pub finish: FinishReason,
    /// The settings tokens were selected with.
    pub sampling: SamplingDefaults,
    /// The seed, when the model samples.
    pub seed: Option<u64>,
    /// The operator's output limit, if one was set.
    pub max_new_tokens: Option<usize>,
    pub max_position_embeddings: usize,
    pub stages: Vec<StageDescriptor>,
    /// Stages whose session did not close or cancel cleanly after the run ended.
    pub cleanup_failures: Vec<String>,
}

fn os_seed() -> Result<u64, GenerationError> {
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes).map_err(|e| {
        GenerationError::Request(format!(
            "no seed was given and the OS random number generator failed: {e}"
        ))
    })?;
    Ok(u64::from_le_bytes(bytes))
}

/// Checks the request against the chain and the model before any stage is called.
fn validate(
    chain: &Chain,
    settings: &GenerationSettings,
    request: &GenerationRequest,
) -> Result<(), GenerationError> {
    sampler::check_declared(&settings.defaults.declared)?;
    sampler::check_sampling(&settings.defaults.sampling)?;
    if settings.text.is_none() && !settings.defaults.stop_strings.is_empty() {
        return Err(GenerationError::UnsupportedSetting {
            key: "stop_strings".into(),
            value: format!("{:?}", settings.defaults.stop_strings),
        });
    }
    if let Some(text) = &settings.text {
        let config = text.model.config();
        if config.vocab_size != chain.vocab_size
            || config.max_position_embeddings != chain.max_position_embeddings
        {
            return Err(GenerationError::Chain(format!(
                "the tokenizer's model has vocabulary {} and context {}; the chain's has {} and {}",
                config.vocab_size,
                config.max_position_embeddings,
                chain.vocab_size,
                chain.max_position_embeddings
            )));
        }
    }
    let prompt = request.prompt.len();
    if prompt == 0 {
        return Err(GenerationError::Request(
            "the prompt holds no tokens".into(),
        ));
    }
    if let Some(&token) = request
        .prompt
        .iter()
        .find(|&&t| t as usize >= chain.vocab_size)
    {
        return Err(GenerationError::Request(format!(
            "prompt token {token} is outside the vocabulary of {}",
            chain.vocab_size
        )));
    }
    let maximum = chain.max_position_embeddings;
    if prompt >= maximum {
        return Err(GenerationError::NoRoomToGenerate { prompt, maximum });
    }
    if let Some(limit) = request.max_new_tokens
        && limit.get() > maximum - prompt
    {
        return Err(GenerationError::LimitAboveMaximum {
            requested: limit.get(),
            maximum: maximum - prompt,
        });
    }
    Ok(())
}

/// Generates across `chain`, sending each token on `events`, until the model stops, the operator
/// limit is reached, the context is full, or `cancel` fires. The chain's sessions are closed when
/// the run completes and cancelled on any other outcome.
pub async fn generate(
    mut chain: Chain,
    settings: &GenerationSettings,
    request: GenerationRequest,
    cancel: &CancelToken,
    events: &mpsc::UnboundedSender<GenerationEvent>,
) -> Result<GenerationReport, GenerationError> {
    if let Err(error) = validate(&chain, settings, &request) {
        chain.cancel_all().await;
        return Err(error);
    }
    let seed = if settings.defaults.sampling.do_sample {
        match request.seed.map_or_else(os_seed, Ok) {
            Ok(seed) => Some(seed),
            Err(error) => {
                chain.cancel_all().await;
                return Err(error);
            }
        }
    } else {
        None
    };
    let mut stream = match &settings.text {
        Some(text) => match TextStream::new(
            &text.model,
            &request.prompt,
            text.skip_special_tokens,
            &settings.defaults.stop_strings,
        ) {
            Ok(stream) => Some(stream),
            Err(error) => {
                chain.cancel_all().await;
                return Err(error);
            }
        },
        None => None,
    };
    let mut sampler = Sampler::new(settings.defaults.sampling.clone(), seed.unwrap_or(0));
    let maximum = chain.max_position_embeddings;
    let limit = request.max_new_tokens.map(NonZeroUsize::get);
    let prompt_tokens = request.prompt.len();
    let mut context = request.prompt;
    let mut generated = Vec::new();
    let mut input = StageInput::Tokens {
        start_position: 0,
        tokens: context.clone(),
    };

    let outcome: Result<(FinishReason, Option<String>), GenerationError> = loop {
        let position = context.len();
        let step = tokio::select! {
            biased;
            () = cancel.cancelled() => None,
            () = events.closed() => None,
            result = chain.step(input) => Some(result),
        };
        let logits = match step {
            None => break Ok((FinishReason::Cancelled, None)),
            Some(Ok(logits)) => logits,
            Some(Err((stage, source))) => {
                let d = chain.stages[stage].descriptor();
                break Err(GenerationError::Stage(Box::new(StageFailure {
                    stage,
                    worker: d.worker.clone(),
                    layers: d.layers(),
                    position,
                    tokens_generated: generated.len(),
                    source,
                    cleanup: Vec::new(),
                })));
            }
        };
        let token = match sampler.select(&logits, &context, position) {
            Ok(token) => token,
            Err(error) => break Err(error),
        };
        generated.push(token);
        context.push(token);
        let eos = settings.defaults.is_eos(token);
        let at_limit = limit == Some(generated.len());
        let full = context.len() == maximum;
        let mut finish = if eos {
            Some(FinishReason::EndOfSequence { token })
        } else if at_limit {
            limit.map(|max_new_tokens| FinishReason::OperatorLimit { max_new_tokens })
        } else if full {
            Some(FinishReason::ContextFull { maximum })
        } else {
            None
        };
        let mut delta = String::new();
        if let Some(stream) = stream.as_mut() {
            let step = if eos {
                stream
                    .finish()
                    .map(|delta| text::Step { delta, stop: None })
            } else {
                stream.push(token)
            };
            match step {
                Ok(step) => {
                    delta = step.delta;
                    if let Some(stop) = step.stop {
                        finish = Some(FinishReason::StopString { stop });
                    } else if finish.is_some() && !eos {
                        match stream.finish() {
                            Ok(rest) => delta.push_str(&rest),
                            Err(error) => break Err(error),
                        }
                    }
                }
                Err(error) => break Err(error),
            }
        }
        let event = GenerationEvent {
            index: generated.len() - 1,
            position,
            token,
            text: delta,
        };
        if events.send(event).is_err() {
            break Ok((FinishReason::Cancelled, None));
        }
        if let Some(finish) = finish {
            break Ok((finish, stream.as_ref().map(|s| s.sent().to_owned())));
        }
        input = StageInput::Tokens {
            start_position: position,
            tokens: vec![token],
        };
    };

    match outcome {
        Ok((finish, text)) => {
            let cleanup_failures = if finish == FinishReason::Cancelled {
                chain.cancel_all().await
            } else {
                chain.close_all().await
            };
            let text = text.or_else(|| stream.as_ref().map(|s| s.sent().to_owned()));
            Ok(GenerationReport {
                prompt_tokens,
                tokens: generated,
                text,
                finish,
                sampling: settings.defaults.sampling.clone(),
                seed,
                max_new_tokens: limit,
                max_position_embeddings: maximum,
                stages: chain.descriptors(),
                cleanup_failures,
            })
        }
        Err(mut error) => {
            let failures = chain.cancel_all().await;
            if let GenerationError::Stage(failure) = &mut error {
                failure.cleanup = failures;
            }
            Err(error)
        }
    }
}

/// A generation running on its own task.
pub struct GenerationRun {
    pub events: mpsc::UnboundedReceiver<GenerationEvent>,
    pub cancel: CancelToken,
    task: tokio::task::JoinHandle<Result<GenerationReport, GenerationError>>,
}

impl GenerationRun {
    /// Waits for the run to end. The receiver stays open until then, so waiting does not cancel
    /// the run; events not yet received are discarded.
    pub async fn finish(self) -> Result<GenerationReport, GenerationError> {
        let Self { events, task, .. } = self;
        let result = task.await;
        drop(events);
        result.map_err(|e| GenerationError::Task(e.to_string()))?
    }
}

/// Starts [`generate`] on a new task. The event channel holds at most one event per position of
/// the native context.
pub fn spawn_generation(
    chain: Chain,
    settings: GenerationSettings,
    request: GenerationRequest,
) -> GenerationRun {
    let (sender, events) = mpsc::unbounded_channel();
    let cancel = CancelToken::new();
    let token = cancel.clone();
    let task =
        tokio::spawn(async move { generate(chain, &settings, request, &token, &sender).await });
    GenerationRun {
        events,
        cancel,
        task,
    }
}
