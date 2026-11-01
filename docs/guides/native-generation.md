# Native generation across a chain of machines

Item 6 of the distributed inference work is the `kadmium_runtime::native_generation` library: the
coordinator loop that generates tokens across a chain of stages, streams them, and can be
cancelled. Each stage holds a contiguous range of a checkpoint's layers. The first stage also holds
the token embedding, and the last holds the final norm and LM head. This is a library integration
point. It adds no CLI, HTTP route or Condenser option.

What runs today: `LocalStage` runs a `native_decoder::DecoderSlice` on a blocking thread in the
coordinator's process. A stage on another machine implements the same `ChainStage` trait over its
worker connection. The authenticated session frames for that connection are #83, which is still
open, so no stage carries activations between machines yet.

```rust,ignore
use kadmium_runtime::model_text::ModelText;
use kadmium_runtime::native_decoder::{DecoderSlice, SlicePlan};
use kadmium_runtime::native_generation::{
    Chain, ChainStage, GenerationRequest, GenerationSettings, LocalStage, spawn_generation,
};
use std::sync::Arc;

let text = Arc::new(ModelText::from_model_dir(dir)?);
let stages: Vec<Box<dyn ChainStage>> = vec![
    Box::new(LocalStage::new("desktop", "my-model", Arc::new(DecoderSlice::load(dir, SlicePlan {
        layers: 0..16, embedding: true, head: false,
    })?))),
    Box::new(LocalStage::new("mac-mini", "my-model", Arc::new(DecoderSlice::load(dir, SlicePlan {
        layers: 16..32, embedding: false, head: true,
    })?))),
];
let prompt = text.encode_chat(&messages, true, None, &variables, template_fuel)?;
let mut run = spawn_generation(
    Chain::new(stages)?,
    GenerationSettings::with_text(Arc::clone(&text), true),
    GenerationRequest::new(prompt),
);
while let Some(event) = run.events.recv().await {
    print!("{}", event.text); // `run.cancel.cancel()` stops the run from any task
}
let report = run.finish().await?;
```

## How a run proceeds

`Chain::new` checks the stages before anything is sent. Every stage must name the same `model_id`
and the same layer count, hidden size, vocabulary and context. The first stage must hold the
embedding and the last the LM head, and the layer ranges must follow one another with no gap and
no overlap.

Each step sends token ids to the first stage, passes each stage's activations to the next, and
takes the logits for the last position from the last stage. The coordinator checks every answer:
activations must start at the position sent, with the rows and width sent, and logits must be for
the last position, one per vocabulary entry. The selected token is sent as a `GenerationEvent`
and fed back as the next step's input. The prefill is the first step.

A run ends with one of these `FinishReason`s:

| Finish reason | Meaning |
| :--- | :--- |
| `EndOfSequence { token }` | The model produced one of its EOS tokens, the last token of the run. |
| `StopString { stop }` | The text reached one of the checkpoint's stop strings. The text ends before it. |
| `OperatorLimit { max_new_tokens }` | The run reached the operator's `max_new_tokens`. |
| `ContextFull { maximum }` | The sequence fills the model's native context. |
| `Cancelled` | `CancelToken::cancel` fired, or the event receiver was dropped. |

When a run completes, every stage's session is closed. On any other outcome every session is
cancelled. A cancellation abandons the stage call in flight at once. A `LocalStage` still finishes
the forward pass on its blocking thread and discards the result.

A stage that fails, or answers with the wrong position, row count, width or logit count, stops the
run with `GenerationError::Stage`. The error names the stage index, its worker, its layers, the
position and the number of tokens already generated, and lists any stage whose cancellation also
failed. Nothing is retried, rerouted to another machine or replaced with a smaller model.

## Quality contract

- **Context.** The model's native context applies. A prompt that leaves no position to generate
  fails with an error that names the prompt length and the context. Nothing is truncated.
- **Output length.** There is no output cap by default. `GenerationRequest::max_new_tokens` is
  blank by default, which means the run continues until an EOS token, a stop string or the full
  context. If you set it, responses stop at that many tokens. A value above what the context leaves
  after the prompt fails with an error that names both numbers. The report records the limit you
  set, and a run that ends at it reports `OperatorLimit`. The checkpoint's declared
  `max_new_tokens` and `max_length` are metadata, not limits.
- **Token selection.** Tokens are selected with the checkpoint's own settings, in Transformers'
  order: the repetition penalty, which also applies to greedy decoding, then, when `do_sample` is
  set, temperature, top-k, top-p, min-p and typical-p, and then a draw. Without `do_sample` the
  token is the first index of the largest logit. Any other declared setting that changes which
  tokens are produced must hold its inactive value; the list is `sampler::UNIMPLEMENTED`, and
  includes beam search, n-gram blocking, suppressed tokens, forced tokens and `max_time`. An active
  one is refused with `UnsupportedSetting`, not ignored. A value Transformers itself rejects, such
  as `typical_p = 0` when sampling, is refused with `InvalidSetting`. A model that declares stop
  strings needs its tokenizer, and is refused without one. Logits are an error if they hold NaN or
  positive infinity, or no finite value, both as the model returns them and after the repetition
  penalty and temperature. A temperature small enough to overflow a logit is an error, not a
  token.
- **Seeds.** Sampling draws from SplitMix64. A run without a seed takes one from the operating
  system, and the report records it. The same seed on the same coordinator selects the same tokens.
  Probabilities are computed in `f64` with the platform's `exp`, so a draw near a boundary could
  differ between coordinator platforms.
- **Cut invariance.** Where the layers are cut cannot change a token. **Proved (model)** as
  `Kadmium.Native.Generation.generate_chain_eq_whole`, building on
  `Kadmium.Native.runChain_eq_runLayers`. The link to the Rust code is the test
  `chain_generation_is_identical_at_every_cut`, which runs four checkpoint layouts at every pair of
  cuts, greedily against an independent decoding loop and with sampling from one seed.

## Streamed text

With a tokenizer, each event carries the text its token adds. The tokenizer is not an incremental
decoder, so each step decodes the whole sequence without the tokenizer's clean-up step, takes the
text after the decoded prompt, and applies the clean-up to that text alone. A clean-up replacement
therefore never reaches across the prompt boundary: a prompt ending in a space followed by a
generated `.` streams `.`, as Transformers' `TextStreamer` does. Text that a later token could
still change is held back:

- a trailing U+FFFD, which the next byte token may complete into a character;
- of what remains, the longest suffix that is the start of a stop string, or of one of the
  tokenizer's clean-up replacements, which remove a space before punctuation.

Held-back text is sent when the run ends, unless the run ends at a stop string. What is sent only
grows, and never reaches into a stop string that a later token completes. **Proved (model)** as
`Kadmium.Native.Generation.emitted_prefix` and `Kadmium.Native.Generation.emitLen_le_stop`, with
the differential fixture `generation_lean_vectors.txt`. If the decoded text ever stops extending
what was already sent, the run fails with `TextUnstable` rather than contradict it. The tokenizer
is outside the proof, so that check runs on every step.

Decoding the whole sequence on every step costs time in proportion to the sequence length, so a
run's total decoding cost grows with the square of its length. Its speed has not been measured.

## Current limitations

- One request per chain, one step at a time. The stages do not overlap work, and requests are not
  batched. Keeping all machines busy is item 8.
- No stage carries activations between machines until #83 lands its session frames and a
  `ChainStage` implementation over them. Placement, item 5, chooses the cuts.
- `LocalStage` cancellation waits for the forward pass in flight to finish on its blocking thread.
  The pass's result is discarded.
- The chain checks model identity by `model_id` and the model's dimensions. Two different
  checkpoints with the same dimensions and the same `model_id` are not told apart.
- No speed figure exists for this loop. Any figure needs a **Measured** entry in
  `docs/BENCHMARKS_2026-10.md`.
