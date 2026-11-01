//! The generation coordinator over chains of stages: cut invariance, streaming, cancellation,
//! stage failure, limits and refused settings.

use kadmium_runtime::model_text::{GenerationDefaults, ModelText, SamplingDefaults};
use kadmium_runtime::native_decoder::{Activations, DecoderConfig, DecoderSlice, SlicePlan};
use kadmium_runtime::native_generation::{
    CancelToken, Chain, ChainStage, FinishReason, GenerationError, GenerationEvent,
    GenerationReport, GenerationRequest, GenerationSettings, LocalStage, StageDescriptor,
    StageError, StageFuture, StageInput, StageOutput, generate, spawn_generation,
};
use serde_json::{Map, Value, json};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{Notify, mpsc};

const FIXTURES: [&str; 4] = ["llama3_bf16", "qwen2_f32", "qwen3_bf16", "mistral_f16"];
const PROMPT: [u32; 6] = [5, 17, 42, 3, 77, 9];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/native_decoder")
        .join(name)
}

fn greedy() -> SamplingDefaults {
    SamplingDefaults {
        do_sample: false,
        temperature: 1.0,
        top_k: 50,
        top_p: 1.0,
        min_p: None,
        typical_p: 1.0,
        repetition_penalty: 1.0,
    }
}

fn defaults(sampling: SamplingDefaults) -> GenerationDefaults {
    GenerationDefaults {
        sampling,
        eos_token_ids: Vec::new(),
        bos_token_id: None,
        pad_token_id: None,
        stop_strings: Vec::new(),
        declared: Map::new(),
    }
}

fn sampled() -> SamplingDefaults {
    SamplingDefaults {
        do_sample: true,
        temperature: 0.7,
        top_k: 20,
        top_p: 0.9,
        min_p: Some(0.01),
        typical_p: 0.95,
        repetition_penalty: 1.1,
    }
}

struct Model {
    dir: PathBuf,
    config: DecoderConfig,
}

impl Model {
    fn open(dir: PathBuf) -> Self {
        let config = DecoderConfig::from_model_dir(&dir).unwrap();
        Self { dir, config }
    }

    fn slice(&self, layers: std::ops::Range<usize>) -> Arc<DecoderSlice> {
        let plan = SlicePlan {
            embedding: layers.start == 0,
            head: layers.end == self.config.num_layers,
            layers,
        };
        Arc::new(DecoderSlice::load(&self.dir, plan).unwrap())
    }

    /// Stages cutting the layers at `cuts`; an empty range is a stage with no layers.
    fn stages(&self, cuts: &[usize]) -> Vec<Box<dyn ChainStage>> {
        let mut bounds = vec![0];
        bounds.extend_from_slice(cuts);
        bounds.push(self.config.num_layers);
        bounds
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                let plan = SlicePlan {
                    layers: w[0]..w[1],
                    embedding: i == 0,
                    head: i + 2 == bounds.len(),
                };
                let slice = Arc::new(DecoderSlice::load(&self.dir, plan).unwrap());
                Box::new(LocalStage::new(format!("machine-{i}"), "fixture", slice))
                    as Box<dyn ChainStage>
            })
            .collect()
    }
}

async fn run(
    stages: Vec<Box<dyn ChainStage>>,
    settings: &GenerationSettings,
    request: GenerationRequest,
) -> (
    Result<GenerationReport, GenerationError>,
    Vec<GenerationEvent>,
) {
    let chain = Chain::new(stages).unwrap();
    let (sender, mut receiver) = mpsc::unbounded_channel();
    let result = generate(chain, settings, request, &CancelToken::new(), &sender).await;
    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event);
    }
    (result, events)
}

/// Greedy decoding written directly against one whole-model slice, as an independent oracle.
fn oracle_greedy(model: &Model, prompt: &[u32]) -> Vec<u32> {
    let slice = model.slice(0..model.config.num_layers);
    let mut cache = slice.new_cache();
    let mut input = slice.embed(prompt, 0).unwrap();
    let mut out = Vec::new();
    let mut length = prompt.len();
    loop {
        let hidden = slice.forward(&input, &mut cache).unwrap();
        let logits = slice.logits(&hidden).unwrap();
        let last = &logits[logits.len() - model.config.vocab_size..];
        let mut best = 0;
        for (i, &v) in last.iter().enumerate() {
            if v > last[best] {
                best = i;
            }
        }
        out.push(best as u32);
        length += 1;
        if length == model.config.max_position_embeddings {
            return out;
        }
        input = slice.embed(&[best as u32], length - 1).unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chain_generation_is_identical_at_every_cut() {
    for name in FIXTURES {
        let model = Model::open(fixture(name));
        let layers = model.config.num_layers;
        let expected = oracle_greedy(&model, &PROMPT);
        let greedy_settings = GenerationSettings::tokens_only(defaults(greedy()));
        let sampled_settings = GenerationSettings::tokens_only(defaults(sampled()));
        let mut sampled_reference = None;
        for a in 0..=layers {
            for b in a..=layers {
                let cuts = [a, b];
                let (report, events) = run(
                    model.stages(&cuts),
                    &greedy_settings,
                    GenerationRequest::new(PROMPT.to_vec()),
                )
                .await;
                let report = report.unwrap();
                assert_eq!(report.tokens, expected, "{name} cuts {cuts:?}");
                assert_eq!(
                    report.finish,
                    FinishReason::ContextFull {
                        maximum: model.config.max_position_embeddings
                    }
                );
                assert_eq!(report.seed, None);
                let streamed: Vec<u32> = events.iter().map(|e| e.token).collect();
                assert_eq!(streamed, report.tokens);
                for (i, event) in events.iter().enumerate() {
                    assert_eq!((event.index, event.position), (i, PROMPT.len() + i));
                }

                let mut request = GenerationRequest::new(PROMPT.to_vec());
                request.seed = Some(0x5EED);
                let (report, _) = run(model.stages(&cuts), &sampled_settings, request).await;
                let report = report.unwrap();
                assert_eq!(report.seed, Some(0x5EED));
                match &sampled_reference {
                    None => sampled_reference = Some(report.tokens),
                    Some(reference) => assert_eq!(&report.tokens, reference, "{name} {cuts:?}"),
                }
            }
        }
    }
}

#[derive(Clone, Default)]
struct Log(Arc<Mutex<Vec<String>>>);

impl Log {
    fn push(&self, entry: String) {
        self.0.lock().unwrap().push(entry);
    }
    fn entries(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
    fn count(&self, what: &str) -> usize {
        self.entries().iter().filter(|e| e.ends_with(what)).count()
    }
}

#[derive(Clone)]
enum Fault {
    Fail,
    Hang(Arc<Notify>),
    WrongPosition,
}

/// A stage that records its calls and misbehaves on its `at`-th forward call.
struct Probe {
    inner: Box<dyn ChainStage>,
    log: Log,
    fault: Option<(usize, Fault)>,
    calls: usize,
}

impl Probe {
    fn wrap(stage: Box<dyn ChainStage>, log: &Log, fault: Option<(usize, Fault)>) -> Box<Self> {
        Box::new(Self {
            inner: stage,
            log: log.clone(),
            fault,
            calls: 0,
        })
    }
    fn name(&self) -> String {
        self.inner.descriptor().worker.clone()
    }
}

impl ChainStage for Probe {
    fn descriptor(&self) -> &StageDescriptor {
        self.inner.descriptor()
    }

    fn forward(&mut self, input: StageInput) -> StageFuture<'_, StageOutput> {
        self.calls += 1;
        self.log.push(format!("{} forward", self.name()));
        let fault = match &self.fault {
            Some((at, fault)) if *at == self.calls => Some(fault.clone()),
            _ => None,
        };
        Box::pin(async move {
            match fault {
                Some(Fault::Fail) => Err(StageError::Worker("connection reset by peer".into())),
                Some(Fault::Hang(entered)) => {
                    entered.notify_one();
                    std::future::pending().await
                }
                Some(Fault::WrongPosition) => match self.inner.forward(input).await? {
                    StageOutput::Hidden(a) => {
                        let shifted = Activations::new(
                            a.start_position() + 1,
                            a.hidden_size(),
                            a.into_data(),
                        )
                        .unwrap();
                        Ok(StageOutput::Hidden(shifted))
                    }
                    other => Ok(other),
                },
                None => self.inner.forward(input).await,
            }
        })
    }

    fn close(&mut self) -> StageFuture<'_, ()> {
        self.log.push(format!("{} close", self.name()));
        self.inner.close()
    }

    fn cancel(&mut self) -> StageFuture<'_, ()> {
        self.log.push(format!("{} cancel", self.name()));
        self.inner.cancel()
    }
}

fn probed(
    model: &Model,
    log: &Log,
    faulty: usize,
    fault: Option<(usize, Fault)>,
) -> Vec<Box<dyn ChainStage>> {
    model
        .stages(&[1, 2])
        .into_iter()
        .enumerate()
        .map(|(i, stage)| {
            let fault = if i == faulty { fault.clone() } else { None };
            Probe::wrap(stage, log, fault) as Box<dyn ChainStage>
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failing_stage_stops_the_run_and_cancels_every_session() {
    let model = Model::open(fixture("qwen2_f32"));
    let log = Log::default();
    let settings = GenerationSettings::tokens_only(defaults(greedy()));
    // The middle stage fails on its third call: prefill and one decode step succeed.
    let stages = probed(&model, &log, 1, Some((3, Fault::Fail)));
    let (result, events) = run(stages, &settings, GenerationRequest::new(PROMPT.to_vec())).await;
    let error = result.unwrap_err();
    match &error {
        GenerationError::Stage(failure) => {
            assert!(matches!(failure.source, StageError::Worker(_)), "{failure}");
            assert_eq!(
                (
                    failure.stage,
                    failure.worker.as_str(),
                    failure.layers.clone()
                ),
                (1, "machine-1", 1..2)
            );
            assert_eq!(
                (failure.position, failure.tokens_generated),
                (PROMPT.len() + 2, 2)
            );
            assert!(failure.cleanup.is_empty());
        }
        other => panic!("{other}"),
    }
    let message = error.to_string();
    assert!(
        message.contains("machine-1") && message.contains("layers 1..2"),
        "{message}"
    );
    assert_eq!(events.len(), 2);
    assert_eq!(log.count(" cancel"), 3);
    assert_eq!(log.count(" close"), 0);
    // The last stage never saw the failed step.
    assert_eq!(
        log.entries()
            .iter()
            .filter(|e| *e == "machine-2 forward")
            .count(),
        2
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stage_answering_for_the_wrong_position_is_refused() {
    let model = Model::open(fixture("qwen2_f32"));
    let log = Log::default();
    let settings = GenerationSettings::tokens_only(defaults(greedy()));
    let stages = probed(&model, &log, 0, Some((2, Fault::WrongPosition)));
    let (result, events) = run(stages, &settings, GenerationRequest::new(PROMPT.to_vec())).await;
    match result.unwrap_err() {
        GenerationError::Stage(failure) => match &failure.source {
            StageError::Contract(message) if failure.stage == 0 => {
                assert!(message.contains("from position"), "{message}")
            }
            other => panic!("{other}"),
        },
        other => panic!("{other}"),
    }
    assert_eq!(events.len(), 1);
    assert_eq!(log.count(" cancel"), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_abandons_the_step_in_flight_and_cancels_every_session() {
    let model = Model::open(fixture("qwen2_f32"));
    let log = Log::default();
    let entered = Arc::new(Notify::new());
    let stages = probed(
        &model,
        &log,
        2,
        Some((4, Fault::Hang(Arc::clone(&entered)))),
    );
    let mut run = spawn_generation(
        Chain::new(stages).unwrap(),
        GenerationSettings::tokens_only(defaults(greedy())),
        GenerationRequest::new(PROMPT.to_vec()),
    );
    entered.notified().await;
    run.cancel.cancel();
    let mut tokens = Vec::new();
    while let Some(event) = run.events.recv().await {
        tokens.push(event.token);
    }
    let report = run.finish().await.unwrap();
    assert_eq!(report.finish, FinishReason::Cancelled);
    assert_eq!(report.tokens, tokens);
    assert_eq!(tokens.len(), 3);
    assert_eq!(log.count(" cancel"), 3);
    assert_eq!(log.count(" close"), 0);
    let expected = oracle_greedy(&model, &PROMPT);
    assert_eq!(tokens, expected[..3]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_the_event_receiver_cancels_the_run() {
    let model = Model::open(fixture("qwen2_f32"));
    let log = Log::default();
    let (sender, receiver) = mpsc::unbounded_channel();
    drop(receiver);
    let report = generate(
        Chain::new(probed(&model, &log, 0, None)).unwrap(),
        &GenerationSettings::tokens_only(defaults(greedy())),
        GenerationRequest::new(PROMPT.to_vec()),
        &CancelToken::new(),
        &sender,
    )
    .await
    .unwrap();
    assert_eq!(report.finish, FinishReason::Cancelled);
    // The receiver is gone before the prefill, so the prefill is abandoned.
    assert!(report.tokens.is_empty());
    assert_eq!(log.count(" cancel"), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_chain_must_hold_one_model_in_order() {
    let model = Model::open(fixture("qwen2_f32"));
    let local = |m: &Model, layers: std::ops::Range<usize>, id: &str| {
        Box::new(LocalStage::new("m", id, m.slice(layers))) as Box<dyn ChainStage>
    };
    let cases: Vec<(Vec<Box<dyn ChainStage>>, &str)> = vec![
        (Vec::new(), "at least one stage"),
        (
            vec![local(&model, 0..1, "a"), local(&model, 2..3, "a")],
            "needs layer 1 next",
        ),
        (
            vec![local(&model, 0..2, "a"), local(&model, 1..3, "a")],
            "needs layer 2 next",
        ),
        (
            vec![local(&model, 0..1, "a"), local(&model, 1..2, "a")],
            "must hold the final norm",
        ),
        (
            vec![local(&model, 1..3, "a")],
            "must hold the token embedding",
        ),
        (
            vec![local(&model, 0..1, "a"), local(&model, 1..3, "b")],
            "holds `b`",
        ),
    ];
    for (stages, needle) in cases {
        match Chain::new(stages) {
            Err(GenerationError::Chain(message)) => {
                assert!(message.contains(needle), "{message} lacks {needle}")
            }
            Err(other) => panic!("{other}"),
            Ok(_) => panic!("accepted a chain that should fail with {needle}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn limits_name_both_numbers_and_a_lowered_limit_is_reported() {
    let model = Model::open(fixture("qwen2_f32"));
    let maximum = model.config.max_position_embeddings;
    let settings = GenerationSettings::tokens_only(defaults(greedy()));

    let (result, _) = run(
        model.stages(&[2]),
        &settings,
        GenerationRequest::new(vec![1; maximum]),
    )
    .await;
    assert!(matches!(
        result,
        Err(GenerationError::NoRoomToGenerate { prompt, maximum: m }) if prompt == maximum && m == maximum
    ));

    let mut request = GenerationRequest::new(PROMPT.to_vec());
    request.max_new_tokens = NonZeroUsize::new(maximum - PROMPT.len() + 1);
    let (result, _) = run(model.stages(&[2]), &settings, request).await;
    let error = result.unwrap_err();
    assert!(
        matches!(error, GenerationError::LimitAboveMaximum { requested, maximum: m }
            if requested == maximum - PROMPT.len() + 1 && m == maximum - PROMPT.len()),
        "{error}"
    );

    let mut request = GenerationRequest::new(PROMPT.to_vec());
    request.max_new_tokens = NonZeroUsize::new(3);
    let (report, events) = run(model.stages(&[2]), &settings, request).await;
    let report = report.unwrap();
    assert_eq!(
        report.finish,
        FinishReason::OperatorLimit { max_new_tokens: 3 }
    );
    assert_eq!(report.max_new_tokens, Some(3));
    assert_eq!((report.tokens.len(), events.len()), (3, 3));

    let (report, _) = run(
        model.stages(&[2]),
        &settings,
        GenerationRequest::new(PROMPT.to_vec()),
    )
    .await;
    let report = report.unwrap();
    assert_eq!(report.max_new_tokens, None);
    assert_eq!(report.tokens.len(), maximum - PROMPT.len());

    let (result, _) = run(
        model.stages(&[2]),
        &settings,
        GenerationRequest::new(vec![96]),
    )
    .await;
    assert!(
        matches!(result, Err(GenerationError::Request(m)) if m.contains("outside the vocabulary"))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn declared_settings_are_implemented_or_refused_before_any_stage_runs() {
    let model = Model::open(fixture("qwen2_f32"));
    let log = Log::default();
    let mut refused = defaults(greedy());
    refused.declared = json!({"num_beams": 4, "max_new_tokens": 17})
        .as_object()
        .unwrap()
        .clone();
    let (result, _) = run(
        probed(&model, &log, 0, None),
        &GenerationSettings::tokens_only(refused),
        GenerationRequest::new(PROMPT.to_vec()),
    )
    .await;
    assert!(
        matches!(result, Err(GenerationError::UnsupportedSetting { key, .. }) if key == "num_beams")
    );
    assert_eq!(log.count(" forward"), 0);
    assert_eq!(log.count(" cancel"), 3);

    let mut stops = defaults(greedy());
    stops.stop_strings = vec!["END".into()];
    let (result, _) = run(
        model.stages(&[1]),
        &GenerationSettings::tokens_only(stops),
        GenerationRequest::new(PROMPT.to_vec()),
    )
    .await;
    assert!(
        matches!(result, Err(GenerationError::UnsupportedSetting { key, .. }) if key == "stop_strings")
    );

    // A declared max_new_tokens is metadata, not a cap.
    let mut declared = defaults(greedy());
    declared.declared = json!({"max_new_tokens": 2, "max_length": 8})
        .as_object()
        .unwrap()
        .clone();
    let (report, _) = run(
        model.stages(&[1]),
        &GenerationSettings::tokens_only(declared),
        GenerationRequest::new(PROMPT.to_vec()),
    )
    .await;
    assert_eq!(
        report.unwrap().tokens.len(),
        model.config.max_position_embeddings - PROMPT.len()
    );
}

// ---- Text ----

fn text_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model_text/byte_bpe")
}

/// The byte-BPE text fixture with random F32 weights for its config, so text and layers belong to
/// one checkpoint.
struct TextCheckpoint(PathBuf);

impl TextCheckpoint {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "kadmium-generation-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(text_fixture()).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
        }
        let c = DecoderConfig::from_model_dir(&dir).unwrap();
        let (h, q, kv, d, i) = (
            c.hidden_size,
            c.q_width(),
            c.kv_width(),
            c.head_dim,
            c.intermediate_size,
        );
        let mut tensors: Vec<(String, Vec<usize>)> =
            vec![("model.embed_tokens.weight".into(), vec![c.vocab_size, h])];
        for layer in 0..c.num_layers {
            let p = format!("model.layers.{layer}.");
            let mut add =
                |name: &str, shape: Vec<usize>| tensors.push((format!("{p}{name}"), shape));
            add("input_layernorm.weight", vec![h]);
            add("post_attention_layernorm.weight", vec![h]);
            add("self_attn.q_proj.weight", vec![q, h]);
            add("self_attn.k_proj.weight", vec![kv, h]);
            add("self_attn.v_proj.weight", vec![kv, h]);
            add("self_attn.o_proj.weight", vec![h, q]);
            if c.qkv_bias {
                add("self_attn.q_proj.bias", vec![q]);
                add("self_attn.k_proj.bias", vec![kv]);
                add("self_attn.v_proj.bias", vec![kv]);
            }
            if c.qk_norm {
                add("self_attn.q_norm.weight", vec![d]);
                add("self_attn.k_norm.weight", vec![d]);
            }
            add("mlp.gate_proj.weight", vec![i, h]);
            add("mlp.up_proj.weight", vec![i, h]);
            add("mlp.down_proj.weight", vec![h, i]);
        }
        tensors.push(("model.norm.weight".into(), vec![h]));
        if !c.tie_word_embeddings {
            tensors.push(("lm_head.weight".into(), vec![c.vocab_size, h]));
        }
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut header = Map::new();
        let mut payload = Vec::new();
        for (name, shape) in tensors {
            let count: usize = shape.iter().product();
            let start = payload.len();
            for _ in 0..count {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                let value = if name.ends_with("norm.weight") {
                    1.0 + unit * 0.2
                } else {
                    unit * 1.5
                };
                payload.extend(value.to_le_bytes());
            }
            header.insert(
                name,
                json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]}),
            );
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        Self(dir)
    }
}

impl TextCheckpoint {
    fn set_tokenizer_config(&self, key: &str, value: Value) {
        let path = self.0.join("tokenizer_config.json");
        let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config[key] = value;
        std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    }
}

impl Drop for TextCheckpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A whole-model stage that returns logits selecting a scripted token at each step.
struct Scripted {
    descriptor: StageDescriptor,
    script: Vec<u32>,
    step: usize,
}

impl ChainStage for Scripted {
    fn descriptor(&self) -> &StageDescriptor {
        &self.descriptor
    }

    fn forward(&mut self, input: StageInput) -> StageFuture<'_, StageOutput> {
        let token = self.script[self.step];
        self.step += 1;
        let mut values = vec![0.0f32; self.descriptor.vocab_size];
        values[token as usize] = 10.0;
        let position = input.start_position() + input.rows() - 1;
        Box::pin(async move { Ok(StageOutput::Logits { position, values }) })
    }

    fn close(&mut self) -> StageFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn cancel(&mut self) -> StageFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}

fn scripted(text: &ModelText, script: Vec<u32>) -> Vec<Box<dyn ChainStage>> {
    let c = text.config();
    vec![Box::new(Scripted {
        descriptor: StageDescriptor {
            worker: "scripted".into(),
            model_id: "byte_bpe".into(),
            plan: SlicePlan::whole(c),
            num_layers: c.num_layers,
            hidden_size: c.hidden_size,
            vocab_size: c.vocab_size,
            max_position_embeddings: c.max_position_embeddings,
        },
        script,
        step: 0,
    })]
}

fn greedy_text(text: &Arc<ModelText>) -> GenerationSettings {
    let mut settings = GenerationSettings::with_text(Arc::clone(text), true);
    settings.defaults.sampling = greedy();
    settings
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streamed_text_never_shows_a_stop_string_or_a_partial_character() {
    let text = Arc::new(ModelText::from_model_dir(&text_fixture()).unwrap());
    assert_eq!(text.generation_defaults().stop_strings, ["END", "结束"]);
    let prompt = text.encode("Q:", false).unwrap();
    let prompt_text = text.decode(&prompt, true).unwrap();
    for (output, stop) in [
        ("价格 is fine END and more", "END"),
        ("an E then EN then 结 then 结束 after", "结束"),
    ] {
        let script = text.encode(output, false).unwrap();
        let mut all = prompt.clone();
        all.extend(&script);
        let generated = text.decode(&all, true).unwrap()[prompt_text.len()..].to_owned();
        let expected = &generated[..generated.find(stop).unwrap()];
        assert!(expected.contains('价') || expected.contains('结'));
        let (report, events) = run(
            scripted(&text, script),
            &greedy_text(&text),
            GenerationRequest::new(prompt.clone()),
        )
        .await;
        let report = report.unwrap();
        assert_eq!(
            report.finish,
            FinishReason::StopString { stop: stop.into() }
        );
        let streamed: String = events.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(streamed, expected);
        assert_eq!(report.text.as_deref(), Some(expected));
        let mut sent = String::new();
        for event in &events {
            sent.push_str(&event.text);
            assert!(!sent.contains('\u{FFFD}'), "{sent:?}");
            assert!(!sent.contains(stop), "{sent:?}");
        }
    }
}

/// A prompt ending in a space followed by a generated `.`: the clean-up would rewrite across the
/// boundary if it ran on the whole sequence. The generated text is cleaned up alone, as
/// Transformers' `TextStreamer` does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clean_up_never_reaches_across_the_prompt_boundary() {
    let checkpoint = TextCheckpoint::new();
    checkpoint.set_tokenizer_config("clean_up_tokenization_spaces", Value::Bool(true));
    let text = Arc::new(ModelText::from_model_dir(&checkpoint.0).unwrap());
    let prompt = text.encode("The answer is ", false).unwrap();
    assert!(text.decode(&prompt, true).unwrap().ends_with(' '));
    for (output, expected) in [(".", "."), (" fine , and done .", " fine, and done.")] {
        let mut script = text.encode(output, false).unwrap();
        // Transformers' TextStreamer decodes the generated ids alone, with clean-up.
        assert_eq!(text.decode(&script, true).unwrap(), expected);
        script.push(2);
        let (report, events) = run(
            scripted(&text, script),
            &greedy_text(&text),
            GenerationRequest::new(prompt.clone()),
        )
        .await;
        let report = report.unwrap();
        assert_eq!(report.finish, FinishReason::EndOfSequence { token: 2 });
        let streamed: String = events.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(streamed, expected);
        assert_eq!(report.text.as_deref(), Some(expected));
    }
}

/// The last stage returns the logits of the last row only. They equal, bit for bit, that row of
/// the logits a whole pass computes over every row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn last_row_logits_equal_the_full_pass_bit_for_bit() {
    for name in FIXTURES {
        let model = Model::open(fixture(name));
        let whole = model.slice(0..model.config.num_layers);
        let mut cache = whole.new_cache();
        let hidden = whole
            .forward(&whole.embed(&PROMPT, 0).unwrap(), &mut cache)
            .unwrap();
        let full = whole.logits(&hidden).unwrap();
        let vocab = model.config.vocab_size;
        let expected: Vec<u32> = full[full.len() - vocab..]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let mut stage = LocalStage::new("m", "fixture", Arc::clone(&whole));
        let output = stage
            .forward(StageInput::Tokens {
                start_position: 0,
                tokens: PROMPT.to_vec(),
            })
            .await
            .unwrap();
        match output {
            StageOutput::Logits { position, values } => {
                assert_eq!(position, PROMPT.len() - 1);
                let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
                assert_eq!(bits, expected, "{name}");
            }
            other => panic!("{other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn held_back_text_is_sent_when_the_model_ends() {
    let text = Arc::new(ModelText::from_model_dir(&text_fixture()).unwrap());
    let prompt = text.encode("Q:", false).unwrap();
    let mut script = text.encode("Hi EN", false).unwrap();
    script.push(2);
    let (report, events) = run(
        scripted(&text, script.clone()),
        &greedy_text(&text),
        GenerationRequest::new(prompt.clone()),
    )
    .await;
    let report = report.unwrap();
    assert_eq!(report.finish, FinishReason::EndOfSequence { token: 2 });
    assert_eq!(report.tokens, script);
    let streamed: String = events.iter().map(|e| e.text.as_str()).collect();
    let expected = {
        let mut all = prompt.clone();
        all.extend(&script[..script.len() - 1]);
        let full = text.decode(&all, true).unwrap();
        full[text.decode(&prompt, true).unwrap().len()..].to_owned()
    };
    assert!(expected.ends_with("EN"), "{expected:?}");
    assert_eq!(streamed, expected);
    assert_eq!(report.text.as_deref(), Some(expected.as_str()));
    // The last token's event carries the text held back for a possible "END".
    assert!(events.last().unwrap().text.ends_with("EN"));

    let mut request = GenerationRequest::new(prompt);
    request.max_new_tokens = NonZeroUsize::new(script.len() - 1);
    let (report, events) = run(scripted(&text, script), &greedy_text(&text), request).await;
    let report = report.unwrap();
    assert_eq!(
        report.finish,
        FinishReason::OperatorLimit {
            max_new_tokens: report.tokens.len()
        }
    );
    let streamed: String = events.iter().map(|e| e.text.as_str()).collect();
    assert_eq!(streamed, expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_real_checkpoint_streams_the_same_text_whole_and_split() {
    let checkpoint = TextCheckpoint::new();
    let text = Arc::new(ModelText::from_model_dir(&checkpoint.0).unwrap());
    let model = Model::open(checkpoint.0.clone());
    // The checkpoint's own settings: sampling, repetition penalty, EOS ids and stop strings.
    let settings = GenerationSettings::with_text(Arc::clone(&text), true);
    assert!(settings.defaults.sampling.do_sample);
    let messages: Vec<kadmium_runtime::model_text::TemplateValue> =
        serde_json::from_str(r#"[{"role":"user","content":"Say something."}]"#).unwrap();
    let prompt = text
        .encode_chat(&messages, true, None, &Default::default(), 1_000_000)
        .unwrap();
    let mut outputs = Vec::new();
    for cuts in [vec![], vec![1], vec![0, 2]] {
        let mut request = GenerationRequest::new(prompt.clone());
        request.seed = Some(7);
        request.max_new_tokens = NonZeroUsize::new(48);
        let (report, events) = run(model.stages(&cuts), &settings, request).await;
        let report = report.unwrap();
        let streamed: String = events.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(report.text.as_deref(), Some(streamed.as_str()));
        assert_eq!(events.len(), report.tokens.len());
        outputs.push((report.tokens, streamed, report.finish));
    }
    assert_eq!(
        outputs[0].2,
        FinishReason::OperatorLimit { max_new_tokens: 48 }
    );
    assert!(outputs.windows(2).all(|w| w[0] == w[1]), "{outputs:?}");
}
