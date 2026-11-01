//! A decoder slice: any contiguous range of a checkpoint's layers, optionally with the token
//! embedding in front and the final norm and LM head behind, loaded from the checkpoint's
//! SafeTensors files at their stored precision, with its own KV cache.
//!
//! A forward pass is a function of the input activations and the slice's KV cache alone, and
//! every row is computed independently of the others, so running layers `[a, c)` gives the same
//! bits as running `[a, b)` and passing the activations to a slice that runs `[b, c)`, and
//! decoding one token at a time gives the same bits as a prefill of the whole sequence.

use super::config::{DecoderConfig, RopeScaling};
use super::math::{self, MAX_ANGLE, Matrix, WeightData};
use super::{NativeDecoderError, Result};
use crate::repops;
use kadmium_data::{SafetensorsDtype, SafetensorsManifest};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

/// What one slice holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlicePlan {
    /// Half-open range of decoder layers.
    pub layers: Range<usize>,
    /// The token embedding, for the first slice of a chain.
    pub embedding: bool,
    /// The final norm and LM head, for the last slice of a chain.
    pub head: bool,
}

impl SlicePlan {
    /// Tensor names required by the loader, including an embedding used as a tied output head.
    /// Validates shapes and dtypes without reading weight payloads.
    pub fn tensor_names(
        &self,
        manifest: &SafetensorsManifest,
        config: &DecoderConfig,
    ) -> Result<Vec<String>> {
        if self.layers.start > self.layers.end || self.layers.end > config.num_layers {
            return Err(NativeDecoderError::LayerRange {
                start: self.layers.start,
                end: self.layers.end,
                layers: config.num_layers,
            });
        }
        check_against_checkpoint(manifest, config)?;
        let mut specs = Vec::new();
        if self.embedding || (self.head && config.tie_word_embeddings) {
            specs.push(Spec {
                name: "model.embed_tokens.weight".into(),
                shape: vec![config.vocab_size, config.hidden_size],
            });
        }
        for layer in self.layers.clone() {
            specs.extend(block_specs(config, layer));
        }
        if self.head {
            specs.push(Spec {
                name: "model.norm.weight".into(),
                shape: vec![config.hidden_size],
            });
            if !config.tie_word_embeddings {
                specs.push(Spec {
                    name: "lm_head.weight".into(),
                    shape: vec![config.vocab_size, config.hidden_size],
                });
            }
        }
        validate_specs(manifest, &specs)?;
        Ok(specs.into_iter().map(|spec| spec.name).collect())
    }

    /// The whole model in one slice.
    pub fn whole(config: &DecoderConfig) -> Self {
        Self {
            layers: 0..config.num_layers,
            embedding: true,
            head: true,
        }
    }
}

/// Hidden states for consecutive positions, row-major `rows x hidden_size`, `f32`. This is
/// what one slice hands the next; its bytes are the whole interface between them.
#[derive(Debug, Clone, PartialEq)]
pub struct Activations {
    start_position: usize,
    hidden_size: usize,
    data: Vec<f32>,
}

impl Activations {
    pub fn new(start_position: usize, hidden_size: usize, data: Vec<f32>) -> Result<Self> {
        if hidden_size == 0 || data.is_empty() || !data.len().is_multiple_of(hidden_size) {
            return Err(NativeDecoderError::Shape(format!(
                "{} activation values do not form whole rows of {hidden_size}",
                data.len()
            )));
        }
        Ok(Self {
            start_position,
            hidden_size,
            data,
        })
    }

    pub fn start_position(&self) -> usize {
        self.start_position
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn rows(&self) -> usize {
        self.data.len() / self.hidden_size
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }
}

/// Keys and values for the layers of one slice, `f32`, one row per position already processed.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Range<usize>,
    kv_width: usize,
    positions: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl KvCache {
    /// Positions already processed; the next forward pass must start here.
    pub fn positions(&self) -> usize {
        self.positions
    }

    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    /// Bytes held by keys and values.
    pub fn bytes(&self) -> usize {
        let elements: usize = self.keys.iter().chain(&self.values).map(Vec::len).sum();
        elements * 4
    }
}

struct Linear {
    weight: Matrix,
    bias: Option<Vec<f32>>,
}

impl Linear {
    fn apply(&self, input: &[f32], rows: usize) -> Vec<f32> {
        math::linear(&self.weight, self.bias.as_deref(), input, rows)
    }

    fn stored_bytes(&self) -> usize {
        self.weight.data.stored_bytes() + self.bias.as_ref().map_or(0, |b| b.len() * 4)
    }
}

struct Block {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Block {
    fn stored_bytes(&self) -> usize {
        let vectors = self.input_norm.len()
            + self.post_norm.len()
            + self.q_norm.as_ref().map_or(0, Vec::len)
            + self.k_norm.as_ref().map_or(0, Vec::len);
        vectors * 4
            + [
                &self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down,
            ]
            .iter()
            .map(|l| l.stored_bytes())
            .sum::<usize>()
    }
}

struct Head {
    norm: Vec<f32>,
    lm_head: Arc<Matrix>,
}

/// A loaded range of decoder layers.
pub struct DecoderSlice {
    config: DecoderConfig,
    plan: SlicePlan,
    embedding: Option<Arc<Matrix>>,
    blocks: Vec<Block>,
    head: Option<Head>,
    inverse_frequencies: Vec<f32>,
}

pub(super) struct Spec {
    pub(super) name: String,
    pub(super) shape: Vec<usize>,
}

/// Validates a tensor's existence, dtype (`F32`, `F16` or `BF16`) and shape against `manifest`
/// and returns the in-memory bytes [`DecoderSlice`] holds for it (1-D norm and bias vectors are
/// widened to `f32`; 2-D weight matrices stay in their stored dtype).
pub(super) fn stored_spec_bytes(manifest: &SafetensorsManifest, spec: &Spec) -> Result<u64> {
    let descriptor = manifest
        .tensor(&spec.name)
        .ok_or_else(|| NativeDecoderError::MissingTensor(spec.name.clone()))?;
    if !matches!(
        descriptor.dtype,
        SafetensorsDtype::F32 | SafetensorsDtype::F16 | SafetensorsDtype::Bf16
    ) {
        return Err(NativeDecoderError::UnsupportedDtype {
            tensor: spec.name.clone(),
            dtype: descriptor.dtype.as_str(),
        });
    }
    let expected: Vec<u64> = spec.shape.iter().map(|&d| d as u64).collect();
    if descriptor.shape != expected {
        return Err(NativeDecoderError::TensorShape {
            tensor: spec.name.clone(),
            expected: spec.shape.clone(),
            actual: descriptor.shape.clone(),
        });
    }
    let elements: u64 = expected
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            NativeDecoderError::Shape(format!("tensor `{}` size overflows", spec.name))
        })?;
    let width: u64 = if descriptor.dtype == SafetensorsDtype::F32 {
        4
    } else {
        2
    };
    let expected_stored = elements.checked_mul(width).ok_or_else(|| {
        NativeDecoderError::Shape(format!("tensor `{}` byte size overflows", spec.name))
    })?;
    if descriptor.byte_len != expected_stored {
        return Err(NativeDecoderError::TensorShape {
            tensor: spec.name.clone(),
            expected: spec.shape.clone(),
            actual: vec![descriptor.byte_len / width],
        });
    }
    if spec.shape.len() == 1 {
        elements.checked_mul(4).ok_or_else(|| {
            NativeDecoderError::Shape(format!("tensor `{}` f32 size overflows", spec.name))
        })
    } else {
        Ok(expected_stored)
    }
}

/// Checks every dtype and shape before a payload is read.
fn validate_specs(manifest: &SafetensorsManifest, specs: &[Spec]) -> Result<()> {
    for spec in specs {
        let _ = stored_spec_bytes(manifest, spec)?;
    }
    Ok(())
}

fn fetch(manifest: &SafetensorsManifest, specs: &[Spec]) -> Result<Vec<WeightData>> {
    validate_specs(manifest, specs)?;
    let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    let (payloads, _) = manifest.read_tensors(&names)?;
    let mut out = Vec::with_capacity(specs.len());
    for (spec, bytes) in specs.iter().zip(payloads) {
        let dtype = manifest
            .tensor(&spec.name)
            .map(|d| d.dtype)
            .ok_or_else(|| NativeDecoderError::MissingTensor(spec.name.clone()))?;
        let elements: usize = spec.shape.iter().product();
        let width = if dtype == SafetensorsDtype::F32 { 4 } else { 2 };
        if bytes.len() != elements * width {
            return Err(NativeDecoderError::TensorShape {
                tensor: spec.name.clone(),
                expected: spec.shape.clone(),
                actual: vec![(bytes.len() / width) as u64],
            });
        }
        out.push(match dtype {
            SafetensorsDtype::F32 => WeightData::F32(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect(),
            ),
            SafetensorsDtype::F16 => WeightData::F16(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c))
                    .collect(),
            ),
            _ => WeightData::Bf16(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c))
                    .collect(),
            ),
        });
    }
    Ok(out)
}

struct Fetched {
    items: std::vec::IntoIter<(Spec, WeightData)>,
}

impl Fetched {
    fn take(&mut self) -> Result<(Spec, WeightData)> {
        self.items
            .next()
            .ok_or_else(|| NativeDecoderError::Shape("tensor list ended early".into()))
    }

    fn matrix(&mut self) -> Result<Matrix> {
        let (spec, data) = self.take()?;
        Ok(Matrix {
            rows: spec.shape[0],
            cols: spec.shape[1],
            data,
        })
    }

    fn vector(&mut self) -> Result<Vec<f32>> {
        Ok(self.take()?.1.widen())
    }
}

fn fetch_all(manifest: &SafetensorsManifest, specs: Vec<Spec>) -> Result<Fetched> {
    let data = fetch(manifest, &specs)?;
    Ok(Fetched {
        items: specs.into_iter().zip(data).collect::<Vec<_>>().into_iter(),
    })
}

pub(super) fn block_specs(c: &DecoderConfig, layer: usize) -> Vec<Spec> {
    let prefix = format!("model.layers.{layer}.");
    let (h, q, kv, d, i) = (
        c.hidden_size,
        c.q_width(),
        c.kv_width(),
        c.head_dim,
        c.intermediate_size,
    );
    let mut specs = Vec::new();
    let mut add = |suffix: &str, shape: &[usize]| {
        specs.push(Spec {
            name: format!("{prefix}{suffix}"),
            shape: shape.to_vec(),
        })
    };
    add("input_layernorm.weight", &[h]);
    add("post_attention_layernorm.weight", &[h]);
    add("self_attn.q_proj.weight", &[q, h]);
    if c.qkv_bias {
        add("self_attn.q_proj.bias", &[q]);
    }
    add("self_attn.k_proj.weight", &[kv, h]);
    if c.qkv_bias {
        add("self_attn.k_proj.bias", &[kv]);
    }
    add("self_attn.v_proj.weight", &[kv, h]);
    if c.qkv_bias {
        add("self_attn.v_proj.bias", &[kv]);
    }
    add("self_attn.o_proj.weight", &[h, q]);
    if c.o_bias {
        add("self_attn.o_proj.bias", &[h]);
    }
    if c.qk_norm {
        add("self_attn.q_norm.weight", &[d]);
        add("self_attn.k_norm.weight", &[d]);
    }
    for (name, shape) in [
        ("gate_proj", [i, h]),
        ("up_proj", [i, h]),
        ("down_proj", [h, i]),
    ] {
        add(&format!("mlp.{name}.weight"), &shape);
        if c.mlp_bias {
            add(&format!("mlp.{name}.bias"), &shape[..1]);
        }
    }
    specs
}

fn load_block(manifest: &SafetensorsManifest, c: &DecoderConfig, layer: usize) -> Result<Block> {
    let specs = block_specs(c, layer);
    let mut f = fetch_all(manifest, specs)?;
    let input_norm = f.vector()?;
    let post_norm = f.vector()?;
    let linear = |f: &mut Fetched, bias: bool| -> Result<Linear> {
        let weight = f.matrix()?;
        let bias = if bias { Some(f.vector()?) } else { None };
        Ok(Linear { weight, bias })
    };
    let q_proj = linear(&mut f, c.qkv_bias)?;
    let k_proj = linear(&mut f, c.qkv_bias)?;
    let v_proj = linear(&mut f, c.qkv_bias)?;
    let o_proj = linear(&mut f, c.o_bias)?;
    let (q_norm, k_norm) = if c.qk_norm {
        (Some(f.vector()?), Some(f.vector()?))
    } else {
        (None, None)
    };
    let gate = linear(&mut f, c.mlp_bias)?;
    let up = linear(&mut f, c.mlp_bias)?;
    let down = linear(&mut f, c.mlp_bias)?;
    Ok(Block {
        input_norm,
        post_norm,
        q: q_proj,
        k: k_proj,
        v: v_proj,
        o: o_proj,
        q_norm,
        k_norm,
        gate,
        up,
        down,
    })
}

fn single_matrix(manifest: &SafetensorsManifest, name: &str, shape: [usize; 2]) -> Result<Matrix> {
    fetch_all(
        manifest,
        vec![Spec {
            name: name.into(),
            shape: shape.to_vec(),
        }],
    )?
    .matrix()
}

/// Ties the config's layer count and attention widths to tensors the checkpoint stores, before
/// anything sized by the config is allocated. A stored layer the config does not count is an
/// error, because running the config's layers would silently skip it. The q, k and v projections
/// of the lowest stored layer must have the shapes the config implies, which bounds
/// `num_attention_heads * head_dim` and `num_key_value_heads * head_dim` by real tensor sizes.
/// A manifest that holds only some layers, such as one staged for a single slice, is accepted.
pub(super) fn check_against_checkpoint(
    manifest: &SafetensorsManifest,
    c: &DecoderConfig,
) -> Result<()> {
    let mut lowest: Option<usize> = None;
    for tensor in manifest.tensors() {
        let Some(index) = tensor
            .name
            .strip_prefix("model.layers.")
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(index, _)| index.parse::<usize>().ok())
        else {
            continue;
        };
        if index >= c.num_layers {
            return Err(NativeDecoderError::Config(format!(
                "the checkpoint stores layer {index}; `num_hidden_layers` is {}",
                c.num_layers
            )));
        }
        lowest = Some(lowest.map_or(index, |l| l.min(index)));
    }
    if let Some(layer) = lowest {
        for (part, rows) in [
            ("q_proj", c.q_width()),
            ("k_proj", c.kv_width()),
            ("v_proj", c.kv_width()),
        ] {
            let name = format!("model.layers.{layer}.self_attn.{part}.weight");
            let descriptor = manifest
                .tensor(&name)
                .ok_or_else(|| NativeDecoderError::MissingTensor(name.clone()))?;
            if descriptor.shape != [rows as u64, c.hidden_size as u64] {
                return Err(NativeDecoderError::TensorShape {
                    tensor: name,
                    expected: vec![rows, c.hidden_size],
                    actual: descriptor.shape.clone(),
                });
            }
        }
    }
    Ok(())
}

/// RoPE inverse frequencies, `f64` then rounded once to `f32`. Allocates `head_dim / 2` values;
/// [`DecoderSlice::load_from`] calls it only after the checkpoint's tensors have confirmed
/// `head_dim`.
pub fn inverse_frequencies(config: &DecoderConfig) -> Vec<f32> {
    let d = config.head_dim;
    let ln_theta = math::ln_f64(config.rope_theta);
    (0..d / 2)
        .map(|i| {
            let base = math::exp_f64(-((2 * i) as f64 / d as f64) * ln_theta);
            let scaled = match config.rope_scaling {
                RopeScaling::Default => base,
                RopeScaling::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position_embeddings: original,
                } => {
                    let wavelength = 2.0 * std::f64::consts::PI / base;
                    if wavelength < original / high_freq_factor {
                        base
                    } else if wavelength > original / low_freq_factor {
                        base / factor
                    } else {
                        let smooth = (original / wavelength - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * base / factor + smooth * base
                    }
                }
            };
            scaled as f32
        })
        .collect()
}

impl DecoderSlice {
    /// Loads a slice from a Hugging Face model directory (`config.json` and SafeTensors).
    pub fn load(model_dir: &Path, plan: SlicePlan) -> Result<Self> {
        let config = DecoderConfig::from_model_dir(model_dir)?;
        let manifest = SafetensorsManifest::open(model_dir)?;
        Self::load_from(&manifest, config, plan)
    }

    pub fn load_from(
        manifest: &SafetensorsManifest,
        config: DecoderConfig,
        plan: SlicePlan,
    ) -> Result<Self> {
        if plan.layers.start > plan.layers.end || plan.layers.end > config.num_layers {
            return Err(NativeDecoderError::LayerRange {
                start: plan.layers.start,
                end: plan.layers.end,
                layers: config.num_layers,
            });
        }
        check_against_checkpoint(manifest, &config)?;
        let embed_shape = [config.vocab_size, config.hidden_size];
        let embedding = if plan.embedding {
            Some(Arc::new(single_matrix(
                manifest,
                "model.embed_tokens.weight",
                embed_shape,
            )?))
        } else {
            None
        };
        // Not presized: the range comes from the config, and each layer is checked against the
        // checkpoint as it loads.
        let mut blocks = Vec::new();
        for layer in plan.layers.clone() {
            blocks.push(load_block(manifest, &config, layer)?);
        }
        let head = if plan.head {
            let norm = fetch_all(
                manifest,
                vec![Spec {
                    name: "model.norm.weight".into(),
                    shape: vec![config.hidden_size],
                }],
            )?
            .vector()?;
            let lm_head = match (&embedding, config.tie_word_embeddings) {
                (Some(shared), true) => Arc::clone(shared),
                (_, true) => Arc::new(single_matrix(
                    manifest,
                    "model.embed_tokens.weight",
                    embed_shape,
                )?),
                (_, false) => Arc::new(single_matrix(manifest, "lm_head.weight", embed_shape)?),
            };
            Some(Head { norm, lm_head })
        } else {
            None
        };
        // Only layers use RoPE, and by now their q and k tensors have confirmed `head_dim`.
        let inverse_frequencies = if blocks.is_empty() {
            Vec::new()
        } else {
            inverse_frequencies(&config)
        };
        if let Some(fastest) = inverse_frequencies.iter().copied().reduce(f32::max)
            && f64::from(fastest) * config.max_position_embeddings as f64 > MAX_ANGLE
        {
            return Err(NativeDecoderError::Unsupported(format!(
                "RoPE angles up to {} at the native context; they are reduced accurately only up to {MAX_ANGLE}",
                f64::from(fastest) * config.max_position_embeddings as f64
            )));
        }
        Ok(Self {
            config,
            plan,
            embedding,
            blocks,
            head,
            inverse_frequencies,
        })
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    pub fn plan(&self) -> &SlicePlan {
        &self.plan
    }

    /// Weight bytes held, equal to the bytes the checkpoint stores for them (norm vectors and
    /// biases are held widened to `f32`). A tied LM head is counted once.
    pub fn stored_weight_bytes(&self) -> usize {
        let embedding = self.embedding.as_ref().map_or(0, |m| m.data.stored_bytes());
        let head = self.head.as_ref().map_or(0, |h| {
            let shared = self
                .embedding
                .as_ref()
                .is_some_and(|e| Arc::ptr_eq(e, &h.lm_head));
            h.norm.len() * 4
                + if shared {
                    0
                } else {
                    h.lm_head.data.stored_bytes()
                }
        });
        embedding + head + self.blocks.iter().map(Block::stored_bytes).sum::<usize>()
    }

    /// An empty cache for this slice's layers.
    pub fn new_cache(&self) -> KvCache {
        let n = self.blocks.len();
        KvCache {
            layers: self.plan.layers.clone(),
            kv_width: self.config.kv_width(),
            positions: 0,
            keys: vec![Vec::new(); n],
            values: vec![Vec::new(); n],
        }
    }

    /// Embeds `tokens` at positions `start_position ..`.
    pub fn embed(&self, tokens: &[u32], start_position: usize) -> Result<Activations> {
        let embedding = self
            .embedding
            .as_ref()
            .ok_or(NativeDecoderError::MissingPart("token embedding"))?;
        if tokens.is_empty() {
            return Err(NativeDecoderError::Shape("no tokens to embed".into()));
        }
        let mut data = Vec::with_capacity(tokens.len() * self.config.hidden_size);
        for &token in tokens {
            let row = token as usize;
            if row >= embedding.rows {
                return Err(NativeDecoderError::TokenOutOfRange {
                    token,
                    vocab_size: embedding.rows,
                });
            }
            data.extend(embedding.row_f32(row));
        }
        Activations::new(start_position, self.config.hidden_size, data)
    }

    /// Runs this slice's layers over `input`, appending its keys and values to `cache`.
    pub fn forward(&self, input: &Activations, cache: &mut KvCache) -> Result<Activations> {
        let c = &self.config;
        if input.hidden_size != c.hidden_size {
            return Err(NativeDecoderError::Shape(format!(
                "activations have hidden size {}, the model {}",
                input.hidden_size, c.hidden_size
            )));
        }
        if cache.layers != self.plan.layers || cache.kv_width != c.kv_width() {
            return Err(NativeDecoderError::Shape(format!(
                "the cache holds layers {:?}, this slice runs {:?}",
                cache.layers, self.plan.layers
            )));
        }
        if input.start_position != cache.positions {
            return Err(NativeDecoderError::CachePosition {
                expected: cache.positions,
                actual: input.start_position,
            });
        }
        let rows = input.rows();
        let end = input.start_position + rows;
        if end > c.max_position_embeddings {
            return Err(NativeDecoderError::ContextExceeded {
                requested: end,
                maximum: c.max_position_embeddings,
            });
        }
        let added = rows * c.kv_width();
        for store in cache.keys.iter_mut().chain(cache.values.iter_mut()) {
            store
                .try_reserve(added)
                .map_err(|_| NativeDecoderError::Memory {
                    bytes: added.saturating_mul(4),
                })?;
        }

        let (cos, sin) = self.rope_table(input.start_position, rows);
        let mut hidden = input.data.clone();
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = self.block_forward(
                block,
                &hidden,
                input.start_position,
                rows,
                &cos,
                &sin,
                &mut cache.keys[index],
                &mut cache.values[index],
            );
        }
        cache.positions = end;
        Activations::new(input.start_position, c.hidden_size, hidden)
    }

    /// Final norm and LM head: `rows x vocab_size` logits.
    pub fn logits(&self, input: &Activations) -> Result<Vec<f32>> {
        let head = self
            .head
            .as_ref()
            .ok_or(NativeDecoderError::MissingPart("final norm and LM head"))?;
        let h = self.config.hidden_size;
        if input.hidden_size != h {
            return Err(NativeDecoderError::Shape(format!(
                "activations have hidden size {}, the model {h}",
                input.hidden_size
            )));
        }
        let rows = input.rows();
        let mut normed = vec![0.0f32; rows * h];
        for r in 0..rows {
            math::rms_norm(
                &input.data[r * h..(r + 1) * h],
                &head.norm,
                self.config.rms_norm_eps,
                &mut normed[r * h..(r + 1) * h],
            );
        }
        Ok(math::linear(&head.lm_head, None, &normed, rows))
    }

    fn rope_table(&self, start: usize, rows: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.inverse_frequencies.len();
        let mut cos = Vec::with_capacity(rows * half);
        let mut sin = Vec::with_capacity(rows * half);
        for position in start..start + rows {
            for &frequency in &self.inverse_frequencies {
                // Exact: a position below 2^20 times a 24-bit significand fits in 53 bits.
                let angle = position as f64 * f64::from(frequency);
                let (s, c) = math::sin_cos_f64(angle);
                cos.push(c as f32);
                sin.push(s as f32);
            }
        }
        (cos, sin)
    }

    #[allow(clippy::too_many_arguments)]
    fn block_forward(
        &self,
        block: &Block,
        x: &[f32],
        start: usize,
        rows: usize,
        cos: &[f32],
        sin: &[f32],
        keys: &mut Vec<f32>,
        values: &mut Vec<f32>,
    ) -> Vec<f32> {
        let c = &self.config;
        let (h, d) = (c.hidden_size, c.head_dim);
        let (qw, kvw) = (c.q_width(), c.kv_width());
        let half = d / 2;
        let eps = c.rms_norm_eps;

        let mut normed = vec![0.0f32; rows * h];
        for r in 0..rows {
            math::rms_norm(
                &x[r * h..(r + 1) * h],
                &block.input_norm,
                eps,
                &mut normed[r * h..(r + 1) * h],
            );
        }
        let mut q = block.q.apply(&normed, rows);
        let mut k = block.k.apply(&normed, rows);
        let v = block.v.apply(&normed, rows);

        let mut scratch = vec![0.0f32; d];
        let mut prepare = |data: &mut [f32], width: usize, norm: &Option<Vec<f32>>| {
            for r in 0..rows {
                let (c_row, s_row) = (
                    &cos[r * half..(r + 1) * half],
                    &sin[r * half..(r + 1) * half],
                );
                for head in data[r * width..(r + 1) * width].chunks_mut(d) {
                    if let Some(weight) = norm {
                        math::rms_norm(head, weight, eps, &mut scratch);
                        head.copy_from_slice(&scratch);
                    }
                    for i in 0..half {
                        let (a, b) = (head[i], head[i + half]);
                        head[i] = a * c_row[i] - b * s_row[i];
                        head[i + half] = b * c_row[i] + a * s_row[i];
                    }
                }
            }
        };
        prepare(&mut q, qw, &block.q_norm);
        prepare(&mut k, kvw, &block.k_norm);
        keys.extend_from_slice(&k);
        values.extend_from_slice(&v);

        let scale = (1.0 / (d as f64).sqrt()) as f32;
        let group = c.num_heads / c.num_kv_heads;
        let mut attended = vec![0.0f32; rows * qw];
        let mut weights = Vec::new();
        for r in 0..rows {
            let visible = start + r + 1;
            for head in 0..c.num_heads {
                let kv_offset = (head / group) * d;
                let query = &q[r * qw + head * d..r * qw + (head + 1) * d];
                weights.clear();
                for j in 0..visible {
                    let key = &keys[j * kvw + kv_offset..j * kvw + kv_offset + d];
                    weights.push(math::lazy_tree_sum(d, &|t| query[t] * key[t]) * scale);
                }
                let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for w in weights.iter_mut() {
                    *w = repops::exp(*w - max);
                }
                let total = repops::tree_sum_f32(&weights);
                for w in weights.iter_mut() {
                    *w /= total;
                }
                let out = &mut attended[r * qw + head * d..r * qw + (head + 1) * d];
                for (t, slot) in out.iter_mut().enumerate() {
                    let column = kv_offset + t;
                    *slot =
                        math::lazy_tree_sum(visible, &|j| weights[j] * values[j * kvw + column]);
                }
            }
        }

        let projected = block.o.apply(&attended, rows);
        let residual: Vec<f32> = x.iter().zip(&projected).map(|(a, b)| a + b).collect();
        for r in 0..rows {
            math::rms_norm(
                &residual[r * h..(r + 1) * h],
                &block.post_norm,
                eps,
                &mut normed[r * h..(r + 1) * h],
            );
        }
        let gate = block.gate.apply(&normed, rows);
        let up = block.up.apply(&normed, rows);
        let mixed: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| repops::silu(g) * u)
            .collect();
        let down = block.down.apply(&mixed, rows);
        residual.iter().zip(&down).map(|(a, b)| a + b).collect()
    }
}
