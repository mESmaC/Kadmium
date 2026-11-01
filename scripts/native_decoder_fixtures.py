"""Writes the reference fixtures for the native decoder tests.

Each fixture is a tiny, randomly initialised model saved by Hugging Face transformers exactly as a
real checkpoint is saved (`config.json` and `model.safetensors`), together with the logits that
transformers computes for a fixed token sequence. The weights are stored in the fixture's dtype;
the reference forward pass runs in float32 on those same stored weights, widened exactly, with
eager attention. `crates/kadmium-runtime/tests/native_decoder.rs` loads the files with the native
decoder and compares its logits against these.

This script is for tests only. Nothing shipped runs Python. It needs torch and transformers:

    python -m venv venv
    venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
    venv/bin/python -m pip install transformers safetensors
    venv/bin/python scripts/native_decoder_fixtures.py crates/kadmium-runtime/tests/fixtures/native_decoder

With `--reference <model directory> <output.json>` it instead writes a reference for a real
checkpoint: the token ids of a short chat prompt, the argmax at every position, and the full
float32 logits of the last position. `real_checkpoint_matches_transformers` reads it when
`KADMIUM_NATIVE_DECODER_MODEL` and `KADMIUM_NATIVE_DECODER_REFERENCE` are set.
"""

import json
import os
import sys

import torch
import transformers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

TOKENS = [1, 17, 42, 5, 88, 3, 64, 29, 71, 12, 0, 95, 33, 50]

COMMON = dict(
    vocab_size=96,
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
    rms_norm_eps=1e-5,
    initializer_range=0.25,
    hidden_act="silu",
)

FIXTURES = [
    (
        "llama3_bf16",
        LlamaConfig,
        LlamaForCausalLM,
        torch.bfloat16,
        dict(
            rope_theta=10000.0,
            rope_scaling=dict(
                rope_type="llama3",
                factor=8.0,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                original_max_position_embeddings=16,
            ),
            tie_word_embeddings=False,
        ),
    ),
    (
        "qwen2_f32",
        Qwen2Config,
        Qwen2ForCausalLM,
        torch.float32,
        dict(rope_theta=1000000.0, tie_word_embeddings=True, use_sliding_window=False),
    ),
    (
        "qwen3_bf16",
        Qwen3Config,
        Qwen3ForCausalLM,
        torch.bfloat16,
        dict(
            head_dim=16,
            rope_theta=1000000.0,
            tie_word_embeddings=False,
            use_sliding_window=False,
            num_key_value_heads=1,
        ),
    ),
    (
        "mistral_f16",
        MistralConfig,
        MistralForCausalLM,
        torch.float16,
        dict(rope_theta=10000.0, sliding_window=None, tie_word_embeddings=False),
    ),
]


def randomise(model, generator):
    """Gives every norm weight and bias a non-trivial value so each one is exercised."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.copy_(1.0 + 0.2 * torch.randn(parameter.shape, generator=generator))
            elif name.endswith(".bias"):
                parameter.copy_(0.2 * torch.randn(parameter.shape, generator=generator))


def write_fixture(root, name, config_class, model_class, dtype, overrides):
    torch.manual_seed(48)
    generator = torch.Generator().manual_seed(4848)
    settings = dict(COMMON)
    settings.update(overrides)
    config = config_class(**settings)
    config._attn_implementation = "eager"
    model = model_class(config).eval()
    randomise(model, generator)
    model = model.to(dtype)
    directory = os.path.join(root, name)
    os.makedirs(directory, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    generation_config = os.path.join(directory, "generation_config.json")
    if os.path.exists(generation_config):
        os.remove(generation_config)

    reference = model_class.from_pretrained(
        directory, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        logits = reference(torch.tensor([TOKENS])).logits[0].to(torch.float32)
    record = dict(
        generator="scripts/native_decoder_fixtures.py",
        torch=torch.__version__,
        transformers=transformers.__version__,
        stored_dtype=str(dtype).removeprefix("torch."),
        tokens=TOKENS,
        vocab_size=config.vocab_size,
        logits=[float(value) for value in logits.flatten().tolist()],
    )
    with open(os.path.join(directory, "reference.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, separators=(",", ":"))
        handle.write("\n")


def write_checkpoint_reference(model_dir, output):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    messages = [{"role": "user", "content": "Name three uses of cadmium and explain one."}]
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if not isinstance(tokens, list):
        tokens = tokens["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        logits = model(torch.tensor([tokens])).logits[0].to(torch.float32)
    record = dict(
        generator="scripts/native_decoder_fixtures.py --reference",
        torch=torch.__version__,
        transformers=transformers.__version__,
        tokens=tokens,
        argmax=[int(i) for i in logits.argmax(dim=-1).tolist()],
        last_logits=[float(v) for v in logits[-1].tolist()],
    )
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(record, handle, separators=(",", ":"))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--reference":
        write_checkpoint_reference(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) != 2:
        sys.exit(
            "usage: native_decoder_fixtures.py <output directory>\n"
            "       native_decoder_fixtures.py --reference <model directory> <output.json>"
        )
    for fixture in FIXTURES:
        write_fixture(sys.argv[1], *fixture)


if __name__ == "__main__":
    main()
