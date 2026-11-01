# Native model text adapter

Item 2 of the distributed inference work is the `kadmium_runtime::model_text` library. Its
entry point is `ModelText::from_model_dir`. This is a library integration point, with no CLI,
HTTP route, sampling engine or distributed generation loop in this change.

The input directory contains `config.json`, `tokenizer.json` and `tokenizer_config.json`.
Optional inputs are `generation_config.json` and `chat_template.jinja`. Named templates may
also live in `tokenizer_config.json`. A standalone Jinja file takes precedence for the default
template. There is no model-name-based template guess.

For legacy exports without `added_tokens_decoder`, `special_tokens_map.json` supplies overrides
used by both chat rendering and the BOS/EOS postprocessor check. Modern exports use the config.
The template receives populated Hugging Face special-token attributes, including
`additional_special_tokens`, without tokenizer configuration flags or unset tokens. These
selection rules are **Proved (model)** by `Kadmium.Native.Text.legacy_template_override`,
`Kadmium.Native.Text.modern_template_special` and `Kadmium.Native.Text.template_visible_iff`;
see the formal inventory for scope and the differential fixtures.

```rust,ignore
use kadmium_runtime::model_text::{ModelText, TemplateValue};
use std::collections::BTreeMap;
use std::path::Path;

let model = ModelText::from_model_dir(Path::new("/models/checkpoint"))?;
let messages: Vec<TemplateValue> = serde_json::from_str(
    r#"[{"role":"user","content":"Explain the result."}]"#,
)?;
// The caller supplies a template execution budget. Exhaustion returns an error.
let tokens = model.encode_chat(&messages, true, None, &BTreeMap::new(), template_fuel)?;
let defaults = model.generation_defaults();
```

Deserialize messages and tool arguments directly into `TemplateValue`: passing them through a
sorted map first loses the insertion order used when the template prints JSON. Model-specific
arguments, such as `enable_thinking`, go in the template-variable map. Call `encode` for plain
text, selecting whether to run the tokenizer's special-token postprocessor. Call `decode` with
the complete token sequence, choosing whether to hide special tokens. It is not an incremental
streaming decoder.

`GenerationDefaults` exposes common sampling fields, the EOS list, BOS/PAD ids, stop strings,
and the complete declared generation metadata. Sampling defaults absent from the file follow
Transformers `GenerationConfig`; no sampler is implemented here. A future generator must
implement or reject every active setting in `declared`, including settings outside the common
fields. The declared `max_length` and `max_new_tokens` remain metadata; this adapter does not
use them to cap generation. `find_stop` searches the complete decoded generated text, excluding
the prompt. Streaming stop buffering and token selection belong to the coordinator work.

## Quality contract and proof scope

The context gate is **Proved (model)** by `Kadmium.Native.Text.accepted_unchanged`,
`Kadmium.Native.Text.full_context_accepted` and `Kadmium.Native.Text.over_context_rejected`.
It accepts the original token list through the native context boundary and rejects a longer
list. EOS membership is **Proved (model)** by `Kadmium.Native.Text.eos_iff_member`.
See [the formal inventory](../../formal/README.md#native-text-adapter-nativemodeltextlean) for the Rust link,
differential fixture and special-token precedence proofs. Tokenization and rendering are
opaque inputs to these proofs; neither library is proved equivalent to Hugging Face.

**Measured comparison**, reproduced by the commands below: the committed synthetic BPE and
Unigram fixtures compare token ids, decoded strings, rendered chats and common sampling fields
against Transformers 4.57.6, tokenizers 0.22.2 and Jinja2 3.1.6. These are test fixtures, not
pretrained models or throughput benchmarks.

```powershell
python scripts/model_text_fixtures.py
cargo test -p kadmium-runtime --test model_text --locked
```

For a downloaded checkpoint's text assets:

```powershell
python scripts/model_text_fixtures.py --model MODEL_DIR --output MODEL_DIR
$env:KADMIUM_MODEL_TEXT_DIR = 'MODEL_DIR'
cargo test -p kadmium-runtime --test model_text --locked real_checkpoint_matches_transformers -- --ignored
```

Text-only comparisons were run on Windows on 2026-10-31 using the pinned Python dependencies:

| Checkpoint | Hugging Face revision |
| :--- | :--- |
| `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` |
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |

## Current limitations

- Only exported `tokenizer.json` assets are accepted; `tokenizer.model` alone needs an export.
- The native decoder's architecture/configuration checks still apply. Text assets do not add
  support for another architecture, sliding-window attention or quantized weights.
- Wrapper options `add_prefix_space=true` and `split_special_tokens=true` are rejected. Conflicting
  added-token metadata or BOS/EOS postprocessor settings require matching exported assets.
  Resolved special tokens must already be registered as special in `tokenizer.json`. Merging
  nonempty legacy additional-token lists containing object descriptors is refused; export modern
  matching assets for that case. String-list merge cases have Transformers differential coverage.
- MiniJinja supports the exercised templates, not every Python/Jinja extension. Missing required
  values, unsupported functions and exhausted template fuel return errors. `strftime_now` is not
  provided. Multimodal message content is rejected.
- The `tojson` adapter accepts default formatting and integer indentation through 16 spaces.
  `ensure_ascii=true`, `sort_keys=true`, custom separators and floating-point JSON values are
  rejected; Python-compatible floating-point formatting remains follow-up work.
- Metadata and template files have a 4 MiB parser guard; tokenizer files have a 256 MiB guard.
  Exceeding either guard is an error. These are artifact parser limits, not context limits.

The runtime dependencies are Rust `tokenizers`, `minijinja` and `minijinja-contrib`. Tokenizers'
default native-regex and C++ trainer features are disabled. Python is fixture tooling only.
