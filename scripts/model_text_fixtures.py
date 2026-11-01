"""Offline differential text fixtures. Test tooling only; the runtime never invokes Python.

Install transformers==4.57.6 tokenizers==0.22.2 jinja2==3.1.6, then run this script.
These are synthetic tokenizers and representative templates, not pretrained model assets.
Use --model DIR --output DIR to compare an actual downloaded checkpoint without its weights.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import jinja2
import tokenizers
import transformers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizerFast


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def references(tokenizer, directory):
    texts = ["Hello, world.", "  tabs\tand\nnewlines  ", "naïve café 中文 🦀", "<s> literal </s>", "I 'm sure ."]
    plain = []
    for text in texts:
        for special in (False, True):
            ids = tokenizer.encode(text, add_special_tokens=special)
            plain.append(dict(text=text, special=special, ids=ids,
                              decoded=tokenizer.decode(ids, skip_special_tokens=False),
                              skipped=tokenizer.decode(ids, skip_special_tokens=True)))
    chats = []
    conversations = [
        [dict(role="user", content="Hello, café 中文 🦀")],
        [dict(role="user", content="first"), dict(role="assistant", content="answer"),
         dict(role="user", content="second\nline")],
    ]
    for messages in conversations:
        for generation in (False, True):
            for variables in ({}, {"enable_thinking": False}):
                prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=generation, **variables)
                ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                                    add_generation_prompt=generation, **variables)
                chats.append(dict(messages=messages, generation=generation, variables=variables,
                                  prompt=prompt, ids=ids))
    if "tool" in str(tokenizer.chat_template):
        tools = [{"type": "function", "function": {"name": "weather", "description": "Weather in café <city>",
                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
        messages = [dict(role="user", content="Weather in Paris"),
                    dict(role="assistant", content="", tool_calls=[{"type": "function", "function": {
                        "name": "weather", "arguments": {"city": "Paris"}}}]),
                    dict(role="tool", content="sunny")]
        variables = {"tools": tools}
        chats.append(dict(messages=messages, generation=True, variables=variables,
                          prompt=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **variables),
                          ids=tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **variables)))
    write_json(directory / "reference.json", dict(
        provenance=dict(generator="scripts/model_text_fixtures.py", transformers=transformers.__version__,
                        tokenizers=tokenizers.__version__, jinja2=jinja2.__version__),
        plain=plain, chats=chats))


def synthetic(root):
    templates = {
        "byte_bpe": "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% if enable_thinking is defined and not enable_thinking %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}",
        "unigram": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('roles must alternate') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'].strip() + ' [/INST]' }}{% else %}{{ ' ' + message['content'] + eos_token }}{% endif %}{% endfor %}",
    }
    special = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"]
    templates["byte_bpe"] = ("{% if tools %}{% for tool in tools %}{{ tool | tojson }}\n{% endfor %}{% endif %}"
                              + templates["byte_bpe"]
                              + "{% for message in messages %}{% if message.tool_calls %}{{ message.tool_calls | tojson(indent=4) }}{% endif %}{% endfor %}")
    corpus = ["Hello world first answer second line tabs and newlines naïve café 中文 🦀 I 'm sure .",
              "user assistant system [INST] [/INST] <think> </think>"]
    for name, template in templates.items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        if name == "byte_bpe":
            backend = Tokenizer(models.BPE())
            backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            backend.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(vocab_size=320, special_tokens=special,
                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        else:
            backend = Tokenizer(models.Unigram())
            backend.pre_tokenizer = pre_tokenizers.Metaspace()
            backend.decoder = decoders.Metaspace()
            trainer = trainers.UnigramTrainer(vocab_size=80, special_tokens=special, unk_token="<unk>", show_progress=False)
        backend.train_from_iterator(corpus, trainer)
        backend.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
                                           unk_token="<unk>", additional_special_tokens=special[3:],
                                           chat_template=template, clean_up_tokenization_spaces=False)
        tokenizer.save_pretrained(directory)
        config = dict(model_type="qwen2", architectures=["Qwen2ForCausalLM"], vocab_size=len(tokenizer),
                      hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=8192, rms_norm_eps=1e-6,
                      eos_token_id=2, bos_token_id=1, tie_word_embeddings=True)
        write_json(directory / "config.json", config)
        generation = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                          repetition_penalty=1.05, eos_token_id=[2, 4], bos_token_id=1,
                          stop_strings=["END", "结束"], max_new_tokens=17)
        write_json(directory / "generation_config.json", generation)
        resolved = GenerationConfig(**generation)
        write_json(directory / "generation_reference.json", {
            key: getattr(resolved, key) for key in ["do_sample", "temperature", "top_p", "top_k", "min_p",
                                                  "typical_p", "repetition_penalty", "eos_token_id"]})
        references(tokenizer, directory)
    special_token_references(root)


def special_token_references(root):
    """Reuse the committed BPE asset so legacy cases do not retrain or renumber its vocabulary."""
    template = ("{% for m in messages %}{{ m.content }}{{ eos_token | default('unset') }}{% endfor %}"
                "|bos={{ bos_token | default('unset') }}|pad={{ pad_token is defined }}"
                "|flags={{ add_bos_token is defined }},{{ add_eos_token is defined }}"
                "|extra={{ additional_special_tokens | default([]) | tojson }}")
    cases = [
        dict(name="legacy_override", modern=False, config={}, special_map={"eos_token": "<|im_end|>"}),
        dict(name="legacy_descriptor_and_bos", modern=False,
             config={"bos_token": "</s>", "add_bos_token": True, "add_eos_token": False},
             special_map={"bos_token": {"content": "<s>"}, "eos_token": {"content": "<|im_end|>"}}),
        dict(name="legacy_missing_map", modern=False, config={}, special_map=None),
        dict(name="legacy_null_and_union", modern=False,
             config={"pad_token": None, "additional_special_tokens": ["<|im_start|>"]},
             special_map={"eos_token": None, "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}),
        dict(name="legacy_empty", modern=False, config={}, special_map={"eos_token": "", "additional_special_tokens": []}),
        dict(name="modern_ignores_map", modern=True, config={}, special_map={"eos_token": "<|im_end|>"}),
        dict(name="modern_context_keys", modern=True,
             config={"pad_token": None, "add_bos_token": True, "add_eos_token": False,
                     "additional_special_tokens": [{"__type": "AddedToken", "content": "<|im_end|>", "special": True}]},
             special_map={"eos_token": "<|im_end|>", "pad_token": "</s>"}),
    ]
    for case in cases:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "model"
            shutil.copytree(root / "byte_bpe", directory)
            config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
            if not case["modern"]:
                config.pop("added_tokens_decoder")
            config.update(case["config"])
            write_json(directory / "tokenizer_config.json", config)
            if case["special_map"] is None:
                (directory / "special_tokens_map.json").unlink()
            else:
                write_json(directory / "special_tokens_map.json", case["special_map"])
            (directory / "chat_template.jinja").write_text(template, encoding="utf-8")
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
            messages = [{"role": "user", "content": "hi"}]
            case["messages"] = messages
            case["template"] = template
            case["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False)
            case["ids"] = tokenizer.apply_chat_template(messages, tokenize=True)
    write_json(root / "special_tokens_reference.json", dict(
        provenance=dict(generator="scripts/model_text_fixtures.py --special-tokens-only", transformers=transformers.__version__,
                        tokenizers=tokenizers.__version__, jinja2=jinja2.__version__), cases=cases))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--special-tokens-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("crates/kadmium-runtime/tests/fixtures/model_text"))
    args = parser.parse_args()
    if args.special_tokens_only:
        special_token_references(args.output)
    elif args.model:
        args.output.mkdir(parents=True, exist_ok=True)
        references(AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False), args.output)
    else:
        synthetic(args.output)
