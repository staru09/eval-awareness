from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

# Where decoder blocks live, in the order we try. Multimodal checkpoints such
# as Muse-Glimmer nest the text stack under language_model.
BLOCK_PATHS = (
    ("model", "layers"),
    ("model", "language_model", "layers"),
    ("language_model", "model", "layers"),
    ("transformer", "h"),
)


@dataclass
class ModelSpec:
    hf_id: str
    enable_thinking: bool = True


def _needs_remote_code(model_path: str) -> bool:
    """Only run a repo's own modeling code for architectures transformers lacks.

    Repos like Phi-3 still ship old modeling files that break on current
    transformers, even though the architecture is built in.
    """
    from transformers import PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    return PretrainedConfig.get_config_dict(model_path)[0].get("model_type") not in CONFIG_MAPPING


def load_tokenizer(model_path: str):
    """AutoTokenizer, except for Mistral repos.

    There transformers picks MistralCommonBackend, which tokenizes a rendered
    prompt's control tokens ([INST], <s>) as plain text and drops the template's
    default system prompt (the one that turns on [THINK] in Ministral Reasoning).
    """
    from transformers import PreTrainedTokenizerFast

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=_needs_remote_code(model_path))
    if type(tok).__name__ == "MistralCommonBackend":
        tok = PreTrainedTokenizerFast.from_pretrained(model_path, fix_mistral_regex=True)
    return tok


def load_model(model_path: str, dtype: str = "bfloat16", **config_overrides):
    tok = load_tokenizer(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs = dict(dtype=getattr(torch, dtype), device_map="auto", output_hidden_states=True,
                  trust_remote_code=_needs_remote_code(model_path), **config_overrides)
    failure = None
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            return cls.from_pretrained(model_path, **kwargs).eval(), tok
        except (ValueError, KeyError, AttributeError) as exc:
            failure = exc
    raise SystemExit(f"could not load {model_path}: {failure}")


def apply_template(tok, messages: list[dict], spec: ModelSpec) -> str:
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=spec.enable_thinking)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_only(tok, text: str, spec: ModelSpec) -> str:
    return apply_template(tok, [{"role": "user", "content": text}], spec)


def followup_prompt(tok, spec: ModelSpec, prompt: str, reply: str, question: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": question},
    ]
    return apply_template(tok, messages, spec)


def blocks(model):
    """The decoder block list, wherever this architecture keeps it."""
    for path in BLOCK_PATHS:
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        else:
            return node
    raise SystemExit(
        f"no decoder blocks found on {type(model).__name__}; add its path to models.BLOCK_PATHS"
    )
