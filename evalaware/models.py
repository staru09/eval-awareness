from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class ModelSpec:
    hf_id: str
    chat_template: bool = True
    enable_thinking: bool = True


def load_model(model_path: str, dtype: str = "bfloat16"):

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtypes[dtype],
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    return model, tok


def apply_template(tok, messages: list[dict], spec: ModelSpec) -> str:
    if not spec.chat_template:
        return messages[-1]["content"]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=spec.enable_thinking,
        )
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


def layer_dims(model) -> tuple[int, int]:
    cfg = model.config.get_text_config()
    return cfg.num_hidden_layers, cfg.hidden_size
