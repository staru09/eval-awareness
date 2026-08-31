"""HF model loading helpers, shared by run_inference.py.

Mirrors the loading conventions used by ../../evaluation-awareness/cls_corr_mi
(device_map="auto", output_hidden_states=True, left-padding, chat-template +
enable_thinking passthrough) so activations/probe scores computed here stay
directly comparable with that pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSpec:
    hf_id: str
    chat_template: bool = True
    enable_thinking: bool = True


QWEN3_8B = ModelSpec(hf_id="Qwen/Qwen3-8B", chat_template=True, enable_thinking=True)

_DTYPES = None  # populated lazily inside load_hf_model, once torch is importable


def load_hf_model(model_path_or_id: str, *, dtype: str = "bfloat16"):
    """Load a HF causal LM + tokenizer, hidden states enabled.

    `device_map="auto"` shards across available GPUs; falls back to CPU if
    none are visible (slow, but works for smoke-testing the pipeline wiring).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path_or_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    return model, tok


def apply_chat_template(tok, user_text: str, spec: ModelSpec) -> str:
    """Wrap a single user turn in the model's chat template."""
    if not spec.chat_template:
        return user_text
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=spec.enable_thinking,
        )
    except TypeError:
        return tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )


def model_weight_summary(model) -> dict:
    """Cheap, storable summary of the model's weights (not the weights
    themselves — those live in the checkpoint). One L2 norm + shape per
    named parameter, so `output/model_summary.json` documents *what* was
    inspected without duplicating multi-GB checkpoint data on disk.
    """
    import torch

    summary = {
        "config": model.config.to_dict(),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "parameters": {},
    }
    with torch.no_grad():
        for name, p in model.named_parameters():
            summary["parameters"][name] = {
                "shape": list(p.shape),
                "dtype": str(p.dtype),
                "norm": float(p.float().norm().item()),
            }
    return summary
