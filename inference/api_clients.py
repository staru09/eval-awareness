"""Thin provider adapters for OpenAI and Anthropic chat models.

Used by run_inference_api.py so the main script doesn't need to know the
per-provider request/response shape. Each adapter returns
(reasoning_trace, final_text): `reasoning_trace` is best-effort — it is
populated only when the provider/model actually exposes one (Anthropic
extended thinking; OpenAI reasoning-capable models via the Responses API),
and is `None` otherwise. This mirrors run_inference.py's
(reasoning_trace, final_answer) split for local Qwen3-8B, so downstream
scripts (verbalize_probe.py, gpt_judge.py) can treat all three sources
uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenResult:
    reasoning_trace: str | None
    final_text: str
    n_output_tokens: int | None = None


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def openai_client():
    from openai import OpenAI

    return OpenAI()


def generate_openai(
    client,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    enable_thinking: bool = False,
) -> GenResult:
    """Generate via the Chat Completions API.

    `enable_thinking=True` is a best-effort request only: most OpenAI
    models don't expose a separate reasoning trace over this API (the
    Responses API's `reasoning` output items would be needed for that, and
    their shape varies by model generation) — this adapter still checks for
    a `reasoning_content` field on the returned message and uses it if
    present, else reasoning_trace stays None.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = resp.choices[0]
    final_text = choice.message.content or ""
    reasoning_trace = getattr(choice.message, "reasoning_content", None)
    n_tokens = getattr(getattr(resp, "usage", None), "completion_tokens", None)
    return GenResult(reasoning_trace=reasoning_trace, final_text=final_text, n_output_tokens=n_tokens)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def anthropic_client():
    from anthropic import Anthropic

    return Anthropic()


def generate_anthropic(
    client,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    enable_thinking: bool = False,
    thinking_budget_tokens: int = 4096,
) -> GenResult:
    """Generate via the Messages API.

    With `enable_thinking=True`, requests extended thinking
    (`thinking={"type": "enabled", "budget_tokens": ...}`); Anthropic
    requires `temperature=1.0` and no `top_p`/`top_k` override when
    thinking is enabled, and `max_tokens` must exceed `thinking_budget_tokens`.
    Response content blocks of type "thinking" become reasoning_trace,
    type "text" becomes final_text.
    """
    kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=messages)
    if enable_thinking:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}
        kwargs["temperature"] = 1.0
        if max_tokens <= thinking_budget_tokens:
            kwargs["max_tokens"] = thinking_budget_tokens + max_tokens
    else:
        kwargs["temperature"] = temperature

    resp = client.messages.create(**kwargs)

    thinking_parts, text_parts = [], []
    for block in resp.content:
        if getattr(block, "type", None) == "thinking":
            thinking_parts.append(getattr(block, "thinking", ""))
        elif getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    reasoning_trace = "\n".join(thinking_parts).strip() or None
    final_text = "\n".join(text_parts).strip()
    n_tokens = getattr(getattr(resp, "usage", None), "output_tokens", None)
    return GenResult(reasoning_trace=reasoning_trace, final_text=final_text, n_output_tokens=n_tokens)


PROVIDERS = {
    "openai": {"client_factory": openai_client, "generate": generate_openai},
    "anthropic": {"client_factory": anthropic_client, "generate": generate_anthropic},
}
