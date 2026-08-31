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

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("api_clients")

_ENV_LOADED = False


def _preview(text: str | None, n: int = 200) -> str:
    if not text:
        return "(empty)"
    text = text.replace("\n", "\\n")
    return text[:n] + ("..." if len(text) > n else "")


def _ensure_env_loaded() -> None:
    """Load ../.env once (if python-dotenv is installed), so OPENAI_API_KEY /
    ANTHROPIC_API_KEY are picked up regardless of the caller's cwd. Never
    overrides an already-exported env var.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    except ImportError:
        pass
    _ENV_LOADED = True


def _require_env(var_name: str, provider: str) -> str:
    _ensure_env_loaded()
    value = os.environ.get(var_name)
    if not value:
        raise SystemExit(
            f"{var_name} is not set. Add it to eval-awareness/.env or export it "
            f"before running the {provider} benchmark."
        )
    return value


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

    api_key = _require_env("OPENAI_API_KEY", "OpenAI")
    return OpenAI(api_key=api_key)


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
    LOG.debug(
        "-> OpenAI request: model=%s max_tokens=%d temperature=%.2f n_messages=%d last_msg=%r",
        model, max_tokens, temperature, len(messages), _preview(messages[-1]["content"]),
    )
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.time() - t0
    choice = resp.choices[0]
    final_text = choice.message.content or ""
    reasoning_trace = getattr(choice.message, "reasoning_content", None)
    n_tokens = getattr(getattr(resp, "usage", None), "completion_tokens", None)
    LOG.debug(
        "<- OpenAI response in %.2fs: id=%s finish_reason=%s completion_tokens=%s "
        "reasoning_trace=%s final_text=%r",
        elapsed, getattr(resp, "id", None), getattr(choice, "finish_reason", None),
        n_tokens, "present" if reasoning_trace else "none", _preview(final_text),
    )
    return GenResult(reasoning_trace=reasoning_trace, final_text=final_text, n_output_tokens=n_tokens)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def anthropic_client():
    from anthropic import Anthropic

    api_key = _require_env("ANTHROPIC_API_KEY", "Anthropic")
    return Anthropic(api_key=api_key)


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

    LOG.debug(
        "-> Anthropic request: model=%s max_tokens=%d thinking=%s n_messages=%d last_msg=%r",
        model, kwargs["max_tokens"], enable_thinking, len(messages), _preview(messages[-1]["content"]),
    )
    t0 = time.time()
    resp = client.messages.create(**kwargs)
    elapsed = time.time() - t0

    thinking_parts, text_parts = [], []
    for block in resp.content:
        if getattr(block, "type", None) == "thinking":
            thinking_parts.append(getattr(block, "thinking", ""))
        elif getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    reasoning_trace = "\n".join(thinking_parts).strip() or None
    final_text = "\n".join(text_parts).strip()
    n_tokens = getattr(getattr(resp, "usage", None), "output_tokens", None)
    LOG.debug(
        "<- Anthropic response in %.2fs: id=%s stop_reason=%s output_tokens=%s "
        "reasoning_trace=%s final_text=%r",
        elapsed, getattr(resp, "id", None), getattr(resp, "stop_reason", None),
        n_tokens, "present" if reasoning_trace else "none", _preview(final_text),
    )
    return GenResult(reasoning_trace=reasoning_trace, final_text=final_text, n_output_tokens=n_tokens)


PROVIDERS = {
    "openai": {"client_factory": openai_client, "generate": generate_openai},
    "anthropic": {"client_factory": anthropic_client, "generate": generate_anthropic},
}
