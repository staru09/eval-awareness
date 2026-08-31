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
    """Generate a response, optionally with a captured reasoning trace.

    `enable_thinking=False` (default): plain Chat Completions call. No
    OpenAI model exposes a reasoning trace over this API regardless of
    model, so `reasoning_trace` is always None here.

    `enable_thinking=True`: routes to `_generate_openai_reasoning`, which
    uses the Responses API with `reasoning.summary` requested. This only
    returns a non-empty trace for an actual reasoning-capable model (o1,
    o3, o3-mini, gpt-5-thinking, ...) — passing a non-reasoning model like
    gpt-4o with --enable-thinking will error or simply come back with no
    reasoning trace, since the `reasoning` param isn't valid for it.
    """
    if enable_thinking:
        return _generate_openai_reasoning(client, model, messages, max_tokens=max_tokens)

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


def _generate_openai_reasoning(client, model: str, messages: list[dict], *, max_tokens: int) -> GenResult:
    """Generate via the Responses API with a reasoning summary requested.

    Reasoning models (o1/o3/o3-mini/gpt-5-thinking/...) generally reject a
    `temperature` override, so none is sent here. `reasoning.summary="auto"`
    asks the API to include a natural-language summary of the model's
    reasoning as a separate "reasoning" output item; plain output text
    comes back as "message" output items (or via the SDK's `output_text`
    convenience property, used as a fallback).
    """
    LOG.debug(
        "-> OpenAI (reasoning) request: model=%s max_output_tokens=%d n_messages=%d last_msg=%r",
        model, max_tokens, len(messages), _preview(messages[-1]["content"]),
    )
    t0 = time.time()
    resp = client.responses.create(
        model=model,
        input=messages,
        reasoning={"summary": "auto"},
        max_output_tokens=max_tokens,
    )
    elapsed = time.time() - t0

    reasoning_parts, text_parts = [], []
    for item in getattr(resp, "output", None) or []:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            for s in getattr(item, "summary", None) or []:
                text = getattr(s, "text", None) or getattr(s, "content", None)
                if text:
                    reasoning_parts.append(text)
        elif item_type == "message":
            for c in getattr(item, "content", None) or []:
                if getattr(c, "type", None) == "output_text":
                    text_parts.append(c.text)

    reasoning_trace = "\n".join(reasoning_parts).strip() or None
    final_text = "\n".join(text_parts).strip() or (getattr(resp, "output_text", None) or "")
    n_tokens = getattr(getattr(resp, "usage", None), "output_tokens", None)
    LOG.debug(
        "<- OpenAI (reasoning) response in %.2fs: id=%s output_tokens=%s "
        "reasoning_trace=%s final_text=%r",
        elapsed, getattr(resp, "id", None), n_tokens,
        "present" if reasoning_trace else "none", _preview(final_text),
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

    `temperature`/`thinking_budget_tokens` are accepted for interface
    parity with generate_openai but NOT forwarded: the installed SDK's
    `Messages.create()` has no `temperature`/`top_p`/`top_k` parameter at
    all (confirmed against the installed anthropic package — a real
    `TypeError: unexpected keyword argument 'temperature'` from an earlier
    run caught this; sampling is apparently not client-configurable on
    this API version).

    With `enable_thinking=True`, requests extended thinking via
    `thinking={"type": "adaptive"}` (the SDK's currently-recommended
    variant; the older `{"type": "enabled", "budget_tokens": ...}` form is
    deprecated per this SDK's own runtime warning). Response content
    blocks of type "thinking" become reasoning_trace, type "text" becomes
    final_text.
    """
    kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=messages)
    if enable_thinking:
        kwargs["thinking"] = {"type": "adaptive"}

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
