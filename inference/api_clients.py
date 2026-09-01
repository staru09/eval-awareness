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
import re
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


def _is_openai_reasoning_model(model: str) -> bool:
    """Whether `model` is a reasoning-capable OpenAI model: the gpt-5.x
    family (gpt-5.4, gpt-5.5, gpt-5.6, gpt-5.6-sol/-terra/-luna, ...) and
    the older o-series (o1, o3, o4-mini, ...). These reject a non-default
    `temperature` over the Chat Completions API entirely -- confirmed via a
    real run against gpt-5.6-terra: "Unsupported value: 'temperature' does
    not support 0.7 with this model. Only the default (1) value is
    supported." -- and this holds even when thinking/reasoning isn't
    actively requested for that particular call (e.g. a plain follow-up
    turn), so this check is name-based, not tied to --enable-thinking.
    """
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def generate_openai(
    client,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    enable_thinking: bool = False,
    reasoning_effort: str = "medium",
) -> GenResult:
    """Generate a response, optionally with a captured reasoning trace.

    `enable_thinking=False` (default): plain Chat Completions call. No
    OpenAI model exposes a reasoning trace over this API regardless of
    model, so `reasoning_trace` is always None here.

    `enable_thinking=True`: routes to `_generate_openai_reasoning`, which
    uses the Responses API with `reasoning.effort`/`reasoning.summary`
    requested. This only works with an actual reasoning-capable model --
    the gpt-5.x family (gpt-5.6, gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna,
    gpt-5.5, gpt-5.4, ...) -- passing a non-reasoning model like gpt-4o
    with --enable-thinking will error, since the `reasoning` param isn't
    valid for it. `reasoning_effort`: one of none/minimal/low/medium/high/
    xhigh/max (model-dependent which are supported; "medium" is a safe
    default matching gpt-5.5's own default).
    """
    if enable_thinking:
        return _generate_openai_reasoning(
            client, model, messages, max_tokens=max_tokens, reasoning_effort=reasoning_effort,
        )

    is_reasoning = _is_openai_reasoning_model(model)
    kwargs: dict = dict(
        model=model,
        messages=messages,
        # Newer models (e.g. gpt-5.6-terra) reject `max_tokens` over Chat
        # Completions with a 400 ("Unsupported parameter") -- confirmed via
        # a real run -- and require `max_completion_tokens` instead.
        max_completion_tokens=max_tokens,
    )
    if not is_reasoning:
        kwargs["temperature"] = temperature

    LOG.debug(
        "-> OpenAI request: model=%s max_tokens=%d temperature=%s n_messages=%d last_msg=%r",
        model, max_tokens, temperature if not is_reasoning else "(omitted, reasoning model)",
        len(messages), _preview(messages[-1]["content"]),
    )
    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
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


def _generate_openai_reasoning(
    client, model: str, messages: list[dict], *, max_tokens: int, reasoning_effort: str = "medium",
) -> GenResult:
    """Generate via the Responses API with reasoning effort + summary requested.

    Reasoning models (the gpt-5.x family -- gpt-5.6, gpt-5.6-sol,
    gpt-5.6-terra, gpt-5.6-luna, gpt-5.5, gpt-5.4, ...) generally reject a
    `temperature` override, so none is sent here. `reasoning.effort` is a
    required-in-practice tuning knob (model-dependent defaults, "medium"
    for gpt-5.5) controlling how much the model reasons; `reasoning.summary
    ="auto"` asks the API to include a natural-language summary of that
    reasoning as a separate "reasoning" output item -- the raw reasoning
    tokens themselves are never exposed, only this summary. Plain output
    text comes back as "message" output items (or via the SDK's
    `output_text` convenience property, used as a fallback).
    """
    LOG.debug(
        "-> OpenAI (reasoning) request: model=%s max_output_tokens=%d effort=%s n_messages=%d last_msg=%r",
        model, max_tokens, reasoning_effort, len(messages), _preview(messages[-1]["content"]),
    )
    t0 = time.time()
    resp = client.responses.create(
        model=model,
        input=messages,
        reasoning={"effort": reasoning_effort, "summary": "auto"},
        max_output_tokens=max_tokens,
    )
    elapsed = time.time() - t0

    reasoning_parts, text_parts = [], []
    saw_reasoning_item = False
    for item in getattr(resp, "output", None) or []:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            saw_reasoning_item = True
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
    if saw_reasoning_item and not reasoning_trace:
        LOG.warning(
            "OpenAI response for %s included a 'reasoning' output item but its summary "
            "was empty -- reasoning_trace will be null. This usually means either (a) the "
            "org isn't verified for reasoning summaries (see "
            "https://platform.openai.com/settings/organization/general), or (b) the model "
            "judged this prompt too trivial to produce a visible summary at "
            "reasoning.effort=%s. Not necessarily a bug.",
            model, reasoning_effort,
        )
    elif not saw_reasoning_item:
        LOG.warning(
            "OpenAI response for %s had no 'reasoning' output item at all despite "
            "reasoning requested -- reasoning_trace will be null.", model,
        )
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
    reasoning_effort: str = "medium",
) -> GenResult:
    """Generate via the Messages API.

    `temperature`/`thinking_budget_tokens`/`reasoning_effort` are accepted
    for interface parity with generate_openai but NOT forwarded (Anthropic
    has no effort-level equivalent; `thinking={"type": "adaptive"}` picks
    its own level internally). The installed SDK's
    `Messages.create()` has no `temperature`/`top_p`/`top_k` parameter at
    all (confirmed against the installed anthropic package — a real
    `TypeError: unexpected keyword argument 'temperature'` from an earlier
    run caught this; sampling is apparently not client-configurable on
    this API version).

    With `enable_thinking=True`, requests extended thinking via
    `thinking={"type": "adaptive", "display": "summarized"}` (the SDK's
    currently-recommended variant; the older `{"type": "enabled",
    "budget_tokens": ...}` form is deprecated per this SDK's own runtime
    warning). `display: "summarized"` is required to get visible text --
    confirmed via docs (platform.claude.com/docs/en/build-with-claude/
    adaptive-thinking): on the newest models, `display` defaults to
    "omitted", which returns a real "thinking" content block but with its
    `thinking` text field deliberately blank, not the block itself missing
    -- matching a real run where reasoning_trace came back null despite
    thinking being enabled. Also note: at adaptive's default effort, Claude
    may skip thinking entirely for a trivial prompt, so an empty trace can
    still legitimately mean "didn't need to think," not just a config
    issue. Response content blocks of type "thinking" become
    reasoning_trace, type "text" becomes final_text.
    """
    kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=messages)
    if enable_thinking:
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}

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


# ---------------------------------------------------------------------------
# vLLM (self-hosted, OpenAI-compatible /v1/chat/completions)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def vllm_client(base_url: str):
    """vLLM's OpenAI-compatible server speaks the same wire protocol as the
    openai package's client, just pointed at a different base_url -- e.g. a
    vLLM instance you're hosting on a cloud GPU VM, per plan.md/README's
    "eval-aware-oss" precedent (vLLM behind an OpenAI-compatible endpoint).
    No real API key needed; vLLM ignores it unless you've configured
    --api-key server-side, in which case set VLLM_API_KEY.
    """
    from openai import OpenAI

    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=api_key)


def generate_vllm(
    client,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    enable_thinking: bool = False,
    reasoning_effort: str = "medium",  # accepted for interface parity, unused
) -> GenResult:
    """Generate via a self-hosted vLLM server's OpenAI-compatible chat
    endpoint. Unlike the real OpenAI API, vLLM serves open-weight models
    (Qwen3, DeepSeek-R1, ...) that emit their reasoning inline as
    `<think>...</think>` in the response text rather than as a separate
    Responses-API output item -- so reasoning capture here works like
    run_inference.py's local-HF path (regex-split the `<think>` block),
    not like generate_openai's Responses-API path.

    `enable_thinking` is forwarded as `chat_template_kwargs.enable_thinking`
    via `extra_body`, matching vLLM's documented mechanism for toggling
    Qwen3's thinking mode per-request over the OpenAI-compatible API.
    `reasoning_effort` has no vLLM/open-weight-model equivalent and is
    ignored. `max_tokens` (not `max_completion_tokens`) is used since vLLM's
    OpenAI-compatible server implements the older/stable Chat Completions
    parameter name.
    """
    LOG.debug(
        "-> vLLM request: model=%s max_tokens=%d temperature=%.2f enable_thinking=%s "
        "n_messages=%d last_msg=%r",
        model, max_tokens, temperature, enable_thinking, len(messages), _preview(messages[-1]["content"]),
    )
    t0 = time.time()
    kwargs: dict = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if enable_thinking:
        # Only attach chat_template_kwargs when actually toggling thinking on.
        # Some served tokenizers (e.g. Mistral's tekken/mistral-common
        # tokenizer, which has no Jinja chat template at all) reject *any*
        # chat_template_kwargs -- even {"enable_thinking": False} -- with a
        # 400 "chat_template is not supported for Mistral tokenizers."
        # confirmed via a real run. Omitting the field entirely for the
        # (default) False case sidesteps that for such models.
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    resp = client.chat.completions.create(**kwargs)
    elapsed = time.time() - t0
    choice = resp.choices[0]
    raw_text = (choice.message.content or "").strip()

    # With a server-side --reasoning-parser configured (recommended: match
    # it to the served model, e.g. `qwen3` for Qwen3), vLLM already splits
    # reasoning out of `content` into a separate, SDK-undeclared `reasoning`
    # field (only reachable via getattr/model_extra) -- there are no raw
    # <think> tags left in `content` to regex-split in that case. Fall back
    # to <think>-tag splitting only if the server has no reasoning parser
    # configured (content then still contains literal <think>...</think>).
    server_side_reasoning = getattr(choice.message, "reasoning", None)
    if server_side_reasoning is not None:
        reasoning_trace = server_side_reasoning.strip() or None
        final_text = raw_text
    else:
        m = _THINK_RE.search(raw_text)
        if m:
            reasoning_trace = m.group(1).strip() or None
            final_text = raw_text[m.end():].strip()
        else:
            reasoning_trace = None
            final_text = raw_text

    n_tokens = getattr(getattr(resp, "usage", None), "completion_tokens", None)
    LOG.debug(
        "<- vLLM response in %.2fs: id=%s finish_reason=%s completion_tokens=%s "
        "reasoning_trace=%s final_text=%r",
        elapsed, getattr(resp, "id", None), getattr(choice, "finish_reason", None),
        n_tokens, "present" if reasoning_trace else "none", _preview(final_text),
    )
    return GenResult(reasoning_trace=reasoning_trace, final_text=final_text, n_output_tokens=n_tokens)


PROVIDERS = {
    "openai": {"client_factory": openai_client, "generate": generate_openai},
    "anthropic": {"client_factory": anthropic_client, "generate": generate_anthropic},
    "vllm": {"client_factory": None, "generate": generate_vllm},  # client built with --base-url, see run_inference_api.py
}
