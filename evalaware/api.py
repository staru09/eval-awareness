import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .parsing import THINK_RE, resolve_answer
from .verbalize import FIELDS


@dataclass
class Generation:
    reasoning: str | None
    text: str


def _key(name: str, provider: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set (required for {provider})")
    return value


def openai_client():
    from openai import OpenAI

    return OpenAI(api_key=_key("OPENAI_API_KEY", "OpenAI"))


def anthropic_client():
    from anthropic import Anthropic

    return Anthropic(api_key=_key("ANTHROPIC_API_KEY", "Anthropic"))


def vllm_client(base_url: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=os.environ.get("VLLM_API_KEY", "EMPTY"))


def _openai_reasoning(client, model, messages, max_tokens, effort) -> Generation:
    resp = client.responses.create(
        model=model, input=messages,
        reasoning={"effort": effort, "summary": "auto"},
        max_output_tokens=max_tokens,
    )
    reasoning, text = [], []
    for item in getattr(resp, "output", None) or []:
        kind = getattr(item, "type", None)
        if kind == "reasoning":
            reasoning += [s.text for s in (getattr(item, "summary", None) or []) if getattr(s, "text", None)]
        elif kind == "message":
            text += [c.text for c in (getattr(item, "content", None) or [])
                     if getattr(c, "type", None) == "output_text"]
    return Generation(
        "\n".join(reasoning).strip() or None,
        "\n".join(text).strip() or (getattr(resp, "output_text", None) or ""),
    )


def generate_openai(client, model, messages, max_tokens=2048, temperature=0.7,
                    enable_thinking=False, effort="medium") -> Generation:
    if enable_thinking:
        return _openai_reasoning(client, model, messages, max_tokens, effort)
    resp = client.chat.completions.create(
        model=model, messages=messages, max_completion_tokens=max_tokens, temperature=temperature,
    )
    message = resp.choices[0].message
    return Generation(getattr(message, "reasoning_content", None), message.content or "")


def generate_anthropic(client, model, messages, max_tokens=2048, temperature=1.0,
                       enable_thinking=False, effort="medium") -> Generation:
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if enable_thinking:
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
    resp = client.messages.create(**kwargs)
    reasoning = [b.thinking for b in resp.content if getattr(b, "type", None) == "thinking"]
    text = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return Generation("\n".join(reasoning).strip() or None, "\n".join(text).strip())


def generate_vllm(client, model, messages, max_tokens=2048, temperature=0.7,
                  enable_thinking=False, effort="medium") -> Generation:
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    message = resp.choices[0].message
    raw = message.content or ""
    server_reasoning = getattr(message, "reasoning", None)
    if server_reasoning is not None:
        reasoning = server_reasoning.strip() or None
        if not raw and reasoning:
            return Generation(None, reasoning)
        return Generation(reasoning, raw)
    m = THINK_RE.search(raw)
    if m:
        return Generation(m.group(1).strip() or None, raw[m.end():].strip())
    return Generation(None, raw)


PROVIDERS = {
    "openai": generate_openai,
    "anthropic": generate_anthropic,
    "vllm": generate_vllm,
}


def make_client(provider: str, base_url: str | None = None):
    if provider == "openai":
        return openai_client()
    if provider == "anthropic":
        return anthropic_client()
    if provider == "vllm":
        if not base_url:
            raise SystemExit("--base-url is required for provider=vllm")
        return vllm_client(base_url)
    raise SystemExit(f"unknown provider {provider!r}")


def generate_traces(client, provider: str, model: str, rows: list[dict], out_dir: Path,
                    max_tokens: int, temperature: float, enable_thinking: bool,
                    effort: str) -> list[dict]:
    generate = PROVIDERS[provider]
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in rows:
        result = generate(client, model, [{"role": "user", "content": row["text"]}],
                          max_tokens=max_tokens, temperature=temperature,
                          enable_thinking=enable_thinking, effort=effort)
        entries.append({
            "id": row["id"],
            "prompt_id": row["prompt_id"],
            "source": row["source"],
            "category": row["category"],
            "ground_truth_label": row["label"],
            "prompt_text": row["text"],
            "generated_text": result.text,
            "reasoning_trace": result.reasoning,
            "final_answer": result.text,
            "model": model,
        })
        (out_dir / "reasoning_traces.json").write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return entries


def verbalize_traces(client, provider: str, model: str, traces: list[dict], questions: list[dict],
                     out_csv: Path, max_tokens: int, temperature: float) -> list[dict]:
    generate = PROVIDERS[provider]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for t in traces:
            for q in questions:
                messages = [
                    {"role": "user", "content": t["prompt_text"]},
                    {"role": "assistant", "content": t["final_answer"]},
                    {"role": "user", "content": q["template_text"]},
                ]
                result = generate(client, model, messages, max_tokens=max_tokens,
                                  temperature=temperature, enable_thinking=False)
                signal, status = resolve_answer(result.text, q["answer"]["real"], q["answer"]["eval"])
                row = {
                    "transcript_id": t["id"],
                    "prompt_id": t["prompt_id"],
                    "probe_id": q["id"],
                    "source": t["source"],
                    "category": t["category"],
                    "ground_truth_label": t["ground_truth_label"],
                    "eval_signal": signal,
                    "parse_status": status,
                    "raw_response": result.text,
                }
                writer.writerow(row)
                rows.append(row)
    return rows
