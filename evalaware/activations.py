import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from .models import ModelSpec, load_tokenizer, prompt_only
from .parsing import split_reasoning

# Where the final norm lives; checkpoint and module names can differ per family.
FINAL_NORMS = ("model.norm", "model.language_model.norm", "language_model.model.norm")


def _save(path: Path, layers, row: dict, n: int | None = None, **extra) -> None:
    """Save the last-token and mean readouts of the first n tokens.

    layers: one [T, d] tensor per decoder layer.
    """
    last = torch.stack([h[:n][-1] for h in layers]).float().cpu().numpy()
    mean = torch.stack([h[:n].float().mean(0) for h in layers]).cpu().numpy()
    np.savez(path, last_token=last, mean_token=mean, label=row["label"],
             prompt_id=row["prompt_id"], source=row["source"],
             n_layers=len(layers), d_model=last.shape[1], **extra)


def _save_both(out_dir: Path, row: dict, layers, n_prompt: int) -> None:
    # The model is causal, so the prompt positions of the prompt + reply pass
    # are the prompt-only readout; no second forward pass needed.
    _save(out_dir / "activations" / f"{row['id']}.npz", layers, row)
    _save(out_dir / "activations_prompt" / f"{row['id']}.npz", layers, row, n_prompt,
          n_prompt_tokens=n_prompt, readout="prompt_only")


def _entry(row: dict, prompt: str, n_prompt: int, ids, tok, spec: ModelSpec,
           config: dict, elapsed: float, **extra) -> dict:
    text = tok.decode(ids, skip_special_tokens=True)
    reasoning, answer = split_reasoning(text, gen_ids=torch.as_tensor(ids), tok=tok)
    return {
        "id": row["id"], "prompt_id": row["prompt_id"], "source": row["source"],
        "category": row["category"], "ground_truth_label": row["label"],
        "prompt_text": row["text"], "templated_prompt": prompt,
        "generated_text": text, "reasoning_trace": reasoning, "final_answer": answer,
        "n_prompt_tokens": n_prompt, "n_generated_tokens": len(ids),
        "generation_config": dict(config), "model": spec.hf_id, "elapsed_s": elapsed, **extra,
    }


def _write_traces(out_dir: Path, entries: list[dict]) -> None:
    (out_dir / "reasoning_traces.json").write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_dirs(out_dir: Path) -> None:
    for d in (out_dir, out_dir / "activations", out_dir / "activations_prompt"):
        d.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def generate_and_capture(model, tok, spec: ModelSpec, rows: list[dict], out_dir: Path,
                         max_new_tokens: int = 4096, temperature: float = 0.7,
                         top_p: float = 0.9, capture: bool = True) -> list[dict]:
    _make_dirs(out_dir)
    config = {"max_new_tokens": max_new_tokens, "temperature": temperature,
              "top_p": top_p, "enable_thinking": spec.enable_thinking}
    entries = []
    for row in rows:
        started = time.time()
        prompt = prompt_only(tok, row["text"], spec)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        n_prompt = inputs["input_ids"].shape[1]
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                             temperature=max(temperature, 1e-5), top_p=top_p,
                             pad_token_id=tok.pad_token_id)
        entries.append(_entry(row, prompt, n_prompt, gen[0, n_prompt:], tok, spec, config,
                              round(time.time() - started, 2)))
        _write_traces(out_dir, entries)
        if capture:
            hidden = model(gen, attention_mask=torch.ones_like(gen), output_hidden_states=True).hidden_states
            _save_both(out_dir, row, [h[0] for h in hidden[1:]], n_prompt)
    return entries


def _final_norm(model_path: str):
    """The model's final norm. vLLM saves the last layer before it, HF after.

    The formula differs by family (Qwen3.5 multiplies by 1 + weight), so the
    model is built on the meta device for its own norm module, and only that
    one weight is loaded.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    from .models import _needs_remote_code

    try:
        shards = json.loads(Path(hf_hub_download(model_path, "model.safetensors.index.json")).read_text())["weight_map"]
    except Exception:  # single-file checkpoint, no index
        with safe_open(hf_hub_download(model_path, "model.safetensors"), "pt") as f:
            shards = {k: "model.safetensors" for k in f.keys()}
    key = next((f"{n}.weight" for n in FINAL_NORMS if f"{n}.weight" in shards), None)

    remote = _needs_remote_code(model_path)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=remote)
    module = None
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            with torch.device("meta"):
                modules = dict(cls.from_config(cfg, trust_remote_code=remote).named_modules())
        except Exception:
            continue
        module = next((modules[n] for n in FINAL_NORMS if n in modules), None)
        if module is not None:
            break
    if key is None or module is None:
        raise SystemExit(f"could not find the final norm of {model_path}")

    norm = module.to_empty(device="cpu")
    with safe_open(hf_hub_download(model_path, shards[key]), "pt") as f:
        norm.weight.data.copy_(f.get_tensor(key))
    norm.float()
    return lambda x: norm(x).detach()


def _shutdown(llm) -> None:
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def generate_and_capture_vllm(model_path: str, spec: ModelSpec, rows: list[dict], out_dir: Path,
                              max_new_tokens: int = 4096, temperature: float = 0.7,
                              top_p: float = 0.9, capture: bool = True, gpu_mem: float = 0.85,
                              hidden_tmp: Path = Path("/nvme/_hidden_states"),
                              sampling: dict | None = None, capture_backend: str = "vllm") -> list[dict]:
    """Same outputs as generate_and_capture, using vLLM.

    Pass 1: a plain engine generates every reply in one batch, then shuts down.
    Pass 2: an engine with extract_hidden_states runs prefill only over
    prompt + reply, which gives both readouts. Two engines because extraction
    reserves cache for every layer's hidden state, which would leave generation
    almost unbatched. Prefix caching stays off: cached blocks return wrong states.

    Extraction memory grows with layers^2 x tokens (about 10 MB per token for an
    8B model), so long thinking replies do not fit. capture_backend="hf" runs
    pass 2 as one HF forward pass per row instead.

    Resumable: generated rows and rows with both .npz files are skipped.
    """
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
        cleanup_hidden_states, load_hidden_states)
    from vllm.inputs import TokensPrompt

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True).get_text_config()
    tok = load_tokenizer(model_path)
    _make_dirs(out_dir)
    hidden_tmp.mkdir(parents=True, exist_ok=True)
    traces_path, ids_path = out_dir / "reasoning_traces.json", out_dir / "generated_token_ids.json"
    gen_ids = json.loads(ids_path.read_text()) if ids_path.exists() else {}
    entries = json.loads(traces_path.read_text(encoding="utf-8")) if traces_path.exists() else []
    entries = [e for e in entries if e["id"] in gen_ids]

    prompts = {r["id"]: prompt_only(tok, r["text"], spec) for r in rows}
    prompt_ids = {k: tok(v, add_special_tokens=False)["input_ids"] for k, v in prompts.items()}
    longest = max(map(len, prompt_ids.values()))
    # Longrope (Phi-3 128k): past original_max_position_embeddings vLLM switches
    # every sequence to the long factors, HF only the longer ones. Stay under it.
    rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    limit = getattr(cfg, "original_max_position_embeddings", None)
    if limit and (rope.get("rope_type") or rope.get("type")) == "longrope":
        max_new_tokens = min(max_new_tokens, limit - longest - 8)
        print(f"longrope: capping max_new_tokens at {max_new_tokens} so sequences stay <= {limit}")

    todo = [r for r in rows if r["id"] not in gen_ids]
    if todo:
        sampling = sampling or {}
        config = {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p,
                  **sampling, "enable_thinking": spec.enable_thinking, "backend": "vllm",
                  "batch_size": len(todo)}
        llm = LLM(model=model_path, dtype="bfloat16", trust_remote_code=True,
                  max_model_len=longest + max_new_tokens + 8, gpu_memory_utilization=gpu_mem)
        started = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids[r["id"]]) for r in todo],
                            SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                                           top_p=top_p, **sampling))
        elapsed = round(time.time() - started, 2)  # wall time of the whole batch
        _shutdown(llm)
        for row, out in zip(todo, outs):
            ids = list(out.outputs[0].token_ids)
            gen_ids[row["id"]] = ids
            entries.append(_entry(row, prompts[row["id"]], len(prompt_ids[row["id"]]), ids, tok,
                                  spec, config, elapsed, finish_reason=out.outputs[0].finish_reason))
        order = {r["id"]: i for i, r in enumerate(rows)}
        entries.sort(key=lambda e: order[e["id"]])
        ids_path.write_text(json.dumps(gen_ids))
        _write_traces(out_dir, entries)
        cut = sum(e.get("finish_reason") == "length" for e in entries)
        print(f"generated {len(todo)} replies in {elapsed}s; "
              f"{cut}/{len(entries)} hit the {max_new_tokens}-token limit")

    todo = [r for r in rows if not ((out_dir / "activations" / f"{r['id']}.npz").exists()
                                    and (out_dir / "activations_prompt" / f"{r['id']}.npz").exists())]
    if not (capture and todo):
        return entries
    if capture_backend == "hf":
        from .models import load_model

        # Mamba layers without the fused kernels build one temporary that grows with
        # tokens x chunk_size and another with (tokens / chunk_size)^2. For an 8k-token
        # row on Nemotron 3 Nano, 128 needs 32 GB and 32 needs 128 GB; 96 peaks near 26 GB.
        # ponytail: fixed 96, sized for ~8k-token rows; the balance point is about
        # (tokens * mamba_head_dim) ** (1/3), recompute if rows get much longer.
        model, _ = load_model(model_path, **({"chunk_size": 96} if getattr(cfg, "chunk_size", None) else {}))
        with torch.no_grad():
            for i, row in enumerate(todo, 1):
                ids = torch.tensor([prompt_ids[row["id"]] + gen_ids[row["id"]]], device=model.device)
                hidden = model(ids, output_hidden_states=True).hidden_states
                _save_both(out_dir, row, [h[0] for h in hidden[1:]], len(prompt_ids[row["id"]]))
                del hidden
                torch.cuda.empty_cache()  # long rows on a big model otherwise run out of memory
                if i % 10 == 0 or i == len(todo):
                    print(f"captured activations for {i}/{len(todo)} rows (hf)")
        return entries
    norm = _final_norm(model_path)
    # Size to the longest real sequence, not the generation cap: extraction
    # reserves cache for every layer's hidden state per token.
    ext_len = max(len(prompt_ids[r["id"]]) + len(gen_ids[r["id"]]) for r in todo) + 8
    llm = LLM(model=model_path, dtype="bfloat16", trust_remote_code=True,
              max_model_len=ext_len, max_num_batched_tokens=ext_len,
              enable_chunked_prefill=False, enable_prefix_caching=False,
              gpu_memory_utilization=gpu_mem,
              speculative_config={"method": "extract_hidden_states", "num_speculative_tokens": 1,
                                  "draft_model_config": {"hf_config": {
                                      "eagle_aux_hidden_state_layer_ids": list(range(1, cfg.num_hidden_layers + 1))}}},
              kv_transfer_config=KVTransferConfig(
                  kv_connector="ExampleHiddenStatesConnector", kv_role="kv_producer",
                  kv_connector_extra_config={"shared_storage_path": str(hidden_tmp)}))
    for start in range(0, len(todo), 16):  # 16 at a time bounds the temp files on disk
        batch = todo[start:start + 16]
        outs = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids[r["id"]] + gen_ids[r["id"]])
                             for r in batch], SamplingParams(max_tokens=1))
        for row, out in zip(batch, outs):
            path = out.kv_transfer_params["hidden_states_path"]
            h = load_hidden_states(path)["hidden_states"].float()  # [T, n_layers, d]; waits for the async write
            cleanup_hidden_states(path)
            h[:, -1] = norm(h[:, -1])
            _save_both(out_dir, row, h.unbind(1), len(prompt_ids[row["id"]]))
        print(f"captured activations for {min(start + 16, len(todo))}/{len(todo)} rows")
    _shutdown(llm)
    return entries


@torch.no_grad()
def capture_prompt_only(model, tok, traces: list[dict], out_dir: Path, model_path: str) -> int:
    """Prompt-only activations for traces made without generate (make_traces.py)."""
    trace_model = next((t.get("model") for t in traces if t.get("model")), None)
    if trace_model and trace_model != model_path:
        raise SystemExit(f"traces came from {trace_model!r}, not {model_path!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in traces:
        if not t.get("templated_prompt"):
            raise SystemExit(f"{t['id']} has no templated_prompt")
        inputs = tok(t["templated_prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
        n_tok = int(inputs["input_ids"].shape[1])
        if t.get("n_prompt_tokens") not in (None, n_tok):
            raise SystemExit(f"{t['id']}: retokenised to {n_tok}, run recorded {t['n_prompt_tokens']}")
        hidden = model(**inputs, output_hidden_states=True).hidden_states
        row = {"label": t["ground_truth_label"], "prompt_id": t["prompt_id"], "source": t["source"]}
        _save(out_dir / f"{t['id']}.npz", [h[0] for h in hidden[1:]], row,
              n_prompt_tokens=n_tok, readout="prompt_only")
    return len(traces)
