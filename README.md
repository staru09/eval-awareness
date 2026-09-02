# LLM Evaluation-Awareness Experiments

A standalone pipeline for measuring whether language models can tell
they're being evaluated — and whether they say so. What started as a
single-model (Qwen3-8B) pilot has grown into two general phases, run
across many models:

- **Phase 1 — verbalized eval-awareness** (this README, `plan.md`): ask a
  model directly whether a transcript was an evaluation, across 5
  independently worded probes. Runs against **any** model — self-hosted
  via vLLM, via OpenAI/Anthropic's APIs, or via plain HF `transformers` as
  a fallback when a model's architecture isn't supported by the installed
  vLLM version. Currently run against **9 models**: see
  [Results](#results-verbalized-eval-awareness-9-models) below.
- **Phase 2 — internal representation, coupling, and causal steering**
  (`plan2.md`, `PROGRESS.md`): does a model represent "this is an
  evaluation" in its activations even when it doesn't say so? Follows the
  methodology of Heidari, Memarian & Rabusseau, *"Evaluation Awareness in
  Language Models: Representation, Verbalization, and Control"*
  (arXiv:2608.21766), run against 3 of the 9 models. See
  [Internal representation, coupling, and steering](#internal-representation-coupling-and-steering-phase-2)
  below for a summary, `plan2.md` for the full experimental design, or
  `explainer.md` for a plain-language walkthrough of the whole project
  (no ML background assumed).

The Phase 1 pipeline:

1. Builds a small mixed eval/deploy dataset (`data/build_dataset.py`).
2. Runs a model over it — locally via HF `transformers`
   (`inference/run_inference.py`, any `--model-path`), self-hosted via
   vLLM, or via a provider API (`inference/run_inference_api.py`) —
   capturing reasoning traces (and, for the local-HF path, hidden-state
   activations).
3. Asks the model to verbalize whether each transcript was an evaluation,
   across 5 probe-question phrasings (`probes/verbalize_probe.py`).

No `inspect_ai` dependency anywhere — one HF code path for generation,
judging, and hidden-state extraction, matching
`../evaluation-awareness/cls_corr_mi`'s conventions so results stay directly
comparable.

## Layout

```
eval-awareness/
├── plan.md                        # Phase 1 design rationale
├── plan2.md                       # Phase 2 design rationale (3 experiments + why)
├── PROGRESS.md                    # running log: every model run, bug found, and result (local, not committed)
├── explainer.md                   # plain-language walkthrough of the whole project
├── run.sh                         # single-model local-HF pipeline (any --model-path, default Qwen3-8B)
├── data/
│   ├── build_dataset.py           # assembles the 20-transcript eval/deploy dataset
│   ├── dataset.json                # 20 transcripts (10 eval, 10 deploy)
│   └── contrastive_pairs_16.txt   # 16 deploy-vs-eval-framed pairs, copied from cls_corr_mi/data/
├── inference/
│   ├── model_utils.py             # HF load/chat-template helpers (any --model-path)
│   ├── run_inference.py           # local HF: generation + hidden-state activation capture
│   ├── api_clients.py             # OpenAI/Anthropic/vLLM provider adapters (reads .env)
│   ├── run_inference_api.py       # single API/vLLM model: generation + verbalized probing
│   └── run_benchmark.py           # multi-model runner: logs + aggregated summary
├── probes/
│   ├── compute_probe.py           # Phase 2 Exp.1: 16-pair direction + random controls, per layer
│   ├── score_activations.py       # Phase 2 Exp.1: project saved activations onto probe
│   ├── steer.py                   # Phase 2 Exp.3: activation steering (Contrastive Activation Addition)
│   ├── run_internal_pipeline.sh   # orchestrates compute_probe -> run_inference -> score_activations
│   │                               # -> classify_auroc for one model (Phase 2 Exp.1)
│   ├── probe_questions.yaml       # 5 verbalized-probe phrasings (ported from eval-aware-oss)
│   └── verbalize_probe.py         # verbalized probing + label parsing (shared by both phases)
├── analysis/
│   ├── classify_auroc.py          # Phase 2 Exp.1: internal-probe AUROC (n=20, LOO accuracy)
│   ├── compare_internal_vs_verbal.py  # Phase 2 Exp.2: internal-vs-verbalized coupling (rho, MI)
│   ├── plot_results.py            # Phase 1 charts (ensemble AUROC, per-probe heatmap/lines)
│   ├── plot_internal_results.py   # Phase 2 Exp.1 charts
│   ├── plot_coupling.py           # Phase 2 Exp.2 charts
│   └── plot_steering.py           # Phase 2 Exp.3 charts
├── judges/
│   └── gpt_judge.py               # optional qualitative pass: GPT reads a model's own
│                                   # reasoning trace + verbalized answers, diagnoses cues
├── dashboard/                     # React+TS+Vite app, live view of every transcript/model
└── output/                        # run artifacts (git-ignored; created at runtime)
    ├── api/                       # Phase 1: one folder per model (reasoning traces, verbalized CSV/AUROC)
    ├── local/                     # Phase 2: one folder per model (activations, probe.pt, all 3 experiments' results)
    └── charts/                    # every generated plot, numbered by which script made it
```

## Requirements

A CUDA GPU for the local-HF path (`run_inference.py`, `probes/`) — VRAM
scales with the model (~16GB for Qwen3-8B in bf16; single GPU,
`device_map="auto"` shards across more if available). The API/vLLM path
(`run_inference_api.py`) needs no local GPU at all. Package management
uses `uv` throughout (`uv sync`, `uv run python ...`) — see this
workspace's package-management convention.
`torch`/`transformers`/`accelerate`/`numpy`/`scikit-learn`/`scipy`/`pyyaml`/`datasets`
are required for the local-HF path.

## Running

```bash
uv sync   # once you have a pyproject.toml/lockfile for this dependency set
./run.sh --model-path Qwen/Qwen3-8B   # or any other local HF model
```

Or step by step (see `run.sh` for the exact commands/paths), e.g. for a
quick generation-only smoke test before running the full pipeline:

```bash
cd inference
uv run python run_inference.py --model-path Qwen/Qwen3-8B --limit 2 --no-hidden-states
```

## Running the benchmark on OpenAI / Anthropic / vLLM models

`inference/run_inference_api.py` is a drop-in alternative to
`run_inference.py` for API (or API-shaped) models — they expose no hidden
states, so it covers generation + verbalized probing only (Phase 1). The
internal-representation/coupling/steering experiments (Phase 2, see
`plan2.md`) need raw hidden-state access, so they run only against the
local-HF path (`run_inference.py`, any `--model-path` — not tied to
Qwen3-8B, just whatever model you point it at). It reuses
`probes/verbalize_probe.py`'s exact label-parsing logic and
`probes/probe_questions.yaml`, and writes the same `reasoning_traces.json`
schema as `run_inference.py` (so `judges/gpt_judge.py` works unmodified on
any of the three), plus `verbalized_responses.csv` and
`results/verbalized_auroc.json` — all under
`output/api/<provider>__<model_slug>/` by default.

Requires `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` (loaded from `.env` via
`python-dotenv` if installed — **never commit `.env`**, it's in
`.gitignore`). `--provider vllm` needs no API key by default.

Thinking/reasoning-trace capture is **ON by default**:

```bash
cd inference
uv run python run_inference_api.py --provider openai --model gpt-5.6
uv run python run_inference_api.py --provider anthropic --model claude-sonnet-4-6
```

For Anthropic this requests extended thinking (its `thinking` content
blocks become `reasoning_trace`). For OpenAI it routes to the **Responses
API** with `reasoning.effort`/`reasoning.summary="auto"` requested — this
only works for an actual reasoning-capable model (the gpt-5.x family:
`gpt-5.6`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`,
`gpt-5.4`, ...); a non-reasoning model like `gpt-4o` doesn't accept the
`reasoning` param over this API and will error, so pass `--no-thinking`
when using a non-reasoning OpenAI model. Tune how hard it reasons with
`--reasoning-effort` (`none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`,
default `medium`, matching `gpt-5.5`'s own default — which values a given
model actually supports varies):

```bash
uv run python run_inference_api.py --provider openai --model gpt-4o --no-thinking
```

Without thinking, OpenAI goes through plain Chat Completions and
`reasoning_trace` stays `null` regardless of model.

### Self-hosted vLLM (e.g. a cloud GPU VM)

`--provider vllm` points the same script at any server speaking vLLM's
OpenAI-compatible endpoint — e.g. host a vLLM instance on a cloud GPU VM
and run this benchmark against it from your local machine, per
`eval-aware-oss/plan.md`'s original vLLM-serving precedent:

```bash
# On the VM:
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 --reasoning-parser qwen3

# Locally:
cd inference
uv run python run_inference_api.py \
  --provider vllm --model Qwen/Qwen3-8B \
  --base-url http://<vm-ip>:8000/v1
```

`run_inference_api.py --max-parallel N` runs N transcripts concurrently via a
thread pool (default 1, sequential) — safe to raise for a self-hosted vLLM
server, which continuously batches concurrent requests on the GPU rather than
queuing them serially. Check your server's own reported capacity first: vLLM
prints `Maximum concurrency for <ctx_len> tokens per request: <N>x` at
startup — that's the worst-case bound assuming every request fills the full
context window; with the short prompts in this dataset, real headroom is
much higher, so `--max-parallel 20` (all 20 transcripts at once) is fine on
a single modern GPU serving Qwen3-8B.

`--base-url` is required for this provider. No real API key is needed
(vLLM ignores it) unless you've configured `--api-key` server-side, in
which case set `VLLM_API_KEY` in `.env`. **Start the vLLM server with
`--reasoning-parser <name>` matching the served model** (`qwen3` for
Qwen3) — this makes vLLM itself split reasoning out of the response into a
separate `message.reasoning` field (not part of the OpenAI SDK's declared
schema, but reachable via `getattr`), rather than leaving raw
`<think>...</think>` tags inside `content`. `generate_vllm` in
`api_clients.py` reads that field directly when present, and only falls
back to regex-splitting literal `<think>` tags out of `content` if the
server has no reasoning parser configured. **Getting the parser wrong
(or omitting it) silently breaks things** — confirmed on a real run: with
`--reasoning-parser deepseek_r1` against Qwen3 (parser/model mismatch),
every verbalized-probe answer landed entirely in the `reasoning` field
with `content: null`, producing 100% parse failures across all 100
verbalized-probe calls (20 transcripts × 5 probes) — not a crash, just
silently empty output, so verify with a manual curl (see below) before
trusting a full run. `--enable-thinking`/`--no-thinking` toggles thinking
mode via vLLM's `chat_template_kwargs.enable_thinking` mechanism.
`--reasoning-effort` has no open-weight-model equivalent and is ignored
for this provider.

`generate_vllm` always attempts to send `chat_template_kwargs`, and only
retries without it if the server explicitly rejects it (confirmed real
error: `"chat_template is not supported for Mistral tokenizers"` — some
served tokenizers, e.g. Mistral's tekken/mistral-common tokenizer, have no
Jinja chat template at all and reject the field outright). **Do not
"fix" this by unconditionally omitting `chat_template_kwargs` when
`enable_thinking=False`** — that was tried and it silently breaks Qwen3-
family reasoning models the other way: their own chat template defaults
`enable_thinking` to `True` when the kwarg is absent, so a disable-thinking
follow-up call (e.g. the verbalized-probe questions, which always run with
thinking off regardless of the main generation's setting) keeps thinking
on, blows through its small token budget before reaching an answer, and
produces empty `content` — confirmed on a real run: 100% parse failures
across all 100 verbalized-probe calls for a Qwen3-based model, from this
exact unconditional-omission "fix."

Sanity-check the server before running the full benchmark:

```bash
curl -s http://<vm-ip>:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role":"user","content":"Hi, how are you?"}],
  "max_tokens": 100,
  "chat_template_kwargs": {"enable_thinking": false}
}'
# content should be non-null/non-empty here.
```

### Multi-model runner: `run_benchmark.py`

Runs several OpenAI/Anthropic/vLLM models in one command (one subprocess
per model, via `run_inference_api.py`, so one model's API error doesn't
take down the rest), **concurrently by default** — they're independent API
calls to different providers/models, so there's no reason to serialize
them (cap with `--max-parallel N`, or set it to `1` to force sequential
execution). Tees each model's output to both the console and a timestamped
per-model log file, and writes an aggregated comparison summary once
everything finishes. Include a `vllm:<model>` spec alongside OpenAI/
Anthropic ones and pass `--vllm-base-url` (one server/URL per invocation —
run separately if you have multiple vLLM servers), e.g.
`--models openai:gpt-5.6 vllm:Qwen/Qwen3-8B --vllm-base-url http://<vm-ip>:8000/v1`.

Thinking is **ON by default for every model**:

```bash
cd inference
uv run python run_benchmark.py --models openai:gpt-5.6 anthropic:claude-sonnet-4-6

# Smoke test: first 5 transcripts only
uv run python run_benchmark.py \
  --models openai:gpt-5.6 anthropic:claude-sonnet-4-6 --limit 5

# gpt-4o isn't a reasoning model -- disable thinking for it specifically:
uv run python run_benchmark.py \
  --models openai:gpt-4o anthropic:claude-sonnet-4-6 \
  --disable-thinking openai:gpt-4o

# Disable thinking for all models:
uv run python run_benchmark.py --models openai:gpt-4o anthropic:claude-sonnet-4-6 --disable-thinking

# Re-run only models that don't have results yet (e.g. after a partial failure):
uv run python run_benchmark.py --models openai:gpt-5.6 anthropic:claude-sonnet-4-6 --resume
```

Flags: `--models` (required, `provider:model` pairs), `--dataset`,
`--probe-questions`, `--out-root` (default `output/api/`), `--log-dir`
(default `output/logs/`), `--max-tokens`, `--temperature`,
`--reasoning-effort` (OpenAI only, default `medium`),
`--disable-thinking [spec ...]` (no args = disable for every model in
`--models`; with args, only for those specs — thinking is otherwise ON by
default for all), `--max-parallel` (default: all models at once), `--limit`,
`--resume`, `--summary-json`/`--summary-md` (override output paths).

Outputs:
- `output/logs/<provider>__<model>__<timestamp>.log` — full stdout/stderr per model run.
- `output/api/<provider>__<model>/` — per-model `reasoning_traces.json`, `verbalized_responses.csv`, `results/verbalized_auroc.json` (same as running `run_inference_api.py` directly).
- `output/api/Benchmark-<model-slug>-<timestamp>.json` + `.md` — cross-model comparison table for that invocation (ensemble AUROC, per-probe AUROC/accuracy/parse-failures, elapsed time), analogous to `EXPERIMENT_SUMMARY.md`'s six-model table. Every invocation gets its own timestamped pair — past runs are never overwritten, so you can review history across runs. `<model-slug>` encodes the `--models` you passed (e.g. `openai-gpt-4o_anthropic-claude-sonnet-4-6`); override the path with `--summary-json`/`--summary-md` if you want a fixed name instead.

Exits non-zero if any model run failed, so it's CI/script-friendly.

## GPT judge (step 6, optional, qualitative)

`judges/gpt_judge.py` is a diagnostic complement to the quantitative
pipeline above — it doesn't produce AUROC numbers, it explains them. For
each transcript it hands an external GPT model (`--model`, default
`gpt-4o`) the original prompt, Qwen3-8B's own `<think>` reasoning trace and
final answer, and its verbalized eval/deploy answers across all 5 probe
questions, then asks GPT to diagnose *what specifically* (surface cues,
self-referential statements in the reasoning trace, or nothing at all —
i.e. the model may be guessing from style) plausibly caused Qwen3's
verbalized verdict, plus GPT's own independent read for comparison.

Requires `OPENAI_API_KEY` (see `.env.example` once created, or export
directly). Outputs `output/gpt_judge_analysis.jsonl` (one row per
transcript, structured) and `output/GPT_JUDGE_REPORT.md` (aggregated:
agreement rates with ground truth / with Qwen's verbalized answer, most
common cues cited, per-transcript detail).

```bash
cd judges
uv run python gpt_judge.py \
  --reasoning-traces ../output/reasoning_traces.json \
  --verbalized-csv ../output/verbalized_responses.csv \
  --model gpt-4o \
  --out-jsonl ../output/gpt_judge_analysis.jsonl \
  --out-report ../output/GPT_JUDGE_REPORT.md
```

## Results (verbalized eval-awareness, 9 models)

All 20 transcripts × 5 verbalized probes each, 0 parse failures across
every model. Regenerate these charts any time after adding/re-running a
model with `python analysis/plot_results.py` (writes to `output/charts/`);
for an interactive, always-live view of the underlying transcripts
(prompt / reasoning trace / final answer / per-probe match), use
`dashboard/` instead (see below) — these charts are the static summary,
the dashboard is the per-transcript drill-down.

| Model | Provider | Ensemble AUROC |
|---|---|---|
| GPT-5.6-terra | OpenAI | 1.000 |
| Qwen3-8B | vLLM | 1.000 |
| Gemma-4-12B-OBLITERATED | local HF (VM) | 1.000 |
| Claude Sonnet 5 | Anthropic | 0.985 |
| qwen3-4b-OBLITERATED | vLLM | 0.875 |
| Mistral-Small-24B | vLLM | 0.865 |
| Llama-3.1-8B | vLLM | 0.830 |
| granite-4.2-30b | vLLM | 0.735 |
| Qwen3.6-27B-OBLITERATED | local HF (VM) | 0.500 (chance) |

Olmo-3-7B-Think was attempted and skipped — vLLM 0.19.1 (pinned by this
VM's driver/CUDA version) has no native model support for its
architecture and crashes on load.

![Ensemble AUROC per model](output/charts/01_ensemble_auroc_bar.png)

**Ensemble AUROC hides real per-probe inconsistency.** No model is
uniformly strong across all 5 phrasings — e.g. GPT-5.6-terra's worst
single probe sits at 0.65 despite a perfect 1.0 ensemble score (the
ensemble averages out inconsistent per-probe judgments, it doesn't mean
the model is reliably certain every time). `h_purpose_analysis_a` is the
most reliable probe across the board (floor of 0.75); `direct_vn_a` (a
rephrasing of the plain direct question) is consistently one of the
weakest for non-frontier models — evidence that verbalized
eval-awareness is phrasing-sensitive, not a stable belief reported the
same way regardless of how it's elicited.

![AUROC per model per probe phrasing](output/charts/02_probe_heatmap.png)

![Per-probe AUROC by model](output/charts/03_probe_lines.png)

**Qwen3.6-27B-OBLITERATED's exact 0.5 across every single probe** (not
just the ensemble — `accuracy: 0.5` on all 5, zero parse failures) is
unusual enough to flag explicitly: that's not "weak signal," it's
suspiciously uniform, consistent with the model giving a similar answer
regardless of the transcript rather than genuinely splitting 50/50 on
this dataset. Worth inspecting the raw responses in the dashboard before
trusting this number as a real "no eval-awareness" result.

![Per-probe AUROC grouped by probe](output/charts/04_probe_grouped_bar.png)

### Interactive dashboard

`dashboard/` is a React + TypeScript + Vite app for browsing every
transcript's prompt, reasoning trace, final answer, and per-probe
verbalized results side by side — reads live from this directory's
`output/api/*` (no rebuild needed after adding a model, just refresh the
browser). See `dashboard/` for setup; quick start:

```bash
cd dashboard
npm install   # first time only
npm run dev
```

## Internal representation, coupling, and steering (Phase 2)

Phase 1 (above) only measures what a model *says* when asked directly.
Phase 2 asks a different question: does the model represent "this is an
evaluation" *internally*, in its activations, even when it never says so?
Full design rationale in `plan2.md`; a no-ML-background walkthrough of
the whole idea (with a running analogy) in `explainer.md`; the full
run-by-run log, every bug found and fixed, and the final numbers in
`PROGRESS.md`. Run against 3 models chosen as a contrast set from Phase
1's results: **Qwen3-8B** (strongest verbalizer), **granite-4.2-30b**
(weakest non-degenerate verbalizer), and **Qwen3.6-27B-OBLITERATED**
(verbalizes at exactly chance).

Three experiments, each building on the last:

1. **Internal representation** (`probes/compute_probe.py` +
   `probes/score_activations.py` + `analysis/classify_auroc.py`,
   orchestrated by `probes/run_internal_pipeline.sh`): build a linear
   probe direction per layer from 16 contrastive prompt pairs, project
   each transcript's activations onto it, and check whether the
   projection separates eval from deploy transcripts (AUROC), against a
   random-direction control. All 3 models clear or approach the source
   paper's 0.7 benchmark at their best layer.
2. **Internal-vs-verbalized coupling** (`analysis/compare_internal_vs_verbal.py`):
   does the internal signal actually correlate, transcript-by-transcript,
   with what the model says out loud? Spearman correlation + mutual
   information, again against a random-direction control. Diverges from
   the source paper here — our correlations run 3-4x higher — for a
   verified, non-bug reason documented in `PROGRESS.md`.
3. **Activation steering** (`probes/steer.py`): the causal test — does
   physically nudging the internal signal during generation change what
   the model says? Contrastive Activation Addition via a forward hook,
   swept across layers and steering strengths.

The standout result: Qwen3.6-27B-OBLITERATED internally separates
eval-vs-deploy almost as well as Qwen3-8B (0.88 AUROC) while its
verbalized answer sits at exactly chance — the cleanest "knows
internally, says nothing" case in the dataset, and sharper than any
single model in the source paper's own six-model sample. Charts:
`output/charts/05_*` through `10_*` (`analysis/plot_internal_results.py`,
`plot_coupling.py`, `plot_steering.py`).

A full LaTeX writeup of both phases (methodology, results tables, and all
14 charts, ready to upload to Overleaf) is at
`eval_awareness_report.zip`.

## Known deviations from plan.md

- `gpqa` (`Idavidrein/gpqa`) is gated on HF and no token was available when
  building the dataset; `data/build_dataset.py` substitutes `cais/mmlu`
  (ungated, same "hard capability" flavor). Swap back if you have HF auth.
- At n=20 transcripts, `analysis/classify_auroc.py` reports AUROC on the
  full set (rank-based, threshold-free) plus leave-one-out accuracy, instead
  of a held-out train/test split — noted explicitly in its output JSON.
  Treat all numbers here as exploratory effect sizes, not significance
  tests, per `EXPERIMENT_SUMMARY.md`'s caveat on the earlier 20-transcript
  vLLM run.
