# Qwen3-8B Evaluation-Awareness Experiment

Standalone pipeline (see `plan.md` for full context/rationale) that:

1. Builds a small mixed eval/deploy dataset (`data/build_dataset.py`).
2. Runs Qwen3-8B inference over it via plain HF `transformers`, capturing
   reasoning traces and hidden-state activations (`inference/run_inference.py`).
3. Computes an internal linear-probe direction from 16 contrastive pairs
   and scores the transcripts' activations against it
   (`probes/compute_probe.py`, `probes/score_activations.py`,
   `analysis/classify_auroc.py`).
4. Asks Qwen3-8B to verbalize whether each transcript was an evaluation,
   across 5 probe-question phrasings (`probes/verbalize_probe.py`).
5. Compares internal vs. stated (verbalized) eval-awareness
   (`analysis/compare_internal_vs_verbal.py`).

No `inspect_ai` dependency anywhere — one HF code path for generation,
judging, and hidden-state extraction, matching
`../evaluation-awareness/cls_corr_mi`'s conventions so results stay directly
comparable.

## Layout

```
eval-awareness/
├── plan.md                        # full design rationale
├── run.sh                         # orchestrates steps 0-5 end-to-end
├── data/
│   ├── build_dataset.py           # step 0
│   ├── dataset.json                # 20 transcripts (10 eval, 10 deploy)
│   └── contrastive_pairs_16.txt   # copied from cls_corr_mi/data/
├── inference/
│   ├── model_utils.py             # HF load/chat-template helpers
│   ├── run_inference.py           # local Qwen3-8B: generation + activation capture
│   ├── api_clients.py             # OpenAI/Anthropic provider adapters (reads .env)
│   ├── run_inference_api.py       # single OpenAI/Anthropic model: generation + verbalized probing
│   └── run_benchmark.py           # multi-model runner: logs + aggregated summary
├── probes/
│   ├── compute_probe.py           # step 1: 16-pair direction + random controls
│   ├── score_activations.py       # step 2: project saved activations onto probe
│   ├── probe_questions.yaml       # 5 verbalized-probe phrasings (ported from eval-aware-oss)
│   └── verbalize_probe.py         # step 4: verbalized probing + label parsing
├── analysis/
│   ├── classify_auroc.py          # step 3: internal-probe AUROC (n=20, LOO accuracy)
│   └── compare_internal_vs_verbal.py  # step 5: internal vs. verbalized comparison
├── judges/
│   └── gpt_judge.py               # step 6 (optional): GPT reads Qwen3's <think>
│                                   # trace + verbalized answers, diagnoses cues
└── output/                        # run artifacts (git-ignored; created at runtime)
```

## Requirements

CUDA GPU with ~16GB+ VRAM for Qwen3-8B in bf16 (single GPU; `device_map="auto"`
shards automatically if more are available). Package management uses `uv`
throughout (`uv sync`, `uv run python ...`) — see this workspace's
package-management convention. `torch`/`transformers`/`accelerate`/`numpy`/
`scikit-learn`/`scipy`/`pyyaml`/`datasets` are required; none of this has
been run/tested in this environment (no GPU/torch available here) — verify
on your GPU machine before trusting the numbers.

## Running

```bash
uv sync   # once you have a pyproject.toml/lockfile for this dependency set
./run.sh
```

Or step by step (see `run.sh` for the exact commands/paths), e.g. for a
quick generation-only smoke test before running the full pipeline:

```bash
cd inference
uv run python run_inference.py --limit 2 --no-hidden-states
```

## Running the benchmark on OpenAI / Anthropic models

`inference/run_inference_api.py` is a drop-in alternative to
`run_inference.py` for API models — they expose no hidden states, so it
covers generation + verbalized probing only (no internal linear-probe
steps 1-3, which are Qwen3-8B/HF-only). It reuses
`probes/verbalize_probe.py`'s exact label-parsing logic and
`probes/probe_questions.yaml`, and writes the same `reasoning_traces.json`
schema as `run_inference.py` (so `judges/gpt_judge.py` works unmodified on
either), plus `verbalized_responses.csv` and `results/verbalized_auroc.json`
— all under `output/api/<provider>__<model_slug>/` by default.

Requires `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` (loaded from `.env` via
`python-dotenv` if installed — **never commit `.env`**, it's in
`.gitignore`).

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

### Multi-model runner: `run_benchmark.py`

Runs several OpenAI/Anthropic models in one command (one subprocess per
model, via `run_inference_api.py`, so one model's API error doesn't take
down the rest), tees each model's output to both the console and a
timestamped per-model log file, and writes an aggregated comparison
summary once everything finishes.

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
default for all), `--limit`, `--resume`, `--summary-json`/`--summary-md`
(override output paths).

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
