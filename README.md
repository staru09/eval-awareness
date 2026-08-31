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
│   └── run_inference.py           # generation + activation capture
├── probes/
│   ├── compute_probe.py           # step 1: 16-pair direction + random controls
│   ├── score_activations.py       # step 2: project saved activations onto probe
│   ├── probe_questions.yaml       # 5 verbalized-probe phrasings (ported from eval-aware-oss)
│   └── verbalize_probe.py         # step 4: verbalized probing + label parsing
├── analysis/
│   ├── classify_auroc.py          # step 3: internal-probe AUROC (n=20, LOO accuracy)
│   └── compare_internal_vs_verbal.py  # step 5: internal vs. verbalized comparison
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
