# Evaluation awareness: representation, verbalization, steering

Does a language model represent the fact that it is being evaluated, does it
say so when asked, and does that internal representation causally drive what it
says? Replication and extension of Heidari, Memarian and Rabusseau,
*Evaluation Awareness in Language Models* (arXiv:2608.21766), on 20 transcripts
(10 evaluation, 10 deployment) in `data/dataset.json`.

Results, corrections and caveats are in `explainer_docs/PROGRESS.md`. Chart by
chart commentary is in `output/charts/CHARTS_EXPLAINER.md`.

## Install

```bash
uv sync                 # or: uv pip install -e ".[api]"
```

## Layout

```
data/            dataset.json, contrastive_pairs_16.txt, probe_questions.yaml, build_dataset.py
evalaware/       the pipeline
output/          generated; gitignored except charts/
```

| module | what it does |
|---|---|
| `models.py` | HF model loading, chat templating |
| `parsing.py` | reasoning/answer split, verbalized label parsing |
| `datasets.py` | dataset, contrastive pairs, probe questions |
| `activations.py` | generation and residual-stream capture |
| `probe.py` | 16-pair probe direction plus random controls |
| `scoring.py` | project activations onto the probe |
| `verbalize.py` | verbalized probing on a local model |
| `steering.py` | contrastive activation addition |
| `metrics.py` | AUROC, band statistic, nulls, coupling |
| `api.py` | OpenAI / Anthropic / vLLM generation and probing |
| `plots.py` | charts |
| `cli.py` | `python -m evalaware.cli <command>` |

## Running the experiments

`$M` is a HF id, `$D` an output directory such as `output/local/qwen3-8b`.

**Phase 1, verbalized probing over an API model.** Generation plus the five
follow-up questions, scored as ensemble AUROC.

```bash
python -m evalaware.cli api --provider vllm --base-url http://localhost:8000/v1 \
    --model "$M" --out-dir "$D"
```

**Experiment 1, internal representation.** Generate with activation capture,
build the probe direction from the 16 contrastive pairs, project, score.

```bash
python -m evalaware.cli generate --model-path "$M" --out-dir "$D"
python -m evalaware.cli probe    --model-path "$M" --out "$D/probe.pt"
python -m evalaware.cli score    --probe "$D/probe.pt" --activations-dir "$D/activations" \
    --out "$D/scores.npz"
python -m evalaware.cli auroc    --scores "$D/scores.npz" --pooling last \
    --out "$D/results/internal_auroc_last.json"
```

**Experiment 1b, prompt-only readout.** The same probe read before the model
generates anything, reusing the stored prompts so the token sequence is
identical.

```bash
python -m evalaware.cli capture-prompt --model-path "$M" \
    --traces "$D/reasoning_traces.json" --out-dir "$D/activations_prompt"
python -m evalaware.cli score --probe "$D/probe.pt" \
    --activations-dir "$D/activations_prompt" --out "$D/scores_prompt.npz"
```

**Experiment 2, coupling.** Internal probe score against verbalized answer, per
layer, with a random-direction band.

```bash
python -m evalaware.cli coupling --scores "$D/scores.npz" \
    --verbalized-csv "$D/verbalized_responses.csv" --pooling last \
    --out "$D/results/coupling_last.json"
```

**Experiment 3, steering.** Add `alpha * m_l * v_l` to the residual stream at
one layer while the model answers the probe question.

```bash
python -m evalaware.cli steer --model-path "$M" --probe "$D/probe.pt" \
    --traces "$D/reasoning_traces.json" --activations-dir "$D/activations" \
    --layers 11,19,23,27 --alphas=-1.0,-0.5,0.5,1.0 --out "$D/results/steering_results.json"
```

**Null models.** A peak taken over 36 to 64 layers at n=20 is inflated by the
selection itself, so this reports a band statistic against a label-permutation
null and fresh random directions.

```bash
python -m evalaware.cli nulls --scores "$D/scores.npz" \
    --activations-dir "$D/activations" --traces "$D/reasoning_traces.json" \
    --pooling last --out "$D/results/nulls_answer_last.json"
```

**Charts.**

```bash
python -m evalaware.cli plots --output-dir output
```

## Notes

The random controls in `probe.py` are seeded from a fixed key so results
reproduce across runs. The key is shared by all models, so two models of equal
width get the same control directions; this affects independence of the control
band across models, not its validity for any one model.

`--alphas` accepts a leading negative number in either spelling; argparse would
otherwise read `-1.0,...` as an option.
