# Evaluation awareness Experiments

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

Thinking is on by default here, matching the local commands; pass
`--no-thinking` to turn it off. The five follow-up calls always run with
thinking off regardless, since they only need a label and a small token budget.

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

`plots` walks `output/local/<model>/results/` and `output/api/<model>/` looking
for the filenames used above. A chart whose inputs are missing is skipped
without a warning, so if you get fewer charts than expected, check the names
first. The two null-model charts need `nulls_answer_<pooling>.json` and
`nulls_prompt_<pooling>.json` specifically.

## Notes

The random controls in `probe.py` are seeded from a fixed key so results
reproduce across runs. The key is shared by all models, so two models of equal
width get the same control directions; this affects independence of the control
band across models, not its validity for any one model.

`--alphas` accepts a leading negative number in either spelling; argparse would
otherwise read `-1.0,...` as an option.

Both generation paths record what produced each transcript: `generation_config`
and `elapsed_s` everywhere, plus `provider` and `n_generated_tokens` for the API
path. Keep these. Recovering the settings behind an old run is otherwise
guesswork, and the sampling config is the first thing you need when a result
does not reproduce.

The probe direction is per model, built from that model's own activations, and
the same `probe.pt` is reused for every experiment on that model. Experiment 3
is only a causal test of Experiment 1 if it steers along the same vector that
was scored.

`enable_thinking` reaches the server as `chat_template_kwargs`, so it does
nothing for a model whose chat template ignores the kwarg. Check the rendered
template before reading anything into a thinking-on versus thinking-off
comparison.
