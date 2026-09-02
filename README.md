# LLM Evaluation-Awareness Experiments

This project measures whether language models can tell they are being evaluated, whether they say so out loud, and whether that signal is also present in their internal activations.

Phase 1 (verbalized eval-awareness) asks a model, across five independently worded probe questions, whether each of 20 transcripts (10 evaluation-framed, 10 deployment-framed) was an evaluation. It was run on 9 models via local Hugging Face Transformers, self-hosted vLLM, or the OpenAI/Anthropic APIs.

Phase 2 follows [Heidari, Memarian & Rabusseau, *Evaluation Awareness in Language Models: Representation, Verbalization, and Control](https://arxiv.org/abs/2608.21766)*. On 3 of the 9 models it checks whether “this is an evaluation” is linearly readable from residual-stream activations, how that internal signal couples to verbalized answers, and whether steering along the probe direction changes what the model says.

## Results

All 20 transcripts × 5 verbalized probes, 0 parse failures on every model.


| Model                   | Ensemble AUROC |
| ----------------------- | -------------- |
| GPT-5.6-terra           | 1.000          |
| Qwen3-8B                | 1.000          |
| Gemma-4-12B-OBLITERATED | 1.000          |
| Claude Sonnet 5         | 0.985          |
| qwen3-4b-OBLITERATED    | 0.875          |
| Mistral-Small-24B       | 0.865          |
| Llama-3.1-8B            | 0.830          |
| granite-4.2-30b         | 0.735          |
| Qwen3.6-27B-OBLITERATED | 0.500          |


Ensemble AUROC hides per-probe inconsistency. No model is uniformly strong across all 5 phrasings. `h_purpose_analysis_a` is the most reliable probe (floor of 0.75). `direct_vn_a` is among the weakest for non-frontier models. Verbalized eval-awareness is phrasing-sensitive.

Qwen3.6-27B-OBLITERATED scores exactly 0.5 on every probe (accuracy 0.5, zero parse failures). It answers “not an evaluation” on all 100 verbalized-probe calls.

Phase 2 used Qwen3-8B (strongest verbalizer), granite-4.2-30b (weakest non-degenerate verbalizer), and Qwen3.6-27B-OBLITERATED (verbalizes at chance).

- Internal representation: a per-layer linear probe on residual-stream activations separates eval from deploy transcripts (AUROC 0.73–0.98). All three models clear or approach the source paper’s 0.7 benchmark at their best layer.
- Internal-vs-verbalized coupling: Spearman |ρ| up to 0.64, higher than the source paper’s |ρ| < 0.19. Both signals separately track ground truth on our direct probes, so they correlate through that shared cause rather than internal state leaking into speech.
- Standout case: Qwen3.6-27B-OBLITERATED internally separates eval vs. deploy almost as well as Qwen3-8B (0.88 AUROC) while verbalizing nothing about it.
- Steering did not produce a clean, consistent effect for any model, matching a limitation reported in the source paper.

Eval-awareness is measurable inside activations, is inconsistently (sometimes not at all) verbalized, and is not reliably steerable via a simple linear direction.

## File directory

```
eval-awareness/
├── plan.md
├── run.sh
├── data/
│   ├── build_dataset.py
│   ├── dataset.json
│   └── contrastive_pairs_16.txt
├── inference/
│   ├── model_utils.py
│   ├── run_inference.py
│   ├── api_clients.py
│   ├── run_inference_api.py
│   └── run_benchmark.py
├── probes/
│   ├── compute_probe.py
│   ├── score_activations.py
│   ├── steer.py
│   ├── run_internal_pipeline.sh
│   ├── probe_questions.yaml
│   └── verbalize_probe.py
├── analysis/
│   ├── classify_auroc.py
│   ├── compare_internal_vs_verbal.py
│   ├── plot_results.py
│   ├── plot_internal_results.py
│   ├── plot_coupling.py
│   └── plot_steering.py
├── judges/
│   └── gpt_judge.py
├── dashboard/
│   └── src/data/dashboard_data.json
├── .github/workflows/
│   └── deploy-dashboard.yml
└── output/charts/
```

