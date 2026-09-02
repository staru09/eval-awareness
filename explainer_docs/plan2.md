# Plan 2: Internal Representation of Evaluation Awareness (3 experiments)

## Context

We've built and run the verbalized-probing half of eval-awareness on 6
models (`output/api/*`), replicating the "external verbalization" facet of
Heidari, Memarian & Rabusseau, *"Evaluation Awareness in Language Models:
Representation, Verbalization, and Control"* (arXiv:2608.21766v1) — read in
full to inform this plan. That paper studies eval-awareness across **four
facets**: (i) internal representation (linear probe on residual stream,
AUROC), (ii) verbalization (LLM-judge 0-3 score), (iii) causal control
(activation steering shifts verbalization), (iv) evolution across training
checkpoints (Olmo3 Base/SFT/DPO/Think).

We've only done (ii) so far, and only at the API/vLLM-generation level (no
hidden-state access). This plan moves into facet (i)+(iii): understanding
what's happening *inside* the models, not just what they say — as **exactly
3 experiments**, each mapped directly to one of the paper's facets, with
explicit reasons for what's in and what's deliberately left out.

**Why 3, and why these 3 specifically:**
- Facet (iv) — evolution across training checkpoints — is **excluded**.
  It requires Base/SFT/DPO/Think checkpoints of the same model family; we
  don't have access to intermediate checkpoints for Qwen3-8B or
  granite-4.2-30b (only final instruct/think releases are public), so this
  facet isn't reproducible with our models. Not a scoping choice, a hard
  constraint.
- Facets (i), (ii)+coupling, and (iii) map cleanly onto **3 experiments**:
  1. Internal representation (linear probe AUROC) — the foundational
     measurement everything else depends on.
  2. Internal-vs-verbalized coupling (Spearman + mutual information) — the
     paper's central surprising finding (weak coupling, |ρ| < 0.19) tested
     against *our own* verbalized data we already collected.
  3. Activation steering — the causal test: does the direction from
     Experiment 1 actually *drive* verbalization, or is it inert?
  This is also the natural build order: 2 needs 1's probe scores, 3 needs
  1's probe direction.

**Models: Qwen3-8B + granite-4.2-30b.** Chosen per the user's explicit
answer: Qwen3-8B as our strongest verbalizer (ensemble AUROC 1.0) and
granite-4.2-30b as the weakest (0.735) — a contrast pair to test whether
"knows internally but doesn't say" or "says but doesn't really know" shows
up differently between them, directly extending the paper's own framing
("a model may internally distinguish evaluation from deployment while
rarely verbalizing that distinction").

**Compute: Azure VM (`azure-btc`, H100 NVL 96GB), new HF `transformers`
venv.** The existing vLLM venv on that VM (`/nvme/venvs/vllm`) doesn't
expose raw hidden states the way we need (`output_hidden_states=True` on a
plain `AutoModelForCausalLM.from_pretrained` call) — vLLM's engine doesn't
return residual-stream activations at all layers per token. There is
already a working recipe for this on the VM: `~/bin/interp-setup.sh` builds
exactly the venv we need (`/nvme/venvs/interp`, torch 2.10 pinned to this
VM's driver, `transformer_lens`, `scikit-learn`, etc.) — reuse it as-is,
don't reinvent.

**Scale realism vs. the paper.** The paper uses 200-1000 prompts per
experiment with 10 reseeds; our dataset has only 20 transcripts (10 eval /
10 deploy), matching our existing `data/dataset.json`. At this scale we
cannot do the paper's held-out calibration/test split — we follow the same
convention already adopted in our own verbalized-AUROC code: leave-one-out
/ full-set AUROC, explicitly flagged as exploratory. This is a known,
already-documented limitation of our existing pipeline (see `README.md`'s
"Known deviations from plan.md"), not a new one introduced here.

## Existing scaffolding to reuse (not rebuild)

- `data/contrastive_pairs_16.txt` — the exact 16 contrastive pairs
  (deployment-framing vs. evaluation-framing), already copied from
  `cls_corr_mi/data/`, matching the paper's Appendix B.2 construction
  method (same idea: hold style/topic fixed, swap only eval/deploy framing).
- `data/dataset.json` — our 20-transcript eval/deploy set, reused
  unchanged as the classification test set for Experiment 1.
- `probes/compute_probe.py`, `probes/score_activations.py`,
  `analysis/classify_auroc.py` — already implement the paper's exact
  method (mean-difference-of-pairs → unit-normalize → per-layer projection
  → AUROC), written for Qwen3-8B only, never run against real weights.
  Needs: (a) generalizing off the Qwen3-8B hardcode to accept
  `--model-path granite-4.2-30b` too, (b) adding the paper's random-direction
  control (Appendix E.1) if not already present — check before writing new
  code.
- `inference/run_inference.py`, `inference/model_utils.py` — local-HF
  generation + hidden-state capture, model-configurable already
  (`--model-path`), reuse directly for granite-4.2-30b too.
- `output/api/vllm__Qwen-Qwen3-8B/verbalized_responses.csv` and
  `.../vllm__ibm-granite-granite-4.2-30b/verbalized_responses.csv` — our
  already-collected verbalized `eval_signal` scores per transcript per
  probe, the exact input Experiment 2 needs. No new generation required
  for that half of Experiment 2.
- `/home/azureuser/bin/interp-setup.sh` (VM) — builds the HF-transformers
  venv we need; run once per VM boot (NVMe wipes on stop/start, same
  gotcha we already hit with the vLLM venv).

## Experiment 1 — Internal representation (linear probe AUROC per layer)

**What:** For each of Qwen3-8B and granite-4.2-30b: build the probe
direction v̂ℓ at every layer from the 16 contrastive pairs (mean difference,
unit-normalized, last-token readout — paper's exact method, §2/Appendix C).
Project our 20 dataset transcripts' last-token residual-stream activations
onto v̂ℓ at every layer. Compute AUROC (eval vs. deploy) per layer, plus a
random-unit-direction control per layer (paper's Appendix E.1) to confirm
the signal isn't a generic property of the activations.

**Why this first:** It's the paper's foundational claim ("linearly
decodable from the residual stream of every model, best AUROC ≥ 0.7") and
the input every other experiment needs (probe direction for steering,
probe score for coupling). Cheapest experiment computationally — one
forward pass per transcript, no generation needed beyond what
`run_inference.py` already captures.

**New/changed files:**
- `probes/compute_probe.py` — generalize `--model-path` (check current
  hardcode), run once per model.
- `probes/score_activations.py` — run against
  `output/<model>/activations/*.npz` from `run_inference.py`.
- `analysis/classify_auroc.py` — add the random-direction control curve if
  not already present; output per-layer AUROC + control band as both a
  JSON table and a plot (matplotlib, matching the paper's Figure 2 style —
  AUROC vs. normalized layer depth).

**Verification:** Peak-layer AUROC for both models should sit noticeably
above 0.5 and above the random-direction band (paper's models all clear
0.7 at their best layer — ours has far fewer transcripts, so treat 0.7 as
a rough target, not a bar). If granite-4.2-30b's peak AUROC is *also* well
above chance despite its weak verbalization (0.735 ensemble), that alone
reproduces the paper's central dissociation finding on our own data.

## Experiment 2 — Internal-vs-verbalized coupling (Spearman + mutual information)

**What:** Pair each transcript's per-layer probe score sℓ(x) (from
Experiment 1) with its already-collected verbalized `eval_signal` (from
`verbalized_responses.csv`) for the same transcript. Compute Spearman rank
correlation and Kraskov k-NN mutual information between the two, per layer,
per probe-question, per model — plus the same random-direction control.

**Why this second:** This is the paper's most important and most
counter-intuitive result — internal representation and verbalization are
almost decoupled (|ρ| < 0.19, MI < 0.04 nats) even though both are
individually above chance. It's cheap to test on our data because *we
already have the verbalized scores* (5 probes × 20 transcripts × 2 models,
zero parse failures, collected earlier this session) — this experiment is
pure analysis, no new model calls.

**New/changed files:**
- `analysis/compare_internal_vs_verbal.py` — already scaffolded per the
  original plan.md but never run; point it at Experiment 1's per-layer
  probe scores + the existing `verbalized_responses.csv` files for both
  models. Output Spearman ρ and MI per layer per probe, plus the
  random-direction baseline, as a JSON table under
  `output/local/<model>/results/coupling.json`.
- `analysis/plot_coupling.py` (new) — mirrors `plot_internal_results.py`'s
  conventions (same palette/style constants, `matplotlib` with `Agg`
  backend, saved to `output/charts/`): per model, |ρ| and MI vs. normalized
  layer depth with the random-direction baseline band (paper's Figure 4
  style), one panel per model side-by-side; plus a peak-coupling summary
  bar comparable to `06_internal_auroc_peak_summary.png`.

**Plotting:** `python analysis/plot_coupling.py` after all
`compare_internal_vs_verbal.py` runs complete →
`output/charts/07_coupling_layers.png`,
`output/charts/08_coupling_peak_summary.png`.

**Verification:** Compare our peak |ρ| / MI against the paper's own
random-direction baseline band — if our real probe's coupling sits inside
that band (as in the paper), that's a genuine replication of the
dissociation finding, worth reporting explicitly as such. If it's instead
clearly above baseline, that's a real difference from the paper worth
flagging (possible small-sample artifact at n=20, or a genuine difference
in our probe-question phrasings vs. their single LLM-judge rubric).

## Experiment 3 — Activation steering (causal test)

**What:** Contrastive Activation Addition (paper's §2/§4): at a chosen
layer ℓ, add α·cℓ to the residual stream at every token position during
generation, where cℓ = mean-activation-norm(ℓ) · v̂ℓ (v̂ℓ from Experiment 1).
Sweep a coarse grid of layers (every 4th, matching the paper's Qwen3-8B
sweep convention) × α ∈ {-1.0, -0.5, 0, 0.5, 1.0} (5 values, coarser than
the paper's 11-point sweep to keep compute bounded). For each (α, ℓ), run
the `direct_a` verbalized probe question (our existing, already-validated
probe from `probe_questions.yaml`) on a fixed subset of ~5-10 transcripts,
and compute the shift in `eval_signal` relative to unsteered (α=0) — our
binary-signal analogue of the paper's ∆J(α, ℓ).

**Why this third and last:** It's the causal test — does the direction
from Experiment 1 actually *move* what the model says, even though
Experiment 2 shows it doesn't naturally correlate with what the model
says? It's also the most expensive experiment (requires forward hooks
during `model.generate()`, a full layer×α grid × multiple transcripts ×
2 models), so it goes last, after the cheaper experiments have already
told us whether a real probe direction exists to steer along at all.

**Simplification vs. the paper, and why:** The paper scores steered
generations with a separate 0-3 GPT-5.4 LLM-judge call per generation
(§2, Appendix C.1) — expensive at their scale (thousands of judge calls).
We instead reuse our own `direct_a` verbalized-probe scoring
(`resolve_answer()` in `probes/verbalize_probe.py`, already built and
validated at 0 parse failures across our whole run) as a cheaper proxy
metric. This trades the paper's finer 0-3 granularity for a binary
eval/deploy signal, but avoids adding a new judge-call dependency and
directly reuses code we've already verified works. Flag this explicitly
in the eventual results write-up as a deliberate metric substitution, not
an oversight.

**New files:**
- `probes/steer.py` (new) — registers a forward hook on the target layer's
  output during `model.generate()`, adds α·cℓ, runs the `direct_a` probe
  question, records `eval_signal` per (α, ℓ, transcript). Reuses
  `model_utils.load_hf_model`/`apply_chat_template` and
  `verbalize_probe.resolve_answer` — no new model-loading or parsing code.
- Output: `output/local/<model>/results/steering_results.json` (per α, ℓ:
  mean `eval_signal` shift vs. baseline).
- `analysis/plot_steering.py` (new) — same palette/style/`Agg` convention
  as the other two plotting scripts, saved to `output/charts/`: ∆-score
  vs. normalized layer depth, one line per α (paper's Figure 5 style), one
  panel per model; plus a summary chart of peak ∆-score per model (which
  direction of α moved verbalization more, and by how much).

**Plotting:** `python analysis/plot_steering.py` after all
`probes/steer.py` runs complete →
`output/charts/09_steering_effect.png`,
`output/charts/10_steering_peak_summary.png`.

**Verification:** For Qwen3-8B, expect steering toward the eval-like side
of v̂ℓ (negative α, per the paper's sign convention: lower score = more
eval-like) to *increase* verbalized eval_signal at some layer/magnitude —
the paper reports Qwen3-8B's steering effect as its strongest of all six
models tested (peak ∆J ≈ 1.5 on their 0-3 scale). If our coarser sweep
reproduces a directionally similar effect (negative α increases
eval_signal, concentrated in a narrow layer band), that's a genuine
within-session replication of the paper's most striking causal result, on
a model we've fully instrumented ourselves.

## Explicitly out of scope for these 3 experiments

- Training-stage evolution (facet iv) — no checkpoint access, as above.
- Mean-token readout (paper's Nemotron3-49B-specific fallback) — not
  needed unless Experiment 1's last-token readout looks anomalous for one
  of our two models the way it did for Nemotron3-49B in the paper.
- A second/fine-grained (0-100) judge rubric (paper's Appendix D
  robustness check) — out of scope; our proxy metric in Experiment 3 is
  already a simplification, adding a second judge scale multiplies cost
  without changing what we're trying to learn in this first pass.
- Multi-layer / multi-direction steering (paper's own listed future work,
  §5) — single-layer-at-a-time steering only, matching the paper's actual
  Experiment 4 (not its "future work" wishlist).

## Verification (end-to-end)

1. Experiment 1: `probe.pt`-equivalent artifacts contain a 16-pair real
   direction + random controls per layer, for both models; per-layer AUROC
   table/plot generated; peak AUROC clears chance for both models.
2. Experiment 2: coupling table/plot generated for both models, real-probe
   curve compared against random-direction baseline band; result (falls
   inside vs. above the band) explicitly stated per model.
3. Experiment 3: steering-results JSON + plot generated for at least
   Qwen3-8B (granite-4.2-30b if compute allows); directional check against
   the paper's sign convention (negative α ⇒ higher eval_signal at some
   layer) explicitly reported as replicated or not.
