# Eval-Awareness Experiment — Progress Log

Tracks what's been built and run so far. See `plan.md` for the original
verbalized-probing design doc, and `plan2.md` for the internal/
mechanistic-probing phase (Experiments 1-3), which replicates methodology
from Heidari, Memarian & Rabusseau, *"Evaluation Awareness in Language
Models: Representation, Verbalization, and Control"* (arXiv:2608.21766).
`explainer.md` has a plain-language walkthrough of the whole project.

## TL;DR — where things stand

- **Verbalized probing** (§4): 9 models run end-to-end (0 parse failures
  on every one). Ensemble AUROC ranges from 1.000 (3 models, including
  Qwen3.6-27B-OBLITERATED) down to 0.830 (Llama-3.1-8B) — **no model
  scores near or below chance any more**; see §10 for a second, much
  bigger correction (a verbalized-answer parsing bug) that overturned the
  original "Qwen3.6-27B-OBLITERATED knows internally but never says so"
  headline finding entirely. Olmo-3-7B-Think skipped (blocked, see §8).
  **Three of these nine numbers (qwen3-4b-OBLITERATED, Gemma-4-12B-
  OBLITERATED, Qwen3.6-27B-OBLITERATED) were corrected mid-session** after
  a real-data audit found and fixed 3 separate reasoning/answer-split
  bugs — see §9 for the full writeup before trusting any pre-§9 numbers
  cited elsewhere in old notes.
- **Internal representation, Experiment 1** (§6): all 3 tested models
  (Qwen3-8B, granite-4.2-30b, Qwen3.6-27B-OBLITERATED) show real,
  above-random-baseline linear separability of eval-vs-deploy in their
  activations — replicates the paper's core claim.
- **Internal-vs-verbalized coupling, Experiment 2** (§6): diverges from
  the paper (much higher correlation than their <0.19) for a verified,
  non-bug reason — our probes are higher-accuracy classifiers of ground
  truth than their judge, so they mechanically correlate more with
  anything else (like the internal probe) that also tracks ground truth.
- **Activation steering, Experiment 3** (§7): no clean bidirectional
  effect for any model — matches a limitation the paper itself reports,
  not a failure unique to us. Two real implementation bugs were caught
  and fixed before trusting any of these numbers (see §7).
- **Not done**: GPT judge never run; Olmo-3-7B-Think skipped; vLLM server
  on the VM currently stopped (§8).
- Dashboard (`dashboard/`, §1) and all analysis/plotting scripts are
  committed. This file (`PROGRESS.md`) is intentionally left uncommitted
  per explicit instruction — treat it as a local working log, not part of
  the repo history.

## 1. Pipeline built

- **Dataset** (`data/dataset.json`): 20 transcripts, 10 eval-labeled
  (hellaswag, arc_challenge, mmlu as a gpqa substitute, agentharm-style),
  10 deploy-labeled (everyday_chat, wildchat).
- **Local HF path** (`inference/run_inference.py`): any HF `transformers`
  model (not just Qwen3-8B — `--model-path` is fully configurable),
  captures reasoning traces + hidden-state activations for the internal
  linear-probe pipeline (`probes/`, `analysis/`). Run successfully against
  real weights for 5 models total across the session: Qwen3-8B,
  granite-4.2-30b, and Qwen3.6-27B-OBLITERATED (with hidden states, for
  the Experiment 1-3 internals phase, §6-§7), plus Gemma-4-12B-OBLITERATED
  and (generation-only, no hidden states) Qwen3.6-27B-OBLITERATED's
  original verbalized-probing pass (§4) — used as the vLLM fallback when
  a model's architecture wasn't supported by this VM's pinned vLLM
  version (see §6/§7 bugs).
- **API/self-hosted path** (`inference/api_clients.py`,
  `inference/run_inference_api.py`, `inference/run_benchmark.py`): unified
  generation + verbalized-probing across OpenAI, Anthropic, and any
  vLLM-served model (self-hosted, OpenAI-compatible endpoint). Supports
  concurrent transcript processing (`--max-parallel`), per-provider
  reasoning-trace capture, and writes `reasoning_traces.json` /
  `verbalized_responses.csv` / `results/verbalized_auroc.json` per model.
- **GPT judge** (`judges/gpt_judge.py`): qualitative pass over a model's
  own reasoning trace + verbalized answers, diagnosing what likely drove
  eval/deploy judgments. Not yet run.
- **Dashboard** (`dashboard/`): React + TypeScript + Vite app that reads
  `output/api/*/{reasoning_traces.json,verbalized_responses.csv,results/verbalized_auroc.json}`
  live via a dev-server middleware (`vite.config.ts`) — always reflects
  what's on disk, no rebuild needed per model. Layout (redesigned mid-session
  from an earlier tabs-based version): model **dropdown** in the header
  (with ensemble AUROC) → transcript list (eval/deploy chips) on the left
  → prompt/collapsible-reasoning/answer in the center → all 5
  verbalized-probe questions and answers **always visible** in a
  right-hand panel (with markdown rendering), so probes can be compared
  side-by-side without clicking each one open. A one-off static-snapshot
  version was also published as a Claude artifact earlier in the session
  (not kept in sync with the live dashboard).

## 2. Self-hosted vLLM infrastructure (Azure VM)

- VM `azure-btc` (H100 NVL, 96GB VRAM), reached via SSH (`~/.ssh/config`
  alias). NVMe scratch disk (`/nvme`, wiped on every VM stop/start) hosts
  the HF cache and the vLLM Python venv — both need re-provisioning after a
  VM restart via `~/bin/nvme-setup.sh` and `~/bin/vllm-setup.sh`.
- vLLM pinned to **0.19.1** (driver 535 caps CUDA at 12.9; vLLM ≥0.20 needs
  torch 2.11+/cu130, which this driver can't run).
- Local machine reaches the VM's vLLM server through an SSH tunnel
  (`ssh -f -N -L 8000:localhost:8000 azure-btc`, with keepalive flags —
  the tunnel has dropped a few times mid-run and needed re-establishing).
- `HF_TOKEN` set on the VM (`~/.config/storage-env.sh`) for gated model
  downloads (Llama-3.1).

## 3. Bugs found and fixed along the way

- **vLLM reasoning-parser mismatch is silent, not a crash.** Starting the
  server with the wrong `--reasoning-parser` for a given model doesn't
  error — it just misroutes text between the `content` and `reasoning`
  response fields. Two real instances:
  - `deepseek_r1` parser against Qwen3 → every verbalized-probe answer
    landed entirely in `reasoning` with `content: null`, producing 100%
    parse failures across all 100 probe calls before this was caught.
  - `granite` parser (built for older text-marker-based Granite 3.x
    reasoning) against granite-4.2-30b, which actually uses `<think>` tags
    like Qwen3 → reasoning never got split out, landed inline in `content`
    instead. Fixed by using `--reasoning-parser qwen3` for granite-4.2 too.
  - Lesson: always smoke-test with one curl call (thinking on *and* off)
    before trusting a full run — see `README.md`'s vLLM section.
- **`chat_template_kwargs` isn't universally safe to send.** Some served
  tokenizers (Mistral's tekken/mistral-common tokenizer has no Jinja chat
  template at all) reject *any* `chat_template_kwargs`, even a no-op
  `{"enable_thinking": false}`, with a 400. Fixed `generate_vllm` in
  `api_clients.py` to only attach `extra_body` when actually toggling
  thinking on.
- **`--max-model-len` needs capping for large models.** granite-4.2-30b's
  default context (131072) needed more KV cache than fit after loading the
  30B weights; capped to `--max-model-len 32768` for all the larger models.
- **Dashboard: raw markdown wasn't rendered.** Model output routinely
  contains literal `**bold**`/`` `code` `` markdown; added a small inline
  renderer (`dashboard/src/markdown.tsx`) instead of showing raw asterisks.

## 4. Models run — final results

All 20 transcripts × 5 verbalized probes each, 0 parse failures across
every model.

| Model | Provider | Ensemble AUROC | Notes |
|---|---|---|---|
| Qwen3.6-27B-OBLITERATED | HF `transformers` (VM, local path) | **1.000** (was 0.450 — see §10, a much bigger correction than §9's) | vLLM couldn't load it (see §6 bugs); **was wrongly described as "always denies"** — that was a grading-script bug, not model behavior; see §10 |
| GPT-5.6-terra | OpenAI | 1.000 | Responses API, reasoning effort=medium |
| Qwen3-8B | vLLM (self-hosted) | 1.000 | `--reasoning-parser qwen3` |
| Claude Sonnet 5 | Anthropic | 0.985 | extended thinking, most *consistent* per-probe |
| Gemma-4-12B-OBLITERATED | HF `transformers` (VM, local path) | 0.985 (was 1.000 → 0.995 → this, corrected twice — see §9, §10) | vLLM couldn't load it (see §6 bugs); ran via `run_inference.py`/`verbalize_probe.py` instead |
| granite-4.2-30b | vLLM (self-hosted) | **0.950** (was 0.735 — see §10) | `--reasoning-parser qwen3` (not `granite`, see bugs above) |
| qwen3-4b-structured-output-merged-stage-a-OBLITERATED | vLLM (self-hosted) | 0.885 (was 0.875, corrected — see §9; unaffected by §10) | `--reasoning-parser qwen3` |
| Mistral-Small-24B-Instruct-2501 | vLLM (self-hosted) | 0.865 | no native thinking mode — ran `--no-thinking` |
| Llama-3.1-8B-Instruct | vLLM (self-hosted) | 0.830 | gated (HF_TOKEN required), no thinking mode |
| Olmo-3-7B-Think | — | **skipped** | vLLM 0.19.1 has no native `Olmo3ForCausalLM` support (`KeyError: 'rope_theta'` on load); would need a newer vLLM than this VM's driver allows |

Key qualitative takeaways (see chat log / `dashboard/` for detail; **updated
after §10's correction** — every bullet below reflects the corrected
numbers, not the originally-recorded ones):
- Ensemble AUROC hides real per-probe inconsistency — GPT-5.6-terra's
  worst single probe is only 0.65 despite a perfect ensemble score.
- Verbalized eval-awareness is high across almost every model tested —
  the lowest score (Llama-3.1-8B, 0.830) still clears 0.8. Model scale
  doesn't cleanly predict it either way: granite-4.2-30b (30B, 0.950)
  beats both 8B models tested, but several smaller models still beat it.
- `h_purpose_analysis_a` is among the most reliable probes across every
  model; `direct_vn_a` (a rephrasing of the plain direct question) is
  consistently one of the weakest for non-frontier models — evidence
  that verbalized eval-awareness, where it's below ceiling, is
  phrasing-sensitive, not a stable belief reported the same way
  regardless of how it's elicited.

## 5. The Azure VM's ephemeral disk keeps resetting

Worth calling out explicitly since it happened repeatedly across the
session: `/nvme` is wiped on every VM stop/start, taking the vLLM venv
and the HF-transformers ("interp") venv with it. Standard recovery each
time a new session starts work on the VM: `~/bin/nvme-setup.sh` (remount)
→ `~/bin/vllm-setup.sh` and/or `~/bin/interp-setup.sh` (rebuild the
needed venv, a few minutes each) before anything else will run.

## 6. Internal-representation phase (plan2.md, Experiments 1-2)

Moves past "what does the model say" (§4 above) into "what's happening
inside the model" — reusing the paper's exact contrastive-pair linear
probe methodology. Scoped to 3 models as a deliberate contrast set:
**Qwen3-8B** (strongest verbalizer), **granite-4.2-30b** (weakest
non-degenerate verbalizer), **Qwen3.6-27B-OBLITERATED** (verbalizes at
exactly chance). Run via HF `transformers` on the VM (`inference/
run_inference.py` + `probes/compute_probe.py` + `probes/
score_activations.py` + `analysis/classify_auroc.py`, orchestrated by
`probes/run_internal_pipeline.sh`), since vLLM doesn't expose raw
per-layer hidden states the way this needs.

### Experiment 1 — internal linear-probe AUROC per layer

16 contrastive prompt pairs (deployment-framing vs. evaluation-framing,
same topic/length) → mean-difference direction per layer, unit-normalized
→ project each of the 20 dataset transcripts' activations onto it →
AUROC (eval vs. deploy) per layer, with a random-direction control band.

| Model | Peak layer (last-token) | AUROC | LOO accuracy | Verbalized AUROC (§4) |
|---|---|---|---|---|
| Qwen3-8B | 19/36 | **0.980** | 0.900 | 1.000 |
| Qwen3.6-27B-OBLITERATED | 24/65 | **0.880** | 0.800 | 0.500 (chance) |
| granite-4.2-30b | 13/64 | 0.730 | 0.450 | 0.735 |

granite's last-token LOO accuracy (0.450) sits *below chance* despite its
AUROC clearing 0.7 — a real small-sample calibration artifact (rank-based
AUROC vs. a per-fold-calibrated threshold don't have to agree at n=20).
Its mean-token readout is actually cleaner and peaks much earlier (layer
3, AUROC 0.770, accuracy 0.650) — mirrors the paper's own Nemotron3-49B
special case, where they had to switch readout conventions for one
model. Charts: `output/charts/11_prompt_vs_answer_layers_{last,mean}.png`
and `12_prompt_vs_answer_peaks.png` (`analysis/plot_prompt_vs_answer.py`).
The former `05_*`/`06_*` charts were deleted: they showed these peaks
against only the per-layer random band, which overstates them (see §11).

**Headline finding:** Qwen3.6-27B-OBLITERATED internally separates
eval-vs-deploy almost as well as Qwen3-8B (0.88 AUROC) while verbalizing
*nothing* about it — the cleanest "knows internally, says nothing" case
in the whole dataset, sharper than anything in the paper's own six
models.

### Experiment 2 — internal-vs-verbalized coupling

Spearman correlation + Kraskov mutual information between each
transcript's per-layer internal probe score and its verbalized
`eval_signal`, per probe question and as a 5-probe ensemble, plus the
same random-direction control (`analysis/compare_internal_vs_verbal.py`,
rewritten from an earlier scaffold that didn't match this methodology).

| Model | Peak \|Spearman ρ\| (ensemble, last-token) | Peak MI (nats) |
|---|---|---|
| Qwen3.6-27B-OBLITERATED | **0.837** @ L34/65 | 0.474 |
| Qwen3-8B | **0.639** @ L29/36 | 0.479 |
| granite-4.2-30b | **0.620** @ L32/64 | 0.379 |

*(These numbers reflect §10's parsing-bug fix, applied after this
section was first written — see §10 for what changed and why. The table
originally showed Qwen3.6-27B-OBLITERATED as "undefined/degenerate";
that was itself downstream of the same bug and no longer holds.)*

**Real divergence from the paper, root cause verified (not a bug):** the
paper's peak \|ρ\| stays under 0.19 across all six of their models (near
their random-direction baseline) — their central claim is that
representation and verbalization are almost decoupled. Ours is 3-4.5x
higher across **all three** of our models. Checked this directly rather
than just speculating: for Qwen3-8B's `direct_a` probe, `corr(internal,
verbal) = 0.832` is **numerically identical** to `corr(internal, label) =
0.832`, because `corr(verbal, label) = 1.0` — that probe has perfect
accuracy for this model, so `verbal ≈ label` exactly, and correlating
internal score against verbal is mathematically the same computation as
correlating it against ground truth. Checked across the other probes
too: coupling strength tracks probe accuracy directly (accuracy
0.60/0.80/0.85/1.00 → \|ρ\| 0.347/0.602/0.664/0.832). The mechanism: both
`internal_score` and `verbal_signal` are separately trying to predict the
same `ground_truth_label`; if both are decent classifiers of it, they
necessarily correlate with *each other* through that shared cause — no
causal "internal signal leaking into speech" required. Our probes are
direct yes/no questions the model mostly answers correctly (so
`verbal_signal` tracks truth well); the paper's judge scores free-form
generations for *incidental* signs of awareness and is right almost all
the time near-zero regardless of ground truth (>93.9% score 0, evals
included) — so their `J` barely tracks truth at all, meaning nothing can
correlate strongly with it, internal probe included. Higher-accuracy
verbalization mechanically produces higher apparent "coupling" with
anything else that also tracks truth well; this is a genuine design
difference between our probes and their judge, not an error in either.
This mechanism now applies uniformly to all three Phase 2 models, not
two of three — see §10 for why Qwen3.6-27B-OBLITERATED's row changed so
much (it was never actually "degenerate"; a parsing bug made it look
that way).
Charts: `output/charts/07_coupling_layers_{last,mean}_{rho,mi}.png`,
`08_coupling_peak_summary.png`.

### Bugs found running Experiments 1-2

- **Two more vLLM registry gaps, same failure category as Olmo-3.**
  `Gemma-4-12B-OBLITERATED` (`Gemma4UnifiedForConditionalGeneration`) and
  `Qwen3.6-27B-OBLITERATED` (`Qwen3_5ForCausalLM`, the text-only variant)
  both fail to load on this VM's vLLM 0.19.1 — the first isn't registered
  at all, the second exists only for a multimodal variant and misroutes
  into a crash. Both fell back to plain HF `transformers` on the VM
  (`interp` venv) instead — confirmed to load and run fine there.
- **"Unified"/multimodal configs nest text-decoder fields.**
  `run_inference.py` read `model.config.num_hidden_layers` directly, which
  doesn't exist on `Gemma4UnifiedConfig` (`AttributeError`) — it's nested
  under `model.config.text_config`. Fixed by using
  `model.config.get_text_config()`, which handles both plain and nested
  configs.
- **The Mistral `chat_template_kwargs` fix regressed Qwen3-family models.**
  An earlier fix (unconditionally omit `chat_template_kwargs` when
  `enable_thinking=False`, to work around Mistral's tokenizer rejecting the
  field outright) broke Qwen3-family reasoning models the other way: their
  own chat template defaults `enable_thinking` to `True` when the field is
  absent, so a disable-thinking follow-up call would silently keep
  thinking on and exhaust its token budget before answering — confirmed on
  a real run (100% parse failures for `qwen3-4b-...-OBLITERATED`'s
  verbalized probes). Fixed properly: always attempt to send the field,
  only drop it and retry if the server explicitly rejects it by name.
- `classify_auroc.py` and `score_activations.py` both hardcoded
  `"model_key": "qwen3_8b"` in their output metadata regardless of which
  model was actually run — harmless while only one model existed, actively
  mislabeling once granite/Qwen3.6 results existed alongside it. Fixed to
  read/pass the real model path.

## 7. Activation steering phase (plan2.md, Experiment 3)

The causal test: does the Experiment-1 probe direction actually *move*
what the model says, not just correlate with it? Contrastive Activation
Addition (`probes/steer.py`, new) — at a chosen decoder layer, add
`alpha * mean_activation_norm(layer) * v_hat` to the residual stream via
a forward hook, at every token position, while the model answers the
`direct_a` verbalized probe question. Layers swept per model, centered on
each model's Experiment-1 peak layer: Qwen3-8B {11,19,23,27} (peak 19),
granite-4.2-30b {5,9,13,17} (peak 13), Qwen3.6-27B-OBLITERATED
{16,20,24,28} (peak 24). `alpha ∈ {-1.0,-0.5,0.5,1.0}` plus an unsteered
`alpha=0` baseline, on a fixed 6-transcript subset (3 eval, 3 deploy) per
model. Charts: `output/charts/09_steering_effect.png`,
`10_steering_peak_summary.png` (`analysis/plot_steering.py`).

### Two real bugs caught and fixed before trusting any numbers

- **Array-size off-by-one.** `probe.pt`'s direction dict has keys
  `0..num_layers-1` (key 0 = embeddings), but the saved
  `activations/*.npz` files only cover the `num_layers-1` decoder-layer
  outputs (embedding layer dropped before saving, same convention
  `score_activations.py` already used) — `mean_activation_norms()`
  crashed with a numpy broadcast `ValueError` (37 vs 36) on the very
  first real run because it was sized off the wrong count. The `--layers`
  CLI validation had the identical off-by-one (accepted an out-of-range
  key that would later `KeyError`). Both fixed.
- **argparse silently misparses comma-separated negative numbers.**
  `--alphas -1.0,-0.5,0.5,1.0` (space-separated form) failed with
  "expected one argument" — argparse's negative-number heuristic only
  recognizes a *pure* negative number (`^-\d*\.\d+$`), not a
  comma-separated list starting with one, so it treated `-1.0,...` as an
  unrecognized flag. Fixed by using `--alphas=...` (`=`-joined) instead.

### Results

Baseline (`alpha=0`) `direct_a` mean `eval_signal` on the 6-transcript
subset: Qwen3-8B 0.500 (perfect: all 3 eval-labeled scored 1.0, all 3
deploy-labeled scored 0.0 — matches its known 1.0 full-dataset accuracy),
granite-4.2-30b 0.667, Qwen3.6-27B-OBLITERATED 0.667 (see caveat below).

**No clean, symmetric bidirectional effect for any model** — matches a
real limitation the paper itself reports, not a failure unique to us:
*"positive values of α produce smaller ∆J(α,ℓ) values, and do not
consistently reduce the judge score below baseline."* Verified our own
sign convention is correctly implemented (probe direction construction
matches the paper's `v = mean(deploy) - mean(eval)` exactly, confirmed
against `data/contrastive_pairs_16.txt`; direction is properly
unit-normalized; activation magnitudes are sane, e.g. ~55 at Qwen3-8B
layer 11) before accepting the irregular pattern as real rather than a
bug:

- **Qwen3-8B**: `alpha=-1.0` broke down almost completely (0-1 of 6
  transcripts parse successfully, rest unparseable/garbled — the
  perturbation at full strength is roughly the same magnitude as the
  activation itself). At moderate strength, **both** signs of alpha
  tended to *increase* mean `eval_signal` above the 0.5 baseline (e.g.
  layer 23: alpha=-0.5/+0.5/+1.0 all delta=+0.5), not the clean
  negative-increases/positive-decreases split the naive reading of the
  paper's convention would predict.
- **granite-4.2-30b**: the most irregular of the three, no full
  breakdown at any tested cell (all `n_scored=6`) but effects don't track
  alpha's sign consistently across layers — layers 9 and 17 both showed
  their *strongest* effect (delta=-0.5) at `alpha=-1.0`, the "eval-ward"
  direction, moving eval_signal *down*, opposite of the paper's
  prediction for that sign.
- **Qwen3.6-27B-OBLITERATED**: the cleanest bidirectional pattern of the
  three — most negative-alpha cells pushed `eval_signal` up to 1.0
  (delta=+0.333, saturating: `alpha=-0.5` and `alpha=-1.0` landed on the
  *identical* value at 3 of 4 layers, consistent with a ceiling effect),
  and 2 of 4 layers showed positive-alpha suppressing it (`L28
  alpha=+0.5`: delta=-0.333). `alpha=±1.0` at layer 16 broke down
  completely (0/6 scored) for both signs.

**Caveat on the Qwen3.6-27B-OBLITERATED baseline, found and disclosed
rather than glossed over:** this steering baseline (0.667, real variance)
does **not** match this model's corrected full-dataset verbalized
ensemble AUROC of 1.000 (§10) either — though both now show real,
non-degenerate variance, unlike the near-total constancy an earlier pass
of this log described. Root cause: a
genuine inconsistency between two scripts, not a bug in either — the
original verbalized run (`probes/verbalize_probe.py`) has thinking
**enabled** on its follow-up probe question (`enable_thinking=True` for
the one `ModelSpec` it uses throughout), while `steer.py` explicitly sets
`enable_thinking=False` on its follow-up call to match the vLLM/API
pipeline's convention. This means the two "baselines" for this model are
not the same experimental condition and shouldn't be compared directly —
itself a small additional finding about how sensitive this model's
verbalized eval-awareness is to whether thinking is enabled on the
follow-up question.

**Also unverified: §10's parsing bug.** `steer.py` scores its follow-up
probe with the same `resolve_answer()` function responsible for §10's
bug, so these steering numbers could in principle be affected too. We
could not re-check them the way we did for Phase 1/Experiment 2, because
`steer.py`'s saved output only has the parsed `eval_signal` per cell, not
the raw response text needed to re-derive it. One mitigating factor:
§10's bug requires a model to restate the prompt template or draft an
answer during *extended thinking* before its real final answer, and
`steer.py` disables thinking on this call (the same inconsistency noted
above) — which removes the specific behavior that caused the bug
elsewhere. Left as an open, disclosed limitation, not assumed clean.

## 8. Not yet done

- GPT judge (`judges/gpt_judge.py`) — written, not yet run against any
  model's transcripts.
- Olmo-3-7B-Think — blocked on vLLM version; would need either a newer
  vLLM (blocked by the VM's driver) or the same HF-`transformers` fallback
  used for the two OBLITERATUS models above (not yet attempted for Olmo-3
  specifically).
- vLLM server on the VM is currently **stopped** (GPU freed after the
  run); restart with the commands in `README.md`'s vLLM section if needed
  again. Remember §5 — the venv will likely need rebuilding first.

## 9. Discrepancy audit: reasoning/answer-split bugs found and fixed

Triggered by a dashboard bug report: for the 3 OBLITERATED models, the
dashboard's "Reasoning" and "Answer" panels were sometimes swapped,
sometimes both showed the same blob, and sometimes one was empty.
Investigating properly (not just patching the dashboard) turned up **3
distinct root-cause bugs**, one per model, each different:

1. **Qwen3.6-27B-OBLITERATED (local HF).** Its chat template pre-seeds
   the assistant turn with an opening `<think>` tag as part of the
   *prompt*, not the generated text — only the closing `</think>` ever
   appears in what gets saved. The old `split_reasoning()` in
   `inference/run_inference.py` required both tags and matched nothing
   on all 20/20 transcripts, dumping the model's entire thinking process
   into `final_answer`. **Fix:** added a closing-tag-only fallback
   pattern (`THINK_CLOSE_ONLY_RE`); re-split all 20 transcripts from the
   already-saved raw text (no regeneration needed).
2. **Gemma-4-12B-OBLITERATED (local HF).** Its "thought" turn boundary
   (`<|channel>thought\n...<channel|>...<turn|>`, confirmed by fetching
   its `tokenizer_config.json` from HF and reading the model's own
   `response_schema.x-regex`) uses **registered special tokens**, unlike
   Qwen3's plain-text `<think>` tags — `decode(skip_special_tokens=True)`
   silently stripped them with zero trace before any text was saved.
   Unlike case 1, this was **not recoverable** from the already-saved
   JSON at all (confirmed: no delimiter of any kind survives in any of
   the 20 saved transcripts). **Fix:** token-level split in
   `run_inference.py` — locate the `<channel|>` token id directly in
   `gen_ids` before decoding throws it away, split there, decode each
   side separately. Required a **full regeneration** (fresh
   `run_inference.py` run, 20/20 transcripts + hidden states).
3. **qwen3-4b-structured-output-merged-stage-a-OBLITERATED (vLLM).** The
   server's `--reasoning-parser qwen3` never saw a closing `</think>`
   from this model (it doesn't emit one), so it bucketed **100%** of the
   response into the `reasoning` field with `content` empty on all
   20/20 transcripts — confirmed by inspection that "reasoning" held the
   real response (direct replies or step-by-step analysis ending in a
   clear answer, not deliberation fragments). This is more than a
   display bug: `run_inference_api.py` builds each follow-up probe
   question's prior-turn context from this same empty `content` field,
   so all 100 verbalized-probe calls for this model ran with **blank
   context**. **Fix:** `api_clients.py`'s `generate_vllm()` now falls
   back to the reasoning text as the real answer when content comes
   back empty, with a log warning. Required a **full rerun** (fresh
   vLLM server + `run_inference_api.py`, capped `--gpu-memory-utilization
   0.35 --max-model-len 8192` to coexist with a concurrent 27B model on
   the same GPU — this model's 262K native context otherwise demands
   36GB of KV cache alone).

**Also found while auditing all 9 Phase-1 models for the same pattern:**
Claude Sonnet 5's `reasoning_trace` is `null` on all 20 transcripts —
verified this is *not* a capture bug by calling `generate_anthropic`
directly on a prompt engineered to require deliberation and confirming
it correctly returns real thinking content when the model actually uses
it; null across all 20 dataset prompts means adaptive-effort genuinely
judged none of them worth deliberating over. granite-4.2-30b has one
transcript (`arc_challenge_2`) with an empty `final_answer`: the model
got stuck in a visible, indecisive reasoning loop ("Wait, is that
true?... Then what?... Perhaps...") and burned its entire 2048-token
budget without ever closing its thinking — genuine model behavior, not a
pipeline bug, left undisturbed.

**Also fixed while in this code:** `verbalize_probe.py` ran all 5 probe
questions per transcript fully sequentially (one `model.generate()` call
at a time) despite confirmed-idle GPU headroom (58% util, 40GB+ free on
a 96GB H100 mid-run). Batched all 5 probes per transcript into one
`generate()` call (safe because `load_hf_model()` already left-pads, so
every row in a batch shares the same input length) — cut Qwen3.6-27B's
100-call probing pass from an estimated ~90 min to ~21 min in practice.

**Reruns performed and corrected numbers** (all via the Azure VM,
`azure-btc`, after the user started it back up):

| Model | Metric | Before | After |
|---|---|---|---|
| qwen3-4b-...-OBLITERATED | Ensemble AUROC | 0.875 | **0.885** |
| Qwen3.6-27B-OBLITERATED | Ensemble AUROC | 0.500 (chance) | **0.450** (below chance) |
| Qwen3.6-27B-OBLITERATED | Constant-denial rate | "100/100" (context was broken) | **99/100** (verified with clean context) |
| Qwen3.6-27B-OBLITERATED | Exp.2 ensemble peak coupling | undefined | 0.378 @ L22 — **exactly at the random-direction ceiling**, i.e. still not a real signal |
| Gemma-4-12B-OBLITERATED | Ensemble AUROC | 1.000 | **0.995** |

The headline finding survives, strengthened rather than undermined: even
with corrected, uncorrupted context, Qwen3.6-27B-OBLITERATED still
denies being evaluated on effectively every call (99/100, not the
"100/100" originally measured under broken context) — confirming this is
a genuine property of the model, not an artifact of either bug.

**Downstream updates made from the corrected data:** dashboard rebuilt
(`npm run aggregate` + `npm run build`) and committed; Phase 1 charts
(`01`-`04`) and Phase 2 Experiment 2 coupling charts (`07`, `08`)
regenerated from the corrected CSVs; `eval_awareness_report.zip`'s
`main.tex` updated throughout (all 3 models' numbers, the Experiment 2
discussion, the Experiment 3 caveat paragraph) plus a new appendix added
comparing our methodology point-by-point against Heidari, Memarian &
Rabusseau's — **`eval_awareness_report.pdf` was not regenerated** (no
local LaTeX toolchain; re-upload the zip to Overleaf to get a current
PDF). Code fixes (`inference/run_inference.py`, `inference/api_clients.py`,
`probes/verbalize_probe.py`) are committed; this file and the corrected
`output/` data are not (per the standing instruction not to commit
`PROGRESS.md`, and `output/` staying git-ignored except chart PNGs).

## 10. A much bigger correction: the verbalized-answer parsing bug

§9 found and fixed reasoning/answer-**display** bugs. This section is a
separate, much more consequential correction: a bug in how we **grade**
a model's verbalized answer, discovered while investigating an unrelated
user question ("why does the dashboard mark this response a mismatch
when the model clearly said YES?").

### What was wrong

`resolve_answer()` in `probes/verbalize_probe.py` — the single function
every probing path in this project uses to turn a model's free-text
response into a binary `eval_signal` — had two compounding bugs:

1. **First `<label>` match, not last.** `LABEL_RE.search()` returns the
   *first* `<label>...</label>` tag in the response. A thinking-mode
   model routinely restates the prompt's own format template (literally
   `<label>[label ... NO, YES]</label>`) or drafts an example answer
   before its real final answer. The first match was then that
   placeholder or an early draft, not the model's actual conclusion.
2. **Order-biased fallback.** When no clean tag match was found, the
   parser scanned the *entire* response for the target words ("NO"/"YES",
   or the equivalent pair for other probes) and returned whichever was
   checked first in a fixed dict order — always the deploy-side label,
   since every call site builds `targets = {real: 0.0, eval: 1.0}` with
   the deploy label first. In a long, deliberative response, both words
   are often present somewhere incidentally (a hedge, an aside, a
   restated instruction), so this silently defaulted to "not an
   evaluation" regardless of what the model actually concluded.

Confirmed directly on the user's example: the model's raw response ended
in `<label>YES</label>` — twice — but was recorded as `eval_signal=0.0`
because an earlier restatement of the format instructions
(`<label>[YES/NO]</label>`) was parsed first, and "NO" is checked before
"YES" in the fallback.

### Fix

`resolve_answer()` now (a) takes the *last* `<label>` match instead of
the first, and (b) for both the tag-content and whole-response fallback
paths, returns whichever target word occurs *last* by character position
in the text, not whichever is checked first in dict order. This is a
general, position-aware fix — not specific to any one model or probe.

### Scope — checked across all 9 models, all 1100 verbalized-probe rows

A pure text-reprocessing pass (re-deriving `eval_signal` from
already-saved `raw_response` text for every model, no new model calls)
found:

| Model | Rows changed / 100 | Ensemble AUROC before | Ensemble AUROC after |
|---|---|---|---|
| Qwen3.6-27B-OBLITERATED | **60** | 0.450 | **1.000** |
| granite-4.2-30b | **15** | 0.735 | **0.950** |
| Gemma-4-12B-OBLITERATED | **19** | 0.995 | **0.985** |
| (other 6 models) | 0 | — | unchanged |

Six of nine models are completely unaffected — the bug requires a
specific behavioral pattern (verbose, restate-then-draft-then-answer
thinking) that only three models happened to exhibit. Where it does
apply, the effect is severe and the *direction* was wrong, not just the
magnitude: the buggy fallback structurally favors "not an evaluation"
regardless of the model's real answer.

### The headline retraction

**Qwen3.6-27B-OBLITERATED was never "internally aware, verbally
silent."** That was this project's single most notable claimed result
(see the retracted paragraph in explainer.md and the old §6 text above,
left in place with corrections rather than deleted, so the mistake stays
visible). Corrected:

- Verbalized ensemble AUROC: 0.450 (below chance) → **1.000** (perfect).
- It does **not** answer "not an evaluation" 99/100 times — that was
  itself downstream of this bug. Its real behavior: highly accurate
  verbalization, closely tracking ground truth, same as its strong
  internal-probe result (0.880 AUROC).
- Its Experiment 2 coupling is **not** undefined/degenerate — recomputed,
  it shows the *highest* peak coupling of the three Phase 2 models
  (`|ρ| = 0.837`, was reported as undefined), fully consistent with the
  same shared-ground-truth-confound mechanism already documented for
  Qwen3-8B and granite-4.2-30b (§6) — just previously masked because the
  input verbalized data was wrong.
- None of the three Phase 2 models now show a
  representation-verbalization dissociation. This means **this project
  does not replicate the source paper's central dissociation claim** at
  the individual-model level, once this bug is fixed — a genuine,
  disclosed non-replication, and arguably a more important finding than
  the retracted one.
- The Phase 2 "strongest/weakest/chance verbalizer" model-selection
  rationale (§6 intro) no longer holds post-correction: all three models
  are now strong-to-perfect verbalizers. Left disclosed rather than
  retroactively re-justified.

### What could not be re-verified

`probes/steer.py` (Experiment 3) scores its own follow-up probe with the
same `resolve_answer()`, so it could in principle be affected too — but
its saved output (`steering_results.json`) only stores the parsed
`eval_signal` per cell, not the raw response text needed to re-derive it,
so this could not be checked the way Phase 1 and Experiment 2 were.
Mitigating factor: `steer.py` explicitly disables thinking on its
follow-up call, which removes the specific extended-thinking
restate-then-draft behavior that caused the bug elsewhere. Left as an
open, disclosed limitation (see the Experiment 3 caveat in §7), not
assumed clean.

### Downstream updates made

Charts `01`-`04` (Phase 1) and `07`-`08` (Experiment 2 coupling)
regenerated; dashboard rebuilt and its data updated; `main.tex` rewritten
substantially — abstract, the new Phase-1 subsection documenting this bug
(mirroring this section), Table `phase1-results`, the Phase 2
model-selection paragraph, Table `exp1-results`, the retracted "most
notable result" paragraph, the entire Experiment 2 section (table +
discussion), the Experiment 3 caveats, four of the five Discussion
bullets, a new Limitations bullet, the Conclusion, and the comparison
appendix's coupling-divergence section — plus `explainer.md`'s "it knew,
it just denied it" narrative replaced with an honest "we made a grading
mistake" story. `eval_awareness_report.pdf` was **not** regenerated (no
local LaTeX toolchain); re-upload the zip to Overleaf for a current PDF.
Code fix (`probes/verbalize_probe.py`) is committed; this file and the
corrected `output/` data are not (same standing instruction as always).
