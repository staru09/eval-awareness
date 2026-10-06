# Todo: linear probes and the Jacobian lens for evaluation awareness

Scope for this round of work. Two models only:

- `Qwen/Qwen3-8B`, 36 layers, 4096 dims, has a reasoning channel
- `meta-llama/Llama-3.2-3B-Instruct`, 28 layers, 3072 dims, no reasoning channel

Methods: linear probes following the Geometry of Truth repo
(https://github.com/saprmarks/geometry-of-truth) and the Jacobian lens
(https://github.com/anthropics/jacobian-lens).

The design questions behind this plan are still open, see **Open decisions** at
the end. Where an answer was needed to write anything down, the assumption is
marked **assumed** and can be changed.

## What we are claiming

**assumed:** the spine is a clean reproduction and correction. The VEA writeups
claim 100 percent probe accuracy on eval versus deployment, and a signal crash
at a late layer. That number is almost certainly inflated by training and
testing inside one source. We reproduce it on two open models under a protocol
that holds out the source, report what survives, and get the layer and token map
out of the same sweep.

Current state of our own result, for context: the internal probe separates eval
from deployment, and the verbalized signal does not track it. Peak correlation
sits inside the random direction band in 5 of 6 model by readout cells. We
cannot yet say where that break happens.

## Phase 0: environment, done 2026-10-06

- `/nvme` was wiped by the VM restart again. Ownership fixed, `/nvme/hf-cache`,
  `/nvme/scratch` and `/nvme/.cache` remade.
- Repo cloned to `/nvme/eval-awareness` at 81e1b99, venv built with uv.
- Qwen3-8B, 16 GB, and Llama-3.2-3B-Instruct, 12 GB, in `/nvme/hf-cache`.
  Configs and tokenizers load.
- `data/dataset_jjpn2_full.json` built: 260 rows, 130 eval and 130 deployment,
  31 sources, median 631 characters.

Open item from this phase: uv resolved `datasets 1.1.1`, which crashes on modern
pyarrow. Forced to 5.1.0 by hand. Pin `datasets>=4.0` in `pyproject.toml` or the
next clean rebuild hits the same wall.

## Phase 1: pinpoint the layer and the token pattern

This comes first because every later arm needs one layer and one token position,
and we currently pick them by taking the best cell of a sweep on the same metric
we then report. That is selection on the test set.

- [ ] Capture per-token activations, not just the four reductions we have now.
      `activations.py` saves only last token and mean token, over the prompt and
      over prompt plus answer. A token pattern sweep needs every position for at
      least one layer, or a named set of positions for all layers. This is new
      code and much more disk.
- [ ] Token patterns to compare: last prompt token, last non-special token,
      final punctuation, mean over the user turn, mean over the whole prompt,
      and the assistant turn's first generated token.
- [ ] Sweep every layer against every token pattern on a tuning split.
- [ ] Report three numbers, not one: the cell chosen on the tuning split scored
      on an untouched test split, the best three layer band from
      `metrics.band`, and a fixed a priori cell.
- [ ] The a priori cell is about 0.4 of depth, which is layer 14 of 36 on
      Qwen3-8B and layer 11 of 28 on Llama-3.2-3B. Geometry of Truth hardcodes
      layer 13 of 32 for Llama-2-7B, 14 of 40 for 13B, 27 of 80 for 70B. Do not
      use two thirds of depth, that was wrong.
- [ ] Compare the two models in relative depth, never by absolute layer index.

**assumed:** the sweep runs on prompt only activations, one forward pass per
prompt, no generation in the loop. Generation is only needed later for the judge
and the lens arms.

## Phase 2: make the probes faithful to the paper

Our `evalaware/linear_probes/` already matches the paper exactly for mass mean
and logistic regression, down to AdamW at lr 1e-3, weight decay 0.1, 1000
epochs, bias free. Four deviations need a decision each.

- [ ] `ccs.py` is a PCA probe, not CCS. The real CCS loss is consistency plus
      confidence on contrast pairs, with a sign fix from labels afterwards.
      Either implement it or stop calling the arm CCS.
- [ ] The covariance corrected mass mean variant exists but no evaluation path
      ever switches it on. The paper uses it for in distribution cells and the
      plain direction out of distribution. Ours therefore reports the paper's
      `iid=False` setting everywhere, including the diagonal.
- [ ] We centre by the training fold's mean. The paper centres each dataset by
      its own mean before splitting. Ours is the leak free version and theirs is
      the published one. Pick one and say which.
- [ ] `cross_dataset` never centres at all, so that path runs raw activations
      against bias free probes. That is a bug, not a choice.
- [ ] We report AUROC. The paper reports accuracy at threshold 0.5 and never
      AUROC. Report both or no number of ours is comparable to theirs.

## Phase 3: datasets

Hard fact: we have exactly one upstream source wired up, `jjpn2/eval_awareness`.
The 31 "sources" inside `dataset_jjpn2.json` are categories within that one
dataset. Nothing else in `vea_datasets.csv` has a loader.

Ceiling of that source: 130 eval and 142 deployment transcripts. 228 of the 500
chat records are dropped by the 200 character minimum in `build_dataset.py`, and
they are mostly genuine one line prompts. Lowering the floor to 50 roughly
doubles the usable set.

- [ ] Decide whether to lower the length floor. It is a one line change and it
      is the cheapest way to double our data.
- [ ] Train on the source labelled split, which is what we have.
- [ ] Add at least one deployment control with a loader: WildChat or
      LMSYS-Chat-1M, topic matched. Both are marked high priority control in
      `vea_datasets.csv`.
- [ ] Add XSTest as the control that matters most: safe prompts that sound
      dangerous, so format says eval and content says harmless.
- [ ] Add a cue controlled set, same content under eval framing, casual framing
      and neutral framing. A probe that separates these cannot be reading topic
      or format. This is the test that decides whether our accuracy is real.
- [ ] Consider the Needham et al. eval awareness transcripts, 1,000 items with
      externally defined labels, for a transfer test we did not construct.
- [ ] Report three accuracies every time: within source, leave one source out,
      and on the cue controlled pairs.

## Phase 4: the Jacobian lens arm

The upstream repo is already cloned at `Z:\Sep-26\VEA_experiments\jacobian-lens`.
It needs `transformers>=5.5`, and the VM has 5.18.0.

- [ ] Push the probe direction at the chosen layer through `J_l` and read the
      vocabulary, with `direction_logits`. That is the dissociation measurement:
      does the eval direction decode to eval words at all.
- [ ] Qwen3-8B has a pre fitted lens in our `PREFITTED` table.
      Llama-3.2-3B-Instruct does not, only Llama-3.1-8B-Instruct does. Fitting
      our own costs one forward and `d_model / 8` backward passes per prompt,
      which is 384 per prompt at 3072 dims, times about 100 prompts. Budget it
      as a one off GPU job. Fitting all layers costs the same as fitting one.
- [ ] Pin the lens version. Our download passes no `revision`, so the file is
      unpinned, and our filenames differ from the ones the upstream walkthrough
      uses. Two runs months apart could silently use different lenses.
- [ ] Controls the README prescribes but no code implements: random directions,
      the logit lens baseline with `use_jacobian=False`, and fixed word list
      ranks.
- [ ] Our wrapper reads one position only. The position by layer viewer from
      2026-09-23 came from a throwaway script that is not in the repo. If we
      want that view again it is new code either way.

## Phase 5: ablations

- [ ] Smaller models of the same family, to see whether the layer sits at the
      same relative depth. Qwen3.5-0.8B already has two runs on disk.
- [ ] Base versus instruct. The VEA claim is that base models detect eval as
      well as aligned ones, which would put the phenomenon in pre training. This
      is the single most interesting ablation and it needs a base checkpoint
      that matches one of our two models.
- [ ] Finetuning stages, if any checkpoint series is available for either model.
- [ ] Steering. Note this is a rewrite, not a rerun. Geometry of Truth adds the
      direction at every layer from 8 up to the probe layer, at two specific
      token positions, scaled so the mean negative activation lands on the mean
      positive one, and reports a normalised indirect effect on
      `P(" TRUE") - P(" FALSE")`. Our `steering.py` adds to the whole residual
      stream at one layer, scales by an alpha sweep times the mean activation
      norm, and measures a parsed free text answer with no random direction
      control. Those are different experiments.
- [ ] OpenAI judge on reasoning chains, as an ablation. `evalaware/judge.py`
      already works and has run on 10 models at 50 traces each with gpt-4o-mini.
      It asks two things: did the model say it was being tested, and can a judge
      tell eval from deployment from the rollout alone with the prompt hidden.
      **assumed:** this arm runs on Qwen3-8B only, because Llama-3.2-3B-Instruct
      has no reasoning channel and forcing it to narrate would change the one
      thing we are measuring.

For reference, the judge results we already have. Rate of the model saying it is
being tested, eval prompts first, 50 transcripts each:

- Qwen3.5-0.8B 0.72 and 0.48, Olmo-3.1-32B 0.60 and 0.40, MiniCPM5-2B 0.60 and
  0.36, Olmo-3-7B 0.44 and 0.24
- Qwen3-8B 0.32 and 0.20, Ministral-3-3B 0.24 and 0.16, Nemotron-30B 0.20 and
  0.16
- granite-4.2-30b 0.36 and 0.48, which is backwards, Llama-3.1-8B 0.08 and 0.04,
  Phi-3-mini 0.04 and 0.04

Confidence against the true label stays near chance everywhere, 0.45 to 0.63.
The blind classifier does better, 0.53 to 0.68. One run is still unjudged:
`output/2026-09-22_Qwen3.5-0.8B_contrastive-probe/local/qwen3.5-0.8b/`.

## Known problems carried in

- `echo_end` in the logit monitor fails on prompts where the model's copy of the
  question diverges from the original. The deployment half of the smoke run is
  invalid because of it. A fix was drafted and rejected, so it is untouched.
- No run kept its activations. Only the legacy `output/local/qwen3-8b/` has
  them, 20 transcripts. Every new capture has to bring activations back from the
  VM, not just the derived scores.
- `.env` holds live looking OpenAI and Anthropic keys in plaintext. It is
  gitignored and untracked, so nothing is exposed in the repo.

## Open decisions

These were asked and not yet answered. Everything above that depends on them is
marked assumed.

1. Is the headline claim the reproduction and correction, the localisation of
   where the signal stops feeding the output, or the methodological point that
   probe accuracy is mostly dataset recognition.
2. Does the eval label come from the source dataset, from cue controlled pairs,
   or from external transcripts.
3. Do we probe the prompt alone, or the prompt plus the model's own output.
4. How do we pick the layer and token without selecting on the reported metric.
5. Which model do we calibrate on, and must the other agree.
6. Does the judge arm run on Qwen3-8B only.
7. How does this file get committed, given `.gitignore` excludes `*.md`.
