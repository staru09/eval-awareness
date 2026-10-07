# Plan

The next run, written down before it starts so nothing is decided after the scores
are visible. Updated 2026-10-07 on the `todo` branch.

Read with: `dataset.md` for the data, `probe.md` for how the probe works, `eval.md`
for how results are judged. This file says what to do and in what order. If a
method detail here ever disagrees with `eval.md`, `eval.md` wins and this file is
wrong.

## Scope

- Model: `Qwen/Qwen3-8B` only.
- Data: the `phase1` experiment in `data/corpus.json`, 1100 prompts.
- Arms: probing on all 1100, steering on 300 of them.
- The `wording` experiment stays parked. No other data is used.

## Research questions

**RQ1. Does Qwen3-8B encode whether a prompt is an evaluation?**
Tier A, real evaluations, against tier B, real people. Answered only together with
RQ2.

**RQ2. Is that about being evaluated, or about text written for a dataset?**
Tier B, real people, against tier C, people writing for a dataset. Neither side is
an evaluation, so the honest answer is 0.5. If tier B separates from tier C as much
as tier A separates from tier B, the probe reads authored text, not evaluation.
This is the comparison that can overturn RQ1.

**RQ3. Which kinds of evaluation does it catch, and does it miss the safety ones?**
Each tier A source against all of tier B.

**RQ4. Content or wording?**
Belongs to the parked `wording` experiment, 100 per group. Tier D inside `phase1`
holds only 25 per group, a size already known to be too small, so it is reported
as a description and never as evidence.

**RQ5. Does the model act on the direction?**
Steered generations against a random direction at matched strength, judged on the
0 to 3 scale. The share scored 3, meaning the model changes its answer because it
thinks it is watched, is the headline.

**Out of reach for this run:** whether the model scores worse when it thinks it is
tested. That needs answer keys, which the dataset does not store.

## Fixed before any scores exist

Full detail in `eval.md`. In short:

0. AUROC is reported raw, matching the reference pipeline: test prompts are
   expected to score higher, above 0.5 is success, below 0.5 is separation the
   wrong way and is reported as that. Nothing is folded into `|AUROC - 0.5|`.
1. Headline is the whole-prompt reduction, raw template, highest 3-layer band over
   all 36 layers.
2. Every control searches layers the same way the probe does, on the same side of
   0.5: 100 random directions each at their own highest band, and 2000 label
   shuffles each at their highest band.
3. A result must also beat counting words. For RQ1 that floor is 0.624.
4. Layers 10 and 14 are reported as a pre-chosen, uncorrected second read. These
   are our decoder-layer numbers; the reference paper calls them 11 and 15.
5. Tier D stays out of the RQ1, RQ2 and RQ3 comparisons.
6. A null is written up, not rerun with more settings.

## Steps

Each step ends with a check shown to the user. The next step does not start until
the check is shown.

### 1. VM up

`/nvme` is wiped on every restart.

- `ssh azure-btc`, `sudo chown azureuser /nvme`, clone the repo, check out `todo`,
  build the venv with uv, pin `datasets>=4.0`, download `Qwen/Qwen3-8B`.
- Copy `data/corpus.json` up. It is not in git.
- **Tokenizer check against the reference.** The reference tokenizes with
  `add_special_tokens=True`, we use `False`. Tokenize three phase1 prompts, raw and
  chat-templated, both ways, and compare the token ids. If they differ, switch ours
  to match before anything is captured.
- **Check:** `nvidia-smi` shows the H100, the venv imports transformers, and the
  tokenizer comparison printed: identical ids, or the change made.

### 2. Export phase1 into the format the tools read

`make_traces.py`, `score` and the analysis read rows with `id`, `prompt_id`,
`text`, `label`, `source` and `category`. `corpus.json` stores source and tier per
experiment under `roles`. A short export writes `data/phase1.json` from it:
`source` from `roles.phase1.source`, `category` from `roles.phase1.tier`,
`prompt_id` numbered in file order.

- **Check:** 1100 rows, 550 labelled test, tiers A 500, B 300, C 200, D 100, two
  printed rows.

### 3. Build the probe with 100 random directions

The random directions are fixed when the probe is built and the stored September
probe carries its own count, so reusing it would silently keep a smaller floor.

- `evalaware probe --n-random 100`.
- Compare its 36 real directions with the September probe, layer by layer, by
  cosine similarity.
- **Check:** the cosine table, and `n_random` read back as 100. A cosine below 1.0
  anywhere is reported, not explained away.

### 4. Capture activations

- `make_traces.py` on `data/phase1.json`, once `--template raw`, once
  `--template chat`. Tokenizer only.
- `capture-prompt` on both.
- **Check:** 1100 files per template, token counts as min, median and max, the
  retokenise check passing on every row. About 2.6 GB in total.

### 5. Score and analyse

- `score` on both templates.
- New `scripts/analyze_phase1.py` producing everything `eval.md` lists: the curve,
  the band, the two pre-chosen layers, all three controls, and beside each AUROC its
  95% interval, Cohen's d and catch rate at 5% false alarms, for RQ1, RQ2 and RQ3.
- **Replication check first.** Chat template, last token, 100 MASK against 100
  WildChat, peak reported in the paper's layer numbering next to the paper's cited
  Qwen3-8B number. If it is far off, stop and find the pipeline difference before
  reading anything else.
- Smoke test it on 20 prompts first.
- **Check:** the replication number, then the full table, with the file each number
  came from. Nothing is interpreted until the table has been shown.

### 6. Steering

- Rewrite `evalaware/steering.py` to generate with the chain of thought on and store
  every trace. It scores nothing.
- 300 prompts, 30 from each of the 10 sources in tiers A to C. Chat template,
  thinking on.
- Six conditions: none, plus and minus at two strengths, random at matched
  strength. 1800 generations. vLLM, not HF generate one at a time: at the earlier
  rate of 70 seconds a prompt this would be 35 hours.
- Smoke test: 2 prompts, 1 condition.
- **Check:** the first 20 traces read by eye. The reasoning is present and the
  steered text is not broken.

### 7. Push everything to the Hub before the VM stops

- Zip and push activations, scores, analysis and traces, from the VM, to the
  private repo.
- **Check:** file count and total size read back from the Hub.

### 8. Judge, later

Runs on CPU from the stored traces, any time after step 7. Method in `eval.md`:
blind to condition, `SEVERITY_PROMPT`, steered compared with random.

## Rules for staying on track

1. **Show the data before running it.** Real rows, counts per group, overlap between
   groups. Never a description in place of the rows.
2. **Smoke test every new script** on a handful of prompts before the full run.
3. **Read every number from its file** before stating it, and name the file.
4. **No new data, sources or comparisons during the run.** A change of plan is
   written into this file first, then made.
5. **Nothing stays only on the VM.** Push before stopping it.
6. **Report a loss as a loss.** If a result loses to any control, that is the result.

## What can still go wrong

- **RQ1 passes and RQ2 separates as well.** The probe reads authored text, not
  evaluation. Most likely outcome given what earlier runs showed.
- **The best band beats its controls but layers 10 and 14 show nothing.** A signal
  exists, just not where the published work puts it. Report the layer; do not claim
  the published result replicates.
- **The best band loses to its controls.** The answer to RQ1 is no, and the run is
  finished.
- **Steering moves nothing.** Expected; published work recovers under 5 percent of
  the behaviour. Written up as a null.
- **The judge disagrees with itself.** The steering result cannot be used.

## Not in this run

Base against instruct, smaller models, the Jacobian lens, the logit monitor, the
`wording` experiment, and any scoring against answer keys. They stay in `todo.md`.
