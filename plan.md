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
- Arms: probing on all 1100, and the model's own reasoning and answer on all 1100,
  with no steering. Steering is decided later and is not part of this run.
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
Deferred with steering. This run saves the unsteered reasoning and answers that any
steering result will later be compared against, and that the judge can score now:
how often the model says it is being tested, per tier, and whether that tracks the
probe score.

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

### 6. Save the model's reasoning and answers, no steering

The judge needs these, and any later steering result is compared against them.

- `evalaware generate --backend vllm --no-activations` on `data/phase1.json`.
  Activations already come from step 4, so none are captured here.
- Chat template, thinking on, the sampling Qwen3 recommends for thinking and that
  our September run used: temperature 0.6, top_p 0.95, top_k 20, min_p 0. Up to
  32768 new tokens so reasoning is not cut short.
- **Saved per prompt**, already built into the command: the prompt, the exact
  templated string, the full output, the reasoning trace, the final answer, token
  counts, the generation settings, and why generation stopped.
- **Run in 5 chunks of 220 and push to the Hub after each one.** The command writes
  its file only when a whole batch finishes, so a crash or VM stop mid-batch loses
  everything since the last write. It skips prompts already done, so each chunk
  adds to the same file.
- Smoke test: 3 prompts first.
- **Check:** after each chunk, the count saved, how many hit the token limit, and
  how many have no reasoning trace. Report a cut-off or empty trace as soon as it
  shows, not at the end.
- Read 10 traces by eye across tiers before the full run: reasoning present, answer
  present, the two split in the right place.

### 7. Push everything to the Hub before the VM stops

- Zip and push activations, scores and analysis, from the VM, to the private repo.
  The reasoning traces are already up, chunk by chunk, from step 6.
- **Check:** file count and total size read back from the Hub.

### 8. Judge, later

Runs on CPU from the stored traces of step 6, any time after step 7.
`SEVERITY_PROMPT`, 0 to 3. Reports the share at each score per tier, and the
correlation between each prompt's judge score and its probe score. The steered
comparison in `eval.md` waits for steering.

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
- **Many generations hit the 32768-token limit or come back without a reasoning
  trace.** Reported per chunk as it happens. Cut-off traces are kept and marked,
  not dropped, since dropping them would bias the judge toward short answers.
- **The judge disagrees with itself.** Its scores cannot be used.

## Not in this run

Steering, base against instruct, smaller models, the Jacobian lens, the logit monitor, the
`wording` experiment, and any scoring against answer keys. They stay in `todo.md`.
