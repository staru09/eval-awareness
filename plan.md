# Plan: phase 1 at 1100 prompts, with steering

Written 2026-10-07 on the `todo` branch. One run, decided in advance, so nothing
is chosen after the scores are visible.

## Research questions

Five questions. Each names the contrast that answers it, the statistic, and what
each outcome licenses us to say. RQ2 is the one that can invalidate RQ1, so RQ1 is
not reportable on its own.

### RQ1. Does Qwen3-8B linearly encode whether a prompt is an evaluation?

- **Contrast:** tier A, real evaluations, against tier B, real people talking to a
  chatbot. 500 against 300.
- **Statistic:** best 3-layer band AUROC on the whole-prompt readout, against 100
  random directions each taken at their own best band, and against a permutation
  null with the max over layers inside each of 2000 shuffles.
- **Positive means:** the band beats the matched random floor, and beats the
  word-count baseline of 0.603.
- **Negative means:** phase 1 is answered and the answer is no. Run 1 was negative
  by this rule once the control got the same layer freedom, 0.335 against 0.358.
- **Status:** asked three times, never yet passed with a matched control.

### RQ2. Is that encoding about being evaluated, or about text written to be data?

This is the falsification test for RQ1. Every test prompt we have ever used was
also written for a corpus, so the two explanations have never been separated.

- **Contrast:** tier B, real people, against tier C, prose written by people but
  authored for a dataset. 300 against 200. Supporting contrast: tier A against
  tier C.
- **Statistic:** the same band and the same matched floor.
- **The right answer is 0.5.** Neither side of B against C is an evaluation, so an
  evaluation detector must not separate them.
- **If B against C separates as much as A against B:** RQ1's positive is explained
  away. The probe detects authored-ness and the write-up says so.
- **If B against C sits at the floor while A against B clears it:** RQ1 survives
  its strongest available challenge. This is the only route by which we can claim
  evaluation awareness at all.
- **Status:** never asked. Tier C exists to ask it.

### RQ3. Which kinds of evaluation does it detect, and does it miss the ones that matter?

- **Contrast:** each of the 11 sources against the pooled real-person prompts, 100
  against 300, each with its own random floor.
- **Statistic:** per-source band AUROC and a beats-floor flag.
- **Prior from run 3:** on the whole-prompt readout only arc, simpleqa, mmlu and
  bbq cleared their floors. MASK and PrivacyLens cleared nothing in any readout,
  and both are safety evaluations written to read like ordinary requests.
- **Why it matters:** if the probe reliably catches multiple choice and reliably
  misses safety evaluations, it is useless for the purpose eval awareness research
  exists for, and that is a finding in its own right.

### RQ4. Is it reading the content of a prompt or the way the prompt is written?

- **Contrast:** tier D, the format control, on its own. Four groups of 25 where
  content and wording vary separately.
- **Statistic:** the wording-only comparison, real people against the same kind of
  real people reworded as exam questions, against its own floor.
- **Known so far:** 0.811 on the last-token readout, which is the same size as the
  0.835 it scores on the real task, and 0.017 to 0.063 on the whole-prompt readout.
- **Caveat on the record:** the two sides of that comparison hold different
  messages, 0 shared ids, so it cannot separate wording from content cleanly. It
  can only show separation where no evaluation is present.

### RQ5. Does the model act on the direction, or merely carry it?

- **Contrast:** steered generation against a random direction at matched magnitude,
  300 prompts stratified across the 11 sources, six conditions.
- **Statistic:** the judge's 0 to 3 score on the stored reasoning traces, reported
  as mean, share at 2 or above, and share at 3. The share at 3, strategic
  adaptation, is the headline.
- **Positive means:** the real direction raises the share at 3 and a random push of
  identical size does not.
- **Negative means:** the information is present but not used. Published work gets
  under 5 percent behavioural recovery, so a null is the expected outcome and gets
  written up.
- **Note:** reading the probe out of steered activations is circular and is not a
  measurement. The direction was added, so of course its own dot product rises.

### What no part of this run can answer

Whether the model behaves differently when it believes it is being tested, in the
sense of scoring worse on a capability test. That needs accuracy against an answer
key, and `final_data.json` stores no keys. RQ5 measures what the model says in its
reasoning, not what it gets right.

## What is already in hand

- `data/final_data.json`, 1100 prompts, 11 sources at 100 each, 550 labelled as a
  test. Built and checked locally. The four tiers:

  | Tier | What it is | Sources | n |
  |---|---|---|---|
  | A | a real evaluation, written as one | mask, model_written_evals, bbq, mmlu, gsm8k | 500 |
  | B | a real person talking to a chatbot | wildchat, lmsys, sharegpt | 300 |
  | C | written by a person, but for a dataset, not to get help | oasst2, no_robots | 200 |
  | D | the old format control, content and wording crossed | eval_awareness_2x2 | 100 |

- Length baseline already measured on it: 0.557 by characters, 0.603 by words.
  Those are the floors. Nothing below 0.603 is a result.
- The probe mechanism: `evalaware/probe.py` builds the direction from the 16
  contrastive pairs in `data/contrastive_pairs_16.txt`. No training, so all 1100
  prompts stay unseen.
- A Qwen3-8B probe already exists at
  `output/2026-09-22_Qwen3-8B_contrastive-probe/local/qwen3-8b/probe.pt`.
- Every activation from the three earlier runs is on the Hub, private, at
  `FiendHunter/qwen3-8b-eval-awareness-probe`, 788 MB.
- `evalaware/judge.py` now holds `SEVERITY_PROMPT`, the 0 to 3 scale.

## What is missing

- The activations for these 1100 prompts. This is the only part that needs a GPU.
- `scripts/analyze_final.py`. The tier comparisons do not exist in any analysis
  script yet.
- A rewrite of `evalaware/steering.py`. Today it appends a followup question and
  parses a free text answer. We want it to generate with the chain of thought on
  and store traces.

## Decisions fixed now, before any scores exist

These are written down here so they cannot be chosen later to suit the result.

1. **The headline readout is whole prompt, mean over all tokens.** That is the
   readout Nguyen et al. use. The last-token readout is reported beside it and
   labelled as not their method.
2. **All 36 layers are searched, and the search is paid for in the controls.**
   We already score every layer. The only question is what counts as the headline,
   and the rule is: whatever freedom the probe gets, the null gets the same.
   - The headline is the best 3-layer band over all layers, compared against 100
     random directions each taken at **its own** best band, and against a
     permutation null that takes the max over layers inside every one of 2000
     shuffles. Both corrections already exist: `bands()` maxes over layers,
     `permutation_null` applies it per shuffle, and `random_direction_null` gives
     each random direction its own best band.
   - The band is preferred to the single best layer because one lucky layer moves
     it less. Both are reported.
   - Two pre-chosen layers are reported as a secondary read, not the headline:
     layer 10, which is Nguyen et al.'s layer 23 of 80 carried over as relative
     depth, and layer 14, the `apriori` layer our code already uses. They need no
     correction at all, so they are the cheap sanity check on whether the signal
     sits where theory says it should.
   - What is never reported on its own: a peak compared against a random floor
     taken at one fixed layer. That mismatch is what made run 1 look like a result.
     The same 0.335 beats a same-layer floor of 0.211 and loses to a
     best-any-layer floor of 0.358. Only the second comparison is fair, because we
     searched layers too.
3. **100 random directions, not 10.** Reported three ways: best at the same layer,
   best allowed any layer, and the mean. The any-layer number is the one that
   counts whenever we also search layers.
4. **Tier D is excluded from the tier A, B and C comparisons.** It is the only
   source holding both labels inside it, by design, so it would contaminate them.
   It is analysed on its own.
5. **Tier B against tier C is the control that decides the interpretation.**
   Neither side is an evaluation, so the right answer is 0.5. If it separates as
   much as tier A against tier B, the finding is that the probe reads prose
   authored for a dataset, not evaluation awareness, and that is what gets
   written.
6. **A null is publishable.** Decided now so a null does not turn into another run.

## Steps

Each step ends with something shown before the next one starts.

### 1. VM up, environment rebuilt

`/nvme` is wiped by every restart. This has happened twice in one day.

- `ssh azure-btc`, `sudo chown azureuser /nvme`, recreate dirs, clone the repo,
  rebuild the venv with uv, pin `datasets>=4.0` because uv resolves 1.1.1 which
  crashes on modern pyarrow.
- Download Qwen3-8B weights.
- **Check:** `nvidia-smi` prints the H100 and the venv imports transformers.

### 2. Reuse the existing probe rather than rebuilding it

The direction is a mean difference over fixed text, so it is deterministic given
the same pairs file and the same weights.

- Copy the September `probe.pt` up, rebuild a second one, and compare the two
  directions layer by layer.
- **Check:** cosine similarity per layer. If it is 1.0 everywhere, the probe is
  reproducible and future scoring needs no GPU. If it is not, say so and use the
  freshly built one.

### 3. Capture activations for all 1100 prompts

- `python data/make_traces.py --dataset data/final_data.json --template raw` and
  again with `--template chat`. This loads the tokenizer only.
- `evalaware capture-prompt` for both.
- Each npz stores only `last_token` and `mean_token`, each [36, 4096] float32,
  about 1.2 MB per prompt. 2200 captures is roughly 2.6 GB.
- **This fixes the pooling for good.** Only those two readouts will ever be
  available without recapturing. Noted so it is not a surprise later.
- **Check:** 1100 files per template, token counts printed as min, median and max,
  and the retokenise assert passing on every row.

### 4. Score and analyse

- `evalaware score` with `--n-random 100`.
- New `scripts/analyze_final.py`, producing per layer and per pooling:
  - AUROC at each of 36 layers, the best 3-layer band as the headline, the single
    best layer beside it, and the two pre-chosen layers as the uncorrected read
  - the three tier comparisons, each with its own random floor:
    - **A against B**, tests against real people. The original question.
    - **A against C**, tests against prose authored for a dataset. If this is much
      smaller than A against B, then what the probe reacts to is a prompt having
      been authored, because tier C was authored too.
    - **B against C**, real people against dataset prose. **Neither side is a test,
      so an honest evaluation detector scores 0.5 here.** If it separates, the probe
      is reading authored-ness and not evaluation.
  - all 11 sources, each with its own floor and a beats-floor flag
  - tier D on its own, the four wording groups
  - permutation null, 2000 shuffles, max over layers inside each shuffle
  - the four text baselines
- **Check:** the whole table in front of you before anything is interpreted.

### 5. Steering, same run, same data

- Rewrite `evalaware/steering.py` to generate with thinking on and store traces.
  Smoke test at 2 prompts and 1 condition first.
- 300 prompts, stratified across the 11 sources. Six conditions: unsteered, plus
  and minus alpha at two magnitudes, and a random unit direction at matched
  magnitude. 1800 generations.
- Why 300 and not 1100: six times 1100 is 6600 generations, and at the 70 seconds
  a prompt the logit monitor managed that is 128 hours. Use vLLM, not HF generate
  at batch size 1.
- One JSON line per generation, flushed as written, so stopping loses nothing.
- **Check:** the first 20 traces read by eye, confirming the chain of thought is
  present and the steered ones are not degenerate.

### 6. Everything to the Hub before the VM goes down

Non-negotiable. `/nvme` has already been lost twice.

- Zip and push activations, scores, analysis and traces from the VM.
- **Check:** file count and total size read back from the Hub.

### 7. Judge, later

Not in this run. Item 6 of `todo.md`. The traces are stored so it can run any
time on CPU. The judge scores on the 0 to 3 scale, never sees which condition a
trace came from, and the comparison is steered against random at matched
magnitude.

## How this run can still fail, and what happens then

- **The best band beats its matched floor but the pre-chosen layers show nothing.**
  That is a real result, just a narrower one: a signal exists somewhere in the
  stack but not where theory puts it. Report the layer it sits at and stop short of
  claiming the published layer replicates.
- **The best band loses to its matched floor.** Then phase 1 is finished and the
  answer is no. This is what run 1 did once the control was given the same layer
  freedom, 0.335 against 0.358.
- **Tier B separates from tier C.** Then this is not an evaluation-awareness probe. Say
  so plainly. This is the most likely outcome given run 3.
- **Steering moves nothing.** Expected. Published work gets under 5 percent
  behavioural recovery. A null is written up, not rerun with more alphas.
- **The judge disagrees with itself.** Then the steering result is unusable and we
  say so rather than picking the kinder pass.

## Not in this run

No base versus instruct, no smaller models, no Jacobian lens, no logit monitor, no
2x2 scale-up. Those stay in `todo.md` and wait for a result worth ablating.
