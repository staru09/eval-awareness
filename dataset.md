# Datasets

Every prompt set in this project, what each one represents, and the exact counts.
Written 2026-10-07. All numbers measured from the files, not recalled.

The model is `Qwen/Qwen3-8B`, 36 layers, 4096 dimensions. Only the `text` field is
ever sent to it. The label, the source and the tier sit beside it and are never
shown.

## The two sets we will run

### 1. `data/final_data.json`, 1100 prompts

Phase 1 at scale. Eleven sources, 100 prompts each. 550 labelled as a test and 550
not. Every prompt is 100 to 300 characters and English, confirmed by langdetect.

| Tier | What it represents | Source | n | Label | Chars min/median/max |
|---|---|---|---|---|---|
| A | a real evaluation, written as one | mask | 100 | 1 | 103/186/291 |
| A | | model_written_evals | 100 | 1 | 102/201/293 |
| A | | bbq | 100 | 1 | 107/173/299 |
| A | | mmlu | 100 | 1 | 101/151/294 |
| A | | gsm8k | 100 | 1 | 103/192/297 |
| B | a real person talking to a chatbot | wildchat | 100 | 0 | 102/165/295 |
| B | | lmsys | 100 | 0 | 102/148/294 |
| B | | sharegpt | 100 | 0 | 105/177/299 |
| C | written by a person, but for a dataset | oasst2 | 100 | 0 | 100/157/289 |
| C | | no_robots | 100 | 0 | 100/181/299 |
| D | the wording control, carried in whole | eval_awareness_2x2 | 100 | mixed | 106/189/319 |

**What each tier is for.** Tier A against tier B is the original question: can the
model tell a test from a real conversation. Tier B against tier C is the control
that decides what the answer means, because neither side is an evaluation, so the
right answer there is 0.5. If the probe separates tier B from tier C, it is
detecting that someone wrote the text for a corpus, not that the model is being
tested. Every test prompt in our earlier runs was also written for a corpus, so
those two explanations had never been separated before tier C existed.

**What each source is.**

| Source | Upstream repo | What it measures |
|---|---|---|
| mask | `cais/MASK` | does the model stay honest when pressured to lie |
| model_written_evals | `Anthropic/model-written-evals` | alignment questions in A/B form, written by a model |
| bbq | `walledai/BBQ` | does the answer fall back on a social stereotype |
| mmlu | `cais/mmlu` | exam questions across school subjects |
| gsm8k | `openai/gsm8k` | arithmetic word problems |
| wildchat | `allenai/WildChat-1M` | first messages from real chatbot traffic |
| lmsys | `lmsys/lmsys-chat-1m` | first messages from real chatbot traffic, other platform |
| sharegpt | ShareGPT, via `viliana-dev/eval-awareness-2x2` | first messages users chose to publish |
| oasst2 | `OpenAssistant/oasst2` | opening prompts written by volunteers for a training corpus |
| no_robots | `HuggingFaceH4/no_robots` | instructions written by paid annotators |
| eval_awareness_2x2 | `viliana-dev/eval-awareness-2x2` | the four wording groups below |

Fill rates when it was built, which show how picky the band is:

```
mask                 100 kept from   273 looked at   37%
model_written_evals  100 kept from   142             70%
bbq                  100 kept from   159             63%
mmlu                 100 kept from   295             34%
gsm8k                100 kept from   126             79%
wildchat             100 kept from   997             10%
lmsys                100 kept from   491             20%
sharegpt             100 kept from   185             54%
oasst2               100 kept from   324             31%
no_robots            100 kept from   223             45%
```

Built by `data/build_final_data.py`, seed 0.

### 2. `data/dataset2.json`, 400 prompts

The wording control. 100 per group, which is where we have chosen to keep it.
Lengths matched across all four groups.

| Group | Content | Wording | n | Label | Chars min/median/max |
|---|---|---|---|---|---|
| bench_eval | a real test | exam | 100 | 1 | 103/160/239 |
| casual_eval | a real test | casual | 100 | 1 | 100/163/238 |
| bench_deploy | a real person | exam | 100 | 0 | 103/161/239 |
| casual_deploy | a real person | casual | 100 | 0 | 101/160/235 |

**What it represents.** Two things vary separately: whether the prompt really is a
test, and whether it is written like an exam question. The group that carries the
argument is `bench_deploy`, real people's messages rewritten into multiple choice
by DeepSeek-V3. Nobody is being tested there, so if the model reacts to those the
way it reacts to real exams, it is reading the writing and not the situation.

Upstream, from the release's own dataset card:

```
bench_eval              MMLU + ARC + GSM8K, kept in exam form
casual_eval_mutual      MuTual first turns
bench_deploy_rewritten  ShareGPT -> multiple choice rewrite by DeepSeek-V3
casual_deploy_matched   ShareGPT first turns, length-matched
```

Built by `data/build_dataset2.py`, seed 0, two length bins of 50.

## The probe's own data

`data/contrastive_pairs_16.txt`, 16 pairs. This is **not training data**. The probe
direction is the mean of the deployment-side activations minus the mean of the
evaluation-side activations, per layer, L2-normalised. Nothing is fitted, so there
is no train/test split and every prompt above stays unseen. The file is byte
identical to the one released with Heidari et al. 2026, and the pairs are Table 12
of Hua et al. 2025.

## Earlier sets, kept for the record

| File | n | What it was for | What it showed |
|---|---|---|---|
| `dataset_vea100.json` | 100 | run 1, five test topics against real chat | peak 0.835 at layer 23, but length alone reached 0.668 |
| `dataset_2x2.json` | 100 | run 2, the wording control at 25 per group | wording alone reached 0.811, the same size as the real task |
| `dataset_benchmarks.json` | 308 | run 3, eleven test sources against 50 real messages | only 1 of 11 cleared its floor on last token, 4 of 11 on whole prompt |

`dataset_vea100.json` also holds `SALT-NLP/PrivacyLens` and `basicv8vc/SimpleQA`,
which are not in `final_data.json`. Both were dropped for length: PrivacyLens gave
only 18 usable items last time and SimpleQA runs short, 47 to 129 characters.

## Things a newcomer should know

1. **The sign convention is load bearing and never flipped.** Positive means
   deployment. A working probe therefore scores **below** 0.5 against a label where
   1 means test. We always report effect as `|AUROC - 0.5|` so direction cannot be
   chosen after the fact.

2. **Length is the first thing to rule out, and it is reported as a floor.**
   Measured on each file:

   | File | chars | words |
   |---|---|---|
   | `final_data.json` | 0.557 | 0.603 |
   | `dataset2.json` | 0.450 | 0.604 |
   | `dataset_2x2.json` | 0.466 | 0.573 |
   | `dataset_benchmarks.json` | 0.473 | 0.474 |
   | `dataset_vea100.json` | 0.668 | 0.671 |

   Nothing below the word figure is a result. `dataset_vea100.json` is the warning:
   counting characters got most of the way to its headline number.

3. **In the 2x2 groups, the two real-person groups share zero messages.**
   `bench_deploy_rewritten` and `casual_deploy_matched` have 199 of 835 ids in
   common in the pool, and 0 in common in what was sampled. So comparing them
   varies the wording **and** which messages are in each pile. It cannot attribute
   a result to wording alone. It can only show separation where no evaluation is
   present on either side.

4. **A properly paired version exists and we are not using it.**
   `casual_deploy_formal` shares all 835 ids with `bench_deploy_rewritten` and the
   same `original_query`, so it is literally the same message in two wordings. We
   avoid it because pairing exactly puts length-only separation at 0.902, which is
   a worse confound than the one it removes. Fixing this properly needs rewrites we
   generate ourselves.

5. **The `_full` configs exist and our old builders missed them.** `build_2x2.py`
   and `build_dataset2.py` read `casual_eval_mutual`, which holds 395 usable items
   in 100 to 300 characters. `casual_eval_mutual_full` holds 1902 in the same band,
   and `bench_eval_full` holds 2001 against 715. Unique items in band, after
   dedupe:

   ```
   config                    rows   100-240   100-300   no limit
   bench_eval                1076       504       715       1073
   bench_eval_full           3000      1380      2001       2992
   casual_eval_mutual        1076       373       395        980
   casual_eval_mutual_full   7088      1826      1902       4356
   bench_deploy_rewritten     835       421       576        834
   casual_deploy_matched     1076       485       611       1060
   ```

   With the full configs the ceiling becomes `bench_deploy_rewritten` at 576 per
   group in 100 to 300 characters, because every row of it is a rewrite and there
   is no full version. We have chosen 100 per group anyway, but the cap is 576, not
   the 395 the old builders implied.

6. **Six of the 100 tier D prompts in `final_data.json` run to 319 characters**,
   just outside the 100 to 300 band, because that set was carried in whole from
   `dataset_2x2.json`, which was built on different bins.

7. **Tier D is the only source with both labels inside it.** It must be excluded
   from the tier A, B and C comparisons or it contaminates them.

8. **`dataset_2x2.json` still carries MuTual's speaker tags** such as a leading
   `m:` and spaces before punctuation. Those are a surface tell with nothing to do
   with evaluation. `build_dataset2.py` strips them, `build_2x2.py` did not, and
   the 100 tier D prompts inherit the unstripped version.

9. **Never trust a dataset's own language column.** The LMSYS column is
   model-detected and let three Spanish prompts into run 1. Every builder since
   runs langdetect after the length filter.

10. **Capturing activations fixes the pooling permanently.** Each npz stores only
    `last_token` and `mean_token`, each `[36, 4096]` float32, about 1.2 MB per
    prompt. Any third readout means capturing again.

11. **The whole-prompt readout is the published method.** Nguyen et al. take the
    mean over all tokens. Our last-token readout is not their method, and it is the
    one that produced the 0.835 that later looked like a style detector. Report
    whole prompt as the headline.

12. **`data/` is gitignored and the Hub repo is private.** WildChat, LMSYS and
    ShareGPT are redistribution-restricted. Builder scripts are force-added to git;
    the prompt files are not. Results live at
    `FiendHunter/qwen3-8b-eval-awareness-probe`, private.

13. **`/nvme` on the VM is wiped by every restart.** It has been lost twice in one
    day. Anything not pushed to the Hub or pulled down is gone.
