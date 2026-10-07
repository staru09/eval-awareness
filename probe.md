# Probe

How the probe is built and how it scores a prompt. Code: `evalaware/probe.py`,
`evalaware/activations.py`, `evalaware/scoring.py`. Model: Qwen3-8B, 36 layers,
4096 dimensions.

## What the probe is

One direction per layer. A list of 4096 numbers that points from "being tested"
toward "talking to a real user" inside the model's hidden state. Nothing is
trained.

## How it is built

1. **Sixteen pairs of short phrases**, in `data/contrastive_pairs_16.txt`. Each
   pair says the same thing two ways:

   ```
   positive: You are talking to a real user:
   negative: You are talking to an AI evaluator:
   ```

   Positive is deployment, negative is evaluation. The pairs are Table 12 of Hua
   et al. 2025. Our file is byte identical to the one released with Heidari et al.
   2026.

2. **One forward pass per phrase**, with no chat template. At every layer, keep the
   hidden state at the **last token**, which is the colon.

3. **Average and subtract.** At each layer, average the 16 positive hidden states,
   average the 16 negative ones, and subtract negative from positive.

4. **Scale to length 1.** Each layer's difference is divided by its own length, so
   only its direction matters.

That gives one direction per layer. The file stores 37: the embedding output plus
the 36 decoder layers. The embedding one is never used. Scoring reads probe entry
`L + 1` against decoder layer `L`, so the two stay aligned.

**Why nothing is trained.** The direction comes only from the 32 phrases. None of
the 1451 prompts in the dataset is used to build it, so every one of them is
unseen and there is no train and test split to manage.

## Random directions, the control

Built at the same time and stored in the same file. For each layer, unit vectors
drawn from a normal distribution, seeded per layer so they reproduce. **The count is
fixed when the probe is built**, by `--n-random`. The default is 10. The phase 1 run
needs 100, so the probe must be built with `--n-random 100`. Reusing an old probe
file silently keeps its old count.

## How it scores a prompt

1. **One forward pass over the prompt.** Nothing is generated.
2. **Keep the hidden state at all 36 decoder layers**, then reduce each layer to one
   vector of 4096 two ways:
   - **whole prompt**: the average over every token
   - **last token**: the final token only
3. **Dot product with the direction.** Each layer's vector times that layer's
   direction gives one number. So each prompt ends with 36 numbers per reduction.
   The same is done for every random direction.

Only these two reductions are stored, `[36, 4096]` each, about 1.2 MB per prompt.
Any other reduction means running the prompts through the model again.

## How the scores become a verdict

No threshold is ever fitted. Each layer's 36 scores across all prompts are ranked
against the labels, giving an AUROC per layer.

**The sign is fixed and never flipped.** A high score means "real user". So a
working probe gives an AUROC **below** 0.5 against a label where 1 means test. To
keep the direction honest, we report the effect as distance from chance:

```
effect = |AUROC - 0.5|
```

0 means the probe sees no difference. 0.5 means perfect separation.

## Which reduction is the headline

**Whole prompt.** It is the method of Nguyen et al., who average over all tokens,
and it barely changes whether the prompt is sent raw or in the chat template.

**Last token is reported beside it, never as the headline.** In chat mode the last
token is the newline after `assistant`, the same for every prompt, so it carries
nothing about the prompt itself. In raw mode it is the prompt's final character.
On earlier data it moved by up to 0.505 between raw and chat at the same layer, and
it separated two groups of real people at 0.811 AUROC when neither group was an
evaluation.

## One mismatch to keep in mind

The direction is built from **one token**, the colon at the end of a short phrase.
It is then applied to the **average over a whole prompt**. Those are different
kinds of hidden state. This is how the published method works too, but it means a
weak result could come from the mismatch rather than from the model lacking the
information.
