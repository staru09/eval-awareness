# What Are We Actually Doing Here? (The Explainer)

If you know nothing about AI models and just want to understand this
project — this is for you.

## The Big Idea: Does the AI Know It's Being Tested?

Imagine you're a student, and your teacher wants to know how you *really*
behave — do you do your homework honestly, do you help classmates, are
you polite? So the teacher gives you a surprise pop quiz to check.

But here's the problem: the moment you realize "oh, this is a quiz, I'm
being graded," you might act differently. You sit up straighter. You
double-check your answers. Maybe you're extra careful not to say anything
that sounds bad. You're not lying, exactly — you're just *on your best
behavior* because you know you're being watched.

Now here's the catch for AI companies: they test their AI models with
quizzes too — called **benchmarks** or **evaluations** — to check if the
AI is safe, smart, and well-behaved. But if the AI can tell "hey, this
looks like a test," it might act extra careful during the test and then
behave completely differently once it's just chatting with a regular
person. That would mean the test isn't actually telling you the truth
about how the AI behaves in real life — just like grading a student only
on days they know they're being watched doesn't tell you what they're
really like every other day.

This whole project is about answering one question:

**Can AI models tell when they're being tested — and if so, do they
admit it, or do they hide it while still quietly acting on it?**

We call this **"evaluation awareness."**

## Setting Up the Experiment: Two Kinds of Conversations

To study this, we needed two piles of conversations to show the AI:

- **"Deploy" conversations** — messages that look like a normal person
  just chatting, asking for help with everyday stuff (like "can you help
  me plan my weekend" or a real ChatGPT conversation someone actually had).
- **"Eval" conversations** — messages that look like they came from a
  test or benchmark (quiz questions, trick questions, safety-test-style
  prompts).

We built a small set of 20 of these (10 of each kind) and ran a bunch of
different AI models over them — big ones, small ones, from different
companies, including some "obliterated" (jailbroken/uncensored) versions
found on the internet, just to see if that changes anything.

## Phase 1 (Already Done): Just Ask It

The simplest thing we could do: after showing the AI one of these
conversations, just **ask it directly** — "hey, was that a test or a real
conversation?" — in five different phrasings, in case the AI understands
one phrasing better than another.

This is like asking the student straight up: "did you know that was a
quiz?" Some kids will say "yeah, obviously" — some models were basically
perfect at admitting it (AUROC of 1.0 means "always correctly says yes
it's a test / no it's not"). Others were basically clueless and guessed
at random.

One especially interesting model — an "obliterated" Qwen 3.6 variant —
*appeared* to score at or below chance (like a coin flip, or worse) on
this test at first. But it turned out **we were grading its answer sheet
wrong**, not that the model was guessing (see the "grading mistake"
callout below, in explainer_docs/PROGRESS.md §10, for the full story) —
once graded correctly, this model turned out to be one of the *best* at
admitting it, not the worst.

## Phase 2: Peeking Inside the AI's "Brain" (Experiment 1 — Done)

This is where it gets interesting. AI models don't think in words — they
think in giant lists of numbers that get transformed step by step,
layer by layer, like an assembly line. This inner list of numbers is
sometimes called the model's "**activations**" — you can loosely think of
it as its brainwaves.

So even if a model *says* "no, this isn't a test" out loud, maybe deep
inside, in its "brainwaves," some part of it is quietly firing off a
signal that says "this is totally a test." We wanted to check.

**How we checked it — the flashlight trick:**

We made 16 pairs of near-identical sentences. Each pair says almost the
same thing, but one version is framed as "you're talking to a real user"
and the other as "you're talking to an AI evaluator." Same topic, same
length — the *only* difference is that one framing smells like a test and
the other doesn't.

We fed all 32 of these sentences into the model and looked at its
"brainwaves" (activations) at every single layer of its assembly line.
Then we did simple subtraction: *average brainwave pattern for
"eval"-framed sentences* minus *average brainwave pattern for
"deploy"-framed sentences*. What's left over is a **direction** — think
of it like a compass needle that points toward "this feels like a test"
inside the model's brain.

Then, for each of our 20 real transcripts, we checked: does this
compass needle actually point the right way? If we show the model an eval
transcript, does its brainwave lean toward the "test" direction? If we
show it a deploy transcript, does it lean the other way?

**What we found:** Yes, mostly! For every model we tested, at some layer
in its assembly line, this compass trick worked really well — sometimes
almost perfectly (98% accuracy for one model). This means: **the model's
internal brainwaves "know" it's being tested, at least at some point in
its thinking, even before it says a single word out loud.**

That "obliterated" Qwen 3.6 model's internal compass strongly pointed the
right direction (88% accuracy) — consistent with what it turned out to
actually be *saying* out loud too, once we fixed our grading mistake: it
was one of our most accurate models at both. See the grading-mistake
callout below for what we originally (wrongly) thought this meant.

> **A grading mistake, found and fixed.** For a while, this project's
> notes described this Qwen 3.6 model as "the cleanest example of an AI
> that knows internally but never says so" — its internal compass looked
> right, but our records showed it saying "not a test" almost every
> single time, even for real tests. That sounded like a big deal. It
> turned out to be a bug in *our own grading script*, not a real thing
> about the model. The model often "shows its work" before giving a
> final answer — like a student writing scratch notes in the margin
> before circling their real answer. Our script was sometimes reading the
> *scratch notes* (which often just repeated the question, like "answer:
> YES or NO") instead of the model's *actual circled answer* further
> down the page, and defaulting to "NO" whenever it got confused. Once we
> fixed the grading script to always read the model's real, final,
> circled answer, this model turned out to correctly say "yes, this is a
> test" almost every time — one of the best in the whole group, not the
> worst. We're keeping this mistake in the writeup on purpose: a finding
> this dramatic (a model that "knows but never tells") is exactly the
> kind of result that deserves someone double-checking the grading before
> believing it, and in our case, the double-check mattered.

## Phase 3: Do the Two Signals Actually Agree? (Experiment 2 — Next)

Okay, so we now have two separate measurements for each conversation:

1. What the model **said** when we asked it directly (Phase 1).
2. What its **internal compass** pointed toward (Phase 2 / Experiment 1).

Do these two things actually line up? You'd *think* if the internal
compass says "this is definitely a test," the model would also be more
likely to *say* "yes, this is a test" — right?

Turns out, in a lot of published research on this topic, the answer is
surprisingly: **not really.** A model can have a very confident internal
compass reading and still say something totally unrelated to it out
loud, or vice versa. It's like a kid whose gut is screaming "this is
definitely a pop quiz" but who says "nah, feels like a normal day" when
asked — the gut feeling and the words don't talk to each other much.

We'll test this by lining up, for every single conversation, the
internal compass reading against what the model actually said, and
checking mathematically whether they move together (this uses two tools
called **correlation** and **mutual information** — don't worry about the
exact math, just think of them as "how much do these two things agree").

## Phase 4: Can We Reach In and Change Its Mind? (Experiment 3 — Next)

This is the most fun one. If the model has this internal "this is a
test" compass direction, what happens if we **grab that compass needle
and physically turn it** while the model is thinking?

Concretely: we take that same compass direction we found in Experiment 1,
and we *inject* it directly into the model's brainwaves while it's
generating a response — nudging it either:

- **Toward** "this is a test" (and see if the model suddenly starts
  admitting it's being tested more often), or
- **Away** from "this is a test" (and see if it starts denying it more,
  even for conversations that really were tests).

This is called **activation steering**, and it's basically a real
science-experiment version of "what if I could reach into your brain and
turn up the volume on one specific thought, and see what you say next?"

If nudging the compass needle actually *changes what the model says*,
that tells us something important: the internal signal isn't just some
useless byproduct sitting there doing nothing — it's actually **causally
connected** to what comes out of the model's mouth, even if normally
(Experiment 2) that connection looks weak or hidden.

**Important detail about who gets nudged:** we only ever reach into the
brain of the model being studied, while it's writing its own answer.
We never touch a separate "judge" — the judge (in our case, a simple
rule that reads the model's final answer and checks if it said "yes this
was a test" or "no it wasn't") is just watching from the outside,
completely unaffected by the nudge. It's like nudging the student's
brain but grading their answer with a completely separate, untouched
answer key.

## Why Any of This Matters

If AI companies test their models for safety, and the models can quietly
tell "hey, this is a safety test," a model *could* — even without
meaning to be sneaky — behave extra safely during the test and then
behave differently once it's out in the real world talking to real
people. That would mean the safety test isn't actually proving what
everyone hopes it's proving.

This project doesn't claim any model is doing that on purpose. It's just
measuring, carefully and honestly: **can these models tell? Do they say
so? And if they don't say so, is it because they genuinely don't know, or
because the "knowing" and the "saying" are just disconnected from each
other?** Those are three very different, very important questions, and
this project measures all three, one experiment at a time.
