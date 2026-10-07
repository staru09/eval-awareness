"""Build final_data.json: 100 prompts from each of 11 sources, 1100 total.

Four tiers, because the question needs more than two sides:

  A  test sets          a real evaluation, written as one
  B  real users         somebody talking to a chatbot
  C  authored as data   written by a person, but for a dataset, not to get help
  D  format controlled  the 2x2 set, where content and wording vary separately

Tier C is the control phase 1 never had. Nobody is tested there and nobody is
casually chatting, so if the probe reacts to a prompt having been authored for a
dataset it lands with tier A, and if it reacts to being tested it lands with
tier B.

Every prompt is 100 to 300 characters and English by langdetect, never by a
dataset's own language column: the LMSYS column is model-detected and leaked
Spanish into our first run.

Prints how many items each source has in the band before it samples, so an
under-filled source is visible rather than silent.

Writes data/final_data.json.
"""
import glob
import gzip
import json
import random
import re
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

HUB = Path.home() / ".cache" / "huggingface" / "hub"
HERE = Path(__file__).resolve().parent
OUT = HERE / "final_data.json"
LO, HI = 100, 300
PER_SOURCE = 100
SEED = 0

SPACE_PUNCT = re.compile(r"\s+([,.!?;:'])")
CONTRACTION = re.compile(r"\b(\w+)\s+n't\b")


def snap(repo):
    hits = glob.glob(str(HUB / f"datasets--{repo.replace('/', '--')}" / "snapshots" / "*"))
    if not hits:
        raise SystemExit(f"{repo} not cached; hf download --repo-type dataset {repo}")
    return hits[0]


def clean(text):
    """Strip tokenisation artefacts only, never a source's framing.

    The framing is the variable under study. Pre-tokenised spacing and speaker
    tags are not, so they go.
    """
    t = re.sub(r"\s+", " ", str(text).replace("\n", " "))
    t = CONTRACTION.sub(r"\1n't", t)
    return SPACE_PUNCT.sub(r"\1", t).strip()


def english(t):
    if not (LO <= len(t) < HI) or " " not in t or "<<" in t:
        return False
    try:
        return detect(t) == "en"
    except Exception:
        return False


def first_user(messages):
    for m in messages:
        if m.get("role") == "user":
            return m.get("content")
    return None


# ---- one reader per source, each returning a list of raw strings ----

def read_parquets(repo, pattern, fn):
    out = []
    for f in sorted(glob.glob(f"{snap(repo)}/{pattern}")):
        out += fn(pd.read_parquet(f))
    return out


def mask():
    return read_parquets("cais/MASK", "*/*.parquet", lambda d: list(d["user_prompt"]))


def model_written_evals():
    out = []
    for f in sorted(glob.glob(f"{snap('Anthropic/model-written-evals')}"
                              "/advanced-ai-risk/human_generated_evals/*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            out += [json.loads(line)["question"] for line in fh if line.strip()]
    return out


def bbq():
    return read_parquets("walledai/BBQ", "data/*.parquet",
                         lambda d: [f"{c} {q}" for c, q in zip(d["context"], d["question"])])


def mmlu():
    return read_parquets("cais/mmlu", "all/test-*.parquet", lambda d: list(d["question"]))


def gsm8k():
    return read_parquets("openai/gsm8k", "main/*.parquet", lambda d: list(d["question"]))


def wildchat():
    return read_parquets("allenai/WildChat-1M", "data/train-00000-of-00014.parquet",
                         lambda d: [c[0]["content"] for c in d["conversation"] if len(c)])


def lmsys():
    return read_parquets("lmsys/lmsys-chat-1m", "data/train-00000-of-00006-*.parquet",
                         lambda d: [c[0]["content"] for c in d["conversation"] if len(c)])


def sharegpt():
    """ShareGPT first turns, taken from the 2x2 release that already extracted them."""
    return read_parquets("viliana-dev/eval-awareness-2x2", "casual_deploy_matched/*.parquet",
                         lambda d: list(d["first_turn"]))


def oasst2():
    """Opening prompter turns only: a reply is a different kind of text."""
    out = []
    for f in sorted(glob.glob(f"{snap('OpenAssistant/oasst2')}/*all.messages.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("role") == "prompter" and not r.get("parent_id") and r.get("lang") == "en":
                    out.append(r["text"])
    return out


def no_robots():
    """The `prompt` column is often the system line, so read the user turn."""
    return read_parquets("HuggingFaceH4/no_robots", "data/*.parquet",
                         lambda d: [first_user(m) for m in d["messages"]])


SOURCES = [
    # name                 tier  label  reader
    ("mask",                 "A", 1, mask),
    ("model_written_evals",  "A", 1, model_written_evals),
    ("bbq",                  "A", 1, bbq),
    ("mmlu",                 "A", 1, mmlu),
    ("gsm8k",                "A", 1, gsm8k),
    ("wildchat",             "B", 0, wildchat),
    ("lmsys",                "B", 0, lmsys),
    ("sharegpt",             "B", 0, sharegpt),
    ("oasst2",               "C", 0, oasst2),
    ("no_robots",            "C", 0, no_robots),
]


def twox2():
    """The existing 100-prompt format control, carried in whole with its labels."""
    rows = json.loads((HERE / "dataset_2x2.json").read_text(encoding="utf-8"))
    return [(clean(r["text"]), r["label"], r["source"]) for r in rows]


def main():
    rng = random.Random(SEED)
    rows, short = [], []
    # One prompt, one place. sharegpt and the 2x2 casual_deploy group both come
    # from casual_deploy_matched, and without this four messages landed in tier B
    # and tier D at once. Seeded with tier D because that set is carried in whole.
    taken = {t.lower() for t, _, _ in twox2()}

    for name, tier, label, reader in SOURCES:
        # Shuffle first and stop at PER_SOURCE. Running the language detector over
        # every row costs minutes on LMSYS at 166k rows, for 100 kept prompts.
        raw_rows = [r for r in reader() if r is not None]
        rng.shuffle(raw_rows)
        pool, seen, looked = [], set(), 0
        for raw in raw_rows:
            looked += 1
            t = clean(raw)
            if t.lower() in seen or t.lower() in taken or not english(t):
                continue
            seen.add(t.lower())
            pool.append(t)
            if len(pool) >= PER_SOURCE:
                break
        rate = f"{len(pool) / looked:.0%}" if looked else "n/a"
        print(f"  {name:21s} tier {tier}  {len(pool):4d} kept from {looked:6d} looked at "
              f"({rate} in {LO}-{HI} chars and English), pool {len(raw_rows)}", end="")
        picked = pool[:PER_SOURCE]
        if len(picked) < PER_SOURCE:
            short.append(f"{name}: {len(picked)} of {PER_SOURCE}")
            print("   UNDER-FILLED")
        else:
            print("")
        taken.update(t.lower() for t in picked)
        for i, t in enumerate(picked):
            rows.append({"id": f"{name}:{i:03d}", "text": t, "label": label,
                         "source": name, "tier": tier, "group": name, "n_chars": len(t)})

    # the 2x2 set keeps its own four groups and its own per-row labels
    pairs = twox2()
    print(f"  {'eval_awareness_2x2':21s} tier D  {len(pairs):6d} carried in whole")
    for i, (t, label, group) in enumerate(pairs):
        rows.append({"id": f"eval_awareness_2x2:{i:03d}", "text": t, "label": label,
                     "source": "eval_awareness_2x2", "tier": "D", "group": group,
                     "n_chars": len(t)})

    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["prompt_id"] = i
    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {OUT}: {len(rows)} prompts, {sum(r['label'] for r in rows)} labelled test")
    for tier in "ABCD":
        sub = [r for r in rows if r["tier"] == tier]
        lens = sorted(r["n_chars"] for r in sub)
        if sub:
            print(f"  tier {tier}  n={len(sub):4d}  chars {lens[0]}/{lens[len(lens) // 2]}/{lens[-1]}")
    if short:
        print("\nUNDER-FILLED:", "; ".join(short))


if __name__ == "__main__":
    main()
