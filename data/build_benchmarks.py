"""One length-matched set holding every group we want to compare.

Eleven evaluation groups, 25 prompts each, plus 50 real user messages as the
shared reference:

  technical   gsm8k, mmlu, arc, hellaswag, truthfulqa, humaneval
  safety      bbq, mask, simpleqa, model_written_evals, privacylens
  deployment  real user first turns

Two deliberate choices. Lengths are matched across every group by sampling the
same count from three length bins, because length alone reached 0.675 on our
first set. And no benchmark scaffolding is added: the multiple-choice items go
in as their bare question stem, with no "Question:/Choices:/Answer:" wrapper,
because the 2x2 run showed that wrapper alone drives the last-token readout.
What varies here is subject matter, not form.
"""
import glob
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0
HUB = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")) + "/hub"
OUT = Path(__file__).resolve().parent / "dataset_benchmarks.json"
BINS = [(100, 160), (160, 240), (240, 320)]
PER_BIN, PER_GROUP, DEPLOY_N, SEED = 9, 25, 50, 0


def snap(repo):
    return glob.glob(f"{HUB}/datasets--{repo}/snapshots/*")[0]


def clean(t):
    return re.sub(r"\s+", " ", str(t)).strip()


def usable(t):
    if "<<" in t or ">>" in t:
        return False
    try:
        return detect(t) == "en"
    except Exception:
        return False


def read(repo, pattern):
    return pd.read_parquet(sorted(glob.glob(f"{snap(repo)}/{pattern}"))[0])


# --------------------------------------------------------- technical benchmarks

def gsm8k():
    return [clean(q) for q in read("openai--gsm8k", "main/test-*.parquet")["question"]]


def mmlu():
    return [clean(q) for q in read("cais--mmlu", "all/test-*.parquet")["question"]]


def arc():
    return [clean(q) for q in read("allenai--ai2_arc", "ARC-Challenge/test-*.parquet")["question"]]


def hellaswag():
    return [clean(c) for c in read("Rowan--hellaswag", "data/validation-*.parquet")["ctx"]]


def truthfulqa():
    return [clean(q) for q in read("truthfulqa--truthful_qa",
                                   "generation/validation-*.parquet")["question"]]


def humaneval():
    """Function signature plus docstring. Newlines collapse, so this reads as
    one long line of code; flagged in the writeup rather than hidden."""
    return [clean(p) for p in read("openai--openai_humaneval",
                                   "openai_humaneval/test-*.parquet")["prompt"]]


# ------------------------------------------------------------- safety behaviour

def bbq():
    out = []
    for f in sorted(glob.glob(f"{snap('walledai--BBQ')}/data/*.parquet")):
        df = pd.read_parquet(f)
        out += [clean(f"{r['context']} {r['question']}") for _, r in df.iterrows()]
    return out


def mask():
    out = []
    for f in sorted(glob.glob(f"{snap('cais--MASK')}/*/*.parquet")):
        out += [clean(p) for p in pd.read_parquet(f)["user_prompt"]]
    return out


def simpleqa():
    df = pd.read_csv(f"{snap('basicv8vc--SimpleQA')}/simple_qa_test_set.csv")
    return [clean(p) for p in df["problem"]]


def model_written_evals():
    out = []
    for path in sorted(glob.glob(f"{snap('Anthropic--model-written-evals')}"
                                 "/advanced-ai-risk/human_generated_evals/*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            out += [clean(json.loads(line)["question"]) for line in fh]
    return out


def privacylens():
    rows = json.loads(Path(f"{snap('SALT-NLP--PrivacyLens')}/main_data.json")
                      .read_text(encoding="utf-8"))
    return [clean(r["trajectory"].get("user_instruction", "")) for r in rows]


def deployment():
    df = read("viliana-dev--eval-awareness-2x2", "casual_deploy_matched/*.parquet")
    return [clean(t) for t in df["first_turn"]]


GROUPS = [
    ("gsm8k", "technical", gsm8k), ("mmlu", "technical", mmlu),
    ("arc", "technical", arc), ("hellaswag", "technical", hellaswag),
    ("truthfulqa", "technical", truthfulqa), ("humaneval", "technical", humaneval),
    ("bbq", "safety", bbq), ("mask", "safety", mask),
    ("simpleqa", "safety", simpleqa), ("model_written_evals", "safety", model_written_evals),
    ("privacylens", "safety", privacylens),
]


def matched_sample(texts, rng, per_bin, cap):
    seen, pool = set(), []
    for t in texts:
        k = t.lower()
        if k in seen or not (BINS[0][0] <= len(t) <= BINS[-1][1]) or not usable(t):
            continue
        seen.add(k)
        pool.append(t)
    picked = []
    for lo, hi in BINS:
        band = [t for t in pool if lo <= len(t) < hi]
        rng.shuffle(band)
        picked += band[:per_bin]
    rng.shuffle(picked)
    return picked[:cap], len(pool)


def main():
    rng = random.Random(SEED)
    rows = []
    for name, family, fn in GROUPS:
        picked, n_pool = matched_sample(fn(), rng, PER_BIN, PER_GROUP)
        for i, t in enumerate(picked):
            rows.append({"id": f"{name}:{i:03d}", "text": t, "label": 1, "source": name,
                         "category": family, "n_chars": len(t)})
        print(f"  {name:20s} {family:9s} pool {n_pool:6d} -> took {len(picked)}")

    picked, n_pool = matched_sample(deployment(), rng, DEPLOY_N // len(BINS) + 1, DEPLOY_N)
    for i, t in enumerate(picked):
        rows.append({"id": f"deployment:{i:03d}", "text": t, "label": 0,
                     "source": "deployment", "category": "deployment", "n_chars": len(t)})
    print(f"  {'deployment':20s} {'real':9s} pool {n_pool:6d} -> took {len(picked)}")

    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["prompt_id"] = i
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {OUT}: {len(rows)} rows, {sum(r['label'] for r in rows)} eval")
    for name in [g[0] for g in GROUPS] + ["deployment"]:
        lens = sorted(r["n_chars"] for r in rows if r["source"] == name)
        print(f"  {name:20s} n={len(lens):2d}  median {lens[len(lens) // 2]} chars")


if __name__ == "__main__":
    main()
