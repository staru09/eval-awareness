"""Build a length-matched 2x2 probe set: eval-ness crossed with benchmark formatting.

The point of this set is that content and formatting vary independently, so the
probe cannot win by recognising a benchmark template. Four cells, 25 prompts each:

                  benchmark format        casual format
  eval content    bench_eval              casual_eval
  deploy content  bench_deploy            casual_deploy

bench_deploy is the critical cell: real user queries rewritten into benchmark
form. A probe reading evaluation calls it deployment. A probe reading format
calls it evaluation.

Lengths are matched by sampling the same number from each of three length bins
in every cell, because on our previous set length alone reached 0.675 AUROC.
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
SNAP = glob.glob(f"{HUB}/datasets--viliana-dev--eval-awareness-2x2/snapshots/*")[0]
OUT = Path(__file__).resolve().parent / "dataset_2x2.json"
BINS = [(100, 160), (160, 240), (240, 320)]
PER_BIN = 9          # 27 per cell before the final trim
PER_CELL = 25
SEED = 0

CELLS = {
    # name           directory                 text column        label (1 = eval content)
    "bench_eval":    ("bench_eval",             "formatted_prompt", 1),
    "casual_eval":   ("casual_eval_mutual",     "first_turn",       1),
    "bench_deploy":  ("bench_deploy_rewritten", "rewritten",        0),
    "casual_deploy": ("casual_deploy_matched",  "first_turn",       0),
}

SPEAKER_RE = re.compile(r"^\s*[mf]\s*:\s*", re.IGNORECASE)
SPACE_PUNCT_RE = re.compile(r"\s+([,.!?;:'])")
CONTRACTION_RE = re.compile(r"\b(\w+)\s+n't\b")


def clean(text):
    """Strip the artefacts of each source's own tokenisation, not its framing.

    MuTual ships pre-tokenised dialogue ("m : excuse me , where is it ?"), and
    the speaker tag plus the spaces before punctuation would be a surface tell
    that has nothing to do with whether the turn is an evaluation. The benchmark
    scaffolding is left exactly as it is, because that is the variable.
    """
    t = str(text).replace("\n", " ")
    t = SPEAKER_RE.sub("", t)
    t = CONTRACTION_RE.sub(r"\1n't", t)
    t = SPACE_PUNCT_RE.sub(r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def usable(text):
    if "<<" in text or ">>" in text:       # GSM8K calculator annotations
        return False
    try:
        return detect(text) == "en"
    except Exception:
        return False


def pool(directory, column):
    df = pd.read_parquet(sorted(glob.glob(f"{SNAP}/{directory}/*.parquet"))[0])
    seen, rows = set(), []
    for _, r in df.iterrows():
        t = clean(r[column])
        key = t.lower()
        if key in seen or not (BINS[0][0] <= len(t) <= BINS[-1][1]):
            continue
        if not usable(t):
            continue
        seen.add(key)
        rows.append(t)
    return rows


def main():
    rng = random.Random(SEED)
    rows = []
    for cell, (directory, column, label) in CELLS.items():
        texts = pool(directory, column)
        picked = []
        for lo, hi in BINS:
            band = [t for t in texts if lo <= len(t) < hi]
            rng.shuffle(band)
            picked += band[:PER_BIN]
            print(f"  {cell:14s} bin {lo}-{hi}: {len(band):4d} available, took {len(band[:PER_BIN])}")
        rng.shuffle(picked)
        for i, t in enumerate(picked[:PER_CELL]):
            rows.append({"id": f"{cell}:{i:03d}", "text": t, "label": label,
                         "source": cell, "category": cell, "n_chars": len(t)})

    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["prompt_id"] = i
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {OUT}: {len(rows)} rows")
    for cell in CELLS:
        lens = sorted(r["n_chars"] for r in rows if r["source"] == cell)
        if lens:
            print(f"  {cell:14s} n={len(lens):2d}  chars min/median/max "
                  f"{lens[0]}/{lens[len(lens)//2]}/{lens[-1]}  label={CELLS[cell][2]}")


if __name__ == "__main__":
    main()
