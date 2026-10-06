"""Build eval.json and deployment.json: short English prompts, 50 of each.

Eval side: 10 prompts from each of the five topic datasets.
Deployment side: 25 each from WildChat-1M and LMSYS-Chat-1M.
Every row keeps the dataset it came from and the subset inside it.
"""
import glob, json, os, random, re
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0  # langdetect is stochastic without this

HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
MIN_CHARS, MAX_CHARS = 40, 300
SEED = 0


def snap(repo):
    return Path(glob.glob(str(HUB / f"datasets--{repo}" / "snapshots" / "*"))[0])


def english(text):
    """Short, clean, and actually English: the detector has the final say."""
    if not text or not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    if text.count("\n") > 3 or "```" in text or "<" in text and ">" in text:
        return False
    if sum(c.isascii() for c in text) / len(text) < 0.98 or " " not in text.strip():
        return False
    try:  # the LMSYS language column is model-detected and lets Spanish through
        return detect(text) == "en"
    except Exception:
        return False


def clean(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def take(rows, n, seed=SEED):
    """Deduplicate on the prompt text, then sample n without replacement."""
    seen, uniq = set(), []
    for r in rows:
        key = r["prompt"].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    random.Random(seed).shuffle(uniq)
    return uniq[:n]


# ---------------------------------------------------------------- eval sources

def model_written_evals():
    out = []
    for path in sorted(snap("Anthropic--model-written-evals").glob(
            "advanced-ai-risk/human_generated_evals/*.jsonl")):
        for line in path.open(encoding="utf-8"):
            q = clean(json.loads(line)["question"])
            if english(q):
                out.append({"prompt": q, "source": "Anthropic/model-written-evals",
                            "subset": path.stem, "topic": "alignment_values"})
    return out


def bbq():
    out = []
    for path in sorted(snap("walledai--BBQ").glob("data/*.parquet")):
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            q = clean(f"{r['context']} {r['question']}")
            if english(q):
                out.append({"prompt": q, "source": "walledai/BBQ",
                            "subset": str(r["category"]), "topic": "bias_fairness",
                            "choices": [str(c) for c in r["choices"]]})
    return out


def mask():
    out = []
    for path in sorted(snap("cais--MASK").glob("*/*.parquet")):
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            q = clean(r["user_prompt"])
            if english(q):
                out.append({"prompt": q, "source": "cais/MASK",
                            "subset": path.parent.name, "topic": "deception_scheming",
                            "system_prompt": clean(r["system_prompt"])})
    return out


def simpleqa():
    df = pd.read_csv(snap("basicv8vc--SimpleQA") / "simple_qa_test_set.csv")
    out = []
    for _, r in df.iterrows():
        q = clean(r["problem"])
        if english(q):
            meta = eval(r["metadata"]) if isinstance(r["metadata"], str) else {}
            out.append({"prompt": q, "source": "basicv8vc/SimpleQA",
                        "subset": str(meta.get("topic", "")), "topic": "calibration_uncertainty",
                        "answer": clean(r["answer"])})
    return out


def privacylens():
    rows = json.loads((snap("SALT-NLP--PrivacyLens") / "main_data.json").read_text(encoding="utf-8"))
    out = []
    for r in rows:
        traj = r["trajectory"]
        q = clean(traj.get("user_instruction", ""))
        if english(q):
            out.append({"prompt": q, "source": "SALT-NLP/PrivacyLens",
                        "subset": r["name"].rstrip("0123456789"), "topic": "privacy",
                        "data_type": r["seed"].get("data_type", "")})
    return out


# ---------------------------------------------------------- deployment sources

def first_english_turns(path, source, text_of, lang_of, limit_rows=40000):
    df = pd.read_parquet(path)
    out = []
    for _, r in df.head(limit_rows).iterrows():
        if str(lang_of(r)).lower() != "english":
            continue
        q = clean(text_of(r))
        if english(q):
            out.append({"prompt": q, "source": source, "subset": path.name,
                        "topic": "real_user_chat"})
    return out


def wildchat():
    path = sorted(snap("allenai--WildChat-1M").glob("data/*.parquet"))[0]
    return first_english_turns(path, "allenai/WildChat-1M",
                               lambda r: r["conversation"][0]["content"],
                               lambda r: r["language"])


def lmsys():
    path = sorted(snap("lmsys--lmsys-chat-1m").glob("data/*.parquet"))[0]
    return first_english_turns(path, "lmsys/lmsys-chat-1m",
                               lambda r: r["conversation"][0]["content"],
                               lambda r: r["language"])


def main():
    eval_rows = []
    for i, fn in enumerate([model_written_evals, bbq, mask, simpleqa, privacylens]):
        pool = fn()
        picked = take(pool, 10, seed=SEED + i)
        print(f"  {picked[0]['topic']:24s} pool={len(pool):6d} picked={len(picked)}")
        eval_rows += picked

    deploy_rows = []
    for i, fn in enumerate([wildchat, lmsys]):
        pool = fn()
        picked = take(pool, 25, seed=SEED + i)
        print(f"  {picked[0]['source']:24s} pool={len(pool):6d} picked={len(picked)}")
        deploy_rows += picked

    out_dir = Path(__file__).resolve().parent
    for name, rows, label in [("eval.json", eval_rows, 1), ("deployment.json", deploy_rows, 0)]:
        for n, r in enumerate(rows):
            r["id"] = f"{r['source'].split('/')[-1]}:{n:03d}"
            r["label"] = label
            r["n_chars"] = len(r["prompt"])
        (out_dir / name).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out_dir / name}: {len(rows)} prompts, "
              f"median {sorted(r['n_chars'] for r in rows)[len(rows) // 2]} chars")


if __name__ == "__main__":
    main()
