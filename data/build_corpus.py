"""Merge every prompt set into one file, data/corpus.json.

One row per unique prompt. Each row says which experiments use it, so an
experiment is a filter rather than its own file:

    phase1   the 1100-prompt scaled run, four tiers
    wording  the 400-prompt format control, 100 per group
    run1     the first 100-prompt topic set, already run
    run2     the 100-prompt format control at 25 per group, already run
    run3     the 308-prompt per-benchmark sweep, already run

A prompt used by two experiments appears once with both names in `sets`, so
nothing is stored twice and nothing is lost.

Reads the five existing files rather than re-downloading, so the prompts already
sent to the model stay byte identical.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus.json"
LO, HI = 100, 300

# file -> experiment name
FILES = {
    "final_data.json": "phase1",
    "dataset2.json": "wording",
    "dataset_vea100.json": "run1",
    "dataset_2x2.json": "run2",
    "dataset_benchmarks.json": "run3",
}

# whatever each file called a source -> one canonical name
CANON = {
    "cais/MASK": "mask", "mask": "mask",
    "Anthropic/model-written-evals": "model_written_evals",
    "model_written_evals": "model_written_evals",
    "walledai/BBQ": "bbq", "bbq": "bbq",
    "SALT-NLP/PrivacyLens": "privacylens", "privacylens": "privacylens",
    "basicv8vc/SimpleQA": "simpleqa", "simpleqa": "simpleqa",
    "lmsys/lmsys-chat-1m": "lmsys", "lmsys": "lmsys",
    "allenai/WildChat-1M": "wildchat", "wildchat": "wildchat",
    "deployment": "wildchat_lmsys_pool",
    "sharegpt": "sharegpt",
    "oasst2": "oasst2", "no_robots": "no_robots",
    "mmlu": "mmlu", "arc": "arc", "gsm8k": "gsm8k", "hellaswag": "hellaswag",
    "humaneval": "humaneval", "truthfulqa": "truthfulqa",
    "bench_eval": "bench_eval", "casual_eval": "casual_eval",
    "bench_deploy": "bench_deploy", "casual_deploy": "casual_deploy",
    "eval_awareness_2x2": "eval_awareness_2x2",
}

UPSTREAM = {
    "mask": "cais/MASK",
    "model_written_evals": "Anthropic/model-written-evals",
    "bbq": "walledai/BBQ",
    "privacylens": "SALT-NLP/PrivacyLens",
    "simpleqa": "basicv8vc/SimpleQA",
    "mmlu": "cais/mmlu",
    "arc": "allenai/ai2_arc",
    "gsm8k": "openai/gsm8k",
    "hellaswag": "Rowan/hellaswag",
    "humaneval": "openai/openai_humaneval",
    "truthfulqa": "truthfulqa/truthful_qa",
    "wildchat": "allenai/WildChat-1M",
    "lmsys": "lmsys/lmsys-chat-1m",
    "wildchat_lmsys_pool": "allenai/WildChat-1M + lmsys/lmsys-chat-1m",
    "sharegpt": "ShareGPT, via viliana-dev/eval-awareness-2x2",
    "oasst2": "OpenAssistant/oasst2",
    "no_robots": "HuggingFaceH4/no_robots",
    "bench_eval": "viliana-dev/eval-awareness-2x2",
    "casual_eval": "viliana-dev/eval-awareness-2x2",
    "bench_deploy": "viliana-dev/eval-awareness-2x2",
    "casual_deploy": "viliana-dev/eval-awareness-2x2",
    "eval_awareness_2x2": "viliana-dev/eval-awareness-2x2",
}

# a real evaluation, a real person, or a person writing for a corpus
TIER = {
    "mask": "A", "model_written_evals": "A", "bbq": "A", "mmlu": "A", "gsm8k": "A",
    "privacylens": "A", "simpleqa": "A", "arc": "A", "hellaswag": "A",
    "humaneval": "A", "truthfulqa": "A",
    "wildchat": "B", "lmsys": "B", "sharegpt": "B", "wildchat_lmsys_pool": "B",
    "oasst2": "C", "no_robots": "C",
    "bench_eval": "D", "casual_eval": "D", "bench_deploy": "D", "casual_deploy": "D",
    "eval_awareness_2x2": "D",
}

# only where wording is fixed by construction. Everywhere else it is unknown,
# and guessing it would invent a variable we never controlled.
WORDING = {"bench_eval": "exam", "bench_deploy": "exam",
           "casual_eval": "casual", "casual_deploy": "casual"}

norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()


def main():
    merged = {}
    for fname, exp in FILES.items():
        for r in json.loads((HERE / fname).read_text(encoding="utf-8")):
            # final_data.json labels its 2x2 rows with the generic source name and
            # keeps the real group in `group`. Use the group, or the four wording
            # groups disappear and RQ4 loses its variable.
            raw_src = r["source"]
            if raw_src == "eval_awareness_2x2":
                raw_src = r.get("group", raw_src)
            src = CANON[raw_src]
            key = norm(r["text"])
            row = merged.get(key)
            if row is None:
                row = merged[key] = {
                    "text": r["text"],
                    "n_chars": len(r["text"]),
                    "n_words": len(r["text"].split()),
                    "label": r["label"],
                    "content": "evaluation" if r["label"] else "real_person",
                    "upstream": UPSTREAM[src],
                    "in_band": LO <= len(r["text"]) < HI,
                    "roles": {},
                    "legacy_ids": [],
                }
            if row["label"] != r["label"]:
                raise SystemExit(f"{r['id']} disagrees on label with {row['legacy_ids']}")
            # The same message can play a different part in a different experiment:
            # a ShareGPT first turn is a real user in phase1 and the casual_deploy
            # group in wording. So the role is stored per experiment, never once.
            if exp in row["roles"]:
                raise SystemExit(f"{r['id']} appears twice inside {exp}")
            row["roles"][exp] = {"source": src, "tier": TIER[src], "wording": WORDING.get(src)}
            row["legacy_ids"].append(f"{exp}/{r['id']}")

    rows = sorted(merged.values(), key=lambda r: r["text"])
    rows = [{"id": f"p{i:04d}", **r} for i, r in enumerate(rows, 1)]
    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {OUT}: {len(rows)} unique prompts, "
          f"{sum(len(r['roles']) > 1 for r in rows)} used by more than one experiment")
    for exp in FILES.values():
        sub = [r for r in rows if exp in r["roles"]]
        n_test = sum(r["label"] for r in sub)
        print(f"\n=== {exp}: {len(sub)} prompts, {n_test} test, {len(sub) - n_test} not, "
              f"{sum(r['in_band'] for r in sub)} in the {LO}-{HI} band")
        for tier in "ABCD":
            for src in sorted({r["roles"][exp]["source"] for r in sub
                               if r["roles"][exp]["tier"] == tier}):
                s2 = [r for r in sub if r["roles"][exp]["source"] == src]
                L = sorted(r["n_chars"] for r in s2)
                print(f"   {tier}  {src:22s} {len(s2):4d}  label {s2[0]['label']}  "
                      f"chars {L[0]}/{L[len(L) // 2]}/{L[-1]}")


if __name__ == "__main__":
    main()
