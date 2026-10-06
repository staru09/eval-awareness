"""Merge eval.json and deployment.json into the schema the pipeline reads.

make_traces.py and datasets.load_transcripts want `text`, `label`, `source`,
`category`, `prompt_id`, `id`. The two scenario files use `prompt` and `topic`,
so this renames and shuffles them into one file.

The model only ever sees `text`. Nothing in it says whether the row is an
evaluation or a real conversation; the label is metadata for scoring only.
"""
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 0
# Words that would hand the label to the model if they leaked into the text.
LEAKS = ("eval.json", "deployment.json", "ground_truth", "label:")


def main():
    rows = []
    for name in ("eval.json", "deployment.json"):
        for r in json.loads((DATA / name).read_text(encoding="utf-8")):
            rows.append({"id": r["id"], "text": r["prompt"], "label": r["label"],
                         "source": r["source"], "category": r["topic"],
                         "subset": r["subset"], "n_chars": r["n_chars"]})

    random.Random(SEED).shuffle(rows)
    for i, r in enumerate(rows):
        r["prompt_id"] = i

    for r in rows:
        assert not any(w in r["text"].lower() for w in LEAKS), r["id"]
        assert "\n" not in r["text"], r["id"]

    out = DATA / "dataset_vea100.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    n_eval = sum(r["label"] for r in rows)
    print(f"wrote {out}: {len(rows)} rows, {n_eval} eval, {len(rows) - n_eval} deployment")
    print(f"  sources: {len(set(r['source'] for r in rows))}, "
          f"single turn: {all(chr(10) not in r['text'] for r in rows)}")
    print(f"  first three ids after shuffle: {[r['id'] for r in rows[:3]]}")


if __name__ == "__main__":
    main()
