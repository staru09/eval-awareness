import re
import sys
from pathlib import Path

import numpy as np
import yaml

SENT_RE = re.compile(r"(?<=[.!?])\s+")
HERE = Path(__file__).parent


def load_sentences(path=None):
    data = yaml.safe_load((path or HERE / "sentences.yaml").read_text(encoding="utf-8"))
    return data["eval_aware"], data["neutral"]


def sentence_cuts(text, n_positions):
    """Character offsets at n_positions evenly spaced sentence ends.

    The end of the text is always the final cut, so the whole trace is scored.
    Fixed width so every transcript yields the same [positions] axis and
    metrics.permutation_null can treat it exactly like the layer axis. Position
    therefore means relative depth through the trace, not absolute index.
    """
    ends = [m.end() for m in SENT_RE.finditer(text)] + [len(text)]
    idx = np.linspace(0, len(ends) - 1, n_positions).round().astype(int)
    return [ends[i] for i in idx]


def cot_prefix(templated_prompt, generated_text, reasoning_trace, cut):
    """Rebuild prompt plus the first `cut` characters of the chain of thought.

    Template agnostic: reasoning_trace is a verbatim substring of generated_text
    for every model checked (Qwen3 offset 8, granite 0, gemma 8 or 9), so the
    offset is looked up rather than assumed. A broken rule raises ValueError
    here, which is the intended hard failure.
    """
    off = generated_text.index(reasoning_trace)
    return templated_prompt + generated_text[: off + cut]


def self_check():
    assert sentence_cuts("One. Two. Three.", 3) == [5, 10, 16]
    assert len(sentence_cuts("Only one sentence", 4)) == 4
    assert cot_prefix("P", "<think>\nabc def.\n</think>\n\nans", "abc def.", 4) == "P<think>\nabc "

    eval_aware, neutral = load_sentences()
    assert len(eval_aware) == len(neutral) >= 5

    import json

    checked = 0
    for path in sorted((HERE.parent / "output" / "local").glob("*/reasoning_traces.json")):
        for trace in json.loads(path.read_text(encoding="utf-8")):
            cot, prompt = trace.get("reasoning_trace"), trace.get("templated_prompt")
            if not cot or not prompt:
                continue
            assert cot_prefix(prompt, trace["generated_text"], cot, len(cot)).endswith(cot), path
            for cut in sentence_cuts(cot, 8):
                assert cot_prefix(prompt, trace["generated_text"], cot, cut).startswith(prompt)
            checked += 1
    assert checked >= 20, f"only {checked} stored traces exercised"
    print(f"self-check ok ({checked} stored traces reconstructed)")


if __name__ == "__main__":
    sys.exit(self_check())
