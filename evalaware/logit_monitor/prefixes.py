import re
from pathlib import Path

import numpy as np
import yaml

SENT_RE = re.compile(r"(?<=[.!?])\s+")
HERE = Path(__file__).parent


def load_sentences(path=None):
    data = yaml.safe_load((path or HERE / "sentences.yaml").read_text(encoding="utf-8"))
    return data["eval_aware"], data["neutral"]


def sentence_cuts(text, n_positions):
    """Character offsets of sentence ends.

    n_positions=0 (the default) scores every boundary, so traces of different
    lengths give different counts and the caller pads. Passing a number
    subsamples to a fixed grid instead, which makes the position axis
    comparable across traces at the cost of skipping boundaries.

    The end of the text is always the final cut, so the whole trace is scored.
    Fixed width so every transcript yields the same [positions] axis and
    metrics.permutation_null can treat it exactly like the layer axis. Position
    therefore means relative depth through the trace, not absolute index.
    """
    ends = [m.end() for m in SENT_RE.finditer(text)] + [len(text)]
    if not n_positions:
        return ends
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
