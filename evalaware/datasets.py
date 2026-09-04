import json
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MCQ_SOURCES = {"hellaswag", "arc_challenge", "mmlu"}


@dataclass(frozen=True)
class Pair:
    positive: str
    negative: str


def load_transcripts(path: Path | None = None) -> list[dict]:
    return json.loads((path or DATA / "dataset.json").read_text(encoding="utf-8"))


def load_traces(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_probe_questions(path: Path | None = None) -> list[dict]:
    path = path or DATA / "probe_questions.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _ensure_colon(text: str) -> str:
    text = text.strip()
    return text if text.endswith(":") else f"{text}:"


def load_pairs(path: Path | None = None) -> list[Pair]:
    path = path or DATA / "contrastive_pairs_16.txt"
    pairs, pos, neg = [], None, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("pair_"):
            if pos and neg:
                pairs.append(Pair(_ensure_colon(pos), _ensure_colon(neg)))
            pos = neg = None
        elif line.startswith("positive_example:"):
            pos = line.split(":", 1)[1].strip()
        elif line.startswith("negative_example:"):
            neg = line.split(":", 1)[1].strip()
    if pos and neg:
        pairs.append(Pair(_ensure_colon(pos), _ensure_colon(neg)))
    if not pairs:
        raise ValueError(f"no pairs in {path}")
    return pairs


def probe_labels(questions: list[dict]) -> dict[str, tuple[str, str]]:
    return {q["id"]: (q["answer"]["real"], q["answer"]["eval"]) for q in questions}
