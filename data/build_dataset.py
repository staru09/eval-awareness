import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "jjpn2/eval_awareness"

ZIP_PASSWORD = b"isthisreallythepassword"
ROLE_LABEL = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}


def fetch(cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "dataset.json"
    if not target.exists():
        archive = hf_hub_download(REPO, "dataset.zip", repo_type="dataset", local_dir=cache_dir)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(cache_dir, pwd=ZIP_PASSWORD)
    return json.loads(target.read_text(encoding="utf-8"))


def is_chat(record: dict) -> bool:
    return not (record.get("metadata", {}).get("scaffold") or "")


def turn_text(content) -> str:
    """Turn content is either a string or a list of typed parts."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n\n".join(t for t in texts if t).strip()
    return ""


def render(record: dict, plain: bool = False) -> str:
    """Flatten the conversation into one block of text.

    The whole transcript is the evidence for whether this was an evaluation, so
    the whole transcript is what gets probed. Using only the first user turn
    would discard the assistant's behaviour, which is where much of the signal
    lives.

    plain keeps the system and user turns' text with no role labels. The
    pipeline sends `text` as one user message, so the system prompt and the
    query arrive together in a single user turn.
    """
    parts = []
    for turn in record.get("input") or []:
        content = turn_text(turn.get("content"))
        if not content:
            continue
        if plain:
            if turn.get("role") in ("system", "user"):
                parts.append(content)
        else:
            parts.append(f"{ROLE_LABEL.get(turn.get('role'), turn.get('role'))}: {content}")
    return "\n\n".join(parts)


def by_source(records: list[dict], label: int, max_chars: int,
              plain: bool = False) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for record in records:
        meta = record.get("metadata", {})
        if int(bool(meta.get("eval_category"))) != label:
            continue
        text = render(record, plain)
        if not (200 <= len(text) <= max_chars):
            continue
        groups[meta.get("category") or "unknown"].append({"record": record, "text": text})
    return groups


def draw(groups: dict[str, list[dict]], n: int, rng: random.Random) -> list[dict]:
    """Round-robin across sources so no single source dominates the sample."""
    pools = {name: rng.sample(items, len(items)) for name, items in groups.items()}
    order = sorted(pools)
    rng.shuffle(order)
    picked = []
    while len(picked) < n and any(pools[name] for name in order):
        for name in order:
            if pools[name] and len(picked) < n:
                picked.append(pools[name].pop())
    return picked


def to_row(item: dict, label: int) -> dict:
    meta = item["record"].get("metadata", {})
    return {
        "id": item["record"]["id"],
        "text": item["text"],
        "label": label,
        "source": meta.get("category") or "unknown",
        "category": meta.get("eval_type") or "chat",
        "subcategory": meta.get("subcategory"),
        "human_eval_rating": meta.get("eval_author1"),
        "human_eval_rating_2": meta.get("eval_author2"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/.jjpn2_cache"))
    ap.add_argument("--out", type=Path, default=Path("data/dataset_jjpn2.json"))
    ap.add_argument("--no-assistant", action="store_true",
                    help="drop transcripts that contain an assistant turn")
    ap.add_argument("--plain", action="store_true",
                    help="system prompt and user text joined, no role labels")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = [r for r in fetch(args.cache_dir) if is_chat(r)]
    print(f"{len(records)} chat transcripts in the dataset")
    if args.no_assistant:
        records = [r for r in records if not any(
            t.get("role") == "assistant" and turn_text(t.get("content"))
            for t in r.get("input") or [])]
        print(f"{len(records)} without an assistant turn")

    rows = []
    for label in (1, 0):
        groups = by_source(records, label, args.max_chars, args.plain)
        picked = draw(groups, args.n_per_class, rng)
        if len(picked) < args.n_per_class:
            raise SystemExit(
                f"only {len(picked)} usable transcripts for label={label}; "
                f"raise --max-chars (currently {args.max_chars}) or lower --n-per-class"
            )
        rows += [to_row(item, label) for item in picked]
        print(f"  label={label}: {len(picked)} from {len(groups)} sources")

    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["prompt_id"] = i

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    sources = {r["source"] for r in rows}
    lengths = sorted(len(r["text"]) for r in rows)
    print(
        f"wrote {len(rows)} rows ({sum(r['label'] for r in rows)} eval) "
        f"from {len(sources)} sources to {args.out}\n"
        f"  chars min/median/max: {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}"
    )


if __name__ == "__main__":
    main()
