from pathlib import Path

import numpy as np


def _project(vec_real, vec_rand, h) -> tuple[float, np.ndarray]:
    return float(np.dot(h, vec_real)), vec_rand @ h


def score(probe: dict, act_dir: Path, rows: list[dict], out: Path, model_key: str | None = None) -> dict:
    real, rand = probe["real"], probe["random"]
    n_random = int(probe["metadata"]["n_random"])

    last_real, mean_real, last_rand, mean_rand = [], [], [], []
    labels, sources, prompt_ids = [], [], []
    n_layers = None

    for row in rows:
        path = act_dir / f"{row['id']}.npz"
        if not path.exists():
            raise SystemExit(f"missing activations for {row['id']} at {path}")
        with np.load(path) as z:
            last_h, mean_h = z["last_token"], z["mean_token"]
        if n_layers is None:
            n_layers = last_h.shape[0]

        lr = np.zeros(n_layers, dtype=np.float32)
        mr = np.zeros(n_layers, dtype=np.float32)
        lx = np.zeros((n_layers, n_random), dtype=np.float32)
        mx = np.zeros((n_layers, n_random), dtype=np.float32)
        for i in range(n_layers):
            v_real = real[i + 1].numpy()
            v_rand = rand[i + 1].numpy()
            lr[i], lx[i] = _project(v_real, v_rand, last_h[i])
            mr[i], mx[i] = _project(v_real, v_rand, mean_h[i])

        last_real.append(lr)
        mean_real.append(mr)
        last_rand.append(lx)
        mean_rand.append(mx)
        labels.append(int(row["label"]))
        sources.append(row["source"])
        prompt_ids.append(int(row["prompt_id"]))

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        scores_last_real=np.stack(last_real),
        scores_mean_real=np.stack(mean_real),
        scores_last_random=np.stack(last_rand),
        scores_mean_random=np.stack(mean_rand),
        labels=np.array(labels, dtype=np.int32),
        source=np.array(sources),
        prompt_ids=np.array(prompt_ids, dtype=np.int64),
        n_layers=n_layers,
        n_random=n_random,
        model_key=model_key or probe["metadata"]["model_path"],
    )
    return {"n_rows": len(rows), "n_layers": n_layers, "n_random": n_random}


def load_scores(path: Path, pooling: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {
            "real": z[f"scores_{pooling}_real"],
            "random": z[f"scores_{pooling}_random"],
            "labels": z["labels"].astype(int),
            "prompt_ids": z["prompt_ids"],
            "n_layers": int(z["n_layers"]),
            "n_random": int(z["n_random"]),
            "model_key": str(z["model_key"]) if "model_key" in z else "unknown",
        }
