import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score

from .parsing import PARSE_FAILURE


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def layer_ranks(scores: np.ndarray) -> np.ndarray:
    return np.column_stack([rankdata(scores[:, l]) for l in range(scores.shape[1])])


def auroc_matrix(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_pos = labels.sum(axis=1)
    n_neg = labels.shape[1] - n_pos
    rank_sums = labels @ ranks
    return (rank_sums - (n_pos * (n_pos + 1) / 2)[:, None]) / (n_pos * n_neg)[:, None]


def per_layer_auroc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return auroc_matrix(layer_ranks(scores), labels.astype(float)[None, :])[0]


def band(values: np.ndarray, width: int) -> tuple[float, int]:
    if width <= 1 or values.size < width:
        i = int(np.argmax(values))
        return float(values[i]), i
    smoothed = np.convolve(values, np.ones(width) / width, mode="valid")
    i = int(np.argmax(smoothed))
    return float(smoothed[i]), i


def bands(matrix: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or matrix.shape[1] < width:
        return matrix.max(axis=1)
    kernel = np.ones(width) / width
    return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, matrix).max(axis=1)


def youden_threshold(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return 0.0
    best_j, best_t = -1.0, 0.0
    for t in np.unique(np.concatenate([pos, neg])):
        sens = (pos >= t).sum() / max(len(pos), 1)
        spec = (neg < t).sum() / max(len(neg), 1)
        if sens + spec - 1.0 > best_j:
            best_j, best_t = sens + spec - 1.0, float(t)
    return best_t


def loo_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    n = len(scores)
    correct = 0
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        rest, y = scores[keep], labels[keep]
        t = youden_threshold(rest[y == 1], rest[y == 0])
        correct += int(int(scores[i] >= t) == labels[i])
    return correct / n if n else 0.0


def layer_auroc_table(real: np.ndarray, rand: np.ndarray, labels: np.ndarray) -> dict:
    out = {}
    for l in range(real.shape[1]):
        rand_aurocs = np.array([auroc(rand[:, l, r], labels) for r in range(rand.shape[2])])
        out[str(l)] = {
            "real": {
                "auroc": auroc(real[:, l], labels),
                "loo_accuracy": loo_accuracy(real[:, l], labels),
            },
            "random": {
                "max_auroc_across_R": float(rand_aurocs.max()),
                "mean_auroc_across_R": float(rand_aurocs.mean()),
            },
        }
    return out


def permutation_null(real: np.ndarray, labels: np.ndarray, n_perm: int, width: int,
                     seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ranks = layer_ranks(real)
    perms = np.stack([rng.permutation(labels) for _ in range(n_perm)]).astype(float)
    return bands(auroc_matrix(ranks, perms), width)


def random_direction_null(hidden: np.ndarray, labels: np.ndarray, width: int, n_dirs: int,
                          seed: int, chunk: int = 100) -> np.ndarray:
    _, n_layers, d_model = hidden.shape
    rng = np.random.default_rng(seed)
    out = np.empty(n_dirs)
    y = labels.astype(float)[None, :]
    for start in range(0, n_dirs, chunk):
        n = min(chunk, n_dirs - start)
        v = rng.standard_normal((n_layers, d_model, n)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        proj = np.einsum("nld,ldr->nlr", hidden, v)
        for j in range(n):
            out[start + j] = bands(auroc_matrix(layer_ranks(proj[:, :, j]), y), width)[0]
    return out


def empirical_p(null: np.ndarray, observed: float) -> float:
    return float((int((null >= observed).sum()) + 1) / (len(null) + 1))


def coupling(internal: np.ndarray, verbal: dict[int, float], prompt_ids: np.ndarray,
             seed: int) -> tuple[float | None, float, bool]:
    pairs = [(internal[i], verbal[int(p)]) for i, p in enumerate(prompt_ids) if int(p) in verbal]
    if len(pairs) < 3:
        return None, 0.0, True
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    if len(np.unique(y)) < 2:
        return None, 0.0, True
    rho, _ = spearmanr(x, y)
    mi = 0.0
    if len(np.unique(x)) > 1:
        mi = float(mutual_info_regression(x.reshape(-1, 1), y, random_state=seed)[0])
    return (float(rho) if rho is not None and not np.isnan(rho) else 0.0), mi, False


def ensemble_auroc(rows: list[dict]) -> float | None:
    by_transcript, truth = {}, {}
    for r in rows:
        if r["eval_signal"] == PARSE_FAILURE:
            continue
        by_transcript.setdefault(r["transcript_id"], []).append(r["eval_signal"])
        truth[r["transcript_id"]] = int(r["ground_truth_label"])
    if len(by_transcript) < 2:
        return None
    labels = np.array([truth[t] for t in by_transcript])
    scores = np.array([np.mean(v) for v in by_transcript.values()])
    return auroc(scores, labels) if np.unique(labels).size > 1 else None


def per_probe_auroc(rows: list[dict]) -> dict:
    out = {}
    for pid in sorted({r["probe_id"] for r in rows}):
        scored = [r for r in rows if r["probe_id"] == pid and r["eval_signal"] != PARSE_FAILURE]
        failures = sum(1 for r in rows if r["probe_id"] == pid and r["eval_signal"] == PARSE_FAILURE)
        labels = np.array([int(r["ground_truth_label"]) for r in scored])
        signals = np.array([r["eval_signal"] for r in scored])
        out[pid] = {
            "auroc": auroc(signals, labels) if np.unique(labels).size > 1 else None,
            "accuracy": float((signals.round() == labels).mean()) if len(scored) else None,
            "n_scored": len(scored),
            "n_parse_failures": failures,
        }
    return out
