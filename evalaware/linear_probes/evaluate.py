import numpy as np
import torch

from .base import as_tensor, cosine


def center(train, *others):
    """Subtract the training rows' mean from every set (Geometry of Truth centres acts)."""
    mu = train.mean(0)
    return (train - mu, *(o - mu for o in others))


def loso_sweep(acts, labels, groups, probe_cls, device="cpu", **kwargs):
    """Leave-one-group-out, per layer, for a pooled probe.

    acts: [n_layers, n_rows, d_model]; labels: 0/1 per row; groups: e.g. the
    source of each row. Each group is held out once; the probe is trained on
    the rest (centred with their own mean) and scores the held-out rows. In
    this data every source is single-class, so holding out a whole source is
    what stops the probe from winning by recognising source style.

    Returns per-layer out-of-fold scores, plus a direction per layer trained on
    all rows for comparing against other directions.
    """
    _check_pooled(probe_cls)
    acts = as_tensor(acts)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    y = torch.as_tensor(labels, dtype=torch.float32)
    scores = np.zeros((len(acts), len(labels)), dtype=np.float32)
    directions = []
    # ponytail: one probe per layer per fold in a Python loop; batch layers into one
    # optimiser if LR sweeps over big models get slow.
    for layer, x in enumerate(acts):
        for g in np.unique(groups):
            test = groups == g
            if len(np.unique(labels[~test])) < 2:
                raise ValueError(f"holding out {g!r} leaves one class in training")
            train_x, test_x = center(x[~test], x[test])
            probe = fit(probe_cls, train_x, y[~test], device=device, **kwargs)
            with torch.no_grad():
                scores[layer, test] = probe(test_x).cpu().numpy()
        directions.append(fit(probe_cls, center(x)[0], y, device=device, **kwargs).direction.cpu())
    return {"scores": scores, "directions": torch.stack(directions)}


def _check_pooled(probe_cls):
    if probe_cls.expects != "pooled":
        raise ValueError(f"{probe_cls.__name__} needs sequence activations; train it directly")


def fit(probe_cls, acts, labels, **kwargs):
    """Uniform entry point: unsupervised probes ignore labels in from_data."""
    return probe_cls.from_data(acts, labels, **kwargs)


def layer_sweep(train_acts, train_labels, test_acts, test_labels, probe_cls, **kwargs):
    """Accuracy per layer. Activations are [n_layers, n_samples, d_model].

    Truth is reported to live in early-to-mid layers rather than at the end, so
    this is the cheapest way to find where to probe before doing anything else.
    """
    _check_pooled(probe_cls)
    out = []
    for layer in range(len(train_acts)):
        probe = fit(probe_cls, train_acts[layer], train_labels, **kwargs)
        out.append({"layer": layer, "accuracy": probe.accuracy(test_acts[layer], test_labels)})
    return out


def cross_dataset(datasets, probe_cls, **kwargs):
    """Train on each dataset, test on all of them.

    `datasets` maps name -> {"train": (acts, labels), "test": (acts, labels)}.
    Off-diagonal entries measure whether the model holds one general direction
    or a separate one per dataset.
    """
    _check_pooled(probe_cls)
    names = list(datasets)
    probes = {n: fit(probe_cls, *datasets[n]["train"], **kwargs) for n in names}
    matrix = [[probes[tr].accuracy(*datasets[te]["test"]) for te in names] for tr in names]
    return {
        "names": names,
        "probe": probe_cls.__name__,
        "accuracy": matrix,
        "cosine": direction_similarity(probes)["cosine"],
    }


def direction_similarity(probes):
    """Pairwise cosine between probe directions. NaN where a probe has none."""
    names = list(probes)
    dirs = {n: probes[n].direction for n in names}
    return {
        "names": names,
        "cosine": [[cosine(dirs[a], dirs[b]) for b in names] for a in names],
    }


def summarise_sweep(sweep):
    accs = [row["accuracy"] for row in sweep]
    best = int(np.argmax(accs))
    return {"best_layer": sweep[best]["layer"], "best_accuracy": accs[best], "per_layer": accs}
