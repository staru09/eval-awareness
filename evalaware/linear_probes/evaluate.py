import numpy as np

from .base import cosine


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
