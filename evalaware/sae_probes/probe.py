import torch

from ..linear_probes.base import Probe, as_tensor, unit
from ..linear_probes.logistic import LRProbe
from .encode import encode, sae_d_in
from .select import top_features


class SAEProbe(Probe):
    """Logistic probe over a handful of sparse SAE features.

    Three variants, all the same object:
      k=1     single best feature, AxBench's SAE-A
      k=n     top-n features, the interpretable middle ground
      k=None  every feature, the dense baseline

    The point of an SAE probe is not accuracy. Difference-of-means beats SAE
    features on concept detection (AxBench: 0.942 vs 0.695) and SAE probes beat
    logistic regression on 2.2% of 113 datasets (Kantamneni et al.). The point is
    that the selected features are inspectable and steerable.
    """

    def __init__(self, sae, feature_idx, head):
        super().__init__()
        self.sae = sae
        self.register_buffer("feature_idx", as_tensor(feature_idx, dtype=torch.long))
        self.head = head

    def features(self, acts):
        return encode(self.sae, acts)[:, self.feature_idx]

    def forward(self, acts):
        return self.head(self.features(acts))

    @property
    def direction(self):
        """Selected features mapped back to residual space through the decoder.

        This is what makes an SAE probe steerable: the returned vector lives in
        d_in and can be added to the residual stream like any other direction.
        """
        weights = self.head.net[0].weight.data[0] / self.head.scaler_scale.clamp(min=1e-12)
        return unit(weights @ self.sae.W_dec[self.feature_idx].to(weights.device))

    @classmethod
    def from_data(cls, acts, labels, sae=None, k=16, method="auroc", min_firing_rate=0.0, C=0.1):
        if sae is None:
            raise ValueError("SAEProbe.from_data needs sae=<loaded SAE>")
        feats = encode(sae, acts)
        labels = as_tensor(labels)
        idx = (torch.arange(feats.shape[1]) if k is None
               else top_features(feats, labels, k, method, min_firing_rate))
        head = LRProbe.from_data(feats[:, idx], labels, C=C)
        return cls(sae, idx, head)

    def report(self, acts, labels):
        """Which features were chosen and how each one separates on its own."""
        from .select import score_features

        feats = self.features(acts)
        scores = score_features(feats, as_tensor(labels), "auroc")
        return {
            "d_in": sae_d_in(self.sae),
            "n_selected": int(len(self.feature_idx)),
            "feature_idx": [int(i) for i in self.feature_idx],
            "feature_separation": [float(s) for s in scores],
            "accuracy": self.accuracy(acts, labels),
        }
