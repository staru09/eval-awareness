import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from .base import Probe, as_tensor, unit


class LRProbe(Probe):
    """Logistic-regression probe with the scaler folded in at inference time.

    C=0.1 and fit_intercept=False match the Geometry of Truth setup (lambda=10).
    """

    def __init__(self, d_in, scaler_mean=None, scaler_scale=None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 1, bias=False), nn.Sigmoid())
        self.register_buffer("scaler_mean", torch.zeros(d_in) if scaler_mean is None else as_tensor(scaler_mean))
        self.register_buffer("scaler_scale", torch.ones(d_in) if scaler_scale is None else as_tensor(scaler_scale))

    def _normalize(self, acts):
        return (acts - self.scaler_mean) / self.scaler_scale.clamp(min=1e-12)

    def forward(self, acts):
        acts = as_tensor(acts).to(self.scaler_mean.device)
        return self.net(self._normalize(acts)).squeeze(-1)

    @property
    def direction(self):
        return unit(self.net[0].weight.data[0])

    @classmethod
    def from_data(cls, acts, labels, C=0.1):
        acts, labels = as_tensor(acts).cpu().numpy(), as_tensor(labels).cpu().numpy()
        scaler = StandardScaler().fit(acts)
        fitted = LogisticRegression(C=C, fit_intercept=False, max_iter=2000)
        fitted.fit(scaler.transform(acts), labels)
        probe = cls(acts.shape[1], scaler.mean_, scaler.scale_)
        probe.net[0].weight.data[0] = as_tensor(fitted.coef_[0])
        return probe
