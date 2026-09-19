import torch
from torch import nn


class Probe(nn.Module):
    """Common interface for every probe in this package.

    expects = "pooled"   -> forward takes [batch, d_model]
    expects = "sequence" -> forward takes [batch, seq, d_model] plus a mask
    """

    expects = "pooled"
    supervised = True

    def forward(self, acts, **kwargs):
        raise NotImplementedError

    def pred(self, acts, **kwargs):
        return (self.forward(acts, **kwargs) > 0.5).long()

    def accuracy(self, acts, labels, **kwargs):
        labels = torch.as_tensor(labels, device=self.device)
        return float((self.pred(acts, **kwargs) == labels).float().mean())

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def direction(self):
        """Unit-norm probe direction, or None when the probe has no single one."""
        return None

    @classmethod
    def from_data(cls, acts, labels, **kwargs):
        raise NotImplementedError


def unit(vector):
    return vector / vector.norm().clamp(min=1e-12)


def as_tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype) if not torch.is_tensor(x) else x.to(dtype)


def cosine(a, b):
    if a is None or b is None:
        return float("nan")
    return float(torch.dot(unit(a.flatten()), unit(b.flatten())))
