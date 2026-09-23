import torch
from torch import nn

from .base import Probe, as_tensor, unit


class LRProbe(Probe):
    """Logistic-regression probe, trained as in Geometry of Truth (probes.py).

    A bias-free linear layer plus sigmoid, fitted by full-batch AdamW
    (lr 1e-3, weight decay 0.1) on BCE for 1000 epochs. Like the paper it
    expects mean-centred activations; evaluate.center does that per fold.
    """

    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 1, bias=False), nn.Sigmoid())

    def forward(self, acts):
        return self.net(as_tensor(acts).to(self.device)).squeeze(-1)

    @property
    def direction(self):
        return unit(self.net[0].weight.data[0])

    @classmethod
    def from_data(cls, acts, labels, lr=1e-3, weight_decay=0.1, epochs=1000, device="cpu"):
        acts, labels = as_tensor(acts).to(device), as_tensor(labels).to(device)
        probe = cls(acts.shape[-1]).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCELoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(probe(acts), labels).backward()
            opt.step()
        return probe.eval()
