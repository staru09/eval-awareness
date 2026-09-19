import torch
from torch import nn

from .base import Probe, as_tensor


class AttnLiteProbe(Probe):
    """Attention probe with a learned, position-independent query.

    Last-token and mean pooling both discard which part of the prompt is
    diagnostic. AttnLite scores every position with one global query, softmaxes
    over the sequence, and classifies the weighted sum. Because the query does
    not depend on the input, the attention weights read directly as a per-token
    relevance score.
    """

    expects = "sequence"

    def __init__(self, d_in, n_heads=1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_heads, d_in) * d_in ** -0.5)
        self.out = nn.Linear(n_heads * d_in, 1)
        self.n_heads = n_heads

    def weights(self, acts, mask=None):
        acts = as_tensor(acts).to(self.query.device)
        scores = torch.einsum("bsd,hd->bhs", acts, self.query) * acts.shape[-1] ** -0.5
        if mask is not None:
            mask = as_tensor(mask, dtype=torch.bool).to(scores.device)
            scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        return scores.softmax(dim=-1)

    def forward(self, acts, mask=None):
        acts = as_tensor(acts).to(self.query.device)
        pooled = torch.einsum("bhs,bsd->bhd", self.weights(acts, mask), acts)
        return torch.sigmoid(self.out(pooled.flatten(1))).squeeze(-1)

    @classmethod
    def from_data(cls, acts, labels, mask=None, n_heads=1, epochs=200, lr=1e-3, weight_decay=1e-2):
        acts, labels = as_tensor(acts), as_tensor(labels)
        probe = cls(acts.shape[-1], n_heads)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCELoss()
        # ponytail: full-batch training, fine at the sizes here. Add minibatching
        # if the activation tensor stops fitting in memory.
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(probe(acts, mask), labels.to(probe.query.device).float())
            loss.backward()
            opt.step()
        probe.eval()
        return probe
