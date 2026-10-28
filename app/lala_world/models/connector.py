import torch.nn as nn


class MLP2x(nn.Module):
    """Two-layer GELU MLP: the ViT->LLM connector and the LLM->target out_head."""

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)
