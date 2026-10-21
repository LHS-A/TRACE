import torch
from torch import nn


class EvidenceAttentionScorer(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, tokens):
        scores = self.net(tokens).squeeze(-1)
        attention = torch.softmax(scores, dim=1)
        aggregated = torch.sum(attention.unsqueeze(-1) * tokens, dim=1)
        return attention, aggregated

