import math

import torch
from torch import nn


class RelationAttentionLayer(nn.Module):
    """Relation-specific multi-head graph attention on a fixed heterogeneous graph."""

    def __init__(self, hidden_dim, heads, relation_names, dropout=0.1):
        super().__init__()
        if hidden_dim % heads: raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim, self.heads, self.head_dim = hidden_dim, heads, hidden_dim // heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.ModuleDict({r: nn.Linear(hidden_dim, hidden_dim, bias=False) for r in relation_names})
        self.v = nn.ModuleDict({r: nn.Linear(hidden_dim, hidden_dim, bias=False) for r in relation_names})
        self.relation_prior = nn.ParameterDict({r: nn.Parameter(torch.ones(heads)) for r in relation_names})
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # x: [B,N,D]. Normalize attention separately for every destination/relation.
        batch, nodes, _ = x.shape
        messages = torch.zeros_like(x)
        query = self.q(x).view(batch, nodes, self.heads, self.head_dim)
        for rel, edges in edge_index.items():
            if edges.numel() == 0: continue
            src, dst = edges[0], edges[1]
            key = self.k[rel](x[:, src]).view(batch, -1, self.heads, self.head_dim)
            value = self.v[rel](x[:, src]).view(batch, -1, self.heads, self.head_dim)
            score = (query[:, dst] * key).sum(-1) / math.sqrt(self.head_dim)
            score = score * self.relation_prior[rel]
            # Stable segmented softmax over incoming edges, vectorized over all
            # destination nodes to avoid a Python loop in every training batch.
            edge_targets = dst.view(1, -1, 1).expand(batch, -1, self.heads)
            maximum = torch.full((batch, nodes, self.heads), -float("inf"), device=x.device,
                                 dtype=score.dtype)
            maximum.scatter_reduce_(1, edge_targets, score, reduce="amax", include_self=True)
            exponent = torch.exp(score - maximum.gather(1, edge_targets))
            denominator = torch.zeros(batch, nodes, self.heads, device=x.device, dtype=score.dtype)
            denominator.scatter_add_(1, edge_targets, exponent)
            attention = exponent / torch.clamp(denominator.gather(1, edge_targets), min=1e-12)
            weighted = attention.unsqueeze(-1) * value.float()
            aggregate = torch.zeros(batch, nodes, self.heads, self.head_dim, device=x.device, dtype=weighted.dtype)
            value_targets = dst.view(1, -1, 1, 1).expand(batch, -1, self.heads, self.head_dim)
            aggregate.scatter_add_(1, value_targets, weighted)
            messages = messages + aggregate.reshape(batch, nodes, self.hidden_dim).to(messages.dtype)
        return self.norm(x + self.dropout(self.out(messages)))


class HeterogeneousGraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, layers, relation_names, num_node_types=3, dropout=0.1):
        super().__init__()
        self.type_projections = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_node_types)])
        self.layers = nn.ModuleList([RelationAttentionLayer(hidden_dim, heads, relation_names, dropout)
                                     for _ in range(layers)])

    def forward(self, node_features, node_types, edge_index):
        x = None
        for node_type, projection in enumerate(self.type_projections):
            mask = node_types == node_type
            if torch.any(mask):
                projected = projection(node_features[:, mask])
                if x is None:
                    x = torch.zeros(node_features.shape[0], node_features.shape[1],
                                    self.type_projections[0].out_features, device=node_features.device,
                                    dtype=projected.dtype)
                x[:, mask] = projected
        if x is None: raise ValueError("Graph contains no recognized node type")
        for layer in self.layers: x = layer(x, edge_index)
        return x
