"""Differentiable MEGAN regression core: see docs/reproducing-megan.md.

Implements the 2023 paper's multi-channel encoder/importance/readout structure.
Residuals, LayerNorm and multi-head averaging are explicitly optional variants.
"""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_remaining_self_loops, scatter, softmax
from torch_geometric.utils.smiles import x_map, e_map

ARCHITECTURE_VERSION = "megan-regression-v2"
EMBEDDED_ARCHITECTURE_VERSION = "megan-regression-embedded-v3"
SUPPORTED_ARCHITECTURES = {ARCHITECTURE_VERSION, EMBEDDED_ARCHITECTURE_VERSION}


class CategoricalEncoder(nn.Module):
    """Separate lookup tables preserve category identity without imposing order."""

    def __init__(self, sizes, embedding_dim):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(size, embedding_dim) for size in sizes])
        for table in self.tables:
            nn.init.xavier_uniform_(table.weight)

    def forward(self, values):
        if values.size(1) != len(self.tables):
            raise ValueError("Categorical feature width does not match the PyG schema")
        indices = values.long()
        if not torch.equal(values, indices.to(values.dtype)):
            raise ValueError("Categorical features must be integer indices")
        for i, table in enumerate(self.tables):
            if bool(((indices[:, i] < 0) | (indices[:, i] >= table.num_embeddings)).any()):
                raise ValueError("Categorical feature index is outside its vocabulary")
        return torch.cat([table(indices[:, i]) for i, table in enumerate(self.tables)], dim=1)


class GATv2WithLogits(nn.Module):
    """Dynamic attention; edge_index[0] sends messages to edge_index[1]."""

    def __init__(self, in_channels, out_channels, edge_dim=0, heads=1,
                 concat=False, use_edge_features=False, dropout=0.0, add_bias=True):
        super().__init__()
        self.heads, self.out_channels = heads, out_channels
        self.concat, self.dropout = concat, dropout
        self.lin_node = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_edge = (nn.Linear(edge_dim, heads * out_channels, bias=False)
                         if use_edge_features and edge_dim > 0 else None)
        # Separate query/target and source/message projections match GATv2.
        self.lin_target = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, out_channels))
        self.out_proj = (nn.Linear(heads * out_channels, out_channels, bias=False)
                         if concat and heads > 1 else None)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if add_bias else None
        self.latest_logits = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_target.weight)
        nn.init.xavier_uniform_(self.att)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        if self.out_proj is not None:
            nn.init.xavier_uniform_(self.out_proj.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        n = x.size(0)
        h = self.lin_node(x).view(n, self.heads, self.out_channels)
        src, dst = edge_index
        query = self.lin_target(x).view(n, self.heads, self.out_channels)
        attention_input = query[dst] + h[src]
        if self.lin_edge is not None:
            if edge_attr is None:
                raise ValueError("Bond features are required by this model")
            attention_input = attention_input + self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        # Nonlinearity BEFORE the attention vector distinguishes GATv2 from GAT.
        attention_features = F.leaky_relu(attention_input, negative_slope=0.2)
        logits = (attention_features * self.att).sum(-1)
        self.latest_logits = logits  # Keep autograd: explanations are trained.
        alpha = softmax(logits, dst, num_nodes=n)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        messages = h[src] * alpha.unsqueeze(-1)
        out = scatter(messages, dst, dim=0, dim_size=n, reduce="sum")
        out = self.out_proj(out.flatten(1)) if self.out_proj is not None else out.mean(1)
        if self.bias is not None:
            out = out + self.bias
        return (out, (edge_index, alpha)) if return_attention_weights else out


class MEGANCore(nn.Module):
    """Prediction plus aligned [E,K] edge and [N,K] node importance masks.

    Default forward returns a prediction Tensor for existing training callers.
    Use return_explanations=True for (prediction, edge_importances, node_importances).
    Output edge explanations align with INPUT edges, excluding added self-loops.
    """

    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, in_channels, hidden_channels=60, out_channels=1, edge_dim=0,
                 num_layers=4, K=2, heads_gat=1, use_edge_features=False,
                 add_self_loops=True, dropout=0.1, layer_norm=True, residual=True,
                 feature_encoding="ordinal", embedding_dim=16):
        super().__init__()
        if min(in_channels, hidden_channels, num_layers, K, heads_gat) < 1:
            raise ValueError("Architecture dimensions must be positive")
        self.K, self.num_layers = K, num_layers
        self.add_self_loops, self.residual = add_self_loops, residual
        if feature_encoding not in ("ordinal", "embedding") or embedding_dim < 1:
            raise ValueError("Require ordinal/embedding encoding and positive embedding_dim")
        self.feature_encoding = feature_encoding
        self.node_encoder = self.edge_encoder = None
        if feature_encoding == "embedding":
            if in_channels != len(x_map) or (use_edge_features and edge_dim != len(e_map)):
                raise ValueError("Embedding models require the maintained PyG SMILES schema")
            self.architecture_version = EMBEDDED_ARCHITECTURE_VERSION
            self.node_encoder = CategoricalEncoder([len(v) for v in x_map.values()], embedding_dim)
            in_channels = len(x_map) * embedding_dim
            if use_edge_features:
                self.edge_encoder = CategoricalEncoder([len(v) for v in e_map.values()], embedding_dim)
                edge_dim = len(e_map) * embedding_dim
        self.attn_layers = nn.ModuleList([
            nn.ModuleList([
                GATv2WithLogits(in_channels if l == 0 else hidden_channels,
                               hidden_channels, edge_dim, heads_gat,
                               use_edge_features=use_edge_features, dropout=dropout)
                for _ in range(K)
            ]) for l in range(num_layers)
        ])
        self.input_proj = (nn.Linear(in_channels, hidden_channels, bias=False)
                           if residual and in_channels != hidden_channels else None)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ] if layer_norm else [])
        self.node_importance_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, K), nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(K * hidden_channels, hidden_channels), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, out_channels),
        )
        self.attention_logits = []
        self.node_importances = self.edge_importances = self.batch_indices = None

    def forward(self, x, edge_index, edge_attr=None, batch=None,
                return_explanations=False, importance_mask: Optional[Tensor] = None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        original_edges = edge_index.size(1)
        if self.node_encoder is not None:
            x = self.node_encoder(x)
        if self.edge_encoder is not None:
            if edge_attr is None:
                raise ValueError("Embedding models require bond category indices")
            # Encode physical bonds first; synthetic loops get ZERO embeddings.
            edge_attr = self.edge_encoder(edge_attr)
        # Molecular graphs have no self-loops; allowing input loops would reorder
        # them in PyG and break the documented explanation alignment.
        if bool((edge_index[0] == edge_index[1]).any()):
            raise ValueError("Pass original molecular edges without self-loops")
        effective_edges = edge_index
        if self.add_self_loops:
            effective_edges, edge_attr = add_remaining_self_loops(
                edge_index, edge_attr, fill_value=0.0, num_nodes=x.size(0)
            )
        self.attention_logits = []
        h = x
        for l, channels in enumerate(self.attn_layers):
            outputs = [F.elu(channel(h, effective_edges, edge_attr)) for channel in channels]
            self.attention_logits.append([channel.latest_logits for channel in channels])
            new_h = torch.stack(outputs).mean(0)
            if self.residual:
                new_h = new_h + (self.input_proj(h) if l == 0 and self.input_proj is not None else h)
            h = self.layer_norms[l](new_h) if self.layer_norms else new_h
        # Eq. 3: sum RAW logits across layers, then sigmoid (not softmax).
        logits = torch.stack([
            torch.stack([a.mean(-1) for a in layer], dim=-1)
            for layer in self.attention_logits
        ]).sum(0)
        effective_importance = torch.sigmoid(logits)
        # Mean pooling keeps the gate in [0,1], unlike degree-dependent sums.
        edge_pool = scatter(effective_importance, effective_edges[1], dim=0,
                            dim_size=x.size(0), reduce="mean")
        node_importance = self.node_importance_mlp(h) * edge_pool
        if importance_mask is not None:
            if importance_mask.shape != node_importance.shape:
                raise ValueError("importance_mask must have shape [nodes, channels]")
            node_importance = node_importance * importance_mask
        self.node_importances = node_importance
        self.edge_importances = effective_importance[:original_edges]
        self.batch_indices = batch
        pooled = [global_add_pool(h * node_importance[:, k:k+1], batch) for k in range(self.K)]
        prediction = self.classifier(torch.cat(pooled, dim=-1))
        if return_explanations:
            return prediction, self.edge_importances, node_importance
        return prediction

    def get_explanations(self, layer_idx=-1):
        """Raw differentiable attention diagnostics; use forward for actual masks."""
        if not self.attention_logits:
            return []
        return self.attention_logits[layer_idx]
