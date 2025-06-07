"""
MEGAN Model Architecture Components
Contains the core neural network layers and model architecture for MEGAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter
from typing import Optional, Tuple, List
from torch import Tensor
import numpy as np


class GATv2WithLogits(nn.Module):
    """
    Custom GATv2‐style attention layer that explicitly stores raw (pre‐softmax)
    attention logits in `latest_logits`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 0,
        heads: int = 1,
        concat: bool = False,
        use_edge_features: bool = False,
        dropout: float = 0.0,
        add_bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.use_edge_features = use_edge_features
        self.dropout = dropout
        
        # Node feature projection
        self.lin_node = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Edge feature projection (if using edge features)
        if use_edge_features and edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None
        
        # Attention mechanism (à la GATv2)
        self.att = nn.Parameter(torch.empty(1, heads, out_channels))
        
        # Output projection and bias
        if concat and heads > 1:
            self.out_proj = nn.Linear(heads * out_channels, out_channels, bias=add_bias)
        else:
            self.out_proj = None
        
        # Store latest attention logits for MEGAN explanation extraction
        self.latest_logits = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
        if self.out_proj is not None:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self, 
        x: Tensor, 
        edge_index: Tensor, 
        edge_attr: Optional[Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tensor:
        """
        Forward pass with attention computation.
        """
        H, C = self.heads, self.out_channels
        N = x.size(0)
        
        # Project node features
        x_proj = self.lin_node(x).view(N, H, C)  # [N, H, C]
        
        # Handle edge features if provided
        if self.use_edge_features and edge_attr is not None and self.lin_edge is not None:
            edge_proj = self.lin_edge(edge_attr).view(-1, H, C)  # [E, H, C]
        else:
            edge_proj = None
        
        # Get source and target node indices
        row, col = edge_index[0], edge_index[1]
        
        # Prepare features for attention computation
        x_i = x_proj[row]  # Source nodes [E, H, C]
        x_j = x_proj[col]  # Target nodes [E, H, C]
        
        # Combine node and edge features (GATv2 style)
        if edge_proj is not None:
            alpha_input = x_i + x_j + edge_proj  # [E, H, C]
        else:
            alpha_input = x_i + x_j  # [E, H, C]
        
        # Apply non-linearity and compute attention scores
        alpha_input = F.leaky_relu(alpha_input, negative_slope=0.2)
        alpha = (alpha_input * self.att).sum(dim=-1)  # [E, H]
        
        # Store raw logits for MEGAN explanation
        self.latest_logits = alpha.detach()
        
        # Apply softmax to get attention weights
        alpha = softmax(alpha, col, num_nodes=N)  # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention weights to target node features
        out = x_j * alpha.unsqueeze(-1)  # [E, H, C]
        out = scatter(out, col, dim=0, dim_size=N, reduce='sum')  # [N, H, C]
        
        # Concatenate or average heads
        if self.concat and self.heads > 1:
            out = out.view(N, H * C)
            if self.out_proj is not None:
                out = self.out_proj(out)
        else:
            out = out.mean(dim=1)  # Average across heads: [N, C]
        
        if return_attention_weights:
            return out, (edge_index, alpha)
        return out


class MEGANCore(nn.Module):
    """
    MEGAN (Multi‐Explanation Graph Attention Network) core architecture.
    Implements L layers, each with K "explanation" heads, uses raw logits for
    E^{im} and V^{im} as in the paper.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 60,
        out_channels: int = 1,
        edge_dim: int = 0,
        num_layers: int = 4,
        K: int = 2,
        heads_gat: int = 1,
        use_edge_features: bool = False,
        add_self_loops: bool = True,
        dropout: float = 0.1,
        layer_norm: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.K = K
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.use_edge_features = use_edge_features
        self.residual = residual
        
        # Build attention layers
        self.attn_layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            curr_in = in_channels if layer_idx == 0 else hidden_channels
            layer_k = nn.ModuleList([
                GATv2WithLogits(
                    in_channels=curr_in,
                    out_channels=hidden_channels,
                    edge_dim=edge_dim,
                    heads=heads_gat,
                    concat=False,
                    use_edge_features=use_edge_features,
                    dropout=dropout,
                )
                for _ in range(K)
            ])
            self.attn_layers.append(layer_k)
        
        # Layer normalization (optional)
        self.layer_norms = nn.ModuleList()
        if layer_norm:
            for _ in range(num_layers):
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # Final prediction head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Store attention logits for explanation
        self.attention_logits = []
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass through MEGAN layers.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, D] (optional)
            batch: Batch assignment [N] (optional)
        
        Returns:
            Graph-level prediction [B, out_channels]
        """
        # Add self loops if requested
        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=x.size(0)
            )
        
        # Clear previous attention logits
        self.attention_logits = []
        
        # Process through MEGAN layers
        h = x
        for layer_idx, layer_k in enumerate(self.attn_layers):
            layer_outputs = []
            layer_logits = []
            
            # Apply K attention heads in parallel
            for k_idx, attn_k in enumerate(layer_k):
                h_k = attn_k(h, edge_index, edge_attr)
                layer_outputs.append(h_k)
                
                # Store attention logits for this head
                if hasattr(attn_k, 'latest_logits') and attn_k.latest_logits is not None:
                    layer_logits.append(attn_k.latest_logits.clone())
            
            # Combine outputs from K heads (average)
            h_new = torch.stack(layer_outputs, dim=0).mean(dim=0)
            
            # Apply residual connection
            if self.residual and h.size(-1) == h_new.size(-1):
                h_new = h + h_new
            
            # Apply layer normalization
            if layer_idx < len(self.layer_norms):
                h_new = self.layer_norms[layer_idx](h_new)
            
            h = h_new
            
            # Store layer attention logits
            if layer_logits:
                self.attention_logits.append(layer_logits)
        
        # Global pooling for graph-level prediction
        if batch is None:
            # Single graph case
            graph_repr = global_add_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))
        else:
            # Batch case
            graph_repr = global_add_pool(h, batch)
        
        # Final prediction
        out = self.classifier(graph_repr)
        
        return out
    
    def get_explanations(self, layer_idx: int = -1) -> List[Tensor]:
        """
        Extract attention logits for explanation analysis.
        
        Args:
            layer_idx: Which layer to extract (-1 for last layer)
        
        Returns:
            List of attention logits for each explanation head
        """
        if not self.attention_logits:
            return []
        
        if layer_idx == -1:
            layer_idx = len(self.attention_logits) - 1
        
        if 0 <= layer_idx < len(self.attention_logits):
            return self.attention_logits[layer_idx]
        
        return []
