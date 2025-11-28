"""
Cross-attention fusion module
Points attend to image features with geometric bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from tools.projection import sample_image_features, compute_geometric_bias


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention from 3D points to 2D image features
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        k_neighbors: int = 5,
        use_geo_bias: bool = True,
        bias_dim: int = 16
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        self.use_geo_bias = use_geo_bias
        self.bias_dim = bias_dim

        # Attention layers
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # FFN
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])

        # Geometric bias encoding
        if use_geo_bias:
            self.bias_encoder = nn.Sequential(
                nn.Linear(bias_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_heads)
            )

        # Projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        point_features: torch.Tensor,
        image_features: torch.Tensor,
        points: torch.Tensor,
        uv: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            point_features: (B, N, C) 3D point features
            image_features: (B, C, H, W) 2D image features
            points: (B, N, 3) 3D point coordinates
            uv: (B, N, 2) projected pixel coordinates (normalized)
            K: (B, 3, 3) camera intrinsic
            T: (B, 4, 4) camera extrinsic
            valid_mask: (B, N) visibility mask

        Returns:
            fused_features: (B, N, C) cross-attention fused features
        """
        B, N, C = point_features.shape

        # Sample image features at projected locations
        img_feats = sample_image_features(
            image_features, uv, k=self.k_neighbors, mode='bilinear'
        )  # (B, N, k, C)

        # Reshape for attention: (B, N, k, C) -> (B*N, k, C)
        img_feats_flat = img_feats.reshape(B * N, self.k_neighbors, C)

        # Point features as queries: (B, N, C) -> (B*N, 1, C)
        queries = self.query_proj(point_features).reshape(B * N, 1, C)

        # Image features as keys and values
        keys = self.key_proj(img_feats_flat)
        values = self.value_proj(img_feats_flat)

        # Geometric bias
        attn_bias = None
        if self.use_geo_bias:
            # Compute geometric bias for each point
            geo_bias_list = []
            for b in range(B):
                bias_b = compute_geometric_bias(
                    points[b], uv[b], K[b], T[b], dim=self.bias_dim
                )
                geo_bias_list.append(bias_b)

            geo_bias = torch.stack(geo_bias_list, dim=0)  # (B, N, bias_dim)
            geo_bias_flat = geo_bias.reshape(B * N, self.bias_dim)

            # Encode to attention bias
            attn_bias = self.bias_encoder(geo_bias_flat)  # (B*N, num_heads)
            attn_bias = attn_bias.unsqueeze(1).unsqueeze(2)  # (B*N, 1, 1, num_heads)

        # Multi-layer cross-attention
        x = queries
        for i in range(self.num_layers):
            # Cross-attention
            attn_out, _ = self.attn_layers[i](
                x, keys, values,
                need_weights=False
            )

            # Residual + norm
            x = self.norms[i](x + attn_out)

            # FFN
            ffn_out = self.ffns[i](x)
            x = x + ffn_out

        # Reshape back: (B*N, 1, C) -> (B, N, C)
        output = x.squeeze(1).reshape(B, N, C)

        # Mask invalid points
        output = output * valid_mask.unsqueeze(-1).float()

        return output


def build_cross_attention(config: dict) -> nn.Module:
    """Build cross-attention module from config"""
    return CrossAttentionFusion(
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 3),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        k_neighbors=config.get('k_neighbors', 5),
        use_geo_bias=config.get('use_geo_bias', True),
        bias_dim=config.get('bias_dim', 16)
    )
