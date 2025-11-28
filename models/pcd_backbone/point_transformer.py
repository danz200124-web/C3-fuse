"""
Point cloud backbone (simplified Point Transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PointTransformerLayer(nn.Module):
    """
    Point Transformer layer with self-attention
    """

    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        # Linear projections
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)

        # Position encoding
        self.pos_enc = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        # Output projection
        self.out_proj = nn.Linear(out_channels, out_channels)

        # Residual projection (when in_channels != out_channels)
        self.residual_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        # Normalization
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.ReLU(),
            nn.Linear(out_channels * 4, out_channels)
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) point features
            pos: (B, N, 3) point coordinates

        Returns:
            out: (B, N, C) output features
        """
        B, N, C = x.shape
        identity = self.residual_proj(x)  # Project identity to match output channels

        # Self-attention
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        # Position encoding
        pos_enc = self.pos_enc(pos[:, :, None, :] - pos[:, None, :, :])  # (B, N, N, C)
        pos_enc = pos_enc.view(B, N, N, self.num_heads, self.head_dim)

        # Attention scores
        attn = torch.einsum('bnhd,bmhd->bnmh', q, k) / (self.head_dim ** 0.5)
        attn = attn + pos_enc.mean(dim=-1)  # Add position bias
        attn = F.softmax(attn, dim=2)

        # Aggregate
        out = torch.einsum('bnmh,bmhd->bnhd', attn, v)
        out = out.reshape(B, N, -1)
        out = self.out_proj(out)

        # Residual
        x = self.norm1(identity + out)

        # Feed-forward
        out = self.ffn(x)
        x = self.norm2(x + out)

        return x


class PointTransformerBackbone(nn.Module):
    """
    Point Transformer backbone for point cloud feature extraction
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_stages: int = 4,
        out_channels: int = 256
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initial embedding
        self.embed = nn.Sequential(
            nn.Linear(in_channels, base_channels),
            nn.ReLU(),
            nn.Linear(base_channels, base_channels)
        )

        # Transformer stages
        self.stages = nn.ModuleList()
        channels = base_channels

        for i in range(num_stages):
            next_channels = channels * 2 if i < num_stages - 1 else out_channels

            self.stages.append(
                PointTransformerLayer(channels, next_channels)
            )

            channels = next_channels

        # Output projection
        self.out_proj = nn.Linear(channels, out_channels)

    def forward(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) point coordinates
            features: (B, N, C) point features, optional

        Returns:
            out: (B, N, out_channels) output features
        """
        B, N, _ = points.shape

        # Initial features
        if features is None:
            x = points  # Use coordinates as features
        else:
            x = features

        # Embed
        x = self.embed(x)

        # Apply stages
        for stage in self.stages:
            x = stage(x, points)

        # Output projection
        x = self.out_proj(x)

        return x


def build_pointcloud_backbone(config: dict) -> nn.Module:
    """
    Build point cloud backbone from config

    Args:
        config: dict with backbone parameters

    Returns:
        backbone: point cloud backbone module
    """
    in_channels = config.get('in_channels', 3)
    base_channels = config.get('base_channels', 32)
    num_stages = config.get('num_stages', 4)
    out_channels = config.get('out_channels', 256)

    return PointTransformerBackbone(
        in_channels=in_channels,
        base_channels=base_channels,
        num_stages=num_stages,
        out_channels=out_channels
    )
