"""
Gating mechanism for adaptive fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGate(nn.Module):
    """
    Adaptive gating to balance cross-attention and BEV features
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        use_confidence: bool = True
    ):
        super().__init__()

        self.use_confidence = use_confidence

        # Gate network
        gate_input_dim = input_dim * 2  # Concatenate two feature paths
        if use_confidence:
            gate_input_dim += 2  # Add modal confidence scores

        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        cross_attn_features: torch.Tensor,
        bev_features: torch.Tensor,
        img_confidence: torch.Tensor = None,
        pcd_confidence: torch.Tensor = None
    ) -> tuple:
        """
        Args:
            cross_attn_features: (B, N, C) features from cross-attention path
            bev_features: (B, N, C) features from BEV path
            img_confidence: (B, N) image modality confidence
            pcd_confidence: (B, N) point cloud modality confidence

        Returns:
            fused_features: (B, N, C) gated fusion output
            gates: (B, N, 1) gate values
        """
        B, N, C = cross_attn_features.shape

        # Concatenate features
        concat_features = torch.cat([cross_attn_features, bev_features], dim=-1)

        # Add confidence scores if available
        if self.use_confidence and img_confidence is not None and pcd_confidence is not None:
            confidence = torch.stack([img_confidence, pcd_confidence], dim=-1)  # (B, N, 2)
            gate_input = torch.cat([concat_features, confidence], dim=-1)
        else:
            gate_input = concat_features

        # Compute gates
        gates = self.gate_net(gate_input)  # (B, N, 1)

        # Gated fusion: γ * f_cross + (1-γ) * f_bev
        fused_features = gates * cross_attn_features + (1 - gates) * bev_features

        return fused_features, gates


class MultiPathGate(nn.Module):
    """
    Multi-path gating for combining multiple fusion paths
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_paths: int = 2,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.num_paths = num_paths

        # Gate network outputs weights for each path
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim * num_paths, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_paths),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )

    def forward(self, feature_list: list) -> tuple:
        """
        Args:
            feature_list: list of (B, N, C) tensors from different paths

        Returns:
            fused_features: (B, N, C) weighted fusion
            weights: (B, N, num_paths) path weights
        """
        assert len(feature_list) == self.num_paths

        B, N, C = feature_list[0].shape

        # Concatenate all paths
        concat_features = torch.cat(feature_list, dim=-1)  # (B, N, C*num_paths)

        # Compute path weights
        weights = self.gate_net(concat_features)  # (B, N, num_paths)

        # Weighted fusion
        fused_features = torch.zeros_like(feature_list[0])
        for i, features in enumerate(feature_list):
            fused_features += weights[:, :, i:i+1] * features

        return fused_features, weights


def compute_modal_confidence(
    features: torch.Tensor,
    mode: str = 'entropy'
) -> torch.Tensor:
    """
    Compute confidence scores for a modality

    Args:
        features: (B, N, C) feature tensor
        mode: 'entropy', 'variance', or 'norm'

    Returns:
        confidence: (B, N) confidence scores
    """
    B, N, C = features.shape

    if mode == 'entropy':
        # Use feature entropy as inverse confidence
        # Normalize features to [0, 1]
        feats_norm = F.softmax(features, dim=-1)
        entropy = -(feats_norm * torch.log(feats_norm + 1e-8)).sum(dim=-1)
        confidence = 1.0 / (1.0 + entropy)

    elif mode == 'variance':
        # Use feature variance
        variance = features.var(dim=-1)
        confidence = 1.0 / (1.0 + variance)

    elif mode == 'norm':
        # Use feature norm
        norm = features.norm(dim=-1)
        confidence = torch.sigmoid(norm)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return confidence


def build_gate(config: dict) -> nn.Module:
    """Build gating module from config"""
    return AdaptiveGate(
        input_dim=config.get('input_dim', 256),
        hidden_dim=config.get('hidden_dim', 128),
        use_confidence=config.get('use_confidence', True)
    )
