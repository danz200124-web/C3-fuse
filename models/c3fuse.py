"""
C3-Fuse Main Network
Combines cross-attention, cylindrical BEV, and gating for image-point cloud fusion
"""

import torch
import torch.nn as nn
import sys
sys.path.append('..')

from models.img_backbone.resnet import build_image_backbone
from models.pcd_backbone.point_transformer import build_pointcloud_backbone
from models.fusion.cross_attn import build_cross_attention
from models.fusion.gate import build_gate, compute_modal_confidence
from tools.cyl_grid import CylindricalBEVFusion


class SegmentationHead(nn.Module):
    """Segmentation head for final predictions"""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dims: list = [256, 128, 64],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        prev_dim = in_channels

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) features

        Returns:
            logits: (B, N, num_classes)
        """
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)
        logits = self.mlp(x_flat)
        return logits.reshape(B, N, -1)


class C3FuseNet(nn.Module):
    """
    C³-Fuse Network: Cross-attention + Cylindrical BEV + Gating
    for robust image-point cloud fusion
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        # Backbones
        self.img_backbone = build_image_backbone(config['model']['img_backbone'])
        self.pcd_backbone = build_pointcloud_backbone(config['model']['pcd_backbone'])

        fusion_config = config['model']['fusion']

        # Fusion paths
        self.use_cross_attn = fusion_config['cross_attn']['enabled']
        self.use_bev = fusion_config['unified_space']['enabled']
        self.use_gate = fusion_config['gate']['enabled']

        hidden_dim = fusion_config['cross_attn']['hidden_dim']

        if self.use_cross_attn:
            self.cross_attn = build_cross_attention(fusion_config['cross_attn'])

        if self.use_bev:
            self.bev_fusion = CylindricalBEVFusion(
                grid_config=fusion_config['unified_space']['grid'],
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                pooling=fusion_config['unified_space']['pooling']
            )

        if self.use_gate:
            self.gate = build_gate(fusion_config['gate'])

        # Segmentation head
        seg_config = config['model']['seg_head']
        self.seg_head = SegmentationHead(
            in_channels=hidden_dim,
            hidden_dims=seg_config['hidden_dims'],
            num_classes=seg_config['num_classes'],
            dropout=seg_config['dropout']
        )

        # Feature projection to unified dimension
        self.img_proj = nn.Linear(
            self.img_backbone.out_channels[-1], hidden_dim
        )
        self.pcd_proj = nn.Linear(
            self.pcd_backbone.out_channels, hidden_dim
        )

    def forward(
        self,
        images: torch.Tensor,
        points: torch.Tensor,
        uv: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        valid_mask: torch.Tensor,
        return_intermediates: bool = False
    ) -> dict:
        """
        Args:
            images: (B, 3, H, W) RGB images
            points: (B, N, 3) point cloud coordinates
            uv: (B, N, 2) projected pixel coordinates (normalized)
            K: (B, 3, 3) camera intrinsic matrices
            T: (B, 4, 4) camera extrinsic matrices
            valid_mask: (B, N) visibility mask
            return_intermediates: whether to return intermediate features

        Returns:
            outputs: dict with keys:
                - logits: (B, N, num_classes) segmentation logits
                - features: (B, N, C) fused features (optional)
                - gates: (B, N, 1) gate values (optional)
        """
        B, N, _ = points.shape

        # Extract image features
        img_feats_list = self.img_backbone(images)
        img_feats = img_feats_list[-1]  # Use finest scale
        img_feats = self.img_proj(img_feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Extract point cloud features
        pcd_feats = self.pcd_backbone(points)  # (B, N, C)
        pcd_feats = self.pcd_proj(pcd_feats)

        # Fusion paths
        outputs = {}

        if self.use_cross_attn and self.use_bev:
            # Both paths active: use gating
            cross_feats = self.cross_attn(
                pcd_feats, img_feats, points, uv, K, T, valid_mask
            )

            bev_feats = self.bev_fusion(points, pcd_feats)

            if self.use_gate:
                # Compute modal confidence
                img_conf = compute_modal_confidence(cross_feats, mode='variance')
                pcd_conf = compute_modal_confidence(bev_feats, mode='variance')

                # Gated fusion
                fused_feats, gates = self.gate(
                    cross_feats, bev_feats, img_conf, pcd_conf
                )

                outputs['gates'] = gates
            else:
                # Simple averaging
                fused_feats = 0.5 * cross_feats + 0.5 * bev_feats

        elif self.use_cross_attn:
            # Only cross-attention path
            fused_feats = self.cross_attn(
                pcd_feats, img_feats, points, uv, K, T, valid_mask
            )

        elif self.use_bev:
            # Only BEV path
            fused_feats = self.bev_fusion(points, pcd_feats)

        else:
            # Fallback: only point cloud features
            fused_feats = pcd_feats

        # Segmentation
        logits = self.seg_head(fused_feats)

        outputs['logits'] = logits

        if return_intermediates:
            outputs['features'] = fused_feats
            outputs['img_features'] = img_feats
            outputs['pcd_features'] = pcd_feats

        return outputs

    def predict(
        self,
        images: torch.Tensor,
        points: torch.Tensor,
        uv: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference method

        Returns:
            pred_labels: (B, N) predicted class labels
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                images, points, uv, K, T, valid_mask, return_intermediates=False
            )
            logits = outputs['logits']
            pred_labels = torch.argmax(logits, dim=-1)

        return pred_labels


def build_c3fuse(config: dict) -> nn.Module:
    """Build C3-Fuse model from config"""
    return C3FuseNet(config)
