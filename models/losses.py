"""
Loss functions for C3-Fuse training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B*N, C) logits or probabilities
            target: (B*N,) labels

        Returns:
            loss: scalar dice loss
        """
        num_classes = pred.shape[1]
        pred_probs = F.softmax(pred, dim=1)

        # One-hot encode targets
        target_one_hot = F.one_hot(target, num_classes).float()

        # Compute dice per class
        intersection = (pred_probs * target_one_hot).sum(dim=0)
        union = pred_probs.sum(dim=0) + target_one_hot.sum(dim=0)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class ContrastiveLoss(nn.Module):
    """Cross-modal contrastive loss (InfoNCE)"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        feat_3d: torch.Tensor,
        feat_2d: torch.Tensor,
        positive_pairs: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            feat_3d: (B, N, C) 3D features
            feat_2d: (B, N, C) 2D features (lifted or projected)
            positive_pairs: (B, N) indices of positive pairs (optional)

        Returns:
            loss: contrastive loss
        """
        B, N, C = feat_3d.shape

        # Normalize features
        feat_3d = F.normalize(feat_3d, dim=-1)
        feat_2d = F.normalize(feat_2d, dim=-1)

        # Flatten
        feat_3d_flat = feat_3d.reshape(B * N, C)
        feat_2d_flat = feat_2d.reshape(B * N, C)

        # Similarity matrix
        sim_matrix = torch.matmul(feat_3d_flat, feat_2d_flat.T) / self.temperature

        # Positive pairs (diagonal by default)
        if positive_pairs is None:
            labels = torch.arange(B * N, device=feat_3d.device)
        else:
            labels = positive_pairs.reshape(-1)

        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class ProjectionConsistencyLoss(nn.Module):
    """Projection consistency between 2D and 3D predictions"""

    def __init__(self, ignore_unlabeled: bool = True):
        super().__init__()
        self.ignore_unlabeled = ignore_unlabeled
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(
        self,
        pred_3d: torch.Tensor,
        pred_2d: torch.Tensor,
        uv: torch.Tensor,
        valid_mask: torch.Tensor,
        image_shape: tuple
    ) -> torch.Tensor:
        """
        Args:
            pred_3d: (B, N, C) 3D predictions (logits)
            pred_2d: (B, C, H, W) 2D predictions (logits)
            uv: (B, N, 2) projected coordinates (normalized to [-1, 1])
            valid_mask: (B, N) visibility mask
            image_shape: (H, W)

        Returns:
            loss: projection consistency loss
        """
        B, N, C = pred_3d.shape

        # Sample 2D predictions at projected locations
        pred_2d_sampled = F.grid_sample(
            pred_2d,
            uv.view(B, N, 1, 2),
            mode='bilinear',
            align_corners=False
        )  # (B, C, N, 1)

        pred_2d_sampled = pred_2d_sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, C)

        # Get 3D labels
        labels_3d = torch.argmax(pred_3d, dim=-1)  # (B, N)

        # Compute consistency loss
        pred_2d_flat = pred_2d_sampled[valid_mask]  # (M, C)
        labels_3d_flat = labels_3d[valid_mask]  # (M,)

        if pred_2d_flat.shape[0] == 0:
            return torch.tensor(0.0, device=pred_3d.device)

        loss = self.ce_loss(pred_2d_flat, labels_3d_flat)

        return loss


class GeometricConsistencyLoss(nn.Module):
    """Geometric plane consistency loss"""

    def __init__(
        self,
        lambda_smooth: float = 0.1,
        min_points_per_plane: int = 50
    ):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.min_points_per_plane = min_points_per_plane

    def forward(
        self,
        points: torch.Tensor,
        pred_labels: torch.Tensor,
        normals: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) point coordinates
            pred_labels: (B, N) predicted plane labels
            normals: (B, N, 3) estimated normals (optional)

        Returns:
            loss: geometric consistency loss
        """
        B, N, _ = points.shape
        device = points.device

        total_loss = 0.0
        num_planes = 0

        for b in range(B):
            pts = points[b]  # (N, 3)
            labels = pred_labels[b]  # (N,)

            unique_labels = torch.unique(labels)

            for label in unique_labels:
                if label == 0:  # Skip background
                    continue

                mask = (labels == label)
                if mask.sum() < self.min_points_per_plane:
                    continue

                plane_pts = pts[mask]  # (M, 3)

                # Fit plane using SVD
                centroid = plane_pts.mean(dim=0)
                centered = plane_pts - centroid

                U, S, V = torch.svd(centered)
                normal = V[:, -1]  # Last column is normal

                # Point-to-plane distance
                d = -torch.dot(normal, centroid)
                distances = torch.abs(torch.matmul(plane_pts, normal) + d)

                # Plane fitting loss
                plane_loss = distances.mean()

                total_loss += plane_loss
                num_planes += 1

        if num_planes == 0:
            return torch.tensor(0.0, device=device)

        # Average over planes
        total_loss = total_loss / num_planes

        # Normal smoothness (if provided)
        if normals is not None and self.lambda_smooth > 0:
            # Compute normal variation
            smooth_loss = 0.0
            # Simple L1 smoothness within each plane
            for b in range(B):
                n = normals[b]  # (N, 3)
                labels = pred_labels[b]

                unique_labels = torch.unique(labels)
                for label in unique_labels:
                    if label == 0:
                        continue

                    mask = (labels == label)
                    if mask.sum() < 2:
                        continue

                    plane_normals = n[mask]
                    mean_normal = plane_normals.mean(dim=0)
                    smooth_loss += (plane_normals - mean_normal).abs().mean()

            total_loss += self.lambda_smooth * smooth_loss

        return total_loss


class GateRegularizationLoss(nn.Module):
    """Regularization loss for gating mechanism"""

    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_sparsity: float = 0.05,
        lambda_balance: float = 0.1
    ):
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.lambda_sparsity = lambda_sparsity
        self.lambda_balance = lambda_balance

    def forward(self, gates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gates: (B, N, 1) gate values in [0, 1]

        Returns:
            loss: gate regularization loss
        """
        B, N, _ = gates.shape

        # Entropy regularization (encourage decisiveness)
        # H(p) = -[p*log(p) + (1-p)*log(1-p)]
        eps = 1e-8
        entropy = -(
            gates * torch.log(gates + eps) +
            (1 - gates) * torch.log(1 - gates + eps)
        )
        entropy_loss = entropy.mean()

        # Sparsity regularization (optional, encourage using fewer gates)
        sparsity_loss = gates.mean()

        # Balance regularization (prevent mode collapse)
        mean_gate = gates.mean()
        balance_loss = (mean_gate - 0.5) ** 2

        # Total loss
        loss = (
            self.lambda_entropy * entropy_loss +
            self.lambda_sparsity * sparsity_loss +
            self.lambda_balance * balance_loss
        )

        return loss


class C3FuseLoss(nn.Module):
    """
    Combined loss for C3-Fuse training
    """

    def __init__(self, config: dict):
        super().__init__()

        loss_config = config['loss']

        # Loss weights
        self.w_seg = loss_config['w_seg']
        self.w_cm = loss_config['w_cm']
        self.w_proj = loss_config['w_proj']
        self.w_plane = loss_config['w_plane']
        self.w_gate = loss_config['w_gate']

        # Loss modules
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=loss_config['temperature'])
        self.proj_consistency_loss = ProjectionConsistencyLoss()
        self.geo_consistency_loss = GeometricConsistencyLoss(
            lambda_smooth=loss_config['lambda_smooth']
        )
        self.gate_reg_loss = GateRegularizationLoss(
            lambda_sparsity=loss_config['lambda_sparse']
        )

    def forward(
        self,
        outputs: dict,
        targets: dict
    ) -> tuple:
        """
        Args:
            outputs: dict from model forward pass
            targets: dict with ground truth data

        Returns:
            total_loss: scalar total loss
            loss_dict: dict of individual loss components
        """
        loss_dict = {}

        # Segmentation loss
        logits = outputs['logits']  # (B, N, C)
        gt_labels = targets['labels']  # (B, N)

        B, N, C = logits.shape
        logits_flat = logits.reshape(B * N, C)
        labels_flat = gt_labels.reshape(B * N)

        loss_ce = self.ce_loss(logits_flat, labels_flat)
        loss_dice = self.dice_loss(logits_flat, labels_flat)
        loss_seg = loss_ce + loss_dice

        loss_dict['seg_ce'] = loss_ce.item()
        loss_dict['seg_dice'] = loss_dice.item()
        loss_dict['seg_total'] = loss_seg.item()

        total_loss = self.w_seg * loss_seg

        # Contrastive loss (if features available)
        if 'pcd_features' in outputs and 'img_features' in targets:
            loss_cm = self.contrastive_loss(
                outputs['pcd_features'],
                targets['img_features']
            )
            loss_dict['contrastive'] = loss_cm.item()
            total_loss += self.w_cm * loss_cm

        # Projection consistency (if 2D predictions available)
        if 'pred_2d' in targets:
            loss_proj = self.proj_consistency_loss(
                logits,
                targets['pred_2d'],
                targets['uv'],
                targets['valid_mask'],
                targets['image_shape']
            )
            loss_dict['proj_consistency'] = loss_proj.item()
            total_loss += self.w_proj * loss_proj

        # Geometric consistency
        if 'points' in targets:
            pred_labels = torch.argmax(logits, dim=-1)
            loss_geo = self.geo_consistency_loss(
                targets['points'],
                pred_labels
            )
            loss_dict['geometric'] = loss_geo.item()
            total_loss += self.w_plane * loss_geo

        # Gate regularization
        if 'gates' in outputs:
            loss_gate = self.gate_reg_loss(outputs['gates'])
            loss_dict['gate_reg'] = loss_gate.item()
            total_loss += self.w_gate * loss_gate

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


def build_loss(config: dict) -> nn.Module:
    """Build loss function from config"""
    return C3FuseLoss(config)
