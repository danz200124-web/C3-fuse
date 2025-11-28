"""
Cylindrical BEV grid utilities
Handles conversion between Cartesian and cylindrical coordinates,
and operations on cylindrical voxel grids
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def points_to_cylindrical(
    points: np.ndarray,
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert Cartesian (x, y, z) to cylindrical (r, theta, z) coordinates

    Args:
        points: (N, 3) Cartesian coordinates
        center: (3,) origin for cylindrical system, defaults to [0, 0, 0]

    Returns:
        cyl_coords: (N, 3) cylindrical coordinates [r, theta, z]
                    theta in [0, 2*pi], r >= 0
    """
    if center is not None:
        points = points - center

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)  # [-pi, pi]
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # [0, 2*pi]

    return np.stack([r, theta, z], axis=1)


def cylindrical_to_cartesian(
    cyl_coords: np.ndarray,
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert cylindrical (r, theta, z) to Cartesian (x, y, z) coordinates

    Args:
        cyl_coords: (N, 3) cylindrical coordinates [r, theta, z]
        center: (3,) origin for cylindrical system

    Returns:
        points: (N, 3) Cartesian coordinates
    """
    r, theta, z = cyl_coords[:, 0], cyl_coords[:, 1], cyl_coords[:, 2]

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    points = np.stack([x, y, z], axis=1)

    if center is not None:
        points = points + center

    return points


class CylindricalGrid:
    """
    Cylindrical BEV grid for unified spatial representation
    """

    def __init__(
        self,
        radial_bins: int = 128,
        theta_bins: int = 256,
        z_bins: int = 64,
        r_min: float = 0.0,
        r_max: float = 50.0,
        z_min: float = -10.0,
        z_max: float = 10.0,
        center: Optional[np.ndarray] = None
    ):
        """
        Args:
            radial_bins: number of radial bins
            theta_bins: number of angular bins
            z_bins: number of vertical bins
            r_min/r_max: radial range
            z_min/z_max: vertical range
            center: (3,) origin of cylindrical coordinate system
        """
        self.Nr = radial_bins
        self.Nt = theta_bins
        self.Nz = z_bins

        self.r_min = r_min
        self.r_max = r_max
        self.z_min = z_min
        self.z_max = z_max

        self.center = center if center is not None else np.zeros(3)

        # Bin edges
        self.r_edges = np.linspace(r_min, r_max, radial_bins + 1)
        self.theta_edges = np.linspace(0, 2 * np.pi, theta_bins + 1)
        self.z_edges = np.linspace(z_min, z_max, z_bins + 1)

        # Bin centers
        self.r_centers = (self.r_edges[:-1] + self.r_edges[1:]) / 2
        self.theta_centers = (self.theta_edges[:-1] + self.theta_edges[1:]) / 2
        self.z_centers = (self.z_edges[:-1] + self.z_edges[1:]) / 2

    def points_to_grid_indices(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert points to grid indices

        Args:
            points: (N, 3) Cartesian coordinates

        Returns:
            indices: (N, 3) grid indices [ir, it, iz]
            valid_mask: (N,) boolean mask for points within grid bounds
        """
        # Convert to cylindrical
        cyl_coords = points_to_cylindrical(points, self.center)
        r, theta, z = cyl_coords[:, 0], cyl_coords[:, 1], cyl_coords[:, 2]

        # Compute indices
        ir = np.digitize(r, self.r_edges) - 1
        it = np.digitize(theta, self.theta_edges) - 1
        iz = np.digitize(z, self.z_edges) - 1

        # Handle theta wrap-around
        it = it % self.Nt

        # Check validity
        valid_mask = (
            (ir >= 0) & (ir < self.Nr) &
            (it >= 0) & (it < self.Nt) &
            (iz >= 0) & (iz < self.Nz)
        )

        indices = np.stack([ir, it, iz], axis=1)
        return indices, valid_mask

    def grid_indices_to_points(
        self,
        indices: np.ndarray
    ) -> np.ndarray:
        """
        Convert grid indices to point coordinates (using bin centers)

        Args:
            indices: (N, 3) grid indices [ir, it, iz]

        Returns:
            points: (N, 3) Cartesian coordinates
        """
        ir, it, iz = indices[:, 0], indices[:, 1], indices[:, 2]

        r = self.r_centers[ir]
        theta = self.theta_centers[it]
        z = self.z_centers[iz]

        cyl_coords = np.stack([r, theta, z], axis=1)
        return cylindrical_to_cartesian(cyl_coords, self.center)

    def voxelize_points(
        self,
        points: np.ndarray,
        features: Optional[np.ndarray] = None,
        pooling: str = 'max'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Voxelize point cloud into cylindrical grid

        Args:
            points: (N, 3) point coordinates
            features: (N, C) point features, optional
            pooling: 'max', 'avg', or 'sum'

        Returns:
            grid: (Nr, Nt, Nz, C) voxelized grid
            occupancy: (Nr, Nt, Nz) occupancy mask
        """
        indices, valid_mask = self.points_to_grid_indices(points)

        valid_points = points[valid_mask]
        valid_indices = indices[valid_mask]

        if features is not None:
            valid_features = features[valid_mask]
            C = features.shape[1]
        else:
            # Use constant feature if none provided
            valid_features = np.ones((valid_mask.sum(), 1))
            C = 1

        # Initialize grid
        grid = np.zeros((self.Nr, self.Nt, self.Nz, C), dtype=np.float32)
        occupancy = np.zeros((self.Nr, self.Nt, self.Nz), dtype=np.float32)

        if pooling == 'max':
            grid.fill(-np.inf)

        # Fill grid
        for idx, feat in zip(valid_indices, valid_features):
            ir, it, iz = idx
            if pooling == 'max':
                grid[ir, it, iz] = np.maximum(grid[ir, it, iz], feat)
            elif pooling == 'avg' or pooling == 'sum':
                grid[ir, it, iz] += feat
            occupancy[ir, it, iz] += 1

        # Average pooling normalization
        if pooling == 'avg':
            mask = occupancy[..., None] > 0
            grid = np.where(mask, grid / (occupancy[..., None] + 1e-8), 0)

        # Handle empty voxels for max pooling
        if pooling == 'max':
            grid = np.where(occupancy[..., None] > 0, grid, 0)

        return grid, occupancy > 0

    def interpolate_features(
        self,
        grid: np.ndarray,
        points: np.ndarray,
        mode: str = 'trilinear'
    ) -> np.ndarray:
        """
        Interpolate features from grid to points

        Args:
            grid: (Nr, Nt, Nz, C) feature grid
            points: (N, 3) query points
            mode: 'trilinear' or 'nearest'

        Returns:
            features: (N, C) interpolated features
        """
        # Convert to cylindrical and normalize to [0, 1]
        cyl_coords = points_to_cylindrical(points, self.center)
        r, theta, z = cyl_coords[:, 0], cyl_coords[:, 1], cyl_coords[:, 2]

        r_norm = (r - self.r_min) / (self.r_max - self.r_min)
        theta_norm = theta / (2 * np.pi)
        z_norm = (z - self.z_min) / (self.z_max - self.z_min)

        # Convert to grid coordinates
        coords = np.stack([
            r_norm * (self.Nr - 1),
            theta_norm * (self.Nt - 1),
            z_norm * (self.Nz - 1)
        ], axis=1)

        N, C = points.shape[0], grid.shape[3]
        features = np.zeros((N, C), dtype=np.float32)

        if mode == 'nearest':
            # Nearest neighbor interpolation
            ir = np.clip(np.round(coords[:, 0]).astype(int), 0, self.Nr - 1)
            it = np.clip(np.round(coords[:, 1]).astype(int), 0, self.Nt - 1)
            iz = np.clip(np.round(coords[:, 2]).astype(int), 0, self.Nz - 1)
            features = grid[ir, it, iz]

        elif mode == 'trilinear':
            # Trilinear interpolation
            ir0 = np.clip(np.floor(coords[:, 0]).astype(int), 0, self.Nr - 2)
            it0 = np.clip(np.floor(coords[:, 1]).astype(int), 0, self.Nt - 2)
            iz0 = np.clip(np.floor(coords[:, 2]).astype(int), 0, self.Nz - 2)

            ir1 = ir0 + 1
            it1 = (it0 + 1) % self.Nt  # Wrap around for theta
            iz1 = iz0 + 1

            # Interpolation weights
            wr = coords[:, 0] - ir0
            wt = coords[:, 1] - it0
            wz = coords[:, 2] - iz0

            # 8 corners
            for i, dr in enumerate([0, 1]):
                for j, dt in enumerate([0, 1]):
                    for k, dz in enumerate([0, 1]):
                        ir = ir0 + dr
                        it = it0 + dt if dt == 0 else it1
                        iz = iz0 + dz

                        w = (
                            (wr if dr else (1 - wr)) *
                            (wt if dt else (1 - wt)) *
                            (wz if dz else (1 - wz))
                        )
                        features += w[:, None] * grid[ir, it, iz]

        return features


class CylindricalBEVFusion(nn.Module):
    """
    Cylindrical BEV fusion module (PyTorch)
    """

    def __init__(
        self,
        grid_config: dict,
        in_channels: int,
        out_channels: int = 256,
        pooling: str = 'max'
    ):
        super().__init__()

        self.grid = CylindricalGrid(**grid_config)
        self.pooling = pooling
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Conv layers for BEV features
        self.conv_bev = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, 3, padding=1),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) point coordinates
            features: (B, N, C) point features

        Returns:
            bev_features: (B, N, out_channels) BEV-fused features
        """
        B, N, _ = points.shape
        device = points.device

        batch_grids = []

        for b in range(B):
            pts = points[b].cpu().numpy()  # (N, 3)
            feats = features[b].cpu().numpy()  # (N, C)

            # Voxelize
            grid, _ = self.grid.voxelize_points(pts, feats, self.pooling)
            batch_grids.append(grid)

        # Stack and convert to tensor
        grid_tensor = torch.from_numpy(
            np.stack(batch_grids, axis=0)
        ).to(device)  # (B, Nr, Nt, Nz, C)

        # Permute for conv3d: (B, C, Nr, Nt, Nz)
        grid_tensor = grid_tensor.permute(0, 4, 1, 2, 3)

        # Apply 3D convolution
        bev_feat_grid = self.conv_bev(grid_tensor)  # (B, out_C, Nr, Nt, Nz)

        # Interpolate back to points
        bev_features = []
        for b in range(B):
            pts = points[b].cpu().numpy()
            grid_np = bev_feat_grid[b].permute(1, 2, 3, 0).cpu().numpy()

            feats = self.grid.interpolate_features(grid_np, pts, mode='trilinear')
            bev_features.append(torch.from_numpy(feats).to(device))

        bev_features = torch.stack(bev_features, dim=0)  # (B, N, out_channels)

        return bev_features
