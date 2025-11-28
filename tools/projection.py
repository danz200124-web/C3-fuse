"""
Projection and visibility utilities
Handles point-to-pixel projection, visibility computation, and feature sampling
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import open3d as o3d


def project_points_to_image(
    points: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    image_shape: Tuple[int, int],
    return_depth: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates

    Args:
        points: (N, 3) point coordinates in world frame
        K: (3, 3) camera intrinsic matrix
        T: (4, 4) camera extrinsic matrix (world -> camera)
        image_shape: (H, W) image dimensions
        return_depth: whether to return depth values

    Returns:
        uv: (N, 2) pixel coordinates
        depth: (N,) depth values
        valid_mask: (N,) boolean mask for points inside image
    """
    N = points.shape[0]

    # Homogeneous coordinates
    points_h = np.hstack([points, np.ones((N, 1))])  # (N, 4)

    # Transform to camera frame
    points_cam = (T @ points_h.T).T[:, :3]  # (N, 3)

    # Get depth
    depth = points_cam[:, 2]  # (N,)

    # Project to image plane
    points_img = (K @ points_cam.T).T  # (N, 3)
    uv = points_img[:, :2] / (points_img[:, 2:3] + 1e-8)  # (N, 2)

    # Check validity (inside image and positive depth)
    H, W = image_shape
    valid_mask = (
        (uv[:, 0] >= 0) & (uv[:, 0] < W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < H) &
        (depth > 0)
    )

    if return_depth:
        return uv, depth, valid_mask
    return uv, valid_mask


def compute_visibility(
    points: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    cameras: List[Tuple[np.ndarray, np.ndarray]],
    image_shape: Tuple[int, int],
    use_zbuffer: bool = True
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute visibility of points from multiple camera viewpoints

    Args:
        points: (N, 3) point coordinates
        mesh: reconstructed mesh for occlusion testing
        cameras: List of (K, T) camera parameters
        image_shape: (H, W)
        use_zbuffer: use Z-buffer for occlusion testing

    Returns:
        visibility_count: (N,) number of views each point is visible from
        visible_views: List of (M,) indices of visible cameras per point
    """
    N = points.shape[0]
    num_cameras = len(cameras)

    visibility_count = np.zeros(N, dtype=np.int32)
    visible_views = [[] for _ in range(N)]

    if use_zbuffer and mesh is not None:
        # Create raycasting scene
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    for cam_idx, (K, T) in enumerate(cameras):
        # Project points
        uv, depth, valid_mask = project_points_to_image(
            points, K, T, image_shape, return_depth=True
        )

        if use_zbuffer and mesh is not None:
            # Ray casting for occlusion test
            cam_center = np.linalg.inv(T)[:3, 3]
            rays = points[valid_mask] - cam_center
            rays = rays / (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-8)

            # Cast rays
            origins = np.tile(cam_center, (valid_mask.sum(), 1)).astype(np.float32)
            directions = rays.astype(np.float32)

            rays_t = o3d.core.Tensor(
                np.hstack([origins, directions]), dtype=o3d.core.Dtype.Float32
            )
            result = scene.cast_rays(rays_t)
            hit_depth = result['t_hit'].numpy()

            # Check if hit depth matches point depth (with tolerance)
            point_depth = depth[valid_mask]
            occlusion_mask = np.abs(hit_depth - point_depth) < 0.1  # 10cm tolerance

            # Update visibility for non-occluded points
            visible_indices = np.where(valid_mask)[0][occlusion_mask]
        else:
            # Simple visibility without occlusion test
            visible_indices = np.where(valid_mask)[0]

        # Update visibility count and view lists
        visibility_count[visible_indices] += 1
        for idx in visible_indices:
            visible_views[idx].append(cam_idx)

    # Convert lists to arrays
    visible_views = [np.array(v) for v in visible_views]

    return visibility_count, visible_views


def sample_image_features(
    features: torch.Tensor,
    uv: torch.Tensor,
    k: int = 5,
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Sample image features at projected point locations with k-NN neighbors

    Args:
        features: (B, C, H, W) image features
        uv: (B, N, 2) pixel coordinates (normalized to [-1, 1])
        k: number of nearest neighbors to sample
        mode: interpolation mode ('bilinear' or 'nearest')

    Returns:
        sampled_features: (B, N, k, C) sampled features
    """
    B, C, H, W = features.shape
    N = uv.shape[1]

    if k == 1:
        # Simple bilinear sampling
        sampled = F.grid_sample(
            features,
            uv.view(B, N, 1, 2),
            mode=mode,
            align_corners=False
        )  # (B, C, N, 1)
        return sampled.permute(0, 2, 3, 1)  # (B, N, 1, C)

    else:
        # Sample k nearest neighbors in a local window
        offset_range = int(np.sqrt(k))
        offsets = []
        for dy in range(-offset_range, offset_range + 1):
            for dx in range(-offset_range, offset_range + 1):
                if len(offsets) >= k:
                    break
                offsets.append([dx / W * 2, dy / H * 2])  # Normalized offsets
        offsets = torch.tensor(offsets[:k], device=uv.device, dtype=uv.dtype)

        # Add offsets to uv coordinates
        uv_neighbors = uv.unsqueeze(2) + offsets.view(1, 1, k, 2)  # (B, N, k, 2)

        # Sample features
        sampled = F.grid_sample(
            features,
            uv_neighbors.reshape(B, N * k, 1, 2),
            mode=mode,
            align_corners=False
        )  # (B, C, N*k, 1)

        sampled = sampled.squeeze(-1).view(B, C, N, k)
        return sampled.permute(0, 2, 3, 1)  # (B, N, k, C)


def compute_geometric_bias(
    points: torch.Tensor,
    uv: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    dim: int = 16
) -> torch.Tensor:
    """
    Compute geometric bias for cross-attention based on:
    - Viewing angle (theta)
    - Distance (rho)
    - Elevation (e)

    Args:
        points: (N, 3) 3D points
        uv: (N, 2) projected pixel coordinates
        K: (3, 3) camera intrinsic
        T: (4, 4) camera extrinsic
        dim: dimension of bias encoding

    Returns:
        bias: (N, dim) geometric bias encoding
    """
    N = points.shape[0]
    device = points.device

    # Camera center
    cam_center = torch.inverse(T)[:3, 3]  # (3,)

    # Vector from camera to point
    vec = points - cam_center  # (N, 3)
    distance = torch.norm(vec, dim=1, keepdim=True)  # (N, 1)

    # Viewing direction (normalized)
    view_dir = vec / (distance + 1e-8)  # (N, 3)

    # Viewing angle (with respect to camera forward direction)
    cam_forward = T[:3, 2]  # Camera looks along +Z
    cos_theta = torch.sum(view_dir * cam_forward, dim=1, keepdim=True)  # (N, 1)

    # Elevation angle
    elevation = torch.atan2(view_dir[:, 2:3],
                           torch.norm(view_dir[:, :2], dim=1, keepdim=True))  # (N, 1)

    # Positional encoding
    freqs = torch.arange(0, dim // 6, device=device, dtype=torch.float32)
    freqs = 2.0 ** freqs

    # Encode distance, angle, elevation
    feats = []
    for val in [distance / 10.0, cos_theta, elevation]:  # Normalize distance
        val_enc = val * freqs.view(1, -1) * np.pi
        feats.append(torch.sin(val_enc))
        feats.append(torch.cos(val_enc))

    bias = torch.cat(feats, dim=1)  # (N, dim)

    # Pad if necessary
    if bias.shape[1] < dim:
        padding = torch.zeros(N, dim - bias.shape[1], device=device)
        bias = torch.cat([bias, padding], dim=1)

    return bias[:, :dim]


def normalize_uv_coordinates(
    uv: np.ndarray,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Normalize pixel coordinates from [0, W/H] to [-1, 1]

    Args:
        uv: (N, 2) pixel coordinates
        image_shape: (H, W)

    Returns:
        uv_norm: (N, 2) normalized coordinates
    """
    H, W = image_shape
    uv_norm = uv.copy()
    uv_norm[:, 0] = (uv[:, 0] / W) * 2 - 1  # x
    uv_norm[:, 1] = (uv[:, 1] / H) * 2 - 1  # y
    return uv_norm
