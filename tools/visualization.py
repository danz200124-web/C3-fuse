"""
Visualization utilities for C3-Fuse
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import open3d as o3d
from typing import Optional, Tuple, List
import matplotlib


def visualize_projection(
    image: np.ndarray,
    points: np.ndarray,
    uv: np.ndarray,
    colors: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize point cloud projection on image

    Args:
        image: (H, W, 3) RGB image
        points: (N, 3) 3D points
        uv: (N, 2) projected pixel coordinates
        colors: (N, 3) point colors, optional
        depth: (N,) depth values for color coding
        valid_mask: (N,) boolean mask
        output_path: save path, optional

    Returns:
        vis_image: visualization result
    """
    vis_image = image.copy()
    H, W = image.shape[:2]

    if valid_mask is None:
        valid_mask = np.ones(len(points), dtype=bool)

    valid_uv = uv[valid_mask]
    valid_depth = depth[valid_mask] if depth is not None else None

    # Color by depth if no colors provided
    if colors is None and valid_depth is not None:
        # Normalize depth to [0, 1]
        d_min, d_max = np.percentile(valid_depth, [5, 95])
        depth_norm = np.clip((valid_depth - d_min) / (d_max - d_min + 1e-8), 0, 1)

        # Colormap
        cmap = plt.get_cmap('jet')
        colors = (cmap(depth_norm)[:, :3] * 255).astype(np.uint8)
    elif colors is None:
        colors = np.tile([0, 255, 0], (len(valid_uv), 1))
    else:
        colors = colors[valid_mask]

    # Draw points
    for (u, v), color in zip(valid_uv.astype(int), colors):
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(vis_image, (u, v), 2, color.tolist(), -1)

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    return vis_image


def visualize_bev(
    grid: np.ndarray,
    occupancy: Optional[np.ndarray] = None,
    title: str = "BEV Grid",
    output_path: Optional[str] = None
) -> None:
    """
    Visualize cylindrical BEV grid

    Args:
        grid: (Nr, Nt, Nz, C) or (Nr, Nt, Nz) grid
        occupancy: (Nr, Nt, Nz) occupancy mask, optional
        title: plot title
        output_path: save path
    """
    if grid.ndim == 4:
        # Take max along feature dimension and z
        grid_2d = np.max(grid, axis=(2, 3))
    elif grid.ndim == 3:
        # Take max along z
        grid_2d = np.max(grid, axis=2)
    else:
        grid_2d = grid

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})

    Nr, Nt = grid_2d.shape
    theta = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
    r = np.arange(Nr)

    theta_grid, r_grid = np.meshgrid(theta, r)

    c = ax.pcolormesh(theta_grid, r_grid, grid_2d, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax)

    ax.set_title(title)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_stereonet(
    normals: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Stereonet",
    output_path: Optional[str] = None
) -> None:
    """
    Plot stereonet (equal-area lower hemisphere projection) for structural geology

    Args:
        normals: (N, 3) unit normal vectors
        labels: (N,) cluster labels, optional
        title: plot title
        output_path: save path
    """
    # Convert normals to dip direction and dip angle
    # Normal points upward from plane, so flip if needed
    normals = normals.copy()
    normals[:, 2] = np.abs(normals[:, 2])  # Project to lower hemisphere

    # Normalize
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    # Convert to dip direction (azimuth) and dip
    dip_direction = np.arctan2(normals[:, 0], normals[:, 1])  # Azimuth
    dip_direction = (dip_direction + 2 * np.pi) % (2 * np.pi)

    dip = np.arccos(np.clip(normals[:, 2], 0, 1))  # Angle from vertical

    # Equal-area projection (Schmidt net)
    r = np.sqrt(2) * np.sin(dip / 2)
    x = r * np.sin(dip_direction)
    y = r * np.cos(dip_direction)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                dip_direction[mask], r[mask],
                c=[colors[i]], label=f'Set {label}',
                s=50, alpha=0.6, edgecolors='black'
            )
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
    else:
        ax.scatter(dip_direction, r, s=50, alpha=0.6, edgecolors='black')

    # Format
    ax.set_ylim(0, np.sqrt(2))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=20)
    ax.grid(True)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    window_name: str = "Point Cloud"
) -> None:
    """
    Visualize point cloud with Open3D

    Args:
        points: (N, 3) point coordinates
        colors: (N, 3) RGB colors in [0, 1], optional
        normals: (N, 3) normal vectors, optional
        window_name: window title
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def create_overlay_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Create overlay of segmentation mask on image

    Args:
        image: (H, W, 3) RGB image
        mask: (H, W) binary mask
        alpha: overlay transparency
        color: RGB color for mask

    Returns:
        overlay: blended image
    """
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    return overlay


def plot_training_curves(
    history: dict,
    output_path: Optional[str] = None
) -> None:
    """
    Plot training curves

    Args:
        history: dict with keys like 'train_loss', 'val_loss', etc.
        output_path: save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    if 'train_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # mIoU
    if 'train_miou' in history:
        axes[0, 1].plot(history['train_miou'], label='Train')
    if 'val_miou' in history:
        axes[0, 1].plot(history['val_miou'], label='Val')
    axes[0, 1].set_title('mIoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1
    if 'train_f1' in history:
        axes[1, 0].plot(history['train_f1'], label='Train')
    if 'val_f1' in history:
        axes[1, 0].plot(history['val_f1'], label='Val')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_attention_weights(
    image: np.ndarray,
    points: np.ndarray,
    uv: np.ndarray,
    attention_weights: np.ndarray,
    point_idx: int,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize cross-attention weights for a specific point

    Args:
        image: (H, W, 3) RGB image
        points: (N, 3) 3D points
        uv: (N, 2) projected coordinates
        attention_weights: (N, K) attention weights for K pixels per point
        point_idx: index of point to visualize
        output_path: save path

    Returns:
        vis_image: visualization
    """
    vis_image = image.copy().astype(np.float32) / 255.0

    # Get attention weights for this point
    weights = attention_weights[point_idx]  # (K,)

    # Highlight attended pixels
    point_uv = uv[point_idx].astype(int)

    # Create heatmap
    heatmap = np.zeros((image.shape[0], image.shape[1]))

    # Simple visualization: mark the point location with weight
    if 0 <= point_uv[0] < image.shape[1] and 0 <= point_uv[1] < image.shape[0]:
        heatmap[point_uv[1], point_uv[0]] = weights.max()

    # Apply Gaussian blur for visualization
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Colormap
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]

    # Blend
    vis_image = 0.6 * vis_image + 0.4 * heatmap_colored

    vis_image = (vis_image * 255).astype(np.uint8)

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    return vis_image
