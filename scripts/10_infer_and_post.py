#!/usr/bin/env python3
"""
C3-Fuse Inference and Post-processing Script
Runs inference and extracts structural plane parameters
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.append('..')
from models.c3fuse import build_c3fuse
from tools.projection import project_points_to_image, normalize_uv_coordinates
from tools.calibration import load_intrinsics


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint"""
    model = build_c3fuse(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def ransac_plane_fit(points: np.ndarray, threshold: float = 0.05, max_iter: int = 1000):
    """
    Fit plane using RANSAC

    Args:
        points: (N, 3) point coordinates
        threshold: inlier distance threshold
        max_iter: maximum iterations

    Returns:
        normal: (3,) plane normal
        d: plane offset (ax + by + cz + d = 0)
        inliers: (N,) boolean mask of inliers
    """
    if len(points) < 3:
        return None, None, None

    best_inliers = None
    best_count = 0
    best_normal = None
    best_d = None

    for _ in range(max_iter):
        # Sample 3 random points
        idx = np.random.choice(len(points), 3, replace=False)
        sample = points[idx]

        # Compute plane
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)

        if np.linalg.norm(normal) < 1e-8:
            continue

        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, sample[0])

        # Compute inliers
        distances = np.abs(points @ normal + d)
        inliers = distances < threshold

        if inliers.sum() > best_count:
            best_count = inliers.sum()
            best_inliers = inliers
            best_normal = normal
            best_d = d

    return best_normal, best_d, best_inliers


def extract_plane_parameters(
    points: np.ndarray,
    labels: np.ndarray,
    min_points: int = 50
) -> list:
    """
    Extract plane parameters from segmented point cloud

    Returns:
        planes: list of dicts with keys:
            - normal: (3,) unit normal
            - d: offset
            - points: (M, 3) inlier points
            - centroid: (3,) plane centroid
            - area: estimated plane area
    """
    planes = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == 0:  # Skip background
            continue

        mask = (labels == label)
        if mask.sum() < min_points:
            continue

        plane_points = points[mask]

        # RANSAC plane fitting
        normal, d, inliers = ransac_plane_fit(plane_points)

        if normal is None:
            continue

        inlier_points = plane_points[inliers]

        if len(inlier_points) < min_points:
            continue

        # Compute plane properties
        centroid = inlier_points.mean(axis=0)

        # Estimate area (convex hull projection)
        from scipy.spatial import ConvexHull
        try:
            # Project to 2D
            u = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
            u = u - np.dot(u, normal) * normal
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)

            points_2d = np.column_stack([
                (inlier_points - centroid) @ u,
                (inlier_points - centroid) @ v
            ])

            hull = ConvexHull(points_2d)
            area = hull.volume  # In 2D, volume is area

        except:
            area = 0.0

        # Dip and dip direction for structural geology
        # Normal points upward from plane
        dip_direction = np.degrees(np.arctan2(normal[0], normal[1])) % 360
        dip = np.degrees(np.arccos(np.abs(normal[2])))

        planes.append({
            'label': int(label),
            'normal': normal,
            'd': d,
            'centroid': centroid,
            'area': area,
            'num_points': len(inlier_points),
            'dip_direction': dip_direction,
            'dip': dip,
            'points': inlier_points
        })

    return planes


def save_plane_csv(planes: list, output_path: str):
    """Save plane parameters to CSV"""
    data = []
    for i, plane in enumerate(planes):
        data.append({
            'PlaneID': i + 1,
            'NumPoints': plane['num_points'],
            'Area_m2': plane['area'],
            'Normal_X': plane['normal'][0],
            'Normal_Y': plane['normal'][1],
            'Normal_Z': plane['normal'][2],
            'Offset_d': plane['d'],
            'Centroid_X': plane['centroid'][0],
            'Centroid_Y': plane['centroid'][1],
            'Centroid_Z': plane['centroid'][2],
            'DipDirection_deg': plane['dip_direction'],
            'Dip_deg': plane['dip']
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved plane parameters to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='C3-Fuse inference and post-processing')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--scene', type=str, required=True, help='Scene data directory')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--post', type=str, default='ransac_plane_fit',
                        choices=['ransac_plane_fit', 'none'], help='Post-processing method')
    parser.add_argument('--export_csv', type=str, default=None, help='Export CSV path')

    args = parser.parse_args()

    # Load configs (should be saved with checkpoint)
    config_path = os.path.join(os.path.dirname(args.ckpt), '..', 'config.yaml')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config not found at {config_path}, using default")
        config = {}  # Use default config

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.ckpt}...")
    model = load_model(args.ckpt, config, device)

    # Load scene data
    print(f"Loading scene from {args.scene}...")

    # Load point cloud
    pcd_path = os.path.join(args.scene, 'pointclouds', 'scene.pcd')
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # Load image
    import cv2
    img_path = os.path.join(args.scene, 'images', 'image_000.jpg')
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load camera parameters
    intrinsics = load_intrinsics(os.path.join(args.scene, 'meta', 'cam_intrinsics.yaml'))
    K = intrinsics['K']

    # Load extrinsics
    with open(os.path.join(args.scene, 'meta', 'extrinsics_refined.json'), 'r') as f:
        import json
        extrinsics = json.load(f)
        T = np.array(extrinsics['T'])

    # Project points to image
    print("Projecting points to image...")
    uv, depth, valid_mask = project_points_to_image(
        points, K, T, (image.shape[0], image.shape[1])
    )

    # Normalize uv
    uv_norm = normalize_uv_coordinates(uv, (image.shape[0], image.shape[1]))

    # Prepare batch
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    points_tensor = torch.from_numpy(points).unsqueeze(0).float()
    uv_tensor = torch.from_numpy(uv_norm).unsqueeze(0).float()
    K_tensor = torch.from_numpy(K).unsqueeze(0).float()
    T_tensor = torch.from_numpy(T).unsqueeze(0).float()
    valid_mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0)

    # Move to device
    image_tensor = image_tensor.to(device)
    points_tensor = points_tensor.to(device)
    uv_tensor = uv_tensor.to(device)
    K_tensor = K_tensor.to(device)
    T_tensor = T_tensor.to(device)
    valid_mask_tensor = valid_mask_tensor.to(device)

    # Inference
    print("Running inference...")
    with torch.no_grad():
        pred_labels = model.predict(
            image_tensor, points_tensor, uv_tensor, K_tensor, T_tensor, valid_mask_tensor
        )

    pred_labels = pred_labels.cpu().numpy()[0]  # (N,)

    # Save predictions
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(args.out, 'pred_labels.npy'), pred_labels)

    # Visualize
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points)

    # Color by labels
    colors = np.zeros((len(points), 3))
    colors[pred_labels == 1] = [1, 0, 0]  # Red for structural planes
    pcd_pred.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(os.path.join(args.out, 'pred_segmentation.ply'), pcd_pred)
    print(f"Saved segmentation to {args.out}/pred_segmentation.ply")

    # Post-processing
    if args.post == 'ransac_plane_fit':
        print("Extracting plane parameters...")
        planes = extract_plane_parameters(points, pred_labels, min_points=50)
        print(f"Found {len(planes)} planes")

        # Save normals
        normals = np.array([p['normal'] for p in planes])
        np.save(os.path.join(args.out, 'plane_normals.npy'), normals)

        # Save CSV
        if args.export_csv:
            save_plane_csv(planes, args.export_csv)
        else:
            save_plane_csv(planes, os.path.join(args.out, 'planes.csv'))

    print("Done!")


if __name__ == '__main__':
    main()
