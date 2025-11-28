"""
Simple data preprocessing script
Handles undistortion, downsampling, and normal estimation
"""

import argparse
import os
import sys
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import yaml

sys.path.append('..')
from tools.calibration import load_intrinsics, undistort_images


def process_images(input_dir: str, output_dir: str, intrinsics: dict = None):
    """Process images (undistortion)"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_files = sorted(Path(input_dir).glob('*.jpg')) + sorted(Path(input_dir).glob('*.png'))

    print(f"Processing {len(image_files)} images...")

    if intrinsics is not None:
        K = intrinsics['K']
        dist = intrinsics['dist']
        img_size = (intrinsics['width'], intrinsics['height'])

        # Compute undistortion maps
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, img_size, cv2.CV_32FC1
        )

    for img_path in tqdm(image_files):
        img = cv2.imread(str(img_path))

        if intrinsics is not None:
            # Undistort
            img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        # Save
        output_path = Path(output_dir) / img_path.name
        cv2.imwrite(str(output_path), img)

    print(f"Saved processed images to {output_dir}")


def process_pointcloud(input_path: str, output_path: str, voxel_size: float = 0.02, estimate_normals: bool = True):
    """Process point cloud"""
    print(f"Loading point cloud from {input_path}...")
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"Original: {len(pcd.points)} points")

    # Voxel downsampling
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"After downsampling: {len(pcd.points)} points")

    # Estimate normals
    if estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        print("Normals estimated")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess scene data')
    parser.add_argument('--scene', type=str, required=True, help='Scene directory')
    parser.add_argument('--tasks', type=str, default='undistort,pcd_voxel_down,pcd_est_normal',
                        help='Comma-separated tasks')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--voxel_size', type=float, default=0.02, help='Voxel size for downsampling')

    args = parser.parse_args()

    tasks = args.tasks.split(',')

    # Load intrinsics if needed
    intrinsics = None
    if 'undistort' in tasks:
        intrinsics_path = os.path.join(args.scene, 'meta', 'cam_intrinsics.yaml')
        if os.path.exists(intrinsics_path):
            intrinsics = load_intrinsics(intrinsics_path)
        else:
            print(f"Warning: Intrinsics not found at {intrinsics_path}")

    # Process images
    if 'undistort' in tasks:
        img_input = os.path.join(args.scene, 'images')
        img_output = os.path.join(args.out, 'images')

        if os.path.exists(img_input):
            process_images(img_input, img_output, intrinsics)

    # Process point cloud
    pcd_input = os.path.join(args.scene, 'pointclouds', 'scene.pcd')
    pcd_output = os.path.join(args.out, 'pointclouds', 'scene.pcd')

    if os.path.exists(pcd_input):
        do_voxel = 'pcd_voxel_down' in tasks
        do_normal = 'pcd_est_normal' in tasks

        voxel_size = args.voxel_size if do_voxel else 0

        process_pointcloud(pcd_input, pcd_output, voxel_size, do_normal)

    print("Preprocessing complete!")


if __name__ == '__main__':
    main()
