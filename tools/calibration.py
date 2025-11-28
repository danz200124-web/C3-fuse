"""
Camera calibration utilities
Zhang's method for intrinsic calibration and image undistortion
"""

import numpy as np
import cv2
import glob
import yaml
from pathlib import Path
from typing import Tuple, List, Dict, Optional


def calibrate_camera(
    images_path: str,
    board_size: Tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    output_path: Optional[str] = None,
    visualize: bool = False
) -> Dict:
    """
    Calibrate camera using Zhang's method with checkerboard pattern

    Args:
        images_path: path pattern for calibration images (e.g., "calib/*.jpg")
        board_size: (cols, rows) number of internal corners
        square_size: size of checkerboard square in meters
        output_path: path to save calibration results (YAML)
        visualize: whether to visualize detected corners

    Returns:
        calib_data: dict with K, dist_coeffs, rvecs, tvecs, rms
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Find images
    image_files = sorted(glob.glob(images_path))
    print(f"Found {len(image_files)} calibration images")

    if len(image_files) < 10:
        print("Warning: At least 10-20 images recommended for good calibration")

    img_shape = None
    successful = 0

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (width, height)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)

            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)

            successful += 1

            if visualize:
                cv2.drawChessboardCorners(img, board_size, corners_refined, ret)
                cv2.imshow('Calibration', img)
                cv2.waitKey(100)

    if visualize:
        cv2.destroyAllWindows()

    print(f"Successfully detected corners in {successful}/{len(image_files)} images")

    if successful < 5:
        raise ValueError("Too few successful detections for calibration")

    # Calibrate camera
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    print(f"Calibration RMS reprojection error: {ret:.4f} pixels")

    calib_data = {
        'rms': float(ret),
        'image_width': int(img_shape[0]),
        'image_height': int(img_shape[1]),
        'camera_matrix': K.tolist(),
        'distortion_coefficients': dist.flatten().tolist(),
        'num_images': successful,
        'board_size': list(board_size),
        'square_size': float(square_size),
    }

    if output_path:
        with open(output_path, 'w') as f:
            yaml.dump(calib_data, f)
        print(f"Calibration saved to {output_path}")

    return calib_data


def undistort_images(
    input_path: str,
    output_path: str,
    intrinsics_file: str,
    crop: bool = False
) -> None:
    """
    Undistort images using calibrated intrinsics

    Args:
        input_path: path pattern for input images
        output_path: directory for output images
        intrinsics_file: path to YAML with calibration data
        crop: whether to crop to valid pixels only
    """
    # Load calibration
    intrinsics = load_intrinsics(intrinsics_file)
    K = intrinsics['K']
    dist = intrinsics['dist']
    img_size = (intrinsics['width'], intrinsics['height'])

    # Compute undistortion maps
    if crop:
        # Get optimal new camera matrix
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, dist, img_size, 1, img_size
        )
    else:
        new_K = K
        roi = None

    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, img_size, cv2.CV_32FC1
    )

    # Process images
    Path(output_path).mkdir(parents=True, exist_ok=True)
    image_files = sorted(glob.glob(input_path))

    print(f"Undistorting {len(image_files)} images...")

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            continue

        # Undistort
        undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        # Crop if needed
        if crop and roi is not None:
            x, y, w, h = roi
            undist = undist[y:y+h, x:x+w]

        # Save
        output_file = Path(output_path) / Path(fname).name
        cv2.imwrite(str(output_file), undist)

    print(f"Saved undistorted images to {output_path}")


def load_intrinsics(yaml_file: str) -> Dict:
    """
    Load camera intrinsics from YAML file

    Args:
        yaml_file: path to YAML file

    Returns:
        intrinsics: dict with K, dist, width, height
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    K = np.array(data['camera_matrix'], dtype=np.float64)
    dist = np.array(data['distortion_coefficients'], dtype=np.float64)

    return {
        'K': K,
        'dist': dist,
        'width': data['image_width'],
        'height': data['image_height'],
        'rms': data.get('rms', None),
    }


def compute_reprojection_error(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute reprojection errors for calibration validation

    Returns:
        mean_error: mean reprojection error across all points
        errors: per-image errors
    """
    errors = []

    for i in range(len(objpoints)):
        # Project 3D points
        imgpoints_proj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, dist
        )

        # Compute error
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
        errors.append(error)

    errors = np.array(errors)
    mean_error = np.mean(errors)

    return mean_error, errors


def estimate_extrinsics_from_checkerboard(
    image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    board_size: Tuple[int, int],
    square_size: float
) -> Optional[np.ndarray]:
    """
    Estimate camera extrinsics from a single checkerboard image

    Returns:
        T: (4, 4) transformation matrix or None if detection fails
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    if not ret:
        return None

    # Refine
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Prepare object points
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Solve PnP
    ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)

    if not ret:
        return None

    # Convert to transformation matrix
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    return T
