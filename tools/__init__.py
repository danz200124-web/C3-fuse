"""
C3-Fuse Toolbox
Essential utilities for image-point cloud fusion
"""

from .projection import (
    project_points_to_image,
    compute_visibility,
    sample_image_features,
)
from .cyl_grid import (
    CylindricalGrid,
    points_to_cylindrical,
    cylindrical_to_cartesian,
)
from .calibration import (
    calibrate_camera,
    undistort_images,
    load_intrinsics,
)
from .visualization import (
    visualize_projection,
    visualize_bev,
    plot_stereonet,
)
from .metrics import (
    compute_iou,
    compute_f1,
    compute_point_to_plane_rmse,
)

__all__ = [
    'project_points_to_image',
    'compute_visibility',
    'sample_image_features',
    'CylindricalGrid',
    'points_to_cylindrical',
    'cylindrical_to_cartesian',
    'calibrate_camera',
    'undistort_images',
    'load_intrinsics',
    'visualize_projection',
    'visualize_bev',
    'plot_stereonet',
    'compute_iou',
    'compute_f1',
    'compute_point_to_plane_rmse',
]
