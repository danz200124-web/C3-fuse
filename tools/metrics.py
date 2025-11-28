"""
Evaluation metrics for C3-Fuse
"""

import numpy as np
import torch
from typing import Optional, Dict, Tuple
from sklearn.metrics import confusion_matrix


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None
) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) for each class

    Args:
        pred: (N,) predicted labels
        target: (N,) ground truth labels
        num_classes: number of classes
        ignore_index: index to ignore (e.g., unlabeled)

    Returns:
        iou: (num_classes,) IoU for each class
    """
    ious = np.zeros(num_classes)

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()

        if union == 0:
            ious[cls] = np.nan
        else:
            ious[cls] = intersection / union

    return ious


def compute_miou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None
) -> float:
    """
    Compute mean IoU

    Args:
        pred: (N,) predicted labels
        target: (N,) ground truth labels
        num_classes: number of classes
        ignore_index: index to ignore

    Returns:
        miou: mean IoU across classes
    """
    ious = compute_iou(pred, target, num_classes, ignore_index)
    valid_ious = ious[~np.isnan(ious)]

    if len(valid_ious) == 0:
        return 0.0

    return np.mean(valid_ious)


def compute_f1(
    pred: np.ndarray,
    target: np.ndarray,
    average: str = 'macro'
) -> float:
    """
    Compute F1 score

    Args:
        pred: (N,) predicted labels
        target: (N,) ground truth labels
        average: 'macro', 'micro', or 'binary'

    Returns:
        f1: F1 score
    """
    from sklearn.metrics import f1_score
    return f1_score(target, pred, average=average, zero_division=0)


def compute_precision_recall(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision and recall for each class

    Args:
        pred: (N,) predicted labels
        target: (N,) ground truth labels
        num_classes: number of classes

    Returns:
        precision: (num_classes,) precision for each class
        recall: (num_classes,) recall for each class
    """
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        tp = np.logical_and(pred_cls, target_cls).sum()
        fp = np.logical_and(pred_cls, ~target_cls).sum()
        fn = np.logical_and(~pred_cls, target_cls).sum()

        if tp + fp > 0:
            precision[cls] = tp / (tp + fp)
        else:
            precision[cls] = 0.0

        if tp + fn > 0:
            recall[cls] = tp / (tp + fn)
        else:
            recall[cls] = 0.0

    return precision, recall


def compute_point_to_plane_rmse(
    points: np.ndarray,
    plane_params: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute point-to-plane RMSE for segmented planes

    Args:
        points: (N, 3) point coordinates
        plane_params: (K, 4) plane parameters [a, b, c, d] for ax+by+cz+d=0
        labels: (N,) point labels indicating which plane each belongs to

    Returns:
        rmse: root mean square error
    """
    errors = []

    for i in range(len(plane_params)):
        mask = (labels == i)
        if mask.sum() == 0:
            continue

        pts = points[mask]
        a, b, c, d = plane_params[i]

        # Point-to-plane distance
        distances = np.abs(pts @ np.array([a, b, c]) + d) / np.sqrt(a**2 + b**2 + c**2)
        errors.extend(distances)

    if len(errors) == 0:
        return 0.0

    return np.sqrt(np.mean(np.array(errors) ** 2))


def compute_normal_angle_error(
    normals_pred: np.ndarray,
    normals_gt: np.ndarray
) -> float:
    """
    Compute angular error between predicted and ground truth normals

    Args:
        normals_pred: (N, 3) predicted normals
        normals_gt: (N, 3) ground truth normals

    Returns:
        mean_angle_error: mean angular error in degrees
    """
    # Normalize
    normals_pred = normals_pred / (np.linalg.norm(normals_pred, axis=1, keepdims=True) + 1e-8)
    normals_gt = normals_gt / (np.linalg.norm(normals_gt, axis=1, keepdims=True) + 1e-8)

    # Dot product
    dot_product = np.sum(normals_pred * normals_gt, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Angle in degrees
    angles = np.arccos(np.abs(dot_product)) * 180 / np.pi

    return np.mean(angles)


def compute_projection_consistency(
    pred_3d: np.ndarray,
    pred_2d: np.ndarray,
    uv: np.ndarray,
    image_shape: Tuple[int, int]
) -> float:
    """
    Compute projection consistency between 3D and 2D predictions

    Args:
        pred_3d: (N,) 3D point predictions
        pred_2d: (H, W) 2D image predictions
        uv: (N, 2) projected coordinates
        image_shape: (H, W)

    Returns:
        consistency: consistency score (0-1)
    """
    H, W = image_shape
    valid_mask = (
        (uv[:, 0] >= 0) & (uv[:, 0] < W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )

    uv_valid = uv[valid_mask].astype(int)
    pred_3d_valid = pred_3d[valid_mask]

    # Sample 2D predictions at projected locations
    pred_2d_sampled = pred_2d[uv_valid[:, 1], uv_valid[:, 0]]

    # Compute agreement
    agreement = (pred_3d_valid == pred_2d_sampled).mean()

    return agreement


class MetricTracker:
    """
    Track metrics during training
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        """
        Update metrics with new values

        Args:
            metrics: dict of metric name -> value
            n: number of samples (for averaging)
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value * n
            self.counts[key] += n

    def get_average(self) -> Dict[str, float]:
        """
        Get average of all tracked metrics

        Returns:
            averages: dict of metric name -> average value
        """
        averages = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                averages[key] = self.metrics[key] / self.counts[key]
            else:
                averages[key] = 0.0

        return averages

    def get_summary(self) -> str:
        """
        Get string summary of metrics

        Returns:
            summary: formatted string
        """
        averages = self.get_average()
        summary = " | ".join([f"{k}: {v:.4f}" for k, v in averages.items()])
        return summary


def compute_all_metrics(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    points: Optional[np.ndarray] = None,
    plane_params: Optional[np.ndarray] = None,
    pred_normals: Optional[np.ndarray] = None,
    gt_normals: Optional[np.ndarray] = None,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Compute all relevant metrics

    Args:
        pred_labels: (N,) predicted labels
        gt_labels: (N,) ground truth labels
        points: (N, 3) point coordinates, optional
        plane_params: (K, 4) plane parameters, optional
        pred_normals: (N, 3) predicted normals, optional
        gt_normals: (N, 3) ground truth normals, optional
        num_classes: number of classes

    Returns:
        metrics: dict of all computed metrics
    """
    metrics = {}

    # Segmentation metrics
    metrics['miou'] = compute_miou(pred_labels, gt_labels, num_classes)
    metrics['f1'] = compute_f1(pred_labels, gt_labels, average='macro')

    precision, recall = compute_precision_recall(pred_labels, gt_labels, num_classes)
    metrics['precision'] = np.mean(precision)
    metrics['recall'] = np.mean(recall)

    # Accuracy
    metrics['accuracy'] = (pred_labels == gt_labels).mean()

    # Geometric metrics
    if points is not None and plane_params is not None:
        metrics['point_to_plane_rmse'] = compute_point_to_plane_rmse(
            points, plane_params, pred_labels
        )

    if pred_normals is not None and gt_normals is not None:
        metrics['normal_angle_error'] = compute_normal_angle_error(
            pred_normals, gt_normals
        )

    return metrics
