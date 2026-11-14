"""Geometric and landmark evaluation metrics."""
from typing import Dict, Tuple
import numpy as np
import torch


def compute_mre(pred_coords: np.ndarray, target_coords: np.ndarray) -> float:
    """Mean Radial Error (MRE) in pixels or mm."""
    # (N, num_points, 2)
    diff = pred_coords - target_coords
    radial_errors = np.linalg.norm(diff, axis=-1)  # (N, num_points)
    return float(np.mean(radial_errors))


def compute_sdr(
    pred_coords: np.ndarray, target_coords: np.ndarray, threshold: float = 2.0
) -> float:
    """Successful Detection Rate (SDR) within threshold (e.g. 2.0mm or 4.0mm).

    Returns percentage [0, 100].
    """
    diff = pred_coords - target_coords
    radial_errors = np.linalg.norm(diff, axis=-1)  # (N, num_points)
    success = (radial_errors <= threshold)
    return float(np.mean(success) * 100.0)


def compute_mae_rmse(
    pred_values: np.ndarray, target_values: np.ndarray
) -> Tuple[float, float]:
    """Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for scalar measurements."""
    abs_errors = np.abs(pred_values - target_values)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(abs_errors**2)))
    return mae, rmse
