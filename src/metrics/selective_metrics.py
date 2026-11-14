"""Selective prediction and risk-coverage metrics."""
from typing import Dict, List, Tuple
import numpy as np


def compute_risk_coverage_curve(
    risk_scores: np.ndarray, errors: np.ndarray, num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Risk-Coverage curve by sorting risk scores ascending.

    Args:
        risk_scores: (N,) predicted risk/uncertainty scores (lower = safer)
        errors: (N,) true evaluation errors (e.g., MAE or binary high-error flag)
    Returns:
        coverages: array of coverage levels in (0, 1]
        risks: corresponding average risk/error of accepted subset
    """
    order = np.argsort(risk_scores)
    sorted_errors = errors[order]

    coverages = []
    risks = []
    n = len(errors)

    for i in range(1, num_points + 1):
        k = int(np.ceil((i / num_points) * n))
        k = max(1, min(k, n))
        cov = k / n
        sub_risk = np.mean(sorted_errors[:k])
        coverages.append(cov)
        risks.append(sub_risk)

    return np.array(coverages), np.array(risks)


def compute_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """Area Under the Risk-Coverage Curve (AURC). Lower is better."""
    return float(np.trapz(risks, coverages))


def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) for binary risk events (e.g., error > threshold)."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(num_bins):
        bin_mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            avg_prob = np.mean(probs[bin_mask])
            avg_label = np.mean(labels[bin_mask])
            ece += (bin_count / total) * np.abs(avg_prob - avg_label)

    return float(ece)
