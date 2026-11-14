"""Geometric Evidence Consistency (GEC) Module."""
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricEvidenceConsistency(nn.Module):
    """Evaluates whether predicted structures are grounded in visible image evidence,

    satisfy anatomical geometric priors, and remain invariant under admissible transformations.
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        # Evidence projection head takes: local visual features + spatial entropy + heatmap peak + residual
        self.evidence_head = nn.Sequential(
            nn.Linear(feature_dim + 3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 4),  # [evidence_strength, spatial_focus, residual_consistency, stability]
            nn.Sigmoid(),
        )

    def compute_dispersion(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Compute spatial entropy (dispersion) of landmark heatmaps.

        Args:
            heatmaps: (B, N, H, W) normalized probability maps
        Returns:
            entropy: (B, N, 1) normalized entropy in [0, 1]
        """
        B, N, H, W = heatmaps.shape
        flat = heatmaps.view(B, N, -1) + 1e-8
        denom = torch.log(torch.tensor(H * W, dtype=torch.float32, device=heatmaps.device))
        entropy = -torch.sum(flat * torch.log(flat), dim=-1, keepdim=True) / denom
        return torch.clamp(entropy, 0.0, 1.0)

    def forward(
        self,
        node_features: torch.Tensor,
        heatmaps: torch.Tensor,
        geometric_residuals: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:

        node_features: (B, N, C) local anatomical visual features
        heatmaps: (B, N, H, W) soft heatmap predictions
        geometric_residuals: (B, N, 1) optional anatomical deviations
        Returns:
            dict containing evidence metrics and compact representation
        """
        B, N, H, W = heatmaps.shape
        dispersion = self.compute_dispersion(heatmaps)  # (B, N, 1)
        heat_peak, _ = heatmaps.view(B, N, -1).max(dim=-1, keepdim=True)  # (B, N, 1)

        if geometric_residuals is None:
            geometric_residuals = torch.zeros(B, N, 1, device=heatmaps.device)

        # Concatenate features
        ev_in = torch.cat([node_features, dispersion, heat_peak, geometric_residuals], dim=-1)
        evidence_vec = self.evidence_head(ev_in)  # (B, N, 4)

        return {
            "evidence_vector": evidence_vec,
            "dispersion": dispersion.squeeze(-1),
            "peak_confidence": heat_peak.squeeze(-1),
            "evidence_strength": evidence_vec[..., 0],
            "spatial_focus": evidence_vec[..., 1],
        }
