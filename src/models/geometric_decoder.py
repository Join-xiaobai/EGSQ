"""Differentiable Geometric Measurement Decoder with Invariant Losses."""
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableGeometricDecoder(nn.Module):
    """Computes exact geometric measurements (Distance, Angle, Ratio) from predicted

    landmark coordinates in a differentiable manner and evaluates geometric consistency.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def compute_distance(
        self, p1: torch.Tensor, p2: torch.Tensor, spacing: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Euclidean distance in physical space (if spacing provided) or pixels.

        Args:
            p1: (B, 2) coords
            p2: (B, 2) coords
            spacing: (B, 1) or scalar physical spacing (mm/pixel)
        """
        diff = p1 - p2
        dist_px = torch.norm(diff, p=2, dim=-1, keepdim=True)
        if spacing is not None:
            return dist_px * spacing
        return dist_px

    def compute_angle(
        self, p_i: torch.Tensor, p_vertex: torch.Tensor, p_k: torch.Tensor
    ) -> torch.Tensor:
        """Angle in degrees at vertex p_vertex formed by rays (p_vertex -> p_i) and (p_vertex -> p_k).

        Scale and translation invariant.
        """
        v1 = p_i - p_vertex
        v2 = p_k - p_vertex

        norm1 = torch.norm(v1, p=2, dim=-1, keepdim=True) + self.eps
        norm2 = torch.norm(v2, p=2, dim=-1, keepdim=True) + self.eps

        u1 = v1 / norm1
        u2 = v2 / norm2

        dot = torch.sum(u1 * u2, dim=-1, keepdim=True)
        dot_clamped = torch.clamp(dot, -1.0 + self.eps, 1.0 - self.eps)
        rad = torch.acos(dot_clamped)
        deg = rad * (180.0 / 3.141592653589793)
        return deg

    def compute_ratio(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """Ratio between two scalar measurements (e.g., CTR = W_heart / W_thorax)."""
        return d1 / (d2 + self.eps)

    def forward(
        self,
        landmarks: torch.Tensor,
        graph_operators: List[Dict[str, Any]],
        spacing: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:

        landmarks: (B, N, 2) predicted 2D coordinates
        graph_operators: list of operator definitions
        spacing: (B, 1) optional scale
        Returns:
            dict of measurement name -> (B, 1) tensor
        """
        measurements = {}
        for op in graph_operators:
            op_type = op["type"]
            name = op["name"]
            ep = op["endpoints"]
            if op_type == "DISTANCE":
                measurements[name] = self.compute_distance(
                    landmarks[:, ep[0]], landmarks[:, ep[1]], spacing=spacing
                )
            elif op_type == "ANGLE":
                measurements[name] = self.compute_angle(
                    landmarks[:, ep[0]], landmarks[:, ep[1]], landmarks[:, ep[2]]
                )
            elif op_type == "RATIO":
                measurements[name] = self.compute_ratio(
                    measurements[ep[0]], measurements[ep[1]]
                )
        return measurements
