"""Full EGSQ Model architecture with Differentiable Geometric Reasoning and Evidence Auditing."""
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .measurement_graph import TaskConditionedGraphEncoder
from .geometric_decoder import DifferentiableGeometricDecoder
from .gec import GeometricEvidenceConsistency
from .rcsq import RiskControlledSelectiveQuantification
from .backbones import ModalityAdapter, SimpleSharedBackbone


class EGSQModel(nn.Module):
    """Evidence-Grounded Selective Quantification Model."""

    def __init__(
        self,
        num_landmarks_max: int = 32,
        feature_dim: int = 128,
        num_modalities: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_landmarks_max = num_landmarks_max

        # 1. Modality Adapters (0: Ultrasound, 1: X-ray)
        self.adapters = nn.ModuleList([
            ModalityAdapter(in_channels=1, out_channels=64) for _ in range(num_modalities)
        ])

        # 2. Shared Visual Backbone
        self.backbone = SimpleSharedBackbone(in_channels=64, feature_dim=feature_dim)

        # 3. Task-Conditioned Measurement Graph Encoder
        self.graph_encoder = TaskConditionedGraphEncoder(embed_dim=feature_dim, num_modalities=num_modalities)

        # 4. Landmark Heatmap Head
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_landmarks_max, kernel_size=1),
        )

        # 5. Differentiable Geometric Decoder
        self.geom_decoder = DifferentiableGeometricDecoder()

        # 6. Geometric Evidence Consistency (GEC)
        self.gec = GeometricEvidenceConsistency(feature_dim=feature_dim)

        # 7. Risk-Controlled Selective Quantification (RCSQ)
        self.rcsq = RiskControlledSelectiveQuantification(evidence_dim=16)

    def soft_argmax_2d(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Compute expected (x, y) coordinates from 2D spatial probability distributions."""
        B, N, H, W = heatmaps.shape
        flat = heatmaps.view(B, N, -1)
        probs = F.softmax(flat, dim=-1).view(B, N, H, W)

        pos_y, pos_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=heatmaps.device),
            torch.linspace(0, 1, W, device=heatmaps.device),
            indexing="ij",
        )

        x_coords = torch.sum(probs * pos_x, dim=(-2, -1))
        y_coords = torch.sum(probs * pos_y, dim=(-2, -1))

        return torch.stack([x_coords, y_coords], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        modality_id: torch.Tensor,
        graph_meta: Dict[str, Any],
        operators: List[Dict[str, Any]],
        spacing: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B = x.shape[0]
        mod_idx = int(modality_id[0].item())
        x_adapted = self.adapters[mod_idx](x)

        # Backbone feature extraction
        feats = self.backbone(x_adapted)  # (B, feature_dim, H', W')

        # Task graph embedding
        q_tau = self.graph_encoder(modality_id, graph_meta)  # (B, feature_dim)
        q_mod = q_tau.unsqueeze(-1).unsqueeze(-1)
        feats_cond = feats * (1.0 + torch.tanh(q_mod))

        # Predict landmark heatmaps
        heatmap_logits = self.heatmap_head(feats_cond)
        coords = self.soft_argmax_2d(heatmap_logits)

        # Scale normalized coordinates to image dimensions
        H, W = x.shape[-2:]
        coords_scaled = coords.clone()
        coords_scaled[..., 0] = coords_scaled[..., 0] * W
        coords_scaled[..., 1] = coords_scaled[..., 1] * H

        # Exact geometric measurement decoding
        measurements = self.geom_decoder(coords_scaled, operators, spacing=spacing)

        # GEC evidence extraction
        B, N, H_m, W_m = heatmap_logits.shape
        flat_feats = F.adaptive_avg_pool2d(feats_cond, (1, 1)).view(B, 1, self.feature_dim)
        node_feats = flat_feats.repeat(1, N, 1)
        normalized_heatmaps = F.softmax(heatmap_logits.view(B, N, -1), dim=-1).view(B, N, H_m, W_m)
        gec_out = self.gec(node_feats, normalized_heatmaps)

        # RCSQ risk prediction & decision
        ev_vec = gec_out["evidence_vector"].view(B, -1)[:, :16]
        if ev_vec.shape[-1] < 16:
            pad = torch.zeros(B, 16 - ev_vec.shape[-1], device=x.device)
            ev_vec = torch.cat([ev_vec, pad], dim=-1)

        risk_scores, accept_decisions = self.rcsq(ev_vec)

        return {
            "coords": coords_scaled,
            "heatmap_logits": heatmap_logits,
            "measurements": measurements,
            "evidence": gec_out,
            "risk_score": risk_scores,
            "accept_decision": accept_decisions,
        }
