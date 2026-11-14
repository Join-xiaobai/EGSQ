"""Task-Conditioned Measurement Graph (TCMG) neural encoder and topology definitions."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphNode:
    id: int
    name: str
    organ: Optional[str] = None
    required_visibility: bool = True


@dataclass
class GraphEdge:
    type: str  # 'DISTANCE', 'ANGLE', 'RATIO', 'CONTOUR'
    endpoints: List[int]
    name: str
    unit: str = "mm"
    max_tolerance_error: float = 2.0


@dataclass
class MeasurementGraph:
    task_name: str
    modality: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    pixel_spacing: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


class TaskConditionedGraphEncoder(nn.Module):
    """Encodes a task measurement graph into a structured conditioning embedding."""

    def __init__(self, embed_dim: int = 256, num_modalities: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_embed = nn.Embedding(num_modalities, embed_dim)
        self.operator_embed = nn.Embedding(8, embed_dim)  # DISTANCE, ANGLE, RATIO, etc.

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, modality_id: torch.Tensor, graph_meta: Dict[str, Any]) -> torch.Tensor:
        """Args:
            modality_id: (B,) int tensor
            graph_meta: dict containing operator_ids, num_nodes, num_edges
        """
        B = modality_id.shape[0]
        mod_feat = self.modality_embed(modality_id)  # (B, embed_dim)

        op_ids = graph_meta.get("operator_ids", torch.zeros(B, dtype=torch.long, device=modality_id.device)).to(modality_id.device)
        if op_ids.dim() == 1 and op_ids.shape[0] != B:
            op_ids = op_ids.repeat(B)
        op_feat = self.operator_embed(op_ids)  # (B, embed_dim)

        num_n = graph_meta.get("num_nodes", torch.zeros(B, 1, device=modality_id.device)).to(modality_id.device)
        if num_n.shape[0] != B:
            num_n = num_n.repeat(B, 1)

        num_e = graph_meta.get("num_edges", torch.zeros(B, 1, device=modality_id.device)).to(modality_id.device)
        if num_e.shape[0] != B:
            num_e = num_e.repeat(B, 1)

        cat_feat = torch.cat([mod_feat, op_feat, num_n, num_e], dim=-1)
        q_tau = self.mlp(cat_feat)
        return q_tau
