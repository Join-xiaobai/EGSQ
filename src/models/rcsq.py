"""Risk-Controlled Selective Quantification (RCSQ) module."""
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


class RiskControlledSelectiveQuantification(nn.Module):
    """Calibrates task-specific error risk and makes Accept / Defer decisions."""

    def __init__(self, evidence_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.risk_net = nn.Sequential(
            nn.Linear(evidence_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Predicted probability of exceeding tolerance error
        )

        # Registered buffer for calibrated thresholds per task
        self.register_buffer("calibrated_threshold", torch.tensor(0.5))

    def predict_risk(self, aggregated_evidence: torch.Tensor) -> torch.Tensor:
        """Args:

        aggregated_evidence: (B, evidence_dim) task-level concatenated evidence
        Returns:
            risk_score: (B, 1) in [0, 1]
        """
        return self.risk_net(aggregated_evidence)

    def forward(
        self,
        aggregated_evidence: torch.Tensor,
        threshold: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns:

        risk_scores: (B, 1)
        decisions: (B, 1) boolean: True for Accept, False for Defer
        """
        risk = self.predict_risk(aggregated_evidence)
        tau = threshold if threshold is not None else self.calibrated_threshold
        accept_mask = (risk <= tau)
        return risk, accept_mask
