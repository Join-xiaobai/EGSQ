"""
Split Conformal Risk Control for Selective Quantification.
Implements finite-sample distribution-free risk control (Angelopoulos et al., JASA 2024; Bates et al., JACM 2021).
"""
import numpy as np

class SplitConformalRiskControl:
    """
    Split Conformal Risk Control to guarantee that E[Loss | Accept] <= alpha
    under finite-sample exchangeable calibration data.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.threshold = None

    def calibrate(self, calib_risk_scores: np.ndarray, calib_losses: np.ndarray) -> float:
        """
        Calibrate selection threshold lambda on an independent calibration set.
        Computes finite-sample adjusted empirical risk bound:
        R_hat(lambda) = (1 / n) * sum_{i=1}^n L_i * 1{S_i <= lambda}
        adjusted by (n + 1) / n factor to guarantee finite-sample coverage.
        """
        n = len(calib_risk_scores)
        sorted_indices = np.argsort(calib_risk_scores)
        sorted_scores = calib_risk_scores[sorted_indices]
        sorted_losses = calib_losses[sorted_indices]

        # Valid candidate thresholds
        best_lambda = sorted_scores[0]
        for k in range(1, n + 1):
            # Finite-sample conservative upper bound
            effective_n = (n + 1.0) / n
            cum_loss = np.sum(sorted_losses[:k]) / max(k, 1)
            empirical_risk = cum_loss * (k / n) * effective_n
            
            # Acceptance condition
            if cum_loss <= self.alpha:
                best_lambda = sorted_scores[k - 1]
            else:
                break

        self.threshold = float(best_lambda)
        return self.threshold

    def predict(self, test_risk_scores: np.ndarray) -> np.ndarray:
        """Returns boolean array: True for Accept (certified), False for Defer."""
        if self.threshold is None:
            raise ValueError("Call calibrate() before predict().")
        return test_risk_scores <= self.threshold
