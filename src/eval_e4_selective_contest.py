import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Experiment E4: Selective Prediction Strategy Contest (GEC Evidence vs. Naive Confidence Baselines)."""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import EGSQModel
from src.data import FetalBiometryDataset
from src.metrics import compute_aurc, compute_risk_coverage_curve, compute_ece
from src.utils import SplitConformalRiskControl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_egsq_multitask.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=== Running Experiment E4: Selective Prediction Baseline Contest ===")
    print("Contenders:")
    print("  1. Baseline A: Heatmap Peak Confidence (Softmax Maximum Probability)")
    print("  2. Baseline B: Spatial Entropy (Dispersion-based Aleatoric Uncertainty)")
    print("  3. Baseline C: Monte-Carlo Heuristic (Uncalibrated Empirical Score)")
    print("  4. Proposed EGSQ: GEC Multi-Faceted Evidence Grounding + Conformal Risk")

    test_csv = "data/Multicentre-Fetal-Biometry/data/annotations/FP/Head_Test.csv"
    img_root = "data/Multicentre-Fetal-Biometry/data/FP/Head"
    dataset = FetalBiometryDataset(csv_file=test_csv, image_root=img_root, anatomy="brain")
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    fetal_operators = [
        {"type": "DISTANCE", "name": "OFD", "endpoints": [0, 1]},
        {"type": "DISTANCE", "name": "BPD", "endpoints": [2, 3]},
    ]
    fetal_graph_meta = {
        "operator_ids": torch.tensor([0]),
        "num_nodes": torch.tensor([[4.0]]),
        "num_edges": torch.tensor([[2.0]]),
    }

    model = EGSQModel(num_landmarks_max=32, feature_dim=128, num_modalities=4).to(args.device)
    model.eval()
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        print(f"Loaded trained checkpoint from {args.checkpoint}")
    else:
        print(f"Warning: checkpoint {args.checkpoint} not found. Running with initial weights.")

    all_preds, all_gts, all_peaks, all_entropies, all_egsq_risks = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(args.device)
            pts = batch["landmarks"].to(args.device)
            sp = batch["spacing"].to(args.device)
            mod = batch["modality_id"].to(args.device)

            out = model(img, mod, fetal_graph_meta, fetal_operators, spacing=sp)
            p = out["coords"][:, :4]

            # 1. Prediction error
            all_preds.append(p.cpu().numpy())
            all_gts.append(pts.cpu().numpy())

            # 2. Risk strategies
            ev = out["evidence"]
            peak = ev["peak_confidence"].mean(dim=-1).cpu().numpy()  # High = confident
            ent = ev["dispersion"].mean(dim=-1).cpu().numpy()         # High = uncertain
            rk_egsq = out["risk_score"].squeeze(-1).cpu().numpy()     # High = risky

            all_peaks.append(peak)
            all_entropies.append(ent)
            all_egsq_risks.append(rk_egsq)

    preds = np.concatenate(all_preds)
    gts = np.concatenate(all_gts)
    errs = np.mean(np.linalg.norm(preds - gts, axis=-1), axis=-1)

    # Score mapping (lower score = model claims sample is safer/more reliable)
    risk_strategies = {
        "Peak_Confidence_Inv": 1.0 - np.concatenate(all_peaks),
        "Spatial_Entropy": np.concatenate(all_entropies),
        "Naive_Uncertainty_Ensemble": (1.0 - np.concatenate(all_peaks) + np.concatenate(all_entropies)) / 2.0,
        "EGSQ_GEC_Risk": np.concatenate(all_egsq_risks),
    }

    contest_results = {}
    print("\n--- Strategy Contest Results (Risk-Coverage Evaluation) ---")
    for strat_name, scores in risk_strategies.items():
        covs, r_curve = compute_risk_coverage_curve(scores, errs, num_points=100)
        aurc = compute_aurc(covs, r_curve)

        # Risk at 80% coverage and 50% coverage
        idx_80 = int(np.argmin(np.abs(covs - 0.80)))
        idx_50 = int(np.argmin(np.abs(covs - 0.50)))
        risk_at_80 = float(r_curve[idx_80])
        risk_at_50 = float(r_curve[idx_50])

        print(f"Strategy [{strat_name:26s}] -> AURC: {aurc:.3f} | Risk@80%Cov: {risk_at_80:.2f}px | Risk@50%Cov: {risk_at_50:.2f}px")
        contest_results[strat_name] = {
            "aurc": float(aurc),
            "risk_at_80": risk_at_80,
            "risk_at_50": risk_at_50,
            "coverages": covs,
            "risks": r_curve,
        }

    os.makedirs("reports", exist_ok=True)
    np.savez("reports/e4_selective_contest_results.npz", **contest_results)
    print("\nSaved E4 selective prediction contest to reports/e4_selective_contest_results.npz")


if __name__ == "__main__":
    main()
