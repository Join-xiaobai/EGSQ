import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Experiment E2: Cross-Center & Leave-Device-Out Generalization Benchmark."""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import EGSQModel
from src.data import FetalBiometryDataset
from src.metrics import compute_mre, compute_sdr, compute_aurc, compute_risk_coverage_curve
from src.utils import SplitConformalRiskControl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_egsq_multitask.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=== Running Experiment E2: Cross-Center & Cross-Device Generalization ===")
    print("Training source: FP (Barcelona multi-device cohort)")
    print("Evaluating domain shift on: 1. In-domain FP Test | 2. External HC18 (Radboud) | 3. External UCL (London)")

    fetal_operators = [
        {"type": "DISTANCE", "name": "OFD", "endpoints": [0, 1]},
        {"type": "DISTANCE", "name": "BPD", "endpoints": [2, 3]},
    ]
    fetal_graph_meta = {
        "operator_ids": torch.tensor([0]),
        "num_nodes": torch.tensor([[4.0]]),
        "num_edges": torch.tensor([[2.0]]),
    }

    # Datasets for evaluation
    subsets = {
        "In_Domain_FP": {
            "csv": "data/Multicentre-Fetal-Biometry/data/annotations/FP/Head_Test.csv",
            "img": "data/Multicentre-Fetal-Biometry/data/FP/Head",
        },
        "Cross_Center_HC18": {
            "csv": "data/Multicentre-Fetal-Biometry/data/annotations/HC18/Head_Test.csv",
            "img": "data/Multicentre-Fetal-Biometry/data/HC18/Head",
        },
        "Cross_Center_UCL": {
            "csv": "data/Multicentre-Fetal-Biometry/data/annotations/UCL/Head_Test.csv",
            "img": "data/Multicentre-Fetal-Biometry/data/UCL/Head",
        },
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

    e2_results = {}

    for name, cfg in subsets.items():
        if not os.path.exists(cfg["csv"]):
            print(f"Skipping {name}: CSV not found at {cfg['csv']}")
            continue

        dataset = FetalBiometryDataset(csv_file=cfg["csv"], image_root=cfg["img"], anatomy="brain")
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        preds_list, gts_list, risks_list = [], [], []
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(args.device)
                pts = batch["landmarks"].to(args.device)
                sp = batch["spacing"].to(args.device)
                mod = batch["modality_id"].to(args.device)

                out = model(img, mod, fetal_graph_meta, fetal_operators, spacing=sp)
                p = out["coords"][:, :4]

                preds_list.append(p.cpu().numpy())
                gts_list.append(pts.cpu().numpy())
                risks_list.append(out["risk_score"].cpu().numpy().squeeze(-1))

        preds = np.concatenate(preds_list)
        gts = np.concatenate(gts_list)
        risks = np.concatenate(risks_list)

        mre = compute_mre(preds, gts)
        sdr2 = compute_sdr(preds, gts, threshold=2.0)
        sdr4 = compute_sdr(preds, gts, threshold=4.0)

        errs = np.mean(np.linalg.norm(preds - gts, axis=-1), axis=-1)
        covs, r_curve = compute_risk_coverage_curve(risks, errs)
        aurc = compute_aurc(covs, r_curve)

        mean_risk = float(np.mean(risks))

        print(f"\nCohort [{name:20s}] (N={len(dataset)}):")
        print(f"  MRE: {mre:.2f}px | SDR@2mm: {sdr2:.1f}% | SDR@4mm: {sdr4:.1f}%")
        print(f"  Selective AURC: {aurc:.2f} | Out-of-Domain Risk Awareness (Mean Risk): {mean_risk:.3f}")

        e2_results[name] = {
            "num_samples": len(dataset),
            "mre": float(mre),
            "sdr2": float(sdr2),
            "sdr4": float(sdr4),
            "aurc": float(aurc),
            "mean_risk": mean_risk,
        }

    os.makedirs("reports", exist_ok=True)
    np.savez("reports/e2_cross_domain_results.npz", **e2_results)
    print("\nSaved E2 cross-domain results to reports/e2_cross_domain_results.npz")


if __name__ == "__main__":
    main()
