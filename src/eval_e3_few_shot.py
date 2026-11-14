import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Experiment E3: Cross-Task Few-Shot Adaptation (Ultrasound Pretraining -> Ceph X-ray Adaptation)."""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.models import EGSQModel
from src.data import ISBICephalometricDataset
from src.metrics import compute_mre, compute_sdr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_egsq_multitask.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=== Running Experiment E3: Cross-Task Few-Shot Transfer ===")
    print("Evaluating adaptation from shared geometric reasoning prior to target Cephalometric X-ray.")
    print("Fractions tested: [1% (2 samples), 5% (7 samples), 10% (15 samples), 20% (30 samples), 100% (Full)]")

    train_csv = "data/isbi2015_cephalometric/train_senior.csv"
    test_csv = "data/isbi2015_cephalometric/test1_senior.csv"
    img_root = "data/isbi2015_cephalometric/cepha400"

    full_train = ISBICephalometricDataset(csv_file=train_csv, image_root=img_root)
    test_dataset = ISBICephalometricDataset(csv_file=test_csv, image_root=img_root)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    ceph_operators = [
        {"type": "ANGLE", "name": "SNA", "endpoints": [0, 1, 4]},
        {"type": "ANGLE", "name": "SNB", "endpoints": [0, 1, 5]},
    ]
    ceph_graph_meta = {
        "operator_ids": torch.tensor([1]),
        "num_nodes": torch.tensor([[19.0]]),
        "num_edges": torch.tensor([[2.0]]),
    }

    fractions = [0.01, 0.05, 0.10, 0.20, 1.0]
    e3_results = {}

    for frac in fractions:
        k = max(2, int(len(full_train) * frac))
        indices = list(range(k))
        sub_dataset = Subset(full_train, indices)
        sub_loader = DataLoader(sub_dataset, batch_size=min(4, k), shuffle=True)

        # Initialize fresh model conditioned on geometric graph
        model = EGSQModel(num_landmarks_max=32, feature_dim=128, num_modalities=4).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

        model.train()
        for epoch in range(1, args.epochs + 1):
            for batch in sub_loader:
                img = batch["image"].to(args.device)
                pts = batch["landmarks"].to(args.device)
                sp = batch["spacing"].to(args.device)
                mod = batch["modality_id"].to(args.device)

                optimizer.zero_grad()
                out = model(img, mod, ceph_graph_meta, ceph_operators, spacing=sp)
                p = out["coords"][:, :19]
                loss = nn.functional.smooth_l1_loss(p, pts)
                loss.backward()
                optimizer.step()

        # Evaluate on full target test set
        model.eval()
        preds_list, gts_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                img = batch["image"].to(args.device)
                pts = batch["landmarks"].to(args.device)
                sp = batch["spacing"].to(args.device)
                mod = batch["modality_id"].to(args.device)

                out = model(img, mod, ceph_graph_meta, ceph_operators, spacing=sp)
                p = out["coords"][:, :19]
                preds_list.append(p.cpu().numpy())
                gts_list.append(pts.cpu().numpy())

        preds = np.concatenate(preds_list)
        gts = np.concatenate(gts_list)

        mre = compute_mre(preds, gts)
        sdr2 = compute_sdr(preds, gts, threshold=2.0)
        sdr4 = compute_sdr(preds, gts, threshold=4.0)

        print(f"Few-Shot [{frac*100:5.1f}% Data ({k:2d} samples)] -> MRE: {mre:.2f}px | SDR@2mm: {sdr2:.1f}% | SDR@4mm: {sdr4:.1f}%")
        e3_results[f"frac_{int(frac*100)}%"] = {
            "fraction": frac,
            "num_samples": k,
            "mre": float(mre),
            "sdr2": float(sdr2),
            "sdr4": float(sdr4),
        }

    os.makedirs("reports", exist_ok=True)
    np.savez("reports/e3_few_shot_results.npz", **e3_results)
    print("\nSaved E3 few-shot results to reports/e3_few_shot_results.npz")


if __name__ == "__main__":
    main()
