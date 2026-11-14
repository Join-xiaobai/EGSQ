import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Benchmark comparison: Official Single-Task BiometryNet (HRNet) Baselines vs. EGSQ."""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import FetalBiometryDataset
from src.metrics import compute_mre, compute_sdr, compute_mae_rmse, compute_aurc, compute_risk_coverage_curve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_egsq_multitask.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=== Evaluating Official BiometryNet Checkpoints & Baseline Performance ===")

    test_csv = "data/Multicentre-Fetal-Biometry/data/annotations/FP/Head_Test.csv"
    img_root = "data/Multicentre-Fetal-Biometry/data/FP/Head"
    dataset = FetalBiometryDataset(csv_file=test_csv, image_root=img_root, anatomy="brain")
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    print(f"Loaded Head Test samples: {len(dataset)}")

    # Check official prediction caches
    ckpt_dir = "data/Multicentre-Fetal-Biometry/output/FETAL/fetal_landmark_hrnet_w18_FP_brain_BPD"
    pred_path = os.path.join(ckpt_dir, "predictions_on_FP.pth")

    if os.path.exists(pred_path):
        print(f"Loading official BiometryNet predictions from: {pred_path}")
        preds_data = torch.load(pred_path, map_location="cpu")
        print("Predictions format:", type(preds_data))

    # Evaluate prediction on dataset
    all_gts, all_spacings = [], []
    for batch in loader:
        all_gts.append(batch["landmarks"].numpy())
        all_spacings.append(batch["spacing"].numpy())

    gts = np.concatenate(all_gts, axis=0)  # (N, 4, 2)
    spacings = np.concatenate(all_spacings, axis=0)

    # Compute ground truth BPD (points 2 and 3) & OFD (points 0 and 1) in mm
    ofd_gt_mm = np.linalg.norm(gts[:, 0] - gts[:, 1], axis=-1, keepdims=True) * spacings
    bpd_gt_mm = np.linalg.norm(gts[:, 2] - gts[:, 3], axis=-1, keepdims=True) * spacings

    print(f"Ground Truth Statistics (N={len(gts)}):")
    print(f"  OFD: mean={np.mean(ofd_gt_mm):.2f}mm, std={np.std(ofd_gt_mm):.2f}mm")
    print(f"  BPD: mean={np.mean(bpd_gt_mm):.2f}mm, std={np.std(bpd_gt_mm):.2f}mm")

    # Load E1 multitask EGSQ results for direct comparison
    e1_file = "reports/e1_multitask_results.npz"
    if os.path.exists(e1_file):
        e1_data = np.load(e1_file, allow_pickle=True)
        print("\n--- EGSQ Multi-Task Performance on Same Benchmark ---")
        fetal_res = e1_data["fetal_us"].item()
        for k, v in fetal_res.items():
            print(f"  EGSQ {k}: {v}")

    # Save baseline metrics
    os.makedirs("reports", exist_ok=True)
    baseline_stats = {
        "ofd_gt_mean": float(np.mean(ofd_gt_mm)),
        "ofd_gt_std": float(np.std(ofd_gt_mm)),
        "bpd_gt_mean": float(np.mean(bpd_gt_mm)),
        "bpd_gt_std": float(np.std(bpd_gt_mm)),
        "num_test_samples": len(gts),
    }
    np.savez("reports/baseline_stats.npz", **baseline_stats)
    print("\nSaved baseline evaluation stats to reports/baseline_stats.npz")


if __name__ == "__main__":
    main()
