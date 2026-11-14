import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Training and Conformal Risk Calibration script for EGSQ."""
import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.models import EGSQModel
from src.data import FetalBiometryDataset
from src.metrics import compute_mae_rmse, compute_mre, compute_sdr, compute_aurc, compute_risk_coverage_curve
from src.utils import SplitConformalRiskControl


def parse_args():
    parser = argparse.ArgumentParser(description="Train EGSQ")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--task-config", type=str, default="configs/fetal_biometry.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def patient_level_split(dataset, train_ratio=0.7, val_ratio=0.15, calib_ratio=0.15, seed=42):
    """Splits dataset strictly by patient ID so images from the same patient never leak."""
    np.random.seed(seed)
    patient_ids = [dataset[i]["patient_id"] for i in range(len(dataset))]
    unique_patients = np.unique(patient_ids)
    np.random.shuffle(unique_patients)

    n_total = len(unique_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pts = set(unique_patients[:n_train])
    val_pts = set(unique_patients[n_train:n_train + n_val])
    calib_pts = set(unique_patients[n_train + n_val:])

    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_pts]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_pts]
    calib_indices = [i for i, pid in enumerate(patient_ids) if pid in calib_pts]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, calib_indices),
    )


def train_epoch(model, loader, optimizer, operators, graph_meta, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        landmarks = batch["landmarks"].to(device)
        spacings = batch["spacing"].to(device)
        modality_ids = batch["modality_id"].to(device)

        optimizer.zero_grad()
        out = model(images, modality_ids, graph_meta, operators, spacing=spacings)

        # 1. Landmark loss (SmoothL1 on available points)
        pred_coords = out["coords"][:, :landmarks.shape[1]]
        loss_lm = nn.functional.smooth_l1_loss(pred_coords, landmarks)

        # 2. Risk loss (predicting high error)
        coord_err = torch.norm(pred_coords - landmarks, p=2, dim=-1).mean(dim=-1, keepdim=True)
        target_risk = (coord_err > 5.0).float()
        loss_risk = nn.functional.binary_cross_entropy(out["risk_score"], target_risk)

        loss = loss_lm + 0.3 * loss_risk
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def evaluate(model, loader, operators, graph_meta, device):
    model.eval()
    all_preds, all_gts, all_risks, all_spacings = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            landmarks = batch["landmarks"].to(device)
            spacings = batch["spacing"].to(device)
            modality_ids = batch["modality_id"].to(device)

            out = model(images, modality_ids, graph_meta, operators, spacing=spacings)
            pred_coords = out["coords"][:, :landmarks.shape[1]]

            all_preds.append(pred_coords.cpu().numpy())
            all_gts.append(landmarks.cpu().numpy())
            all_risks.append(out["risk_score"].cpu().numpy().squeeze(-1))
            all_spacings.append(spacings.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    gts = np.concatenate(all_gts, axis=0)
    risks = np.concatenate(all_risks, axis=0)

    mre = compute_mre(preds, gts)
    sdr_2mm = compute_sdr(preds, gts, threshold=2.0)
    sdr_4mm = compute_sdr(preds, gts, threshold=4.0)

    # Compute radial errors per sample
    sample_errors = np.mean(np.linalg.norm(preds - gts, axis=-1), axis=-1)

    return mre, sdr_2mm, sdr_4mm, risks, sample_errors


def main():
    args = parse_args()
    print(f"Starting EGSQ training on device: {args.device}")

    # Dummy graph meta and operators for fetal BPD/OFD
    graph_meta = {
        "operator_ids": torch.tensor([0]),
        "num_nodes": torch.tensor([[4.0]]),
        "num_edges": torch.tensor([[2.0]]),
    }
    operators = [
        {"type": "DISTANCE", "name": "OFD", "endpoints": [0, 1]},
        {"type": "DISTANCE", "name": "BPD", "endpoints": [2, 3]},
    ]

    csv_path = "data/Multicentre-Fetal-Biometry/data/annotations/FP/Head_Train.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found at {csv_path}. Please check data path.")
        return

    full_dataset = FetalBiometryDataset(
        csv_file=csv_path,
        image_root="data/Multicentre-Fetal-Biometry/data/FP/Head",
        anatomy="brain",
    )
    print(f"Total dataset loaded: {len(full_dataset)} samples.")

    train_set, val_set, calib_set = patient_level_split(full_dataset)
    print(f"Patient-level split -> Train: {len(train_set)}, Val: {len(val_set)}, Calib: {len(calib_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    calib_loader = DataLoader(calib_set, batch_size=args.batch_size, shuffle=False)

    model = EGSQModel(num_landmarks_max=8, feature_dim=128, num_modalities=2).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, operators, graph_meta, args.device)
        mre, sdr2, sdr4, _, _ = evaluate(model, val_loader, operators, graph_meta, args.device)
        print(f"Epoch [{epoch}/{args.epochs}] Loss: {loss:.4f} | Val MRE: {mre:.2f}px | SDR@2: {sdr2:.1f}% | SDR@4: {sdr4:.1f}%")

    # Conformal Risk Calibration on Calibration Set
    print("\n--- Running Conformal Risk Control Calibration ---")
    _, _, _, calib_risks, calib_errors = evaluate(model, calib_loader, operators, graph_meta, args.device)
    conformal = SplitConformalRiskControl(alpha=10.0)  # tolerance threshold in pixels
    lam = conformal.calibrate(calib_risks, calib_errors)
    print(f"Calibrated Risk Threshold (lambda): {lam:.4f}")

    # Risk-Coverage Curve
    covs, risks_curve = compute_risk_coverage_curve(calib_risks, calib_errors)
    aurc = compute_aurc(covs, risks_curve)
    print(f"Area Under Risk-Coverage Curve (AURC): {aurc:.4f}")

    os.makedirs("reports", exist_ok=True)
    np.savez("reports/fetal_risk_coverage.npz", coverages=covs, risks=risks_curve, aurc=aurc, threshold=lam)
    print("Calibration & risk-coverage metrics saved to reports/fetal_risk_coverage.npz")


if __name__ == "__main__":
    main()
