"""Multi-task Cross-Modality Training with Early Stopping & Conformal Risk Calibration (E1 Experiment)."""
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import EGSQModel
from src.data import FetalBiometryDataset, ISBICephalometricDataset
from src.metrics import compute_mre, compute_sdr, compute_aurc, compute_risk_coverage_curve
from src.utils import SplitConformalRiskControl, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best_egsq_multitask.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"=== EGSQ Multi-Task Experiment (E1): Fetal US + Lateral Cephalometric X-ray ===")
    print(f"Running on: {args.device} | Max Epochs: {args.epochs} | Early Stopping Patience: {args.patience}")

    # 1. Datasets & Loaders
    # Task A: Fetal Ultrasound (Train / Test)
    fetal_train = FetalBiometryDataset(
        csv_file="data/Multicentre-Fetal-Biometry/data/annotations/FP/Head_Train.csv",
        image_root="data/Multicentre-Fetal-Biometry/data/FP/Head",
        anatomy="brain",
    )
    fetal_test = FetalBiometryDataset(
        csv_file="data/Multicentre-Fetal-Biometry/data/annotations/FP/Head_Test.csv",
        image_root="data/Multicentre-Fetal-Biometry/data/FP/Head",
        anatomy="brain",
    )

    # Task B: Cephalometric X-ray (ISBI 2015 Train / Test1)
    ceph_train = ISBICephalometricDataset(
        csv_file="data/isbi2015_cephalometric/train_senior.csv",
        image_root="data/isbi2015_cephalometric/cepha400",
    )
    ceph_test = ISBICephalometricDataset(
        csv_file="data/isbi2015_cephalometric/test1_senior.csv",
        image_root="data/isbi2015_cephalometric/cepha400",
    )

    print(f"Loaded Fetal US: Train={len(fetal_train)}, Test={len(fetal_test)}")
    print(f"Loaded Ceph X-ray: Train={len(ceph_train)}, Test={len(ceph_test)}")

    loader_fetal_tr = DataLoader(fetal_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_ceph_tr = DataLoader(ceph_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    loader_fetal_te = DataLoader(fetal_test, batch_size=args.batch_size, shuffle=False)
    loader_ceph_te = DataLoader(ceph_test, batch_size=args.batch_size, shuffle=False)

    # Task Graph Metadata & Operators
    fetal_operators = [
        {"type": "DISTANCE", "name": "OFD", "endpoints": [0, 1]},
        {"type": "DISTANCE", "name": "BPD", "endpoints": [2, 3]},
    ]
    fetal_graph_meta = {
        "operator_ids": torch.tensor([0]),  # 0: DISTANCE
        "num_nodes": torch.tensor([[4.0]]),
        "num_edges": torch.tensor([[2.0]]),
    }

    ceph_operators = [
        {"type": "ANGLE", "name": "SNA", "endpoints": [0, 1, 4]},  # Sella(1), Nasion(2), A-point(5)
        {"type": "ANGLE", "name": "SNB", "endpoints": [0, 1, 5]},  # Sella(1), Nasion(2), B-point(6)
    ]
    ceph_graph_meta = {
        "operator_ids": torch.tensor([1]),  # 1: ANGLE
        "num_nodes": torch.tensor([[19.0]]),
        "num_edges": torch.tensor([[2.0]]),
    }

    # 2. Model Initialization (supports up to 32 landmarks, 4 modalities)
    model = EGSQModel(num_landmarks_max=32, feature_dim=128, num_modalities=4).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=0.05,
        mode="min",
        checkpoint_path=args.checkpoint_path,
        warmup_epochs=args.warmup,
        verbose=True,
    )

    # Helper evaluation function
    def evaluate_mre(loader, max_pts, operators, graph_meta):
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(args.device)
                pts = batch["landmarks"].to(args.device)
                sp = batch["spacing"].to(args.device)
                mod = batch["modality_id"].to(args.device)

                out = model(img, mod, graph_meta, operators, spacing=sp)
                p = out["coords"][:, :max_pts]
                preds.append(p.cpu().numpy())
                gts.append(pts.cpu().numpy())
        p_all = np.concatenate(preds, axis=0)
        g_all = np.concatenate(gts, axis=0)
        return float(compute_mre(p_all, g_all))

    # 3. Multi-task Training Loop with Early Stopping
    print("--- Training Phase with Early Stopping & LR Decay ---")
    final_epoch = args.epochs
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        # Alternate batches between Fetal US and Ceph X-ray
        iter_ceph = iter(loader_ceph_tr)
        for batch_f in loader_fetal_tr:
            try:
                batch_c = next(iter_ceph)
            except StopIteration:
                iter_ceph = iter(loader_ceph_tr)
                batch_c = next(iter_ceph)

            # Step A: Fetal US
            img_f = batch_f["image"].to(args.device)
            pts_f = batch_f["landmarks"].to(args.device)
            sp_f = batch_f["spacing"].to(args.device)
            mod_f = batch_f["modality_id"].to(args.device)

            optimizer.zero_grad()
            out_f = model(img_f, mod_f, fetal_graph_meta, fetal_operators, spacing=sp_f)
            pred_f = out_f["coords"][:, :pts_f.shape[1]]
            loss_lm_f = nn.functional.smooth_l1_loss(pred_f, pts_f)
            err_f = torch.norm(pred_f - pts_f, p=2, dim=-1).mean(dim=-1, keepdim=True)
            loss_rk_f = nn.functional.binary_cross_entropy(out_f["risk_score"], (err_f > 5.0).float())

            # Step B: Ceph X-ray
            img_c = batch_c["image"].to(args.device)
            pts_c = batch_c["landmarks"].to(args.device)
            sp_c = batch_c["spacing"].to(args.device)
            mod_c = batch_c["modality_id"].to(args.device)

            out_c = model(img_c, mod_c, ceph_graph_meta, ceph_operators, spacing=sp_c)
            pred_c = out_c["coords"][:, :pts_c.shape[1]]
            loss_lm_c = nn.functional.smooth_l1_loss(pred_c, pts_c)
            err_c = torch.norm(pred_c - pts_c, p=2, dim=-1).mean(dim=-1, keepdim=True)
            loss_rk_c = nn.functional.binary_cross_entropy(out_c["risk_score"], (err_c > 10.0).float())

            loss = (loss_lm_f + 0.3 * loss_rk_f) + (loss_lm_c + 0.3 * loss_rk_c)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)

        # Validate on Ceph & Fetal to compute joint monitor score
        val_mre_ceph = evaluate_mre(loader_ceph_te, 19, ceph_operators, ceph_graph_meta)
        val_mre_fetal = evaluate_mre(loader_fetal_te, 4, fetal_operators, fetal_graph_meta)
        joint_val_score = val_mre_ceph + val_mre_fetal

        scheduler.step(joint_val_score)
        curr_lr = optimizer.param_groups[0]["lr"]

        if epoch % 1 == 0:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] Loss: {avg_loss:.4f} | Ceph MRE: {val_mre_ceph:.2f}px | Fetal MRE: {val_mre_fetal:.2f}px | Joint: {joint_val_score:.2f} | LR: {curr_lr:.2e}")

        stop = early_stopping(joint_val_score, model, epoch)
        if stop:
            print(f"[Early Stopping Triggered] Stopping training at epoch {epoch}. Best epoch was {early_stopping.best_epoch}.")
            final_epoch = epoch
            break

    # 4. Evaluation & Conformal Calibration using Best Checkpoint
    print(f"--- Loading Best Model Checkpoint from {args.checkpoint_path} ---")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded Best Checkpoint from Epoch {checkpoint['epoch']} (Validation Metric: {checkpoint['metric_value']:.4f})")
    model.eval()

    def eval_task(loader, operators, graph_meta, max_pts):
        preds, gts, risks = [], [], []
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(args.device)
                pts = batch["landmarks"].to(args.device)
                sp = batch["spacing"].to(args.device)
                mod = batch["modality_id"].to(args.device)

                out = model(img, mod, graph_meta, operators, spacing=sp)
                p = out["coords"][:, :max_pts]
                preds.append(p.cpu().numpy())
                gts.append(pts.cpu().numpy())
                risks.append(out["risk_score"].cpu().numpy().squeeze(-1))

        p_all = np.concatenate(preds, axis=0)
        g_all = np.concatenate(gts, axis=0)
        r_all = np.concatenate(risks, axis=0)

        mre = compute_mre(p_all, g_all)
        sdr2 = compute_sdr(p_all, g_all, threshold=2.0)
        sdr4 = compute_sdr(p_all, g_all, threshold=4.0)

        sample_errs = np.mean(np.linalg.norm(p_all - g_all, axis=-1), axis=-1)
        covs, r_curve = compute_risk_coverage_curve(r_all, sample_errs)
        aurc = compute_aurc(covs, r_curve)

        n = len(sample_errs)
        calib_r, test_r = r_all[:n//2], r_all[n//2:]
        calib_e, test_e = sample_errs[:n//2], sample_errs[n//2:]

        conformal = SplitConformalRiskControl(alpha=float(np.percentile(calib_e, 30)))
        lam = conformal.calibrate(calib_r, calib_e)
        decisions = conformal.predict(test_r)
        accepted_err = test_e[decisions]
        emp_risk = float(np.mean(accepted_err)) if len(accepted_err) > 0 else 0.0
        coverage = float(np.mean(decisions) * 100.0)

        return float(mre), float(sdr2), float(sdr4), float(aurc), float(lam), emp_risk, coverage

    # Fetal US Test
    f_mre, f_sdr2, f_sdr4, f_aurc, f_lam, f_risk, f_cov = eval_task(
        loader_fetal_te, fetal_operators, fetal_graph_meta, max_pts=4
    )
    print("[Results: Fetal Ultrasound Biometry]")
    print(f"MRE: {f_mre:.2f}px | SDR@2mm: {f_sdr2:.1f}% | SDR@4mm: {f_sdr4:.1f}%")
    print(f"Selective AURC: {f_aurc:.2f} | Conformal Coverage: {f_cov:.1f}% | Controlled Risk: {f_risk:.2f}px")

    # Ceph X-ray Test
    c_mre, c_sdr2, c_sdr4, c_aurc, c_lam, c_risk, c_cov = eval_task(
        loader_ceph_te, ceph_operators, ceph_graph_meta, max_pts=19
    )
    print("[Results: ISBI 2015 Cephalometric X-ray]")
    print(f"MRE: {c_mre:.2f}px | SDR@2mm: {c_sdr2:.1f}% | SDR@4mm: {c_sdr4:.1f}%")
    print(f"Selective AURC: {c_aurc:.2f} | Conformal Coverage: {c_cov:.1f}% | Controlled Risk: {c_risk:.2f}px")

    # Save summary report
    os.makedirs("reports", exist_ok=True)
    report_dict = {
        "fetal_us": {"mre": f_mre, "sdr2": f_sdr2, "sdr4": f_sdr4, "aurc": f_aurc, "coverage": f_cov, "risk": f_risk},
        "ceph_xray": {"mre": c_mre, "sdr2": c_sdr2, "sdr4": c_sdr4, "aurc": c_aurc, "coverage": c_cov, "risk": c_risk},
        "best_epoch": early_stopping.best_epoch,
        "final_epoch": final_epoch,
    }
    np.savez("reports/e1_multitask_results.npz", **report_dict)
    print("Saved E1 results to reports/e1_multitask_results.npz")


if __name__ == "__main__":
    main()
