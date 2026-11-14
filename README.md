# EGSQ: Evidence-Grounded Selective Quantification in Safety-Critical Visual Informatics

Official PyTorch implementation of **EGSQ** (Evidence-Grounded Selective Quantification), a unified framework for task-conditioned geometric reasoning and distribution-free risk-controlled selective prediction.

---

## Method Overview

Quantitative measurement from noisy, heterogeneous visual sensors requires topological relational priors and certified risk control. EGSQ introduces three synergistic components:
1. **Task-Conditioned Measurement Graphs (TCMG)**: Formalizes structural landmarks, metric operators (caliper distances, angles, morphological ratios), and calibration specifications as invariant structured graphs.
2. **Differentiable Geometric Decoders (DGD)**: Lossless analytical mapping from spatial probability mass expectations to continuous Euclidean metric spaces via soft-argmax operators.
3. **Geometric Evidence Consistency (GEC) & Conformal Risk Control (CRC)**: Finite-sample distribution-free statistical risk bounds guaranteeing that empirical risk under accepted predictions satisfies user-specified risk tolerance $\alpha$.

---

## Directory Structure

```text
├── configs/            # Task configurations and training hyperparameters
│   ├── base_config.yaml
│   ├── fetal_biometry.yaml
│   └── cephalometric_aariz.yaml
├── src/                # Core framework source code
│   ├── data/           # Dataset loaders and geometry transformations
│   ├── metrics/        # MRE, SDR, and Risk-Coverage evaluation metrics
│   ├── models/         # TCMG, DGD, and GEC network architectures
│   └── utils/          # Finite-sample split conformal risk control
├── scripts/            # Environment setup and execution scripts
├── reports/            # Quantitative evaluation reports (.npz) and curves
└── README.md
```

---

## Installation & Environment

```bash
git clone https://github.com/Join-xiaobai/EGSQ.git
cd EGSQ
pip install torch torchvision numpy scipy pyyaml
```

---

## Reproducing Paper Benchmarks

### 1. Modular Ablation Evaluation (Table 3)
```bash
python src/eval_ablation.py
```

### 2. Sensor Corruption & Stress Testing (Table 4)
```bash
python src/eval_stress_test.py
```

### 3. Multi-Task Training
```bash
python src/train_multitask.py --config configs/base_config.yaml
```

---

## License

This project is released under the MIT License.
