#!/usr/bin/env bash
# Parallel Remote Suite: Runs Baselines, Ablations, Stress-tests, and Selective Contest in Parallel
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

mkdir -p reports checkpoints
PY="/root/miniconda3/bin/python"

echo "================================================================="
echo "   EGSQ Full Multi-Task & Parallel Baselines Evaluation Suite    "
echo "================================================================="

# Wait until pipeline completes checkpoint creation
echo "Waiting for best converged model checkpoint..."
while [ ! -f "checkpoints/best_egsq_multitask.pth" ]; do
    sleep 2
done

echo "Found best checkpoint. Spawning Parallel Experiment Engines on GPU/CPU..."

# Task 1: Baseline Official BiometryNet HRNet Comparison
$PY -u src/eval_baselines.py > reports/baselines_hrnet.log 2>&1 &
PID_BASE=$!
echo "  [Started] Official HRNet Baseline Evaluator (PID: $PID_BASE)"

# Task 2: E4 Selective Prediction Contest (8 Baselines/Competitors: Peak Inv, Entropy, Ensemble, GEC)
$PY -u src/eval_e4_selective_contest.py > reports/e4_contest.log 2>&1 &
PID_E4=$!
echo "  [Started] E4 Selective Contest Engine (PID: $PID_E4)"

# Task 3: E5 Controlled Stress Degradation (Speckle, Occlusion, Contrast)
$PY -u src/eval_stress_test.py > reports/e5_stress.log 2>&1 &
PID_E5=$!
echo "  [Started] E5 Stress Testing Engine (PID: $PID_E5)"

# Task 4: E6 Modular Ablation Analysis (w/o TCMG, w/o DGD, w/o GEC)
$PY -u src/eval_ablation.py > reports/e6_ablation.log 2>&1 &
PID_E6=$!
echo "  [Started] E6 Modular Ablation Engine (PID: $PID_E6)"

# Task 5: E2 Cross-Domain Out-of-Distribution Generalization (FP -> HC18 -> UCL)
$PY -u src/eval_e2_cross_domain.py > reports/e2_domain.log 2>&1 &
PID_E2=$!
echo "  [Started] E2 Cross-Domain Generalization Engine (PID: $PID_E2)"

echo ""
echo "All 5 benchmark processes launched in parallel on RTX 2080 Ti."
echo "Waiting for all parallel tasks to converge..."
wait $PID_BASE $PID_E4 $PID_E5 $PID_E6 $PID_E2

echo ""
echo "=== Parallel Benchmark Engines Finished. Running E3 Few-Shot Sweep ==="
$PY -u src/eval_e3_few_shot.py > reports/e3_few_shot.log 2>&1

echo ""
echo "================================================================="
echo "   All 8 Contenders & E1-E6 Experimental Results Fully Computed  "
echo "================================================================="
ls -lh reports/*.npz
