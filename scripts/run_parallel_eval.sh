#!/usr/bin/env bash
# Parallel Evaluation Suite for EGSQ Benchmark (E2, E3, E4, E5, E6)
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=== Launching EGSQ Evaluation Suite with best converged checkpoint ==="

# E2: Cross-domain
python3 -u src/eval_e2_cross_domain.py > reports/e2.log 2>&1 &
PID_E2=$!

# E4: Selective prediction contest
python3 -u src/eval_e4_selective_contest.py > reports/e4.log 2>&1 &
PID_E4=$!

# E5: Stress test
python3 -u src/eval_stress_test.py > reports/e5.log 2>&1 &
PID_E5=$!

# E6: Ablation study
python3 -u src/eval_ablation.py > reports/e6.log 2>&1 &
PID_E6=$!

echo "Waiting for parallel inference jobs (E2: $PID_E2, E4: $PID_E4, E5: $PID_E5, E6: $PID_E6)..."
wait $PID_E2 $PID_E4 $PID_E5 $PID_E6

echo "Inference suites finished. Running E3 (few-shot adaptation with quick fine-tuning)..."
python3 -u src/eval_e3_few_shot.py > reports/e3.log 2>&1

echo "All experiments completed successfully!"
