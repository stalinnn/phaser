#!/usr/bin/env bash
set -euo pipefail

# Run all paper experiments sequentially and save logs.
# Usage:
#   bash run_all_experiments.sh
#
# Notes:
# - This is a .sh script (for Git Bash / WSL / Linux / macOS).
# - On Windows PowerShell, prefer: `powershell -ExecutionPolicy Bypass -File run_all_experiments.ps1`

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="paper_archive/results/logs"
mkdir -p "$LOG_DIR"

# Set SKIP_LARGE=1 to skip the slowest "Large (454M)" config runs
# Example:
#   SKIP_LARGE=1 bash run_all_experiments.sh
export SKIP_LARGE="${SKIP_LARGE:-0}"

run_one () {
  local name="$1"
  local script="$2"
  local log="$LOG_DIR/${name}.log"
  echo "=== Running: ${name} ==="
  echo "Script: ${script}"
  echo "Log: ${log}"
  echo "Env: SKIP_LARGE=${SKIP_LARGE}"
  echo "Started: $(date -Iseconds)"
  # -u for unbuffered logs
  "${PYTHON_BIN}" -u "${script}" 2>&1 | tee "${log}"
  echo "Finished: $(date -Iseconds)"
  echo
}

run_one "01_experiment_scaling_law_baseline_transformer" "paper_archive/code/experiment_scaling_law_baseline_transformer.py"
run_one "02_experiment_scaling_law_sparse_tgn" "paper_archive/code/experiment_scaling_law_sparse_tgn.py"
run_one "03_experiment_scaling_law_tgnblock" "paper_archive/code/experiment_scaling_law.py"
run_one "04_experiment_tgn_topk_training" "paper_archive/code/experiment_tgn_topk_training.py"

echo "All done. Logs in: ${LOG_DIR}"

