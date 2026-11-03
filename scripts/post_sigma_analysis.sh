#!/usr/bin/env bash
set -euo pipefail
SWEEP_JOB="${1:?sigma sweep job ID is required}"
OUTPUT_ROOT="${2:?analysis output directory is required}"
PYTHON_BIN="${3:-/home/qijunrong/anaconda3/bin/python}"
BASELINE_INPUT="${4:-/home/qijunrong/03-proj/HIPE/outputs/table2_20261103/raw_runs.json}"
REPO_ROOT="$PWD"
test -f "${REPO_ROOT}/analysis/table2/best_sigma.py"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}"
"${PYTHON_BIN}" analysis/table2/collect_campaign.py --jobs "${SWEEP_JOB}" --output "${OUTPUT_ROOT}/new_sigma_runs.json"
"${PYTHON_BIN}" analysis/table2/best_sigma.py --input "${BASELINE_INPUT}" "${OUTPUT_ROOT}/new_sigma_runs.json" --output "${OUTPUT_ROOT}"
