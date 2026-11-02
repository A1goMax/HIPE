#!/usr/bin/env bash
set -euo pipefail
MANIFEST="${1:?campaign manifest is required}"
PYTHON_BIN="${2:?absolute Python executable is required}"
# sbatch copies this script into its spool directory. The submitted --chdir
# selects the frozen source checkout, so BASH_SOURCE is not a repository path.
REPO_ROOT="$PWD"
if [[ ! -f "${REPO_ROOT}/hipe/experiments.py" ]]; then
  echo "SLURM working directory is not the frozen HIPE checkout: ${REPO_ROOT}" >&2
  exit 2
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m hipe.experiments worker --manifest "${MANIFEST}"
