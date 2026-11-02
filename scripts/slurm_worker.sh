#!/usr/bin/env bash
set -euo pipefail
MANIFEST="${1:?campaign manifest is required}"
PYTHON_BIN="${2:?absolute Python executable is required}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m hipe.experiments worker --manifest "${MANIFEST}"
