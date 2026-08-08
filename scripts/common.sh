#!/usr/bin/env bash

set -euo pipefail

HIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${HIPE_DATA_DIR:-${HIPE_ROOT}/data}"
ARTIFACT_ROOT="${HIPE_ARTIFACTS_DIR:-${HIPE_ROOT}/artifacts}"

export PYTHONPATH="${HIPE_ROOT}:${PYTHONPATH:-}"

mkdir -p "${ARTIFACT_ROOT}"

