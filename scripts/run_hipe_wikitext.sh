#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL_SIZE="${MODEL_SIZE:-60M}"
SEQ_LEN="${SEQ_LEN:-1024}"
SIGMA="${SIGMA:-500}"
THRESHOLD="${THRESHOLD:-3}"
SEED="${SEED:-6198}"
RUN_NAME="${RUN_NAME:-hipe_wikitext_${MODEL_SIZE}_L${SEQ_LEN}}"
DATA_DIR="${WIKITEXT_DIR:-${DATA_ROOT}/wikitext}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/${RUN_NAME}/seed_${SEED}}"

FLASH_ARGS=()
if [[ "${FLASH_ATTENTION:-0}" == "1" ]]; then
  FLASH_ARGS+=(--flash-attention)
fi

"${PYTHON_BIN}" -m hipe.train \
  --output-dir "${OUTPUT_DIR}" \
  --data-path "${DATA_DIR}/raw" \
  --tokenizer-path "${DATA_DIR}/tokenizer" \
  --model-size "${MODEL_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --sigma "${SIGMA}" \
  --rope-scaling-threshold "${THRESHOLD}" \
  --seed "${SEED}" \
  "${FLASH_ARGS[@]}"
