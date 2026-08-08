#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL_SIZE="${MODEL_SIZE:-300M}"
SEQ_LEN="${SEQ_LEN:-2048}"
SIGMA="${SIGMA:-700}"
THRESHOLD="${THRESHOLD:-7}"
SEED="${SEED:-6198}"
RUN_NAME="${RUN_NAME:-hipe_c4_${MODEL_SIZE}_L${SEQ_LEN}}"
C4_DIR="${C4_DIR:-${DATA_ROOT}/c4}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${DATA_ROOT}/wikitext/tokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/${RUN_NAME}/seed_${SEED}}"

FLASH_ARGS=()
if [[ "${FLASH_ATTENTION:-0}" == "1" ]]; then
  FLASH_ARGS+=(--flash-attention)
fi

"${PYTHON_BIN}" -m hipe.train \
  --output-dir "${OUTPUT_DIR}" \
  --data-path "${C4_DIR}/c4_30M_train" \
  --validation-path "${C4_DIR}/c4_30M_validation" \
  --tokenizer-path "${TOKENIZER_DIR}" \
  --model-size "${MODEL_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --sigma "${SIGMA}" \
  --rope-scaling-threshold "${THRESHOLD}" \
  --seed "${SEED}" \
  "${FLASH_ARGS[@]}"
