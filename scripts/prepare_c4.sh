#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

C4_DIR="${C4_DIR:-${DATA_ROOT}/c4}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${DATA_ROOT}/wikitext/tokenizer}"
C4_COUNT_ARGS=()
if [[ -n "${TRAIN_DOCS:-}" || -n "${VALIDATION_DOCS:-}" ]]; then
  if [[ -z "${TRAIN_DOCS:-}" || -z "${VALIDATION_DOCS:-}" ]]; then
    echo "TRAIN_DOCS and VALIDATION_DOCS must be set together" >&2
    exit 2
  fi
  C4_COUNT_ARGS+=(--train-docs "${TRAIN_DOCS}" --validation-docs "${VALIDATION_DOCS}")
else
  C4_COUNT_ARGS+=(
    --total-docs "${TOTAL_DOCS:-30000000}"
    --train-fraction "${TRAIN_FRACTION:-0.90}"
    --validation-fraction "${VALIDATION_FRACTION:-0.05}"
  )
fi

CACHE_ARGS=()
if [[ -n "${HF_CACHE_DIR:-}" ]]; then
  CACHE_ARGS+=(--cache-dir "${HF_CACHE_DIR}")
fi

"${PYTHON_BIN}" -m hipe.prepare_c4 \
  --output-dir "${C4_DIR}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --tokenizer-name "${TOKENIZER_NAME:-EleutherAI/gpt-neox-20b}" \
  --dataset-name "${C4_DATASET_NAME:-allenai/c4}" \
  --dataset-config "${C4_DATASET_CONFIG:-en}" \
  --shard-size "${SHARD_SIZE:-100000}" \
  --seed "${SEED:-6198}" \
  "${C4_COUNT_ARGS[@]}" \
  "${CACHE_ARGS[@]}"
