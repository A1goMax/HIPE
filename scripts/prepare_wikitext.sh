#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

"${PYTHON_BIN}" -m hipe.prepare_wikitext \
  --output-dir "${WIKITEXT_DIR:-${DATA_ROOT}/wikitext}" \
  --tokenizer-name "${TOKENIZER_NAME:-EleutherAI/gpt-neox-20b}"
