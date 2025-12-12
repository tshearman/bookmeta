#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-}
shift || true
LOG_FILE="batch_failures.log"

if [[ -z "${ROOT_DIR}" ]]; then
  echo "Usage: $0 <root-directory> [app.py args...]" >&2
  exit 1
fi

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "Root directory not found: ${ROOT_DIR}" >&2
  exit 1
fi

PDF_COUNT=$(find "${ROOT_DIR}" -type f -name '*.pdf' | wc -l | tr -d ' ')
echo "Found ${PDF_COUNT} PDF(s) to process under ${ROOT_DIR}"

find "${ROOT_DIR}" -type f -name '*.pdf' -print0 | while IFS= read -r -d '' pdf; do
  echo "Processing ${pdf}"
  if ! python app.py "${pdf}" --base-dir="${ROOT_DIR}" "$@"; then
    echo "Failed to process ${pdf}, continuing..." >&2
    echo "${pdf}" >> "${LOG_FILE}"
  fi
done
