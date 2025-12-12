#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-}
shift || true
LOG_FILE="batch_failures.log"

if [[ -z "${ROOT_DIR}" ]]; then
  echo "Usage: $0 <root-directory> [provider] [app.py args...]" >&2
  exit 1
fi

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "Root directory not found: ${ROOT_DIR}" >&2
  exit 1
fi

PROVIDER="openai"
EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -gt 0 && "${EXTRA_ARGS[0]}" != --* ]]; then
  PROVIDER="${EXTRA_ARGS[0]}"
  EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
fi

has_provider_flag=false
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[i]}"
  if [[ "${arg}" == --provider || "${arg}" == --provider=* ]]; then
    has_provider_flag=true
    break
  fi
done

PROVIDER_ARGS=()
if [[ "${has_provider_flag}" == false ]]; then
  PROVIDER_ARGS=(--provider "${PROVIDER}")
fi

PDF_COUNT=$(find "${ROOT_DIR}" -type f -name '*.pdf' | wc -l | tr -d ' ')
echo "Found ${PDF_COUNT} PDF(s) to process under ${ROOT_DIR}"

find "${ROOT_DIR}" -type f -name '*.pdf' -print0 | while IFS= read -r -d '' pdf; do
  echo "Processing ${pdf}"
  if ! python app.py "${pdf}" --base-dir="${ROOT_DIR}" "${PROVIDER_ARGS[@]}" "${EXTRA_ARGS[@]}"; then
    echo "Failed to process ${pdf}, continuing..." >&2
    echo "${pdf}" >> "${LOG_FILE}"
  fi
done
