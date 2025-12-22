#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"

if ! command -v qpdf >/dev/null 2>&1; then
  echo "qpdf is required (install via your package manager)." >&2
  exit 1
fi

find "$root" -type f -iname '*.pdf' -print0 |
while IFS= read -r -d '' pdf; do
  if ! qpdf --check "$pdf" >/dev/null 2>&1; then
    echo "Invalid PDF: $pdf"
  fi
done
