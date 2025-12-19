#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: recursive_unarchive.sh <root_dir>

Recursively finds archives (7z, zip, tar, tar.gz/tgz, tar.bz2/tbz/tbz2)
under the provided directory and expands each archive into a directory
named after the archive (minus its extension) within the parent folder.
Existing files will be overwritten during extraction.
USAGE
}

err() {
  echo "Error: $*" >&2
  exit 1
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Required command '$1' is not available in PATH"
  fi
}

extract_archive() {
  local archive="$1"
  local parent_dir dest_name dest_dir
  local base lower

  parent_dir=$(dirname "$archive")
  base=$(basename "$archive")
  lower=$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')
  dest_name=${base%.*}

  case "$lower" in
    *.tar.gz)
      dest_name=${dest_name%.*}
      dest_dir="$parent_dir/$dest_name"
      mkdir -p "$dest_dir"
      require_cmd tar
      tar -xzf "$archive" -C "$dest_dir"
      ;;
    *.tgz)
      dest_dir="$parent_dir/$dest_name"
      mkdir -p "$dest_dir"
      require_cmd tar
      tar -xzf "$archive" -C "$dest_dir"
      ;;
    *.tar.bz2)
      dest_name=${dest_name%.*}
      dest_dir="$parent_dir/$dest_name"
      mkdir -p "$dest_dir"
      require_cmd tar
      tar -xjf "$archive" -C "$dest_dir"
      ;;
    *.tbz|*.tbz2)
      dest_dir="$parent_dir/$dest_name"
      mkdir -p "$dest_dir"
      require_cmd tar
      tar -xjf "$archive" -C "$dest_dir"
      ;;
    *.tar)
      dest_dir="$parent_dir/$dest_name"
      mkdir -p "$dest_dir"
      require_cmd tar
      tar -xf "$archive" -C "$dest_dir"
      ;;
    *.zip)
      dest_dir="$parent_dir/$dest_name"
      mkdir -p "$dest_dir"
      require_cmd unzip
      unzip -oq "$archive" -d "$dest_dir"
      ;;
    *.7z)
      dest_dir="$parent_dir/$dest_name"
      mkdir -p "$dest_dir"
      require_cmd 7z
      7z x -aoa -y -o"$dest_dir" "$archive" >/dev/null
      ;;
    *)
      return 1
      ;;
  esac

  printf 'Extracted %s -> %s\n' "$archive" "$dest_dir"
  return 0
}

if [ "$#" -ne 1 ]; then
  usage >&2
  exit 1
fi

root_dir=${1%/}

if [ ! -d "$root_dir" ]; then
  err "Directory '$root_dir' does not exist or is not a directory"
fi

root_dir=$(cd "$root_dir" && pwd)

find "$root_dir" -type f -print0 | while IFS= read -r -d '' file; do
  extract_archive "$file" || true
done
