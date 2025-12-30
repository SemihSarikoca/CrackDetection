#!/usr/bin/env bash
set -euo pipefail

#.Resolve repository root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

IMAGE_ROOT="${1:-dataset}"
OUTPUT_DIR="${2:-outputs}"

IMAGE_PATH="$ROOT_DIR/$IMAGE_ROOT"
OUTPUT_PATH="$ROOT_DIR/$OUTPUT_DIR"

if [[ ! -d "$IMAGE_PATH" ]]; then
  echo "Image root '$IMAGE_PATH' not found. Provide a valid directory as the first argument." >&2
  exit 1
fi

mkdir -p "$OUTPUT_PATH"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

exec "$PYTHON_BIN" -m crack_detection.cli \
  --image-root "$IMAGE_PATH" \
  --visualize \
  --output-dir "$OUTPUT_PATH"
