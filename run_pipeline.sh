#!/usr/bin/env bash
set -euo pipefail

#.Resolve repository root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

DATASET_ROOT="${1:-dataset/cracktile}"
OUTPUT_ROOT="${2:-outputs/cracktile}"
EXTRA_ARGS=("${@:3}")

FULL_DATASET_ROOT="$ROOT_DIR/$DATASET_ROOT"
if [[ ! -d "$FULL_DATASET_ROOT" ]]; then
  echo "Dataset root '$FULL_DATASET_ROOT' not found. Provide a valid directory as the first argument." >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

declare -a SPLITS=("train" "test")
for split in "${SPLITS[@]}"; do
  split_root="$FULL_DATASET_ROOT/$split"
  if [[ ! -d "$split_root" ]]; then
    echo "Skipping '$split_root' because it does not exist."
    continue
  fi

  output_dir="$ROOT_DIR/$OUTPUT_ROOT/$split"
  mkdir -p "$output_dir"

  echo "Running pipeline for $split_root -> $output_dir"
  "$PYTHON_BIN" -m crack_detection.cli \
    --image-root "$split_root" \
    --visualize \
    --output-dir "$output_dir" \
    "${EXTRA_ARGS[@]}"
done
