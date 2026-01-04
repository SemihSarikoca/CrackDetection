#!/usr/bin/env bash
set -euo pipefail

# 1. Klasör kök dizinini belirle
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

# 2. Giriş ve Çıkış yollarını tanımla
# Eğer scripti çalıştırırken argüman vermezsen varsayılan olarak senin istediğin yolları kullanır
INPUT_DIR="${1:-test_building/images}"
OUTPUT_DIR="${2:-test_building_outputs}"
EXTRA_ARGS=("${@:3}")

FULL_INPUT_PATH="$ROOT_DIR/$INPUT_DIR"
FULL_OUTPUT_PATH="$ROOT_DIR/$OUTPUT_DIR"

# 3. Giriş klasörü var mı kontrol et
if [[ ! -d "$FULL_INPUT_PATH" ]]; then
    echo "Error: Input directory not found: '$FULL_INPUT_PATH'" >&2
    exit 1
fi

# 4. Çıkış klasörünü oluştur (yoksa)
mkdir -p "$FULL_OUTPUT_PATH"

# 5. PYTHONPATH ayarını yap
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

# 6. İşlemi başlat
echo "Starting..."
echo "Input: $FULL_INPUT_PATH"
echo "Output: $FULL_OUTPUT_PATH"

"$PYTHON_BIN" -m crack_detection.cli \
    --image-root "$FULL_INPUT_PATH" \
    --visualize \
    --output-dir "$FULL_OUTPUT_PATH" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "Completed!"