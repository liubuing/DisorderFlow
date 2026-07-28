#!/bin/bash
# Resume and verify the official SAbDab2 machine-learning dataset.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$PROJECT_DIR/data/sabdab2_current/splits.tar.gz"
URL="https://zenodo.org/api/records/20083995/files/splits.tar.gz/content"
EXPECTED_SIZE=876381859
EXPECTED_MD5="0dbb4cc499e9eb77f14008b232f2c38c"

if [[ "${1:-}" == "--wait-for-v5" ]]; then
    while pgrep -f '[b]uild_conformation_cf.py' >/dev/null; do
        sleep 300
    done
fi

mkdir -p "$(dirname "$ARCHIVE")"
curl -L --fail --retry 10 --retry-delay 10 --retry-all-errors \
    --continue-at - --output "$ARCHIVE" "$URL"

ACTUAL_SIZE=$(stat -c '%s' "$ARCHIVE")
if [[ "$ACTUAL_SIZE" -ne "$EXPECTED_SIZE" ]]; then
    echo "Size mismatch: $ACTUAL_SIZE != $EXPECTED_SIZE" >&2
    exit 1
fi

echo "$EXPECTED_MD5  $ARCHIVE" | md5sum --check --status
tar -tzf "$ARCHIVE" >/dev/null
echo "SAbDab2 ML dataset verified: $ARCHIVE"
