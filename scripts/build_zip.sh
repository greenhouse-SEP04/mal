#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"

rm -rf "$BUILD"
mkdir -p "$BUILD"

# Put the handler at build/handler.py
cp "$ROOT/src/greenhouse_ml_service.py" "$BUILD/handler.py"

# Only zip the handler (deps come from layers)
cd "$BUILD"
zip -qr ml_service.zip handler.py
echo "Created $BUILD/ml_service.zip"
