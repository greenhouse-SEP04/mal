#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"

rm -rf "$BUILD"
mkdir -p "$BUILD"

cd "$ROOT/src"
zip -qr "$BUILD/ml_service.zip" handler.py
echo "Created $BUILD/ml_service.zip"
