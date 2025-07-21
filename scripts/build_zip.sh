#!/usr/bin/env bash
set -e
rm -rf build
mkdir -p build
zip -r build/ml_service.zip src