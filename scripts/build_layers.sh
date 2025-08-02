#!/usr/bin/env bash
# Build Lambda Layers for scientific Python stack and keep the handler ZIP tiny.
# Creates:
#   mal/layers/sk1.zip  (numpy, scipy, joblib, threadpoolctl)
#   mal/layers/sk2.zip  (pandas, scikit-learn)  -- installed with --no-deps
# Usage (WSL):
#   dos2unix scripts/build_layers.sh && chmod +x scripts/build_layers.sh
#   bash scripts/build_layers.sh
set -Eeuo pipefail

# ---- Settings ---------------------------------------------------------------
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAYERS_DIR="${LAYERS_DIR:-$ROOT/.lambda_layers}"

# Python / Lambda base
PYVER="${PYVER:-3.11}"
BASE_IMAGE="public.ecr.aws/lambda/python:${PYVER}"

# Allow forcing Docker architecture if needed (e.g., DOCKER_PLATFORM=--platform=linux/amd64)
DOCKER_PLATFORM="${DOCKER_PLATFORM:-}"

# Pin versions to Lambda-compatible manylinux wheels
NUMPY="${NUMPY:-1.26.4}"
SCIPY="${SCIPY:-1.11.4}"
PANDAS="${PANDAS:-2.1.4}"
SKLEARN="${SKLEARN:-1.3.2}"
JOBLIB="${JOBLIB:-1.3.2}"
THREADPOOLCTL="${THREADPOOLCTL:-3.2.0}"

# ---- Workspace --------------------------------------------------------------
WORKDIR="$(mktemp -d "${ROOT}/.layers_tmp_XXXXXX")"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

# Ensure final target is a real directory (handle broken symlinks or stray files)
if [ -L "$LAYERS_DIR" ] || [ -f "$LAYERS_DIR" ] || { [ -e "$LAYERS_DIR" ] && [ ! -d "$LAYERS_DIR" ]; }; then
  rm -f -- "$LAYERS_DIR"
fi
mkdir -p -- "$LAYERS_DIR"

echo "→ Building layers in: $WORKDIR"
echo "→ Output directory   : $LAYERS_DIR"
echo "→ Lambda base image  : $BASE_IMAGE (PY $PYVER)"
echo

# ---- Build inside Lambda base image for ABI-compatible wheels ---------------
docker run --rm $DOCKER_PLATFORM \
  --entrypoint /bin/bash \
  -e PIP_ONLY_BINARY=":all:" \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -v "$WORKDIR:/layers" \
  "$BASE_IMAGE" -lc "
    set -Eeuo pipefail
    echo ':: Python:'; python -V
    python -m pip install --upgrade pip wheel setuptools

    mkdir -p /layers/sk1/python/lib/python${PYVER}/site-packages
    mkdir -p /layers/sk2/python/lib/python${PYVER}/site-packages

    echo ':: Installing layer sk1 (numpy, scipy, joblib, threadpoolctl)'
    python -m pip install --no-cache-dir \
      -t /layers/sk1/python/lib/python${PYVER}/site-packages \
      numpy==${NUMPY} \
      scipy==${SCIPY} \
      joblib==${JOBLIB} \
      threadpoolctl==${THREADPOOLCTL}

    echo ':: Installing layer sk2 (pandas, scikit-learn) with --no-deps'
    python -m pip install --no-cache-dir --no-deps \
      -t /layers/sk2/python/lib/python${PYVER}/site-packages \
      pandas==${PANDAS} \
      scikit-learn==${SKLEARN}

    echo ':: Zipping layers with correct internal layout (python/...)'
    cd /layers/sk1 && python -m zipfile -c ../sk1.zip python
    cd /layers/sk2 && python -m zipfile -c ../sk2.zip python
  "

# ---- Move artifacts out of temp dir -----------------------------------------
mv -f "$WORKDIR"/sk1.zip "$LAYERS_DIR"/
mv -f "$WORKDIR"/sk2.zip "$LAYERS_DIR"/

echo
echo "✅ Built layers:"
ls -lh "$LAYERS_DIR"/sk*.zip
echo
echo "Tip: If Lambda can't import modules, verify the layer structure via:"
echo "  unzip -l \"$LAYERS_DIR/sk1.zip\" | head"
echo "  unzip -l \"$LAYERS_DIR/sk2.zip\" | head"
