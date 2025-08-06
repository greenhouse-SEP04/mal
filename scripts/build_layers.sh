#!/usr/bin/env bash
# Build Lambda Layers
# Produces:
#   .lambda_layers/sk1.zip  (numpy, scipy, joblib, threadpoolctl)
#   .lambda_layers/sk2.zip  (pandas, scikit-learn)
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAYERS_DIR="${LAYERS_DIR:-$ROOT/.lambda_layers}"
PYVER="${PYVER:-3.11}"

# Versions (keep in sync with requirements.txt)
NUMPY="${NUMPY:-1.26.4}"
SCIPY="${SCIPY:-1.11.4}"
PANDAS="${PANDAS:-2.1.4}"
SKLEARN="${SKLEARN:-1.3.2}"
JOBLIB="${JOBLIB:-1.3.2}"
THREADPOOLCTL="${THREADPOOLCTL:-3.2.0}"

TMP="$(mktemp -d "${ROOT}/.layers_tmp_XXXXXX")"
cleanup(){
  rm -rf "$TMP" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$LAYERS_DIR" \
         "$TMP/wheels" \
         "$TMP/sk1/python/lib/python${PYVER}/site-packages" \
         "$TMP/sk2/python/lib/python${PYVER}/site-packages"

echo "→ Downloading wheels (manylinux) to $TMP/wheels"
python -m pip install --upgrade pip wheel setuptools >/dev/null
python -m pip download --only-binary=:all: \
  -d "$TMP/wheels" \
  numpy=="$NUMPY" scipy=="$SCIPY" joblib=="$JOBLIB" threadpoolctl=="$THREADPOOLCTL" \
  pandas=="$PANDAS" scikit-learn=="$SKLEARN"

echo "→ Installing into layer directories (from local wheels only)"
python -m pip install --no-index --find-links "$TMP/wheels" \
  -t "$TMP/sk1/python/lib/python${PYVER}/site-packages" \
  numpy=="$NUMPY" scipy=="$SCIPY" joblib=="$JOBLIB" threadpoolctl=="$THREADPOOLCTL"

python -m pip install --no-index --find-links "$TMP/wheels" --no-deps \
  -t "$TMP/sk2/python/lib/python${PYVER}/site-packages" \
  pandas=="$PANDAS" scikit-learn=="$SKLEARN"

echo "→ Pruning tests and caches"
for d in "$TMP/sk1/python/lib/python${PYVER}/site-packages" "$TMP/sk2/python/lib/python${PYVER}/site-packages"; do
  find "$d" -type d \( -iname tests -o -iname testing -o -iname __pycache__ \) -prune -exec rm -rf {} + || true
  find "$d" -type f -name '*.pyc' -delete || true
done

echo "→ Zipping layers"
( cd "$TMP/sk1" && python -m zipfile -c "$LAYERS_DIR/sk1.zip" python )
( cd "$TMP/sk2" && python -m zipfile -c "$LAYERS_DIR/sk2.zip" python )

echo "✅ Built:"
ls -lh "$LAYERS_DIR"/sk*.zip
