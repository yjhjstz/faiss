#!/usr/bin/env bash
# Build the faiss-gpu delivery image.
# Run from repo root.

set -euo pipefail

IMAGE="${IMAGE:-faiss-gpu-sq:1.8.0}"

# 1) Ensure wheel is up-to-date
if [[ ! -f build/faiss/python/dist/faiss-*.whl ]]; then
  echo "[build] no wheel found, generating one..."
  (cd build/faiss/python && python setup.py bdist_wheel)
fi

# 2) Build Docker image
echo "[build] docker build -> ${IMAGE}"
docker build -f docker/Dockerfile -t "${IMAGE}" .

echo ""
echo "[build] done. Run with:"
echo "  docker run --rm --gpus all -v \$PWD:/workspace -it ${IMAGE}"
