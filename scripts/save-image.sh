#!/usr/bin/env bash
# Save the locally built RTX PRO 6000 image to a tarball.
#
# Usage:
#   ./scripts/save-image.sh [TAG] [OUTPUT]
set -euo pipefail

TAG="${1:-v0.1}"
IMAGE="vllm-rtx6000-qwen36-27b:${TAG}"
OUT="${2:-vllm-rtx6000-qwen36-27b-${TAG}.tar}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "ERROR: image ${IMAGE} not found locally. Run ./scripts/build.sh ${TAG} first." >&2
  exit 1
fi

echo "== Saving ${IMAGE} -> ${OUT} =="
docker save "${IMAGE}" -o "${OUT}"
ls -lh "${OUT}"
