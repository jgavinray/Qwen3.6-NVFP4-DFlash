#!/usr/bin/env bash
# Build vllm-rtx6000-qwen36-27b for RTX PRO 6000 / GB202.
#
# Usage:
#   ./scripts/build.sh [TAG] [VLLM_REF]
#
# Defaults:
#   TAG=v0.1
#   VLLM_REF=refs/pull/40898/head
set -euo pipefail

TAG="${1:-v0.1}"
VLLM_REF="${2:-refs/pull/40898/head}"
IMAGE_LOCAL="vllm-rtx6000-qwen36-27b:${TAG}"

cd "$(dirname "$0")/.."

echo "== Building ${IMAGE_LOCAL} =="
echo "  source: ./Dockerfile"
echo "  vllm ref: ${VLLM_REF}"
echo "  context: $(pwd)"
echo "  cores: $(nproc)"
echo "  ram: $(free -g | awk 'NR==2{print $2}') GB"
echo

if ! command -v docker >/dev/null; then
  echo "ERROR: docker not installed" >&2
  exit 1
fi

if ! docker info | grep -qi nvidia; then
  echo "WARN: nvidia runtime not detected; build may work but runtime GPU tests will not" >&2
fi

LOG="/tmp/vllm-rtx6000-qwen36-27b-build-$(date +%s).log"
echo "== Log: $LOG =="
echo

docker build \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  -t "${IMAGE_LOCAL}" \
  -f Dockerfile \
  . 2>&1 | tee "$LOG"

echo
echo "== Done =="
echo "  image: ${IMAGE_LOCAL}"
echo "  size:  $(docker images --format '{{.Size}}' "${IMAGE_LOCAL}" | head -1)"
echo "  log:   $LOG"
echo
echo "Next:"
echo "  docker compose -f examples/docker-compose.yml up -d"
echo "  ./scripts/save-image.sh ${TAG}      # optional local export"
echo "  ./scripts/push-ghcr.sh ${TAG}       # optional, only if you have GHCR access"
