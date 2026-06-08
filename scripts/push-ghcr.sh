#!/usr/bin/env bash
# Optional helper: tag and push vllm-rtx6000-qwen36-27b to GitHub Container Registry.
#
# You do NOT need this for local use. Run the image directly after build, or use
# ./scripts/save-image.sh if you want a local tarball.
#
# Prerequisites:
#   export GITHUB_TOKEN=ghp_xxxxxxxxx
#   export GITHUB_USER=<github-user>
#
# Usage:
#   ./scripts/push-ghcr.sh [TAG]
set -euo pipefail

TAG="${1:-v0.1}"
GH_USER="${GITHUB_USER:-jgavinray}"
IMAGE="vllm-rtx6000-qwen36-27b"
LOCAL="${IMAGE}:${TAG}"
REMOTE="ghcr.io/${GH_USER}/${IMAGE}:${TAG}"
REMOTE_LATEST="ghcr.io/${GH_USER}/${IMAGE}:latest"

if ! docker image inspect "${LOCAL}" >/dev/null 2>&1; then
  echo "ERROR: image ${LOCAL} not found locally. Run ./scripts/build.sh first." >&2
  exit 1
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: GHCR push is optional but requires an account/token." >&2
  echo "For local use, run the image directly or export it with:" >&2
  echo "  ./scripts/save-image.sh ${TAG}" >&2
  echo "To push to GHCR, set GITHUB_TOKEN to a PAT with write:packages scope." >&2
  exit 1
fi

echo "== Pushing ${LOCAL} -> ${REMOTE} =="
echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${GH_USER}" --password-stdin
docker tag "${LOCAL}" "${REMOTE}"
docker push "${REMOTE}"
docker tag "${LOCAL}" "${REMOTE_LATEST}"
docker push "${REMOTE_LATEST}"
docker logout ghcr.io

echo
echo "== Done =="
echo "  ${REMOTE}"
echo "  ${REMOTE_LATEST}"
