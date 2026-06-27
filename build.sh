#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-fraserpricee/vllm:dspark-cu132-20260627}"
docker build --tag "$TAG" --file Dockerfile .
echo "built $TAG"
