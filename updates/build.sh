#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${INSITUANN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BUILD_ROOT="${INSITUANN_BUILD:-${ROOT}/build}"
BUILD_DIR="${SCRIPT_DIR}/build"
JOBS="${JOBS:-12}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DINSITUANN_ROOT="${ROOT}" \
  -DINSITUANN_BUILD="${BUILD_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-120}"

cmake --build "${BUILD_DIR}" -j"${JOBS}"
