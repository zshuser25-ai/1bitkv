#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
venv=${OSCAR_VENV:-/root/oscar-env}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${repo}/rotation/_triton_per_rank:${repo}/sglang-research/python"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/oscar_int1_triton}
cd "${repo}"
exec "${venv}/bin/python" -m pytest -p no:cacheprovider -q -s \
  experiments/qwen3_4b_instruct_int1/tests/test_pure_int1_gpu.py
