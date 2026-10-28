#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export OSCAR_PYTHON=${OSCAR_PYTHON:-/usr/bin/python}
export OSCAR_MODEL_DIR=${OSCAR_MODEL_DIR:-/nfs/AE/zsh/models/Qwen3-4B-Instruct-2507}
export OSCAR_ASSET_ROOT=${OSCAR_ASSET_ROOT:-${repo}/assets/qwen3_4b_instruct_2507}
export OSCAR_FISHER_PATH=${OSCAR_FISHER_PATH:-${OSCAR_ASSET_ROOT}/anchor_fisher_diag_gpqa198_v2.pt}
export OSCAR_SERVED_MODEL_NAME=${OSCAR_SERVED_MODEL_NAME:-qwen3-4b-instruct-2507-fa-p16r64a240}
export SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE=${SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE:-2048}
export OSCAR_TRITON_PER_RANK_BASE=${OSCAR_TRITON_PER_RANK_BASE:-/tmp/fa_qwen240_a800_triton}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
exec bash "${repo}/experiments/qwen3_4b_instruct_int1/scripts/launch_server.sh"
