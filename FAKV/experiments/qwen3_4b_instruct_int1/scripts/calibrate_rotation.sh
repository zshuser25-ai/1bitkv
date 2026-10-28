#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
venv=${OSCAR_VENV:-/root/oscar-env}

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="${CUDA_HOME}/bin:${PATH}"
export PY=${PY:-${venv}/bin/python}
export PY_EVAL=${PY_EVAL:-${venv}/bin/python}
export CONDA_ENV_DIR=${CONDA_ENV_DIR:-${venv}}
export SGLANG_DUMP_DIR=${SGLANG_DUMP_DIR:-${repo}/sglang-research}
export MODEL=${OSCAR_MODEL_DIR:-/root/oscar-assets/Qwen3-4B-Instruct-2507}
export DUMP_KVCACHE_TOKENS=${DUMP_KVCACHE_TOKENS:-20000}

# Bootstrap through the mixed-KV path already validated on this POD. The dump
# hook captures Q/K/V before these temporary rotations are applied.
bootstrap_rotation=${OSCAR_BOOTSTRAP_ROTATION_BASE:-/root/oscar-assets/rotation/Qwen3-4B-Thinking-2507/seq20000_prompt85_group128}
export SGLANG_ENABLE_MIXED_KV_WINDOWS=1
export SGLANG_MIXED_KV_PREFIX_TOKENS=64
export SGLANG_MIXED_KV_RECENT_TOKENS=256
export SGLANG_MIXED_KV_HP_DTYPE=bfloat16
export SGLANG_MIXED_KV_SCALE_DTYPE=float32
export SGLANG_MIXED_KV_HP_MAX_SPLITS=8
export SGLANG_OSCAR_ABSORB_V_ROTATION=1
export SGLANG_OSCAR_K_ROTATION_PATH="${bootstrap_rotation}/k_rotation_qqt_r_h_pbr.pt"
export SGLANG_OSCAR_V_ROTATION_PATH="${bootstrap_rotation}/v_rotation_sst_r_h_pbr.pt"
export SGLANG_OSCAR_K_CLIP_RATIO=0.96
export SGLANG_OSCAR_V_CLIP_RATIO=0.92
export SGLANG_OSCAR_AA_K_MODE=deployed
export SGLANG_OSCAR_AA_K_HIGH_DIM=0
export SGLANG_OSCAR_V_MIXED_MODE=mixed_int1
export SGLANG_OSCAR_V_HIGH_DIM=0
export SGLANG_LLOYD_MAX=0

cd "${repo}"
bash rotation/qwen3-4B-instruct-2507/save_qkv_4b_instruct.sh
bash rotation/qwen3-4B-instruct-2507/compute_rotation.sh
