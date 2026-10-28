#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${repo}/experiments/qwen3_4b_instruct_int1
venv=/root/oscar-env
model_dir=/root/oscar-assets/Qwen3-4B-Instruct-2507
calibration_base=${repo}/rotation/qwen3-4B-instruct-2507/GPQA/seq20000_prompt88_group128
rotation_base=${calibration_base}/rotations

test -f "${model_dir}/config.json"
test -f "${rotation_base}/k_rotation_qqt_r_h_pbr.pt"
test -f "${rotation_base}/v_rotation_sst_r_h_pbr.pt"
test -f "${calibration_base}/anchor_fisher_diag_v1.pt"
mkdir -p "${work}/logs"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${repo}/rotation/_triton_per_rank:${repo}/sglang-research/python"
export OSCAR_TRITON_PER_RANK_BASE=/tmp/oscar_int1_triton_a800
export SGLANG_ENABLE_MIXED_KV_WINDOWS=1
export SGLANG_MIXED_KV_PREFIX_TOKENS=${SGLANG_MIXED_KV_PREFIX_TOKENS:-16}
export SGLANG_MIXED_KV_RECENT_TOKENS=${SGLANG_MIXED_KV_RECENT_TOKENS:-248}
export SGLANG_MIXED_KV_HP_DTYPE=bfloat16
export SGLANG_MIXED_KV_SCALE_DTYPE=float32
export SGLANG_MIXED_KV_HP_MAX_SPLITS=8
export SGLANG_OSCAR_ABSORB_V_ROTATION=1
export SGLANG_OSCAR_K_ROTATION_PATH="${rotation_base}/k_rotation_qqt_r_h_pbr.pt"
export SGLANG_OSCAR_V_ROTATION_PATH="${rotation_base}/v_rotation_sst_r_h_pbr.pt"
export SGLANG_OSCAR_K_CLIP_RATIO=0.96
export SGLANG_OSCAR_V_CLIP_RATIO=0.92
export SGLANG_OSCAR_AA_K_MODE=deployed
export SGLANG_OSCAR_AA_K_HIGH_DIM=0
export SGLANG_OSCAR_V_MIXED_MODE=mixed_int1
export SGLANG_OSCAR_V_HIGH_DIM=0
export SGLANG_OSCAR_FP16_ANCHOR_TOKENS=${SGLANG_OSCAR_FP16_ANCHOR_TOKENS:-56}
export SGLANG_OSCAR_ANCHOR_FISHER_PATH="${calibration_base}/anchor_fisher_diag_v1.pt"
export SGLANG_LLOYD_MAX=0

cd "${repo}"
nohup "${venv}/bin/python" -m sglang.launch_server \
  --model-path "${model_dir}" \
  --served-model-name qwen3-4b-instruct-2507-oscar-int1 \
  --tensor-parallel-size 4 \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --kv-cache-dtype int2 \
  --kv-cache-quant-group-size 128 \
  --context-length 32768 \
  --mem-fraction-static 0.80 \
  --max-running-requests 32 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --disable-custom-all-reduce \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  >"${work}/logs/server_fisher_a800.log" 2>&1 &
echo $! >"${work}/server_fisher_a800.pid"
echo "server_pid=$(cat "${work}/server_fisher_a800.pid") log=${work}/logs/server_fisher_a800.log"
