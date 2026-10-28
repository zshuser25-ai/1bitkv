#!/usr/bin/env bash
set -euo pipefail

mode=${1:-baseline}
case "${mode}" in
  baseline|fisher) ;;
  *) echo "usage: $0 [baseline|fisher]" >&2; exit 2 ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${repo}/experiments/qwen3_8b_int1
venv=${OSCAR_VENV:-/root/oscar-env}
model_dir=${OSCAR_MODEL_DIR:-/root/oscar-assets/Qwen3-8B}
rotation_base=${OSCAR_ROTATION_BASE:-${repo}/rotation/qwen3-8B/rotation_zoo_download/Qwen3-8B/seq20000_prompt83_group128}
fisher_path=${OSCAR_FISHER_PATH:-${repo}/rotation/qwen3-8B/GPQA/seq20000_prompt83_group128/anchor_fisher_diag_v1_4090.pt}
served_model_name=${OSCAR_SERVED_MODEL_NAME:-qwen3-8b-oscar-int1-nonthinking}

test -f "${model_dir}/config.json"
test -f "${rotation_base}/k_rotation_qqt_r_h_pbr.pt"
test -f "${rotation_base}/v_rotation_sst_r_h_pbr.pt"
if [[ "${mode}" == fisher ]]; then
  test -f "${fisher_path}"
fi
mkdir -p "${work}/logs"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${repo}/rotation/_triton_per_rank:${repo}/sglang-research/python"
export OSCAR_TRITON_PER_RANK_BASE=${OSCAR_TRITON_PER_RANK_BASE:-/tmp/oscar_qwen3_8b_int1_triton}
export SGLANG_ENABLE_MIXED_KV_WINDOWS=1
export SGLANG_MIXED_KV_PREFIX_TOKENS=64
export SGLANG_MIXED_KV_RECENT_TOKENS=256
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
export SGLANG_LLOYD_MAX=0

if [[ "${mode}" == fisher ]]; then
  export SGLANG_OSCAR_FP16_ANCHOR_TOKENS=28
  export SGLANG_OSCAR_ANCHOR_FISHER_PATH="${fisher_path}"
else
  export SGLANG_OSCAR_FP16_ANCHOR_TOKENS=0
  export SGLANG_OSCAR_ANCHOR_FISHER_PATH=
fi

extra_server_args=()
if [[ "${OSCAR_DISABLE_OVERLAP:-0}" == 1 ]]; then
  extra_server_args+=(--disable-overlap-schedule)
fi

cd "${repo}"
nohup "${venv}/bin/python" -m sglang.launch_server \
  --model-path "${model_dir}" \
  --served-model-name "${served_model_name}" \
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
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  "${extra_server_args[@]}" \
  >"${work}/logs/server_${mode}.log" 2>&1 &
echo $! >"${work}/server_${mode}.pid"
echo "server_pid=$(cat "${work}/server_${mode}.pid") mode=${mode} log=${work}/logs/server_${mode}.log"
