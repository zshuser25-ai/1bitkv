#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${repo}/experiments/qwen3_4b_instruct_int1
if [[ -n "${OSCAR_PYTHON:-}" ]]; then
  python_bin=${OSCAR_PYTHON}
elif [[ -n "${OSCAR_VENV:-}" ]]; then
  python_bin=${OSCAR_VENV}/bin/python
else
  python_bin=$(command -v python3)
fi
model_dir=${OSCAR_MODEL_DIR:-${repo}/models/Qwen3-4B-Instruct-2507}
asset_root=${OSCAR_ASSET_ROOT:-${repo}/assets/qwen3_4b_instruct_2507}
rotation_base=${OSCAR_ROTATION_BASE:-${asset_root}}
fisher_path=${OSCAR_FISHER_PATH:-${asset_root}/anchor_fisher_diag_gpqa198_v2.pt}
kernel_base=${OSCAR_TRITON_KERNEL_BASE:-${repo}/rotation/_triton_per_rank}

for required_file in \
  "${model_dir}/config.json" \
  "${rotation_base}/k_rotation_qqt_r_h_pbr.pt" \
  "${rotation_base}/v_rotation_sst_r_h_pbr.pt" \
  "${fisher_path}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    echo "Set OSCAR_MODEL_DIR, OSCAR_ROTATION_BASE, or OSCAR_FISHER_PATH as needed." >&2
    exit 1
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 1
fi
mkdir -p "${work}/logs"
server_log=${OSCAR_SERVER_LOG:-${work}/logs/server.log}
server_pid_file=${OSCAR_SERVER_PID_FILE:-${work}/server.pid}
server_port=${OSCAR_SERVER_PORT:-8000}
served_model_name=${OSCAR_SERVED_MODEL_NAME:-qwen3-4b-instruct-2507-erakv-v2}
max_running_requests=${OSCAR_MAX_RUNNING_REQUESTS:-32}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
export PYTHONPATH="${kernel_base}:${repo}/sglang-research/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE=${SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE:-2048}
export OSCAR_TRITON_PER_RANK_BASE=${OSCAR_TRITON_PER_RANK_BASE:-/tmp/oscar_int1_triton}
export SGLANG_ENABLE_MIXED_KV_WINDOWS=1
export SGLANG_MIXED_KV_PREFIX_TOKENS=${SGLANG_MIXED_KV_PREFIX_TOKENS:-16}
export SGLANG_MIXED_KV_RECENT_TOKENS=${SGLANG_MIXED_KV_RECENT_TOKENS:-64}
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
export SGLANG_OSCAR_PURE_INT1_COMPACT_SCALES="${SGLANG_OSCAR_PURE_INT1_COMPACT_SCALES:-1}"
export SGLANG_OSCAR_FP16_ANCHOR_TOKENS=${SGLANG_OSCAR_FP16_ANCHOR_TOKENS:-240}
export SGLANG_OSCAR_ANCHOR_FISHER_PATH="${fisher_path}"
export SGLANG_LLOYD_MAX=0

cd "${repo}"
nohup "${python_bin}" -m sglang.launch_server \
  --model-path "${model_dir}" \
  --served-model-name "${served_model_name}" \
  --tensor-parallel-size 4 \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --kv-cache-dtype int2 \
  --kv-cache-quant-group-size 128 \
  --context-length 32768 \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.80 \
  --max-running-requests "${max_running_requests}" \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --disable-custom-all-reduce \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --enable-deterministic-inference \
  --host 127.0.0.1 \
  --port "${server_port}" \
  --trust-remote-code \
  >"${server_log}" 2>&1 &
echo $! >"${server_pid_file}"
echo "server_pid=$(cat "${server_pid_file}") log=${server_log}"
