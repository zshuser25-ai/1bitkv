#!/usr/bin/env bash
set -euo pipefail

venv=${OSCAR_VENV:-/root/oscar-env}
asset_root=${OSCAR_ASSET_ROOT:-/root/oscar-assets}
model_dir=${OSCAR_MODEL_DIR:-${asset_root}/Qwen3-4B-Thinking-2507}
rotation_dir=${OSCAR_ROTATION_DIR:-${asset_root}/rotation}

mkdir -p "${model_dir}" "${rotation_dir}"
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}

"${venv}/bin/hf" download Qwen/Qwen3-4B-Thinking-2507 \
  --revision 768f209d9ea81521153ed38c47d515654e938aea \
  --local-dir "${model_dir}" &
model_pid=$!

"${venv}/bin/hf" download Zhongzhu/OSCAR-RotationZoo \
  --include 'Qwen3-4B-Thinking-2507/seq20000_prompt85_group128/*' \
  --local-dir "${rotation_dir}" &
rotation_pid=$!

cleanup() {
  kill "${model_pid}" "${rotation_pid}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
wait "${model_pid}"
wait "${rotation_pid}"
trap - INT TERM EXIT
