#!/usr/bin/env bash
set -euo pipefail

venv=${OSCAR_VENV:-/root/oscar-env}
asset_root=${OSCAR_ASSET_ROOT:-/root/oscar-assets}
model_dir=${OSCAR_MODEL_DIR:-${asset_root}/Qwen3-4B-Instruct-2507}

mkdir -p "${model_dir}"
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}

"${venv}/bin/hf" download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir "${model_dir}"

echo "RotationZoo does not contain an Instruct-2507 rotation; run calibrate_rotation.sh next."
