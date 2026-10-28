#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
model_dir=${OSCAR_MODEL_DIR:-/root/oscar-assets/Qwen3-8B}
rotation_base=${OSCAR_ROTATION_BASE:-${repo}/rotation/qwen3-8B/rotation_zoo_download/Qwen3-8B/seq20000_prompt83_group128}
output_base=${repo}/rotation/qwen3-8B/GPQA/seq20000_prompt83_group128
thinking_mode=${OSCAR_THINKING_MODE:-nonthinking}
case "${thinking_mode}" in
  thinking)
    default_output=${output_base}/anchor_fisher_diag_v1_thinking_4090.pt
    thinking_args=(--enable-thinking)
    ;;
  nonthinking)
    default_output=${output_base}/anchor_fisher_diag_v1_4090.pt
    thinking_args=(--disable-thinking)
    ;;
  *) echo "OSCAR_THINKING_MODE must be thinking or nonthinking" >&2; exit 2 ;;
esac
output_path=${OSCAR_FISHER_OUTPUT:-${default_output}}
mkdir -p "${output_base}" "${repo}/experiments/qwen3_8b_int1/logs"

cd "${repo}"
exec /root/oscar-env/bin/python \
  experiments/qwen3_4b_instruct_int1/scripts/calibrate_anchor_fisher.py \
  --model "${model_dir}" \
  --model-name Qwen3-8B \
  --source gpqa_diamond \
  --k-rotation "${rotation_base}/k_rotation_qqt_r_h_pbr.pt" \
  --v-rotation "${rotation_base}/v_rotation_sst_r_h_pbr.pt" \
  --output "${output_path}" \
  --max-used-samples 16 \
  --max-length 512 \
  --tail-loss-tokens 256 \
  "${thinking_args[@]}" \
  --prefix-tokens 64 \
  --recent-tokens 256 \
  --shrinkage 0.05 \
  --winsor-quantile 0.995 \
  --seed 20260921 \
  --device-map balanced
