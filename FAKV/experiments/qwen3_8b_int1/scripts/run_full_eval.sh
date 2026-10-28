#!/usr/bin/env bash
set -euo pipefail

mode=${1:-baseline}
case "${mode}" in
  baseline|fisher) ;;
  *) echo "usage: $0 [baseline|fisher]" >&2; exit 2 ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${repo}/experiments/qwen3_8b_int1
suffix=${OSCAR_RUN_SUFFIX:-}
thinking_mode=${OSCAR_THINKING_MODE:-nonthinking}
served_model_name=${OSCAR_SERVED_MODEL_NAME:-qwen3-8b-oscar-int1-nonthinking}
case "${thinking_mode}" in
  thinking)
    export SGLANG_ENABLE_THINKING=1
    export SGLANG_DISABLE_THINKING=0
    ;;
  nonthinking)
    export SGLANG_ENABLE_THINKING=0
    export SGLANG_DISABLE_THINKING=1
    ;;
  *) echo "OSCAR_THINKING_MODE must be thinking or nonthinking" >&2; exit 2 ;;
esac
run_name=${mode}${suffix:+_${suffix}}
output=${work}/results/official_v5_${run_name}
log=${work}/logs/eval_${run_name}.log
mkdir -p "${work}/logs"

cd "${repo}"
export OFFICIAL_V5_ROOT="${repo}/vllm_turbo_baseline_acc"
export OFFICIAL_V5_REQUEST_TIMEOUT_MIN=600
export SGLANG_TOKENIZER_PATH=/root/oscar-assets/Qwen3-8B
nohup /root/oscar-env/bin/python \
  "${repo}/experiments/qwen3_4b_instruct_int1/scripts/run_accuracy_suite_sglang.py" \
  --suite-dir vllm_turbo_baseline_acc/accuracy_suites/model_agnostic_accuracy_official_v5 \
  --output-dir "${output}" \
  --base-url http://127.0.0.1:8000/v1 \
  --model "${served_model_name}" \
  --no-reasoning-effort \
  --concurrency 32 \
  --benchmarks GSM8K IFEval "LiveCodeBench v6" MultiPL-E \
  --code-eval-isolated \
  --resume \
  >"${log}" 2>&1 &
echo $! >"${work}/eval_${run_name}.pid"
echo "eval_pid=$(cat "${work}/eval_${run_name}.pid") mode=${mode} run=${run_name} log=${log} output=${output}"
