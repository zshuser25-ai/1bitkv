#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${repo}/experiments/qwen3_4b_instruct_int1
venv=${OSCAR_VENV:-/root/oscar-env}
model_dir=${OSCAR_MODEL_DIR:-/root/oscar-assets/Qwen3-4B-Instruct-2507}
mkdir -p "${work}/results/official_v5" "${work}/logs"

export OFFICIAL_V5_ROOT="${repo}/vllm_turbo_baseline_acc"
export SGLANG_TOKENIZER_PATH="${model_dir}"
cd "${repo}"
exec "${venv}/bin/python" \
  experiments/qwen3_4b_instruct_int1/scripts/run_accuracy_suite_sglang.py \
  --suite-dir vllm_turbo_baseline_acc/accuracy_suites/model_agnostic_accuracy_official_v5 \
  --output-dir experiments/qwen3_4b_instruct_int1/results/official_v5 \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3-4b-instruct-2507-oscar-int1 \
  --no-reasoning-effort \
  --concurrency 32 \
  --benchmarks GSM8K IFEval 'LiveCodeBench v6' MultiPL-E \
  --code-eval-isolated \
  --resume 2>&1 | tee "${work}/logs/eval.log"
