#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${repo}/experiments/qwen3_4b_instruct_int1
log=${work}/logs/eval_anchor28_fisher_v1_a800.log
output=${work}/results/official_v5_anchor28_fisher_v1_a800

cd "${repo}"
export OFFICIAL_V5_REQUEST_TIMEOUT_MIN=600
export OFFICIAL_V5_ROOT="${repo}/vllm_turbo_baseline_acc"
export SGLANG_TOKENIZER_PATH=/root/oscar-assets/Qwen3-4B-Instruct-2507
nohup /root/oscar-env/bin/python \
  "${work}/scripts/run_accuracy_suite_sglang.py" \
  --suite-dir vllm_turbo_baseline_acc/accuracy_suites/model_agnostic_accuracy_official_v5 \
  --output-dir "${output}" \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3-4b-instruct-2507-oscar-int1 \
  --no-reasoning-effort \
  --concurrency 32 \
  --benchmarks GSM8K IFEval "LiveCodeBench v6" MultiPL-E \
  --code-eval-isolated \
  --resume \
  >"${log}" 2>&1 &
echo $! >"${work}/eval_fisher_v1_a800.pid"
echo "eval_pid=$(cat "${work}/eval_fisher_v1_a800.pid") log=${log} output=${output}"
