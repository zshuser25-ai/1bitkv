#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${repo}/experiments/qwen3_8b_int1
baseline_result=${work}/results/official_v5_baseline_nonthinking/summary.json
fisher_result=${work}/results/official_v5_fisher_nonthinking/summary.json
log=${work}/logs/continue_on_4090.log

exec >>"${log}" 2>&1
echo "[$(date '+%F %T')] 4090 continuation started"

wait_pid_file() {
  local pid_file=$1
  local label=$2
  local pid
  pid=$(cat "${pid_file}")
  echo "[$(date '+%F %T')] waiting for ${label} pid=${pid}"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date '+%F %T')] ${label} exited"
}

validate_summary() {
  local path=$1
  /root/oscar-env/bin/python - "${path}" <<'PY'
import json, sys
path = sys.argv[1]
summary = json.load(open(path, encoding="utf-8"))
assert summary["valid"] is True, summary
assert summary["total"] == 2360, summary
assert summary["scored"] == 2360, summary
assert summary["status_counts"] == {"scored": 2360}, summary
print("validated", path)
PY
}

stop_server() {
  local pid_file=$1
  local pid
  pid=$(cat "${pid_file}")
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[$(date '+%F %T')] stopping server pid=${pid}"
    kill -TERM "${pid}"
    for _ in $(seq 1 60); do
      kill -0 "${pid}" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL "${pid}"
  fi
}

wait_health() {
  local server_pid=$1
  /root/oscar-env/bin/python - "${server_pid}" <<'PY'
import os, sys, time, urllib.request
pid = int(sys.argv[1])
for _ in range(900):
    try:
        if urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3).status == 200:
            print("health=200")
            raise SystemExit(0)
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise SystemExit(f"server pid {pid} exited before health: {exc}")
    time.sleep(1)
raise SystemExit("server health timeout")
PY
}

long_smoke() {
  /root/oscar-env/bin/python - <<'PY'
import json, time, urllib.request
prompt = ("The quick brown fox jumps over the lazy dog. " * 2600) + "\nReply with only OK."
body = json.dumps({
    "model": "qwen3-8b-oscar-int1-nonthinking",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 8,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
started = time.time()
with urllib.request.urlopen(request, timeout=600) as response:
    result = json.load(response)
usage = result.get("usage") or {}
assert usage.get("prompt_tokens", 0) > 20000, usage
assert result.get("choices"), result
assert "<think>" not in (result["choices"][0]["message"].get("content") or "")
print("long_smoke=PASS", "seconds=", round(time.time() - started, 2), "usage=", usage)
PY
}

wait_pid_file "${work}/eval_baseline_nonthinking.pid" baseline_eval
validate_summary "${baseline_result}"
stop_server "${work}/server_baseline.pid"

echo "[$(date '+%F %T')] starting Qwen3-8B Fisher calibration on 4090"
OSCAR_FISHER_OUTPUT=${repo}/rotation/qwen3-8B/GPQA/seq20000_prompt83_group128/anchor_fisher_diag_v1_4090.pt \
  bash "${work}/scripts/calibrate_fisher.sh" \
  >"${work}/logs/calibrate_fisher_4090.log" 2>&1
test -s "${repo}/rotation/qwen3-8B/GPQA/seq20000_prompt83_group128/anchor_fisher_diag_v1_4090.pt"
sha256sum "${repo}/rotation/qwen3-8B/GPQA/seq20000_prompt83_group128/anchor_fisher_diag_v1_4090.pt"

echo "[$(date '+%F %T')] starting Fisher server on 4090"
bash "${work}/scripts/launch_server.sh" fisher
fisher_server_pid=$(cat "${work}/server_fisher.pid")
wait_health "${fisher_server_pid}"
long_smoke

echo "[$(date '+%F %T')] starting Fisher full evaluation on 4090"
OSCAR_RUN_SUFFIX=nonthinking bash "${work}/scripts/run_full_eval.sh" fisher
wait_pid_file "${work}/eval_fisher_nonthinking.pid" fisher_eval
validate_summary "${fisher_result}"
echo "[$(date '+%F %T')] ALL_4090_RUNS_COMPLETE"
