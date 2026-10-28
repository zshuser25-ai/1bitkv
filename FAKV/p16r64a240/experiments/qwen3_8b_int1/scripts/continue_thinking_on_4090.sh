#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
work=${repo}/experiments/qwen3_8b_int1
baseline_result=${work}/results/official_v5_baseline_thinking/summary.json
fisher_result=${work}/results/official_v5_fisher_thinking/summary.json
fisher_path=${repo}/rotation/qwen3-8B/GPQA/seq20000_prompt83_group128/anchor_fisher_diag_v1_thinking_4090.pt
served_model=qwen3-8b-oscar-int1-thinking
log=${work}/logs/continue_thinking_on_4090.log

exec >>"${log}" 2>&1
echo "[$(date '+%F %T')] thinking continuation started"

wait_pid_file() {
  local pid_file=$1 label=$2 pid
  pid=$(cat "${pid_file}")
  echo "[$(date '+%F %T')] waiting for ${label} pid=${pid}"
  while kill -0 "${pid}" 2>/dev/null; do sleep 30; done
  echo "[$(date '+%F %T')] ${label} exited"
}

validate_summary() {
  /root/oscar-env/bin/python - "$1" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["valid"] is True, summary
assert summary["total"] == 2360, summary
assert summary["scored"] == 2360, summary
assert summary["status_counts"] == {"scored": 2360}, summary
print("validated", sys.argv[1])
PY
}

stop_server() {
  local pid pid_file=$1
  pid=$(cat "${pid_file}")
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}"
    for _ in $(seq 1 60); do
      kill -0 "${pid}" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL "${pid}"
  fi
}

wait_health() {
  /root/oscar-env/bin/python - "$1" <<'PY'
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

thinking_long_smoke() {
  /root/oscar-env/bin/python - "${served_model}" <<'PY'
import json, sys, time, urllib.request
prompt = ("The quick brown fox jumps over the lazy dog. " * 2600) + "\nReply with only OK."
body = json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 8,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": True},
}).encode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json"},
)
started = time.time()
with urllib.request.urlopen(request, timeout=600) as response:
    result = json.load(response)
usage = result.get("usage") or {}
message = result["choices"][0]["message"]
content = message.get("content") or ""
reasoning = message.get("reasoning_content") or ""
assert usage.get("prompt_tokens", 0) > 20000, usage
assert "<think>" in content or reasoning, message
print("thinking_long_smoke=PASS", "seconds=", round(time.time() - started, 2), "usage=", usage)
PY
}

wait_pid_file "${work}/eval_baseline_thinking.pid" baseline_thinking_eval
validate_summary "${baseline_result}"
stop_server "${work}/server_baseline.pid"

echo "[$(date '+%F %T')] starting Fisher thinking server"
OSCAR_FISHER_PATH="${fisher_path}" \
OSCAR_SERVED_MODEL_NAME="${served_model}" \
  bash "${work}/scripts/launch_server.sh" fisher
fisher_server_pid=$(cat "${work}/server_fisher.pid")
wait_health "${fisher_server_pid}"
thinking_long_smoke

echo "[$(date '+%F %T')] starting Fisher thinking full evaluation"
OSCAR_FISHER_PATH="${fisher_path}" \
OSCAR_RUN_SUFFIX=thinking \
OSCAR_THINKING_MODE=thinking \
OSCAR_SERVED_MODEL_NAME="${served_model}" \
  bash "${work}/scripts/run_full_eval.sh" fisher
wait_pid_file "${work}/eval_fisher_thinking.pid" fisher_thinking_eval
validate_summary "${fisher_result}"
echo "[$(date '+%F %T')] ALL_THINKING_4090_RUNS_COMPLETE"
