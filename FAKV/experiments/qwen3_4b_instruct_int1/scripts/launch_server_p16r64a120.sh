#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=$(cd -- "${script_dir}/.." && pwd)

export SGLANG_MIXED_KV_PREFIX_TOKENS=16
export SGLANG_MIXED_KV_RECENT_TOKENS=64
export SGLANG_OSCAR_FP16_ANCHOR_TOKENS=120
export OSCAR_SERVER_LOG=${OSCAR_SERVER_LOG:-${work}/logs/server_p16r64a120.log}
export OSCAR_SERVER_PID_FILE=${OSCAR_SERVER_PID_FILE:-${work}/server_p16r64a120.pid}
export OSCAR_SERVED_MODEL_NAME=${OSCAR_SERVED_MODEL_NAME:-qwen3-4b-instruct-2507-erakv-v2-p16r64a120}

exec bash "${script_dir}/launch_server.sh"
