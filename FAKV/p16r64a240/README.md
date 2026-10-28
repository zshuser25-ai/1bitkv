# FA p16r64a240

This isolated directory branch retains the original FA source and assets. Its Qwen launcher defaults to prefix16/recent64/anchor240. V2 uses all198 GPQA Diamond prompts to learn diagonal Fisher weights in the fixed rotated coordinate system. Calibration prefix/recent are16/64; max_length512, tail_loss256, seed20260921, shrinkage0.05, Winsor0.995. GPQA evaluation overlaps calibration and must be labeled accordingly.

The rollover selector pads candidate tiles to next_power_of_2(anchor_capacity+flush_interval), so240 anchors plus8 flushed rows require256 entries. Decode correction already loops over32-anchor tiles.

Deterministic inference is enabled in the Qwen launchers. Set SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE=2048 to match chunked_prefill_size2048; the upstream4096 alignment would prevent long-request admission. The FA pre-rotation/cache-write path remains active under deterministic inference. Small repeated probes measure reproducibility; the option alone does not prove full-run bitwise equality.

Original upstream documentation below describes the parent branch defaults.

# ERAKV_V2: OSCAR INT1 + FP16 Anchors in Chunked Prefill

## GitLab 克隆后部署 Qwen3-4B-Instruct-2507

仓库必须包含完整的 `sglang-research/python` 修改版源码、`rotation/_triton_per_rank`、
`experiments/qwen3_4b_instruct_int1/scripts/launch_server.sh` 和
`assets/qwen3_4b_instruct_2507`。单独上传 `requirements.txt` 无法安装本地修改版 SGLang。
`assets` 内三个文件是本次四卡实验使用的 K/V 旋转矩阵与 GPQA 16 条校准得到的
Fisher 张量。Qwen 模型权重没有包含在仓库中，需要单独下载或提供。

已验证环境为 Ubuntu 22.04 x86_64、Python 3.11、四张 RTX 4090、支持 CUDA 12.8
PyTorch wheel 的 NVIDIA 驱动（实测 575.57.08），并需有 C++ 编译工具链与可用的
CUDA 开发环境。安装 Python 包时需要能访问 PyPI 或等价的镜像源。其他 GPU 或
Python 版本需要另行验证。

```bash
git clone <your-gitlab-url> FAKV_V2
cd FAKV_V2
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export OSCAR_MODEL_DIR=/path/to/Qwen3-4B-Instruct-2507
bash experiments/qwen3_4b_instruct_int1/scripts/launch_server.sh
curl -fsS http://127.0.0.1:8000/health
```

`OSCAR_MODEL_DIR` 必须指向有 `config.json`、tokenizer 和全部权重文件的模型目录。
启动脚本会优先使用 `OSCAR_PYTHON`，其次使用 `OSCAR_VENV/bin/python`，否则使用
当前 PATH 中的 `python3`；在激活的 `.venv` 中运行即可。若需覆盖默认资产路径，
设置 `OSCAR_ROTATION_BASE`（包含两个旋转矩阵文件）和 `OSCAR_FISHER_PATH`。
服务默认监听 `127.0.0.1:8000`，日志写入
`experiments/qwen3_4b_instruct_int1/logs/server.log`。

评测 official v8 还需单独提供 `/nfs/AE/zsh/dataset` 对应的数据集、测试协议和
评测脚本；这些内容不属于本仓库的模型服务安装。不要把模型权重、数据集、
运行日志或 Python 虚拟环境放进普通 Git 提交。

This directory is an independent implementation tree for Route B.  It is
based on the current OSCAR/SGLang code but adds FP16-anchor replacement to
chunked-prefill attention.

## Supported cache layouts

Current launcher defaults (2026-10-08):

- first 16 tokens: BF16;
- middle region: INT1;
- most recent 248 tokens: BF16;
- middle-region anchors: **56 FP16 entries per layer and KV head**.

`SGLANG_OSCAR_FP16_ANCHOR_TOKENS` supports `0` (disabled), `28` with
prefix/recent `64/256`, or `56` with prefix/recent `16/248`. Other layouts
fail at startup. Launchers accept explicit environment overrides for all
three values; use `64/256/28` to reproduce the previous configuration.

## Route B semantics

For chunk 0 there is no historical prefix, so no prefill correction is
needed.  Before each later chunk writes its K/V, the code snapshots that
request's existing anchor directory.  It then dequantizes the historical
prefix, replaces each selected INT1 row/head with the exact FP16 anchor, and
runs FlashAttention over:

```text
corrected historical prefix + exact current chunk
```

Taking the snapshot before the cache write is mandatory: candidates admitted
from the current chunk must not be corrected again because their exact K/V are
already part of the current FlashAttention input.

Decode keeps the existing exact-minus-INT1 correction and is supports both 28 and 56 anchors.

## Main files

- `sglang-research/python/sglang/srt/layers/attention/quantized_kv_prefill.py`
  implements historical-prefix replacement.
- `sglang-research/python/sglang/srt/layers/attention/triton_backend.py`
  snapshots the pre-write directory and invokes replacement.
- `sglang-research/python/sglang/srt/mem_cache/unified_kv_pool.py`
  owns and updates the configured-capacity directory.
- `experiments/qwen3_4b_instruct_int1/tests/smoke_fp16_anchors.py`
  checks selection, decode correction, rollover, and Route B prefill.

The launcher loads the modified SGLang source and the per-rank Triton cache
helper from this repository.  The three small experiment assets are in
`assets/qwen3_4b_instruct_2507`; model weights are supplied separately.

The launcher fixes chunked prefill at 2048 and defaults to the 16/248/56 layout;
it does not accept a 320-anchor override.
