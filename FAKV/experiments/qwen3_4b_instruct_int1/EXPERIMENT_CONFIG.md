# OSCAR INT1 实验配置

## 1. 实验目标

在 `Qwen/Qwen3-4B-Instruct-2507` 上测试 OSCAR 混合精度 KV Cache。只运行一个实验组：序列前 64 个 token 和最近 256 个 token 使用 BF16，中间 token 的 K/V Cache 全部压缩为 INT1。

本实验不运行 BF16 基线、OSCAR INT2 或其他 INT1/INT2 混合子维度配置。

## 2. 模型

| 项目 | 配置 |
|---|---|
| Hugging Face 模型 | `Qwen/Qwen3-4B-Instruct-2507` |
| 固定 revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| 层数 | 36 |
| hidden size | 2560 |
| attention heads | 32 |
| KV heads | 8 |
| head dimension | 128 |
| 权重 dtype | BF16 |
| 服务名称 | `qwen3-4b-instruct-2507-oscar-int1` |
| tensor parallel size | 4 |
| 最大上下文长度 | 32768 |

## 3. OSCAR KV Cache 配置

每个请求的缓存按 token 位置分为三段：

| 区域 | token 范围 | K/V 存储格式 |
|---|---|---|
| 前缀窗口 | 前 64 个 token | BF16 |
| 中间区域 | 前缀窗口与最近窗口之间 | INT1 |
| 最近窗口 | 最近 256 个 token | BF16 |

中间区域的 K 和 V 使用相同的纯 INT1 格式：

- 128 个 head 维度全部量化，不保留 INT2 子维度。
- 每个值存储一个符号位；8 个维度打包到一个字节。
- 每个 128 维 K/V 行占 16 个打包字节。
- 每个 token、每个 KV head 额外保存一个 BF16 缩放值。
- 缩放值为该行旋转并裁剪后数值的平均绝对值：`scale = mean(abs(x))`。
- 反量化值为 `+scale` 或 `-scale`。
- K 与 V 都执行该规则。

对应环境变量：

```text
SGLANG_ENABLE_MIXED_KV_WINDOWS=1
SGLANG_MIXED_KV_PREFIX_TOKENS=64
SGLANG_MIXED_KV_RECENT_TOKENS=256
SGLANG_MIXED_KV_HP_DTYPE=bfloat16
SGLANG_MIXED_KV_SCALE_DTYPE=float32
SGLANG_OSCAR_AA_K_MODE=deployed
SGLANG_OSCAR_AA_K_HIGH_DIM=0
SGLANG_OSCAR_V_MIXED_MODE=mixed_int1
SGLANG_OSCAR_V_HIGH_DIM=0
SGLANG_OSCAR_PURE_INT1_COMPACT_SCALES=1
SGLANG_OSCAR_FP16_ANCHOR_TOKENS=28
```

纯 INT1 专用分支由 `SGLANG_OSCAR_PURE_INT1_COMPACT_SCALES=1` 开启（默认开启）：中间区域只保存 1-bit 符号和每行一个 BF16 magnitude，不再分配继承自 INT2、但在纯 INT1 中未被读取的 FP32 affine scale/zero。因此中间区域由旧布局的 1.625 bit/element 降为 1.125 bit/element。`SGLANG_MIXED_KV_SCALE_DTYPE=float32` 仍用于非纯 INT1 路径；设专用开关为 `0` 可恢复旧缓冲布局做兼容性对照。

## 4. 旋转与裁剪

| 项目 | 配置 |
|---|---|
| Rotation 来源 | 本模型 GPQA Q/K/V 离线校准 |
| 校准目录 | `rotation/qwen3-4B-instruct-2507/GPQA/seq20000_prompt88_group128/rotations` |
| K rotation | `k_rotation_qqt_r_h_pbr.pt` |
| V rotation | `v_rotation_sst_r_h_pbr.pt` |
| K clip ratio | 0.96 |
| V clip ratio | 0.92 |
| quant group size | 128 |
| absorb V rotation | 开启 |
| Lloyd-Max | 关闭 |

RotationZoo 没有 Instruct-2507 的预计算文件，因此先用 20000-token GPQA 校准集采集 Q/K/V，并按 OSCAR 的 `qqt_sst`、`r_h_pbr` 方法拟合专用旋转矩阵。

## 5. 数据集和评测范围

数据集目录：

```text
/nfs/AE/zsh/oscar/vllm_turbo_baseline_acc/accuracy_suites/model_agnostic_accuracy_official_v5
```

运行 2360 个生成样本：

| Benchmark | 样本数 | 最大输出 token | 指标 |
|---|---:|---:|---|
| GSM8K | 1319 | 8192 | 数值答案正确率 |
| IFEval | 541 | 4096 | 官方 strict/loose 指令遵循指标 |
| LiveCodeBench v6 | 175 | 8192 | 官方公开和私有测试通过率 |
| MultiPL-E | 325 | 4096 | Python/C++ 程序测试通过率 |

`WikiText-2` 的 1 个 perplexity 任务不运行。它需要本地 perplexity 路径，且不会测试本实验关注的 OpenAI chat 自回归 OSCAR KV Cache 路径。

## 6. 解码参数

| 参数 | 值 |
|---|---:|
| `n` | 1 |
| `temperature` | 0.0 |
| `top_p` | 1.0 |
| `seed` | 42 |
| `reasoning_effort` | 不发送 |
| 请求并发 | 32 |

Instruct-2507 是非思考模型，评测使用 `--no-reasoning-effort`，避免把 Thinking 模型参数带入本实验。

## 7. 服务参数

```text
--tensor-parallel-size 4
--prefill-attention-backend triton
--decode-attention-backend triton
--kv-cache-dtype int2
--kv-cache-quant-group-size 128
--context-length 32768
--mem-fraction-static 0.80
--max-running-requests 32
--disable-cuda-graph
--disable-piecewise-cuda-graph
--disable-custom-all-reduce
--disable-radix-cache
```

`--kv-cache-dtype int2` 是 OSCAR 现有缓存调度入口的名称。启用 `K_HIGH_DIM=0` 和 `V_HIGH_DIM=0` 后，中间区域实际分配和读写的是 16 字节纯 INT1 K/V 行。

## 8. 远端运行环境

| 项目 | 配置 |
|---|---|
| POD | `root@192.168.16.44:19837` |
| GPU | NVIDIA GeForce RTX 4090 24GB |
| 使用 GPU | GPU 0、1、2、3（TP=4） |
| Python 环境 | `/root/oscar-env` |
| Python | 3.12.14 |
| Torch | 2.9.1 |
| Torchvision | 0.24.1 |
| Transformers | 5.3.0 |
| FlashInfer | 0.6.7.post3 |
| Kernels | 0.14.1（与 Transformers 5.3.0 兼容） |
| CUDA toolkit | 12.8.93 (`/usr/local/cuda-12.8`) |
| NVIDIA driver | 575.57.08 |
| OSCAR 代码 | `/nfs/AE/zsh/oscar` |

依赖按 OSCAR 仓库的 `sglang-research/python/pyproject.toml` 安装。

## 9. 正式实验前检查

正式评测开始前依次完成：

1. 下载固定 revision 的模型。
2. 采集 20000-token GPQA Q/K/V，并拟合 Instruct-2507 专用 rotation。
3. 在 GPU 0 运行 GPU 内核测试，比较纯 INT1 flush 结果、参考打包结果、decode 输出和稠密反量化参考输出。
3. 启动服务并检查 `/health`、`/v1/models`。
4. 发送一个长度超过 320 token 的冒烟请求，确保实际触发中间 INT1 区域。
5. 检查服务日志，确认前缀 64、最近 256、K/V high dimension 0 和 packed dimension 16。
6. 运行 2360 个正式样本，启用 `--resume` 保存中断进度。

任一检查失败时，先修复并重新执行该检查，再开始正式评测。

## 10. 执行命令

登录 POD 后：

```bash
cd /nfs/AE/zsh/oscar
bash experiments/qwen3_4b_instruct_int1/scripts/download_assets.sh
bash experiments/qwen3_4b_instruct_int1/scripts/calibrate_rotation.sh
bash experiments/qwen3_4b_instruct_int1/scripts/run_kernel_test.sh
bash experiments/qwen3_4b_instruct_int1/scripts/launch_server.sh
bash experiments/qwen3_4b_instruct_int1/scripts/run_eval.sh
```

## 11. 输出位置

```text
experiments/qwen3_4b_instruct_int1/logs/server.log
experiments/qwen3_4b_instruct_int1/logs/eval.log
experiments/qwen3_4b_instruct_int1/results/official_v5/
experiments/qwen3_4b_instruct_int1/server.pid
```

最终报告记录各 benchmark 的原生指标、成功/失败样本数、实际模型 revision、旋转文件哈希、代码 commit、依赖版本、GPU 型号和完整启动参数。
