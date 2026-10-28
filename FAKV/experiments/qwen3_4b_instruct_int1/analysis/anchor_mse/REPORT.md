# OSCAR INT1 + 28 FP16 anchors：逐层 MSE 分析

## 分析口径

- 模型：Qwen3-4B-Instruct-2507，共 36 层、8 个 KV heads、head dim 128。
- 布局：前 64 token BF16，中间 INT1，最近 256 token BF16；仅分析中间区域。
- 量化：K clip=0.96、V clip=0.92，逐 token/head BF16 scale，对称 sign INT1。
- anchor：每层、每请求、每 KV head 选 28 个 K/V 对并保存为 FP16。
- 校准数据：GPQA Q/K/V dump 约 2 万 token；其中 17 个 prompt 存在非空中间区域，共 1747 个中间 token。

当前排序公式为：

```text
eK_i = ||K_i - Khat_i||²
eV_i = ||V_i - Vhat_i||²
s_current(i) = 0.5 * (eK_i / max(||K_i||², eps)
                    + eV_i / max(||V_i||², eps))
```

## 结果

按 6 层分组如下；`efficiency` 是当前公式实际消除的绝对 K/V SSE，占同预算绝对-MSE oracle 的比例。

| 层 | 原始 joint MSE | 当前消除 | 绝对-MSE oracle 消除 | efficiency |
|---|---:|---:|---:|---:|
| 0–5 | 1.6489 | 24.02% | 29.05% | 80.99% |
| 6–11 | 0.9856 | 24.02% | 26.70% | 89.29% |
| 12–17 | 0.7940 | 24.79% | 26.74% | 92.43% |
| 18–23 | 0.8737 | 24.95% | 26.44% | 94.23% |
| 24–29 | 0.9959 | 24.91% | 28.13% | 87.17% |
| 30–35 | 3.1401 | 24.96% | 33.84% | 72.03% |

全层按 SSE 加权后：

- anchor 前 joint MSE：1.40637。
- 当前公式 anchor 后：1.06054，消除 24.59%。
- 绝对-MSE oracle anchor 后：0.97879，消除 30.40%。
- 在 anchor 数量、存储和 correction 都不变时，oracle 使 anchor 后 MSE 再下降 7.71%。
- 当前选择与 oracle 的平均集合重叠率只有 48.91%。

主要异常层：

| 层 | K MSE | V MSE | 当前消除 | oracle 消除 | efficiency |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.4907 | 0.00035 | 24.48% | 35.58% | 68.79% |
| 29 | 1.2830 | 1.6079 | 24.09% | 32.09% | 75.06% |
| 31 | 1.2816 | 2.4606 | 25.31% | 33.36% | 75.87% |
| 32 | 1.2438 | 3.7081 | 24.53% | 34.37% | 71.37% |
| 33 | 1.1429 | 8.9430 | 24.26% | 37.22% | 65.19% |
| 34 | 1.3860 | 7.2420 | 25.38% | 35.43% | 71.64% |
| 35 | 3.2592 | 3.7857 | 25.01% | 32.26% | 77.52% |

前层几乎完全由 K 的绝对误差主导，末层则转为 V 主导。当前逐 token 相对归一化会抹掉这种层间、K/V 间的绝对幅值差异，因此总是稳定消除约 25%，却错过高能量异常 token。

## 理论优化空间

### 1. 如果目标是缓存张量 MSE

固定每个 KV head 选择 28 个 K/V 对时，精确最优分数就是：

```text
s_cache(i) = eK_i + eV_i
```

这是可证明的：选中一个 token 后，其 K/V reconstruction SSE 被移除；总收益可加，故直接取 `eK+eV` 最大的 28 个。无需相对归一化，也无需调 K/V alpha。建议把它作为缓存 MSE 的 oracle 和首个 A/B 候选。

### 2. 如果目标是 attention 输出误差或任务精度

缓存 MSE 并不是 attention 的真实目标。对单个 query 做一阶展开，token `i` 的误差影响近似为：

```text
u_i(q) = p_i² * ||delta_v_i
          + ((q · delta_k_i) / sqrt(d)) * (v_i - o)||²
```

其中 `p_i` 是 attention probability，`o` 是精确 attention 输出。这个式子说明：

- K 应按 query 投影误差 `q·delta_k` 衡量，而不是欧氏 MSE；
- V 误差要乘 attention probability；
- K/V 权重随 layer、head、query 改变，不存在普适的固定 0.5；
- 同一 token 的 K/V 误差可能相互抵消或增强。

可实现的静态近似是用校准 query 协方差 `Cq`，令 K 项为 `delta_k^T Cq delta_k / d`；更准确的动态版本则在 decode 时使用当前 query 和近似 attention probability。动态方法计算更贵，应先离线证明收益。

### 3. clip 与 scale

对称一比特 reconstruction `xhat=s*sign(x)` 的原始张量 MSE 最优 scale 为：

```text
s* = mean(abs(x))
```

当前 clip 不改变符号，却用 `mean(abs(clipped_x))` 缩小 scale；从“重建原始 x”的 MSE 目标看不可能优于 `mean(abs(x))`。实测取消 clip 后 K MSE 由 1.88228 降至 1.88135，V 由 0.93046 降至 0.92771，joint 只改善约 0.13%。BF16 scale 换 FP32 也几乎无收益。因此这里理论上可改，但不是当前主优化点。

若允许改变格式，可尝试每 32/64 维一个 scale，或二中心/带 offset 的 binary quantizer；这会增加 scale 元数据或修改 kernel，不能再视为零成本公式替换。

## 针对不同处理方法的校验方法

| 改动类型 | 主校验指标 | 必须做的对照/不变量 |
|---|---|---|
| clip/scale/INT1 quantizer | 每层、每 head 的 reconstruction MSE；P50/P90/P99 | Torch 与 Triton bit/scale/dequant 逐元素一致；分别报告 clipping error 和 quantization error |
| 缓存-MSE anchor 排序 | removed SSE、anchor 后 MSE、相对 oracle regret | 同一输入、同一 28 配额；batch top-k 与流式 rollover top-k 完全一致；稳定 tie、请求隔离、head 隔离 |
| query-covariance 静态排序 | `q·delta_k` 的预测误差、attention-output MSE | 与 absolute-SSE/current 三方比较；按层/头报告 Spearman、top-28 regret，不能只看 cache MSE |
| query-aware 动态排序 | attention output MSE、LSE 误差、logit KL | 用 dense FP32 attention 作 reference；覆盖高/低 attention token、GQA 四个 query heads、softmax 极值与抵消情况 |
| 分层/分头自适应预算 | 固定总 anchor 内存下的全模型目标 | 总 FP16 条目数严格相同；比较均匀 28、按层 MSE、按 attention sensitivity 分配 |
| correction kernel | 输出与显式“替换 28 行后重算 dense attention”的误差 | 检查先减 INT1 再加 FP16、无重复位置、count<28、rollover、并发请求、请求复用 |
| 端到端方法选择 | 固定解码下 token agreement、logit KL、各数据集指标和 Strict Prompt 宏平均 | paired prompts、相同 seed/采样参数；用 paired bootstrap/McNemar 给置信区间，同时报告吞吐、显存与 TTFT |

## 建议实验顺序

1. 不改存储和 correction，只把 selector A/B 为 `relative` 与 `absolute eK+eV`；先做本地 attention-output replay。
2. 若 absolute 同时降低 attention-output MSE，再跑完整 `vllm_turbo_baseline_acc`。
3. 若 cache MSE 降但 attention 不降，转向一阶 query-aware score，并用 dense attention leave-one-token-out 校准，而不是继续调固定 K/V alpha。
4. clip/scale 仅作为次要消融；预期收益远小于 selector。

逐层完整数据见 `layerwise_anchor_mse.csv`，机器可读汇总见 `summary.json`。当前 GPQA 校准集中只有 17 个长 prompt，因此上述结果足以定位公式问题，但正式切换 selector 前还应从目标评测集补充一批长上下文 replay。
