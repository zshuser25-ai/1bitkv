# Qwen3-8B OSCAR pure-INT1 A/B

Fixed settings for both groups: prefix 64 BF16, middle pure INT1 K/V, recent
256 BF16, OSCAR RotationZoo rotations, K clip 0.96, V clip 0.92, TP=4,
context 32768, and official_v5 2360-sample accuracy suite.
The Qwen3 chat template is explicitly rendered with `enable_thinking=False`
for calibration, tokenization preflight, smoke requests, and generation.

- `baseline`: no FP16 anchors (`SGLANG_OSCAR_FP16_ANCHOR_TOKENS=0`).
- `fisher`: 28 FP16 anchors per request/layer/KV-head, selected with a
  Qwen3-8B-specific diagonal-Fisher checkpoint calibrated on independent
  GPQA-diamond prompts.

Calibration and both formal A/B runs are executed sequentially on the same
4x RTX 4090 pod. A800 outputs are not used.

Formal result directories use the `_nonthinking` suffix so they cannot be
confused with the interrupted default-thinking run.
