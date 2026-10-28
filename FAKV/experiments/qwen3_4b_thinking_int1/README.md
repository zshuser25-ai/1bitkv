# Qwen3-4B-Thinking-2507 OSCAR INT1

This experiment has one configuration:

- model: `Qwen/Qwen3-4B-Thinking-2507`, revision
  `768f209d9ea81521153ed38c47d515654e938aea`
- OSCAR window: first 64 tokens BF16, latest 256 tokens BF16
- middle K/V cache: symmetric INT1 for all 128 head dimensions
- INT1 representation: one sign bit per value and one BF16 mean-absolute
  scale per token and KV head, so each 128-dimensional K or V row occupies
  16 packed bytes plus its scale
- official RotationZoo calibration: `seq20000_prompt85_group128`
- K/V clipping: 0.96 / 0.92
- dataset: `model_agnostic_accuracy_official_v5`
- evaluated benchmarks: GSM8K, IFEval, LiveCodeBench v6, MultiPL-E
- reasoning effort: `high` (SGLang's supported equivalent for the suite's
  otherwise unsupported `max` value)
- execution: tensor parallel across GPU 0-3, evaluation concurrency 32

WikiText-2 is excluded because perplexity scoring does not exercise the
autoregressive OSCAR KV path used by this experiment.

Run on the POD in this order:

```bash
bash experiments/qwen3_4b_thinking_int1/scripts/download_assets.sh
bash experiments/qwen3_4b_thinking_int1/scripts/run_kernel_test.sh
bash experiments/qwen3_4b_thinking_int1/scripts/launch_server.sh
bash experiments/qwen3_4b_thinking_int1/scripts/run_eval.sh
```

The server PID and log are written under `experiments/qwen3_4b_thinking_int1`.
Accuracy outputs are written to `results/official_v5` and support `--resume`.
