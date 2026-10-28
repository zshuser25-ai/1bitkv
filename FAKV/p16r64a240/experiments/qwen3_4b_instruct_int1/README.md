# Qwen3-4B-Instruct-2507 OSCAR INT1

This experiment has one configuration:

- model: `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`
- OSCAR window: first 64 tokens BF16, latest 256 tokens BF16
- middle K/V cache: symmetric INT1 for all 128 head dimensions
- INT1 representation: one sign bit per value and one BF16 mean-absolute
  scale per token and KV head, so each 128-dimensional K or V row occupies
  16 packed bytes plus its scale
- model-specific GPQA calibration: `seq20000_prompt88_group128`
- K/V clipping: 0.96 / 0.92
- dataset: `model_agnostic_accuracy_official_v5`
- evaluated benchmarks: GSM8K, IFEval, LiveCodeBench v6, MultiPL-E
- reasoning effort: disabled (the Instruct checkpoint is non-thinking)
- execution: tensor parallel across GPU 0-3, evaluation concurrency 32

WikiText-2 is excluded because perplexity scoring does not exercise the
autoregressive OSCAR KV path used by this experiment.

Run on the POD in this order:

```bash
bash experiments/qwen3_4b_instruct_int1/scripts/download_assets.sh
bash experiments/qwen3_4b_instruct_int1/scripts/calibrate_rotation.sh
bash experiments/qwen3_4b_instruct_int1/scripts/run_kernel_test.sh
bash experiments/qwen3_4b_instruct_int1/scripts/launch_server.sh
bash experiments/qwen3_4b_instruct_int1/scripts/run_eval.sh
```

The server PID and log are written under `experiments/qwen3_4b_instruct_int1`.
Accuracy outputs are written to `results/official_v5` and support `--resume`.

## Calibrated Fisher anchor selector (v1)

The optional v1 selector replaces ERAKV's fixed relative-MSE ranking with
the diagonal final-loss sensitivity score

```text
score = sum(FK[layer, head, dim] * (K - K_int1)^2)
      + sum(FV[layer, head, dim] * (V - V_int1)^2)
```

`FK/FV` are calibrated once from tail next-token cross-entropy gradients on
independent GPQA-diamond prompts.  Calibration captures post-RoPE K/V,
transforms their gradients into the deployed OSCAR rotation coordinates,
and writes a roughly 300 KiB FP32 checkpoint.  The runtime does no backward
pass and keeps the same 28 FP16 anchors.

```bash
/root/oscar-env/bin/pip install accelerate
CUDA_VISIBLE_DEVICES=0,1,2,3 /root/oscar-env/bin/python \
  experiments/qwen3_4b_instruct_int1/scripts/calibrate_anchor_fisher.py \
  --model /root/oscar-assets/Qwen3-4B-Instruct-2507 \
  --source gpqa_diamond --max-used-samples 16

SGLANG_OSCAR_ANCHOR_FISHER_PATH=$PWD/rotation/qwen3-4B-instruct-2507/GPQA/seq20000_prompt88_group128/anchor_fisher_diag_v1.pt \
  bash experiments/qwen3_4b_instruct_int1/scripts/launch_server.sh
```

Leaving `SGLANG_OSCAR_ANCHOR_FISHER_PATH` empty preserves the original
relative-MSE selector.  Checkpoint loading fails closed on model, shape,
coordinate-system, K/V rotation hash, or TP head-shard mismatch.
