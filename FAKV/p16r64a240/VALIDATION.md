# ERAKV_V2 Validation

Date: 2026-09-22  
GPU: 4 × RTX 4090  
Model: Qwen3-4B-Instruct-2507  
Fisher checkpoint: `anchor_fisher_diag_v1.pt` for Qwen3-4B-Instruct-2507

## Static validation

The four modified runtime modules and the anchor smoke test passed Python
bytecode compilation.  The launcher passed shell parsing.

The implementation rejects every positive anchor capacity other than 28.
The enabled layout additionally requires:

```text
BF16 prefix = 64
INT1 middle = remaining historical middle region
BF16 recent = 256
FP16 anchor capacity = 28 per layer / KV head
```

## GPU operator smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/nfs/AE/zsh/ERAKV_V2/sglang-research/python \
/root/oscar-env/bin/python \
  experiments/qwen3_4b_instruct_int1/tests/smoke_fp16_anchors.py
```

Result:

```text
PASS anchor_count=28
selection=ERAKV_joint_relative_error
fisher_selection=diagonal_final_loss_sensitivity
correction_max_abs=1.788e-07
fused_max_abs=2.980e-07
rollover=fused_top28
prefill=historical_replacement
```

This covers fixed-capacity selection, Fisher selection, decode correction,
decode rollover, request/head-isolated prefill replacement, and preservation
of non-anchor rows.

## Real chunked-prefill integration smoke

The server was started with `chunked_prefill_size=2048`, the Fisher selector,
and the fixed `64 / INT1 / 256 + 28 anchors` layout.  Startup confirmed on all
four TP ranks:

```text
ERAKV FP16 anchors enabled (capacity=28 per request/layer/KV-head)
calibrated diagonal-Fisher anchor selector enabled
chunked_prefill_size=2048
```

A 7005-token prompt forced multiple prefill chunks.  The scheduler processed
three 2048-token chunks followed by the tail chunk, returned HTTP 200, and
generated four tokens without an anchor-position/slot invariant failure.
Consequently, later chunks successfully traversed the Route B historical
replacement path using the pre-write 28-anchor snapshots.

The smoke server was then terminated gracefully.  No evaluation job was
resumed and GPU memory returned to zero.
