"""GPU smoke for ERAKV_V2's supported anchor layouts."""

import math
import os

CAPACITY = int(os.environ.get("SGLANG_OSCAR_FP16_ANCHOR_TOKENS", "56"))
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_kv_pool import (
    UnifiedInt2HPKVPool,
    dequantize_attention_mixed_k_torch,
    pack_attention_mixed_k_torch,
)
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _apply_fp16_anchor_correction,
)
from sglang.srt.layers.attention.quantized_kv_prefill import (
    replace_dequantized_prefix_with_fp16_anchors,
)


def make_pool(exact_k: torch.Tensor, exact_v: torch.Tensor, capacity: int = CAPACITY):
    if capacity not in (28, 56, 120):
        raise ValueError("ERAKV_V2 permits 28, 56, or 120 anchors")
    rows, heads, dim = exact_k.shape
    packed_k, high_k, low_k = pack_attention_mixed_k_torch(
        exact_k, 0, torch.float32
    )
    packed_v, high_v, low_v = pack_attention_mixed_k_torch(
        exact_v, 0, torch.float32
    )
    pool = object.__new__(UnifiedInt2HPKVPool)
    pool.anchor_enabled = True
    pool.anchor_capacity = capacity
    pool.anchor_fisher_k = None
    pool.anchor_fisher_v = None
    pool.device = exact_k.device
    pool.layer_num = 1
    pool.start_layer = 0
    pool.max_req_slots = 2
    pool.head_num = heads
    pool.head_dim = dim
    pool.v_head_dim = dim
    pool.k_buffer = [packed_k]
    pool.v_buffer = [packed_v]
    pool.k_scales_zeros = [high_k]
    pool.v_scales_zeros = [high_v]
    pool.k_low_scales = [low_k]
    pool.v_low_scales = [low_v]
    pool._R_k = torch.eye(dim, device=pool.device)[None]
    pool._R_v = torch.eye(dim, device=pool.device)[None]
    shape = (pool.max_req_slots, heads, pool.anchor_capacity)
    pool.anchor_k_buffer = [
        torch.zeros((*shape, dim), dtype=torch.float16, device=pool.device)
    ]
    pool.anchor_v_buffer = [
        torch.zeros((*shape, dim), dtype=torch.float16, device=pool.device)
    ]
    pool.anchor_scores = [
        torch.full(shape, -float("inf"), dtype=torch.float32, device=pool.device)
    ]
    pool.anchor_positions = [
        torch.full(shape, -1, dtype=torch.int32, device=pool.device)
    ]
    pool.anchor_quant_slots = [
        torch.full(shape, -1, dtype=torch.int64, device=pool.device)
    ]
    pool.anchor_counts = [
        torch.zeros((pool.max_req_slots, heads), dtype=torch.int16, device=pool.device)
    ]
    return pool


def main():
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    rows, heads, groups, dim = CAPACITY + 15, 2, 4, 128
    exact_k = torch.randn(rows, heads, dim, device=device)
    exact_v = torch.randn(rows, heads, dim, device=device)
    # Create deterministic ties at positions 0 and 1; earlier position wins.
    exact_k[1] = exact_k[0]
    exact_v[1] = exact_v[0]
    pool = make_pool(exact_k, exact_v)
    slots = torch.arange(rows, device=device, dtype=torch.int64)
    req = torch.zeros(rows, device=device, dtype=torch.int64)
    positions = slots.clone()
    pool._update_fp16_anchors_from_rows(
        0, slots, exact_k, exact_v, req, positions
    )

    reconstructed_k = dequantize_attention_mixed_k_torch(
        pool.k_buffer[0], pool.k_scales_zeros[0], pool.k_low_scales[0], 0,
        torch.float32,
    )
    reconstructed_v = dequantize_attention_mixed_k_torch(
        pool.v_buffer[0], pool.v_scales_zeros[0], pool.v_low_scales[0], 0,
        torch.float32,
    )
    k_error = (exact_k - reconstructed_k).square().sum(-1)
    v_error = (exact_v - reconstructed_v).square().sum(-1)
    score = 0.5 * (
        k_error / exact_k.square().sum(-1).clamp_min(2.0**-24)
        + v_error / exact_v.square().sum(-1).clamp_min(2.0**-24)
    )
    for head in range(heads):
        position_order = torch.argsort(positions, stable=True)
        expected = position_order[
            torch.argsort(score[position_order, head], descending=True, stable=True)
        ][:CAPACITY]
        actual = pool.anchor_positions[0][0, head].to(torch.int64)
        torch.testing.assert_close(actual, expected)
        assert int(pool.anchor_counts[0][0, head]) == CAPACITY

    # The calibrated selector must use the diagonal weights rather than the
    # legacy relative MSE, independently for every KV head.
    fisher_pool = make_pool(exact_k, exact_v)
    fisher_k = torch.linspace(0.1, 2.0, dim, device=device).repeat(heads, 1)
    fisher_v = torch.linspace(2.0, 0.1, dim, device=device).repeat(heads, 1)
    fisher_k[1] = fisher_k[1].flip(0)
    fisher_v[1] = fisher_v[1].flip(0)
    fisher_pool.anchor_fisher_k = fisher_k[None]
    fisher_pool.anchor_fisher_v = fisher_v[None]
    fisher_pool._update_fp16_anchors_from_rows(
        0, slots, exact_k, exact_v, req, positions
    )
    fisher_score = (
        (exact_k - reconstructed_k).square() * fisher_k[None]
    ).sum(-1) + (
        (exact_v - reconstructed_v).square() * fisher_v[None]
    ).sum(-1)
    for head in range(heads):
        position_order = torch.argsort(positions, stable=True)
        expected = position_order[
            torch.argsort(
                fisher_score[position_order, head], descending=True, stable=True
            )
        ][:CAPACITY]
        torch.testing.assert_close(
            fisher_pool.anchor_positions[0][0, head].long(), expected
        )

    q = torch.randn(1, heads * groups, dim, device=device)
    q4 = q.reshape(1, heads, groups, dim)
    approx_logits = torch.einsum("bhgd,thd->bhgt", q4, reconstructed_k)
    approx_logits *= 1.0 / math.sqrt(dim)
    base_weights = torch.softmax(approx_logits, dim=-1)
    base_o = torch.einsum("bhgt,thd->bhgd", base_weights, reconstructed_v)
    base_lse = torch.logsumexp(approx_logits, dim=-1).reshape(1, heads * groups)
    corrected = base_o.reshape(1, heads * groups, dim).clone()
    pool.apply_fp16_anchor_correction(
        0, q, corrected, base_lse, torch.tensor([0], device=device),
        1.0 / math.sqrt(dim),
    )

    target_k = reconstructed_k.clone()
    target_v = reconstructed_v.clone()
    for head in range(heads):
        anchor_pos = pool.anchor_positions[0][0, head].long()
        target_k[anchor_pos, head] = pool.anchor_k_buffer[0][0, head].float()
        target_v[anchor_pos, head] = pool.anchor_v_buffer[0][0, head].float()
    target_logits = torch.einsum("bhgd,thd->bhgt", q4, target_k)
    target_logits *= 1.0 / math.sqrt(dim)
    target = torch.einsum(
        "bhgt,thd->bhgd", torch.softmax(target_logits, dim=-1), target_v
    ).reshape_as(corrected)
    torch.testing.assert_close(corrected, target, rtol=2e-5, atol=2e-5)
    fused = base_o.reshape(1, heads * groups, dim).clone()
    _apply_fp16_anchor_correction(
        q,
        fused,
        base_lse,
        torch.tensor([0], device=device, dtype=torch.int64),
        pool.k_buffer[0],
        pool.v_buffer[0],
        pool.k_low_scales[0],
        pool.v_low_scales[0],
        pool.anchor_k_buffer[0],
        pool.anchor_v_buffer[0],
        pool.anchor_quant_slots[0],
        pool.anchor_counts[0],
        1.0 / math.sqrt(dim),
    )
    torch.testing.assert_close(fused, target, rtol=2e-5, atol=2e-5)

    # Exercise the fused 8-token decode rollover selector as well.
    rollover_k = torch.cat(
        [exact_k, torch.randn(8, heads, dim, device=device)], dim=0
    )
    rollover_v = torch.cat(
        [exact_v, torch.randn(8, heads, dim, device=device)], dim=0
    )
    flush_pool = make_pool(rollover_k, rollover_v)
    flush_pool.anchor_fisher_k = fisher_k[None]
    flush_pool.anchor_fisher_v = fisher_v[None]
    flush_pool.hp_k_buffer = [rollover_k[-8:].to(torch.bfloat16)]
    flush_pool.hp_v_buffer = [rollover_v[-8:].to(torch.bfloat16)]
    flush_pool._update_fp16_anchors_from_rows(
        0, slots, rollover_k[:rows], rollover_v[:rows], req, positions
    )
    plan = SimpleNamespace(
        bs=1,
        flush_interval=8,
        src_hp_slot=torch.arange(8, device=device, dtype=torch.int64),
        dst_quant_slots=torch.arange(rows, rows + 8, device=device, dtype=torch.int64),
        flush_pos=torch.arange(rows, rows + 8, device=device, dtype=torch.int32),
        valid_mask=torch.ones(8, device=device, dtype=torch.int8),
    )
    flush_pool.update_fp16_anchors_from_flush(
        plan, torch.tensor([0], device=device, dtype=torch.int64)
    )
    rk = dequantize_attention_mixed_k_torch(
        flush_pool.k_buffer[0], flush_pool.k_scales_zeros[0],
        flush_pool.k_low_scales[0], 0, torch.float32,
    )
    rv = dequantize_attention_mixed_k_torch(
        flush_pool.v_buffer[0], flush_pool.v_scales_zeros[0],
        flush_pool.v_low_scales[0], 0, torch.float32,
    )
    # Flush exact values originate in the BF16 recent arena.
    scoring_k = rollover_k.clone()
    scoring_v = rollover_v.clone()
    scoring_k[-8:] = scoring_k[-8:].to(torch.bfloat16).float()
    scoring_v[-8:] = scoring_v[-8:].to(torch.bfloat16).float()
    all_score = (
        (scoring_k - rk).square() * fisher_k[None]
    ).sum(-1) + (
        (scoring_v - rv).square() * fisher_v[None]
    ).sum(-1)
    all_positions = torch.arange(rows + 8, device=device)
    for head in range(heads):
        pos_order = torch.argsort(all_positions, stable=True)
        expected = pos_order[
            torch.argsort(all_score[pos_order, head], descending=True, stable=True)
        ][:CAPACITY]
        torch.testing.assert_close(
            flush_pool.anchor_positions[0][0, head].long(), expected
        )

    # Route B prefill: selected historical INT1 rows are replaced in-place,
    # independently by request and KV head, before FlashAttention.
    prefix_indices = torch.tensor([10, 11, 12, 20, 21], device=device)
    prefix_indptr = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    prefix_k = torch.zeros(5, heads, dim, device=device, dtype=torch.float16)
    prefix_v = torch.zeros_like(prefix_k)
    anchor_k = torch.randn(2, heads, CAPACITY, dim, device=device, dtype=torch.float16)
    anchor_v = torch.randn_like(anchor_k)
    anchor_slots = torch.full(
        (2, heads, CAPACITY), -1, device=device, dtype=torch.int64
    )
    anchor_positions = torch.full_like(anchor_slots, -1)
    anchor_counts = torch.tensor([[2, 1], [1, 1]], device=device, dtype=torch.int16)
    anchor_slots[0, 0, :2] = torch.tensor([10, 12], device=device)
    anchor_slots[0, 1, 0] = 11
    anchor_slots[1, 0, 0] = 21
    anchor_slots[1, 1, 0] = 20
    anchor_positions[0, 0, :2] = torch.tensor([0, 2], device=device)
    anchor_positions[0, 1, 0] = 1
    anchor_positions[1, 0, 0] = 1
    anchor_positions[1, 1, 0] = 0
    replaced = replace_dequantized_prefix_with_fp16_anchors(
        prefix_k,
        prefix_v,
        prefix_indices,
        prefix_indptr,
        anchor_k,
        anchor_v,
        anchor_slots,
        anchor_positions,
        anchor_counts,
    )
    assert replaced == 5
    torch.testing.assert_close(prefix_k[0, 0], anchor_k[0, 0, 0])
    torch.testing.assert_close(prefix_k[2, 0], anchor_k[0, 0, 1])
    torch.testing.assert_close(prefix_k[1, 1], anchor_k[0, 1, 0])
    torch.testing.assert_close(prefix_k[4, 0], anchor_k[1, 0, 0])
    torch.testing.assert_close(prefix_k[3, 1], anchor_k[1, 1, 0])
    assert torch.count_nonzero(prefix_k[3, 0]) == 0
    print(
        f"PASS anchor_count={CAPACITY} selection=ERAKV_joint_relative_error "
        "fisher_selection=diagonal_final_loss_sensitivity "
        f"correction_max_abs={(corrected - target).abs().max().item():.3e} "
        f"fused_max_abs={(fused - target).abs().max().item():.3e} "
        f"rollover=fused_top{CAPACITY} prefill=historical_replacement"
    )


if __name__ == "__main__":
    main()
