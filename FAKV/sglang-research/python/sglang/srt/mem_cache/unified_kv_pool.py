"""
Unified HP + int2 KV cache pool.

Quant arena: paged with ``N_Q`` slots per page. HP arena: shared HP-prefix
pool (paged) followed by per-request HP-recent ring slabs. Slot id namespace
is flat (``[0, num_quant_pages*N_Q)`` quant, ``[HP_OFFSET, ...)`` HP), and
kernels dispatch by ``slot >= HP_OFFSET``.
"""

from __future__ import annotations

import logging
import os
import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.QuantKernel.fused_hadamard_int2_kv import (
    quantized_set_kv_int2_pretransformed_triton,
)
from sglang.QuantKernel.oscar_rotation_clip_int2_kv import (
    quantized_set_kv_int2_oscar_rotate_k_clip_triton,
    quantized_set_kv_int2_pretransformed_clip_triton,
)
from sglang.srt.mem_cache.kv_quant_kernels import _get_num_scale_groups
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)
from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    OscarRotationConfig,
    _set_kv_buffer_impl,
    get_tensor_size_bytes,
    load_oscar_rotation_config,
    load_oscar_rotations,
)

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_oscar_k_layout(
    mode: str, *, head_dim: int, base_group_size: Optional[int]
) -> dict:
    """Resolve K-only OSCAR layout without accidentally changing V groups."""
    normalized = mode.strip().lower()
    allowed = (
        "off",
        "attention",
        "random",
        "deployed",
        "split_k2_separate",
        "product_vq",
    )
    if normalized not in allowed:
        raise ValueError(
            "SGLANG_OSCAR_AA_K_MODE must be " + "/".join(allowed) + f", got {normalized!r}"
        )
    group_size = head_dim if base_group_size is None else int(base_group_size)
    if group_size <= 0 or head_dim % group_size:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by base_group_size ({group_size})"
        )
    split_k2 = normalized == "split_k2_separate"
    product_vq = normalized == "product_vq"
    k_group_size = head_dim // 2 if split_k2 else group_size
    return {
        "mode": normalized,
        "mixed_int1": normalized in ("attention", "random", "deployed"),
        "product_vq": product_vq,
        "split_k2_separate": split_k2,
        "k_group_size": k_group_size,
        "k_num_groups": head_dim // k_group_size,
        "v_group_size": group_size,
        "v_num_groups": head_dim // group_size,
    }


def _clip_rows_torch(x: torch.Tensor, ratio: float) -> torch.Tensor:
    """Reference per-row symmetric clipping matching the OSCAR index rule."""
    if ratio <= 0.0:
        return x
    dim = x.shape[-1]
    clip_index = min(max(int(ratio * dim), 0), dim - 1)
    threshold = torch.kthvalue(x.float().abs(), clip_index + 1, dim=-1).values
    return x.float().clamp(-threshold[..., None], threshold[..., None]).to(x.dtype)


def _pack_int2_rows_torch(
    x: torch.Tensor, scale_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one asymmetric INT2 group per row using OSCAR quarter layout."""
    dim = x.shape[-1]
    if dim % 4 != 0:
        raise ValueError(f"INT2 dimension must be divisible by four, got {dim}")
    x32 = x.float()
    val_min = x32.amin(dim=-1, keepdim=True)
    val_max = x32.amax(dim=-1, keepdim=True)
    scale = (val_max - val_min).clamp(min=1e-8) / 3.0
    zero = -val_min / scale
    q = (x32 / scale + zero + 0.5).clamp(0, 3).to(torch.uint8)
    bq = dim // 4
    packed = (
        q[..., :bq]
        | (q[..., bq : 2 * bq] << 2)
        | (q[..., 2 * bq : 3 * bq] << 4)
        | (q[..., 3 * bq :] << 6)
    ).contiguous()
    scales_zeros = torch.cat([scale, zero], dim=-1).to(scale_dtype).contiguous()
    return packed, scales_zeros


def pack_attention_mixed_k_torch(
    x: torch.Tensor,
    high_dim: int,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack K as high-subspace INT2 plus low-subspace symmetric INT1."""
    low_dim = x.shape[-1] - high_dim
    if high_dim % 4 or low_dim % 8:
        raise ValueError(
            f"mixed K requires high_dim % 4 == 0 and low_dim % 8 == 0, "
            f"got {high_dim}+{low_dim}"
        )
    if high_dim:
        high_packed, high_sz = _pack_int2_rows_torch(
            x[..., :high_dim], scale_dtype
        )
    else:
        high_packed = torch.empty(
            *x.shape[:-1], 0, dtype=torch.uint8, device=x.device
        )
        # Pure INT1 has no affine-INT2 subspace. Keep the tensor rank/API,
        # but do not materialize a legacy scale+zero pair per row.
        high_sz = torch.empty(
            *x.shape[:-1], 0, dtype=scale_dtype, device=x.device
        )
    low = x[..., high_dim:].float()
    low_scale = low.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    bits = (low >= 0).to(torch.uint8).reshape(*low.shape[:-1], low_dim // 8, 8)
    weights = (1 << torch.arange(8, dtype=torch.uint8, device=x.device))
    low_packed = (bits * weights).sum(dim=-1).to(torch.uint8).contiguous()
    return (
        torch.cat([high_packed, low_packed], dim=-1).contiguous(),
        high_sz,
        low_scale.to(torch.bfloat16).contiguous(),
    )


def dequantize_attention_mixed_k_torch(
    packed: torch.Tensor,
    high_scales_zeros: torch.Tensor,
    low_scales: torch.Tensor,
    high_dim: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference inverse of :func:`pack_attention_mixed_k_torch`."""
    high_bytes = high_dim // 4
    low_bytes = packed.shape[-1] - high_bytes
    p2 = packed[..., :high_bytes]
    scale = high_scales_zeros[..., 0:1].float()
    zero = high_scales_zeros[..., 1:2].float()
    high = torch.cat(
        [
            ((p2 >> shift) & 0x03).float()
            for shift in (0, 2, 4, 6)
        ],
        dim=-1,
    )
    high = (high - zero) * scale
    p1 = packed[..., high_bytes:]
    bit_ids = torch.arange(8, dtype=torch.uint8, device=packed.device)
    low_bits = ((p1[..., None] >> bit_ids) & 1).reshape(
        *packed.shape[:-1], low_bytes * 8
    )
    low = torch.where(low_bits.bool(), 1.0, -1.0) * low_scales.float()
    return torch.cat([high, low], dim=-1).to(output_dtype)


def _validate_product_vq_layout(
    x: torch.Tensor,
    high_dim: int,
    codebooks: torch.Tensor,
    distance_weights: Optional[torch.Tensor],
) -> tuple[int, int, int, int]:
    if x.ndim < 3:
        raise ValueError("product VQ expects [..., heads, dim] input")
    low_dim = x.shape[-1] - high_dim
    if high_dim % 4 or low_dim <= 0:
        raise ValueError("product VQ requires an INT2-compatible high subspace")
    if codebooks.ndim != 4 or codebooks.shape[0] != x.shape[-2]:
        raise ValueError("codebooks must have [heads, groups, codes, vector_dim]")
    groups, codes, vector_dim = codebooks.shape[1:]
    if low_dim != groups * vector_dim or codes not in (16, 256):
        raise ValueError(
            "product VQ requires low_dim == groups*vector_dim and 16/256 codes"
        )
    index_bits = 4 if codes == 16 else 8
    if groups * index_bits % 8:
        raise ValueError("VQ indices must occupy an integer number of bytes")
    if distance_weights is not None and tuple(distance_weights.shape) != (
        x.shape[-2],
        groups,
        vector_dim,
    ):
        raise ValueError("distance_weights must have [heads, groups, vector_dim]")
    return groups, codes, vector_dim, index_bits


def pack_product_vq_k_torch(
    x: torch.Tensor,
    high_dim: int,
    scale_dtype: torch.dtype,
    codebooks: torch.Tensor,
    distance_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack high INT2 plus low product-VQ with 4-bit or 8-bit indices."""
    groups, _, vector_dim, index_bits = _validate_product_vq_layout(
        x, high_dim, codebooks, distance_weights
    )
    high_packed, high_sz = _pack_int2_rows_torch(x[..., :high_dim], scale_dtype)
    low = x[..., high_dim:].float()
    low_scale = low.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    stored_scale = low_scale.to(torch.bfloat16)
    normalized = (low / stored_scale.float()).reshape(
        -1, x.shape[-2], groups, vector_dim
    )
    centers = codebooks.to(device=x.device, dtype=torch.float32)
    differences = normalized[..., None, :] - centers[None, ...]
    if distance_weights is None:
        distances = differences.square().sum(dim=-1)
    else:
        weights = distance_weights.to(device=x.device, dtype=torch.float32)
        distances = (
            differences.square() * weights[None, :, :, None, :]
        ).sum(dim=-1)
    indices = distances.argmin(dim=-1).to(torch.uint8)
    if index_bits == 4:
        low_packed = (indices[..., 0::2] | (indices[..., 1::2] << 4)).reshape(
            *x.shape[:-2], x.shape[-2], groups // 2
        )
    else:
        low_packed = indices.reshape(*x.shape[:-2], x.shape[-2], groups)
    return (
        torch.cat([high_packed, low_packed], dim=-1).contiguous(),
        high_sz,
        stored_scale.contiguous(),
    )


def dequantize_product_vq_k_torch(
    packed: torch.Tensor,
    high_scales_zeros: torch.Tensor,
    low_scale: torch.Tensor,
    high_dim: int,
    codebooks: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference inverse of :func:`pack_product_vq_k_torch`."""
    high_bytes = high_dim // 4
    low_packed = packed[..., high_bytes:]
    if codebooks.ndim != 4 or codebooks.shape[0] != packed.shape[-2]:
        raise ValueError("codebook shape does not match packed VQ layout")
    groups, codes, _ = codebooks.shape[1:]
    if codes not in (16, 256):
        raise ValueError("product VQ supports 16/256 codes")
    index_bits = 4 if codes == 16 else 8
    expected_bytes = groups * index_bits // 8
    if low_packed.shape[-1] != expected_bytes:
        raise ValueError("codebook shape does not match packed VQ index bytes")
    p2 = packed[..., :high_bytes]
    scale = high_scales_zeros[..., 0:1].float()
    zero = high_scales_zeros[..., 1:2].float()
    high = torch.cat(
        [((p2 >> shift) & 0x03).float() for shift in (0, 2, 4, 6)],
        dim=-1,
    )
    high = (high - zero) * scale
    if index_bits == 4:
        indices = torch.stack(
            [low_packed & 0x0F, (low_packed >> 4) & 0x0F], dim=-1
        ).reshape(-1, packed.shape[-2], groups)
    else:
        indices = low_packed.reshape(-1, packed.shape[-2], groups)
    centers = codebooks.to(device=packed.device, dtype=torch.float32)
    expanded = centers.unsqueeze(0).expand(indices.shape[0], -1, -1, -1, -1)
    chosen = torch.gather(
        expanded,
        3,
        indices[..., None, None].long().expand(
            -1, -1, -1, 1, centers.shape[-1]
        ),
    ).squeeze(3)
    low = chosen.reshape(*packed.shape[:-2], packed.shape[-2], -1)
    low = low * low_scale.float()
    return torch.cat([high, low], dim=-1).to(output_dtype)


@triton.jit
def _set_mixed_hp_buffer_kernel(
    src_ptr,
    dst_ptr,
    loc_ptr,
    num_tokens,
    row_dim: tl.constexpr,
    src_stride_token: tl.constexpr,
    src_stride_dim: tl.constexpr,
    dst_stride_loc: tl.constexpr,
    dst_stride_dim: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offs = block_idx * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    loc = tl.load(loc_ptr + token_idx)
    is_hp = loc >= HP_OFFSET
    hp_loc = loc - HP_OFFSET
    mask = is_hp & (token_idx < num_tokens) & (offs < row_dim)
    vals = tl.load(
        src_ptr + token_idx * src_stride_token + offs * src_stride_dim,
        mask=mask,
        other=0.0,
    )
    tl.store(
        dst_ptr + hp_loc * dst_stride_loc + offs * dst_stride_dim,
        vals,
        mask=mask,
    )


@triton.jit
def _update_fp16_anchors_from_flush_kernel(
    HP_K,
    HP_V,
    Quant_K,
    Quant_V,
    K_Low_Scales,
    V_Low_Scales,
    Fisher_K,
    Fisher_V,
    Src_HP,
    Dst_Quant,
    Flush_Pos,
    Valid,
    Req_Indices,
    Anchor_K,
    Anchor_V,
    Anchor_Scores,
    Anchor_Positions,
    Anchor_Slots,
    Anchor_Counts,
    stride_hkl,
    stride_hkh,
    stride_hkd,
    stride_hvl,
    stride_hvh,
    stride_hvd,
    stride_qkl,
    stride_qkh,
    stride_qkd,
    stride_qvl,
    stride_qvh,
    stride_qvd,
    stride_ksl,
    stride_ksh,
    stride_vsl,
    stride_vsh,
    stride_fkh,
    stride_fkd,
    stride_fvh,
    stride_fvd,
    stride_akr,
    stride_akh,
    stride_aka,
    stride_akd,
    stride_avr,
    stride_avh,
    stride_ava,
    stride_avd,
    stride_adr,
    stride_adh,
    stride_ada,
    stride_acr,
    stride_ach,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    FLUSH_INTERVAL: tl.constexpr,
    ANCHOR_CAPACITY: tl.constexpr,
    USE_FISHER: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Score one rollover tile and retain ERAKV's stable top anchors in-place."""
    batch = tl.program_id(0)
    head = tl.program_id(1)
    req = tl.load(Req_Indices + batch).to(tl.int64)
    old_count = tl.load(
        Anchor_Counts + req * stride_acr + head * stride_ach
    ).to(tl.int32)
    offs_c = tl.arange(0, BLOCK_C)
    offs_d = tl.arange(0, BLOCK_D)
    new_idx = offs_c - ANCHOR_CAPACITY
    old_mask = (offs_c < ANCHOR_CAPACITY) & (offs_c < old_count)
    new_range = (new_idx >= 0) & (new_idx < FLUSH_INTERVAL)
    plan_idx = batch * FLUSH_INTERVAL + tl.maximum(new_idx, 0)
    new_valid = new_range & (
        tl.load(Valid + plan_idx, mask=new_range, other=0).to(tl.int32) != 0
    )
    candidate_valid = old_mask | new_valid
    safe_src = tl.maximum(
        tl.load(Src_HP + plan_idx, mask=new_range, other=0).to(tl.int64), 0
    )
    safe_dst = tl.maximum(
        tl.load(Dst_Quant + plan_idx, mask=new_range, other=0).to(tl.int64), 0
    )
    new_pos = tl.load(Flush_Pos + plan_idx, mask=new_range, other=0).to(tl.int32)

    exact_k = tl.load(
        HP_K
        + safe_src[:, None] * stride_hkl
        + head * stride_hkh
        + offs_d[None, :] * stride_hkd,
        mask=new_valid[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    k_bytes = tl.load(
        Quant_K
        + safe_dst[:, None] * stride_qkl
        + head * stride_qkh
        + (offs_d[None, :] // 8) * stride_qkd,
        mask=new_valid[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0,
    ).to(tl.int32)
    k_bits = (k_bytes >> (offs_d[None, :] % 8)) & 1
    k_scale = tl.load(
        K_Low_Scales + safe_dst * stride_ksl + head * stride_ksh,
        mask=new_valid,
        other=0.0,
    ).to(tl.float32)
    reconstructed_k = tl.where(k_bits != 0, 1.0, -1.0) * k_scale[:, None]
    k_delta = exact_k - reconstructed_k
    if USE_FISHER:
        fisher_k = tl.load(
            Fisher_K + head * stride_fkh + offs_d * stride_fkd,
            mask=offs_d < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        k_error = tl.sum(k_delta * k_delta * fisher_k[None, :], axis=1)
    else:
        k_error = tl.sum(k_delta * k_delta, axis=1)
        k_norm = tl.maximum(tl.sum(exact_k * exact_k, axis=1), 2.0**-24)

    exact_v = tl.load(
        HP_V
        + safe_src[:, None] * stride_hvl
        + head * stride_hvh
        + offs_d[None, :] * stride_hvd,
        mask=new_valid[:, None] & (offs_d[None, :] < V_HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    v_bytes = tl.load(
        Quant_V
        + safe_dst[:, None] * stride_qvl
        + head * stride_qvh
        + (offs_d[None, :] // 8) * stride_qvd,
        mask=new_valid[:, None] & (offs_d[None, :] < V_HEAD_DIM),
        other=0,
    ).to(tl.int32)
    v_bits = (v_bytes >> (offs_d[None, :] % 8)) & 1
    v_scale = tl.load(
        V_Low_Scales + safe_dst * stride_vsl + head * stride_vsh,
        mask=new_valid,
        other=0.0,
    ).to(tl.float32)
    reconstructed_v = tl.where(v_bits != 0, 1.0, -1.0) * v_scale[:, None]
    v_delta = exact_v - reconstructed_v
    if USE_FISHER:
        fisher_v = tl.load(
            Fisher_V + head * stride_fvh + offs_d * stride_fvd,
            mask=offs_d < V_HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        v_error = tl.sum(v_delta * v_delta * fisher_v[None, :], axis=1)
        new_score = k_error + v_error
    else:
        v_error = tl.sum(v_delta * v_delta, axis=1)
        v_norm = tl.maximum(tl.sum(exact_v * exact_v, axis=1), 2.0**-24)
        new_score = 0.5 * (k_error / k_norm + v_error / v_norm)

    old_score = tl.load(
        Anchor_Scores
        + req * stride_adr
        + head * stride_adh
        + offs_c * stride_ada,
        mask=offs_c < ANCHOR_CAPACITY,
        other=-float("inf"),
    )
    old_position = tl.load(
        Anchor_Positions
        + req * stride_adr
        + head * stride_adh
        + offs_c * stride_ada,
        mask=offs_c < ANCHOR_CAPACITY,
        other=0,
    )
    scores = tl.where(old_mask, old_score, tl.where(new_valid, new_score, -float("inf")))
    positions = tl.where(old_mask, old_position, tl.where(new_valid, new_pos, 0))
    better = (scores[None, :] > scores[:, None]) | (
        (scores[None, :] == scores[:, None])
        & (positions[None, :] < positions[:, None])
    )
    better &= candidate_valid[None, :]
    rank = tl.sum(better.to(tl.int32), axis=1)
    selected = tl.sum(
        tl.where(
            (rank[:, None] == offs_c[None, :]) & candidate_valid[:, None],
            offs_c[:, None],
            0,
        ),
        axis=0,
    )
    out_mask = offs_c < ANCHOR_CAPACITY
    selected_old = selected < ANCHOR_CAPACITY
    selected_new = selected - ANCHOR_CAPACITY
    selected_plan = batch * FLUSH_INTERVAL + tl.maximum(selected_new, 0)
    selected_src = tl.maximum(
        tl.load(Src_HP + selected_plan, mask=~selected_old & out_mask, other=0), 0
    ).to(tl.int64)
    selected_dst = tl.maximum(
        tl.load(Dst_Quant + selected_plan, mask=~selected_old & out_mask, other=0), 0
    ).to(tl.int64)
    selected_score = tl.sum(
        tl.where(offs_c[:, None] == selected[None, :], scores[:, None], 0.0),
        axis=0,
    )
    selected_position = tl.sum(
        tl.where(offs_c[:, None] == selected[None, :], positions[:, None], 0),
        axis=0,
    )
    old_slot = tl.load(
        Anchor_Slots
        + req * stride_adr
        + head * stride_adh
        + selected * stride_ada,
        mask=selected_old & out_mask,
        other=0,
    )
    selected_slot = tl.where(selected_old, old_slot, selected_dst)

    old_k = tl.load(
        Anchor_K
        + req * stride_akr
        + head * stride_akh
        + selected[:, None] * stride_aka
        + offs_d[None, :] * stride_akd,
        mask=selected_old[:, None] & out_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    new_k = tl.load(
        HP_K
        + selected_src[:, None] * stride_hkl
        + head * stride_hkh
        + offs_d[None, :] * stride_hkd,
        mask=(~selected_old)[:, None] & out_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    old_v = tl.load(
        Anchor_V
        + req * stride_avr
        + head * stride_avh
        + selected[:, None] * stride_ava
        + offs_d[None, :] * stride_avd,
        mask=selected_old[:, None] & out_mask[:, None] & (offs_d[None, :] < V_HEAD_DIM),
        other=0.0,
    )
    new_v = tl.load(
        HP_V
        + selected_src[:, None] * stride_hvl
        + head * stride_hvh
        + offs_d[None, :] * stride_hvd,
        mask=(~selected_old)[:, None] & out_mask[:, None] & (offs_d[None, :] < V_HEAD_DIM),
        other=0.0,
    )
    tl.store(
        Anchor_Scores + req * stride_adr + head * stride_adh + offs_c * stride_ada,
        selected_score,
        mask=out_mask,
    )
    tl.store(
        Anchor_Positions + req * stride_adr + head * stride_adh + offs_c * stride_ada,
        selected_position,
        mask=out_mask,
    )
    tl.store(
        Anchor_Slots + req * stride_adr + head * stride_adh + offs_c * stride_ada,
        selected_slot,
        mask=out_mask,
    )
    tl.store(
        Anchor_K
        + req * stride_akr
        + head * stride_akh
        + offs_c[:, None] * stride_aka
        + offs_d[None, :] * stride_akd,
        tl.where(selected_old[:, None], old_k, new_k),
        mask=out_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
    )
    tl.store(
        Anchor_V
        + req * stride_avr
        + head * stride_avh
        + offs_c[:, None] * stride_ava
        + offs_d[None, :] * stride_avd,
        tl.where(selected_old[:, None], old_v, new_v),
        mask=out_mask[:, None] & (offs_d[None, :] < V_HEAD_DIM),
    )
    new_count = tl.minimum(
        ANCHOR_CAPACITY,
        old_count + tl.sum(new_valid.to(tl.int32), axis=0),
    )
    tl.store(
        Anchor_Counts + req * stride_acr + head * stride_ach,
        new_count,
    )


def _resolve_torch_dtype(name: str, *, kind: str) -> torch.dtype:
    """Map a friendly dtype name (``bf16``/``bfloat16``/``fp16``/``half``/``fp32``)
    to the corresponding ``torch.dtype``. ``kind`` is only used in the error
    message ("scale" / "HP" / etc.) so the caller's intent surfaces in the
    failure mode.
    """
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(
        f"Unsupported {kind} dtype: {name}. Expected bf16/fp16/fp32."
    )


def resolve_scale_dtype(name: str) -> torch.dtype:
    return _resolve_torch_dtype(name, kind="scale")


def resolve_hp_dtype(name: str) -> torch.dtype:
    return _resolve_torch_dtype(name, kind="HP")


def compute_page_geometry(hp_dtype: torch.dtype) -> Tuple[int, int]:
    """Return ``(N_H, N_Q)`` for int2 + ``hp_dtype``.

    ``N_Q`` is the int2 page size used by the paged quant allocator (and
    ``--page-size``). ``N_H`` is retained as ``1`` for documentation and for
    legacy callers, but no longer carries the LCM byte-equivalence invariant —
    HP and quant arenas are decoupled allocations under the slab design.
    """
    hp_itemsize = torch.empty(0, dtype=hp_dtype).element_size()
    return 1, 4 * hp_itemsize


def compute_recent_ring_size(hp_recent_tokens: int, n_q: int) -> int:
    # Max HP-recent occupancy between flushes is hp_recent + (N_Q - 1); the
    # ring reuses slots after the oldest N_Q have been demoted to quant.
    return int(hp_recent_tokens) + int(n_q) - 1


class UnifiedInt2HPKVPool(KVCache):
    """Unified HP + int2 MHA KV cache.

    The pool exposes:
      * ``k_buffer[l]``, ``v_buffer[l]``           – quant (int2 packed uint8) views
      * ``hp_k_buffer[l]``, ``hp_v_buffer[l]``     – HP (``hp_dtype``) views
      * ``k_scales_zeros[l]``, ``v_scales_zeros[l]`` – per-group scales+zeros in
        ``scale_dtype`` (bf16/fp16/fp32)

    The quant and HP views alias the same byte arena. Callers must treat a
    physical page as homogeneous (either tier) at any given time; this invariant
    is enforced by the ``UnifiedInt2HPKVAllocator`` that hands out slot ids into
    these views.
    """

    def __init__(
        self,
        num_quant_pages: int,
        hp_dtype: torch.dtype,
        hp_prefix_tokens: int,
        hp_recent_tokens: int,
        dtype: str,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_req_slots: int,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        model_dtype: Optional[torch.dtype] = None,
        kv_cache_quant_group_size: Optional[int] = None,
        scale_dtype: torch.dtype = torch.bfloat16,
        num_hp_prefix_slots: int = 0,
    ):
        assert dtype == "int2", (
            "UnifiedInt2HPKVPool supports only int2 quant tier; got %s" % dtype
        )
        # Work around KVCache.__init__ dtype validation: it stores ``dtype`` as
        # a string and sets ``store_dtype=torch.uint8`` for int2.
        super().__init__(
            size=num_quant_pages,  # used by base class for sizing heuristics only
            page_size=1,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            model_dtype=model_dtype,
        )

        self.num_quant_pages = int(num_quant_pages)
        self.hp_dtype = hp_dtype
        self.scale_dtype = scale_dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        self.hp_prefix_tokens = int(hp_prefix_tokens)
        self.hp_recent_tokens = int(hp_recent_tokens)
        self.kv_cache_quant_group_size = kv_cache_quant_group_size
        self.attention_aware_k_mode = os.environ.get(
            "SGLANG_OSCAR_AA_K_MODE", "off"
        ).strip().lower()
        layout = resolve_oscar_k_layout(
            self.attention_aware_k_mode,
            head_dim=self.head_dim,
            base_group_size=self.kv_cache_quant_group_size,
        )
        self.attention_aware_k_mixed = bool(layout["mixed_int1"])
        self.product_vq_k = bool(layout["product_vq"])
        self.split_k2_separate = bool(layout["split_k2_separate"])
        self.k_compressed_subspace = self.attention_aware_k_mixed or self.product_vq_k
        self.k_high_dim = int(os.environ.get("SGLANG_OSCAR_AA_K_HIGH_DIM", "64"))
        self.k_low_dim = self.head_dim - self.k_high_dim
        self.k_vq_vector_dim = int(
            os.environ.get("SGLANG_OSCAR_K_VQ_VECTOR_DIM", "8")
        )
        self.k_vq_codes = int(os.environ.get("SGLANG_OSCAR_K_VQ_CODES", "16"))
        self.k_vq_groups = 0
        self.k_vq_index_bits = 0
        self.k_vq_index_bytes = 0
        if self.k_compressed_subspace:
            if self.k_high_dim < 0 or self.k_high_dim >= self.head_dim:
                raise ValueError("compressed K high dimension must be in [0, head_dim)")
            if self.k_high_dim % 4 or self.k_low_dim % 8:
                raise ValueError(
                    "compressed K requires INT2 high_dim % 4 and low_dim % 8"
                )
        if self.product_vq_k and (
            self.head_dim != 128 or self.k_high_dim != 64 or self.head_num != 8
        ):
            raise ValueError(
                "product VQ currently requires head_dim=128, high_dim=64 and 8 KV heads"
            )
        if self.product_vq_k:
            if (
                self.k_vq_vector_dim <= 0
                or self.k_low_dim % self.k_vq_vector_dim
                or self.k_vq_codes not in (16, 256)
            ):
                raise ValueError("invalid K VQ vector_dim/codes")
            self.k_vq_groups = self.k_low_dim // self.k_vq_vector_dim
            self.k_vq_index_bits = 4 if self.k_vq_codes == 16 else 8
            if self.k_vq_groups * self.k_vq_index_bits % 8:
                raise ValueError("K VQ indices must occupy whole bytes")
            self.k_vq_index_bytes = (
                self.k_vq_groups * self.k_vq_index_bits // 8
            )
        self.k_vq_codebooks: Optional[torch.Tensor] = None
        self.k_vq_distance_weights: Optional[torch.Tensor] = None
        self.v_mixed_mode = os.environ.get(
            "SGLANG_OSCAR_V_MIXED_MODE", "off"
        ).strip().lower()
        if self.v_mixed_mode not in ("off", "mixed_int1", "product_vq"):
            raise ValueError(
                "SGLANG_OSCAR_V_MIXED_MODE must be off/mixed_int1/product_vq, got "
                f"{self.v_mixed_mode!r}"
            )
        self.product_vq_v = self.v_mixed_mode == "product_vq"
        self.v_compressed_subspace = self.v_mixed_mode in ("mixed_int1", "product_vq")
        self.v_high_dim = int(os.environ.get("SGLANG_OSCAR_V_HIGH_DIM", "64"))
        self.v_low_dim = self.v_head_dim - self.v_high_dim
        self.v_vq_vector_dim = int(
            os.environ.get("SGLANG_OSCAR_V_VQ_VECTOR_DIM", "8")
        )
        self.v_vq_codes = int(os.environ.get("SGLANG_OSCAR_V_VQ_CODES", "16"))
        self.v_vq_groups = 0
        self.v_vq_index_bits = 0
        self.v_vq_index_bytes = 0
        if self.v_compressed_subspace:
            if self.v_high_dim < 0 or self.v_high_dim >= self.v_head_dim:
                raise ValueError("compressed V high dimension must be in [0, v_head_dim)")
            if self.v_high_dim % 4 or self.v_low_dim % 8:
                raise ValueError(
                    "compressed V requires INT2 high_dim % 4 and low_dim % 8"
                )
        if self.product_vq_v and (
            self.v_head_dim != 128 or self.v_high_dim != 64 or self.head_num != 8
        ):
            raise ValueError(
                "product VQ V currently requires v_head_dim=128, high_dim=64 "
                "and 8 KV heads"
            )
        if self.product_vq_v:
            if (
                self.v_vq_vector_dim <= 0
                or self.v_low_dim % self.v_vq_vector_dim
                or self.v_vq_codes not in (16, 256)
            ):
                raise ValueError("invalid V VQ vector_dim/codes")
            self.v_vq_groups = self.v_low_dim // self.v_vq_vector_dim
            self.v_vq_index_bits = 4 if self.v_vq_codes == 16 else 8
            if self.v_vq_groups * self.v_vq_index_bits % 8:
                raise ValueError("V VQ indices must occupy whole bytes")
            self.v_vq_index_bytes = (
                self.v_vq_groups * self.v_vq_index_bits // 8
            )
        self.v_vq_codebooks: Optional[torch.Tensor] = None
        self.v_vq_distance_weights: Optional[torch.Tensor] = None
        compact_pure_int1 = envs.SGLANG_OSCAR_PURE_INT1_COMPACT_SCALES.get()
        self.pure_int1_k = (
            compact_pure_int1
            and self.attention_aware_k_mixed
            and self.k_high_dim == 0
        )
        self.pure_int1_v = (
            compact_pure_int1
            and self.v_mixed_mode == "mixed_int1"
            and self.v_high_dim == 0
        )

        self.anchor_capacity = envs.SGLANG_OSCAR_FP16_ANCHOR_TOKENS.get()
        supported_anchor_layouts = {
            0: None,
            28: (64, 256),
            56: (16, 248),
            120: (16, 64),
        }
        if self.anchor_capacity not in supported_anchor_layouts:
            raise ValueError(
                "SGLANG_OSCAR_FP16_ANCHOR_TOKENS currently supports only "
                "0 (disabled), 28 (64/256), 56 (16/248), or 120 (16/64) in ERAKV_V2"
            )
        self.anchor_enabled = self.anchor_capacity > 0
        if self.anchor_enabled:
            # Keep this implementation intentionally scoped to the experiment
            # it was built for.  In particular, the anchor reconstruction path
            # below assumes the whole middle row uses the mixed-INT1 codec.
            expected_windows = supported_anchor_layouts[self.anchor_capacity]
            if (self.hp_prefix_tokens, self.hp_recent_tokens) != expected_windows:
                raise ValueError(
                    f"{self.anchor_capacity} FP16 anchors require OSCAR "
                    f"prefix={expected_windows[0]} and recent={expected_windows[1]}"
                )
            if not (
                self.attention_aware_k_mixed
                and self.k_high_dim == 0
                and self.v_compressed_subspace
                and not self.product_vq_v
                and self.v_high_dim == 0
            ):
                raise ValueError(
                    "FP16 anchors require pure INT1 K/V middle rows "
                    "(K/V high_dim=0, mixed_int1 codecs)"
                )

        self.N_H, self.N_Q = compute_page_geometry(hp_dtype)
        self.hp_recent_ring_size = compute_recent_ring_size(
            self.hp_recent_tokens, self.N_Q
        )
        if num_hp_prefix_slots > 0 and num_hp_prefix_slots % self.N_Q != 0:
            num_hp_prefix_slots = (
                (num_hp_prefix_slots + self.N_Q - 1) // self.N_Q * self.N_Q
            )
        self.num_hp_prefix_slots = int(num_hp_prefix_slots)
        self.slab_size = self.hp_recent_ring_size  # back-compat alias
        self._hp_offset = self.num_quant_pages * self.N_Q
        self._hp_recent_base = self.num_hp_prefix_slots

        # Window sizes must be N_Q-aligned so radix tree (page_size=N_Q) and
        # the flush kernel land on page boundaries.
        if self.hp_prefix_tokens % self.N_Q != 0:
            raise ValueError(
                f"SGLANG_MIXED_KV_PREFIX_TOKENS ({self.hp_prefix_tokens}) "
                f"must be a multiple of N_Q ({self.N_Q})."
            )
        if self.hp_recent_tokens % self.N_Q != 0:
            raise ValueError(
                f"SGLANG_MIXED_KV_RECENT_TOKENS ({self.hp_recent_tokens}) "
                f"must be a multiple of N_Q ({self.N_Q})."
            )
        # Flush every N_Q decode steps demotes exactly N_Q HP-recent slots
        # into one quant page. Per-request counter (initialized at admission
        # to (hp_recent+N_Q-1)-H_0) keeps every flush whole-page.
        self.flush_interval = self.N_Q
        self.max_req_slots = int(max_req_slots)
        self._flush_counter = torch.zeros(
            (self.max_req_slots,), dtype=torch.int32, device=self.device
        )
        self._next_slab_offset = torch.zeros(
            (self.max_req_slots,), dtype=torch.int32, device=self.device
        )

        # Forward-done event stashed each iteration; consumed by
        # ``wait_pending_forward`` inside the flush apply phase.
        self._pending_forward_done = None

        # Grouping for quantization.
        self.k_quant_group_size, self.k_num_scale_groups = self._resolve_quant_grouping(
            self.head_dim, "K"
        )
        self.v_quant_group_size, self.v_num_scale_groups = self._resolve_quant_grouping(
            self.v_head_dim, "V"
        )
        if self.split_k2_separate:
            if self.head_dim != 128 or self.k_high_dim != 64:
                raise ValueError(
                    "split_k2_separate currently requires head_dim=128 and high_dim=64"
                )
            self.k_quant_group_size = self.k_high_dim
            self.k_num_scale_groups = 2
        assert self.head_dim % 4 == 0, (
            f"head_dim={self.head_dim} must be divisible by 4 for int2 packing"
        )
        assert self.v_head_dim % 4 == 0, (
            f"v_head_dim={self.v_head_dim} must be divisible by 4 for int2 packing"
        )

        self._create_arenas()

        # Cached attributes used by the rest of the stack.
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = None
        self.row_dim = self.head_num * self.head_dim  # for store_cache helpers
        self.same_kv_dim = self.head_dim == self.v_head_dim

        # Oscar rotation + clip. Per-layer orthogonal matrices [head_dim,
        # head_dim] / [v_head_dim, v_head_dim] are loaded in ``hp_dtype`` so
        # the ``rows @ R`` pre-pass and ``result @ R.T`` inverse are plain
        # bf16 GEMMs.
        self._oscar_cfg: OscarRotationConfig = load_oscar_rotation_config()
        self._k_clip_ratio: float = self._oscar_cfg.k_clip_ratio
        self._v_clip_ratio: float = self._oscar_cfg.v_clip_ratio
        self._lloyd_max: bool = envs.SGLANG_LLOYD_MAX.get()
        self._R_k: torch.Tensor = load_oscar_rotations(
            self._oscar_cfg.k_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=self.head_dim,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
        )
        self._R_v: torch.Tensor = load_oscar_rotations(
            self._oscar_cfg.v_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=self.v_head_dim,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
        )
        self.anchor_fisher_k: Optional[torch.Tensor] = None
        self.anchor_fisher_v: Optional[torch.Tensor] = None
        fisher_path = envs.SGLANG_OSCAR_ANCHOR_FISHER_PATH.get().strip()
        if fisher_path:
            if not self.anchor_enabled:
                raise ValueError(
                    "SGLANG_OSCAR_ANCHOR_FISHER_PATH requires "
                    "SGLANG_OSCAR_FP16_ANCHOR_TOKENS > 0"
                )
            self.anchor_fisher_k, self.anchor_fisher_v = (
                self._load_anchor_fisher_checkpoint(fisher_path)
            )
        if self.product_vq_k:
            self.k_vq_codebooks, self.k_vq_distance_weights = (
                self._load_product_vq_checkpoint("k")
            )
        if self.product_vq_v:
            self.v_vq_codebooks, self.v_vq_distance_weights = (
                self._load_product_vq_checkpoint("v")
            )
        logger.info(
            "UnifiedInt2HPKVPool: Oscar rotation enabled (k_clip=%.4f v_clip=%.4f lloyd_max=%s)",
            self._k_clip_ratio,
            self._v_clip_ratio,
            self._lloyd_max,
        )
        if self.attention_aware_k_mixed:
            logger.info(
                "UnifiedInt2HPKVPool: attention-aware mixed K enabled "
                "(mode=%s high=%d@INT2 low=%d@INT1 packed_bytes=%d "
                "low_scale=bf16 affine_scale_zero_bytes=%d)",
                self.attention_aware_k_mode,
                self.k_high_dim,
                self.k_low_dim,
                self.k_high_dim // 4 + self.k_low_dim // 8,
                self.k_scales_zeros[0].shape[-1]
                * self.k_scales_zeros[0].element_size(),
            )
        elif self.product_vq_k:
            logger.info(
                "UnifiedInt2HPKVPool: product-VQ K enabled "
                "(high=%d@INT2 low=%d@VQ D=%d K=%d packed_bytes=%d low_scale=bf16 "
                "codebook_bytes=%d distance_weight_bytes=%d)",
                self.k_high_dim,
                self.k_low_dim,
                self.k_vq_vector_dim,
                self.k_vq_codes,
                self.k_high_dim // 4 + self.k_vq_index_bytes,
                self.k_vq_codebooks.numel() * self.k_vq_codebooks.element_size(),
                self.k_vq_distance_weights.numel()
                * self.k_vq_distance_weights.element_size(),
            )
        elif self.split_k2_separate:
            logger.info(
                "UnifiedInt2HPKVPool: split-K2 separate K scales enabled "
                "(high=%d low=%d K_groups=2 V_groups=%d)",
                self.k_high_dim,
                self.k_low_dim,
                self.v_num_scale_groups,
            )
        if self.product_vq_v:
            logger.info(
                "UnifiedInt2HPKVPool: product-VQ V enabled "
                "(high=%d@INT2 low=%d@VQ D=%d K=%d packed_bytes=%d low_scale=bf16 "
                "codebook_bytes=%d distance_weight_bytes=%d)",
                self.v_high_dim,
                self.v_low_dim,
                self.v_vq_vector_dim,
                self.v_vq_codes,
                self.v_high_dim // 4 + self.v_vq_index_bytes,
                self.v_vq_codebooks.numel() * self.v_vq_codebooks.element_size(),
                self.v_vq_distance_weights.numel()
                * self.v_vq_distance_weights.element_size(),
            )
        elif self.v_compressed_subspace:
            logger.info(
                "UnifiedInt2HPKVPool: mixed V enabled "
                "(high=%d@INT2 low=%d@INT1 packed_bytes=%d "
                "low_scale=bf16 affine_scale_zero_bytes=%d)",
                self.v_high_dim,
                self.v_low_dim,
                self.v_high_dim // 4 + self.v_low_dim // 8,
                self.v_scales_zeros[0].shape[-1]
                * self.v_scales_zeros[0].element_size(),
            )

        hp_total_slots = (
            self.num_hp_prefix_slots
            + self.max_req_slots * self.hp_recent_ring_size
        )
        self._finalize_allocation_log(hp_total_slots)
        hp_itemsize = torch.empty(0, dtype=self.hp_dtype).element_size()
        hp_bytes = (
            hp_total_slots
            * self.layer_num
            * self.head_num
            * (self.head_dim + self.v_head_dim)
            * hp_itemsize
        )
        logger.info(
            "UnifiedInt2HPKVPool: HP arena reserves %.2f GB "
            "(hp_prefix_pool_slots=%d, max_req_slots=%d, recent_ring=%d "
            "= R=%d + N_Q-1=%d, P=%d, layers=%d, head_num=%d, "
            "head_dim+v_head_dim=%d, hp_dtype=%s)",
            hp_bytes / GB,
            self.num_hp_prefix_slots,
            self.max_req_slots,
            self.hp_recent_ring_size,
            self.hp_recent_tokens,
            self.N_Q - 1,
            self.hp_prefix_tokens,
            self.layer_num,
            self.head_num,
            self.head_dim + self.v_head_dim,
            str(self.hp_dtype),
        )
        if self.anchor_enabled:
            anchor_bytes = sum(
                get_tensor_size_bytes(t)
                for tensors in (
                    self.anchor_k_buffer,
                    self.anchor_v_buffer,
                    self.anchor_scores,
                    self.anchor_positions,
                    self.anchor_quant_slots,
                    self.anchor_counts,
                )
                for t in tensors
            )
            logger.info(
                "UnifiedInt2HPKVPool: ERAKV FP16 anchors enabled "
                "(capacity=%d per request/layer/KV-head, %.2f MiB per TP rank)",
                self.anchor_capacity,
                anchor_bytes / (1024 * 1024),
            )

    # -- Configuration accessors -------------------------------------------

    def mixed_kv_enabled(self) -> bool:
        return True

    def _load_anchor_fisher_checkpoint(
        self, checkpoint_path: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load calibrated diagonal Fisher weights in OSCAR coordinates."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"anchor Fisher checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported anchor Fisher format_version")
        checkpoint_model = checkpoint.get("model")
        if not isinstance(checkpoint_model, str) or not checkpoint_model.strip():
            raise ValueError("anchor Fisher checkpoint is missing its model identifier")
        if checkpoint.get("coordinate_system") != "oscar_rotated_post_rope":
            raise ValueError("anchor Fisher checkpoint has wrong coordinate system")
        expected_hashes = {
            "k_rotation_sha256": _sha256_file(self._oscar_cfg.k_rotation_path),
            "v_rotation_sha256": _sha256_file(self._oscar_cfg.v_rotation_path),
        }
        for key, expected in expected_hashes.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"anchor Fisher checkpoint {key} mismatch")

        k = checkpoint.get("k_weights")
        v = checkpoint.get("v_weights")
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise ValueError("anchor Fisher checkpoint is missing K/V weights")
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("anchor Fisher weights must have [layers, heads, dim]")
        if k.shape[0] < self.end_layer or v.shape[0] < self.end_layer:
            raise ValueError("anchor Fisher checkpoint does not cover all local layers")
        k = k[self.start_layer : self.end_layer]
        v = v[self.start_layer : self.end_layer]

        if model_parallel_is_initialized():
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
        else:
            tp_rank = 0
            tp_size = 1
        expected_global_heads = self.head_num * tp_size
        if k.shape[1] != expected_global_heads or v.shape[1] != expected_global_heads:
            raise ValueError(
                "anchor Fisher global KV-head count mismatch: "
                f"checkpoint K/V={k.shape[1]}/{v.shape[1]}, "
                f"runtime local={self.head_num}, TP={tp_size}"
            )
        head_start = tp_rank * self.head_num
        head_end = head_start + self.head_num
        k = k[:, head_start:head_end]
        v = v[:, head_start:head_end]
        if tuple(k.shape[2:]) != (self.head_dim,) or tuple(v.shape[2:]) != (
            self.v_head_dim,
        ):
            raise ValueError("anchor Fisher head-dimension mismatch")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise ValueError("anchor Fisher weights contain non-finite values")
        if (k < 0).any() or (v < 0).any() or k.sum() <= 0 or v.sum() <= 0:
            raise ValueError("anchor Fisher weights must be nonnegative and nonzero")
        logger.info(
            "UnifiedInt2HPKVPool: calibrated diagonal-Fisher anchor selector "
            "enabled (model=%s path=%s sha256=%s)",
            checkpoint_model,
            path,
            _sha256_file(str(path)),
        )
        return (
            k.to(device=self.device, dtype=torch.float32).contiguous(),
            v.to(device=self.device, dtype=torch.float32).contiguous(),
        )

    def _load_product_vq_checkpoint(
        self, side: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if side not in ("k", "v"):
            raise ValueError(f"invalid VQ checkpoint side: {side!r}")
        env_name = (
            "SGLANG_OSCAR_VQ_CODEBOOK_PATH"
            if side == "k"
            else "SGLANG_OSCAR_V_VQ_CODEBOOK_PATH"
        )
        checkpoint_path = os.environ.get(env_name, "").strip()
        if not checkpoint_path:
            raise ValueError(f"product_vq {side.upper()} mode requires {env_name}")
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"VQ codebook checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported VQ checkpoint format_version")
        if checkpoint.get("model") != "Qwen3-4B-Instruct-2507":
            raise ValueError("VQ checkpoint model does not match Qwen3-4B-Instruct-2507")
        if checkpoint.get("codec") not in (
            "low64_product_vq_8d_16codes_int4",
            "low64_product_vq_global_config",
        ):
            raise ValueError("unsupported VQ checkpoint codec")
        groups = self.k_vq_groups if side == "k" else self.v_vq_groups
        codes = self.k_vq_codes if side == "k" else self.v_vq_codes
        vector_dim = (
            self.k_vq_vector_dim if side == "k" else self.v_vq_vector_dim
        )
        contract = checkpoint.get("shape_contract", {})
        expected_contract = {
            "head_dim": 128,
            "high_dim": 64,
            "kv_heads": 8,
            "groups": groups,
            "codes": codes,
            "vector_dim": vector_dim,
            "codebook_dtype": "bfloat16",
            "distance_weight_dtype": "bfloat16",
        }
        if contract != expected_contract:
            raise ValueError(f"VQ shape contract mismatch: {contract!r}")
        rotation_path = (
            self._oscar_cfg.k_rotation_path
            if side == "k"
            else self._oscar_cfg.v_rotation_path
        )
        if checkpoint.get("rotation_sha256") != _sha256_file(rotation_path):
            raise ValueError(f"VQ checkpoint {side.upper()} rotation SHA256 mismatch")
        layers = checkpoint.get("layers", {})
        layer_ids = list(range(self.start_layer, self.end_layer))
        codebooks = []
        weights = []
        for layer_id in layer_ids:
            entry = layers.get(layer_id, layers.get(str(layer_id)))
            if entry is None:
                raise ValueError(f"VQ checkpoint missing layer {layer_id}")
            codebook = entry["codebooks"]
            distance_weight = entry["distance_weights"]
            if tuple(codebook.shape) != (8, groups, codes, vector_dim):
                raise ValueError(f"invalid VQ codebook shape at layer {layer_id}")
            if tuple(distance_weight.shape) != (8, groups, vector_dim):
                raise ValueError(f"invalid VQ distance-weight shape at layer {layer_id}")
            codebooks.append(codebook.to(torch.bfloat16))
            weights.append(distance_weight.to(torch.bfloat16))
        return (
            torch.stack(codebooks)
            .to(device=self.device, dtype=torch.bfloat16)
            .contiguous(),
            torch.stack(weights)
            .to(device=self.device, dtype=torch.bfloat16)
            .contiguous(),
        )

    def stash_pending_forward(self, event) -> None:
        """Record the most recent forward-stream completion event.

        Called once per iteration from the scheduler. The event is consumed
        by :meth:`wait_pending_forward` at the apply boundary inside
        ``_alloc_for_decode_mixed``.
        """
        self._pending_forward_done = event

    def wait_pending_forward(self) -> None:
        """Order the current stream after the stashed forward-done event.

        Must be issued *before* the apply phase of the flush
        (``gpu_flush_int2_apply``), since the remap kernel writes
        ``req_to_token`` at positions the previous forward's attention is
        concurrently reading. Pre-apply work (allocator free, plan kernel)
        runs ahead of this wait so its host syncs don't block on the
        previous forward.
        """
        if self._pending_forward_done is None:
            return
        torch.cuda.current_stream().wait_event(self._pending_forward_done)
        self._pending_forward_done = None

    @property
    def hp_global_offset(self) -> int:
        return self._hp_offset

    @property
    def hp_size(self) -> int:
        return (
            self.num_hp_prefix_slots
            + self.max_req_slots * self.hp_recent_ring_size
        )

    @property
    def quant_size(self) -> int:
        return self.num_quant_pages * self.N_Q

    @property
    def hp_prefix_pool_slots(self) -> int:
        return self.num_hp_prefix_slots

    @property
    def hp_recent_base(self) -> int:
        """First HP-buffer index reserved for per-req recent slabs."""
        return self._hp_recent_base

    def release_req_slab(self, req_pool_idx) -> None:
        # Reset the per-req HP-recent cursor and flush counter so the next
        # request taking over ``req_pool_idx`` starts clean.
        if isinstance(req_pool_idx, torch.Tensor):
            idx = req_pool_idx.to(self._next_slab_offset.device).to(torch.int64)
            if idx.numel() == 0:
                return
            self._next_slab_offset[idx] = 0
            self._flush_counter[idx] = 0
        else:
            i = int(req_pool_idx)
            self._next_slab_offset[i] = 0
            self._flush_counter[i] = 0
            idx = i
        if self.anchor_enabled:
            for layer_idx in range(self.layer_num):
                self.anchor_counts[layer_idx][idx] = 0
                self.anchor_scores[layer_idx][idx] = -float("inf")
                self.anchor_positions[layer_idx][idx] = -1
                self.anchor_quant_slots[layer_idx][idx] = -1

    def _update_fp16_anchors_from_rows(
        self,
        layer_idx: int,
        quant_slots: torch.Tensor,
        exact_k: torch.Tensor,
        exact_v: torch.Tensor,
        req_indices: torch.Tensor,
        positions: torch.Tensor,
        req_values=None,
    ) -> None:
        """Admit quantized middle rows into the bounded ERAKV directory.

        This is a correctness-first eager implementation for the fixed OSCAR
        experiment.  Selection is independent for every request/KV head and
        uses ERAKV's FP32 joint relative K/V reconstruction error.  Stable
        ties retain the earlier logical position.
        """
        if not self.anchor_enabled or quant_slots.numel() == 0:
            return
        quant_slots = quant_slots.to(device=self.device, dtype=torch.int64)
        req_indices = req_indices.to(device=self.device, dtype=torch.int64)
        positions = positions.to(device=self.device, dtype=torch.int64)
        exact_k = exact_k.reshape(-1, self.head_num, self.head_dim)
        exact_v = exact_v.reshape(-1, self.head_num, self.v_head_dim)
        reconstructed_k = dequantize_attention_mixed_k_torch(
            self.k_buffer[layer_idx][quant_slots],
            self.k_scales_zeros[layer_idx][quant_slots],
            self.k_low_scales[layer_idx][quant_slots],
            0,
            torch.float32,
        )
        reconstructed_v = dequantize_attention_mixed_k_torch(
            self.v_buffer[layer_idx][quant_slots],
            self.v_scales_zeros[layer_idx][quant_slots],
            self.v_low_scales[layer_idx][quant_slots],
            0,
            torch.float32,
        )
        k32 = exact_k.float()
        v32 = exact_v.float()
        k_delta2 = (k32 - reconstructed_k).square()
        v_delta2 = (v32 - reconstructed_v).square()
        if self.anchor_fisher_k is not None:
            scores = (
                k_delta2 * self.anchor_fisher_k[layer_idx][None]
            ).sum(dim=-1) + (
                v_delta2 * self.anchor_fisher_v[layer_idx][None]
            ).sum(dim=-1)
        else:
            k_error = k_delta2.sum(dim=-1)
            v_error = v_delta2.sum(dim=-1)
            k_norm = k32.square().sum(dim=-1).clamp_min(2.0**-24)
            v_norm = v32.square().sum(dim=-1).clamp_min(2.0**-24)
            scores = 0.5 * (k_error / k_norm + v_error / v_norm)

        # Prefill calls this once per layer, and decode calls it once per
        # N_Q-token rollover.  Host loops keep the selector simple and exact;
        # all large tensor work remains on GPU.
        if req_values is None:
            req_values = torch.unique(req_indices).tolist()
        for req in req_values:
            row_ids = torch.nonzero(req_indices == req, as_tuple=False).flatten()
            if row_ids.numel() == 0:
                continue
            old_counts = self.anchor_counts[layer_idx][req].to(torch.int64)
            old_valid = (
                torch.arange(self.anchor_capacity, device=self.device)[None, :]
                < old_counts[:, None]
            )
            old_scores = self.anchor_scores[layer_idx][req].clone().masked_fill(
                ~old_valid, -float("inf")
            )
            old_positions = self.anchor_positions[layer_idx][req].to(
                torch.int64
            ).clone()
            old_slots = self.anchor_quant_slots[layer_idx][req].clone()
            old_k = self.anchor_k_buffer[layer_idx][req].clone()
            old_v = self.anchor_v_buffer[layer_idx][req].clone()
            new_scores = scores[row_ids].transpose(0, 1)
            new_positions = positions[row_ids][None, :].expand(self.head_num, -1)
            new_slots = quant_slots[row_ids][None, :].expand(self.head_num, -1)
            new_k = exact_k[row_ids].permute(1, 0, 2).to(torch.float16)
            new_v = exact_v[row_ids].permute(1, 0, 2).to(torch.float16)
            candidate_scores = torch.cat([old_scores, new_scores], dim=1)
            candidate_positions = torch.cat(
                [old_positions, new_positions], dim=1
            )
            candidate_slots = torch.cat([old_slots, new_slots], dim=1)
            candidate_k = torch.cat([old_k, new_k], dim=1)
            candidate_v = torch.cat([old_v, new_v], dim=1)

            # Stable lexicographic order: descending score, then ascending
            # absolute position.  This matches ERAKV's frozen rule.
            position_order = torch.argsort(
                candidate_positions, dim=1, stable=True
            )
            scores_by_position = torch.gather(
                candidate_scores, 1, position_order
            )
            score_order = torch.argsort(
                scores_by_position, dim=1, descending=True, stable=True
            )
            selected = torch.gather(position_order, 1, score_order)[
                :, : self.anchor_capacity
            ]
            gather_vec = selected[..., None]
            self.anchor_scores[layer_idx][req] = torch.gather(
                candidate_scores, 1, selected
            )
            self.anchor_positions[layer_idx][req] = torch.gather(
                candidate_positions, 1, selected
            ).to(torch.int32)
            self.anchor_quant_slots[layer_idx][req] = torch.gather(
                candidate_slots, 1, selected
            )
            self.anchor_k_buffer[layer_idx][req] = torch.gather(
                candidate_k, 1, gather_vec.expand(-1, -1, self.head_dim)
            )
            self.anchor_v_buffer[layer_idx][req] = torch.gather(
                candidate_v, 1, gather_vec.expand(-1, -1, self.v_head_dim)
            )
            self.anchor_counts[layer_idx][req] = torch.clamp(
                old_counts + row_ids.numel(), max=self.anchor_capacity
            ).to(torch.int16)

    def update_fp16_anchors_from_flush(self, plan, req_pool_indices) -> None:
        """Observe exact HP rows immediately after their INT1 rollover."""
        if not self.anchor_enabled:
            return
        if self.anchor_capacity not in (28, 56, 120):
            raise RuntimeError("ERAKV_V2 invariant violated: unsupported anchor capacity")
        for layer_idx in range(self.layer_num):
            k = self.anchor_k_buffer[layer_idx]
            v = self.anchor_v_buffer[layer_idx]
            directory = self.anchor_scores[layer_idx]
            counts = self.anchor_counts[layer_idx]
            use_fisher = self.anchor_fisher_k is not None
            fisher_k = (
                self.anchor_fisher_k[layer_idx]
                if use_fisher
                else self._R_k[layer_idx]
            )
            fisher_v = (
                self.anchor_fisher_v[layer_idx]
                if use_fisher
                else self._R_v[layer_idx]
            )
            _update_fp16_anchors_from_flush_kernel[
                (plan.bs, self.head_num)
            ](
                self.hp_k_buffer[layer_idx],
                self.hp_v_buffer[layer_idx],
                self.k_buffer[layer_idx],
                self.v_buffer[layer_idx],
                self.k_low_scales[layer_idx],
                self.v_low_scales[layer_idx],
                fisher_k,
                fisher_v,
                plan.src_hp_slot,
                plan.dst_quant_slots,
                plan.flush_pos,
                plan.valid_mask,
                req_pool_indices,
                k,
                v,
                directory,
                self.anchor_positions[layer_idx],
                self.anchor_quant_slots[layer_idx],
                counts,
                self.hp_k_buffer[layer_idx].stride(0),
                self.hp_k_buffer[layer_idx].stride(1),
                self.hp_k_buffer[layer_idx].stride(2),
                self.hp_v_buffer[layer_idx].stride(0),
                self.hp_v_buffer[layer_idx].stride(1),
                self.hp_v_buffer[layer_idx].stride(2),
                self.k_buffer[layer_idx].stride(0),
                self.k_buffer[layer_idx].stride(1),
                self.k_buffer[layer_idx].stride(2),
                self.v_buffer[layer_idx].stride(0),
                self.v_buffer[layer_idx].stride(1),
                self.v_buffer[layer_idx].stride(2),
                self.k_low_scales[layer_idx].stride(0),
                self.k_low_scales[layer_idx].stride(1),
                self.v_low_scales[layer_idx].stride(0),
                self.v_low_scales[layer_idx].stride(1),
                fisher_k.stride(0),
                fisher_k.stride(1),
                fisher_v.stride(0),
                fisher_v.stride(1),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                directory.stride(0), directory.stride(1), directory.stride(2),
                counts.stride(0), counts.stride(1),
                NUM_HEADS=self.head_num,
                HEAD_DIM=self.head_dim,
                V_HEAD_DIM=self.v_head_dim,
                FLUSH_INTERVAL=plan.flush_interval,
                ANCHOR_CAPACITY=self.anchor_capacity,
                USE_FISHER=use_fisher,
                BLOCK_C=triton.next_power_of_2(self.anchor_capacity + plan.flush_interval),
                BLOCK_D=128,
                num_warps=8,
                num_stages=1,
            )

    def apply_fp16_anchor_correction(
        self,
        layer_id: int,
        q: torch.Tensor,
        o: torch.Tensor,
        base_lse: torch.Tensor,
        req_pool_indices: torch.Tensor,
        sm_scale: float,
    ) -> None:
        """Replace selected INT1 contributions with their exact FP16 K/V."""
        if not self.anchor_enabled:
            return
        layer_idx = self._layer_index(layer_id)
        req = req_pool_indices.to(device=self.device, dtype=torch.int64)
        counts = self.anchor_counts[layer_idx][req].to(torch.int64)
        slots = self.anchor_quant_slots[layer_idx][req]
        bs, kv_heads, capacity = slots.shape
        q_heads = q.shape[1]
        groups = q_heads // kv_heads
        if groups * kv_heads != q_heads:
            raise ValueError("query heads must be divisible by KV heads")

        valid = (
            torch.arange(capacity, device=self.device)[None, None, :]
            < counts[:, :, None]
        )
        safe_slots = slots.clamp_min(0)
        flat_slots = safe_slots.reshape(-1)
        flat_heads = (
            torch.arange(kv_heads, device=self.device)[None, :, None]
            .expand(bs, kv_heads, capacity)
            .reshape(-1)
        )
        approx_k = dequantize_attention_mixed_k_torch(
            self.k_buffer[layer_idx][flat_slots, flat_heads],
            self.k_scales_zeros[layer_idx][flat_slots, flat_heads],
            self.k_low_scales[layer_idx][flat_slots, flat_heads],
            0,
            torch.float32,
        ).reshape(bs, kv_heads, capacity, self.head_dim)
        approx_v = dequantize_attention_mixed_k_torch(
            self.v_buffer[layer_idx][flat_slots, flat_heads],
            self.v_scales_zeros[layer_idx][flat_slots, flat_heads],
            self.v_low_scales[layer_idx][flat_slots, flat_heads],
            0,
            torch.float32,
        ).reshape(bs, kv_heads, capacity, self.v_head_dim)
        exact_k = self.anchor_k_buffer[layer_idx][req].float()
        exact_v = self.anchor_v_buffer[layer_idx][req].float()
        q4 = q.float().reshape(bs, kv_heads, groups, self.head_dim)
        exact_logits = torch.einsum("bhgd,bhad->bhga", q4, exact_k) * sm_scale
        approx_logits = torch.einsum("bhgd,bhad->bhga", q4, approx_k) * sm_scale
        valid4 = valid[:, :, None, :]
        exact_logits = exact_logits.masked_fill(~valid4, -float("inf"))
        approx_logits = approx_logits.masked_fill(~valid4, -float("inf"))

        base_lse4 = base_lse.float().reshape(bs, kv_heads, groups)
        base_o4 = o.float().reshape(bs, kv_heads, groups, self.v_head_dim)
        maximum = torch.maximum(base_lse4, exact_logits.max(dim=-1).values)
        base_weight = torch.exp(base_lse4 - maximum)
        approx_weight = torch.exp(approx_logits - maximum[..., None])
        exact_weight = torch.exp(exact_logits - maximum[..., None])
        denominator = (
            base_weight
            - approx_weight.sum(dim=-1)
            + exact_weight.sum(dim=-1)
        ).clamp_min(2.0**-24)
        numerator = base_o4 * base_weight[..., None]
        numerator -= torch.einsum("bhga,bhad->bhgd", approx_weight, approx_v)
        numerator += torch.einsum("bhga,bhad->bhgd", exact_weight, exact_v)
        o.copy_((numerator / denominator[..., None]).reshape_as(o).to(o.dtype))

    def _resolve_quant_grouping(self, head_dim: int, tensor_name: str) -> tuple[int, int]:
        group_size = (
            head_dim
            if self.kv_cache_quant_group_size is None
            else self.kv_cache_quant_group_size
        )
        if group_size <= 0:
            raise ValueError(
                f"{tensor_name} kv_cache_quant_group_size must be positive, got {group_size}"
            )
        if head_dim % group_size != 0:
            raise ValueError(
                f"{tensor_name} head_dim ({head_dim}) must be divisible by "
                f"kv_cache_quant_group_size ({group_size})"
            )
        return group_size, head_dim // group_size

    # -- Arena construction ------------------------------------------------

    def _create_arenas(self):
        # HP arena layout: [shared prefix pool] [per-req recent slab 0]
        # [per-req recent slab 1] ... Quant arena is paged with N_Q slots
        # per page; scales/zeros are quant-only.
        hp_total_slots = (
            self.num_hp_prefix_slots
            + self.max_req_slots * self.hp_recent_ring_size
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                k_packed_dim = (
                    self.k_high_dim // 4
                    + (
                        self.k_vq_index_bytes
                        if self.product_vq_k
                        else self.k_low_dim // 8
                    )
                    if self.k_compressed_subspace
                    else self.head_dim // 4
                )
                self.k_buffer = [
                    torch.zeros(
                        (self.num_quant_pages * self.N_Q, self.head_num, k_packed_dim),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (
                            self.num_quant_pages * self.N_Q,
                            self.head_num,
                            self.v_high_dim // 4
                            + (
                                self.v_vq_index_bytes
                                if self.product_vq_v
                                else self.v_low_dim // 8
                            )
                            if self.v_compressed_subspace
                            else self.v_head_dim // 4,
                        ),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                k_scale_zero_dim = (
                    0 if self.pure_int1_k else 2 * self.k_num_scale_groups
                )
                v_scale_zero_dim = (
                    0 if self.pure_int1_v else 2 * self.v_num_scale_groups
                )
                self.k_scales_zeros = [
                    torch.zeros(
                        (
                            self.num_quant_pages * self.N_Q,
                            self.head_num,
                            k_scale_zero_dim,
                        ),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scales_zeros = [
                    torch.zeros(
                        (
                            self.num_quant_pages * self.N_Q,
                            self.head_num,
                            v_scale_zero_dim,
                        ),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.k_low_scales = (
                    [
                        torch.zeros(
                            (self.num_quant_pages * self.N_Q, self.head_num, 1),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    if self.k_compressed_subspace
                    else []
                )
                self.v_low_scales = (
                    [
                        torch.zeros(
                            (self.num_quant_pages * self.N_Q, self.head_num, 1),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    if self.v_compressed_subspace
                    else []
                )
                self.hp_k_buffer = [
                    torch.zeros(
                        (hp_total_slots, self.head_num, self.head_dim),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.hp_v_buffer = [
                    torch.zeros(
                        (hp_total_slots, self.head_num, self.v_head_dim),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                if self.anchor_enabled:
                    anchor_shape_k = (
                        self.max_req_slots,
                        self.head_num,
                        self.anchor_capacity,
                        self.head_dim,
                    )
                    anchor_shape_v = (
                        self.max_req_slots,
                        self.head_num,
                        self.anchor_capacity,
                        self.v_head_dim,
                    )
                    directory_shape = (
                        self.max_req_slots,
                        self.head_num,
                        self.anchor_capacity,
                    )
                    self.anchor_k_buffer = [
                        torch.zeros(anchor_shape_k, dtype=torch.float16, device=self.device)
                        for _ in range(self.layer_num)
                    ]
                    self.anchor_v_buffer = [
                        torch.zeros(anchor_shape_v, dtype=torch.float16, device=self.device)
                        for _ in range(self.layer_num)
                    ]
                    self.anchor_scores = [
                        torch.full(
                            directory_shape,
                            -float("inf"),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.anchor_positions = [
                        torch.full(
                            directory_shape, -1, dtype=torch.int32, device=self.device
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.anchor_quant_slots = [
                        torch.full(
                            directory_shape, -1, dtype=torch.int64, device=self.device
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.anchor_counts = [
                        torch.zeros(
                            (self.max_req_slots, self.head_num),
                            dtype=torch.int16,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                else:
                    self.anchor_k_buffer = []
                    self.anchor_v_buffer = []
                    self.anchor_scores = []
                    self.anchor_positions = []
                    self.anchor_quant_slots = []
                    self.anchor_counts = []

        # Cached device pointer arrays for the fused decode-flush kernel. The
        # flush kernel loops over layers inside the kernel, so it needs
        # per-layer base pointers as an int64 GPU tensor. Strides are identical
        # across layers (we enforce that below); the flush kernel reads the
        # single set at launch time via tl.constexpr.
        def _base_ptrs(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [t.data_ptr() for t in tensors],
                dtype=torch.int64,
                device=self.device,
            )

        self._flush_hp_k_ptrs = _base_ptrs(self.hp_k_buffer)
        self._flush_hp_v_ptrs = _base_ptrs(self.hp_v_buffer)
        self._flush_quant_k_ptrs = _base_ptrs(self.k_buffer)
        self._flush_quant_v_ptrs = _base_ptrs(self.v_buffer)
        self._flush_k_sz_ptrs = _base_ptrs(self.k_scales_zeros)
        self._flush_v_sz_ptrs = _base_ptrs(self.v_scales_zeros)
        self._flush_k_low_scale_ptrs = (
            _base_ptrs(self.k_low_scales) if self.k_compressed_subspace else None
        )
        self._flush_v_low_scale_ptrs = (
            _base_ptrs(self.v_low_scales) if self.v_compressed_subspace else None
        )

        # Strides (elements, not bytes) for each kind of buffer. The arenas are
        # all contiguous, so every layer shares the same strides; assert to be
        # safe.
        def _strides(t: torch.Tensor) -> tuple:
            return (int(t.stride(0)), int(t.stride(1)), int(t.stride(2)))

        hp_k_stride = _strides(self.hp_k_buffer[0])
        hp_v_stride = _strides(self.hp_v_buffer[0])
        q_k_stride = _strides(self.k_buffer[0])
        q_v_stride = _strides(self.v_buffer[0])
        k_sz_stride = _strides(self.k_scales_zeros[0])
        v_sz_stride = _strides(self.v_scales_zeros[0])
        k_low_scale_stride = (
            _strides(self.k_low_scales[0]) if self.k_compressed_subspace else None
        )
        v_low_scale_stride = (
            _strides(self.v_low_scales[0]) if self.v_compressed_subspace else None
        )
        for l in range(self.layer_num):
            assert _strides(self.hp_k_buffer[l]) == hp_k_stride
            assert _strides(self.hp_v_buffer[l]) == hp_v_stride
            assert _strides(self.k_buffer[l]) == q_k_stride
            assert _strides(self.v_buffer[l]) == q_v_stride
            assert _strides(self.k_scales_zeros[l]) == k_sz_stride
            assert _strides(self.v_scales_zeros[l]) == v_sz_stride
            if self.k_compressed_subspace:
                assert _strides(self.k_low_scales[l]) == k_low_scale_stride
            if self.v_compressed_subspace:
                assert _strides(self.v_low_scales[l]) == v_low_scale_stride

        self._flush_hp_k_stride = hp_k_stride
        self._flush_hp_v_stride = hp_v_stride
        self._flush_quant_k_stride = q_k_stride
        self._flush_quant_v_stride = q_v_stride
        self._flush_k_sz_stride = k_sz_stride
        self._flush_v_sz_stride = v_sz_stride
        self._flush_k_low_scale_stride = k_low_scale_stride
        self._flush_v_low_scale_stride = v_low_scale_stride

    # -- KVCache interface -------------------------------------------------

    def get_kv_size_bytes(self):
        k = sum(get_tensor_size_bytes(t) for t in self.k_buffer)
        k += sum(get_tensor_size_bytes(s) for s in self.k_scales_zeros)
        k += sum(get_tensor_size_bytes(s) for s in self.k_low_scales)
        if self.k_vq_codebooks is not None:
            k += get_tensor_size_bytes(self.k_vq_codebooks)
            k += get_tensor_size_bytes(self.k_vq_distance_weights)
        k += sum(get_tensor_size_bytes(t) for t in self.hp_k_buffer)
        k += sum(get_tensor_size_bytes(t) for t in self.anchor_k_buffer)
        k += sum(get_tensor_size_bytes(t) for t in self.anchor_scores)
        k += sum(get_tensor_size_bytes(t) for t in self.anchor_positions)
        k += sum(get_tensor_size_bytes(t) for t in self.anchor_quant_slots)
        k += sum(get_tensor_size_bytes(t) for t in self.anchor_counts)
        v = sum(get_tensor_size_bytes(t) for t in self.v_buffer)
        v += sum(get_tensor_size_bytes(s) for s in self.v_scales_zeros)
        v += sum(get_tensor_size_bytes(s) for s in self.v_low_scales)
        if self.v_vq_codebooks is not None:
            v += get_tensor_size_bytes(self.v_vq_codebooks)
            v += get_tensor_size_bytes(self.v_vq_distance_weights)
        v += sum(get_tensor_size_bytes(t) for t in self.hp_v_buffer)
        v += sum(get_tensor_size_bytes(t) for t in self.anchor_v_buffer)
        return k, v

    def _layer_index(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        # Triton backend asks for the quant view in the mixed path; HP view is
        # accessed via ``get_hp_key_buffer``.
        return self.k_buffer[self._layer_index(layer_id)]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._layer_index(layer_id)]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_raw_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[self._layer_index(layer_id)]

    def get_raw_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._layer_index(layer_id)]

    def get_key_scales_zeros(self, layer_id: int) -> torch.Tensor:
        return self.k_scales_zeros[self._layer_index(layer_id)]

    def get_value_scales_zeros(self, layer_id: int) -> torch.Tensor:
        return self.v_scales_zeros[self._layer_index(layer_id)]

    def get_key_low_scales(self, layer_id: int) -> Optional[torch.Tensor]:
        if not self.k_compressed_subspace:
            return None
        return self.k_low_scales[self._layer_index(layer_id)]

    def get_key_vq_codebook(self, layer_id: int) -> Optional[torch.Tensor]:
        if not self.product_vq_k:
            return None
        return self.k_vq_codebooks[self._layer_index(layer_id)]

    def get_value_low_scales(self, layer_id: int) -> Optional[torch.Tensor]:
        if not self.v_compressed_subspace:
            return None
        return self.v_low_scales[self._layer_index(layer_id)]

    def get_value_vq_codebook(self, layer_id: int) -> Optional[torch.Tensor]:
        if not self.product_vq_v:
            return None
        return self.v_vq_codebooks[self._layer_index(layer_id)]

    def dequantize_attention_mixed_key(
        self,
        layer_id: int,
        global_indices: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize mixed HP/quant K prefix rows in their original order."""
        if not self.attention_aware_k_mixed:
            raise RuntimeError("attention-aware mixed K is not enabled")
        idx = self._layer_index(layer_id)
        global_indices = global_indices.to(torch.int64)
        is_hp = global_indices >= self._hp_offset
        out = torch.empty(
            (global_indices.numel(), self.head_num, self.head_dim),
            dtype=output_dtype,
            device=global_indices.device,
        )
        if bool(is_hp.any()):
            hp_idx = global_indices[is_hp] - self._hp_offset
            out[is_hp] = self.hp_k_buffer[idx][hp_idx].to(output_dtype)
        is_quant = ~is_hp
        if bool(is_quant.any()):
            quant_idx = global_indices[is_quant]
            out[is_quant] = dequantize_attention_mixed_k_torch(
                self.k_buffer[idx][quant_idx],
                self.k_scales_zeros[idx][quant_idx],
                self.k_low_scales[idx][quant_idx],
                self.k_high_dim,
                output_dtype,
            )
        return out

    def dequantize_product_vq_key(
        self,
        layer_id: int,
        global_indices: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize product-VQ HP/quant K prefix rows in original order."""
        if not self.product_vq_k:
            raise RuntimeError("product VQ K is not enabled")
        idx = self._layer_index(layer_id)
        global_indices = global_indices.to(torch.int64)
        is_hp = global_indices >= self._hp_offset
        out = torch.empty(
            (global_indices.numel(), self.head_num, self.head_dim),
            dtype=output_dtype,
            device=global_indices.device,
        )
        if bool(is_hp.any()):
            hp_idx = global_indices[is_hp] - self._hp_offset
            out[is_hp] = self.hp_k_buffer[idx][hp_idx].to(output_dtype)
        is_quant = ~is_hp
        if bool(is_quant.any()):
            quant_idx = global_indices[is_quant]
            out[is_quant] = dequantize_product_vq_k_torch(
                self.k_buffer[idx][quant_idx],
                self.k_scales_zeros[idx][quant_idx],
                self.k_low_scales[idx][quant_idx],
                self.k_high_dim,
                self.k_vq_codebooks[idx],
                output_dtype,
            )
        return out

    def dequantize_attention_mixed_value(
        self,
        layer_id: int,
        global_indices: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize mixed HP/quant V prefix rows in their original order."""
        if not self.v_compressed_subspace:
            raise RuntimeError("mixed V is not enabled")
        idx = self._layer_index(layer_id)
        global_indices = global_indices.to(torch.int64)
        is_hp = global_indices >= self._hp_offset
        out = torch.empty(
            (global_indices.numel(), self.head_num, self.v_head_dim),
            dtype=output_dtype,
            device=global_indices.device,
        )
        if bool(is_hp.any()):
            hp_idx = global_indices[is_hp] - self._hp_offset
            out[is_hp] = self.hp_v_buffer[idx][hp_idx].to(output_dtype)
        is_quant = ~is_hp
        if bool(is_quant.any()):
            quant_idx = global_indices[is_quant]
            if self.product_vq_v:
                out[is_quant] = dequantize_product_vq_k_torch(
                    self.v_buffer[idx][quant_idx],
                    self.v_scales_zeros[idx][quant_idx],
                    self.v_low_scales[idx][quant_idx],
                    self.v_high_dim,
                    self.v_vq_codebooks[idx],
                    output_dtype,
                )
            else:
                out[is_quant] = dequantize_attention_mixed_k_torch(
                    self.v_buffer[idx][quant_idx],
                    self.v_scales_zeros[idx][quant_idx],
                    self.v_low_scales[idx][quant_idx],
                    self.v_high_dim,
                    output_dtype,
                )
        return out

    def get_hp_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.hp_k_buffer[self._layer_index(layer_id)]

    def get_hp_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.hp_v_buffer[self._layer_index(layer_id)]

    def get_raw_kv_buffer(self, layer_id: int):
        idx = self._layer_index(layer_id)
        return {
            "k_buffer": self.k_buffer[idx],
            "v_buffer": self.v_buffer[idx],
            "k_scales_zeros": self.k_scales_zeros[idx],
            "v_scales_zeros": self.v_scales_zeros[idx],
            "k_low_scales": self.get_key_low_scales(layer_id),
            "v_low_scales": self.get_value_low_scales(layer_id),
            "k_vq_codebook": self.get_key_vq_codebook(layer_id),
            "v_vq_codebook": self.get_value_vq_codebook(layer_id),
            "dtype": "int2",
        }

    def _split_global_locs(self, loc: torch.Tensor):
        loc64 = loc.to(torch.int64)
        hp_mask = loc64 >= self._hp_offset
        quant_loc = loc64[~hp_mask]
        hp_loc_global = loc64[hp_mask] - self._hp_offset
        return quant_loc, hp_loc_global, hp_mask

    def _rotate_kv_inplace(
        self,
        layer_id: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        v_rotation_absorbed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the per-layer Oscar rotation ``rows @ R`` to HP K/V tiles.

        Returns tensors in ``self.hp_dtype`` ready to be stored or packed.
        ``R_k`` / ``R_v`` are ``[head_dim, head_dim]`` bf16 on the KV device,
        loaded in ``__init__``.
        """
        idx = self._layer_index(layer_id)
        k_hp = cache_k.to(self.hp_dtype) @ self._R_k[idx]
        if v_rotation_absorbed:
            v_hp = cache_v.to(self.hp_dtype)
        else:
            v_hp = cache_v.to(self.hp_dtype) @ self._R_v[idx]
        return k_hp, v_hp

    def _prepare_hp_kv_tensors(
        self,
        layer_id: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        already_rotated: bool,
        v_rotation_absorbed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the Oscar rotation to HP K/V and cast to ``hp_dtype``.
        ``already_rotated`` skips the rotation pre-pass.
        """
        if already_rotated:
            return cache_k.to(self.hp_dtype), cache_v.to(self.hp_dtype)
        return self._rotate_kv_inplace(
            layer_id, cache_k, cache_v, v_rotation_absorbed
        )

    def _set_hp_kv_buffer(
        self,
        layer_id: int,
        hp_loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        idx = self._layer_index(layer_id)
        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.hp_k_buffer[idx],
            self.hp_v_buffer[idx],
            hp_loc,
            row_dim=self.row_dim,
            store_dtype=self.hp_dtype,
            device_module=self.device_module,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def _set_quant_kv_buffer_extend(
        self,
        layer_id: int,
        quant_loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        already_hadamard_transformed: bool,
        mixed_hp_offset: Optional[int] = None,
        v_rotation_absorbed: bool = False,
    ):
        """Prefill/extend-only: rotate (oscar R) + optional per-row clip +
        int2-pack + write quant slots.

        Decode-time flushes go through the dedicated GPU flush kernel
        (see ``gpu_flush_int2``); this method is *not* used for those.
        """
        idx = self._layer_index(layer_id)
        clip_on = self._k_clip_ratio > 0.0 or self._v_clip_ratio > 0.0

        if self.k_compressed_subspace or self.v_compressed_subspace:
            if self.k_num_scale_groups != 1 or self.v_num_scale_groups != 1:
                raise NotImplementedError(
                    "compressed-subspace K/V prototype requires group_size=head_dim"
                )
            if not already_hadamard_transformed:
                cache_k, cache_v = self._rotate_kv_inplace(
                    layer_id, cache_k, cache_v, v_rotation_absorbed
                )
            else:
                cache_k = cache_k.to(self.hp_dtype)
                cache_v = cache_v.to(self.hp_dtype)
            quant_mask = quant_loc.to(torch.int64) < int(mixed_hp_offset)
            if not bool(quant_mask.any()):
                return
            dst = quant_loc.to(torch.int64)[quant_mask]
            k_rows = cache_k.reshape(-1, self.head_num, self.head_dim)[quant_mask]
            v_rows = cache_v.reshape(-1, self.head_num, self.v_head_dim)[quant_mask]
            k_rows = _clip_rows_torch(k_rows, self._k_clip_ratio)
            v_rows = _clip_rows_torch(v_rows, self._v_clip_ratio)
            if self.product_vq_k:
                k_packed, k_sz, k_low_scale = pack_product_vq_k_torch(
                    k_rows,
                    self.k_high_dim,
                    self.scale_dtype,
                    self.k_vq_codebooks[idx],
                    self.k_vq_distance_weights[idx],
                )
            elif self.attention_aware_k_mixed:
                k_packed, k_sz, k_low_scale = pack_attention_mixed_k_torch(
                    k_rows, self.k_high_dim, self.scale_dtype
                )
            else:
                k_packed, k_sz = _pack_int2_rows_torch(k_rows, self.scale_dtype)
                k_low_scale = None
            if self.product_vq_v:
                v_packed, v_sz, v_low_scale = pack_product_vq_k_torch(
                    v_rows,
                    self.v_high_dim,
                    self.scale_dtype,
                    self.v_vq_codebooks[idx],
                    self.v_vq_distance_weights[idx],
                )
            elif self.v_compressed_subspace:
                v_packed, v_sz, v_low_scale = pack_attention_mixed_k_torch(
                    v_rows, self.v_high_dim, self.scale_dtype
                )
            else:
                v_packed, v_sz = _pack_int2_rows_torch(v_rows, self.scale_dtype)
                v_low_scale = None
            self.k_buffer[idx][dst] = k_packed
            # Pure INT1 has no affine INT2 parameters. In compatibility mode
            # the legacy two-float arena still exists, but remains unused.
            if k_sz.shape[-1] > 0:
                self.k_scales_zeros[idx][dst] = k_sz
            if k_low_scale is not None:
                self.k_low_scales[idx][dst] = k_low_scale
            self.v_buffer[idx][dst] = v_packed
            if v_sz.shape[-1] > 0:
                self.v_scales_zeros[idx][dst] = v_sz
            if v_low_scale is not None:
                self.v_low_scales[idx][dst] = v_low_scale
            return

        # Fused rotate(K) + clip(KV) + quantize(KV) + set(KV). Skips the
        # standalone ``K @ R_k`` GEMM and its bf16 staging tensor by doing
        # the rotation inside the int2 pack kernel via ``tl.dot``. V must
        # already be in R_v space (rotation absorbed) — the kernel does
        # not rotate V. Requires single-scale layout (num_groups == 1) for
        # both K and V scales/zeros.
        if envs.SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT.get():
            assert v_rotation_absorbed, (
                "V rotation must be absorbed for fused oscar K-rotation + clip + quant + set"
            )

        use_fused_rotate = (
            envs.SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT.get()
            and not already_hadamard_transformed
            and v_rotation_absorbed
            and clip_on
            and _get_num_scale_groups(self.k_scales_zeros[idx]) == 1
            and _get_num_scale_groups(self.v_scales_zeros[idx]) == 1
        )
        if use_fused_rotate:
            quantized_set_kv_int2_oscar_rotate_k_clip_triton(
                cache_k.to(self.hp_dtype),
                cache_v.to(self.hp_dtype),
                self._R_k[idx],
                quant_loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scales_zeros[idx],
                self.v_scales_zeros[idx],
                self._k_clip_ratio,
                self._v_clip_ratio,
                hp_global_offset=mixed_hp_offset,
            )
            return

        if not already_hadamard_transformed:
            cache_k, cache_v = self._rotate_kv_inplace(
                layer_id, cache_k, cache_v, v_rotation_absorbed
            )
        else:
            cache_k = cache_k.to(self.hp_dtype)
            cache_v = cache_v.to(self.hp_dtype)

        if not clip_on:
            quantized_set_kv_int2_pretransformed_triton(
                cache_k,
                cache_v,
                quant_loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scales_zeros[idx],
                self.v_scales_zeros[idx],
                hp_global_offset=mixed_hp_offset,
            )
            return

        quantized_set_kv_int2_pretransformed_clip_triton(
            cache_k,
            cache_v,
            quant_loc,
            self.k_buffer[idx],
            self.v_buffer[idx],
            self.k_scales_zeros[idx],
            self.v_scales_zeros[idx],
            self._k_clip_ratio,
            self._v_clip_ratio,
            hp_global_offset=mixed_hp_offset,
            lloyd_max=self._lloyd_max,
        )

    def _set_mixed_hp_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        idx = self._layer_index(layer_id)

        def _launch(src: torch.Tensor, dst: torch.Tensor):
            src2 = src.reshape(src.shape[0], -1)
            dst2 = dst.reshape(dst.shape[0], -1)
            row_dim = src2.shape[1]
            if row_dim == 0 or src2.shape[0] == 0:
                return
            block_row = min(1024, triton.next_power_of_2(row_dim))
            grid = (src2.shape[0], triton.cdiv(row_dim, block_row))
            _set_mixed_hp_buffer_kernel[grid](
                src2,
                dst2,
                loc,
                src2.shape[0],
                row_dim,
                src2.stride(0),
                src2.stride(1),
                dst2.stride(0),
                dst2.stride(1),
                HP_OFFSET=int(self._hp_offset),
                BLOCK_ROW=block_row,
                num_warps=4,
                num_stages=1,
            )

        _launch(cache_k, self.hp_k_buffer[idx])
        _launch(cache_v, self.hp_v_buffer[idx])

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        already_hadamard_transformed: bool = False,
        is_decode: bool = False,
        anchor_req_indices: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
    ):
        """Write K/V to the unified pool.

        ``is_decode`` selects the path:
            * ``True`` -- single-token decode write. Caller fills ``loc`` with
              valid HP slot ids (from ``allocator.alloc_hp_recent``); we
              write only the HP buffer, with no boolean masking (safe under
              CUDA-graph capture).
            * ``False`` (extend / prefill) -- mixed write: int2 quant slots get
              the rotated+clipped pack, HP slots get the bf16 row.

        Callers must pass ``is_decode`` based on
        ``forward_batch.forward_mode.is_decode_or_idle()``. Capture state is
        *not* a reliable proxy: piecewise CUDA graph captures parts of prefill
        (would mis-route to the HP-only branch) and ``--disable-cuda-graph``
        runs decode eagerly (would mis-route to the quant+HP branch).
        """
        if loc.numel() == 0:
            return

        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        v_rotation_absorbed = bool(getattr(layer, "oscar_v_rotation_absorbed", False))

        if is_decode:
            hp_local = loc.to(torch.int64) - self._hp_offset
            cache_k_hp, cache_v_hp = self._prepare_hp_kv_tensors(
                layer_id,
                cache_k,
                cache_v,
                already_hadamard_transformed,
                v_rotation_absorbed,
            )
            self._set_hp_kv_buffer(layer_id, hp_local, cache_k_hp, cache_v_hp)
            return

        self._set_quant_kv_buffer_extend(
            layer_id,
            loc,
            cache_k,
            cache_v,
            already_hadamard_transformed,
            mixed_hp_offset=int(self._hp_offset),
            v_rotation_absorbed=v_rotation_absorbed,
        )
        cache_k_hp, cache_v_hp = self._prepare_hp_kv_tensors(
            layer_id,
            cache_k,
            cache_v,
            already_hadamard_transformed,
            v_rotation_absorbed,
        )
        if self.anchor_enabled:
            if anchor_req_indices is None or anchor_positions is None:
                raise ValueError(
                    "anchor-enabled prefill requires request indices and positions"
                )
            quant_mask = loc.to(torch.int64) < int(self._hp_offset)
            if bool(quant_mask.any().item()):
                self._update_fp16_anchors_from_rows(
                    self._layer_index(layer_id),
                    loc.to(torch.int64)[quant_mask],
                    cache_k_hp.reshape(-1, self.head_num, self.head_dim)[quant_mask],
                    cache_v_hp.reshape(-1, self.head_num, self.v_head_dim)[quant_mask],
                    anchor_req_indices[quant_mask],
                    anchor_positions[quant_mask],
                )
        self._set_mixed_hp_kv_buffer(layer_id, loc, cache_k_hp, cache_v_hp)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if tgt_loc.numel() == 0:
            return
        # Moves only make sense within the same tier. The allocator ensures
        # that; callers should split by tier before calling here.
        tgt_q, tgt_hp, tgt_mask = self._split_global_locs(tgt_loc)
        src_q, src_hp, src_mask = self._split_global_locs(src_loc)
        assert torch.equal(tgt_mask, src_mask), (
            "move_kv_cache requires src/tgt tiers to match"
        )
        for l in range(self.layer_num):
            if tgt_q.numel() > 0:
                self.k_buffer[l][tgt_q] = self.k_buffer[l][src_q]
                self.v_buffer[l][tgt_q] = self.v_buffer[l][src_q]
                self.k_scales_zeros[l][tgt_q] = self.k_scales_zeros[l][src_q]
                self.v_scales_zeros[l][tgt_q] = self.v_scales_zeros[l][src_q]
                if self.k_compressed_subspace:
                    self.k_low_scales[l][tgt_q] = self.k_low_scales[l][src_q]
                if self.v_compressed_subspace:
                    self.v_low_scales[l][tgt_q] = self.v_low_scales[l][src_q]
            if tgt_hp.numel() > 0:
                self.hp_k_buffer[l][tgt_hp] = self.hp_k_buffer[l][src_hp]
                self.hp_v_buffer[l][tgt_hp] = self.hp_v_buffer[l][src_hp]

    def get_cpu_copy(self, indices):
        raise NotImplementedError("CPU offload is not supported by UnifiedInt2HPKVPool")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("CPU offload is not supported by UnifiedInt2HPKVPool")
