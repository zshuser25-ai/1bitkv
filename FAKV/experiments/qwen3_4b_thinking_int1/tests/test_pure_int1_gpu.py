import math
import os
import sys
import types
from pathlib import Path

import pytest
import torch

# Load the vendored runtime modules without importing SGLang's optional
# frontend dependencies. The full server environment installs those packages;
# this kernel test only needs the namespace path.
repo_root = Path(__file__).resolve().parents[3]
sglang_root = repo_root / "sglang-research" / "python" / "sglang"
if "sglang" not in sys.modules:
    package = types.ModuleType("sglang")
    package.__path__ = [os.fspath(sglang_root)]
    sys.modules["sglang"] = package
utils = types.ModuleType("sglang.srt.utils")
utils.is_hip = lambda: False
sys.modules["sglang.srt.utils"] = utils
environ = types.ModuleType("sglang.srt.environ")
environ.envs = types.SimpleNamespace()
sys.modules["sglang.srt.environ"] = environ

from sglang.QuantKernel.gpu_flush_int2 import FlushPlan, gpu_flush_int2_apply
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _decode_grouped_att_m_fwd_quant_attention_mixed,
    _decode_grouped_att_m_fwd_quant_v_mixed,
)


def pack_attention_mixed_k_torch(x, high_dim, scale_dtype):
    assert high_dim == 0
    low = x.float()
    scale = low.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    bits = (low >= 0).to(torch.uint8).reshape(*low.shape[:-1], 16, 8)
    weights = 1 << torch.arange(8, dtype=torch.uint8, device=x.device)
    packed = (bits * weights).sum(dim=-1).to(torch.uint8)
    high_sz = torch.zeros(*x.shape[:-1], 2, device=x.device, dtype=scale_dtype)
    return packed, high_sz, scale.to(torch.bfloat16)


def dequantize_attention_mixed_k_torch(
    packed, high_scales_zeros, low_scales, high_dim, output_dtype
):
    assert high_dim == 0
    bit_ids = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = ((packed[..., None] >> bit_ids) & 1).reshape(*packed.shape[:-1], 128)
    return torch.where(bits.bool(), 1.0, -1.0).mul(low_scales.float()).to(output_dtype)


def _ptrs(tensors):
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors], device="cuda", dtype=torch.int64
    )


def _strides(tensor):
    return tuple(int(value) for value in tensor.stride())


def _dequantize_int2(packed, scales_zeros):
    scale = scales_zeros[..., 0:1].float()
    zero = scales_zeros[..., 1:2].float()
    return torch.cat(
        [((packed >> shift) & 0x03).float() for shift in (0, 2, 4, 6)], dim=-1
    ).sub(zero).mul(scale)


@pytest.mark.parametrize("mixed_k,k_high_dim,v_high_dim", [(True, 0, 0)])
def test_pure_int1_flush_and_decode_match_dense_reference(
    mixed_k, k_high_dim, v_high_dim
):
    torch.manual_seed(20260914 + k_high_dim + v_high_dim)
    layers, tokens, kv_heads, q_heads, dim = 2, 129, 8, 32, 128
    dense_k = [
        torch.randn(tokens, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    dense_v = [
        torch.randn(tokens, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    k_bytes = (
        k_high_dim // 4 + (dim - k_high_dim) // 8 if mixed_k else dim // 4
    )
    v_bytes = v_high_dim // 4 + (dim - v_high_dim) // 8
    packed_k = [
        torch.empty(tokens, kv_heads, k_bytes, device="cuda", dtype=torch.uint8)
        for _ in range(layers)
    ]
    packed_v = [
        torch.empty(tokens, kv_heads, v_bytes, device="cuda", dtype=torch.uint8)
        for _ in range(layers)
    ]
    k_scales_zeros = [
        torch.empty(tokens, kv_heads, 2, device="cuda", dtype=torch.float32)
        for _ in range(layers)
    ]
    v_scales_zeros = [
        torch.empty(tokens, kv_heads, 2, device="cuda", dtype=torch.float32)
        for _ in range(layers)
    ]
    k_low_scales = [
        torch.empty(tokens, kv_heads, 1, device="cuda", dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    v_low_scales = [
        torch.empty(tokens, kv_heads, 1, device="cuda", dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    plan = FlushPlan(
        returned_slot_ids=positions,
        src_hp_slot=positions,
        flush_pos=torch.arange(tokens, device="cuda", dtype=torch.int32),
        valid_mask=torch.ones(tokens, device="cuda", dtype=torch.int8),
        dst_quant_slots=positions,
        bs=1,
        flush_interval=tokens,
    )
    req_to_token = torch.full((1, tokens), -1, device="cuda", dtype=torch.int32)
    gpu_flush_int2_apply(
        plan,
        req_pool_indices=torch.tensor([0], device="cuda", dtype=torch.int64),
        req_to_token=req_to_token,
        hp_k_ptrs=_ptrs(dense_k),
        hp_v_ptrs=_ptrs(dense_v),
        quant_k_ptrs=_ptrs(packed_k),
        quant_v_ptrs=_ptrs(packed_v),
        k_sz_ptrs=_ptrs(k_scales_zeros),
        v_sz_ptrs=_ptrs(v_scales_zeros),
        hp_k_sample=dense_k[0],
        hp_v_sample=dense_v[0],
        quant_k_sample=packed_k[0],
        quant_v_sample=packed_v[0],
        k_sz_sample=k_scales_zeros[0],
        v_sz_sample=v_scales_zeros[0],
        hp_k_strides=_strides(dense_k[0]),
        hp_v_strides=_strides(dense_v[0]),
        quant_k_strides=_strides(packed_k[0]),
        quant_v_strides=_strides(packed_v[0]),
        k_sz_strides=_strides(k_scales_zeros[0]),
        v_sz_strides=_strides(v_scales_zeros[0]),
        num_heads=kv_heads,
        head_dim=dim,
        v_head_dim=dim,
        k_num_scale_groups=1,
        v_num_scale_groups=1,
        num_layers=layers,
        k_low_scale_ptrs=_ptrs(k_low_scales),
        k_low_scale_sample=k_low_scales[0],
        k_low_scale_strides=_strides(k_low_scales[0]),
        v_low_scale_ptrs=_ptrs(v_low_scales),
        v_low_scale_sample=v_low_scales[0],
        v_low_scale_strides=_strides(v_low_scales[0]),
        attention_aware_k_mixed=mixed_k,
        mixed_v=True,
        k_high_dim=k_high_dim,
        v_high_dim=v_high_dim,
    )
    torch.cuda.synchronize()

    for layer in range(layers):
        if mixed_k:
            ref_k, ref_ksz, ref_klow = pack_attention_mixed_k_torch(
                dense_k[layer], k_high_dim, torch.float32
            )
        else:
            ref_k, ref_ksz = _pack_int2_rows_torch(
                dense_k[layer], torch.float32
            )
            ref_klow = None
        ref_v, ref_vsz, ref_vlow = pack_attention_mixed_k_torch(
            dense_v[layer], v_high_dim, torch.float32
        )
        assert torch.equal(packed_k[layer], ref_k)
        assert torch.equal(packed_v[layer], ref_v)
        assert torch.allclose(k_scales_zeros[layer], ref_ksz, atol=1e-6, rtol=1e-5)
        assert torch.allclose(v_scales_zeros[layer], ref_vsz, atol=1e-6, rtol=1e-5)
        if mixed_k:
            assert torch.equal(k_low_scales[layer], ref_klow)
        assert torch.equal(v_low_scales[layer], ref_vlow)

    kd = (
        dequantize_attention_mixed_k_torch(
            packed_k[0], k_scales_zeros[0], k_low_scales[0], k_high_dim,
            torch.float32,
        )
        if mixed_k
        else _dequantize_int2(packed_k[0], k_scales_zeros[0])
    )
    vd = dequantize_attention_mixed_k_torch(
        packed_v[0], v_scales_zeros[0], v_low_scales[0], v_high_dim, torch.float32
    )
    query = torch.randn(1, q_heads, dim, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, q_heads, 1, dim, device="cuda", dtype=torch.float32)
    lse = torch.empty(1, q_heads, 1, device="cuda", dtype=torch.float32)
    indptr = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    indices = torch.arange(tokens, device="cuda", dtype=torch.int32)
    splits = torch.ones(1, device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(dim)
    if mixed_k:
        _decode_grouped_att_m_fwd_quant_attention_mixed(
            query, packed_k[0], packed_v[0], k_scales_zeros[0],
            k_low_scales[0], v_scales_zeros[0], v_low_scales[0], out, lse,
            indptr, indices, splits, 1, scale, 0.0, -1,
            k_high_dim=k_high_dim, v_high_dim=v_high_dim,
        )
    else:
        _decode_grouped_att_m_fwd_quant_v_mixed(
            query, packed_k[0], packed_v[0], k_scales_zeros[0],
            v_scales_zeros[0], v_low_scales[0], out, lse, indptr, indices,
            splits, 1, scale, 0.0, -1, v_high_dim=v_high_dim,
        )
    torch.cuda.synchronize()

    references = []
    q_per_kv = q_heads // kv_heads
    for head in range(q_heads):
        key = kd[:, head // q_per_kv]
        value = vd[:, head // q_per_kv]
        probability = torch.softmax((query[0, head].float() @ key.T) * scale, dim=-1)
        references.append(probability @ value)
    reference = torch.stack(references)
    assert torch.allclose(out[0, :, 0], reference, atol=2e-2, rtol=2e-2)
