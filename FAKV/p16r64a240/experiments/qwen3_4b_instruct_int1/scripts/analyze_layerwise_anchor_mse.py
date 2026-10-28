#!/usr/bin/env python3
"""Layerwise INT1/FP16-anchor error analysis for the fixed OSCAR layout."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
CALIB = (
    ROOT
    / "rotation/qwen3-4B-instruct-2507/GPQA/seq20000_prompt88_group128"
)
DUMPS = CALIB / "qkv_dumps/gpqa"
OUT = ROOT / "experiments/qwen3_4b_instruct_int1/analysis/anchor_mse"
FISHER = CALIB / "anchor_fisher_diag_v1.pt"
CAPACITY = 28
PREFIX = 64
RECENT = 256
K_CLIP = 0.96
V_CLIP = 0.92
HEAD_DIM = 128


def load_rotations(path: Path, device: torch.device) -> list[torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [
        payload["layers"][layer]["rotation"].to(
            device=device, dtype=torch.bfloat16
        )
        for layer in range(36)
    ]


def clip_rows(x: torch.Tensor, ratio: float) -> torch.Tensor:
    index = min(max(int(ratio * x.shape[-1]), 0), x.shape[-1] - 1)
    threshold = torch.kthvalue(x.float().abs(), index + 1, dim=-1).values
    return x.float().clamp(-threshold[..., None], threshold[..., None]).to(x.dtype)


def int1_reconstruct(clipped: torch.Tensor) -> torch.Tensor:
    # Runtime stores this scale in BF16, including for FP16 anchors.
    scale = clipped.float().abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = scale.to(torch.bfloat16).float()
    return torch.where(clipped.float() >= 0, 1.0, -1.0) * scale


def top_indices(score: torch.Tensor, count: int) -> torch.Tensor:
    # score [tokens, heads] -> [heads, count]
    return torch.topk(score.transpose(0, 1), k=count, dim=1).indices


def gather_by_head(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # values [tokens, heads], indices [heads, selected]
    return torch.gather(values.transpose(0, 1), 1, indices)


def analyze_layer(
    layer: int,
    r_k: torch.Tensor,
    r_v: torch.Tensor,
    fisher_k: torch.Tensor,
    fisher_v: torch.Tensor,
    device: torch.device,
) -> dict:
    layer_dir = DUMPS / f"layer_{layer}"
    chunk_ids = sorted(int(path.stem) for path in (layer_dir / "k").glob("*.pt"))

    # Query second moments in the same rotated coordinate system used by decode.
    q2_sum = torch.zeros((8, HEAD_DIM), dtype=torch.float64, device=device)
    q_count = 0
    for chunk_id in chunk_ids:
        q = torch.load(layer_dir / "q" / f"{chunk_id}.pt", map_location="cpu")
        q = (q.to(device=device, dtype=torch.bfloat16) @ r_k).reshape(
            -1, 8, 4, HEAD_DIM
        )
        q2_sum += q.float().square().sum(dim=(0, 2)).double()
        q_count += q.shape[0] * q.shape[2]
    q2 = (q2_sum / q_count).float()

    scalar_parts: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "k_sse",
            "v_sse",
            "k_norm",
            "v_norm",
            "k_clip_sse",
            "v_clip_sse",
            "k_quant_sse",
            "v_quant_sse",
            "k_q_sse",
            "k_q_norm",
            "k_no_clip_sse",
            "v_no_clip_sse",
            "k_current_scale_fp32_sse",
            "v_current_scale_fp32_sse",
            "fisher_score",
        )
    }
    prompt_spans: list[tuple[int, int]] = []
    cursor = 0

    for chunk_id in chunk_ids:
        k = torch.load(layer_dir / "k" / f"{chunk_id}.pt", map_location="cpu")
        v = torch.load(layer_dir / "v" / f"{chunk_id}.pt", map_location="cpu")
        seq_lens = torch.load(
            layer_dir / "seq_lens" / f"{chunk_id}.pt", map_location="cpu"
        ).tolist()
        k = k.to(device=device, dtype=torch.bfloat16) @ r_k
        v = v.to(device=device, dtype=torch.bfloat16) @ r_v
        k_clip = clip_rows(k, K_CLIP)
        v_clip = clip_rows(v, V_CLIP)
        rk = int1_reconstruct(k_clip)
        rv = int1_reconstruct(v_clip)

        start = 0
        selected_rows: list[torch.Tensor] = []
        for seq_len in seq_lens:
            middle_start = start + PREFIX
            middle_end = start + max(PREFIX, seq_len - RECENT)
            if middle_end > middle_start:
                rows = torch.arange(middle_start, middle_end, device=device)
                selected_rows.append(rows)
                width = middle_end - middle_start
                prompt_spans.append((cursor, cursor + width))
                cursor += width
            start += seq_len
        if not selected_rows:
            continue
        rows = torch.cat(selected_rows)
        exact_k = k[rows].float()
        exact_v = v[rows].float()
        clipped_k = k_clip[rows].float()
        clipped_v = v_clip[rows].float()
        recon_k = rk[rows]
        recon_v = rv[rows]
        # Clipping does not change a coordinate's sign.  Therefore the
        # MSE-optimal symmetric INT1 scale for the *original* row remains
        # mean(abs(original)); compare that with the runtime's clipped scale.
        k_sign = torch.where(exact_k >= 0, 1.0, -1.0)
        v_sign = torch.where(exact_v >= 0, 1.0, -1.0)
        k_no_clip_scale = exact_k.abs().mean(-1, keepdim=True).to(torch.bfloat16).float()
        v_no_clip_scale = exact_v.abs().mean(-1, keepdim=True).to(torch.bfloat16).float()
        k_current_scale_fp32 = clipped_k.abs().mean(-1, keepdim=True)
        v_current_scale_fp32 = clipped_v.abs().mean(-1, keepdim=True)
        ek = exact_k - recon_k
        ev = exact_v - recon_v
        scalar_parts["k_sse"].append(ek.square().sum(-1))
        scalar_parts["v_sse"].append(ev.square().sum(-1))
        scalar_parts["k_norm"].append(exact_k.square().sum(-1))
        scalar_parts["v_norm"].append(exact_v.square().sum(-1))
        scalar_parts["k_clip_sse"].append((exact_k - clipped_k).square().sum(-1))
        scalar_parts["v_clip_sse"].append((exact_v - clipped_v).square().sum(-1))
        scalar_parts["k_quant_sse"].append((clipped_k - recon_k).square().sum(-1))
        scalar_parts["v_quant_sse"].append((clipped_v - recon_v).square().sum(-1))
        scalar_parts["k_q_sse"].append((ek.square() * q2[None]).sum(-1))
        scalar_parts["k_q_norm"].append((exact_k.square() * q2[None]).sum(-1))
        scalar_parts["k_no_clip_sse"].append(
            (exact_k - k_sign * k_no_clip_scale).square().sum(-1)
        )
        scalar_parts["v_no_clip_sse"].append(
            (exact_v - v_sign * v_no_clip_scale).square().sum(-1)
        )
        scalar_parts["k_current_scale_fp32_sse"].append(
            (exact_k - k_sign * k_current_scale_fp32).square().sum(-1)
        )
        scalar_parts["v_current_scale_fp32_sse"].append(
            (exact_v - v_sign * v_current_scale_fp32).square().sum(-1)
        )
        scalar_parts["fisher_score"].append(
            (ek.square() * fisher_k[None]).sum(-1)
            + (ev.square() * fisher_v[None]).sum(-1)
        )
        del k, v, k_clip, v_clip, rk, rv

    values = {name: torch.cat(parts) for name, parts in scalar_parts.items()}
    eps = 2.0**-24
    k_rel = values["k_sse"] / values["k_norm"].clamp_min(eps)
    v_rel = values["v_sse"] / values["v_norm"].clamp_min(eps)
    current = 0.5 * (k_rel + v_rel)
    absolute = values["k_sse"] + values["v_sse"]
    head_norm = 0.5 * (
        values["k_sse"] / values["k_norm"].mean(0).clamp_min(eps)
        + values["v_sse"] / values["v_norm"].mean(0).clamp_min(eps)
    )
    query_aware = 0.5 * (
        values["k_q_sse"] / values["k_q_norm"].clamp_min(eps) + v_rel
    )
    methods = {
        "current_rel_50_50": current,
        "absolute_joint_mse": absolute,
        "head_normalized": head_norm,
        "query_diag_rel_50_50": query_aware,
        "calibrated_fisher_v1": values["fisher_score"],
    }
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        methods[f"relative_alpha_{alpha:.2f}"] = alpha * k_rel + (1 - alpha) * v_rel

    removed = {name: 0.0 for name in methods}
    removed_k = {name: 0.0 for name in methods}
    removed_v = {name: 0.0 for name in methods}
    overlap_numer = 0
    overlap_denom = 0
    selected_total = 0
    for start, end in prompt_spans:
        count = min(CAPACITY, end - start)
        if count == 0:
            continue
        indices = {
            name: top_indices(score[start:end], count) for name, score in methods.items()
        }
        utility_k = values["k_sse"][start:end]
        utility_v = values["v_sse"][start:end]
        for name, chosen in indices.items():
            rk_removed = gather_by_head(utility_k, chosen).sum().item()
            rv_removed = gather_by_head(utility_v, chosen).sum().item()
            removed_k[name] += rk_removed
            removed_v[name] += rv_removed
            removed[name] += rk_removed + rv_removed
        cur = indices["current_rel_50_50"]
        oracle = indices["absolute_joint_mse"]
        intersection = (cur[:, :, None] == oracle[:, None, :]).any(-1).sum().item()
        overlap_numer += intersection
        overlap_denom += cur.numel()
        selected_total += cur.numel()

    rows = values["k_sse"].shape[0]
    heads = values["k_sse"].shape[1]
    elements_per_side = rows * heads * HEAD_DIM
    total_k = values["k_sse"].sum().item()
    total_v = values["v_sse"].sum().item()
    total = total_k + total_v
    current_removed = removed["current_rel_50_50"]
    oracle_removed = removed["absolute_joint_mse"]
    result = {
        "layer": layer,
        "middle_tokens": rows,
        "prompts_with_middle": len(prompt_spans),
        "anchor_entries": selected_total,
        "k_mse": total_k / elements_per_side,
        "v_mse": total_v / elements_per_side,
        "joint_mse": total / (2 * elements_per_side),
        "k_clip_mse": values["k_clip_sse"].sum().item() / elements_per_side,
        "v_clip_mse": values["v_clip_sse"].sum().item() / elements_per_side,
        "k_quant_after_clip_mse": values["k_quant_sse"].sum().item() / elements_per_side,
        "v_quant_after_clip_mse": values["v_quant_sse"].sum().item() / elements_per_side,
        "k_no_clip_mse": values["k_no_clip_sse"].sum().item() / elements_per_side,
        "v_no_clip_mse": values["v_no_clip_sse"].sum().item() / elements_per_side,
        "k_current_scale_fp32_mse": values["k_current_scale_fp32_sse"].sum().item()
        / elements_per_side,
        "v_current_scale_fp32_mse": values["v_current_scale_fp32_sse"].sum().item()
        / elements_per_side,
        "k_relative_mean": k_rel.mean().item(),
        "v_relative_mean": v_rel.mean().item(),
        "joint_relative_mean": current.mean().item(),
        "joint_relative_p50": current.quantile(0.50).item(),
        "joint_relative_p90": current.quantile(0.90).item(),
        "joint_relative_p99": current.quantile(0.99).item(),
        "kv_relative_correlation": torch.corrcoef(
            torch.stack([k_rel.flatten(), v_rel.flatten()])
        )[0, 1].item(),
        "current_anchor_removed_mse_pct": 100 * current_removed / total,
        "oracle_anchor_removed_mse_pct": 100 * oracle_removed / total,
        "current_vs_oracle_efficiency_pct": 100 * current_removed / oracle_removed,
        "current_oracle_selection_overlap_pct": 100 * overlap_numer / overlap_denom,
        "post_current_joint_mse": (total - current_removed) / (2 * elements_per_side),
        "post_oracle_joint_mse": (total - oracle_removed) / (2 * elements_per_side),
    }
    for name in methods:
        result[f"{name}_removed_mse_pct"] = 100 * removed[name] / total
        result[f"{name}_removed_k_pct"] = 100 * removed_k[name] / total_k
        result[f"{name}_removed_v_pct"] = 100 * removed_v[name] / total_v
    return result


def main() -> None:
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    k_rotations = load_rotations(CALIB / "rotations/k_rotation_qqt_r_h_pbr.pt", device)
    v_rotations = load_rotations(CALIB / "rotations/v_rotation_sst_r_h_pbr.pt", device)
    fisher = torch.load(FISHER, map_location="cpu", weights_only=False)
    fisher_k = fisher["k_weights"].to(device=device, dtype=torch.float32)
    fisher_v = fisher["v_weights"].to(device=device, dtype=torch.float32)
    results = []
    for layer in range(36):
        result = analyze_layer(
            layer,
            k_rotations[layer],
            v_rotations[layer],
            fisher_k[layer],
            fisher_v[layer],
            device,
        )
        results.append(result)
        print(
            f"layer={layer:02d} mse={result['joint_mse']:.6g} "
            f"current_removed={result['current_anchor_removed_mse_pct']:.2f}% "
            f"oracle_removed={result['oracle_anchor_removed_mse_pct']:.2f}% "
            f"eff={result['current_vs_oracle_efficiency_pct']:.2f}%",
            flush=True,
        )
    (OUT / "layerwise_anchor_mse.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (OUT / "layerwise_anchor_mse.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    numeric = [key for key, value in results[0].items() if isinstance(value, float)]
    summary = {
        "layers": len(results),
        "calibration": str(CALIB.relative_to(ROOT)),
        "formula": "0.5*(K_SSE/K_norm + V_SSE/V_norm)",
        "means": {key: sum(row[key] for row in results) / len(results) for key in numeric},
        "min_layers": {key: min(results, key=lambda row: row[key])["layer"] for key in numeric},
        "max_layers": {key: max(results, key=lambda row: row[key])["layer"] for key in numeric},
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
