#!/usr/bin/env python3
"""Calibrate diagonal final-loss Fisher weights for OSCAR FP16 anchors.

The calibration objective is teacher-forced next-token cross entropy on the
tail of representative prompts.  K/V gradients are captured after RoPE,
rotated into the exact coordinate system used by OSCAR, squared, averaged,
regularized, and saved as a small runtime checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import repeat_kv


NUM_LAYERS = 36
KV_HEADS = 8
HEAD_DIM = 128
CAPTURED: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fisher_eager_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager GQA attention that retains post-RoPE K/V gradients."""
    if key.requires_grad:
        key.retain_grad()
        value.retain_grad()
        CAPTURED[int(module.layer_idx)] = (key, value)
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
    return output, weights


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    calib = root / "rotation/qwen3-4B-instruct-2507/GPQA/seq20000_prompt88_group128"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model identifier stored in the checkpoint (defaults to --model basename).",
    )
    parser.add_argument(
        "--source",
        choices=("gpqa_diamond", "jsonl"),
        default="gpqa_diamond",
        help="Use independent GPQA calibration prompts by default; jsonl is for custom held-out data.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=root
        / "vllm_turbo_baseline_acc/accuracy_suites/"
        "model_agnostic_accuracy_official_v5/manifest.jsonl",
    )
    parser.add_argument(
        "--k-rotation",
        type=Path,
        default=calib / "rotations/k_rotation_qqt_r_h_pbr.pt",
    )
    parser.add_argument(
        "--v-rotation",
        type=Path,
        default=calib / "rotations/v_rotation_sst_r_h_pbr.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=calib / "anchor_fisher_diag_gpqa198_v2.pt",
    )
    parser.add_argument("--samples-per-benchmark", type=int, default=8)
    parser.add_argument("--gpqa-prompts", type=int, default=198)
    parser.add_argument("--max-used-samples", type=int, default=0)
    parser.add_argument("--gpqa-csv", type=Path, required=True)
    parser.add_argument("--anchor-tokens", type=int, default=240)
    parser.add_argument("--benchmarks", default="GSM8K,IFEval,LiveCodeBench v6,MultiPL-E")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--tail-loss-tokens", type=int, default=256)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Render calibration prompts with enable_thinking=False.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Render calibration prompts with enable_thinking=True explicitly.",
    )
    parser.add_argument("--prefix-tokens", type=int, default=16)
    parser.add_argument("--recent-tokens", type=int, default=64)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument("--winsor-quantile", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def gpqa_prompts(
    root: Path,
    tokenizer,
    count: int,
    seed: int,
    disable_thinking: bool = False,
    enable_thinking: bool = False,
    gpqa_csv: Path | None = None,
) -> list[dict[str, str]]:
    """Reuse the exact independent GPQA-diamond calibration prompt builder."""
    if gpqa_csv is None:
        raise ValueError("gpqa_csv is required for frozen calibration")
    with gpqa_csv.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != 198 or count != 198:
        raise ValueError("V2 must use all 198 GPQA Diamond records")
    rng = random.Random(seed)
    template = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    raw = []
    for row in source_rows:
        perm = rng.sample(range(4), 4)
        choices = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        body = dict(zip("ABCD", [choices[i] for i in perm]))
        body["Question"] = row["Question"]
        raw.append(template.format(**body))
    rows = [
        {
            "benchmark": "GPQA-diamond-calibration",
            "id": f"gpqa_diamond:{index:06d}",
            "text": tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **(
                    {"enable_thinking": enable_thinking}
                    if disable_thinking or enable_thinking
                    else {}
                ),
            ),
        }
        for index, prompt in enumerate(raw)
    ]
    random.Random(seed).shuffle(rows)
    return rows


def row_text(
    row: dict[str, Any],
    tokenizer,
    disable_thinking: bool = False,
    enable_thinking: bool = False,
) -> str:
    template_kwargs = (
        {"enable_thinking": enable_thinking}
        if disable_thinking or enable_thinking
        else {}
    )
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    text = row.get("prompt") or row.get("text")
    if not isinstance(text, str):
        return ""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def reservoir_prompts(
    path: Path,
    tokenizer,
    benchmarks: set[str],
    per_benchmark: int,
    seed: int,
    max_records: int,
    disable_thinking: bool = False,
    enable_thinking: bool = False,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    reservoirs: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    with path.open(encoding="utf-8") as source:
        for record_idx, line in enumerate(source):
            if max_records and record_idx >= max_records:
                break
            row = json.loads(line)
            benchmark = str(row.get("benchmark") or row.get("dataset") or "default")
            if benchmarks and benchmark not in benchmarks:
                continue
            text = row_text(row, tokenizer, disable_thinking, enable_thinking)
            if not text:
                continue
            seen[benchmark] += 1
            candidate = {"benchmark": benchmark, "id": str(row.get("id", record_idx)), "text": text}
            bucket = reservoirs[benchmark]
            if len(bucket) < per_benchmark:
                bucket.append(candidate)
            else:
                replacement = rng.randrange(seen[benchmark])
                if replacement < per_benchmark:
                    bucket[replacement] = candidate
    selected = [row for benchmark in sorted(reservoirs) for row in reservoirs[benchmark]]
    rng.shuffle(selected)
    return selected


def load_rotations(path: Path) -> list[torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    rotations = []
    for layer in range(NUM_LAYERS):
        entry = state["layers"].get(layer, state["layers"].get(str(layer)))
        if entry is None or tuple(entry["rotation"].shape) != (HEAD_DIM, HEAD_DIM):
            raise ValueError(f"invalid rotation for layer {layer} in {path}")
        rotations.append(entry["rotation"].float().contiguous())
    return rotations


def regularize_fisher(
    k: torch.Tensor,
    v: torch.Tensor,
    shrinkage: float,
    winsor_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    if not 0 < winsor_quantile <= 1:
        raise ValueError("winsor_quantile must be in (0, 1]")
    if winsor_quantile < 1:
        k_cap = torch.quantile(k, winsor_quantile, dim=-1, keepdim=True)
        v_cap = torch.quantile(v, winsor_quantile, dim=-1, keepdim=True)
        k = torch.minimum(k, k_cap)
        v = torch.minimum(v, v_cap)
    joint_mean = 0.5 * (k.mean(-1, keepdim=True) + v.mean(-1, keepdim=True))
    if (joint_mean <= 0).any() or not torch.isfinite(joint_mean).all():
        raise RuntimeError("calibration produced empty or non-finite Fisher weights")
    k = (1 - shrinkage) * k + shrinkage * joint_mean
    v = (1 - shrinkage) * v + shrinkage * joint_mean
    normalizer = 0.5 * (k.mean(-1, keepdim=True) + v.mean(-1, keepdim=True))
    return (k / normalizer).float().contiguous(), (v / normalizer).float().contiguous()


def main() -> None:
    args = parse_args()
    if args.disable_thinking and args.enable_thinking:
        raise ValueError("--disable-thinking and --enable-thinking are mutually exclusive")
    root = Path(__file__).resolve().parents[3]
    if args.max_length <= args.prefix_tokens + args.recent_tokens:
        raise ValueError("max-length must leave a non-empty INT1 middle region")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    benchmarks = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    if args.source == "gpqa_diamond":
        prompts = gpqa_prompts(
            root,
            tokenizer,
            args.gpqa_prompts,
            args.seed,
            args.disable_thinking,
            args.enable_thinking,
            args.gpqa_csv,
        )
        source_description = "all 198 GPQA Diamond prompts; overlaps GPQA evaluation"
    else:
        prompts = reservoir_prompts(
            args.input_jsonl,
            tokenizer,
            benchmarks,
            args.samples_per_benchmark,
            args.seed,
            args.max_records,
            args.disable_thinking,
            args.enable_thinking,
        )
        source_description = str(args.input_jsonl)
    if not prompts:
        raise RuntimeError("no calibration prompts selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = args.output.parent / "gpqa198_rendered_prompts.jsonl"
    prompt_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts))
    print(
        f"selected_prompts={len(prompts)} source={source_description}", flush=True
    )

    AttentionInterface.register("fisher_eager", fisher_eager_attention)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation="fisher_eager",
        trust_remote_code=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.enable_input_require_grads()
    input_device = model.get_input_embeddings().weight.device
    k_rotations = load_rotations(args.k_rotation)
    v_rotations = load_rotations(args.v_rotation)
    k_fisher = torch.zeros((NUM_LAYERS, KV_HEADS, HEAD_DIM), dtype=torch.float64)
    v_fisher = torch.zeros_like(k_fisher)
    observations = torch.zeros(NUM_LAYERS, dtype=torch.int64)
    used = []
    original = torch.load(root / "assets/qwen3_4b_instruct_2507/anchor_fisher_diag_gpqa16.pt", map_location="cpu", weights_only=False)
    control_ids = {sample["id"] for sample in original["calibration"]["samples"]}
    assert len(control_ids) == 16
    control_k = torch.zeros_like(k_fisher)
    control_v = torch.zeros_like(v_fisher)
    control_observations = torch.zeros_like(observations)
    control_used = []
    raw_dir = args.output.parent / "raw"
    raw_dir.mkdir(exist_ok=True)

    for sample_idx, sample in enumerate(prompts):
        if args.max_used_samples and len(used) >= args.max_used_samples:
            break
        encoded = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )
        input_ids = encoded.input_ids.to(input_device)
        seq_len = input_ids.shape[1]
        middle_end = seq_len - args.recent_tokens
        if middle_end <= args.prefix_tokens:
            print(f"skip id={sample['id']} tokens={seq_len} no_middle", flush=True)
            continue
        attention_mask = encoded.attention_mask.to(input_device)
        CAPTURED.clear()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        shift_logits = outputs.logits[:, :-1].float()
        shift_labels = input_ids[:, 1:].to(shift_logits.device)
        target_start = max(0, shift_labels.shape[1] - args.tail_loss_tokens)
        labels = torch.full_like(shift_labels, -100)
        labels[:, target_start:] = shift_labels[:, target_start:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        if len(CAPTURED) != NUM_LAYERS:
            raise RuntimeError(f"captured {len(CAPTURED)} layers, expected {NUM_LAYERS}")

        sample_k = torch.zeros_like(k_fisher)
        sample_v = torch.zeros_like(v_fisher)
        for layer in range(NUM_LAYERS):
            key, value = CAPTURED[layer]
            if key.grad is None or value.grad is None:
                raise RuntimeError(f"missing K/V gradient at layer {layer}")
            kg = key.grad[:, :, args.prefix_tokens:middle_end].float()
            vg = value.grad[:, :, args.prefix_tokens:middle_end].float()
            rk = k_rotations[layer].to(device=kg.device)
            rv = v_rotations[layer].to(device=vg.device)
            # Runtime stores K'=K@Rk and V'=V@Rv, hence g'=g@R.
            kg = kg @ rk
            vg = vg @ rv
            sample_k[layer] = kg.square().sum(dim=(0, 2)).double().cpu()
            sample_v[layer] = vg.square().sum(dim=(0, 2)).double().cpu()
            observations[layer] += kg.shape[0] * kg.shape[2]
        k_fisher += sample_k
        v_fisher += sample_v
        sample_info = {"benchmark": sample["benchmark"], "id": sample["id"], "tokens": seq_len, "loss": loss.item(), "middle_tokens": middle_end - args.prefix_tokens}
        used.append(sample_info)
        if sample["id"] in control_ids:
            control_k += sample_k
            control_v += sample_v
            control_observations += middle_end - args.prefix_tokens
            control_used.append(sample_info)
        torch.save({"sample": sample_info, "input_ids": input_ids.cpu(), "k_gradient_square_sum": sample_k, "v_gradient_square_sum": sample_v}, raw_dir / (sample["id"].replace(":", "_") + ".pt"))
        (args.output.parent / "status.json").write_text(json.dumps({"used": len(used), "expected": len(prompts), "last_sample": sample_info}))
        print(
            f"sample={sample_idx + 1}/{len(prompts)} id={sample['id']} "
            f"tokens={seq_len} middle={middle_end - args.prefix_tokens} loss={loss.item():.6f}",
            flush=True,
        )
        del outputs, shift_logits, loss
        CAPTURED.clear()
        gc.collect()
        torch.cuda.empty_cache()

    if not used:
        raise RuntimeError("all selected prompts were too short for the INT1 middle region")
    denom = observations[:, None, None].clamp_min(1).double()
    k_fisher /= denom
    v_fisher /= denom
    k_weights, v_weights = regularize_fisher(
        k_fisher, v_fisher, args.shrinkage, args.winsor_quantile
    )
    checkpoint = {
        "format_version": 1,
        "model": args.model_name or args.model.name,
        "method": "diagonal_empirical_fisher_tail_next_token_ce_v1",
        "coordinate_system": "oscar_rotated_post_rope",
        "shape_contract": {"layers": NUM_LAYERS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM},
        "k_rotation_sha256": sha256_file(args.k_rotation),
        "v_rotation_sha256": sha256_file(args.v_rotation),
        "k_weights": k_weights,
        "v_weights": v_weights,
        "calibration": {
            "source": args.source,
            "input": source_description,
            "samples": used,
            "max_length": args.max_length,
            "tail_loss_tokens": args.tail_loss_tokens,
            "enable_thinking": True if args.enable_thinking else not args.disable_thinking,
            "prefix_tokens": args.prefix_tokens,
            "recent_tokens": args.recent_tokens,
            "anchor_tokens": args.anchor_tokens,
            "gpqa_csv_sha256": sha256_file(args.gpqa_csv),
            "shrinkage": args.shrinkage,
            "winsor_quantile": args.winsor_quantile,
            "seed": args.seed,
            "observations_per_layer": observations.tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    assert {sample["id"] for sample in used} == {sample["id"] for sample in prompts}, "all198 must contribute"
    assert {sample["id"] for sample in control_used} == control_ids
    ck, cv = regularize_fisher(control_k / control_observations[:, None, None], control_v / control_observations[:, None, None], args.shrinkage, args.winsor_quantile)
    control_checkpoint = dict(checkpoint)
    control_checkpoint["k_weights"] = ck
    control_checkpoint["v_weights"] = cv
    control_checkpoint["calibration"] = dict(checkpoint["calibration"], samples=control_used, observations_per_layer=control_observations.tolist(), input="original16 GPQA identities reaggregated with p16r64 middle mask")
    torch.save(control_checkpoint, args.output.parent / "anchor_fisher_diag_gpqa16_p16r64a240.pt")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples_used": len(used),
                "k_minmax": [k_weights.min().item(), k_weights.max().item()],
                "v_minmax": [v_weights.min().item(), v_weights.max().item()],
                "k_mean": k_weights.mean().item(),
                "v_mean": v_weights.mean().item(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
