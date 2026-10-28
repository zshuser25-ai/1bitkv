#!/usr/bin/env python3
"""Run official_v5 against SGLang's prompt-based /tokenize endpoint.

The fixed official_v5 runner sends chat messages to /tokenize, matching vLLM.
SGLang's endpoint accepts a rendered prompt instead. This wrapper changes only
the preflight tokenization transport; generation, scoring, fingerprints, retry
rules, and output files remain implemented by the unmodified suite runner.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


SUITE_ROOT = Path(os.environ.get("OFFICIAL_V5_ROOT", "/suite"))
RUNNER_PATH = SUITE_ROOT / "tools" / "run_accuracy_suite.py"
TOKENIZER_PATH = os.environ["SGLANG_TOKENIZER_PATH"]

sys.path.insert(0, str(SUITE_ROOT))
spec = importlib.util.spec_from_file_location("official_v5_runner", RUNNER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load official runner: {RUNNER_PATH}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)


@lru_cache(maxsize=None)
def server_max_model_len(base_url: str, model: str, timeout: int) -> int:
    response = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    response.raise_for_status()
    models = response.json()["data"]
    selected = next((item for item in models if item.get("id") == model), None)
    if selected is None and len(models) == 1:
        selected = models[0]
    if selected is None or "max_model_len" not in selected:
        raise RuntimeError(f"cannot resolve max_model_len for model={model!r}")
    return int(selected["max_model_len"])


def tokenize_prompt_sglang(
    *,
    base_url: str,
    model: str,
    sample: dict[str, Any],
    decoding: dict[str, Any],
    timeout: int,
) -> dict[str, int]:
    server_root = base_url.rstrip("/")
    if server_root.endswith("/v1"):
        server_root = server_root[:-3]

    template_kwargs: dict[str, Any] = {}
    if decoding.get("reasoning_effort"):
        template_kwargs["reasoning_effort"] = decoding["reasoning_effort"]
    prompt = tokenizer.apply_chat_template(
        runner.sample_messages(sample),
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    body = {"model": model, "prompt": prompt, "add_special_tokens": False}

    for attempt in range(1, 4):
        try:
            response = requests.post(
                f"{server_root}/tokenize", json=body, timeout=timeout
            )
            response.raise_for_status()
            break
        except Exception as exc:
            retryable = isinstance(exc, (requests.ConnectionError, requests.Timeout))
            if isinstance(exc, requests.HTTPError):
                status_code = getattr(exc.response, "status_code", None)
                retryable = status_code is not None and status_code >= 500
            if not retryable or attempt == 3:
                detail = getattr(getattr(exc, "response", None), "text", "")
                raise RuntimeError(f"SGLang tokenize failed: {exc!r}; body={detail}") from exc
            time.sleep(2 ** (attempt - 1))

    data = response.json()
    return {
        "input_tokens": int(data["count"]),
        "server_max_model_len": server_max_model_len(base_url, model, timeout),
    }


runner.tokenize_prompt = tokenize_prompt_sglang

if __name__ == "__main__":
    raise SystemExit(runner.main())
