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
DISABLE_THINKING = os.environ.get("SGLANG_DISABLE_THINKING", "0") == "1"
ENABLE_THINKING = os.environ.get("SGLANG_ENABLE_THINKING", "0") == "1"
if DISABLE_THINKING and ENABLE_THINKING:
    raise RuntimeError("SGLANG_DISABLE_THINKING and SGLANG_ENABLE_THINKING cannot both be 1")


def explicit_thinking_kwargs() -> dict[str, bool]:
    if ENABLE_THINKING:
        return {"enable_thinking": True}
    if DISABLE_THINKING:
        return {"enable_thinking": False}
    return {}


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
    template_kwargs: dict[str, Any] = explicit_thinking_kwargs()
    if decoding.get("reasoning_effort"):
        template_kwargs["reasoning_effort"] = decoding["reasoning_effort"]
    prompt = tokenizer.apply_chat_template(
        runner.sample_messages(sample),
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    # Use the exact served tokenizer locally. LongBench v2 can contain
    # multi-million-word documents; serializing their complete token-id arrays
    # through /tokenize repeatedly makes the official middle-truncation
    # preflight needlessly expensive without changing any evaluation semantics.
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return {
        "input_tokens": len(input_ids),
        "server_max_model_len": server_max_model_len(base_url, model, timeout),
    }


runner.tokenize_prompt = tokenize_prompt_sglang


def tokenize_text_ids_local(
    *, base_url: str, model: str, text: str, timeout: int
) -> list[int]:
    del base_url, model, timeout
    return tokenizer.encode(text, add_special_tokens=False)


def detokenize_ids_local(
    *, base_url: str, model: str, ids: list[int], timeout: int
) -> str:
    del base_url, model, timeout
    return tokenizer.decode(ids, skip_special_tokens=False)


# Keep the runner's official first-half + last-half algorithm, but avoid HTTP
# round trips for the tokenizer operations used by that algorithm.
runner.tokenize_text_ids = tokenize_text_ids_local
runner.detokenize_ids = detokenize_ids_local

# Keep generation rendering identical to the local tokenization preflight.
# This is opt-in so existing official_v5 runs retain their frozen behavior.
_requests_post = requests.post


def post_with_template_mode(url: str, *args: Any, **kwargs: Any):
    if (DISABLE_THINKING or ENABLE_THINKING) and url.rstrip("/").endswith("/chat/completions"):
        body = dict(kwargs.get("json") or {})
        template_kwargs = dict(body.get("chat_template_kwargs") or {})
        template_kwargs.update(explicit_thinking_kwargs())
        body["chat_template_kwargs"] = template_kwargs
        kwargs["json"] = body
    return _requests_post(url, *args, **kwargs)


runner.requests.post = post_with_template_mode

# A small number of deterministic long-form generations can legitimately need
# more than the suite's 300-second transport timeout on this TP=4 setup.  The
# opt-in override changes only the HTTP wait time; token budgets, decoding,
# scoring, and the protocol fingerprint remain unchanged.
_request_completion = runner.request_completion


def request_completion_with_timeout_floor(*args: Any, **kwargs: Any) -> dict[str, Any]:
    timeout_floor = int(os.environ.get("OFFICIAL_V5_REQUEST_TIMEOUT_MIN", "0"))
    if timeout_floor:
        kwargs["timeout"] = max(int(kwargs["timeout"]), timeout_floor)
    return _request_completion(*args, **kwargs)


runner.request_completion = request_completion_with_timeout_floor

if __name__ == "__main__":
    raise SystemExit(runner.main())
