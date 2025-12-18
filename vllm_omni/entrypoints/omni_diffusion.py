# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import time
import os
import socket
import json
from dataclasses import fields
from typing import Any
from datetime import datetime

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.entrypoints.stage_utils import append_jsonl


logging.basicConfig(level=logging.INFO)
logger = init_logger(__name__)


def _make_run_stats_path() -> str:
    """
    Create a per-run jsonl path in a fixed directory, named by timestamp.
    If OMNI_DIFFUSION_STATS is set, use it as an explicit file path.
    """
    explicit = os.getenv("OMNI_DIFFUSION_STATS")
    if explicit:
        # User explicitly provided a file path -> respect it.
        parent = os.path.dirname(explicit)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return explicit

    stats_dir = os.getenv("OMNI_DIFFUSION_STATS_DIR", "omni_diffusion_stats")
    os.makedirs(stats_dir, exist_ok=True)

    # Local time; include milliseconds + pid to avoid collisions
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    pid = os.getpid()

    filename = f"omni_diffusion_{ts}_pid{pid}.jsonl"
    return os.path.join(stats_dir, filename)


_STATS_PATH = _make_run_stats_path()

_PRINT_DIFFUSION_METRICS = os.getenv("OMNI_DIFFUSION_PRINT", "1") == "1"




def _record(event: str, **kv: Any) -> None:
    """
    Record a structured profiling event to jsonl.
    """
    kv["event"] = event
    kv["ts"] = time.time()
    kv["pid"] = os.getpid()
    kv["host"] = socket.gethostname()
    try:
        append_jsonl(_STATS_PATH, kv)
    except Exception as e:
        logger.error("Failed to write omni diffusion stats: %s", e)


def _extract_kwargs_detail(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and sanitize selected kwargs for logging / jsonl.
    """
    keys_of_interest = (
        "height",
        "width",
        "generator",
        "true_cfg_scale",
        "num_inference_steps",
        "num_outputs_per_prompt",
    )

    out: dict[str, Any] = {}
    for k in keys_of_interest:
        if k not in kwargs:
            continue
        v = kwargs[k]
        try:
            json.dumps(v)
            out[k] = v
        except Exception:
            # torch.Generator / custom objects
            out[k] = repr(v)
    return out


def prepare_requests(prompt: str | list[str], **kwargs):
    field_names = {f.name for f in fields(OmniDiffusionRequest)}
    init_kwargs = {"prompt": prompt}
    for key, value in kwargs.items():
        if key in field_names:
            init_kwargs[key] = value
    return OmniDiffusionRequest(**init_kwargs)


class OmniDiffusion:
    """
    High-level entrypoint for vLLM-Omni diffusion models.
    """

    def __init__(self, od_config: OmniDiffusionConfig | None = None, **kwargs):
        t0 = time.perf_counter()

        if od_config is None:
            od_config = OmniDiffusionConfig.from_kwargs(**kwargs)
        elif isinstance(od_config, dict):
            od_config = OmniDiffusionConfig.from_kwargs(**od_config)

        self.od_config = od_config
        self._default_num_steps = getattr(
            od_config,
            "num_inference_steps",
            None,
        )

        model_index = get_hf_file_to_dict(
            "model_index.json",
            od_config.model,
        )
        od_config.model_class_name = model_index.get("_class_name")

        tf_config_dict = get_hf_file_to_dict(
            "transformer/config.json",
            od_config.model,
        )
        od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)

        self.engine: DiffusionEngine = DiffusionEngine.make_engine(od_config)

        init_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "OmniDiffusion initialized: model=%s, class=%s, init_ms=%.2f",
            od_config.model,
            od_config.model_class_name,
            init_ms,
        )

        _record(
            "engine_load",
            model=od_config.model,
            model_class=od_config.model_class_name,
            init_ms=init_ms,
        )

    def generate(
        self,
        prompt: str | list[str],
        **kwargs,
    ):
        t_total_begin = time.perf_counter()

        if isinstance(prompt, str):
            prompts = [prompt]
            prompt_chars = len(prompt)
        elif isinstance(prompt, list):
            prompts = list(prompt)
            prompt_chars = sum(len(p) for p in prompt)
        else:
            raise ValueError("Prompt must be a string or a list of strings")

        requests: list[OmniDiffusionRequest] = [
            prepare_requests(p, **kwargs) for p in prompts
        ]

        kwargs_detail = _extract_kwargs_detail(kwargs)

        _record(
            "request_scheduled",
            n_requests=len(requests),
            prompt_chars=prompt_chars,
            **kwargs_detail,
        )

        t_engine_begin = time.perf_counter()
        out = self.engine.step(requests)
        engine_ms = (time.perf_counter() - t_engine_begin) * 1000

        total_ms = (time.perf_counter() - t_total_begin) * 1000

        num_steps = kwargs_detail.get("num_inference_steps")
        denoise_avg_ms = (
            engine_ms / num_steps
            if num_steps
            else None
        )

        input_tokens = prompt_chars  # approximation
        input_tokens_per_s = (
            input_tokens / (total_ms / 1000)
            if total_ms > 0
            else None
        )

        logger.info(
            "request_finished: n_requests=%d, total_ms=%.2f",
            len(requests),
            total_ms,
        )

        if _PRINT_DIFFUSION_METRICS:
            logger.info(
            "request_scheduled: n_requests=%d, kwargs_keys=%s, kwargs_detail=%s",
            len(requests),
            list(kwargs.keys()),
            kwargs_detail,
        )
            logger.info(
                "OMNI_DIFFUSION_METRICS %s",
                json.dumps(
                    {
                        "prompt_chars": prompt_chars,
                        "input_tokens": input_tokens,
                        "input_tokens_per_s": input_tokens_per_s,
                        "num_inference_steps": num_steps,
                        "diffusion_total_ms": engine_ms,
                        "denoise_avg_ms": denoise_avg_ms,
                        "total_ms": total_ms,
                    },
                    ensure_ascii=False,
                ),
            )

        _record(
            "request_finished",
            n_requests=len(requests),
            total_ms=total_ms,
            diffusion_total_ms=engine_ms,
            denoise_avg_ms=denoise_avg_ms,
            input_tokens=input_tokens,
            input_tokens_per_s=input_tokens_per_s,
        )

        return out

    def close(self) -> None:
        self.engine.close()

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass