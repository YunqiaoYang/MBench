# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""OpenAI-compatible VLM judge for MBench-T trigger-conditioned metrics."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from mbench.metrics.judge import JudgeResult, TriggerJudgeResult
from mbench.metrics.schema import PromptContext
from mbench.metrics.video_io import sample_bgr_frames


# No default endpoint or model is hard-coded: providers vary, and shipping a
# specific URL in source would leak the maintainer's preferred provider.
# Configure via the CLI flags ``--vlm-base-url``/``--vlm-model`` or the
# corresponding ``MBENCH_VLM_*`` environment variables.
DEFAULT_BASE_URL = ""
DEFAULT_MODEL = ""
ENV_BASE_URL = "MBENCH_VLM_BASE_URL"
ENV_API_KEY = "MBENCH_VLM_API_KEY"
ENV_MODEL = "MBENCH_VLM_MODEL"
ENV_ENDPOINT_PATH = "MBENCH_VLM_ENDPOINT_PATH"
ENV_MAX_TOKENS = "MBENCH_VLM_MAX_TOKENS"
ENV_API_STYLE = "MBENCH_VLM_API_STYLE"


@dataclass
class OpenAICompatVLMConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    n_frames: int = 8
    jpeg_quality: int = 85
    timeout_s: float = 120.0
    temperature: float = 0.0
    endpoint_path: str | None = None
    max_tokens: int = 1024
    api_style: str = "chat_completions"
    max_retries: int = 3
    retry_backoff: float = 2.0

    @classmethod
    def from_env(cls, **overrides: Any) -> OpenAICompatVLMConfig:
        api_key = overrides.pop("api_key", None) or os.environ.get(ENV_API_KEY, "")
        if not api_key:
            raise ValueError(
                f"Missing VLM API key. Set {ENV_API_KEY} or pass api_key= explicitly."
            )
        base_url = (
            overrides.pop("base_url", None)
            or os.environ.get(ENV_BASE_URL, "")
            or DEFAULT_BASE_URL
        )
        if not base_url:
            raise ValueError(
                f"Missing VLM base URL. Set {ENV_BASE_URL} or pass base_url= explicitly "
                "(e.g. an OpenAI-compatible /v1 endpoint)."
            )
        model = (
            overrides.pop("model", None)
            or os.environ.get(ENV_MODEL, "")
            or DEFAULT_MODEL
        )
        if not model:
            raise ValueError(
                f"Missing VLM model name. Set {ENV_MODEL} or pass model= explicitly."
            )
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            endpoint_path=overrides.pop("endpoint_path", None)
            or os.environ.get(ENV_ENDPOINT_PATH)
            or None,
            max_tokens=int(overrides.pop("max_tokens", None) or os.environ.get(ENV_MAX_TOKENS, 1024)),
            api_style=str(overrides.pop("api_style", None) or os.environ.get(ENV_API_STYLE, "chat_completions")),
            **overrides,
        )


class OpenAICompatMBenchTJudge:
    """VLM judge backed by an OpenAI-style chat/completions endpoint."""

    def __init__(self, config: OpenAICompatVLMConfig | None = None, **config_kwargs: Any):
        self.config = config or OpenAICompatVLMConfig.from_env(**config_kwargs)

    def judge_trigger(
        self,
        *,
        context: PromptContext,
        video_path: str,
        prompt: str,
    ) -> TriggerJudgeResult:
        raw = self._judge_video(
            video_path=video_path,
            prompt=prompt,
            response_hint=(
                "Return compact JSON with keys: triggered (boolean), confidence (0-1), reason (string under 10 words)."
            ),
        )
        triggered = _coerce_bool(raw.get("triggered"))
        confidence = _coerce_optional_float(raw.get("confidence"))
        return TriggerJudgeResult(
            triggered=triggered,
            score=1.0 if triggered else 0.0,
            confidence=confidence,
            reason=str(raw.get("reason") or ""),
            metadata={"raw_response": raw, "model": self.config.model},
        )

    def judge_consistency(
        self,
        *,
        context: PromptContext,
        video_path: str,
        prompt: str,
        score_keys: tuple[str, ...] | None = None,
    ) -> JudgeResult:
        keys = score_keys or (
            "progress_trigger_score",
            "evolution_correctness_score",
        )
        hint_parts = [f"{key} (1-5)" for key in keys]
        raw = self._judge_video(
            video_path=video_path,
            prompt=prompt,
            response_hint=(
                "Return JSON with keys: "
                + ", ".join(hint_parts)
                + ", confidence (0-1 optional), reason (string)."
            ),
        )
        scores: dict[str, float] = {}
        for key in keys:
            if key in raw:
                scores[key] = float(raw[key])
        confidence = _coerce_optional_float(raw.get("confidence"))
        return JudgeResult(
            score=0.0,
            scores=scores,
            confidence=confidence,
            reason=str(raw.get("reason") or ""),
            metadata={"raw_response": raw, "model": self.config.model},
        )

    def _judge_video(self, *, video_path: str, prompt: str, response_hint: str) -> dict[str, Any]:
        text = "\n".join(
            [
                prompt,
                "",
                response_hint,
                "Respond with JSON only. Do not wrap the JSON in markdown fences.",
            ]
        )
        image_urls = _video_frame_data_urls(
            video_path,
            n_frames=self.config.n_frames,
            jpeg_quality=self.config.jpeg_quality,
        )
        if self.config.api_style == "responses":
            payload = _build_responses_payload(
                model=self.config.model,
                text=text,
                image_urls=image_urls,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif self.config.api_style == "chat_completions":
            payload = _build_chat_completions_payload(
                model=self.config.model,
                text=text,
                image_urls=image_urls,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported VLM API style: {self.config.api_style!r}")
        response_text = self._post_vlm(payload)
        return _parse_json_object(response_text)

    def _post_vlm(self, payload: dict[str, Any]) -> str:
        import time as _time

        url = _api_url(self.config.base_url, self.config.endpoint_path, self.config.api_style)
        max_retries = getattr(self.config, 'max_retries', 3)
        retry_backoff = getattr(self.config, 'retry_backoff', 2.0)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return _extract_response_text(body, self.config.api_style)
            except urllib.error.HTTPError as e:
                last_error = e
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                if e.code in (429, 503, 502, 504) and attempt < max_retries:
                    wait = retry_backoff * (2 ** attempt)
                    _time.sleep(wait)
                    continue
                raise RuntimeError(f"VLM request failed ({e.code}): {detail}") from e
            except urllib.error.URLError as e:
                last_error = e
                if attempt < max_retries:
                    _time.sleep(retry_backoff * (2 ** attempt))
                    continue
        raise RuntimeError(f"VLM request failed after {max_retries + 1} attempts: {last_error}")


def _build_chat_completions_payload(
    *,
    model: str,
    text: str,
    image_urls: list[str],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image_url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _build_responses_payload(
    *,
    model: str,
    text: str,
    image_urls: list[str],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    for image_url in image_urls:
        content.append({"type": "input_image", "image_url": image_url})
    return {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }


def _api_url(base_url: str, endpoint_path: str | None, api_style: str) -> str:
    """Resolve an OpenAI-compatible VLM endpoint.

    If ``endpoint_path`` is not provided, bare hosts are treated as services
    whose OpenAI-compatible endpoint lives under ``/v1/chat/completions``;
    base URLs that already contain a path append ``/chat/completions``.
    """

    base = base_url.rstrip("/")
    if endpoint_path:
        return base + "/" + endpoint_path.strip("/")
    parsed = urllib.parse.urlparse(base)
    path = parsed.path.rstrip("/")
    if api_style == "responses":
        suffix = "/v1/responses" if path in {"", "/"} else "/responses"
    else:
        suffix = "/v1/chat/completions" if path in {"", "/"} else "/chat/completions"
    return base + suffix


def _extract_response_text(body: dict[str, Any], api_style: str) -> str:
    if api_style == "responses":
        if isinstance(body.get("output_text"), str):
            return str(body["output_text"])
        try:
            output = body["output"]
            for item in output:
                for content in item.get("content", []):
                    if isinstance(content, dict) and isinstance(content.get("text"), str):
                        return str(content["text"])
        except (KeyError, TypeError):
            pass
        raise RuntimeError(f"Unexpected VLM response shape: {body}")
    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected VLM response shape: {body}") from e


def _video_frame_data_urls(
    video_path: str,
    *,
    n_frames: int,
    jpeg_quality: int,
) -> list[str]:
    import cv2

    frames_bgr, frame_indices = sample_bgr_frames(video_path, n_frames=n_frames)
    if not frames_bgr:
        raise ValueError(f"cannot sample frames from video: {video_path}")

    urls: list[str] = []
    for frame_idx, frame in zip(frame_indices, frames_bgr):
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            continue
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        urls.append(f"data:image/jpeg;base64,{b64}")
    if not urls:
        raise ValueError(f"failed to encode sampled frames from video: {video_path}")
    return urls


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("VLM returned empty response")

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        raise ValueError(f"VLM response is not JSON: {text[:500]}")
    candidate = text[start : end + 1] if end > start else text[start:]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        recovered = _recover_flat_json_fields(candidate)
        if recovered:
            return recovered
        raise ValueError(f"VLM response is not JSON: {text[:500]}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"VLM JSON must be an object, got: {type(parsed).__name__}")
    return parsed


def _recover_flat_json_fields(text: str) -> dict[str, Any]:
    """Recover simple fields from a truncated flat JSON object."""

    recovered: dict[str, Any] = {}
    bool_match = re.search(r'"triggered"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
    if bool_match:
        recovered["triggered"] = bool_match.group(1).lower() == "true"
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if confidence_match:
        recovered["confidence"] = float(confidence_match.group(1))
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', text, flags=re.DOTALL)
    if reason_match:
        recovered["reason"] = reason_match.group(1).replace("\n", " ").strip()
    if "triggered" in recovered:
        recovered["_recovered_from_truncated_json"] = True
        return recovered
    return {}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"cannot coerce triggered flag from {value!r}")


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
