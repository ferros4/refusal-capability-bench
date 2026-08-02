"""Shared OpenAI-compatible chat client (Ollama, llama.cpp, vLLM)."""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any

import httpx

from harness.logging_config import get_logger

try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

log = get_logger(__name__)

# Generic local defaults (Ollama). No machine-specific hosts.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
RETRYABLE_STATUS_CODES = frozenset({502, 503})
RETRY_SLEEP_S = 5.0
DEFAULT_MAX_RETRIES = 10

# Streaming delta keys used by thinking / reasoning models (Qwen, DeepSeek, etc.)
_REASONING_DELTA_KEYS = (
    "reasoning_content",
    "reasoning",
    "thinking",
    "thought",
)


def resolve_base_url(
    base_url: str | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    scheme: str = "http",
) -> str:
    """
    Build the OpenAI-compatible API root (…/v1).

    Prefer an explicit base_url when provided. Otherwise:
    http://{host}:{port}/v1 with host default 127.0.0.1 and port default 11434.
    """
    if base_url:
        url = base_url.strip().rstrip("/")
        if not url:
            raise ValueError("base_url must not be empty")
        return url

    resolved_host = (host or DEFAULT_HOST).strip() or DEFAULT_HOST
    resolved_port = DEFAULT_PORT if port is None else int(port)
    if resolved_port <= 0 or resolved_port > 65535:
        raise ValueError(f"port out of range: {resolved_port}")
    return f"{scheme}://{resolved_host}:{resolved_port}/v1"


# Back-compat alias used by some imports; resolves to local Ollama default.
DEFAULT_BASE_URL = resolve_base_url()


@dataclass
class ChatResult:
    content: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_estimated: bool = False
    reasoning_content: str = ""
    used_reasoning_fallback: bool = False

    @property
    def tokens_per_sec(self) -> float:
        """Completion tokens per second (generation throughput)."""
        if self.latency_s <= 0:
            return 0.0
        return round(self.completion_tokens / self.latency_s, 3)


def estimate_tokens(text: str) -> int:
    """Rough token estimate when the API omits usage (~4 chars/token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def parse_usage(data: dict, prompt: str, content: str) -> tuple[int, int, int, bool]:
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    total_tokens = int(usage.get("total_tokens") or 0)
    estimated = False
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(content)
        estimated = True
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens, estimated


def _delta_content(delta: dict[str, Any]) -> str:
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if content is None:
        return ""
    return str(content)


def _delta_reasoning(delta: dict[str, Any]) -> str:
    if not isinstance(delta, dict):
        return ""
    for key in _REASONING_DELTA_KEYS:
        value = delta.get(key)
        if value:
            return str(value)
    return ""


def parse_sse_chat_stream(lines: Any) -> tuple[str, str, dict[str, Any]]:
    """
    Parse OpenAI-style SSE chat.completion.chunk lines.

    Returns (content, reasoning_content, usage_dict).
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    chunk_count = 0
    for raw in lines:
        if raw is None:
            continue
        line = (
            raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        )
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            if line == "[DONE]":
                break
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            log.debug("Skipping non-JSON SSE line: %s", line[:200])
            continue
        if not isinstance(chunk, dict):
            continue
        chunk_count += 1
        if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            text = _delta_content(delta)
            if text:
                content_parts.append(text)
            reason = _delta_reasoning(delta)
            if reason:
                reasoning_parts.append(reason)
            message = choice.get("message") or {}
            if isinstance(message, dict):
                if message.get("content") and not delta.get("content"):
                    content_parts.append(str(message.get("content") or ""))
                for key in _REASONING_DELTA_KEYS:
                    if message.get(key) and not delta.get(key):
                        reasoning_parts.append(str(message.get(key) or ""))
    log.debug(
        "SSE stream parsed chunks=%s content_chars=%s reasoning_chars=%s usage=%s",
        chunk_count,
        sum(len(part) for part in content_parts),
        sum(len(part) for part in reasoning_parts),
        bool(usage),
    )
    return "".join(content_parts), "".join(reasoning_parts), usage


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str = "",
        api_key: str = "ollama",
        timeout: float = 600.0,
        verify: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_sleep_s: float = RETRY_SLEEP_S,
    ):
        if not base_url or not str(base_url).strip():
            raise ValueError("base_url is required (e.g. http://127.0.0.1:11434/v1)")
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.max_retries = max(0, int(max_retries))
        self.retry_sleep_s = float(retry_sleep_s)
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
            verify=verify,
        )
        log.debug(
            "ChatClient init base_url=%s model=%s timeout=%s verify=%s max_retries=%s",
            self.base_url,
            self.model,
            timeout,
            verify,
            self.max_retries,
        )

    def chat(
        self,
        user: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        model: str | None = None,
    ) -> ChatResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        prompt_blob = (system + "\n" if system else "") + user
        model_id = model or self.model
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            # Request final usage chunk when the server supports it (OpenAI-compatible).
            "stream_options": {"include_usage": True},
        }
        log.debug(
            "chat start model=%s max_tokens=%s temperature=%s prompt_chars=%s",
            model_id,
            max_tokens,
            temperature,
            len(prompt_blob),
        )
        started = time.perf_counter()
        attempt = 0
        content = ""
        reasoning = ""
        usage: dict[str, Any] = {}

        while True:
            try:
                with self.client.stream(
                    "POST", "/chat/completions", json=payload
                ) as response:
                    log.debug(
                        "chat HTTP status=%s attempt=%s",
                        response.status_code,
                        attempt,
                    )
                    if response.status_code in RETRYABLE_STATUS_CODES:
                        response.read()
                        if attempt >= self.max_retries:
                            log.error(
                                "chat failed after retries status=%s model=%s",
                                response.status_code,
                                model_id,
                            )
                            response.raise_for_status()
                        attempt += 1
                        log.warning(
                            "chat retryable status=%s attempt=%s/%s sleep=%.1fs model=%s",
                            response.status_code,
                            attempt,
                            self.max_retries,
                            self.retry_sleep_s,
                            model_id,
                        )
                        time.sleep(self.retry_sleep_s)
                        continue

                    response.raise_for_status()
                    content, reasoning, usage = parse_sse_chat_stream(
                        response.iter_lines()
                    )
                    break
            except httpx.ConnectError as exc:
                msg = str(exc)
                log.error("chat connect error model=%s: %s", model_id, exc)
                if "WRONG_VERSION_NUMBER" in msg or "SSL" in msg:
                    raise ConnectionError(
                        f"{exc}\nHint: API base is {self.base_url!r}. "
                        "WRONG_VERSION_NUMBER usually means you used https:// against an HTTP server. "
                        "Try --base-url http://HOST:PORT/v1 or --host/--port"
                    ) from exc
                raise
            except httpx.HTTPStatusError:
                log.exception("chat HTTP error model=%s", model_id)
                raise
            except httpx.TimeoutException:
                log.error(
                    "chat timeout model=%s after %.1fs",
                    model_id,
                    time.perf_counter() - started,
                )
                raise

        latency = time.perf_counter() - started
        used_fallback = False
        if not content.strip() and reasoning.strip():
            # Thinking models often burn max_tokens on reasoning_* deltas and leave
            # content empty — keep the trace so results.csv is not blank.
            content = reasoning
            used_fallback = True
            log.warning(
                "chat empty content; using reasoning stream as response "
                "(model=%s latency=%.2fs reasoning_chars=%s). "
                "Consider raising --max-tokens so a final answer can be emitted.",
                model_id,
                latency,
                len(reasoning),
            )
        elif not content.strip():
            log.warning(
                "chat empty content and empty reasoning model=%s latency=%.2fs usage=%s",
                model_id,
                latency,
                usage,
            )

        prompt_tokens, completion_tokens, total_tokens, estimated = parse_usage(
            {"usage": usage}, prompt_blob, content
        )
        log.info(
            "chat done model=%s latency=%.2fs content_chars=%s reasoning_chars=%s "
            "tokens=%s/%s fallback=%s",
            model_id,
            latency,
            len(content),
            len(reasoning),
            completion_tokens,
            prompt_tokens,
            used_fallback,
        )
        return ChatResult(
            content=content,
            latency_s=round(latency, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tokens_estimated=estimated,
            reasoning_content=reasoning,
            used_reasoning_fallback=used_fallback,
        )

    def close(self) -> None:
        log.debug("ChatClient close base_url=%s", self.base_url)
        self.client.close()
