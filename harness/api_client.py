"""Shared OpenAI-compatible chat client (Ollama, llama.cpp, vLLM)."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import httpx

try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

# Generic local defaults (Ollama). No machine-specific hosts.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434


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
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    estimated = False
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(content)
        estimated = True
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens, estimated


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str = "",
        api_key: str = "ollama",
        timeout: float = 300.0,
        verify: bool = False,
    ):
        if not base_url or not str(base_url).strip():
            raise ValueError("base_url is required (e.g. http://127.0.0.1:11434/v1)")
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
            verify=verify,
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
        started = time.perf_counter()
        try:
            response = self.client.post(
                "/chat/completions",
                json={
                    "model": model or self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
        except httpx.ConnectError as exc:
            msg = str(exc)
            if "WRONG_VERSION_NUMBER" in msg or "SSL" in msg:
                raise ConnectionError(
                    f"{exc}\nHint: API base is {self.base_url!r}. "
                    "WRONG_VERSION_NUMBER usually means you used https:// against an HTTP server. "
                    "Try --base-url http://HOST:PORT/v1 or --host/--port"
                ) from exc
            raise
        latency = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"] or ""
        prompt_tokens, completion_tokens, total_tokens, estimated = parse_usage(
            data, prompt_blob, content
        )
        return ChatResult(
            content=content,
            latency_s=round(latency, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tokens_estimated=estimated,
        )

    def close(self) -> None:
        self.client.close()
